from dataclasses import dataclass

import numpy as np
import pyvista as pv
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
from torch import cosine_similarity

from beaty_common.pose_utils import sixd_to_quat
from beaty_common.torch_nets import MapTensors, ReplayTensors
from vendor.poselib.poselib import quat_inverse

from .utils.pose_utils import quat_rotate, unity_to_zup

torch.set_printoptions(precision=3, sci_mode=False)

plane_width = 1.8
plane_height = 1.05
plane_left = -plane_width / 2
plane_bottom = 0
plane_right = plane_width / 2
plane_top = plane_height
note_x_offset = 0.9
note_y_offset = 0.0
note_z_offset = 0.0
PLAYER_HEIGHT = 1.5044

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@dataclass(slots=True)
class PlayerMotion:
    carry_3p: torch.Tensor
    trajectory_3p: torch.Tensor


@dataclass(slots=True)
class SaberSimulation:
    motion: PlayerMotion
    replay: ReplayTensors
    map_profiles: MapTensors
    note_jump_speed: float


@dataclass(slots=True)
class MapObjectIds:
    notes: torch.Tensor
    bombs: torch.Tensor


@dataclass(slots=True)
class NoteCollisionMasks:
    appeared: torch.Tensor
    good_cut: torch.Tensor
    bad_cut: torch.Tensor
    cut_angles: torch.Tensor


@dataclass(slots=True)
class SimulationMasks:
    notes: NoteCollisionMasks
    bomb_appeared: torch.Tensor
    bomb_collided: torch.Tensor
    head_obstacle_distances: torch.Tensor


@dataclass(slots=True)
class SaberScore:
    normalized_score: torch.Tensor
    n_opportunities: torch.Tensor
    n_hits: torch.Tensor
    n_misses: torch.Tensor
    n_goods: torch.Tensor


@dataclass(slots=True)
class CandidateFeedback:
    normalized_score: torch.Tensor
    hit_note_mask: torch.Tensor
    bomb_penalty: torch.Tensor
    obstacle_penalty: torch.Tensor


@dataclass(slots=True)
class SaberEvaluation:
    score: SaberScore
    feedback: CandidateFeedback


@dataclass(slots=True)
class BoxGeometry:
    vertices: torch.Tensor
    quats: torch.Tensor


class TorchSaber:
    @staticmethod
    def evaluate_and_simulate(
        simulation: SaberSimulation,
        note_alive_mask: torch.Tensor,
    ) -> CandidateFeedback:
        masks = TorchSaber.simulate(simulation)
        evaluation = TorchSaber.evaluate(
            note_alive_mask.to(device),
            masks,
        )
        return evaluation.feedback

    @staticmethod
    def evaluate(
        note_alive_mask: torch.Tensor,
        masks: SimulationMasks,
        batch_size: int | None = None,
    ) -> SaberEvaluation:
        note_appeared_mask = masks.notes.appeared.to(device)
        bomb_appeared_mask = masks.bomb_appeared.to(device)
        bomb_collided_mask = masks.bomb_collided.to(device)
        head_obstacle_signed_distances = masks.head_obstacle_distances.to(device)
        good_cut_mask = masks.notes.good_cut.to(device)
        bad_cut_mask = masks.notes.bad_cut.to(device)
        cut_angles = masks.notes.cut_angles.to(device)

        max_int8 = torch.iinfo(torch.int8).max
        min_int8 = torch.iinfo(torch.int8).min
        note_batches = torch.split(
            torch.arange(note_alive_mask.shape[2], device=device),
            batch_size if batch_size is not None else note_alive_mask.shape[2],
        )

        n_opportunities = (note_alive_mask & note_appeared_mask.any(-2)).sum(-1)
        n_hits = torch.zeros_like(n_opportunities)
        n_goods = torch.zeros_like(n_opportunities)
        total_score = torch.zeros_like(n_opportunities, dtype=torch.float32)
        hit_note_batches = []
        pre_window_size = 24
        post_window_size = 24
        score_shape = list(cut_angles.shape[:3]) + [1]
        frame_index_grid = torch.arange(cut_angles.shape[2], device=device)[None, None, :, None]

        for note_batch in note_batches:
            hit_mask = good_cut_mask[..., note_batch] | bad_cut_mask[..., note_batch]
            any_good_cut = good_cut_mask[..., note_batch].any(-2)
            any_bad_cut = bad_cut_mask[..., note_batch].any(-2)

            good_cumsum = any_good_cut.cumsum(2, dtype=torch.uint8)
            bad_cumsum = any_bad_cut.cumsum(2, dtype=torch.uint8)
            good_run_mask = ((good_cumsum > 0) & ~(bad_cumsum > 0)).cumsum(2) > 0
            good_run_mask = good_run_mask & note_alive_mask[..., None, note_batch]
            final_good_mask = good_run_mask[:, :, -1]
            first_good_frame = good_run_mask.cumsum(2) == 1

            torch.cuda.empty_cache()
            batch_cut_angles = cut_angles[..., note_batch] * 1

            pre_indices = torch.arange(pre_window_size, device=device)[None, None].repeat(score_shape)
            pre_indices += frame_index_grid
            pre_indices -= pre_window_size
            pre_indices = pre_indices.clamp(0, cut_angles.shape[2] - 1)
            cut_angles_for_min = batch_cut_angles.nan_to_num(max_int8).to(torch.int8)
            relevant_pre_angles = torch.take_along_dim(cut_angles_for_min.unsqueeze(-2), pre_indices.unsqueeze(-1), -3)
            min_pre_angles = relevant_pre_angles.min(-2)[0]
            del relevant_pre_angles
            torch.cuda.empty_cache()
            min_pre_angles = torch.where(min_pre_angles == max_int8, torch.nan, min_pre_angles).to(torch.float)
            pre_score = torch.clamp(min_pre_angles, min=-100) / -100
            pre_score = torch.clamp(pre_score, max=1.0, min=0.0)

            post_indices = torch.arange(post_window_size + 1, device=device)[None, None].repeat(score_shape)
            post_indices += frame_index_grid
            post_indices = post_indices.clamp(0, cut_angles.shape[2] - 1)
            cut_angles_for_max = batch_cut_angles.nan_to_num(min_int8).to(torch.int8)
            relevant_post_angles = torch.take_along_dim(cut_angles_for_max.unsqueeze(-2), post_indices.unsqueeze(-1), -3)
            max_post_angles = relevant_post_angles.max(-2)[0]
            del relevant_post_angles
            torch.cuda.empty_cache()
            max_post_angles = torch.where(max_post_angles == min_int8, torch.nan, max_post_angles).to(torch.float)
            post_score = torch.clamp(max_post_angles, max=60) / 60
            post_score = torch.clamp(post_score, max=1.0, min=0.0)

            swing_score = 0.7 * pre_score + 0.3 * post_score
            swing_score = torch.where(first_good_frame, swing_score, torch.nan)
            total_score += swing_score.nansum(-1).nansum(-1).nan_to_num(0)

            hit_note_mask = hit_mask.any(-2).any(2)
            hit_note_batches.append(hit_note_mask)
            n_hits += hit_note_mask.sum(-1)
            n_goods += final_good_mask.sum(-1)

        denominator = torch.where(n_opportunities == 0, torch.ones_like(n_opportunities), n_opportunities)
        normalized_score = total_score / denominator
        n_misses = n_opportunities - n_hits
        bomb_penalty = bomb_collided_mask.sum(2).sum(-1) / (bomb_appeared_mask.sum(2).sum(-1) + 1e-7)
        obstacle_penalty = head_obstacle_signed_distances.nanmean(-1).nanmean(-1).nan_to_num(10)
        obstacle_penalty[obstacle_penalty < 0] = -torch.inf

        return SaberEvaluation(
            score=SaberScore(
                normalized_score=normalized_score,
                n_opportunities=n_opportunities,
                n_hits=n_hits,
                n_misses=n_misses,
                n_goods=n_goods,
            ),
            feedback=CandidateFeedback(
                normalized_score=normalized_score,
                hit_note_mask=torch.cat(hit_note_batches, dim=-1),
                bomb_penalty=bomb_penalty,
                obstacle_penalty=obstacle_penalty,
            ),
        )

    @staticmethod
    def simulate(
        simulation: SaberSimulation,
        batch_size: int | None = None,
    ) -> SimulationMasks:
        replay = simulation.replay
        trajectory_3p = simulation.motion.trajectory_3p
        carry_3p = simulation.motion.carry_3p
        assert replay.note_ids is not None
        assert replay.bomb_ids is not None
        frame_batches = torch.split(
            torch.arange(trajectory_3p.shape[2], device=device),
            batch_size if batch_size is not None else trajectory_3p.shape[2],
        )

        note_appeared_batches = []
        bomb_appeared_batches = []
        bomb_collided_batches = []
        obstacle_distance_batches = []
        good_cut_batches = []
        bad_cut_batches = []
        cut_angle_batches = []

        map_object_ids = MapObjectIds(
            notes=torch.arange(simulation.map_profiles.notes.shape[1], device=device),
            bombs=torch.arange(simulation.map_profiles.bombs.shape[1], device=device),
        )

        for frame_batch in frame_batches:
            replay_window = ReplayTensors(
                replay.notes[:, :, frame_batch],
                replay.bombs[:, :, frame_batch],
                replay.obstacles[:, :, frame_batch],
                note_ids=replay.note_ids[:, :, frame_batch],
                bomb_ids=replay.bomb_ids[:, :, frame_batch],
            )
            masks = TorchSaber.get_collision_masks(
                PlayerMotion(carry_3p, trajectory_3p[:, :, frame_batch]),
                replay_window,
                simulation.note_jump_speed,
                map_object_ids,
            )
            note_appeared_batches.append(masks.notes.appeared)
            bomb_appeared_batches.append(masks.bomb_appeared)
            bomb_collided_batches.append(masks.bomb_collided)
            obstacle_distance_batches.append(masks.head_obstacle_distances)
            good_cut_batches.append(masks.notes.good_cut)
            bad_cut_batches.append(masks.notes.bad_cut)
            cut_angle_batches.append(masks.notes.cut_angles)
            carry_3p = trajectory_3p[:, :, frame_batch[[-1]]]

        return SimulationMasks(
            notes=NoteCollisionMasks(
                appeared=torch.cat(note_appeared_batches, dim=2),
                good_cut=torch.cat(good_cut_batches, dim=2),
                bad_cut=torch.cat(bad_cut_batches, dim=2),
                cut_angles=torch.cat(cut_angle_batches, dim=2),
            ),
            bomb_appeared=torch.cat(bomb_appeared_batches, dim=2),
            bomb_collided=torch.cat(bomb_collided_batches, dim=2),
            head_obstacle_distances=torch.cat(obstacle_distance_batches, dim=2),
        )

    @staticmethod
    def get_note_geometry(
        note_bags: torch.Tensor,
        note_jump_speed: float,
        collider_type: int,
    ) -> BoxGeometry:
        n_songs, n_cands, n_frames, n_notes = note_bags.shape[:4]
        if collider_type == 0:
            note_collider_mesh = pv.Cube(x_length=1.0, y_length=0.8, z_length=0.5)
        elif collider_type == 1:
            note_collider_mesh = pv.Cube(x_length=1.0, y_length=0.8, z_length=0.8)
        else:
            note_collider_mesh = pv.Cube(x_length=0.4, y_length=0.4, z_length=0.4)

        note_vertices = torch.tensor(note_collider_mesh.points, dtype=torch.float, device=device)
        note_vertices = note_vertices[None, None, None, None].repeat(n_songs, n_cands, n_frames, n_notes, 1, 1)

        note_angle_degrees = np.array([0, 180, -90, 90, -45, 45, -135, 135, 0])
        note_values = note_bags.detach().cpu().numpy() * 1
        note_values = np.where(np.isnan(note_values), 0, note_values).astype(int)
        note_angles = np.where(
            np.isnan(note_bags[..., -2].detach().cpu().numpy()),
            np.nan,
            note_angle_degrees[note_values[..., -2]],
        )
        note_quats = Rotation.from_euler("x", note_angles.reshape(-1), degrees=True).as_quat()
        note_quats = torch.tensor(note_quats.reshape(n_songs, n_cands, n_frames, n_notes, 4), dtype=torch.float, device=device)
        note_vertices = quat_rotate(note_quats[..., None, :].repeat_interleave(8, -2), note_vertices)

        plane_grid = np.array(
            np.meshgrid(
                np.linspace(plane_right, plane_left, 4),
                np.linspace(plane_bottom, plane_top, 3),
            )
        ).transpose((2, 1, 0))
        plane_grid = torch.tensor(plane_grid, dtype=torch.float, device=device)

        note_positions = plane_grid[note_values[..., 3], note_values[..., 4]]
        note_positions = torch.concatenate([note_bags[..., [0]] * note_jump_speed + note_x_offset, note_positions], dim=-1)
        note_positions[..., 1] += note_y_offset
        note_positions[..., 2] += PLAYER_HEIGHT / 2
        note_positions[..., 2] += note_z_offset

        note_vertices += note_positions[..., None, :]
        if collider_type == 0 or collider_type == 1:
            note_vertices[..., 0] -= 0.25

        return BoxGeometry(note_vertices, note_quats)

    @staticmethod
    def get_obstacle_verts_and_normals(
        obstacle_bags: torch.Tensor,
        note_jump_speed: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_songs, n_cands, n_frames, n_obstacles = obstacle_bags.shape[:4]
        obstacle_values = obstacle_bags.detach().cpu().numpy() * 1
        obstacle_values = np.where(np.isnan(obstacle_values), 0, obstacle_values).astype(int)

        plane_leftright = np.linspace(plane_right, plane_left, 4)
        plane_bottomtop = np.linspace(plane_bottom, plane_top, 3)
        plane_width_interval = plane_leftright[0] - plane_leftright[1]
        plane_height_interval = plane_bottomtop[1] - plane_bottomtop[0]
        plane_grid = np.array(np.meshgrid(plane_leftright, plane_bottomtop)).transpose((2, 1, 0))
        plane_grid = torch.tensor(plane_grid, dtype=torch.float, device=device)

        obstacle_mesh = pv.Cube(x_length=0.4, y_length=plane_width_interval, z_length=plane_height_interval)
        obstacle_vertices = torch.tensor(obstacle_mesh.points, dtype=torch.float, device=device)
        obstacle_vertices = obstacle_vertices[None, None, None, None].repeat(n_songs, n_cands, n_frames, n_obstacles, 1, 1)

        obstacle_positions = plane_grid[0, 0] + (plane_grid[1, 1] - plane_grid[0, 0]) * torch.as_tensor(
            obstacle_values[..., [5, 6]],
            device=device,
        )
        obstacle_positions = torch.concatenate([obstacle_bags[..., [0]] * note_jump_speed + note_x_offset, obstacle_positions], dim=-1)
        obstacle_positions[..., 2] += PLAYER_HEIGHT / 2
        obstacle_positions[..., 2] += note_z_offset
        obstacle_positions[..., 1] += note_y_offset
        obstacle_vertices[..., [1, 2]] += obstacle_positions[..., None, [1, 2]]

        obstacle_vertices[..., 0] = obstacle_bags[..., [0]] * note_jump_speed + note_x_offset
        obstacle_vertices[..., -4:, 0] += obstacle_bags[..., [4]] * note_jump_speed + note_x_offset
        obstacle_vertices[..., [1, 2, 6, 7], 2] += (obstacle_bags[..., [-1]] - 1) * plane_height_interval
        obstacle_vertices[..., [0, 1, 4, 7], 1] = (
            obstacle_vertices[..., [2, 3, 5, 6], 1] - obstacle_bags[..., [-2]] * plane_width_interval
        )
        obstacle_vertices[obstacle_vertices.isnan().any(-1)] = torch.nan

        obstacle_normals = torch.tensor(obstacle_mesh.face_normals, dtype=torch.float, device=device)
        obstacle_normals = obstacle_normals[None, None, None, None].repeat(n_songs, n_cands, n_frames, n_obstacles, 1, 1)
        obstacle_normals[obstacle_vertices.isnan().any(-1).any(-1)] = torch.nan
        return obstacle_vertices, obstacle_normals

    @staticmethod
    def box_trail_collision_from_verts(
        box_geometry: BoxGeometry,
        saber_xyzs: torch.Tensor,
        saber_quats: torch.Tensor,
    ) -> torch.Tensor:
        box_verts = box_geometry.vertices
        box_quats = box_geometry.quats
        n_songs, n_cands, n_frames = box_verts.shape[:3]
        note_positions = box_verts.mean(-2)

        saber_xyz_pairs = torch.stack([saber_xyzs[:, :, :-1], saber_xyzs[:, :, 1:]], -2)
        saber_quat_pairs = torch.stack([saber_quats[:, :, :-1], saber_quats[:, :, 1:]], -2)

        hilt_positions = saber_xyz_pairs
        tip_offset = torch.tensor([[[[[1.0, 0, 0]]]]], dtype=torch.float, device=device).repeat(n_songs, n_cands, n_frames, 2, 2, 1)
        tip_positions = hilt_positions + quat_rotate(saber_quat_pairs.contiguous(), tip_offset)

        note_to_hilt = hilt_positions.unsqueeze(-2) - note_positions.unsqueeze(-3).unsqueeze(-3)
        note_to_tip = tip_positions.unsqueeze(-2) - note_positions.unsqueeze(-3).unsqueeze(-3)
        note_quats = box_quats.unsqueeze(-3).unsqueeze(-3).repeat(1, 1, 1, 2, 2, 1, 1)
        inverse_note_quats = quat_inverse(note_quats)
        hilt_local = quat_rotate(inverse_note_quats, note_to_hilt)
        tip_local = quat_rotate(inverse_note_quats, note_to_tip)

        repeated_box_verts = box_verts.unsqueeze(-4).repeat_interleave(2, 3)
        repeated_box_quats = box_quats.unsqueeze(-2).unsqueeze(-4).repeat(1, 1, 1, 2, 1, 8, 1)
        inverse_box_quats = quat_inverse(repeated_box_quats)
        repeated_box_positions = repeated_box_verts.mean(-2, keepdim=True)
        unrotated_box_verts = quat_rotate(inverse_box_quats, repeated_box_verts - repeated_box_positions)

        hilt_to_tip = tip_local - hilt_local
        lerp_shape = [1] * (len(tip_positions.shape) + 1)
        lerp_shape[-2] = -1
        sampled_tip_positions = hilt_local.unsqueeze(-2) + hilt_to_tip.unsqueeze(-2) * torch.linspace(0, 1, 5, device=device).view(lerp_shape)
        sampled_tip_start = sampled_tip_positions[:, :, :, :, 0]
        sampled_tip_end = sampled_tip_positions[:, :, :, :, 1]
        sampled_tip_delta = sampled_tip_end - sampled_tip_start

        box_mins = unrotated_box_verts.min(-2)[0]
        box_maxs = unrotated_box_verts.max(-2)[0]

        slab_entries = (box_mins.unsqueeze(-2) - sampled_tip_start) / sampled_tip_delta
        slab_exits = (box_maxs.unsqueeze(-2) - sampled_tip_start) / sampled_tip_delta
        near_hits = torch.minimum(slab_entries, slab_exits)
        far_hits = torch.maximum(slab_entries, slab_exits)
        entry_time = near_hits.max(-1)[0]
        exit_time = far_hits.min(-1)[0]

        ray_collision_mask = entry_time < exit_time
        in_segment_mask = ((entry_time > 0) & (entry_time < 1)) | ((exit_time > 0) & (exit_time < 1))
        saber_box_collision_mask = torch.any(ray_collision_mask & in_segment_mask, dim=-1)

        interior_collision_mask = (
            (sampled_tip_end > box_mins.unsqueeze(-2)) & (sampled_tip_end < box_maxs.unsqueeze(-2))
        ).all(-1).any(-1)
        return saber_box_collision_mask | interior_collision_mask

    @staticmethod
    def get_collision_masks(
        motion: PlayerMotion,
        replay: ReplayTensors,
        note_jump_speed: float,
        map_object_ids: MapObjectIds,
    ) -> SimulationMasks:
        assert replay.note_ids is not None
        assert replay.bomb_ids is not None
        note_bags = replay.notes
        bomb_bags = replay.bombs
        obstacle_bags = replay.obstacles
        note_ids = replay.note_ids
        bomb_ids = replay.bomb_ids

        n_songs, n_cands, n_frames = motion.trajectory_3p.shape[:3]
        n_notes = note_bags.shape[-2]
        full_trajectory = torch.cat([motion.carry_3p, motion.trajectory_3p], dim=2)

        good_cut_geometry = TorchSaber.get_note_geometry(note_bags, note_jump_speed, 0)
        bad_cut_geometry = TorchSaber.get_note_geometry(note_bags, note_jump_speed, 2)
        bomb_geometry = TorchSaber.get_note_geometry(bomb_bags, note_jump_speed, 2)
        obstacle_verts, obstacle_normals = TorchSaber.get_obstacle_verts_and_normals(obstacle_bags, note_jump_speed)

        note_values = note_bags.detach().cpu().numpy() * 1
        note_values = np.where(np.isnan(note_values), 0, note_values).astype(int)

        trajectory_xyz = full_trajectory[..., :3] * 1
        trajectory_sixd = full_trajectory[..., 3:] * 1
        trajectory_quat = sixd_to_quat(trajectory_sixd)
        trajectory_xyz, trajectory_quat = unity_to_zup(trajectory_xyz, trajectory_quat)
        saber_xyz = trajectory_xyz[..., [1, 2], :]
        saber_quat = trajectory_quat[..., [1, 2], :]

        good_cut_collision = TorchSaber.box_trail_collision_from_verts(
            good_cut_geometry,
            saber_xyz,
            saber_quat,
        )
        bad_cut_collision = TorchSaber.box_trail_collision_from_verts(
            bad_cut_geometry,
            saber_xyz,
            saber_quat,
        )
        bomb_collision = TorchSaber.box_trail_collision_from_verts(
            bomb_geometry,
            saber_xyz,
            saber_quat,
        )

        note_colors = note_values[..., -3]
        color_onehots = torch.eye(2, dtype=torch.bool, device=device)[note_colors].swapaxes(-2, -1)
        matching_color_collision = good_cut_collision & color_onehots
        wrong_color_collision = bad_cut_collision & ~color_onehots

        saber_tip_offset = torch.tensor([[[[[1.0, 0, 0]]]]], dtype=torch.float, device=device).repeat(n_songs, n_cands, n_frames + 1, 2, 1)
        saber_tip_offset = quat_rotate(saber_quat.contiguous(), saber_tip_offset)
        saber_tip_positions = saber_xyz + saber_tip_offset

        if saber_tip_positions.shape[2] > 1:
            offset_velocities = np.gradient(saber_tip_positions.detach().cpu().numpy(), axis=2) * 60
            offset_velocities = gaussian_filter1d(offset_velocities, 2, axis=2, mode="nearest")
            offset_velocities = torch.as_tensor(offset_velocities, device=saber_tip_positions.device)[:, :, 1:]
        else:
            offset_velocities = saber_tip_positions * 0
        offset_velocities[..., 0] = 0
        normalized_offset_velocities = offset_velocities / (offset_velocities.norm(dim=-1, keepdim=True) + 1e-10)

        cut_direction_vectors = torch.tensor([[[[[0, 0, 1]]]]], dtype=torch.float, device=device).repeat(
            n_songs,
            n_cands,
            n_frames,
            n_notes,
            1,
        )
        cut_direction_vectors = quat_rotate(good_cut_geometry.quats, cut_direction_vectors)
        one_zero_zero = torch.zeros_like(cut_direction_vectors, device=device)
        one_zero_zero[..., 0] = 1
        zero_one_zero = torch.zeros_like(cut_direction_vectors, device=device)
        zero_one_zero[..., 1] = 1

        cut_basis = torch.stack([one_zero_zero, cut_direction_vectors], dim=-2)
        hilt_tip_projection = torch.einsum("...ab,...b->...a", cut_basis.unsqueeze(-4), saber_tip_offset[:, :, 1:].unsqueeze(-2))
        hilt_tip_projection = torch.cat([hilt_tip_projection, torch.zeros_like(hilt_tip_projection[..., [0]])], dim=-1)
        cosine_values = cosine_similarity(one_zero_zero[:, :, :, None], hilt_tip_projection, dim=-1)
        cut_angles = torch.sign(torch.sum(zero_one_zero[:, :, :, None] * hilt_tip_projection, dim=-1))
        cut_angles = cut_angles * torch.acos(cosine_values) * 180 / np.pi
        cut_angles = cut_angles.to(torch.float16)
        cut_angles = torch.take_along_dim(cut_angles, torch.as_tensor(note_colors[..., None, :], device=device), dim=-2)[..., 0, :]

        velocity_alignment = torch.sum(
            cut_direction_vectors.unsqueeze(-3) * normalized_offset_velocities.unsqueeze(-2),
            dim=-1,
        )
        good_direction = (velocity_alignment > 0.0) | (note_bags[..., [-2, -2]].swapaxes(-1, -2) == 8)
        bad_direction = ~good_direction

        bad_cut_across_time = bad_cut_collision & (wrong_color_collision | bad_direction)
        good_cut_across_time = good_cut_collision & matching_color_collision & good_direction

        note_appeared_global_to_local = map_object_ids.notes[None, None, None, :, None] == note_ids[..., None, :]
        note_appeared_local_to_global = note_appeared_global_to_local.swapaxes(-2, -1)
        note_appeared_mask = note_appeared_global_to_local.any(-1).detach().cpu()

        bomb_appeared_mask = (map_object_ids.bombs[None, None, None, :, None] == bomb_ids[..., None, :]).any(-1).detach().cpu()
        bomb_collided_ids = torch.where(bomb_collision.any(-2), bomb_ids, torch.nan)
        bomb_collided_mask = (map_object_ids.bombs[None, None, None, :, None] == bomb_collided_ids[..., None, :]).any(-1).detach().cpu()

        head_xyz = trajectory_xyz[:, :, 1:, [0]]
        obstacle_mins = obstacle_verts.min(-2)[0]
        obstacle_maxs = obstacle_verts.max(-2)[0]
        head_inside_obstacle = torch.all((head_xyz > obstacle_mins) & (head_xyz < obstacle_maxs), dim=-1)
        obstacle_centers = obstacle_verts.mean(-2)
        obstacle_to_head = head_xyz - obstacle_centers
        head_obstacle_dots = torch.sum((obstacle_to_head[..., None, :] * obstacle_normals)[..., [1, 2]], dim=-1)
        head_obstacle_distances = head_obstacle_dots.max(-1)[0]
        head_obstacle_signed_distances = head_obstacle_distances * torch.where(head_inside_obstacle, 0, 1)

        good_cut_note_ids = torch.where(good_cut_across_time, note_ids.unsqueeze(-2), torch.nan)
        good_cut_mask = (map_object_ids.notes[None, None, None, :, None] == good_cut_note_ids[..., None, :]).any(-1).detach().cpu()
        bad_cut_note_ids = torch.where(bad_cut_across_time, note_ids.unsqueeze(-2), torch.nan)
        bad_cut_mask = (map_object_ids.notes[None, None, None, :, None] == bad_cut_note_ids[..., None, :]).any(-1).detach().cpu()

        note_cut_angles = torch.full(note_appeared_mask.shape, torch.nan, dtype=torch.float16, device=device)
        note_cut_angles[torch.where(note_appeared_global_to_local)[:-1]] = cut_angles[torch.where(note_appeared_local_to_global)[:-1]]

        return SimulationMasks(
            notes=NoteCollisionMasks(
                appeared=note_appeared_mask,
                good_cut=good_cut_mask,
                bad_cut=bad_cut_mask,
                cut_angles=note_cut_angles.detach().cpu(),
            ),
            bomb_appeared=bomb_appeared_mask,
            bomb_collided=bomb_collided_mask,
            head_obstacle_distances=head_obstacle_signed_distances,
        )
