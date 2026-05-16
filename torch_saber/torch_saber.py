from functools import reduce

import torch
import numpy as np
import pyvista as pv
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
from torch import cosine_similarity
from tqdm import tqdm

from beaty_common.pose_utils import sixd_to_quat, slerp, interpolate_xyzsixd
from poselib.poselib import quat_inverse
from torch_saber.utils.pose_utils import unity_to_zup, quat_rotate, expm_to_quat_torch

torch.set_printoptions(precision=3, sci_mode=False)

plane_width = 1.8
plane_height = 1.05
plane_left = 0 - plane_width / 2
plane_bottom = 0
plane_right = 0 + plane_width / 2
plane_top = plane_height
# note_speed_mult = 20
note_x_offset = 0.9
note_y_offset = 0.0
note_z_offset = 0.0

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def nanmin_(tensor, dim=None, keepdim=False):
    max_value = torch.finfo(tensor.dtype).max
    allnan = tensor.isnan().all(dim=dim)
    tensor = tensor.nan_to_num_(
        max_value,
    ).min(
        dim=dim, keepdim=keepdim
    )[0]
    tensor[allnan] = torch.nan
    return tensor


def nanmax_(tensor, dim=None, keepdim=False):
    min_value = torch.finfo(tensor.dtype).min
    allnan = tensor.isnan().all(dim=dim)
    tensor = tensor.nan_to_num_(min_value).max(dim=dim, keepdim=keepdim)[0]
    tensor[allnan] = torch.nan
    return tensor


class TorchSaber:

    @staticmethod
    def evaluate_and_simulate(
        carry_3p: torch.Tensor,
        my_3p_traj: torch.Tensor,
        note_bags: torch.Tensor,
        bomb_bags: torch.Tensor,
        obstacle_bags: torch.Tensor,
        note_ids: torch.Tensor,
        bomb_ids: torch.Tensor,
        obstacle_ids: torch.Tensor,
        note_profiles: torch.Tensor,
        bomb_profiles: torch.Tensor,
        obstacle_profiles: torch.Tensor,
        note_alive_mask: torch.Tensor,
        player_height: float,
        njs: float,
        batch_size: int = None,
    ):
        """
        Given a batch of 3p trajectories (normalized to match PHC's height) and the sequence of note bags,
        Evaluate the f1 score of the trajectory, assuming the dimensions of the hitboxes.
        Many simplifying assumptions are made here, but as a baseline this should be fine.

        my_3p_traj has shape (n_songs, n_cands, n_frames, 27)
        where the 6-dim feature is xyz and expm
        note_bags has shape (B, T, 20, 5)
        where the 5-dim feature is time, x, y, color, angle
        """
        (
            note_appeared_yes_mask,
            bomb_appeared_yes_mask,
            obstacle_appeared_mask,
            bomb_collided_yes_mask,
            head_obstacle_signed_dists,
            gc_yes_mask,
            bc_yes_mask,
            ms_yes_mask,
            offset_vels,
        ) = TorchSaber.simulate(
            carry_3p,
            my_3p_traj,
            note_bags,
            bomb_bags,
            obstacle_bags,
            note_ids,
            bomb_ids,
            obstacle_ids,
            note_profiles,
            bomb_profiles,
            obstacle_profiles,
            player_height,
            njs,
            batch_size,
        )

        (
            normalized_score,
            n_opportunities,
            n_hits,
            n_misses,
            n_goods,
            collided_note_masks,
            bomb_penalty,
            obstacle_penalty,
            scores_at_collision,
            final_good_yes,
        ) = TorchSaber.evaluate(
            note_alive_mask.to(device),
            note_appeared_yes_mask.to(device),
            bomb_appeared_yes_mask.to(device),
            bomb_collided_yes_mask.to(device),
            head_obstacle_signed_dists.to(device),
            gc_yes_mask.to(device),
            bc_yes_mask.to(device),
            offset_vels.to(device),
        )
        return normalized_score, n_opportunities, n_hits, n_misses, n_goods, collided_note_masks, bomb_penalty, obstacle_penalty

    @staticmethod
    def evaluate(
        note_alive_mask,
        note_appeared_yes_mask,
        bomb_appeared_yes_mask,
        bomb_collided_yes_mask,
        head_obstacle_signed_dists,
        gc_yes_mask,
        bc_yes_mask,
        degs,
        batch_size: int = None,
    ):
        max_value = torch.iinfo(torch.int8).max
        min_value = torch.iinfo(torch.int8).min

        idxs = torch.arange(note_alive_mask.shape[2], device=device)
        batch_idxs = torch.split(idxs, batch_size if batch_size is not None else note_alive_mask.shape[2])

        # hit_yes_mask = gc_yes_mask | bc_yes_mask
        # g = gc_yes_mask.any(-2)
        # b = bc_yes_mask.any(-2)
        # good_cumsum = g.cumsum(2, dtype=torch.uint8)
        # bad_cumsum = b.cumsum(2, dtype=torch.uint8)
        # # g = torch.where(g.any(2), g.to(torch.uint8).argmax(2), torch.inf)
        # # b = torch.where(g.any(2), b.to(torch.uint8).argmax(2), torch.inf)
        # # hit_and_good_mask = g < b
        # # hit_and_good_mask = g <= b
        #
        n_opportunities = (note_alive_mask & note_appeared_yes_mask.any(-2)).sum(-1)
        n_hits = 0
        n_goods = 0
        # hit_and_good_mask = ((good_cumsum > 0) & ~(bad_cumsum > 0)).cumsum(2) > 0
        # # hit_and_good_mask = hit_and_good_mask | ((gc_yes_mask.any(-2) & bc_yes_mask.any(-2)).cumsum(2) > 0)
        # hit_and_good_mask = hit_and_good_mask & note_alive_mask[..., None, :]
        # final_good_yes = hit_and_good_mask[:, :, -1]
        # hit_and_good_first_mask = hit_and_good_mask.cumsum(2) == 1

        shape = list(degs.shape[:3])
        shape += [1]
        arange_res = torch.arange(degs.shape[2], device=device)[None, None, :, None]
        res = []
        pre_window_size = 24
        post_window_size = 24
        total_score = 0
        for batch_i in batch_idxs:
            hit_yes_mask = gc_yes_mask[..., batch_i] | bc_yes_mask[..., batch_i]
            g = gc_yes_mask[..., batch_i].any(-2)
            b = bc_yes_mask[..., batch_i].any(-2)
            good_cumsum = g.cumsum(2, dtype=torch.uint8)
            bad_cumsum = b.cumsum(2, dtype=torch.uint8)

            hit_and_good_mask = ((good_cumsum > 0) & ~(bad_cumsum > 0)).cumsum(2) > 0
            hit_and_good_mask = hit_and_good_mask & note_alive_mask[..., None, batch_i]
            final_good_yes = hit_and_good_mask[:, :, -1]
            hit_and_good_first_mask = hit_and_good_mask.cumsum(2) == 1

            torch.cuda.empty_cache()
            batch_degs = degs[..., batch_i] * 1
            idxs = torch.arange(pre_window_size, device=device)[None, None].repeat(shape)
            idxs += arange_res
            idxs -= pre_window_size
            idxs = idxs.clamp(0, degs.shape[2] - 1)
            # degs_for_min = degs.nan_to_num(max_value).to(torch.int8)
            degs_for_min = batch_degs.nan_to_num(max_value).to(torch.int8)
            relevant_degs = torch.take_along_dim(degs_for_min.unsqueeze(-2), idxs.unsqueeze(-1), -3)
            min_degs = relevant_degs.min(-2)[0]
            del relevant_degs
            torch.cuda.empty_cache()
            min_degs = torch.where(min_degs == max_value, torch.nan, min_degs).to(torch.float)
            pre_score = torch.clamp(min_degs, min=-100) / -100
            pre_score = torch.clamp(pre_score, max=1.0, min=0.0)

            idxs = torch.arange(post_window_size + 1, device=device)[None, None].repeat(shape)
            idxs += arange_res
            idxs = idxs.clamp(0, degs.shape[2] - 1)
            degs_for_max = batch_degs.nan_to_num(min_value).to(torch.int8)
            relevant_degs = torch.take_along_dim(degs_for_max.unsqueeze(-2), idxs.unsqueeze(-1), -3)
            max_degs = relevant_degs.max(-2)[0]
            del relevant_degs
            torch.cuda.empty_cache()
            max_degs = torch.where(max_degs == min_value, torch.nan, max_degs).to(torch.float)
            post_score = torch.clamp(max_degs, max=60) / 60
            post_score = torch.clamp(post_score, max=1.0, min=0.0)

            swing_score = 0.7 * pre_score + 0.3 * post_score  # NOTE: accuracy score is not included in the current version
            swing_score = torch.where(hit_and_good_first_mask, swing_score, torch.nan)
            score = swing_score.nansum(-1).nansum(-1).nan_to_num(0)
            total_score += score
            n_hits += hit_yes_mask.any(-2).any(2).sum(-1)
            n_goods += final_good_yes.sum(-1)

            res.append(swing_score)

        # swing_score = torch.cat(res, dim=-1)
        # swing_score = torch.where(hit_and_good_first_mask, swing_score, torch.nan)
        # normalized_score = swing_score.nansum(-1).nansum(-1).nan_to_num(0) / n_opportunities
        # safe division: if total_score or n_opportunities is 0, then return 0
        denominator = torch.where(n_opportunities == 0, torch.ones_like(n_opportunities), n_opportunities)
        # print(f"{denominator=}")
        normalized_score = total_score / denominator

        # n_goods = torch.sum(final_good_yes, dim=-1)
        n_misses = n_opportunities - n_hits
        n_bomb_opportunities = bomb_appeared_yes_mask.sum(2).sum(-1)
        n_bomb_hits = bomb_collided_yes_mask.sum(2).sum(-1)
        bomb_penalty = n_bomb_hits / (n_bomb_opportunities + 1e-7)
        obstacle_penalty = head_obstacle_signed_dists.nanmean(-1).nanmean(-1).nan_to_num(10)
        obstacle_penalty[obstacle_penalty < 0] = -torch.inf

        return (
            normalized_score,
            n_opportunities,
            n_hits,
            n_misses,
            n_goods,
            hit_yes_mask.any(-2).any(-2),
            bomb_penalty,
            obstacle_penalty,
            swing_score,
            final_good_yes,
        )

    @staticmethod
    def simulate(
        carry_3p: torch.Tensor,
        my_3p_traj: torch.Tensor,
        note_bags: torch.Tensor,
        bomb_bags: torch.Tensor,
        obstacle_bags: torch.Tensor,
        note_ids: torch.Tensor,
        bomb_ids: torch.Tensor,
        obstacle_ids: torch.Tensor,
        note_profiles: torch.Tensor,
        bomb_profiles: torch.Tensor,
        obstacle_profiles: torch.Tensor,
        player_height: float,
        njs: float,
        batch_size: int = None,
    ):
        idxs = torch.arange(my_3p_traj.shape[2], device=device)
        batch_idxs = torch.split(idxs, batch_size if batch_size is not None else my_3p_traj.shape[2])

        batch_reses = []
        n_notes = note_profiles.shape[1]
        n_bombs = bomb_profiles.shape[1]
        n_obstacles = obstacle_profiles.shape[1]
        unique_note_ids = torch.arange(n_notes, device=device)
        unique_bomb_ids = torch.arange(n_bombs, device=device)
        unique_obstacle_ids = torch.arange(n_obstacles, device=device)
        for batch_i in batch_idxs:
            cpu_res = []
            res = TorchSaber.get_collision_masks(
                carry_3p,
                my_3p_traj[:, :, batch_i],
                note_bags[:, :, batch_i],
                bomb_bags[:, :, batch_i],
                obstacle_bags[:, :, batch_i],
                note_ids[:, :, batch_i],
                bomb_ids[:, :, batch_i],
                obstacle_ids[:, :, batch_i],
                player_height,
                unique_note_ids,
                unique_bomb_ids,
                unique_obstacle_ids,
                njs,
            )
            for i, v in enumerate(res):
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu()
            #     cpu_res.append(v)
            batch_reses.append(res)
            # torch.cuda.empty_cache()
            carry_3p = my_3p_traj[:, :, batch_i[[-1]]]
        (
            note_appeared_yes_mask,
            bomb_appeared_yes_mask,
            obstacle_appeared_mask,
            bomb_collided_yes_mask,
            head_obstacle_signed_dists,
            gc_yes_mask,
            bc_yes_mask,
            ms_yes_mask,
            offset_vels,
        ) = list(reduce(lambda acc, res: [torch.cat([a, r], dim=2) for a, r in zip(acc, res)], batch_reses))
        return (
            note_appeared_yes_mask,
            bomb_appeared_yes_mask,
            obstacle_appeared_mask,
            bomb_collided_yes_mask,
            head_obstacle_signed_dists,
            gc_yes_mask,
            bc_yes_mask,
            ms_yes_mask,
            offset_vels,
        )

    @staticmethod
    def get_note_verts_and_normals_and_quats(note_bags: torch.Tensor, player_height: float, njs: float, type: int):
        """
        Get the vertices and face normals of notes in the bag post transform
        """
        n_songs, n_cands, n_frames, n_notes, *rest = note_bags.shape
        # https://github.com/MetaGuard/SimSaber/tree/main
        if type == 0:  # good cut collider
            note_collider_mesh = pv.Cube(x_length=1.0, y_length=0.8, z_length=0.5)
            # note_collider_mesh = pv.Cube(x_length=0.4, y_length=0.4, z_length=0.4)
        elif type == 1:  # dot collider
            note_collider_mesh = pv.Cube(x_length=1.0, y_length=0.8, z_length=0.8)
        else:
            note_collider_mesh = pv.Cube(x_length=0.4, y_length=0.4, z_length=0.4)

        note_collider_verts = torch.tensor(note_collider_mesh.points, dtype=torch.float, device=device)
        note_verts = note_collider_verts[None, None, None, None].repeat(n_songs, n_cands, n_frames, n_notes, 1, 1)

        note_angle_degrees = np.array([0, 180, -90, 90, -45, 45, -135, 135, 0])
        tmp = note_bags.detach().cpu().numpy() * 1
        tmp = np.where(np.isnan(tmp), 0, tmp)
        tmp = tmp.astype(int)
        note_angles = np.where(
            np.isnan(note_bags[..., -2].detach().cpu().numpy()),
            np.nan,
            note_angle_degrees[tmp[..., -2]],
        )
        note_quat = Rotation.from_euler("x", note_angles.reshape(-1), degrees=True).as_quat().reshape((n_songs, n_cands, n_frames, n_notes, 4))
        note_quat = torch.tensor(note_quat, dtype=torch.float, device=device)
        note_verts = quat_rotate(
            note_quat[..., None, :].repeat_interleave(8, -2),
            note_verts,
        )

        note_collider_normals = torch.tensor(note_collider_mesh.face_normals, dtype=torch.float, device=device)
        note_face_normals = note_collider_normals[None, None, None, None].repeat(n_songs, n_cands, n_frames, n_notes, 1, 1)
        note_face_normals = quat_rotate(
            note_quat[..., None, :].repeat_interleave(6, -2),
            note_face_normals,
        )

        plane_grid = np.array(
            np.meshgrid(
                np.linspace(plane_right, plane_left, 4),  # note: left and right are flipped for unity
                np.linspace(plane_bottom, plane_top, 3),
            )
        ).transpose((2, 1, 0))
        plane_grid = torch.tensor(plane_grid, dtype=torch.float, device=device)

        note_pos = plane_grid[tmp[..., 3], tmp[..., 4]]
        note_pos = torch.concatenate([note_bags[..., [0]] * njs + note_x_offset, note_pos], dim=-1)

        # note_pos[..., 2] += 1.5044 - 0.333
        note_pos[..., 1] += note_y_offset
        note_pos[..., 2] += player_height / 2
        note_pos[..., 2] += note_z_offset
        note_verts += note_pos[..., None, :]
        if type == 0 or type == 1:
            note_verts[..., 0] -= 0.25

        return note_verts, note_face_normals, note_quat

    @staticmethod
    def get_obstacle_verts_and_normals(obstacle_bags: torch.Tensor, player_height: float, njs: float):
        """
        Get the vertices and face normals of notes in the bag post transform
        """
        n_songs, n_cands, n_frames, n_obstacles, *rest = obstacle_bags.shape
        # obstacle_verts = obstacle_collider_verts[None, None, None].repeat(obstacle_bags.shape[0], 0).repeat(obstacle_bags.shape[1], 1).repeat(obstacle_bags.shape[-2], 2)
        # obstacle_verts = torch.tensor(obstacle_verts, dtype=torch.float, device=device)
        tmp = obstacle_bags.detach().cpu().numpy() * 1
        tmp = np.where(np.isnan(tmp), 0, tmp)
        tmp = tmp.astype(int)

        plane_leftright = np.linspace(plane_right, plane_left, 4)  # note: left and right are flipped for unity
        plane_bottomtop = np.linspace(plane_bottom, plane_top, 3)
        plane_width_interval = plane_leftright[0] - plane_leftright[1]
        plane_height_interval = plane_bottomtop[1] - plane_bottomtop[0]

        plane_grid = np.array(
            np.meshgrid(
                plane_leftright,
                plane_bottomtop,
            )
        ).transpose((2, 1, 0))
        plane_grid = torch.tensor(plane_grid, dtype=torch.float, device=device)

        obstacle_collider_mesh = pv.Cube(x_length=0.4, y_length=plane_width_interval, z_length=plane_height_interval)
        obstacle_collider_verts = torch.tensor(obstacle_collider_mesh.points, dtype=torch.float, device=device)
        obstacle_verts = obstacle_collider_verts[None, None, None, None].repeat(n_songs, n_cands, n_frames, n_obstacles, 1, 1)

        # Locate base to the correct place based on x and y
        # obstacle_pos = plane_grid[tmp[..., 5], tmp[..., 6]]
        obstacle_pos = plane_grid[0, 0] + (plane_grid[1, 1] - plane_grid[0, 0]) * torch.as_tensor(tmp[..., [5, 6]], device=device)
        obstacle_pos = torch.concatenate([obstacle_bags[..., [0]] * njs + note_x_offset, obstacle_pos], dim=-1)
        # obstacle_pos[..., 2] += 1.5044 - 0.333
        obstacle_pos[..., 2] += player_height / 2
        obstacle_pos[..., 2] += note_z_offset
        obstacle_pos[..., 1] += note_y_offset
        # obstacle_verts += obstacle_pos[..., None, :]
        obstacle_verts[..., [1, 2]] += obstacle_pos[..., None, [1, 2]]

        # Depth based on duration (index 4)
        obstacle_verts[..., 0] = obstacle_bags[..., [0]] * njs + note_x_offset
        obstacle_verts[..., -4:, 0] += obstacle_bags[..., [4]] * njs + note_x_offset
        # obstacle_verts[..., :-4:, 0] -= 0.3 * njs
        # Top height based on index -1
        # If height is 1 then no offset added. If height is 2, then add 1 * height_interval
        obstacle_verts[..., [1, 2, 6, 7], 2] += (obstacle_bags[..., [-1]] - 1) * plane_height_interval

        # left-right based on index -2
        # If width is 1 then no offset added. If width is 2, then add 1 * width_interval
        # left vertices
        # obstacle_verts[..., [2, 3, 5, 6], 1] = obstacle_verts[..., [0, 1, 4, 7], 1] + (obstacle_bags[..., [-2]] - 0) * np.clip(plane_width_interval, a_min=0, a_max=np.inf)
        # right vertices
        obstacle_verts[..., [0, 1, 4, 7], 1] = obstacle_verts[..., [2, 3, 5, 6], 1] - (obstacle_bags[..., [-2]] - 0) * plane_width_interval

        obstacle_verts[obstacle_verts.isnan().any(-1)] = torch.nan

        obstacle_collider_normals = torch.tensor(obstacle_collider_mesh.face_normals, dtype=torch.float, device=device)
        obstacle_face_normals = obstacle_collider_normals[None, None, None, None].repeat(n_songs, n_cands, n_frames, n_obstacles, 1, 1)
        # obstacle_face_normals = obstacle_face_normals[None, None, None].repeat(obstacle_bags.shape[0], 0).repeat(obstacle_bags.shape[1], 1).repeat(obstacle_bags.shape[-2], 2)
        # obstacle_face_normals = torch.tensor(obstacle_face_normals, dtype=torch.float, device=device)
        obstacle_face_normals[obstacle_verts.isnan().any(-1).any(-1)] = torch.nan

        return obstacle_verts, obstacle_face_normals

    @staticmethod
    def box_box_collision_from_verts_and_normals(verts1: torch.Tensor, normals1: torch.Tensor, verts2: torch.Tensor, normals2: torch.Tensor):
        edge_idxs = pv.Cube().regular_faces.reshape(-1, 2)
        edges1 = verts1[..., edge_idxs[:, 0], :] - verts1[..., edge_idxs[:, 1], :]
        edges2 = verts2[..., edge_idxs[:, 0], :] - verts2[..., edge_idxs[:, 1], :]
        edge_crosses = (
            torch.cross(
                edges1.unsqueeze(-3).unsqueeze(-2),
                edges2.unsqueeze(-4).unsqueeze(-3),
                dim=-1,
            )
            # .permute((0, 1, 2, 4, 3, 5, 6))
            .contiguous()
        )
        edge_crosses = edge_crosses.flatten(-3, -2)
        # The end-all-be-all for collision precompute
        all_cand_axes_across_time = torch.concatenate(
            [
                edge_crosses,
                normals1.unsqueeze(-3).repeat_interleave(normals2.shape[-3], -3),
                normals2.unsqueeze(-4).repeat_interleave(normals1.shape[-3], -4),
            ],
            dim=-2,
        )
        all_cand_axes_across_time /= torch.norm(all_cand_axes_across_time, dim=-1, keepdim=True) + 1e-10

        proj1 = torch.sum(
            verts1.unsqueeze(-3).unsqueeze(-3) * all_cand_axes_across_time.unsqueeze(-2),
            dim=-1,
        )
        proj2 = torch.sum(
            verts2.unsqueeze(-4).unsqueeze(-3) * all_cand_axes_across_time.unsqueeze(-2),
            dim=-1,
        )
        min1 = proj1.min(-1)[0]
        max1 = proj1.max(-1)[0]
        min2 = proj2.min(-1)[0]
        max2 = proj2.max(-1)[0]
        collide_yes = torch.all(max1 >= min2, dim=-1) & torch.all(max2 >= min1, dim=-1)
        return collide_yes

    @staticmethod
    def box_trail_collision_from_verts(box_verts: torch.Tensor, box_quats: torch.Tensor, saber_xyzs: torch.Tensor, saber_quats: torch.Tensor):
        n_songs, n_cands, n_frames, *rest = box_verts.shape
        note_pos = box_verts.mean(-2)

        saber_xyzs_prevcur = torch.stack([saber_xyzs[:, :, :-1], saber_xyzs[:, :, 1:]], -2)
        saber_quats_prevcur = torch.stack([saber_quats[:, :, :-1], saber_quats[:, :, 1:]], -2)

        hilt = saber_xyzs_prevcur
        tip = torch.tensor([[[[[1.0, 0, 0]]]]], dtype=torch.float, device=device).repeat(n_songs, n_cands, n_frames, 2, 2, 1)
        tip = hilt + quat_rotate(saber_quats_prevcur.contiguous(), tip)

        # Apply rotation to project to each note's local coordinate system
        note_to_hilt = hilt.unsqueeze(-2) - note_pos.unsqueeze(-3).unsqueeze(-3)
        note_to_tip = tip.unsqueeze(-2) - note_pos.unsqueeze(-3).unsqueeze(-3)
        note_quat_repped = box_quats.unsqueeze(-3).unsqueeze(-3).repeat(1, 1, 1, 2, 2, 1, 1)
        inv_note_quat_repped = quat_inverse(note_quat_repped)
        hilt_loc = quat_rotate(inv_note_quat_repped, note_to_hilt)
        tip_loc = quat_rotate(inv_note_quat_repped, note_to_tip)

        note_verts_repped = box_verts.unsqueeze(-4).repeat_interleave(2, 3)
        note_quat_repped = box_quats.unsqueeze(-2).unsqueeze(-4).repeat(1, 1, 1, 2, 1, 8, 1)
        inv_note_quat_repped = quat_inverse(note_quat_repped)
        # inv_note_quat_repped = inv_note_quat_repped.unsqueeze(-2).repeat_interleave(8, -2)
        note_pos_repped = note_verts_repped.mean(-2, keepdim=True)
        note_verts_unrotated = quat_rotate(inv_note_quat_repped, note_verts_repped - note_pos_repped)
        # note_verts_unrotated = note_verts_repped

        hilt_tip = tip_loc - hilt_loc
        shape = [1] * (len(tip.shape) + 1)
        shape[-2] = -1
        hilt_tip_lerp = hilt_loc.unsqueeze(-2) + hilt_tip.unsqueeze(-2) * torch.linspace(0, 1, 5, device=device).view(shape)

        # hilt_tip_lerp = torch.cat([hilt_tip_lerp[:, :, [0]], hilt_tip_lerp], dim=2)
        hilt_tip_lerp_0 = hilt_tip_lerp[:, :, :, :, 0]
        hilt_tip_lerp_1 = hilt_tip_lerp[:, :, :, :, 1]
        hilt_tip_lerp_01 = hilt_tip_lerp_1 - hilt_tip_lerp_0

        note_slab_mins = note_verts_unrotated.min(-2)[0]
        note_slab_maxs = note_verts_unrotated.max(-2)[0]

        slab_entries = (note_slab_mins.unsqueeze(-2) - hilt_tip_lerp_0) / hilt_tip_lerp_01
        slab_exits = (note_slab_maxs.unsqueeze(-2) - hilt_tip_lerp_0) / hilt_tip_lerp_01

        earlier = torch.minimum(slab_entries, slab_exits)
        later = torch.maximum(slab_entries, slab_exits)
        tclose = earlier.max(-1)[0]
        tfar = later.min(-1)[0]
        ray_collide_yes = tclose < tfar
        range_yes = torch.logical_or((tclose > 0) & (tclose < 1), (tfar > 0) & (tfar < 1))

        saber_note_collision_yeses = torch.any(ray_collide_yes & range_yes, dim=-1)

        # Additionally, check if the current keypoints are interior points
        interior_yeses = ((hilt_tip_lerp_1 > note_slab_mins.unsqueeze(-2)) & (hilt_tip_lerp_1 < note_slab_maxs.unsqueeze(-2))).all(-1).any(-1)

        return saber_note_collision_yeses | interior_yeses

    @staticmethod
    def box_point_collision_from_verts_and_normals(box_verts: torch.Tensor, box_normals: torch.Tensor, box_quats: torch.Tensor, point_xyzs: torch.Tensor):
        n_songs, n_cands, n_frames, *rest = box_verts.shape
        note_pos = box_verts.mean(-2)
        pass

    @staticmethod
    def get_collision_masks(
        carry_3p: torch.Tensor,
        my_3p_traj: torch.Tensor,
        note_bags: torch.Tensor,
        bomb_bags: torch.Tensor,
        obstacle_bags: torch.Tensor,
        note_ids: torch.Tensor,
        bomb_ids: torch.Tensor,
        obstacle_ids: torch.Tensor,
        player_height: float,
        unique_note_ids: torch.Tensor,
        unique_bomb_ids: torch.Tensor,
        unique_obstacle_ids: torch.Tensor,
        njs: float,
        # bomb_bags: torch.Tensor,
    ):
        # (n_songs, n_cands, n_frames, ...)
        n_songs, n_cands, n_frames, *rest = my_3p_traj.shape
        n_notes = note_bags.shape[-2]
        # n_interp_frames = 1
        my_3p_traj_ = torch.cat([carry_3p, my_3p_traj], dim=2)

        note_gc_verts, _, note_gc_quats = TorchSaber.get_note_verts_and_normals_and_quats(note_bags, player_height, njs, 0)
        note_bc_verts, _, note_bc_quats = TorchSaber.get_note_verts_and_normals_and_quats(note_bags, player_height, njs, 2)
        bomb_verts, bomb_collider_normals_across_time, bomb_quats = TorchSaber.get_note_verts_and_normals_and_quats(bomb_bags, player_height, njs, 2)

        # three_p_verts_across_time, three_p_normals_across_time = TorchSaber.get_3p_verts_and_normals(my_3p_traj)
        obstacle_verts_across_time, obstacle_normals_across_time = TorchSaber.get_obstacle_verts_and_normals(obstacle_bags, player_height, njs)

        # note_pos = note_gc_verts.mean(-2)

        tmp = note_bags.detach().cpu().numpy() * 1
        tmp = np.where(np.isnan(tmp), 0, tmp)
        tmp = tmp.astype(int)

        # saber_note_collision_yeses = TorchSaber.box_box_collision_from_verts_and_normals(saber_collider_verts_across_time, saber_collider_normals_across_time, note_collider_verts_across_time, note_collider_normals_across_time)
        # saber_note_collision_yeses = saber_note_collision_yeses.reshape((n_songs, n_cands, n_frames, n_interp_frames, 2, -1))
        # saber_note_collision_yeses = saber_note_collision_yeses.any(3)

        # Be non-intrusive, implement the segment box collision as a separate thing
        my_3p_xyz, my_3p_sixd = (
            my_3p_traj_[..., :3] * 1,
            my_3p_traj_[..., 3:] * 1,
        )
        my_3p_quat = sixd_to_quat(my_3p_sixd)
        my_3p_xyz, my_3p_quat = unity_to_zup(my_3p_xyz, my_3p_quat)
        saber_xyzs = my_3p_xyz[..., [1, 2], :]
        saber_quats = my_3p_quat[..., [1, 2], :]

        saber_gc_collision_yeses = TorchSaber.box_trail_collision_from_verts(note_gc_verts, note_gc_quats, saber_xyzs, saber_quats)
        saber_bc_collision_yeses = TorchSaber.box_trail_collision_from_verts(note_bc_verts, note_bc_quats, saber_xyzs, saber_quats)
        saber_bomb_collision_yeses = TorchSaber.box_trail_collision_from_verts(bomb_verts, bomb_quats, saber_xyzs, saber_quats)

        # head_obstacle_collision_yes = TorchSaber.box_box_collision_from_verts_and_normals(three_p_verts_across_time, three_p_normals_across_time, obstacle_verts_across_time, obstacle_normals_across_time)
        # head_obstacle_collision_yes = head_obstacle_collision_yes[..., 0, :]

        # Instead of box-box collision, do point-box collision
        # head_obstacle_vert_deltas = (three_p_verts_across_time[..., [0], :, :].unsqueeze(-2).unsqueeze(-2) - obstacle_verts_across_time.unsqueeze(-4).unsqueeze(-4))[:, :, :, 0]
        # # Check if the head is inside the box
        # head_obstacle_vert_delta_dots = torch.sum(head_obstacle_vert_deltas[..., None, :] * obstacle_normals_across_time[..., None, :, :], dim=-1)
        # head_obstacle_collision_yes = torch.all(head_obstacle_vert_delta_dots < 0, dim=(-1, -2))

        colors = tmp[..., -3]
        color_onehots = torch.eye(2, dtype=torch.bool, device=device)[colors].swapaxes(-2, -1)
        color_yes_across_time = saber_gc_collision_yeses & color_onehots
        color_bad_across_time = saber_bc_collision_yeses & ~color_onehots

        tip = torch.tensor([[[[[1.0, 0, 0]]]]], dtype=torch.float, device=device).repeat(n_songs, n_cands, n_frames + 1, 2, 1)
        hilt_tip = quat_rotate(saber_quats.contiguous(), tip)
        tip = saber_xyzs + hilt_tip

        if tip.shape[2] > 1:
            offset_vels = np.gradient(tip.detach().cpu().numpy(), axis=2) * 60
            offset_vels = torch.as_tensor(gaussian_filter1d(offset_vels, 2, axis=2, mode="nearest"), device=tip.device)
            offset_vels = offset_vels[:, :, 1:]
        else:
            offset_vels = tip * 0
        offset_vels[..., 0] = 0
        normalized_offset_vels = offset_vels / (offset_vels.norm(dim=-1, keepdim=True) + 1e-10)

        cut_dir_vecs_across_time = torch.tensor([[[[[0, 0, 1]]]]], dtype=torch.float, device=device).repeat(n_songs, n_cands, n_frames, n_notes, 1)
        cut_dir_vecs_across_time = quat_rotate(note_gc_quats, cut_dir_vecs_across_time)
        one_zero_zero = torch.zeros_like(cut_dir_vecs_across_time, device=device)
        one_zero_zero[..., 0] = 1
        zero_one_zero = torch.zeros_like(cut_dir_vecs_across_time, device=device)
        zero_one_zero[..., 1] = 1
        # up = torch.cross(one_zero_zero, cut_dir_vecs_across_time, dim=-1)
        basis = torch.stack([one_zero_zero, cut_dir_vecs_across_time], dim=-2)
        # up = torch.cross(basis[..., 0, :], basis[..., 1, :], dim=-1)
        projs = torch.einsum("...ab,...b->...a", basis.unsqueeze(-4), hilt_tip[:, :, 1:].unsqueeze(-2))
        projs_with_third_zero = torch.cat([projs, torch.zeros_like(projs[..., [0]])], dim=-1)
        cossims = cosine_similarity(one_zero_zero[:, :, :, None], projs_with_third_zero, dim=-1)
        arccoss = torch.acos(cossims)
        # signs = torch.sign(torch.sum(torch.cross(one_zero_zero[:, :, :, None], projs_with_third_zero, dim=-1) * up[:, :, :, None], dim=-1))
        signs = torch.sign(torch.sum(zero_one_zero[:, :, :, None] * projs_with_third_zero, dim=-1))
        degs = signs * arccoss * 180 / np.pi
        # degs = arccoss * 180 / np.pi
        degs = degs.to(torch.float16)
        degs = torch.take_along_dim(degs, torch.as_tensor(colors[..., None, :], device=device), dim=-2)[..., 0, :]

        dots_across_time = torch.sum(cut_dir_vecs_across_time.unsqueeze(-3) * normalized_offset_vels.unsqueeze(-2), dim=-1)
        direction_yes_across_time = torch.logical_or(dots_across_time > 0.0, note_bags[..., [-2, -2]].swapaxes(-1, -2) == 8)
        # direction_bad_across_time = torch.logical_and(dots_across_time <= 0.0, note_bags[..., [-2, -2]].swapaxes(-1, -2) != 8)
        direction_bad_across_time = ~direction_yes_across_time

        bc_yes_across_time = saber_bc_collision_yeses & (color_bad_across_time | direction_bad_across_time)
        gc_yes_across_time = saber_gc_collision_yeses & color_yes_across_time & direction_yes_across_time
        # break tie between the two

        ms_yes_across_time = ~(bc_yes_across_time | gc_yes_across_time)

        note_appeared_global_to_local = unique_note_ids[None, None, None, :, None] == note_ids[..., None, :]
        note_appeared_local_to_global = note_appeared_global_to_local.swapaxes(-2, -1)
        note_appeared_yes_mask = note_appeared_global_to_local.any(-1).detach().cpu()

        bomb_appeared_yes_mask = (unique_bomb_ids[None, None, None, :, None] == bomb_ids[..., None, :]).any(-1).detach().cpu()
        obstacle_appeared_global_to_local = unique_obstacle_ids[None, None, None, :, None] == obstacle_ids[..., None, :]
        obstacle_appeared_yes_mask = obstacle_appeared_global_to_local.any(-1).detach().cpu()

        # Collect note destruction masks (this is one of the most genius things I've ever done)

        bomb_collided_idx = torch.where(saber_bomb_collision_yeses.any(-2), bomb_ids, torch.nan)
        bomb_collided_yes_mask = (unique_bomb_ids[None, None, None, :, None] == bomb_collided_idx[..., None, :]).any(-1).detach().cpu()
        # print(bomb_collided_yes_mask)

        # Compute signed distance between obstacles and head
        head_xyzs = my_3p_xyz[:, :, 1:, [0]]
        obst_mins = obstacle_verts_across_time.min(-2)[0]
        obst_maxs = obstacle_verts_across_time.max(-2)[0]
        head_obst_yes = torch.all((head_xyzs > obst_mins) & (head_xyzs < obst_maxs), dim=-1)

        obst_xyzs = obstacle_verts_across_time.mean(-2)
        obst_to_head = head_xyzs - obst_xyzs
        head_obst_dots_yz = torch.sum((obst_to_head[..., None, :] * obstacle_normals_across_time)[..., [1, 2]], dim=-1)
        head_obst_dist_yz = head_obst_dots_yz.max(-1)[0]

        # head_yzs = my_3p_xyz[:, :, 1:, [0], [1, 2]]
        # This is not the correct box-point distance
        # head_yz_obstacle_min_max_dists = torch.abs(head_yzs[..., None, None, :] - obstacle_verts_across_time[..., [1, 2]])[..., [0]].min(-1)[0].min(-1)[0]
        # This is currently incorrect; I need to instead compute dot product wrt normals...

        # head_yz_obstacle_min_max_dists = head_obstacle_vert_deltas[..., [1, 2]].abs().min(-1)[0].min(-1)[0].min(-2)[0]
        # head_yz_obstacle_min_max_dists = head_obstacle_vert_deltas[..., [2]].abs().min(-1)[0].min(-1)[0].min(-2)[0]
        # collided_local_to_global_where = torch.where(collided_global_to_local)
        # head_obstacle_signed_dists = torch.full(head_obst_yes.shape, torch.nan, dtype=torch.float, device=device)
        # head_obstacle_signed_dists[collided_global_to_local
        # a = torch.where(obstacle_appeared_global_to_local)[:-1]
        # b = torch.where(obstacle_appeared_local_to_global)[:-1]
        # head_obstacle_signed_dists[a] = head_obst_dist_yz[b]
        # head_obstacle_signed_dists *= torch.where(head_obst_yes, 0, 1)
        # head_obstacle_signed_dists = torch.where(head_obstacle_signed_dists.isnan(), torch.zeros_like(head_obstacle_signed_dists), head_obstacle_signed_dists)
        head_obstacle_signed_dists = head_obst_dist_yz * torch.where(head_obst_yes, 0, 1)

        gc_note_ids = torch.where(gc_yes_across_time, note_ids.unsqueeze(-2), torch.nan)
        gc_yes_masks = (unique_note_ids[None, None, None, :, None] == gc_note_ids[..., None, :]).any(-1).detach().cpu()
        bc_note_ids = torch.where(bc_yes_across_time, note_ids.unsqueeze(-2), torch.nan)
        bc_yes_masks = (unique_note_ids[None, None, None, :, None] == bc_note_ids[..., None, :]).any(-1).detach().cpu()
        ms_note_ids = torch.where(ms_yes_across_time, note_ids.unsqueeze(-2), torch.nan)
        ms_yes_masks = (unique_note_ids[None, None, None, :, None] == ms_note_ids[..., None, :]).any(-1).detach().cpu()

        my_degs = torch.full(note_appeared_yes_mask.shape, torch.nan, dtype=torch.float16, device=device)
        my_degs[torch.where(note_appeared_global_to_local)[:-1]] = degs[torch.where(note_appeared_local_to_global)[:-1]]
        my_degs = my_degs.detach().cpu()

        return (
            note_appeared_yes_mask,
            bomb_appeared_yes_mask,
            obstacle_appeared_yes_mask,
            bomb_collided_yes_mask,
            head_obstacle_signed_dists,
            gc_yes_masks,
            bc_yes_masks,
            ms_yes_masks,
            my_degs,
        )

    @staticmethod
    def get_saber_verts_and_normals_and_quats(my_3p_traj: torch.Tensor):
        n_songs, n_cands, n_frames, *rest = my_3p_traj.shape
        my_3p_traj = my_3p_traj.reshape((n_songs, n_cands, n_frames, 3, 9))
        # my_3p_xyz, my_3p_expm = (
        my_3p_xyz, my_3p_sixd = (
            my_3p_traj[..., :3] * 1,
            my_3p_traj[..., 3:] * 1,
        )
        # my_3p_quat = expm_to_quat_torch(my_3p_expm)
        my_3p_quat = sixd_to_quat(my_3p_sixd)
        my_3p_xyz, my_3p_quat = unity_to_zup(my_3p_xyz, my_3p_quat)

        # Pre-compute collision stuff using vertices
        saber_collider_mesh = pv.Cube(x_length=1.2, y_length=0.2, z_length=0.2)
        saber_collider_verts = torch.tensor(saber_collider_mesh.points, dtype=torch.float, device=device)
        saber_collider_verts_across_time = saber_collider_verts[None, None, None, None].repeat(n_songs, n_cands, n_frames, 2, 1, 1)
        saber_quats = my_3p_quat[..., 1:, :]
        saber_xyzs = my_3p_xyz[..., 1:, :]

        saber_collider_verts_across_time += torch.tensor([0.5, 0, 0], dtype=torch.float, device=device)[None, None, None, None]
        saber_collider_verts_across_time = quat_rotate(
            saber_quats[..., None, :].repeat_interleave(8, -2),
            saber_collider_verts_across_time,
        )
        saber_collider_verts_across_time += saber_xyzs[..., None, :]

        saber_collider_normals = torch.tensor(saber_collider_mesh.face_normals, dtype=torch.float, device=device)
        saber_collider_normals_across_time = saber_collider_normals[None, None, None, None].repeat(n_songs, n_cands, n_frames, 2, 1, 1)
        saber_collider_normals_across_time = quat_rotate(
            saber_quats[..., None, :].repeat_interleave(6, -2),
            saber_collider_normals_across_time,
        )

        return (
            saber_collider_verts_across_time,
            saber_collider_normals_across_time,
            saber_quats,
        )

    @staticmethod
    def get_3p_verts_and_normals(my_3p_traj: torch.Tensor):
        n_songs, n_cands, n_frames, *rest = my_3p_traj.shape
        collider_mesh = pv.Cube(x_length=0.75, y_length=0.75, z_length=0.75)
        collider_verts = torch.tensor(collider_mesh.points, dtype=torch.float, device=device)
        collider_verts_across_time = collider_verts[None, None, None, None].repeat(n_songs, n_cands, n_frames, 3, 1, 1)
        # collider_verts_across_time = collider_verts[None, None, None].repeat(my_3p_traj.shape[0], 0).repeat(my_3p_traj.shape[1], 1).repeat(3, 2)
        # collider_verts_across_time = torch.tensor(collider_verts_across_time, dtype=torch.float, device=device)
        my_3p_xyz, my_3p_sixd = (
            my_3p_traj[..., :3] * 1,
            my_3p_traj[..., 3:] * 1,
        )
        my_3p_quat = sixd_to_quat(my_3p_sixd)
        my_3p_xyz, my_3p_quat = unity_to_zup(my_3p_xyz, my_3p_quat)
        collider_verts_across_time += my_3p_xyz[..., None, :]
        collider_mesh_normals = torch.tensor(collider_mesh.face_normals, dtype=torch.float, device=device)
        collider_normals_across_time = collider_mesh_normals[None, None, None, None].repeat(n_songs, n_cands, n_frames, 3, 1, 1)

        return collider_verts_across_time, collider_normals_across_time
