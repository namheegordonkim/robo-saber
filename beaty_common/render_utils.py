import imageio
import numpy as np
import pyvista as pv
import torch
from datasets import Dataset
from pyvista import Plotter
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from beaty_common.bsmg_xror_utils import open_beatmap_from_unpacked_xror, load_cbo_and_3p, device, get_cbo_np
from beaty_common.data_utils import SegmentSampler
from beaty_common.pose_utils import sixd_to_quat, unity_to_zup
from beaty_common.train_utils import nanpad_collate_fn

from proj_utils.dirs import proj_dir
from torch_saber import TorchSaber
from torch_saber.viz.visual_data import GenericVisual, NoteVisual, ObstacleVisual
from xror.xror import XROR


# def evaluate_boxrr_row(arrow_files, x, left_handed=False):
#     xror_unpacked = get_xror_for_row(arrow_files, x)
#     return evaluate_row_against_accompanying_map(xror_unpacked, left_handed=left_handed)


def get_xror_for_row(arrow_files, x):
    shard_idx = x["Shard Index"]
    datapoint_idx = x["Datapoint Index"]
    shard_file = arrow_files[shard_idx]
    ds = Dataset.from_file(shard_file)
    xror_unpacked = XROR.unpack(ds[datapoint_idx]["xror"])
    return xror_unpacked


def valuate_row_against_accompanying_map(arrow_files, x, left_handed=False):
    xror_unpacked = get_xror_for_row(arrow_files, x)
    beatmap, map_info = open_beatmap_from_unpacked_xror(xror_unpacked)
    return evaluate_xror_on_map(xror_unpacked, beatmap, map_info, left_handed=left_handed)


def evaluate_3p_on_map(_3p, difficulty, characteristic, beatmap, map_info, song_duration, left_handed=False):
    segment_sampler = SegmentSampler()
    notes_np, bombs_np, obstacles_np = get_cbo_np(beatmap, map_info)
    # song_duration = notes_np[:, 0].max()
    timestamps = np.arange(0, song_duration, 1 / 60)
    d = {
        "notes_np": notes_np,
        "bombs_np": bombs_np,
        "obstacles_np": obstacles_np,
        "timestamps": timestamps,
    }
    d = nanpad_collate_fn([[d]])
    length = d["timestamps"].shape[1]
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            d[k] = v.to(device=device)
    game_segments = segment_sampler.sample_for_evaluation(
        d["notes_np"],
        d["bombs_np"],
        d["obstacles_np"],
        d["timestamps"],
        d["lengths"],
        length,
        1,
        512,
        1,
        4.0,
        80,
        -0.5,
    )
    # Get NJS
    njs = get_njs(map_info, difficulty, characteristic)
    note_alive_mask = torch.where(torch.isnan(d["notes_np"][..., 0]), torch.as_tensor(False), torch.as_tensor(True))[..., None, :]
    (
        note_appeared_yes_mask,
        bomb_appeared_yes_mask,
        obstacle_appeared_mask,
        bomb_collided_yes_mask,
        obstacle_collided_yes_mask,
        gc_mask,
        bc_mask,
        ms_mask,
        offset_vels,
    ) = TorchSaber.simulate(
        _3p[:, :, [0]],
        _3p,
        game_segments.notes[:, None],
        game_segments.bombs[:, None],
        game_segments.obstacles[:, None],
        game_segments.note_ids[:, None],
        game_segments.bomb_ids[:, None],
        game_segments.obstacle_ids[:, None],
        d["notes_np"],
        d["bombs_np"],
        d["obstacles_np"],
        1.5044,
        njs,
        1024,
    )
    (
        ts,
        n_opportunities,
        n_hits,
        n_misses,
        n_goods,
        collided_note_masks,
        bomb_penalty,
        obstacle_penalty,
        scores_at_collisions,
        final_good_yes,
    ) = TorchSaber.evaluate(
        note_alive_mask,
        bomb_appeared_yes_mask,
        bomb_collided_yes_mask,
        obstacle_collided_yes_mask,
        gc_mask,
        bc_mask,
        offset_vels,
        batch_size=1024,
    )
    return ts, n_opportunities, n_goods, n_hits, n_misses


def get_njs(map_info, difficulty, characteristic):
    found = False
    njs = 18
    for dbs in map_info["_difficultyBeatmapSets"]:
        if dbs["_beatmapCharacteristicName"] == characteristic:
            for db in dbs["_difficultyBeatmaps"]:
                if db["_difficulty"] == difficulty:
                    found = True
                    njs = db["_noteJumpMovementSpeed"]
                    break
    assert found
    if njs == 0:
        njs = map_info["_beatsPerMinute"] / 10
    return njs


def evaluate_xror_on_map(xror_unpacked, beatmap, map_info, left_handed=False, njs_offset=0.0):
    segment_sampler = SegmentSampler()
    difficulty = xror_unpacked.data["info"]["software"]["activity"]["difficulty"]
    characteristic = xror_unpacked.data["info"]["software"]["activity"]["mode"]
    d = load_cbo_and_3p(xror_unpacked, beatmap, map_info, left_handed=left_handed, rescale_yes=True)
    found = False
    njs = get_njs(map_info, difficulty, characteristic) + njs_offset
    d = nanpad_collate_fn([[d]])
    length = d["timestamps"].shape[1]
    # length = n_frames
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            d[k] = v.to(device=device)
    # d["lengths"][:] = n_frames # cap it at here
    # d["gt_3p_np"] = d["gt_3p_np"][:, :n_frames]
    game_segments = segment_sampler.sample_for_evaluation(
        d["notes_np"],
        d["bombs_np"],
        d["obstacles_np"],
        d["timestamps"],
        d["lengths"],
        length,
        1,
        512,
        1,
        4.0,
        80,
        -0.5,
    )
    gt_3p_np = d["gt_3p_np"]
    gt_3p_np = gt_3p_np.reshape((*gt_3p_np.shape[:-1], 3, -1))
    # height = torch.nanmedian(gt_3p_np[:, -1000:, 0, 1]).item()
    # height = np.nanmedian(np.array(xror_unpacked.data['events'][5]['floatData'])[:, -1]).item()
    note_alive_mask = torch.where(torch.isnan(d["notes_np"][..., 0]), torch.as_tensor(False), torch.as_tensor(True))[..., None, :]
    # njs = x["njs"]
    # njs = 10
    n_notes = d["notes_np"].shape[1]
    # print(f"{n_notes=}")
    (
        note_appeared_yes_mask,
        bomb_appeared_yes_mask,
        obstacle_appeared_mask,
        bomb_collided_yes_mask,
        obstacle_collided_yes_mask,
        gc_mask,
        bc_mask,
        ms_mask,
        offset_vels,
    ) = TorchSaber.simulate(
        gt_3p_np[:, None, [0]],
        gt_3p_np[:, None],
        game_segments.notes[:, None],
        game_segments.bombs[:, None],
        game_segments.obstacles[:, None],
        game_segments.note_ids[:, None],
        game_segments.bomb_ids[:, None],
        game_segments.obstacle_ids[:, None],
        d["notes_np"],
        d["bombs_np"],
        d["obstacles_np"],
        1.5044,
        njs,
        1024,
    )
    (
        ts,
        n_opportunities,
        n_hits,
        n_misses,
        n_goods,
        collided_note_masks,
        bomb_penalty,
        obstacle_penalty,
        scores_at_collisions,
        final_good_yes,
    ) = TorchSaber.evaluate(
        note_alive_mask,
        bomb_appeared_yes_mask,
        bomb_collided_yes_mask,
        obstacle_collided_yes_mask,
        gc_mask,
        bc_mask,
        offset_vels,
        batch_size=1024,
    )
    return ts, n_opportunities, n_goods, n_hits, n_misses


def method_name2(args, xror_unpacked, map_info, d, my_3p_traj, shard_idx, datapoint_idx, seed):
    difficulty = xror_unpacked.data["info"]["software"]["activity"]["difficulty"]
    characteristic = xror_unpacked.data["info"]["software"]["activity"]["mode"]
    njs = 0
    for dbs in map_info["_difficultyBeatmapSets"]:
        if dbs["_beatmapCharacteristicName"] == characteristic:
            for db in dbs["_difficultyBeatmaps"]:
                if db["_difficulty"] == difficulty:
                    found = True
                    njs = db["_noteJumpMovementSpeed"]
                    break
    if njs == 0:
        njs = map_info["_beatsPerMinute"] / 10

    purview_notes = 200
    timestamps = d["timestamps"]
    note_bags = d["notes_np"]
    bomb_bags = d["bombs_np"]
    obstacle_bags = d["obstacles_np"]

    height = 1.5044
    length = timestamps.shape[1]
    lengths = torch.tensor([length], dtype=torch.long, device=device)
    segment_sampler = SegmentSampler()
    game_segments = segment_sampler.sample_for_evaluation(
        note_bags,
        bomb_bags,
        obstacle_bags,
        timestamps,
        lengths,
        length,
        1,
        2048,
        1,
        10.0,
        200,
        -0.1,
    )
    note_verts, note_face_normals, note_quat = TorchSaber.get_note_verts_and_normals_and_quats(game_segments.notes[:, None], height, njs, 2)
    bomb_verts, bomb_face_normals, bomb_quat = TorchSaber.get_note_verts_and_normals_and_quats(game_segments.bombs[:, None], height, njs, 2)
    obstacle_verts, obstacle_face_normals = TorchSaber.get_obstacle_verts_and_normals(game_segments.obstacles[:, None], height, njs)
    three_p_verts, three_p_face_normals = TorchSaber.get_3p_verts_and_normals(my_3p_traj[:, None])
    (
        note_appeared_yes_mask,
        bomb_appeared_yes_mask,
        obstacle_appeared_mask,
        bomb_collided_yes_mask,
        obstacle_collided_yes_mask,
        gc_mask,
        bc_mask,
        ms_mask,
        offset_vels,
    ) = TorchSaber.simulate(
        my_3p_traj[:, None, [0]],
        my_3p_traj[:, None],
        game_segments.notes[:, None],
        game_segments.bombs[:, None],
        game_segments.obstacles[:, None],
        game_segments.note_ids[:, None],
        game_segments.bomb_ids[:, None],
        game_segments.obstacle_ids[:, None],
        d["notes_np"],
        d["bombs_np"],
        d["obstacles_np"],
        1.5044,
        njs,
        1024,
    )
    # collide_yes_across_time = gc_collided_yes_mask.detach().cpu().numpy()
    good_yes_across_time = gc_mask.detach().cpu().numpy()
    bad_yes_across_time = bc_mask.detach().cpu().numpy()
    three_p_obstacle_collision_yeses = obstacle_collided_yes_mask.detach().cpu().numpy()
    note_verts = note_verts.detach().cpu().numpy()
    note_quat = note_quat.detach().cpu().numpy()
    note_xyzs = note_verts.mean(-2)
    bomb_verts = bomb_verts.detach().cpu().numpy()
    bomb_quat = bomb_quat.detach().cpu().numpy()
    bomb_xyzs = bomb_verts.mean(-2)
    obstacle_verts = obstacle_verts.detach().cpu().numpy()
    notes = game_segments.notes.detach().cpu().numpy()
    three_p = my_3p_traj.detach().cpu().numpy()
    three_p_xyz = three_p_verts.detach().cpu().numpy().mean(-2)
    # Get red/green mask
    good_cumsum = gc_mask.any(-2).cumsum(2)
    bad_cumsum = bc_mask.any(-2).cumsum(2)
    hit_and_good_mask = ((good_cumsum > 0) & ~(bad_cumsum > 0)).cumsum(2) > 0
    hit_and_good_mask = hit_and_good_mask | ((gc_mask.any(-2) & bc_mask.any(-2)).cumsum(2) > 0)
    note_ids = torch.where(game_segments.note_ids == -1, 0, game_segments.note_ids)
    green_red_mask = torch.take_along_dim(hit_and_good_mask.to(device), note_ids[None], -1)

    # Visualize rollout ygts on the left, rollout wins on the right
    pl = Plotter(window_size=(608, 608), lighting="three lights", off_screen=True)
    pl.camera.position = (-7, 0, 7)
    pl.camera.focal_point = (3, 0, 0)
    pl.camera.up = (0, 0, 1)
    pl.set_background("#FFFFFF")
    pl.add_axes()
    plane = pv.Cube(center=(1, 0, 0), x_length=10, y_length=3, z_length=0.1)
    pl.add_mesh(plane, color="#FFFFFF")
    n_saber_visuals = 2
    n_axes_visuals = 3
    n_note_visuals = purview_notes
    n_bomb_visuals = purview_notes
    n_obstacle_visuals = purview_notes
    n_3p_visuals = 3
    note_visuals = np.empty(n_note_visuals, dtype=object)
    bomb_visuals = np.empty(n_bomb_visuals, dtype=object)
    obstacle_visuals = np.empty(n_obstacle_visuals, dtype=object)
    three_p_visuals = np.empty(n_3p_visuals, dtype=object)
    seg_visuals = np.empty((5, 2), dtype=object)
    for i in range(5):
        for j in range(2):
            seg_mesh = pv.Line()
            seg_actor = pl.add_mesh(seg_mesh, color="black", line_width=5, render_lines_as_tubes=True)
            seg_visual = GenericVisual(seg_mesh, seg_actor)
            seg_visuals[i, j] = seg_visual
    for i in range(n_note_visuals):
        note_visual = NoteVisual(pl)
        note_visuals[i] = note_visual

        bomb_visual = NoteVisual(pl)
        bomb_visual.bloq_mesh.cell_data["color"] = np.array([[0, 0, 0]]).repeat(bomb_visual.bloq_mesh.n_cells, 0)
        bomb_visuals[i] = bomb_visual

        obstacle_visual = ObstacleVisual(pl)
        obstacle_visuals[i] = obstacle_visual
    saber_visuals = np.empty(n_saber_visuals, dtype=object)
    axes_visuals = np.empty(n_axes_visuals, dtype=object)
    colors = {
        0: np.array([[255.0, 0.0, 132.0]]) / 255,
        1: np.array([[0.0, 229.0, 255.0]]) / 255,
        3: np.array([[0.0, 0.0, 0.0]]) / 255,
    }
    for i in range(n_saber_visuals):
        # collider_mesh = pv.Cube(x_length=1.0, y_length=0.1, z_length=0.1)
        collider_mesh = pv.Line([0, 0, 0], [1, 0, 0])
        collider_mesh.cell_data["color"] = colors[i].repeat(collider_mesh.n_cells, 0) * 1
        # collider_mesh.points += np.array([[0.5, 0, 0]])
        collider_actor = pl.add_mesh(collider_mesh, scalars="color", rgb=True, show_scalar_bar=False, render_lines_as_tubes=True, line_width=5)
        saber_visual = GenericVisual(collider_mesh, collider_actor)
        saber_visuals[i] = saber_visual
    for i in range(n_axes_visuals):
        sphere_colors = {
            0: np.array([1.0, 0.0, 0.0]),
            1: np.array([0.0, 1.0, 0.0]),
            2: np.array([0.0, 0.0, 1.0]),
        }
        sphere_mesh = pv.Sphere(radius=0.05)
        sphere_mesh.cell_data["color"] = np.array([sphere_colors[i]]).repeat(sphere_mesh.n_cells, 0)
        x_axis_mesh = pv.Arrow((0, 0, 0), (1, 0, 0), tip_radius=0.025, shaft_radius=0.01)
        x_axis_mesh.cell_data["color"] = np.array([[1.0, 0.0, 0.0]]).repeat(x_axis_mesh.n_cells, 0)
        y_axis_mesh = pv.Arrow((0, 0, 0), (0, 1, 0), tip_radius=0.025, shaft_radius=0.01)
        y_axis_mesh.cell_data["color"] = np.array([[0.0, 1.0, 0.0]]).repeat(y_axis_mesh.n_cells, 0)
        z_axis_mesh = pv.Arrow((0, 0, 0), (0, 0, 1), tip_radius=0.025, shaft_radius=0.01)
        z_axis_mesh.cell_data["color"] = np.array([[0.0, 0.0, 1.0]]).repeat(z_axis_mesh.n_cells, 0)
        axes_mesh = sphere_mesh + x_axis_mesh + y_axis_mesh + z_axis_mesh
        axes_actor = pl.add_mesh(axes_mesh, scalars="color", rgb=True)

        axes_visual = GenericVisual(axes_mesh, axes_actor)
        axes_visuals[i] = axes_visual
    for i in range(n_axes_visuals):
        cube_colors = {
            0: np.array([1.0, 0.0, 0.0]),
            1: np.array([0.0, 1.0, 0.0]),
            2: np.array([0.0, 0.0, 1.0]),
        }
        cube_mesh = pv.Cube(x_length=0.1, y_length=0.1, z_length=0.1)
        cube_mesh.cell_data["color"] = np.array([cube_colors[i]]).repeat(cube_mesh.n_cells, 0)
        three_p_mesh = cube_mesh
        three_p_actor = pl.add_mesh(three_p_mesh, scalars="color", rgb=True)
        three_p_visual = GenericVisual(three_p_mesh, three_p_actor)
        three_p_visuals[i] = three_p_visual
    imgs = []
    n_frames = three_p.shape[1]
    # n_frames = 500
    for state_frame in tqdm(np.arange(0, n_frames, 1)):
        for i, xyzsixd in enumerate(three_p[0, state_frame]):
            xyz = xyzsixd[:3]
            quat = sixd_to_quat(xyzsixd[3:])
            xyz, quat = unity_to_zup(xyz, quat)
            # pos = xyz
            pos = three_p_xyz[0, 0, state_frame, i]
            rot = Rotation.from_quat(quat).as_matrix()
            m = np.eye(4)
            m[:3, 3] = pos
            m[:3, :3] = rot
            # Sabers
            if 1 <= i <= 2:
                saber_visuals[i - 1].actor.user_matrix = m
                saber_visuals[i - 1].actor.SetVisibility(True)
                if bad_yes_across_time[0, 0, state_frame, i - 1].any():
                    saber_color = np.array([[1.0, 0.0, 0.0]])
                elif good_yes_across_time[0, 0, state_frame, i - 1].any():
                    saber_color = np.array([[0.0, 1.0, 0.0]])
                else:
                    saber_color = colors[i - 1]
                # saber_color = colors[i - 1]
                saber_visuals[i - 1].mesh.cell_data["color"] = saber_color.repeat(saber_visuals[i - 1].mesh.n_cells, 0) * 1

                if state_frame > 0:
                    old_xyzsixd = three_p[0, state_frame - 1, i] * 1
                    xyz = old_xyzsixd[:3]
                    quat = sixd_to_quat(old_xyzsixd[3:])
                    xyz, quat = unity_to_zup(xyz, quat)

                    old_hilt = xyz
                    old_tip = old_hilt + Rotation.from_quat(quat).as_matrix() @ np.array([1.0, 0, 0])
                    old_hilt_tip_lerp = np.linspace(old_hilt, old_tip, 5)

                    hilt = pos
                    tip = hilt + rot @ np.array([1.0, 0, 0])
                    hilt_tip_lerp = np.linspace(hilt, tip, 5)

                    for j in range(5):
                        seg_visuals[j, i - 1].mesh.points[0] = old_hilt_tip_lerp[j]
                        seg_visuals[j, i - 1].mesh.points[1] = hilt_tip_lerp[j]
                        seg_visuals[j, i - 1].actor.SetVisibility(True)

            # Axes
            axes_visuals[i].actor.SetVisibility(False)

            three_p_colors = {
                0: np.array([1.0, 0.0, 0.0]),
                1: np.array([0.0, 1.0, 0.0]),
                2: np.array([0.0, 0.0, 1.0]),
            }
            if three_p_obstacle_collision_yeses[0, 0, state_frame, i].any():
                three_p_color = np.array([1.0, 0.0, 0.0])[None]
            else:
                three_p_color = three_p_colors[i][None]
            three_p_visuals[i].mesh.cell_data["color"] = three_p_color.repeat(three_p_visuals[i].mesh.n_cells, 0) * 1

            three_p_visuals[i].actor.user_matrix = m
            three_p_visuals[i].actor.SetVisibility(True)

        # Note, bomb, and obstacle visualization
        for i in range(note_xyzs.shape[-2]):
            m = np.eye(4)
            note_xyz = note_xyzs[0, 0, state_frame, i]
            if ~np.isnan(note_xyz).any():
                note_info = notes[0, state_frame, i]
                # Bloq color
                color = colors[int(note_info[-3].item())]
                if green_red_mask[0, 0, state_frame, i]:
                    color = np.array([[0.0, 1.0, 0.0]])
                # else:
                #     color = np.array([[1.0, 0.0, 0.0]])

                note_visuals[i].bloq_mesh.cell_data["color"] = color.repeat(note_visuals[i].bloq_mesh.n_cells, 0) * 1

                m[:3, 3] = note_xyz
                # m[0, 3] += app_state.x_offset
                # m[2, 3] += app_state.z_offset
                m[:3, :3] = Rotation.from_quat(note_quat[0, 0, state_frame, i]).as_matrix()
                note_visuals[i].bloq_actor.user_matrix = m
                note_visuals[i].collider_actor.user_matrix = m
                note_visuals[i].arrow_actor.user_matrix = m
                note_visuals[i].bloq_actor.SetVisibility(True)
                note_visuals[i].collider_actor.SetVisibility(True)
                note_visuals[i].arrow_actor.SetVisibility(True)

                # if i == 5:
                #     print(f"{note_xyz=}")
                #     print(f"{tip=}")

                if game_segments.notes[0, state_frame, i, -2] == 8:
                    note_visuals[i].arrow_actor.SetVisibility(False)
            else:
                note_visuals[i].bloq_actor.SetVisibility(False)
                note_visuals[i].arrow_actor.SetVisibility(False)
                note_visuals[i].collider_actor.SetVisibility(False)

            # bombs
            bomb_xyz = bomb_xyzs[0, 0, state_frame, i]
            if ~np.isnan(bomb_xyz).any():
                m = np.eye(4)
                m[:3, 3] = bomb_xyz
                # m[:3, :3] = Rotation.from_quat(note_quat[0, 0, state_frame, i]).as_matrix()
                bomb_visuals[i].bloq_actor.user_matrix = m
                bomb_visuals[i].collider_actor.user_matrix = m
                bomb_visuals[i].arrow_actor.user_matrix = m
                bomb_visuals[i].bloq_actor.SetVisibility(True)
                bomb_visuals[i].collider_actor.SetVisibility(True)
                bomb_visuals[i].arrow_actor.SetVisibility(False)
            else:
                bomb_visuals[i].bloq_actor.SetVisibility(False)
                bomb_visuals[i].arrow_actor.SetVisibility(False)
                bomb_visuals[i].collider_actor.SetVisibility(False)

            if ~np.isnan(obstacle_verts[0, 0, state_frame, i]).any():
                obstacle_visuals[i].collider_mesh.points = obstacle_verts[0, 0, state_frame, i]
                obstacle_visuals[i].collider_actor.SetVisibility(True)
            else:
                obstacle_visuals[i].collider_actor.SetVisibility(False)

        pl.render()
        img = np.array(pl.screenshot())
        imgs.append(img)
    video_path = f"{proj_dir}/out/{args.run_name}/{args.out_name}/video_{shard_idx}_{datapoint_idx}.mp4"
    w = imageio.get_writer(
        video_path,
        format="FFMPEG",
        mode="I",
        fps=60,
        codec="h264",
        pixelformat="yuv420p",
    )
    for img in tqdm(imgs):
        w.append_data(img)
    w.close()
