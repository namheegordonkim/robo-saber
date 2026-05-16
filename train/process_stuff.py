import os
from argparse import ArgumentParser

import numpy as np
import torch

from proj_utils import my_logging
from proj_utils.dirs import proj_dir
from beaty_common.bsmg_xror_utils import open_beatmap_from_bsmg_or_boxrr, get_cbo_np, extract_3p_with_60fps
from xror.xror import XROR


def main(args, remaining_args):
    if args.debug_yes:
        import pydevd_pycharm

        pydevd_pycharm.settrace(
            "localhost",
            port=12346,
            stdout_to_server=True,
            stderr_to_server=True,
            suspend=False,
        )

    logdir = f"{proj_dir}/logdir/{args.run_name}/{args.out_name}"
    os.makedirs(logdir, exist_ok=True)
    logger = my_logging.get_logger(args.run_name, args.out_name, logdir)
    logger.info(f"Starting")

    beatmap, song_info = open_beatmap_from_bsmg_or_boxrr(None, args.in_boxrr)
    with open(args.in_boxrr, "rb") as f:
        file = f.read()
    xror = XROR.unpack(file)
    notes_np, bombs_np, obstacles_np = get_cbo_np(beatmap, song_info)
    frames_np = np.array(xror.data["frames"])
    my_pos_expm, my_pos_sixd, timestamps = extract_3p_with_60fps(frames_np)

    d = {
        "notes_np": notes_np,
        "bombs_np": bombs_np,
        "obstacles_np": obstacles_np,
        "my_pos_sixd": my_pos_sixd,
        "my_pos_expm": my_pos_expm,
        "timestamps": timestamps,
    }
    out_dir = os.path.join(proj_dir, "out", args.run_name, args.out_name)
    out_path = f"{out_dir}/{args.out_path}"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(d, out_path)
    logger.info(f"Saved to {out_path}")

    # pl = pv.Plotter(off_screen=True, window_size=(608, 608))
    # pl.add_axes()
    # sphere_actors = []
    # axes_actors = []
    # plane = pv.Plane(center=(0, 0, 0), i_size=10, j_size=10)
    # sphere = pv.Sphere(radius=0.05)
    # for i in range(3):
    #     colors = {0: "red", 1: "green", 2: "blue"}
    #     sphere_actor = pl.add_mesh(sphere, color=colors[i])
    #     sphere_actors.append(sphere_actor)
    #
    #     x_axis_mesh = pv.Arrow(
    #         (0, 0, 0), (1, 0, 0), tip_radius=0.025, shaft_radius=0.01
    #     )
    #     x_axis_mesh.cell_data["color"] = (
    #         np.array([[1, 0, 0]]).repeat(x_axis_mesh.n_cells, 0) * 255
    #     )
    #     y_axis_mesh = pv.Arrow(
    #         (0, 0, 0), (0, 1, 0), tip_radius=0.025, shaft_radius=0.01
    #     )
    #     y_axis_mesh.cell_data["color"] = (
    #         np.array([[0, 1, 0]]).repeat(y_axis_mesh.n_cells, 0) * 255
    #     )
    #     z_axis_mesh = pv.Arrow(
    #         (0, 0, 0), (0, 0, 1), tip_radius=0.025, shaft_radius=0.01
    #     )
    #     z_axis_mesh.cell_data["color"] = (
    #         np.array([[0, 0, 1]]).repeat(z_axis_mesh.n_cells, 0) * 255
    #     )
    #     axes_mesh = x_axis_mesh + y_axis_mesh + z_axis_mesh
    #     axes_actor = pl.add_mesh(axes_mesh, scalars="color", rgb=True)
    #     axes_actors.append(axes_actor)
    #
    # note_actors = []
    # note_meshes = []
    # for i in range(20):
    #     note_mesh = pv.Cube(center=(0, 0, 0), x_length=0.1, y_length=0.1, z_length=0.1)
    #     # note_mesh = pv.Sphere(radius=0.1)
    #     note_mesh.cell_data["color"] = np.array([[0, 0, 0]]).repeat(
    #         note_mesh.n_cells, 0
    #     )
    #     note_actor = pl.add_mesh(note_mesh, rgb=True, scalars="color")
    #     note_actors.append(note_actor)
    #     note_meshes.append(note_mesh)
    #
    # pl.add_mesh(plane)
    # pl.enable_shadows()
    #
    # n_frames = timestamps.shape[0]
    # imgs = []
    # for t in tqdm(range(0, n_frames, 4)):
    #     # for t in tqdm(range(0, 200, 4)):
    #     for i, sphere_actor in enumerate(sphere_actors):
    #         m = np.eye(4)
    #         m[:3, 3] = my_pos[t, i]
    #         m[:3, :3] = Rotation.from_quat(my_quat[t, i]).as_matrix()
    #         sphere_actor.user_matrix = m
    #         axes_actors[i].user_matrix = m
    #
    #     stamp = timestamps[t]
    #     delta_times = song_and_xror_merged[:, 5] - stamp
    #     within_purview_yes = np.logical_and(delta_times < 2.0, delta_times > 0.0)
    #     notes_within = song_and_xror_merged[within_purview_yes]
    #     plane_width = 1.5
    #     plane_height = 1.0
    #     plane_left = 0 - plane_width / 2
    #     plane_bottom = 0 - plane_height / 2
    #     plane_right = 0 + plane_width / 2
    #     plane_top = 0 + plane_height / 2
    #     # plane_grid = np.array(np.meshgrid(np.linspace(plane_bottom, plane_top, 3), np.linspace(plane_left, plane_right, 4)))
    #     plane_grid = np.array(
    #         np.meshgrid(
    #             np.linspace(plane_left, plane_right, 4),
    #             np.linspace(plane_bottom, plane_top, 3),
    #         )
    #     ).transpose((0, 2, 1))
    #
    #     for i, note_within in enumerate(notes_within):
    #         note_actor = note_actors[i]
    #         note_mesh = note_meshes[i]
    #         delta_times = note_within[5] - stamp
    #         note_pos = plane_grid[
    #             :, note_within[1].astype(int), note_within[2].astype(int)
    #         ]
    #         note_pos = np.array([0.5, *note_pos])
    #         note_pos += my_pos[t, 0]
    #         note_pos[0] += delta_times * 2
    #
    #         color = {0: np.array([[0, 1, 0]]), 1: np.array([[0, 0, 1]])}[
    #             note_within[3].astype(int)
    #         ]
    #         note_mesh.cell_data["color"] = color.repeat(note_mesh.n_cells, 0) * 255
    #
    #         m = np.eye(4)
    #         m[:3, 3] = note_pos
    #         note_actor.user_matrix = m
    #         note_actor
    #
    #     for i in range(len(notes_within), 20):
    #         note_actor = note_actors[i]
    #         m = np.eye(4)
    #         m[:3, 3] = (0, 0, 0)
    #         note_actor.user_matrix = m
    #
    #     pl.camera.position = (-7, -7, 7)
    #     pl.camera.focal_point = (0, 0, 0)
    #     pl.render()
    #     img = np.array(pl.screenshot())
    #     imgs.append(img)
    #
    # w = imageio.get_writer(
    #     f"{proj_dir}/dump/videooo.mp4",
    #     format="FFMPEG",
    #     mode="I",
    #     fps=15,
    #     codec="h264",
    #     pixelformat="yuv420p",
    # )
    # for img in imgs:
    #     w.append_data(img)
    # w.close()

    logger.info(f"Done")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--out_name", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    # parser.add_argument("--in_bsmg_info", type=str, required=True)
    # parser.add_argument("--in_bsmg_level", type=str, required=True)
    parser.add_argument("--in_boxrr", type=str, required=True)

    parser.add_argument(
        "--debug_yes", "-d", action="store_true"
    )  # if set, will pause the program
    args, remaining_args = parser.parse_known_args()
    main(args, remaining_args)
