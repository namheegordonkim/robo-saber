import glob
import json
import os
from argparse import ArgumentParser

import imageio
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from proj_utils.dirs import proj_dir
from xror.xror import XROR
import pyvista as pv


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

    filepaths = glob.glob(
        f"{proj_dir}/data/beaterson/025bcf95-9be6-4620-8d68-6b5903e62aee/*.xror"
    )
    filepath = filepaths[-1]
    with open(filepath, "rb") as f:
        file = f.read()
    xror = XROR.unpack(file)
    frames_np = np.array(xror.data["frames"])

    timestamps = frames_np[1:, ..., 0]
    part_data = frames_np[1:, ..., 1:]
    part_data = part_data.reshape(part_data.shape[0], 3, -1)
    my_pos = part_data[..., :3]
    my_quat = part_data[..., 3:]
    my_rot = Rotation.from_quat(my_quat.reshape(-1, 4))

    correction = Rotation.from_euler("X", 0.5 * np.pi)
    correction_matrix = correction.as_matrix()
    my_pos = np.einsum("nkj,ij->nki", my_pos, correction_matrix)
    my_rot = correction * my_rot

    correction = Rotation.from_euler("Z", 0.5 * np.pi)
    correction_matrix = correction.as_matrix()
    my_pos = np.einsum("nkj,ij->nki", my_pos, correction_matrix)
    my_rot = correction * my_rot

    my_rotmat = my_rot.as_matrix().reshape((my_quat.shape[0], 3, 3, 3))
    my_rotmat[..., [0, 1, 2]] = my_rotmat[..., [2, 0, 1]]
    my_rotmat[:, 1:, ..., [1, 2]] *= -1
    my_rot = Rotation.from_matrix(my_rotmat.reshape((-1, 3, 3)))

    my_quat = my_rot.as_quat().reshape(*my_quat.shape)
    my_sixd = my_rot.as_matrix()[..., :2].reshape(*my_quat.shape[:-1], 6)
    my_pos[..., [0, 1]] -= my_pos[0][0, [0, 1]]

    my_pos_sixd = np.concatenate([my_pos, my_sixd], axis=-1)
    my_pos_sixd[:, [0, 1, 2]] = my_pos_sixd[:, [0, 2, 1]]
    my_pos_sixd = my_pos_sixd.reshape(-1, 27)

    d = {
        "my_pos_sixd": my_pos_sixd,
        "timestamps": timestamps,
    }
    out_dir = os.path.join(proj_dir, "out", args.run_name, args.out_name)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(d, f"{out_dir}/pos_sixd_time.pkl")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--out_name", type=str, required=True)
    parser.add_argument(
        "--debug_yes", "-d", action="store_true"
    )  # if set, will pause the program
    args, remaining_args = parser.parse_known_args()
    main(args, remaining_args)
