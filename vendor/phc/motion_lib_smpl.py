import os.path as osp
import random

import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot

from .flags import flags
from .motion_lib_base import (
    MotionLibBase,
    compute_motion_dof_vels,
    FixHeightMode,
)
from vendor.poselib.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState


def to_torch(tensor):
    if torch.is_tensor(tensor):
        return tensor
    return torch.from_numpy(tensor)


USE_CACHE = False
print("MOVING MOTION DATA TO GPU, USING CACHE:", USE_CACHE)

if not USE_CACHE:
    old_numpy = torch.Tensor.numpy

    class Patch:

        def numpy(self):
            if self.is_cuda:
                return self.to("cpu").numpy()
            else:
                return old_numpy(self)

    torch.Tensor.numpy = Patch.numpy


class MotionLibSMPL(MotionLibBase):

    def __init__(self, motion_lib_cfg):
        super().__init__(motion_lib_cfg=motion_lib_cfg)
        self.mesh_parsers = None
        return

    @staticmethod
    def fix_trans_height(
        pose_aa, trans, curr_gender_betas, mesh_parsers, fix_height_mode
    ):
        if fix_height_mode == FixHeightMode.no_fix:
            return trans, 0

        with torch.no_grad():
            frame_check = 30
            gender = curr_gender_betas[0]
            betas = curr_gender_betas[1:]
            mesh_parser = mesh_parsers[gender.item()]
            height_tolorance = 0.0
            vertices_curr, joints_curr = mesh_parser.get_joints_verts(
                pose_aa[:frame_check], betas[None,], trans[:frame_check]
            )

            offset = (
                joints_curr[:, 0] - trans[:frame_check]
            )  # account for SMPL root offset. since the root trans we pass in has been processed, we have to "add it back".

            if fix_height_mode == FixHeightMode.ankle_fix:
                assignment_indexes = mesh_parser.lbs_weights.argmax(axis=1)
                pick = (
                    (
                        (
                            (
                                assignment_indexes
                                != mesh_parser.joint_names.index("L_Toe")
                            ).int()
                            + (
                                assignment_indexes
                                != mesh_parser.joint_names.index("R_Toe")
                            ).int()
                            + (
                                assignment_indexes
                                != mesh_parser.joint_names.index("R_Hand")
                            ).int()
                            + +(
                                assignment_indexes
                                != mesh_parser.joint_names.index("L_Hand")
                            ).int()
                        )
                        == 4
                    )
                    .nonzero()
                    .squeeze()
                )
                diff_fix = (
                    (vertices_curr[:, pick] - offset[:, None])[:frame_check, ..., -1]
                    .min(dim=-1)
                    .values
                    - height_tolorance
                ).min()  # Only acount the first 30 frames, which usually is a calibration phase.
            elif fix_height_mode == FixHeightMode.full_fix:

                diff_fix = (
                    (vertices_curr - offset[:, None])[:frame_check, ..., -1]
                    .min(dim=-1)
                    .values
                    - height_tolorance
                ).min()  # Only acount the first 30 frames, which usually is a calibration phase.

            trans[..., -1] -= diff_fix
            return trans, diff_fix

    @staticmethod
    def load_motion_with_skeleton(
        ids,
        motion_data_list,
        skeleton_trees,
        shape_params,
        mesh_parsers,
        config,
        queue,
        pid,
    ):
        # ZL: loading motion with the specified skeleton. Perfoming forward kinematics to get the joint positions
        max_len = config.max_length
        fix_height = config.fix_height
        np.random.seed(np.random.randint(5000) * pid)
        res = {}
        assert len(ids) == len(motion_data_list)
        for f in range(len(motion_data_list)):
            curr_id = ids[f]  # id for this datasample
            curr_file = motion_data_list[f]
            if not isinstance(curr_file, dict) and osp.isfile(curr_file):
                key = motion_data_list[f].split("/")[-1].split(".")[0]
                curr_file = joblib.load(curr_file)[key]
            seq_len = curr_file["trans_orig"].shape[0]
            if max_len == -1 or seq_len < max_len:
                start, end = 0, seq_len
            else:
                start = random.randint(0, seq_len - max_len)
                end = start + max_len
            trans = curr_file["trans_orig"][start:end]
            pose_quat_global = curr_file["pose_quat_global"][start:end]

            trans = to_torch(trans)

            B, J, N = pose_quat_global.shape
            ##### ZL: randomize the heading ######
            if (not flags.im_eval) and (not flags.test):
                # if True:
                random_rot = np.zeros(3)
                random_rot[2] = np.pi * (2 * np.random.random() - 1.0)
                random_heading_rot = sRot.from_euler("xyz", random_rot)
                pose_quat_global = (
                    (
                        random_heading_rot
                        * sRot.from_quat(pose_quat_global.reshape(-1, 4))
                    )
                    .as_quat()
                    .reshape(B, J, N)
                )
                trans = torch.matmul(
                    trans, torch.from_numpy(random_heading_rot.as_matrix().T)
                )
            else:
                init_heading_rot = sRot.from_quat(pose_quat_global[0, 0].reshape(-1, 4))
                pose_quat_global = (
                    (
                        init_heading_rot.inv()
                        * sRot.from_quat(pose_quat_global.reshape(-1, 4))
                    )
                    .as_quat()
                    .reshape(B, J, N)
                )
                trans = np.matmul(trans, init_heading_rot.inv().as_matrix()[0].T)

            ##### ZL: randomize the heading ######
            pose_quat_global = to_torch(pose_quat_global)
            trans = to_torch(trans)
            sk_state = SkeletonState.from_rotation_and_root_translation(
                skeleton_trees[f], pose_quat_global, trans, is_local=False
            )
            curr_motion = SkeletonMotion.from_skeleton_state(
                sk_state, curr_file.get("fps", 30)
            )
            curr_dof_vels = compute_motion_dof_vels(curr_motion)
            curr_motion.dof_vels = curr_dof_vels
            # res[curr_id] = (curr_file, curr_motion)
            res[curr_id] = curr_motion
            print(pid, f)

        if queue is not None:
            queue.put(res)
        else:
            return res

    # @staticmethod
    # def load_motion_with_skeleton(
    #     ids,
    #     motion_data_list,
    #     skeleton_trees,
    #     shape_params,
    #     mesh_parsers,
    #     config,
    #     queue,
    #     pid,
    # ):
    #     # ZL: loading motion with the specified skeleton. Perfoming forward kinematics to get the joint positions
    #     max_len = config.max_length
    #     fix_height = config.fix_height
    #     np.random.seed(np.random.randint(5000) * pid)
    #     res = {}
    #     assert len(ids) == len(motion_data_list)
    #     for f in range(len(motion_data_list)):
    #         curr_id = ids[f]  # id for this datasample
    #         curr_file = motion_data_list[f]
    #         if not isinstance(curr_file, dict) and osp.isfile(curr_file):
    #             key = motion_data_list[f].split("/")[-1].split(".")[0]
    #             curr_file = joblib.load(curr_file)[key]
    #         curr_gender_beta = shape_params[f]
    #
    #         seq_len = curr_file["root_trans_offset"].shape[0]
    #         if max_len == -1 or seq_len < max_len:
    #             start, end = 0, seq_len
    #         else:
    #             start = random.randint(0, seq_len - max_len)
    #             end = start + max_len
    #
    #         trans = curr_file["root_trans_offset"].clone()[start:end]
    #         pose_aa = to_torch(curr_file["pose_aa"][start:end])
    #         pose_quat_global = curr_file["pose_quat_global"][start:end]
    #
    #         B, J, N = pose_quat_global.shape
    #
    #         ##### ZL: randomize the heading ######
    #         if (not flags.im_eval) and (not flags.test):
    #             # if True:
    #             random_rot = np.zeros(3)
    #             random_rot[2] = np.pi * (2 * np.random.random() - 1.0)
    #             random_heading_rot = sRot.from_euler("xyz", random_rot)
    #             pose_aa[:, :3] = torch.tensor(
    #                 (random_heading_rot * sRot.from_rotvec(pose_aa[:, :3])).as_rotvec()
    #             )
    #             pose_quat_global = (
    #                 (
    #                     random_heading_rot
    #                     * sRot.from_quat(pose_quat_global.reshape(-1, 4))
    #                 )
    #                 .as_quat()
    #                 .reshape(B, J, N)
    #             )
    #             trans = torch.matmul(
    #                 trans, torch.from_numpy(random_heading_rot.as_matrix().T)
    #             )
    #         ##### ZL: randomize the heading ######
    #
    #         if not mesh_parsers is None:
    #             trans, trans_fix = MotionLibSMPL.fix_trans_height(
    #                 pose_aa,
    #                 trans,
    #                 curr_gender_beta,
    #                 mesh_parsers,
    #                 fix_height_mode=fix_height,
    #             )
    #         else:
    #             trans_fix = 0
    #
    #         correction = Rotation.from_euler("Z", -0.5 * np.pi)
    #         pose_rots = Rotation.from_quat(pose_quat_global.reshape(-1, 4))
    #         corrected_pose_rots = (correction * pose_rots)
    #         pose_quat_global = corrected_pose_rots.as_quat().reshape(pose_quat_global.shape)
    #
    #         pose_quat_global = to_torch(pose_quat_global)
    #         sk_state = SkeletonState.from_rotation_and_root_translation(
    #             skeleton_trees[f], pose_quat_global, trans, is_local=False
    #         )
    #         curr_motion = SkeletonMotion.from_skeleton_state(
    #             sk_state, curr_file.get("fps", 30)
    #         )
    #
    #         # d = torch.load(
    #         #     f"{proj_dir}/data/motionlib/from_adas/turn_me_on_expert_curr_motion.pkl"
    #         # )
    #         # pose_quat_global = to_torch(d["pose_quat_global"])
    #         # trans = to_torch(d["trans"])
    #         # curr_motion = SkeletonMotion.from_skeleton_state(
    #         #     sk_state, 60
    #         # )
    #         # curr_motion = torch.load(
    #         #     f"{proj_dir}/data/motionlib/from_adas/turn_me_on_expert_curr_motion.pkl"
    #         # )
    #         # curr_motion = torch.load(
    #         #     config["curr_motion_path"]
    #         # )
    #         curr_dof_vels = compute_motion_dof_vels(curr_motion)
    #
    #         # trans_np = trans.cpu().numpy()
    #         # pkg_np = pose_quat_global.cpu().numpy()
    #         # cmgt_np = curr_motion.global_translation.cpu().numpy()
    #         # import matplotlib.pyplot as plt
    #         # fig = plt.figure()
    #         # ax = fig.add_subplot(projection='3d')
    #         # # ax.scatter(trans_np[:, 0], trans_np[:, 1], trans_np[:, 2])
    #         # ax.scatter(cmgt_np[0, :, 0], cmgt_np[0, :, 1], cmgt_np[0, :, 2])
    #         # ax.set_xlim([-1, 1])
    #         # ax.set_ylim([-1, 1])
    #         # ax.set_zlim([-1, 1])
    #         # plt.show()
    #         # plt.close()
    #
    #         if flags.real_traj:
    #             quest_sensor_data = to_torch(curr_file["quest_sensor_data"])
    #             quest_trans = quest_sensor_data[..., :3]
    #             quest_rot = quest_sensor_data[..., 3:]
    #
    #             quest_trans[..., -1] -= trans_fix  # Fix trans
    #
    #             global_angular_vel = SkeletonMotion._compute_angular_velocity(
    #                 quest_rot, time_delta=1 / curr_file["fps"]
    #             )
    #             linear_vel = SkeletonMotion._compute_velocity(
    #                 quest_trans, time_delta=1 / curr_file["fps"]
    #             )
    #             quest_motion = {
    #                 "global_angular_vel": global_angular_vel,
    #                 "linear_vel": linear_vel,
    #                 "quest_trans": quest_trans,
    #                 "quest_rot": quest_rot,
    #             }
    #             curr_motion.quest_motion = quest_motion
    #
    #         curr_motion.dof_vels = curr_dof_vels
    #         curr_motion.gender_beta = curr_gender_beta
    #
    #         # torch.save(curr_motion, f"{proj_dir}/data/motionlib/from_adas/curr_motion.pkl")
    #
    #         res[curr_id] = (curr_file, curr_motion)
    #
    #     if not queue is None:
    #         queue.put(res)
    #     else:
    #         return res
