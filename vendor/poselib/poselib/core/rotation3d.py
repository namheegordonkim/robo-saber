# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
# Trimmed for robo-saber: only quaternion helpers referenced by callers remain.

import torch


@torch.jit.script
def quat_conjugate(x):
    return torch.cat([-x[..., :3], x[..., 3:]], dim=-1)


@torch.jit.script
def quat_inverse(x):
    return quat_conjugate(x)
