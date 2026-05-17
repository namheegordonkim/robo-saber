import numpy as np
import torch
from scipy.spatial.transform import Rotation


def sixd_to_quat(sixd: np.ndarray or torch.Tensor) -> np.ndarray or torch.Tensor:
    if isinstance(sixd, torch.Tensor):
        sixd_array = sixd.detach().cpu().numpy()
    else:
        sixd_array = sixd * 1
    sixd_array = sixd_array.reshape((*sixd_array.shape[:-1], 2, 3)).swapaxes(-2, -1)
    m = np.concatenate([sixd_array, np.cross(sixd_array[..., 0], sixd_array[..., 1], axis=-1)[..., None]], axis=-1)
    r = Rotation.from_matrix(m.reshape((-1, 3, 3)))
    quat = r.as_quat().reshape(*sixd.shape[:-1], 4)
    if isinstance(sixd, torch.Tensor):
        quat = torch.as_tensor(quat, device=sixd.device, dtype=sixd.dtype)
    return quat


def quat_to_sixd(quat: np.ndarray or torch.Tensor) -> np.ndarray or torch.Tensor:
    r = Rotation.from_quat(quat.reshape(-1, 4))
    m = r.as_matrix().reshape((*quat.shape[:-1], 3, 3))
    sixd = m[..., :2].swapaxes(-2, -1).reshape(*quat.shape[:-1], 6)
    if isinstance(quat, torch.Tensor):
        sixd = torch.as_tensor(sixd, device=quat.device, dtype=quat.dtype)
    return sixd


def slerp(q0, q1, t):
    cos_half_theta = torch.sum(q0 * q1, dim=-1)

    neg_mask = cos_half_theta < 0
    q1 = q1.clone()
    q1[neg_mask] = -q1[neg_mask]
    cos_half_theta = torch.abs(cos_half_theta)
    cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

    half_theta = torch.acos(cos_half_theta)
    sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

    ratioA = torch.sin((1 - t) * half_theta) / sin_half_theta
    ratioB = torch.sin(t * half_theta) / sin_half_theta

    new_q = ratioA * q0 + ratioB * q1

    new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
    new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)

    return new_q


def interpolate_xyzsixd(keypoints, stride):
    device = keypoints.device
    if keypoints.shape[2] == 1:
        return keypoints[:, :, :, None].repeat_interleave(stride, -3)
    lefts = keypoints[:, :, :-1]
    rights = keypoints[:, :, 1:]
    lefts = lefts.reshape((lefts.shape[0], lefts.shape[1], lefts.shape[2], 3, -1))
    rights = rights.reshape((rights.shape[0], rights.shape[1], rights.shape[2], 3, -1))
    left_xyz = lefts[..., :3]
    right_xyz = rights[..., :3]
    lerp_t = torch.linspace(0, 1, stride + 1, device=device)[1:]
    interpolated_xyz = left_xyz[:, :, :, None] * (1 - lerp_t[None, None, None, :, None, None]) + right_xyz[:, :, :, None] * lerp_t[None, None, None, :, None, None]
    left_sixd = lefts[..., 3:]
    right_sixd = rights[..., 3:]
    left_quat = sixd_to_quat(left_sixd)
    right_quat = sixd_to_quat(right_sixd)
    slerp_t = torch.linspace(0, 1, stride + 1, device=device)[1:]
    interpolated_quat = slerp(left_quat[:, :, :, None], right_quat[:, :, :, None], slerp_t[None, None, None, :, None, None])
    interpolated_sixd = quat_to_sixd(interpolated_quat.detach().cpu()).to(device=device)
    interpolated = torch.cat([interpolated_xyz, interpolated_sixd], dim=-1)
    interpolated = interpolated.flatten(2, 3).flatten(-2, -1)
    return interpolated
