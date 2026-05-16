import numpy as np
import torch


def expm_to_quat(expm: np.ndarray, eps: float = 1e-8):
    """ """
    half_angle = np.linalg.norm(expm, axis=-1)[..., None]
    c = np.cos(half_angle)
    s = np.sin(half_angle)
    quat = np.where(
        half_angle < eps,
        np.concatenate([expm, np.ones_like(half_angle)], axis=-1),
        np.concatenate([expm * s / half_angle, c], axis=-1),
    )
    return quat


def quat_to_expm_torch(quat: torch.Tensor, eps: float = 1e-8):
    """
    Quaternion is (x, y, z, w)
    """
    im = quat[..., :3]
    im_norm = torch.norm(im, dim=-1)
    half_angle = torch.arctan2(im_norm, quat[..., 3])
    expm = torch.where(
        im_norm[..., None] < eps,
        im,
        half_angle[..., None] * (im / im_norm[..., None]),
    )
    return expm


def expm_to_quat_torch(expm: torch.Tensor, eps: float = 1e-8):
    """ """
    half_angle = torch.linalg.norm(expm, dim=-1)[..., None]
    c = torch.cos(half_angle)
    s = torch.sin(half_angle)
    quat = torch.where(
        half_angle < eps,
        torch.cat([expm, torch.ones_like(half_angle)], dim=-1),
        torch.cat([expm * s / half_angle, c], dim=-1),
    )
    return quat


def quat_rotate(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)

    qvec = q[..., :-1]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[..., -1:] * uv + uuv)).view(original_shape)


def unity_to_zup(xyz: np.ndarray, quat: np.ndarray):
    my_pos = xyz
    my_pos[..., [0, 1, 2]] = my_pos[..., [2, 0, 1]]
    my_pos[..., 1] *= -1

    my_quat = quat
    my_quat[..., [0, 1, 2, 3]] = my_quat[..., [2, 0, 1, 3]]
    my_quat[..., [0, 2]] *= -1

    return my_pos, my_quat
