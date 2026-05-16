import numpy as np
import torch
from scipy.spatial.transform import Rotation


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
    my_pos = xyz * 1
    my_pos[..., [0, 1, 2]] = my_pos[..., [2, 0, 1]]
    my_pos[..., 1] *= -1

    my_quat = quat * 1
    my_quat[..., [0, 1, 2, 3]] = my_quat[..., [2, 0, 1, 3]]
    my_quat[..., [0, 2]] *= -1

    return my_pos, my_quat


def zup_to_unity(xyz: np.ndarray, quat: np.ndarray):
    my_pos = xyz
    my_pos[..., 1] *= -1
    my_pos[..., [0, 1, 2]] = my_pos[..., [1, 2, 0]]

    my_quat = quat
    my_quat[..., [0, 2]] *= -1
    my_quat[..., [0, 1, 2, 3]] = my_quat[..., [1, 2, 0, 3]]

    return my_pos, my_quat


def rotate_thumb_to_fingertip(quat: np.ndarray):
    # Local angle corrections
    my_lhand_rot = Rotation.from_quat(quat[..., 1, :].reshape((-1, 4)))

    my_lhand_eul = my_lhand_rot.as_euler("ZYX", degrees=True)
    my_lhand_eul[..., -1] -= 90
    my_lhand_rot = Rotation.from_euler("ZYX", my_lhand_eul, degrees=True)

    my_lhand_eul = my_lhand_rot.as_euler("XYZ", degrees=True)
    my_lhand_eul[..., -1] -= 90
    my_lhand_rot = Rotation.from_euler("XYZ", my_lhand_eul, degrees=True)

    quat[..., 1, :] = my_lhand_rot.as_quat()

    my_rhand_rot = Rotation.from_quat(quat[..., 2, :].reshape((-1, 4)))

    my_rhand_eul = my_rhand_rot.as_euler("ZYX", degrees=True)
    my_rhand_eul[..., -1] += 90
    my_rhand_rot = Rotation.from_euler("ZYX", my_rhand_eul, degrees=True)

    my_rhand_eul = my_rhand_rot.as_euler("XYZ", degrees=True)
    my_rhand_eul[..., -1] += 90
    my_rhand_rot = Rotation.from_euler("XYZ", my_rhand_eul, degrees=True)

    quat[..., 2, :] = my_rhand_rot.as_quat()
    return quat


def rotate_fingertip_to_thumb(quat: np.ndarray):
    # Local angle corrections
    my_lhand_rot = Rotation.from_quat(quat[..., 1, :].reshape((-1, 4)))

    my_lhand_eul = my_lhand_rot.as_euler("XYZ", degrees=True)
    my_lhand_eul[..., -1] += 90
    my_lhand_rot = Rotation.from_euler("XYZ", my_lhand_eul, degrees=True)

    my_lhand_eul = my_lhand_rot.as_euler("ZYX", degrees=True)
    my_lhand_eul[..., -1] += 90
    my_lhand_rot = Rotation.from_euler("ZYX", my_lhand_eul, degrees=True)

    quat[..., 1, :] = my_lhand_rot.as_quat()

    my_rhand_rot = Rotation.from_quat(quat[..., 2, :].reshape((-1, 4)))

    my_rhand_eul = my_rhand_rot.as_euler("XYZ", degrees=True)
    my_rhand_eul[..., -1] -= 90
    my_rhand_rot = Rotation.from_euler("XYZ", my_rhand_eul, degrees=True)

    my_rhand_eul = my_rhand_rot.as_euler("ZYX", degrees=True)
    my_rhand_eul[..., -1] -= 90
    my_rhand_rot = Rotation.from_euler("ZYX", my_rhand_eul, degrees=True)

    quat[..., 2, :] = my_rhand_rot.as_quat()
    return quat


# @torch.jit.script
def quat_to_mat(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


# @torch.jit.script
def mat_to_quat(m):
    """
    Construct a 3D rotation from a valid 3x3 rotation matrices.
    Reference can be found here:
    http://www.cg.info.hiroshima-cu.ac.jp/~miyazaki/knowledge/teche52.html

    :param m: 3x3 orthogonal rotation matrices.
    :type m: Tensor

    :rtype: Tensor
    """
    m = m.unsqueeze(0)
    diag0 = m[..., 0, 0]
    diag1 = m[..., 1, 1]
    diag2 = m[..., 2, 2]

    # Math stuff.
    w = (((diag0 + diag1 + diag2 + 1.0) / 4.0).clamp(0.0, None)) ** 0.5
    x = (((diag0 - diag1 - diag2 + 1.0) / 4.0).clamp(0.0, None)) ** 0.5
    y = (((-diag0 + diag1 - diag2 + 1.0) / 4.0).clamp(0.0, None)) ** 0.5
    z = (((-diag0 - diag1 + diag2 + 1.0) / 4.0).clamp(0.0, None)) ** 0.5

    # Only modify quaternions where w > x, y, z.
    c0 = (w >= x) & (w >= y) & (w >= z)
    x[c0] *= (m[..., 2, 1][c0] - m[..., 1, 2][c0]).sign()
    y[c0] *= (m[..., 0, 2][c0] - m[..., 2, 0][c0]).sign()
    z[c0] *= (m[..., 1, 0][c0] - m[..., 0, 1][c0]).sign()

    # Only modify quaternions where x > w, y, z
    c1 = (x >= w) & (x >= y) & (x >= z)
    w[c1] *= (m[..., 2, 1][c1] - m[..., 1, 2][c1]).sign()
    y[c1] *= (m[..., 1, 0][c1] + m[..., 0, 1][c1]).sign()
    z[c1] *= (m[..., 0, 2][c1] + m[..., 2, 0][c1]).sign()

    # Only modify quaternions where y > w, x, z.
    c2 = (y >= w) & (y >= x) & (y >= z)
    w[c2] *= (m[..., 0, 2][c2] - m[..., 2, 0][c2]).sign()
    x[c2] *= (m[..., 1, 0][c2] + m[..., 0, 1][c2]).sign()
    z[c2] *= (m[..., 2, 1][c2] + m[..., 1, 2][c2]).sign()

    # Only modify quaternions where z > w, x, y.
    c3 = (z >= w) & (z >= x) & (z >= y)
    w[c3] *= (m[..., 1, 0][c3] - m[..., 0, 1][c3]).sign()
    x[c3] *= (m[..., 2, 0][c3] + m[..., 0, 2][c3]).sign()
    y[c3] *= (m[..., 2, 1][c3] + m[..., 1, 2][c3]).sign()

    return quat_normalize(torch.stack([x, y, z, w], dim=-1)).squeeze(0)


# @torch.jit.script
def quat_normalize(q):
    """
    Construct 3D rotation from quaternion (the quaternion needs not to be normalized).
    """
    q = quat_unit(quat_pos(q))  # normalized to positive and unit quaternion
    return q


# @torch.jit.script
def quat_unit(x):
    """
    normalized quaternion with norm of 1
    """
    norm = quat_abs(x).unsqueeze(-1)
    return x / (norm.clamp(min=1e-9))


# @torch.jit.script
def quat_pos(x):
    """
    make all the real part of the quaternion positive
    """
    q = x
    z = (q[..., 3:] < 0).float()
    q = (1 - 2 * z) * q
    return q


# @torch.jit.script
def quat_abs(x):
    """
    quaternion norm (unit quaternion represents a 3D rotation, which has norm of 1)
    """
    x = x.norm(p=2, dim=-1)
    return x


def mat_to_sixd(mat: torch.Tensor or np.ndarray):
    if torch.is_tensor(mat):
        curr_pose = mat.to(mat.device).float().reshape(*mat.shape[:-2], 3, 3)
    else:
        curr_pose = torch.tensor(mat).to(mat.device).float().reshape(*mat.shape[:-1], 3, 3)

    sixd = curr_pose[:, :, :2].transpose(1, 2).flatten(-1)
    sixd = sixd.reshape(mat.shape[0], -1, 6)
    return sixd


def sixd_to_mat(sixd: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalisation per Section B of [1].
    Args:
        sixd: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = sixd[..., :3], sixd[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def quat_to_sixd_torch(quat: torch.Tensor) -> torch.Tensor:
    return mat_to_sixd(quat_to_mat(quat))


def sixd_to_quat_torch(sixd: torch.Tensor) -> torch.Tensor:
    return mat_to_quat(sixd_to_mat(sixd))


def quat_to_sixd(quat: np.ndarray or torch.Tensor) -> np.ndarray or torch.Tensor:
    r = Rotation.from_quat(quat.reshape(-1, 4))
    m = r.as_matrix().reshape((*quat.shape[:-1], 3, 3))
    sixd = m[..., :2].swapaxes(-2, -1).reshape(*quat.shape[:-1], 6)
    if isinstance(quat, torch.Tensor):
        sixd = torch.as_tensor(sixd, device=quat.device, dtype=quat.dtype)
    return sixd


def sixd_to_quat(sixd: np.ndarray or torch.Tensor) -> np.ndarray or torch.Tensor:
    if isinstance(sixd, torch.Tensor):
        _sixd = sixd.detach().cpu().numpy()
    else:
        _sixd = sixd * 1
    _sixd = _sixd.reshape((*_sixd.shape[:-1], 2, 3)).swapaxes(-2, -1)
    m = np.concatenate([_sixd, np.cross(_sixd[..., 0], _sixd[..., 1], axis=-1)[..., None]], axis=-1)
    r = Rotation.from_matrix(m.reshape((-1, 3, 3)))
    quat = r.as_quat().reshape(*sixd.shape[:-1], 4)
    if isinstance(sixd, torch.Tensor):
        quat = torch.as_tensor(quat, device=sixd.device, dtype=sixd.dtype)
    return quat


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


def interpolate_xyzquat(keypoints, stride):
    # (n_songs, n_cands, n_frames, ...)
    # use n_frames as n_keypoints
    device = keypoints.device
    if keypoints.shape[2] == 1:
        return keypoints[:, :, :, None].repeat_interleave(stride, -3)

    # Lerp
    lefts = keypoints[:, :, :-1]
    rights = keypoints[:, :, 1:]
    # lefts = lefts.reshape((lefts.shape[0], lefts.shape[1], lefts.shape[2], 3, -1))
    # rights = rights.reshape((rights.shape[0], rights.shape[1], rights.shape[2], 3, -1))
    left_xyz = lefts[..., :3]
    right_xyz = rights[..., :3]
    lerp_t = torch.linspace(0, 1, stride + 1, device=device)[1:]
    interpolated_xyz = left_xyz[:, :, :, None] * (1 - lerp_t[None, None, None, :, None, None]) + right_xyz[:, :, :, None] * lerp_t[None, None, None, :, None, None]

    # Slerp
    left_quat = lefts[..., 3:]
    right_quat = rights[..., 3:]
    slerp_t = torch.linspace(0, 1, stride + 1, device=device)[1:]
    interpolated_quat = slerp(left_quat[:, :, :, None], right_quat[:, :, :, None], slerp_t[None, None, None, :, None, None])
    interpolated = torch.cat([interpolated_xyz, interpolated_quat], dim=-1)
    interpolated = interpolated.flatten(2, 3)

    return interpolated


def interpolate_xyzsixd(keypoints, stride):
    # (n_songs, n_cands, n_frames, ...)
    # use n_frames as n_keypoints
    device = keypoints.device
    if keypoints.shape[2] == 1:
        return keypoints[:, :, :, None].repeat_interleave(stride, -3)
    # Lerp
    lefts = keypoints[:, :, :-1]
    rights = keypoints[:, :, 1:]
    lefts = lefts.reshape((lefts.shape[0], lefts.shape[1], lefts.shape[2], 3, -1))
    rights = rights.reshape((rights.shape[0], rights.shape[1], rights.shape[2], 3, -1))
    left_xyz = lefts[..., :3]
    right_xyz = rights[..., :3]
    # slopes = (right_xyz - left_xyz)
    lerp_t = torch.linspace(0, 1, stride + 1, device=device)[1:]
    # lerp_t = torch.zeros_like(lerp_t)
    # lerp_t = torch.ones_like(lerp_t)
    interpolated_xyz = left_xyz[:, :, :, None] * (1 - lerp_t[None, None, None, :, None, None]) + right_xyz[:, :, :, None] * lerp_t[None, None, None, :, None, None]
    # Slerp
    left_sixd = lefts[..., 3:]
    right_sixd = rights[..., 3:]
    left_quat = sixd_to_quat(left_sixd)
    right_quat = sixd_to_quat(right_sixd)
    slerp_t = torch.linspace(0, 1, stride + 1, device=device)[1:]
    # slerp_t = torch.zeros_like(slerp_t)
    # slerp_t = torch.ones_like(slerp_t)
    interpolated_quat = slerp(left_quat[:, :, :, None], right_quat[:, :, :, None], slerp_t[None, None, None, :, None, None])
    interpolated_sixd = quat_to_sixd(interpolated_quat.detach().cpu()).to(device=device)
    interpolated = torch.cat([interpolated_xyz, interpolated_sixd], dim=-1)
    # cat_me = lefts[:, :, [[0]]].repeat_interleave(stride, -3)
    # interpolated = torch.cat([cat_me, interpolated], dim=2)
    interpolated = interpolated.flatten(2, 3).flatten(-2, -1)
    return interpolated
