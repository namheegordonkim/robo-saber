import numpy as np
import torch


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
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


def unity_to_zup(xyz: np.ndarray | torch.Tensor, quat: np.ndarray | torch.Tensor) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
    my_pos = xyz
    my_pos[..., [0, 1, 2]] = my_pos[..., [2, 0, 1]]
    my_pos[..., 1] *= -1

    my_quat = quat
    my_quat[..., [0, 1, 2, 3]] = my_quat[..., [2, 0, 1, 3]]
    my_quat[..., [0, 2]] *= -1

    return my_pos, my_quat
