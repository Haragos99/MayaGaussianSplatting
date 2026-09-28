import torch

def quaternion_to_rotation_matrix(q: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Torch version (N, 4) quaternions -> (N, 3, 3) rotation matrices."""
    if normalize:
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    w, x, y, z = q.unbind(dim=-1)

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    rows = torch.stack(
        [
            1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
            2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
            2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
        ],
        dim=-1,
    )

    return rows.reshape(*q.shape[:-1], 3, 3)
