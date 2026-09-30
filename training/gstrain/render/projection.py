"""Batched world -> camera -> image projection (guide sections 10 and 11)."""

import torch

from ..geometry.camera import TorchCamera


def world_to_camera(camera: TorchCamera, xyz: torch.Tensor) -> torch.Tensor:
    """(N, 3) world points -> (N, 3) camera points, COLMAP convention X_c = R X_w + t."""
    return xyz @ camera.R.transpose(-1, -2) + camera.t


def project_points_Torch(
    camera: TorchCamera,
    points_camera: torch.Tensor,
    near: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(N, 3) camera points -> pixel means (N, 2), depths (N,), visibility mask (N,)."""
    x, y, z = points_camera.unbind(dim=-1)

    visible = z > near

    # Clamping keeps the divide (and its gradient) finite for culled Gaussians.
    z_safe = z.clamp_min(near)

    u = camera.fx * x / z_safe + camera.cx
    v = camera.fy * y / z_safe + camera.cy

    return torch.stack([u, v], dim=-1), z, visible
