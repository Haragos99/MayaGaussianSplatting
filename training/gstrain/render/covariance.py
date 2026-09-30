import torch

from ..geometry.camera import TorchCamera
from ..geometry.quaternion import quaternion_to_rotation_matrix

def covariance_3d(scale: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """(N, 3) scales + (N, 4) quaternions -> (N, 3, 3) world covariances."""
    R = quaternion_to_rotation_matrix(rotation)

    # R @ diag(s^2) scales column j of R by s_j^2.
    RS = R * scale.square().unsqueeze(-2)

    return RS @ R.transpose(-1, -2)


def perspective_jacobian(
    camera: TorchCamera,
    points_camera: torch.Tensor,
    near: float = 0.01,
) -> torch.Tensor:
    """(N, 3) camera points -> (N, 2, 3) local linearisation of the projection."""
    x, y, z = points_camera.unbind(dim=-1)

    z = z.clamp_min(near)
    z2 = z * z
    zero = torch.zeros_like(z)

    row_u = torch.stack([camera.fx / z, zero, -camera.fx * x / z2], dim=-1)
    row_v = torch.stack([zero, camera.fy / z, -camera.fy * y / z2], dim=-1)

    return torch.stack([row_u, row_v], dim=-2)

def covariance_2d(
    camera: TorchCamera,
    points_camera: torch.Tensor,
    cov3d: torch.Tensor,
    near: float = 0.01,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """World covariances -> (N, 2, 2) screen-space covariances in pixel units."""
    R = camera.R

    cov_camera = R @ cov3d @ R.transpose(-1, -2)

    J = perspective_jacobian(camera, points_camera, near=near)
    cov = J @ cov_camera @ J.transpose(-1, -2)

    # Keeps thin/degenerate splats invertible and at least sub-pixel wide.
    eye = torch.eye(2, dtype=cov.dtype, device=cov.device)

    return cov + epsilon * eye


def covariance_to_conic(cov2d: torch.Tensor, epsilon: float = 1e-12) -> tuple[torch.Tensor, torch.Tensor]:
    """(N, 2, 2) -> conic (N, 3) as [A, B, C] and the determinant (N,).

    Closed-form inverse of a symmetric 2x2 - no torch.linalg.inv needed.
    """
    a = cov2d[..., 0, 0]
    b = cov2d[..., 0, 1]
    c = cov2d[..., 1, 1]

    det = a * c - b * b
    inv_det = 1.0 / det.clamp_min(epsilon)

    conic = torch.stack([c * inv_det, -b * inv_det, a * inv_det], dim=-1)

    return conic, det
