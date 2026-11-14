from dataclasses import dataclass

import torch

from ..geometry.camera import TorchCamera
from ..model_torch import TorchGaussianModel
from .binning import build_tile_bins, cutoff_sigma, screen_radius
from .covariance import covariance_2d, covariance_3d, covariance_to_conic
from .projection import project_points_Torch, world_to_camera
from .rasterizer import ALPHA_MAX, ALPHA_MIN, rasterize_tiles

# 3DGS culls splats closer than 0.2 to the camera: their projection is
# numerically wild and they are almost never meant to be seen.
NEAR = 0.2
# EWA screen-space low-pass filter (pixels^2), as in 3DGS.
DILATION = 0.3


@dataclass
class ProjectedGaussians:
    """Screen-space state for all N Gaussians - tensors, not one object each."""

    mean: torch.Tensor       # (N, 2) pixel centers
    cov2d: torch.Tensor      # (N, 2, 2)
    conic: torch.Tensor      # (N, 3) inverse covariance as [A, B, C]
    radius: torch.Tensor     # (N, 2) pixel radii where alpha reaches the cutoff
    color: torch.Tensor      # (N, 3)
    opacity: torch.Tensor    # (N,)
    depth: torch.Tensor      # (N,)
    visible: torch.Tensor    # (N,) bool


def project_gaussians(
    camera: TorchCamera,
    gaussians: TorchGaussianModel,
    near: float = NEAR,
    dilation: float = DILATION,
    sigma_factor: float | None = None,
    max_radius: float | None = None,
    alpha_min: float = ALPHA_MIN,
) -> ProjectedGaussians:
    """sigma_factor=None sizes each footprint from its opacity (see cutoff_sigma);
    a float forces a fixed sigma box for every splat."""
    points_camera = world_to_camera(camera, gaussians.xyz)
    mean, depth, visible = project_points_Torch(camera, points_camera, near=near)

    cov3d = covariance_3d(gaussians.scale, gaussians.rotation)
    cov2d = covariance_2d(camera, points_camera, cov3d, near=near, dilation=dilation)
    conic, _ = covariance_to_conic(cov2d)

    opacity = gaussians.opacity
    sigma = cutoff_sigma(opacity, alpha_min) if sigma_factor is None else sigma_factor

    return ProjectedGaussians(
        mean=mean,
        cov2d=cov2d,
        conic=conic,
        radius=screen_radius(cov2d.detach(), sigma, max_radius=max_radius),
        color=gaussians.colors_for(camera.camera_center),
        opacity=opacity,
        depth=depth,
        visible=visible,
    )


def render_gaussians(
    camera: TorchCamera,
    gaussians: TorchGaussianModel,
    **kwargs,
) -> torch.Tensor:
    """-> (H, W, 3) image on the camera's device. Options as render_gaussians_with_aux."""
    image, _ = render_gaussians_with_aux(camera, gaussians, **kwargs)
    return image


def render_gaussians_with_aux(
    camera: TorchCamera,
    gaussians: TorchGaussianModel,
    tile_size: int = 16,
    tile_chunk: int = 32,
    near: float = NEAR,
    dilation: float = DILATION,
    sigma_factor: float | None = None,
    max_per_tile: int | None = None,
    max_radius: float | None = None,
    alpha_max: float = ALPHA_MAX,
    alpha_min: float = ALPHA_MIN,
    depth_chunk: int = 256,
    background: torch.Tensor | None = None,
    contribution: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ProjectedGaussians]:
    """Return the rendered image and screen-space data used by training.

    `background` (3,) is composited behind the splats (black when None).
    `contribution` (N,) accumulates each splat's sum(T * alpha) over the image.
    """
    # A footprint wider than the image already covers every tile, so capping
    # here bounds the pair count without changing what is drawn.
    if max_radius is None:
        max_radius = float(max(camera.width, camera.height))

    projected = project_gaussians(
        camera, gaussians, near=near, dilation=dilation, sigma_factor=sigma_factor,
        max_radius=max_radius, alpha_min=alpha_min,
    )
    if projected.mean.requires_grad:
        projected.mean.retain_grad()

    bins = build_tile_bins(
        projected.mean,
        projected.cov2d,
        projected.depth,
        projected.visible,
        camera.width,
        camera.height,
        tile_size=tile_size,
        radius=projected.radius,
    )

    image = rasterize_tiles(
        bins,
        projected.mean,
        projected.conic,
        projected.opacity,
        projected.color,
        camera.width,
        camera.height,
        tile_size=tile_size,
        tile_chunk=tile_chunk,
        max_per_tile=max_per_tile,
        alpha_max=alpha_max,
        alpha_min=alpha_min,
        depth_chunk=depth_chunk,
        background=background,
        contribution=contribution,
    )

    return image.clamp(0.0, 1.0), projected
