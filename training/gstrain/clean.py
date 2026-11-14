from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .dataset import TrainView
from .model_torch import PARAMETER_ATTRIBUTES, TorchGaussianModel
from .render.renderer_torch import render_gaussians
from .strategy import splat_contribution


@dataclass
class CleanResult:
    model: TorchGaussianModel
    keep: torch.Tensor                      # (N_original,) bool
    removed: dict[str, int] = field(default_factory=dict)
    focus: torch.Tensor | None = None       # (3,) world point the cameras look at
    crop_radius: float | None = None        # world units, when cropping


@torch.no_grad()
def scene_focus(views: list[TrainView]) -> torch.Tensor:
    """(3,) the point closest (least squares) to every camera's viewing axis.

    For a capture around a subject this is the subject. Each camera looks
    along +z of its own frame, i.e. world direction R[2].
    """
    device = views[0].camera.R.device
    normal = torch.zeros(3, 3, device=device, dtype=torch.float64)
    rhs = torch.zeros(3, device=device, dtype=torch.float64)
    for view in views:
        axis = view.camera.R[2].double()
        projector = torch.eye(3, device=device, dtype=torch.float64) - torch.outer(axis, axis)
        normal += projector
        rhs += projector @ view.camera.camera_center.double()
    return torch.linalg.solve(normal, rhs).float()


@torch.no_grad()
def orbit_camera(camera, focus: torch.Tensor, degrees: float, views: list[TrainView]):
    """`camera` swung `degrees` around the subject, about the rig's up axis.

    Shows the model from a position no photo was taken from - roughly what an
    orbit in Maya shows.
    """
    from .geometry.camera import TorchCamera

    if degrees == 0:
        return camera
    # COLMAP cameras have +y pointing down, so "up" is minus their mean y axis.
    up = -torch.stack([view.camera.R[1] for view in views]).mean(dim=0)
    up = up / up.norm()
    angle = torch.tensor(float(degrees), device=up.device).deg2rad()
    cross = torch.tensor(
        [[0.0, -up[2], up[1]], [up[2], 0.0, -up[0]], [-up[1], up[0], 0.0]], device=up.device
    )
    rotation = torch.eye(3, device=up.device) + angle.sin() * cross + (1 - angle.cos()) * cross @ cross
    center = focus + rotation @ (camera.camera_center - focus)
    world_to_camera = camera.R @ rotation.T
    return TorchCamera(
        width=camera.width, height=camera.height,
        fx=camera.fx, fy=camera.fy, cx=camera.cx, cy=camera.cy,
        R=world_to_camera, t=-world_to_camera @ center,
    )


def select(model: TorchGaussianModel, keep: torch.Tensor) -> TorchGaussianModel:
    """A new model holding only the splats where `keep` is true."""
    keep = keep.to(model.device)
    count = int(keep.sum())
    device = model.device
    subset = TorchGaussianModel(
        xyz=torch.zeros(count, 3, device=device),
        scale=torch.ones(count, 3, device=device),
        rotation=torch.zeros(count, 4, device=device),
        opacity=torch.full((count,), 0.5, device=device),
        color=torch.zeros(count, 3, device=device),
        sh_degree=model.max_sh_degree,
    )
    for attribute in PARAMETER_ATTRIBUTES.values():
        setattr(subset, attribute, nn.Parameter(getattr(model, attribute).detach()[keep].clone()))
    subset.active_sh_degree = model.active_sh_degree
    return subset


def strongest_share(contribution: torch.Tensor, share: float) -> torch.Tensor:
    """(N,) bool: the fewest splats whose contribution adds up to `share` of the total."""
    keep = torch.zeros_like(contribution, dtype=torch.bool)
    total = float(contribution.sum())
    if total <= 0.0:
        return keep
    order = torch.argsort(contribution, descending=True)
    cumulative = torch.cumsum(contribution[order], dim=0) / total
    count = int(torch.searchsorted(cumulative, torch.tensor(share, device=cumulative.device))) + 1
    keep[order[:min(count, len(order))]] = True
    return keep & (contribution > 0)


@torch.no_grad()
def clean_model(
    model: TorchGaussianModel,
    views: list[TrainView],
    min_opacity: float = 0.05,
    keep_contribution: float = 0.99,
    crop_radius: float | None = None,
    render_options: dict | None = None,
) -> CleanResult:
    """Apply the filters; `views` are the training views (contribution, focus)."""
    if not 0.0 < keep_contribution <= 1.0:
        raise ValueError("keep_contribution must be in (0, 1]")
    render_options = render_options or {}
    n = len(model)
    keep = torch.ones(n, dtype=torch.bool, device=model.device)
    removed: dict[str, int] = {}

    if min_opacity > 0.0:
        transparent = model.opacity < min_opacity
        removed["opacity"] = int((transparent & keep).sum())
        keep &= ~transparent

    if keep_contribution < 1.0:
        contribution = splat_contribution(
            model,
            views,
            lambda camera, m, contribution: render_gaussians(
                camera, m, contribution=contribution, **render_options
            ),
        )
        weak = ~strongest_share(contribution, keep_contribution)
        removed["contribution"] = int((weak & keep).sum())
        keep &= ~weak

    focus = scene_focus(views)
    radius = None
    if crop_radius is not None and crop_radius > 0.0:
        camera_distance = torch.stack(
            [(view.camera.camera_center - focus).norm() for view in views]
        ).median()
        radius = float(crop_radius * camera_distance)
        outside = (model.xyz - focus).norm(dim=-1) > radius
        removed["crop"] = int((outside & keep).sum())
        keep &= ~outside

    return CleanResult(select(model, keep), keep, removed, focus, radius)
