import math
from dataclasses import dataclass, field
from typing import Callable

import torch
from tqdm.auto import tqdm

from .geometry.camera import TorchCamera
from .densification import (
    DensificationStats,
    DensificationConfig,
    densify_and_prune,
    reset_opacity,
    should_densify,
    should_reset_opacity,
)
from .loss import photometric_loss
from .model_torch import TorchGaussianModel
from .render.renderer_torch import render_gaussians, render_gaussians_with_aux
from .dataset import TrainView




@dataclass
class TrainConfig:
    iterations: int = 7_000

    # Multiplied by the scene extent - see create_optimizer.
    position_lr_init: float = 1.6e-4
    position_lr_final: float = 1.6e-6

    scale_lr: float = 5e-3
    rotation_lr: float = 1e-3
    opacity_lr: float = 5e-2
    color_lr: float = 2.5e-3

    lambda_dssim: float = 0.2

    tile_size: int = 16
    tile_chunk: int = 64
    max_per_tile: int | None = 256

    seed: int = 0
    densification: DensificationConfig = field(default_factory=DensificationConfig)


def scene_extent(views: list[TrainView]) -> float:
    """Radius of the camera rig, used to put the position LR in world units."""
    centers = torch.stack([view.camera.camera_center for view in views])
    radius = (centers - centers.mean(dim=0)).norm(dim=-1).max()

    extent = float(radius) * 1.1
    if extent <= 0.0:
        raise ValueError(
            "camera centres coincide, so the scene scale is unknown - "
            "pass extent= explicitly (a zero extent would freeze all positions)"
        )

    return extent


def exponential_lr(step: int, lr_init: float, lr_final: float, max_steps: int) -> float:
    """Log-space interpolation, so the LR decays by a constant *factor*."""
    if lr_init <= 0.0 or lr_final <= 0.0:
        return 0.0

    t = min(max(step / max(max_steps, 1), 0.0), 1.0)

    return math.exp(math.log(lr_init) * (1.0 - t) + math.log(lr_final) * t)

"Optimaz the training with Aam "
def create_optimizer(
    model: TorchGaussianModel,
    config: TrainConfig,
    extent: float,
) -> torch.optim.Adam:
    return torch.optim.Adam(
        [
            {"params": [model.xyz], "lr": config.position_lr_init * extent, "name": "xyz"},
            {"params": [model.scale_raw], "lr": config.scale_lr, "name": "scale"},
            {"params": [model.rotation], "lr": config.rotation_lr, "name": "rotation"},
            {"params": [model.opacity_raw], "lr": config.opacity_lr, "name": "opacity"},
            {"params": [model.color_raw], "lr": config.color_lr, "name": "color"},
        ],
        eps=1e-15,
    )


def update_position_lr(
    optimizer: torch.optim.Adam,
    config: TrainConfig,
    extent: float,
    step: int,
) -> float:
    """Only the position LR decays; the others stay fixed, as in 3DGS."""
    lr = exponential_lr(
        step,
        config.position_lr_init * extent,
        config.position_lr_final * extent,
        config.iterations,
    )

    for group in optimizer.param_groups:
        if group["name"] == "xyz":
            group["lr"] = lr

    return lr


def render_view(
    model: TorchGaussianModel,
    view: TrainView,
    config: TrainConfig,
) -> torch.Tensor:
    return render_gaussians(
        view.camera,
        model,
        tile_size=config.tile_size,
        tile_chunk=config.tile_chunk,
        max_per_tile=config.max_per_tile,
    )


def train_step(
    model: TorchGaussianModel,
    views:  list[TrainView],
    optimizer: torch.optim.Adam,
    config: TrainConfig,
    densification_stats: DensificationStats | None = None,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    mean_loss = 0.0

    for current_view in views:
        target = current_view.image.to(model.device, non_blocking=True)
        if densification_stats is None:
            rendered = render_view(model, current_view, config)
        else:
            rendered, projected = render_gaussians_with_aux(
                current_view.camera,
                model,
                tile_size=config.tile_size,
                tile_chunk=config.tile_chunk,
                max_per_tile=config.max_per_tile,
            )
        loss = photometric_loss(rendered, target, config.lambda_dssim)
        loss.backward()
        if densification_stats is not None:
            densification_stats.add(projected, current_view.camera.width, current_view.camera.height)
        mean_loss += loss.detach() / len(views)

    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(len(views))

    optimizer.step()

    return float(mean_loss)


def train(
    model: TorchGaussianModel,
    views: list[TrainView],
    config: TrainConfig | None = None,
    extent: float | None = None,
    on_iteration: Callable[[int, float], None] | None = None,
) -> list[float]:
    config = config or TrainConfig()
    if not views:
        raise ValueError("training requires at least one view")
    extent = scene_extent(views) if extent is None else extent

    optimizer = create_optimizer(model, config, extent)
    densification_stats = DensificationStats(len(model), model.device)

    #store the loss of the training
    history: list[float] = []

    # Training Loop
    progress = tqdm(range(config.iterations), desc="Training", unit="iter")
    for step in progress:
        update_position_lr(optimizer, config, extent, step)

        iteration = step + 1
        collect_stats = iteration < config.densification.end_iteration
        loss = train_step(
            model,
            views,
            optimizer,
            config,
            densification_stats if collect_stats else None,
        )
        history.append(loss)
        if should_densify(iteration, config.densification):
            result = densify_and_prune(
                model,
                optimizer,
                densification_stats,
                config.densification,
                extent,
                iteration,
            )
            progress.write(
                f"Densification at iteration {iteration}: Gaussians "
                f"{result['before']} -> {result['after']} "
                f"(candidates clone={result['clone_candidates']}, split={result['split_candidates']}; "
                f"added clone={result['cloned']}, split={result['split']}; "
                f"pruned={result['pruned']} [opacity={result['opacity_pruned']}, "
                f"screen={result['screen_pruned']}, world={result['world_pruned']}])"
            )
        if should_reset_opacity(iteration, config.densification):
            reset_count = reset_opacity(
                model, optimizer, config.densification.opacity_reset_value
            )
            progress.write(
                f"Opacity reset at iteration {iteration}: "
                f"{reset_count} splats capped at {config.densification.opacity_reset_value:g}"
            )
        progress.set_postfix(loss=f"{loss:.4f}", splats=len(model))
        if on_iteration is not None:
            on_iteration(step, loss)

    return history
