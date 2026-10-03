import math
import random
from dataclasses import dataclass
from typing import Callable

import torch
from tqdm.auto import tqdm

from .geometry.camera import TorchCamera
from .loss import photometric_loss
from .model_torch import TorchGaussianModel
from .render.renderer_torch import render_gaussians

"Store the data of the camera and tatget img "
@dataclass
class TrainView:
    camera: TorchCamera
    image: torch.Tensor  # (H, W, 3) in [0, 1]


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
    view: TrainView,
    optimizer: torch.optim.Adam,
    config: TrainConfig,
) -> float:
    optimizer.zero_grad(set_to_none=True)

    target = view.image.to(model.device, non_blocking=True)
    loss = photometric_loss(render_view(model, view, config), target, config.lambda_dssim)

    loss.backward()
    optimizer.step()

    return float(loss.detach())


def train(
    model: TorchGaussianModel,
    views: list[TrainView],
    config: TrainConfig | None = None,
    extent: float | None = None,
    on_iteration: Callable[[int, float], None] | None = None,
) -> list[float]:
    config = config or TrainConfig()
    extent = scene_extent(views) if extent is None else extent

    optimizer = create_optimizer(model, config, extent)
    rng = random.Random(config.seed)

    #store the loss of the training
    history: list[float] = []

    # Training Loop
    progress = tqdm(range(config.iterations), desc="Training", unit="iter")
    for step in progress:
        update_position_lr(optimizer, config, extent, step)

        loss = train_step(model, rng.choice(views), optimizer, config)
        history.append(loss)
        progress.set_postfix(loss=f"{loss:.4f}")
        if on_iteration is not None:
            on_iteration(step, loss)

    return history
