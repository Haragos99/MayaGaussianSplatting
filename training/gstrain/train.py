import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable

import torch
from tqdm.auto import tqdm

from .dataset import TrainView
from .densification import DensificationConfig
from .loss import l1_loss, photometric_loss, psnr, ssim
from .model_torch import TorchGaussianModel
from .render.renderer_torch import render_gaussians, render_gaussians_with_aux
from .run_log import RunLog
from .strategy import (
    DefaultStrategy,
    NoStrategy,
    prune_by_contribution,
    splat_contribution,
    splat_statistics,
)
from .strategy_mcmc import MCMCConfig, MCMCStrategy

BACKGROUNDS = ("black", "white", "random")
STRATEGIES = ("default", "mcmc")


@dataclass
class TrainConfig:
    iterations: int = 7_000

    # Multiplied by the scene extent - see create_optimizer.
    position_lr_init: float = 1.6e-4
    position_lr_final: float = 1.6e-6

    scale_lr: float = 5e-3
    rotation_lr: float = 1e-3
    opacity_lr: float = 5e-2
    # SH band 0 (the colour); higher bands use a 20x smaller rate, as in 3DGS.
    color_lr: float = 2.5e-3

    lambda_dssim: float = 0.2

    tile_size: int = 16
    tile_chunk: int = 64
    # No hard cap: a cap drops visible splats in crowded tiles (square seams).
    max_per_tile: int | None = None
    depth_chunk: int = 256

    # Composited behind the splats. "random" draws a new colour every
    # iteration, so splats cannot hide transparency behind a fixed background.
    background: str = "black"
    # One more SH band every this many iterations, up to the model's degree.
    sh_degree_interval: int = 1_000

    strategy: str = "default"
    densification: DensificationConfig = field(default_factory=DensificationConfig)
    mcmc: MCMCConfig = field(default_factory=MCMCConfig)

    # LightGaussian-style cleanup after training: drop this fraction of the
    # least-contributing splats, then fine-tune without densification.
    prune_contribution: float = 0.0
    prune_finetune: int = 500

    eval_every: int = 500
    log_every: int = 10
    stats_every: int = 100
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
            {"params": [model.sh_dc], "lr": config.color_lr, "name": "f_dc"},
            {"params": [model.sh_rest], "lr": config.color_lr / 20.0, "name": "f_rest"},
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


def background_color(name: str, device: torch.device, generator: torch.Generator | None = None) -> torch.Tensor:
    if name == "black":
        return torch.zeros(3, device=device)
    if name == "white":
        return torch.ones(3, device=device)
    if name == "random":
        return torch.rand(3, device=device, generator=generator)
    raise ValueError(f"background must be one of {BACKGROUNDS}, got {name!r}")


def render_options(config: TrainConfig) -> dict:
    return {
        "tile_size": config.tile_size,
        "tile_chunk": config.tile_chunk,
        "max_per_tile": config.max_per_tile,
        "depth_chunk": config.depth_chunk,
    }


def render_view(
    model: TorchGaussianModel,
    view: TrainView,
    config: TrainConfig,
    background: torch.Tensor | None = None,
) -> torch.Tensor:
    return render_gaussians(view.camera, model, background=background, **render_options(config))


@torch.no_grad()
def evaluate(
    model: TorchGaussianModel,
    views: list[TrainView],
    config: TrainConfig,
) -> dict[str, float]:
    """Mean PSNR / SSIM / L1 over `views` (normally the held-out test views).

    Random backgrounds are a training aid only; evaluation uses black.
    """
    name = "black" if config.background == "random" else config.background
    background = background_color(name, model.device)
    totals = {"psnr": 0.0, "ssim": 0.0, "l1": 0.0}
    for view in views:
        rendered = render_view(model, view, config, background)
        target = view.image.to(model.device)
        totals["psnr"] += float(psnr(rendered, target))
        totals["ssim"] += float(ssim(rendered, target))
        totals["l1"] += float(l1_loss(rendered, target))
    return {key: value / max(len(views), 1) for key, value in totals.items()}


def build_strategy(model, optimizer, config: TrainConfig, extent: float):
    if config.strategy == "default":
        return DefaultStrategy(model, optimizer, config.densification, extent)
    if config.strategy == "mcmc":
        return MCMCStrategy(model, optimizer, config.mcmc, config.iterations)
    raise ValueError(f"strategy must be one of {STRATEGIES}, got {config.strategy!r}")


class ViewSampler:
    """Every training view once per epoch, in a new random order each epoch."""

    def __init__(self, views: list[TrainView], seed: int = 0) -> None:
        self.views = list(views)
        self.random = random.Random(seed)
        self.stack: list[TrainView] = []

    def next(self) -> TrainView:
        if not self.stack:
            self.stack = list(self.views)
            self.random.shuffle(self.stack)
        return self.stack.pop()


def train(
    model: TorchGaussianModel,
    views: list[TrainView],
    config: TrainConfig | None = None,
    extent: float | None = None,
    on_iteration: Callable[[int, float], None] | None = None,
    test_views: list[TrainView] | None = None,
    run_log: RunLog | None = None,
    checkpoint: Callable[[int, TorchGaussianModel, torch.optim.Optimizer], None] | None = None,
    checkpoint_every: int = 0,
    resume: dict | None = None,
) -> list[float]:
    """Optimise `model` on `views`. Returns the per-iteration training loss.

    `on_iteration(step, loss)` is called after every iteration (step is
    0-based). `test_views` are evaluated every `config.eval_every` iterations
    and at the end. `resume` is a checkpoint dict from checkpoint.load_checkpoint
    (model parameters must already be restored into `model`).
    """
    config = config or TrainConfig()
    if not views:
        raise ValueError("training requires at least one view")
    extent = scene_extent(views + list(test_views or [])) if extent is None else extent

    optimizer = create_optimizer(model, config, extent)
    start_iteration = 1
    if resume is not None:
        optimizer.load_state_dict(resume["optimizer"])
        start_iteration = int(resume["iteration"]) + 1
        model.active_sh_degree = int(resume.get("active_sh_degree", 0))

    strategy = build_strategy(model, optimizer, config, extent)
    sampler = ViewSampler(views, config.seed)
    background_generator = torch.Generator(device=model.device)
    background_generator.manual_seed(config.seed)

    history: list[float] = []
    started = time.time()
    last_test: dict[str, float] = {}

    total = config.iterations + (config.prune_finetune if config.prune_contribution > 0 else 0)
    progress = tqdm(
        range(start_iteration, total + 1), desc="Training", unit="iter",
        initial=start_iteration - 1, total=total,
    )
    progress.write(strategy.describe())

    def note(message: str) -> None:
        progress.write(message)
        if run_log is not None:
            run_log.log({"iteration": iteration, "event": message})

    def run_eval() -> None:
        nonlocal last_test
        if not test_views:
            return
        last_test = evaluate(model, test_views, config)
        record = {"iteration": iteration, "splats": len(model), **{f"test_{k}": v for k, v in last_test.items()}}
        if run_log is not None:
            run_log.log(record)
        progress.write(
            f"iteration {iteration}: test PSNR {last_test['psnr']:.2f} dB, "
            f"SSIM {last_test['ssim']:.3f}, {len(model):,} splats"
        )

    iteration = start_iteration - 1
    try:
        for iteration in progress:
            finetuning = iteration > config.iterations
            if iteration == config.iterations + 1:
                # Training proper is over: prune by contribution, then fine-tune.
                contribution = splat_contribution(
                    model, views,
                    lambda camera, m, contribution: render_gaussians(
                        camera, m, contribution=contribution, **render_options(config)
                    ),
                )
                removed = prune_by_contribution(model, optimizer, contribution, config.prune_contribution)
                strategy = NoStrategy()
                note(
                    f"iteration {iteration}: pruned {removed:,} least-contributing splats "
                    f"({config.prune_contribution:.0%}), fine-tuning {config.prune_finetune} iterations "
                    f"on {len(model):,}"
                )

            lr_xyz = update_position_lr(optimizer, config, extent, min(iteration, config.iterations))
            if (
                config.sh_degree_interval > 0
                and iteration % config.sh_degree_interval == 0
                and model.oneup_sh_degree()
            ):
                note(f"iteration {iteration}: SH degree -> {model.active_sh_degree}")

            view = sampler.next()
            target = view.image.to(model.device, non_blocking=True)
            background = background_color(config.background, model.device, background_generator)

            optimizer.zero_grad(set_to_none=True)
            rendered, projected = render_gaussians_with_aux(
                view.camera, model, background=background, **render_options(config)
            )
            loss = photometric_loss(rendered, target, config.lambda_dssim)
            loss = loss + strategy.regularization(model)
            loss.backward()

            strategy.after_backward(iteration, projected, view.camera)
            optimizer.step()
            for message in strategy.after_step(iteration, lr_xyz):
                note(message)

            loss_value = float(loss.detach())
            history.append(loss_value)

            if run_log is not None and (iteration % config.log_every == 0 or iteration == total):
                record = {
                    "iteration": iteration,
                    "loss": loss_value,
                    "psnr": float(psnr(rendered.detach(), target)),
                    "splats": len(model),
                    "lr_xyz": lr_xyz,
                    "sh_degree": model.active_sh_degree,
                    "elapsed_s": round(time.time() - started, 2),
                }
                if iteration % config.stats_every == 0:
                    record.update(splat_statistics(model, extent))
                run_log.log(record)
                run_log.status("running", iteration=iteration, total=total, splats=len(model))

            if config.eval_every > 0 and iteration % config.eval_every == 0 and iteration != total:
                run_eval()
            if checkpoint is not None and checkpoint_every > 0 and iteration % checkpoint_every == 0:
                checkpoint(iteration, model, optimizer)

            postfix = {"loss": f"{loss_value:.4f}", "splats": len(model)}
            if last_test:
                postfix["test_psnr"] = f"{last_test['psnr']:.2f}"
            if finetuning:
                postfix["phase"] = "finetune"
            progress.set_postfix(postfix)

            if on_iteration is not None:
                on_iteration(iteration - 1, loss_value)

        run_eval()
    except KeyboardInterrupt:
        if run_log is not None:
            run_log.status("cancelled", iteration=iteration)
        raise
    finally:
        progress.close()

    return history


def last_test_metrics(model: TorchGaussianModel, test_views: list[TrainView], config: TrainConfig) -> dict:
    """Convenience for callers that want the final numbers without a run log."""
    return evaluate(model, test_views, config) if test_views else {}
