import torch

from .densification import (
    PRUNE_WARNING_FRACTION,
    DensificationConfig,
    DensificationStats,
    densify_and_prune,
    prune,
    reset_opacity,
    should_densify,
    should_reset_opacity,
)
from .model_torch import TorchGaussianModel

"""`DefaultStrategy` is 3DGS adaptive density control (densification.py).
`MCMCStrategy` (strategy_mcmc.py) is 3DGS-as-MCMC."""
class DefaultStrategy:
    """3DGS adaptive density control: clone / split by gradient, prune, opacity reset."""

    name = "default"

    def __init__(
        self,
        model: TorchGaussianModel,
        optimizer: torch.optim.Optimizer,
        config: DensificationConfig,
        extent: float,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.extent = extent
        self.stats = DensificationStats(len(model), model.device)

    def describe(self) -> str:
        return self.config.describe()

    def regularization(self, model: TorchGaussianModel) -> torch.Tensor | float:
        return 0.0

    def after_backward(self, iteration: int, projected, camera) -> None:
        if iteration < self.config.end_iteration:
            self.stats.add(projected, camera.width, camera.height)

    @torch.no_grad()
    def after_step(self, iteration: int, lr_xyz: float) -> list[str]:
        messages: list[str] = []
        if should_densify(iteration, self.config):
            result = densify_and_prune(
                self.model, self.optimizer, self.stats, self.config, self.extent, iteration
            )
            messages.append(
                f"iteration {iteration}: densify {result['before']} -> {result['after']} splats "
                f"(clone {result['cloned']}, split {result['split']}, pruned {result['pruned']} "
                f"[opacity {result['opacity_pruned']}, screen {result['screen_pruned']}, "
                f"world {result['world_pruned']}])"
            )
            if result["before"] and result["pruned"] > PRUNE_WARNING_FRACTION * result["before"]:
                messages.append(
                    f"WARNING iteration {iteration}: one round pruned {result['pruned']:,} of "
                    f"{result['before']:,} splats ({result['pruned'] / result['before']:.0%}). "
                    "Large holes usually follow; check the size-pruning settings."
                )
        if should_reset_opacity(iteration, self.config):
            count = reset_opacity(self.model, self.optimizer, self.config.opacity_reset_value)
            messages.append(
                f"iteration {iteration}: opacity reset, {count} splats capped at "
                f"{self.config.opacity_reset_value:g}"
            )
        return messages

    def on_population_change(self) -> None:
        """Called after anything outside the strategy added or removed splats."""
        self.stats.reset(len(self.model), self.model.device)


class NoStrategy:
    """Fixed set of splats (used for the fine-tune after contribution pruning)."""

    name = "none"

    def describe(self) -> str:
        return "Densification: off"

    def regularization(self, model: TorchGaussianModel) -> float:
        return 0.0

    def after_backward(self, iteration: int, projected, camera) -> None:
        pass

    def after_step(self, iteration: int, lr_xyz: float) -> list[str]:
        return []

    def on_population_change(self) -> None:
        pass



# Contribution pruning (LightGaussian, Fan et al. 2024)
@torch.no_grad()
def splat_contribution(model: TorchGaussianModel, views, render) -> torch.Tensor:
    """(N,) sum of T * alpha over every pixel of every view: how much each
    splat actually shows up in the training images."""
    contribution = torch.zeros(len(model), device=model.device)
    for view in views:
        render(view.camera, model, contribution=contribution)
    return contribution


@torch.no_grad()
def prune_by_contribution(
    model: TorchGaussianModel,
    optimizer: torch.optim.Optimizer,
    contribution: torch.Tensor,
    fraction: float,
) -> int:
    """Remove the `fraction` of splats with the smallest contribution."""
    if not 0.0 <= fraction < 1.0:
        raise ValueError("prune fraction must be in [0, 1)")
    count = int(len(model) * fraction)
    if count == 0:
        return 0
    lowest = torch.argsort(contribution)[:count]
    keep = torch.ones(len(model), dtype=torch.bool, device=model.device)
    keep[lowest] = False
    return prune(model, optimizer, keep)



# Splat statistics for the log
@torch.no_grad()
def splat_statistics(model: TorchGaussianModel, extent: float) -> dict[str, float]:
    """Is the model getting too many / too large / too transparent splats?"""
    largest = model.scale.amax(dim=-1) / extent
    opacity = model.opacity
    return {
        "splats": len(model),
        "scale_median": float(largest.median()),
        "scale_p95": float(torch.quantile(largest.float(), 0.95)) if len(model) else 0.0,
        "low_opacity_fraction": float((opacity < 0.1).float().mean()),
    }
