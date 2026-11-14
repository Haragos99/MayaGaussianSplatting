from dataclasses import dataclass

import torch

from .densification import _rebuild_parameters
from .geometry.quaternion import quaternion_to_rotation_matrix
from .model_torch import PARAMETER_ATTRIBUTES, TorchGaussianModel

# Relocation splits a target into at most this many copies (gsplat's n_max).
MAX_COPIES = 51

"""3D Gaussian Splatting as Markov Chain Monte Carlo (Kheradmand et al., NeurIPS 2024)."""
@dataclass(frozen=True)
class MCMCConfig:
    cap_max: int = 300_000
    refine_start_iteration: int = 500
    # Stop growing / relocating after this fraction of the run (gsplat: 25k of 30k).
    refine_stop_fraction: float = 0.8
    refine_every: int = 100
    growth_rate: float = 0.05
    min_opacity: float = 0.005
    noise_lr: float = 5e5
    opacity_reg: float = 0.01
    scale_reg: float = 0.01
    seed: int = 0

    # gsplat's MCMC preset starts splats more opaque and smaller than 3DGS.
    initial_opacity: float = 0.5
    initial_scale_multiplier: float = 0.1


def relocation_values(
    opacity: torch.Tensor,
    scale: torch.Tensor,
    copies: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Opacity and scale for each of `copies` coincident splats replacing one.

    o' = 1 - (1 - o)^(1/n)
    s' = s * o / sum_{k=0}^{n-1} (-1)^k C(n, k+1) o'^(k+1) / sqrt(k+1)

    The paper writes the denominator as a double sum over i and k; summing over
    i first (hockey-stick identity) leaves this single sum. For n = 1 both
    values are unchanged.
    """
    n = copies.clamp(1, MAX_COPIES).to(torch.float64)
    o = opacity.to(torch.float64)
    new_opacity = 1.0 - (1.0 - o) ** (1.0 / n)

    k = torch.arange(MAX_COPIES, dtype=torch.float64, device=opacity.device)  # k = 0..50
    n_col = n.unsqueeze(-1)
    valid = k < n_col
    # log C(n, k+1) via lgamma, so large n cannot overflow.
    log_binom = (
        torch.lgamma(n_col + 1) - torch.lgamma(k + 2) - torch.lgamma((n_col - k - 1).clamp_min(0) + 1)
    )
    terms = (
        torch.where(k % 2 == 0, 1.0, -1.0)
        * torch.exp(log_binom)
        * new_opacity.unsqueeze(-1) ** (k + 1)
        / torch.sqrt(k + 1)
    )
    denominator = torch.where(valid, terms, torch.zeros_like(terms)).sum(dim=-1)

    coefficient = o / denominator.clamp_min(1e-12)
    new_scale = scale.to(torch.float64) * coefficient.unsqueeze(-1)
    return new_opacity.to(opacity.dtype), new_scale.to(scale.dtype)


class MCMCStrategy:
    name = "mcmc"

    def __init__(
        self,
        model: TorchGaussianModel,
        optimizer: torch.optim.Optimizer,
        config: MCMCConfig,
        iterations: int,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.stop_iteration = int(iterations * config.refine_stop_fraction)
        self.generator = torch.Generator(device=model.device)
        self.generator.manual_seed(config.seed)

    def describe(self) -> str:
        c = self.config
        return (
            f"Strategy MCMC: cap {c.cap_max:,} splats, +{c.growth_rate:.0%} and relocation every "
            f"{c.refine_every} iterations from {c.refine_start_iteration} to {self.stop_iteration}; "
            f"opacity reg {c.opacity_reg:g}, scale reg {c.scale_reg:g}, noise lr {c.noise_lr:g}"
        )

    def regularization(self, model: TorchGaussianModel) -> torch.Tensor:
        return (
            self.config.opacity_reg * model.opacity.mean()
            + self.config.scale_reg * model.scale.mean()
        )

    def after_backward(self, iteration: int, projected, camera) -> None:
        pass

    def on_population_change(self) -> None:
        pass

    def is_refine_step(self, iteration: int) -> bool:
        c = self.config
        return (
            c.refine_start_iteration < iteration < self.stop_iteration
            and iteration % c.refine_every == 0
        )

    @torch.no_grad()
    def after_step(self, iteration: int, lr_xyz: float) -> list[str]:
        messages: list[str] = []
        if self.is_refine_step(iteration):
            relocated = self.relocate()
            added = self.add()
            messages.append(
                f"iteration {iteration}: MCMC relocated {relocated}, added {added} "
                f"-> {len(self.model):,} splats"
            )
        self.inject_noise(lr_xyz)
        return messages

    # ------------------------------------------------------------------ #

    def _sample_by_opacity(self, candidates: torch.Tensor, count: int) -> torch.Tensor:
        weights = self.model.opacity.detach()[candidates].clamp_min(1e-12)
        picks = torch.multinomial(weights, count, replacement=True, generator=self.generator)
        return candidates[picks]

    def _shared_values(self, sources: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """New raw opacity / scale for sampled sources, given how often each was picked."""
        copies = torch.bincount(sources, minlength=len(self.model))[sources] + 1
        opacity, scale = relocation_values(
            self.model.opacity.detach()[sources], self.model.scale.detach()[sources], copies
        )
        opacity = opacity.clamp(self.config.min_opacity, 1.0 - 1e-6)
        return opacity.logit(), scale.clamp_min(1e-10).log()

    @torch.no_grad()
    def relocate(self) -> int:
        """Move dead splats onto live ones sampled by opacity."""
        opacity = self.model.opacity.detach()
        dead = torch.nonzero(opacity <= self.config.min_opacity, as_tuple=True)[0]
        alive = torch.nonzero(opacity > self.config.min_opacity, as_tuple=True)[0]
        if dead.numel() == 0 or alive.numel() == 0:
            return 0

        sources = self._sample_by_opacity(alive, dead.numel())
        opacity_raw, scale_raw = self._shared_values(sources)

        def build(name: str, tensor: torch.Tensor):
            updated = tensor.clone()
            if name == "opacity":
                updated[sources] = opacity_raw
            elif name == "scale":
                updated[sources] = scale_raw
            updated[dead] = updated[sources]

            def state(moment: torch.Tensor) -> torch.Tensor:
                moment = moment.clone()
                moment[sources] = 0.0
                return moment

            return updated, state

        _rebuild_parameters(self.model, self.optimizer, build)
        return int(dead.numel())

    @torch.no_grad()
    def add(self) -> int:
        """Grow by `growth_rate` (up to the cap) with copies of live splats."""
        current = len(self.model)
        target = min(self.config.cap_max, int((1.0 + self.config.growth_rate) * current))
        count = max(0, target - current)
        if count == 0:
            return 0

        everyone = torch.arange(current, device=self.model.device)
        sources = self._sample_by_opacity(everyone, count)
        opacity_raw, scale_raw = self._shared_values(sources)

        def build(name: str, tensor: torch.Tensor):
            updated = tensor.clone()
            if name == "opacity":
                updated[sources] = opacity_raw
            elif name == "scale":
                updated[sources] = scale_raw
            grown = torch.cat([updated, updated[sources]], dim=0)

            def state(moment: torch.Tensor) -> torch.Tensor:
                moment = moment.clone()
                moment[sources] = 0.0
                return torch.cat([moment, torch.zeros_like(moment[sources])], dim=0)

            return grown, state

        _rebuild_parameters(self.model, self.optimizer, build)
        return count

    @torch.no_grad()
    def inject_noise(self, lr_xyz: float) -> None:
        """xyz += lr * noise_lr * sigmoid(-100 (o - 0.005)) * Sigma @ eta."""
        model = self.model
        opacity = model.opacity.detach()
        gate = torch.sigmoid(-100.0 * (opacity - self.config.min_opacity))
        if float(gate.max()) < 1e-8:
            return

        rotation = quaternion_to_rotation_matrix(model.rotation.detach())
        scale = model.scale.detach()
        covariance = (rotation * scale.square().unsqueeze(-2)) @ rotation.transpose(-1, -2)

        eta = torch.randn(
            model.xyz.shape, device=model.device, dtype=model.xyz.dtype, generator=self.generator
        )
        noise = torch.einsum("nij,nj->ni", covariance, eta)
        noise = noise * (gate * lr_xyz * self.config.noise_lr).unsqueeze(-1)
        model.xyz.add_(noise)


__all__ = ["MCMCConfig", "MCMCStrategy", "relocation_values", "PARAMETER_ATTRIBUTES"]
