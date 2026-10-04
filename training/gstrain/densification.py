from dataclasses import dataclass
import torch
import torch.nn as nn

from .geometry.quaternion import quaternion_to_rotation_matrix
from .model_torch import PARAMETER_ATTRIBUTES, TorchGaussianModel
from .render.renderer_torch import ProjectedGaussians


@dataclass(frozen=True)
class DensificationConfig:
    grad_threshold: float = 2e-4
    min_opacity: float = 0.005
    percent_dense: float = 0.01
    max_scale_fraction: float | None = 0.1
    split_into: int = 2
    max_screen_radius: float | None = 20.0
    start_iteration: int = 50
    end_iteration: int = 15_000
    interval: int = 100
    opacity_reset_interval: int | None = 3_000
    opacity_reset_value: float = 0.01
    max_gaussians: int = 2_000_000
    delay_size_pruning_until_opacity_reset: bool = True
    seed: int = 0

class DensificationStats:
    """Running screen-position gradient mean and maximum radius per splat."""

    def __init__(self, count: int, device: torch.device) -> None:
        self.accumulated = torch.zeros(count, device=device)
        self.denominator = torch.zeros(count, device=device)
        self.max_radius = torch.zeros(count, device=device)

    @torch.no_grad()
    def add(self, projected: ProjectedGaussians, width: int, height: int) -> None:
        if projected.mean.grad is None:
            return

        center = projected.mean.detach()
        radius = projected.radius.detach()
        overlaps_image = (
            (center[:, 0] + radius[:, 0] >= 0)
            & (center[:, 0] - radius[:, 0] < width)
            & (center[:, 1] + radius[:, 1] >= 0)
            & (center[:, 1] - radius[:, 1] < height)
        )
        seen = projected.visible & overlaps_image
        if not seen.any():
            return

        pixel_to_normalized = projected.mean.new_tensor([0.5 * width, 0.5 * height])
        gradient = (projected.mean.grad * pixel_to_normalized).norm(dim=-1)
        self.accumulated[seen] += gradient[seen]
        self.denominator[seen] += 1.0
        radius_1d = radius.amax(dim=-1)
        self.max_radius[seen] = torch.maximum(self.max_radius[seen], radius_1d[seen])

    def average_gradient(self) -> torch.Tensor:
        average = self.accumulated / self.denominator.clamp_min(1.0)
        return torch.nan_to_num(average, nan=0.0, posinf=0.0, neginf=0.0)

    def reset(self, count: int, device: torch.device) -> None:
        self.__init__(count, device)


@torch.no_grad()
def _rebuild_parameters(model: TorchGaussianModel, optimizer, build) -> None:
    """Resize Gaussian parameters and the matching Adam moments together."""
    for group in optimizer.param_groups:
        if len(group["params"]) != 1:
            raise ValueError("densification expects one Gaussian parameter per optimizer group")

        name = group["name"]
        attribute = PARAMETER_ATTRIBUTES[name]
        old_parameter = group["params"][0]
        new_value, state_builder = build(name, old_parameter)
        new_parameter = nn.Parameter(new_value.detach().contiguous())

        old_state = optimizer.state.pop(old_parameter, None)
        if old_state is not None:
            new_state = dict(old_state)
            for key in ("exp_avg", "exp_avg_sq"):
                if key in new_state:
                    new_state[key] = state_builder(new_state[key])
            optimizer.state[new_parameter] = new_state

        group["params"][0] = new_parameter
        setattr(model, attribute, new_parameter)


@torch.no_grad()
def prune(model: TorchGaussianModel, optimizer, keep: torch.Tensor) -> int:
    """Remove rows where keep is false, preserving surviving Adam moments."""
    if keep.ndim != 1 or keep.numel() != len(model):
        raise ValueError("keep mask must have one entry per Gaussian")
    keep = keep.to(device=model.device, dtype=torch.bool)
    removed = int((~keep).sum())
    if removed == 0:
        return 0

    _rebuild_parameters(
        model,
        optimizer,
        lambda _name, tensor: (tensor[keep], lambda state: state[keep]),
    )
    return removed


@torch.no_grad()
def append(model: TorchGaussianModel, optimizer, additions: dict[str, torch.Tensor]) -> None:
    """Append splats; new rows start with zeroed Adam moments."""
    missing = set(PARAMETER_ATTRIBUTES) - additions.keys()
    if missing:
        raise ValueError(f"missing added Gaussian parameters: {sorted(missing)}")

    def build(name: str, tensor: torch.Tensor):
        extra = additions[name].to(device=tensor.device, dtype=tensor.dtype)
        if extra.shape[1:] != tensor.shape[1:]:
            raise ValueError(f"added {name} rows have incompatible shape {extra.shape}")
        return (
            torch.cat((tensor, extra), dim=0),
            lambda state: torch.cat((state, torch.zeros_like(extra)), dim=0),
        )

    _rebuild_parameters(model, optimizer, build)


def _selected(
    model: TorchGaussianModel,
    gradient: torch.Tensor,
    observed: torch.Tensor,
    config: DensificationConfig,
    extent: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    wants_growth = (gradient >= config.grad_threshold) & observed
    largest_scale = model.scale.detach().amax(dim=-1)
    is_small = largest_scale <= config.percent_dense * extent
    return wants_growth & is_small, wants_growth & ~is_small


@torch.no_grad()
def clone(model: TorchGaussianModel, optimizer, mask: torch.Tensor) -> int:
    """Duplicate high-gradient, small splats without removing their parents."""
    mask = mask.to(device=model.device, dtype=torch.bool)
    count = int(mask.sum())
    if count == 0:
        return 0

    additions = {
        name: getattr(model, attribute).detach()[mask].clone()
        for name, attribute in PARAMETER_ATTRIBUTES.items()
    }
    append(model, optimizer, additions)
    return count


@torch.no_grad()
def split(
    model: TorchGaussianModel,
    optimizer,
    mask: torch.Tensor,
    config: DensificationConfig,
) -> int:
    """Replace selected large splats with smaller children sampled from them."""
    original_count = len(model)
    if mask.ndim != 1 or mask.numel() != original_count:
        raise ValueError("split mask must have one entry per current Gaussian")
    mask = mask.to(device=model.device, dtype=torch.bool)
    count = int(mask.sum())
    if count == 0:
        return 0

    parts = config.split_into
    scale = model.scale.detach()[mask].repeat(parts, 1)
    rotation = model.rotation.detach()[mask].repeat(parts, 1)
    center = model.xyz.detach()[mask].repeat(parts, 1)

    generator = torch.Generator(device=model.device)
    generator.manual_seed(config.seed)
    offset = torch.randn(
        scale.shape,
        device=scale.device,
        dtype=scale.dtype,
        generator=generator,
    ) * scale
    rotations = quaternion_to_rotation_matrix(rotation)
    displaced = torch.bmm(rotations, offset.unsqueeze(-1)).squeeze(-1)

    additions = {
        "xyz": center + displaced,
        "scale": (scale / (0.8 * parts)).log(),
        "rotation": rotation,
        "opacity": model.opacity_raw.detach()[mask].repeat(parts),
        "color": model.color_raw.detach()[mask].repeat(parts, 1),
    }
    append(model, optimizer, additions)

    keep = torch.ones(len(model), dtype=torch.bool, device=model.device)
    keep[:original_count] = ~mask
    prune(model, optimizer, keep)
    return count


@torch.no_grad()
def reset_opacity(
    model: TorchGaussianModel,
    optimizer,
    value: float = 0.01,
) -> int:
    """Cap opacity and clear its Adam moments so splats must re-earn visibility."""
    if not 0 < value < 1:
        raise ValueError("opacity reset value must be between zero and one")

    ceiling = torch.logit(model.opacity_raw.new_tensor(value))
    reset_count = int((model.opacity.detach() > value).sum())

    def build(name: str, tensor: torch.Tensor):
        if name != "opacity":
            return tensor, lambda state: state
        return tensor.clamp_max(ceiling), torch.zeros_like

    _rebuild_parameters(model, optimizer, build)
    return reset_count


def _pad_mask(mask: torch.Tensor, size: int) -> torch.Tensor:
    if mask.numel() >= size:
        return mask[:size]
    padded = torch.zeros(size, dtype=torch.bool, device=mask.device)
    padded[: mask.numel()] = mask
    return padded


@torch.no_grad()
def densify_and_prune(
    model: TorchGaussianModel,
    optimizer,
    stats: DensificationStats,
    config: DensificationConfig,
    extent: float,
    iteration: int,
) -> dict[str, int]:
    """Run one round of gradient-guided growth and scheduled pruning."""
    before = len(model)
    if stats.accumulated.numel() != before:
        raise ValueError("densification statistics do not match model population")

    gradient = stats.average_gradient()
    opacity = model.opacity.detach().reshape(-1)
    largest_scale = model.scale.detach().amax(dim=-1)
    size_pruning_enabled = (
        not config.delay_size_pruning_until_opacity_reset
        or config.opacity_reset_interval is None
        or iteration > config.opacity_reset_interval
    )

    opacity_drop = opacity <= config.min_opacity
    world_drop = torch.zeros_like(opacity_drop)
    if size_pruning_enabled and config.max_scale_fraction is not None:
        world_drop = largest_scale > config.max_scale_fraction * extent
    screen_drop = torch.zeros_like(opacity_drop)
    if size_pruning_enabled and config.max_screen_radius is not None:
        screen_drop = stats.max_radius > config.max_screen_radius

    screen_drop &= ~opacity_drop
    world_drop &= ~opacity_drop & ~screen_drop
    original_keep = ~(opacity_drop | screen_drop | world_drop)
    if before and not original_keep.any():
        protected = int(opacity.argmax())
        opacity_drop[protected] = False
        screen_drop[protected] = False
        world_drop[protected] = False
        original_keep[protected] = True

    clone_mask, split_mask = _selected(
        model, gradient, stats.denominator > 0, config, extent
    )
    clone_mask &= original_keep
    split_mask &= original_keep
    clone_candidate_count = int(clone_mask.sum())
    split_candidate_count = int(split_mask.sum())

    # Limit net growth while counting split as (children - 1) new splats.
    capacity = max(config.max_gaussians - int(original_keep.sum()), 0)
    clone_indices = torch.nonzero(clone_mask, as_tuple=False).flatten()[:capacity]
    clone_mask = torch.zeros_like(clone_mask)
    clone_mask[clone_indices] = True
    capacity -= len(clone_indices)

    split_capacity = capacity // (config.split_into - 1)
    split_indices = torch.nonzero(split_mask, as_tuple=False).flatten()[:split_capacity]
    split_mask = torch.zeros_like(split_mask)
    split_mask[split_indices] = True

    cloned = clone(model, optimizer, clone_mask)
    split_parent_mask = _pad_mask(split_mask, len(model))
    split_count = split(model, optimizer, split_parent_mask, config)

    original_after_split = original_keep[~split_mask]
    added_count = len(model) - original_after_split.numel()
    zeros_for_added = torch.zeros(added_count, dtype=torch.bool, device=model.device)
    opacity_drop = torch.cat((opacity_drop[~split_mask], zeros_for_added))
    screen_drop = torch.cat((screen_drop[~split_mask], zeros_for_added))
    world_drop = torch.cat((world_drop[~split_mask], zeros_for_added))

    # Existing rows were classified before growth; apply these checks to new rows.
    added_start = original_after_split.numel()
    if added_count:
        current_opacity = model.opacity.detach().reshape(-1)
        opacity_drop[added_start:] = current_opacity[added_start:] <= config.min_opacity
        if size_pruning_enabled and config.max_scale_fraction is not None:
            current_scale = model.scale.detach().amax(dim=-1)
            world_drop[added_start:] = current_scale[added_start:] > config.max_scale_fraction * extent

    keep = ~(opacity_drop | screen_drop | world_drop)

    if len(model) and not keep.any():
        protected = int(model.opacity.detach().argmax())
        keep[protected] = True
        opacity_drop[protected] = False
        screen_drop[protected] = False
        world_drop[protected] = False

    pruned = prune(model, optimizer, keep)
    stats.reset(len(model), model.device)

    return {
        "cloned": cloned,
        "split": split_count,
        "pruned": pruned,
        "before": before,
        "after": len(model),
        "clone_candidates": clone_candidate_count,
        "split_candidates": split_candidate_count,
        "opacity_pruned": int(opacity_drop.sum()),
        "screen_pruned": int(screen_drop.sum()),
        "world_pruned": int(world_drop.sum()),
        "added": len(model) - (before - pruned - split_count),
    }


def should_densify(iteration: int, config: DensificationConfig) -> bool:
    return (
        config.start_iteration < iteration < config.end_iteration
        and iteration % config.interval == 0
    )


def should_reset_opacity(iteration: int, config: DensificationConfig) -> bool:
    interval = config.opacity_reset_interval
    return interval is not None and iteration > 0 and iteration % interval == 0
