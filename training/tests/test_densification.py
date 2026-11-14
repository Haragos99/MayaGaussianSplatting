import torch

from gstrain.densification import (
    DensificationStats,
    DensificationConfig,
    densify_and_prune,
    reset_opacity,
    should_densify,
)
from gstrain.dataset import TrainView
from gstrain.geometry.camera import TorchCamera
from gstrain.model_torch import PARAMETER_ATTRIBUTES, TorchGaussianModel
from gstrain.render.renderer_torch import render_gaussians_with_aux
from gstrain.train import TrainConfig, create_optimizer, train


def _model(
    xyz: torch.Tensor,
    scale: torch.Tensor,
    opacity: torch.Tensor,
) -> TorchGaussianModel:
    count = len(xyz)
    rotation = torch.zeros(count, 4, dtype=xyz.dtype, device=xyz.device)
    rotation[:, 0] = 1.0
    color = torch.full((count, 3), 0.5, dtype=xyz.dtype, device=xyz.device)
    return TorchGaussianModel(xyz, scale, rotation, opacity, color)


def _camera(tx: float = 0.0, device: torch.device | None = None) -> TorchCamera:
    device = device or torch.device("cpu")
    return TorchCamera(
        width=16,
        height=16,
        fx=torch.tensor(12.0, device=device),
        fy=torch.tensor(12.0, device=device),
        cx=torch.tensor(8.0, device=device),
        cy=torch.tensor(8.0, device=device),
        R=torch.eye(3, device=device),
        t=torch.tensor([tx, 0.0, 0.0], device=device),
    )


def test_renderer_accumulates_projected_gradient_stats() -> None:
    model = _model(
        torch.tensor([[0.2, 0.0, 2.0]]),
        torch.tensor([[0.5, 0.5, 0.5]]),
        torch.tensor([0.8]),
    )
    image, projected = render_gaussians_with_aux(_camera(), model, tile_size=4)
    image[8, 8, 0].backward()

    stats = DensificationStats(len(model), model.device)
    stats.add(projected, width=16, height=16)

    assert projected.mean.grad is not None
    assert torch.isfinite(projected.mean.grad).all()
    assert stats.accumulated.sum() > 0
    assert stats.denominator.sum() == 1
    assert stats.max_radius.max() > 0


def test_functional_clone_split_prune_preserves_adam_state() -> None:
    model = _model(
        torch.tensor([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [2.0, 0.0, 2.0]]),
        torch.tensor([[0.05, 0.05, 0.05], [0.5, 0.5, 0.5], [0.05, 0.05, 0.05]]),
        torch.tensor([0.8, 0.8, 0.001]),
    )
    optimizer = create_optimizer(model, TrainConfig(iterations=2), extent=1.0)
    sum(parameter.square().sum() for parameter in model.parameters()).backward()
    optimizer.step()

    stats = DensificationStats(len(model), model.device)
    stats.accumulated[:] = torch.tensor([1.0, 1.0, 0.0])
    stats.denominator[:] = 1.0
    config = DensificationConfig(
        grad_threshold=0.1,
        min_opacity=0.01,
        percent_dense=0.1,
        max_scale_fraction=None,
        start_iteration=1,
        end_iteration=10,
        max_gaussians=10,
        seed=5,
    )

    result = densify_and_prune(model, optimizer, stats, config, extent=1.0, iteration=2)

    assert (result["cloned"], result["split"], result["pruned"], result["after"]) == (1, 1, 1, 4)
    assert torch.all(model.scale[2:] < 0.5)
    assert stats.accumulated.shape == (len(model),)
    for group in optimizer.param_groups:
        name = group["name"]
        parameter = group["params"][0]
        assert parameter is getattr(model, PARAMETER_ATTRIBUTES[name])
        assert optimizer.state[parameter]["exp_avg"].shape == parameter.shape

    reset_count = reset_opacity(model, optimizer)
    assert reset_count == len(model)
    assert model.opacity.max() <= 0.010001
    for group in optimizer.param_groups:
        parameter = group["params"][0]
        assert optimizer.state[parameter]["exp_avg"].shape == parameter.shape
    assert torch.count_nonzero(optimizer.state[model.opacity_raw]["exp_avg"]) == 0


def test_training_runs_scheduled_functional_densification() -> None:
    model = _model(
        torch.tensor([[0.15, 0.0, 3.0]]),
        torch.tensor([[0.05, 0.05, 0.05]]),
        torch.tensor([0.8]),
    )
    views = [
        TrainView(_camera(), torch.zeros(16, 16, 3)),
        TrainView(_camera(tx=0.2), torch.zeros(16, 16, 3)),
    ]
    config = TrainConfig(
        iterations=2,
        densification=DensificationConfig(
            start_iteration=1,
            end_iteration=3,
            interval=1,
            grad_threshold=0.0,
            min_opacity=0.0,
            percent_dense=100.0,
            max_scale_fraction=None,
            max_screen_radius=None,
            max_gaussians=2,
        ),
    )

    losses = train(model, views, config=config)

    assert len(losses) == 2
    assert len(model) == 2


def test_densification_schedule_waits_then_uses_global_interval() -> None:
    config = DensificationConfig(start_iteration=50, end_iteration=200, interval=100)

    assert not should_densify(50, config)
    assert should_densify(100, config)
    assert not should_densify(150, config)
    assert not should_densify(200, config)


def test_opacity_reset_stops_with_densification() -> None:
    from gstrain.densification import should_reset_opacity

    config = DensificationConfig(end_iteration=7_000, opacity_reset_interval=3_000)
    assert should_reset_opacity(3_000, config)
    assert should_reset_opacity(6_000, config)
    assert not should_reset_opacity(9_000, config)


def test_schedule_scales_with_run_length() -> None:
    from gstrain.densification import should_reset_opacity

    # A 30k run reproduces the reference 3DGS schedule.
    full = DensificationConfig.for_iterations(30_000)
    assert (full.start_iteration, full.interval, full.opacity_reset_interval) == (500, 100, 3_000)

    for n in (500, 2_000, 7_000):
        config = DensificationConfig.for_iterations(n)
        rounds = [i for i in range(1, n + 1) if should_densify(i, config)]
        resets = [i for i in range(1, n + 1) if should_reset_opacity(i, config)]
        assert rounds, n
        # New splats get at least 100 Adam steps before they are judged again.
        assert all(b - a >= 100 for a, b in zip(rounds, rounds[1:]))
        # Opacity gets at least 200 steps to recover between resets.
        assert all(b - a >= 200 for a, b in zip(resets, resets[1:]))
        assert all(r < config.end_iteration for r in resets)

    config = DensificationConfig.for_iterations(500)
    assert [i for i in range(1, 501) if should_densify(i, config)] == [100, 200]
    assert [i for i in range(1, 501) if should_reset_opacity(i, config)] == []
    # Too short to densify usefully.
    assert not any(should_densify(i, DensificationConfig.for_iterations(100)) for i in range(1, 101))


def test_split_children_use_fresh_noise_each_round() -> None:
    from gstrain.densification import split

    def children(seed: int) -> torch.Tensor:
        model = _model(
            torch.zeros(1, 3),
            torch.tensor([[0.5, 0.5, 0.5]]),
            torch.tensor([0.8]),
        )
        optimizer = create_optimizer(model, TrainConfig(iterations=1), extent=1.0)
        generator = torch.Generator().manual_seed(seed)
        split(model, optimizer, torch.tensor([True]), DensificationConfig(), generator)
        return model.xyz.detach().clone()

    assert not torch.allclose(children(1), children(2))
    assert torch.allclose(children(3), children(3))


def test_capacity_keeps_highest_gradient_candidates() -> None:
    model = _model(
        torch.tensor([[0.0, 0.0, 2.0], [0.5, 0.0, 2.0], [1.0, 0.0, 2.0]]),
        torch.full((3, 3), 0.01),
        torch.full((3,), 0.8),
    )
    optimizer = create_optimizer(model, TrainConfig(iterations=1), extent=1.0)
    stats = DensificationStats(len(model), model.device)
    stats.accumulated[:] = torch.tensor([0.2, 0.9, 0.5])
    stats.denominator[:] = 1.0
    config = DensificationConfig(
        grad_threshold=0.1, percent_dense=0.1, max_scale_fraction=None,
        max_screen_radius=None, min_opacity=0.0, max_gaussians=4,
    )

    result = densify_and_prune(model, optimizer, stats, config, extent=1.0, iteration=100)

    assert result["cloned"] == 1
    assert torch.allclose(model.xyz[-1], torch.tensor([0.5, 0.0, 2.0]))


def test_size_pruning_waits_until_after_opacity_warmup() -> None:
    config = DensificationConfig(
        start_iteration=1,
        end_iteration=5_000,
        interval=1,
        min_opacity=0.0,
        max_screen_radius=20.0,
        max_scale_fraction=0.1,
        opacity_reset_interval=3_000,
        max_gaussians=4,
    )

    def oversized_model() -> tuple[TorchGaussianModel, torch.optim.Adam, DensificationStats]:
        model = _model(
            torch.tensor([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0]]),
            torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]),
            torch.tensor([0.8, 0.8]),
        )
        optimizer = create_optimizer(model, TrainConfig(iterations=1), extent=1.0)
        stats = DensificationStats(len(model), model.device)
        stats.max_radius.fill_(100.0)
        return model, optimizer, stats

    early_model, early_optimizer, early_stats = oversized_model()
    early = densify_and_prune(
        early_model, early_optimizer, early_stats, config, extent=1.0, iteration=100
    )
    assert early["pruned"] == 0

    late_model, late_optimizer, late_stats = oversized_model()
    late = densify_and_prune(
        late_model, late_optimizer, late_stats, config, extent=1.0, iteration=3_001
    )
    assert late["pruned"] == 1
    assert late["screen_pruned"] >= 1

    # No opacity reset in the schedule (short runs) -> never prune by size.
    no_reset = DensificationConfig(**{**config.__dict__, "opacity_reset_interval": None})
    model, optimizer, stats = oversized_model()
    result = densify_and_prune(model, optimizer, stats, no_reset, extent=1.0, iteration=4_000)
    assert result["pruned"] == 0
    assert (
        late["opacity_pruned"] + late["screen_pruned"] + late["world_pruned"]
        == late["pruned"]
    )

def _big_on_screen_model():
    model = _model(
        torch.tensor([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0]]),
        torch.tensor([[0.05, 0.05, 0.05], [0.05, 0.05, 0.05]]),
        torch.tensor([0.8, 0.8]),
    )
    optimizer = create_optimizer(model, TrainConfig(iterations=1), extent=1.0)
    stats = DensificationStats(len(model), model.device)
    stats.max_radius.fill_(400.0)            # huge on screen ...
    stats.max_radius_fraction.fill_(0.8)     # ... 80% of the image
    return model, optimizer, stats


def test_default_config_never_prunes_by_screen_size() -> None:
    """Regression: a fixed 20 px limit deleted the wall/background splats at 480 px
    (test PSNR 20.8 -> 6.4 dB on MeSP). Screen-size pruning is now opt-in."""
    model, optimizer, stats = _big_on_screen_model()
    config = DensificationConfig(min_opacity=0.0, max_scale_fraction=None, opacity_reset_interval=100)
    result = densify_and_prune(model, optimizer, stats, config, extent=1.0, iteration=5_000)
    assert result["screen_pruned"] == 0 and result["pruned"] == 0


def test_screen_fraction_pruning_is_resolution_independent() -> None:
    model, optimizer, stats = _big_on_screen_model()
    stats.max_radius_fraction[1] = 0.05      # second splat is small relative to the image
    config = DensificationConfig(
        min_opacity=0.0, max_scale_fraction=None, opacity_reset_interval=100, max_screen_fraction=0.15
    )
    result = densify_and_prune(model, optimizer, stats, config, extent=1.0, iteration=5_000)
    assert result["screen_pruned"] == 1 and len(model) == 1


def test_first_opacity_reset_waits_for_densification() -> None:
    from gstrain.densification import should_reset_opacity

    for n in (7_000, 15_000, 30_000):
        config = DensificationConfig.for_iterations(n)
        rounds = [i for i in range(1, n + 1) if should_densify(i, config)]
        resets = [i for i in range(1, n + 1) if should_reset_opacity(i, config)]
        assert resets and resets[0] == 3_000
        # The round at the reset iteration runs before the reset (as in 3DGS).
        assert sum(r <= resets[0] for r in rounds) >= 25
    # Too short to rebuild the scene after a reset -> no reset (and no size pruning).
    for n in (1_000, 2_000, 5_000):
        config = DensificationConfig.for_iterations(n)
        assert config.opacity_reset_interval is None


def test_strategy_warns_when_one_round_prunes_a_quarter() -> None:
    from gstrain.strategy import DefaultStrategy

    model, optimizer, _ = _big_on_screen_model()
    config = DensificationConfig(
        start_iteration=0, end_iteration=1_000, interval=10, min_opacity=0.0,
        max_scale_fraction=None, opacity_reset_interval=5, max_screen_radius=20.0,
    )
    strategy = DefaultStrategy(model, optimizer, config, extent=1.0)
    strategy.stats.max_radius.fill_(400.0)
    messages = strategy.after_step(10, lr_xyz=1e-4)
    assert any(message.startswith("WARNING") for message in messages)


# --- added: split / clone details, no-collapse regression ----------------- #

import pytest

from gstrain.densification import clone, split
from gstrain.geometry.quaternion import quaternion_to_rotation_matrix
from gstrain.loss import psnr
from gstrain.render.renderer_torch import render_gaussians


def _sh_model() -> TorchGaussianModel:
    g = torch.Generator().manual_seed(0)
    model = _model(torch.rand(4, 3, generator=g) + torch.tensor([0.0, 0.0, 2.0]),
                   0.05 + 0.2 * torch.rand(4, 3, generator=g), torch.full((4,), 0.6))
    with torch.no_grad():
        model.sh_rest.copy_(torch.randn(model.sh_rest.shape, generator=g))
        model.rotation.copy_(torch.nn.functional.normalize(torch.randn(4, 4, generator=g), dim=-1))
    return model


def test_split_children_smaller_and_inside_parent() -> None:
    model = _sh_model()
    parent = {name: getattr(model, attr).detach().clone() for name, attr in PARAMETER_ATTRIBUTES.items()}
    optimizer = create_optimizer(model, TrainConfig(iterations=2), extent=1.0)
    mask = torch.tensor([False, True, False, True])
    config = DensificationConfig()
    assert split(model, optimizer, mask, config, torch.Generator().manual_seed(1)) == 2
    assert len(model) == 4 - 2 + 2 * config.split_into

    children = slice(2, None)  # survivors first, then the children (parent order, repeated)
    parents = torch.tensor([1, 3]).repeat(config.split_into)
    parent_scale = parent["scale"][parents].exp()
    assert torch.allclose(model.scale[children], parent_scale / (0.8 * config.split_into), rtol=1e-5)
    # Children are sampled from the parent's Gaussian: within a few sigma, in the parent's frame.
    R = quaternion_to_rotation_matrix(parent["rotation"][parents])
    local = torch.einsum("nji,nj->ni", R, model.xyz[children] - parent["xyz"][parents])
    assert torch.all((local / parent_scale).abs() < 5.0)
    # Everything else is copied from the parent.
    for name in ("opacity", "f_dc", "f_rest"):
        assert torch.equal(getattr(model, PARAMETER_ATTRIBUTES[name])[children], parent[name][parents]), name


def test_clone_keeps_all_parameters_including_sh() -> None:
    model = _sh_model()
    model.active_sh_degree = 2
    optimizer = create_optimizer(model, TrainConfig(iterations=2), extent=1.0)
    mask = torch.tensor([True, False, False, True])
    before = {attr: getattr(model, attr).detach().clone() for attr in PARAMETER_ATTRIBUTES.values()}
    assert clone(model, optimizer, mask) == 2
    assert len(model) == 6 and model.active_sh_degree == 2
    for attr, value in before.items():
        assert torch.equal(getattr(model, attr)[:4], value), attr
        assert torch.equal(getattr(model, attr)[4:], value[mask]), attr


@pytest.mark.slow
def test_no_collapse_after_opacity_reset() -> None:
    """Regression for the 480 px collapse: a reset plus pruning must not wipe the scene.

    A large splat makes the background (it is wider than 20 px on screen, which the
    old default screen-size pruning deleted). After the reset the model must recover.
    """
    def scene() -> TorchGaussianModel:
        xyz = torch.tensor([[0.0, 0.0, 6.0], [0.1, 0.05, 2.0], [-0.15, -0.1, 2.2], [0.05, -0.15, 1.9]])
        scale = torch.tensor([[3.0, 3.0, 0.05], [0.06] * 3, [0.05] * 3, [0.04] * 3])
        color = torch.tensor([[0.2, 0.4, 0.8], [0.9, 0.2, 0.1], [0.1, 0.9, 0.2], [0.9, 0.9, 0.1]])
        return TorchGaussianModel(xyz, scale, torch.tensor([[1.0, 0, 0, 0]]).repeat(4, 1),
                                  torch.tensor([0.95, 0.9, 0.9, 0.9]), color, sh_degree=0)

    cameras = [_camera(tx) for tx in (-0.1, 0.0, 0.1)]
    with torch.no_grad():
        targets = [render_gaussians(c, scene()) for c in cameras]
    views = [TrainView(c, t) for c, t in zip(cameras, targets)]
    model = scene()
    with torch.no_grad():
        model.xyz.add_(0.02 * torch.randn(model.xyz.shape, generator=torch.Generator().manual_seed(0)))

    densification = DensificationConfig(start_iteration=10, end_iteration=200, interval=20,
                                        opacity_reset_interval=100, max_gaussians=200)
    counts: list[int] = []
    quality: dict[int, float] = {}

    def on_iteration(step: int, _loss: float) -> None:
        counts.append(len(model))
        if step + 1 in (99, 300):
            with torch.no_grad():
                quality[step + 1] = float(sum(psnr(render_gaussians(v.camera, model), v.image) for v in views) / 3)

    # extent 40: the background splat (scale 3) stays below the 10 %-of-extent world-size limit.
    train(model, views, TrainConfig(iterations=300, eval_every=0, densification=densification),
          extent=40.0, on_iteration=on_iteration)

    drops = [b / a for a, b in zip(counts, counts[1:]) if b < a]
    assert all(d > 0.75 for d in drops), min(drops)
    assert quality[300] >= quality[99] - 1.0, quality
    assert quality[300] > 20.0, quality
