import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gstrain.model import GaussianModel, OptimizationConfig  # noqa: E402
from gstrain.params import create_from_point_cloud  # noqa: E402


def build_model(n: int = 8, scale: float = 0.001) -> GaussianModel:
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(n, 3))
    rgb = rng.integers(0, 256, size=(n, 3), dtype=np.uint8)
    params = create_from_point_cloud(xyz, rgb, sh_degree=1)
    model = GaussianModel.from_params(params, spatial_lr_scale=1.0)
    model.training_setup(OptimizationConfig())
    model._scaling.data.fill_(math.log(scale))
    return model


def prime_optimizer_state(model: GaussianModel) -> None:
    """One step so Adam actually allocates exp_avg / exp_avg_sq."""
    for group in model.optimizer.param_groups:
        param = group["params"][0]
        param.grad = torch.ones_like(param)
    model.optimizer.step()
    model.optimizer.zero_grad(set_to_none=True)


def assert_state_matches_params(model: GaussianModel) -> None:
    for group in model.optimizer.param_groups:
        param = group["params"][0]
        state = model.optimizer.state.get(param)
        assert state is not None, f"group {group['name']} lost its Adam state"
        assert state["exp_avg"].shape == param.shape, group["name"]
        assert state["exp_avg_sq"].shape == param.shape, group["name"]


def test_clone_grows_the_model_and_zeroes_new_momentum():
    model = build_model(n=8, scale=0.001)
    prime_optimizer_state(model)
    before = model.optimizer.state[model._xyz]["exp_avg"].clone()

    model.xyz_gradient_accum.fill_(1.0)
    model.denom.fill_(1.0)
    model.densify_and_prune(extent=1.0, max_screen_size=None)

    assert len(model) == 16
    assert_state_matches_params(model)
    state = model.optimizer.state[model._xyz]
    torch.testing.assert_close(state["exp_avg"][:8], before)
    assert torch.all(state["exp_avg"][8:] == 0)


def test_split_replaces_parents_with_smaller_children():
    model = build_model(n=8, scale=0.5)
    prime_optimizer_state(model)

    model.xyz_gradient_accum.fill_(1.0)
    model.denom.fill_(1.0)
    model.densify_and_prune(extent=1.0, max_screen_size=None)

    assert len(model) == 16  # 8 parents removed, 2 children each
    assert_state_matches_params(model)
    assert float(model.get_scaling.detach().max()) < 0.5


def test_stats_buffers_are_resized_with_the_model():
    model = build_model(n=8, scale=0.001)
    prime_optimizer_state(model)
    model.xyz_gradient_accum.fill_(1.0)
    model.denom.fill_(1.0)
    model.densify_and_prune(extent=1.0, max_screen_size=None)

    assert model.xyz_gradient_accum.shape == (len(model), 1)
    assert model.denom.shape == (len(model), 1)
    assert model.max_radii2D.shape == (len(model),)


def test_prune_keeps_the_momentum_of_survivors():
    model = build_model(n=8)
    prime_optimizer_state(model)
    before = model.optimizer.state[model._xyz]["exp_avg"].clone()

    mask = torch.zeros(8, dtype=torch.bool)
    mask[[1, 4, 6]] = True
    model.prune_points(mask)

    assert len(model) == 5
    assert_state_matches_params(model)
    torch.testing.assert_close(model.optimizer.state[model._xyz]["exp_avg"], before[~mask])


def test_low_opacity_points_are_pruned():
    model = build_model(n=8)
    prime_optimizer_state(model)
    model._opacity.data[:4] = -20.0  # sigmoid(-20) is far below min_opacity
    model.densify_and_prune(extent=1.0, max_screen_size=None)

    assert len(model) == 4
    assert_state_matches_params(model)


def test_reset_opacity_caps_and_clears_momentum():
    model = build_model(n=8)
    prime_optimizer_state(model)
    model.reset_opacity()

    assert float(model.get_opacity.detach().max()) <= 0.01 + 1e-6
    state = model.optimizer.state[model._opacity]
    assert torch.all(state["exp_avg"] == 0)
    assert_state_matches_params(model)


def test_optimizer_still_steps_after_densification():
    model = build_model(n=8, scale=0.001)
    prime_optimizer_state(model)
    model.xyz_gradient_accum.fill_(1.0)
    model.denom.fill_(1.0)
    model.densify_and_prune(extent=1.0, max_screen_size=None)

    before = model._xyz.detach().clone()
    for group in model.optimizer.param_groups:
        group["params"][0].grad = torch.ones_like(group["params"][0])
    model.optimizer.step()
    assert not torch.allclose(model._xyz.detach(), before)


def test_learning_rate_decays():
    model = build_model()
    first = model.update_learning_rate(1)
    last = model.update_learning_rate(model.cfg.position_lr_max_steps)
    assert last < first
