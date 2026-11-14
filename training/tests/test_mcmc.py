import torch
from gstrain.dataset import TrainView
from gstrain.geometry.camera import TorchCamera
from gstrain.model_torch import PARAMETER_ATTRIBUTES, TorchGaussianModel
from gstrain.render.renderer_torch import render_gaussians
from gstrain.strategy import prune_by_contribution, splat_contribution
from gstrain.strategy_mcmc import MCMCConfig, MCMCStrategy, relocation_values
from gstrain.train import TrainConfig, create_optimizer, train


def _camera(size: int = 32) -> TorchCamera:
    return TorchCamera(
        width=size, height=size,
        fx=torch.tensor(float(size)), fy=torch.tensor(float(size)),
        cx=torch.tensor(size / 2), cy=torch.tensor(size / 2),
        R=torch.eye(3), t=torch.zeros(3),
    )


def _model(xyz, scale, opacity) -> TorchGaussianModel:
    count = len(xyz)
    rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1)
    return TorchGaussianModel(xyz, scale, rotation, opacity, torch.full((count, 3), 0.8))


def _with_optimizer(model):
    optimizer = create_optimizer(model, TrainConfig(iterations=10), extent=1.0)
    sum(p.square().sum() for p in model.parameters()).backward()
    optimizer.step()
    optimizer.zero_grad()
    return optimizer


def test_relocation_values_keep_single_splat_unchanged() -> None:
    opacity = torch.tensor([0.3, 0.9])
    scale = torch.tensor([[0.1, 0.2, 0.3], [0.5, 0.5, 0.5]])
    new_opacity, new_scale = relocation_values(opacity, scale, torch.tensor([1, 1]))
    assert torch.allclose(new_opacity, opacity)
    assert torch.allclose(new_scale, scale)


def test_relocated_copies_render_like_the_original() -> None:
    """n coincident copies with the relocation opacity/scale ~ the original splat."""
    camera = _camera()
    xyz = torch.tensor([[0.0, 0.0, 3.0]])
    scale = torch.full((1, 3), 0.3)
    for opacity in (0.3, 0.7):
        original = render_gaussians(camera, _model(xyz, scale, torch.tensor([opacity])))
        for copies in (2, 3):
            new_opacity, new_scale = relocation_values(
                torch.tensor([opacity]), scale, torch.tensor([copies])
            )
            model = _model(xyz.repeat(copies, 1), new_scale.repeat(copies, 1), new_opacity.repeat(copies))
            assert (render_gaussians(camera, model) - original).abs().max() < 2e-2


def test_relocate_moves_dead_splats_onto_live_ones() -> None:
    model = _model(
        torch.tensor([[0.0, 0.0, 3.0], [5.0, 5.0, 5.0], [0.3, 0.0, 3.0]]),
        torch.full((3, 3), 0.1),
        torch.tensor([0.8, 0.001, 0.6]),
    )
    optimizer = _with_optimizer(model)
    strategy = MCMCStrategy(model, optimizer, MCMCConfig(seed=1), iterations=1000)

    assert strategy.relocate() == 1
    # The dead splat now sits exactly on one of the live ones and is visible.
    assert any(torch.allclose(model.xyz[1], model.xyz[i]) for i in (0, 2))
    assert model.opacity[1] > 0.005
    for group in optimizer.param_groups:
        parameter = group["params"][0]
        assert parameter is getattr(model, PARAMETER_ATTRIBUTES[group["name"]])
        assert optimizer.state[parameter]["exp_avg"].shape == parameter.shape


def test_growth_is_five_percent_and_never_exceeds_cap() -> None:
    model = _model(torch.rand(100, 3) + torch.tensor([0.0, 0.0, 3.0]), torch.full((100, 3), 0.05),
                   torch.full((100,), 0.5))
    optimizer = _with_optimizer(model)
    strategy = MCMCStrategy(model, optimizer, MCMCConfig(cap_max=108), iterations=1000)

    assert strategy.add() == 5 and len(model) == 105
    assert strategy.add() == 3 and len(model) == 108   # capped
    assert strategy.add() == 0
    for group in optimizer.param_groups:
        parameter = group["params"][0]
        assert optimizer.state[parameter]["exp_avg"].shape == parameter.shape


def test_noise_only_moves_transparent_splats() -> None:
    model = _model(torch.tensor([[0.0, 0.0, 3.0], [1.0, 0.0, 3.0]]), torch.full((2, 3), 0.1),
                   torch.tensor([0.9, 0.001]))
    optimizer = _with_optimizer(model)
    strategy = MCMCStrategy(model, optimizer, MCMCConfig(), iterations=1000)
    before = model.xyz.detach().clone()
    strategy.inject_noise(lr_xyz=1e-4)
    moved = (model.xyz.detach() - before).norm(dim=-1)
    assert moved[0] < 1e-9
    assert moved[1] > 1e-6


def test_mcmc_regularization_is_mean_opacity_and_scale() -> None:
    model = _model(torch.zeros(2, 3), torch.tensor([[0.1] * 3, [0.3] * 3]), torch.tensor([0.2, 0.6]))
    strategy = MCMCStrategy(model, _with_optimizer(model), MCMCConfig(), iterations=10)
    expected = 0.01 * model.opacity.mean() + 0.01 * model.scale.mean()
    assert torch.allclose(strategy.regularization(model), expected)


def test_mcmc_training_grows_to_cap() -> None:
    camera = _camera(16)
    model = _model(torch.rand(20, 3) * 0.4 - 0.2 + torch.tensor([0.0, 0.0, 3.0]),
                   torch.full((20, 3), 0.05), torch.full((20,), 0.5))
    views = [TrainView(camera, torch.full((16, 16, 3), 0.5)) for _ in range(2)]
    config = TrainConfig(
        iterations=40, strategy="mcmc", eval_every=0,
        mcmc=MCMCConfig(cap_max=22, refine_start_iteration=5, refine_every=5),
    )
    train(model, views, config=config, extent=1.0)
    assert 20 < len(model) <= 22


def test_contribution_pruning_drops_the_hidden_splat() -> None:
    camera = _camera()
    # Splat 1 sits exactly behind the opaque splat 0; splat 2 is off-screen.
    model = _model(
        torch.tensor([[0.0, 0.0, 3.0], [0.0, 0.0, 6.0], [50.0, 0.0, 3.0], [0.4, 0.0, 3.0]]),
        torch.tensor([[0.5] * 3, [0.05] * 3, [0.1] * 3, [0.1] * 3]),
        torch.tensor([0.99, 0.5, 0.5, 0.5]),
    )
    optimizer = _with_optimizer(model)
    views = [TrainView(camera, torch.zeros(32, 32, 3))]
    contribution = splat_contribution(model, views, render_gaussians)
    assert contribution[2] == 0 and contribution[1] < 0.05 * contribution[0]

    assert prune_by_contribution(model, optimizer, contribution, 0.5) == 2
    assert len(model) == 2
    kept = model.xyz.detach()
    assert torch.allclose(kept[0], torch.tensor([0.0, 0.0, 3.0]), atol=1e-2)
    assert torch.allclose(kept[1], torch.tensor([0.4, 0.0, 3.0]), atol=1e-2)


def test_training_with_contribution_pruning_finetunes() -> None:
    camera = _camera(16)
    model = _model(torch.rand(10, 3) * 0.4 - 0.2 + torch.tensor([0.0, 0.0, 3.0]),
                   torch.full((10, 3), 0.05), torch.full((10,), 0.5))
    views = [TrainView(camera, torch.full((16, 16, 3), 0.5))]
    config = TrainConfig(iterations=4, prune_contribution=0.3, prune_finetune=3, eval_every=0)
    history = train(model, views, config=config, extent=1.0)
    assert len(history) == 7
    assert len(model) == 7


# --- added: Adam moments, checkpoint / resume, no opacity reset ----------- #

from gstrain.checkpoint import load_checkpoint, save_checkpoint


def test_relocation_resets_adam_moments() -> None:
    model = _model(torch.tensor([[0.0, 0.0, 3.0], [0.2, 0.0, 3.0], [0.4, 0.0, 3.0]]),
                   torch.full((3, 3), 0.1), torch.tensor([0.8, 0.001, 0.6]))
    optimizer = _with_optimizer(model)
    strategy = MCMCStrategy(model, optimizer, MCMCConfig(seed=1), iterations=1000)
    before = model.xyz.detach().clone()
    assert strategy.relocate() == 1
    # The dead splat now sits on one of the live ones; that source's moments are zero.
    source = int((model.xyz[1] - before).norm(dim=-1).argmin())
    assert source in (0, 2)
    other = 2 - source
    for group in optimizer.param_groups:
        state = optimizer.state[group["params"][0]]
        assert torch.count_nonzero(state["exp_avg"][source]) == 0, group["name"]
        assert torch.count_nonzero(state["exp_avg_sq"][source]) == 0, group["name"]
    # The untouched live splat keeps its moments.
    assert torch.count_nonzero(optimizer.state[model.xyz]["exp_avg_sq"][other]) > 0


def test_mcmc_checkpoint_resume(tmp_path) -> None:
    views = [TrainView(_camera(), torch.full((32, 32, 3), 0.4)) for _ in range(2)]
    # 5 % growth needs >= 20 splats to add one (int(1.05 * N)), as in gsplat.
    mcmc = MCMCConfig(cap_max=50, refine_start_iteration=1, refine_every=2, refine_stop_fraction=1.0)
    config = TrainConfig(iterations=20, strategy="mcmc", mcmc=mcmc, eval_every=0)
    xyz = torch.stack([torch.linspace(-0.4, 0.4, 40), torch.zeros(40), torch.full((40,), 3.0)], dim=-1)
    model = _model(xyz, torch.full((40, 3), 0.05), torch.full((40,), 0.5))
    saved = {}

    def checkpoint(iteration, m, optimizer):
        saved[iteration] = save_checkpoint(tmp_path / f"c{iteration}.pt", m, optimizer, iteration)

    train(model, views, config, extent=1.0, checkpoint=checkpoint, checkpoint_every=10)
    restored, state = load_checkpoint(saved[10], torch.device("cpu"))
    assert 40 < len(restored) <= 50
    assert all(s["exp_avg"].shape[0] == len(restored) for s in state["optimizer"]["state"].values())

    history = train(restored, views, config, extent=1.0, resume=state)
    assert len(history) == 10
    assert len(restored) <= 50


def test_mcmc_never_resets_opacity() -> None:
    model = _model(torch.tensor([[0.0, 0.0, 3.0], [0.2, 0.0, 3.0]]), torch.full((2, 3), 0.1),
                   torch.tensor([0.9, 0.7]))
    optimizer = _with_optimizer(model)
    strategy = MCMCStrategy(model, optimizer, MCMCConfig(refine_start_iteration=10**9), iterations=30_000)
    before = model.opacity.detach().clone()
    for iteration in (3000, 6000, 9000):  # the 3DGS reset iterations
        strategy.after_step(iteration, lr_xyz=1e-4)
    assert torch.allclose(model.opacity, before)
