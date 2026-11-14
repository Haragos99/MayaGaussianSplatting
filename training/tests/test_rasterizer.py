import torch

from gstrain.geometry.camera import TorchCamera
from gstrain.model_torch import TorchGaussianModel
from gstrain.render.binning import cutoff_sigma
from gstrain.render.rasterizer import ALPHA_MIN
from gstrain.render.renderer_torch import render_gaussians


def _camera(size: int = 32) -> TorchCamera:
    return TorchCamera(
        width=size,
        height=size,
        fx=torch.tensor(float(size)),
        fy=torch.tensor(float(size)),
        cx=torch.tensor(size / 2),
        cy=torch.tensor(size / 2),
        R=torch.eye(3),
        t=torch.zeros(3),
    )


def _random_model(count: int, seed: int = 0) -> TorchGaussianModel:
    g = torch.Generator().manual_seed(seed)
    xyz = torch.rand(count, 3, generator=g) - 0.5
    xyz[:, 2] = 2.0 + torch.rand(count, generator=g)
    scale = 0.02 + 0.15 * torch.rand(count, 3, generator=g)
    rotation = torch.randn(count, 4, generator=g)
    opacity = 0.2 + 0.79 * torch.rand(count, generator=g)
    color = torch.rand(count, 3, generator=g)
    return TorchGaussianModel(xyz, scale, rotation, opacity, color)


def test_cutoff_sigma_matches_alpha_threshold() -> None:
    opacity = torch.tensor([0.999, 0.5, 0.05, ALPHA_MIN / 2])
    d = cutoff_sigma(opacity)
    alpha_at_edge = opacity * torch.exp(-0.5 * d * d)
    assert torch.allclose(alpha_at_edge[:3], torch.full((3,), ALPHA_MIN), rtol=1e-4)
    assert d[3] == 0


def test_render_does_not_depend_on_tile_size() -> None:
    """The old 3-sigma box clip and per-tile cap both made tile seams."""
    model = _random_model(200)
    camera = _camera()
    with torch.no_grad():
        images = [render_gaussians(camera, model, tile_size=t) for t in (4, 8, 16)]
    for image in images[1:]:
        assert torch.allclose(images[0], image, atol=1e-5)


def test_depth_slices_match_single_pass() -> None:
    model = _random_model(300, seed=1)
    camera = _camera()
    with torch.no_grad():
        whole = render_gaussians(camera, model, tile_size=8, depth_chunk=10_000)
        sliced = render_gaussians(camera, model, tile_size=8, depth_chunk=7)
    # Early exit may skip splats behind T < 1e-4, so allow that much error.
    assert torch.allclose(whole, sliced, atol=2e-4)


def test_gradients_flow_through_slices() -> None:
    model = _random_model(100, seed=2)
    model.active_sh_degree = model.max_sh_degree  # so the higher SH bands are used too
    render_gaussians(_camera(), model, tile_size=8, depth_chunk=5).mean().backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


# --- added: signature, reference renderer, early stop, gradients, contribution --- #

import inspect

import numpy as np
import torch.nn as nn

from gstrain.cameras import TrainingCamera
from gstrain.model import GaussianModel
from gstrain.render.rasterizer import rasterize_tiles
from gstrain.render.renderer_numpy import renderGS


def test_signature_has_all_parameters() -> None:
    """A manual edit once dropped these and broke every caller."""
    parameters = inspect.signature(rasterize_tiles).parameters
    for name in ("alpha_min", "alpha_max", "depth_chunk", "transmittance_min", "background", "contribution"):
        assert name in parameters, name
    assert parameters["transmittance_min"].default == 1e-4
    assert parameters["alpha_min"].default == ALPHA_MIN


def test_matches_numpy_reference() -> None:
    """The slow per-splat NumPy renderer is the readable reference for the tiled one."""
    size = 32
    xyz = np.array([[0.0, 0.0, 3.0], [0.15, -0.1, 2.5], [-0.2, 0.1, 3.5]], dtype=np.float32)
    scale = np.array([[0.08, 0.05, 0.05], [0.04, 0.06, 0.05], [0.1, 0.1, 0.1]], dtype=np.float32)
    rotation = np.array([[1.0, 0, 0, 0], [0.9, 0.2, 0.1, 0.3], [0.8, -0.3, 0.4, 0.1]], dtype=np.float32)
    rotation /= np.linalg.norm(rotation, axis=1, keepdims=True)
    opacity = np.array([0.7, 0.5, 0.9], dtype=np.float32)
    color = np.array([[0.9, 0.1, 0.1], [0.1, 0.8, 0.2], [0.2, 0.3, 0.9]], dtype=np.float32)

    torch_image = render_gaussians(
        _camera(size),
        TorchGaussianModel(*(torch.tensor(a) for a in (xyz, scale, rotation, opacity, color)), sh_degree=0),
        dilation=1e-6, alpha_min=1e-6,
    ).detach().numpy()

    # The NumPy renderer samples pixel i at i (not i + 0.5): shift its principal point to match.
    c = size / 2 - 0.5
    numpy_camera = TrainingCamera(
        image_id=0, name="", width=size, height=size, fx=float(size), fy=float(size), cx=c, cy=c,
        K=np.eye(3), R=np.eye(3), t=np.zeros(3), C=np.zeros(3),
    )
    numpy_image = renderGS(numpy_camera, GaussianModel(xyz, scale, rotation, opacity, color))

    difference = np.abs(torch_image - numpy_image)
    # The reference cuts each splat at a 3-sigma box; the tiled renderer does not.
    assert difference.max() < 2e-2
    assert difference.mean() < 1e-3


def test_early_stop_transmittance() -> None:
    """Splats behind an opaque wall change nothing, with or without the early break."""
    wall = [[0.0, 0.0, 2.0 + 0.01 * i] for i in range(5)]
    def model(with_hidden: bool) -> TorchGaussianModel:
        xyz = torch.tensor(wall + ([[0.0, 0.0, 4.0]] if with_hidden else []))
        n = len(xyz)
        color = torch.full((n, 3), 0.3)
        if with_hidden:
            color[-1] = torch.tensor([1.0, 0.0, 0.0])
        return TorchGaussianModel(xyz, torch.full((n, 3), 5.0), torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1),
                                  torch.full((n,), 0.999), color, sh_degree=0)
    for depth_chunk in (1, 2, 256):
        a = render_gaussians(_camera(), model(False), depth_chunk=depth_chunk)
        b = render_gaussians(_camera(), model(True), depth_chunk=depth_chunk)
        assert (a - b).abs().max() < 1e-3, depth_chunk


def _double_model() -> TorchGaussianModel:
    model = TorchGaussianModel(
        torch.tensor([[0.05, -0.03, 2.0], [-0.1, 0.08, 2.6]]),
        torch.tensor([[0.15, 0.1, 0.12], [0.2, 0.15, 0.1]]),
        torch.tensor([[0.95, 0.1, 0.2, 0.05], [0.9, -0.2, 0.1, 0.3]]),
        torch.tensor([0.6, 0.5]),
        torch.tensor([[0.8, 0.2, 0.3], [0.1, 0.7, 0.6]]),
        sh_degree=0,
    )
    for name, parameter in list(model.named_parameters()):
        setattr(model, name, nn.Parameter(parameter.detach().double()))
    return model


def test_gradcheck_all_parameters() -> None:
    """Autograd gradients agree with central finite differences (float64)."""
    size = 8
    camera = _camera(size)
    camera = TorchCamera(size, size, camera.fx.double(), camera.fy.double(), camera.cx.double(),
                         camera.cy.double(), camera.R.double(), camera.t.double())
    weights = torch.rand(size, size, 3, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    model = _double_model()

    def objective() -> torch.Tensor:
        # alpha_min tiny: no cutoff discontinuity inside the finite-difference step.
        return (render_gaussians(camera, model, tile_size=8, alpha_min=1e-12) * weights).sum()

    objective().backward()
    eps = 1e-6
    for name in ("xyz", "scale_raw", "rotation", "opacity_raw", "sh_dc"):
        parameter = getattr(model, name)
        analytic = parameter.grad.clone()
        flat = parameter.data.view(-1)
        for index in range(flat.numel()):
            original = float(flat[index])
            with torch.no_grad():
                flat[index] = original + eps
                plus = float(objective())
                flat[index] = original - eps
                minus = float(objective())
                flat[index] = original
            numeric = (plus - minus) / (2 * eps)
            assert abs(numeric - float(analytic.view(-1)[index])) < 1e-4 * max(1.0, abs(numeric)), (name, index)


def test_contribution_sums_to_coverage() -> None:
    """sum over splats of T*alpha = sum over pixels of (1 - final transmittance)."""
    model = _random_model(40, seed=4)
    with torch.no_grad():
        model.sh_dc.zero_()  # colour 0.5, so neither render is clipped
    camera = _camera()
    contribution = torch.zeros(len(model))
    black = render_gaussians(camera, model, contribution=contribution, depth_chunk=7)
    white = render_gaussians(camera, model, background=torch.ones(3), depth_chunk=7)
    transmittance = (white - black)[..., 0]
    assert torch.isclose(contribution.sum(), (1 - transmittance).sum(), rtol=1e-4)
    assert torch.all(contribution >= 0)
