
import numpy as np
import torch

from gstrain.checkpoint import load_checkpoint, save_checkpoint
from gstrain.dataset import TrainView, split_views
from gstrain.densification import DensificationConfig
from gstrain.export_ply import read_ply, verify_ply, write_ply
from gstrain.geometry.camera import TorchCamera
from gstrain.loss import psnr
from gstrain.model import initialize_gaussians, knn_mean_distance
from gstrain.colmap import Points3D
from gstrain.model_torch import TorchGaussianModel
from gstrain.render.covariance import covariance_2d, covariance_3d
from gstrain.render.projection import world_to_camera
from gstrain.render.renderer_torch import render_gaussians
from gstrain.sh import eval_sh, rgb_to_sh, sh_to_rgb
from gstrain.train import TrainConfig, ViewSampler, create_optimizer, train
from gstrain.undistort import distort, undistort_image


def _camera(size: int = 16, focal: float = 16.0) -> TorchCamera:
    return TorchCamera(
        width=size, height=size,
        fx=torch.tensor(focal), fy=torch.tensor(focal),
        cx=torch.tensor(size / 2), cy=torch.tensor(size / 2),
        R=torch.eye(3), t=torch.zeros(3),
    )


def _splat_at(u: float, v: float, camera: TorchCamera, z: float = 2.0, size: float = 0.005,
              opacity: float = 0.9) -> TorchGaussianModel:
    x = (u - float(camera.cx)) * z / float(camera.fx)
    y = (v - float(camera.cy)) * z / float(camera.fy)
    return TorchGaussianModel(
        torch.tensor([[x, y, z]]), torch.full((1, 3), size),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]), torch.tensor([opacity]), torch.ones(1, 3),
    )


# --- pixel centres -------------------------------------------------------- #

def test_splat_on_pixel_centre_peaks_at_that_pixel() -> None:
    camera = _camera()
    image = render_gaussians(camera, _splat_at(5.5, 9.5, camera))[..., 0]
    row, col = divmod(int(image.argmax()), camera.width)
    assert (row, col) == (9, 5)
    # Exactly centred, so the left/right and up/down neighbours match.
    assert torch.isclose(image[9, 4], image[9, 6], atol=1e-6)
    assert torch.isclose(image[8, 5], image[10, 5], atol=1e-6)


def test_rescaled_camera_keeps_pixel_centres() -> None:
    camera = _camera()
    model = _splat_at(5.5, 9.5, camera)  # world point fixed by the 16 px camera
    small = camera.rescaled(8, 8)        # same point is at (2.75, 4.75) there
    image = render_gaussians(small, model)[..., 0]
    row, col = divmod(int(image.argmax()), small.width)
    assert (row, col) == (4, 2)


# --- EWA details ---------------------------------------------------------- #

def test_dilation_makes_tiny_splats_at_least_subpixel() -> None:
    camera = _camera()
    points = torch.tensor([[0.0, 0.0, 2.0]])
    cov = covariance_2d(camera, points, covariance_3d(torch.full((1, 3), 1e-6), torch.tensor([[1.0, 0, 0, 0]])))
    assert torch.allclose(torch.diagonal(cov[0]), torch.full((2,), 0.3), atol=1e-4)


def test_jacobian_clamp_bounds_offscreen_footprints() -> None:
    camera = _camera()
    rotation = torch.tensor([[1.0, 0, 0, 0]])
    cov3d = covariance_3d(torch.full((1, 3), 0.05), rotation)
    near_edge = covariance_2d(camera, torch.tensor([[0.6, 0.0, 1.0]]), cov3d)   # just outside
    far_out = covariance_2d(camera, torch.tensor([[50.0, 0.0, 1.0]]), cov3d)    # way outside
    # Without the clamp the far splat's footprint would be ~5000x larger.
    assert far_out[0, 0, 0] < 2 * near_edge[0, 0, 0]


# --- background ----------------------------------------------------------- #

def test_background_fills_uncovered_pixels() -> None:
    camera = _camera()
    model = _splat_at(5.5, 9.5, camera)
    image = render_gaussians(camera, model, background=torch.tensor([0.0, 1.0, 0.0]))
    assert torch.allclose(image[0, 15], torch.tensor([0.0, 1.0, 0.0]))       # empty tile
    assert torch.allclose(image[9, 0, 1], torch.tensor(1.0), atol=1e-3)      # busy tile, no splat there


# --- undistortion --------------------------------------------------------- #

def test_undistort_identity_and_sampling_position() -> None:
    height, width = 80, 100
    fx = fy = 60.0
    cx, cy = 50.0, 40.0
    ramp = np.tile(np.arange(width, dtype=np.float64)[None, :, None], (height, 1, 3))

    assert undistort_image(ramp, fx, fy, cx, cy, (0.0, 0.0, 0.0, 0.0)) is ramp

    distortion = (0.1, 0.0, 0.0, 0.0)
    out = undistort_image(ramp, fx, fy, cx, cy, distortion)
    row, col = 10, 80
    xd, _ = distort(np.array((col + 0.5 - cx) / fx), np.array((row + 0.5 - cy) / fy), distortion)
    # The ramp's value is the column index, so the output shows where it sampled.
    assert np.isclose(out[row, col, 0], fx * xd + cx - 0.5, atol=1e-6)


# --- metrics, split, sampler ---------------------------------------------- #

def test_psnr_values() -> None:
    a = torch.zeros(4, 4, 3)
    assert float(psnr(a, a)) >= 119.9
    assert np.isclose(float(psnr(a, a + 0.1)), 20.0, atol=1e-4)


def test_split_holds_out_every_eighth_view() -> None:
    views = [TrainView(camera=None, image=None, name=str(i)) for i in range(17)]
    train_views, test_views = split_views(views, 8)
    assert [v.name for v in test_views] == ["0", "8", "16"]
    assert len(train_views) == 14
    assert split_views(views, 0) == (views, [])


def test_view_sampler_covers_every_view_each_epoch() -> None:
    views = [TrainView(camera=None, image=None, name=str(i)) for i in range(7)]
    sampler = ViewSampler(views, seed=3)
    for _ in range(3):
        assert sorted(sampler.next().name for _ in range(7)) == sorted(v.name for v in views)


# --- initialization ------------------------------------------------------- #

def test_init_has_no_zero_size_splats_for_duplicate_points() -> None:
    xyz = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    points = Points3D(ids=np.arange(5), xyz=xyz, rgb=np.full((5, 3), 128), error=np.zeros(5))
    model = initialize_gaussians(points)
    assert (model.scale > 0.1).all()          # 1-NN would give 0 for the duplicate pair
    assert np.allclose(model.opacity, 0.1)
    # (1, 0, 0): the duplicate pair at distance 1 twice, then (0, 1, 0) at sqrt(2).
    assert np.isclose(knn_mean_distance(xyz)[2], (1 + 1 + np.sqrt(2)) / 3)


# --- spherical harmonics -------------------------------------------------- #

def test_sh_degree_zero_is_plain_rgb() -> None:
    rgb = torch.rand(10, 3)
    assert torch.allclose(sh_to_rgb(rgb_to_sh(rgb)), rgb, atol=1e-6)

    model = TorchGaussianModel(torch.rand(10, 3) + 2, torch.full((10, 3), 0.1),
                               torch.tensor([[1.0, 0, 0, 0]]).repeat(10, 1), torch.full((10,), 0.5), rgb)
    assert torch.allclose(model.colors_for(torch.zeros(3)), rgb, atol=1e-6)

    # With all higher bands zero, every degree gives the same colour.
    dirs = torch.nn.functional.normalize(torch.randn(10, 3), dim=-1)
    assert torch.allclose(eval_sh(3, model.features, dirs), eval_sh(0, model.features, dirs), atol=1e-6)


def test_ply_roundtrip_with_sh_and_maya_axis(tmp_path) -> None:
    model = TorchGaussianModel(torch.rand(5, 3), torch.full((5, 3), 0.1),
                               torch.tensor([[1.0, 0, 0, 0]]).repeat(5, 1), torch.full((5,), 0.5),
                               torch.rand(5, 3))
    with torch.no_grad():
        model.sh_rest.copy_(torch.randn_like(model.sh_rest))

    path = write_ply(tmp_path / "a.ply", model)
    fields = read_ply(path)
    rest = model.sh_rest.detach().numpy()
    # Channel-major: f_rest_0..14 are red bands 1..15, f_rest_15 is green band 1.
    assert np.allclose(fields["f_rest_0"], rest[:, 0, 0])
    assert np.allclose(fields["f_rest_15"], rest[:, 0, 1])
    assert np.allclose(fields["f_rest_44"], rest[:, 14, 2])
    assert verify_ply(path)["sh_degree"] == 3

    maya = read_ply(write_ply(tmp_path / "b.ply", model, up_axis="maya-y"))
    assert np.allclose(maya["y"], -fields["y"]) and np.allclose(maya["z"], -fields["z"])


# --- checkpoints ---------------------------------------------------------- #

def test_checkpoint_roundtrip_and_resume(tmp_path) -> None:
    model = _splat_at(5.5, 9.5, _camera())
    optimizer = create_optimizer(model, TrainConfig(iterations=2), extent=1.0)
    model.active_sh_degree = 2
    path = save_checkpoint(tmp_path / "c.pt", model, optimizer, iteration=5)

    restored, state = load_checkpoint(path, torch.device("cpu"))
    assert state["iteration"] == 5 and restored.active_sh_degree == 2
    for name in ("xyz", "scale_raw", "opacity_raw", "sh_dc", "sh_rest"):
        assert torch.equal(getattr(restored, name), getattr(model, name))

    views = [TrainView(_camera(), torch.zeros(16, 16, 3)), TrainView(_camera(), torch.zeros(16, 16, 3))]
    config = TrainConfig(iterations=7, eval_every=0,
                         densification=DensificationConfig(start_iteration=100, end_iteration=101))
    history = train(restored, views, config=config, extent=1.0, resume=state)
    assert len(history) == 2  # iterations 6 and 7


def test_training_reports_held_out_metrics(tmp_path) -> None:
    from gstrain.run_log import RunLog
    import json

    camera = _camera()
    target = render_gaussians(camera, _splat_at(5.5, 9.5, camera)).detach()
    views = [TrainView(camera, target), TrainView(camera, target)]
    model = _splat_at(6.5, 8.5, camera, opacity=0.5)
    with RunLog(tmp_path) as log:
        train(model, views[:1], TrainConfig(iterations=4, eval_every=2, log_every=1), extent=1.0,
              test_views=views[1:], run_log=log)
    records = [json.loads(line) for line in (tmp_path / "progress.jsonl").read_text().splitlines()]
    evals = [r for r in records if "test_psnr" in r]
    assert [r["iteration"] for r in evals] == [2, 4]
    assert all(np.isfinite(r["test_psnr"]) for r in evals)


# --- added: undistortion field, view selection, LR schedule, backgrounds -- #

import pytest

from gstrain.colmap import load_colmap
from gstrain.dataset import build_views
from gstrain.train import background_color, exponential_lr


def test_undistort_whole_image_samples_distorted_positions() -> None:
    """Every interior output pixel samples where k1 = 0.025 moves its ideal ray (MeSP-like)."""
    height, width, f, k1 = 60, 90, 70.0, 0.025
    cx, cy = width / 2, height / 2
    u, v = np.meshgrid(np.arange(width) + 0.0, np.arange(height) + 0.0)
    coordinates = np.stack([u, v, np.zeros_like(u)], axis=-1)  # value = array index
    out = undistort_image(coordinates, f, f, cx, cy, (k1, 0.0, 0.0, 0.0))

    # Independent formula: x_d = x (1 + k1 r^2), pixel centres at i + 0.5.
    x = (u + 0.5 - cx) / f
    y = (v + 0.5 - cy) / f
    radial = 1 + k1 * (x * x + y * y)
    expected_u = f * x * radial + cx - 0.5
    expected_v = f * y * radial + cy - 0.5
    inside = (expected_u >= 0) & (expected_u <= width - 1) & (expected_v >= 0) & (expected_v <= height - 1)
    assert inside.mean() > 0.9
    assert np.abs(out[..., 0] - expected_u)[inside].max() < 1e-6
    assert np.abs(out[..., 1] - expected_v)[inside].max() < 1e-6


def test_build_views_sorted_and_limited(tiny_scene) -> None:
    data = load_colmap(tiny_scene)
    device = torch.device("cpu")
    views = build_views(data, tiny_scene / "images", max_edge=16, device=device)
    names = [v.name for v in views]
    assert names == sorted(names) and len(names) == 8
    assert (views[0].camera.width, views[0].camera.height) == (16, 12)
    assert views[0].image.shape == (12, 16, 3)
    assert 0.0 <= float(views[0].image.min()) and float(views[0].image.max()) <= 1.0

    limited = build_views(data, tiny_scene / "images", max_edge=16, device=device, limit=3)
    assert [v.name for v in limited] == [names[0], names[2], names[5]]


def test_exponential_lr_endpoints() -> None:
    values = [exponential_lr(step, 1e-2, 1e-4, 100) for step in range(0, 101, 10)]
    assert np.isclose(values[0], 1e-2) and np.isclose(values[-1], 1e-4)
    assert all(a > b for a, b in zip(values, values[1:]))
    assert np.isclose(values[5], 1e-3)  # halfway in log space
    assert exponential_lr(500, 1e-2, 1e-4, 100) == pytest.approx(1e-4)  # clamped after the end
    assert exponential_lr(10, 0.0, 1e-4, 100) == 0.0


def test_random_background_changes_per_iteration() -> None:
    device = torch.device("cpu")
    g = torch.Generator().manual_seed(7)
    a, b = background_color("random", device, g), background_color("random", device, g)
    assert not torch.equal(a, b)
    assert torch.equal(a, background_color("random", device, torch.Generator().manual_seed(7)))
    assert torch.equal(background_color("white", device), torch.ones(3))
    with pytest.raises(ValueError):
        background_color("green", device)
