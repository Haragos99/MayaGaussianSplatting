import numpy as np
import torch

from gstrain.cameras import create_training_camera
from gstrain.colmap import Camera, Image, quaternion_to_rotmatrix
from gstrain.geometry.camera import TorchCamera, fit_resolution
from gstrain.geometry.quaternion import quaternion_to_rotation_matrix
from gstrain.render.covariance import covariance_3d
from gstrain.render.projection import project_points_Torch, world_to_camera


def test_quaternion_matrix_normalizes_and_ignores_sign() -> None:
    q = torch.randn(20, 4, generator=torch.Generator().manual_seed(0))
    unit = q / q.norm(dim=-1, keepdim=True)
    R = quaternion_to_rotation_matrix(q * 3.7)
    assert torch.allclose(R, quaternion_to_rotation_matrix(unit), atol=1e-6)
    assert torch.allclose(R, quaternion_to_rotation_matrix(-unit), atol=1e-6)
    assert torch.allclose(R @ R.transpose(-1, -2), torch.eye(3).expand(20, 3, 3), atol=1e-5)
    # Same convention as the NumPy COLMAP reader.
    assert np.allclose(R[0].numpy(), quaternion_to_rotmatrix(unit[0].numpy()), atol=1e-6)


def test_covariance_3d_is_symmetric_psd() -> None:
    g = torch.Generator().manual_seed(1)
    scale = 0.01 + torch.rand(10, 3, generator=g)
    rotation = torch.randn(10, 4, generator=g)
    cov = covariance_3d(scale, rotation)
    assert torch.allclose(cov, cov.transpose(-1, -2), atol=1e-6)
    eigen = torch.linalg.eigvalsh(cov.double())
    expected = scale.double().square().sort(dim=-1).values
    assert torch.allclose(eigen, expected, rtol=1e-4, atol=1e-7)


def test_fit_resolution() -> None:
    assert fit_resolution(1920, 1080, 480) == (480, 270)
    assert fit_resolution(1080, 1920, 480) == (270, 480)
    assert fit_resolution(320, 200, 480) == (320, 200)  # never upscales
    assert fit_resolution(5000, 3, 100) == (100, 1)     # never zero


def test_downscaled_projects_same_point() -> None:
    camera = TorchCamera(
        width=1920, height=1080,
        fx=torch.tensor(1500.0), fy=torch.tensor(1490.0),
        cx=torch.tensor(955.0), cy=torch.tensor(545.0),
        R=torch.eye(3), t=torch.zeros(3),
    )
    small = camera.downscaled(480)
    assert (small.width, small.height) == (480, 270)
    points = torch.tensor([[0.3, -0.2, 2.0], [-1.0, 0.5, 4.0]])
    uv, _, _ = project_points_Torch(camera, world_to_camera(camera, points))
    uv_small, _, _ = project_points_Torch(small, world_to_camera(small, points))
    relative = uv / torch.tensor([camera.width, camera.height])
    relative_small = uv_small / torch.tensor([small.width, small.height])
    assert torch.allclose(relative, relative_small, atol=1e-6)


def test_from_training_camera_world_to_camera() -> None:
    q = np.array([0.9, 0.1, -0.3, 0.2])
    q = q / np.linalg.norm(q)
    t = np.array([0.5, -0.2, 3.0])
    training = create_training_camera(
        Camera(1, "PINHOLE", 40, 30, np.array([35.0, 36.0, 20.0, 15.0])), Image(1, q, t, 1, "a.png")
    )
    camera = TorchCamera.from_training_camera(training, device=torch.device("cpu"))
    assert np.allclose(camera.R.numpy(), quaternion_to_rotmatrix(q), atol=1e-6)
    assert np.allclose(camera.t.numpy(), t)
    assert np.allclose(camera.camera_center.numpy(), training.C, atol=1e-5)
    assert (float(camera.fx), float(camera.fy), float(camera.cx), float(camera.cy)) == (35.0, 36.0, 20.0, 15.0)
