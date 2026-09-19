import numpy as np
import pytest

from gstrain.cameras import CameraInfo, cameras_from_model
from gstrain.colmap import Distortion, read_model
from gstrain.undistort import undistort_image


def checkerboard(width=64, height=48, square=8):
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    pattern = ((u // square + v // square) % 2).astype(np.float32)
    return np.stack([pattern, pattern, pattern], axis=-1)


def test_identity_distortion_is_a_no_op():
    image = checkerboard()
    out = undistort_image(image, 50.0, 50.0, 32.0, 24.0, Distortion())
    np.testing.assert_array_equal(out, image)


def test_distortion_changes_the_image_but_not_the_principal_point():
    image = checkerboard()
    out = undistort_image(image, 50.0, 50.0, 32.0, 24.0, Distortion(k1=-0.3))
    assert not np.allclose(out, image)
    np.testing.assert_allclose(out[24, 32], image[24, 32], atol=1e-6)


def test_pincushion_and_barrel_pull_in_opposite_directions():
    image = checkerboard()
    barrel = undistort_image(image, 50.0, 50.0, 32.0, 24.0, Distortion(k1=-0.3))
    pincushion = undistort_image(image, 50.0, 50.0, 32.0, 24.0, Distortion(k1=0.3))
    assert not np.allclose(barrel, pincushion)


def test_output_keeps_shape_and_range():
    image = checkerboard()
    out = undistort_image(image, 50.0, 50.0, 32.0, 24.0, Distortion(k1=0.1, k2=0.05, p1=0.01, p2=0.01))
    assert out.shape == image.shape
    assert out.dtype == image.dtype
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_camera_info_scales_the_principal_point(colmap_binary):
    infos = cameras_from_model(read_model(colmap_binary), colmap_binary / "images", eval_hold=0)
    full = infos[0]
    assert full.principal_x == pytest.approx(32.0)
    half = full.scaled(2)
    assert (half.width, half.height) == (32, 24)
    assert half.principal_x == pytest.approx(16.0)
    assert half.principal_y == pytest.approx(12.0)
    assert half.fov_x == pytest.approx(full.fov_x)


def test_principal_point_defaults_to_the_image_centre():
    info = CameraInfo(0, "a", None, 64, 48, 1.0, 0.8, np.eye(3), np.zeros(3))
    assert info.principal_x == 32.0
    assert info.principal_y == 24.0
    assert info.distortion.is_identity
