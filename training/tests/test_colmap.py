import numpy as np
import pytest

from gstrain import cameras as cam_math
from gstrain import colmap


def test_binary_and_text_models_agree(colmap_binary, colmap_text):
    a = colmap.read_model(colmap_binary)
    b = colmap.read_model(colmap_text)

    assert len(a.cameras) == len(b.cameras) == 1
    assert len(a.images) == len(b.images) == 6
    assert len(a.points.ids) == len(b.points.ids) == 40

    for key in a.images:
        np.testing.assert_allclose(a.images[key].qvec, b.images[key].qvec, atol=1e-9)
        np.testing.assert_allclose(a.images[key].tvec, b.images[key].tvec, atol=1e-9)
        assert a.images[key].name == b.images[key].name
    np.testing.assert_allclose(a.points.xyz, b.points.xyz, atol=1e-9)
    np.testing.assert_array_equal(a.points.rgb, b.points.rgb)


def test_camera_center_matches_minus_r_transpose_t(colmap_binary):
    model = colmap.read_model(colmap_binary)
    for image in model.images.values():
        R = image.R_w2c
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)
        np.testing.assert_allclose(image.camera_center, -R.T @ image.tvec, atol=1e-12)


def test_quaternion_roundtrip(colmap_binary):
    model = colmap.read_model(colmap_binary)
    for image in model.images.values():
        q = colmap.rotmat_to_qvec(colmap.qvec_to_rotmat(image.qvec))
        if np.dot(q, image.qvec) < 0:
            q = -q
        np.testing.assert_allclose(q, image.qvec, atol=1e-9)


def test_cameras_look_at_the_origin(colmap_binary):
    """The synthetic rig points at the origin, so origin must project near the principal point."""
    model = colmap.read_model(colmap_binary)
    cam = model.cameras[1]
    fx, fy, cx, cy = cam.intrinsics()
    for image in model.images.values():
        p_view = image.R_w2c @ np.zeros(3) + image.tvec
        assert p_view[2] > 0, "the origin must be in front of the camera"
        assert fx * p_view[0] / p_view[2] + cx == pytest.approx(cx, abs=1e-6)
        assert fy * p_view[1] / p_view[2] + cy == pytest.approx(cy, abs=1e-6)


def test_radial_model_is_supported_and_carries_distortion():
    cam = colmap.Camera(1, "SIMPLE_RADIAL", 64, 48, np.array([50.0, 32.0, 24.0, -0.02]))
    assert not cam.is_pinhole
    assert cam.intrinsics() == (50.0, 50.0, 32.0, 24.0)
    assert cam.distortion() == colmap.Distortion(k1=-0.02)


def test_opencv_model_maps_all_four_coefficients():
    cam = colmap.Camera(1, "OPENCV", 64, 48, np.array([50.0, 51.0, 32.0, 24.0, 0.1, 0.2, 0.3, 0.4]))
    assert cam.intrinsics() == (50.0, 51.0, 32.0, 24.0)
    assert cam.distortion() == colmap.Distortion(k1=0.1, k2=0.2, p1=0.3, p2=0.4)


def test_unsupported_model_is_rejected():
    cam = colmap.Camera(1, "OPENCV_FISHEYE", 64, 48, np.zeros(8))
    with pytest.raises(ValueError, match="image_undistorter"):
        cam.intrinsics()


def test_scene_normalization_and_split(colmap_binary):
    model = colmap.read_model(colmap_binary)
    infos = cam_math.cameras_from_model(model, colmap_binary / "images", eval_hold=8)
    assert len(infos) == 6
    center, radius = cam_math.scene_normalization(infos)
    np.testing.assert_allclose(center, [0.0, 0.5, 0.0], atol=1e-6)
    assert radius == pytest.approx(3.0 * 1.1, abs=1e-6)
    assert sum(c.is_test for c in infos) == 1


def test_fov_focal_roundtrip():
    fov = cam_math.focal_to_fov(50.0, 64)
    assert cam_math.fov_to_focal(fov, 64) == pytest.approx(50.0)


def test_world_view_transform_is_row_vector(colmap_binary):
    model = colmap.read_model(colmap_binary)
    image = next(iter(model.images.values()))
    R = image.R_w2c.T
    m = cam_math.world_view_transform(R, image.tvec)
    p_world = np.array([0.1, -0.2, 0.3])
    expected = image.R_w2c @ p_world + image.tvec
    got = np.append(p_world, 1.0) @ m
    np.testing.assert_allclose(got[:3], expected, atol=1e-6)
    assert got[3] == pytest.approx(1.0)


def test_projection_matrix_maps_near_and_far_to_zero_and_one():
    p = cam_math.projection_matrix(1.0, 0.8, znear=0.01, zfar=100.0)
    for z, expected in ((0.01, 0.0), (100.0, 1.0)):
        clip = np.array([0.0, 0.0, z, 1.0]) @ p
        assert clip[2] / clip[3] == pytest.approx(expected, abs=1e-5)
