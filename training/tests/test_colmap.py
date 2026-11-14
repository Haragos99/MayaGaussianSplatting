import struct
import numpy as np
import pytest
from gstrain.cameras import create_training_camera
from gstrain.colmap import (
    CAMERA_MODEL_IDS,
    Camera,
    Image,
    find_colmap_dir,
    load_colmap,
    quaternion_to_rotmatrix,
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary,
    rotmatrix_to_quaternionc,
)


def _random_quaternions(count: int, seed: int = 0) -> np.ndarray:
    q = np.random.default_rng(seed).normal(size=(count, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def test_quaternion_rotation_roundtrip() -> None:
    for q in _random_quaternions(50):
        R = quaternion_to_rotmatrix(q)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(R), 1.0)
        back = rotmatrix_to_quaternionc(R)
        assert back[0] >= 0
        # q and -q are the same rotation.
        assert np.isclose(abs(np.dot(back, q)), 1.0, atol=1e-9)
    # All four branches of the matrix -> quaternion conversion.
    for R in (np.eye(3), np.diag([1.0, -1.0, -1.0]), np.diag([-1.0, 1.0, -1.0]), np.diag([-1.0, -1.0, 1.0])):
        assert np.allclose(quaternion_to_rotmatrix(rotmatrix_to_quaternionc(R)), R)


def test_camera_center_is_minus_Rt_t() -> None:
    q = _random_quaternions(1, seed=3)[0]
    t = np.array([0.3, -1.0, 2.0])
    image = Image(1, q, t, 1, "a.png")
    R = quaternion_to_rotmatrix(q)
    center = image.camera_center()
    assert np.allclose(center, -R.T @ t)
    # The centre maps to the camera origin.
    assert np.allclose(R @ center + t, 0.0)


def _write_binary(directory, data) -> None:
    with open(directory / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", len(data.cameras)))
        for camera in data.cameras.values():
            f.write(struct.pack("<iiQQ", camera.id, CAMERA_MODEL_IDS[camera.model], camera.width, camera.height))
            f.write(struct.pack("<" + "d" * len(camera.params), *camera.params))
    with open(directory / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", len(data.images)))
        for image in data.images.values():
            f.write(struct.pack("<idddddddi", image.id, *image.qvec, *image.tvec, image.camera_id))
            f.write(image.name.encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", 1))
            f.write(struct.pack("<ddq", 1.0, 2.0, -1))  # one 2D observation, skipped by the reader
    with open(directory / "points3D.bin", "wb") as f:
        points = data.points3D
        f.write(struct.pack("<Q", len(points.ids)))
        for i in range(len(points.ids)):
            f.write(struct.pack("<QdddBBBd", int(points.ids[i]), *points.xyz[i], *(int(c) for c in points.rgb[i]),
                                float(points.error[i])))
            f.write(struct.pack("<Q", 2))
            f.write(struct.pack("<iiii", 1, 0, 2, 0))  # track, skipped by the reader


def test_txt_and_binary_readers_agree(tiny_scene, tmp_path) -> None:
    text = load_colmap(tiny_scene)
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    _write_binary(binary_dir, text)

    cameras = read_cameras_binary(binary_dir / "cameras.bin")
    images = read_images_binary(binary_dir / "images.bin")
    points = read_points3d_binary(binary_dir / "points3D.bin")

    assert cameras.keys() == text.cameras.keys()
    for key, camera in cameras.items():
        expected = text.cameras[key]
        assert (camera.model, camera.width, camera.height) == (expected.model, expected.width, expected.height)
        assert np.allclose(camera.params, expected.params)
    assert images.keys() == text.images.keys()
    for key, image in images.items():
        expected = text.images[key]
        assert image.name == expected.name and image.camera_id == expected.camera_id
        assert np.allclose(image.qvec, expected.qvec) and np.allclose(image.tvec, expected.tvec)
    assert np.array_equal(points.ids, text.points3D.ids)
    assert np.allclose(points.xyz, text.points3D.xyz)
    assert np.array_equal(points.rgb, text.points3D.rgb)

    # load_colmap prefers .bin when both exist.
    assert load_colmap(binary_dir).path == binary_dir


def test_truncated_binary_raises(tmp_path) -> None:
    (tmp_path / "cameras.bin").write_bytes(struct.pack("<Q", 2) + b"\x00" * 10)
    with pytest.raises(EOFError):
        read_cameras_binary(tmp_path / "cameras.bin")


def test_find_colmap_dir_layouts(tmp_path) -> None:
    for layout in ("", "sparse", "sparse/0"):
        root = tmp_path / (layout.replace("/", "_") or "root")
        model = root / layout if layout else root
        model.mkdir(parents=True)
        (model / "cameras.txt").write_text("")
        assert find_colmap_dir(root) == model
    with pytest.raises(FileNotFoundError, match="no COLMAP model"):
        find_colmap_dir(tmp_path / "empty")


@pytest.mark.parametrize(
    "model, params, expected, distortion",
    [
        ("PINHOLE", [10, 11, 5, 6], (10, 11, 5, 6), (0, 0, 0, 0)),
        ("SIMPLE_PINHOLE", [10, 5, 6], (10, 10, 5, 6), (0, 0, 0, 0)),
        ("SIMPLE_RADIAL", [10, 5, 6, 0.02], (10, 10, 5, 6), (0.02, 0, 0, 0)),
        ("RADIAL", [10, 5, 6, 0.02, -0.01], (10, 10, 5, 6), (0.02, -0.01, 0, 0)),
        ("OPENCV", [10, 11, 5, 6, 0.1, 0.2, 0.3, 0.4], (10, 11, 5, 6), (0.1, 0.2, 0.3, 0.4)),
    ],
)
def test_training_camera_models(model, params, expected, distortion) -> None:
    camera = Camera(1, model, 20, 12, np.array(params, dtype=np.float64))
    image = Image(7, np.array([1.0, 0, 0, 0]), np.array([0.0, 0.0, 2.0]), 1, "x.png")
    training = create_training_camera(camera, image)
    assert (training.fx, training.fy, training.cx, training.cy) == expected
    assert np.allclose(training.distortion, distortion)
    assert np.allclose(training.K, [[expected[0], 0, expected[2]], [0, expected[1], expected[3]], [0, 0, 1]])
    assert np.allclose(training.C, [0.0, 0.0, -2.0])


def test_unsupported_camera_model_raises() -> None:
    camera = Camera(1, "OPENCV_FISHEYE", 20, 12, np.zeros(8))
    image = Image(1, np.array([1.0, 0, 0, 0]), np.zeros(3), 1, "x.png")
    with pytest.raises(NotImplementedError):
        create_training_camera(camera, image)
