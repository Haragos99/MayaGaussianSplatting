import numpy as np
import pytest

from gstrain import colmap, verify_ply
from gstrain.export_ply import attribute_names, flatten_features_rest, write_ply
from gstrain.params import create_from_point_cloud, sigmoid

MODELS = __import__("pathlib").Path(__file__).resolve().parents[2] / "models"
PIPEWORK_PLY = MODELS / "pipework.ply"
SUPERSPLAT_PLY = MODELS / "CA.ply"


@pytest.fixture
def params(colmap_binary):
    model = colmap.read_model(colmap_binary)
    return create_from_point_cloud(model.points.xyz, model.points.rgb, sh_degree=3)


def test_initial_params_are_sane(params):
    assert len(params) == 40
    assert params.sh_degree == 3
    assert params.features_rest.shape == (40, 15, 3)
    np.testing.assert_allclose(params.features_rest, 0.0)
    np.testing.assert_allclose(params.rotation, np.tile([1.0, 0.0, 0.0, 0.0], (40, 1)))
    np.testing.assert_allclose(sigmoid(params.opacity), 0.1, atol=1e-6)
    assert np.isfinite(params.scaling).all()
    assert params.activated_scale().max() < 10.0


def test_degree_three_writes_62_properties(params, tmp_path):
    path = write_ply(tmp_path / "out.ply", params)
    _, names = verify_ply.read_raw(path)
    assert len(names) == 62
    assert names == attribute_names(45)
    assert names[:6] == ["x", "y", "z", "nx", "ny", "nz"]
    assert names[9:12] == ["f_rest_0", "f_rest_1", "f_rest_2"]
    assert names[-8:] == ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]


def test_roundtrip_matches_the_cpp_activations(params, tmp_path):
    path = write_ply(tmp_path / "out.ply", params)
    decoded = verify_ply.decode(path)

    assert len(decoded) == len(params)
    np.testing.assert_allclose(decoded.xyz, params.xyz, atol=1e-6)
    np.testing.assert_allclose(decoded.scale, params.activated_scale(), rtol=1e-6)
    np.testing.assert_allclose(decoded.opacity, params.activated_opacity()[:, 0], atol=1e-6)
    np.testing.assert_allclose(decoded.color, params.activated_color(), atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(decoded.rotation, axis=1), 1.0, atol=1e-6)


def test_decoded_colour_returns_the_source_rgb(colmap_binary, tmp_path):
    model = colmap.read_model(colmap_binary)
    params = create_from_point_cloud(model.points.xyz, model.points.rgb)
    decoded = verify_ply.decode(write_ply(tmp_path / "out.ply", params))
    np.testing.assert_allclose(decoded.color, model.points.rgb / 255.0, atol=2e-3)


def test_f_rest_is_channel_major():
    rest = np.arange(2 * 15 * 3, dtype=np.float32).reshape(2, 15, 3)
    flat = flatten_features_rest(rest)
    assert flat.shape == (2, 45)
    np.testing.assert_allclose(flat[0, :15], rest[0, :, 0])
    np.testing.assert_allclose(flat[0, 15:30], rest[0, :, 1])
    np.testing.assert_allclose(flat[0, 30:], rest[0, :, 2])


def test_header_is_binary_little_endian(params, tmp_path):
    path = write_ply(tmp_path / "out.ply", params)
    head = path.read_bytes()[:64]
    assert head.startswith(b"ply\nformat binary_little_endian 1.0\nelement vertex 40\n")


def test_maya_y_up_axis_flips_y_and_z(params, tmp_path):
    straight = verify_ply.decode(write_ply(tmp_path / "a.ply", params, up_axis="colmap"))
    flipped = verify_ply.decode(write_ply(tmp_path / "b.ply", params, up_axis="maya-y"))
    np.testing.assert_allclose(flipped.xyz[:, 0], straight.xyz[:, 0], atol=1e-6)
    np.testing.assert_allclose(flipped.xyz[:, 1], -straight.xyz[:, 1], atol=1e-6)
    np.testing.assert_allclose(flipped.xyz[:, 2], -straight.xyz[:, 2], atol=1e-6)
    np.testing.assert_allclose(flipped.scale, straight.scale, rtol=1e-6)


def test_up_axis_rotation_is_consistent_with_the_covariance(params, tmp_path):
    """Sigma' must equal R Sigma R^T, matching src/splatCalculator.cpp."""
    rng = np.random.default_rng(3)
    params.rotation[:] = rng.normal(size=params.rotation.shape).astype(np.float32)
    params.scaling[:] = rng.normal(scale=0.2, size=params.scaling.shape).astype(np.float32)

    straight = verify_ply.decode(write_ply(tmp_path / "a.ply", params, up_axis="colmap"))
    flipped = verify_ply.decode(write_ply(tmp_path / "b.ply", params, up_axis="maya-y"))
    world = np.diag([1.0, -1.0, -1.0])

    for i in range(0, len(params), 7):
        m = colmap.qvec_to_rotmat(straight.rotation[i]) @ np.diag(straight.scale[i])
        sigma = m @ m.T
        m2 = colmap.qvec_to_rotmat(flipped.rotation[i]) @ np.diag(flipped.scale[i])
        np.testing.assert_allclose(m2 @ m2.T, world @ sigma @ world.T, atol=1e-5)


@pytest.mark.skipif(not PIPEWORK_PLY.exists(), reason="models/pipework.ply not present")
def test_reader_handles_the_plain_rgb_point_cloud_fallback():
    """The C++ decoder falls back to red/green/blue uchar when f_dc_* is absent."""
    decoded = verify_ply.decode(PIPEWORK_PLY)
    assert len(decoded) > 0
    assert decoded.color.min() >= 0.0 and decoded.color.max() <= 1.0
    np.testing.assert_allclose(decoded.opacity, 1.0)


@pytest.mark.skipif(not SUPERSPLAT_PLY.exists(), reason="models/CA.ply not present")
def test_compressed_ply_is_reported_clearly():
    with pytest.raises(ValueError, match="SuperSplat compressed"):
        verify_ply.decode(SUPERSPLAT_PLY)
