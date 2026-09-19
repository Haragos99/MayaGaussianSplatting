import numpy as np
import pytest

from gstrain import sh


def test_dc_roundtrip():
    rgb = np.array([[0.0, 0.5, 1.0], [0.25, 0.75, 0.1]], dtype=np.float32)
    np.testing.assert_allclose(sh.sh_to_rgb(sh.rgb_to_sh(rgb)), rgb, atol=1e-6)


def test_c0_matches_the_cpp_decoder():
    assert sh.C0 == pytest.approx(0.28209479177387814, abs=0.0)


def test_num_coeffs():
    assert [sh.num_sh_coeffs(d) for d in range(4)] == [1, 4, 9, 16]


def test_degree_zero_eval_equals_dc_formula():
    coeffs = np.zeros((2, 16, 3), dtype=np.float32)
    coeffs[:, 0, :] = sh.rgb_to_sh(np.array([[0.2, 0.4, 0.6], [0.9, 0.1, 0.5]], dtype=np.float32))
    dirs = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    value = sh.eval_sh(0, coeffs, dirs) + 0.5
    np.testing.assert_allclose(value, [[0.2, 0.4, 0.6], [0.9, 0.1, 0.5]], atol=1e-6)


def test_higher_bands_are_direction_dependent():
    coeffs = np.zeros((1, 16, 3), dtype=np.float32)
    coeffs[:, 2, :] = 1.0  # the z band of degree 1
    up = sh.eval_sh(1, coeffs, np.array([[0.0, 0.0, 1.0]], dtype=np.float32))
    down = sh.eval_sh(1, coeffs, np.array([[0.0, 0.0, -1.0]], dtype=np.float32))
    np.testing.assert_allclose(up, -down, atol=1e-6)
    assert np.abs(up).max() > 0.1


def test_eval_rejects_too_few_coefficients():
    with pytest.raises(ValueError):
        sh.eval_sh(3, np.zeros((1, 9, 3), dtype=np.float32), np.zeros((1, 3), dtype=np.float32))
