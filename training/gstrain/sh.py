"""Real spherical harmonics helpers (numpy only).

C0 must stay in sync with src/ply/StandardSplatDecoder.cpp, which decodes the
stored DC coefficient back to colour as `0.5 + C0 * f_dc`.
"""

from __future__ import annotations

import numpy as np

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)


def num_sh_coeffs(degree: int) -> int:
    """Total coefficients per colour channel for an SH of the given degree."""
    return (degree + 1) ** 2


def rgb_to_sh(rgb: np.ndarray) -> np.ndarray:
    """Linear RGB in [0,1] -> degree-0 SH coefficient."""
    return (np.asarray(rgb, dtype=np.float32) - 0.5) / C0


def sh_to_rgb(sh: np.ndarray) -> np.ndarray:
    """Degree-0 SH coefficient -> linear RGB (unclamped)."""
    return np.asarray(sh, dtype=np.float32) * C0 + 0.5


def eval_sh(degree: int, sh: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Evaluate SH of `degree` at unit directions.

    sh:   (N, K, 3) coefficients, K >= (degree+1)**2
    dirs: (N, 3) unit vectors
    """
    sh = np.asarray(sh, dtype=np.float32)
    dirs = np.asarray(dirs, dtype=np.float32)
    if sh.shape[1] < num_sh_coeffs(degree):
        raise ValueError(f"need {num_sh_coeffs(degree)} coefficients for degree {degree}")

    result = C0 * sh[:, 0]
    if degree == 0:
        return result

    x = dirs[:, 0:1]
    y = dirs[:, 1:2]
    z = dirs[:, 2:3]
    result = result - C1 * y * sh[:, 1] + C1 * z * sh[:, 2] - C1 * x * sh[:, 3]
    if degree == 1:
        return result

    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    result = (
        result
        + C2[0] * xy * sh[:, 4]
        + C2[1] * yz * sh[:, 5]
        + C2[2] * (2.0 * zz - xx - yy) * sh[:, 6]
        + C2[3] * xz * sh[:, 7]
        + C2[4] * (xx - yy) * sh[:, 8]
    )
    if degree == 2:
        return result

    result = (
        result
        + C3[0] * y * (3.0 * xx - yy) * sh[:, 9]
        + C3[1] * xy * z * sh[:, 10]
        + C3[2] * y * (4.0 * zz - xx - yy) * sh[:, 11]
        + C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh[:, 12]
        + C3[4] * x * (4.0 * zz - xx - yy) * sh[:, 13]
        + C3[5] * z * (xx - yy) * sh[:, 14]
        + C3[6] * x * (xx - 3.0 * yy) * sh[:, 15]
    )
    return result
