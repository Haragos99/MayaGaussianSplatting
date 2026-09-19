"""Torch port of gstrain.sh. Constants are imported so the two cannot drift."""

from __future__ import annotations

import torch

from .sh import C0, C1, C2, C3, num_sh_coeffs


def eval_sh(degree: int, sh: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """sh: (N,K,3) coefficients, dirs: (N,3) unit vectors -> (N,3)."""
    if sh.shape[1] < num_sh_coeffs(degree):
        raise ValueError(f"need {num_sh_coeffs(degree)} coefficients for degree {degree}")

    result = C0 * sh[:, 0]
    if degree == 0:
        return result

    x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
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

    return (
        result
        + C3[0] * y * (3.0 * xx - yy) * sh[:, 9]
        + C3[1] * xy * z * sh[:, 10]
        + C3[2] * y * (4.0 * zz - xx - yy) * sh[:, 11]
        + C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh[:, 12]
        + C3[4] * x * (4.0 * zz - xx - yy) * sh[:, 13]
        + C3[5] * z * (xx - yy) * sh[:, 14]
        + C3[6] * x * (xx - 3.0 * yy) * sh[:, 15]
    )
