"""Small torch helpers shared by the model and the rasterizer."""

from __future__ import annotations

import math

import torch


def build_rotation(quaternions: torch.Tensor) -> torch.Tensor:
    """Normalised [w,x,y,z] quaternions -> (N,3,3) rotation matrices.

    Same convention as SplatCalculator::buildCovariance in the C++ plugin.
    """
    q = torch.nn.functional.normalize(quaternions, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], dim=-1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], dim=-1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=-1),
        ],
        dim=1,
    )


def build_covariance(scales: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    """Sigma = M M^T with M = R diag(s), matching splatCalculator.cpp."""
    m = build_rotation(rotations) * scales.unsqueeze(1)
    return m @ m.transpose(1, 2)


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, 1e-7, 1.0 - 1e-7)
    return torch.log(x / (1.0 - x))


def get_expon_lr_func(lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1_000_000):
    """Log-linear decay with an optional warm-up damp, as in the 3DGS reference."""

    def helper(step: int) -> float:
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * math.sin(
                0.5 * math.pi * min(max(step / lr_delay_steps, 0.0), 1.0)
            )
        else:
            delay_rate = 1.0
        t = min(max(step / max_steps, 0.0), 1.0)
        return delay_rate * math.exp(math.log(lr_init) * (1 - t) + math.log(lr_final) * t)

    return helper
