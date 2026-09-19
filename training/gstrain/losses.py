"""Photometric losses: L1 + D-SSIM, as in the 3DGS paper (lambda = 0.2)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

DSSIM_LAMBDA = 0.2
_WINDOW_SIZE = 11
_WINDOW_SIGMA = 1.5


def l1_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.abs(prediction - target).mean()


def _gaussian_window(channels: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(_WINDOW_SIZE, device=device, dtype=dtype) - _WINDOW_SIZE // 2
    g = torch.exp(-(coords**2) / (2.0 * _WINDOW_SIGMA**2))
    g = g / g.sum()
    window = g.unsqueeze(1) @ g.unsqueeze(0)
    return window.expand(channels, 1, _WINDOW_SIZE, _WINDOW_SIZE).contiguous()


def ssim(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean SSIM over a (C,H,W) or (B,C,H,W) pair in [0,1]."""
    if prediction.dim() == 3:
        prediction = prediction.unsqueeze(0)
        target = target.unsqueeze(0)
    channels = prediction.shape[1]
    window = _gaussian_window(channels, prediction.device, prediction.dtype)
    pad = _WINDOW_SIZE // 2

    mu1 = F.conv2d(prediction, window, padding=pad, groups=channels)
    mu2 = F.conv2d(target, window, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = F.conv2d(prediction * prediction, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(prediction * target, window, padding=pad, groups=channels) - mu1_mu2

    c1, c2 = 0.01**2, 0.03**2
    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    return (numerator / denominator).mean()


def photometric_loss(prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (total loss, plain L1) so the L1 term can be logged separately."""
    l1 = l1_loss(prediction, target)
    total = (1.0 - DSSIM_LAMBDA) * l1 + DSSIM_LAMBDA * (1.0 - ssim(prediction, target))
    return total, l1


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = ((prediction - target) ** 2).mean()
    return 20.0 * torch.log10(1.0 / torch.sqrt(torch.clamp(mse, min=1e-12)))


def mse_to_psnr(mse: float) -> float:
    return 20.0 * math.log10(1.0 / math.sqrt(max(mse, 1e-12)))
