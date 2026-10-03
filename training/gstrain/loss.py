import torch
import torch.nn.functional as F

# Stabilisers from the original SSIM paper, for data in [0, 1].
C1 = 0.01**2
C2 = 0.03**2


def l1_loss(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (rendered - target).abs().mean()


def _to_nchw(image: torch.Tensor) -> torch.Tensor:
    """(H, W, 3) -> (1, 3, H, W), the layout conv2d expects."""
    return image.permute(2, 0, 1).unsqueeze(0)


def _gaussian_window(
    window_size: int,
    sigma: float,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    line = torch.exp(-coords.square() / (2 * sigma**2))
    line = line / line.sum()

    window = line.outer(line)

    # One window per channel, applied independently via groups=channels.
    return window.expand(channels, 1, window_size, window_size).contiguous()


def ssim(
    rendered: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Mean structural similarity in [-1, 1]; 1 means identical."""
    x = _to_nchw(rendered)
    y = _to_nchw(target)

    channels = x.shape[1]
    window = _gaussian_window(window_size, sigma, channels, x.device, x.dtype)
    pad = window_size // 2

    def blur(image: torch.Tensor) -> torch.Tensor:
        return F.conv2d(image, window, padding=pad, groups=channels)

    mu_x = blur(x)
    mu_y = blur(y)

    mu_xx = mu_x * mu_x
    mu_yy = mu_y * mu_y
    mu_xy = mu_x * mu_y

    # Var(X) = E[X^2] - E[X]^2, computed with the same weighted window.
    sigma_xx = blur(x * x) - mu_xx
    sigma_yy = blur(y * y) - mu_yy
    sigma_xy = blur(x * y) - mu_xy

    numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_xx + mu_yy + C1) * (sigma_xx + sigma_yy + C2)

    return (numerator / denominator).mean()

"L = (1 - lambda) * L1 + lambda * (1 - SSIM), lambda = 0.2"
def photometric_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
    lambda_dssim: float = 0.2,
) -> torch.Tensor:
    d_ssim = 1.0 - ssim(rendered, target)

    return (1.0 - lambda_dssim) * l1_loss(rendered, target) + lambda_dssim * d_ssim
