import torch

from gstrain.loss import C1, C2, l1_loss, photometric_loss, psnr, ssim


def _image(seed: int = 0, size: int = 24) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    # Smooth structure + texture, so SSIM has something to measure.
    y, x = torch.meshgrid(torch.linspace(0, 1, size), torch.linspace(0, 1, size), indexing="ij")
    base = torch.stack([x, y, (x + y) / 2], dim=-1)
    return (0.6 * base + 0.4 * torch.rand(size, size, 3, generator=g)).clamp(0, 1)


def test_ssim_identical_is_one() -> None:
    image = _image()
    assert torch.isclose(ssim(image, image), torch.tensor(1.0), atol=1e-5)


def test_ssim_ranks_distortions() -> None:
    image = _image()
    g = torch.Generator().manual_seed(1)
    slightly = (image + 0.02 * torch.randn(image.shape, generator=g)).clamp(0, 1)
    heavily = (image + 0.3 * torch.randn(image.shape, generator=g)).clamp(0, 1)
    assert 1.0 > float(ssim(image, slightly)) > float(ssim(image, heavily))


def test_ssim_matches_reference_value_for_constant_images() -> None:
    # Constant images: sigma terms are 0 inside, so SSIM = (2 a b + C1) / (a^2 + b^2 + C1) away
    # from the zero-padded border. Compare the formula on a large image where the border is small.
    a, b = 0.3, 0.6
    expected = (2 * a * b + C1) * C2 / ((a * a + b * b + C1) * C2)
    x = torch.full((64, 64, 3), a, dtype=torch.float64)
    y = torch.full((64, 64, 3), b, dtype=torch.float64)
    from gstrain import loss

    # Interior only: crop away the 5-pixel padding halo before averaging.
    window = loss._gaussian_window(11, 1.5, 3, x.device, x.dtype)
    mu_x = torch.nn.functional.conv2d(loss._to_nchw(x), window, padding=5, groups=3)[..., 5:-5, 5:-5]
    assert torch.allclose(mu_x, torch.full_like(mu_x, a))
    assert abs(float(ssim(x, y)) - expected) < 0.05  # whole-image mean includes the border
    assert float(ssim(x, y)) < 1.0


def test_photometric_loss_mix() -> None:
    rendered, target = _image(0), _image(5)
    expected = 0.8 * l1_loss(rendered, target) + 0.2 * (1 - ssim(rendered, target))
    assert torch.isclose(photometric_loss(rendered, target, 0.2), expected)
    assert torch.isclose(photometric_loss(rendered, target, 0.0), l1_loss(rendered, target))


def test_loss_gradients_finite() -> None:
    rendered = _image(0).clone().requires_grad_(True)
    photometric_loss(rendered, _image(5)).backward()
    assert torch.isfinite(rendered.grad).all()
    assert rendered.grad.abs().sum() > 0


def test_psnr_known_value() -> None:
    target = torch.zeros(8, 8, 3)
    # MSE 0.01 -> 20 dB.
    assert torch.isclose(psnr(torch.full((8, 8, 3), 0.1), target), torch.tensor(20.0), atol=1e-4)
    assert torch.isfinite(psnr(target, target))
