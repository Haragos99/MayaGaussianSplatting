import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gstrain.cameras import world_view_transform  # noqa: E402
from gstrain.dataset import Camera  # noqa: E402
from gstrain.render import rasterize  # noqa: E402


def make_camera(size: int, fov: float = 0.8, distance: float = 3.0, dtype=torch.float32) -> Camera:
    """Camera at (0,0,-distance) looking down +Z, so the origin projects to the centre."""
    rotation = np.eye(3, dtype=np.float32)
    translation = np.array([0.0, 0.0, distance], dtype=np.float32)
    return Camera(
        uid=0,
        name="test",
        width=size,
        height=size,
        fov_x=fov,
        fov_y=fov,
        world_view_transform=torch.from_numpy(world_view_transform(rotation, translation)).to(dtype),
        camera_center=torch.from_numpy(-(rotation @ translation)).to(dtype),
        image=None,
    )


def single_gaussian(scale=0.05, opacity=0.5, position=(0.0, 0.0, 0.0), dtype=torch.float32):
    means = torch.tensor([position], dtype=dtype)
    scales = torch.full((1, 3), scale, dtype=dtype)
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype)
    opacities = torch.tensor([[opacity]], dtype=dtype)
    colors = torch.ones((1, 3), dtype=dtype)
    return means, scales, rotations, opacities, colors


def test_centred_gaussian_peaks_at_the_centre_pixel():
    size = 31  # odd, so the projected centre lands exactly on a pixel centre
    camera = make_camera(size)
    out = rasterize(*single_gaussian(), camera=camera, bg_color=torch.zeros(3))

    image = out.image
    assert image.shape == (3, size, size)
    peak = torch.argmax(image[0])
    assert (int(peak) // size, int(peak) % size) == (size // 2, size // 2)
    assert float(image.detach()[0, size // 2, size // 2]) == pytest.approx(0.5, abs=1e-5)
    assert bool(out.visibility_filter[0])
    assert int(out.radii[0]) > 0


def test_footprint_is_symmetric():
    size = 31
    camera = make_camera(size)
    image = rasterize(*single_gaussian(scale=0.2), camera=camera, bg_color=torch.zeros(3)).image[0]
    torch.testing.assert_close(image, torch.flip(image, dims=[1]), atol=1e-6, rtol=0)
    torch.testing.assert_close(image, torch.flip(image, dims=[0]), atol=1e-6, rtol=0)


def test_anisotropic_gaussian_is_wider_than_it_is_tall():
    size = 31
    camera = make_camera(size)
    means, _, rotations, opacities, colors = single_gaussian()
    scales = torch.tensor([[0.4, 0.05, 0.05]])
    image = rasterize(means, scales, rotations, opacities, colors, camera, torch.zeros(3)).image[0]

    lit = image.detach() > 0.5 * float(image.detach().max())
    rows = torch.any(lit, dim=1).sum()
    cols = torch.any(lit, dim=0).sum()
    assert int(cols) > int(rows)


def test_gaussian_behind_the_camera_is_culled():
    camera = make_camera(16, distance=3.0)
    out = rasterize(*single_gaussian(position=(0.0, 0.0, -5.0)), camera=camera, bg_color=torch.zeros(3))
    assert not bool(out.visibility_filter.any())
    torch.testing.assert_close(out.image, torch.zeros_like(out.image))


def test_background_fills_empty_regions():
    camera = make_camera(16)
    bg = torch.tensor([0.2, 0.4, 0.6])
    out = rasterize(*single_gaussian(scale=0.01), camera=camera, bg_color=bg)
    corner = out.image[:, 0, 0]
    torch.testing.assert_close(corner, bg, atol=1e-6, rtol=0)


def test_empty_model_returns_background():
    camera = make_camera(16)
    bg = torch.tensor([1.0, 0.0, 0.0])
    out = rasterize(
        torch.zeros((0, 3)),
        torch.zeros((0, 3)),
        torch.zeros((0, 4)),
        torch.zeros((0, 1)),
        torch.zeros((0, 3)),
        camera,
        bg,
    )
    assert out.image.shape == (3, 16, 16)
    torch.testing.assert_close(out.image[:, 5, 5], bg)


def test_opacity_composites_front_to_back():
    """A near opaque splat in front must hide the one behind it."""
    size = 31
    camera = make_camera(size)
    means = torch.tensor([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]])
    scales = torch.full((2, 3), 0.05)
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    opacities = torch.tensor([[0.99], [0.99]])
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # front is red

    centre = rasterize(means, scales, rotations, opacities, colors, camera, torch.zeros(3)).image.detach()[
        :, size // 2, size // 2
    ]
    assert float(centre[0]) > 0.9
    assert float(centre[1]) < 0.05


def test_gradcheck():
    dtype = torch.float64
    camera = make_camera(16, fov=0.8, distance=4.0, dtype=dtype)
    generator = torch.Generator().manual_seed(0)

    means = (torch.rand((4, 3), generator=generator, dtype=dtype) - 0.5) * 0.6
    # Wide splats keep every pixel well above the 1/255 cutoff, which is a hard
    # threshold and would otherwise make the finite differences disagree.
    scales = torch.full((4, 3), 1.6, dtype=dtype)
    rotations = torch.tensor([[1.0, 0.1, 0.0, 0.0]], dtype=dtype).repeat(4, 1)
    opacities = torch.full((4, 1), 0.5, dtype=dtype)
    colors = torch.rand((4, 3), generator=generator, dtype=dtype) * 0.8 + 0.1

    inputs = tuple(t.clone().requires_grad_(True) for t in (means, scales, rotations, opacities, colors))

    def fn(*args):
        return rasterize(*args, camera=camera, bg_color=torch.zeros(3, dtype=dtype)).image

    assert torch.autograd.gradcheck(fn, inputs, eps=1e-6, atol=1e-5, rtol=1e-3)


def test_viewspace_gradient_is_populated():
    camera = make_camera(31)
    means, scales, rotations, opacities, colors = single_gaussian(scale=0.2)
    means = means.clone().requires_grad_(True)
    out = rasterize(means, scales, rotations, opacities, colors, camera, torch.zeros(3))
    out.image.sum().backward()
    assert out.viewspace_points.grad is not None
    assert out.viewspace_points.grad.shape == (1, 2)


def naive_render(means, scales, rotations, opacities, colors, camera, bg):
    """Straightforward per-pixel loop over depth-sorted Gaussians, no tiles."""
    from gstrain.render.torch_raster import COV2D_BLUR
    from gstrain.torch_utils import build_covariance

    viewmat = camera.world_view_transform.to(means.dtype)
    fx, fy = camera.focal_x, camera.focal_y
    p_view = means @ viewmat[:3, :3] + viewmat[3, :3]
    z = p_view[:, 2]
    u = fx * p_view[:, 0] / z + camera.principal_x
    v = fy * p_view[:, 1] / z + camera.principal_y

    j = torch.zeros((means.shape[0], 3, 3), dtype=means.dtype)
    j[:, 0, 0] = fx / z
    j[:, 0, 2] = -fx * p_view[:, 0] / z**2
    j[:, 1, 1] = fy / z
    j[:, 1, 2] = -fy * p_view[:, 1] / z**2
    t = j @ viewmat[:3, :3].T
    cov = t @ build_covariance(scales, rotations) @ t.transpose(1, 2)
    a = cov[:, 0, 0] + COV2D_BLUR
    b = cov[:, 0, 1]
    c = cov[:, 1, 1] + COV2D_BLUR
    det = a * c - b * b
    conic = torch.stack([c / det, -b / det, a / det], dim=-1)

    image = torch.zeros((3, camera.height, camera.width), dtype=means.dtype)
    for y in range(camera.height):
        for x in range(camera.width):
            transmittance = 1.0
            accumulated = torch.zeros(3, dtype=means.dtype)
            for i in torch.argsort(z).tolist():
                dx = x + 0.5 - u[i]
                dy = y + 0.5 - v[i]
                power = -0.5 * (conic[i, 0] * dx * dx + conic[i, 2] * dy * dy) - conic[i, 1] * dx * dy
                if power > 0:
                    continue
                alpha = min(0.99, float(opacities[i, 0] * torch.exp(power)))
                if alpha <= 1.0 / 255.0:
                    continue
                accumulated = accumulated + colors[i] * alpha * transmittance
                transmittance *= 1.0 - alpha
            image[:, y, x] = accumulated + transmittance * bg
    return image


def test_matches_a_naive_per_pixel_reference():
    """The tiled, depth-blocked implementation must agree with the obvious one."""
    camera = make_camera(24, fov=0.9, distance=3.0)
    generator = torch.Generator().manual_seed(7)
    means = (torch.rand((6, 3), generator=generator) - 0.5) * 1.2
    scales = torch.rand((6, 3), generator=generator) * 0.25 + 0.05
    rotations = torch.rand((6, 4), generator=generator)
    opacities = torch.rand((6, 1), generator=generator) * 0.7 + 0.2
    colors = torch.rand((6, 3), generator=generator)
    bg = torch.tensor([0.1, 0.2, 0.3])

    fast = rasterize(means, scales, rotations, opacities, colors, camera, bg).image
    slow = naive_render(means, scales, rotations, opacities, colors, camera, bg)
    torch.testing.assert_close(fast, slow, atol=1e-5, rtol=1e-4)


def test_depth_block_size_does_not_change_the_image(monkeypatch):
    """Blocking is an implementation detail: results must not depend on it."""
    from gstrain.render import torch_raster

    camera = make_camera(24, fov=0.9)
    generator = torch.Generator().manual_seed(11)
    args = (
        (torch.rand((40, 3), generator=generator) - 0.5) * 1.2,
        torch.rand((40, 3), generator=generator) * 0.2 + 0.05,
        torch.rand((40, 4), generator=generator),
        torch.rand((40, 1), generator=generator) * 0.5 + 0.1,
        torch.rand((40, 3), generator=generator),
    )

    monkeypatch.setattr(torch_raster, "DEPTH_BLOCK", 64)
    whole = torch_raster.rasterize(*args, camera=camera, bg_color=torch.zeros(3)).image
    monkeypatch.setattr(torch_raster, "DEPTH_BLOCK", 3)
    split = torch_raster.rasterize(*args, camera=camera, bg_color=torch.zeros(3)).image
    torch.testing.assert_close(whole, split, atol=1e-6, rtol=1e-5)


def test_tile_chunking_does_not_change_the_image():
    camera = make_camera(48, fov=0.9)
    generator = torch.Generator().manual_seed(13)
    args = (
        (torch.rand((60, 3), generator=generator) - 0.5) * 1.5,
        torch.rand((60, 3), generator=generator) * 0.2 + 0.05,
        torch.rand((60, 4), generator=generator),
        torch.rand((60, 1), generator=generator) * 0.5 + 0.1,
        torch.rand((60, 3), generator=generator),
    )
    roomy = rasterize(*args, camera=camera, bg_color=torch.zeros(3), element_budget=4_000_000).image
    cramped = rasterize(*args, camera=camera, bg_color=torch.zeros(3), element_budget=16_384).image
    torch.testing.assert_close(roomy, cramped, atol=1e-6, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_matches_cpu():
    generator = torch.Generator().manual_seed(17)
    cpu_args = (
        (torch.rand((200, 3), generator=generator) - 0.5) * 2.0,
        torch.rand((200, 3), generator=generator) * 0.2 + 0.02,
        torch.rand((200, 4), generator=generator),
        torch.rand((200, 1), generator=generator) * 0.8 + 0.1,
        torch.rand((200, 3), generator=generator),
    )
    bg = torch.tensor([0.2, 0.3, 0.4])

    camera = make_camera(64, fov=0.9)
    on_cpu = rasterize(*cpu_args, camera=camera, bg_color=bg).image
    on_gpu = rasterize(
        *(a.cuda() for a in cpu_args), camera=camera.to("cuda"), bg_color=bg.cuda()
    ).image
    torch.testing.assert_close(on_gpu.cpu(), on_cpu, atol=1e-4, rtol=1e-3)
