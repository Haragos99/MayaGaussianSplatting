import torch

from .binning import TileBins

# 3DGS skips any splat whose alpha at a pixel is below 1/255: it cannot change
# an 8-bit pixel, and cutting there (instead of at a fixed box) means the edge
# of a splat never depends on which tiles it was binned into.
ALPHA_MIN = 1.0 / 255.0

# 3DGS clamps alpha to 0.99 so a single splat never makes a pixel fully opaque
# (the transmittance, and with it every gradient behind, would vanish).
ALPHA_MAX = 0.99


def rasterize_tiles(
    bins: TileBins,
    uv: torch.Tensor,
    conic: torch.Tensor,
    opacity: torch.Tensor,
    color: torch.Tensor,
    width: int,
    height: int,
    tile_size: int = 16,
    tile_chunk: int = 32,
    max_per_tile: int | None = None,
    alpha_max: float = ALPHA_MAX,
    alpha_min: float = ALPHA_MIN,
    depth_chunk: int = 256,
    transmittance_min: float = 1e-4,
    background: torch.Tensor | None = None,
    contribution: torch.Tensor | None = None,
) -> torch.Tensor:
    """-> (H, W, 3) image.

    Pixel (i, j) is sampled at its centre (i + 0.5, j + 0.5), the COLMAP
    convention the camera intrinsics are expressed in.

    Each tile's depth-sorted list is composited front to back in slices of
    `depth_chunk` splats, carrying transmittance between slices; once every
    pixel of a tile chunk is opaque (T < transmittance_min) the rest is
    skipped. Tiles are processed in order of how many splats they hold, so a
    chunk is padded only to tiles of similar depth, not to the busiest one.

    `background` (3,) is composited behind the splats (C = sum + T * bg).
    `contribution` (N,), if given, receives sum(T * alpha) per splat over all
    pixels - the LightGaussian significance score - without gradients.
    """
    device = uv.device
    dtype = uv.dtype
    num_pixels = tile_size * tile_size

    if background is None:
        background = torch.zeros(3, dtype=dtype, device=device)
    background = background.to(device=device, dtype=dtype).reshape(3)

    tile_counts = bins.tile_end - bins.tile_start
    if max_per_tile is not None:
        # Pairs are depth sorted, so truncating keeps the front-most Gaussians.
        tile_counts = tile_counts.clamp_max(max_per_tile)

    # Pixels of tiles with no splats show only the background.
    pixel_tile = (
        torch.arange(height, device=device).unsqueeze(-1) // tile_size * bins.grid_x
        + torch.arange(width, device=device) // tile_size
    ).reshape(-1)
    empty_pixel = (tile_counts[pixel_tile] == 0).to(dtype).unsqueeze(-1)
    image = empty_pixel * background

    if bins.num_pairs == 0:
        return image.view(height, width, 3)

    busy = torch.nonzero(tile_counts, as_tuple=True)[0]
    busy = busy[torch.argsort(tile_counts[busy])]

    offset = torch.arange(tile_size, device=device)
    shaded_index: list[torch.Tensor] = []
    shaded_rgb: list[torch.Tensor] = []

    for first in range(0, busy.numel(), tile_chunk):
        tile_index = busy[first:first + tile_chunk]

        start = bins.tile_start[tile_index]
        counts = tile_counts[tile_index]
        depth_slots = int(counts.max())
        num_tiles = tile_index.numel()

        tile_x = (tile_index % bins.grid_x) * tile_size
        tile_y = (tile_index // bins.grid_x) * tile_size

        pixel_x = (tile_x.unsqueeze(-1) + offset).unsqueeze(-2).expand(num_tiles, tile_size, tile_size)
        pixel_y = (tile_y.unsqueeze(-1) + offset).unsqueeze(-1).expand(num_tiles, tile_size, tile_size)
        pixel_x = pixel_x.reshape(num_tiles, num_pixels)
        pixel_y = pixel_y.reshape(num_tiles, num_pixels)
        sample_x = (pixel_x.to(dtype) + 0.5).unsqueeze(-1)
        sample_y = (pixel_y.to(dtype) + 0.5).unsqueeze(-1)

        transmittance = torch.ones(num_tiles, num_pixels, dtype=dtype, device=device)
        rgb = torch.zeros(num_tiles, num_pixels, 3, dtype=dtype, device=device)

        for slice_start in range(0, depth_slots, depth_chunk):
            slot = torch.arange(slice_start, min(slice_start + depth_chunk, depth_slots), device=device)
            occupied = slot < counts.unsqueeze(-1)
            pair = (start.unsqueeze(-1) + slot).clamp_max(bins.num_pairs - 1)
            gaussian = bins.gaussian_ids[pair]

            g_uv = uv[gaussian]
            g_conic = conic[gaussian]
            g_color = color[gaussian]
            g_opacity = opacity[gaussian] * occupied

            dx = sample_x - g_uv[..., 0].unsqueeze(-2)
            dy = sample_y - g_uv[..., 1].unsqueeze(-2)

            a = g_conic[..., 0].unsqueeze(-2)
            b = g_conic[..., 1].unsqueeze(-2)
            c = g_conic[..., 2].unsqueeze(-2)

            # The conic is positive definite, so power <= 0; the clamp only
            # guards against round-off producing alpha > opacity.
            power = (-0.5 * (a * dx * dx + 2 * b * dx * dy + c * dy * dy)).clamp_max(0.0)
            alpha = (g_opacity.unsqueeze(-2) * power.exp()).clamp_max(alpha_max)
            alpha = torch.where(alpha >= alpha_min, alpha, torch.zeros_like(alpha))

            # Exclusive cumulative product inside the slice, scaled by the
            # transmittance left over from the slices in front of it.
            survives = 1.0 - alpha
            local = torch.cat(
                [torch.ones_like(survives[..., :1]), survives[..., :-1]], dim=-1
            ).cumprod(dim=-1)

            weight = transmittance.unsqueeze(-1) * local * alpha
            rgb = rgb + torch.einsum("cpk,ckj->cpj", weight, g_color)
            transmittance = transmittance * local[..., -1] * survives[..., -1]

            if contribution is not None:
                with torch.no_grad():
                    inside = ((pixel_x < width) & (pixel_y < height)).to(dtype).unsqueeze(-1)
                    per_splat = (weight.detach() * inside).sum(dim=1) * occupied
                    contribution.index_add_(0, gaussian.reshape(-1), per_splat.reshape(-1))

            more_slices = slice_start + depth_chunk < depth_slots
            if more_slices and float(transmittance.detach().max()) < transmittance_min:
                break

        rgb = rgb + transmittance.unsqueeze(-1) * background

        # Edge tiles hang over the image border; those pixels are dropped.
        inside = (pixel_x < width) & (pixel_y < height)
        flat = (pixel_y * width + pixel_x).clamp_max(height * width - 1)

        shaded_index.append(flat[inside])
        shaded_rgb.append(rgb[inside])

    # Scattering once keeps a single full-size image in the autograd graph;
    # accumulating per chunk would retain one copy per chunk instead.
    image = image.index_add(0, torch.cat(shaded_index), torch.cat(shaded_rgb))

    return image.view(height, width, 3)
