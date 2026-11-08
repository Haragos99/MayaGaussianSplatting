import torch

from .binning import TileBins

# 3DGS skips any splat whose alpha at a pixel is below 1/255: it cannot change
# an 8-bit pixel, and cutting there (instead of at a fixed box) means the edge
# of a splat never depends on which tiles it was binned into.
ALPHA_MIN = 1.0 / 255.0


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
    alpha_max: float = 0.999,
) -> torch.Tensor:
    """-> (H, W, 3) image in [0, 1]."""
    device = uv.device
    dtype = uv.dtype

    image = torch.zeros(height * width, 3, dtype=dtype, device=device)
    if bins.num_pairs == 0:
        return image.view(height, width, 3)

    offset = torch.arange(tile_size, device=device)
    num_pixels = tile_size * tile_size
    shaded_index: list[torch.Tensor] = []
    shaded_rgb: list[torch.Tensor] = []

    for first in range(0, bins.num_tiles, tile_chunk):
        tile_index = torch.arange(first, min(first + tile_chunk, bins.num_tiles), device=device)

        start = bins.tile_start[tile_index]
        counts = bins.tile_end[tile_index] - start

        # Pairs are depth sorted, so truncating keeps the front-most Gaussians.
        if max_per_tile is not None:
            counts = counts.clamp_max(max_per_tile)

        depth_slots = int(counts.max())
        if depth_slots == 0:
            continue

        num_tiles = tile_index.numel()

        tile_x = (tile_index % bins.grid_x) * tile_size
        tile_y = (tile_index // bins.grid_x) * tile_size

        pixel_x = (tile_x.unsqueeze(-1) + offset).unsqueeze(-2).expand(num_tiles, tile_size, tile_size)
        pixel_y = (tile_y.unsqueeze(-1) + offset).unsqueeze(-1).expand(num_tiles, tile_size, tile_size)
        pixel_x = pixel_x.reshape(num_tiles, num_pixels)
        pixel_y = pixel_y.reshape(num_tiles, num_pixels)
        sample_x = pixel_x.to(dtype).unsqueeze(-1)
        sample_y = pixel_y.to(dtype).unsqueeze(-1)

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

            more_slices = slice_start + depth_chunk < depth_slots
            if more_slices and float(transmittance.detach().max()) < transmittance_min:
                break

        # Edge tiles hang over the image border; those pixels are dropped.
        inside = (pixel_x < width) & (pixel_y < height)
        flat = (pixel_y * width + pixel_x).clamp_max(height * width - 1)

        shaded_index.append(flat[inside])
        shaded_rgb.append(rgb[inside])

    if not shaded_index:
        return image.view(height, width, 3)

    # Scattering once keeps a single full-size image in the autograd graph;
    # accumulating per chunk would retain one copy per chunk instead.
    image = image.index_add(0, torch.cat(shaded_index), torch.cat(shaded_rgb))

    return image.view(height, width, 3)
