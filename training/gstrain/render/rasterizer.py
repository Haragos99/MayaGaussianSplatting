import torch

from .binning import TileBins


def rasterize_tiles(
    bins: TileBins,
    uv: torch.Tensor,
    conic: torch.Tensor,
    opacity: torch.Tensor,
    color: torch.Tensor,
    radius: torch.Tensor,
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
        num_pixels = tile_size * tile_size

        slot = torch.arange(depth_slots, device=device)
        occupied = slot < counts.unsqueeze(-1)
        pair = (start.unsqueeze(-1) + slot).clamp_max(bins.num_pairs - 1)
        gaussian = bins.gaussian_ids[pair]

        g_uv = uv[gaussian]
        g_conic = conic[gaussian]
        g_color = color[gaussian]
        g_radius = radius[gaussian]
        g_opacity = opacity[gaussian] * occupied

        tile_x = (tile_index % bins.grid_x) * tile_size
        tile_y = (tile_index // bins.grid_x) * tile_size

        pixel_x = (tile_x.unsqueeze(-1) + offset).unsqueeze(-2).expand(num_tiles, tile_size, tile_size)
        pixel_y = (tile_y.unsqueeze(-1) + offset).unsqueeze(-1).expand(num_tiles, tile_size, tile_size)
        pixel_x = pixel_x.reshape(num_tiles, num_pixels)
        pixel_y = pixel_y.reshape(num_tiles, num_pixels)

        dx = pixel_x.unsqueeze(-1) - g_uv[..., 0].unsqueeze(-2)
        dy = pixel_y.unsqueeze(-1) - g_uv[..., 1].unsqueeze(-2)

        a = g_conic[..., 0].unsqueeze(-2)
        b = g_conic[..., 1].unsqueeze(-2)
        c = g_conic[..., 2].unsqueeze(-2)

        # A tile is coarser than the 3-sigma box, so clip per pixel as well -
        # otherwise the visible tail would depend on tile_size.
        footprint = _inside_footprint(pixel_x, pixel_y, g_uv, g_radius)

        power = -0.5 * (a * dx * dx + 2 * b * dx * dy + c * dy * dy)
        alpha = (g_opacity.unsqueeze(-2) * power.exp() * footprint).clamp(0.0, alpha_max)

        # Exclusive cumulative product: T_i = prod_{j<i} (1 - alpha_j).
        survives = 1.0 - alpha
        transmittance = torch.cat(
            [torch.ones_like(survives[..., :1]), survives[..., :-1]], dim=-1
        ).cumprod(dim=-1)

        weight = transmittance * alpha
        rgb = torch.einsum("cpk,ckj->cpj", weight, g_color)

        # Edge tiles hang over the image border; those pixels are dropped.
        inside = (pixel_x < width) & (pixel_y < height)
        flat = (pixel_y * width + pixel_x).clamp_max(height * width - 1)

        image = image.index_add(0, flat[inside], rgb[inside])

    return image.view(height, width, 3)


def _inside_footprint(
    pixel_x: torch.Tensor,
    pixel_y: torch.Tensor,
    g_uv: torch.Tensor,
    g_radius: torch.Tensor,
) -> torch.Tensor:
    """(C, P, K) mask using the same floor/ceil bounds as the NumPy reference."""
    lo = (g_uv - g_radius).floor().unsqueeze(-3)
    hi = (g_uv + g_radius).ceil().unsqueeze(-3)

    x = pixel_x.unsqueeze(-1)
    y = pixel_y.unsqueeze(-1)

    inside = (
        (x >= lo[..., 0]) & (x <= hi[..., 0]) & (y >= lo[..., 1]) & (y <= hi[..., 1])
    )

    return inside.to(g_uv.dtype)
