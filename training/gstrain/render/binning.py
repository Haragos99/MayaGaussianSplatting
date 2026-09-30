import math
from dataclasses import dataclass

import torch

"Tile bounds, binning and depth sorting "
@dataclass
class TileBins:
    gaussian_ids: torch.Tensor  # (P,) which Gaussian
    tile_ids: torch.Tensor      # (P,) which tile, ascending
    tile_start: torch.Tensor    # (T,) first pair index of each tile
    tile_end: torch.Tensor      # (T,) one past the last pair index
    grid_x: int
    grid_y: int

    @property
    def num_tiles(self) -> int:
        return self.grid_x * self.grid_y

    @property
    def num_pairs(self) -> int:
        return int(self.gaussian_ids.numel())


def tile_grid(width: int, height: int, tile_size: int) -> tuple[int, int]:
    return math.ceil(width / tile_size), math.ceil(height / tile_size)


def screen_radius(cov2d: torch.Tensor, sigma_factor: float = 3.0) -> torch.Tensor:
    """(N, 2, 2) -> (N, 2) axis-aligned 3-sigma pixel radii."""
    rx = cov2d[..., 0, 0].clamp_min(1e-8).sqrt()
    ry = cov2d[..., 1, 1].clamp_min(1e-8).sqrt()

    return sigma_factor * torch.stack([rx, ry], dim=-1)


def tile_ranges(
    uv: torch.Tensor,
    radius: torch.Tensor,
    width: int,
    height: int,
    tile_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """-> (N, 4) inclusive tile box [x0, x1, y0, y1] and (N,) on-screen mask."""
    grid_x, grid_y = tile_grid(width, height, tile_size)

    # floor/ceil to whole pixels first: the rasterizer's per-pixel footprint
    # uses the same rounding, and a tile missed here can never be shaded.
    lo = (uv - radius).floor()
    hi = (uv + radius).ceil()

    # Overlap is decided before clamping, otherwise off-screen splats survive.
    touches = (hi[..., 0] >= 0) & (lo[..., 0] <= width - 1) & (hi[..., 1] >= 0) & (lo[..., 1] <= height - 1)

    x0 = (lo[..., 0] / tile_size).floor().long().clamp(0, grid_x - 1)
    x1 = (hi[..., 0] / tile_size).floor().long().clamp(0, grid_x - 1)
    y0 = (lo[..., 1] / tile_size).floor().long().clamp(0, grid_y - 1)
    y1 = (hi[..., 1] / tile_size).floor().long().clamp(0, grid_y - 1)

    return torch.stack([x0, x1, y0, y1], dim=-1), touches


def build_tile_bins(
    uv: torch.Tensor,
    cov2d: torch.Tensor,
    depth: torch.Tensor,
    visible: torch.Tensor,
    width: int,
    height: int,
    tile_size: int = 16,
    sigma_factor: float = 3.0,
) -> TileBins:
    device = uv.device
    grid_x, grid_y = tile_grid(width, height, tile_size)

    box, touches = tile_ranges(uv, screen_radius(cov2d, sigma_factor), width, height, tile_size)
    keep = visible & touches

    x0, x1, y0, y1 = box.unbind(dim=-1)
    span_x = (x1 - x0 + 1) * keep
    span_y = (y1 - y0 + 1) * keep
    counts = span_x * span_y

    total = int(counts.sum())
    if total == 0:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        zeros = torch.zeros(grid_x * grid_y, dtype=torch.long, device=device)
        return TileBins(empty, empty, zeros, zeros.clone(), grid_x, grid_y)

    # One entry per (Gaussian, tile) pair, expanded without a Python loop.
    gaussian_ids = torch.repeat_interleave(torch.arange(len(uv), device=device), counts)

    # Index of each pair within its own Gaussian's block: 0, 1, 2, ... span-1.
    block_start = counts.cumsum(0) - counts
    local = torch.arange(total, device=device) - block_start.repeat_interleave(counts)

    span_x_rep = span_x.repeat_interleave(counts).clamp_min(1)
    tile_x = x0.repeat_interleave(counts) + local % span_x_rep
    tile_y = y0.repeat_interleave(counts) + local // span_x_rep
    tile_ids = tile_y * grid_x + tile_x

    # Sort by tile, then by depth: depth first, then a stable sort on tile id.
    order = depth.repeat_interleave(counts).argsort()
    order = order[tile_ids[order].argsort(stable=True)]

    tile_ids = tile_ids[order]
    gaussian_ids = gaussian_ids[order]

    tile_index = torch.arange(grid_x * grid_y, device=device)
    tile_start = torch.searchsorted(tile_ids, tile_index)
    tile_end = torch.searchsorted(tile_ids, tile_index, right=True)

    return TileBins(gaussian_ids, tile_ids, tile_start, tile_end, grid_x, grid_y)
