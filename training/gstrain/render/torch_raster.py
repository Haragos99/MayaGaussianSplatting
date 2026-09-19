"""Differentiable tile rasterizer for 3D Gaussians, written in plain PyTorch.

Follows Kerbl et al. 2023 (SIGGRAPH): EWA projection (Eq. 5), a 16x16 tile grid,
one global sort on (tile, depth), then front-to-back alpha compositing. The C++
plugin does the same sort in src/render/SplatDepthSorter.cpp.

Blending walks each tile in depth blocks so that peak memory is bounded by
`element_budget` regardless of how crowded a tile is, tiles are grouped by
occupancy so padding stays local, and a tile stops once it is opaque.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..sh_torch import eval_sh
from ..torch_utils import build_covariance

BLOCK = 16
PIXELS_PER_TILE = BLOCK * BLOCK

#: Gaussians composited per pass over a tile; caps the size of the blend tensors.
DEPTH_BLOCK = 64

#: A tile this opaque cannot be changed by anything further back.
MIN_TRANSMITTANCE = 1e-4

#: Screen-space low-pass filter from EWA splatting; without it small Gaussians
#: alias to single pixels and the optimisation diverges.
COV2D_BLUR = 0.3


@dataclass
class RenderOutput:
    image: torch.Tensor  # (3,H,W)
    viewspace_points: torch.Tensor  # (N,2) leaf; .grad is dL/d(pixel position)
    visibility_filter: torch.Tensor  # (N,) bool
    radii: torch.Tensor  # (N,) int64, screen-space radius in pixels


def render(camera, model, bg_color: torch.Tensor, scaling_modifier: float = 1.0, **kwargs) -> RenderOutput:
    """Evaluate the model's SH for this view, then rasterize."""
    means3D = model.get_xyz
    dirs = torch.nn.functional.normalize(
        means3D - camera.camera_center.to(means3D.device, means3D.dtype), dim=-1
    )
    colors = torch.clamp_min(eval_sh(model.active_sh_degree, model.get_features, dirs) + 0.5, 0.0)
    return rasterize(
        means3D=means3D,
        scales=model.get_scaling * scaling_modifier,
        rotations=model.get_rotation,
        opacities=model.get_opacity,
        colors=colors,
        camera=camera,
        bg_color=bg_color,
        **kwargs,
    )


def rasterize(
    means3D: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    camera,
    bg_color: torch.Tensor,
    element_budget: int = 4_000_000,
) -> RenderOutput:
    device, dtype = means3D.device, means3D.dtype
    n = means3D.shape[0]
    height, width = int(camera.height), int(camera.width)
    bg = bg_color.to(device=device, dtype=dtype).reshape(3)

    viewspace = torch.zeros((n, 2), device=device, dtype=dtype, requires_grad=True)
    if n == 0:
        return RenderOutput(
            image=bg.reshape(3, 1, 1).expand(3, height, width).clone(),
            viewspace_points=viewspace,
            visibility_filter=torch.zeros((0,), dtype=torch.bool, device=device),
            radii=torch.zeros((0,), dtype=torch.int64, device=device),
        )

    viewmat = camera.world_view_transform.to(device=device, dtype=dtype)
    fx, fy = float(camera.focal_x), float(camera.focal_y)

    # --- project ----------------------------------------------------------- #
    p_view = means3D @ viewmat[:3, :3] + viewmat[3, :3]
    depth = p_view[:, 2]
    safe_z = torch.where(depth > 1e-6, depth, torch.full_like(depth, 1e-6))
    u = fx * p_view[:, 0] / safe_z + camera.principal_x + viewspace[:, 0]
    v = fy * p_view[:, 1] / safe_z + camera.principal_y + viewspace[:, 1]

    # --- 2D covariance (EWA, paper Eq. 5) ---------------------------------- #
    sigma = build_covariance(scales, rotations)
    lim_x = 1.3 * math.tan(camera.fov_x * 0.5)
    lim_y = 1.3 * math.tan(camera.fov_y * 0.5)
    tx = torch.clamp(p_view[:, 0] / safe_z, -lim_x, lim_x) * safe_z
    ty = torch.clamp(p_view[:, 1] / safe_z, -lim_y, lim_y) * safe_z

    zero = torch.zeros_like(depth)
    jacobian = torch.stack(
        [
            torch.stack([fx / safe_z, zero, -fx * tx / (safe_z * safe_z)], dim=-1),
            torch.stack([zero, fy / safe_z, -fy * ty / (safe_z * safe_z)], dim=-1),
            torch.stack([zero, zero, zero], dim=-1),
        ],
        dim=1,
    )
    world_to_view = viewmat[:3, :3].transpose(0, 1)  # R_w2c for column vectors
    t = jacobian @ world_to_view
    cov = t @ sigma @ t.transpose(1, 2)
    cov_a = cov[:, 0, 0] + COV2D_BLUR
    cov_b = cov[:, 0, 1]
    cov_c = cov[:, 1, 1] + COV2D_BLUR
    det = cov_a * cov_c - cov_b * cov_b

    with torch.no_grad():
        valid = (depth > 0.2) & (det > 1e-9)
        mid = 0.5 * (cov_a + cov_c)
        lam = mid + torch.sqrt(torch.clamp(mid * mid - det, min=0.1))
        radii = torch.ceil(3.0 * torch.sqrt(torch.clamp(lam, min=0.0))).to(torch.int64)
        radii = torch.where(valid, radii, torch.zeros_like(radii))

    det_safe = torch.where(valid, det, torch.ones_like(det))
    conic_a = cov_c / det_safe
    conic_b = -cov_b / det_safe
    conic_c = cov_a / det_safe

    # --- tile binning ------------------------------------------------------ #
    grid_x = (width + BLOCK - 1) // BLOCK
    grid_y = (height + BLOCK - 1) // BLOCK
    num_tiles = grid_x * grid_y

    with torch.no_grad():
        ud, vd = u.detach(), v.detach()
        r = radii.to(dtype)
        x0 = torch.clamp(torch.floor((ud - r) / BLOCK).to(torch.int64), 0, grid_x)
        x1 = torch.clamp(torch.floor((ud + r + BLOCK - 1) / BLOCK).to(torch.int64), 0, grid_x)
        y0 = torch.clamp(torch.floor((vd - r) / BLOCK).to(torch.int64), 0, grid_y)
        y1 = torch.clamp(torch.floor((vd + r + BLOCK - 1) / BLOCK).to(torch.int64), 0, grid_y)
        span_x = (x1 - x0).clamp(min=0)
        span_y = (y1 - y0).clamp(min=0)
        counts = torch.where(valid, span_x * span_y, torch.zeros_like(span_x))
        total = int(counts.sum())

    visibility = radii > 0
    if total == 0:
        return RenderOutput(
            image=bg.reshape(3, 1, 1).expand(3, height, width).clone(),
            viewspace_points=viewspace,
            visibility_filter=visibility,
            radii=radii,
        )

    with torch.no_grad():
        gauss_idx = torch.repeat_interleave(torch.arange(n, device=device), counts)
        starts = torch.cumsum(counts, 0) - counts
        offset = torch.arange(total, device=device) - starts[gauss_idx]
        span = span_x[gauss_idx]
        tile_x = x0[gauss_idx] + offset % span
        tile_y = y0[gauss_idx] + offset // span
        tile_id = tile_y * grid_x + tile_x

        # One global sort on (tile, depth), the same key the C++ sorter builds.
        rank = torch.empty(n, dtype=torch.int64, device=device)
        rank[torch.argsort(depth.detach())] = torch.arange(n, device=device)
        order = torch.argsort(tile_id * n + rank[gauss_idx])
        tile_sorted = tile_id[order]
        gauss_sorted = gauss_idx[order]

        tile_counts = torch.bincount(tile_sorted, minlength=num_tiles)
        tile_starts = torch.cumsum(tile_counts, 0) - tile_counts

        # Group tiles by occupancy so a chunk only pads to its own depth rather than
        # to the busiest tile on screen.
        busy = torch.nonzero(tile_counts, as_tuple=True)[0]
        busy = busy[torch.argsort(tile_counts[busy])]
        empty = torch.nonzero(tile_counts == 0, as_tuple=True)[0]

        oy, ox = torch.meshgrid(
            torch.arange(BLOCK, device=device), torch.arange(BLOCK, device=device), indexing="ij"
        )
        ox = ox.reshape(1, -1)
        oy = oy.reshape(1, -1)

    # --- blend ------------------------------------------------------------- #
    opacity = opacities.reshape(-1)
    flat = torch.zeros((height * width, 3), device=device, dtype=dtype)

    if empty.numel() and bool(bg.any()):
        px = (empty % grid_x).unsqueeze(1) * BLOCK + ox
        py = (empty // grid_x).unsqueeze(1) * BLOCK + oy
        visible_px = ((px < width) & (py < height)).reshape(-1)
        pixel_id = (py * width + px).reshape(-1)[visible_px]
        flat = flat.index_add(0, pixel_id, bg.reshape(1, 3).expand(pixel_id.numel(), 3))

    tiles_per_chunk = max(1, element_budget // (PIXELS_PER_TILE * DEPTH_BLOCK))

    for begin in range(0, int(busy.numel()), tiles_per_chunk):
        sel = busy[begin : begin + tiles_per_chunk]
        counts_c = tile_counts[sel]
        starts_c = tile_starts[sel]
        deepest = int(counts_c.max())

        px = (sel % grid_x).unsqueeze(1) * BLOCK + ox
        py = (sel // grid_x).unsqueeze(1) * BLOCK + oy
        inside = (px < width) & (py < height)
        pxf = px.to(dtype) + 0.5
        pyf = py.to(dtype) + 0.5

        acc = torch.zeros((sel.numel(), PIXELS_PER_TILE, 3), device=device, dtype=dtype)
        trans = torch.ones((sel.numel(), PIXELS_PER_TILE, 1), device=device, dtype=dtype)

        for first in range(0, deepest, DEPTH_BLOCK):
            k = torch.arange(first, min(first + DEPTH_BLOCK, deepest), device=device)
            present = k.unsqueeze(0) < counts_c.unsqueeze(1)
            g = gauss_sorted[(starts_c.unsqueeze(1) + k.unsqueeze(0)).clamp(max=total - 1)]

            dx = pxf.unsqueeze(2) - u[g].unsqueeze(1)
            dy = pyf.unsqueeze(2) - v[g].unsqueeze(1)
            power = (
                -0.5 * (conic_a[g].unsqueeze(1) * dx * dx + conic_c[g].unsqueeze(1) * dy * dy)
                - conic_b[g].unsqueeze(1) * dx * dy
            )
            alpha = torch.clamp(opacity[g].unsqueeze(1) * torch.exp(torch.clamp(power, max=0.0)), max=0.99)
            keep = present.unsqueeze(1) & (power <= 0.0) & (alpha > 1.0 / 255.0)
            alpha = torch.where(keep, alpha, torch.zeros_like(alpha))

            cumulative = torch.cumprod(1.0 - alpha, dim=2)
            exclusive = torch.cat([torch.ones_like(cumulative[:, :, :1]), cumulative[:, :, :-1]], dim=2)
            acc = acc + torch.einsum("tpk,tkc->tpc", alpha * exclusive * trans, colors[g])
            trans = trans * cumulative[:, :, -1:]

            # Nothing behind an opaque tile can matter; skipping costs one sync.
            if first + DEPTH_BLOCK < deepest and float(trans.detach().max()) < MIN_TRANSMITTANCE:
                break

        color = acc + trans * bg.reshape(1, 1, 3)
        visible_px = inside.reshape(-1)
        flat = flat.index_add(
            0, (py * width + px).reshape(-1)[visible_px], color.reshape(-1, 3)[visible_px]
        )

    image = flat.reshape(height, width, 3).permute(2, 0, 1)
    return RenderOutput(
        image=image,
        viewspace_points=viewspace,
        visibility_filter=visibility,
        radii=radii,
    )
