import torch

from gstrain.render.binning import build_tile_bins, screen_radius, tile_grid


def _bins(uv, sigma, depth, visible=None, width=48, height=32, tile=16):
    uv = torch.tensor(uv, dtype=torch.float32)
    n = len(uv)
    cov = torch.diag_embed(torch.tensor(sigma, dtype=torch.float32).square().unsqueeze(-1).expand(n, 2))
    depth = torch.tensor(depth, dtype=torch.float32)
    visible = torch.ones(n, dtype=torch.bool) if visible is None else torch.tensor(visible)
    return build_tile_bins(uv, cov, depth, visible, width, height, tile_size=tile, sigma_factor=3.0)


def _tiles_of(bins, gaussian: int) -> set[int]:
    return {int(t) for t, g in zip(bins.tile_ids, bins.gaussian_ids) if int(g) == gaussian}


def test_tile_grid_rounds_up() -> None:
    assert tile_grid(33, 17, 16) == (3, 2)
    assert tile_grid(32, 16, 16) == (2, 1)


def test_tile_ranges_cover_footprint() -> None:
    width, height, tile = 48, 32, 16
    uv, sigma = [[20.0, 15.0]], [2.5]
    bins = _bins(uv, sigma, [1.0], width=width, height=height, tile=tile)
    listed = _tiles_of(bins, 0)
    radius = 3.0 * sigma[0]
    grid_x = bins.grid_x
    for y in range(height):
        for x in range(width):
            if (x + 0.5 - uv[0][0]) ** 2 + (y + 0.5 - uv[0][1]) ** 2 <= radius ** 2:
                assert (y // tile) * grid_x + x // tile in listed, (x, y)


def test_offscreen_and_behind_camera_splats_are_not_binned() -> None:
    bins = _bins(
        [[24.0, 16.0], [-100.0, 16.0], [24.0, 500.0], [24.0, 16.0]],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, -1.0],
        visible=[True, True, True, False],
    )
    assert set(int(g) for g in bins.gaussian_ids) == {0}


def test_bins_sorted_by_depth_within_tile() -> None:
    depth = [3.0, 1.0, 2.0, 0.5]
    bins = _bins([[8.0, 8.0]] * 4, [1.0] * 4, depth)
    for tile in range(bins.num_tiles):
        ids = bins.gaussian_ids[bins.tile_start[tile]:bins.tile_end[tile]]
        depths = torch.tensor(depth)[ids]
        assert torch.all(depths[1:] >= depths[:-1])
    assert bins.num_pairs == 4
    assert torch.all(bins.tile_ids[1:] >= bins.tile_ids[:-1])


def test_empty_scene_gives_empty_bins() -> None:
    bins = _bins([[-50.0, -50.0]], [1.0], [1.0])
    assert bins.num_pairs == 0
    assert torch.all(bins.tile_start == bins.tile_end)


def test_screen_radius_tensor_sigma() -> None:
    cov = torch.diag_embed(torch.tensor([[4.0, 9.0], [1.0, 16.0]]))
    sigma = torch.tensor([2.0, 3.0])
    radius = screen_radius(cov, sigma)
    assert torch.allclose(radius[0], screen_radius(cov[:1], 2.0)[0])
    assert torch.allclose(radius[1], screen_radius(cov[1:], 3.0)[0])
    assert torch.allclose(radius, torch.tensor([[4.0, 6.0], [3.0, 12.0]]))
    assert screen_radius(cov, 3.0, max_radius=5.0).max() <= 5.0
