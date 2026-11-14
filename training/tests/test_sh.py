import pytest
import torch

from gstrain.sh import C0, C1, eval_sh, num_sh_coeffs, rgb_to_sh, sh_to_rgb

from conftest import make_model


def _dirs(count: int, seed: int = 0) -> torch.Tensor:
    return torch.nn.functional.normalize(torch.randn(count, 3, generator=torch.Generator().manual_seed(seed)), dim=-1)


def test_num_sh_coeffs() -> None:
    assert [num_sh_coeffs(d) for d in range(4)] == [1, 4, 9, 16]


def test_rgb_sh_roundtrip() -> None:
    rgb = torch.rand(50, 3)
    assert torch.allclose(sh_to_rgb(rgb_to_sh(rgb)), rgb, atol=1e-6)
    assert torch.allclose(rgb_to_sh(torch.full((1, 3), 0.5)), torch.zeros(1, 3))


def test_eval_sh_matches_explicit_basis() -> None:
    sh = torch.randn(30, 4, 3)
    dirs = _dirs(30)
    x, y, z = dirs[:, :1], dirs[:, 1:2], dirs[:, 2:3]
    expected = C0 * sh[:, 0] - C1 * y * sh[:, 1] + C1 * z * sh[:, 2] - C1 * x * sh[:, 3]
    assert torch.allclose(eval_sh(1, sh, dirs), expected, atol=1e-6)


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_eval_sh_higher_bands_view_dependent(degree) -> None:
    count = num_sh_coeffs(degree)
    sh = torch.zeros(1, count, 3)
    sh[:, 0] = 0.4
    flat_a = eval_sh(degree, sh, torch.tensor([[0.0, 0.0, 1.0]]))
    flat_b = eval_sh(degree, sh, torch.tensor([[1.0, 0.0, 0.0]]))
    assert torch.allclose(flat_a, flat_b)  # only band 0: same colour from every side

    # A coefficient in the highest band makes the colour depend on direction.
    sh[:, count - 1] = 0.5
    colours = eval_sh(degree, sh.expand(20, -1, -1), _dirs(20, seed=degree))
    assert colours.std(dim=0).max() > 1e-3


def test_eval_sh_rejects_bad_degree_or_shape() -> None:
    with pytest.raises(ValueError):
        eval_sh(4, torch.zeros(1, 25, 3), _dirs(1))
    with pytest.raises(ValueError):
        eval_sh(2, torch.zeros(1, 4, 3), _dirs(1))


def test_oneup_sh_degree_caps_at_max() -> None:
    model = make_model([[0.0, 0.0, 2.0]], sh_degree=2)
    assert model.active_sh_degree == 0
    assert model.oneup_sh_degree() and model.oneup_sh_degree()
    assert model.active_sh_degree == 2
    assert not model.oneup_sh_degree()
    assert model.active_sh_degree == 2
    assert model.sh_rest.shape == (1, 8, 3)


def test_colors_for_uses_view_direction() -> None:
    model = make_model([[0.0, 0.0, 0.0]], color=[[0.5, 0.5, 0.5]], sh_degree=1)
    with torch.no_grad():
        model.sh_rest[:, 1] = 1.0  # band-1 z coefficient (features index 2; sh_rest skips band 0)
    model.active_sh_degree = 1
    from_front = model.colors_for(torch.tensor([0.0, 0.0, -2.0]))  # looking along +z
    from_back = model.colors_for(torch.tensor([0.0, 0.0, 2.0]))
    assert torch.all(from_front > from_back)
    assert torch.all(from_back >= 0.0)  # clamped at zero
