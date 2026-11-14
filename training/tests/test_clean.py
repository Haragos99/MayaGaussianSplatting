import math
import torch
from gstrain.clean import clean_model, orbit_camera, scene_focus, strongest_share
from gstrain.dataset import TrainView
from gstrain.export_ply import load_ply, write_ply
from gstrain.geometry.camera import TorchCamera
from gstrain.model_torch import PARAMETER_ATTRIBUTES, TorchGaussianModel


def _look_at(center, target, size: int = 32) -> TorchCamera:
    """COLMAP-style camera at `center` looking at `target` (+z forward, +y down)."""
    center = torch.tensor(center, dtype=torch.float32)
    forward = torch.nn.functional.normalize(torch.tensor(target, dtype=torch.float32) - center, dim=0)
    down = torch.tensor([0.0, -1.0, 0.0])
    right = torch.nn.functional.normalize(torch.linalg.cross(down, forward), dim=0)
    down = torch.linalg.cross(forward, right)
    R = torch.stack([right, down, forward])
    return TorchCamera(
        width=size, height=size,
        fx=torch.tensor(float(size)), fy=torch.tensor(float(size)),
        cx=torch.tensor(size / 2), cy=torch.tensor(size / 2),
        R=R, t=-R @ center,
    )


def _ring_views(target=(0.0, 0.0, 0.0), radius: float = 3.0, count: int = 6) -> list[TrainView]:
    views = []
    for i in range(count):
        a = 2 * math.pi * i / count
        center = (target[0] + radius * math.sin(a), target[1], target[2] + radius * math.cos(a))
        views.append(TrainView(_look_at(center, target), torch.zeros(32, 32, 3)))
    return views


def _model(xyz, opacity, scale: float = 0.1) -> TorchGaussianModel:
    count = len(xyz)
    return TorchGaussianModel(
        torch.tensor(xyz, dtype=torch.float32), torch.full((count, 3), scale),
        torch.tensor([[1.0, 0, 0, 0]]).repeat(count, 1), torch.tensor(opacity), torch.rand(count, 3),
    )


def test_scene_focus_finds_the_subject() -> None:
    views = _ring_views(target=(0.5, -0.2, 1.0))
    assert torch.allclose(scene_focus(views), torch.tensor([0.5, -0.2, 1.0]), atol=1e-4)


def test_strongest_share_keeps_the_fewest_splats() -> None:
    contribution = torch.tensor([1.0, 5.0, 0.0, 3.0, 1.0])
    assert strongest_share(contribution, 0.8).tolist() == [False, True, False, True, False]
    # Never keeps splats that contribute nothing, even at 100 %.
    assert strongest_share(contribution, 1.0).tolist() == [True, True, False, True, True]


def test_clean_filters_opacity_hidden_and_far_splats() -> None:
    views = _ring_views()
    model = _model(
        [[0.0, 0.0, 0.0],      # subject, opaque          -> keep
         [0.2, 0.0, 0.0],      # subject, transparent     -> opacity
         [0.0, 0.0, 0.0],      # tiny, inside the subject -> contribution
         [1.8, 0.0, 0.0]],     # visible, but far from it -> crop
        [0.9, 0.01, 0.9, 0.9],
    )
    with torch.no_grad():
        model.scale_raw[2] = torch.full((3,), 0.001).log()
        model.scale_raw[0] = torch.full((3,), 0.3).log()
        model.scale_raw[3] = torch.full((3,), 0.3).log()

    result = clean_model(model, views, min_opacity=0.05, keep_contribution=0.99, crop_radius=0.5)
    assert result.keep.tolist() == [True, False, False, False]
    assert result.removed == {"opacity": 1, "contribution": 1, "crop": 1}
    assert len(result.model) == 1 and abs(result.crop_radius - 1.5) < 1e-4
    for attribute in PARAMETER_ATTRIBUTES.values():
        assert torch.equal(getattr(result.model, attribute)[0], getattr(model, attribute)[0])


def test_filters_can_be_switched_off() -> None:
    model = _model([[0.0, 0.0, 0.0], [0.0, 9.0, 0.0]], [0.01, 0.9])
    result = clean_model(model, _ring_views(), min_opacity=0.0, keep_contribution=1.0, crop_radius=None)
    assert len(result.model) == 2 and result.removed == {}


def test_load_ply_roundtrip(tmp_path) -> None:
    model = _model([[0.1, 0.2, 0.3], [1.0, -1.0, 2.0]], [0.3, 0.8])
    with torch.no_grad():
        model.sh_rest.copy_(torch.randn_like(model.sh_rest))
        model.rotation.copy_(torch.nn.functional.normalize(torch.randn(2, 4), dim=-1))
    for axis in ("colmap", "maya-y"):
        loaded = load_ply(write_ply(tmp_path / f"{axis}.ply", model, axis), up_axis=axis)
        assert loaded.max_sh_degree == 3 and loaded.active_sh_degree == 3
        for name in ("xyz", "scale_raw", "opacity_raw", "sh_dc", "sh_rest"):
            assert torch.allclose(getattr(loaded, name), getattr(model, name), atol=1e-6), (axis, name)
        # A quaternion and its negative are the same rotation.
        dot = (loaded.rotation * model.rotation).sum(dim=-1).abs()
        assert torch.allclose(dot, torch.ones(2), atol=1e-6)


def test_orbit_camera_turns_around_the_subject() -> None:
    views = _ring_views()
    camera = views[0].camera
    focus = scene_focus(views)
    assert orbit_camera(camera, focus, 0, views) is camera
    turned = orbit_camera(camera, focus, 60, views)
    # Same distance, still looking at the subject, now where the next ring camera is.
    assert torch.allclose((turned.camera_center - focus).norm(), torch.tensor(3.0), atol=1e-4)
    assert torch.allclose(turned.camera_center, views[1].camera.camera_center, atol=1e-4) or \
        torch.allclose(turned.camera_center, views[-1].camera.camera_center, atol=1e-4)
    looking = torch.nn.functional.normalize(focus - turned.camera_center, dim=0)
    assert torch.allclose(turned.R[2], looking, atol=1e-4)


# --- added: edge cases ---------------------------------------------------- #

import pytest

from gstrain.clean import select


def test_strongest_share_edge_cases() -> None:
    assert not strongest_share(torch.zeros(5), 0.9).any()
    ties = torch.tensor([1.0, 1.0, 1.0, 1.0])
    first = strongest_share(ties, 0.5)
    assert int(first.sum()) == 2
    assert torch.equal(first, strongest_share(ties, 0.5))
    assert strongest_share(torch.tensor([2.0]), 0.01).tolist() == [True]


@pytest.mark.parametrize("share", [0.0, -0.1, 1.5])
def test_clean_rejects_bad_share(share) -> None:
    with pytest.raises(ValueError):
        clean_model(_model([[0.0, 0.0, 0.0]], [0.9]), _ring_views(), keep_contribution=share)


def test_select_preserves_sh_degree_and_values() -> None:
    model = _model([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.3, 0.0]], [0.9, 0.5, 0.7])
    with torch.no_grad():
        model.sh_rest.copy_(torch.randn_like(model.sh_rest))
    model.active_sh_degree = 3
    keep = torch.tensor([True, False, True])
    subset = select(model, keep)
    assert subset.active_sh_degree == 3 and subset.max_sh_degree == 3
    eye = torch.tensor([1.0, 2.0, -3.0])
    assert torch.allclose(subset.colors_for(eye), model.colors_for(eye)[keep])
    # The subset is a copy: changing it does not touch the original.
    with torch.no_grad():
        subset.xyz.add_(1.0)
    assert torch.equal(model.xyz[0], torch.zeros(3))
