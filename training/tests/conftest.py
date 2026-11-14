import math
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from gstrain.colmap import rotmatrix_to_quaternionc
from gstrain.geometry.camera import TorchCamera
from gstrain.model_torch import TorchGaussianModel


def pytest_configure(config) -> None:
    config.addinivalue_line("markers", "slow: end-to-end tests (deselect with -m \"not slow\")")


# --- cameras and models --------------------------------------------------- #

def make_camera(size: int = 16, focal: float | None = None, height: int | None = None) -> TorchCamera:
    """Pinhole camera at the origin looking down +z."""
    height = height or size
    focal = float(size) if focal is None else float(focal)
    return TorchCamera(
        width=size, height=height,
        fx=torch.tensor(focal), fy=torch.tensor(focal),
        cx=torch.tensor(size / 2), cy=torch.tensor(height / 2),
        R=torch.eye(3), t=torch.zeros(3),
    )


def look_at_rotation(center: np.ndarray, target: np.ndarray) -> np.ndarray:
    """World -> camera rotation of a COLMAP camera (+z forward, +y down)."""
    forward = target - center
    forward = forward / np.linalg.norm(forward)
    down = np.array([0.0, 1.0, 0.0])
    right = np.cross(down, forward)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.stack([right, down, forward])


def make_model(xyz, scale=0.1, opacity=0.8, color=None, sh_degree: int = 3) -> TorchGaussianModel:
    xyz = torch.as_tensor(xyz, dtype=torch.float32).reshape(-1, 3)
    count = len(xyz)
    scale = torch.as_tensor(scale, dtype=torch.float32)
    scale = scale.expand(count, 3).clone() if scale.ndim < 2 else scale
    opacity = torch.as_tensor(opacity, dtype=torch.float32).expand(count).clone()
    color = torch.full((count, 3), 0.7) if color is None else torch.as_tensor(color, dtype=torch.float32)
    rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1)
    return TorchGaussianModel(xyz, scale, rotation, opacity, color, sh_degree=sh_degree)


# --- a tiny COLMAP project on disk ---------------------------------------- #

SCENE_WIDTH = 32
SCENE_HEIGHT = 24
SCENE_IMAGES = 8


def write_tiny_scene(root: Path, distortion: float = 0.01) -> Path:
    """<root>/sparse/0/*.txt + <root>/images/*.png: 8 cameras on a ring around a
    small coloured point cloud. Camera 1 is PINHOLE, camera 2 SIMPLE_RADIAL."""
    sparse = root / "sparse" / "0"
    images = root / "images"
    sparse.mkdir(parents=True)
    images.mkdir(parents=True)

    f, cx, cy = 30.0, SCENE_WIDTH / 2, SCENE_HEIGHT / 2
    (sparse / "cameras.txt").write_text(
        "# id model width height params\n"
        f"1 PINHOLE {SCENE_WIDTH} {SCENE_HEIGHT} {f} {f} {cx} {cy}\n"
        f"2 SIMPLE_RADIAL {SCENE_WIDTH} {SCENE_HEIGHT} {f} {cx} {cy} {distortion}\n"
    )

    lines = ["# image list"]
    rng = np.random.default_rng(0)
    for i in range(SCENE_IMAGES):
        angle = 2 * math.pi * i / SCENE_IMAGES
        center = np.array([3.0 * math.sin(angle), 0.0, -3.0 * math.cos(angle)])
        R = look_at_rotation(center, np.zeros(3))
        q = rotmatrix_to_quaternionc(R)
        t = -R @ center
        name = f"img_{i:02d}.png"
        camera_id = 1 + i % 2
        lines.append(" ".join(str(v) for v in [i + 1, *q, *t, camera_id, name]))
        lines.append("")  # no 2D observations
        pixels = (rng.random((SCENE_HEIGHT, SCENE_WIDTH, 3)) * 80 + 60).astype(np.uint8)
        pixels[SCENE_HEIGHT // 3: 2 * SCENE_HEIGHT // 3, SCENE_WIDTH // 3: 2 * SCENE_WIDTH // 3] = (200, 80, 40)
        Image.fromarray(pixels).save(images / name)
    (sparse / "images.txt").write_text("\n".join(lines) + "\n")

    points = ["# points"]
    for i in range(60):
        x, y, z = rng.normal(0.0, 0.4, size=3)
        r, g, b = rng.integers(0, 255, size=3)
        points.append(f"{i + 1} {x} {y} {z} {r} {g} {b} 0.5")
    (sparse / "points3D.txt").write_text("\n".join(points) + "\n")
    return root


@pytest.fixture
def tiny_scene(tmp_path) -> Path:
    return write_tiny_scene(tmp_path / "scene")
