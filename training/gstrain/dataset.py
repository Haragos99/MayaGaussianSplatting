from pathlib import Path
from PIL import Image
import numpy as np
import torch
from dataclasses import dataclass
from .cameras import TrainingCamera
from .colmap import Camera, Points3D
from .model import initialize_gaussians
from .torch_utils import choose_device

def load_image(images_dir: Path, image_name: str) -> torch.Tensor:
    image_path = images_dir / image_name

    image = Image.open(image_path).convert("RGB")

    return np.asarray(image)


def load_target_image(
    images_dir: Path,
    image_name: str,
    width: int,
    height: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """-> (H, W, 3) float tensor in [0, 1] on `device`.

    Resized to match a rescaled camera; the two must always agree or the render
    is offset from its target and the loss cannot go to zero.
    """
    image = Image.open(images_dir / image_name).convert("RGB")

    if image.size != (width, height):
        image = image.resize((width, height), Image.LANCZOS)

    pixels = np.asarray(image, dtype=np.float32) / 255.0

    return torch.as_tensor(pixels, device=device or choose_device())

@dataclass
class ProjectedGaussian:
    mean: np.ndarray          # (2,)
    covariance: np.ndarray    # (2, 2)
    color: np.ndarray         # (3,)
    opacity: float
    depth: float


class Scene:
     def __init__(self,train_cameras: list[TrainingCamera],test_cameras: list[Camera]) -> None:
        self.train_cameras = train_cameras
        self.test_cameras = test_cameras


     def load(points3D: Points3D):
         # TODO impelemt the loading here
         gaussians_model = initialize_gaussians(points3D)
         
