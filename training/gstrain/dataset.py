from pathlib import Path
from PIL import Image
import numpy as np
import torch
from dataclasses import dataclass
from training.gstrain.cameras import TrainingCamera
from training.gstrain.colmap import Camera, Points3D
from training.gstrain.model import initialize_gaussians

def load_image(images_dir: Path, image_name: str) -> torch.Tensor:
    image_path = images_dir / image_name

    image = Image.open(image_path).convert("RGB")

    return np.asarray(image)

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
         
