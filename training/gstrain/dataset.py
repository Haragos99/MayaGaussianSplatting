from pathlib import Path
from PIL import Image
import numpy as np
import torch
from dataclasses import dataclass
from .cameras import TrainingCamera, create_training_cameras
from .colmap import Camera, ColmapData, Points3D
from .geometry.camera import TorchCamera
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


"Store the data of the camera and tatget img "
@dataclass
class TrainView:
    """One camera and the photograph it should reproduce."""
    camera: TorchCamera
    image: torch.Tensor  # (H, W, 3) in [0, 1]


def views_memory_mb(count: int, width: int, height: int) -> float:
    return count * width * height * 3 * 4 / 1024**2


def build_views(
    data: ColmapData,
    images_dir: Path,
    max_edge: int = 800,
    device: torch.device | None = None,
    image_device: torch.device | None = None,
    limit: int | None = None,
) -> list[TrainView]:
    """Every COLMAP image as a training view, downscaled to `max_edge`.

    Targets are cached on `image_device` (the render device by default). A full
    scene at high resolution will not fit next to the render, so pass
    torch.device("cpu") to keep them on the host - `train_step` moves one view
    at a time, and a pinned 2 MB copy costs far less than the render itself.
    """
    device = device or choose_device()
    image_device = device if image_device is None else image_device
    pin = image_device.type == "cpu" and device.type == "cuda"

    views: list[TrainView] = []
    training_cameras = create_training_cameras(data)
    for image_id, training_camera in list(training_cameras.items())[:limit]:
        camera = TorchCamera.from_training_camera(training_camera, device=device)
        camera = camera.downscaled(max_edge)

        image = load_target_image(
            images_dir, data.images[image_id].name,
            camera.width, camera.height, device=image_device,
        )

        views.append(TrainView(camera=camera, image=image.pin_memory() if pin else image))

    return views

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
         
