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
from .undistort import Distortion, is_identity, undistort_image

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
    camera: TorchCamera | None = None,
    distortion: Distortion = (0.0, 0.0, 0.0, 0.0),
) -> torch.Tensor:
    """-> (H, W, 3) float tensor in [0, 1] on `device`.

    Resized to match a rescaled camera; the two must always agree or the render
    is offset from its target and the loss cannot go to zero. With a
    `distortion` and the (resized) `camera`, the photo is rectified to the
    pinhole model the rasterizer uses.
    """
    with Image.open(images_dir / image_name) as image:
        image = image.convert("RGB")
        if image.size != (width, height):
            # Downscale first: distortion lives in normalised coordinates, so
            # rectifying the small image is equivalent and much cheaper.
            image = image.resize((width, height), Image.LANCZOS)
        pixels = np.asarray(image, dtype=np.float32) / 255.0

    if camera is not None and not is_identity(distortion):
        pixels = undistort_image(
            pixels,
            float(camera.fx), float(camera.fy), float(camera.cx), float(camera.cy),
            distortion,
        )

    return torch.as_tensor(np.ascontiguousarray(pixels), device=device or choose_device())


"Store the data of the camera and target img "
@dataclass
class TrainView:
    """One camera and the photograph it should reproduce."""
    camera: TorchCamera
    image: torch.Tensor  # (H, W, 3) in [0, 1]
    name: str = ""


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
    """Every COLMAP image as a view, sorted by image name, downscaled to `max_edge`.

    `limit` keeps that many views evenly spaced over the sorted list, so a
    reduced set still surrounds the scene. Lens distortion is rectified on load.

    Targets are cached on `image_device` (the render device by default). Pass
    torch.device("cpu") to keep a large scene on the host; the training loop
    moves one view at a time.
    """
    device = device or choose_device()
    image_device = device if image_device is None else image_device
    pin = image_device.type == "cpu" and device.type == "cuda"

    training_cameras = sorted(create_training_cameras(data).values(), key=lambda c: c.name)
    if limit is not None and 0 < limit < len(training_cameras):
        step = len(training_cameras) / limit
        training_cameras = [training_cameras[int(i * step)] for i in range(limit)]

    views: list[TrainView] = []
    for training_camera in training_cameras:
        camera = TorchCamera.from_training_camera(training_camera, device=device)
        camera = camera.downscaled(max_edge)

        image = load_target_image(
            images_dir, training_camera.name,
            camera.width, camera.height, device=image_device,
            camera=camera, distortion=training_camera.distortion,
        )

        views.append(TrainView(
            camera=camera,
            image=image.pin_memory() if pin else image,
            name=training_camera.name,
        ))

    return views


def split_views(views: list[TrainView], eval_hold: int = 8) -> tuple[list[TrainView], list[TrainView]]:
    """(train, test): every `eval_hold`-th view (by position) is held out, as in 3DGS.

    eval_hold <= 0 trains on everything and returns no test views.
    """
    if eval_hold <= 0:
        return list(views), []
    test = [view for index, view in enumerate(views) if index % eval_hold == 0]
    train = [view for index, view in enumerate(views) if index % eval_hold != 0]
    if not train:
        raise ValueError("no training views left; lower eval_hold or add images")
    return train, test

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
         
