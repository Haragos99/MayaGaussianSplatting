from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
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





from dataclasses import dataclass


@dataclass
class TorchCamera:
    width: int
    height: int

    fx: torch.Tensor
    fy: torch.Tensor
    cx: torch.Tensor
    cy: torch.Tensor

    R: torch.Tensor
    t: torch.Tensor
    C: torch.Tensor

    @classmethod
    def from_training_camera(cls,camera, device: torch.device):
        dtype = torch.float32
        return cls(
            width=int(camera.width),
            height=int(camera.height),

            fx=torch.tensor(
                camera.fx,
                dtype=dtype,
                device=device,
            ),

            fy=torch.tensor(
                camera.fy,
                dtype=dtype,
                device=device,
            ),

            cx=torch.tensor(
                camera.cx,
                dtype=dtype,
                device=device,
            ),

            cy=torch.tensor(
                camera.cy,
                dtype=dtype,
                device=device,
            ),

            R=torch.as_tensor(
                camera.R,
                dtype=dtype,
                device=device,
            ),

            t=torch.as_tensor(
                camera.t,
                dtype=dtype,
                device=device,
            ),
        )













class Scene:
     def __init__(self,train_cameras: list[TrainingCamera],test_cameras: list[Camera]) -> None:
        self.train_cameras = train_cameras
        self.test_cameras = test_cameras


     def load(points3D: Points3D):
         # TODO impelemt the loading here
         gaussians_model = initialize_gaussians(points3D)







def world_to_camera(camera,xyz,):
    return xyz @ camera.R.T + camera.t


def quaternion_to_rotation_matrix(q: torch.Tensor,) -> torch.Tensor:
    """Calaculate all quaternions in torch"""
    # Normalize quaternion.
    q = F.normalize(q,dim=-1,eps=1e-8,)

    qw, qx, qy, qz = q.unbind(dim=-1)

    R = torch.empty(
        (*q.shape[:-1], 3, 3),
        dtype=q.dtype,
        device=q.device,
    )

    R[..., 0, 0] = (1- 2 * (qy * qy + qz * qz))
    R[..., 0, 1] = (2 * (qx * qy - qz * qw))
    R[..., 0, 2] = (2 * (qx * qz + qy * qw))
    R[..., 1, 0] = (2 * (qx * qy + qz * qw))
    R[..., 1, 1] = (1 - 2 * (qx * qx + qz * qz))
    R[..., 1, 2] = (2 * (qy * qz - qx * qw))
    R[..., 2, 0] = (2 * (qx * qz - qy * qw))
    R[..., 2, 1] = (2 * (qy * qz + qx * qw))
    R[..., 2, 2] = (1 - 2 * (qx * qx + qy * qy))

    return R


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu"
    )