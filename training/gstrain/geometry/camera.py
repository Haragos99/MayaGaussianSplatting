from dataclasses import dataclass
import torch
from ..torch_utils import choose_device

"""Camera stored as CUDA tensors"""
@dataclass
class TorchCamera:
    width: int
    height: int

    fx: torch.Tensor
    fy: torch.Tensor
    cx: torch.Tensor
    cy: torch.Tensor

    R: torch.Tensor  # (3, 3) world -> camera rotation
    t: torch.Tensor  # (3,)   world -> camera translation

    @property
    def device(self) -> torch.device:
        return self.R.device

    @classmethod
    def from_training_camera(cls, camera, device: torch.device | None = None) -> "TorchCamera":
        device = device or choose_device()

        def scalar(value) -> torch.Tensor:
            return torch.tensor(float(value), dtype=torch.float32, device=device)

        def array(value) -> torch.Tensor:
            return torch.as_tensor(value, dtype=torch.float32, device=device)

        return cls(
            width=int(camera.width),
            height=int(camera.height),
            fx=scalar(camera.fx),
            fy=scalar(camera.fy),
            cx=scalar(camera.cx),
            cy=scalar(camera.cy),
            R=array(camera.R),
            t=array(camera.t),
        )
