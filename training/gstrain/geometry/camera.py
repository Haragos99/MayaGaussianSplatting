from dataclasses import dataclass
import torch
from ..torch_utils import choose_device


def fit_resolution(width: int, height: int, max_edge: int) -> tuple[int, int]:
    """Largest size with the same aspect ratio whose longest edge is max_edge."""
    longest = max(width, height)
    if longest <= max_edge:
        return int(width), int(height)

    scale = max_edge / longest

    return max(1, round(width * scale)), max(1, round(height * scale))

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

    @property
    def camera_center(self) -> torch.Tensor:
        """(3,) camera position in world space, from X_c = R X_w + t."""
        return -self.R.transpose(-1, -2) @ self.t

    def rescaled(self, width: int, height: int) -> "TorchCamera":
        """Same view, different pixel grid.

        Pixel centres sit on integer coordinates, so the image spans
        [-0.5, W - 0.5] and the principal point maps as (c + 0.5) * s - 0.5.
        Scaling c by s alone would shift the image by half a pixel per axis.
        """
        scale_x = width / self.width
        scale_y = height / self.height

        return TorchCamera(
            width=int(width),
            height=int(height),
            fx=self.fx * scale_x,
            fy=self.fy * scale_y,
            cx=(self.cx + 0.5) * scale_x - 0.5,
            cy=(self.cy + 0.5) * scale_y - 0.5,
            R=self.R,
            t=self.t,
        )

    def downscaled(self, max_edge: int) -> "TorchCamera":
        return self.rescaled(*fit_resolution(self.width, self.height, max_edge))

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
