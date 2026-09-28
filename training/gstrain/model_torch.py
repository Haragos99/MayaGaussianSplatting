import numpy as np
import torch
import torch.nn as nn

from .torch_utils import choose_device

EPS = 1e-6

"""It is a Torch version of the GSM insted of use numoy"""
class TorchGaussianModel(nn.Module):
    def __init__(
        self,
        xyz: torch.Tensor,
        scale: torch.Tensor,
        rotation: torch.Tensor,
        opacity: torch.Tensor,
        color: torch.Tensor,
    ) -> None:
        super().__init__()

        self.xyz = nn.Parameter(xyz.float())

        self.scale_raw = nn.Parameter(scale.float().clamp_min(EPS).log())

        # Quaternion [w, x, y, z], normalized lazily when it is used.
        self.rotation = nn.Parameter(rotation.float())

        opacity = opacity.float().clamp(1e-4, 1.0 - 1e-4)
        self.opacity_raw = nn.Parameter(opacity.logit())

        color = color.float().clamp(1e-4, 1.0 - 1e-4)
        self.color_raw = nn.Parameter(color.logit())

    @property
    def scale(self) -> torch.Tensor:
        return self.scale_raw.exp()

    @property
    def opacity(self) -> torch.Tensor:
        return self.opacity_raw.sigmoid()

    @property
    def color(self) -> torch.Tensor:
        return self.color_raw.sigmoid()

    @property
    def device(self) -> torch.device:
        return self.xyz.device

    def __len__(self) -> int:
        return self.xyz.shape[0]

    def extra_repr(self) -> str:
        return f"num_gaussians={len(self)}, device={self.device}"

    @classmethod
    def from_numpy(cls, gaussians, device: torch.device | None = None) -> "TorchGaussianModel":
        """Build from the NumPy `GaussianModel` produced by `initialize_gaussians`."""
        device = device or choose_device()

        def to_tensor(array: np.ndarray) -> torch.Tensor:
            if isinstance(array, torch.Tensor):
                return array.detach().to(device=device, dtype=torch.float32).contiguous().clone()
            return torch.as_tensor(np.ascontiguousarray(array), dtype=torch.float32, device=device)

        model = cls(
            xyz=to_tensor(gaussians.xyz),
            scale=to_tensor(gaussians.scale),
            rotation=to_tensor(gaussians.rotation),
            opacity=to_tensor(gaussians.opacity),
            color=to_tensor(gaussians.color),
        )

        return model.to(device)
