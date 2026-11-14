import numpy as np
import torch
import torch.nn as nn

from .sh import eval_sh, num_sh_coeffs, rgb_to_sh, sh_to_rgb
from .torch_utils import choose_device

EPS = 1e-6

# Optimizer-group / densification name -> model attribute. Every per-splat
# parameter is listed here, so clone / split / prune / relocate carry all of
# them without knowing what they are.
PARAMETER_ATTRIBUTES = {
    "xyz": "xyz",
    "scale": "scale_raw",
    "rotation": "rotation",
    "opacity": "opacity_raw",
    "f_dc": "sh_dc",
    "f_rest": "sh_rest",
}


class TorchGaussianModel(nn.Module):
    """Trainable splats. Colour is spherical harmonics up to `sh_degree`.

    Raw storage matches the 3DGS .ply: log scale, logit opacity, SH
    coefficients (colour = 0.5 + SH(view direction), clamped at 0).
    `active_sh_degree` starts at 0 and is raised during training.
    """

    def __init__(
        self,
        xyz: torch.Tensor,
        scale: torch.Tensor,
        rotation: torch.Tensor,
        opacity: torch.Tensor,
        color: torch.Tensor,
        sh_degree: int = 3,
    ) -> None:
        super().__init__()
        if not 0 <= sh_degree <= 3:
            raise ValueError("sh_degree must be between 0 and 3")

        self.max_sh_degree = int(sh_degree)
        self.active_sh_degree = 0

        self.xyz = nn.Parameter(xyz.float())

        self.scale_raw = nn.Parameter(scale.float().clamp_min(EPS).log())

        # Quaternion [w, x, y, z], normalized lazily when it is used.
        self.rotation = nn.Parameter(rotation.float())

        opacity = opacity.float().clamp(1e-4, 1.0 - 1e-4)
        self.opacity_raw = nn.Parameter(opacity.logit())

        count = xyz.shape[0]
        self.sh_dc = nn.Parameter(rgb_to_sh(color.float()).reshape(count, 1, 3))
        rest = num_sh_coeffs(self.max_sh_degree) - 1
        self.sh_rest = nn.Parameter(torch.zeros(count, rest, 3, dtype=torch.float32, device=xyz.device))

    @property
    def scale(self) -> torch.Tensor:
        return self.scale_raw.exp()

    @property
    def opacity(self) -> torch.Tensor:
        return self.opacity_raw.sigmoid()

    @property
    def color(self) -> torch.Tensor:
        """(N, 3) view-independent (band-0) colour."""
        return sh_to_rgb(self.sh_dc[:, 0]).clamp_min(0.0)

    @property
    def features(self) -> torch.Tensor:
        """(N, K, 3) all SH coefficients."""
        return torch.cat([self.sh_dc, self.sh_rest], dim=1)

    def colors_for(self, camera_center: torch.Tensor) -> torch.Tensor:
        """(N, 3) colour seen from `camera_center`, using the active SH bands."""
        if self.active_sh_degree == 0:
            return self.color
        directions = self.xyz - camera_center.to(self.xyz)
        directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return (eval_sh(self.active_sh_degree, self.features, directions) + 0.5).clamp_min(0.0)

    def oneup_sh_degree(self) -> bool:
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            return True
        return False

    @property
    def device(self) -> torch.device:
        return self.xyz.device

    def __len__(self) -> int:
        return self.xyz.shape[0]

    def extra_repr(self) -> str:
        return (
            f"num_gaussians={len(self)}, sh_degree={self.active_sh_degree}/{self.max_sh_degree}, "
            f"device={self.device}"
        )

    @classmethod
    def from_numpy(
        cls, gaussians, device: torch.device | None = None, sh_degree: int = 3
    ) -> "TorchGaussianModel":
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
            sh_degree=sh_degree,
        )

        return model.to(device)
