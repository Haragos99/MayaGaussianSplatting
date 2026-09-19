"""Raw (pre-activation) Gaussian parameters and their initialisation.

These are exactly the values stored in the .ply. The C++ loader
(src/ply/StandardSplatDecoder.cpp) applies the activations:
    opacity -> sigmoid, scale -> exp, colour -> 0.5 + C0 * f_dc.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sh import num_sh_coeffs, rgb_to_sh


def inverse_sigmoid(x):
    x = np.clip(np.asarray(x, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return np.log(x / (1.0 - x)).astype(np.float32)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


@dataclass
class GaussianParams:
    xyz: np.ndarray  # (N,3)
    features_dc: np.ndarray  # (N,1,3)
    features_rest: np.ndarray  # (N,K-1,3)
    opacity: np.ndarray  # (N,1)   logit
    scaling: np.ndarray  # (N,3)   log
    rotation: np.ndarray  # (N,4)  quaternion [w,x,y,z]

    def __post_init__(self) -> None:
        n = self.xyz.shape[0]
        shapes = {
            "xyz": (n, 3),
            "opacity": (n, 1),
            "scaling": (n, 3),
            "rotation": (n, 4),
        }
        for name, expected in shapes.items():
            got = getattr(self, name).shape
            if got != expected:
                raise ValueError(f"{name} has shape {got}, expected {expected}")
        if self.features_dc.shape != (n, 1, 3):
            raise ValueError(f"features_dc has shape {self.features_dc.shape}, expected {(n, 1, 3)}")
        if self.features_rest.ndim != 3 or self.features_rest.shape[0] != n or self.features_rest.shape[2] != 3:
            raise ValueError(f"features_rest has shape {self.features_rest.shape}, expected (N,K-1,3)")

    def __len__(self) -> int:
        return int(self.xyz.shape[0])

    def select(self, index: np.ndarray) -> "GaussianParams":
        return GaussianParams(
            xyz=self.xyz[index],
            features_dc=self.features_dc[index],
            features_rest=self.features_rest[index],
            opacity=self.opacity[index],
            scaling=self.scaling[index],
            rotation=self.rotation[index],
        )

    @property
    def sh_degree(self) -> int:
        total = self.features_rest.shape[1] + 1
        degree = int(round(np.sqrt(total))) - 1
        if num_sh_coeffs(degree) != total:
            raise ValueError(f"{total} SH coefficients is not a square number")
        return degree

    def activated_scale(self) -> np.ndarray:
        return np.clip(np.exp(self.scaling.astype(np.float64)), 1e-4, 10.0)

    def activated_opacity(self) -> np.ndarray:
        return np.clip(sigmoid(self.opacity), 0.0, 1.0)

    def activated_color(self) -> np.ndarray:
        from .sh import sh_to_rgb

        return np.clip(sh_to_rgb(self.features_dc[:, 0, :]), 0.0, 1.0)


def knn_mean_distance(xyz: np.ndarray, k: int = 3) -> np.ndarray:
    """Mean distance to the k nearest neighbours, used to size the initial splats."""
    from scipy.spatial import cKDTree

    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.shape[0] <= k:
        return np.full((xyz.shape[0],), 0.01, dtype=np.float64)
    tree = cKDTree(xyz)
    # k+1 because the first hit is the point itself.
    dist, _ = tree.query(xyz, k=k + 1, workers=-1)
    return dist[:, 1:].mean(axis=1)


def create_from_point_cloud(
    xyz: np.ndarray,
    rgb: np.ndarray,
    sh_degree: int = 3,
    initial_opacity: float = 0.1,
) -> GaussianParams:
    """Initialise Gaussians from the COLMAP sparse cloud (rgb is uint8 0..255)."""
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    n = xyz.shape[0]
    if n == 0:
        raise ValueError("point cloud is empty; COLMAP reconstruction produced no 3D points")

    colors = np.asarray(rgb, dtype=np.float32) / 255.0
    features = np.zeros((n, num_sh_coeffs(sh_degree), 3), dtype=np.float32)
    features[:, 0, :] = rgb_to_sh(colors)

    dist = np.clip(knn_mean_distance(xyz), 1e-7, None)
    scaling = np.log(dist).astype(np.float32)[:, None].repeat(3, axis=1)

    rotation = np.zeros((n, 4), dtype=np.float32)
    rotation[:, 0] = 1.0

    opacity = np.full((n, 1), inverse_sigmoid(initial_opacity), dtype=np.float32)

    return GaussianParams(
        xyz=xyz,
        features_dc=features[:, :1, :],
        features_rest=features[:, 1:, :],
        opacity=opacity,
        scaling=scaling,
        rotation=rotation,
    )
