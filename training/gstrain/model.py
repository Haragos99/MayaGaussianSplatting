import numpy as np
from dataclasses import dataclass
from scipy.spatial import cKDTree
from .colmap import Points3D

# 3DGS starts every splat at 10% opacity: faint enough that overlapping
# initial splats do not hide each other, so all of them receive gradients.
INITIAL_OPACITY = 0.1


@dataclass
class GaussianModel:
    xyz: np.ndarray
    scale: np.ndarray
    rotation: np.ndarray
    opacity: np.ndarray
    color: np.ndarray

    def __len__(self) -> int:
        return len(self.xyz)

    def select(self, index: np.ndarray) -> "GaussianModel":
        return GaussianModel(
            xyz=self.xyz[index],
            scale=self.scale[index],
            rotation=self.rotation[index],
            opacity=self.opacity[index],
            color=self.color[index],
        )


def knn_mean_distance(xyz: np.ndarray, k: int = 3) -> np.ndarray:
    """Mean distance from each point to its k nearest neighbours."""
    xyz = np.asarray(xyz, dtype=np.float64)
    if len(xyz) <= k:
        return np.full(len(xyz), 0.01)
    # k + 1 because each point's nearest hit is itself.
    distances, _ = cKDTree(xyz).query(xyz, k=k + 1)
    return distances[:, 1:].mean(axis=1)


def initialize_gaussians(
    points: Points3D,
    initial_opacity: float = INITIAL_OPACITY,
    max_points: int | None = None,
    seed: int = 0,
) -> GaussianModel:
    """One isotropic splat per COLMAP point, as in 3DGS.

    Size is the mean distance to the 3 nearest neighbours. A single neighbour
    is zero for duplicate points (8% of MeSP), which made those splats
    invisible forever. Sizes are computed on the full cloud *before*
    `max_points` subsampling: subsampling first would inflate the spacing and
    start every splat far too large.
    """
    xyz = points.xyz.astype(np.float32, copy=True)
    color = points.rgb.astype(np.float32) / 255.0

    size = np.clip(knn_mean_distance(xyz), 1e-7, None)
    scale = np.repeat(size[:, None], 3, axis=1).astype(np.float32)

    rotation = np.zeros((len(xyz), 4), dtype=np.float32)
    rotation[:, 0] = 1.0

    opacity = np.full(len(xyz), initial_opacity, dtype=np.float32)

    model = GaussianModel(xyz=xyz, scale=scale, rotation=rotation, opacity=opacity, color=color)
    if max_points and len(model) > max_points:
        pick = np.random.default_rng(seed).choice(len(model), max_points, replace=False)
        model = model.select(np.sort(pick))
    return model
