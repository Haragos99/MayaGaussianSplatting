import numpy as np
from dataclasses import dataclass
from scipy.spatial import cKDTree
from.colmap import  Points3D

@dataclass
class GaussianModel:
    xyz: np.ndarray
    scale: np.ndarray
    rotation: np.ndarray
    opacity: np.ndarray
    color: np.ndarray

   
def initialize_gaussians(points: Points3D) -> "GaussianModel":
    xyz = points.xyz.astype(np.float32, copy=True)
    color = (points.rgb.astype(np.float32)/ 255.0)
    tree = cKDTree(xyz)

    distances, _ = tree.query(xyz, k=2)
    nearest_distance = distances[:, 1]
    scale_value = nearest_distance * 0.5
    scale = np.repeat(scale_value[:, None], 3, axis=1).astype(np.float32)
    rotation = np.zeros((len(xyz), 4),dtype=np.float32)
    rotation[:, 0] = 1.0

    opacity = np.full(len(xyz), 0.5, dtype=np.float32)

    return GaussianModel(
        xyz=xyz,
        scale=scale,
        rotation=rotation,
        opacity=opacity,
        color=color,
    )