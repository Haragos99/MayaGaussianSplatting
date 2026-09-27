import numpy as np
from.colmap import  Points3D
from.cameras import TrainingCamera
import matplotlib.pyplot as plt

def world_to_camera(camera: TrainingCamera, points: Points3D) -> np.ndarray:
    """
    Transform world-space points to camera space.

    points.xyz: (N, 3)
    returns:    (N, 3)
    """
    return (camera.R @ points.xyz.T).T + camera.t


def project_points(camera: TrainingCamera, points: Points3D):
    """
    points: (N, 3) world coordinates

    returns:
        pixels: (N, 2)
        depth:  (N,)
    """
    points_camera = world_to_camera(camera, points)

    X = points_camera[:, 0]
    Y = points_camera[:, 1]
    Z = points_camera[:, 2]

    u = camera.fx * X / Z + camera.cx
    v = camera.fy * Y / Z + camera.cy

    pixels = np.stack([u, v], axis=1)

    return pixels, Z



def plot_points3D(points: Points3D):
    xyz = points.xyz
    rgb = points.rgb / 255.0

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        c=rgb,
        s=1,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title("COLMAP 3D Point Cloud")

    return  plt.show()