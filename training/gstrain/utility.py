import numpy as np
from.colmap import Camera, Points3D

def world_to_camera(camera: Camera, points: Points3D) ->  np.ndarray:
    """ points: (N, 3) returns: (N, 3)"""
    return (camera.R @ points.T).T + camera.t


def project_points(camera: Camera, points: Points3D):
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