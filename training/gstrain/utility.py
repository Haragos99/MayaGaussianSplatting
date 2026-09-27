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


# Plots -----------------------------------------

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


def plot_scene(points: Points3D, cameras: dict[int, TrainingCamera]):
    xyz = points.xyz
    rgb = points.rgb / 255.0

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 3D points
    ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        c=rgb,
        s=1,
    )

    # Camera centers
    centers = np.array([
        camera.C
        for camera in cameras.values()
    ])

    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        centers[:, 2],
        marker="^",
        s=30,
        label="Cameras",
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title("COLMAP Scene")
    ax.legend()

    plt.show()


def plot_projection(camera: TrainingCamera, points: Points3D):
    pixels, depth = project_points(camera, points)

    print(np.shape(pixels))

    # Check 3D points are actually visible inside the camera's image
    in_front = depth > 0

    u = pixels[:, 0]
    v = pixels[:, 1]

    inside_width = (u >= 0) & (u < camera.width)
    inside_height = (v >= 0) & (v < camera.height)

    valid = (
        in_front
        & inside_width
        & inside_height
    )

    plt.figure(figsize=(10, 8))

    plt.scatter(
        pixels[valid, 0],
        pixels[valid, 1],
        s=1,
    )

    plt.xlim(0, camera.width)
    plt.ylim(camera.height, 0)

    plt.xlabel("u")
    plt.ylabel("v")
    plt.title(f"Projected COLMAP points: {camera.name}")

    plt.show()





def show_rendered_image(image: np.ndarray) -> None:

    plt.figure(figsize=(12, 8))

    plt.imshow(image)

    plt.axis("off")

    plt.title("Gaussian Render")

    plt.show()