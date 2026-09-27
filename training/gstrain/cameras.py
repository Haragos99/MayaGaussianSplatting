from dataclasses import dataclass
import numpy as np
from.colmap import Camera, ColmapData, Image, quaternion_to_rotmatrix

@dataclass
class TrainingCamera:
    image_id: int
    name: str

    width: int
    height: int

    fx: float
    fy: float
    cx: float
    cy: float

    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    C: np.ndarray






def create_training_camera(camera: Camera, image: Image) -> TrainingCamera:
    if camera.model == "PINHOLE":
        fx, fy, cx, cy = camera.params
    elif camera.model == "SIMPLE_PINHOLE":
        f, cx, cy = camera.params
        fx = f
        fy = f
    elif camera.model == "SIMPLE_RADIAL":
        f, cx, cy, k = camera.params

        fx = f
        fy = f

    else:
        raise NotImplementedError( f"Camera model {camera.model} " "is not supported yet.")

    # Camera Intrinsic Matrix
    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # Rotation Matrix
    R = quaternion_to_rotmatrix(image.qvec)

    # Trasnlate Vector
    t = image.tvec

    # Camera center in world coordinates
    C = -R.T @ t

    return TrainingCamera(
        image_id=image.id,
        name=image.name,
        width=camera.width,
        height=camera.height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        R=R,
        t=t,
        K=K,
        C=C
    )



def create_training_cameras(data: ColmapData) -> TrainingCamera:

    training_cameras = {}

    for image_id, image in data.images.items():

        # Find the camera intrinsics
        camera = data.cameras[image.camera_id]

        training_camera = create_training_camera(camera, image)
        training_cameras[image_id] = training_camera

    return training_cameras