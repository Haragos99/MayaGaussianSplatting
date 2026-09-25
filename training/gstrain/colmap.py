
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import struct

@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray


@dataclass
class Image:
    id: int
    qvec: np.ndarray       # qw, qx, qy, qz
    tvec: np.ndarray       # tx, ty, tz
    camera_id: int
    name: str

    def rotation(self):
        return quaternion_to_rotmatrix(self.qvec)

    # fromula: C = -Rᵀt
    def camera_center(self):
        R = self.rotation()
        return -R.T @ self.tvec



def quaternion_to_rotmatrix(qvec) -> np.ndarray:
    """COLMAP quaternion [w, x, y, z] -> 3x3 rotation matrix (Hamilton)."""
    w, x, y, z = (float(v) for v in qvec)
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def rotmatrix_to_quaternionc(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> quaternion [w, x, y, z]."""
    R = np.asarray(R, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q) # norlamlize quaternion



@dataclass
class Points3D:
    ids: np.ndarray  # (N,) int64
    xyz: np.ndarray  # (N,3) float64
    rgb: np.ndarray  # (N,3) uint8
    error: np.ndarray  # (N,) float64


@dataclass
class ColmapData:
    cameras: dict[int, Camera]
    images: dict[int, Image]
    points3D: dict[int, Points3D]
    path: Path



# COLMAP CAMERA MODELS
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}

CAMERA_MODEL_IDS = {
    name: model_id
    for model_id, (name, _) in CAMERA_MODELS.items()
}


# BINARY HELPER
def read_bytes(fid, num_bytes: int, fmt: str):
    data = fid.read(num_bytes)
    if len(data) != num_bytes:
        raise EOFError(f"truncated COLMAP file: wanted {num_bytes} bytes, got {len(data)}")
    return struct.unpack("<" + fmt, data)


def read_cameras_txt(path: Path)->dict[int, Camera]:
    cameras = {}
    with open(path) as f:
        for line in f:

            if line.startswith("#") or not line.strip():
                continue

            x = line.split()

            camera = Camera(
                id=int(x[0]),
                model=x[1],
                width=int(x[2]),
                height=int(x[3]),
                params=np.asarray(x[4:],dtype=np.float64)
            )

            cameras[camera.id] = camera

    return cameras


def read_cameras_binary(path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    with open(path, "rb") as fid:

        (num_cameras,) = read_bytes(fid, 8, "Q")
        for _ in range(num_cameras):
            camera_id, model_id, width, height = read_bytes(fid, 24, "iiQQ")
            if model_id not in CAMERA_MODELS:
                raise ValueError(f"unknown COLMAP camera model id {model_id}")
            
            model, num_params = CAMERA_MODELS[model_id]
            params = read_bytes(fid, 8 * num_params, "d" * num_params)
            cameras[camera_id] = Camera(camera_id, model, width, height, np.array(params, dtype=np.float64))
    return cameras

# IMAGES
def read_images_txt(path)->dict[int, Image]:
    images: dict[int, Image] = {}

    with open(path) as f:

        while True:
            line = f.readline()

            if not line:
                break

            if line.startswith("#") or not line.strip():
                continue

            # Image header
            x = line.split()
            image_id = int(x[0])
            qvec = np.asarray(x[1:5], dtype=np.float64)
            tvec = np.asarray(x[5:8], dtype=np.float64)
            camera_id = int(x[8])
            name = x[9]

            # 2D observations
            line = f.readline()
            values = line.split()

            if values:
                values = np.asarray(values,dtype=np.float64)
                values = values.reshape(-1, 3)
                xys = values[:, :2]
                point3D_ids = values[:, 2].astype(np.int64)

            else:
                xys = np.empty((0, 2), dtype=np.float64)
                point3D_ids = np.empty((0,),dtype=np.int64)

            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=name,
                xys=xys,
                point3D_ids=point3D_ids
            )

    return images

def read_images_binary(path) -> dict[int, Image]:
    images: dict[int, Image] = {}

    with open(path, "rb") as fid:
        (num_images,) = read_bytes(fid, 8, "Q")
        for _ in range(num_images):
            props = read_bytes(fid, 64, "idddddddi")
            image_id = props[0]
            qvec = np.array(props[1:5], dtype=np.float64)
            tvec = np.array(props[5:8], dtype=np.float64)
            camera_id = props[8]
            name_bytes = bytearray()
            while True:
                char = fid.read(1)
                if char == b"\x00" or char == b"":
                    break

                name_bytes += char

            (num_points2d,) = read_bytes(fid, 8, "Q")
            fid.seek(24 * num_points2d, 1)  # 2D observations are not needed for training
            images[image_id] = Image(image_id, qvec, tvec, camera_id, name_bytes.decode("utf-8"))

    return images




# POINTS3D
def read_points3D_txt(path: Path)->Points3D:
    ids = []
    xyz = []
    rgb = []
    error = []

    with open(path, "r") as f:
        for line in f:

            if line.startswith("#") or not line.strip():
                continue

            x = line.split()
            ids.append(int(x[0]))
            xyz.append([float(x[1]),float(x[2]),float(x[3])])
            rgb.append([int(x[4]),int(x[5]),int(x[6])])
            error.append(float(x[7]))

    return Points3D(
        ids=np.asarray(ids, dtype=np.int64),
        xyz=np.asarray(xyz, dtype=np.float64),
        rgb=np.asarray(rgb, dtype=np.uint8),
        error=np.asarray(error, dtype=np.float64),
    )




def read_points3d_binary(path) -> Points3D:
    ids = []
    xyz = []
    rgb = []
    error = []

    with open(path, "rb") as fid:
        (num_points,) = read_bytes(fid, 8, "Q")
        for _ in range(num_points):
            props = read_bytes(fid, 43, "QdddBBBd")
            ids.append(props[0])
            xyz.append(props[1:4])
            rgb.append(props[4:7])
            error.append(props[7])
            (track_length,) = read_bytes(fid, 8, "Q")
            fid.seek(8 * track_length, 1)
    return Points3D(
        np.array(ids, dtype=np.int64).reshape(-1),
        np.array(xyz, dtype=np.float64).reshape(-1, 3),
        np.array(rgb, dtype=np.uint8).reshape(-1, 3),
        np.array(error, dtype=np.float64).reshape(-1),
    )





def find_colmap_dir(source: Path) -> Path:
    """Accept either the model dir itself or a COLMAP project root."""
    source = Path(source)
    candidates = [source, source / "sparse" / "0", source / "sparse"]
    for candidate in candidates:
        if (candidate / "cameras.bin").exists() or (candidate / "cameras.txt").exists():
            return candidate
        
    raise FileNotFoundError(
        f"no COLMAP model (cameras.bin/.txt) under {source}; looked in: "
        + ", ".join(str(c) for c in candidates)
    )

# LOADER
def load_colmap(source) -> ColmapData:
    path = find_colmap_dir(Path(source))

    # Load
    if (path / "cameras.bin").exists():
        cameras = read_cameras_binary(path / "cameras.bin")
        images = read_images_binary(path / "images.bin")
        points3D = read_points3d_binary(path / "points3D.bin")

    else:
        cameras = read_cameras_txt(path / "cameras.txt")
        images = read_images_txt(path / "images.txt")
        points3D = read_points3D_txt(path / "points3D.txt")

    return ColmapData(cameras, images, points3D, path)



