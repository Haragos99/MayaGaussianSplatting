"""COLMAP sparse model reader (binary + text), numpy only.

Format reference: https://colmap.github.io/format.html
- all binary data is little endian
- images store the WORLD -> CAMERA pose as (qw,qx,qy,qz, tx,ty,tz), Hamilton
  convention; the camera centre is `-R^T @ t`
- the camera frame is X right, Y down, Z forward
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# model_id -> (name, number of params)
CAMERA_MODELS: dict[int, tuple[str, int]] = {
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
CAMERA_MODEL_IDS: dict[str, int] = {name: mid for mid, (name, _) in CAMERA_MODELS.items()}

#: Models whose intrinsics are exactly a pinhole (no distortion to undo).
UNDISTORTED_MODELS = frozenset({"SIMPLE_PINHOLE", "PINHOLE"})

#: Models we can rectify ourselves; anything else needs `colmap image_undistorter`.
SUPPORTED_MODELS = UNDISTORTED_MODELS | {"SIMPLE_RADIAL", "RADIAL", "OPENCV"}


@dataclass(frozen=True)
class Distortion:
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0

    @property
    def is_identity(self) -> bool:
        return self.k1 == 0.0 and self.k2 == 0.0 and self.p1 == 0.0 and self.p2 == 0.0

    def apply(self, x, y):
        """Normalised undistorted coords -> distorted coords (COLMAP's OpenCV model)."""
        r2 = x * x + y * y
        radial = 1.0 + self.k1 * r2 + self.k2 * r2 * r2
        xd = x * radial + 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x)
        yd = y * radial + self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y
        return xd, yd


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    @property
    def is_pinhole(self) -> bool:
        return self.model in UNDISTORTED_MODELS

    def _check_supported(self) -> None:
        if self.model not in SUPPORTED_MODELS:
            raise ValueError(
                f"camera {self.id} uses model {self.model}, which this trainer cannot rectify. "
                "Run `colmap image_undistorter` and train on the undistorted output."
            )

    def intrinsics(self) -> tuple[float, float, float, float]:
        """Ideal pinhole (fx, fy, cx, cy); distortion is handled separately."""
        self._check_supported()
        p = [float(v) for v in self.params]
        if self.model == "PINHOLE" or self.model == "OPENCV":
            return p[0], p[1], p[2], p[3]
        return p[0], p[0], p[1], p[2]

    def distortion(self) -> Distortion:
        self._check_supported()
        p = [float(v) for v in self.params]
        if self.model == "SIMPLE_RADIAL":
            return Distortion(k1=p[3])
        if self.model == "RADIAL":
            return Distortion(k1=p[3], k2=p[4])
        if self.model == "OPENCV":
            return Distortion(k1=p[4], k2=p[5], p1=p[6], p2=p[7])
        return Distortion()


@dataclass
class Image:
    id: int
    qvec: np.ndarray  # (4,) world->camera rotation, [w, x, y, z]
    tvec: np.ndarray  # (3,) world->camera translation
    camera_id: int
    name: str

    @property
    def R_w2c(self) -> np.ndarray:
        return qvec_to_rotmat(self.qvec)

    @property
    def camera_center(self) -> np.ndarray:
        return -self.R_w2c.T @ self.tvec


@dataclass
class Points3D:
    ids: np.ndarray  # (N,) int64
    xyz: np.ndarray  # (N,3) float64
    rgb: np.ndarray  # (N,3) uint8
    error: np.ndarray  # (N,) float64


@dataclass
class SparseModel:
    cameras: dict[int, Camera]
    images: dict[int, Image]
    points: Points3D
    path: Path


def qvec_to_rotmat(qvec) -> np.ndarray:
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


def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
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
    return q / np.linalg.norm(q)


# --------------------------------------------------------------------------- #
# binary
# --------------------------------------------------------------------------- #


def _read(fid, num_bytes: int, fmt: str):
    data = fid.read(num_bytes)
    if len(data) != num_bytes:
        raise EOFError(f"truncated COLMAP file: wanted {num_bytes} bytes, got {len(data)}")
    return struct.unpack("<" + fmt, data)


def read_cameras_binary(path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    with open(path, "rb") as fid:
        (num_cameras,) = _read(fid, 8, "Q")
        for _ in range(num_cameras):
            camera_id, model_id, width, height = _read(fid, 24, "iiQQ")
            if model_id not in CAMERA_MODELS:
                raise ValueError(f"unknown COLMAP camera model id {model_id}")
            model, num_params = CAMERA_MODELS[model_id]
            params = _read(fid, 8 * num_params, "d" * num_params)
            cameras[camera_id] = Camera(camera_id, model, width, height, np.array(params, dtype=np.float64))
    return cameras


def read_images_binary(path) -> dict[int, Image]:
    images: dict[int, Image] = {}
    with open(path, "rb") as fid:
        (num_images,) = _read(fid, 8, "Q")
        for _ in range(num_images):
            props = _read(fid, 64, "idddddddi")
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
            (num_points2d,) = _read(fid, 8, "Q")
            fid.seek(24 * num_points2d, 1)  # 2D observations are not needed for training
            images[image_id] = Image(image_id, qvec, tvec, camera_id, name_bytes.decode("utf-8"))
    return images


def read_points3d_binary(path) -> Points3D:
    ids, xyz, rgb, err = [], [], [], []
    with open(path, "rb") as fid:
        (num_points,) = _read(fid, 8, "Q")
        for _ in range(num_points):
            props = _read(fid, 43, "QdddBBBd")
            ids.append(props[0])
            xyz.append(props[1:4])
            rgb.append(props[4:7])
            err.append(props[7])
            (track_length,) = _read(fid, 8, "Q")
            fid.seek(8 * track_length, 1)
    return Points3D(
        np.array(ids, dtype=np.int64).reshape(-1),
        np.array(xyz, dtype=np.float64).reshape(-1, 3),
        np.array(rgb, dtype=np.uint8).reshape(-1, 3),
        np.array(err, dtype=np.float64).reshape(-1),
    )


# --------------------------------------------------------------------------- #
# text
# --------------------------------------------------------------------------- #


def _data_lines(path):
    with open(path, "r", encoding="utf-8") as fid:
        for line in fid:
            line = line.strip()
            if line and not line.startswith("#"):
                yield line


def read_cameras_text(path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    for line in _data_lines(path):
        parts = line.split()
        camera_id = int(parts[0])
        cameras[camera_id] = Camera(
            camera_id,
            parts[1],
            int(parts[2]),
            int(parts[3]),
            np.array([float(p) for p in parts[4:]], dtype=np.float64),
        )
    return cameras


def read_images_text(path) -> dict[int, Image]:
    images: dict[int, Image] = {}
    lines = list(_data_lines(path))
    # Every image occupies two lines; the second holds the 2D observations.
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        image_id = int(parts[0])
        images[image_id] = Image(
            image_id,
            np.array([float(p) for p in parts[1:5]], dtype=np.float64),
            np.array([float(p) for p in parts[5:8]], dtype=np.float64),
            int(parts[8]),
            " ".join(parts[9:]),
        )
    return images


def read_points3d_text(path) -> Points3D:
    ids, xyz, rgb, err = [], [], [], []
    for line in _data_lines(path):
        parts = line.split()
        ids.append(int(parts[0]))
        xyz.append([float(p) for p in parts[1:4]])
        rgb.append([int(p) for p in parts[4:7]])
        err.append(float(parts[7]))
    return Points3D(
        np.array(ids, dtype=np.int64).reshape(-1),
        np.array(xyz, dtype=np.float64).reshape(-1, 3),
        np.array(rgb, dtype=np.uint8).reshape(-1, 3),
        np.array(err, dtype=np.float64).reshape(-1),
    )


# --------------------------------------------------------------------------- #


def find_sparse_dir(source: Path) -> Path:
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


def read_model(source) -> SparseModel:
    """Read a sparse model, preferring binary like COLMAP itself does."""
    path = find_sparse_dir(Path(source))
    if (path / "cameras.bin").exists():
        cameras = read_cameras_binary(path / "cameras.bin")
        images = read_images_binary(path / "images.bin")
        points = read_points3d_binary(path / "points3D.bin")
    else:
        cameras = read_cameras_text(path / "cameras.txt")
        images = read_images_text(path / "images.txt")
        points = read_points3d_text(path / "points3D.txt")
    return SparseModel(cameras, images, points, path)
