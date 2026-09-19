"""Camera math shared by the dataset and the rasterizer (numpy only).

Matrix convention: every 4x4 returned here is ROW-VECTOR, i.e. `p_row @ M`,
matching Maya and src/splatCalculator.cpp rather than the usual `M @ p_col`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import colmap


@dataclass
class CameraInfo:
    uid: int
    name: str
    image_path: Path
    width: int
    height: int
    fov_x: float
    fov_y: float
    R: np.ndarray  # (3,3) camera->world rotation (= R_w2c.T)
    T: np.ndarray  # (3,) world->camera translation (COLMAP tvec)
    is_test: bool = False
    cx: float | None = None
    cy: float | None = None
    distortion: colmap.Distortion = field(default_factory=colmap.Distortion)

    @property
    def camera_center(self) -> np.ndarray:
        return -self.R @ self.T

    @property
    def focal_x(self) -> float:
        return fov_to_focal(self.fov_x, self.width)

    @property
    def focal_y(self) -> float:
        return fov_to_focal(self.fov_y, self.height)

    @property
    def principal_x(self) -> float:
        return self.width * 0.5 if self.cx is None else self.cx

    @property
    def principal_y(self) -> float:
        return self.height * 0.5 if self.cy is None else self.cy

    def scaled(self, factor: int) -> "CameraInfo":
        """Downscale the image plane; FoV is resolution independent."""
        if factor == 1:
            return self
        width = max(1, self.width // factor)
        height = max(1, self.height // factor)
        return CameraInfo(
            self.uid,
            self.name,
            self.image_path,
            width,
            height,
            self.fov_x,
            self.fov_y,
            self.R,
            self.T,
            self.is_test,
            self.principal_x * width / self.width,
            self.principal_y * height / self.height,
            self.distortion,
        )


def focal_to_fov(focal: float, pixels: int) -> float:
    return 2.0 * np.arctan(pixels / (2.0 * focal))


def fov_to_focal(fov: float, pixels: int) -> float:
    return pixels / (2.0 * np.tan(fov * 0.5))


def world_view_transform(R: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Row-vector world->camera matrix. R is camera->world, T is the COLMAP tvec."""
    m = np.zeros((4, 4), dtype=np.float32)
    m[:3, :3] = R  # row-vector form of R_w2c is R_w2c.T, which is exactly R
    m[3, :3] = T
    m[3, 3] = 1.0
    return m


def projection_matrix(fov_x: float, fov_y: float, znear: float = 0.01, zfar: float = 100.0) -> np.ndarray:
    """Row-vector perspective projection with z mapped to [0, 1]."""
    top = np.tan(fov_y * 0.5) * znear
    right = np.tan(fov_x * 0.5) * znear
    p = np.zeros((4, 4), dtype=np.float32)
    p[0, 0] = znear / right
    p[1, 1] = znear / top
    p[2, 2] = zfar / (zfar - znear)
    p[2, 3] = 1.0
    p[3, 2] = -(zfar * znear) / (zfar - znear)
    return p


def scene_normalization(cameras: list[CameraInfo]) -> tuple[np.ndarray, float]:
    """(centre, radius) of the camera rig; radius is the scene extent used by
    the position learning rate and by densification thresholds."""
    centers = np.stack([c.camera_center for c in cameras], axis=0)
    center = centers.mean(axis=0)
    radius = float(np.linalg.norm(centers - center, axis=1).max()) * 1.1
    return center, max(radius, 1e-6)


def cameras_from_model(
    model: colmap.SparseModel,
    images_dir: Path,
    eval_hold: int = 8,
) -> list[CameraInfo]:
    """Build CameraInfos from a COLMAP model, sorted by image name.

    `eval_hold` > 0 marks every Nth image as a held-out test view.
    """
    infos: list[CameraInfo] = []
    for uid, image in enumerate(sorted(model.images.values(), key=lambda im: im.name)):
        cam = model.cameras[image.camera_id]
        fx, fy, cx, cy = cam.intrinsics()
        infos.append(
            CameraInfo(
                uid=uid,
                name=image.name,
                image_path=images_dir / image.name,
                width=cam.width,
                height=cam.height,
                fov_x=focal_to_fov(fx, cam.width),
                fov_y=focal_to_fov(fy, cam.height),
                R=image.R_w2c.T,
                T=image.tvec.copy(),
                is_test=eval_hold > 0 and uid % eval_hold == 0,
                cx=cx,
                cy=cy,
                distortion=cam.distortion(),
            )
        )
    return infos
