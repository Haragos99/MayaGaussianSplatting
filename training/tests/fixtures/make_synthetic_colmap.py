"""Generate a tiny synthetic COLMAP model so tests need no real dataset."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from gstrain.colmap import CAMERA_MODEL_IDS, qvec_to_rotmat, rotmat_to_qvec


def look_at(eye: np.ndarray, target: np.ndarray, up=(0.0, -1.0, 0.0)):
    """COLMAP-style pose: X right, Y down, Z forward. Returns (qvec_wxyz, tvec)."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, dtype=np.float64))
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    R_w2c = np.stack([right, down, forward], axis=0)
    tvec = -R_w2c @ eye
    return rotmat_to_qvec(R_w2c), tvec


def make_scene(num_views: int = 6, num_points: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    xyz = rng.normal(scale=0.3, size=(num_points, 3))
    rgb = rng.integers(0, 256, size=(num_points, 3), dtype=np.uint8)

    poses = []
    for i in range(num_views):
        angle = 2.0 * np.pi * i / num_views
        eye = np.array([3.0 * np.cos(angle), 0.5, 3.0 * np.sin(angle)])
        poses.append(look_at(eye, np.zeros(3)))
    return xyz, rgb, poses


def write_model(root: Path, binary: bool = True, num_views: int = 6, num_points: int = 40) -> Path:
    """Write sparse/0 (+ empty images/) under `root`. Returns the project root."""
    root = Path(root)
    sparse = root / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    (root / "images").mkdir(parents=True, exist_ok=True)

    xyz, rgb, poses = make_scene(num_views, num_points)
    width, height, fx, fy, cx, cy = 64, 48, 50.0, 50.0, 32.0, 24.0
    names = [f"view_{i:03d}.png" for i in range(len(poses))]

    if binary:
        _write_cameras_bin(sparse / "cameras.bin", width, height, fx, fy, cx, cy)
        _write_images_bin(sparse / "images.bin", poses, names)
        _write_points_bin(sparse / "points3D.bin", xyz, rgb)
    else:
        _write_cameras_txt(sparse / "cameras.txt", width, height, fx, fy, cx, cy)
        _write_images_txt(sparse / "images.txt", poses, names)
        _write_points_txt(sparse / "points3D.txt", xyz, rgb)
    return root


def write_images(root: Path, num_views: int = 6, num_points: int = 40) -> Path:
    """Render the same point cloud as soft blobs, independently of our rasterizer."""
    from PIL import Image

    xyz, rgb, poses = make_scene(num_views, num_points)
    width, height, fx, fy, cx, cy = 64, 48, 50.0, 50.0, 32.0, 24.0
    images_dir = Path(root) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    px, py = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    sigma, alpha = 2.0, 0.9

    for i, (qvec, tvec) in enumerate(poses):
        p_view = xyz @ qvec_to_rotmat(qvec).T + tvec
        canvas = np.zeros((height, width, 3), dtype=np.float64)
        transmittance = np.ones((height, width), dtype=np.float64)
        for point in np.argsort(p_view[:, 2]):  # front to back
            z = p_view[point, 2]
            if z <= 0.1:
                continue
            u = fx * p_view[point, 0] / z + cx
            v = fy * p_view[point, 1] / z + cy
            weight = alpha * np.exp(-0.5 * (((px - u) ** 2 + (py - v) ** 2) / sigma**2))
            canvas += (weight * transmittance)[..., None] * (rgb[point] / 255.0)
            transmittance *= 1.0 - weight
        pixels = (np.clip(canvas, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        Image.fromarray(pixels).save(images_dir / f"view_{i:03d}.png")
    return images_dir


def _write_cameras_bin(path, width, height, fx, fy, cx, cy):
    with open(path, "wb") as fid:
        fid.write(struct.pack("<Q", 1))
        fid.write(struct.pack("<iiQQ", 1, CAMERA_MODEL_IDS["PINHOLE"], width, height))
        fid.write(struct.pack("<dddd", fx, fy, cx, cy))


def _write_images_bin(path, poses, names):
    with open(path, "wb") as fid:
        fid.write(struct.pack("<Q", len(poses)))
        for i, (qvec, tvec) in enumerate(poses):
            fid.write(struct.pack("<idddddddi", i + 1, *qvec, *tvec, 1))
            fid.write(names[i].encode("utf-8") + b"\x00")
            fid.write(struct.pack("<Q", 2))
            for j in range(2):
                fid.write(struct.pack("<ddq", float(j), float(j), j))


def _write_points_bin(path, xyz, rgb):
    with open(path, "wb") as fid:
        fid.write(struct.pack("<Q", xyz.shape[0]))
        for i in range(xyz.shape[0]):
            fid.write(struct.pack("<QdddBBBd", i + 1, *xyz[i], *(int(c) for c in rgb[i]), 0.5))
            fid.write(struct.pack("<Q", 2))
            fid.write(struct.pack("<iiii", 1, 0, 2, 1))


def _write_cameras_txt(path, width, height, fx, fy, cx, cy):
    with open(path, "w", encoding="utf-8") as fid:
        fid.write("# Camera list\n")
        fid.write(f"1 PINHOLE {width} {height} {fx!r} {fy!r} {cx!r} {cy!r}\n")


def _write_images_txt(path, poses, names):
    with open(path, "w", encoding="utf-8") as fid:
        fid.write("# Image list\n")
        for i, (qvec, tvec) in enumerate(poses):
            q = " ".join(repr(float(v)) for v in qvec)
            t = " ".join(repr(float(v)) for v in tvec)
            fid.write(f"{i + 1} {q} {t} 1 {names[i]}\n")
            fid.write("0.0 0.0 1 1.0 1.0 2\n")


def _write_points_txt(path, xyz, rgb):
    with open(path, "w", encoding="utf-8") as fid:
        fid.write("# 3D point list\n")
        for i in range(xyz.shape[0]):
            p = " ".join(repr(float(v)) for v in xyz[i])
            c = " ".join(str(int(v)) for v in rgb[i])
            fid.write(f"{i + 1} {p} {c} 0.5 1 0 2 1\n")
