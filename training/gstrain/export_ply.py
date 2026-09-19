"""Write 3DGS .ply files that src/ply/StandardSplatDecoder.cpp can read.

Contract (verified against the C++ loader):
    format binary_little_endian 1.0   (big endian is rejected by PlyHeaderParser)
    element vertex N
    float x y z nx ny nz f_dc_0..2 f_rest_0..M opacity scale_0..2 rot_0..3
All values are RAW / pre-activation; the loader applies exp, sigmoid and the SH
DC constant itself. rot_0 is the quaternion W component.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .params import GaussianParams

#: 180 degrees about X: turns a COLMAP-style Y-down world into Maya's Y-up.
_MAYA_Y_ROT = np.diag([1.0, -1.0, -1.0]).astype(np.float64)


def attribute_names(num_rest: int) -> list[str]:
    names = ["x", "y", "z", "nx", "ny", "nz"]
    names += [f"f_dc_{i}" for i in range(3)]
    names += [f"f_rest_{i}" for i in range(num_rest)]
    names += ["opacity"]
    names += [f"scale_{i}" for i in range(3)]
    names += [f"rot_{i}" for i in range(4)]
    return names


def flatten_features_rest(features_rest: np.ndarray) -> np.ndarray:
    """(N,K-1,3) -> (N,3*(K-1)) in channel-major order: all R, then all G, then all B."""
    return np.ascontiguousarray(features_rest.transpose(0, 2, 1).reshape(features_rest.shape[0], -1))


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of [w,x,y,z] quaternions, broadcasting over the leading axis."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def rotate_world(params: GaussianParams, rotation: np.ndarray) -> GaussianParams:
    """Apply a world-space rotation to positions and orientations (scales are invariant)."""
    from .colmap import rotmat_to_qvec

    q_world = rotmat_to_qvec(rotation).astype(np.float32)
    return GaussianParams(
        xyz=(params.xyz.astype(np.float64) @ rotation.T).astype(np.float32),
        features_dc=params.features_dc,
        features_rest=params.features_rest,
        opacity=params.opacity,
        scaling=params.scaling,
        rotation=quat_multiply(np.broadcast_to(q_world, params.rotation.shape), params.rotation).astype(
            np.float32
        ),
    )


def normalize_quaternions(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    out = np.where(norm > 1e-8, q / np.maximum(norm, 1e-8), np.array([1.0, 0.0, 0.0, 0.0], dtype=q.dtype))
    return out.astype(np.float32)


def write_ply(path, params: GaussianParams, up_axis: str = "colmap") -> Path:
    """Write `params` as a binary little-endian 3DGS .ply. Returns the path."""
    if up_axis not in ("colmap", "maya-y"):
        raise ValueError(f"unknown up_axis {up_axis!r}, expected 'colmap' or 'maya-y'")
    if up_axis == "maya-y":
        params = rotate_world(params, _MAYA_Y_ROT)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = len(params)
    rest = flatten_features_rest(params.features_rest)
    columns = [
        params.xyz.astype(np.float32),
        np.zeros((n, 3), dtype=np.float32),  # normals are unused by 3DGS but part of the layout
        params.features_dc[:, 0, :].astype(np.float32),
        rest.astype(np.float32),
        params.opacity.astype(np.float32),
        params.scaling.astype(np.float32),
        normalize_quaternions(params.rotation.astype(np.float32)),
    ]
    data = np.concatenate(columns, axis=1)

    names = attribute_names(rest.shape[1])
    if data.shape[1] != len(names):
        raise AssertionError(f"packed {data.shape[1]} floats but declared {len(names)} properties")

    header = "ply\nformat binary_little_endian 1.0\n"
    header += f"element vertex {n}\n"
    header += "".join(f"property float {name}\n" for name in names)
    header += "end_header\n"

    with open(path, "wb") as fid:
        fid.write(header.encode("ascii"))
        np.ascontiguousarray(data, dtype="<f4").tofile(fid)
    return path
