from pathlib import Path

import numpy as np
import torch

from .model_torch import TorchGaussianModel
from .sh import C0 as SH_C0

UP_AXES = ("colmap", "maya-y")

# 180 degrees about X: turns COLMAP's Y-down world into Maya's Y-up.
_MAYA_Y_ROTATION = np.diag([1.0, -1.0, -1.0])
# The same rotation as a [w, x, y, z] quaternion.
_MAYA_Y_QUATERNION = np.array([0.0, 1.0, 0.0, 0.0])


def rgb_to_sh_dc(color: np.ndarray) -> np.ndarray:
    """Inverse of the decoder's 0.5 + C0 * f_dc."""
    return (color - 0.5) / SH_C0


def sh_dc_to_rgb(f_dc: np.ndarray) -> np.ndarray:
    return 0.5 + SH_C0 * f_dc


def attribute_names(num_rest: int = 0) -> list[str]:
    names = ["x", "y", "z", "nx", "ny", "nz"]
    names += [f"f_dc_{i}" for i in range(3)]
    names += [f"f_rest_{i}" for i in range(num_rest)]
    names += ["opacity"]
    names += [f"scale_{i}" for i in range(3)]
    names += [f"rot_{i}" for i in range(4)]

    return names


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of [w, x, y, z] quaternions (broadcasting)."""
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def gaussian_attributes(model: TorchGaussianModel, up_axis: str = "colmap") -> np.ndarray:
    """-> (N, 14 + 3 * rest) float32 in the decoder's expected order."""
    if up_axis not in UP_AXES:
        raise ValueError(f"up_axis must be one of {UP_AXES}, got {up_axis!r}")

    with torch.no_grad():
        xyz = model.xyz.detach().cpu().double().numpy()
        f_dc = model.sh_dc.detach().cpu().numpy()[:, 0, :]
        rest = model.sh_rest.detach().cpu().numpy()
        f_rest = rest.transpose(0, 2, 1).reshape(len(model), -1)
        opacity = model.opacity_raw.detach().cpu().numpy().reshape(-1, 1)
        scale = model.scale_raw.detach().cpu().numpy()

        rotation = model.rotation.detach()
        rotation = rotation / rotation.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        rotation = rotation.cpu().double().numpy()

    if up_axis == "maya-y":
        # Rotating the world rotates positions and orientations; scales and
        # colours are unchanged. (View-dependent SH would strictly need its
        # bands rotated too; band 0 - the colour - is rotation invariant.)
        xyz = xyz @ _MAYA_Y_ROTATION.T
        rotation = _quat_multiply(_MAYA_Y_QUATERNION, rotation)

    normals = np.zeros_like(xyz)
    columns = [xyz, normals, f_dc, f_rest, opacity, scale, rotation]
    return np.concatenate([c.astype(np.float32) for c in columns], axis=1)


def write_ply(path: str | Path, model: TorchGaussianModel, up_axis: str = "colmap") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = gaussian_attributes(model, up_axis=up_axis)
    names = attribute_names(model.sh_rest.shape[1] * 3)
    if data.shape[1] != len(names):
        raise AssertionError(f"packed {data.shape[1]} floats but declared {len(names)} properties")

    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(data)}"]
    header += [f"property float {name}" for name in names]
    header += ["end_header", ""]

    with path.open("wb") as handle:
        handle.write("\n".join(header).encode("ascii"))
        handle.write(np.ascontiguousarray(data, dtype="<f4").tobytes())

    return path


def read_ply(path: str | Path) -> dict[str, np.ndarray]:
    """Minimal reader for verifying what write_ply produced."""
    path = Path(path)
    raw = path.read_bytes()

    marker = b"end_header\n"
    end = raw.index(marker) + len(marker)
    header = raw[:end].decode("ascii").splitlines()

    if "format binary_little_endian 1.0" not in header:
        raise ValueError(f"{path} is not a binary little-endian .ply")

    count = next(int(line.split()[-1]) for line in header if line.startswith("element vertex"))
    names = [line.split()[-1] for line in header if line.startswith("property")]

    values = np.frombuffer(raw[end:], dtype="<f4", count=count * len(names))
    values = values.reshape(count, len(names))

    return {name: values[:, i] for i, name in enumerate(names)}


def verify_ply(path: str | Path) -> dict[str, object]:
    """Decode a written file the way the Maya loader would, and sanity-check it."""
    fields = read_ply(path)

    num_rest = sum(1 for name in fields if name.startswith("f_rest_"))
    missing = [name for name in attribute_names(num_rest) if name not in fields]
    if missing:
        raise ValueError(f"{path} is missing {missing}")
    if num_rest not in (0, 9, 24, 45):
        raise ValueError(f"{path} has {num_rest} f_rest values; expected 0, 9, 24 or 45")

    f_dc = np.stack([fields[f"f_dc_{i}"] for i in range(3)], axis=1)
    scale_raw = np.stack([fields[f"scale_{i}"] for i in range(3)], axis=1)
    quaternion = np.stack([fields[f"rot_{i}"] for i in range(4)], axis=1)
    xyz = np.stack([fields["x"], fields["y"], fields["z"]], axis=1)

    all_values = np.stack(list(fields.values()), axis=1)
    if not np.isfinite(all_values).all():
        raise ValueError(f"{path} contains NaN or infinite values")

    sh_degree = {0: 0, 9: 1, 24: 2, 45: 3}[num_rest]
    return {
        "count": len(fields["x"]),
        "sh_degree": sh_degree,
        "xyz": xyz,
        "color": np.clip(sh_dc_to_rgb(f_dc), 0.0, 1.0),
        "opacity": 1.0 / (1.0 + np.exp(-fields["opacity"])),
        "scale": np.exp(scale_raw),
        "rotation": quaternion,
    }


def load_ply(path: str | Path, device: torch.device | None = None, up_axis: str = "colmap") -> TorchGaussianModel:
    """Read a .ply written by write_ply back into a trainable model.

    `up_axis` is how the file was written; a "maya-y" file is rotated back
    into the COLMAP frame (the rotation is its own inverse). All SH bands in
    the file become active.
    """
    import torch.nn as nn

    if up_axis not in UP_AXES:
        raise ValueError(f"up_axis must be one of {UP_AXES}, got {up_axis!r}")
    device = device or torch.device("cpu")
    fields = read_ply(path)
    count = len(fields["x"])
    rest = sum(1 for name in fields if name.startswith("f_rest_")) // 3
    degree = {0: 0, 3: 1, 8: 2, 15: 3}.get(rest)
    if degree is None:
        raise ValueError(f"{path} has {rest * 3} f_rest values, which is not an SH degree 0-3 layout")

    def stack(names: list[str]) -> np.ndarray:
        return np.stack([fields[name] for name in names], axis=1).astype(np.float64)

    xyz = stack(["x", "y", "z"])
    rotation = stack([f"rot_{i}" for i in range(4)])
    if up_axis == "maya-y":
        xyz = xyz @ _MAYA_Y_ROTATION.T
        rotation = _quat_multiply(_MAYA_Y_QUATERNION, rotation)

    def parameter(array: np.ndarray) -> nn.Parameter:
        return nn.Parameter(torch.as_tensor(np.ascontiguousarray(array), dtype=torch.float32, device=device))

    model = TorchGaussianModel(
        xyz=torch.zeros(count, 3, device=device),
        scale=torch.ones(count, 3, device=device),
        rotation=torch.zeros(count, 4, device=device),
        opacity=torch.full((count,), 0.5, device=device),
        color=torch.zeros(count, 3, device=device),
        sh_degree=degree,
    )
    model.xyz = parameter(xyz)
    model.scale_raw = parameter(stack([f"scale_{i}" for i in range(3)]))
    model.rotation = parameter(rotation)
    model.opacity_raw = parameter(fields["opacity"].astype(np.float64))
    model.sh_dc = parameter(stack([f"f_dc_{i}" for i in range(3)]).reshape(count, 1, 3))
    # f_rest is channel-major on disk: (N, 3, K-1) -> (N, K-1, 3).
    f_rest = stack([f"f_rest_{i}" for i in range(rest * 3)]).reshape(count, 3, rest).transpose(0, 2, 1)
    model.sh_rest = parameter(f_rest)
    model.active_sh_degree = degree
    return model
