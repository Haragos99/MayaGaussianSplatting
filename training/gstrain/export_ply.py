from pathlib import Path

import numpy as np
import torch

from .model_torch import TorchGaussianModel

# Zeroth-order spherical harmonic, the constant basis function.
SH_C0 = 0.28209479177387814


def rgb_to_sh_dc(color: np.ndarray) -> np.ndarray:
    """Inverse of the decoder's 0.5 + C0 * f_dc."""
    return (color - 0.5) / SH_C0


def sh_dc_to_rgb(f_dc: np.ndarray) -> np.ndarray:
    return 0.5 + SH_C0 * f_dc


def attribute_names() -> list[str]:
    names = ["x", "y", "z", "nx", "ny", "nz"]
    names += [f"f_dc_{i}" for i in range(3)]
    names += ["opacity"]
    names += [f"scale_{i}" for i in range(3)]
    names += [f"rot_{i}" for i in range(4)]

    return names


def gaussian_attributes(model: TorchGaussianModel) -> np.ndarray:
    """-> (N, 14) float32 in the decoder's expected order."""
    with torch.no_grad():
        xyz = model.xyz.detach().cpu().numpy()
        f_dc = rgb_to_sh_dc(model.color.detach().cpu().numpy())
        opacity = model.opacity_raw.detach().cpu().numpy().reshape(-1, 1)
        scale = model.scale_raw.detach().cpu().numpy()

        rotation = model.rotation.detach()
        rotation = rotation / rotation.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        rotation = rotation.cpu().numpy()

    normals = np.zeros_like(xyz)

    return np.concatenate([xyz, normals, f_dc, opacity, scale, rotation], axis=1).astype(np.float32)


def write_ply(path: str | Path, model: TorchGaussianModel) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = gaussian_attributes(model)

    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(data)}"]
    header += [f"property float {name}" for name in attribute_names()]
    header += ["end_header", ""]

    with path.open("wb") as handle:
        handle.write("\n".join(header).encode("ascii"))
        handle.write(data.tobytes())

    return path


def read_ply(path: str | Path) -> dict[str, np.ndarray]:
    """Minimal reader for verifying what write_ply produced."""
    path = Path(path)
    raw = path.read_bytes()

    marker = b"end_header\n"
    end = raw.index(marker) + len(marker)
    header = raw[:end].decode("ascii").splitlines()

    count = next(int(line.split()[-1]) for line in header if line.startswith("element vertex"))
    names = [line.split()[-1] for line in header if line.startswith("property")]

    values = np.frombuffer(raw[end:], dtype="<f4", count=count * len(names))
    values = values.reshape(count, len(names))

    return {name: values[:, i] for i, name in enumerate(names)}


def verify_ply(path: str | Path) -> dict[str, object]:
    """Decode a written file the way the Maya loader would."""
    fields = read_ply(path)

    missing = [name for name in attribute_names() if name not in fields]
    if missing:
        raise ValueError(f"{path} is missing {missing}")

    f_dc = np.stack([fields[f"f_dc_{i}"] for i in range(3)], axis=1)
    scale_raw = np.stack([fields[f"scale_{i}"] for i in range(3)], axis=1)
    quaternion = np.stack([fields[f"rot_{i}"] for i in range(4)], axis=1)

    return {
        "count": len(fields["x"]),
        "xyz": np.stack([fields["x"], fields["y"], fields["z"]], axis=1),
        "color": np.clip(sh_dc_to_rgb(f_dc), 0.0, 1.0),
        "opacity": 1.0 / (1.0 + np.exp(-fields["opacity"])),
        "scale": np.exp(scale_raw),
        "rotation": quaternion,
    }
