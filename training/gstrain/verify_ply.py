"""Independent .ply reader that mirrors src/ply/StandardSplatDecoder.cpp.

Used to prove an exported file decodes to the values we meant, without going
through Maya. Deliberately does NOT share code with export_ply.py.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

C0 = 0.28209479177387814

_PLY_TO_NUMPY = {
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
}


@dataclass
class DecodedSplats:
    xyz: np.ndarray
    color: np.ndarray  # (N,3) 0..1
    opacity: np.ndarray  # (N,) 0..1
    scale: np.ndarray  # (N,3) linear
    rotation: np.ndarray  # (N,4) normalised [w,x,y,z]
    property_names: list[str]

    def __len__(self) -> int:
        return int(self.xyz.shape[0])


def read_raw(path) -> tuple[dict[str, np.ndarray], list[str]]:
    """Read the `vertex` element of a binary little-endian / ascii ply."""
    path = Path(path)
    with open(path, "rb") as fid:
        if fid.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a ply file")
        fmt = None
        elements: list[tuple[str, int, list[tuple[str, str]]]] = []
        while True:
            line = fid.readline()
            if not line:
                raise ValueError("unexpected end of header")
            tokens = line.decode("ascii", "replace").strip().split()
            if not tokens or tokens[0] == "comment":
                continue
            if tokens[0] == "format":
                fmt = tokens[1]
            elif tokens[0] == "element":
                elements.append((tokens[1], int(tokens[2]), []))
            elif tokens[0] == "property":
                if tokens[1] == "list":
                    raise ValueError("list properties are not supported")
                elements[-1][2].append((tokens[1], tokens[2]))
            elif tokens[0] == "end_header":
                break

        if fmt == "binary_big_endian":
            raise ValueError("big endian ply is rejected by the Maya loader too")
        if fmt not in ("binary_little_endian", "ascii"):
            raise ValueError(f"unsupported ply format {fmt!r}")

        data: dict[str, np.ndarray] = {}
        names: list[str] = []
        for name, count, props in elements:
            dtype = np.dtype([(p_name, "<" + _PLY_TO_NUMPY[p_type]) for p_type, p_name in props])
            if fmt == "ascii":
                rows = [fid.readline().split() for _ in range(count)]
                block = np.array([tuple(row) for row in rows], dtype=dtype) if count else np.zeros(0, dtype)
            else:
                block = np.fromfile(fid, dtype=dtype, count=count)
            if block.shape[0] != count:
                raise ValueError(f"element {name}: expected {count} rows, read {block.shape[0]}")
            if name == "vertex":
                names = [p_name for _, p_name in props]
                data = {p_name: block[p_name].astype(np.float64) for p_name in names}
        if not data:
            raise ValueError("no `vertex` element found")
        return data, names


def decode(path) -> DecodedSplats:
    """Apply the same activations the C++ decoder applies."""
    raw, names = read_raw(path)
    if "x" not in raw:
        if "packed_position" in raw:
            raise ValueError(
                f"{Path(path)} is a SuperSplat compressed .ply; this verifier only handles the "
                "standard 3DGS layout (see src/ply/SuperSplatDecoder.cpp for that format)"
            )
        raise ValueError(f"{Path(path)} has no x/y/z properties")
    n = raw["x"].shape[0]

    xyz = np.stack([raw["x"], raw["y"], raw["z"]], axis=1)

    if "f_dc_0" in raw:
        dc = np.stack([raw[f"f_dc_{i}"] for i in range(3)], axis=1)
        color = np.clip(0.5 + C0 * dc, 0.0, 1.0)
    else:
        color = np.stack([raw[c] for c in ("red", "green", "blue")], axis=1) / 255.0
        color = np.clip(color, 0.0, 1.0)

    opacity = np.clip(1.0 / (1.0 + np.exp(-raw["opacity"])), 0.0, 1.0) if "opacity" in raw else np.ones(n)

    if "scale_0" in raw:
        scale = np.stack([raw[f"scale_{i}"] for i in range(3)], axis=1)
    else:
        scale = np.zeros((n, 3))
    scale = np.clip(np.exp(scale), 1e-4, 10.0)

    if "rot_0" in raw:
        rot = np.stack([raw[f"rot_{i}"] for i in range(4)], axis=1)
        norm = np.linalg.norm(rot, axis=1, keepdims=True)
        rot = np.where(norm > 1e-8, rot / np.maximum(norm, 1e-8), np.array([1.0, 0.0, 0.0, 0.0]))
    else:
        rot = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))

    return DecodedSplats(xyz, color, opacity, scale, rot, names)


def summarize(path) -> str:
    splats = decode(path)
    lines = [
        f"file          : {Path(path)}",
        f"splats        : {len(splats)}",
        f"properties    : {len(splats.property_names)} ({', '.join(splats.property_names[:9])}, ...)",
        f"bbox min      : {np.array2string(splats.xyz.min(axis=0), precision=4)}",
        f"bbox max      : {np.array2string(splats.xyz.max(axis=0), precision=4)}",
        f"opacity       : min {splats.opacity.min():.4f} mean {splats.opacity.mean():.4f} max {splats.opacity.max():.4f}",
        f"scale (linear): min {splats.scale.min():.6f} mean {splats.scale.mean():.6f} max {splats.scale.max():.6f}",
        f"colour        : mean {np.array2string(splats.color.mean(axis=0), precision=4)}",
        f"quat norm err : {np.abs(np.linalg.norm(splats.rotation, axis=1) - 1.0).max():.3e}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Decode a 3DGS .ply the way the Maya plugin does.")
    parser.add_argument("ply", type=Path)
    args = parser.parse_args(argv)
    print(summarize(args.ply))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
