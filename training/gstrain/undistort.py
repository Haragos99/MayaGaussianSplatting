"""Lens undistortion, so COLMAP models that are not already pinhole can be trained.

The rasterizer only knows about pinhole cameras. For each output pixel we take the
ideal ray, push it through the distortion model and sample the source image there,
which is the same forward mapping `colmap image_undistorter` performs.
"""

from __future__ import annotations

import numpy as np

from .colmap import Distortion


def undistort_image(
    image: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    distortion: Distortion,
) -> np.ndarray:
    """Rectify an (H,W,C) float image in place of its distorted original."""
    if distortion.is_identity:
        return image

    from scipy.ndimage import map_coordinates

    height, width = image.shape[:2]
    u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    xd, yd = distortion.apply((u - cx) / fx, (v - cy) / fy)
    coords = np.stack([fy * yd + cy, fx * xd + cx])

    out = np.empty_like(image)
    for channel in range(image.shape[2]):
        out[..., channel] = map_coordinates(
            image[..., channel], coords, order=1, mode="constant", cval=0.0
        )
    return out
