import numpy as np

Distortion = tuple[float, float, float, float]  # k1, k2, p1, p2 (COLMAP OpenCV model)


def is_identity(distortion: Distortion) -> bool:
    return all(value == 0.0 for value in distortion)


def distort(x: np.ndarray, y: np.ndarray, distortion: Distortion) -> tuple[np.ndarray, np.ndarray]:
    """Ideal normalised image coords -> distorted normalised coords."""
    k1, k2, p1, p2 = distortion
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return xd, yd


def undistort_image(
    image: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    distortion: Distortion,
) -> np.ndarray:
    """(H, W, C) float image -> its pinhole rectification, same size.

    Intrinsics are those of `image` (already resized), in the COLMAP convention
    where pixel i has its centre at i + 0.5. Distortion is defined on
    normalised coordinates, so it is the same at every resolution.
    """
    if is_identity(distortion):
        return image

    from scipy.ndimage import map_coordinates

    height, width = image.shape[:2]
    u, v = np.meshgrid(
        np.arange(width, dtype=np.float64) + 0.5,
        np.arange(height, dtype=np.float64) + 0.5,
    )
    xd, yd = distort((u - cx) / fx, (v - cy) / fy, distortion)

    # Back to array indices (centre of pixel i is index i).
    rows = fy * yd + cy - 0.5
    cols = fx * xd + cx - 0.5

    out = np.empty_like(image)
    for channel in range(image.shape[2]):
        # 'nearest' repeats the border instead of painting black corners that
        # the splats would then try to reproduce.
        out[..., channel] = map_coordinates(
            image[..., channel], [rows, cols], order=1, mode="nearest"
        )
    return out
