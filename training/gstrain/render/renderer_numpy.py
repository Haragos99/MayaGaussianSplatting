import numpy as np

from ..colmap import quaternion_to_rotmatrix
from ..dataset import ProjectedGaussian
from ..model import GaussianModel
from ..cameras import TrainingCamera


def covariance_from_scale_rotation(scale: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    R = quaternion_to_rotmatrix(rotation)
    S = np.diag(scale)
    covariance = (
        R
        @ S
        @ S.T
        @ R.T
    )

    return covariance

def project_gaussian(
    camera: TrainingCamera,
    xyz: np.ndarray,
    covariance: np.ndarray,
    color: np.ndarray,
    opacity: float,
) -> ProjectedGaussian | None:
    # World -> camera
    point_camera = (camera.R @ xyz + camera.t)

    X, Y, Z = point_camera

    if Z <= 0:
        return None

    # Project mean
    u = (camera.fx * X / Z + camera.cx)
    v = (camera.fy * Y / Z + camera.cy)

    mean = np.array( [u, v],dtype=np.float64)

    # World covariance -> camera covariance
    covariance_camera = (
        camera.R
        @ covariance
        @ camera.R.T
    )

    # Perspective Jacobian
    J = np.array([
        [
            camera.fx / Z,
            0.0,
            -camera.fx * X / (Z * Z),
        ],
        [
            0.0,
            camera.fy / Z,
            -camera.fy * Y / (Z * Z),
        ],
    ])


    # Camera covariance -> image covariance
    covariance_2d = (
        J
        @ covariance_camera
        @ J.T
    )

    # Numerical protection.
    covariance_2d += (np.eye(2)* 1e-6)

    return ProjectedGaussian(
        mean=mean,
        covariance=covariance_2d,
        color=color,
        opacity=float(opacity),
        depth=float(Z),
    )





def project_gaussians(camera: TrainingCamera, gaussians: GaussianModel) -> list[ProjectedGaussian]:
    projected = []

    for i in range(len(gaussians.xyz)):

        covariance = (
            covariance_from_scale_rotation(
                gaussians.scale[i],
                gaussians.rotation[i],
            )
        )

        gaussian = project_gaussian(
            camera=camera,
            xyz=gaussians.xyz[i],
            covariance=covariance,
            color=gaussians.color[i],
            opacity=gaussians.opacity[i],
        )

        if gaussian is None:
            continue

        projected.append(gaussian)

    return projected





def gaussian_radius(covariance: np.ndarray) -> np.ndarray:
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-8)

    sigma = np.sqrt(eigenvalues)

    radius = (3.0 * sigma)

    return radius





def compute_splat_bounds(gaussian: ProjectedGaussian, width: int, height: int, sigma_factor: float = 3.0) -> tuple[int, int, int, int] | None:
    u, v = gaussian.mean

    sigma_x = np.sqrt(max(gaussian.covariance[0, 0], 1e-8,))
    sigma_y = np.sqrt(max(gaussian.covariance[1, 1],1e-8,))

    radius_x = (sigma_factor * sigma_x)
    radius_y = (sigma_factor * sigma_y)

    xmin = max(0, int(np.floor(u - radius_x)))
    xmax = min(width - 1,int(np.ceil(u + radius_x)))
    ymin = max(0, int(np.floor(v - radius_y)))
    ymax = min(height - 1, int(np.ceil(v + radius_y)))

    if xmin > xmax or ymin > ymax:
        return None

    return (xmin, xmax, ymin, ymax)



def evaluate_gaussian(
    gaussian: ProjectedGaussian,
    xx: np.ndarray,
    yy: np.ndarray,
) -> np.ndarray:

    pixels = np.stack(
        [
            xx,
            yy,
        ],
        axis=-1,
    )

    delta = (pixels - gaussian.mean)
    covariance_inv = np.linalg.inv(gaussian.covariance)
    exponent = -0.5 * np.einsum(
        "...i,ij,...j->...",
        delta,
        covariance_inv,
        delta,
    )

    return np.exp(exponent)



def rasterize_gaussian(gaussian: ProjectedGaussian, image: np.ndarray, transmittance: np.ndarray) -> None:

    height, width, _ = image.shape
    bounds = compute_splat_bounds(gaussian, width, height)

    if bounds is None:
        return

    xmin, xmax, ymin, ymax = bounds


    # Pixel coordinates
    ys = np.arange(ymin,ymax + 1)
    xs = np.arange(xmin, xmax + 1)
    xx, yy = np.meshgrid(xs, ys)


    # Evaluate Gaussian
    gaussian_value = evaluate_gaussian(gaussian,xx, yy)


    # Alpha
    alpha = (gaussian.opacity * gaussian_value)

    alpha = np.clip(alpha,0.0, 0.99)


    # Existing transmittance
    T = transmittance[
        ymin:ymax + 1,
        xmin:xmax + 1,
    ]


    # Contribution
    weight = (T * alpha)


    # Color accumulation
    image[ymin:ymax + 1, xmin:xmax + 1] += (weight[..., None]* gaussian.color)


    # Update transmittance
    transmittance[ymin:ymax + 1, xmin:xmax + 1,] *= (1.0 - alpha)





def renderGS(
    camera: TrainingCamera,
    gaussians: GaussianModel,
) -> np.ndarray:

    height = camera.height
    width = camera.width

    # Project Gaussians
    projected = project_gaussians(camera,gaussians)

    # Depth sorting
    projected.sort(key=lambda gaussian: gaussian.depth)

    # Output buffers
    image = np.zeros((height, width, 3),dtype=np.float32,)

    transmittance = np.ones((height, width),dtype=np.float32,)

    # Rasterize
    for gaussian in projected:
        rasterize_gaussian(gaussian,image,transmittance)


    # Final image
    return np.clip(image,0.0,1.0,)