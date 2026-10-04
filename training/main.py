import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from .gstrain.colmap import load_colmap
from .gstrain.utility import project_points, show_rendered_image, plot_points3D, plot_scene, plot_projection
from .gstrain.model import initialize_gaussians
from .gstrain.render.renderer_numpy import renderGS
from gstrain.model_torch import TorchGaussianModel
from gstrain.torch_utils import choose_device, describe_device
from gstrain.colmap import quaternion_to_rotmatrix
from gstrain.geometry.quaternion import quaternion_to_rotation_matrix
from gstrain.render.covariance import covariance_2d, covariance_3d, covariance_to_conic
from gstrain.geometry.camera import TorchCamera
from gstrain.render.projection import project_points_Torch, world_to_camera
from gstrain.render.binning import build_tile_bins
from gstrain.render.renderer_torch import render_gaussians
from .gstrain.cameras import create_training_cameras
from .gstrain.colmap import Points3D, load_colmap
from .gstrain.dataset import build_views
from .gstrain.export_ply import write_ply
from .gstrain.geometry.camera import TorchCamera
from .gstrain.model import initialize_gaussians
from .gstrain.model_torch import TorchGaussianModel
from .gstrain.preview import save_progress_preview
from .gstrain.torch_utils import choose_device, describe_device
from .gstrain.train import TrainConfig, TrainView, train
import time
import torch

def run_training_test(
    colmap_dir: Path,
    images_dir: Path | None = None,
    iterations: int = 3,
    max_views: int | None = None,
    max_edge: int = 256,
    max_gaussians: int = 5_000,
    output_dir: Path = Path("out/previews"),
    preview_every: int = 100,
    ply_output: Path | None = None,
) -> list[float]:
    """Run a small end-to-end training pass against a COLMAP scene."""
    if (
        iterations < 1
        or (max_views is not None and max_views < 2)
        or max_edge < 1
        or max_gaussians < 1
        or preview_every < 1
    ):
        raise ValueError(
            "iterations, max_edge, max_gaussians, and preview_every must be positive; "
            "max_views must be at least 2"
        )

    data = load_colmap(colmap_dir)
    images_dir = images_dir or data.path.parent.parent / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {images_dir}")
    if len(data.points3D.xyz) == 0:
        raise ValueError("COLMAP scene contains no 3D points to initialize Gaussians")

    device = choose_device()
    point_count = len(data.points3D.xyz)
    if point_count > max_gaussians:
        indices = np.linspace(0, point_count - 1, max_gaussians, dtype=np.int64)
        points = Points3D(
            ids=data.points3D.ids[indices],
            xyz=data.points3D.xyz[indices],
            rgb=data.points3D.rgb[indices],
            error=data.points3D.error[indices],
        )
    else:
        points = data.points3D

    model = TorchGaussianModel.from_numpy(initialize_gaussians(points), device=device)
    views = build_views(
        data,
        images_dir,
        max_edge=max_edge,
        device=device,
        limit=max_views,
    )

    print(f"Training views: {len(views)} / {len(data.images)}")
    if len(views) < 2:
        raise ValueError("training needs at least two COLMAP images")
    first_image = next(iter(data.images.values()))
    with Image.open(images_dir / first_image.name) as original_image:
        original_width, original_height = original_image.size
    resized_width = views[0].camera.width
    resized_height = views[0].camera.height
    print(
        f"First image size: {original_width}x{original_height} -> "
        f"{resized_width}x{resized_height}"
    )

    def save_periodic_preview(step: int, _loss: float) -> None:
        completed_iterations = step + 1
        if completed_iterations % preview_every == 0:
            preview_path = save_progress_preview(
                model, views[0], output_dir, completed_iterations
            )
            print(f"\nSaved preview: {preview_path}")

    losses = train(
        model,
        views,
        config=TrainConfig(iterations=iterations),
        on_iteration=save_periodic_preview,
    )
    ply_output = ply_output or output_dir.parent / "point_cloud.ply"
    ply_path = write_ply(ply_output, model)
    ply_size_bytes = ply_path.stat().st_size

    print(f"Training smoke test completed: {len(losses)} iterations, {len(model)} Gaussians")
    print(f"Loss: {losses[0]:.6f} -> {losses[-1]:.6f}")
    print(f"Exported {len(model)} splats to: {ply_path}")
    print(f"PLY size: {ply_size_bytes:,} bytes ({ply_size_bytes / (1024 ** 2):.2f} MiB)")
    return losses


def main() -> None:
    print(describe_device())
    colmapData = load_colmap(r"C:\Users\Geri\Documents\Projects\CG\MeSP\sparse\0")

    print(len(colmapData.images))
    training_cameras = create_training_cameras(colmapData)
    print(len(training_cameras))
    train_camera = training_cameras[1]

    pixels, depth = project_points(train_camera,colmapData.points3D)


    # Test
    cpu_model = initialize_gaussians(colmapData.points3D)
    gpu_model = TorchGaussianModel.from_numpy(cpu_model, device=choose_device())

    print(gpu_model)                    # num_gaussians=..., device=cuda:0
    print(gpu_model.xyz.device)         # cuda:0
    print(gpu_model.opacity[:5])        # 0.5 recovered through sigmoid
    print(gpu_model.scale_raw[:2])      # log-space values Adam will optimize




    q = np.random.default_rng(0).normal(size=(200_000, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)

    t = time.perf_counter()
    np.stack([quaternion_to_rotmatrix(qi) for qi in q])
    print("numpy loop:", time.perf_counter() - t)

    qg = torch.from_numpy(q).cuda()
    quaternion_to_rotation_matrix(qg); torch.cuda.synchronize()   # warm-up
    t = time.perf_counter()
    R = quaternion_to_rotation_matrix(qg); torch.cuda.synchronize()
    print("torch cuda:", time.perf_counter() - t)




    # A flat, disc-like Gaussian: wide in x/y, thin in z
    scale = torch.tensor([[1.0, 1.0, 0.05]])
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    tilt = torch.tensor([[0.9239, 0.3827, 0.0, 0.0]])   # 45 deg about X

    print(covariance_3d(scale, identity)[0])
    print(covariance_3d(scale, tilt)[0])                # off-diagonal terms appear

    # The extent is unchanged by rotation - only the orientation moved:
    for q in (identity, tilt):
        print(torch.linalg.eigvalsh(covariance_3d(scale, q)[0]))




    cam = TorchCamera.from_training_camera(create_training_cameras(colmapData)[1])

    xyz = torch.as_tensor(colmapData.points3D.xyz, dtype=torch.float32, device=cam.device)
    uv, depth, visible = project_points_Torch(cam, world_to_camera(cam, xyz))

    print("in front of camera:", visible.sum().item(), "/", len(xyz))
    print("depth range:", depth[visible].min().item(), depth[visible].max().item())

    inside = visible & (uv[:, 0] >= 0) & (uv[:, 0] < cam.width) \
                    & (uv[:, 1] >= 0) & (uv[:, 1] < cam.height)
    print("inside the image:", inside.sum().item())


    uv    = torch.tensor([[32.0, 32.0]])
    depth = torch.tensor([1.0])
    vis   = torch.tensor([True])

    for sigma in (0.5, 2.0, 6.0, 20.0):
        cov = torch.tensor([[[sigma**2, 0.0], [0.0, sigma**2]]])
        bins = build_tile_bins(uv, cov, depth, vis, 64, 64, tile_size=16)
        print(f"sigma={sigma:5}  tiles touched: {bins.num_pairs}")


    bins = build_tile_bins(uv, cov, depth, vis, 640, 480, tile_size=16)
    print(bins.num_pairs / len(uv), "pairs per Gaussian")

    print(len(pixels))



    cam = TorchCamera.from_training_camera(create_training_cameras(colmapData)[1])
    model = TorchGaussianModel.from_numpy(initialize_gaussians(colmapData.points3D), device=cam.device)

    with torch.no_grad():
        image = render_gaussians(cam, model)

    print(image.shape, image.device)
    show_rendered_image(image.cpu().numpy())

    #plot_points3D(colmapData.points3D)
    #plot_scene(colmapData.points3D,training_cameras)
    #plot_projection(train_camera, colmapData.points3D)

    #gaussians = initialize_gaussians(colmapData.points3D)
    #rendered = renderGS(train_camera,gaussians)
    #print(rendered.shape)

    #show_rendered_image(rendered)



    # Test Training
    parser = argparse.ArgumentParser(description="Run a short Gaussian-splat training smoke test.")
    parser.add_argument(
        "colmap_dir",
        nargs="?",
        type=Path,
        default=Path(r"C:\Users\Geri\Documents\Projects\CG\MeSP\sparse\0"),
        help="COLMAP model directory (or project root)",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        help="image directory; defaults to the COLMAP project's images folder",
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--max-views", type=int, help="optional cap; defaults to using every COLMAP view")
    parser.add_argument("--max-edge", type=int, default=128)
    parser.add_argument("--max-gaussians", type=int, default=5_000)
    parser.add_argument("--output-dir", type=Path, default=Path("out/previews"))
    parser.add_argument("--preview-every", type=int, default=100)
    parser.add_argument("--ply-output", type=Path, help="PLY output path; defaults beside the preview directory")
    args = parser.parse_args()

    run_training_test(
        args.colmap_dir,
        images_dir=args.images_dir,
        iterations=args.iterations,
        max_views=args.max_views,
        max_edge=args.max_edge,
        max_gaussians=args.max_gaussians,
        output_dir=args.output_dir,
        preview_every=args.preview_every,
        ply_output=args.ply_output,
    )


if __name__ == "__main__":
    main()