import numpy as np, torch
from .gstrain.cameras import create_training_cameras
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
import time
print(describe_device())







if __name__ == "__main__": 
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


