import numpy as np
from .gstrain.cameras import create_training_cameras
from .gstrain.colmap import load_colmap
from .gstrain.utility import project_points, show_rendered_image, plot_points3D, plot_scene, plot_projection
from .gstrain.model import initialize_gaussians
from .gstrain.render.renderer_numpy import renderGS
from gstrain.model_torch import TorchGaussianModel
from gstrain.torch_utils import choose_device, describe_device




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

    print(len(pixels))

    #plot_points3D(colmapData.points3D)
    #plot_scene(colmapData.points3D,training_cameras)
    #plot_projection(train_camera, colmapData.points3D)

    gaussians = initialize_gaussians(colmapData.points3D)
    rendered = renderGS(train_camera,gaussians)
    print(rendered.shape)

    show_rendered_image(rendered)


