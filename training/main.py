import numpy as np
from .gstrain.cameras import create_training_cameras
from .gstrain.colmap import load_colmap
from .gstrain.utility import project_points, show_rendered_image, plot_points3D, plot_scene, plot_projection
from .gstrain.model import initialize_gaussians
from .gstrain.render.render import renderGS

if __name__ == "__main__": 
    colmapData = load_colmap(r"C:\Users\Geri\Documents\Projects\CG\MeSP\sparse\0")

    print(len(colmapData.images))
    training_cameras = create_training_cameras(colmapData)
    print(len(training_cameras))
    train_camera = training_cameras[1]

    pixels, depth = project_points(train_camera,colmapData.points3D)

    print(len(pixels))

    #plot_points3D(colmapData.points3D)
    #plot_scene(colmapData.points3D,training_cameras)
    #plot_projection(train_camera, colmapData.points3D)

    gaussians = initialize_gaussians(colmapData.points3D)
    rendered = renderGS(train_camera,gaussians)
    print(rendered.shape)

    show_rendered_image(rendered)


