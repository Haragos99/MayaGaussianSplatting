import numpy as np
from .gstrain.cameras import create_training_cameras
from .gstrain.colmap import load_colmap
from .gstrain.utility import project_points, plot_points3D


if __name__ == "__main__": 
    colmapData = load_colmap(r"C:\Users\Geri\Documents\Projects\CG\MeSP\sparse\0")

    print(len(colmapData.images))
    training_cameras = create_training_cameras(colmapData)
    print(len(training_cameras))
    train_camera = training_cameras[1]

    pixels, depth = project_points(train_camera,colmapData.points3D)

    print(len(pixels))


    plot_points3D(colmapData.points3D)


