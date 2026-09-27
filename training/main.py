import numpy as np
from .gstrain.cameras import create_training_cameras
from .gstrain.colmap import load_colmap
from .gstrain.utility import project_points


if __name__ == "__main__": 
    colmapData = load_colmap(r"C:\Users\Geri\Documents\Projects\CG\MeSP\sparse\0")

    print(len(colmapData.images))
    trianing_cameras = create_training_cameras(colmapData)
    print(len(trianing_cameras))
    points = np.array([
        [0.0, 0.0, 5.0],
        [1.0, 0.0, 5.0],
        [0.0, 1.0, 5.0],
    ])


    pixels, depth = project_points(trianing_cameras[10],points)

    print(pixels)
    print("Test")
    print(depth)
