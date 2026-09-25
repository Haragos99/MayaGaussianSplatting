from .gstrain.colmap import load_colmap


if __name__ == "__main__": 
    colmapData = load_colmap(r"C:\Users\Geri\Documents\Projects\CG\MeSP\sparse\0")

    print(len(colmapData.images))