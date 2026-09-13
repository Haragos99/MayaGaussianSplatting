#pragma once

#include "MeshData.h"
#include "VoxelGrid.h"

#include <vector>

namespace GS::Mesh
{
    // Walks every cell of a voxel grid, looks up which of its twelve edges the
    // iso surface crosses and emits the triangles for that configuration.
    // Depends on VoxelGrid alone, so it meshes any volume.
    class MarchingCubes
    {
    public:
        MeshData generate(const VoxelGrid& grid, float isoLevel) const;

    private:
        // Index of the shared vertex on `edge` of cell (x, y, z). `cache` holds
        // only the two grid slices the current cell slice can reach.
        static int edgeVertex(const VoxelGrid& grid, float isoLevel,
                              int x, int y, int z, int edge,
                              std::vector<int>& cache, MeshData& out);
    };
}
