#include "MarchingCubes.h"

#include "MarchingCubesTables.h"

#include <algorithm>
#include <cmath>

namespace GS::Mesh
{
    namespace
    {
        using namespace MarchingCubesTables;

        // Below this the two corner samples count as equal and the crossing is
        // put in the middle of the edge.
        constexpr float kDenominatorEpsilon = 1e-6f;
    }

    MeshData MarchingCubes::generate(const VoxelGrid& grid, float isoLevel) const
    {
        MeshData out;
        if (grid.isEmpty())
            return out;

        // Cells on slice z only ever own edges on grid slices z and z+1, so two
        // slices of vertex slots are enough to weld the surface.
        const size_t sliceSlots =
            static_cast<size_t>(grid.sizeX()) * grid.sizeY() * 3;

        std::vector<int> cache(sliceSlots * 2, -1);

        for (int z = 0; z < grid.sizeZ() - 1; ++z)
        {
            for (int y = 0; y < grid.sizeY() - 1; ++y)
            {
                for (int x = 0; x < grid.sizeX() - 1; ++x)
                {
                    int cubeIndex = 0;
                    for (int corner = 0; corner < 8; ++corner)
                    {
                        const int* offset = kCornerOffset[corner];
                        const float value = grid.valueAt(
                            x + offset[0], y + offset[1], z + offset[2]);

                        if (value < isoLevel)
                            cubeIndex |= 1 << corner;
                    }

                    if (kEdgeTable[cubeIndex] == 0)
                        continue;

                    const int* triangles = kTriangleTable[cubeIndex];
                    for (int i = 0; triangles[i] != -1; i += 3)
                    {
                        const int a = edgeVertex(grid, isoLevel, x, y, z, triangles[i], cache, out);
                        const int b = edgeVertex(grid, isoLevel, x, y, z, triangles[i + 1], cache, out);
                        const int c = edgeVertex(grid, isoLevel, x, y, z, triangles[i + 2], cache, out);

                        // Two corners landing on the same crossing collapse it.
                        if (a == b || b == c || a == c)
                            continue;

                        // Reversed against the table: density is high inside,
                        // the opposite of what the table assumes.
                        out.polygonCounts.append(3);
                        out.polygonConnects.append(a);
                        out.polygonConnects.append(c);
                        out.polygonConnects.append(b);
                    }
                }
            }

            std::copy(cache.begin() + sliceSlots, cache.end(), cache.begin());
            std::fill(cache.begin() + sliceSlots, cache.end(), -1);
        }

        return out;
    }

    int MarchingCubes::edgeVertex(const VoxelGrid& grid, float isoLevel,
                                  int x, int y, int z, int edge,
                                  std::vector<int>& cache, MeshData& out)
    {
        const int* owner = kEdgeOwner[edge];
        const size_t slot =
            ((static_cast<size_t>(owner[2]) * grid.sizeY() + (y + owner[1]))
             * grid.sizeX() + (x + owner[0])) * 3 + owner[3];

        if (cache[slot] >= 0)
            return cache[slot];

        const int* corners = kEdgeCorners[edge];
        const int* offsetA = kCornerOffset[corners[0]];
        const int* offsetB = kCornerOffset[corners[1]];

        const int ax = x + offsetA[0], ay = y + offsetA[1], az = z + offsetA[2];
        const int bx = x + offsetB[0], by = y + offsetB[1], bz = z + offsetB[2];

        const float valueA = grid.valueAt(ax, ay, az);
        const float valueB = grid.valueAt(bx, by, bz);
        const float denominator = valueB - valueA;

        const float t = std::abs(denominator) > kDenominatorEpsilon
            ? std::clamp((isoLevel - valueA) / denominator, 0.0f, 1.0f)
            : 0.5f;

        const MFloatVector positionA = grid.positionAt(ax, ay, az);
        const MFloatVector positionB = grid.positionAt(bx, by, bz);
        const MFloatVector position = positionA + (positionB - positionA) * t;

        const MFloatVector gradientA = grid.gradientAt(ax, ay, az);
        const MFloatVector gradientB = grid.gradientAt(bx, by, bz);
        const MFloatVector gradient = gradientA + (gradientB - gradientA) * t;

        MFloatVector normal(0.0f, 1.0f, 0.0f);
        if (gradient * gradient > kDenominatorEpsilon)
            normal = -gradient.normal();

        const int index = static_cast<int>(out.points.length());
        out.points.append(MFloatPoint(position.x, position.y, position.z));
        out.normals.append(MVector(normal.x, normal.y, normal.z));
        cache[slot] = index;

        return index;
    }
}
