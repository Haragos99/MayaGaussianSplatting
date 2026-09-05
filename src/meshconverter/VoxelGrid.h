#pragma once

#include <maya/MFloatVector.h>

#include <vector>

namespace GS::Mesh
{
    // Dense scalar volume on a regular lattice: the only input marching cubes
    // understands.
    class VoxelGrid
    {
    public:
        VoxelGrid() = default;
        VoxelGrid(int sizeX, int sizeY, int sizeZ,
                  const MFloatVector& origin,
                  const MFloatVector& cellSize);

        int sizeX() const { return m_size[0]; }
        int sizeY() const { return m_size[1]; }
        int sizeZ() const { return m_size[2]; }

        const MFloatVector& origin() const { return m_origin; }
        const MFloatVector& cellSize() const { return m_cellSize; }

        bool isEmpty() const;

        float valueAt(int x, int y, int z) const { return m_values[index(x, y, z)]; }
        void setValue(int x, int y, int z, float value) { m_values[index(x, y, z)] = value; }

        MFloatVector positionAt(int x, int y, int z) const;

        // Central difference, clamped at the borders. Points towards denser
        // samples, so the outward normal is its negation.
        MFloatVector gradientAt(int x, int y, int z) const;

    private:
        size_t index(int x, int y, int z) const;
        int clampAxis(int value, int axis) const;

        int m_size[3] = { 0, 0, 0 };
        MFloatVector m_origin{ 0.0f, 0.0f, 0.0f };
        MFloatVector m_cellSize{ 1.0f, 1.0f, 1.0f };
        std::vector<float> m_values;
    };
}
