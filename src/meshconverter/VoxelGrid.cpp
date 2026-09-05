#include "VoxelGrid.h"

#include <algorithm>

namespace GS::Mesh
{
    VoxelGrid::VoxelGrid(int sizeX, int sizeY, int sizeZ,
                         const MFloatVector& origin,
                         const MFloatVector& cellSize)
        : m_origin(origin)
        , m_cellSize(cellSize)
    {
        m_size[0] = std::max(sizeX, 0);
        m_size[1] = std::max(sizeY, 0);
        m_size[2] = std::max(sizeZ, 0);

        m_values.assign(
            static_cast<size_t>(m_size[0]) * m_size[1] * m_size[2], 0.0f);
    }

    bool VoxelGrid::isEmpty() const
    {
        return m_size[0] < 2 || m_size[1] < 2 || m_size[2] < 2;
    }

    size_t VoxelGrid::index(int x, int y, int z) const
    {
        return (static_cast<size_t>(z) * m_size[1] + y) * m_size[0] + x;
    }

    int VoxelGrid::clampAxis(int value, int axis) const
    {
        return std::clamp(value, 0, m_size[axis] - 1);
    }

    MFloatVector VoxelGrid::positionAt(int x, int y, int z) const
    {
        return MFloatVector(m_origin.x + x * m_cellSize.x,
                            m_origin.y + y * m_cellSize.y,
                            m_origin.z + z * m_cellSize.z);
    }

    MFloatVector VoxelGrid::gradientAt(int x, int y, int z) const
    {
        const float dx = valueAt(clampAxis(x + 1, 0), y, z) - valueAt(clampAxis(x - 1, 0), y, z);
        const float dy = valueAt(x, clampAxis(y + 1, 1), z) - valueAt(x, clampAxis(y - 1, 1), z);
        const float dz = valueAt(x, y, clampAxis(z + 1, 2)) - valueAt(x, y, clampAxis(z - 1, 2));

        return MFloatVector(dx / (2.0f * m_cellSize.x),
                            dy / (2.0f * m_cellSize.y),
                            dz / (2.0f * m_cellSize.z));
    }
}
