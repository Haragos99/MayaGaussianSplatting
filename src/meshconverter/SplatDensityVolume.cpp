#include "SplatDensityVolume.h"

#include <algorithm>
#include <cmath>

namespace GS::Mesh
{
    namespace
    {
        // Splats fainter than one 8 bit alpha step cannot lift a cell over any
        // useful iso level.
        constexpr float kMinOpacity = 1.0f / 255.0f;

        // A Gaussian thinner than the grid would fall between two samples.
        constexpr float kMinDeviationInCells = 0.5f;

        // A single background Gaussian can be as wide as the whole capture.
        // Letting it scatter would cost more than the rest of the cloud and
        // would only smear the surface, so anything wider is dropped.
        constexpr int kMaxCellSpan = 64;

        // Empty rim so splats sitting on the bounding box still close.
        constexpr int kBorderCells = 2;

        // Rotation matrix rows from the {w, x, y, z} quaternion.
        void quaternionAxes(const float q[4], float axes[3][3])
        {
            const float w = q[0], x = q[1], y = q[2], z = q[3];
            const float lengthSquared = w * w + x * x + y * y + z * z;
            const float s = lengthSquared > 0.0f ? 2.0f / lengthSquared : 0.0f;

            const float xx = x * x * s, yy = y * y * s, zz = z * z * s;
            const float xy = x * y * s, xz = x * z * s, yz = y * z * s;
            const float wx = w * x * s, wy = w * y * s, wz = w * z * s;

            axes[0][0] = 1.0f - (yy + zz); axes[0][1] = xy + wz;          axes[0][2] = xz - wy;
            axes[1][0] = xy - wz;          axes[1][1] = 1.0f - (xx + zz); axes[1][2] = yz + wx;
            axes[2][0] = xz + wy;          axes[2][1] = yz - wx;          axes[2][2] = 1.0f - (xx + yy);
        }
    }

    SplatDensityVolume::SplatDensityVolume(const std::vector<GaussianSplat>& splats,
                                           const MBoundingBox& bounds,
                                           float reachInDeviations)
        : m_splats(splats)
        , m_bounds(bounds)
        , m_reachInDeviations(std::max(reachInDeviations, 0.5f))
    {
    }

    VoxelGrid SplatDensityVolume::build(int resolution) const
    {
        resolution = std::max(resolution, 2);

        if (m_splats.empty())
            return VoxelGrid();

        const MPoint minimum = m_bounds.min();
        const MPoint maximum = m_bounds.max();

        const float extent[3] = {
            static_cast<float>(maximum.x - minimum.x),
            static_cast<float>(maximum.y - minimum.y),
            static_cast<float>(maximum.z - minimum.z)
        };

        const float longest = std::max({ extent[0], extent[1], extent[2] });
        if (longest <= 0.0f)
            return VoxelGrid();

        const float step = longest / (resolution - 1);

        const MFloatVector origin(
            static_cast<float>(minimum.x) - kBorderCells * step,
            static_cast<float>(minimum.y) - kBorderCells * step,
            static_cast<float>(minimum.z) - kBorderCells * step);

        const auto samplesAlong = [step](float length) {
            return std::max(2,
                static_cast<int>(std::ceil(length / step)) + 1 + 2 * kBorderCells);
        };

        VoxelGrid grid(samplesAlong(extent[0]),
                       samplesAlong(extent[1]),
                       samplesAlong(extent[2]),
                       origin,
                       MFloatVector(step, step, step));

        for (const GaussianSplat& splat : m_splats)
            scatter(splat, grid);

        return grid;
    }

    void SplatDensityVolume::scatter(const GaussianSplat& splat, VoxelGrid& grid) const
    {
        if (splat.opacity <= kMinOpacity)
            return;

        const MFloatVector& cell = grid.cellSize();

        const float deviation[3] = {
            std::max(splat.scale[0], cell.x * kMinDeviationInCells),
            std::max(splat.scale[1], cell.y * kMinDeviationInCells),
            std::max(splat.scale[2], cell.z * kMinDeviationInCells)
        };

        const float reach = m_reachInDeviations *
            std::max({ deviation[0], deviation[1], deviation[2] });

        if (reach > kMaxCellSpan * 0.5f * cell.x)
            return;

        const MFloatVector& origin = grid.origin();
        const int size[3] = { grid.sizeX(), grid.sizeY(), grid.sizeZ() };
        const float center[3] = {
            static_cast<float>(splat.center.x),
            static_cast<float>(splat.center.y),
            static_cast<float>(splat.center.z)
        };
        const float originAxis[3] = { origin.x, origin.y, origin.z };
        const float cellAxis[3] = { cell.x, cell.y, cell.z };

        int lower[3];
        int upper[3];
        for (int axis = 0; axis < 3; ++axis)
        {
            const float relative = center[axis] - originAxis[axis];

            lower[axis] = std::max(0,
                static_cast<int>(std::floor((relative - reach) / cellAxis[axis])));

            upper[axis] = std::min(size[axis] - 1,
                static_cast<int>(std::ceil((relative + reach) / cellAxis[axis])));

            if (lower[axis] > upper[axis])
                return;
        }

        float axes[3][3];
        quaternionAxes(splat.rotation, axes);

        const float inverseDeviation[3] = {
            1.0f / deviation[0], 1.0f / deviation[1], 1.0f / deviation[2]
        };

        const float cutoff = m_reachInDeviations * m_reachInDeviations;

        for (int z = lower[2]; z <= upper[2]; ++z)
        {
            for (int y = lower[1]; y <= upper[1]; ++y)
            {
                for (int x = lower[0]; x <= upper[0]; ++x)
                {
                    const MFloatVector point = grid.positionAt(x, y, z);
                    const float offset[3] = {
                        point.x - center[0], point.y - center[1], point.z - center[2]
                    };

                    // Into the splat's own frame, where the covariance is
                    // diagonal and the falloff is a plain squared distance.
                    float squaredDistance = 0.0f;
                    for (int axis = 0; axis < 3; ++axis)
                    {
                        const float local =
                            (offset[0] * axes[axis][0] +
                             offset[1] * axes[axis][1] +
                             offset[2] * axes[axis][2]) * inverseDeviation[axis];

                        squaredDistance += local * local;
                    }

                    if (squaredDistance > cutoff)
                        continue;

                    const float density =
                        splat.opacity * std::exp(-0.5f * squaredDistance);

                    if (density > grid.valueAt(x, y, z))
                        grid.setValue(x, y, z, density);
                }
            }
        }
    }
}
