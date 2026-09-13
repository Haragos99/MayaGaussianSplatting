#pragma once

#include "VoxelGrid.h"

#include <maya/MBoundingBox.h>

#include <vector>

#include "../data.h"

namespace GS::Mesh
{
    // Voxelises a splat capture into a density volume, which is what lets
    // marching cubes turn a cloud into a surface.
    //
    // Each splat is scattered into the cells it reaches rather than each cell
    // asking which splats cover it: with a million anisotropic Gaussians the
    // second direction would be hopeless. Cells keep the strongest
    // contribution, so the iso level reads as an opacity.
    class SplatDensityVolume
    {
    public:
        SplatDensityVolume(const std::vector<GaussianSplat>& splats,
                           const MBoundingBox& bounds,
                           float reachInDeviations);

        VoxelGrid build(int resolution) const;

    private:
        void scatter(const GaussianSplat& splat, VoxelGrid& grid) const;

        const std::vector<GaussianSplat>& m_splats;
        MBoundingBox m_bounds;
        // How far out, in standard deviations, a splat still contributes.
        float m_reachInDeviations;
    };
}
