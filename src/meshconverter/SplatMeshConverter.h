#pragma once

#include <maya/MBoundingBox.h>
#include <maya/MObject.h>
#include <maya/MString.h>

#include <vector>

#include "../data.h"

namespace GS::Mesh
{
    struct ConversionSettings
    {
        int resolution = 128;     // Samples along the longest axis.
        float isoLevel = 0.20f;   // Density treated as the surface.
        float reach = 2.0f;       // Splat support, in standard deviations.
    };

    // Splats -> density volume -> marching cubes -> DAG mesh.
    class SplatMeshConverter
    {
    public:
        static MObject convert(const std::vector<GaussianSplat>& splats,
                               const MBoundingBox& bounds,
                               const ConversionSettings& settings,
                               const MString& baseName);
    };
}
