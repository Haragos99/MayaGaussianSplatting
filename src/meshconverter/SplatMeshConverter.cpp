#include "SplatMeshConverter.h"

#include "MarchingCubes.h"
#include "MayaMeshBuilder.h"
#include "SplatDensityVolume.h"

#include <maya/MGlobal.h>

namespace GS::Mesh
{
    MObject SplatMeshConverter::convert(const std::vector<GaussianSplat>& splats,
                                        const MBoundingBox& bounds,
                                        const ConversionSettings& settings,
                                        const MString& baseName)
    {
        if (splats.empty())
        {
            MGlobal::displayWarning("Splat to mesh: no splats to convert.");
            return MObject::kNullObj;
        }

        const SplatDensityVolume volume(splats, bounds, settings.reach);
        const VoxelGrid grid = volume.build(settings.resolution);

        if (grid.isEmpty())
        {
            MGlobal::displayWarning("Splat to mesh: the density volume is empty.");
            return MObject::kNullObj;
        }

        const MeshData data = MarchingCubes().generate(grid, settings.isoLevel);

        if (data.isEmpty())
        {
            MGlobal::displayWarning(
                "Splat to mesh: no surface at the requested iso level.");
            return MObject::kNullObj;
        }

        MGlobal::displayInfo(
            MString("Splat to mesh: ") + static_cast<int>(data.points.length()) +
            " vertices, " + static_cast<int>(data.polygonCounts.length()) +
            " triangles.");

        return MayaMeshBuilder::create(data, baseName);
    }
}
