#include "SplatDataSource.h"

#include <maya/MGlobal.h>

#include <cmath>
#include <string>
#include <utility>

#include "../gaussianSplatPlyLoader.h"

namespace
{
    // Used only when the node has no usable file yet.
    const MString kBackupFilePath(
        "C:\\Users\\Geri\\Documents\\Projects\\CG\\MayaGaussianSplatting\\models\\Tree.ply");

    const MString kProceduralSourceName("<procedural backup>");
}

namespace GS
{
    bool SplatDataSource::loadFromFile(const MString& filePath)
    {
        if (filePath.length() > 0)
        {
            std::vector<GaussianSplat> loaded;
            std::string error;

            if (GaussianSplatPlyLoader::load(filePath.asChar(), loaded, &error) &&
                !loaded.empty())
            {
                adopt(std::move(loaded), filePath);

                MGlobal::displayInfo(
                    "Gaussian splats loaded from: " + filePath);

                return true;
            }

            MGlobal::displayWarning(
                "Gaussian splat file could not be loaded: " + filePath +
                " (" + MString(error.c_str()) + ")");
        }

        // A bad path must not throw away splats that are already valid.
        if (hasSplats())
        {
            return false;
        }

        return loadBackup();
    }


    bool SplatDataSource::loadBackup()
    {
        std::vector<GaussianSplat> loaded;
        std::string error;

        if (GaussianSplatPlyLoader::load(kBackupFilePath.asChar(), loaded, &error) &&
            !loaded.empty())
        {
            adopt(std::move(loaded), kBackupFilePath);

            MGlobal::displayInfo(
                "Gaussian splats loaded from backup file: " + kBackupFilePath);

            return true;
        }

        buildProceduralBackup();

        return true;
    }


    void SplatDataSource::buildProceduralBackup()
    {
        constexpr int countX = 20;
        constexpr int countY = 20;

        std::vector<GaussianSplat> generated;
        generated.reserve(countX * countY);

        for (int y = 0; y < countY; ++y)
        {
            for (int x = 0; x < countX; ++x)
            {
                const float fx =
                    static_cast<float>(x) / static_cast<float>(countX - 1);

                const float fy =
                    static_cast<float>(y) / static_cast<float>(countY - 1);

                GaussianSplat splat;

                splat.center = MPoint(
                    (fx - 0.5f) * 6.0f,
                    (fy - 0.5f) * 4.0f,
                    std::sin(fx * 6.2831853f) * 0.5f);

                splat.color = MColor(fx, fy, 1.0f - fx, 0.45f);
                splat.scaleX = 0.08f;
                splat.scaleY = 0.08f;
                splat.scaleZ = 0.08f;
                splat.opacity = 0.45f;

                generated.push_back(splat);
            }
        }

        adopt(std::move(generated), kProceduralSourceName);

        MGlobal::displayWarning(
            "No valid Gaussian splat file. Using the procedural backup grid.");
    }


    void SplatDataSource::adopt(
        std::vector<GaussianSplat>&& splats,
        const MString& sourcePath)
    {
        m_splats = std::move(splats);
        m_bounds = computeBounds(m_splats);
        m_sourcePath = sourcePath;

        ++m_version;
    }


    MBoundingBox SplatDataSource::computeBounds(
        const std::vector<GaussianSplat>& splats)
    {
        MBoundingBox bounds;

        for (const GaussianSplat& splat : splats)
        {
            const double r =
                std::max(splat.scaleX, splat.scaleY) * 2.0;

            bounds.expand(splat.center + MVector(r, r, r));
            bounds.expand(splat.center + MVector(-r, -r, -r));
        }

        return bounds;
    }


    void SplatDataSource::clear()
    {
        m_splats.clear();
        m_splats.shrink_to_fit();

        m_bounds.clear();
        m_sourcePath.clear();

        ++m_version;
    }
}
