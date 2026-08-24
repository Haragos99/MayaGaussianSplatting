#pragma once

#include <maya/MBoundingBox.h>
#include <maya/MString.h>

#include <vector>

#include "../data.h"

namespace GS
{
    // Owns the splat set of one locator and the policy used to refresh it:
    // a failed load never destroys data that is already on screen.
    class SplatDataSource
    {
    public:
        // Returns true when the splat set actually changed.
        bool loadFromFile(const MString& filePath);

        const std::vector<GaussianSplat>& splats() const { return m_splats; }
        const MBoundingBox& bounds() const { return m_bounds; }
        const MString& sourcePath() const { return m_sourcePath; }

        bool hasSplats() const { return !m_splats.empty(); }

        // Incremented on every successful change so consumers can detect reloads.
        unsigned int version() const { return m_version; }

        void clear();

    private:
        bool loadBackup();
        void buildProceduralBackup();

        void adopt(
            std::vector<GaussianSplat>&& splats,
            const MString& sourcePath);

        static MBoundingBox computeBounds(
            const std::vector<GaussianSplat>& splats);

        std::vector<GaussianSplat> m_splats;
        MBoundingBox m_bounds;
        MString m_sourcePath;
        unsigned int m_version = 0;
    };
}
