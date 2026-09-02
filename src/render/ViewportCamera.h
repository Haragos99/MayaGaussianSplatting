#pragma once

#include <maya/MMatrix.h>
#include <maya/MPoint.h>
#include <maya/MVector.h>

namespace GS
{
    // Orthonormal basis of a viewport camera, in world space.
    struct CameraState
    {
        MPoint  position{ 0.0, 0.0, 10.0 };
        MVector right{ 1.0, 0.0, 0.0 };
        MVector up{ 0.0, 1.0, 0.0 };
        MVector forward{ 0.0, 0.0, -1.0 };
    };


    // Reads the active viewport camera and decides when a cached view-dependent
    // result (splat sorting, billboard orientation) has become stale.
    class ViewportCamera
    {
    public:
        // Fills state with the active 3d view camera. Leaves state untouched on failure.
        static bool readActive(CameraState& state);

        // Re-expresses the camera in another space, for example object space
        // through the world-to-object matrix. Assumes a rigid transform.
        static CameraState toSpace(
            const CameraState& state,
            const MMatrix& transform);

        // Signed distance of a world point along the camera view direction.
        static double depthAlongView(
            const MPoint& worldPoint,
            const CameraState& state);

        bool hasReference() const { return m_hasReference; }

        // True when the camera moved/rotated past the resort thresholds.
        bool needsResort(const CameraState& state) const;

        // Stores state as the new comparison reference.
        void commit(const CameraState& state);

        void invalidate();

    private:
        // Larger values mean fewer sorts: faster viewport, less accurate transparency.
        static constexpr double kPositionThreshold = 0.05;
        static constexpr double kDirectionThreshold = 0.002;

        CameraState m_reference;
        bool m_hasReference = false;
    };
}
