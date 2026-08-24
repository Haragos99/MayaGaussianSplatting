#include "ViewportCamera.h"

#include <maya/M3dView.h>
#include <maya/MDagPath.h>
#include <maya/MFnCamera.h>
#include <maya/MMatrix.h>
#include <maya/MStatus.h>

#include <algorithm>
#include <cmath>

namespace GS
{
    bool ViewportCamera::readActive(CameraState& state)
    {
        M3dView view = M3dView::active3dView();

        MDagPath cameraPath;
        MStatus status = view.getCamera(cameraPath);

        if (!status || !cameraPath.isValid())
        {
            return false;
        }

        MFnCamera camera(cameraPath, &status);
        if (!status)
        {
            return false;
        }

        const MPoint eye = camera.eyePoint(MSpace::kWorld, &status);
        if (!status)
        {
            return false;
        }

        const MMatrix cameraMatrix = cameraPath.inclusiveMatrix();

        CameraState result;

        result.position = eye;

        result.right = MVector(
            cameraMatrix[0][0],
            cameraMatrix[0][1],
            cameraMatrix[0][2]);

        result.up = MVector(
            cameraMatrix[1][0],
            cameraMatrix[1][1],
            cameraMatrix[1][2]);

        // Maya cameras look down their negative local Z axis.
        result.forward = MVector(
            -cameraMatrix[2][0],
            -cameraMatrix[2][1],
            -cameraMatrix[2][2]);

        result.right.normalize();
        result.up.normalize();
        result.forward.normalize();

        state = result;

        return true;
    }


    double ViewportCamera::depthAlongView(
        const MPoint& worldPoint,
        const CameraState& state)
    {
        const MVector toPoint = worldPoint - state.position;

        return toPoint * state.forward;
    }


    bool ViewportCamera::needsResort(const CameraState& state) const
    {
        if (!m_hasReference)
        {
            return true;
        }

        const double positionDelta =
            state.position.distanceTo(m_reference.position);

        const double directionDot =
            std::clamp(
                state.forward.normal() * m_reference.forward.normal(),
                -1.0,
                1.0);

        const double directionDelta =
            1.0 - std::abs(directionDot);

        return
            positionDelta > kPositionThreshold ||
            directionDelta > kDirectionThreshold;
    }


    void ViewportCamera::commit(const CameraState& state)
    {
        m_reference = state;
        m_hasReference = true;
    }


    void ViewportCamera::invalidate()
    {
        m_hasReference = false;
    }
}
