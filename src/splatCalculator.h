#pragma once

#include <maya/MBoundingBox.h>
#include <maya/MVector.h>

#include <vector>

#include "data.h"

// Symmetric 3D covariance, upper triangle only.
// Sigma = R * S * S^T * R^T, [KKLD23] Eq. 6. Camera independent.
struct SplatCovariance3
{
    float xx = 0.0f;
    float xy = 0.0f;
    float xz = 0.0f;
    float yy = 0.0f;
    float yz = 0.0f;
    float zz = 0.0f;
};


// Camera plane basis expressed in the space the splat centers live in.
struct SplatProjectionBasis
{
    MVector right{ 1.0, 0.0, 0.0 };
    MVector up{ 0.0, 1.0, 0.0 };
};


class SplatCalculator {

public:
    // The quad covers 3 sigma. GaussianSplat.ogsfx uses the same constant.
    static constexpr float kSigmaExtent = 3.0f;

    // Emits the four corners of one splat quad.
    // Returns false when the splat is culled and nothing was written.
    static bool buildSplatVertices(
        const GS::GaussianSplat& splat,
        const SplatProjectionBasis& basis,
        std::vector<GS::SplatVertex>& vertices,
        float splatSize,
        MBoundingBox& boundingBox
    );

private:
    static SplatCovariance3 buildCovariance(
        const GS::GaussianSplat& splat,
        float splatSize
    );

    // Sigma' = W * Sigma * W^T reduced to the camera plane, [KKLD23] Eq. 5.
    static void projectCovarianceToCamera(
        const SplatCovariance3& covariance,
        const SplatProjectionBasis& basis,
        float& c00,
        float& c01,
        float& c11
    );

    // Closed form eigen decomposition of the 2x2 conic, no trigonometry.
    static bool calculateEllipseAxes(
        float c00,
        float c01,
        float c11,
        const SplatProjectionBasis& basis,
        ProjectedEllipse& ellipse
    );

    static void appendSplatQuad(
        const GS::GaussianSplat& splat,
        const MVector& axisX,
        const MVector& axisY,
        std::vector<GS::SplatVertex>& vertices,
        MBoundingBox& boundingBox
    );

    // Splats below this alpha never reach the framebuffer ([KKLD23] uses 1/255).
    static constexpr float kMinAlpha = 1.0f / 255.0f;

    // Low pass filter: keeps needle thin splats from collapsing to zero area.
    static constexpr float kMinAxisRatio = 0.01f;
};
