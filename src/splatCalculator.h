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


class SplatCalculator {

public:
    // The quad covers 3 sigma. GaussianSplat.ogsfx uses the same constant.
    static constexpr float kSigmaExtent = 3.0f;

    // Emits the four corners of one splat quad. The corners carry no camera
    // information: the vertex shader projects the covariance every frame.
    // Returns false when the splat is culled and nothing was written.
    static bool buildSplatVertices(
        const GS::GaussianSplat& splat,
        std::vector<GS::SplatVertex>& vertices,
        float splatSize,
        MBoundingBox& boundingBox
    );

private:
    static SplatCovariance3 buildCovariance(
        const GS::GaussianSplat& splat,
        float splatSize
    );

    static void appendSplatQuad(
        const GS::GaussianSplat& splat,
        const SplatCovariance3& covariance,
        std::vector<GS::SplatVertex>& vertices,
        MBoundingBox& boundingBox
    );

    // Splats below this alpha never reach the framebuffer ([KKLD23] uses 1/255).
    static constexpr float kMinAlpha = 1.0f / 255.0f;
};
