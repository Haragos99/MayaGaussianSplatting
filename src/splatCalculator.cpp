#include "splatCalculator.h"

#include <algorithm>
#include <cmath>

bool SplatCalculator::buildSplatVertices(
    const GS::GaussianSplat& splat,
    std::vector<GS::SplatVertex>& vertices,
    float splatSize,
    MBoundingBox& boundingBox
)
{
    // A splat under 1/255 alpha can never change a pixel, so it is dropped
    // before it costs any vertex, index or sorting work.
    if (splat.opacity < kMinAlpha)
    {
        return false;
    }

    const SplatCovariance3 covariance = buildCovariance(splat, splatSize);

    // Zero volume splat: the vertex shader could only build a degenerate quad.
    if (covariance.xx <= 0.0f &&
        covariance.yy <= 0.0f &&
        covariance.zz <= 0.0f)
    {
        return false;
    }

    appendSplatQuad(
        splat,
        covariance,
        vertices,
        boundingBox
    );

    return true;
}

// Build the 3D Gaussian covariance straight from the quaternion and the scale.
//
//     Sigma = R * S * S^T * R^T = M * M^T   with   M = R * S      [KKLD23] Eq. 6
//
// Only the six unique components are kept, everything stays in float and no
// MQuaternion / MMatrix temporary is built per splat.
SplatCovariance3 SplatCalculator::buildCovariance(
    const GS::GaussianSplat& splat,
    float splatSize)
{
    // The PLY decoders store the quaternion as {w, x, y, z}.
    float w = splat.rotation[0];
    float x = splat.rotation[1];
    float y = splat.rotation[2];
    float z = splat.rotation[3];

    const float lengthSquared =
        w * w + x * x + y * y + z * z;

    if (lengthSquared > 1.0e-12f)
    {
        const float inverseLength =
            1.0f / std::sqrt(lengthSquared);

        w *= inverseLength;
        x *= inverseLength;
        y *= inverseLength;
        z *= inverseLength;
    }
    else
    {
        w = 1.0f;
        x = 0.0f;
        y = 0.0f;
        z = 0.0f;
    }


    const float sx = splat.scaleX * splatSize;
    const float sy = splat.scaleY * splatSize;
    const float sz = splat.scaleZ * splatSize;


    // Columns of the rotation matrix, already scaled: M = R * S.
    const float m00 = (1.0f - 2.0f * (y * y + z * z)) * sx;
    const float m10 = (2.0f * (x * y + w * z)) * sx;
    const float m20 = (2.0f * (x * z - w * y)) * sx;

    const float m01 = (2.0f * (x * y - w * z)) * sy;
    const float m11 = (1.0f - 2.0f * (x * x + z * z)) * sy;
    const float m21 = (2.0f * (y * z + w * x)) * sy;

    const float m02 = (2.0f * (x * z + w * y)) * sz;
    const float m12 = (2.0f * (y * z - w * x)) * sz;
    const float m22 = (1.0f - 2.0f * (x * x + y * y)) * sz;


    SplatCovariance3 covariance;

    covariance.xx = m00 * m00 + m01 * m01 + m02 * m02;
    covariance.xy = m00 * m10 + m01 * m11 + m02 * m12;
    covariance.xz = m00 * m20 + m01 * m21 + m02 * m22;
    covariance.yy = m10 * m10 + m11 * m11 + m12 * m12;
    covariance.yz = m10 * m20 + m11 * m21 + m12 * m22;
    covariance.zz = m20 * m20 + m21 * m21 + m22 * m22;

    return covariance;
}


// Emit the four camera independent corners of one splat.
//
// Every corner carries the same center and the same covariance; only the
// (+-1, +-1) corner code differs. GaussianSplat.ogsfx projects the covariance
// and offsets the corner, so nothing here has to be rebuilt when the camera moves.
void SplatCalculator::appendSplatQuad(
    const GS::GaussianSplat& splat,
    const SplatCovariance3& covariance,
    std::vector<GS::SplatVertex>& vertices,
    MBoundingBox& boundingBox
)
{
    GS::SplatVertex vertex{};

    vertex.center[0] = static_cast<float>(splat.center.x);
    vertex.center[1] = static_cast<float>(splat.center.y);
    vertex.center[2] = static_cast<float>(splat.center.z);

    vertex.covA[0] = covariance.xx;
    vertex.covA[1] = covariance.xy;
    vertex.covA[2] = covariance.xz;

    vertex.covB[0] = covariance.yy;
    vertex.covB[1] = covariance.yz;
    vertex.covB[2] = covariance.zz;

    vertex.color[0] = splat.color.r;
    vertex.color[1] = splat.color.g;
    vertex.color[2] = splat.color.b;

    vertex.color[3] =
        std::clamp(
            splat.opacity,
            0.0f,
            1.0f
        );


    static constexpr float kCorners[GS::kVerticesPerSplatQuad][2] =
    {
        { -1.0f, -1.0f },
        {  1.0f, -1.0f },
        {  1.0f,  1.0f },
        { -1.0f,  1.0f }
    };

    for (const auto& corner : kCorners)
    {
        vertex.corner[0] = corner[0];
        vertex.corner[1] = corner[1];

        vertices.push_back(vertex);
    }


    // Axis aligned bound of the 3 sigma ellipsoid. Also camera independent,
    // so Maya frustum culls against a box that stays valid while orbiting.
    const MVector extent(
        kSigmaExtent * std::sqrt(std::max(0.0f, covariance.xx)),
        kSigmaExtent * std::sqrt(std::max(0.0f, covariance.yy)),
        kSigmaExtent * std::sqrt(std::max(0.0f, covariance.zz))
    );

    boundingBox.expand(splat.center + extent);
    boundingBox.expand(splat.center - extent);
}




