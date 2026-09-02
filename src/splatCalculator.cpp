#include "splatCalculator.h"

#include <algorithm>
#include <cmath>

bool SplatCalculator::buildSplatVertices(
    const GS::GaussianSplat& splat,
    const SplatProjectionBasis& basis,
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

    float c00 = 0.0f;
    float c01 = 0.0f;
    float c11 = 0.0f;

    projectCovarianceToCamera(covariance, basis, c00, c01, c11);

    ProjectedEllipse ellipse;

    if (!calculateEllipseAxes(c00, c01, c11, basis, ellipse))
    {
        return false;
    }

    appendSplatQuad(
        splat,
        ellipse.axisX,
        ellipse.axisY,
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


// Project the 3D covariance onto the camera plane.
//
//     Sigma' = W * Sigma * W^T      [KKLD23] Eq. 5 / [ZPBG01] EWA
//
// Returns the three unique entries of:
//
//     [ c00  c01 ]
//     [ c01  c11 ]

void SplatCalculator::projectCovarianceToCamera(
    const SplatCovariance3& covariance,
    const SplatProjectionBasis& basis,
    float& c00,
    float& c01,
    float& c11)
{
    const float rx = static_cast<float>(basis.right.x);
    const float ry = static_cast<float>(basis.right.y);
    const float rz = static_cast<float>(basis.right.z);

    const float ux = static_cast<float>(basis.up.x);
    const float uy = static_cast<float>(basis.up.y);
    const float uz = static_cast<float>(basis.up.z);


    // Sigma * right and Sigma * up, shared by the three quadratic forms.
    const float sr0 = covariance.xx * rx + covariance.xy * ry + covariance.xz * rz;
    const float sr1 = covariance.xy * rx + covariance.yy * ry + covariance.yz * rz;
    const float sr2 = covariance.xz * rx + covariance.yz * ry + covariance.zz * rz;

    const float su0 = covariance.xx * ux + covariance.xy * uy + covariance.xz * uz;
    const float su1 = covariance.xy * ux + covariance.yy * uy + covariance.yz * uz;
    const float su2 = covariance.xz * ux + covariance.yz * uy + covariance.zz * uz;


    c00 = rx * sr0 + ry * sr1 + rz * sr2;
    c01 = rx * su0 + ry * su1 + rz * su2;
    c11 = ux * su0 + uy * su1 + uz * su2;
}


// Eigen decomposition of the 2x2 conic.
//
// The eigenvector of the larger eigenvalue can be read directly out of the
// matrix, which removes the atan2 + cos + sin the old version ran per splat.
bool SplatCalculator::calculateEllipseAxes(
    float c00,
    float c01,
    float c11,
    const SplatProjectionBasis& basis,
    ProjectedEllipse& ellipse)
{
    const float middle = 0.5f * (c00 + c11);

    const float determinant = c00 * c11 - c01 * c01;

    const float discriminant =
        std::sqrt(
            std::max(
                0.0f,
                middle * middle - determinant
            )
        );

    const float lambdaMajor = middle + discriminant;
    const float lambdaMinor = middle - discriminant;

    // Degenerate splat: no visible area at all.
    if (lambdaMajor <= 0.0f)
    {
        return false;
    }


    // Eigenvector of lambdaMajor.
    float ex = lambdaMajor - c11;
    float ey = c01;

    const float eigenLengthSquared = ex * ex + ey * ey;

    if (eigenLengthSquared > 1.0e-20f)
    {
        const float inverseLength =
            1.0f / std::sqrt(eigenLengthSquared);

        ex *= inverseLength;
        ey *= inverseLength;
    }
    else
    {
        ex = 1.0f;
        ey = 0.0f;
    }


    // Eigenvalue = variance, sqrt(variance) = sigma.
    const float sigmaMajor = std::sqrt(lambdaMajor);

    // Low pass filter so needle thin splats keep a non degenerate quad.
    const float sigmaMinor =
        std::max(
            std::sqrt(std::max(0.0f, lambdaMinor)),
            sigmaMajor * kMinAxisRatio
        );


    // basis.right and basis.up are orthonormal, so rotating them by the
    // eigenvector already yields unit vectors: no normalize() needed.
    ellipse.axisX =
        (basis.right * ex + basis.up * ey) *
        static_cast<double>(sigmaMajor * kSigmaExtent);

    ellipse.axisY =
        (basis.right * -ey + basis.up * ex) *
        static_cast<double>(sigmaMinor * kSigmaExtent);

    return true;
}


// Append the two triangles forming one splat quad
void SplatCalculator::appendSplatQuad(
    const GS::GaussianSplat& splat,
    const MVector& axisX,
    const MVector& axisY,
    std::vector<GS::SplatVertex>& vertices,
    MBoundingBox& boundingBox
)
{
    const float alpha =
        std::clamp(
            splat.opacity,
            0.0f,
            1.0f
        );

    // Vertex helper
    auto makeVertex =
        [&](const MPoint& position,
            float u,
            float v)
        {
            GS::SplatVertex vertex{};

            vertex.position[0] =
                static_cast<float>(position.x);

            vertex.position[1] =
                static_cast<float>(position.y);

            vertex.position[2] =
                static_cast<float>(position.z);

            vertex.color[0] =
                splat.color.r;

            vertex.color[1] =
                splat.color.g;

            vertex.color[2] =
                splat.color.b;

            vertex.color[3] =
                alpha;

            vertex.uv[0] = u;
            vertex.uv[1] = v;

            return vertex;
        };


    // Four corners
    const MPoint center =
        splat.center;


    const MPoint bottomLeft =
        center -
        axisX -
        axisY;

    const MPoint bottomRight =
        center +
        axisX -
        axisY;

    const MPoint topRight =
        center +
        axisX +
        axisY;

    const MPoint topLeft =
        center -
        axisX +
        axisY;


    // Four shared corners. The index buffer stitches the two triangles, which
    // is a third less vertex data than emitting six standalone vertices.
    vertices.push_back(makeVertex(bottomLeft, -1.0f, -1.0f));
    vertices.push_back(makeVertex(bottomRight, 1.0f, -1.0f));
    vertices.push_back(makeVertex(topRight, 1.0f, 1.0f));
    vertices.push_back(makeVertex(topLeft, -1.0f, 1.0f));

    // Bounding box
    boundingBox.expand(bottomLeft);
    boundingBox.expand(bottomRight);
    boundingBox.expand(topRight);
    boundingBox.expand(topLeft);
}




