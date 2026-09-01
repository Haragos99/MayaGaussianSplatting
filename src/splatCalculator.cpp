#include "splatCalculator.h"

void SplatCalculator::buildSplatVertices(
    const GS::GaussianSplat& splat,
    const MVector& cameraRight,
    const MVector& cameraUp,
    std::vector<GS::SplatVertex>& vertices,
    float splatSize,
    MBoundingBox& boundingBox
)
{
    // Build covariance from rotation + scale.
    const MMatrix covariance = buildCovariance(splat, splatSize);

    // Project the 3D covariance onto the camera plane.
    const MMatrix covariance2D = projectCovarianceToCamera(
            covariance,
            cameraRight,
            cameraUp
        );

    // Convert the 2D covariance into ellipse axes.
    constexpr double sigmaMultiplier = 3.0;

    const ProjectedEllipse ellipse = calculateEllipseAxes(
            covariance2D,
            cameraRight,
            cameraUp,
            sigmaMultiplier
        );

    // Create the actual quad.
    appendSplatQuad(
        splat,
        ellipse.axisX,
        ellipse.axisY,
        vertices,
        boundingBox
    );
}

// Build 3D Gaussian covariance
// Sigma = R * S * S^T * R^T
MMatrix SplatCalculator::buildCovariance(const GS::GaussianSplat& splat, float splatSize)
{
    MQuaternion q(
        splat.rotation[0],
        splat.rotation[1],
        splat.rotation[2],
        splat.rotation[3]
    );

    q.normalizeIt();


    // Rotation matrix
    const MMatrix rotation =
        q.asMatrix();

    // S^2 diagonal matrix
    // Instead of explicitly creating:
    // S * S.transpose()
    // directly create:
    // diag(sx^2, sy^2, sz^2)
    MMatrix scaleSquared;
    scaleSquared.setToIdentity();

    const double sx =
        static_cast<double>(splat.scaleX) * splatSize;

    const double sy =
        static_cast<double>(splat.scaleY) * splatSize;

    const double sz =
        static_cast<double>(splat.scaleZ) * splatSize;

    scaleSquared[0][0] = sx * sx;
    scaleSquared[1][1] = sy * sy;
    scaleSquared[2][2] = sz * sz;


    // Covariance
    return
        rotation *
        scaleSquared *
        rotation.transpose();
}


// Project 3D covariance into camera plane
//
// Returns:
//
//     [ c00  c01 ]
//     [ c01  c11 ]
//
// Stored in the upper-left 2x2 portion of an MMatrix.

MMatrix SplatCalculator::projectCovarianceToCamera(
    const MMatrix& covariance,
    const MVector& cameraRight,
    const MVector& cameraUp) 
{
    const double c00 =
        covarianceQuadraticForm(
            covariance,
            cameraRight,
            cameraRight
        );


    const double c01 =
        covarianceQuadraticForm(
            covariance,
            cameraRight,
            cameraUp
        );


    const double c11 =
        covarianceQuadraticForm(
            covariance,
            cameraUp,
            cameraUp
        );


    MMatrix covariance2D;
    covariance2D.setToIdentity();

    covariance2D[0][0] = c00;
    covariance2D[0][1] = c01;
    covariance2D[1][0] = c01;
    covariance2D[1][1] = c11;

    return covariance2D;
}


// Calculate v^T * Sigma * w
double SplatCalculator::covarianceQuadraticForm(
    const MMatrix& covariance,
    const MVector& v,
    const MVector& w) 
{
    return
        v.x * (
            covariance[0][0] * w.x +
            covariance[0][1] * w.y +
            covariance[0][2] * w.z
            )
        +
        v.y * (
            covariance[1][0] * w.x +
            covariance[1][1] * w.y +
            covariance[1][2] * w.z
            )
        +
        v.z * (
            covariance[2][0] * w.x +
            covariance[2][1] * w.y +
            covariance[2][2] * w.z
            );
}


// Calculate ellipse axes from 2D covariance
ProjectedEllipse SplatCalculator::calculateEllipseAxes(
    const MMatrix& covariance2D,
    const MVector& cameraRight,
    const MVector& cameraUp,
    double sigmaMultiplier) 
{
    const double c00 =
        covariance2D[0][0];

    const double c01 =
        covariance2D[0][1];

    const double c11 =
        covariance2D[1][1];


    // Eigenvalues of:
    //     [ c00 c01 ]
    //     [ c01 c11 ]
    const double trace =
        c00 + c11;

    const double diff =
        c00 - c11;

    const double discriminant =
        std::sqrt(
            std::max(
                0.0,
                diff * diff +
                4.0 * c01 * c01
            )
        );


    const double lambdaMajor =
        0.5 * (trace + discriminant);

    const double lambdaMinor =
        0.5 * (trace - discriminant);


    // Eigenvalue = variance
    // sqrt(variance) = sigma
    const double sigmaMajor =
        std::sqrt(
            std::max(
                0.0,
                lambdaMajor
            )
        );

    const double sigmaMinor =
        std::sqrt(
            std::max(
                0.0,
                lambdaMinor
            )
        );


    // Eigenvector angle
    const double angle =
        0.5 * std::atan2(
            2.0 * c01,
            c00 - c11
        );


    const double c =
        std::cos(angle);

    const double s =
        std::sin(angle);


    // Rotate camera basis into ellipse basis
    MVector axisX =
        cameraRight * c +
        cameraUp * s;

    MVector axisY =
        cameraRight * -s +
        cameraUp * c;


    axisX.normalize();
    axisY.normalize();


    // ------------------------------------------------------------------------
    // Scale to desired Gaussian radius.
    //
    // 3 sigma is a reasonable visualization extent.
    // ------------------------------------------------------------------------

    axisX *=
        sigmaMajor * sigmaMultiplier;

    axisY *=
        sigmaMinor * sigmaMultiplier;


    return {
        axisX,
        axisY
    };
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


    // Triangle 1
    vertices.push_back(
        makeVertex(
            bottomLeft,
            -1.0f,
            -1.0f
        )
    );

    vertices.push_back(
        makeVertex(
            bottomRight,
            1.0f,
            -1.0f
        )
    );

    vertices.push_back(
        makeVertex(
            topRight,
            1.0f,
            1.0f
        )
    );

    // Triangle 2
    vertices.push_back(
        makeVertex(
            bottomLeft,
            -1.0f,
            -1.0f
        )
    );

    vertices.push_back(
        makeVertex(
            topRight,
            1.0f,
            1.0f
        )
    );

    vertices.push_back(
        makeVertex(
            topLeft,
            -1.0f,
            1.0f
        )
    );

    // Bounding box
    boundingBox.expand(bottomLeft);
    boundingBox.expand(bottomRight);
    boundingBox.expand(topRight);
    boundingBox.expand(topLeft);
}




