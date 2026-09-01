#include <maya/MMatrix.h>
#include "data.h"
#include <algorithm>
#include <maya/MQuaternion.h>
#include <maya/MBoundingBox.h>

class SplatCalculator {

public:
    static void buildSplatVertices(
        const GS::GaussianSplat& splat,
        const MVector& cameraRight,
        const MVector& cameraUp,
        std::vector<GS::SplatVertex>& vertices,
		float splatSize,
		MBoundingBox& boundingBox
    );
private:
    static MMatrix buildCovariance(const GS::GaussianSplat& splat, float splatSize);

    static MMatrix projectCovarianceToCamera(
        const MMatrix& covariance,
        const MVector& cameraRight,
        const MVector& cameraUp
    );

    static double covarianceQuadraticForm(
        const MMatrix& covariance,
        const MVector& v,
        const MVector& w
    );

    static ProjectedEllipse calculateEllipseAxes(
        const MMatrix& covariance2D,
        const MVector& cameraRight,
        const MVector& cameraUp,
        double sigmaMultiplier
    );

    static void appendSplatQuad(
        const GS::GaussianSplat& splat,
        const MVector& axisX,
        const MVector& axisY,
        std::vector<GS::SplatVertex>& vertices,
        MBoundingBox& boundingBox
    );

};
