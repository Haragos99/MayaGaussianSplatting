#pragma once
#include <maya/MPxDrawOverride.h>
#include <maya/MPxGeometryOverride.h>
#include <maya/MUserData.h>
#include <maya/MGlobal.h>
#include <maya/MPxLocatorNode.h>
#include <maya/MDrawRegistry.h>
#include <maya/MHWGeometry.h>
#include <random>
#include <maya/MPxSubSceneOverride.h>
#include <maya/MShaderManager.h>
#include <maya/MBoundingBox.h>
#include <maya/MPoint.h>
#include <maya/MFnDependencyNode.h>
#include <maya/MFnDagNode.h>
#include <maya/MFnCamera.h>
#include <maya/M3dView.h>
#include <maya/MDagPath.h>
#include "data.h"
#include <chrono>

using namespace MHWRender;


struct TestPoint
{
    float pos[3];
    float color[4]; // rgba
};

// MyLocator node is drawn using MyLocatorDrawOverride
class GaussianSplattingLocator : public MPxLocatorNode
{
public:
	GaussianSplattingLocator() = default;
    static MTypeId id;
    static MObject locatorMsgAttr;
    static void* creator() { return new GaussianSplattingLocator(); }
    static MStatus initialize();
    MStatus connectionMade(const MPlug& plug, const MPlug& otherPlug, bool asSrc) override;
    
    bool isBounded() const override { return true; }


};


class GaussianSplattingSubSceneOverride final : public MHWRender::MPxSubSceneOverride
{
public:
    static MHWRender::MPxSubSceneOverride* creator(const MObject& obj)
    {
        return new GaussianSplattingSubSceneOverride(obj);
    }

    GaussianSplattingSubSceneOverride(const MObject& obj);

    MObject m_object;
    MPxNode* m_node = nullptr;

    ~GaussianSplattingSubSceneOverride() override;

    MHWRender::DrawAPI supportedDrawAPIs() const override;

    bool requiresUpdate(
        const MHWRender::MSubSceneContainer& container,
        const MHWRender::MFrameContext& frameContext) const override;

    void update(
        MHWRender::MSubSceneContainer& container,
        const MHWRender::MFrameContext& frameContext) override;

    bool furtherUpdateRequired(
        const MHWRender::MFrameContext& frameContext) override;

    bool hasUIDrawables() const override;

    bool areUIDrawablesDirty() const override;

    void addUIDrawables(
        MHWRender::MUIDrawManager& drawManager,
        const MHWRender::MFrameContext& frameContext) override;

    bool enableUpdateForSelection() const override;

    bool getSelectionPath(
        const MHWRender::MRenderItem& renderItem,
        MDagPath& dagPath) const override;

    bool getInstancedSelectionPath(
        const MHWRender::MRenderItem& renderItem,
        const MHWRender::MIntersection& intersection,
        MDagPath& dagPath) const override;

    void updateSelectionGranularity(
        const MDagPath& path,
        MHWRender::MSelectionContext& selectionContext) override;

    void markDirty();

private:
    void createOrUpdateRenderItem(
        MHWRender::MSubSceneContainer& container);

    void createShader();

    void releaseShader();

    void loadSplatsFromNodeOrDemoData();

    bool readCamera(
        MPoint& cameraWorldPosition,
        MVector& cameraWorldRight,
        MVector& cameraWorldUp,
        MVector& cameraWorldForward) const;

    bool cameraChanged(
        const MPoint& cameraWorldPosition,
        const MVector& cameraWorldForward) const;

    void buildVertexBuffer(
        const std::vector<GS::SplatVertex>& vertices);

    void buildIndexBuffer(
        const std::vector<unsigned int>& indices);


    void buildStaticVertexBuffersOnce();

    void rebuildSortedIndexBufferOnly(
        const MPoint& cameraWorldPosition,
        const MVector& cameraWorldForward);

    bool cameraChangedEnoughForIndexRebuild(
        const MPoint& cameraWorldPosition,
        const MVector& cameraWorldForward) const;

    void uploadVertexBuffers(
        const std::vector<GS::SplatVertex>& vertices);

    void uploadIndexBuffer(
        const std::vector<unsigned int>& indices);




    static float depthFromCamera(
        const MPoint& worldPoint,
        const MPoint& cameraWorldPosition,
        const MVector& cameraWorldForward);

private:

    void buildSplatVertices(
        const GS::GaussianSplat& splat,
        const MVector& cameraRight,
        const MVector& cameraUp,
        std::vector<GS::SplatVertex>& vertices);
    
    MMatrix buildCovariance(const GS::GaussianSplat& splat) const;

    MMatrix projectCovarianceToCamera(
        const MMatrix& covariance,
        const MVector& cameraRight,
        const MVector& cameraUp) const;

    double covarianceQuadraticForm(
        const MMatrix& covariance,
        const MVector& v,
        const MVector& w) const;

    ProjectedEllipse calculateEllipseAxes(
        const MMatrix& covariance2D,
        const MVector& cameraRight,
        const MVector& cameraUp,
        double sigmaMultiplier) const;


    void appendSplatQuad(
        const GS::GaussianSplat& splat,
        const MVector& axisX,
        const MVector& axisY,
        std::vector<GS::SplatVertex>& vertices);


    static const MString kRenderItemName;
    unsigned int CircleSegments = 16;
    MObject m_nodeObj;
    MDagPath m_dagPath;

    bool m_dirty = true;
    bool m_geometryDirty = true;
    bool m_shaderDirty = true;
    bool m_uiDirty = true;
    bool m_vertexBufferDirty = true;   // Rebuild only when PLY/data changes.
    bool m_indexBufferDirty = true;    // Rebuild when camera sorting changes.

    std::chrono::high_resolution_clock::time_point m_lastFrame;
    double m_fps = 0.0;

    std::vector<GS::GaussianSplat> m_splats;

    MBoundingBox m_boundingBox;

    std::unique_ptr<MHWRender::MVertexBuffer> m_positionBuffer;
    std::unique_ptr<MHWRender::MVertexBuffer> m_colorBuffer;
    std::unique_ptr<MHWRender::MVertexBuffer> m_uvBuffer;
    std::unique_ptr<MHWRender::MIndexBuffer>  m_indexBuffer;

    MHWRender::MShaderInstance* m_shader = nullptr;

    mutable bool m_haveLastCamera = false;
    mutable MPoint m_lastCameraPosition;
    mutable MVector m_lastCameraForward;

    unsigned int m_vertexCount = 0;
    unsigned int m_indexCount = 0;
};


