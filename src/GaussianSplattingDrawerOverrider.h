#pragma once
#include <maya/MPxDrawOverride.h>
#include <maya/MPxGeometryOverride.h>
#include <maya/MUserData.h>
#include <maya/MGlobal.h>
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
#include "Data.h"
#include <chrono>
#include "GaussianSplattingNode.h"
#include "render/SplatBufferManager.h"
#include "render/SplatDepthSorter.h"
#include "render/ViewportCamera.h"

using namespace MHWRender;

// This class is responsible for overriding the default drawing behavior of the GaussianSplattingNode in Maya's viewport. 
// It manages the creation and updating of render items, shaders, and vertex/index buffers for efficient rendering of Gaussian splats. 
// It also handles selection and UI drawing for the node.
class GaussianSplattingSubSceneOverride final : public MHWRender::MPxSubSceneOverride
{
public:
    static MHWRender::MPxSubSceneOverride* creator(const MObject& obj)
    {
        return new GaussianSplattingSubSceneOverride(obj);
    }

    GaussianSplattingSubSceneOverride(const MObject& obj);

    MObject m_object;
    MPxNode* m_node;

    ~GaussianSplattingSubSceneOverride() override;

    float getSpaltSize() const;
    
	bool isBoundingBoxActive() const;

	bool isSpaltInfoActive() const;

    MHWRender::DrawAPI supportedDrawAPIs() const override;

    bool requiresUpdate(
        const MHWRender::MSubSceneContainer& container,
        const MHWRender::MFrameContext& frameContext) const override;

    void update(
        MHWRender::MSubSceneContainer& container,
        const MHWRender::MFrameContext& frameContext) override;

    bool furtherUpdateRequired(const MHWRender::MFrameContext& frameContext) override;

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
    void initialize(const MObject& obj);

    void createOrUpdateRenderItem(MHWRender::MSubSceneContainer& container);

    void createShader();

    void releaseShader();

    // Drops every cached GPU/CPU result so the next update rebuilds from scratch.
    void discardCachedGeometry();

    const std::vector<GS::GaussianSplat>& splats() const;

    // Camera independent, so it only runs on data or splat size changes.
    void buildVertexBuffers();

    void rebuildSortedIndexBufferOnly(const GS::CameraState& camera);

    // Camera re-expressed in the node's object space, where the splats live.
    GS::CameraState objectSpaceCamera(const GS::CameraState& camera) const;

    void bindGeometry(MHWRender::MRenderItem& item);

    static const MString kRenderItemName;
    MObject m_nodeObj;
    MDagPath m_dagPath;

    bool m_dirty;
    bool m_geometryDirty;
    bool m_shaderDirty;
    bool m_uiDirty;
    bool m_vertexBufferDirty;   // Rebuild only when PLY/data changes.
    bool m_indexBufferDirty;    // Rebuild when camera sorting changes.

	bool sliderDirty;
    std::chrono::high_resolution_clock::time_point m_lastFrame;
    double m_fps;

    GaussianSplattingLocator* m_locator;
    unsigned int m_dataVersion;

	float m_splatSize;
    MHWRender::MShaderInstance* m_splatShader;

    MBoundingBox m_boundingBox;

    GS::SplatBufferManager m_buffers;
    GS::ViewportCamera m_camera;
    GS::SplatDepthSorter m_sorter;

};


