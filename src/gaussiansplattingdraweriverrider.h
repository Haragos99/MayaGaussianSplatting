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
#include "data.h"
#include <chrono>
#include "gaussiansplattingnode.h"
#include "render/SplatBufferManager.h"
#include "render/SplatDepthSorter.h"
#include "render/ViewportCamera.h"

using namespace MHWRender;


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

    float getSpaltSize() const;
    

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

    // Drops every cached GPU/CPU result so the next update rebuilds from scratch.
    void discardCachedGeometry();

    const std::vector<GS::GaussianSplat>& splats() const;

    void buildStaticVertexBuffersOnce(const GS::CameraState& camera);

    void rebuildSortedIndexBufferOnly(const GS::CameraState& camera);

    // Camera re-expressed in the node's object space, where the splats live.
    GS::CameraState objectSpaceCamera(const GS::CameraState& camera) const;

    void bindGeometry(MHWRender::MRenderItem& item);

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

	bool sliderDirty = true;
    std::chrono::high_resolution_clock::time_point m_lastFrame;
    double m_fps = 0.0;

    GaussianSplattingLocator* m_locator = nullptr;
    unsigned int m_dataVersion = 0;

	float m_splatSize;
    MHWRender::MShaderInstance* m_splatShader = nullptr;

    MBoundingBox m_boundingBox;

    GS::SplatBufferManager m_buffers;
    GS::ViewportCamera m_camera;
    GS::SplatDepthSorter m_sorter;

};


