#include "gaussiansplattingdraweriverrider.h"
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnNumericAttribute.h>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include "gaussianSplatPlyLoader.h"
#include <maya/MQuaternion.h>
#include "splatCalculator.h"

const MString GaussianSplattingSubSceneOverride::kRenderItemName("gaussianSplatRenderItem");

GaussianSplattingSubSceneOverride::GaussianSplattingSubSceneOverride(const MObject& obj)
    : MPxSubSceneOverride(obj)
    , m_nodeObj(obj)
{
    MStatus status;
    MFnDagNode dagNode(obj, &status);
    m_lastFrame = std::chrono::high_resolution_clock::now();
    if (status)
    {
        dagNode.getPath(m_dagPath);
    }
	m_splatSize = 1.0f;
    MFnDependencyNode fnNode(obj);
    m_locator = dynamic_cast<GaussianSplattingLocator*>(fnNode.userNode());
    if (!m_locator)
    {
		MGlobal::displayError("Failed to get GaussianSplattingLocator node from MObject.");
    }

    m_dirty = true;
    m_geometryDirty = true;
    m_shaderDirty = true;
    m_uiDirty = true;
}

const std::vector<GS::GaussianSplat>& GaussianSplattingSubSceneOverride::splats() const
{
    static const std::vector<GS::GaussianSplat> kNoSplats;

    return m_locator ? m_locator->splats() : kNoSplats;
}

void GaussianSplattingSubSceneOverride::discardCachedGeometry()
{
    m_buffers.releaseAll();
    m_camera.invalidate();
    m_boundingBox.clear();

    m_vertexBufferDirty = true;
    m_indexBufferDirty = true;
    m_uiDirty = true;
}

float GaussianSplattingSubSceneOverride::getSpaltSize() const
{
    MPlug plug(m_nodeObj, GaussianSplattingLocator::aSplatSize);

    return plug.asFloat();
}

GaussianSplattingSubSceneOverride::~GaussianSplattingSubSceneOverride()
{
    releaseShader();

    m_buffers.releaseAll();
}

MHWRender::DrawAPI GaussianSplattingSubSceneOverride::supportedDrawAPIs() const
{
    return MHWRender::kOpenGLCoreProfile | MHWRender::kDirectX11;
}

bool GaussianSplattingSubSceneOverride::requiresUpdate(
    const MHWRender::MSubSceneContainer& container,
    const MHWRender::MFrameContext& frameContext) const
{
    GS::CameraState camera;

    const bool cameraNeedsResort =
        GS::ViewportCamera::readActive(camera) &&
        m_camera.needsResort(camera);

    return
        container.count() == 0 ||
        m_splatSize != getSpaltSize()||
        m_dirty ||
        m_vertexBufferDirty ||
        m_indexBufferDirty ||
        m_shaderDirty ||
        cameraNeedsResort ||
        sliderDirty ||
        !m_locator ||
        m_locator->needsReload() ||
        m_locator->dataVersion() != m_dataVersion ||
        !m_buffers.isRenderable();
}

void GaussianSplattingSubSceneOverride::update(
    MHWRender::MSubSceneContainer& container,
    const MHWRender::MFrameContext& frameContext)
{
    auto now = std::chrono::high_resolution_clock::now();

    double dt = std::chrono::duration<double>(now - m_lastFrame).count();

    m_fps = 1.0 / dt;

    sliderDirty = m_splatSize != getSpaltSize();


    m_splatSize = getSpaltSize();
	m_splatSize = std::max(0.1f, std::min(m_splatSize, 1.0f));

    // A new file path invalidates every cached buffer.
    if (m_locator)
    {
        const unsigned int dataVersion = m_locator->syncSplatData();

        if (dataVersion != m_dataVersion)
        {
            m_dataVersion = dataVersion;
            discardCachedGeometry();
        }
    }

    createOrUpdateRenderItem(container);

    MHWRender::MRenderItem* item = container.find(kRenderItemName);
    if (!item)
    {
        return;
    }

    if (!m_splatShader || m_shaderDirty)
    {
        createShader();
        item->setShader(m_splatShader);
    }

    // Falls back to the CameraState defaults when no viewport camera is available.
    GS::CameraState camera;
    GS::ViewportCamera::readActive(camera);

    // Build static vertex buffers only when splat data changed.
    if (m_vertexBufferDirty ||
        sliderDirty ||
        !m_buffers.hasVertices())
    {
        buildStaticVertexBuffersOnce(camera);

        // If vertices changed, the index buffer must also be rebuilt.
        m_indexBufferDirty = true;
    }
    
    // Rebuild only the sorted index buffer when the camera changed enough.
    if (m_indexBufferDirty ||
        !m_buffers.hasIndices() ||
        m_camera.needsResort(camera))
    {
        rebuildSortedIndexBufferOnly(camera);
    }

    bindGeometry(*item);

    item->enable(m_buffers.isRenderable());

    m_dirty = false;
    m_vertexBufferDirty = false;
    m_indexBufferDirty = false;
    m_shaderDirty = false;
    m_uiDirty = true;
    m_lastFrame = now;
}

bool GaussianSplattingSubSceneOverride::furtherUpdateRequired(
    const MHWRender::MFrameContext& frameContext)
{
    // Return true only if you are doing progressive/async loading.
    // Be careful: always returning true can hurt Maya interactivity.
    return false;
}

bool GaussianSplattingSubSceneOverride::hasUIDrawables() const
{
    return true;
}

bool GaussianSplattingSubSceneOverride::areUIDrawablesDirty() const
{
    return m_uiDirty;
}

void GaussianSplattingSubSceneOverride::addUIDrawables(
    MHWRender::MUIDrawManager& drawManager,
    const MHWRender::MFrameContext& frameContext)
{
    drawManager.beginDrawable();

    drawManager.setColor(MColor(1.0f, 0.8f, 0.1f, 1.0f));

    MString splatSizeLabel;
    splatSizeLabel += "Gaussian splats: ";
    splatSizeLabel += static_cast<int>(splats().size());

    drawManager.text(
        MPoint(0.0, 1.5, 0.0),
        splatSizeLabel,
        MHWRender::MUIDrawManager::kCenter);

    MString fpsLabel;
    fpsLabel += "FPS: ";
    fpsLabel += static_cast<int>(m_fps);

    drawManager.text(
        MPoint(0.0, 2.5, 0.0),
        fpsLabel,
        MHWRender::MUIDrawManager::kLeft);

    drawManager.setColor(MColor(0.2f, 0.8f, 1.0f, 1.0f));
    drawManager.box(
        m_boundingBox.center(),
        MVector(1.0, 0.0, 0.0),
        MVector(0.0, 1.0, 0.0),
        m_boundingBox.width(),
        m_boundingBox.height(),
        m_boundingBox.depth());

    drawManager.endDrawable();

    m_uiDirty = false;
}

bool GaussianSplattingSubSceneOverride::enableUpdateForSelection() const
{
    return true;
}

bool GaussianSplattingSubSceneOverride::getSelectionPath(
    const MHWRender::MRenderItem& renderItem,
    MDagPath& dagPath) const
{
    if (!m_dagPath.isValid())
    {
        return false;
    }

    dagPath = m_dagPath;
    return true;
}

bool GaussianSplattingSubSceneOverride::getInstancedSelectionPath(
    const MHWRender::MRenderItem& renderItem,
    const MHWRender::MIntersection& intersection,
    MDagPath& dagPath) const
{
    if (!m_dagPath.isValid())
    {
        return false;
    }

    dagPath = m_dagPath;
    return true;
}

void GaussianSplattingSubSceneOverride::updateSelectionGranularity(
    const MDagPath& path,
    MHWRender::MSelectionContext& selectionContext)
{
    selectionContext.setSelectionLevel(MHWRender::MSelectionContext::kObject);
}

void GaussianSplattingSubSceneOverride::markDirty()
{
    m_dirty = true;
    m_geometryDirty = true;
    m_uiDirty = true;
}

void GaussianSplattingSubSceneOverride::createOrUpdateRenderItem(
    MHWRender::MSubSceneContainer& container)
{
    MRenderItem* item = container.find(kRenderItemName);

    if (!item)
    {
        item = MRenderItem::Create(
            kRenderItemName,
            MRenderItem::DecorationItem,
            MGeometry::kTriangles);

        if (!item)
        {
            return;
        }

        item->setDrawMode(MGeometry::kAll);
        item->enable(true);

        // If your Maya version supports these, they are useful for transparent splats.
        item->setTreatAsTransparent(true);
        item->castsShadows(false);
        item->receivesShadows(false);

        container.add(item);
    }
}

void GaussianSplattingSubSceneOverride::createShader()
{
    releaseShader();

    MRenderer* renderer = MRenderer::theRenderer();
    if (!renderer)
    {
        return;
    }

    const MShaderManager* shaderManager = renderer->getShaderManager();
    if (!shaderManager)
    {
        return;
    }


    auto api =
        renderer->drawAPI();

    if (api != MHWRender::kOpenGLCoreProfile)
    {
        MGlobal::displayError(
            "GaussianSplat.ogsfx requires "
            "Maya OpenGL Core Profile."
        );

        return;
    }

    MStringArray techniques;

     
        shaderManager->getEffectsTechniques(
            "C:\\Users\\Geri\\Documents\\Projects\\CG\\MayaGaussianSplatting\\src\\shaders\\GaussianSplat.ogsfx",
            techniques,
            nullptr,
            0,
            false
        );


        if (techniques.length() == 0)
        {
            MGlobal::displayError(
                "Maya found no techniques in GaussianSplat.ogsfx."
            );

            MGlobal::displayError(
                shaderManager->getLastError()
            );

            MGlobal::displayError(
                shaderManager->getLastErrorSource(
                    true,
                    true,
                    10
                )
            );

            return;
        };
    
        MGlobal::displayInfo(
            "GaussianSplat.ogsfx techniques:"
        );


        for (unsigned int i = 0;
            i < techniques.length();
            ++i)
        {
            MGlobal::displayInfo(
                "  " + techniques[i]
            );
        }


        m_splatShader =
            shaderManager->getEffectsFileShader(
                "C:\\Users\\Geri\\Documents\\Projects\\CG\\MayaGaussianSplatting\\src\\shaders\\GaussianSplat.ogsfx",
                "Main",
                nullptr,
                0,
                false
            );


        if (!m_splatShader)
        {
            MGlobal::displayError(
                "getEffectsFileShader() returned null."
            );

            MGlobal::displayError(
                shaderManager->getLastError()
            );

            MGlobal::displayError(
                shaderManager->getLastErrorSource(
                    true,
                    true,
                    10
                )
            );

            return;
        }

    MGlobal::displayInfo(
        "Gaussian splat OGSFX shader loaded."
    );
}

void GaussianSplattingSubSceneOverride::releaseShader()
{
    if (!m_splatShader)
    {
        return;
    }

    MRenderer* renderer = MRenderer::theRenderer();
    if (renderer)
    {
        const MShaderManager* shaderManager = renderer->getShaderManager();
        if (shaderManager)
        {
            shaderManager->releaseShader(m_splatShader);
        }
    }

    m_splatShader = nullptr;
}

void GaussianSplattingSubSceneOverride::bindGeometry(
    MHWRender::MRenderItem& item)
{
    if (!m_buffers.isRenderable())
    {
        return;
    }

    MHWRender::MVertexBufferArray vertexBuffers;

    if (!m_buffers.fillVertexBufferArray(vertexBuffers))
    {
        return;
    }

    setGeometryForRenderItem(
        item,
        vertexBuffers,
        *m_buffers.indexBuffer(),
        &m_boundingBox);
}


void GaussianSplattingSubSceneOverride::rebuildSortedIndexBufferOnly(
    const GS::CameraState& camera)
{
    // Create splat IDs
    std::vector<unsigned int> sortedSplatIds;
    const std::vector<GS::GaussianSplat>& splatList = splats();

    sortedSplatIds.reserve(splatList.size());

    for (unsigned int i = 0;
        i < static_cast<unsigned int>(splatList.size());
        ++i)
    {
        sortedSplatIds.push_back(i);
    }

    // Object -> world matrix
    MMatrix objectToWorld;

    if (m_dagPath.isValid())
    {
        objectToWorld = m_dagPath.inclusiveMatrix();
    }

    // Sort splats back-to-front
    //
    // Important for alpha blending.
    
    std::sort(
        sortedSplatIds.begin(),
        sortedSplatIds.end(),
        [&](unsigned int a, unsigned int b)
        {
            const MPoint worldA =
                splatList[a].center * objectToWorld;

            const MPoint worldB =
                splatList[b].center * objectToWorld;

            const double depthA =
                GS::ViewportCamera::depthAlongView(
                    worldA,
                    camera
                );

            const double depthB =
                GS::ViewportCamera::depthAlongView(
                    worldB,
                    camera
                );

            return depthA > depthB;
        }
    );
    
    // Each splat now has exactly:
    //     6 vertices
    // representing:
    //     triangle 1: 0,1,2
    //     triangle 2: 3,4,5
    // Therefore:
    //     indices per splat = 6
    constexpr unsigned int VerticesPerSplat = 6;

    std::vector<unsigned int> indices;

    indices.reserve(
        splatList.size() * VerticesPerSplat
    );

    for (unsigned int splatId : sortedSplatIds)
    {
        const unsigned int base =
            splatId * VerticesPerSplat;

        // Triangle 1
        indices.push_back(base + 0);
        indices.push_back(base + 1);
        indices.push_back(base + 2);

        // Triangle 2
        indices.push_back(base + 3);
        indices.push_back(base + 4);
        indices.push_back(base + 5);
    }

    m_buffers.uploadIndices(indices);

    m_indexBufferDirty = false;

    m_camera.commit(camera);
}

void GaussianSplattingSubSceneOverride::buildStaticVertexBuffersOnce(
    const GS::CameraState& camera)
{
    std::vector<GS::SplatVertex> vertices;

    const std::vector<GS::GaussianSplat>& splatList = splats();

    // Two triangles / 6 vertices per splat.
    vertices.reserve(splatList.size() * 6);

    m_boundingBox.clear();

    for (const GS::GaussianSplat& splat : splatList)
    {
        SplatCalculator::buildSplatVertices(
            splat,
            camera.right,
            camera.up,
            vertices,
            m_splatSize,
            m_boundingBox
        );
    }

    m_buffers.uploadVertices(vertices);

    m_vertexBufferDirty = false;
}
