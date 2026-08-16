#include "gaussiansplattingdraweriverrider.h"
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnNumericAttribute.h>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include "gaussianSplatPlyLoader.h"
#include <maya/MQuaternion.h>

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
    loadSplatsFromNodeOrDemoData();
}

float GaussianSplattingSubSceneOverride::getSpaltSize() const
{
    MPlug plug(m_nodeObj, GaussianSplattingLocator::aSplatSize);

    return plug.asFloat();
}

GaussianSplattingSubSceneOverride::~GaussianSplattingSubSceneOverride()
{
    releaseShader();

    m_positionBuffer.reset();
    m_colorBuffer.reset();
    m_uvBuffer.reset();
    m_indexBuffer.reset();
}

MHWRender::DrawAPI GaussianSplattingSubSceneOverride::supportedDrawAPIs() const
{
    return MHWRender::kOpenGLCoreProfile | MHWRender::kDirectX11;
}

bool GaussianSplattingSubSceneOverride::requiresUpdate(
    const MHWRender::MSubSceneContainer& container,
    const MHWRender::MFrameContext& frameContext) const
{
    MPoint cameraPosition;
    MVector cameraRight;
    MVector cameraUp;
    MVector cameraForward;


    const bool haveCamera = readCamera(
        cameraPosition,
        cameraRight,
        cameraUp,
        cameraForward);

    const bool cameraNeedsResort =
        haveCamera && cameraChangedEnoughForIndexRebuild(
            cameraPosition,
            cameraForward);

    return
        container.count() == 0 ||
        m_splatSize != getSpaltSize()||
        m_dirty ||
        m_vertexBufferDirty ||
        m_indexBufferDirty ||
        m_shaderDirty ||
        cameraNeedsResort ||
        sliderDirty ||
        !m_positionBuffer ||
        !m_colorBuffer ||
        !m_uvBuffer ||
        !m_indexBuffer;
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


    MGlobal::displayInfo(
        "Current splat size: " + MString() + m_splatSize
    );

    createOrUpdateRenderItem(container);

    MHWRender::MRenderItem* item = container.find(kRenderItemName);
    if (!item)
    {
        return;
    }

    if (!m_shader || m_shaderDirty)
    {
        createShader();
    }

    if (m_splatShader)
    {
        item->setShader(m_splatShader);
    }

    MPoint cameraPosition;
    MVector cameraRight;
    MVector cameraUp;
    MVector cameraForward;

    const bool haveCamera = readCamera(
        cameraPosition,
        cameraRight,
        cameraUp,
        cameraForward);

    if (!haveCamera)
    {
        cameraPosition = MPoint(0.0, 0.0, 10.0);
        cameraForward = MVector(0.0, 0.0, -1.0);
    }

    // Build static vertex buffers only when splat data changed.
    if (
        m_vertexBufferDirty ||
        sliderDirty ||
        !m_positionBuffer ||
        !m_colorBuffer ||
        !m_uvBuffer)
    {
       
        buildStaticVertexBuffersOnce();
        // If vertices changed, the index buffer must also be rebuilt.
        m_indexBufferDirty = true;
    }
    
    // Rebuild only the sorted index buffer when the camera changed enough.
    if (
        m_indexBufferDirty ||
        !m_indexBuffer ||
        cameraChangedEnoughForIndexRebuild(cameraPosition, cameraForward))
    {
        rebuildSortedIndexBufferOnly(cameraPosition, cameraForward);
    }

    if (
        m_positionBuffer &&
        m_colorBuffer &&
        m_uvBuffer &&
        m_indexBuffer &&
        m_vertexCount > 0 &&
        m_indexCount > 0)
    {
        MHWRender::MVertexBufferArray vertexBuffers;

        vertexBuffers.addBuffer("positions", m_positionBuffer.get());
        vertexBuffers.addBuffer("colors", m_colorBuffer.get());
        vertexBuffers.addBuffer("uvs", m_uvBuffer.get());

        setGeometryForRenderItem(
            *item,
            vertexBuffers,
            *m_indexBuffer,
            &m_boundingBox);
    }

    item->enable(true);

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
    splatSizeLabel += static_cast<int>(m_splats.size());

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

    // Simple stock shader fallback.
    // For real Gaussian splatting, replace this with an effect/custom shader
    // that reads UV/color streams and computes:
    //
    //     alpha = exp(-dot(uv, uv) * falloff) * opacity
    //
   // m_shader = shaderManager->getStockShader(MShaderManager::k3dCPVSolidShader);
    if (m_shader)
    {
        float solidColor[4] = { 0.0f, 1.0f, 0.8f, 0.45f };
       // m_shader->setParameter("solidColor", solidColor);
    }
}

void GaussianSplattingSubSceneOverride::releaseShader()
{
    if (!m_shader)
    {
        return;
    }

    MRenderer* renderer = MRenderer::theRenderer();
    if (renderer)
    {
        const MShaderManager* shaderManager = renderer->getShaderManager();
        if (shaderManager)
        {
            shaderManager->releaseShader(m_shader);
        }
    }

    m_shader = nullptr;
}

void GaussianSplattingSubSceneOverride::loadSplatsFromNodeOrDemoData()
{
    // Replace this with reading data from your GaussianSplatShape node:
    //
    //   - file path attr
    //   - loaded PLY/splat data
    //   - positions
    //   - SH/color
    //   - opacity
    //   - scale
    //   - rotation/covariance
    //
    // This demo creates a small cloud.

    m_splats.clear();

    const int countX = 20;
    const int countY = 20;
    const std::string filePath = "C:\\Users\\Geri\\Documents\\Projects\\CG\\MayaGaussianSplatting\\models\\Tree.ply";
    std::string error;

    if (!GaussianSplatPlyLoader::load(filePath, m_splats, &error))
    { 
        for (int y = 0; y < countY; ++y)
        {
            for (int x = 0; x < countX; ++x)
            {
                const float fx = static_cast<float>(x) / static_cast<float>(countX - 1);
                const float fy = static_cast<float>(y) / static_cast<float>(countY - 1);

                GS::GaussianSplat splat;
                splat.center = MPoint(
                    (fx - 0.5f) * 6.0f,
                    (fy - 0.5f) * 4.0f,
                    std::sin(fx * 6.2831853f) * 0.5f);

                splat.color = MColor(fx, fy, 1.0f - fx, 0.45f);
                splat.scaleX = 0.08f;
                splat.scaleY = 0.08f;
				splat.scaleZ = 0.08f;
                splat.opacity = 0.45f;

                m_splats.push_back(splat);
            }
        }
    }
    m_boundingBox.clear();

    for (const GS::GaussianSplat& splat : m_splats)
    {
        const double r = std::max(splat.scaleX, splat.scaleY) * 2.0;
        m_boundingBox.expand(splat.center + MVector(r, r, r));
        m_boundingBox.expand(splat.center + MVector(-r, -r, -r));
    }

    m_dirty = true;
    m_geometryDirty = true;
    m_shaderDirty = true;
    m_uiDirty = true;
}

bool GaussianSplattingSubSceneOverride::readCamera(
    MPoint& cameraWorldPosition,
    MVector& cameraWorldRight,
    MVector& cameraWorldUp,
    MVector& cameraWorldForward) const
{
    M3dView view = M3dView::active3dView();

    MDagPath cameraPath;
    MStatus status = view.getCamera(cameraPath);

    if (!status || !cameraPath.isValid())
    {
        return false;
    }

    MFnCamera camera(cameraPath, &status);
    if (!status)
    {
        return false;
    }

    cameraWorldPosition = camera.eyePoint(MSpace::kWorld, &status);
    if (!status)
    {
        return false;
    }

    const MMatrix cameraMatrix = cameraPath.inclusiveMatrix();

    // These are camera local basis vectors transformed to world.
    // Depending on your camera convention, forward may need sign adjustment.
    cameraWorldRight = MVector(
        cameraMatrix[0][0],
        cameraMatrix[0][1],
        cameraMatrix[0][2]);

    cameraWorldUp = MVector(
        cameraMatrix[1][0],
        cameraMatrix[1][1],
        cameraMatrix[1][2]);

    cameraWorldForward = MVector(
        -cameraMatrix[2][0],
        -cameraMatrix[2][1],
        -cameraMatrix[2][2]);

    cameraWorldRight.normalize();
    cameraWorldUp.normalize();
    cameraWorldForward.normalize();

    return true;
}

bool GaussianSplattingSubSceneOverride::cameraChanged(
    const MPoint& cameraWorldPosition,
    const MVector& cameraWorldForward) const
{
    if (!m_haveLastCamera)
    {
        return true;
    }

    const double positionDelta =
        cameraWorldPosition.distanceTo(m_lastCameraPosition);

    const double directionDelta =
        1.0 - std::abs(cameraWorldForward.normal() * m_lastCameraForward.normal());

    return positionDelta > 0.001 || directionDelta > 0.0001;
}

void GaussianSplattingSubSceneOverride::buildVertexBuffer(
    const std::vector<GS::SplatVertex>& vertices)
{
    m_positionBuffer.reset();
    m_colorBuffer.reset();
    m_uvBuffer.reset();

    const unsigned int vertexCount =
        static_cast<unsigned int>(vertices.size());

    if (vertexCount == 0)
    {
        return;
    }

    MVertexBufferDescriptor positionDesc(
        "",
        MGeometry::kPosition,
        MGeometry::kFloat,
        3);

    MVertexBufferDescriptor colorDesc(
        "",
        MGeometry::kColor,
        MGeometry::kFloat,
        4);

    MVertexBufferDescriptor uvDesc(
        "",
        MGeometry::kTexture,
        MGeometry::kFloat,
        2);

    m_positionBuffer =
        std::make_unique<MHWRender::MVertexBuffer>(positionDesc);

    m_colorBuffer =
        std::make_unique<MHWRender::MVertexBuffer>(colorDesc);

    m_uvBuffer =
        std::make_unique<MHWRender::MVertexBuffer>(uvDesc);

    float* positions =
        static_cast<float*>(m_positionBuffer->acquire(vertexCount, true));

    float* colors =
        static_cast<float*>(m_colorBuffer->acquire(vertexCount, true));

    float* uvs =
        static_cast<float*>(m_uvBuffer->acquire(vertexCount, true));

    if (!positions || !colors || !uvs)
    {
        return;
    }

    for (unsigned int i = 0; i < vertexCount; ++i)
    {
        const GS::SplatVertex& vertex = vertices[i];

        positions[i * 3 + 0] = vertex.position[0];
        positions[i * 3 + 1] = vertex.position[1];
        positions[i * 3 + 2] = vertex.position[2];

        colors[i * 4 + 0] = vertex.color[0];
        colors[i * 4 + 1] = vertex.color[1];
        colors[i * 4 + 2] = vertex.color[2];
        colors[i * 4 + 3] = vertex.color[3];

        uvs[i * 2 + 0] = vertex.uv[0];
        uvs[i * 2 + 1] = vertex.uv[1];
    }

    m_positionBuffer->commit(positions);
    m_colorBuffer->commit(colors);
    m_uvBuffer->commit(uvs);
}

void GaussianSplattingSubSceneOverride::buildIndexBuffer(
    const std::vector<unsigned int>& indices)
{
    m_indexBuffer.reset();

    const unsigned int indexCount =
        static_cast<unsigned int>(indices.size());

    if (indexCount == 0)
    {
        return;
    }

    m_indexBuffer =
        std::make_unique<MIndexBuffer>(MGeometry::kUnsignedInt32);

    unsigned int* dst =
        static_cast<unsigned int*>(m_indexBuffer->acquire(indexCount, true));

    if (!dst)
    {
        return;
    }

    for (unsigned int i = 0; i < indexCount; ++i)
    {
        dst[i] = indices[i];
    }

    m_indexBuffer->commit(dst);
}

float GaussianSplattingSubSceneOverride::depthFromCamera(
    const MPoint& worldPoint,
    const MPoint& cameraWorldPosition,
    const MVector& cameraWorldForward)
{
    const MVector toPoint = worldPoint - cameraWorldPosition;
    return static_cast<float>(toPoint * cameraWorldForward);
}


void GaussianSplattingSubSceneOverride::rebuildSortedIndexBufferOnly(
    const MPoint& cameraWorldPosition,
    const MVector& cameraWorldForward)
{
    // Create splat IDs
    std::vector<unsigned int> sortedSplatIds;
    sortedSplatIds.reserve(m_splats.size());

    for (unsigned int i = 0;
        i < static_cast<unsigned int>(m_splats.size());
        ++i)
    {
        sortedSplatIds.push_back(i);
    }


    // Object -> world matrix
    MMatrix objectToWorld;

    if (m_dagPath.isValid())
    {
        objectToWorld =
            m_dagPath.inclusiveMatrix();
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
                m_splats[a].center * objectToWorld;

            const MPoint worldB =
                m_splats[b].center * objectToWorld;

            const double depthA =
                depthFromCamera(
                    worldA,
                    cameraWorldPosition,
                    cameraWorldForward
                );

            const double depthB =
                depthFromCamera(
                    worldB,
                    cameraWorldPosition,
                    cameraWorldForward
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
        m_splats.size() * VerticesPerSplat
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

    uploadIndexBuffer(indices);

    m_indexCount =
        static_cast<unsigned int>(
            indices.size()
            );

    m_indexBufferDirty = false;


    // Save camera state
    m_haveLastCamera = true;

    m_lastCameraPosition =
        cameraWorldPosition;

    m_lastCameraForward =
        cameraWorldForward;
}


void GaussianSplattingSubSceneOverride::uploadVertexBuffers(
    const std::vector<GS::SplatVertex>& vertices)
{
    m_positionBuffer.reset();
    m_colorBuffer.reset();
    m_uvBuffer.reset();

    const unsigned int vertexCount =
        static_cast<unsigned int>(vertices.size());

    if (vertexCount == 0)
    {
        m_vertexCount = 0;
        return;
    }

    MHWRender::MVertexBufferDescriptor positionDesc(
        "",
        MHWRender::MGeometry::kPosition,
        MHWRender::MGeometry::kFloat,
        3);

    MHWRender::MVertexBufferDescriptor colorDesc(
        "",
        MHWRender::MGeometry::kColor,
        MHWRender::MGeometry::kFloat,
        4);

    MHWRender::MVertexBufferDescriptor uvDesc(
        "",
        MHWRender::MGeometry::kTexture,
        MHWRender::MGeometry::kFloat,
        2);

    m_positionBuffer =
        std::make_unique<MHWRender::MVertexBuffer>(positionDesc);

    m_colorBuffer =
        std::make_unique<MHWRender::MVertexBuffer>(colorDesc);

    m_uvBuffer =
        std::make_unique<MHWRender::MVertexBuffer>(uvDesc);

    float* positions =
        static_cast<float*>(m_positionBuffer->acquire(vertexCount, true));

    float* colors =
        static_cast<float*>(m_colorBuffer->acquire(vertexCount, true));

    float* uvs =
        static_cast<float*>(m_uvBuffer->acquire(vertexCount, true));

    if (!positions || !colors || !uvs)
    {
        m_positionBuffer.reset();
        m_colorBuffer.reset();
        m_uvBuffer.reset();

        m_vertexCount = 0;
        return;
    }

    for (unsigned int i = 0; i < vertexCount; ++i)
    {
        const GS::SplatVertex& vertex = vertices[i];

        positions[i * 3 + 0] = vertex.position[0];
        positions[i * 3 + 1] = vertex.position[1];
        positions[i * 3 + 2] = vertex.position[2];

        colors[i * 4 + 0] = vertex.color[0];
        colors[i * 4 + 1] = vertex.color[1];
        colors[i * 4 + 2] = vertex.color[2];
        colors[i * 4 + 3] = vertex.color[3];

        uvs[i * 2 + 0] = vertex.uv[0];
        uvs[i * 2 + 1] = vertex.uv[1];
    }

    m_positionBuffer->commit(positions);
    m_colorBuffer->commit(colors);
    m_uvBuffer->commit(uvs);

    m_vertexCount = vertexCount;
}


bool GaussianSplattingSubSceneOverride::cameraChangedEnoughForIndexRebuild(
    const MPoint& cameraWorldPosition,
    const MVector& cameraWorldForward) const
{
    if (!m_haveLastCamera)
    {
        return true;
    }

    const double positionDelta =
        cameraWorldPosition.distanceTo(m_lastCameraPosition);

    const MVector currentForward = cameraWorldForward.normal();
    const MVector lastForward = m_lastCameraForward.normal();

    const double directionDot =
        std::max(-1.0, std::min(1.0, currentForward * lastForward));

    const double directionDelta =
        1.0 - std::abs(directionDot);

    // Tune these values.
    // Larger values = fewer sorts, faster viewport, less accurate transparency.
    constexpr double kPositionThreshold = 0.05;
    constexpr double kDirectionThreshold = 0.002;

    return
        positionDelta > kPositionThreshold ||
        directionDelta > kDirectionThreshold;
}


void GaussianSplattingSubSceneOverride::uploadIndexBuffer(
    const std::vector<unsigned int>& indices)
{
    m_indexBuffer.reset();

    const unsigned int indexCount =
        static_cast<unsigned int>(indices.size());

    if (indexCount == 0)
    {
        m_indexCount = 0;
        return;
    }

    m_indexBuffer =
        std::make_unique<MHWRender::MIndexBuffer>(
            MHWRender::MGeometry::kUnsignedInt32);

    unsigned int* dst =
        static_cast<unsigned int*>(m_indexBuffer->acquire(indexCount, true));

    if (!dst)
    {
        m_indexBuffer.reset();
        m_indexCount = 0;
        return;
    }

    for (unsigned int i = 0; i < indexCount; ++i)
    {
        dst[i] = indices[i];
    }

    m_indexBuffer->commit(dst);

    m_indexCount = indexCount;
}


void GaussianSplattingSubSceneOverride::buildStaticVertexBuffersOnce()
{
    std::vector<GS::SplatVertex> vertices;

    // Two triangles / 6 vertices per splat.
    vertices.reserve(m_splats.size() * 6);

    m_boundingBox.clear();

    // Camera is common to all splats.
    MPoint cameraPosition;
    MVector cameraRight;
    MVector cameraUp;
    MVector cameraForward;

    readCamera(
        cameraPosition,
        cameraRight,
        cameraUp,
        cameraForward
    );

    cameraRight.normalize();
    cameraUp.normalize();
    cameraForward.normalize();

    for (const GS::GaussianSplat& splat : m_splats)
    {
        buildSplatVertices(
            splat,
            cameraRight,
            cameraUp,
            vertices
        );
    }

    uploadVertexBuffers(vertices);

    m_vertexCount =
        static_cast<unsigned int>(vertices.size());

    m_vertexBufferDirty = false;
}

void GaussianSplattingSubSceneOverride::buildSplatVertices(
    const GS::GaussianSplat& splat,
    const MVector& cameraRight,
    const MVector& cameraUp,
    std::vector<GS::SplatVertex>& vertices)
{

    // Build covariance from rotation + scale.
    const MMatrix covariance =
        buildCovariance(splat);


    // Project the 3D covariance onto the camera plane.
    const MMatrix covariance2D =
        projectCovarianceToCamera(
            covariance,
            cameraRight,
            cameraUp
        );


    // Convert the 2D covariance into ellipse axes.
    constexpr double sigmaMultiplier = 3.0;

    const ProjectedEllipse ellipse =
        calculateEllipseAxes(
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
        vertices
    );
}

// Build 3D Gaussian covariance
// Sigma = R * S * S^T * R^T
MMatrix GaussianSplattingSubSceneOverride::buildCovariance(
    const GS::GaussianSplat& splat) const
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
        static_cast<double>(splat.scaleX) * m_splatSize;

    const double sy =
        static_cast<double>(splat.scaleY) * m_splatSize;

    const double sz =
        static_cast<double>(splat.scaleZ) * m_splatSize;

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

MMatrix GaussianSplattingSubSceneOverride::projectCovarianceToCamera(
    const MMatrix& covariance,
    const MVector& cameraRight,
    const MVector& cameraUp) const
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
double GaussianSplattingSubSceneOverride::covarianceQuadraticForm(
    const MMatrix& covariance,
    const MVector& v,
    const MVector& w) const
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
ProjectedEllipse
GaussianSplattingSubSceneOverride::calculateEllipseAxes(
    const MMatrix& covariance2D,
    const MVector& cameraRight,
    const MVector& cameraUp,
    double sigmaMultiplier) const
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
void GaussianSplattingSubSceneOverride::appendSplatQuad(
    const GS::GaussianSplat& splat,
    const MVector& axisX,
    const MVector& axisY,
    std::vector<GS::SplatVertex>& vertices)
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
    m_boundingBox.expand(bottomLeft);
    m_boundingBox.expand(bottomRight);
    m_boundingBox.expand(topRight);
    m_boundingBox.expand(topLeft);
}