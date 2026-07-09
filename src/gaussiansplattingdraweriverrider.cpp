#include "gaussiansplattingdraweriverrider.h"
#include <maya/MFnTypedAttribute.h>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include "gaussianSplatPlyLoader.h"

MTypeId GaussianSplattingLocator::id(0x7802aaaa);
MObject GaussianSplattingLocator::locatorMsgAttr;

MStatus GaussianSplattingLocator::connectionMade(const MPlug& plug, const MPlug& otherPlug, bool asSrc)
{
    // Check if the connection is made to the locatorMsgAttr
    if (plug == locatorMsgAttr || otherPlug == locatorMsgAttr) {
        MGlobal::displayInfo("Connection made to locatorMsgAttr.");
    }
	return MPxLocatorNode::connectionMade(plug, otherPlug, asSrc);
}

MStatus GaussianSplattingLocator::initialize()
{
    MFnTypedAttribute typedAttr;
    locatorMsgAttr = typedAttr.create("locatorMsg", "locatorMsg", MFnData::kString);
	addAttribute(locatorMsgAttr);

    return MS::kSuccess;
}


//////////////////////////////////////////////////////////////////////////////////
const MString GsTestSubSceneOverride::kRenderItemName("gaussianSplatRenderItem");



GsTestSubSceneOverride::GsTestSubSceneOverride(const MObject& obj)
    : MPxSubSceneOverride(obj)
    , m_nodeObj(obj)
{
    MStatus status;
    MFnDagNode dagNode(obj, &status);

    if (status)
    {
        dagNode.getPath(m_dagPath);
    }

    loadSplatsFromNodeOrDemoData();
}

GsTestSubSceneOverride::~GsTestSubSceneOverride()
{
    releaseShader();

    m_positionBuffer.reset();
    m_colorBuffer.reset();
    m_uvBuffer.reset();
    m_indexBuffer.reset();
}

MHWRender::DrawAPI GsTestSubSceneOverride::supportedDrawAPIs() const
{
    return MHWRender::kOpenGLCoreProfile | MHWRender::kDirectX11;
}

bool GsTestSubSceneOverride::requiresUpdate(
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

    const bool cameraIsDifferent =
        haveCamera && cameraChanged(cameraPosition, cameraForward);

    return
        container.count() == 0 ||
        m_dirty ||
        m_geometryDirty ||
        m_shaderDirty ||
        cameraIsDifferent;
}

void GsTestSubSceneOverride::update(
    MHWRender::MSubSceneContainer& container,
    const MHWRender::MFrameContext& frameContext)
{
    createOrUpdateRenderItem(container);

    MRenderItem* item = container.find(kRenderItemName);
    if (!item)
    {
        return;
    }

    if (!m_shader || m_shaderDirty)
    {
        createShader();
    }

    if (m_shader)
    {
        item->setShader(m_shader);
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
        cameraRight = MVector(1.0, 0.0, 0.0);
        cameraUp = MVector(0.0, 1.0, 0.0);
        cameraForward = MVector(0.0, 0.0, -1.0);
    }

    const bool mustRebuildGeometry =
        m_geometryDirty ||
        cameraChanged(cameraPosition, cameraForward) ||
        !m_positionBuffer ||
        !m_colorBuffer ||
        !m_uvBuffer ||
        !m_indexBuffer;

    if (mustRebuildGeometry)
    {
        buildGeometry(
            cameraPosition,
            cameraRight,
            cameraUp,
            cameraForward);
    }

    if (
        m_positionBuffer &&
        m_colorBuffer &&
        m_uvBuffer &&
        m_indexBuffer &&
        m_vertexCount > 0 &&
        m_indexCount > 0)
    {
        MVertexBufferArray vertexBuffers;

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
    m_geometryDirty = false;
    m_shaderDirty = false;
    m_uiDirty = true;

    m_haveLastCamera = true;
    m_lastCameraPosition = cameraPosition;
    m_lastCameraForward = cameraForward;
}

bool GsTestSubSceneOverride::furtherUpdateRequired(
    const MHWRender::MFrameContext& frameContext)
{
    // Return true only if you are doing progressive/async loading.
    // Be careful: always returning true can hurt Maya interactivity.
    return false;
}

bool GsTestSubSceneOverride::hasUIDrawables() const
{
    return true;
}

bool GsTestSubSceneOverride::areUIDrawablesDirty() const
{
    return m_uiDirty;
}

void GsTestSubSceneOverride::addUIDrawables(
    MHWRender::MUIDrawManager& drawManager,
    const MHWRender::MFrameContext& frameContext)
{
    drawManager.beginDrawable();

    drawManager.setColor(MColor(1.0f, 0.8f, 0.1f, 1.0f));

    MString label;
    label += "Gaussian splats: ";
    label += static_cast<int>(m_splats.size());

    drawManager.text(
        MPoint(0.0, 1.5, 0.0),
        label,
        MHWRender::MUIDrawManager::kCenter);

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

bool GsTestSubSceneOverride::enableUpdateForSelection() const
{
    return true;
}

bool GsTestSubSceneOverride::getSelectionPath(
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

bool GsTestSubSceneOverride::getInstancedSelectionPath(
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

void GsTestSubSceneOverride::updateSelectionGranularity(
    const MDagPath& path,
    MHWRender::MSelectionContext& selectionContext)
{
    selectionContext.setSelectionLevel(MHWRender::MSelectionContext::kObject);
}

void GsTestSubSceneOverride::markDirty()
{
    m_dirty = true;
    m_geometryDirty = true;
    m_uiDirty = true;
}

void GsTestSubSceneOverride::createOrUpdateRenderItem(
    MHWRender::MSubSceneContainer& container)
{
    MRenderItem* item = container.find(kRenderItemName);

    if (!item)
    {
        item = MRenderItem::Create(
            kRenderItemName,
            MRenderItem::MaterialSceneItem,
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

void GsTestSubSceneOverride::createShader()
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

    // Simple stock shader fallback.
    // For real Gaussian splatting, replace this with an effect/custom shader
    // that reads UV/color streams and computes:
    //
    //     alpha = exp(-dot(uv, uv) * falloff) * opacity
    //
    m_shader = shaderManager->getStockShader(MShaderManager::k3dCPVSolidShader);
    if (m_shader)
    {
        float solidColor[4] = { 0.0f, 1.0f, 0.8f, 0.45f };
        m_shader->setParameter("solidColor", solidColor);
    }
}

void GsTestSubSceneOverride::releaseShader()
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


void GsTestSubSceneOverride::loadSplatsFromNodeOrDemoData()
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

bool GsTestSubSceneOverride::readCamera(
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

bool GsTestSubSceneOverride::cameraChanged(
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

void GsTestSubSceneOverride::buildGeometry(
    const MPoint& cameraWorldPosition,
    const MVector& cameraWorldRight,
    const MVector& cameraWorldUp,
    const MVector& cameraWorldForward)
{
    std::vector<size_t> order;
    order.reserve(m_splats.size());

    for (size_t i = 0; i < m_splats.size(); ++i)
    {
        order.push_back(i);
    }

    MMatrix objectToWorld;
    MMatrix worldToObject;

    if (m_dagPath.isValid())
    {
        objectToWorld = m_dagPath.inclusiveMatrix();
        worldToObject = objectToWorld.inverse();
    }

    std::sort(
        order.begin(),
        order.end(),
        [&](size_t a, size_t b)
        {
            const MPoint worldA = m_splats[a].center * objectToWorld;
            const MPoint worldB = m_splats[b].center * objectToWorld;

            const float depthA = depthFromCamera(
                worldA,
                cameraWorldPosition,
                cameraWorldForward);

            const float depthB = depthFromCamera(
                worldB,
                cameraWorldPosition,
                cameraWorldForward);

            // Back-to-front for transparency.
            return depthA > depthB;
        });

    MVector localRight = cameraWorldRight * worldToObject;
    MVector localUp = cameraWorldUp * worldToObject;

    localRight.normalize();
    localUp.normalize();

    std::vector<GS::SplatVertex> vertices;
    std::vector<unsigned int> indices;

    constexpr unsigned int CircleSegments = 16;

    vertices.reserve(m_splats.size() * (CircleSegments + 1));
    indices.reserve(m_splats.size() * CircleSegments * 3);

    m_boundingBox.clear();

    for (size_t sortedIndex = 0; sortedIndex < order.size(); ++sortedIndex)
    {
        const GS::GaussianSplat& splat = m_splats[order[sortedIndex]];

        const unsigned int base =
            static_cast<unsigned int>(vertices.size());

        const MVector rx = localRight * splat.scaleX / 10;
        const MVector uy = localUp * splat.scaleY / 10;

        const float alpha = std::clamp(splat.opacity, 0.0f, 1.0f);

        auto makeVertex =
            [&](const MPoint& p, float u, float v) -> GS::SplatVertex
            {
                GS::SplatVertex vertex;

                vertex.position[0] = static_cast<float>(p.x);
                vertex.position[1] = static_cast<float>(p.y);
                vertex.position[2] = static_cast<float>(p.z);

                vertex.color[0] = splat.color.r;
                vertex.color[1] = splat.color.g;
                vertex.color[2] = splat.color.b;
                vertex.color[3] = alpha;

                vertex.uv[0] = u;
                vertex.uv[1] = v;

                return vertex;
            };

        // Center vertex
        vertices.push_back(makeVertex(splat.center, 0.0f, 0.0f));

        // Rim vertices
        for (unsigned int i = 0; i < CircleSegments; ++i)
        {
            const float angle = static_cast<float>(2.0 * M_PI * i / CircleSegments);

            const float c = std::cos(angle);
            const float s = std::sin(angle);

            MPoint p = splat.center + rx * c + uy * s;

            vertices.push_back(makeVertex(p, c, s));

            m_boundingBox.expand(p);
        }

        m_boundingBox.expand(splat.center);

        // Triangle fan
        for (unsigned int i = 0; i < CircleSegments; ++i)
        {
            indices.push_back(base);
            indices.push_back(base + 1 + i);
            indices.push_back(base + 1 + ((i + 1) % CircleSegments));
        }
    }

    buildVertexBuffer(vertices);
    buildIndexBuffer(indices);

    m_vertexCount = static_cast<unsigned int>(vertices.size());
    m_indexCount = static_cast<unsigned int>(indices.size());
}

void GsTestSubSceneOverride::buildVertexBuffer(
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

void GsTestSubSceneOverride::buildIndexBuffer(
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

float GsTestSubSceneOverride::depthFromCamera(
    const MPoint& worldPoint,
    const MPoint& cameraWorldPosition,
    const MVector& cameraWorldForward)
{
    const MVector toPoint = worldPoint - cameraWorldPosition;
    return static_cast<float>(toPoint * cameraWorldForward);
}