#include "gaussiansplatbuffer_manager.h"

#include <algorithm>

namespace GS
{

void GaussianSplatBufferManager::clear()
{
    m_positionBuffer.reset();
    m_colorBuffer.reset();
    m_uvBuffer.reset();
    m_indexBuffer.reset();

    m_vertexCount = 0;
    m_indexCount = 0;
}

bool GaussianSplatBufferManager::uploadVertices(
    const std::vector<Vertex>& vertices)
{
    m_positionBuffer.reset();
    m_colorBuffer.reset();
    m_uvBuffer.reset();
    m_vertexCount = 0;

    if (vertices.empty())
    {
        return true;
    }

    const unsigned int count =
        static_cast<unsigned int>(vertices.size());

    const MHWRender::MVertexBufferDescriptor positionDesc(
        "positions",
        MHWRender::MGeometry::kPosition,
        MHWRender::MGeometry::kFloat,
        3);

    const MHWRender::MVertexBufferDescriptor colorDesc(
        "colors",
        MHWRender::MGeometry::kColor,
        MHWRender::MGeometry::kFloat,
        4);

    const MHWRender::MVertexBufferDescriptor uvDesc(
        "uvs",
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
        static_cast<float*>(m_positionBuffer->acquire(count, true));

    float* colors =
        static_cast<float*>(m_colorBuffer->acquire(count, true));

    float* uvs =
        static_cast<float*>(m_uvBuffer->acquire(count, true));

    if (!positions || !colors || !uvs)
    {
        clear();
        return false;
    }

    for (unsigned int i = 0; i < count; ++i)
    {
        const Vertex& v = vertices[i];

        const unsigned int p = i * 3;
        positions[p + 0] = v.position[0];
        positions[p + 1] = v.position[1];
        positions[p + 2] = v.position[2];

        const unsigned int c = i * 4;
        colors[c + 0] = v.color[0];
        colors[c + 1] = v.color[1];
        colors[c + 2] = v.color[2];
        colors[c + 3] = v.color[3];

        const unsigned int uv = i * 2;
        uvs[uv + 0] = v.uv[0];
        uvs[uv + 1] = v.uv[1];
    }

    m_positionBuffer->commit(positions);
    m_colorBuffer->commit(colors);
    m_uvBuffer->commit(uvs);

    m_vertexCount = count;
    return true;
}

bool GaussianSplatBufferManager::uploadIndices(
    const std::vector<Index>& indices)
{
    m_indexBuffer.reset();
    m_indexCount = 0;

    if (indices.empty())
    {
        return true;
    }

    const unsigned int count =
        static_cast<unsigned int>(indices.size());

    m_indexBuffer =
        std::make_unique<MHWRender::MIndexBuffer>(
            MHWRender::MGeometry::kUnsignedInt32);

    auto* dst =
        static_cast<unsigned int*>(
            m_indexBuffer->acquire(count, true));

    if (!dst)
    {
        m_indexBuffer.reset();
        return false;
    }

    std::copy(indices.begin(), indices.end(), dst);
    m_indexBuffer->commit(dst);

    m_indexCount = count;
    return true;
}

void GaussianSplatBufferManager::buildVertexBufferArray(
    MHWRender::MVertexBufferArray& out) const
{
    out.addBuffer("positions", m_positionBuffer.get());
    out.addBuffer("colors", m_colorBuffer.get());
    out.addBuffer("uvs", m_uvBuffer.get());
}

bool GaussianSplatBufferManager::ready() const noexcept
{
    return
        m_positionBuffer &&
        m_colorBuffer &&
        m_uvBuffer &&
        m_indexBuffer &&
        m_vertexCount > 0 &&
        m_indexCount > 0;
}

} // namespace GS
