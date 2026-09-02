#include "SplatBufferManager.h"

namespace GS
{
    bool SplatBufferManager::uploadVertices(
        const std::vector<SplatVertex>& vertices)
    {
        releaseVertices();

        const unsigned int vertexCount =
            static_cast<unsigned int>(vertices.size());

        if (vertexCount == 0)
        {
            return false;
        }

        const MHWRender::MVertexBufferDescriptor positionDesc(
            "",
            MHWRender::MGeometry::kPosition,
            MHWRender::MGeometry::kFloat,
            3);

        const MHWRender::MVertexBufferDescriptor colorDesc(
            "",
            MHWRender::MGeometry::kColor,
            MHWRender::MGeometry::kFloat,
            4);

        const MHWRender::MVertexBufferDescriptor uvDesc(
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
            releaseVertices();
            return false;
        }

        for (unsigned int i = 0; i < vertexCount; ++i)
        {
            const SplatVertex& vertex = vertices[i];

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

        return true;
    }


    bool SplatBufferManager::uploadQuadIndices(
        const std::vector<unsigned int>& quadOrder)
    {
        releaseIndices();

        const unsigned int quadCount =
            static_cast<unsigned int>(quadOrder.size());

        if (quadCount == 0)
        {
            return false;
        }

        const unsigned int indexCount =
            quadCount * kIndicesPerSplatQuad;

        m_indexBuffer =
            std::make_unique<MHWRender::MIndexBuffer>(
                MHWRender::MGeometry::kUnsignedInt32);

        unsigned int* destination =
            static_cast<unsigned int*>(m_indexBuffer->acquire(indexCount, true));

        if (!destination)
        {
            releaseIndices();
            return false;
        }

        for (unsigned int i = 0; i < quadCount; ++i)
        {
            const unsigned int base =
                quadOrder[i] * kVerticesPerSplatQuad;

            unsigned int* triangles =
                destination + i * kIndicesPerSplatQuad;

            triangles[0] = base + 0;
            triangles[1] = base + 1;
            triangles[2] = base + 2;

            triangles[3] = base + 0;
            triangles[4] = base + 2;
            triangles[5] = base + 3;
        }

        m_indexBuffer->commit(destination);

        m_indexCount = indexCount;

        return true;
    }


    bool SplatBufferManager::hasVertices() const
    {
        return
            m_positionBuffer &&
            m_colorBuffer &&
            m_uvBuffer &&
            m_vertexCount > 0;
    }


    bool SplatBufferManager::hasIndices() const
    {
        return m_indexBuffer && m_indexCount > 0;
    }


    bool SplatBufferManager::isRenderable() const
    {
        return hasVertices() && hasIndices();
    }


    bool SplatBufferManager::fillVertexBufferArray(
        MHWRender::MVertexBufferArray& buffers) const
    {
        if (!hasVertices())
        {
            return false;
        }

        buffers.addBuffer("positions", m_positionBuffer.get());
        buffers.addBuffer("colors", m_colorBuffer.get());
        buffers.addBuffer("uvs", m_uvBuffer.get());

        return true;
    }


    void SplatBufferManager::releaseVertices()
    {
        m_positionBuffer.reset();
        m_colorBuffer.reset();
        m_uvBuffer.reset();

        m_vertexCount = 0;
    }


    void SplatBufferManager::releaseIndices()
    {
        m_indexBuffer.reset();

        m_indexCount = 0;
    }


    void SplatBufferManager::releaseAll()
    {
        releaseVertices();
        releaseIndices();
    }
}
