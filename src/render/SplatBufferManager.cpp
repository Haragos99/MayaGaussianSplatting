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

        const MHWRender::MVertexBufferDescriptor centerDesc(
            "",
            MHWRender::MGeometry::kPosition,
            MHWRender::MGeometry::kFloat,
            3);

        const MHWRender::MVertexBufferDescriptor covADesc(
            "",
            MHWRender::MGeometry::kNormal,
            MHWRender::MGeometry::kFloat,
            3);

        const MHWRender::MVertexBufferDescriptor covBDesc(
            "",
            MHWRender::MGeometry::kTangent,
            MHWRender::MGeometry::kFloat,
            3);

        const MHWRender::MVertexBufferDescriptor cornerDesc(
            "",
            MHWRender::MGeometry::kTexture,
            MHWRender::MGeometry::kFloat,
            2);

        const MHWRender::MVertexBufferDescriptor colorDesc(
            "",
            MHWRender::MGeometry::kColor,
            MHWRender::MGeometry::kFloat,
            4);

        m_centerBuffer =
            std::make_unique<MHWRender::MVertexBuffer>(centerDesc);

        m_covABuffer =
            std::make_unique<MHWRender::MVertexBuffer>(covADesc);

        m_covBBuffer =
            std::make_unique<MHWRender::MVertexBuffer>(covBDesc);

        m_cornerBuffer =
            std::make_unique<MHWRender::MVertexBuffer>(cornerDesc);

        m_colorBuffer =
            std::make_unique<MHWRender::MVertexBuffer>(colorDesc);

        float* centers =
            static_cast<float*>(m_centerBuffer->acquire(vertexCount, true));

        float* covA =
            static_cast<float*>(m_covABuffer->acquire(vertexCount, true));

        float* covB =
            static_cast<float*>(m_covBBuffer->acquire(vertexCount, true));

        float* corners =
            static_cast<float*>(m_cornerBuffer->acquire(vertexCount, true));

        float* colors =
            static_cast<float*>(m_colorBuffer->acquire(vertexCount, true));

        if (!centers || !covA || !covB || !corners || !colors)
        {
            releaseVertices();
            return false;
        }

        for (unsigned int i = 0; i < vertexCount; ++i)
        {
            const SplatVertex& vertex = vertices[i];

            centers[i * 3 + 0] = vertex.center[0];
            centers[i * 3 + 1] = vertex.center[1];
            centers[i * 3 + 2] = vertex.center[2];

            covA[i * 3 + 0] = vertex.covA[0];
            covA[i * 3 + 1] = vertex.covA[1];
            covA[i * 3 + 2] = vertex.covA[2];

            covB[i * 3 + 0] = vertex.covB[0];
            covB[i * 3 + 1] = vertex.covB[1];
            covB[i * 3 + 2] = vertex.covB[2];

            corners[i * 2 + 0] = vertex.corner[0];
            corners[i * 2 + 1] = vertex.corner[1];

            colors[i * 4 + 0] = vertex.color[0];
            colors[i * 4 + 1] = vertex.color[1];
            colors[i * 4 + 2] = vertex.color[2];
            colors[i * 4 + 3] = vertex.color[3];
        }

        m_centerBuffer->commit(centers);
        m_covABuffer->commit(covA);
        m_covBBuffer->commit(covB);
        m_cornerBuffer->commit(corners);
        m_colorBuffer->commit(colors);

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
            m_centerBuffer &&
            m_covABuffer &&
            m_covBBuffer &&
            m_cornerBuffer &&
            m_colorBuffer &&
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

        buffers.addBuffer("centers", m_centerBuffer.get());
        buffers.addBuffer("covA", m_covABuffer.get());
        buffers.addBuffer("covB", m_covBBuffer.get());
        buffers.addBuffer("corners", m_cornerBuffer.get());
        buffers.addBuffer("colors", m_colorBuffer.get());

        return true;
    }


    void SplatBufferManager::releaseVertices()
    {
        m_centerBuffer.reset();
        m_covABuffer.reset();
        m_covBBuffer.reset();
        m_cornerBuffer.reset();
        m_colorBuffer.reset();

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
