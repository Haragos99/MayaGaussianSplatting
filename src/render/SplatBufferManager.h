#pragma once

#include <maya/MHWGeometry.h>

#include <memory>
#include <vector>

#include "../data.h"

namespace GS
{
    // Owns the GPU buffers of the splat render item.
    // The override feeds it CPU-side data and never touches MVertexBuffer directly.
    class SplatBufferManager
    {
    public:
        SplatBufferManager() = default;
        ~SplatBufferManager() = default;

        SplatBufferManager(const SplatBufferManager&) = delete;
        SplatBufferManager& operator=(const SplatBufferManager&) = delete;

        // Uploads interleaved splat vertices into the position/color/uv streams.
        bool uploadVertices(const std::vector<SplatVertex>& vertices);

        bool uploadIndices(const std::vector<unsigned int>& indices);

        // True when every stream plus the index buffer holds drawable data.
        bool isRenderable() const;

        bool hasVertices() const;
        bool hasIndices() const;

        unsigned int vertexCount() const { return m_vertexCount; }
        unsigned int indexCount() const { return m_indexCount; }

        // Collects the vertex streams for MPxSubSceneOverride::setGeometryForRenderItem.
        bool fillVertexBufferArray(MHWRender::MVertexBufferArray& buffers) const;

        MHWRender::MIndexBuffer* indexBuffer() const { return m_indexBuffer.get(); }

        void releaseVertices();
        void releaseIndices();
        void releaseAll();

    private:
        std::unique_ptr<MHWRender::MVertexBuffer> m_positionBuffer;
        std::unique_ptr<MHWRender::MVertexBuffer> m_colorBuffer;
        std::unique_ptr<MHWRender::MVertexBuffer> m_uvBuffer;
        std::unique_ptr<MHWRender::MIndexBuffer>  m_indexBuffer;

        unsigned int m_vertexCount = 0;
        unsigned int m_indexCount = 0;
    };
}
