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

        // Uploads the camera independent splat streams. Only has to run when
        // the splat data or the splat size changes, never on camera movement.
        bool uploadVertices(const std::vector<SplatVertex>& vertices);

        // Expands a back-to-front quad order into triangle indices, writing
        // directly into the mapped GPU buffer.
        bool uploadQuadIndices(const std::vector<unsigned int>& quadOrder);

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
        // Each stream uses a distinct semantic so Maya can bind it to the
        // matching GaussianSplat.ogsfx attribute. NORMAL and TANGENT are used
        // as raw float3 channels, they do not carry a normal or a tangent.
        std::unique_ptr<MHWRender::MVertexBuffer> m_centerBuffer;   // POSITION
        std::unique_ptr<MHWRender::MVertexBuffer> m_covABuffer;     // NORMAL
        std::unique_ptr<MHWRender::MVertexBuffer> m_covBBuffer;     // TANGENT
        std::unique_ptr<MHWRender::MVertexBuffer> m_cornerBuffer;   // TEXCOORD0
        std::unique_ptr<MHWRender::MVertexBuffer> m_colorBuffer;    // COLOR0
        std::unique_ptr<MHWRender::MIndexBuffer>  m_indexBuffer;

        unsigned int m_vertexCount = 0;
        unsigned int m_indexCount = 0;
    };
}
