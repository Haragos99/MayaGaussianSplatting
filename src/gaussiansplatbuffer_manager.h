#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include <maya/MHWGeometry.h>

#include "data.h"

namespace GS
{

// Owns the GPU buffers used by the current quad-based splat renderer.
// The renderer can later extend this class with quaternion/scale streams when
// the covariance projection is moved fully into the vertex shader.
class GaussianSplatBufferManager final
{
public:
    using Vertex = SplatVertex;
    using Index = unsigned int;

    GaussianSplatBufferManager() = default;
    ~GaussianSplatBufferManager() = default;

    GaussianSplatBufferManager(const GaussianSplatBufferManager&) = delete;
    GaussianSplatBufferManager& operator=(const GaussianSplatBufferManager&) = delete;

    void clear();

    bool uploadVertices(const std::vector<Vertex>& vertices);
    bool uploadIndices(const std::vector<Index>& indices);

    void buildVertexBufferArray(MHWRender::MVertexBufferArray& out) const;

    MHWRender::MVertexBuffer* positionBuffer() const noexcept { return m_positionBuffer.get(); }
    MHWRender::MVertexBuffer* colorBuffer() const noexcept { return m_colorBuffer.get(); }
    MHWRender::MVertexBuffer* uvBuffer() const noexcept { return m_uvBuffer.get(); }
    MHWRender::MIndexBuffer* indexBuffer() const noexcept { return m_indexBuffer.get(); }

    std::size_t vertexCount() const noexcept { return m_vertexCount; }
    std::size_t indexCount() const noexcept { return m_indexCount; }

    bool ready() const noexcept;

private:
    std::unique_ptr<MHWRender::MVertexBuffer> m_positionBuffer;
    std::unique_ptr<MHWRender::MVertexBuffer> m_colorBuffer;
    std::unique_ptr<MHWRender::MVertexBuffer> m_uvBuffer;
    std::unique_ptr<MHWRender::MIndexBuffer> m_indexBuffer;

    std::size_t m_vertexCount = 0;
    std::size_t m_indexCount = 0;
};

} // namespace GS
