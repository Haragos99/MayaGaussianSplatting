#pragma once
#include <vector>
#include <maya/MColor.h>
#include <maya/MPoint.h>
struct Vertex
{
    float x, y, z;
};



struct Point
{
    float x, y, z;
    float r, g, b;
};

static std::vector<Point> g_points =
{
    {0.0f, 0.0f, 0.0f, 1, 0, 0},
    {1.0f, 0.0f, 0.0f, 0, 1, 0},
    {0.0f, 1.0f, 0.0f, 0, 0, 1},
};

static const int g_count = 3;

struct Splat {
    float pos[3];
    float scale[3];
    float rot[4];      // quaternion
    float opacity;
    float shDC[3];
    std::vector<float> shRest; // 0, 9, 24, or 45 floats depending on SH degree
};


namespace GS
{
    // One splat quad = 4 shared corners referenced by 2 triangles.
    constexpr unsigned int kVerticesPerSplatQuad = 4;
    constexpr unsigned int kIndicesPerSplatQuad = 6;

    struct GaussianSplat
    {
        MPoint center;      // Object-space center.
        MColor color;       // RGBA color.
        float scaleX;       // Billboard half-size X.
        float scaleY;       // Billboard half-size Y.
		float scaleZ;       // Billboard half-size Z.
        float opacity;      // 0..1.
        float scale[3] = { 0.03f, 0.03f, 0.03f };   // Full 3D scale (linear).
        float rotation[4] = { 1.0f, 0.0f, 0.0f, 0.0f }; // Quaternion (w, x, y, z), normalized.
    };

    struct SplatVertex
    {
        float center[3];    // Splat center, object space. Shared by all 4 corners.
        float covA[3];      // Sigma xx, xy, xz.
        float covB[3];      // Sigma yy, yz, zz.
        float corner[2];    // (+-1, +-1), the only per-vertex value.
        float color[4];
    };
}