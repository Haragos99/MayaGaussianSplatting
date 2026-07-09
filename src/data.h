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
    struct GaussianSplat
    {
        MPoint center;      // Object-space center.
        MColor color;       // RGBA color.
        float scaleX;       // Billboard half-size X.
        float scaleY;       // Billboard half-size Y.
        float opacity;      // 0..1.
    };

    struct SplatVertex
    {
        float position[3];
        float color[4];
        float uv[2];
    };
}