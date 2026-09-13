#pragma once

#include <maya/MFloatPointArray.h>
#include <maya/MIntArray.h>
#include <maya/MVectorArray.h>

namespace GS::Mesh
{
    // Triangle soup in the layout MFnMesh::create expects.
    struct MeshData
    {
        MFloatPointArray points;
        MVectorArray normals;
        MIntArray polygonCounts;
        MIntArray polygonConnects;

        bool isEmpty() const { return polygonCounts.length() == 0; }
    };
}
