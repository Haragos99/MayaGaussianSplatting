#pragma once

#include "MeshData.h"

#include <maya/MObject.h>
#include <maya/MString.h>

namespace GS::Mesh
{
    // Turns generated geometry into a DAG mesh. Must not be called from a
    // compute() or from the viewport draw pass.
    class MayaMeshBuilder
    {
    public:
        static MObject create(const MeshData& data, const MString& baseName);
    };
}
