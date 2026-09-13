#pragma once

#include <maya/MPxCommand.h>
#include <maya/MSyntax.h>

namespace GS::Mesh
{
    // gsSplatToMesh -node <locator> [-resolution int] [-isoLevel float]
    class SplatToMeshCommand : public MPxCommand
    {
    public:
        static const MString kName;
        static void* creator() { return new SplatToMeshCommand(); }
        static MSyntax newSyntax();

        MStatus doIt(const MArgList& args) override;
    };
}
