#include "SplatToMeshCommand.h"

#include <maya/MArgDatabase.h>
#include <maya/MFnDependencyNode.h>
#include <maya/MSelectionList.h>

#include "../gaussiansplattingnode.h"
#include "SplatMeshConverter.h"

namespace GS::Mesh
{
    const MString SplatToMeshCommand::kName("gsSplatToMesh");

    MSyntax SplatToMeshCommand::newSyntax()
    {
        MSyntax syntax;
        syntax.addFlag("-n", "-node", MSyntax::kString);
        syntax.addFlag("-r", "-resolution", MSyntax::kLong);
        syntax.addFlag("-iso", "-isoLevel", MSyntax::kDouble);
        return syntax;
    }

    MStatus SplatToMeshCommand::doIt(const MArgList& args)
    {
        MStatus status;
        MArgDatabase argData(syntax(), args, &status);
        if (!status)
            return status;

        MString nodeName;
        if (argData.isFlagSet("-node"))
            argData.getFlagArgument("-node", 0, nodeName);

        MSelectionList selection;
        if (!selection.add(nodeName) || selection.isEmpty())
        {
            displayError("gsSplatToMesh: node not found: " + nodeName);
            return MS::kFailure;
        }

        MObject node;
        selection.getDependNode(0, node);

        MFnDependencyNode fnNode(node);
        auto* locator = dynamic_cast<GaussianSplattingLocator*>(fnNode.userNode());
        if (!locator)
        {
            displayError("gsSplatToMesh: not a GaussianSplattingLocator: " + nodeName);
            return MS::kFailure;
        }

        locator->syncSplatData();

        ConversionSettings settings;

        if (argData.isFlagSet("-resolution"))
            argData.getFlagArgument("-resolution", 0, settings.resolution);

        if (argData.isFlagSet("-isoLevel"))
        {
            double isoLevel = settings.isoLevel;
            argData.getFlagArgument("-isoLevel", 0, isoLevel);
            settings.isoLevel = static_cast<float>(isoLevel);
        }

        const MObject mesh = SplatMeshConverter::convert(
            locator->splats(),
            locator->splatBounds(),
            settings,
            fnNode.name() + "_mesh");

        return mesh.isNull() ? MS::kFailure : MS::kSuccess;
    }
}
