#include <maya/MFnPlugin.h>
#include <maya/MDrawRegistry.h>
#include "gaussiansplattingnode.h"
#include "gaussiansplattingdraweriverrider.h"
#include "data.h"
#include "meshconverter/SplatToMeshCommand.h"



MStatus initializePlugin(MObject obj)
{
	MFnPlugin plugin(obj, "Haragos", "1.0", "Any");
    MStatus status;
    const MString myClassification("drawdb/subscene/GaussianSplattingLocator");
    status = plugin.registerNode(
        "GaussianSplattingLocator",               // name used in Maya (createNode myLocator)
        GaussianSplattingLocator::id,             // unique type ID
        GaussianSplattingLocator::creator,        // function pointer that creates an instance
        GaussianSplattingLocator::initialize,     // initialization (attributes, etc.)
        MPxNode::kLocatorNode,      // tells Maya it's a locator
        &myClassification
    );
    status.perror("registerNode"); // This will print an error if registration fails

    status.perror("REWGregisterNode"); // This will print an error if registration fails

    status = MHWRender::MDrawRegistry::registerSubSceneOverrideCreator(
        "drawdb/subscene/GaussianSplattingLocator",
        "GaussianSplattingSubSceneOverride",
        GaussianSplattingSubSceneOverride::creator);

    status = plugin.registerCommand(
        GS::Mesh::SplatToMeshCommand::kName,
        GS::Mesh::SplatToMeshCommand::creator,
        GS::Mesh::SplatToMeshCommand::newSyntax);
    status.perror("registerCommand gsSplatToMesh");

	return MS::kSuccess;
}


MStatus uninitializePlugin(MObject obj)
{
	MFnPlugin plugin(obj);
    plugin.deregisterCommand(GS::Mesh::SplatToMeshCommand::kName);
    plugin.deregisterNode(GaussianSplattingLocator::id);

    MHWRender::MDrawRegistry::deregisterSubSceneOverrideCreator(
        "drawdb/subscene/GaussianSplattingLocator",
        "GaussianSplattingSubSceneOverride");

	return MS::kSuccess;
}