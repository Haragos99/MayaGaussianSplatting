#include <maya/MFnPlugin.h>
#include <maya/MDrawRegistry.h>
#include "gaussiansplattingnode.h"
#include "gaussiansplattingdraweriverrider.h"
#include "data.h"



MStatus initializePlugin(MObject obj)
{
	MFnPlugin plugin(obj, "Haragos", "1.0", "Any");
	// Register commands, nodes, etc. here
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

    /*
    status = MDrawRegistry::registerDrawOverrideCreator(
        myClassification,
        "GaussianSplattingDrawOverride",
        GaussianSplattingDrawOverride::Creator
    );
    */
    status.perror("REWGregisterNode"); // This will print an error if registration fails

    status = plugin.registerNode(
        "gaussiansplattingNode",               //  the type name Maya knows
        GaussiansplattingNode::id,             // the unique typeId (MTypeId)
        GaussiansplattingNode::creator,
        GaussiansplattingNode::initialize
    );

    /*
    MHWRender::MDrawRegistry::registerGeometryOverrideCreator(
        "drawdb/geometry/GaussianSplattingLocator",
        "GSDrawOverride",
        GSDrawOverride::creator);
   */
    status = MHWRender::MDrawRegistry::registerSubSceneOverrideCreator(
        "drawdb/subscene/GaussianSplattingLocator",
        "GsTestPlugin",
        GsTestSubSceneOverride::creator);


         

	return MS::kSuccess;
}


MStatus uninitializePlugin(MObject obj)
{
	MFnPlugin plugin(obj);
	// Deregister commands, nodes, etc. here
    plugin.deregisterNode(GaussiansplattingNode::id);
    plugin.deregisterNode(GaussianSplattingLocator::id);

    /*
    MDrawRegistry::deregisterDrawOverrideCreator(
        "drawdb/geometry/myInitials_myLocator",
        "MyInitials_myLocatorDrawOverride"
    );
    */

    /*
    MHWRender::MDrawRegistry::deregisterGeometryOverrideCreator(
        "drawdb/geometry/GSNode",
        "GSDrawOverride");
    

    */
    MHWRender::MDrawRegistry::deregisterSubSceneOverrideCreator(
        "drawdb/subscene/GaussianSplattingLocator",
        "GsTestPlugin");

	return MS::kSuccess;
}