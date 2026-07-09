#include "gaussiansplattingnode.h"

MTypeId GaussiansplattingNode::id(0x001226C1);

void* GaussiansplattingNode::creator()
{
	return new GaussiansplattingNode();
}

MStatus GaussiansplattingNode::initialize()
{
	// Initialize attributes here
	return MS::kSuccess;
}


void GaussiansplattingNode::loadPointCloud(const std::string& filePath)
{

}