#include <maya/MFnNumericAttribute.h>
#include "gaussiansplattingnode.h"

MTypeId GaussianSplattingLocator::id(0x7802aaaa);
MObject GaussianSplattingLocator::locatorMsgAttr;
MObject GaussianSplattingLocator::aSplatSize;

MStatus GaussianSplattingLocator::connectionMade(const MPlug& plug, const MPlug& otherPlug, bool asSrc)
{
    // Check if the connection is made to the locatorMsgAttr
    if (plug == locatorMsgAttr || otherPlug == locatorMsgAttr) {
        MGlobal::displayInfo("Connection made to locatorMsgAttr.");
    }
    return MPxLocatorNode::connectionMade(plug, otherPlug, asSrc);
}

MStatus GaussianSplattingLocator::initialize()
{
    MFnNumericAttribute  typedAttr;

    aSplatSize = typedAttr.create("splatSize", "splatSize", MFnNumericData::kFloat);
    typedAttr.setMin(0.1f);     // slider min
    typedAttr.setMax(1.0f);    // slider max
    typedAttr.setKeyable(true);
    typedAttr.setStorable(true);
    typedAttr.setReadable(true);
    typedAttr.setWritable(true);
    addAttribute(aSplatSize);
    return MS::kSuccess;
}
