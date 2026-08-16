#pragma once
#include <maya/MPxLocatorNode.h>
#include "data.h"
#include <maya/MGlobal.h>

// MyLocator node is drawn using MyLocatorDrawOverride
class GaussianSplattingLocator : public MPxLocatorNode
{
public:
    GaussianSplattingLocator() = default;
    static MTypeId id;
    static MObject locatorMsgAttr;
    static void* creator() { return new GaussianSplattingLocator(); }
    static MStatus initialize();
    MStatus connectionMade(const MPlug& plug, const MPlug& otherPlug, bool asSrc) override;
    static MObject aSplatSize;
    bool isBounded() const override { return true; }


};
