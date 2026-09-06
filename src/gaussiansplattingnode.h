#pragma once
#include <maya/MPxLocatorNode.h>
#include "data.h"
#include <maya/MGlobal.h>
#include "scene/SplatDataSource.h"

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
    MStatus compute(const MPlug& plug, MDataBlock& dataBlock);
    MStatus setDependentsDirty(const MPlug& plug, MPlugArray& affectedPlugs);
    static MObject aSplatSize;
	static MObject aFileName;
    static MObject outputAttr;
    bool isBounded() const override { return true; }

    // Reloads the file when the path attribute changed. Returns the data version.
    unsigned int syncSplatData();

    bool needsReload() const { return m_fileDirty; }

    unsigned int dataVersion() const { return m_data.version(); }

    const std::vector<GS::GaussianSplat>& splats() const { return m_data.splats(); }

    const MBoundingBox& splatBounds() const { return m_data.bounds(); }



private:

    GS::SplatDataSource m_data;
    bool m_fileDirty = true;
};
