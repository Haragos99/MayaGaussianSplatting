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
    static MObject aGenerateMesh;
    static MObject aMeshResolution;
    static MObject aMeshIsoLevel;
    bool isBounded() const override { return true; }

    // Reloads the file when the path attribute changed. Returns the data version.
    unsigned int syncSplatData();

    bool needsReload() const { return m_fileDirty; }

    unsigned int dataVersion() const { return m_data.version(); }

    const std::vector<GS::GaussianSplat>& splats() const { return m_data.splats(); }

    const MBoundingBox& splatBounds() const { return m_data.bounds(); }

    int meshResolution() const;
    float meshIsoLevel() const;

private:
    // Meshing builds DAG nodes, which is illegal from the DG or the draw pass,
    // so the conversion is queued and runs on idle instead.
    void scheduleMeshRebuild();

    GS::SplatDataSource m_data;
    bool m_fileDirty = true;
};
