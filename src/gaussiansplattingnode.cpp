#include <maya/MFnNumericAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MViewport2Renderer.h>
#include "gaussiansplattingnode.h"

MTypeId GaussianSplattingLocator::id(0x7802aaaa);
MObject GaussianSplattingLocator::locatorMsgAttr;
MObject GaussianSplattingLocator::aSplatSize;
MObject GaussianSplattingLocator::outputAttr;
MObject GaussianSplattingLocator::aFileName;

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


    MFnTypedAttribute tAttr;

    aFileName = tAttr.create(
        "fileName",
        "fn",
        MFnData::kString
    );


    tAttr.setStorable(true);
    tAttr.setWritable(true);
    tAttr.setReadable(true);
    tAttr.setKeyable(true);
    tAttr.setUsedAsFilename(true);

    addAttribute(aFileName);

    outputAttr = typedAttr.create(
        "output",
        "out",
        MFnNumericData::kFloat,
        0.0
    );

    typedAttr.setWritable(false);
    typedAttr.setReadable(true);

    addAttribute(outputAttr);

    // Filename change -> output becomes dirty
    attributeAffects(aFileName, outputAttr);


    return MS::kSuccess;
}


unsigned int GaussianSplattingLocator::syncSplatData()
{
    if (m_fileDirty)
    {
        m_fileDirty = false;

        const MPlug filePlug(thisMObject(), aFileName);

        m_data.loadFromFile(filePlug.asString());
    }

    return m_data.version();
}


MStatus GaussianSplattingLocator::compute(
    const MPlug& plug,
    MDataBlock& dataBlock)
{
    if (plug == outputAttr)
    {
        MString fileName =
            dataBlock.inputValue(aFileName).asString();

        MDataHandle outputHandle =
            dataBlock.outputValue(outputAttr);

        outputHandle.setClean();

        return MS::kSuccess;
    }

    return MS::kUnknownParameter;
}


MStatus GaussianSplattingLocator::setDependentsDirty(
    const MPlug& plug,
    MPlugArray& affectedPlugs)
{
    if (plug == aFileName)
    {
        m_fileDirty = true;

        // Forces the sub-scene override to run update() and rebuild its buffers.
        MHWRender::MRenderer::setGeometryDrawDirty(thisMObject(), true);
    }

    return MPxLocatorNode::setDependentsDirty(plug, affectedPlugs);
}