#include <maya/MFnNumericAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include "gaussiansplattingnode.h"
#include "gaussianSplatPlyLoader.h"

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


std::pair<std::vector<GS::GaussianSplat>,MBoundingBox> GaussianSplattingLocator::loadSplatsFromFile()
{
    MBoundingBox m_boundingBox;
    std::vector<GS::GaussianSplat> m_splats;

    const int countX = 20;
    const int countY = 20;
    const std::string filePath = "C:\\Users\\Geri\\Documents\\Projects\\CG\\MayaGaussianSplatting\\models\\Tree.ply";
    std::string error;

    if (!GaussianSplatPlyLoader::load(filePath, m_splats, &error))
    {
        for (int y = 0; y < countY; ++y)
        {
            for (int x = 0; x < countX; ++x)
            {
                const float fx = static_cast<float>(x) / static_cast<float>(countX - 1);
                const float fy = static_cast<float>(y) / static_cast<float>(countY - 1);

                GS::GaussianSplat splat;
                splat.center = MPoint(
                    (fx - 0.5f) * 6.0f,
                    (fy - 0.5f) * 4.0f,
                    std::sin(fx * 6.2831853f) * 0.5f);

                splat.color = MColor(fx, fy, 1.0f - fx, 0.45f);
                splat.scaleX = 0.08f;
                splat.scaleY = 0.08f;
                splat.scaleZ = 0.08f;
                splat.opacity = 0.45f;

                m_splats.push_back(splat);
            }
        }
    };

    for (const GS::GaussianSplat& splat : m_splats)
    {
        const double r = std::max(splat.scaleX, splat.scaleY) * 2.0;
        m_boundingBox.expand(splat.center + MVector(r, r, r));
        m_boundingBox.expand(splat.center + MVector(-r, -r, -r));
    }

	return  std::make_pair( m_splats, m_boundingBox);
}


MStatus GaussianSplattingLocator::compute(
    const MPlug& plug,
    MDataBlock& dataBlock)
{
    if (plug == outputAttr)
    {
        MString fileName =
            dataBlock.inputValue(aFileName).asString();

        MGlobal::displayInfo(
            "Filename: " + fileName
        );

        // Do something with the file...

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
        MString fileName =
            plug.asString();

        MGlobal::displayInfo(
            "Filename changed: " + fileName
        );
    }

    return MS::kSuccess;
}