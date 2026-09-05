#include "MayaMeshBuilder.h"

#include <maya/MFnDependencyNode.h>
#include <maya/MFnMesh.h>
#include <maya/MGlobal.h>
#include <maya/MIntArray.h>

namespace GS::Mesh
{
    MObject MayaMeshBuilder::create(const MeshData& data, const MString& baseName)
    {
        if (data.isEmpty())
        {
            return MObject::kNullObj;
        }

        MStatus status;
        MFnMesh fnMesh;

        MObject transform = fnMesh.create(
            data.points.length(),
            data.polygonCounts.length(),
            data.points,
            data.polygonCounts,
            data.polygonConnects,
            MObject::kNullObj,
            &status);

        if (!status)
        {
            MGlobal::displayError(
                "Gaussian splat mesh could not be created: " + status.errorString());
            return MObject::kNullObj;
        }

        if (data.normals.length() == data.points.length())
        {
            MIntArray vertexIds(data.normals.length());
            for (unsigned int i = 0; i < data.normals.length(); ++i)
            {
                vertexIds[i] = i;
            }
                
			auto normals = data.normals;
            fnMesh.setVertexNormals(normals, vertexIds);
        }

        MFnDependencyNode fnTransform(transform);
        fnTransform.setName(baseName);

        // Without a shading group the mesh renders black in the viewport.
        MGlobal::executeCommand("sets -edit -forceElement initialShadingGroup " + fnTransform.name());

        return transform;
    }
}
