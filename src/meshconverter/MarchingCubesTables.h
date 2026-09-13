#pragma once

// Classic marching cubes lookup tables (Lorensen & Cline, as tabulated by Paul
// Bourke). Numbering used everywhere in this folder:
//   corners 0..7 : (0,0,0) (1,0,0) (1,0,1) (0,0,1) (0,1,0) (1,1,0) (1,1,1) (0,1,1)
//   edges   0..11: 0-1 1-2 2-3 3-0 4-5 5-6 6-7 7-4 0-4 1-5 2-6 3-7
namespace GS::Mesh::MarchingCubesTables
{
    // Bit i is set when edge i is crossed by the surface for that cube index.
    extern const int kEdgeTable[256];

    // Up to five triangles per cube, given as edge indices and terminated by -1.
    extern const int kTriangleTable[256][16];

    // Cell relative offset of each corner.
    extern const int kCornerOffset[8][3];

    // The two corners each edge connects.
    extern const int kEdgeCorners[12][2];

    // Grid vertex (cell relative offset) and axis (0=X, 1=Y, 2=Z) that owns each
    // edge. Two neighbouring cells map a shared edge onto the same owner, which
    // is what lets them reuse one interpolated vertex.
    extern const int kEdgeOwner[12][4];
}
