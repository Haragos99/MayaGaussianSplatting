#pragma once

#include <maya/MPoint.h>
#include <maya/MVector.h>

#include <vector>

namespace GS
{
    // Produces the back-to-front draw order required by alpha blending.
    //
    // The previous std::sort did O(n log n) comparisons and multiplied a point
    // by the object-to-world matrix twice per comparison. This class caches the
    // centers, builds one 32 bit depth key per splat and runs a 4 pass LSD
    // radix sort, which is linear and comparison free.
    class SplatDepthSorter
    {
    public:
        void reserve(size_t quadCount);

        void addCenter(const MPoint& center);

        void clear();

        size_t quadCount() const { return m_centers.size() / 3; }

        // viewDirection must live in the same space as the cached centers.
        const std::vector<unsigned int>& sortBackToFront(const MVector& viewDirection);

    private:
        void radixSort();

        std::vector<float>        m_centers;
        std::vector<unsigned int> m_keys;
        std::vector<unsigned int> m_keysScratch;
        std::vector<unsigned int> m_order;
        std::vector<unsigned int> m_orderScratch;
    };
}
