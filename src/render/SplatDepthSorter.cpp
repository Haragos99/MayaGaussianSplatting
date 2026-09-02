#include "SplatDepthSorter.h"

#include <cstring>

namespace GS
{
    namespace
    {
        constexpr unsigned int kRadixPasses = 4;
        constexpr unsigned int kRadixBuckets = 256;

        // Order preserving float -> unsigned int mapping, then inverted so that
        // an ascending radix sort yields a descending (back-to-front) depth order.
        inline unsigned int makeDepthKey(float depth)
        {
            unsigned int bits = 0;
            std::memcpy(&bits, &depth, sizeof(bits));

            bits = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);

            return ~bits;
        }
    }


    void SplatDepthSorter::reserve(size_t quadCount)
    {
        m_centers.reserve(quadCount * 3);
    }


    void SplatDepthSorter::addCenter(const MPoint& center)
    {
        m_centers.push_back(static_cast<float>(center.x));
        m_centers.push_back(static_cast<float>(center.y));
        m_centers.push_back(static_cast<float>(center.z));
    }


    void SplatDepthSorter::clear()
    {
        m_centers.clear();
        m_keys.clear();
        m_keysScratch.clear();
        m_order.clear();
        m_orderScratch.clear();
    }


    const std::vector<unsigned int>& SplatDepthSorter::sortBackToFront(
        const MVector& viewDirection)
    {
        const size_t count = quadCount();

        m_keys.resize(count);
        m_order.resize(count);

        const float dx = static_cast<float>(viewDirection.x);
        const float dy = static_cast<float>(viewDirection.y);
        const float dz = static_cast<float>(viewDirection.z);

        for (size_t i = 0; i < count; ++i)
        {
            const float* center = &m_centers[i * 3];

            const float depth =
                center[0] * dx +
                center[1] * dy +
                center[2] * dz;

            m_keys[i] = makeDepthKey(depth);
            m_order[i] = static_cast<unsigned int>(i);
        }

        radixSort();

        return m_order;
    }


    void SplatDepthSorter::radixSort()
    {
        const size_t count = m_keys.size();

        if (count < 2)
        {
            return;
        }

        size_t histogram[kRadixPasses][kRadixBuckets] = {};

        for (const unsigned int key : m_keys)
        {
            ++histogram[0][key & 0xFFu];
            ++histogram[1][(key >> 8) & 0xFFu];
            ++histogram[2][(key >> 16) & 0xFFu];
            ++histogram[3][(key >> 24) & 0xFFu];
        }

        m_keysScratch.resize(count);
        m_orderScratch.resize(count);

        for (unsigned int pass = 0; pass < kRadixPasses; ++pass)
        {
            const unsigned int shift = pass * 8;

            // Every key shares this byte, so the pass cannot change the order.
            if (histogram[pass][(m_keys[0] >> shift) & 0xFFu] == count)
            {
                continue;
            }

            size_t offsets[kRadixBuckets];
            size_t running = 0;

            for (unsigned int bucket = 0; bucket < kRadixBuckets; ++bucket)
            {
                offsets[bucket] = running;
                running += histogram[pass][bucket];
            }

            for (size_t i = 0; i < count; ++i)
            {
                const unsigned int key = m_keys[i];
                const size_t destination = offsets[(key >> shift) & 0xFFu]++;

                m_keysScratch[destination] = key;
                m_orderScratch[destination] = m_order[i];
            }

            m_keys.swap(m_keysScratch);
            m_order.swap(m_orderScratch);
        }
    }
}
