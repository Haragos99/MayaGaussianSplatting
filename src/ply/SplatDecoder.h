#pragma once

#include <istream>
#include <string>
#include <vector>

#include "PlyTypes.h"
#include "../data.h"

namespace GS::Ply
{
    class SplatDecoder
    {
    public:
        virtual ~SplatDecoder() = default;

        virtual bool canDecode(const Header& header) const = 0;

        virtual bool decode(
            std::istream& input,
            const Header& header,
            std::vector<GaussianSplat>& outSplats,
            std::string* outError) const = 0;
    };
}
