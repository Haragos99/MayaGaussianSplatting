#pragma once

#include "SplatDecoder.h"

namespace GS::Ply
{
    class StandardSplatDecoder final : public SplatDecoder
    {
    public:
        bool canDecode(const Header& header) const override;

        bool decode(
            std::istream& input,
            const Header& header,
            std::vector<GaussianSplat>& outSplats,
            std::string* outError) const override;
    };
}
