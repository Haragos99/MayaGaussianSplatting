#include "gaussianSplatPlyLoader.h"

#include <fstream>
#include <vector>

#include "ply/PlyHeaderParser.h"
#include "ply/SplatDecoder.h"
#include "ply/StandardSplatDecoder.h"
#include "ply/SuperSplatDecoder.h"

namespace
{
    bool decodeWithRegisteredDecoder(
        std::istream& input,
        const GS::Ply::Header& header,
        std::vector<GS::GaussianSplat>& outSplats,
        std::string* outError)
    {
        const std::vector<std::shared_ptr<GS::Ply::SplatDecoder>> decoders =
        {
            std::make_shared<GS::Ply::SuperSplatDecoder>(),
            std::make_shared<GS::Ply::StandardSplatDecoder>()
        };

        for (const std::shared_ptr<GS::Ply::SplatDecoder>& decoder : decoders)
        {
            if (decoder->canDecode(header))
            {
                return decoder->decode(input, header, outSplats, outError);
            }
        }

        if (outError)
        {
            *outError = "PLY file does not match a supported Gaussian splat format.";
        }

        return false;
    }
}

bool GaussianSplatPlyLoader::load(
    const std::string& filePath,
    std::vector<GS::GaussianSplat>& outSplats,
    std::string* outError)
{
    outSplats.clear();

    std::ifstream input(filePath, std::ios::binary);
    if (!input)
    {
        if (outError) *outError = "Could not open PLY file: " + filePath;
        return false;
    }

    GS::Ply::Header header;
    if (!GS::Ply::parseHeader(input, header, outError)) return false;
    if (header.format == GS::Ply::Format::Unknown)
    {
        if (outError) *outError = "PLY format is missing or unsupported.";
        return false;
    }

    if (!decodeWithRegisteredDecoder(input, header, outSplats, outError))
    {
        outSplats.clear();
        return false;
    }

    return true;
}
