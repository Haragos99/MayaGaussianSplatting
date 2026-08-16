#include "StandardSplatDecoder.h"

#include <algorithm>
#include <cmath>

#include "PlyElementReader.h"
#include "PlyHeaderParser.h"

namespace GS::Ply
{
    namespace
    {
        constexpr float kSphericalHarmonicsC0 = 0.28209479177387814f;

        float clamp01(float value)
        {
            return std::max(0.0f, std::min(1.0f, value));
        }

        float sigmoid(float value)
        {
            return 1.0f / (1.0f + std::exp(-value));
        }

        double getProperty(const Record& record, int propertyIndex, double fallback = 0.0)
        {
            if (propertyIndex < 0) 
            { 
                return fallback; 
            }

            const std::size_t index = static_cast<std::size_t>(propertyIndex);
            return index < record.size() ? record[index] : fallback;
        }

        GaussianSplat makeSplat(const Record& record, const Element& vertexElement)
        {
            const std::vector<Property>& properties = vertexElement.properties;
            const int positionXIndex = findPropertyIndex(properties, "x");
            const int positionYIndex = findPropertyIndex(properties, "y");
            const int positionZIndex = findPropertyIndex(properties, "z");
            const int opacityIndex = findPropertyIndex(properties, "opacity");
            const int scaleXIndex = findPropertyIndex(properties, "scale_0");
            const int scaleYIndex = findPropertyIndex(properties, "scale_1");
            const int scaleZIndex = findPropertyIndex(properties, "scale_2");
            const int rotationWIndex = findPropertyIndex(properties, "rot_0");
            const int rotationXIndex = findPropertyIndex(properties, "rot_1");
            const int rotationYIndex = findPropertyIndex(properties, "rot_2");
            const int rotationZIndex = findPropertyIndex(properties, "rot_3");
            const int sphericalRedIndex = findPropertyIndex(properties, "f_dc_0");
            const int sphericalGreenIndex = findPropertyIndex(properties, "f_dc_1");
            const int sphericalBlueIndex = findPropertyIndex(properties, "f_dc_2");
            const int redIndex = findPropertyIndex(properties, "red");
            const int greenIndex = findPropertyIndex(properties, "green");
            const int blueIndex = findPropertyIndex(properties, "blue");
            const int alphaIndex = findPropertyIndex(properties, "alpha");

            GaussianSplat splat;
            splat.center = MPoint(
                getProperty(record, positionXIndex),
                getProperty(record, positionYIndex),
                getProperty(record, positionZIndex)
            );

            float red = 1.0f;
            float green = 1.0f;
            float blue = 1.0f;

            if (sphericalRedIndex >= 0 && sphericalGreenIndex >= 0 && sphericalBlueIndex >= 0)
            {
                red = clamp01(0.5f + kSphericalHarmonicsC0 * static_cast<float>(getProperty(record, sphericalRedIndex)));
                green = clamp01(0.5f + kSphericalHarmonicsC0 * static_cast<float>(getProperty(record, sphericalGreenIndex)));
                blue = clamp01(0.5f + kSphericalHarmonicsC0 * static_cast<float>(getProperty(record, sphericalBlueIndex)));
            }
            else if (redIndex >= 0 && greenIndex >= 0 && blueIndex >= 0)
            {
                red = clamp01(static_cast<float>(getProperty(record, redIndex)) / 255.0f);
                green = clamp01(static_cast<float>(getProperty(record, greenIndex)) / 255.0f);
                blue = clamp01(static_cast<float>(getProperty(record, blueIndex)) / 255.0f);
            }

            float opacity = 1.0f;
            if (opacityIndex >= 0) 
            {
                opacity = sigmoid(static_cast<float>(getProperty(record, opacityIndex)));
            }
            else if (alphaIndex >= 0) 
            {
                opacity = clamp01(static_cast<float>(getProperty(record, alphaIndex)) / 255.0f); 
            }

            float scaleX = 0.03f;
            float scaleY = 0.03f;
            float scaleZ = 0.03f;
            if (scaleXIndex >= 0 && scaleYIndex >= 0)
            {
                scaleX = std::exp(static_cast<float>(getProperty(record, scaleXIndex)));
                scaleY = std::exp(static_cast<float>(getProperty(record, scaleYIndex)));
                scaleZ = scaleZIndex >= 0 ? std::exp(static_cast<float>(getProperty(record, scaleZIndex))) : scaleX;
            }
            else if (scaleZIndex >= 0)
            {
                scaleX = scaleY = scaleZ = std::exp(static_cast<float>(getProperty(record, scaleZIndex)));
            }

            splat.scale[0] = std::max(0.0001f, std::min(scaleX, 10.0f));
            splat.scale[1] = std::max(0.0001f, std::min(scaleY, 10.0f));
            splat.scale[2] = std::max(0.0001f, std::min(scaleZ, 10.0f));
            splat.scaleX = splat.scale[0];
            splat.scaleY = splat.scale[1];
			splat.scaleZ = splat.scale[2];
            splat.opacity = opacity;
            splat.color = MColor(red, green, blue, opacity);

            if (rotationWIndex >= 0 && rotationXIndex >= 0 && rotationYIndex >= 0 && rotationZIndex >= 0)
            {
                float rotationW = static_cast<float>(getProperty(record, rotationWIndex));
                float rotationX = static_cast<float>(getProperty(record, rotationXIndex));
                float rotationY = static_cast<float>(getProperty(record, rotationYIndex));
                float rotationZ = static_cast<float>(getProperty(record, rotationZIndex));
                const float rotationLength = std::sqrt(
                    rotationW * rotationW + rotationX * rotationX + rotationY * rotationY + rotationZ * rotationZ);

                if (rotationLength > 1e-8f)
                {
                    rotationW /= rotationLength;
                    rotationX /= rotationLength;
                    rotationY /= rotationLength;
                    rotationZ /= rotationLength;
                }
                else
                {
                    rotationW = 1.0f;
                    rotationX = rotationY = rotationZ = 0.0f;
                }

                splat.rotation[0] = rotationW;
                splat.rotation[1] = rotationX;
                splat.rotation[2] = rotationY;
                splat.rotation[3] = rotationZ;
            }

            return splat;
        }
    }

    bool StandardSplatDecoder::canDecode(const Header& header) const
    {
        const Element* vertexElement = findElement(header, "vertex");
        return vertexElement != nullptr
            && findPropertyIndex(vertexElement->properties, "x") >= 0
            && findPropertyIndex(vertexElement->properties, "y") >= 0
            && findPropertyIndex(vertexElement->properties, "z") >= 0;
    }

    bool StandardSplatDecoder::decode(
        std::istream& input,
        const Header& header,
        std::vector<GaussianSplat>& outSplats,
        std::string* outError) const
    {
        const Element* vertexElement = findElement(header, "vertex");
        if (vertexElement == nullptr || vertexElement->count == 0)
        {
            if (outError) 
            {
                *outError = "PLY file has no vertices."; 
            }

            return false;
        }

        Records records;
        const bool readSuccess = header.format == Format::Ascii
            ? readAsciiElement(input, *vertexElement, records, outError)
            : readBinaryElement(input, *vertexElement, records, outError);

        if (!readSuccess)
        {
            return false;
        }

        outSplats.clear();
        outSplats.reserve(records.size());
        for (const Record& record : records)
        {
            outSplats.push_back(makeSplat(record, *vertexElement));
        }

        return true;
    }
}
