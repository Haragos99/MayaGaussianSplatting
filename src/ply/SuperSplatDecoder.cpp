#include "SuperSplatDecoder.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "PlyElementReader.h"
#include "PlyHeaderParser.h"

namespace GS::Ply
{
    namespace
    {
        constexpr std::size_t kChunkSize = 256;

        float clamp01(float value)
        {
            return std::max(0.0f, std::min(1.0f, value));
        }

        float lerp(float minimum, float maximum, float normalizedValue)
        {
            return minimum + (maximum - minimum) * normalizedValue;
        }

        std::uint32_t getPackedValue(const Record& record, int propertyIndex)
        {
            return static_cast<std::uint32_t>(record[static_cast<std::size_t>(propertyIndex)]);
        }

        float getRangeValue(
            const Record& chunkRecord,
            const std::vector<Property>& properties,
            const std::string& propertyName,
            bool maximum,
            float fallback)
        {
            const std::string rangeName = (maximum ? "max_" : "min_") + propertyName;
            const int propertyIndex = findPropertyIndex(properties, rangeName);
            return propertyIndex >= 0
                ? static_cast<float>(chunkRecord[static_cast<std::size_t>(propertyIndex)])
                : fallback;
        }

        void unpackPositionOrScale(
            std::uint32_t packedValue,
            float& normalizedX,
            float& normalizedY,
            float& normalizedZ)
        {
            constexpr uint32_t X_MASK = 0x7FF; // 11 bits
            constexpr uint32_t Y_MASK = 0x3FF; // 10 bits
            constexpr uint32_t Z_MASK = 0x7FF; // 11 bits

            constexpr float X_MAX = 2047.0f;
            constexpr float Y_MAX = 1023.0f;
            constexpr float Z_MAX = 2047.0f;

            const uint32_t packedX = (packedValue >> 21) & X_MASK;
            const uint32_t packedY = (packedValue >> 11) & Y_MASK;
            const uint32_t packedZ = packedValue & Z_MASK;

            normalizedX = static_cast<float>(packedX) / X_MAX;
            normalizedY = static_cast<float>(packedY) / Y_MAX;
            normalizedZ = static_cast<float>(packedZ) / Z_MAX;
        }

        void unpackRotation(
            std::uint32_t packedValue,
            float& rotationW,
            float& rotationX,
            float& rotationY,
            float& rotationZ)
        {
            constexpr uint32_t COMPONENT_MASK = 0x3FFu; // 10 bits
            constexpr float COMPONENT_MAX = 1023.0f;
            constexpr float SQRT_TWO = 1.4142135623730951f;

            const uint32_t packedA = (packedValue >> 20) & COMPONENT_MASK;
            const uint32_t packedB = (packedValue >> 10) & COMPONENT_MASK;
            const uint32_t packedC = packedValue & COMPONENT_MASK;

            const float componentA =
                (static_cast<float>(packedA) / COMPONENT_MAX - 0.5f) * SQRT_TWO;

            const float componentB =
                (static_cast<float>(packedB) / COMPONENT_MAX - 0.5f) * SQRT_TWO;

            const float componentC =
                (static_cast<float>(packedC) / COMPONENT_MAX - 0.5f) * SQRT_TWO;

            const float squaredSum =
                componentA * componentA +
                componentB * componentB +
                componentC * componentC;

            const float missingComponent =
                std::sqrt(std::max(0.0f, 1.0f - squaredSum));

            const uint32_t omittedComponentIndex = packedValue >> 30;


            switch (omittedComponentIndex)
            {
            case 0:
                rotationW = missingComponent;
                rotationX = componentA;
                rotationY = componentB;
                rotationZ = componentC;
                break;

            case 1:
                rotationW = componentA;
                rotationX = missingComponent;
                rotationY = componentB;
                rotationZ = componentC;
                break;

            case 2:
                rotationW = componentA;
                rotationX = componentB;
                rotationY = missingComponent;
                rotationZ = componentC;
                break;

            case 3:
            default:
                rotationW = componentA;
                rotationX = componentB;
                rotationY = componentC;
                rotationZ = missingComponent;
                break;
            }


        }
    }

    bool SuperSplatDecoder::canDecode(const Header& header) const
    {
        const Element* chunkElement = findElement(header, "chunk");
        const Element* vertexElement = findElement(header, "vertex");

        return header.format == Format::BinaryLittleEndian
            && chunkElement != nullptr
            && vertexElement != nullptr
            && findPropertyIndex(vertexElement->properties, "packed_position") >= 0
            && findPropertyIndex(vertexElement->properties, "packed_scale") >= 0
            && findPropertyIndex(vertexElement->properties, "packed_color") >= 0;
    }

    bool SuperSplatDecoder::decode(
        std::istream& input,
        const Header& header,
        std::vector<GaussianSplat>& outSplats,
        std::string* outError) const
    {
        const Element* chunkElement = findElement(header, "chunk");
        const Element* vertexElement = findElement(header, "vertex");

        if (chunkElement == nullptr || vertexElement == nullptr)
        {
            if (outError) 
            { 
                *outError = "Compressed PLY is missing chunk or vertex element."; 
            }

            return false;
        }

        Records chunkRecords;
        Records vertexRecords;

        for (const Element& element : header.elements)
        {
            Records ignoredRecords;
            Records* targetRecords = &ignoredRecords;
            if (element.name == "chunk")
            {
                targetRecords = &chunkRecords;
            }
            else if (element.name == "vertex")
            {
                targetRecords = &vertexRecords;
            }

            if (!readBinaryElement(input, element, *targetRecords, outError))
            {
                return false;
            }
            if (element.name == "vertex")
            {
                break;
            }
        }

        const std::vector<Property>& chunkProperties = chunkElement->properties;
        const std::vector<Property>& vertexProperties = vertexElement->properties;
        const int packedPositionIndex = findPropertyIndex(vertexProperties, "packed_position");
        const int packedScaleIndex = findPropertyIndex(vertexProperties, "packed_scale");
        const int packedRotationIndex = findPropertyIndex(vertexProperties, "packed_rotation");
        const int packedColorIndex = findPropertyIndex(vertexProperties, "packed_color");

        outSplats.clear();
        outSplats.reserve(vertexRecords.size());

        for (std::size_t vertexIndex = 0; vertexIndex < vertexRecords.size(); ++vertexIndex)
        {
            const std::size_t chunkIndex = vertexIndex / kChunkSize;
            if (chunkIndex >= chunkRecords.size())
            {
                if (outError)
                {
                    *outError = "Compressed PLY vertex references a missing chunk.";
                }

                return false;
            }

            const Record& vertexRecord = vertexRecords[vertexIndex];
            const Record& chunkRecord = chunkRecords[chunkIndex];

            float normalizedX = 0.0f;
            float normalizedY = 0.0f;
            float normalizedZ = 0.0f;
            unpackPositionOrScale(getPackedValue(vertexRecord, packedPositionIndex), normalizedX, normalizedY, normalizedZ);

            GaussianSplat splat;
            splat.center = MPoint(
                lerp(getRangeValue(chunkRecord, chunkProperties, "x", false, 0.0f), getRangeValue(chunkRecord, chunkProperties, "x", true, 0.0f), normalizedX),
                lerp(getRangeValue(chunkRecord, chunkProperties, "y", false, 0.0f), getRangeValue(chunkRecord, chunkProperties, "y", true, 0.0f), normalizedY),
                lerp(getRangeValue(chunkRecord, chunkProperties, "z", false, 0.0f), getRangeValue(chunkRecord, chunkProperties, "z", true, 0.0f), normalizedZ)
            );

            float scaleX = 0.03f;
            float scaleY = 0.03f;
            float scaleZ = 0.03f;
            unpackPositionOrScale(getPackedValue(vertexRecord, packedScaleIndex), normalizedX, normalizedY, normalizedZ);
            scaleX = std::exp(lerp(getRangeValue(chunkRecord, chunkProperties, "scale_x", false, 0.0f), getRangeValue(chunkRecord, chunkProperties, "scale_x", true, 0.0f), normalizedX));
            scaleY = std::exp(lerp(getRangeValue(chunkRecord, chunkProperties, "scale_y", false, 0.0f), getRangeValue(chunkRecord, chunkProperties, "scale_y", true, 0.0f), normalizedY));
            scaleZ = std::exp(lerp(getRangeValue(chunkRecord, chunkProperties, "scale_z", false, 0.0f), getRangeValue(chunkRecord, chunkProperties, "scale_z", true, 0.0f), normalizedZ));

            splat.scale[0] = std::max(0.0001f, std::min(scaleX, 10.0f));
            splat.scale[1] = std::max(0.0001f, std::min(scaleY, 10.0f));
            splat.scale[2] = std::max(0.0001f, std::min(scaleZ, 10.0f));
            splat.scaleX = splat.scale[0];
            splat.scaleY = splat.scale[1];
			splat.scaleZ = splat.scale[2];

            const std::uint32_t colorValue = getPackedValue(vertexRecord, packedColorIndex);
            const float red = clamp01(lerp(getRangeValue(chunkRecord, chunkProperties, "r", false, 0.0f), getRangeValue(chunkRecord, chunkProperties, "r", true, 1.0f), static_cast<float>((colorValue >> 24) & 0xFFu) / 255.0f));
            const float green = clamp01(lerp(getRangeValue(chunkRecord, chunkProperties, "g", false, 0.0f), getRangeValue(chunkRecord, chunkProperties, "g", true, 1.0f), static_cast<float>((colorValue >> 16) & 0xFFu) / 255.0f));
            const float blue = clamp01(lerp(getRangeValue(chunkRecord, chunkProperties, "b", false, 0.0f), getRangeValue(chunkRecord, chunkProperties, "b", true, 1.0f), static_cast<float>((colorValue >> 8) & 0xFFu) / 255.0f));
            splat.opacity = static_cast<float>(colorValue & 0xFFu) / 255.0f;
            splat.color = MColor(red, green, blue, splat.opacity);

            if (packedRotationIndex >= 0)
            {
                unpackRotation(
                    getPackedValue(vertexRecord, packedRotationIndex),
                    splat.rotation[0], splat.rotation[1], splat.rotation[2], splat.rotation[3]
                );
            }

            outSplats.push_back(splat);
        }

        return true;
    }
}
