#include "gaussianSplatPlyLoader.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <unordered_map>

namespace
{
    enum class PlyFormat
    {
        Unknown,
        Ascii,
        BinaryLittleEndian
    };

    enum class PlyScalarType
    {
        Invalid,
        Int8,
        UInt8,
        Int16,
        UInt16,
        Int32,
        UInt32,
        Float32,
        Float64
    };

    struct PlyProperty
    {
        std::string name;
        PlyScalarType type = PlyScalarType::Invalid;
    };

    struct PlyHeader
    {
        PlyFormat format = PlyFormat::Unknown;
        std::size_t vertexCount = 0;
        std::vector<PlyProperty> vertexProperties;
    };

    static constexpr float kSHC0 = 0.28209479177387814f;

    float clamp01(float v)
    {
        return std::max(0.0f, std::min(1.0f, v));
    }

    float sigmoid(float v)
    {
        return 1.0f / (1.0f + std::exp(-v));
    }

    bool isLittleEndianHost()
    {
        const std::uint16_t value = 1;
        return *reinterpret_cast<const std::uint8_t*>(&value) == 1;
    }

    PlyScalarType parseScalarType(const std::string& text)
    {
        if (text == "char" || text == "int8")
        {
            return PlyScalarType::Int8;
        }

        if (text == "uchar" || text == "uint8")
        {
            return PlyScalarType::UInt8;
        }

        if (text == "short" || text == "int16")
        {
            return PlyScalarType::Int16;
        }

        if (text == "ushort" || text == "uint16")
        {
            return PlyScalarType::UInt16;
        }

        if (text == "int" || text == "int32")
        {
            return PlyScalarType::Int32;
        }

        if (text == "uint" || text == "uint32")
        {
            return PlyScalarType::UInt32;
        }

        if (text == "float" || text == "float32")
        {
            return PlyScalarType::Float32;
        }

        if (text == "double" || text == "float64")
        {
            return PlyScalarType::Float64;
        }

        return PlyScalarType::Invalid;
    }

    std::size_t scalarTypeSize(PlyScalarType type)
    {
        switch (type)
        {
        case PlyScalarType::Int8:
        case PlyScalarType::UInt8:
            return 1;

        case PlyScalarType::Int16:
        case PlyScalarType::UInt16:
            return 2;

        case PlyScalarType::Int32:
        case PlyScalarType::UInt32:
        case PlyScalarType::Float32:
            return 4;

        case PlyScalarType::Float64:
            return 8;

        default:
            return 0;
        }
    }

    template <typename T>
    T readBinaryValue(std::istream& input)
    {
        T value{};
        input.read(reinterpret_cast<char*>(&value), sizeof(T));
        return value;
    }

    double readBinaryScalar(std::istream& input, PlyScalarType type)
    {
        // This loader expects binary_little_endian.
        // On normal Windows/Linux/macOS x86/x64 machines this is fine.
        // If you need big-endian support, add byte swapping here.
        switch (type)
        {
        case PlyScalarType::Int8:
            return static_cast<double>(readBinaryValue<std::int8_t>(input));

        case PlyScalarType::UInt8:
            return static_cast<double>(readBinaryValue<std::uint8_t>(input));

        case PlyScalarType::Int16:
            return static_cast<double>(readBinaryValue<std::int16_t>(input));

        case PlyScalarType::UInt16:
            return static_cast<double>(readBinaryValue<std::uint16_t>(input));

        case PlyScalarType::Int32:
            return static_cast<double>(readBinaryValue<std::int32_t>(input));

        case PlyScalarType::UInt32:
            return static_cast<double>(readBinaryValue<std::uint32_t>(input));

        case PlyScalarType::Float32:
            return static_cast<double>(readBinaryValue<float>(input));

        case PlyScalarType::Float64:
            return readBinaryValue<double>(input);

        default:
            return 0.0;
        }
    }

    bool parseHeader(
        std::istream& input,
        PlyHeader& header,
        std::string* outError)
    {
        std::string line;

        if (!std::getline(input, line) || line != "ply")
        {
            if (outError)
            {
                *outError = "File is not a valid PLY file. Missing 'ply' header.";
            }

            return false;
        }

        bool insideVertexElement = false;

        while (std::getline(input, line))
        {
            if (!line.empty() && line.back() == '\r')
            {
                line.pop_back();
            }

            if (line == "end_header")
            {
                return true;
            }

            if (line.empty() || line.rfind("comment", 0) == 0)
            {
                continue;
            }

            std::istringstream ss(line);
            std::string token;
            ss >> token;

            if (token == "format")
            {
                std::string formatName;
                ss >> formatName;

                if (formatName == "ascii")
                {
                    header.format = PlyFormat::Ascii;
                }
                else if (formatName == "binary_little_endian")
                {
                    header.format = PlyFormat::BinaryLittleEndian;
                }
                else
                {
                    if (outError)
                    {
                        *outError = "Unsupported PLY format: " + formatName;
                    }

                    return false;
                }
            }
            else if (token == "element")
            {
                std::string elementName;
                std::size_t elementCount = 0;

                ss >> elementName >> elementCount;

                insideVertexElement = elementName == "vertex";

                if (insideVertexElement)
                {
                    header.vertexCount = elementCount;
                }
            }
            else if (token == "property" && insideVertexElement)
            {
                std::string typeName;
                std::string propertyName;

                ss >> typeName;

                // This loader only supports scalar vertex properties.
                // If the header says "property list ...", skip it.
                if (typeName == "list")
                {
                    continue;
                }

                ss >> propertyName;

                PlyProperty property;
                property.name = propertyName;
                property.type = parseScalarType(typeName);

                if (property.type == PlyScalarType::Invalid)
                {
                    if (outError)
                    {
                        *outError = "Unsupported PLY property type: " + typeName;
                    }

                    return false;
                }

                header.vertexProperties.push_back(property);
            }
        }

        if (outError)
        {
            *outError = "PLY header ended unexpectedly.";
        }

        return false;
    }

    int findPropertyIndex(
        const std::vector<PlyProperty>& properties,
        const std::string& name)
    {
        for (std::size_t i = 0; i < properties.size(); ++i)
        {
            if (properties[i].name == name)
            {
                return static_cast<int>(i);
            }
        }

        return -1;
    }

    double getProperty(
        const std::vector<double>& values,
        int index,
        double fallback = 0.0)
    {
        if (index < 0)
        {
            return fallback;
        }

        const std::size_t i = static_cast<std::size_t>(index);

        if (i >= values.size())
        {
            return fallback;
        }

        return values[i];
    }

    GS::GaussianSplat makeSplatFromProperties(
        const std::vector<double>& values,
        const std::vector<PlyProperty>& properties)
    {
        const int idxX = findPropertyIndex(properties, "x");
        const int idxY = findPropertyIndex(properties, "y");
        const int idxZ = findPropertyIndex(properties, "z");

        const int idxOpacity = findPropertyIndex(properties, "opacity");

        const int idxScale0 = findPropertyIndex(properties, "scale_0");
        const int idxScale1 = findPropertyIndex(properties, "scale_1");
        const int idxScale2 = findPropertyIndex(properties, "scale_2");

        const int idxFdc0 = findPropertyIndex(properties, "f_dc_0");
        const int idxFdc1 = findPropertyIndex(properties, "f_dc_1");
        const int idxFdc2 = findPropertyIndex(properties, "f_dc_2");

        const int idxRed = findPropertyIndex(properties, "red");
        const int idxGreen = findPropertyIndex(properties, "green");
        const int idxBlue = findPropertyIndex(properties, "blue");
        const int idxAlpha = findPropertyIndex(properties, "alpha");

        GS::GaussianSplat splat;

        splat.center = MPoint(
            getProperty(values, idxX),
            getProperty(values, idxY),
            getProperty(values, idxZ));

        float r = 1.0f;
        float g = 1.0f;
        float b = 1.0f;

        if (idxFdc0 >= 0 && idxFdc1 >= 0 && idxFdc2 >= 0)
        {
            // Common 3D Gaussian Splatting color conversion:
            // RGB = 0.5 + C0 * SH_DC
            r = clamp01(0.5f + kSHC0 * static_cast<float>(getProperty(values, idxFdc0)));
            g = clamp01(0.5f + kSHC0 * static_cast<float>(getProperty(values, idxFdc1)));
            b = clamp01(0.5f + kSHC0 * static_cast<float>(getProperty(values, idxFdc2)));
        }
        else if (idxRed >= 0 && idxGreen >= 0 && idxBlue >= 0)
        {
            // Standard PLY vertex color.
            // Usually red/green/blue are 0..255.
            r = static_cast<float>(getProperty(values, idxRed)) / 255.0f;
            g = static_cast<float>(getProperty(values, idxGreen)) / 255.0f;
            b = static_cast<float>(getProperty(values, idxBlue)) / 255.0f;

            r = clamp01(r);
            g = clamp01(g);
            b = clamp01(b);
        }

        float opacity = 1.0f;

        if (idxOpacity >= 0)
        {
            // 3DGS usually stores opacity in logit space.
            opacity = sigmoid(static_cast<float>(getProperty(values, idxOpacity)));
        }
        else if (idxAlpha >= 0)
        {
            opacity = static_cast<float>(getProperty(values, idxAlpha)) / 255.0f;
            opacity = clamp01(opacity);
        }

        float sx = 0.03f;
        float sy = 0.03f;

        if (idxScale0 >= 0 && idxScale1 >= 0)
        {
            // 3DGS usually stores scale in log space.
            sx = std::exp(static_cast<float>(getProperty(values, idxScale0)));
            sy = std::exp(static_cast<float>(getProperty(values, idxScale1)));

            // For a simple billboard renderer, keep this conservative.
            // You can expose this as a node attribute later.
            constexpr float viewportScaleMultiplier = 1.0f;

            sx *= viewportScaleMultiplier;
            sy *= viewportScaleMultiplier;
        }
        else if (idxScale2 >= 0)
        {
            const float s = std::exp(static_cast<float>(getProperty(values, idxScale2)));
            sx = s;
            sy = s;
        }

        // Safety clamp. Some PLYs can contain extremely large splats.
        sx = std::max(0.0001f, std::min(sx, 10.0f));
        sy = std::max(0.0001f, std::min(sy, 10.0f));

        splat.opacity = opacity;
        splat.color = MColor(r, g, b, opacity);
        splat.scaleX = sx;
        splat.scaleY = sy;

        return splat;
    }

    bool readAsciiVertices(
        std::istream& input,
        const PlyHeader& header,
        std::vector<GS::GaussianSplat>& outSplats,
        std::string* outError)
    {
        outSplats.clear();
        outSplats.reserve(header.vertexCount);

        std::string line;

        for (std::size_t vertexIndex = 0; vertexIndex < header.vertexCount; ++vertexIndex)
        {
            if (!std::getline(input, line))
            {
                if (outError)
                {
                    *outError = "Unexpected end of ASCII PLY vertex data.";
                }

                return false;
            }

            std::istringstream ss(line);
            std::vector<double> values;
            values.resize(header.vertexProperties.size(), 0.0);

            for (std::size_t propertyIndex = 0;
                propertyIndex < header.vertexProperties.size();
                ++propertyIndex)
            {
                ss >> values[propertyIndex];

                if (!ss)
                {
                    if (outError)
                    {
                        *outError = "Failed to read ASCII PLY vertex property.";
                    }

                    return false;
                }
            }

            outSplats.push_back(
                makeSplatFromProperties(values, header.vertexProperties));
        }

        return true;
    }

    bool readBinaryLittleEndianVertices(
        std::istream& input,
        const PlyHeader& header,
        std::vector<GS::GaussianSplat>& outSplats,
        std::string* outError)
    {
        if (!isLittleEndianHost())
        {
            if (outError)
            {
                *outError = "binary_little_endian PLY loading on big-endian CPU is not implemented.";
            }

            return false;
        }

        outSplats.clear();
        outSplats.reserve(header.vertexCount);

        for (std::size_t vertexIndex = 0; vertexIndex < header.vertexCount; ++vertexIndex)
        {
            std::vector<double> values;
            values.resize(header.vertexProperties.size(), 0.0);

            for (std::size_t propertyIndex = 0;
                propertyIndex < header.vertexProperties.size();
                ++propertyIndex)
            {
                const PlyProperty& property = header.vertexProperties[propertyIndex];

                if (scalarTypeSize(property.type) == 0)
                {
                    if (outError)
                    {
                        *outError = "Invalid binary PLY scalar property.";
                    }

                    return false;
                }

                values[propertyIndex] = readBinaryScalar(input, property.type);

                if (!input)
                {
                    if (outError)
                    {
                        *outError = "Unexpected end of binary PLY vertex data.";
                    }

                    return false;
                }
            }

            outSplats.push_back(
                makeSplatFromProperties(values, header.vertexProperties));
        }

        return true;
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
        if (outError)
        {
            *outError = "Could not open PLY file: " + filePath;
        }

        return false;
    }

    PlyHeader header;

    if (!parseHeader(input, header, outError))
    {
        return false;
    }

    if (header.format == PlyFormat::Unknown)
    {
        if (outError)
        {
            *outError = "PLY format is missing or unsupported.";
        }

        return false;
    }

    if (header.vertexCount == 0)
    {
        if (outError)
        {
            *outError = "PLY file has no vertices.";
        }

        return false;
    }

    if (header.vertexProperties.empty())
    {
        if (outError)
        {
            *outError = "PLY file has no vertex properties.";
        }

        return false;
    }

    const int idxX = findPropertyIndex(header.vertexProperties, "x");
    const int idxY = findPropertyIndex(header.vertexProperties, "y");
    const int idxZ = findPropertyIndex(header.vertexProperties, "z");

    if (idxX < 0 || idxY < 0 || idxZ < 0)
    {
        if (outError)
        {
            *outError = "PLY file is missing required x/y/z vertex properties.";
        }

        return false;
    }

    bool success = false;

    if (header.format == PlyFormat::Ascii)
    {
        success = readAsciiVertices(input, header, outSplats, outError);
    }
    else if (header.format == PlyFormat::BinaryLittleEndian)
    {
        success = readBinaryLittleEndianVertices(input, header, outSplats, outError);
    }

    if (!success)
    {
        outSplats.clear();
        return false;
    }

    return true;
}