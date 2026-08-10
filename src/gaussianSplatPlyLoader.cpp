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

    struct PlyElement
    {
        std::string name;
        std::size_t count = 0;
        std::vector<PlyProperty> properties;
    };

    struct PlyHeader
    {
        PlyFormat format = PlyFormat::Unknown;
        std::vector<PlyElement> elements;
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

        int currentElementIndex = -1;

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
                PlyElement element;
                ss >> element.name >> element.count;

                header.elements.push_back(element);
                currentElementIndex = static_cast<int>(header.elements.size()) - 1;
            }
            else if (token == "property" && currentElementIndex >= 0)
            {
                std::string typeName;
                std::string propertyName;

                ss >> typeName;

                // This loader only supports scalar properties.
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

                header.elements[static_cast<std::size_t>(currentElementIndex)]
                    .properties.push_back(property);
            }
        }

        if (outError)
        {
            *outError = "PLY header ended unexpectedly.";
        }

        return false;
    }

    const PlyElement* findElement(const PlyHeader& header, const std::string& name)
    {
        for (const PlyElement& element : header.elements)
        {
            if (element.name == name)
            {
                return &element;
            }
        }

        return nullptr;
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
        const PlyElement& vertexElement,
        std::vector<GS::GaussianSplat>& outSplats,
        std::string* outError)
    {
        outSplats.clear();
        outSplats.reserve(vertexElement.count);

        std::string line;

        for (std::size_t vertexIndex = 0; vertexIndex < vertexElement.count; ++vertexIndex)
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
            values.resize(vertexElement.properties.size(), 0.0);

            for (std::size_t propertyIndex = 0;
                propertyIndex < vertexElement.properties.size();
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
                makeSplatFromProperties(values, vertexElement.properties));
        }

        return true;
    }

    bool readBinaryLittleEndianVertices(
        std::istream& input,
        const PlyElement& vertexElement,
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
        outSplats.reserve(vertexElement.count);

        for (std::size_t vertexIndex = 0; vertexIndex < vertexElement.count; ++vertexIndex)
        {
            std::vector<double> values;
            values.resize(vertexElement.properties.size(), 0.0);

            for (std::size_t propertyIndex = 0;
                propertyIndex < vertexElement.properties.size();
                ++propertyIndex)
            {
                const PlyProperty& property = vertexElement.properties[propertyIndex];

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
                makeSplatFromProperties(values, vertexElement.properties));
        }

        return true;
    }

    // ---------------------------------------------------------------------
    // SuperSplat / PlayCanvas "compressed" PLY support.
    //
    // Layout:
    //   element chunk  N   -> per-chunk min/max ranges (256 gaussians / chunk)
    //   element vertex M   -> packed_position/rotation/scale/color (uint32 each)
    //   element sh     M   -> optional higher-order SH (ignored here)
    // ---------------------------------------------------------------------

    constexpr std::size_t kCompressedChunkSize = 256;

    bool isCompressedSuperSplat(const PlyHeader& header)
    {
        const PlyElement* chunk = findElement(header, "chunk");
        const PlyElement* vertex = findElement(header, "vertex");

        if (chunk == nullptr || vertex == nullptr)
        {
            return false;
        }

        return findPropertyIndex(vertex->properties, "packed_position") >= 0
            && findPropertyIndex(vertex->properties, "packed_scale") >= 0
            && findPropertyIndex(vertex->properties, "packed_color") >= 0;
    }

    // Read one element's records into a flat [record][property] value table.
    bool readElementValues(
        std::istream& input,
        const PlyElement& element,
        std::vector<std::vector<double>>& outRecords,
        std::string* outError)
    {
        outRecords.assign(element.count, std::vector<double>(element.properties.size(), 0.0));

        for (std::size_t record = 0; record < element.count; ++record)
        {
            for (std::size_t propertyIndex = 0;
                propertyIndex < element.properties.size();
                ++propertyIndex)
            {
                const PlyProperty& property = element.properties[propertyIndex];

                if (scalarTypeSize(property.type) == 0)
                {
                    if (outError)
                    {
                        *outError = "Invalid binary PLY scalar property.";
                    }

                    return false;
                }

                outRecords[record][propertyIndex] = readBinaryScalar(input, property.type);

                if (!input)
                {
                    if (outError)
                    {
                        *outError = "Unexpected end of binary PLY element data.";
                    }

                    return false;
                }
            }
        }

        return true;
    }

    float lerp(float a, float b, float t)
    {
        return a + (b - a) * t;
    }

    // Unpack an 11/10/11-bit normalized triple from a 32-bit value.
    void unpack111011(std::uint32_t value, float& x, float& y, float& z)
    {
        x = static_cast<float>((value >> 21) & 0x7FFu) / 2047.0f;
        y = static_cast<float>((value >> 11) & 0x3FFu) / 1023.0f;
        z = static_cast<float>(value & 0x7FFu) / 2047.0f;
    }

    bool readCompressedVertices(
        std::istream& input,
        const PlyHeader& header,
        std::vector<GS::GaussianSplat>& outSplats,
        std::string* outError)
    {
        if (header.format != PlyFormat::BinaryLittleEndian)
        {
            if (outError)
            {
                *outError = "Compressed SuperSplat PLY must be binary_little_endian.";
            }

            return false;
        }

        if (!isLittleEndianHost())
        {
            if (outError)
            {
                *outError = "Compressed PLY loading on big-endian CPU is not implemented.";
            }

            return false;
        }

        outSplats.clear();

        // Elements are stored back-to-back in header order; read them in order.
        std::vector<std::vector<double>> chunkRecords;
        std::vector<std::vector<double>> vertexRecords;

        const PlyElement* vertexElement = nullptr;
        const PlyElement* chunkElement = nullptr;

        for (const PlyElement& element : header.elements)
        {
            if (element.name == "chunk")
            {
                chunkElement = &element;

                if (!readElementValues(input, element, chunkRecords, outError))
                {
                    return false;
                }
            }
            else if (element.name == "vertex")
            {
                vertexElement = &element;

                if (!readElementValues(input, element, vertexRecords, outError))
                {
                    return false;
                }

                // Everything after the vertex element (e.g. sh) is not needed
                // for the billboard renderer, so we can stop reading here.
                break;
            }
            else
            {
                // Skip any element preceding vertex that we do not consume.
                std::vector<std::vector<double>> discard;

                if (!readElementValues(input, element, discard, outError))
                {
                    return false;
                }
            }
        }

        if (vertexElement == nullptr || chunkElement == nullptr)
        {
            if (outError)
            {
                *outError = "Compressed PLY is missing chunk or vertex element.";
            }

            return false;
        }

        const std::vector<PlyProperty>& cp = chunkElement->properties;

        const int cMinX = findPropertyIndex(cp, "min_x");
        const int cMinY = findPropertyIndex(cp, "min_y");
        const int cMinZ = findPropertyIndex(cp, "min_z");
        const int cMaxX = findPropertyIndex(cp, "max_x");
        const int cMaxY = findPropertyIndex(cp, "max_y");
        const int cMaxZ = findPropertyIndex(cp, "max_z");

        const int cMinSx = findPropertyIndex(cp, "min_scale_x");
        const int cMinSy = findPropertyIndex(cp, "min_scale_y");
        const int cMinSz = findPropertyIndex(cp, "min_scale_z");
        const int cMaxSx = findPropertyIndex(cp, "max_scale_x");
        const int cMaxSy = findPropertyIndex(cp, "max_scale_y");
        const int cMaxSz = findPropertyIndex(cp, "max_scale_z");

        const int cMinR = findPropertyIndex(cp, "min_r");
        const int cMinG = findPropertyIndex(cp, "min_g");
        const int cMinB = findPropertyIndex(cp, "min_b");
        const int cMaxR = findPropertyIndex(cp, "max_r");
        const int cMaxG = findPropertyIndex(cp, "max_g");
        const int cMaxB = findPropertyIndex(cp, "max_b");

        if (cMinX < 0 || cMinY < 0 || cMinZ < 0 || cMaxX < 0 || cMaxY < 0 || cMaxZ < 0)
        {
            if (outError)
            {
                *outError = "Compressed PLY chunk is missing position range properties.";
            }

            return false;
        }

        const std::vector<PlyProperty>& vp = vertexElement->properties;

        const int vPos = findPropertyIndex(vp, "packed_position");
        const int vScale = findPropertyIndex(vp, "packed_scale");
        const int vColor = findPropertyIndex(vp, "packed_color");

        if (vPos < 0 || vColor < 0)
        {
            if (outError)
            {
                *outError = "Compressed PLY vertex is missing packed_position/packed_color.";
            }

            return false;
        }

        outSplats.reserve(vertexRecords.size());

        for (std::size_t i = 0; i < vertexRecords.size(); ++i)
        {
            const std::vector<double>& v = vertexRecords[i];

            const std::size_t chunkIndex = i / kCompressedChunkSize;

            if (chunkIndex >= chunkRecords.size())
            {
                if (outError)
                {
                    *outError = "Compressed PLY vertex references a missing chunk.";
                }

                return false;
            }

            const std::vector<double>& c = chunkRecords[chunkIndex];

            GS::GaussianSplat splat;

            // --- Position ---
            {
                float nx = 0.0f;
                float ny = 0.0f;
                float nz = 0.0f;
                unpack111011(static_cast<std::uint32_t>(v[static_cast<std::size_t>(vPos)]), nx, ny, nz);

                const float x = lerp(
                    static_cast<float>(c[static_cast<std::size_t>(cMinX)]),
                    static_cast<float>(c[static_cast<std::size_t>(cMaxX)]), nx);
                const float y = lerp(
                    static_cast<float>(c[static_cast<std::size_t>(cMinY)]),
                    static_cast<float>(c[static_cast<std::size_t>(cMaxY)]), ny);
                const float z = lerp(
                    static_cast<float>(c[static_cast<std::size_t>(cMinZ)]),
                    static_cast<float>(c[static_cast<std::size_t>(cMaxZ)]), nz);

                splat.center = MPoint(x, y, z);
            }

            // --- Scale (stored in log space, same as standard 3DGS) ---
            float sx = 0.03f;
            float sy = 0.03f;

            if (vScale >= 0 && cMinSx >= 0 && cMinSy >= 0 && cMaxSx >= 0 && cMaxSy >= 0)
            {
                float nsx = 0.0f;
                float nsy = 0.0f;
                float nsz = 0.0f;
                unpack111011(static_cast<std::uint32_t>(v[static_cast<std::size_t>(vScale)]), nsx, nsy, nsz);

                const float logSx = lerp(
                    static_cast<float>(c[static_cast<std::size_t>(cMinSx)]),
                    static_cast<float>(c[static_cast<std::size_t>(cMaxSx)]), nsx);
                const float logSy = lerp(
                    static_cast<float>(c[static_cast<std::size_t>(cMinSy)]),
                    static_cast<float>(c[static_cast<std::size_t>(cMaxSy)]), nsy);

                sx = std::exp(logSx);
                sy = std::exp(logSy);
            }

            sx = std::max(0.0001f, std::min(sx, 10.0f));
            sy = std::max(0.0001f, std::min(sy, 10.0f));

            // --- Color + opacity (packed_color = 8.8.8.8 RGBA) ---
            const std::uint32_t packedColor =
                static_cast<std::uint32_t>(v[static_cast<std::size_t>(vColor)]);

            float r = static_cast<float>((packedColor >> 24) & 0xFFu) / 255.0f;
            float g = static_cast<float>((packedColor >> 16) & 0xFFu) / 255.0f;
            float b = static_cast<float>((packedColor >> 8) & 0xFFu) / 255.0f;
            const float a = static_cast<float>(packedColor & 0xFFu) / 255.0f;

            if (cMinR >= 0 && cMaxR >= 0 && cMinG >= 0 && cMaxG >= 0 && cMinB >= 0 && cMaxB >= 0)
            {
                r = lerp(
                    static_cast<float>(c[static_cast<std::size_t>(cMinR)]),
                    static_cast<float>(c[static_cast<std::size_t>(cMaxR)]), r);
                g = lerp(
                    static_cast<float>(c[static_cast<std::size_t>(cMinG)]),
                    static_cast<float>(c[static_cast<std::size_t>(cMaxG)]), g);
                b = lerp(
                    static_cast<float>(c[static_cast<std::size_t>(cMinB)]),
                    static_cast<float>(c[static_cast<std::size_t>(cMaxB)]), b);
            }

            r = clamp01(r);
            g = clamp01(g);
            b = clamp01(b);

            const float opacity = clamp01(a);

            splat.opacity = opacity;
            splat.color = MColor(r, g, b, opacity);
            splat.scaleX = sx;
            splat.scaleY = sy;

            outSplats.push_back(splat);
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

    // Compressed SuperSplat / PlayCanvas PLY (packed chunk/vertex/sh layout).
    if (isCompressedSuperSplat(header))
    {
        if (!readCompressedVertices(input, header, outSplats, outError))
        {
            outSplats.clear();
            return false;
        }

        return true;
    }

    // Standard (uncompressed) PLY path.
    const PlyElement* vertexElement = findElement(header, "vertex");

    if (vertexElement == nullptr || vertexElement->count == 0)
    {
        if (outError)
        {
            *outError = "PLY file has no vertices.";
        }

        return false;
    }

    if (vertexElement->properties.empty())
    {
        if (outError)
        {
            *outError = "PLY file has no vertex properties.";
        }

        return false;
    }

    const int idxX = findPropertyIndex(vertexElement->properties, "x");
    const int idxY = findPropertyIndex(vertexElement->properties, "y");
    const int idxZ = findPropertyIndex(vertexElement->properties, "z");

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
        success = readAsciiVertices(input, *vertexElement, outSplats, outError);
    }
    else if (header.format == PlyFormat::BinaryLittleEndian)
    {
        success = readBinaryLittleEndianVertices(input, *vertexElement, outSplats, outError);
    }

    if (!success)
    {
        outSplats.clear();
        return false;
    }

    return true;
}