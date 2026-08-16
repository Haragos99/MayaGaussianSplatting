#include "PlyElementReader.h"

#include <cstdint>
#include <sstream>

namespace GS::Ply
{
    namespace
    {
        std::size_t scalarTypeSize(ScalarType type)
        {
            switch (type)
            {
            case ScalarType::Int8:
            case ScalarType::UInt8: 
                return 1;
            case ScalarType::Int16:
            case ScalarType::UInt16: 
                return 2;
            case ScalarType::Int32:
            case ScalarType::UInt32:
            case ScalarType::Float32:
                return 4;
            case ScalarType::Float64: 
                return 8;
            default: 
                return 0;
            }
        }

        template <typename ValueType>
        ValueType readBinaryValue(std::istream& input)
        {
            ValueType value{};
            input.read(reinterpret_cast<char*>(&value), sizeof(ValueType));
            return value;
        }

        double readBinaryScalar(std::istream& input, ScalarType type)
        {
            switch (type)
            {
            case ScalarType::Int8: 
                return static_cast<double>(readBinaryValue<std::int8_t>(input));
            case ScalarType::UInt8: 
                return static_cast<double>(readBinaryValue<std::uint8_t>(input));
            case ScalarType::Int16: 
                return static_cast<double>(readBinaryValue<std::int16_t>(input));
            case ScalarType::UInt16:
                return static_cast<double>(readBinaryValue<std::uint16_t>(input));
            case ScalarType::Int32:
                return static_cast<double>(readBinaryValue<std::int32_t>(input));
            case ScalarType::UInt32:
                return static_cast<double>(readBinaryValue<std::uint32_t>(input));
            case ScalarType::Float32:
                return static_cast<double>(readBinaryValue<float>(input));
            case ScalarType::Float64: 
                return readBinaryValue<double>(input);
            default: return 0.0;
            }
        }
    }

    bool readAsciiElement(
        std::istream& input,
        const Element& element,
        Records& records,
        std::string* outError)
    {
        records.clear();
        records.reserve(element.count);

        std::string line;
        for (std::size_t recordIndex = 0; recordIndex < element.count; ++recordIndex)
        {
            if (!std::getline(input, line))
            {
                if (outError)
                {
                    *outError = "Unexpected end of ASCII PLY element data.";
                }

                return false;
            }

            std::istringstream lineStream(line);
            Record record(element.properties.size(), 0.0);

            for (std::size_t propertyIndex = 0; propertyIndex < element.properties.size(); ++propertyIndex)
            {
                lineStream >> record[propertyIndex];
                if (!lineStream)
                {
                    if (outError)
                    {
                        *outError = "Failed to read ASCII PLY element property.";
                    }

                    return false;
                }
            }

            records.push_back(record);
        }

        return true;
    }

    bool readBinaryElement(
        std::istream& input,
        const Element& element,
        Records& records,
        std::string* outError)
    {
        records.assign(element.count, Record(element.properties.size(), 0.0));

        for (Record& record : records)
        {
            for (std::size_t propertyIndex = 0; propertyIndex < element.properties.size(); ++propertyIndex)
            {
                const Property& property = element.properties[propertyIndex];
                if (scalarTypeSize(property.type) == 0)
                {
                    if (outError)
                    {
                        *outError = "Invalid binary PLY scalar property.";
                    }

                    return false;
                }

                record[propertyIndex] = readBinaryScalar(input, property.type);
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
}
