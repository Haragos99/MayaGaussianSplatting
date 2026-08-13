#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace GS::Ply
{
    enum class Format
    {
        Unknown,
        Ascii,
        BinaryLittleEndian
    };

    enum class ScalarType
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

    struct Property
    {
        std::string name;
        ScalarType type = ScalarType::Invalid;
    };

    struct Element
    {
        std::string name;
        std::size_t count = 0;
        std::vector<Property> properties;
    };

    struct Header
    {
        Format format = Format::Unknown;
        std::vector<Element> elements;
    };
}
