#pragma once

#include <istream>
#include <string>

#include "PlyTypes.h"

namespace GS::Ply
{
    bool parseHeader(
        std::istream& input,
        Header& header,
        std::string* outError);

    const Element* findElement(
        const Header& header,
        const std::string& elementName);

    int findPropertyIndex(
        const std::vector<Property>& properties,
        const std::string& propertyName);
}
