#pragma once

#include <istream>
#include <string>
#include <vector>

#include "PlyTypes.h"

namespace GS::Ply
{
    using Record = std::vector<double>;
    using Records = std::vector<Record>;

    bool readAsciiElement(
        std::istream& input,
        const Element& element,
        Records& records,
        std::string* outError);

    bool readBinaryElement(
        std::istream& input,
        const Element& element,
        Records& records,
        std::string* outError);
}
