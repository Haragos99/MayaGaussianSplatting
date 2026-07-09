#pragma once
#include "data.h"


class GaussianSplatPlyLoader
{
public:
    static bool load(
        const std::string& filePath,
        std::vector<GS::GaussianSplat>& outSplats,
        std::string* outError = nullptr);

private:
    GaussianSplatPlyLoader() = delete;
};
