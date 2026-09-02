cmake_minimum_required(VERSION 3.19)

project(MayaGaussianSplatting)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Maya is built with /MD
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")

# -----------------------------------------------------------------------------
# Find Maya SDK
# -----------------------------------------------------------------------------

if(NOT DEFINED MAYA_SDK_ROOT)
    if(WIN32)
        file(GLOB MAYA_VERSIONS "C:/Program Files/Autodesk/Maya*")
        list(SORT MAYA_VERSIONS ORDER DESCENDING)

        if(MAYA_VERSIONS)
            list(GET MAYA_VERSIONS 0 MAYA_SDK_ROOT)
            set(MAYA_SDK_ROOT "${MAYA_SDK_ROOT}" CACHE PATH "Path to Maya SDK")
        else()
            message(FATAL_ERROR
                "No Maya installation found.\n"
                "Please specify -DMAYA_SDK_ROOT=\"C:/Program Files/Autodesk/Maya2026\"")
        endif()

    elseif(UNIX)

        file(GLOB MAYA_VERSIONS "/usr/autodesk/maya*")
        list(SORT MAYA_VERSIONS ORDER DESCENDING)

        if(MAYA_VERSIONS)
            list(GET MAYA_VERSIONS 0 MAYA_SDK_ROOT)
            set(MAYA_SDK_ROOT "${MAYA_SDK_ROOT}" CACHE PATH "Path to Maya SDK")
        else()
            message(FATAL_ERROR
                "No Maya installation found.\n"
                "Please specify -DMAYA_SDK_ROOT=/usr/autodesk/maya2026")
        endif()

    endif()
endif()

message(STATUS "Using Maya SDK: ${MAYA_SDK_ROOT}")

# Plugin target

add_library(MayaGaussianSplatting SHARED
    src/main.cpp
 "src/gaussiansplattingnode.h" 
 "src/gaussiansplattingnode.cpp" 
 "src/gaussiansplattingdraweriverrider.h" 
 "src/gaussiansplattingdraweriverrider.cpp"
 "src/data.h" "src/gaussianSplatPlyLoader.h" "src/gaussianSplatPlyLoader.cpp"
 "src/render/SplatBufferManager.h" "src/render/SplatBufferManager.cpp"
 "src/render/SplatDepthSorter.h" "src/render/SplatDepthSorter.cpp"
 "src/render/ViewportCamera.h" "src/render/ViewportCamera.cpp"
 "src/scene/SplatDataSource.h" "src/scene/SplatDataSource.cpp"
 "src/ply/PlyTypes.h"
 "src/ply/PlyHeaderParser.h" "src/ply/PlyHeaderParser.cpp"
 "src/ply/PlyElementReader.h" "src/ply/PlyElementReader.cpp"
 "src/ply/SplatDecoder.h"
 "src/ply/StandardSplatDecoder.h" "src/ply/StandardSplatDecoder.cpp"
 "src/ply/SuperSplatDecoder.h" "src/ply/SuperSplatDecoder.cpp" "src/splatCalculator.h" "src/splatCalculator.cpp")


# Include directories
target_include_directories(MayaGaussianSplatting PRIVATE
    "${MAYA_SDK_ROOT}/include"
)

# Compiler definitions required by Maya

target_compile_definitions(MayaGaussianSplatting PRIVATE
    NT_PLUGIN
    REQUIRE_IOSTREAM
    _BOOL
)


# Link libraries
target_link_libraries(MayaGaussianSplatting PRIVATE
    "${MAYA_SDK_ROOT}/lib/Foundation.lib"
    "${MAYA_SDK_ROOT}/lib/OpenMaya.lib"
    "${MAYA_SDK_ROOT}/lib/OpenMayaAnim.lib"
    "${MAYA_SDK_ROOT}/lib/OpenMayaUI.lib"
    "${MAYA_SDK_ROOT}/lib/OpenMayaFX.lib"
    "${MAYA_SDK_ROOT}/lib/OpenMayaRender.lib"
    "${MAYA_SDK_ROOT}/lib/Image.lib"
    opengl32.lib
)

# Export plugin entry points
if(MSVC)
    target_link_options(MayaGaussianSplatting PRIVATE
        /export:initializePlugin
        /export:uninitializePlugin
    )
endif()


# Maya plugin extension
set_target_properties(MayaGaussianSplatting PROPERTIES
    PREFIX ""
    SUFFIX ".mll"
)