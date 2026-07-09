# MayaGaussianSplatting

A C++ plugin for **Autodesk Maya** that enables **3D Gaussian Splatting (3DGS)** visualization directly in **Viewport 2.0** using the **Maya API**. The plugin introduces a custom Gaussian Splat render node that renders Gaussian splats efficiently on the GPU while integrating seamlessly into the Maya scene.

## Features

- Native Gaussian Splat rendering inside Autodesk Maya
- Custom Maya render node for Gaussian Splat datasets
- GPU-accelerated rendering using Maya Viewport 2.0
- Built entirely in C++ using the Autodesk Maya API
- Uses Maya's rendering framework to leverage the GPU for high-performance visualization
- Interactive rendering directly in the Maya viewport
- Integrates with standard Maya workflows and scene management

## Demo

![Gaussian Splatting Demo](resources/GF.gif)

## How It Works

The plugin adds a custom Gaussian Splat render node to Maya. Instead of converting Gaussian splats into polygonal geometry, the node renders them directly through **Viewport 2.0**.

Rendering is performed using the **Autodesk Maya Viewport 2.0 API**, which provides access to the graphics pipeline and GPU resources. Vertex buffers, shaders, textures, and render items are managed through the Maya rendering API, allowing the plugin to utilize the GPU while remaining fully integrated with Maya's rendering architecture.

## Technologies

- C++
- Autodesk Maya API
- Maya Viewport 2.0 (VP2)
- GPU Rendering
- CMake

## Requirements

- Autodesk Maya
- C++17 compatible compiler
- CMake
- Autodesk Maya C++ SDK

## Future Work

- Selection and manipulation in the viewport
- Animation support
- Level-of-detail (LOD) rendering
- Improved rendering performance
- Additional import/export formats

