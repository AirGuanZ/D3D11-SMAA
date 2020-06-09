D3D11-SMAA is a header-only implementation of MLAA and SMAA for DirectX 11.

![pic](./gallery/hello_world.png)

## Getting Started

Simply include `./include/agz/smaa/mlaa.h` or `./include/agz/smaa/smaa.h` in your project.

## Building Example

Run following cmds in project directory:

```
mkdir build
cd build
cmake ..
```

The algorithm implementation in `mlaa.h` and `smaa.h` is C++11 compatible. However, the example project and its dependencies require a compiler with C++17 support and Win10 SDK. Thus, `Visual Studio 2017/2019` is recommended for building the example.

## Usage

1. Create a SMAA/MLAA instance with D3D device/deviceContext and algorithm parameters:

   ```cpp
   // for MLAA:
   auto mlaa = std::make_unique<agz::mlaa::MLAA>(
               	device, deviceContext,
               	targetSize.x, targetSize.y,
               	agz::mlaa::EdgeDetectionMode::Lum,
               	edgeThreshold, maxSearchDistanceLen);
   
   // or for SMAA:
   auto smaa = std::make_unique<agz::smaa::SMAA>(
               	device, deviceContext,
               	targetSize.x, targetSize.y,
               	agz::smaa::EdgeDetectionMode::Lum,
               	edgeThreshold, edgeLocalContrastFactor,
               	maxSearchDistanceLen, cornerAreaFactor);
   ```

2. In each frame:

   ```cpp
   bind and clear render target for edge detection
   smaa/* or mlaa */.detectEdge(inputImage/* or depthTexture */)
       
   bind and clear render target for blending weight computation
   smaa/* or mlaa */.computeBlendingWeight(edgeTexture)
       
   bind and clear render target for final result
   smaa/* or mlaa */.blend(weightTexture, inputImage)
   ```

The meanings of algorithm parameters and requirements of render targets are explained in `mlaa.h` and `smaa.h`. However, reading the original paper is still highly recommended.

## TODO

- [ ] Diagonal pattern handling

## Reference

http://www.iryoku.com/mlaa/

http://www.iryoku.com/smaa/