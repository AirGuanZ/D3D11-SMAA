#pragma once

#include <vector>

#include <wrl/client.h>
#include <d3d11.h>

#define AGZ_SMAA_BEGIN namespace agz { namespace smaa {
#define AGZ_SMAA_END   } }

AGZ_SMAA_BEGIN

using Microsoft::WRL::ComPtr;

/*
    MLAA algorithm:
        1. edge detection
        2. blending weight computation
        3. blending
    each pass is performed by drawing a full-screen triangle

    MLAA.performOnePass():
        bind vertex buffer, shader and set uniforms variables
        emit a drawcall

    the user should bind and clear the corresponding render target
*/

class MLAA
{
public:

    enum class Mode
    {
        Depth,
        Lum
    };

    MLAA(
        ID3D11Device        *device,
        ID3D11DeviceContext *deviceContext,
        int                  maxEdgeDetectionLen,
        float                edgeThreshold,
        Mode                 edgeDetectionMode,
        int                  framebufferWidth,
        int                  framebufferHeight);

    void DetectEdgeWithLum(ID3D11ShaderResourceView *img);

    void DetectEdgeWithDepth(ID3D11ShaderResourceView *depth);

    void ComputeBlendingWeight(ID3D11ShaderResourceView *edge);

    void PerformBlending(
        ID3D11ShaderResourceView *img,
        ID3D11ShaderResourceView *weight);

    ID3D11Texture2D *GetInnerAreaTexture() noexcept;

    ID3D11ShaderResourceView *GetInnerAreaTextureSRV() noexcept;

private:

    ComPtr<ID3D10Blob> InitVertexShader(ID3D11Device *device);

    void InitEdgeShader(
        ID3D11Device *device,
        float         edgeThreshold,
        Mode          mode);

    void InitWeightShader(
        ID3D11Device *device,
        int           maxEdgeDetectionLen);

    void InitBlendingShader(ID3D11Device *device);

    void InitVertexBuffer(ID3D11Device *device);

    void InitInputLayout(
        ID3D11Device      *device,
        ComPtr<ID3D10Blob> shaderByteCode);

    void InitSamplers(ID3D11Device *device);

    void InitInnerAreaTexture(
        ID3D11Device *device,
        int           maxEdgeDetectionLen);

    void InitPixelSizeConstant(
        ID3D11Device *device,
        int           framebufferWidth,
        int           framebufferHeight);

    std::vector<float> GenerateInnerAreaTexture(
        int maxEdgeDetectionLen,
        int *width, int *height) const noexcept;

    // device context

    ID3D11DeviceContext *DC_;

    // shaders

    ComPtr<ID3D10Blob>         vertexShaderByteCode_;
    ComPtr<ID3D11VertexShader> vertexShader_;

    ComPtr<ID3D11PixelShader> edgeShader_;
    ComPtr<ID3D11PixelShader> weightShader_;
    ComPtr<ID3D11PixelShader> blendingShader_;

    // vertex buffer & input layout

    struct Vertex
    {
        float x, y;
        float u, v;
    };

    ComPtr<ID3D11Buffer> vertexBuffer_;

    ComPtr<ID3D11InputLayout> inputLayout_;

    // samplers

    ComPtr<ID3D11SamplerState> pointSampler_;
    ComPtr<ID3D11SamplerState> linearSampler_;

    // inner area texture

    ComPtr<ID3D11Texture2D>          innerAreaTexture_;
    ComPtr<ID3D11ShaderResourceView> innerAreaTextureSRV_;

    // constant buffer in weight shader

    struct PixelSizeConstant
    {
        float PIXEL_SIZE_IN_TEXCOORD_X;
        float PIXEL_SIZE_IN_TEXCOORD_Y;
        float pad0;
        float pad1;
    };

    ComPtr<ID3D11Buffer> pixelSizeConstantBuffer_;
};

AGZ_SMAA_END
