#pragma once

#include <wrl/client.h>

#include <d3d11.h>

namespace agz { namespace smaa {

using Microsoft::WRL::ComPtr;

class Common
{
public:

    Common(
        ID3D11Device        *device,
        ID3D11DeviceContext *deviceContext);

    void bindVertex() const;

    void unbindVertex() const;

    ID3D11Device        *D;
    ID3D11DeviceContext *DC;

    ComPtr<ID3D11VertexShader> vertexShader;
    ComPtr<ID3D11Buffer>       vertexBuffer;
    ComPtr<ID3D11InputLayout>  inputLayout;

    ComPtr<ID3D11SamplerState> pointSampler;
    ComPtr<ID3D11SamplerState> linearSampler;
};

class EdgeDetection
{
public:

    EdgeDetection(
        ID3D11Device *device,
        float         edgeThreshold,
        float         localContractFactor);

    void detectEdge(
        const Common             &common,
        ID3D11ShaderResourceView *img) const;

    ComPtr<ID3D11PixelShader> pixelShader;
};

class BlendingWeight
{
public:

    BlendingWeight(
        ID3D11Device *device,
        int           maxSearchDistanceLen,
        float         cornerAreaFactor,
        int           width,
        int           height);

    void computeBlendingWeight(
        const Common             &common,
        ID3D11ShaderResourceView *edgeTexture);

    ComPtr<ID3D11PixelShader> pixelShader;

    ComPtr<ID3D11Texture2D>          innerAreaTexture;
    ComPtr<ID3D11ShaderResourceView> innerAreaTextureSRV;
};

class Blending
{
public:

    Blending(
        ID3D11Device *device,
        int           width,
        int           height);

    void blend(
        const Common             &common,
        ID3D11ShaderResourceView *weightTexture,
        ID3D11ShaderResourceView *img) const;

    ComPtr<ID3D11PixelShader> pixelShader;
};

//  ========= SMAA =========

class SMAA
{
public:

    SMAA(
        ID3D11Device        *device,
        ID3D11DeviceContext *deviceContext,
        float                edgeThreshold,
        float                localContractFactor,
        int                  maxSearchDistanceLen,
        float                cornerAreaFactor,
        int                  width,
        int                  height);
    
    void detectEdge(
        ID3D11ShaderResourceView *img) const;
    
    void computeBlendingWeight(
        ID3D11ShaderResourceView *edgeTexture);

    void blend(
        ID3D11ShaderResourceView *weightTexture,
        ID3D11ShaderResourceView *img);

private:

    Common         common_;
    EdgeDetection  edgeDetection_;
    BlendingWeight blendingWeight_;
    Blending       blending_;
};

} } // namespace agz::smaa
