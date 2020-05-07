#pragma once

#include <wrl/client.h>

#include <d3d11.h>

namespace agz { namespace smaa {

using Microsoft::WRL::ComPtr;

// ========= common =========

class Common
{
public:

    Common(
        ID3D11Device *device,
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

// ========= edge detection =========

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

//  ========= SMAA =========

class SMAA
{
public:

    SMAA(
        ID3D11Device        *device,
        ID3D11DeviceContext *deviceContext,
        float                edgeThreshold,
        float                localContractFactor);
    
    void detectEdge(
        ID3D11ShaderResourceView *img) const;

private:

    Common common_;
    EdgeDetection edgeDetection_;
};

} } // namespace agz::smaa
