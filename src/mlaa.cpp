#include <cmath>
#include <string>

#include <agz/smaa/mlaa.h>

#include "./helper.h"

AGZ_SMAA_BEGIN

namespace
{
    struct Vec2
    {
        float x = 0, y = 0;

        Vec2 operator+(const Vec2 &rhs) const noexcept
            { return { x + rhs.x, y + rhs.y }; }
        
        Vec2 operator-(const Vec2 &rhs) const noexcept
            { return { x - rhs.x, y - rhs.y }; }
    };

    Vec2 operator*(float lhs, const Vec2 &rhs) noexcept
        { return { lhs * rhs.x, lhs * rhs.y }; }

    /*
      given line segment ab and a pixel where pixel.x \in (a.x, b.x),
      compute areas of divided-by-line parts of the pixel

      returns: (a1, a2)
        a1: area in the lower pixel that
            should be covered by color of the upper pixel
        a2: area in the upper pixel that
            should be covered by color of the lower pixel
    */
    std::pair<float, float> ComputePixelInnerArea(
        const Vec2 &a, const Vec2 &b, int pixel)
    {
        const float xL = static_cast<float>(pixel);
        const float xR = static_cast<float>(pixel + 1);

        const float x0 = a.x, y0 = a.y;
        const float x1 = b.x, y1 = b.y;

        const float yL = y0 + (xL - x0) * (y1 - y0) / (x1 - x0);
        const float yR = y0 + (xR - x0) * (y1 - y0) / (x1 - x0);

        if((xL < x0 || xL >= x1) && (xR <= x0 || xR > x1))
            return { 0.0f, 0.0f };

        // case 1. one trapezoid

        if(std::abs(yL) < 1e-4f ||
           std::abs(yR) < 1e-4f ||
           ((yL > 0) == (yR > 0)))
        {
            const float area = (yL + yR) / 2;
            if(area < 0)
                return { 0.0f, -area };
            return { area, 0.0f };
        }

        // case 2. two triangles

        const float xM = -y0 * (x1 - x0) / (y1 - y0) + x0;
        
        float areaLeft  = std::abs(0.5f * yL * (xM - xL));
        float areaRight = std::abs(0.5f * yR * (xR - xM));

        if(xM <= x0)
            areaLeft = 0;

        if(xM >= x1)
            areaRight = 0;

        // left is higher than right
        if(yL < yR)
            return { areaRight, areaLeft };

        // otherwise
        return { areaLeft, areaRight };
    }

    std::pair<float, float> ComputePixelInnerArea(
        int dist1, int cross1, int dist2, int cross2)
    {
        const float dist = static_cast<float>(dist1) +
                           static_cast<float>(dist2) + 1;

        if(cross1 == 0 && cross2 == 0)
        {
            //
            // ----------
            //
            return { 0.0f, 0.0f };
        }

        if(cross1 == 0 && cross2 == 1)
        {
            //          |
            // ----------
            //
            return ComputePixelInnerArea(
                { dist / 2, 0 }, { dist, -0.5f }, dist1);
        }

        if(cross1 == 0 && cross2 == 3)
        {
            //
            // ----------
            //          |
            return ComputePixelInnerArea(
                { dist / 2, 0 }, { dist, 0.5f }, dist1);
        }

        if(cross1 == 0 && cross2 == 4)
        {
            //          |
            // ----------
            //          |
            return { 0.0f, 0.0f };
        }

        if(cross1 == 1 && cross2 == 0)
        {
            // |
            // ----------
            //
            return ComputePixelInnerArea(
                { 0, -0.5f }, { dist / 2, 0 }, dist1);
        }

        if(cross1 == 1 && cross2 == 1)
        {
            // |        |
            // ----------
            //
            const auto aL = ComputePixelInnerArea(
                { 0, -0.5f }, { dist / 2, 0.0f }, dist1);
            const auto aR = ComputePixelInnerArea(
                { dist / 2, 0.0f }, { dist, -0.5f }, dist1);
            return { aL.first + aR.first, aL.second + aR.second };
        }

        if(cross1 == 1 && cross2 == 3)
        {
            // |
            // ----------
            //          |
            return ComputePixelInnerArea(
                { 0, -0.5f }, { dist, 0.5f }, dist1);
        }

        if(cross1 == 1 && cross2 == 4)
        {
            // |        |
            // ----------
            //          |
            return ComputePixelInnerArea(
                { 0, -0.5f }, { dist, 0.5f }, dist1);
        }

        if(cross1 == 3 && cross2 == 0)
        {
            //
            // ----------
            // |
            return ComputePixelInnerArea(
                { 0, 0.5f }, { dist / 2, 0 }, dist1);
        }

        if(cross1 == 3 && cross2 == 1)
        {
            //          |
            // ----------
            // |
            return ComputePixelInnerArea(
                { 0, 0.5f }, { dist, -0.5f }, dist1);
        }

        if(cross1 == 3 && cross2 == 3)
        {
            //
            // ----------
            // |        |
            const auto aL = ComputePixelInnerArea(
                { 0, 0.5f }, { dist / 2, 0.0f }, dist1);
            const auto aR = ComputePixelInnerArea(
                { dist / 2, 0.0f }, { dist, 0.5f }, dist1);
            return { aL.first + aR.first, aL.second + aR.second };
        }

        if(cross1 == 3 && cross2 == 4)
        {
            //          |
            // ----------
            // |        |
            return ComputePixelInnerArea(
                { 0, 0.5f }, { dist, -0.5f }, dist1);
        }

        if(cross1 == 4 && cross2 == 0)
        {
            // |
            // ----------
            // |
            return { 0.0f, 0.0f };
        }

        if(cross1 == 4 && cross2 == 1)
        {
            // |        |
            // ----------
            // |
            return ComputePixelInnerArea(
                { 0, 0.5f }, { dist, -0.5f }, dist1);
        }

        if(cross1 == 4 && cross2 == 3)
        {
            // |
            // ----------
            // |        |
            return ComputePixelInnerArea(
                { 0, -0.5f }, { dist, 0.5f }, dist1);
        }

        if(cross1 == 4 && cross2 == 4)
        {
            // |        |
            // ----------
            // |        |
            return { 0.0f, 0.0f };
        }

        return { 0.0f, 0.0f };
    }
}

MLAA::MLAA(
    ID3D11Device        *device,
    ID3D11DeviceContext *deviceContext,
    int                  maxEdgeDetectionLen,
    float                edgeThreshold,
    Mode                 mode,
    int                  framebufferWidth,
    int                  framebufferHeight)
    : DC_(deviceContext)
{
    auto vsByteCode = InitVertexShader(device);

    InitEdgeShader       (device, edgeThreshold, mode);
    InitWeightShader     (device, maxEdgeDetectionLen);
    InitBlendingShader   (device);
    InitVertexBuffer     (device);
    InitInputLayout      (device, vsByteCode);
    InitSamplers         (device);
    InitInnerAreaTexture (device, maxEdgeDetectionLen);
    InitPixelSizeConstant(device, framebufferWidth, framebufferHeight);
}

void MLAA::DetectEdgeWithLum(ID3D11ShaderResourceView *img)
{
    // bind shader/texture/sampler

    DC_->VSSetShader(vertexShader_.Get(), nullptr, 0);
    DC_->PSSetShader(edgeShader_.Get(), nullptr, 0);
    DC_->PSSetSamplers(0, 1, pointSampler_.GetAddressOf());
    DC_->PSSetShaderResources(0, 1, &img);

    // bind vertex buffer/input layout

    const UINT stride = sizeof(Vertex), offset = 0;
    DC_->IASetVertexBuffers(
        0, 1, vertexBuffer_.GetAddressOf(), &stride, &offset);

    DC_->IASetInputLayout(inputLayout_.Get());
    DC_->IASetPrimitiveTopology(
        D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    // emit drawcall

    DC_->Draw(3, 0);

    // clear vertex buffer/input layout

    ID3D11Buffer *NULL_VERTEX_BUFFER = nullptr;
    DC_->IASetVertexBuffers(
        0, 1, &NULL_VERTEX_BUFFER, &stride, &offset);

    DC_->IASetInputLayout(nullptr);

    // clear shader/texture/sampler

    ID3D11SamplerState *NULL_SAMPLER = nullptr;
    DC_->PSSetSamplers(0, 1, &NULL_SAMPLER);

    ID3D11ShaderResourceView *NULL_IMG = nullptr;
    DC_->PSSetShaderResources(0, 1, &NULL_IMG);

    DC_->PSSetShader(nullptr, nullptr, 0);
    DC_->VSSetShader(nullptr, nullptr, 0);
}

void MLAA::DetectEdgeWithDepth(ID3D11ShaderResourceView *depth)
{
    // bind shader/texture/sampler

    DC_->VSSetShader(vertexShader_.Get(), nullptr, 0);
    DC_->PSSetShader(edgeShader_.Get(), nullptr, 0);
    DC_->PSSetSamplers(0, 1, pointSampler_.GetAddressOf());
    DC_->PSSetShaderResources(0, 1, &depth);

    // bind vertex buffer/input layout

    const UINT stride = sizeof(Vertex), offset = 0;
    DC_->IASetVertexBuffers(
        0, 1, vertexBuffer_.GetAddressOf(), &stride, &offset);

    DC_->IASetInputLayout(inputLayout_.Get());
    DC_->IASetPrimitiveTopology(
        D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    // emit drawcall

    DC_->Draw(3, 0);

    // clear vertex buffer/input layout

    ID3D11Buffer *NULL_VERTEX_BUFFER = nullptr;
    DC_->IASetVertexBuffers(
        0, 1, &NULL_VERTEX_BUFFER, &stride, &offset);

    DC_->IASetInputLayout(nullptr);

    // clear shader/texture/sampler

    ID3D11SamplerState *NULL_SAMPLER = nullptr;
    DC_->PSSetSamplers(0, 1, &NULL_SAMPLER);

    ID3D11ShaderResourceView *NULL_IMG = nullptr;
    DC_->PSSetShaderResources(0, 1, &NULL_IMG);

    DC_->PSSetShader(nullptr, nullptr, 0);
    DC_->VSSetShader(nullptr, nullptr, 0);
}

void MLAA::ComputeBlendingWeight(ID3D11ShaderResourceView *edge)
{
    // bind shader/texture/sampler/constant buffer

    DC_->VSSetShader(vertexShader_.Get(), nullptr, 0);
    DC_->PSSetShader(weightShader_.Get(), nullptr, 0);
    DC_->PSSetSamplers(0, 1, pointSampler_.GetAddressOf());
    DC_->PSSetSamplers(1, 1, linearSampler_.GetAddressOf());
    DC_->PSSetShaderResources(0, 1, &edge);
    DC_->PSSetShaderResources(1, 1, innerAreaTextureSRV_.GetAddressOf());
    DC_->PSSetConstantBuffers(0, 1, pixelSizeConstantBuffer_.GetAddressOf());

    // bind vertex buffer/input layout

    const UINT stride = sizeof(Vertex), offset = 0;
    DC_->IASetVertexBuffers(
        0, 1, vertexBuffer_.GetAddressOf(), &stride, &offset);

    DC_->IASetInputLayout(inputLayout_.Get());
    DC_->IASetPrimitiveTopology(
        D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    // emit drawcall

    DC_->Draw(3, 0);

    // clear vertex buffer/input layout

    ID3D11Buffer *NULL_VERTEX_BUFFER = nullptr;
    DC_->IASetVertexBuffers(
        0, 1, &NULL_VERTEX_BUFFER, &stride, &offset);

    DC_->IASetInputLayout(nullptr);

    // clear shader/texture/sampler

    ID3D11Buffer *NULL_BUFFER = nullptr;
    DC_->PSSetConstantBuffers(0, 1, &NULL_BUFFER);

    ID3D11SamplerState *NULL_SAMPLER = nullptr;
    DC_->PSSetSamplers(0, 1, &NULL_SAMPLER);
    DC_->PSSetSamplers(1, 1, &NULL_SAMPLER);

    ID3D11ShaderResourceView *NULL_IMG = nullptr;
    DC_->PSSetShaderResources(0, 1, &NULL_IMG);
    DC_->PSSetShaderResources(1, 1, &NULL_IMG);

    DC_->PSSetShader(nullptr, nullptr, 0);
    DC_->VSSetShader(nullptr, nullptr, 0);
}

void MLAA::PerformBlending(
    ID3D11ShaderResourceView *img, ID3D11ShaderResourceView *weight)
{
    // bind shader/texture/sampler/constant buffer

    DC_->VSSetShader(vertexShader_.Get(), nullptr, 0);
    DC_->PSSetShader(blendingShader_.Get(), nullptr, 0);
    DC_->PSSetSamplers(0, 1, pointSampler_.GetAddressOf());
    DC_->PSSetSamplers(1, 1, linearSampler_.GetAddressOf());
    DC_->PSSetShaderResources(0, 1, &img);
    DC_->PSSetShaderResources(1, 1, &weight);
    DC_->PSSetConstantBuffers(0, 1, pixelSizeConstantBuffer_.GetAddressOf());

    // bind vertex buffer/input layout

    const UINT stride = sizeof(Vertex), offset = 0;
    DC_->IASetVertexBuffers(
        0, 1, vertexBuffer_.GetAddressOf(), &stride, &offset);

    DC_->IASetInputLayout(inputLayout_.Get());
    DC_->IASetPrimitiveTopology(
        D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    // emit drawcall

    DC_->Draw(3, 0);

    // clear vertex buffer/input layout

    ID3D11Buffer *NULL_VERTEX_BUFFER = nullptr;
    DC_->IASetVertexBuffers(
        0, 1, &NULL_VERTEX_BUFFER, &stride, &offset);

    DC_->IASetInputLayout(nullptr);

    // clear shader/texture/sampler

    ID3D11Buffer *NULL_BUFFER = nullptr;
    DC_->PSSetConstantBuffers(0, 1, &NULL_BUFFER);

    ID3D11SamplerState *NULL_SAMPLER = nullptr;
    DC_->PSSetSamplers(0, 1, &NULL_SAMPLER);
    DC_->PSSetSamplers(1, 1, &NULL_SAMPLER);

    ID3D11ShaderResourceView *NULL_IMG = nullptr;
    DC_->PSSetShaderResources(0, 1, &NULL_IMG);
    DC_->PSSetShaderResources(1, 1, &NULL_IMG);

    DC_->PSSetShader(nullptr, nullptr, 0);
    DC_->VSSetShader(nullptr, nullptr, 0);
}

ID3D11Texture2D *MLAA::GetInnerAreaTexture() noexcept
{
    return innerAreaTexture_.Get();
}

ID3D11ShaderResourceView *MLAA::GetInnerAreaTextureSRV() noexcept
{
    return innerAreaTextureSRV_.Get();
}

ComPtr<ID3D10Blob> MLAA::InitVertexShader(ID3D11Device *device)
{
    const char *COMMON_VERTEX_SHADER_SOURCE =
#include "./mlaa_vs.inl"
        ;

    ComPtr<ID3D10Blob> vertexShaderByteCode = detail::CompileToByteCode(
        COMMON_VERTEX_SHADER_SOURCE, "vs_5_0", nullptr);

    vertexShader_ = detail::CreateVertexShader(
        device,
        vertexShaderByteCode->GetBufferPointer(),
        vertexShaderByteCode->GetBufferSize());

    return vertexShaderByteCode;
}

void MLAA::InitEdgeShader(
    ID3D11Device *device,
    float         edgeThreshold,
    Mode          mode)
{
    const std::string edgeThresholdStr =
        std::to_string(edgeThreshold);

    const D3D_SHADER_MACRO EDGE_MACROS[] = {
        { "EDGE_THRESHOLD"  , edgeThresholdStr.c_str() },
        { nullptr           , nullptr                  }
    };

    const char *EDGE_DEPTH_SHADER_SOURCE =
#include "mlaa_depth_edge.inl"
        ;
    const char *EDGE_LUM_SHADER_SOURCE = 
#include "mlaa_lum_edge.inl"
        ;
    const char *SHADER_SOURCE = mode == Mode::Depth ?
                                EDGE_DEPTH_SHADER_SOURCE :
                                EDGE_LUM_SHADER_SOURCE;

    ComPtr<ID3D10Blob> edgeShaderByteCode = detail::CompileToByteCode(
        SHADER_SOURCE, "ps_5_0", EDGE_MACROS);

    edgeShader_ = detail::CreatePixelShader(
        device,
        edgeShaderByteCode->GetBufferPointer(),
        edgeShaderByteCode->GetBufferSize());
}

void MLAA::InitWeightShader(ID3D11Device *device, int maxEdgeDetectionLen)
{
    const std::string maxEdgeDetectionLenStr =
        std::to_string(maxEdgeDetectionLen);

    const D3D_SHADER_MACRO WEIGHT_MACROS[] = {
        { "EDGE_DETECTION_MAX_LEN" , maxEdgeDetectionLenStr.c_str()  },
        { nullptr                  , nullptr                        }
    };

    const char *WEIGHT_SHADER_SOURCE =
#include "./mlaa_weight.inl"
        ;

    ComPtr<ID3D10Blob> weightShaderByteCode = detail::CompileToByteCode(
        WEIGHT_SHADER_SOURCE, "ps_5_0", WEIGHT_MACROS);

    weightShader_ = detail::CreatePixelShader(
        device,
        weightShaderByteCode->GetBufferPointer(),
        weightShaderByteCode->GetBufferSize());
}

void MLAA::InitBlendingShader(ID3D11Device *device)
{
    const char *BLENDING_SHADER_SOURCE =
#include "mlaa_blend.inl"
        ;

    ComPtr<ID3D10Blob> blendingShaderByteCode = detail::CompileToByteCode(
        BLENDING_SHADER_SOURCE, "ps_5_0", nullptr);

    blendingShader_ = detail::CreatePixelShader(
        device,
        blendingShaderByteCode->GetBufferPointer(),
        blendingShaderByteCode->GetBufferSize());
}

void MLAA::InitVertexBuffer(ID3D11Device *device)
{
    Vertex vertexData[] = {
        { -1, -1, +0, +1 },
        { -1, +3, +0, -1 },
        { +3, -1, +2, +1 }
    };

    D3D11_BUFFER_DESC bufferDesc;
    bufferDesc.Usage               = D3D11_USAGE_IMMUTABLE;
    bufferDesc.BindFlags           = D3D11_BIND_VERTEX_BUFFER;
    bufferDesc.ByteWidth           = static_cast<UINT>(sizeof(Vertex)) * 3;
    bufferDesc.CPUAccessFlags      = 0;
    bufferDesc.MiscFlags           = 0;
    bufferDesc.StructureByteStride = 0;

    D3D11_SUBRESOURCE_DATA subrscData;
    subrscData.pSysMem          = vertexData;
    subrscData.SysMemPitch      = 0;
    subrscData.SysMemSlicePitch = 0;

    const HRESULT hr = device->CreateBuffer(
        &bufferDesc, &subrscData, vertexBuffer_.GetAddressOf());
    if(FAILED(hr))
        throw std::runtime_error("MLAA: failed to initialize vertex buffer");
}

void MLAA::InitInputLayout(
    ID3D11Device *device, ComPtr<ID3D10Blob> shaderByteCode)
{
    D3D11_INPUT_ELEMENT_DESC inputDesc[] = {
        {
            "POSITION", 0, DXGI_FORMAT_R32G32_FLOAT,
            0, offsetof(Vertex, x),
            D3D11_INPUT_PER_VERTEX_DATA, 0
        },
        {
            "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,
            0, offsetof(Vertex, u),
            D3D11_INPUT_PER_VERTEX_DATA, 0
        }
    };

    const HRESULT hr = device->CreateInputLayout(
        inputDesc, 2,
        shaderByteCode->GetBufferPointer(),
        shaderByteCode->GetBufferSize(),
        inputLayout_.GetAddressOf());
    if(FAILED(hr))
        throw std::runtime_error("MLAA: failed to create input layout");
}

void MLAA::InitSamplers(ID3D11Device *device)
{
    const float BORDER_COLOR[4] = { 0, 0, 0, 0 };

    D3D11_SAMPLER_DESC samplerDesc;
    samplerDesc.Filter         = D3D11_FILTER_MIN_MAG_MIP_POINT;
    samplerDesc.AddressU       = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDesc.AddressV       = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDesc.AddressW       = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDesc.MipLODBias     = 0;
    samplerDesc.MaxAnisotropy  = 0;
    samplerDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
    samplerDesc.MinLOD         = -FLT_MAX;
    samplerDesc.MaxLOD         = FLT_MAX;
    memcpy(samplerDesc.BorderColor, BORDER_COLOR, sizeof(BORDER_COLOR));

    HRESULT hr = device->CreateSamplerState(
        &samplerDesc, pointSampler_.GetAddressOf());
    if(FAILED(hr))
        throw std::runtime_error("MLAA: failed to initialize point sampler");

    samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    hr = device->CreateSamplerState(
        &samplerDesc, linearSampler_.GetAddressOf());
    if(FAILED(hr))
        throw std::runtime_error("MLAA: failed to initialize point sampler");
}

void MLAA::InitInnerAreaTexture(
    ID3D11Device *device, int maxEdgeDetectionLen)
{
    // generate texture data

    int width, height;
    const auto texData = GenerateInnerAreaTexture(
        maxEdgeDetectionLen, &width, &height);

    // create d3d11 texture

    D3D11_TEXTURE2D_DESC texDesc;
    texDesc.Width              = width;
    texDesc.Height             = height;
    texDesc.MipLevels          = 1;
    texDesc.ArraySize          = 1;
    texDesc.Format             = DXGI_FORMAT_R32G32B32A32_FLOAT;
    texDesc.SampleDesc.Count   = 1;
    texDesc.SampleDesc.Quality = 0;
    texDesc.Usage              = D3D11_USAGE_IMMUTABLE;
    texDesc.BindFlags          = D3D11_BIND_SHADER_RESOURCE;
    texDesc.CPUAccessFlags     = 0;
    texDesc.MiscFlags          = 0;

    D3D11_SUBRESOURCE_DATA initData;
    initData.pSysMem          = texData.data();
    initData.SysMemPitch      = width * sizeof(float) * 4;
    initData.SysMemSlicePitch = 0;

    HRESULT hr = device->CreateTexture2D(
        &texDesc, &initData, innerAreaTexture_.GetAddressOf());
    if(FAILED(hr))
        throw std::runtime_error("failed to create inner area texture");

    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
    srvDesc.Format                    = DXGI_FORMAT_R32G32B32A32_FLOAT;
    srvDesc.ViewDimension             = D3D11_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MostDetailedMip = 0;
    srvDesc.Texture2D.MipLevels       = 1;

    hr = device->CreateShaderResourceView(
        innerAreaTexture_.Get(), &srvDesc, innerAreaTextureSRV_.GetAddressOf());
    if(FAILED(hr))
    {
        throw std::runtime_error(
            "failed to create shader resource view for inner area texture");
    }
}

void MLAA::InitPixelSizeConstant(
    ID3D11Device *device, int framebufferWidth, int framebufferHeight)
{
    D3D11_BUFFER_DESC bufferDesc;
    bufferDesc.ByteWidth           = sizeof(PixelSizeConstant);
    bufferDesc.Usage               = D3D11_USAGE_IMMUTABLE;
    bufferDesc.BindFlags           = D3D11_BIND_CONSTANT_BUFFER;
    bufferDesc.CPUAccessFlags      = 0;
    bufferDesc.MiscFlags           = 0;
    bufferDesc.StructureByteStride = 0;

    PixelSizeConstant data = {
        1.0f / framebufferWidth,
        1.0f / framebufferHeight,
        0, 0
    };

    D3D11_SUBRESOURCE_DATA initData;
    initData.pSysMem          = &data;
    initData.SysMemPitch      = 0;
    initData.SysMemSlicePitch = 0;

    const HRESULT hr = device->CreateBuffer(
        &bufferDesc, &initData, pixelSizeConstantBuffer_.GetAddressOf());
    if(FAILED(hr))
    {
        throw std::runtime_error(
            "failed to create constant buffer for weight shader");
    }
}

// this method will be called for only once during construction
// so it's not well optimized
std::vector<float> MLAA::GenerateInnerAreaTexture(
    int maxEdgeDetectionLen,
    int *width, int *height) const noexcept
{
    // the inner area texture contains 25 grids
    // grid [i, j] <=> [cross1, cross2] = [i / 4, j / 4]
    // each grid contains 2*maxEdgeDetectionLen+1 texels
    // texel [m, n] <=> [dist1, dist2] = [m, n]

    const int gridSidelen = 2 * maxEdgeDetectionLen + 1;
    const int sidelen     = 5 * gridSidelen;

    *width  = sidelen;
    *height = sidelen;

    // texel data

    std::vector<float> ret(sidelen * sidelen * 4, 0.0f);

    auto texel = [&](int dist1, int cross1, int dist2, int cross2)
        -> float*
    {
        const int base_u = gridSidelen * cross1;
        const int base_v = gridSidelen * cross2;
        
        const int u = base_u + dist1;
        const int v = base_v + dist2;

        const int idx = v * sidelen + u;
        return &ret[4 * idx];
    };

    // fill texels

    for(int cross1 = 0; cross1 < 5; ++cross1)
    {
        for(int cross2 = 0; cross2 < 5; ++cross2)
        {
            for(int dist1 = 0; dist1 < gridSidelen; ++dist1)
            {
                for(int dist2 = 0; dist2 < gridSidelen; ++dist2)
                {
                    const auto rg = ComputePixelInnerArea(
                        dist1, cross1, dist2, cross2);

                    auto pTexel = texel(dist1, cross1, dist2, cross2);
                    pTexel[0] = rg.first;
                    pTexel[1] = rg.second;
                    pTexel[3] = 1;
                }
            }
        }
    }

    return ret;
}

AGZ_SMAA_END
