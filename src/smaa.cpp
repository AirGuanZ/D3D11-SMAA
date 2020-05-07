#include <stdexcept>
#include <string>

#include <d3dcompiler.h>

#include <agz/smaa/smaa.h>

namespace agz { namespace smaa {

namespace detail
{

// ========= compile shader =========

inline ComPtr<ID3D10Blob> CompileToByteCode(
    const char *source, const char *target,
    const D3D_SHADER_MACRO *macros = nullptr)
{
#ifdef _DEBUG
    constexpr UINT COMPILER_FLAGS = D3DCOMPILE_DEBUG |
                                    D3DCOMPILE_SKIP_OPTIMIZATION;
#else
    constexpr UINT COMPILER_FLAGS = 0;
#endif

    const size_t sourceLen = std::strlen(source);

    ComPtr<ID3D10Blob> ret, err;
    const HRESULT hr = D3DCompile(
        source, sourceLen,
        nullptr, macros, nullptr,
        "main", target,
        COMPILER_FLAGS, 0,
        ret.GetAddressOf(), err.GetAddressOf());

    if(FAILED(hr))
    {
        auto rawErrMsg = reinterpret_cast<const char *>(
            err->GetBufferPointer());
        throw std::runtime_error(rawErrMsg);
    }

    return ret;
}

// ========= create shader =========

inline ComPtr<ID3D11VertexShader> CreateVertexShader(
    ID3D11Device *device, void *byteCode, size_t len)
{
    ComPtr<ID3D11VertexShader> shader;
    const HRESULT hr = device->CreateVertexShader(
        byteCode, len, nullptr, shader.GetAddressOf());
    return FAILED(hr) ? nullptr : shader;
}

inline ComPtr<ID3D11PixelShader> CreatePixelShader(
    ID3D11Device *device, void *byteCode, size_t len)
{
    ComPtr<ID3D11PixelShader> shader;
    const HRESULT hr = device->CreatePixelShader(
        byteCode, len, nullptr, shader.GetAddressOf());
    return FAILED(hr) ? nullptr : shader;
}

// ========= shader sources =========

static const char *COMMON_VERTEX_SHADER_SOURCE = R"___(
struct VSInput
{
    float2 position : POSITION;
    float2 texCoord : TEXCOORD;
};

struct VSOutput
{
    float4 position : SV_POSITION;
    float2 texCoord : TEXCOORD;
};

VSOutput main(VSInput input)
{
    VSOutput output = (VSOutput)0;
    output.position = float4(input.position, 0.5, 1);
    output.texCoord = input.texCoord;
    return output;
}
)___";

static const char *EDGE_DETECTION_SHADER_SOURCE = R"___(
// #define EDGE_THRESHOLD  XXX
// #define CONTRAST_FACTOR XXX

struct PSInput
{
    float4 position : SV_POSITION;
    float2 texCoord : TEXCOORD;
};

Texture2D<float3> ImageTexture : register(t0);
SamplerState      PointSampler : register(s0);

float4 main(PSInput input) : SV_TARGET
{
    const float3 LUM_FACTOR = float3(0.2126, 0.7152, 0.0722);

    // sample image pixels

    float3 c = ImageTexture.SampleLevel(
        PointSampler, input.texCoord, 0);
    float3 c_left = ImageTexture.SampleLevel(
        PointSampler, input.texCoord, 0, int2(-1, 0));
    float3 c_top = ImageTexture.SampleLevel(
        PointSampler, input.texCoord, 0, int2(0, -1));

    // eval left/top edge

    float2 delta = float2(
        dot(LUM_FACTOR, abs(c - c_left)),
        dot(LUM_FACTOR, abs(c - c_top)));
    
    float2 is_edge = step(EDGE_THRESHOLD, delta);

    if(dot(is_edge, 1) == 0)
        discard;

    // eval local constract

    float3 c_left2 = ImageTexture.SampleLevel(
        PointSampler, input.texCoord, 0, int2(-2, 0));
    float3 c_top2 = ImageTexture.SampleLevel(
        PointSampler, input.texCoord, 0, int2(0, -2));
    float3 c_right = ImageTexture.SampleLevel(
        PointSampler, input.texCoord, 0, int2(1, 0));
    float3 c_bottom = ImageTexture.SampleLevel(
        PointSampler, input.texCoord, 0, int2(0, 1));

    float4 delta_local = float4(
        dot(LUM_FACTOR, abs(c_left - c_left2)),
        dot(LUM_FACTOR, abs(c_top  - c_top2)),
        dot(LUM_FACTOR, abs(c      - c_right)),
        dot(LUM_FACTOR, abs(c      - c_bottom)));

    // local constract adaptation

    float max_delta_local = max(max(delta_local.r, delta_local.g),
                                max(delta_local.b, delta_local.a));

    is_edge *= step(CONTRAST_FACTOR * max_delta_local, delta);

    return float4(is_edge, 0, 0);
}
)___";

} // namespace detail

Common::Common(
    ID3D11Device *device, ID3D11DeviceContext *deviceContext)
{
    D  = device;
    DC = deviceContext;

    // vertex shader

    ComPtr<ID3D10Blob> vertexShaderByteCode = detail::CompileToByteCode(
        detail::COMMON_VERTEX_SHADER_SOURCE, "vs_5_0", nullptr);

    vertexShader = detail::CreateVertexShader(
        device,
        vertexShaderByteCode->GetBufferPointer(),
        vertexShaderByteCode->GetBufferSize());

    // vertex buffer

    struct Vertex { float x, y, u, v; };

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

    HRESULT hr = device->CreateBuffer(
        &bufferDesc, &subrscData, vertexBuffer.GetAddressOf());
    if(FAILED(hr))
        throw std::runtime_error("MLAA: failed to initialize vertex buffer");

    // input layout

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

    hr = device->CreateInputLayout(
        inputDesc, 2,
        vertexShaderByteCode->GetBufferPointer(),
        vertexShaderByteCode->GetBufferSize(),
        inputLayout.GetAddressOf());
    if(FAILED(hr))
        throw std::runtime_error("MLAA: failed to create input layout");

    // samplers
    
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

    hr = device->CreateSamplerState(
        &samplerDesc, pointSampler.GetAddressOf());
    if(FAILED(hr))
        throw std::runtime_error("MLAA: failed to initialize point sampler");

    samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    hr = device->CreateSamplerState(
        &samplerDesc, linearSampler.GetAddressOf());
    if(FAILED(hr))
        throw std::runtime_error("MLAA: failed to initialize point sampler");
}

void Common::bindVertex() const
{
    const UINT stride = sizeof(float) * 4, offset = 0;
    DC->IASetVertexBuffers(
        0, 1, vertexBuffer.GetAddressOf(), &stride, &offset);

    DC->IASetInputLayout(inputLayout.Get());
    DC->IASetPrimitiveTopology(
        D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
}

void Common::unbindVertex() const
{
    const UINT stride = 0, offset = 0;

    ID3D11Buffer *NULL_VERTEX_BUFFER = nullptr;
    DC->IASetVertexBuffers(
        0, 1, &NULL_VERTEX_BUFFER, &stride, &offset);

    DC->IASetInputLayout(nullptr);
}

EdgeDetection::EdgeDetection(
    ID3D11Device *device,
    float         edgeThreshold,
    float         localContrastFactor)
{
    const std::string edgeThresholdStr =
        std::to_string(edgeThreshold);
    const std::string localContrastFactorStr =
        std::to_string(localContrastFactor);

    const D3D_SHADER_MACRO MACROS[] = {
        { "EDGE_THRESHOLD" , edgeThresholdStr.c_str()       },
        { "CONTRAST_FACTOR", localContrastFactorStr.c_str() },
        { nullptr          , nullptr                        }
    };

    ComPtr<ID3D10Blob> shaderByteCode = detail::CompileToByteCode(
        detail::EDGE_DETECTION_SHADER_SOURCE, "ps_5_0", MACROS);

    pixelShader = detail::CreatePixelShader(
        device,
        shaderByteCode->GetBufferPointer(),
        shaderByteCode->GetBufferSize());
}

void EdgeDetection::detectEdge(
    const Common             &common,
    ID3D11ShaderResourceView *img) const
{
    // bind shader/texture/sampler

    common.DC->VSSetShader(common.vertexShader.Get(), nullptr, 0);
    common.DC->PSSetShader(pixelShader.Get(), nullptr, 0);
    common.DC->PSSetSamplers(0, 1, common.pointSampler.GetAddressOf());
    common.DC->PSSetShaderResources(0, 1, &img);

    // bind vertex buffer/input layout

    common.bindVertex();

    // emit drawcall

    common.DC->Draw(3, 0);

    // clear vertex buffer/input layout

    common.unbindVertex();

    // clear shader/texture/sampler

    ID3D11SamplerState *NULL_SAMPLER = nullptr;
    common.DC->PSSetSamplers(0, 1, &NULL_SAMPLER);

    ID3D11ShaderResourceView *NULL_IMG = nullptr;
    common.DC->PSSetShaderResources(0, 1, &NULL_IMG);

    common.DC->PSSetShader(nullptr, nullptr, 0);
    common.DC->VSSetShader(nullptr, nullptr, 0);
}

SMAA::SMAA(
    ID3D11Device        *device,
    ID3D11DeviceContext *deviceContext,
    float                edgeThreshold,
    float                localContractFactor)
    : common_(device, deviceContext),
      edgeDetection_(device, edgeThreshold, localContractFactor)
{
    
}

void SMAA::detectEdge(ID3D11ShaderResourceView *img) const
{
    edgeDetection_.detectEdge(common_, img);
}

} } // namespace agz::smaa
