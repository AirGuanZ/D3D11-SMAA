#pragma once

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <wrl/client.h>

#include <d3dcompiler.h>
#include <d3d11.h>

namespace agz { namespace mlaa {

using Microsoft::WRL::ComPtr;

class Common
{
public:

    Common(
        ID3D11Device        *device,
        ID3D11DeviceContext *deviceContext);

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

    enum class Mode
    {
        Depth,
        Lum
    };

    explicit EdgeDetection(
        ID3D11Device *device,
        Mode          mode,
        float         threshold);

    void detectEdge(
        const Common         &common,
        ID3D11ShaderResourceView *source) const;

    ComPtr<ID3D11PixelShader> pixelShader;
};

class BlendingWeight
{
public:

    BlendingWeight(
        ID3D11Device *device,
        int           maxSearchDistanceLen,
        int           width,
        int           height);

    void computeBlendingWeight(
        const Common         &common,
        ID3D11ShaderResourceView *edgeTexture) const;

    ComPtr<ID3D11PixelShader> pixelShader;

    ComPtr<ID3D11Texture2D>          innerAreaTexture;
    ComPtr<ID3D11ShaderResourceView> innerAreaTextureSRV;
};

class Blending
{
public:

    explicit Blending(
        ID3D11Device *device,
        int           width,
        int           height);

    void blend(
        const Common             &common,
        ID3D11ShaderResourceView *weightTexture,
        ID3D11ShaderResourceView *img) const;

    ComPtr<ID3D11PixelShader> pixelShader;
};

// ========= MLAA =========

class MLAA
{
public:

    using EdgeDetectionMode = EdgeDetection::Mode;

    MLAA(
        ID3D11Device        *device,
        ID3D11DeviceContext *deviceContext,
        int                  width,
        int                  height,
        EdgeDetectionMode    mode                   = EdgeDetectionMode::Lum,
        float                edgeDetectionThreshold = 0.1f,
        int                  maxSearchDistanceLen   = 8);

    void detectEdge(
        ID3D11ShaderResourceView *source) const;

    void computeBlendingWeight(
        ID3D11ShaderResourceView *edgeTexture) const;

    void blend(
        ID3D11ShaderResourceView *weightTexture,
        ID3D11ShaderResourceView *img) const;

    Common         &_common        () noexcept { return common_;         }
    EdgeDetection  &_edgeDetection () noexcept { return edgeDetection_;  }
    BlendingWeight &_blendingWeight() noexcept { return blendingWeight_; }
    Blending       &_blending      () noexcept { return blending_;       }

private:

    Common         common_;
    EdgeDetection  edgeDetection_;
    BlendingWeight blendingWeight_;
    Blending       blending_;
};


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

static const char *EDGE_DEPTH_SHADER_SOURCE = R"___(
// #define EDGE_THRESHOLD XXX

struct PSInput
{
    float4 position : SV_POSITION;
    float2 texCoord : TEXCOORD;
};

Texture2D<float> DepthTexture : register(t0);
SamplerState     PointSampler : register(s0);

float4 main(PSInput input) : SV_TARGET
{
    // sample depth texture

    float d = DepthTexture.SampleLevel(
        PointSampler, input.texCoord, 0);
    float d_left = DepthTexture.SampleLevel(
        PointSampler, input.texCoord, 0, int2(-1, 0));
    float d_top = DepthTexture.SampleLevel(
        PointSampler, input.texCoord, 0, int2(0, -1));

    // compute delta depth

    float2 delta_d = abs(d.xx - float2(d_left, d_top));

    float2 is_edge = step(EDGE_THRESHOLD, delta_d);
    if(is_edge.x == 0 && is_edge.y == 0)
        discard;
    return float4(is_edge, 0, 0);
}
)___";

static const char *EDGE_LUM_SHADER_SOURCE = R"___(
// #define EDGE_THRESHOLD XXX

struct PSInput
{
    float4 position : SV_POSITION;
    float2 texCoord : TEXCOORD;
};

Texture2D<float3> ImageTexture : register(t0);
SamplerState      PointSampler : register(s0);

float to_lum(float3 color)
{
    return 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b;
}

float diff_lum(float3 a, float3 b)
{
    return to_lum(abs(a - b));
}

float4 main(PSInput input) : SV_TARGET
{
    // sample texture

    float3 d = ImageTexture.SampleLevel(
        PointSampler, input.texCoord, 0);
    float3 d_left = ImageTexture.SampleLevel(
        PointSampler, input.texCoord, 0, int2(-1, 0));
    float3 d_top = ImageTexture.SampleLevel(
        PointSampler, input.texCoord, 0, int2(0, -1));

    // compute delta lum

    float2 delta_d = float2(diff_lum(d_left, d), diff_lum(d_top, d));
    
    float2 is_edge = step(EDGE_THRESHOLD, delta_d);
    if(is_edge.x == 0 && is_edge.y == 0)
        discard;
    return float4(is_edge, 0, 0);
}
)___";

static const char *WEIGHT_SHADER_SOURCE = R"___(
// #define EDGE_DETECTION_MAX_LEN
// #define PIXEL_SIZE_IN_TEXCOORD

struct PSInput
{
    float4 position : SV_POSITION;
    float2 texCoord : TEXCOORD;
};

Texture2D<float4> EdgeTexture      : register(t0);
Texture2D<float4> InnerAreaTexture : register(t1);

SamplerState PointSampler  : register(s0);
SamplerState LinearSampler : register(s1);

float find_left_end(float2 center)
{
    center -= float2(1.5, 0) * PIXEL_SIZE_IN_TEXCOORD;
    float e = 0;

    for(int p2 = 0; p2 < EDGE_DETECTION_MAX_LEN; ++p2)
    {
        e = EdgeTexture.SampleLevel(LinearSampler, center, 0).g;
        
        [flatten]
        if(e < 0.9)
            break;

        center -= float2(2, 0) * PIXEL_SIZE_IN_TEXCOORD;
    }

    return max(-2 * p2 - 2 * e, -2 * EDGE_DETECTION_MAX_LEN);
}

float find_right_end(float2 center)
{
    center += float2(1.5, 0) * PIXEL_SIZE_IN_TEXCOORD;
    float e = 0;

    for(int p2 = 0; p2 < EDGE_DETECTION_MAX_LEN; ++p2)
    {
        e = EdgeTexture.SampleLevel(LinearSampler, center, 0).g;

        [flatten]
        if(e < 0.9)
            break;

        center += float2(2, 0) * PIXEL_SIZE_IN_TEXCOORD;
    }

    return min(2 * p2 + 2 * e, 2 * EDGE_DETECTION_MAX_LEN);
}

float find_top_end(float2 center)
{
    center -= float2(0, 1.5) * PIXEL_SIZE_IN_TEXCOORD;
    float e = 0;

    for(int p2 = 0; p2 < EDGE_DETECTION_MAX_LEN; ++p2)
    {
        e = EdgeTexture.SampleLevel(LinearSampler, center, 0).r;

        [flatten]
        if(e < 0.9)
            break;

        center -= float2(0, 2) * PIXEL_SIZE_IN_TEXCOORD;
    }

    return max(-2 * p2 - 2 * e, -2 * EDGE_DETECTION_MAX_LEN);
}

float find_bottom_end(float2 center)
{
    center += float2(0, 1.5) * PIXEL_SIZE_IN_TEXCOORD;
    float e = 0;

    for(int p2 = 0; p2 < EDGE_DETECTION_MAX_LEN; ++p2)
    {
        e = EdgeTexture.SampleLevel(LinearSampler, center, 0).r;

        [flatten]
        if(e < 0.9)
            break;

        center += float2(0, 2) * PIXEL_SIZE_IN_TEXCOORD;
    }

    return min(2 * p2 + 2 * e, 2 * EDGE_DETECTION_MAX_LEN);
}

float2 inner_area(float dist1, float cross1, float dist2, float cross2)
{
    // dist1: [0, 2 * EDGE_DETECTION_MAX_LEN]
    // dist2: [0, 2 * EDGE_DETECTION_MAX_LEN]
    
    // cross1: 0, 0.25, 0.75, 1
    // cross2: 0, 0.25, 0.75, 1

    float base_u = (2 * EDGE_DETECTION_MAX_LEN + 1) * round(4 * cross1);
    float base_v = (2 * EDGE_DETECTION_MAX_LEN + 1) * round(4 * cross2);

    float pixel_u = base_u + dist1;
    float pixel_v = base_v + dist2;

    float u = (pixel_u + 0.5) / ((2 * EDGE_DETECTION_MAX_LEN + 1) * 5);
    float v = (pixel_v + 0.5) / ((2 * EDGE_DETECTION_MAX_LEN + 1) * 5);

    return InnerAreaTexture.SampleLevel(PointSampler, float2(u, v), 0).rg;
}

float4 main(PSInput input) : SV_TARGET
{
    float4 output = (float4)0;

    float2 e = EdgeTexture.SampleLevel(PointSampler, input.texCoord, 0).rg;

    // edge at left side
    if(e.r)
    {
        float top_end    = find_top_end   (input.texCoord);
        float bottom_end = find_bottom_end(input.texCoord);

        float2 coord_top = float2(
            input.texCoord.x - 0.25    * PIXEL_SIZE_IN_TEXCOORD.x,
            input.texCoord.y + top_end * PIXEL_SIZE_IN_TEXCOORD.y);
        float2 coord_bottom = float2(
            input.texCoord.x - 0.25             * PIXEL_SIZE_IN_TEXCOORD.x,
            input.texCoord.y + (bottom_end + 1) * PIXEL_SIZE_IN_TEXCOORD.y);

        float cross_top = EdgeTexture.SampleLevel(
            LinearSampler, coord_top, 0).g;
        float cross_bottom = EdgeTexture.SampleLevel(
            LinearSampler, coord_bottom, 0).g;

        output.ba = inner_area(
            -top_end, cross_top, bottom_end, cross_bottom);
    }

    // edge at top side
    if(e.g)
    {
        float left_end  = find_left_end(input.texCoord);
        float right_end = find_right_end(input.texCoord);

        float2 coord_left = float2(
            input.texCoord.x + left_end * PIXEL_SIZE_IN_TEXCOORD.x,
            input.texCoord.y - 0.25     * PIXEL_SIZE_IN_TEXCOORD.y);
        float2 coord_right = float2(
            input.texCoord.x + (right_end + 1) * PIXEL_SIZE_IN_TEXCOORD.x,
            input.texCoord.y - 0.25            * PIXEL_SIZE_IN_TEXCOORD.y);

        float cross_left = EdgeTexture.SampleLevel(
            LinearSampler, coord_left, 0).r;
        float cross_right = EdgeTexture.SampleLevel(
            LinearSampler, coord_right, 0).r;

        output.rg = inner_area(
            -left_end, cross_left, right_end, cross_right);
    }

    return output;
}
)___";

static const char *BLENDING_SHADER_SOURCE = R"___(
//#define PIXEL_SIZE_IN_TEXCOORD XXX

struct PSInput
{
    float4 position : SV_POSITION;
    float2 texCoord : TEXCOORD;
};

Texture2D<float4> ImageTexture  : register(t0);
Texture2D<float4> WeightTexture : register(t1);

SamplerState      PointSampler  : register(s0);
SamplerState      LinearSampler : register(s1);

float4 main(PSInput input) : SV_TARGET
{
    float2 w_up_left = WeightTexture.SampleLevel(
        PointSampler, input.texCoord, 0).rb;
    float w_right = WeightTexture.SampleLevel(
        PointSampler, input.texCoord, 0, int2(1, 0)).a;
    float w_down = WeightTexture.SampleLevel(
        PointSampler, input.texCoord, 0, int2(0, 1)).g;

    float w_sum = dot(float4(w_up_left, w_right, w_down), 1);
    if(w_sum == 0)
        return ImageTexture.SampleLevel(PointSampler, input.texCoord, 0);

    float4 up = ImageTexture.SampleLevel(
        LinearSampler, input.texCoord + PIXEL_SIZE_IN_TEXCOORD * float2(0, -w_up_left.r), 0);
    float4 right = ImageTexture.SampleLevel(
        LinearSampler, input.texCoord + PIXEL_SIZE_IN_TEXCOORD * float2(w_right, 0), 0);
    float4 down = ImageTexture.SampleLevel(
        LinearSampler, input.texCoord + PIXEL_SIZE_IN_TEXCOORD * float2(0, w_down), 0);
    float4 left = ImageTexture.SampleLevel(
        LinearSampler, input.texCoord + PIXEL_SIZE_IN_TEXCOORD * float2(-w_up_left.g, 0), 0);

    return (up    * w_up_left.r +
            right * w_right     +
            down  * w_down      +
            left  * w_up_left.g) / w_sum;
}
)___";

// ========= inner area texture =========

struct Vec2
{
    float x = 0, y = 0;

    Vec2 operator+(const Vec2 &rhs) const noexcept
        { return { x + rhs.x, y + rhs.y }; }
    
    Vec2 operator-(const Vec2 &rhs) const noexcept
        { return { x - rhs.x, y - rhs.y }; }
};

inline Vec2 operator*(float lhs, const Vec2 &rhs) noexcept
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
inline std::pair<float, float> ComputePixelInnerArea(
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
    return { areaLeft, areaRight };
}

inline std::pair<float, float> ComputePixelInnerArea(
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

// this method will be called for only once during construction
// so it's not well optimized
inline std::vector<float> GenerateInnerAreaTexture(
    int maxEdgeDetectionLen,
    int *width, int *height)
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

} // namespace detail

inline Common::Common(
    ID3D11Device        *device,
    ID3D11DeviceContext *deviceContext)
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

inline EdgeDetection::EdgeDetection(
    ID3D11Device *device, Mode mode, float threshold)
{
    const std::string edgeThresholdStr =
        std::to_string(threshold);

    const D3D_SHADER_MACRO EDGE_MACROS[] = {
        { "EDGE_THRESHOLD"  , edgeThresholdStr.c_str() },
        { nullptr           , nullptr                  }
    };

    const char *shaderSource = mode == Mode::Depth ?
                               detail::EDGE_DEPTH_SHADER_SOURCE :
                               detail::EDGE_LUM_SHADER_SOURCE;

    ComPtr<ID3D10Blob> edgeShaderByteCode = detail::CompileToByteCode(
        shaderSource, "ps_5_0", EDGE_MACROS);

    pixelShader = detail::CreatePixelShader(
        device,
        edgeShaderByteCode->GetBufferPointer(),
        edgeShaderByteCode->GetBufferSize());
}

inline void EdgeDetection::detectEdge(
    const Common &common, ID3D11ShaderResourceView *source) const
{
    // bind shader/texture/sampler

    common.DC->VSSetShader(common.vertexShader.Get(), nullptr, 0);
    common.DC->PSSetShader(pixelShader.Get(), nullptr, 0);
    common.DC->PSSetSamplers(0, 1, common.pointSampler.GetAddressOf());
    common.DC->PSSetShaderResources(0, 1, &source);

    // bind vertex buffer/input layout

    const UINT stride = sizeof(float) * 4, offset = 0;
    common.DC->IASetVertexBuffers(
        0, 1, common.vertexBuffer.GetAddressOf(), &stride, &offset);

    common.DC->IASetInputLayout(common.inputLayout.Get());
    common.DC->IASetPrimitiveTopology(
        D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    // emit drawcall

    common.DC->Draw(3, 0);

    // clear vertex buffer/input layout

    ID3D11Buffer *NULL_VERTEX_BUFFER = nullptr;
    common.DC->IASetVertexBuffers(
        0, 1, &NULL_VERTEX_BUFFER, &stride, &offset);

    common.DC->IASetInputLayout(nullptr);

    // clear shader/texture/sampler

    ID3D11SamplerState *NULL_SAMPLER = nullptr;
    common.DC->PSSetSamplers(0, 1, &NULL_SAMPLER);

    ID3D11ShaderResourceView *NULL_IMG = nullptr;
    common.DC->PSSetShaderResources(0, 1, &NULL_IMG);

    common.DC->PSSetShader(nullptr, nullptr, 0);
    common.DC->VSSetShader(nullptr, nullptr, 0);
}

inline BlendingWeight::BlendingWeight(
    ID3D11Device *device, int maxSearchDistanceLen, int width, int height)
{
    // pixel shader

    const std::string maxEdgeDetectionLenStr =
        std::to_string(maxSearchDistanceLen);

    const std::string pixelSizeInTexCoordStr = 
        "float2(" + std::to_string(1.0f / width) + ", "
                  + std::to_string(1.0f / height) + ")";

    const D3D_SHADER_MACRO WEIGHT_MACROS[] = {
        { "EDGE_DETECTION_MAX_LEN" , maxEdgeDetectionLenStr.c_str() },
        { "PIXEL_SIZE_IN_TEXCOORD", pixelSizeInTexCoordStr.c_str()  },
        { nullptr                  , nullptr                        }
    };

    ComPtr<ID3D10Blob> weightShaderByteCode = detail::CompileToByteCode(
        detail::WEIGHT_SHADER_SOURCE, "ps_5_0", WEIGHT_MACROS);

    pixelShader = detail::CreatePixelShader(
        device,
        weightShaderByteCode->GetBufferPointer(),
        weightShaderByteCode->GetBufferSize());

    // inner area texture

    int innerWidth, innerHeight;
    const auto texData = detail::GenerateInnerAreaTexture(
        maxSearchDistanceLen, &innerWidth, &innerHeight);

    // create d3d11 texture

    D3D11_TEXTURE2D_DESC texDesc;
    texDesc.Width              = innerWidth;
    texDesc.Height             = innerHeight;
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
    initData.SysMemPitch      = innerWidth * sizeof(float) * 4;
    initData.SysMemSlicePitch = 0;

    HRESULT hr = device->CreateTexture2D(
        &texDesc, &initData, innerAreaTexture.GetAddressOf());
    if(FAILED(hr))
        throw std::runtime_error("failed to create inner area texture");

    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
    srvDesc.Format                    = DXGI_FORMAT_R32G32B32A32_FLOAT;
    srvDesc.ViewDimension             = D3D11_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MostDetailedMip = 0;
    srvDesc.Texture2D.MipLevels       = 1;

    hr = device->CreateShaderResourceView(
        innerAreaTexture.Get(), &srvDesc, innerAreaTextureSRV.GetAddressOf());
    if(FAILED(hr))
    {
        throw std::runtime_error(
            "failed to create shader resource view for inner area texture");
    }
}

inline void BlendingWeight::computeBlendingWeight(
    const Common &common, ID3D11ShaderResourceView *edgeTexture) const
{
    // bind shader/texture/sampler/constant buffer

    common.DC->VSSetShader(common.vertexShader.Get(), nullptr, 0);
    common.DC->PSSetShader(pixelShader.Get(), nullptr, 0);
    common.DC->PSSetSamplers(0, 1, common.pointSampler.GetAddressOf());
    common.DC->PSSetSamplers(1, 1, common.linearSampler.GetAddressOf());
    common.DC->PSSetShaderResources(0, 1, &edgeTexture);
    common.DC->PSSetShaderResources(1, 1, innerAreaTextureSRV.GetAddressOf());

    // bind vertex buffer/input layout

    const UINT stride = sizeof(float) * 4, offset = 0;
    common.DC->IASetVertexBuffers(
        0, 1, common.vertexBuffer.GetAddressOf(), &stride, &offset);

    common.DC->IASetInputLayout(common.inputLayout.Get());
    common.DC->IASetPrimitiveTopology(
        D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    // emit drawcall

    common.DC->Draw(3, 0);

    // clear vertex buffer/input layout

    ID3D11Buffer *NULL_VERTEX_BUFFER = nullptr;
    common.DC->IASetVertexBuffers(
        0, 1, &NULL_VERTEX_BUFFER, &stride, &offset);

    common.DC->IASetInputLayout(nullptr);

    // clear shader/texture/sampler

    ID3D11SamplerState *NULL_SAMPLER = nullptr;
    common.DC->PSSetSamplers(0, 1, &NULL_SAMPLER);
    common.DC->PSSetSamplers(1, 1, &NULL_SAMPLER);

    ID3D11ShaderResourceView *NULL_IMG = nullptr;
    common.DC->PSSetShaderResources(0, 1, &NULL_IMG);
    common.DC->PSSetShaderResources(1, 1, &NULL_IMG);

    common.DC->PSSetShader(nullptr, nullptr, 0);
    common.DC->VSSetShader(nullptr, nullptr, 0);
}

inline Blending::Blending(
    ID3D11Device *device,
    int           width,
    int           height)
{
    const std::string pixelSizeInTexCoordStr =
        "float2(" + std::to_string(1.0f / width) + ", "
        + std::to_string(1.0f / height) + ")";

    const D3D_SHADER_MACRO MACROS[] = {
        { "PIXEL_SIZE_IN_TEXCOORD", pixelSizeInTexCoordStr.c_str()  },
        { nullptr                  , nullptr                        }
    };

    ComPtr<ID3D10Blob> blendingShaderByteCode = detail::CompileToByteCode(
        detail::BLENDING_SHADER_SOURCE, "ps_5_0", MACROS);

    pixelShader = detail::CreatePixelShader(
        device,
        blendingShaderByteCode->GetBufferPointer(),
        blendingShaderByteCode->GetBufferSize());
}

inline void Blending::blend(
    const Common         &common,
    ID3D11ShaderResourceView *weightTexture,
    ID3D11ShaderResourceView *img) const
{
    // bind shader/texture/sampler/constant buffer

    common.DC->VSSetShader(common.vertexShader.Get(), nullptr, 0);
    common.DC->PSSetShader(pixelShader.Get(), nullptr, 0);
    common.DC->PSSetSamplers(0, 1, common.pointSampler.GetAddressOf());
    common.DC->PSSetSamplers(1, 1, common.linearSampler.GetAddressOf());
    common.DC->PSSetShaderResources(0, 1, &img);
    common.DC->PSSetShaderResources(1, 1, &weightTexture);
    
    // bind vertex buffer/input layout

    const UINT stride = sizeof(float) * 4, offset = 0;
    common.DC->IASetVertexBuffers(
        0, 1, common.vertexBuffer.GetAddressOf(), &stride, &offset);

    common.DC->IASetInputLayout(common.inputLayout.Get());
    common.DC->IASetPrimitiveTopology(
        D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    // emit drawcall

    common.DC->Draw(3, 0);

    // clear vertex buffer/input layout

    ID3D11Buffer *NULL_VERTEX_BUFFER = nullptr;
    common.DC->IASetVertexBuffers(
        0, 1, &NULL_VERTEX_BUFFER, &stride, &offset);

    common.DC->IASetInputLayout(nullptr);

    // clear shader/texture/sampler

    ID3D11SamplerState *NULL_SAMPLER = nullptr;
    common.DC->PSSetSamplers(0, 1, &NULL_SAMPLER);
    common.DC->PSSetSamplers(1, 1, &NULL_SAMPLER);

    ID3D11ShaderResourceView *NULL_IMG = nullptr;
    common.DC->PSSetShaderResources(0, 1, &NULL_IMG);
    common.DC->PSSetShaderResources(1, 1, &NULL_IMG);

    common.DC->PSSetShader(nullptr, nullptr, 0);
    common.DC->VSSetShader(nullptr, nullptr, 0);
}

inline MLAA::MLAA(
    ID3D11Device        *device,
    ID3D11DeviceContext *deviceContext,
    int                  width,
    int                  height,
    EdgeDetectionMode    mode,
    float                edgeDetectionThreshold,
    int                  maxSearchDistanceLen)
    : common_        (device, deviceContext),
      edgeDetection_ (device, mode, edgeDetectionThreshold),
      blendingWeight_(device, maxSearchDistanceLen, width, height),
      blending_      (device, width, height)
{

}

inline void MLAA::detectEdge(
    ID3D11ShaderResourceView *source) const
{
    edgeDetection_.detectEdge(common_, source);
}

inline void MLAA::computeBlendingWeight(
    ID3D11ShaderResourceView *edgeTexture) const
{
    blendingWeight_.computeBlendingWeight(common_, edgeTexture);
}

inline void MLAA::blend(
    ID3D11ShaderResourceView *weightTexture,
    ID3D11ShaderResourceView *img) const
{
    blending_.blend(common_, weightTexture, img);
}

} } // namespace agz::mlaa
