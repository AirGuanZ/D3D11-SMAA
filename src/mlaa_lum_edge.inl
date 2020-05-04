R"___(

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

float4 main(PSInput input) : SV_TARGET
{
    // sample depth texture

    float d = to_lum(ImageTexture.SampleLevel(
        PointSampler, input.texCoord, 0));
    float d_left = to_lum(ImageTexture.SampleLevel(
        PointSampler, input.texCoord, 0, int2(-1, 0)));
    float d_right = to_lum(ImageTexture.SampleLevel(
        PointSampler, input.texCoord, 0, int2(+1, 0)));
    float d_top = to_lum(ImageTexture.SampleLevel(
        PointSampler, input.texCoord, 0, int2(0, -1)));
    float d_bottom = to_lum(ImageTexture.SampleLevel(
        PointSampler, input.texCoord, 0, int2(0, 1)));

    // compute delta lum

    float4 delta_d = abs(d.xxxx - float4(d_left, d_top, d_right, d_bottom));
    
    float4 is_edge = step(EDGE_THRESHOLD, delta_d);
    if(dot(is_edge, 1) == 0)
        discard;

    return is_edge;
}

)___"
