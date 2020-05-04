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

float diff_lum(float3 a, float3 b)
{
    return to_lum(abs(a - b));
}

float4 main(PSInput input) : SV_TARGET
{
    // sample depth texture

    float3 d = ImageTexture.SampleLevel(
        PointSampler, input.texCoord, 0);
    float3 d_left = ImageTexture.SampleLevel(
        PointSampler, input.texCoord, 0, int2(-1, 0));
    float3 d_right = ImageTexture.SampleLevel(
        PointSampler, input.texCoord, 0, int2(+1, 0));
    float3 d_top = ImageTexture.SampleLevel(
        PointSampler, input.texCoord, 0, int2(0, -1));
    float3 d_bottom = ImageTexture.SampleLevel(
        PointSampler, input.texCoord, 0, int2(0, +1));

    // compute delta lum

    //float4 delta_d = abs(d.xxxx - float4(
    //                                to_lum(d_left),
    //                                to_lum(d_top),
    //                                to_lum(d_right),
    //                                to_lum(d_bottom)));

    float4 delta_d = float4(
                        diff_lum(d_left  , d),
                        diff_lum(d_top   , d),
                        diff_lum(d_right , d),
                        diff_lum(d_bottom, d));
    
    float4 is_edge = step(EDGE_THRESHOLD, delta_d);
    if(dot(is_edge, 1) == 0)
        discard;

    return is_edge;
}

)___"
