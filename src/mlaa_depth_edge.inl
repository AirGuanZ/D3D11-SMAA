R"___(

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
    float d_right = DepthTexture.SampleLevel(
        PointSampler, input.texCoord, 0, int2(+1, 0));
    float d_top = DepthTexture.SampleLevel(
        PointSampler, input.texCoord, 0, int2(0, -1));;
    float d_bottom = DepthTexture.SampleLevel(
        PointSampler, input.texCoord, 0, int2(0, 1));

    // compute delta depth

    float4 delta_d = abs(d.xxxx - float4(d_left, d_top, d_right, d_bottom));
    
    float4 is_edge = step(EDGE_THRESHOLD, delta_d);
    if(dot(is_edge, 1) == 0)
        discard;

    return is_edge;
}

)___"
