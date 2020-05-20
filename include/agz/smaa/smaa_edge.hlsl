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
