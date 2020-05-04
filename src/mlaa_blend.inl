R"___(

struct PSInput
{
    float4 position : SV_POSITION;
    float2 texCoord : TEXCOORD;
};

cbuffer PSConstant : register(b0)
{
    float2 PIXEL_SIZE_IN_TEXCOORD;
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

)___"
