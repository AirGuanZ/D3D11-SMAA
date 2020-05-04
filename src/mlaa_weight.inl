R"___(

// #define EDGE_DETECTION_MAX_LEN

struct PSInput
{
    float4 position : SV_POSITION;
    float2 texCoord : TEXCOORD;
};

cbuffer PSConstant : register(b0)
{
    float2 PIXEL_SIZE_IN_TEXCOORD;
};

Texture2D<float4> EdgeTexture      : register(t0);
Texture2D<float4> InnerAreaTexture : register(t1);

SamplerState      PointSampler  : register(s0);
SamplerState      LinearSampler : register(s1);

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

)___"
