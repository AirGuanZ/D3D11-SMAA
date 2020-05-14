// #define EDGE_DETECTION_MAX_LEN XXX
// #define PIXEL_SIZE_IN_TEXCOORD XXX

// IMPROVE: performance optimization

struct PSInput
{
    float4 position : SV_POSITION;
    float2 texCoord : TEXCOORD;
};

Texture2D<float4> EdgeTexture      : register(t0);
Texture2D<float4> InnerAreaTexture : register(t1);

SamplerState PointSampler  : register(s0);
SamplerState LinearSampler : register(s1);

bool is_edge_end(float e, float ce)
{
    return e < 0.87 || ce > 0.01;
}

float end_len_high(float e, float ce)
{
    // 4 weighted contributor of e:
    // 0.6525, 0.21875, 0.09375, 0.03125
    if(e < 0.64)
        return 0;
    float ans_e = e > 0.87 ? 2 : 1;

    // there is 16 value cases of ce
    // in which only 4 'segments' of end len
    // TODO: LUT optimization
    float ans_ce;
    if(ce < 0.09)
        ans_ce = 2;
    else if(ce < 0.2)
        ans_ce = 1;
    else if(ce < 0.3)
        ans_ce = 2;
    else
        ans_ce = 1;

    return min(ans_e, ans_ce);
}

float end_len_low(float e, float ce)
{
    if(e < 0.64)
        return 0;
    float ans_e = e > 0.87 ? 2 : 1;

    // TODO: LUT optimization
    float ans_ce;
    if(ce < 0.03)
        ans_ce = 2;
    else if(ce < 0.09)
        ans_ce = 1;
    else if(ce < 0.2)
        ans_ce = 0;
    else if(ce < 0.3)
        ans_ce = 1;
    else
        ans_ce = 0;

    return min(ans_e, ans_ce);
}

float find_left_end(float2 c)
{
    c += float2(-0.25, 0.125) * PIXEL_SIZE_IN_TEXCOORD;

    for(int p2 = 0; p2 < EDGE_DETECTION_MAX_LEN; ++p2)
    {
        float2 ce_e = EdgeTexture.SampleLevel(LinearSampler, c, 0).rg;

        if(is_edge_end(ce_e.g, ce_e.r))
        {
            float ans = -2 * p2 + 1 - end_len_high(ce_e.g, ce_e.r);
            return min(0, ans);
        }

        c += float2(-2, 0) * PIXEL_SIZE_IN_TEXCOORD;
    }

    return -2 * EDGE_DETECTION_MAX_LEN + 1;
}

float find_right_end(float2 c)
{
    c += float2(1.25, 0.125) * PIXEL_SIZE_IN_TEXCOORD;

    for(int p2 = 0; p2 < EDGE_DETECTION_MAX_LEN; ++p2)
    {
        float2 ce_e = EdgeTexture.SampleLevel(LinearSampler, c, 0).rg;

        if(is_edge_end(ce_e.g, ce_e.r))
            return 2 * p2 + end_len_low(ce_e.g, ce_e.r);

        c += float2(2, 0) * PIXEL_SIZE_IN_TEXCOORD;
    }

    return 2 * EDGE_DETECTION_MAX_LEN;
}

float find_top_end(float2 c)
{
    c += float2(-0.125, -0.25) * PIXEL_SIZE_IN_TEXCOORD;

    for(int p2 = 0; p2 < EDGE_DETECTION_MAX_LEN; ++p2)
    {
        float2 e_ce = EdgeTexture.SampleLevel(LinearSampler, c, 0).rg;

        if(is_edge_end(e_ce.r, e_ce.g))
        {
            float ans = -2 * p2 + 1 - end_len_high(e_ce.r, e_ce.g);
            return min(0, ans);
        }

        c += float2(0, -2) * PIXEL_SIZE_IN_TEXCOORD;
    }

    return -2 * EDGE_DETECTION_MAX_LEN + 1;
}

float find_bottom_end(float2 c)
{
    c += float2(-0.125, 1.25) * PIXEL_SIZE_IN_TEXCOORD;

    for(int p2 = 0; p2 < EDGE_DETECTION_MAX_LEN; ++p2)
    {
        float2 e_ce = EdgeTexture.SampleLevel(LinearSampler, c, 0).rg;

        if(is_edge_end(e_ce.r, e_ce.g))
            return 2 * p2 + end_len_low(e_ce.r, e_ce.g);

        c += float2(0, 2) * PIXEL_SIZE_IN_TEXCOORD;
    }

    return 2 * EDGE_DETECTION_MAX_LEN;
}

float2 inner_area(float dist1, float cross1, float dist2, float cross2)
{
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
        float top_end = find_top_end(input.texCoord);
        float bottom_end = find_bottom_end(input.texCoord);

        float2 coord_top = float2(
            input.texCoord.x - 0.25 * PIXEL_SIZE_IN_TEXCOORD.x,
            input.texCoord.y + top_end * PIXEL_SIZE_IN_TEXCOORD.y);
        float2 coord_bottom = float2(
            input.texCoord.x - 0.25 * PIXEL_SIZE_IN_TEXCOORD.x,
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
        float left_end = find_left_end(input.texCoord);
        float right_end = find_right_end(input.texCoord);

        float2 coord_left = float2(
            input.texCoord.x + left_end * PIXEL_SIZE_IN_TEXCOORD.x,
            input.texCoord.y - 0.25 * PIXEL_SIZE_IN_TEXCOORD.y);
        float2 coord_right = float2(
            input.texCoord.x + (right_end + 1) * PIXEL_SIZE_IN_TEXCOORD.x,
            input.texCoord.y - 0.25 * PIXEL_SIZE_IN_TEXCOORD.y);

        float cross_left = EdgeTexture.SampleLevel(
            LinearSampler, coord_left, 0).r;
        float cross_right = EdgeTexture.SampleLevel(
            LinearSampler, coord_right, 0).r;

        output.rg = inner_area(
            -left_end, cross_left, right_end, cross_right);
    }

    return output;
}
