// #define EDGE_DETECTION_MAX_LEN XXX
// #define PIXEL_SIZE_IN_TEXCOORD XXX
// #define CORNER_AREA_FACTOR     XXX

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

float at_corner_factor(float dl, float dr, float e1, float e2)
{
    if(dl < dr && e1 > 0.1)
        return CORNER_AREA_FACTOR;
    if(dl > dr && e2 > 0.1)
        return CORNER_AREA_FACTOR;
    return 1;
}

float ab_corner_factor(float dl, float dr, float e3, float e4)
{
    if(dl < dr && e3 > 0.1)
        return CORNER_AREA_FACTOR;
    if(dl > dr && e4 > 0.1)
        return CORNER_AREA_FACTOR;
    return 1;
}

float al_corner_factor(float db, float dt, float e1, float e2)
{
    if(db < dt && e1 > 0.1)
        return CORNER_AREA_FACTOR;
    if(db > dt && e2 > 0.1)
        return CORNER_AREA_FACTOR;
    return 1;
}

float ar_corner_factor(float db, float dt, float e3, float e4)
{
    if(db < dt && e3 > 0.1)
        return CORNER_AREA_FACTOR;
    if(db > dt && e4 > 0.1)
        return CORNER_AREA_FACTOR;
    return 1;
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

        float2 e1_coord = input.texCoord + float2(-2, bottom_end + 1) * PIXEL_SIZE_IN_TEXCOORD;
        float2 e2_coord = input.texCoord + float2(-2, top_end) * PIXEL_SIZE_IN_TEXCOORD;
        float2 e3_coord = input.texCoord + float2(1, bottom_end + 1) * PIXEL_SIZE_IN_TEXCOORD;
        float2 e4_coord = input.texCoord + float2(1, top_end) * PIXEL_SIZE_IN_TEXCOORD;

        float e1 = EdgeTexture.SampleLevel(PointSampler, e1_coord, 0).g;
        float e2 = EdgeTexture.SampleLevel(PointSampler, e2_coord, 0).g;
        float e3 = EdgeTexture.SampleLevel(PointSampler, e3_coord, 0).g;
        float e4 = EdgeTexture.SampleLevel(PointSampler, e4_coord, 0).g;

        output.b *= ar_corner_factor(bottom_end, -top_end, e3, e4);
        output.a *= al_corner_factor(bottom_end, -top_end, e1, e2);
    }

    // edge at top side
    if(e.g)
    {
        // find left/right edge length

        float left_end = find_left_end(input.texCoord);
        float right_end = find_right_end(input.texCoord);

        // compute at & ab

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

        // corner area factor

        float2 e1_coord = input.texCoord + float2(left_end,      -2) * PIXEL_SIZE_IN_TEXCOORD;
        float2 e2_coord = input.texCoord + float2(right_end + 1, -2) * PIXEL_SIZE_IN_TEXCOORD;
        float2 e3_coord = input.texCoord + float2(left_end,       1) * PIXEL_SIZE_IN_TEXCOORD;
        float2 e4_coord = input.texCoord + float2(right_end + 1,  1) * PIXEL_SIZE_IN_TEXCOORD;

        float e1 = EdgeTexture.SampleLevel(PointSampler, e1_coord, 0).r;
        float e2 = EdgeTexture.SampleLevel(PointSampler, e2_coord, 0).r;
        float e3 = EdgeTexture.SampleLevel(PointSampler, e3_coord, 0).r;
        float e4 = EdgeTexture.SampleLevel(PointSampler, e4_coord, 0).r;

        output.r *= ab_corner_factor(-left_end, right_end, e3, e4);
        output.g *= at_corner_factor(-left_end, right_end, e1, e2);
    }

    return output;
}