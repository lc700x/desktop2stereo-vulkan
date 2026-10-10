#pragma once

// Shared display-only FXAA. Detect encoded RGB edges; blend in linear RGB.
static const char *D2S_FXAA_MSL = R"MSL(
#include <metal_stdlib>
using namespace metal;

// Match the six uint32 fields passed by the native and torch.mps wrappers.
struct FxaaParams {
    uint width;
    uint height;
    uint channels;
    uint layout; // 0: interleaved uint8, 1: planar float32, 2: planar uint8.
    uint eye_width;
    uint batches;
};

static inline uint aa_index(int2 point, uint batch, uint channel,
                             constant FxaaParams& p) {
    uint pixel = uint(point.y) * p.width + uint(point.x);
    uint plane = p.width * p.height;
    return p.layout == 0u
        ? (batch * plane + pixel) * p.channels + channel
        : (batch * p.channels + channel) * plane + pixel;
}

static inline uint aa_pixel_index(int2 point, uint batch,
                                  constant FxaaParams& p) {
    return batch * p.width * p.height + uint(point.y) * p.width + uint(point.x);
}

static inline float3 aa_read_rgb(device const uchar *image,
                                  device const float *image_float, int2 point,
                                  uint batch, int eye_first, int eye_last,
                                  constant FxaaParams& p) {
    point.x = clamp(point.x, eye_first, eye_last);
    point.y = clamp(point.y, 0, int(p.height) - 1);
    if (p.layout == 0u && p.channels == 4u) {
        // CoreML's native packer emits interleaved RGBA8. Load all channels
        // once so the many FXAA luma taps do not issue three byte loads.
        device const uchar4 *pixels = reinterpret_cast<device const uchar4 *>(image);
        return float3(pixels[aa_pixel_index(point, batch, p)].xyz) / 255.0f;
    }
    uint r = aa_index(point, batch, 0u, p);
    uint g = aa_index(point, batch, 1u, p);
    uint b = aa_index(point, batch, 2u, p);
    return p.layout == 1u ? float3(image_float[r], image_float[g], image_float[b])
                         : float3(image[r], image[g], image[b]) / 255.0f;
}

static inline float aa_luma(float3 rgb) {
    return dot(rgb, float3(0.299f, 0.587f, 0.114f));
}

static inline float aa_sample_luma(device const uchar *image, device const float *image_float, float2 point, uint batch,
                                    int eye_first, int eye_last,
                                    constant FxaaParams& p) {
    int2 base = int2(floor(point));
    float2 fraction = fract(point);
    float a = aa_luma(aa_read_rgb(image, image_float, base, batch, eye_first, eye_last, p));
    float b = aa_luma(aa_read_rgb(image, image_float, base + int2(1, 0), batch, eye_first, eye_last, p));
    float c = aa_luma(aa_read_rgb(image, image_float, base + int2(0, 1), batch, eye_first, eye_last, p));
    float d = aa_luma(aa_read_rgb(image, image_float, base + int2(1, 1), batch, eye_first, eye_last, p));
    return mix(mix(a, b, fraction.x), mix(c, d, fraction.x), fraction.y);
}

static inline float3 aa_decode_srgb(float3 rgb) {
    return select(rgb / 12.92f, pow((rgb + 0.055f) / 1.055f, float3(2.4f)),
                  rgb > float3(0.04045f));
}

static inline float3 aa_encode_srgb(float3 rgb) {
    rgb = clamp(rgb, float3(0.0f), float3(1.0f));
    return select(rgb * 12.92f, 1.055f * pow(rgb, float3(1.0f / 2.4f)) - 0.055f,
                  rgb > float3(0.0031308f));
}

static inline float3 aa_sample_linear_rgb(device const uchar *image, device const float *image_float, float2 point, uint batch,
                                           int eye_first, int eye_last,
                                           constant FxaaParams& p) {
    int2 base = int2(floor(point));
    float2 fraction = fract(point);
    float3 a = aa_decode_srgb(aa_read_rgb(image, image_float, base, batch, eye_first, eye_last, p));
    // The FXAA normal is axis-aligned; only two endpoints have nonzero weight.
    int2 next = fraction.x == 0.0f ? int2(0, 1) : int2(1, 0);
    float weight = fraction.x == 0.0f ? fraction.y : fraction.x;
    float3 b = aa_decode_srgb(aa_read_rgb(image, image_float, base + next, batch, eye_first, eye_last, p));
    return aa_encode_srgb(mix(a, b, weight));
}

static inline float3 aa_fxaa_pixel(
    device const uchar *source, device const float *source_float,
    int2 point, uint batch, constant FxaaParams& p) {
    int eye_width = int(p.eye_width > 0u && p.eye_width < p.width ? p.eye_width : p.width);
    int eye_first = point.x < eye_width ? 0 : eye_width;
    int eye_last = eye_first == 0 ? eye_width - 1 : int(p.width) - 1;
    float3 center_rgb = aa_read_rgb(source, source_float, point, batch, eye_first, eye_last, p);
    float M = aa_luma(center_rgb);
    float N = aa_luma(aa_read_rgb(source, source_float, point + int2(0, -1), batch, eye_first, eye_last, p));
    float S = aa_luma(aa_read_rgb(source, source_float, point + int2(0, 1), batch, eye_first, eye_last, p));
    float W = aa_luma(aa_read_rgb(source, source_float, point + int2(-1, 0), batch, eye_first, eye_last, p));
    float E = aa_luma(aa_read_rgb(source, source_float, point + int2(1, 0), batch, eye_first, eye_last, p));
    float minimum = min(M, min(min(N, S), min(W, E)));
    float maximum = max(M, max(max(N, S), max(W, E)));
    float range = maximum - minimum;
    if (range < max(0.0312f, maximum * 0.125f) + 1.0e-5f) {
        return center_rgb;
    }
    float NW = aa_luma(aa_read_rgb(source, source_float, point + int2(-1, -1), batch, eye_first, eye_last, p));
    float NE = aa_luma(aa_read_rgb(source, source_float, point + int2(1, -1), batch, eye_first, eye_last, p));
    float SW = aa_luma(aa_read_rgb(source, source_float, point + int2(-1, 1), batch, eye_first, eye_last, p));
    float SE = aa_luma(aa_read_rgb(source, source_float, point + int2(1, 1), batch, eye_first, eye_last, p));
    float horizontal = 2.0f * abs(N + S - 2.0f * M) +
                       abs(NW + SW - 2.0f * W) + abs(NE + SE - 2.0f * E);
    float vertical = 2.0f * abs(W + E - 2.0f * M) +
                     abs(NW + NE - 2.0f * N) + abs(SW + SE - 2.0f * S);
    bool horizontal_edge = horizontal >= vertical - 1.0e-6f;
    float neighbor_negative = horizontal_edge ? N : W;
    float neighbor_positive = horizontal_edge ? S : E;
    float gradient_negative = neighbor_negative - M;
    float gradient_positive = neighbor_positive - M;
    bool use_negative = abs(gradient_negative) >= abs(gradient_positive) - 1.0e-6f;
    float gradient = max(abs(gradient_negative), abs(gradient_positive));
    float edge_luma = 0.5f * (M + (use_negative ? neighbor_negative : neighbor_positive));
    float2 normal = horizontal_edge ? float2(0.0f, 1.0f) : float2(1.0f, 0.0f);
    if (use_negative) normal = -normal;
    float2 tangent = horizontal_edge ? float2(1.0f, 0.0f) : float2(0.0f, 1.0f);
    float2 edge_center = float2(point) + normal * 0.5f;
    float distance_negative = 1.0f;
    float distance_positive = 1.0f;
    float delta_negative = aa_sample_luma(source, source_float, edge_center - tangent, batch, eye_first, eye_last, p) - edge_luma;
    float delta_positive = aa_sample_luma(source, source_float, edge_center + tangent, batch, eye_first, eye_last, p) - edge_luma;
    bool done_negative = abs(delta_negative) >= gradient * 0.25f - 1.1e-4f;
    bool done_positive = abs(delta_positive) >= gradient * 0.25f - 1.1e-4f;
    const float quality_steps[6] = {1.5f, 2.0f, 2.0f, 2.0f, 4.0f, 8.0f};
    for (int step = 0; step < 6; ++step) {
        if (done_negative && done_positive) break;
        if (!done_negative) {
            distance_negative += quality_steps[step];
            delta_negative = aa_sample_luma(source, source_float, edge_center - tangent * distance_negative, batch,
                                            eye_first, eye_last, p) - edge_luma;
            done_negative = abs(delta_negative) >= gradient * 0.25f - 1.1e-4f;
        }
        if (!done_positive) {
            distance_positive += quality_steps[step];
            delta_positive = aa_sample_luma(source, source_float, edge_center + tangent * distance_positive, batch,
                                            eye_first, eye_last, p) - edge_luma;
            done_positive = abs(delta_positive) >= gradient * 0.25f - 1.1e-4f;
        }
    }
    bool nearest_negative = distance_negative <= distance_positive;
    float nearest_distance = min(distance_negative, distance_positive);
    float nearest_delta = nearest_negative ? delta_negative : delta_positive;
    bool sign_valid = (nearest_delta < 0.0f) != (M < edge_luma);
    float edge_offset = sign_valid
        ? 0.5f - nearest_distance / (distance_negative + distance_positive) : 0.0f;
    float neighborhood = (2.0f * (N + S + W + E) + NW + NE + SW + SE) / 12.0f;
    float subpixel = clamp(abs(neighborhood - M) / max(range, 1.0e-6f), 0.0f, 1.0f);
    subpixel = subpixel * subpixel * (3.0f - 2.0f * subpixel);
    subpixel = subpixel * subpixel * 1.25f;
    float offset = max(edge_offset, subpixel);
    float3 rgb = aa_sample_linear_rgb(source, source_float, float2(point) + normal * offset, batch,
                                       eye_first, eye_last, p);
    return rgb;
}

kernel void d2s_sbs_fxaa(
    device const uchar *source [[buffer(0)]],
    device const float *source_float [[buffer(1)]],
    device uchar *output [[buffer(2)]],
    device float *output_float [[buffer(3)]],
    constant FxaaParams& p [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.width || gid.y >= p.height || gid.z >= p.batches) return;
    int2 point = int2(gid.xy);
    float3 rgb = aa_fxaa_pixel(source, source_float, point, gid.z, p);
    for (uint channel = 0u; channel < 3u; ++channel) {
        uint index = aa_index(point, gid.z, channel, p);
        if (p.layout == 1u) output_float[index] = rgb[channel];
        else output[index] = uchar(clamp(rgb[channel] * 255.0f + 0.5f, 0.0f, 255.0f));
    }
    if (p.channels == 4u) {
        uint index = aa_index(point, gid.z, 3u, p);
        if (p.layout == 1u) output_float[index] = source_float[index];
        else output[index] = source[index];
    }
}

// Filter complete eye pixels before Half-SBS area reduction. Keeping both
// operations in one kernel avoids another full-resolution eye allocation/pass.
kernel void d2s_sbs_fxaa_half(
    device const uchar *source [[buffer(0)]],
    device const float *source_float [[buffer(1)]],
    device uchar *output [[buffer(2)]],
    device float *output_float [[buffer(3)]],
    constant FxaaParams& p [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {
    uint output_width = p.width / 2u;
    if (gid.x >= output_width || gid.y >= p.height || gid.z >= p.batches) return;
    uint left_output_width = p.eye_width / 2u;
    bool right_eye = gid.x >= left_output_width;
    uint output_eye_x = right_eye ? gid.x - left_output_width : gid.x;
    uint eye_output_width = right_eye ? p.eye_width - left_output_width : left_output_width;
    uint eye_source_x = right_eye ? p.eye_width : 0u;
    float source_begin = float(output_eye_x) * float(p.eye_width) / float(eye_output_width);
    float source_end = float(output_eye_x + 1u) * float(p.eye_width) / float(eye_output_width);
    float3 linear_sum = float3(0.0f);
    float alpha_sum = 0.0f;
    float total_weight = 0.0f;
    uint first_source_x = uint(floor(source_begin));
    uint past_source_x = uint(ceil(source_end));
    for (uint tap = 0u; tap < 3u; ++tap) {
        uint source_x = first_source_x + tap;
        if (source_x >= past_source_x) break;
        float weight = max(0.0f, min(source_end, float(source_x + 1u)) -
                                 max(source_begin, float(source_x)));
        int2 point = int2(eye_source_x + source_x, gid.y);
        float3 encoded = aa_fxaa_pixel(source, source_float, point, gid.z, p);
        linear_sum += aa_decode_srgb(encoded) * weight;
        if (p.channels == 4u) {
            uint alpha_index = aa_index(point, gid.z, 3u, p);
            alpha_sum += (p.layout == 1u ? source_float[alpha_index]
                                        : float(source[alpha_index]) / 255.0f) * weight;
        }
        total_weight += weight;
    }
    float3 rgb = aa_encode_srgb(linear_sum / max(total_weight, 1.0e-6f));
    uint pixel = gid.y * output_width + gid.x;
    uint plane = output_width * p.height;
    for (uint channel = 0u; channel < 3u; ++channel) {
        uint index = p.layout == 0u ? (gid.z * plane + pixel) * p.channels + channel
                                   : (gid.z * p.channels + channel) * plane + pixel;
        if (p.layout == 1u) output_float[index] = rgb[channel];
        else output[index] = uchar(clamp(rgb[channel] * 255.0f + 0.5f, 0.0f, 255.0f));
    }
    if (p.channels == 4u) {
        uint index = p.layout == 0u ? (gid.z * plane + pixel) * p.channels + 3u
                                   : (gid.z * p.channels + 3u) * plane + pixel;
        float alpha = alpha_sum / max(total_weight, 1.0e-6f);
        if (p.layout == 1u) output_float[index] = alpha;
        else output[index] = uchar(clamp(alpha * 255.0f + 0.5f, 0.0f, 255.0f));
    }
}


)MSL";
