#import "macos_coreml_io.h"

#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <pthread.h>
#include <time.h>

enum {
    D2S_COREML_OK = 0,
    D2S_COREML_ERROR = -1,
    D2S_COREML_BUSY = -2,
    D2S_COREML_UNSUPPORTED = -3,
    D2S_COREML_OUTPUT_BACKING = -4,
};

typedef struct {
    uint32_t source_width;
    uint32_t source_height;
    uint32_t input_width;
    uint32_t input_height;
    float mean0;
    float mean1;
    float mean2;
    float std0;
    float std1;
    float std2;
} D2SPreprocessParams;

typedef struct {
    uint32_t count;
    float lo;
    float hi;
} D2SNormalizeParams;

typedef struct {
    uint32_t depth_width;
    uint32_t depth_height;
    float strength;
    float sigma_color;
    float max_keep;
    float depth_pop;
    float edge_threshold;
    int32_t axis;
} D2SDepthFilterParams;

typedef struct {
    uint32_t depth_width;
    uint32_t depth_height;
    float depth_pop;
} D2SDepthGaussianParams;

typedef struct {
    uint32_t source_width;
    uint32_t source_height;
    uint32_t depth_width;
    uint32_t depth_height;
    float edge_threshold;
} D2SDepthUpsampleParams;

typedef struct {
    uint32_t source_width;
    uint32_t source_height;
    uint32_t depth_width;
    uint32_t depth_height;
    uint32_t output_width;
    uint32_t output_height;
    uint32_t output_format;
    uint32_t output_channels;
    D2SCoreMLIOWarpConfig stereo;
} D2SWarpParams;

static const char *D2SMetalSource = R"D2S(
#include <metal_stdlib>
using namespace metal;

struct PreprocessParams {
    uint source_width;
    uint source_height;
    uint input_width;
    uint input_height;
    float mean0;
    float mean1;
    float mean2;
    float std0;
    float std1;
    float std2;
};

struct NormalizeParams {
    uint count;
    float lo;
    float hi;
};

struct WarpParams {
    uint source_width;
    uint source_height;
    uint depth_width;
    uint depth_height;
    uint output_width;
    uint output_height;
    uint output_format;
    uint output_channels;
    float depth_strength;
    float max_disparity_px;
    float convergence;
    float edge_threshold;
    float fill_strength;
    int fill_radius;
    int mask_feather_radius;
    int symmetric;
    int layers;
    float softness;
    float foreground_scale;
    float midground_scale;
    float background_scale;
    int edge_dilation;
    int screen_edge_suppression;
    int hole_fill_mode;
    int occlusion_enabled;
    float depth_pop;
    float antialias_strength;
    int anaglyph_method;
    int edge_aa_enabled;
};

kernel void d2s_preprocess(
    texture2d<float, access::sample> source [[texture(0)]],
    device float *output [[buffer(0)]],
    constant PreprocessParams& p [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.input_width || gid.y >= p.input_height) return;
    constexpr sampler s(address::clamp_to_edge, filter::linear);
    float2 uv = (float2(gid) + 0.5f) /
        float2(p.input_width, p.input_height);
    float4 pixel = source.sample(s, uv);
    uint plane = p.input_width * p.input_height;
    uint offset = gid.y * p.input_width + gid.x;
    // MTLPixelFormatBGRA8Unorm exposes logical RGB components to Metal shaders.
    output[offset] = (pixel.r - p.mean0) / p.std0;
    output[plane + offset] = (pixel.g - p.mean1) / p.std1;
    output[2u * plane + offset] = (pixel.b - p.mean2) / p.std2;
}

kernel void d2s_normalize_half(
    device half *source [[buffer(0)]],
    device float *output [[buffer(1)]],
    device atomic_uint *status [[buffer(2)]],
    constant NormalizeParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.count) return;
    float value = float(source[gid]);
    if (!isfinite(value)) {
        value = 1.0f;
        atomic_fetch_add_explicit(status, 1u, memory_order_relaxed);
    }
    output[gid] = clamp((value - p.lo) / max(p.hi - p.lo, 1.0e-6f), 0.0f, 1.0f);
}

kernel void d2s_normalize_float(
    device float *source [[buffer(0)]],
    device float *output [[buffer(1)]],
    device atomic_uint *status [[buffer(2)]],
    constant NormalizeParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.count) return;
    float value = source[gid];
    if (!isfinite(value)) {
        value = 1.0f;
        atomic_fetch_add_explicit(status, 1u, memory_order_relaxed);
    }
    output[gid] = clamp((value - p.lo) / max(p.hi - p.lo, 1.0e-6f), 0.0f, 1.0f);
}

static inline float depth_sample(device float *depth, uint width, uint height,
                                 float u, float v) {
    float x = clamp(u * float(width) - 0.5f, 0.0f, float(width) - 1.001f);
    float y = clamp(v * float(height) - 0.5f, 0.0f, float(height) - 1.001f);
    int x0 = int(floor(x));
    int y0 = int(floor(y));
    int x1 = min(x0 + 1, int(width) - 1);
    int y1 = min(y0 + 1, int(height) - 1);
    float fx = x - float(x0);
    float fy = y - float(y0);
    float a = depth[y0 * width + x0];
    float b = depth[y0 * width + x1];
    float c = depth[y1 * width + x0];
    float d = depth[y1 * width + x1];
    return mix(mix(a, b, fx), mix(c, d, fx), fy);
}

static inline float depth_at(device float *depth, float x, float y,
                             constant WarpParams& p) {
    float u = (x + 0.5f) / float(p.source_width);
    float v = (y + 0.5f) / float(p.source_height);
    return clamp(depth_sample(depth, p.depth_width, p.depth_height, u, v),
                 0.0f, 1.0f);
}

static inline float depth_pop(float value, constant WarpParams& p) {
    float pop = p.depth_pop;
    if (abs(pop) < 1.0e-6f) return value;
    float centered = value - 0.5f;
    if (pop < 0.0f) {
        return clamp(0.5f + centered * (1.0f - clamp(-pop, 0.0f, 1.0f)),
                     0.0f, 1.0f);
    }
    float exponent = 1.0f / (1.0f + pop);
    float magnitude = pow(abs(centered), exponent);
    return clamp(0.5f + (centered < 0.0f ? -magnitude : magnitude), 0.0f, 1.0f);
}

static inline float conservative_depth_at(device float *depth, float x, float y,
                                         constant WarpParams& p) {
    float sx = clamp(x, 0.0f, float(p.source_width) - 1.0f);
    float sy = clamp(y, 0.0f, float(p.source_height) - 1.0f);
    float dx = clamp((sx + 0.5f) * float(p.depth_width) /
                     float(p.source_width) - 0.5f,
                     0.0f, float(p.depth_width - 1u));
    float dy = clamp((sy + 0.5f) * float(p.depth_height) /
                     float(p.source_height) - 0.5f,
                     0.0f, float(p.depth_height - 1u));
    uint x0 = uint(floor(dx));
    uint x1 = min(x0 + 1u, p.depth_width - 1u);
    uint iy = min(uint(floor(dy + 0.5f)), p.depth_height - 1u);
    float a = depth_pop(depth[iy * p.depth_width + x0], p);
    float b = depth_pop(depth[iy * p.depth_width + x1], p);
    return max(a, b);
}

static inline float processed_depth_at(device float *depth, float x, float y,
                                       constant WarpParams& p) {
    // The native output is already percentile-normalized. Apply the same
    // optional depth-pop control as postprocess_depth before stereo sampling.
    if (p.antialias_strength <= 0.0f) {
        return depth_pop(depth_at(depth, x, y, p), p);
    }
    int radius = min(int(3.0f * p.antialias_strength) / 2, 3);
    if (radius <= 0) return depth_pop(depth_at(depth, x, y, p), p);
    float sigma = max(0.5f * p.antialias_strength, 1.0e-4f);
    float total = 0.0f;
    float weight_total = 0.0f;
    for (int offset = -3; offset <= 3; ++offset) {
        if (abs(offset) <= radius) {
            float distance = float(offset);
            float weight = exp(-(distance * distance) / (2.0f * sigma * sigma));
            total += depth_pop(depth_at(depth, x + distance, y, p), p) * weight;
            weight_total += weight;
        }
    }
    float horizontal = total / max(weight_total, 1.0e-6f);
    total = 0.0f;
    weight_total = 0.0f;
    for (int offset = -3; offset <= 3; ++offset) {
        if (abs(offset) <= radius) {
            float distance = float(offset);
            float weight = exp(-(distance * distance) / (2.0f * sigma * sigma));
            float value = depth_pop(depth_at(depth, x, y + distance, p), p);
            total += (offset == 0 ? horizontal : value) * weight;
            weight_total += weight;
        }
    }
    return clamp(total / max(weight_total, 1.0e-6f), 0.0f, 1.0f);
}

static inline float layered_depth_scale(float value, constant WarpParams& p) {
    float normalized = clamp(value, 0.0f, 1.0f);
    float background_weight = clamp((0.5f - normalized) * 2.0f, 0.0f, 1.0f);
    float foreground_weight = clamp((normalized - 0.5f) * 2.0f, 0.0f, 1.0f);
    float midground_weight = clamp(1.0f - background_weight - foreground_weight,
                                   0.0f, 1.0f);
    return background_weight * max(p.background_scale, 0.0f) +
           midground_weight * max(p.midground_scale, 0.0f) +
           foreground_weight * max(p.foreground_scale, 0.0f);
}

static inline float shift_from_depth(float value, constant WarpParams& p) {
    return -(value - p.convergence) * layered_depth_scale(value, p) *
           max(p.depth_strength, 0.0f) * p.max_disparity_px * 0.5f;
}

static inline float layer_weight(float value, int layer_index,
                                 constant WarpParams& p) {
    int count = clamp(p.layers, 1, 4);
    if (count == 1) return 1.0f;
    float center = float(layer_index) / float(count - 1);
    float delta = clamp(value, 0.0f, 1.0f) - center;
    return exp(-(delta * delta) / max(p.softness, 1.0e-4f));
}

static inline float layer_shift_scale(int layer_index, constant WarpParams& p) {
    int count = clamp(p.layers, 1, 4);
    return 0.75f + 0.25f * float(layer_index + 1) / float(count);
}

static inline float effective_layer_shift_scale(float value,
                                                 constant WarpParams& p) {
    int count = clamp(p.layers, 1, 4);
    if (count == 1) return 1.0f;
    if (count == 2) {
        float depth = clamp(value, 0.0f, 1.0f);
        float foreground_weight = 1.0f /
            (1.0f + exp((1.0f - 2.0f * depth) /
                        max(p.softness, 1.0e-4f)));
        return 0.875f + 0.125f * foreground_weight;
    }
    float weighted_scale = 0.0f;
    float total_weight = 0.0f;
    for (int layer_index = 0; layer_index < 4; ++layer_index) {
        if (layer_index >= count) break;
        float weight = layer_weight(value, layer_index, p);
        weighted_scale += weight * layer_shift_scale(layer_index, p);
        total_weight += weight;
    }
    return weighted_scale / max(total_weight, 1.0e-6f);
}

static inline float2 clamp_source(float2 point, constant WarpParams& p) {
    return clamp(point, float2(0.0f),
                 float2(max(float(p.source_width) - 1.0f, 0.0f),
                        max(float(p.source_height) - 1.0f, 0.0f)));
}

static inline float3 sample_color(texture2d<float, access::sample> color,
                                  float x, float y, constant WarpParams& p) {
    constexpr sampler s(address::clamp_to_edge, filter::linear);
    float2 point = clamp_source(float2(x, y), p);
    float2 uv = (point + 0.5f) /
                float2(p.source_width, p.source_height);
    return color.sample(s, uv).rgb;
}

static inline float projected_target_x(float source_x, float depth_value,
                                       float eye_sign,
                                       constant WarpParams& p) {
    return source_x - shift_from_depth(depth_value, p) *
                       effective_layer_shift_scale(depth_value, p) * eye_sign;
}

static inline float dibr_depth_at(
    texture2d<float, access::sample> color, device float *depth,
    float x, float y, constant WarpParams& p) {
    // A negative strength marks model-grid AA already completed by the pack
    // pass. Match CUDA/ROCm's bilinear depth sampling without re-sharpening it.
    if (p.antialias_strength < 0.0f) {
        return depth_pop(depth_at(depth, x, y, p), p);
    }
    float sx = clamp(x, 0.0f, float(p.source_width) - 1.0f);
    float sy = clamp(y, 0.0f, float(p.source_height) - 1.0f);
    float dx = clamp((sx + 0.5f) * float(p.depth_width) /
                     float(p.source_width) - 0.5f,
                     0.0f, float(p.depth_width - 1u));
    float dy = clamp((sy + 0.5f) * float(p.depth_height) /
                     float(p.source_height) - 0.5f,
                     0.0f, float(p.depth_height - 1u));
    uint x0 = uint(floor(dx));
    uint x1 = min(x0 + 1u, p.depth_width - 1u);
    uint y0 = uint(floor(dy));
    uint y1 = min(y0 + 1u, p.depth_height - 1u);
    float fx = dx - float(x0);
    float fy = dy - float(y0);
    uint i00 = y0 * p.depth_width + x0;
    uint i10 = y0 * p.depth_width + x1;
    uint i01 = y1 * p.depth_width + x0;
    uint i11 = y1 * p.depth_width + x1;
    float d00 = depth[i00];
    float d10 = depth[i10];
    float d01 = depth[i01];
    float d11 = depth[i11];
    float minimum = min(min(d00, d10), min(d01, d11));
    float maximum = max(max(d00, d10), max(d01, d11));
    float w00 = (1.0f - fy) * (1.0f - fx);
    float w10 = (1.0f - fy) * fx;
    float w01 = fy * (1.0f - fx);
    float w11 = fy * fx;
    float bilinear = w00 * d00 + w10 * d10 + w01 * d01 + w11 * d11;
    if (maximum - minimum <= max(p.edge_threshold, 0.02f)) return bilinear;

    float source_y0 = (float(y0) + 0.5f) * float(p.source_height) /
                      float(p.depth_height) - 0.5f;
    float source_y1 = (float(y1) + 0.5f) * float(p.source_height) /
                      float(p.depth_height) - 0.5f;
    float source_x0 = (float(x0) + 0.5f) * float(p.source_width) /
                      float(p.depth_width) - 0.5f;
    float source_x1 = (float(x1) + 0.5f) * float(p.source_width) /
                      float(p.depth_width) - 0.5f;
    float3 target_color = sample_color(color, sx, sy, p);
    float3 c00 = sample_color(color, source_x0, source_y0, p);
    float3 c10 = sample_color(color, source_x1, source_y0, p);
    float3 c01 = sample_color(color, source_x0, source_y1, p);
    float3 c11 = sample_color(color, source_x1, source_y1, p);
    float delta00 = dot(abs(target_color - c00), float3(1.0f / 3.0f));
    float delta10 = dot(abs(target_color - c10), float3(1.0f / 3.0f));
    float delta01 = dot(abs(target_color - c01), float3(1.0f / 3.0f));
    float delta11 = dot(abs(target_color - c11), float3(1.0f / 3.0f));
    float q00 = w00 * exp(-delta00 / 0.12f);
    float q10 = w10 * exp(-delta10 / 0.12f);
    float q01 = w01 * exp(-delta01 / 0.12f);
    float q11 = w11 * exp(-delta11 / 0.12f);
    float midpoint = (minimum + maximum) * 0.5f;
    float low00 = 1.0f - smoothstep(midpoint - 0.02f, midpoint + 0.02f, d00);
    float low10 = 1.0f - smoothstep(midpoint - 0.02f, midpoint + 0.02f, d10);
    float low01 = 1.0f - smoothstep(midpoint - 0.02f, midpoint + 0.02f, d01);
    float low11 = 1.0f - smoothstep(midpoint - 0.02f, midpoint + 0.02f, d11);
    float q_low00 = q00 * low00, q_low10 = q10 * low10;
    float q_low01 = q01 * low01, q_low11 = q11 * low11;
    float q_high00 = q00 * (1.0f - low00), q_high10 = q10 * (1.0f - low10);
    float q_high01 = q01 * (1.0f - low01), q_high11 = q11 * (1.0f - low11);
    float low_total = q_low00 + q_low10 + q_low01 + q_low11;
    float high_total = q_high00 + q_high10 + q_high01 + q_high11;
    float low_depth = (q_low00 * d00 + q_low10 * d10 + q_low01 * d01 + q_low11 * d11) /
                      max(low_total, 1.0e-6f);
    float high_depth = (q_high00 * d00 + q_high10 * d10 + q_high01 * d01 + q_high11 * d11) /
                       max(high_total, 1.0e-6f);
    float low_delta = (q_low00 * delta00 + q_low10 * delta10 +
                       q_low01 * delta01 + q_low11 * delta11) / max(low_total, 1.0e-6f);
    float high_delta = (q_high00 * delta00 + q_high10 * delta10 +
                        q_high01 * delta01 + q_high11 * delta11) / max(high_total, 1.0e-6f);
    float3 low_color = (q_low00 * c00 + q_low10 * c10 + q_low01 * c01 + q_low11 * c11) /
                       max(low_total, 1.0e-6f);
    float3 high_color = (q_high00 * c00 + q_high10 * c10 +
                         q_high01 * c01 + q_high11 * c11) / max(high_total, 1.0e-6f);
    float prefer_high = smoothstep(-0.01f, 0.01f, low_delta - high_delta);
    float selected = mix(low_depth, high_depth, prefer_high);
    float depth_confidence = smoothstep(max(p.edge_threshold * 0.5f, 0.01f),
                                        max(p.edge_threshold * 1.5f, 0.06f),
                                        maximum - minimum);
    float color_contrast = dot(abs(low_color - high_color), float3(1.0f / 3.0f));
    float color_confidence = smoothstep(0.005f, 0.03f, color_contrast);
    float class_support = smoothstep(1.0e-4f, 0.01f, min(low_total, high_total));
    return mix(bilinear, selected,
               depth_confidence * color_confidence * clamp(class_support, 0.0f, 1.0f));
}

static inline int dibr_search_radius(constant WarpParams& p) {
    float maximum_scale = max(max(p.foreground_scale, p.midground_scale),
                              max(p.background_scale, 1.0f));
    float depth_span = max(abs(p.convergence), abs(1.0f - p.convergence));
    float shift_bound = depth_span * max(p.depth_strength, 0.0f) *
                        p.max_disparity_px * 0.5f * maximum_scale;
    return min(int(ceil(shift_bound)) + 1, int(p.source_width));
}

static inline float3 sample_color_depth_class(
    texture2d<float, access::sample> color, device float *depth,
    float x, float y, float class_depth, constant WarpParams& p) {
    float sx = clamp(x, 0.0f, float(p.source_width) - 1.0f);
    float sy = clamp(y, 0.0f, float(p.source_height) - 1.0f);
    int x0 = int(floor(sx));
    int y0 = int(floor(sy));
    int x1 = min(x0 + 1, int(p.source_width) - 1);
    int y1 = min(y0 + 1, int(p.source_height) - 1);
    float fx = sx - float(x0);
    float fy = sy - float(y0);
    float w00 = (1.0f - fx) * (1.0f - fy);
    float w10 = fx * (1.0f - fy);
    float w01 = (1.0f - fx) * fy;
    float w11 = fx * fy;
    float3 total = float3(0.0f);
    float weight = 0.0f;
    float threshold = max(p.edge_threshold, 1.0e-6f);
    float d = dibr_depth_at(color, depth, float(x0), float(y0), p);
    float keep = 1.0f - smoothstep(0.5f * threshold, threshold, abs(d - class_depth));
    total += sample_color(color, float(x0), float(y0), p) * w00 * keep;
    weight += w00 * keep;
    d = dibr_depth_at(color, depth, float(x1), float(y0), p);
    keep = 1.0f - smoothstep(0.5f * threshold, threshold, abs(d - class_depth));
    total += sample_color(color, float(x1), float(y0), p) * w10 * keep;
    weight += w10 * keep;
    d = dibr_depth_at(color, depth, float(x0), float(y1), p);
    keep = 1.0f - smoothstep(0.5f * threshold, threshold, abs(d - class_depth));
    total += sample_color(color, float(x0), float(y1), p) * w01 * keep;
    weight += w01 * keep;
    d = dibr_depth_at(color, depth, float(x1), float(y1), p);
    keep = 1.0f - smoothstep(0.5f * threshold, threshold, abs(d - class_depth));
    total += sample_color(color, float(x1), float(y1), p) * w11 * keep;
    weight += w11 * keep;
    return mix(sample_color(color, x, y, p), total / max(weight, 1.0e-6f), min(weight, 1.0f));
}

static inline float3 dibr_gather_at(
    texture2d<float, access::sample> color, device float *depth,
    float target_x, float target_y, float eye_sign,
    constant WarpParams& p) {
    int radius = dibr_search_radius(p);
    // Depth 0..1 projects inside these conservative endpoints; the one-pixel
    // margin covers bilinear splat support and radius caps the search globally.
    float maximum_scale = max(max(p.foreground_scale, p.midground_scale),
                              max(p.background_scale, 1.0f));
    float disparity_bound = max(p.depth_strength, 0.0f) *
                            p.max_disparity_px * 0.5f * maximum_scale;
    float at_zero = target_x + p.convergence * disparity_bound * eye_sign;
    float at_one = target_x - (1.0f - p.convergence) * disparity_bound * eye_sign;
    float search_min = max(min(at_zero, at_one) - 1.0f,
                           target_x - float(radius));
    float search_max = min(max(at_zero, at_one) + 1.0f,
                           target_x + float(radius));
    // Use fixed source-pixel centers for all target subpixel samples. Sliding
    // this lattice with target_x makes the selected depth/color jump over time.
    int first_source_x = int(ceil(search_min));
    int last_source_x = int(floor(search_max));
    float best_depth = -1.0f;
    float best_coverage = 0.0f;
    float best_x = target_x;
    float left_error = 1.0e6f;
    float right_error = 1.0e6f;
    float left_depth = -1.0f;
    float right_depth = -1.0f;
    float left_x = target_x;
    float right_x = target_x;
    for (int source_ix = first_source_x; source_ix <= last_source_x; ++source_ix) {
        float source_x = float(source_ix);
        if (source_x >= 0.0f && source_x < float(p.source_width)) {
            float value = dibr_depth_at(color, depth, source_x, target_y, p);
            float projected_x = projected_target_x(source_x, value, eye_sign, p);
            float error = projected_x - target_x;
            float coverage = max(0.0f, 1.0f - abs(error));
            if (coverage > 0.0f &&
                (value > best_depth ||
                 (value == best_depth && coverage > best_coverage))) {
                best_depth = value;
                best_coverage = coverage;
                best_x = source_x;
            }
            if (error <= 0.0f && abs(error) < left_error) {
                left_error = abs(error);
                left_depth = value;
                left_x = source_x;
            }
            if (error >= 0.0f && error < right_error) {
                right_error = error;
                right_depth = value;
                right_x = source_x;
            }
        }
    }

    if (left_error >= 1.0e5f) {
        left_error = right_error;
        left_depth = right_depth;
        left_x = right_x;
    }
    if (right_error >= 1.0e5f) {
        right_error = left_error;
        right_depth = left_depth;
        right_x = left_x;
    }

    float threshold = max(p.edge_threshold, 1.0e-6f);
    float3 left = sample_color_depth_class(color, depth, left_x, target_y, left_depth, p);
    float3 right = sample_color_depth_class(color, depth, right_x, target_y, right_depth, p);
    float side_fraction = left_error / max(left_error + right_error, 1.0e-6f);
    float3 background_linear = mix(left, right, side_fraction);
    float background_edge = smoothstep(0.5f * threshold, threshold, abs(left_depth - right_depth));
    float3 background = mix(background_linear, left_depth <= right_depth ? left : right, background_edge);
    if (best_depth < 0.0f) return background;

    float3 foreground = float3(0.0f);
    float foreground_coverage = 0.0f;
    for (int offset = -1; offset <= 1; ++offset) {
        float source_x = best_x + float(offset);
        if (source_x >= 0.0f && source_x < float(p.source_width)) {
            float value = dibr_depth_at(color, depth, source_x, target_y, p);
            float projected_x = projected_target_x(source_x, value, eye_sign, p);
            float class_weight = 1.0f - smoothstep(0.5f * threshold, threshold, best_depth - value);
            float coverage = max(0.0f, 1.0f - abs(projected_x - target_x)) * class_weight;
            if (coverage > 0.0f) {
                foreground += sample_color_depth_class(
                    color, depth, source_x, target_y, best_depth, p
                ) * coverage;
                foreground_coverage += coverage;
            }
        }
    }
    foreground /= max(foreground_coverage, 1.0e-6f);
    float background_confidence = smoothstep(0.5f * threshold, threshold,
        best_depth - min(left_depth, right_depth));
    return mix(foreground, background,
        (1.0f - clamp(foreground_coverage, 0.0f, 1.0f)) * background_confidence);
}

static inline float3 dibr_warp_at(
    texture2d<float, access::sample> color, device float *depth,
    float target_x, float target_y, float eye_sign,
    constant WarpParams& p) {
    float target_depth = dibr_depth_at(color, depth, target_x, target_y, p);
    float target_shift = shift_from_depth(target_depth, p) *
                         effective_layer_shift_scale(target_depth, p);
    float source_x = clamp(target_x + target_shift * eye_sign, 0.0f,
                           float(p.source_width) - 1.0f);
    float source_depth = dibr_depth_at(color, depth, source_x, target_y, p);
    float source_shift = shift_from_depth(source_depth, p) *
                         effective_layer_shift_scale(source_depth, p);
    float refined_x = clamp(target_x + source_shift * eye_sign, 0.0f,
                            float(p.source_width) - 1.0f);
    float source_residual = abs(refined_x - source_x);
    float front_confidence = 0.0f;
    float threshold = max(p.edge_threshold, 1.0e-6f);
    // No depth can occlude a sample already within the visibility threshold of 1.
    if (source_depth < 1.0f - 0.5f * threshold) {
        float front_shift = shift_from_depth(1.0f, p);
        float front_x = clamp(target_x + front_shift * eye_sign, 0.0f,
                              float(p.source_width) - 1.0f);
        float front_depth = p.antialias_strength != 0.0f
            ? dibr_depth_at(color, depth, front_x, target_y, p)
            : conservative_depth_at(depth, front_x, target_y, p);
        front_confidence = smoothstep(0.5f * threshold, threshold, front_depth - source_depth);
    }

    float confidence = max(front_confidence,
        max(smoothstep(0.5f * threshold, threshold, abs(source_depth - target_depth)),
            smoothstep(0.125f, 0.5f, source_residual)));
    float3 linear = sample_color(color, refined_x, target_y, p);
    if (confidence > 0.0f) {
        return mix(linear, dibr_gather_at(color, depth, target_x, target_y, eye_sign, p), confidence);
    }
    return linear;
}

// Avoid expanding the gather, nested occlusion, and per-eye routines at every
// output-mode call site: that made Metal pipeline compilation take minutes.
static __attribute__((noinline)) float3 layered_warp_at(
    texture2d<float, access::sample> color, device float *depth,
    float x, float y, float eye_sign, constant WarpParams& p) {
    return dibr_warp_at(color, depth, x, y, eye_sign, p);
}

static inline float max_shift_magnitude(constant WarpParams& p) {
    float scale = max(p.depth_strength, 0.0f) * p.max_disparity_px * 0.5f;
    return max(max(abs(p.convergence * scale),
                   abs((1.0f - p.convergence) * scale)), 1.0e-6f);
}

static inline float edge_at(device float *depth, float x, float y,
                            constant WarpParams& p) {
    float center_depth = processed_depth_at(depth, x, y, p);
    float depth_edge = abs(processed_depth_at(depth, x + 1.0f, y, p) - center_depth) +
                       abs(processed_depth_at(depth, x, y + 1.0f, p) - center_depth);
    float center_shift = abs(shift_from_depth(center_depth, p));
    float shift_edge = abs(abs(shift_from_depth(processed_depth_at(depth, x + 1.0f, y, p), p)) - center_shift) +
                       abs(abs(shift_from_depth(processed_depth_at(depth, x, y + 1.0f, p), p)) - center_shift);
    return (depth_edge > p.edge_threshold ||
            shift_edge / max_shift_magnitude(p) > 0.05f) ? 1.0f : 0.0f;
}

static inline bool antialias_edge_at(device float *depth, float x, float y,
                                     constant WarpParams& p) {
    float center_depth = processed_depth_at(depth, x, y, p);
    float center_shift = shift_from_depth(center_depth, p);
    float depth_edge = max(
        max(abs(processed_depth_at(depth, x - 1.0f, y, p) - center_depth),
            abs(processed_depth_at(depth, x + 1.0f, y, p) - center_depth)),
        max(abs(processed_depth_at(depth, x, y - 1.0f, p) - center_depth),
            abs(processed_depth_at(depth, x, y + 1.0f, p) - center_depth)));
    float shift_edge = max(
        max(abs(shift_from_depth(processed_depth_at(depth, x - 1.0f, y, p), p) - center_shift),
            abs(shift_from_depth(processed_depth_at(depth, x + 1.0f, y, p), p) - center_shift)),
        max(abs(shift_from_depth(processed_depth_at(depth, x, y - 1.0f, p), p) - center_shift),
            abs(shift_from_depth(processed_depth_at(depth, x, y + 1.0f, p), p) - center_shift)));
    return depth_edge > clamp(p.edge_threshold, 0.01f, 0.025f) ||
           shift_edge / max_shift_magnitude(p) > 0.025f;
}

static __attribute__((noinline)) float occlusion_at(device float *depth, float x, float y,
                                 constant WarpParams& p) {
    if (p.occlusion_enabled == 0) return 0.0f;
    int radius = min(p.edge_dilation, 3);
    float found = 0.0f;
    for (int dy = -3; dy <= 3; ++dy) {
        for (int dx = -3; dx <= 3; ++dx) {
            if (abs(dx) <= radius && abs(dy) <= radius) {
                found = max(found, edge_at(depth, x + float(dx), y + float(dy), p));
            }
        }
    }
    return found;
}

static inline float feathered_mask_at(device float *depth, float x, float y,
                                       constant WarpParams& p) {
    int radius = min(p.mask_feather_radius, 3);
    if (radius <= 0) return occlusion_at(depth, x, y, p);
    float total = 0.0f;
    float count = 0.0f;
    for (int dy = -3; dy <= 3; ++dy) {
        for (int dx = -3; dx <= 3; ++dx) {
            if (abs(dx) <= radius && abs(dy) <= radius) {
                total += occlusion_at(depth, x + float(dx), y + float(dy), p);
                count += 1.0f;
            }
        }
    }
    return total / max(count, 1.0f);
}

static inline float3 box_average(texture2d<float, access::sample> color,
                                  device float *depth, float x, float y,
                                  float eye_sign, constant WarpParams& p) {
    int radius = min(p.fill_radius, 3);
    float3 total = float3(0.0f);
    float count = 0.0f;
    for (int dy = -3; dy <= 3; ++dy) {
        for (int dx = -3; dx <= 3; ++dx) {
            if (abs(dx) <= radius && abs(dy) <= radius) {
                total += layered_warp_at(color, depth, x + float(dx), y + float(dy), eye_sign, p);
                count += 1.0f;
            }
        }
    }
    return total / max(count, 1.0f);
}

static inline float3 fill_eye(texture2d<float, access::sample> color,
                               device float *depth, float x, float y,
                               float eye_sign, float mask,
                               constant WarpParams& p) {
    float3 original = layered_warp_at(color, depth, x, y, eye_sign, p);
    if (p.hole_fill_mode == 2 || mask <= 1.0e-5f || p.fill_strength <= 1.0e-5f) {
        return original;
    }
    float3 blurred = box_average(color, depth, x, y, eye_sign, p);
    if (p.hole_fill_mode == 0) {
        return mix(original, blurred, clamp(mask * p.fill_strength, 0.0f, 1.0f));
    }
    float left_depth = processed_depth_at(depth, x - 1.0f, y, p);
    float right_depth = processed_depth_at(depth, x + 1.0f, y, p);
    float left_shift = shift_from_depth(left_depth, p);
    float right_shift = shift_from_depth(right_depth, p);
    bool reliable = abs(right_depth - left_depth) > p.edge_threshold ||
                    abs(right_shift - left_shift) > 0.05f;
    int direction = right_depth < left_depth ? 1 : -1;
    int radius = clamp(p.fill_radius, 1, 3);
    float3 directional = float3(0.0f);
    for (int step = 1; step <= 3; ++step) {
        if (step <= radius) {
            directional += layered_warp_at(color, depth, x + float(direction * step), y, eye_sign, p);
        }
    }
    directional /= float(radius);
    float3 content_aware = reliable ? directional * 0.75f + blurred * 0.25f : blurred;
    float3 current = original;
    float current_luma = (current.r + current.g + current.b) / 3.0f;
    float3 previous_x = layered_warp_at(color, depth, x - 1.0f, y, eye_sign, p);
    float3 previous_y = layered_warp_at(color, depth, x, y - 1.0f, eye_sign, p);
    float rgb_edge_x = abs(current_luma - (previous_x.r + previous_x.g + previous_x.b) / 3.0f);
    float rgb_edge_y = abs(current_luma - (previous_y.r + previous_y.g + previous_y.b) / 3.0f);
    float protection = clamp((max(rgb_edge_x, rgb_edge_y) - 0.20f) / 0.30f, 0.0f, 1.0f);
    float current_depth = processed_depth_at(depth, x, y, p);
    float depth_protection = clamp((max(abs(current_depth - left_depth),
                                        abs(current_depth - right_depth)) - 0.04f) / 0.12f,
                                   0.0f, 1.0f) * 0.5f;
    protection = max(protection, depth_protection);
    float blend = clamp(mask * p.fill_strength * (1.0f - protection * 0.70f), 0.0f, 1.0f);
    return mix(original, content_aware, blend);
}

static __attribute__((noinline)) float3 eye_pixel(texture2d<float, access::sample> color,
                               device float *depth, float x, float y,
                               float eye_sign, bool screen_edge,
                               constant WarpParams& p) {
    bool fill_enabled = p.hole_fill_mode != 2 && p.fill_strength > 1.0e-5f;
    float mask = (!fill_enabled || screen_edge)
        ? 0.0f : feathered_mask_at(depth, x, y, p);
    float3 center = fill_eye(color, depth, x, y, eye_sign, mask, p);
    if (p.edge_aa_enabled == 0 ||
        (p.output_format != 0u && p.output_format != 1u) ||
        !antialias_edge_at(depth, x, y, p)) {
        return clamp(center, float3(0.0f), float3(1.0f));
    }
    float3 covered = (
        fill_eye(color, depth, x - 0.25f, y - 0.25f, eye_sign, mask, p) +
        fill_eye(color, depth, x + 0.25f, y - 0.25f, eye_sign, mask, p) +
        fill_eye(color, depth, x - 0.25f, y + 0.25f, eye_sign, mask, p) +
        fill_eye(color, depth, x + 0.25f, y + 0.25f, eye_sign, mask, p)
    ) * 0.25f;
    return clamp(covered, float3(0.0f), float3(1.0f));
}

static inline float3 output_eye_pixel(
    texture2d<float, access::sample> color, device float *depth,
    float x, float y, float eye_sign, bool screen_edge,
    constant WarpParams& p) {
    // Half-SBS averages two complete eye pixels with positive weights.
    if (p.output_format != 0u || p.source_width != p.output_width) {
        return eye_pixel(color, depth, x, y, eye_sign, screen_edge, p);
    }
    float base = floor(x);
    float3 left = eye_pixel(color, depth, base, y, eye_sign, screen_edge, p);
    float3 right = eye_pixel(color, depth, base + 1.0f, y, eye_sign, screen_edge, p);
    return (left + right) * 0.5f;
}

static inline float3 output_eye_pixel_vertical(
    texture2d<float, access::sample> color, device float *depth,
    float x, float y, float eye_sign, bool screen_edge,
    uint eye_height, constant WarpParams& p) {
    if (p.output_format != 2u || p.source_height != eye_height * 2u) {
        return eye_pixel(color, depth, x, y, eye_sign, screen_edge, p);
    }
    float base = floor(y);
    float3 ym1 = eye_pixel(color, depth, x, base - 1.0f, eye_sign, screen_edge, p);
    float3 y0 = eye_pixel(color, depth, x, base, eye_sign, screen_edge, p);
    float3 y1 = eye_pixel(color, depth, x, base + 1.0f, eye_sign, screen_edge, p);
    float3 y2 = eye_pixel(color, depth, x, base + 2.0f, eye_sign, screen_edge, p);
    return (-ym1 + 9.0f * y0 + 9.0f * y1 - y2) * 0.0625f;
}

static inline float3 finite_color(float3 value) {
    if (!isfinite(value.r)) value.r = 0.0f;
    if (!isfinite(value.g)) value.g = 0.0f;
    if (!isfinite(value.b)) value.b = 0.0f;
    return clamp(value, float3(0.0f), float3(1.0f));
}

kernel void d2s_warp_pack(
    texture2d<float, access::sample> color [[texture(0)]],
    device float *depth [[buffer(0)]],
    device uchar *output [[buffer(1)]],
    constant WarpParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    uint total = p.output_width * p.output_height;
    if (gid >= total) return;
    uint x = gid % p.output_width;
    uint y = gid / p.output_width;
    float3 pixel = float3(0.0f);
    if (p.output_format == 5u) {
        float value = processed_depth_at(depth, float(x), float(y), p);
        pixel = float3(isfinite(value) ? value : 1.0f);
    } else if (p.output_format == 4u) {
        uint edge = min(uint(max(p.screen_edge_suppression, 0)),
                        min(p.source_width, p.source_height));
        bool screen_edge = edge > 0u &&
            (x < edge || y < edge || x >= p.source_width - edge ||
             y >= p.source_height - edge);
        pixel = eye_pixel(color, depth, float(x), float(y), 1.0f, screen_edge, p);
    } else if (p.output_format == 6u) {
        uint edge = min(uint(max(p.screen_edge_suppression, 0)),
                        min(p.source_width, p.source_height));
        bool screen_edge = edge > 0u &&
            (x < edge || y < edge || x >= p.source_width - edge ||
             y >= p.source_height - edge);
        float3 left_pixel = eye_pixel(color, depth, float(x), float(y), 1.0f,
                                      screen_edge, p);
        float3 right_pixel = eye_pixel(color, depth, float(x), float(y),
                                       p.symmetric != 0 ? -1.0f : -0.9f,
                                       screen_edge, p);
        if (p.anaglyph_method == 1) {
            pixel = float3(right_pixel.r, left_pixel.g, right_pixel.b);
        } else if (p.anaglyph_method == 2) {
            pixel = float3(left_pixel.r, left_pixel.g, right_pixel.b);
        } else if (p.anaglyph_method == 3) {
            float left_gray = (left_pixel.r + left_pixel.g + left_pixel.b) / 3.0f;
            float right_gray = (right_pixel.r + right_pixel.g + right_pixel.b) / 3.0f;
            pixel = float3(left_gray, right_gray, right_gray);
        } else {
            pixel = float3(left_pixel.r, right_pixel.g, right_pixel.b);
        }
    } else if (p.output_format == 7u || p.output_format == 8u) {
        uint edge = min(uint(max(p.screen_edge_suppression, 0)),
                        min(p.source_width, p.source_height));
        bool screen_edge = edge > 0u &&
            (x < edge || y < edge || x >= p.source_width - edge ||
             y >= p.source_height - edge);
        // Match CUDA Triton: Interleaved is row-interleaved, Leia is
        // column-interleaved.
        bool left = p.output_format == 7u ? (y % 2u == 0u) : (x % 2u == 0u);
        pixel = eye_pixel(color, depth, float(x), float(y),
                          left ? 1.0f : (p.symmetric != 0 ? -1.0f : -0.9f),
                          screen_edge, p);
    } else if (p.output_format == 2u || p.output_format == 3u) {
        uint eye_height = p.output_height / 2u;
        bool left = y < eye_height;
        uint eye_y = left ? y : y - eye_height;
        float source_x = float(x);
        float source_y = p.output_format == 2u
            ? (float(eye_y) + 0.5f) * (float(p.source_height) / float(eye_height)) - 0.5f
            : float(eye_y);
        uint actual_eye_height = left ? eye_height : p.output_height - eye_height;
        uint edge = min(uint(max(p.screen_edge_suppression, 0)),
                        min(p.output_width, actual_eye_height));
        bool screen_edge = edge > 0u &&
            (x < edge || eye_y < edge || x >= p.output_width - edge ||
             eye_y >= actual_eye_height - edge);
        pixel = output_eye_pixel_vertical(
            color, depth, source_x, source_y,
            left ? 1.0f : (p.symmetric != 0 ? -1.0f : -0.9f),
            screen_edge, actual_eye_height, p);
    } else {
        uint eye_width = p.output_width / 2u;
        bool left = x < eye_width;
        uint eye_x = left ? x : x - eye_width;
        float source_x = (float(eye_x) + 0.5f) *
                             (float(p.source_width) / float(eye_width)) - 0.5f;
        float source_y = float(y);
        uint display_eye_width = eye_width;
        uint edge = min(uint(max(p.screen_edge_suppression, 0)),
                        min(display_eye_width, p.output_height));
        uint horizontal_edge = edge;
        bool screen_edge = edge > 0u &&
            (eye_x < horizontal_edge || y < edge ||
             eye_x >= eye_width - horizontal_edge || y >= p.output_height - edge);
        pixel = output_eye_pixel(
            color, depth, source_x, source_y,
            left ? 1.0f : (p.symmetric != 0 ? -1.0f : -0.9f),
            screen_edge, p);
    }
    pixel = finite_color(pixel);
    uint offset = gid * p.output_channels;
    output[offset + 0u] = uchar(clamp(pixel.r * 255.0f + 0.5f, 0.0f, 255.0f));
    output[offset + 1u] = uchar(clamp(pixel.g * 255.0f + 0.5f, 0.0f, 255.0f));
    output[offset + 2u] = uchar(clamp(pixel.b * 255.0f + 0.5f, 0.0f, 255.0f));
    if (p.output_channels == 4u) output[offset + 3u] = 255u;
}


)D2S";

// Optional model-grid separable smoothing. Keeping this in a separate
// library lets the existing CoreML inference and warp pipelines initialize
// even if a device cannot compile this optional filter.
static const char *D2SDepthFilterMetalSource = R"D2S_FILTER(
#include <metal_stdlib>
using namespace metal;

struct DepthFilterParams {
    uint depth_width;
    uint depth_height;
    float strength;
    float sigma_color;
    float max_keep;
    float depth_pop;
    float edge_threshold;
    int axis;
};

static inline float d2s_filter_depth_pop(float value, float pop) {
    if (abs(pop) < 1.0e-6f) return clamp(value, 0.0f, 1.0f);
    float centered = value - 0.5f;
    if (pop < 0.0f) {
        return clamp(0.5f + centered * (1.0f - clamp(-pop, 0.0f, 1.0f)),
                     0.0f, 1.0f);
    }
    float exponent = 1.0f / (1.0f + pop);
    float magnitude = pow(abs(centered), exponent);
    return clamp(0.5f + (centered < 0.0f ? -magnitude : magnitude), 0.0f, 1.0f);
}

kernel void d2s_filter_depth_guided(
    device const float *source_depth [[buffer(0)]],
    device float *filtered_depth [[buffer(1)]],
    constant DepthFilterParams& p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.depth_width || gid.y >= p.depth_height) return;
    uint output_index = gid.y * p.depth_width + gid.x;
    float center = d2s_filter_depth_pop(source_depth[output_index], p.depth_pop);
    int kernel_size = int(3.0f * p.strength) | 1;
    int radius = min(max((kernel_size - 1) / 2, 0), 6);
    if (radius <= 0) {
        filtered_depth[output_index] = center;
        return;
    }

    float sigma = max(0.5f * p.strength, 1.0e-4f);
    float inverse_two_sigma_sq = 0.5f / (sigma * sigma);
    float smooth_sum = 0.0f;
    float weight_sum = 0.0f;
    for (int offset = -6; offset <= 6; ++offset) {
        if (abs(offset) > radius) continue;
        float distance = float(offset);
        float weight = exp(-(distance * distance) * inverse_two_sigma_sq);
        int sx = int(gid.x) + (p.axis == 0 ? offset : 0);
        int sy = int(gid.y) + (p.axis == 1 ? offset : 0);
        sx = clamp(sx, 0, int(p.depth_width) - 1);
        sy = clamp(sy, 0, int(p.depth_height) - 1);
        float sample_depth = d2s_filter_depth_pop(
            source_depth[uint(sy) * p.depth_width + uint(sx)], p.depth_pop);
        smooth_sum += sample_depth * weight;
        weight_sum += weight;
    }
    filtered_depth[output_index] = clamp(smooth_sum / max(weight_sum, 1.0e-6f), 0.0f, 1.0f);
}

struct DepthUpsampleParams {
    uint source_width;
    uint source_height;
    uint depth_width;
    uint depth_height;
    float edge_threshold;
};

static inline float3 d2s_upsample_color(
    texture2d<float, access::sample> source_rgb, float x, float y,
    constant DepthUpsampleParams& p) {
    constexpr sampler s(address::clamp_to_edge, filter::linear);
    float2 uv = (float2(x, y) + 0.5f) /
                float2(p.depth_width, p.depth_height);
    return source_rgb.sample(s, uv).rgb;
}

static inline float d2s_upsample_curvature(
    device const float* depth, int x, int y, uint height, uint width) {
    float curvatureX = 0.0f, curvatureY = 0.0f;
    if (width >= 3u) {
        uint mx = uint(clamp(x, 1, int(width) - 2));
        uint row = uint(y) * width;
        curvatureX = abs(depth[row + mx - 1u] - 2.0f * depth[row + mx] + depth[row + mx + 1u]);
    }
    if (height >= 3u) {
        uint my = uint(clamp(y, 1, int(height) - 2));
        uint column = uint(x);
        curvatureY = abs(depth[column + (my - 1u) * width] -
                         2.0f * depth[column + my * width] + depth[column + (my + 1u) * width]);
    }
    return max(curvatureX, curvatureY);
}

kernel void d2s_upsample_depth_joint_bilateral(
    texture2d<float, access::sample> source_rgb [[texture(0)]],
    device const float *source_depth [[buffer(0)]],
    device float *upsampled_depth [[buffer(1)]],
    constant DepthUpsampleParams& p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.source_width || gid.y >= p.source_height) return;
    constexpr sampler s(address::clamp_to_edge, filter::linear);
    float dx = clamp((float(gid.x) + 0.5f) * float(p.depth_width) /
                     float(p.source_width) - 0.5f,
                     0.0f, float(p.depth_width) - 1.0f);
    float dy = clamp((float(gid.y) + 0.5f) * float(p.depth_height) /
                     float(p.source_height) - 0.5f,
                     0.0f, float(p.depth_height) - 1.0f);
    int x0 = int(floor(dx));
    int y0 = int(floor(dy));
    int x1 = min(x0 + 1, int(p.depth_width) - 1);
    int y1 = min(y0 + 1, int(p.depth_height) - 1);
    float fx = dx - float(x0);
    float fy = dy - float(y0);
    uint i00 = uint(y0) * p.depth_width + uint(x0);
    uint i10 = uint(y0) * p.depth_width + uint(x1);
    uint i01 = uint(y1) * p.depth_width + uint(x0);
    uint i11 = uint(y1) * p.depth_width + uint(x1);
    float d00 = source_depth[i00];
    float d10 = source_depth[i10];
    float d01 = source_depth[i01];
    float d11 = source_depth[i11];
    float s00 = (1.0f - fy) * (1.0f - fx);
    float s10 = (1.0f - fy) * fx;
    float s01 = fy * (1.0f - fx);
    float s11 = fy * fx;
    float bilinear = s00 * d00 + s10 * d10 + s01 * d01 + s11 * d11;

    float2 rgb_uv = (float2(gid) + 0.5f) /
                    float2(p.source_width, p.source_height);
    float edgeSignal =
        s00 * d2s_upsample_curvature(source_depth, x0, y0, p.depth_height, p.depth_width) +
        s10 * d2s_upsample_curvature(source_depth, x1, y0, p.depth_height, p.depth_width) +
        s01 * d2s_upsample_curvature(source_depth, x0, y1, p.depth_height, p.depth_width) +
        s11 * d2s_upsample_curvature(source_depth, x1, y1, p.depth_height, p.depth_width);
    float edgeWeight = smoothstep(max(p.edge_threshold, 1.0e-6f) * 0.25f,
                                  max(p.edge_threshold, 1.0e-6f) * 0.5f, edgeSignal);
    if (edgeWeight <= 0.0f) {
        upsampled_depth[gid.y * p.source_width + gid.x] = bilinear;
        return;
    }
    float3 target = source_rgb.sample(s, rgb_uv).rgb;
    int cx = int(floor(dx + 0.5f)), cy = int(floor(dy + 0.5f));
    float spatial_total = 0.0f, depth_sum = 0.0f, depth_square_sum = 0.0f;
    float3 color_sum = float3(0.0f);
    float color_square_sum = 0.0f, total_weight = 0.0f, guided_sum = 0.0f;
    float max_log_weight = -1.0e30f;
    float matching_distance_sum = 0.0f;
    float low_count = 0.0f, high_count = 0.0f, low_delta_sum = 0.0f, high_delta_sum = 0.0f;
    float low_total = 0.0f, high_total = 0.0f, low_sum = 0.0f, high_sum = 0.0f;
    float threshold = max(p.edge_threshold, 1.0e-6f);
    float sigma = 0.04f;
    for (int oy = -1; oy <= 1; ++oy) {
        for (int ox = -1; ox <= 1; ++ox) {
            uint sx = uint(clamp(cx + ox, 0, int(p.depth_width) - 1));
            uint sy = uint(clamp(cy + oy, 0, int(p.depth_height) - 1));
            float spatial = (1.0f - smoothstep(0.5f, 1.5f, abs(dx - float(cx + ox)))) *
                            (1.0f - smoothstep(0.5f, 1.5f, abs(dy - float(cy + oy))));
            uint sample_index = sy * p.depth_width + sx;
            float sample = source_depth[sample_index];
            float3 color = d2s_upsample_color(source_rgb, float(sx), float(sy), p);
            float3 difference = target - color;
            float distance = dot(difference, difference) / 3.0f;
            float log_weight = spatial > 0.0f
                ? log(max(spatial, 1.0e-30f)) - distance / (2.0f * sigma * sigma) : -1.0e30f;
            float new_max = max(max_log_weight, log_weight);
            float rescale = exp(max_log_weight - new_max);
            float weight = spatial > 0.0f ? exp(log_weight - new_max) : 0.0f;
            total_weight = total_weight * rescale + weight;
            guided_sum = guided_sum * rescale + weight * sample;
            matching_distance_sum = matching_distance_sum * rescale + weight * distance;
            max_log_weight = new_max;
            float delta = dot(abs(difference), float3(1.0f / 3.0f));
            float low = 1.0f - smoothstep(-0.5f * threshold, 0.5f * threshold, sample - bilinear);
            float high = 1.0f - low;
            low_count += spatial * low;
            high_count += spatial * high;
            low_delta_sum += spatial * low * delta;
            high_delta_sum += spatial * high * delta;
            low_total = low_total * rescale + weight * low;
            high_total = high_total * rescale + weight * high;
            low_sum = low_sum * rescale + weight * low * sample;
            high_sum = high_sum * rescale + weight * high * sample;
            spatial_total += spatial;
            depth_sum += spatial * sample;
            depth_square_sum += spatial * sample * sample;
            color_sum += spatial * color;
            color_square_sum += spatial * dot(color, color) / 3.0f;
        }
    }
    float denominator = max(spatial_total, 1.0e-8f);
    float mean_depth = depth_sum / denominator;
    float3 mean_color = color_sum / denominator;
    float depth_variance = max(depth_square_sum / denominator - mean_depth * mean_depth, 0.0f);
    float color_variance = max(color_square_sum / denominator - dot(mean_color, mean_color) / 3.0f, 0.0f);
    float confidence = smoothstep(threshold * threshold * 0.0625f,
                                  threshold * threshold * 0.25f, depth_variance) *
                       smoothstep(0.000025f, 0.0004f, color_variance);
    float guided = total_weight > 1.0e-8f ? guided_sum / total_weight : bilinear;
    float low_delta = low_delta_sum / max(low_count, 1.0e-8f);
    float high_delta = high_delta_sum / max(high_count, 1.0e-8f);
    float prefer_high = smoothstep(-0.02f, 0.02f, low_delta - high_delta);
    float selected = mix(low_sum / max(low_total, 1.0e-8f),
                         high_sum / max(high_total, 1.0e-8f), prefer_high);
    float support = smoothstep(0.001f, 0.1f, min(low_count, high_count));
    float class_confidence = smoothstep(0.0025f, 0.03f, abs(low_delta - high_delta)) * support;
    class_confidence *= 1.0f - smoothstep(0.000025f, 0.0004f,
                                        matching_distance_sum / max(total_weight, 1.0e-8f));
    guided = mix(guided, selected, class_confidence);
    upsampled_depth[gid.y * p.source_width + gid.x] =
        clamp(mix(bilinear, guided, edgeWeight * confidence), 0.0f, 1.0f);
}

struct DepthGaussianParams {
    uint depth_width;
    uint depth_height;
    float depth_pop;
};

static inline float d2s_gaussian_depth_pop(float value, float pop) {
    if (abs(pop) < 1.0e-6f) return clamp(value, 0.0f, 1.0f);
    float centered = value - 0.5f;
    if (pop < 0.0f) {
        return clamp(0.5f + centered * (1.0f - clamp(-pop, 0.0f, 1.0f)),
                     0.0f, 1.0f);
    }
    float magnitude = pow(abs(centered), 1.0f / (1.0f + pop));
    return clamp(0.5f + (centered < 0.0f ? -magnitude : magnitude), 0.0f, 1.0f);
}

kernel void d2s_filter_depth_global_gaussian(
    device const float *source_depth [[buffer(0)]],
    device float *filtered_depth [[buffer(1)]],
    constant DepthGaussianParams& p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.depth_width || gid.y >= p.depth_height) return;
    constexpr float sample_weights[5] = {
        0.15246914f, 0.22184130f, 0.25137912f,
        0.22184130f, 0.15246914f
    };
    float long_edge = float(max(p.depth_width, p.depth_height));
    float spacing = 1.0f + 0.5f * clamp((long_edge - 1920.0f) / 1920.0f, 0.0f, 1.0f);
    float axis_weights[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int offset = -2; offset <= 2; ++offset) {
        float position = float(offset) * spacing;
        int lower = int(floor(position));
        float fraction = position - float(lower);
        float weight = sample_weights[offset + 2];
        axis_weights[lower + 3] += weight * (1.0f - fraction);
        if (fraction > 0.0f) axis_weights[lower + 4] += weight * fraction;
    }
    float sum = 0.0f;
    float weight_sum = 0.0f;
    for (int oy = -3; oy <= 3; ++oy) {
        float wy = axis_weights[oy + 3];
        if (wy <= 0.0f) continue;
        uint sy = uint(clamp(int(gid.y) + oy, 0, int(p.depth_height) - 1));
        for (int ox = -3; ox <= 3; ++ox) {
            float wx = axis_weights[ox + 3];
            if (wx <= 0.0f) continue;
            uint sx = uint(clamp(int(gid.x) + ox, 0, int(p.depth_width) - 1));
            float sample = d2s_gaussian_depth_pop(
                source_depth[sy * p.depth_width + sx], p.depth_pop);
            float weight = wx * wy;
            sum += sample * weight;
            weight_sum += weight;
        }
    }
    filtered_depth[gid.y * p.depth_width + gid.x] = clamp(
        sum / max(weight_sum, 1.0e-8f), 0.0f, 1.0f);
}
)D2S_FILTER";

typedef struct {
    __strong id<MTLBuffer> input_buffer;
    __strong id<MTLBuffer> raw_depth_buffer;
    __strong id<MTLBuffer> normalized_depth_buffer;
    // Retained across slot release so each model-depth allocation is reused.
    __strong id<MTLBuffer> filtered_depth_buffer;
    __strong id<MTLBuffer> filter_intermediate_buffer;
    __strong id<MTLBuffer> status_buffer;
    __strong id<MTLBuffer> packed_buffer;
    __strong id<MTLTexture> color_texture;
    __strong MLMultiArray *input_array;
    __strong MLMultiArray *output_array;
    __strong id<MLFeatureProvider> features;
    __strong MLPredictionOptions *prediction_options;
    __strong id<MLFeatureProvider> prediction;
    CVMetalTextureRef color_cv_texture;
    CVPixelBufferRef pixel_buffer;
    int32_t state;
    uint64_t frame_id;
    int32_t output_is_float16;
    float normalize_lo;
    float normalize_hi;
} D2SSlot;

typedef struct {
    __strong MLModel *model;
    __strong id<MTLDevice> device;
    __strong id<MTLCommandQueue> queue;
    __strong id<MTLCommandQueue> pack_queue;
    __strong id<MTLComputePipelineState> preprocess_pipeline;
    __strong id<MTLComputePipelineState> normalize_half_pipeline;
    __strong id<MTLComputePipelineState> normalize_float_pipeline;
    __strong id<MTLComputePipelineState> depth_filter_pipeline;
    __strong id<MTLComputePipelineState> depth_upsample_pipeline;
    __strong id<MTLComputePipelineState> depth_global_gaussian_pipeline;
    __strong id<MTLComputePipelineState> warp_pipeline;
    __strong id<MTLBuffer> upsampled_depth_buffer;
    __strong id<MTLBuffer> global_gaussian_depth_buffer;
    CVMetalTextureCacheRef texture_cache;
    __strong NSString *input_name;
    __strong NSString *output_name;
    __strong NSArray<NSNumber *> *input_shape;
    __strong NSArray<NSNumber *> *output_shape;
    MLMultiArrayDataType output_type;
    int32_t input_width;
    int32_t input_height;
    int32_t depth_width;
    int32_t depth_height;
    size_t depth_count;
    D2SSlot slots[3];
    pthread_mutex_t mutex;
    int32_t mutex_initialized;
    pthread_mutex_t pack_mutex;
    int32_t pack_mutex_initialized;
    uint64_t depth_range_frame_id;
    float depth_range_lo;
    float depth_range_hi;
    int32_t depth_range_initialized;
    char error[512];
} D2SCoreMLIO;

struct D2SLockGuard {
    pthread_mutex_t *mutex;
    explicit D2SLockGuard(pthread_mutex_t *value) : mutex(value) {
        pthread_mutex_lock(mutex);
    }
    ~D2SLockGuard() {
        pthread_mutex_unlock(mutex);
    }
};

static void d2s_set_error(D2SCoreMLIO *ctx, NSString *message) {
    const char *text = message.UTF8String ?: "unknown native CoreML error";
    snprintf(ctx->error, sizeof(ctx->error), "%s", text);
}

static void d2s_set_create_error(char *buffer, size_t capacity, NSString *message) {
    if (buffer == NULL || capacity == 0) return;
    const char *text = message.UTF8String ?: "unknown native CoreML error";
    snprintf(buffer, capacity, "%s", text);
}

static double d2s_now_ms(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &value) != 0) return 0.0;
    return (double)value.tv_sec * 1000.0 + (double)value.tv_nsec / 1000000.0;
}

static MLComputeUnits d2s_compute_units(int32_t value) {
    switch (value) {
        case 0: return MLComputeUnitsCPUOnly;
        case 1: return MLComputeUnitsCPUAndGPU;
        case 3: return MLComputeUnitsCPUAndNeuralEngine;
        default: return MLComputeUnitsAll;
    }
}

static size_t d2s_shape_count(NSArray<NSNumber *> *shape) {
    size_t count = 1;
    for (NSNumber *value in shape) count *= (size_t)value.unsignedIntegerValue;
    return count;
}

static void d2s_release_slot(D2SSlot *slot) {
    // Core ML may reset its execution stream asynchronously after prediction
    // returns. Drop the bound objects before their Metal storage, and only
    // after the viewer has finished consuming this slot.
    slot->prediction = nil;
    slot->prediction_options = nil;
    slot->features = nil;
    slot->output_array = nil;
    slot->input_array = nil;
    // Core ML may finish releasing a backing array asynchronously after the
    // prediction call returns. Do not reuse these buffers across predictions.
    slot->input_buffer = nil;
    slot->raw_depth_buffer = nil;
    slot->normalized_depth_buffer = nil;
    slot->status_buffer = nil;
    slot->color_texture = nil;
    if (slot->color_cv_texture != NULL) {
        CFRelease(slot->color_cv_texture);
        slot->color_cv_texture = NULL;
    }
    if (slot->pixel_buffer != NULL) {
        CVPixelBufferRelease(slot->pixel_buffer);
        slot->pixel_buffer = NULL;
    }
    slot->state = 0;
    slot->frame_id = 0;
}

static void d2s_destroy_context(D2SCoreMLIO *ctx) {
    if (ctx == NULL) return;
    if (ctx->pack_mutex_initialized) pthread_mutex_lock(&ctx->pack_mutex);
    if (ctx->mutex_initialized) pthread_mutex_lock(&ctx->mutex);
    for (int i = 0; i < 3; ++i) {
        d2s_release_slot(&ctx->slots[i]);
        ctx->slots[i].packed_buffer = nil;
        ctx->slots[i].filtered_depth_buffer = nil;
        ctx->slots[i].filter_intermediate_buffer = nil;
    }
    if (ctx->texture_cache != NULL) {
        CFRelease(ctx->texture_cache);
        ctx->texture_cache = NULL;
    }
    ctx->model = nil;
    ctx->device = nil;
    ctx->queue = nil;
    ctx->pack_queue = nil;
    ctx->preprocess_pipeline = nil;
    ctx->normalize_half_pipeline = nil;
    ctx->normalize_float_pipeline = nil;
    ctx->depth_filter_pipeline = nil;
    ctx->depth_upsample_pipeline = nil;
    ctx->depth_global_gaussian_pipeline = nil;
    ctx->upsampled_depth_buffer = nil;
    ctx->global_gaussian_depth_buffer = nil;
    ctx->warp_pipeline = nil;
    ctx->input_name = nil;
    ctx->output_name = nil;
    ctx->input_shape = nil;
    ctx->output_shape = nil;
    if (ctx->mutex_initialized) {
        pthread_mutex_unlock(&ctx->mutex);
        pthread_mutex_destroy(&ctx->mutex);
        ctx->mutex_initialized = 0;
    }
    if (ctx->pack_mutex_initialized) {
        pthread_mutex_unlock(&ctx->pack_mutex);
        pthread_mutex_destroy(&ctx->pack_mutex);
        ctx->pack_mutex_initialized = 0;
    }
    free(ctx);
}

static BOOL d2s_compile_pipelines(D2SCoreMLIO *ctx, NSError **error) {
    id<MTLLibrary> library = [ctx->device newLibraryWithSource:
        [NSString stringWithUTF8String:D2SMetalSource] options:nil error:error];
    if (library == nil) return NO;
    id<MTLFunction> preprocess = [library newFunctionWithName:@"d2s_preprocess"];
    id<MTLFunction> norm_half = [library newFunctionWithName:@"d2s_normalize_half"];
    id<MTLFunction> norm_float = [library newFunctionWithName:@"d2s_normalize_float"];
    id<MTLFunction> warp = [library newFunctionWithName:@"d2s_warp_pack"];
    if (!preprocess || !norm_half || !norm_float || !warp) return NO;
    ctx->preprocess_pipeline = [ctx->device newComputePipelineStateWithFunction:preprocess error:error];
    ctx->normalize_half_pipeline = [ctx->device newComputePipelineStateWithFunction:norm_half error:error];
    ctx->normalize_float_pipeline = [ctx->device newComputePipelineStateWithFunction:norm_float error:error];
    ctx->warp_pipeline = [ctx->device newComputePipelineStateWithFunction:warp error:error];
    if (!(ctx->preprocess_pipeline && ctx->normalize_half_pipeline &&
           ctx->normalize_float_pipeline && ctx->warp_pipeline)) return NO;

    NSError *filter_error = nil;
    id<MTLLibrary> filter_library = [ctx->device newLibraryWithSource:
        [NSString stringWithUTF8String:D2SDepthFilterMetalSource]
        options:nil error:&filter_error];
    if (filter_library != nil) {
        id<MTLFunction> filter_function =
            [filter_library newFunctionWithName:@"d2s_filter_depth_guided"];
        id<MTLFunction> upsample_function =
            [filter_library newFunctionWithName:@"d2s_upsample_depth_joint_bilateral"];
        id<MTLFunction> global_gaussian_function =
            [filter_library newFunctionWithName:@"d2s_filter_depth_global_gaussian"];
        if (filter_function != nil) {
            ctx->depth_filter_pipeline = [ctx->device
                newComputePipelineStateWithFunction:filter_function error:&filter_error];
        }
        if (upsample_function != nil) {
            ctx->depth_upsample_pipeline = [ctx->device
                newComputePipelineStateWithFunction:upsample_function error:&filter_error];
        }
        if (global_gaussian_function != nil) {
            ctx->depth_global_gaussian_pipeline = [ctx->device
                newComputePipelineStateWithFunction:global_gaussian_function
                error:&filter_error];
        }
    }
    if (ctx->depth_global_gaussian_pipeline == nil) {
        if (error != NULL) *error = filter_error;
        return NO;
    }
    // Guided depth filtering is optional; global Gaussian is required whenever
    // the shared display-AA switch is enabled.
    return YES;
}

static BOOL d2s_get_model_contract(D2SCoreMLIO *ctx, NSError **error) {
    MLModelDescription *description = ctx->model.modelDescription;
    NSDictionary<NSString *, MLFeatureDescription *> *inputs = description.inputDescriptionsByName;
    NSDictionary<NSString *, MLFeatureDescription *> *outputs = description.outputDescriptionsByName;
    if (inputs.count != 1 || outputs.count != 1) {
        d2s_set_error(ctx, @"native CoreML bridge requires exactly one input and one output");
        return NO;
    }
    ctx->input_name = inputs.allKeys.firstObject;
    ctx->output_name = outputs.allKeys.firstObject;
    MLFeatureDescription *input = inputs[ctx->input_name];
    MLFeatureDescription *output = outputs[ctx->output_name];
    if (input.type != MLFeatureTypeMultiArray || output.type != MLFeatureTypeMultiArray ||
        input.multiArrayConstraint.shape.count != 4 ||
        output.multiArrayConstraint.shape.count < 2) {
        d2s_set_error(ctx, @"native CoreML bridge requires array input/output features");
        return NO;
    }
    ctx->input_shape = input.multiArrayConstraint.shape;
    ctx->output_shape = output.multiArrayConstraint.shape;
    if (ctx->input_shape[0].intValue != 1 || ctx->input_shape[1].intValue != 3 ||
        ctx->input_shape[2].intValue != ctx->input_height ||
        ctx->input_shape[3].intValue != ctx->input_width) {
        d2s_set_error(ctx, @"CoreML input shape does not match the requested model resolution");
        return NO;
    }
    ctx->depth_height = ctx->output_shape[ctx->output_shape.count - 2].intValue;
    ctx->depth_width = ctx->output_shape.lastObject.intValue;
    ctx->depth_count = d2s_shape_count(ctx->output_shape);
    if (ctx->depth_count != (size_t)ctx->depth_height * (size_t)ctx->depth_width &&
        ctx->output_shape.count != 3) {
        d2s_set_error(ctx, @"unsupported CoreML depth output shape");
        return NO;
    }
    ctx->output_type = output.multiArrayConstraint.dataType;
    if (ctx->output_type != MLMultiArrayDataTypeFloat16 &&
        ctx->output_type != MLMultiArrayDataTypeFloat32) {
        d2s_set_error(ctx, @"native CoreML bridge supports Float16 or Float32 depth output only");
        return NO;
    }
    return YES;
}

void *d2s_coreml_io_create(const char *model_path, int32_t input_width,
                           int32_t input_height, int32_t compute_units,
                           char *error_buffer, size_t error_capacity) {
    @autoreleasepool {
        if (model_path == NULL || input_width <= 0 || input_height <= 0) {
            d2s_set_create_error(error_buffer, error_capacity, @"invalid native CoreML bridge arguments");
            return NULL;
        }
        D2SCoreMLIO *ctx = (D2SCoreMLIO *)calloc(1, sizeof(D2SCoreMLIO));
        if (ctx == NULL) {
            d2s_set_create_error(error_buffer, error_capacity, @"native CoreML bridge allocation failed");
            return NULL;
        }
        if (pthread_mutex_init(&ctx->mutex, NULL) != 0) {
            d2s_set_create_error(error_buffer, error_capacity, @"native CoreML bridge lock initialization failed");
            free(ctx);
            return NULL;
        }
        ctx->mutex_initialized = 1;
        if (pthread_mutex_init(&ctx->pack_mutex, NULL) != 0) {
            d2s_set_create_error(error_buffer, error_capacity,
                                 @"native CoreML pack lock initialization failed");
            d2s_destroy_context(ctx);
            return NULL;
        }
        ctx->pack_mutex_initialized = 1;
        ctx->input_width = input_width;
        ctx->input_height = input_height;
        ctx->device = MTLCreateSystemDefaultDevice();
        if (ctx->device == nil) {
            d2s_set_create_error(error_buffer, error_capacity, @"Metal device unavailable");
            d2s_destroy_context(ctx);
            return NULL;
        }
        ctx->queue = [ctx->device newCommandQueue];
        if (ctx->queue == nil) {
            d2s_set_create_error(error_buffer, error_capacity, @"Metal command queue unavailable");
            d2s_destroy_context(ctx);
            return NULL;
        }
        ctx->pack_queue = [ctx->device newCommandQueue];
        if (ctx->pack_queue == nil) {
            d2s_set_create_error(error_buffer, error_capacity, @"Metal pack command queue unavailable");
            d2s_destroy_context(ctx);
            return NULL;
        }
        NSError *error = nil;
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:model_path]];
        MLModelConfiguration *configuration = [MLModelConfiguration new];
        configuration.computeUnits = d2s_compute_units(compute_units);
        NSURL *load_url = url;
        if ([url.pathExtension.lowercaseString isEqualToString:@"mlpackage"]) {
            load_url = [MLModel compileModelAtURL:url error:&error];
        }
        if (load_url != nil) {
            ctx->model = [MLModel modelWithContentsOfURL:load_url
                                             configuration:configuration
                                                    error:&error];
        }
        if (ctx->model == nil) {
            d2s_set_create_error(error_buffer, error_capacity, error.localizedDescription ?: @"CoreML model load failed");
            d2s_destroy_context(ctx);
            return NULL;
        }
        if (!d2s_get_model_contract(ctx, &error)) {
            d2s_set_create_error(error_buffer, error_capacity, [NSString stringWithUTF8String:ctx->error]);
            d2s_destroy_context(ctx);
            return NULL;
        }
        if (!d2s_compile_pipelines(ctx, &error)) {
            d2s_set_create_error(error_buffer, error_capacity, error.localizedDescription ?: @"Metal pipeline compilation failed");
            d2s_destroy_context(ctx);
            return NULL;
        }
        CVReturn status = CVMetalTextureCacheCreate(kCFAllocatorDefault, NULL,
                                                     ctx->device, NULL,
                                                     &ctx->texture_cache);
        if (status != kCVReturnSuccess || ctx->texture_cache == NULL) {
            d2s_set_create_error(error_buffer, error_capacity, @"CVMetalTextureCache creation failed");
            d2s_destroy_context(ctx);
            return NULL;
        }
        for (int i = 0; i < 3; ++i) ctx->slots[i].state = 0;
        return ctx;
    }
}

static int32_t d2s_find_slot(D2SCoreMLIO *ctx) {
    for (int32_t i = 0; i < 3; ++i) {
        if (ctx->slots[i].state == 0) return i;
    }
    return -1;
}

static BOOL d2s_dispatch_preprocess(D2SCoreMLIO *ctx, D2SSlot *slot,
                                    uint32_t source_width, uint32_t source_height) {
    id<MTLCommandBuffer> command = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:ctx->preprocess_pipeline];
    D2SPreprocessParams params = {
        source_width, source_height,
        (uint32_t)ctx->input_width, (uint32_t)ctx->input_height,
        0.485f, 0.456f, 0.406f,
        0.229f, 0.224f, 0.225f,
    };
    [encoder setTexture:slot->color_texture atIndex:0];
    [encoder setBuffer:slot->input_buffer offset:0 atIndex:0];
    [encoder setBytes:&params length:sizeof(params) atIndex:1];
    NSUInteger w = 8;
    NSUInteger h = 8;
    MTLSize grid = MTLSizeMake(ctx->input_width, ctx->input_height, 1);
    MTLSize groups = MTLSizeMake(
        (grid.width + w - 1) / w, (grid.height + h - 1) / h, 1);
    [encoder dispatchThreadgroups:groups threadsPerThreadgroup:MTLSizeMake(w, h, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    return command.status == MTLCommandBufferStatusCompleted;
}

static void d2s_encode_normalize(D2SCoreMLIO *ctx, D2SSlot *slot,
                                 id<MTLComputeCommandEncoder> encoder,
                                 float lo, float hi) {
    uint32_t count = (uint32_t)ctx->depth_count;
    D2SNormalizeParams params = {count, lo, hi};
    *(uint32_t *)slot->status_buffer.contents = 0;
    id<MTLComputePipelineState> pipeline = slot->output_is_float16
        ? ctx->normalize_half_pipeline : ctx->normalize_float_pipeline;
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:slot->raw_depth_buffer offset:0 atIndex:0];
    [encoder setBuffer:slot->normalized_depth_buffer offset:0 atIndex:1];
    [encoder setBuffer:slot->status_buffer offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];
    NSUInteger w = 64;
    MTLSize grid = MTLSizeMake(count, 1, 1);
    [encoder dispatchThreads:grid
      threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
}

static float d2s_half_to_float(uint16_t value) {
    uint32_t sign = ((uint32_t)value & 0x8000u) << 16;
    uint32_t exponent = ((uint32_t)value >> 10) & 0x1fu;
    uint32_t mantissa = (uint32_t)value & 0x3ffu;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) bits = sign;
        else {
            exponent = 1;
            while ((mantissa & 0x400u) == 0) { mantissa <<= 1; --exponent; }
            mantissa &= 0x3ffu;
            bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static int d2s_float_compare(const void *left, const void *right) {
    const float a = *(const float *)left;
    const float b = *(const float *)right;
    return (a > b) - (a < b);
}

static uint32_t d2s_depth_range(D2SCoreMLIO *ctx, D2SSlot *slot,
                                float *lo, float *hi) {
    // Percentile clipping is only used to reject sparse model outliers. A
    // bounded, evenly spaced sample preserves that behavior without sorting
    // the full model output on the runtime thread.
    const size_t sample_capacity = ctx->depth_count < 2048 ? ctx->depth_count : 2048;
    const size_t sample_stride = (ctx->depth_count + sample_capacity - 1) /
                                 sample_capacity;
    float *values = (float *)malloc(sample_capacity * sizeof(float));
    if (values == NULL) {
        *lo = 0.0f;
        *hi = 1.0f;
        return 0;
    }
    uint32_t finite = 0;
    size_t sampled = 0;
    uint32_t nonfinite = 0;
    if (slot->output_is_float16) {
        const uint16_t *source = (const uint16_t *)slot->raw_depth_buffer.contents;
        for (size_t i = 0; i < ctx->depth_count; ++i) {
            float value = d2s_half_to_float(source[i]);
            if (isfinite(value)) {
                ++finite;
                if ((i % sample_stride) == 0 && sampled < sample_capacity) {
                    values[sampled++] = value;
                }
            }
            else ++nonfinite;
        }
    } else {
        const float *source = (const float *)slot->raw_depth_buffer.contents;
        for (size_t i = 0; i < ctx->depth_count; ++i) {
            float value = source[i];
            if (isfinite(value)) {
                ++finite;
                if ((i % sample_stride) == 0 && sampled < sample_capacity) {
                    values[sampled++] = value;
                }
            }
            else ++nonfinite;
        }
    }
    if (finite == 0 || sampled == 0) {
        *lo = 0.0f;
        *hi = 1.0f;
        free(values);
        return nonfinite;
    }
    qsort(values, sampled, sizeof(float), d2s_float_compare);
    size_t lo_index = (size_t)floor(0.02 * (double)(sampled - 1));
    size_t hi_index = (size_t)floor(0.98 * (double)(sampled - 1));
    *lo = values[lo_index];
    *hi = values[hi_index];
    if (!isfinite(*lo) || !isfinite(*hi) || *hi <= *lo) {
        *lo = 0.0f;
        *hi = 1.0f;
    }
    free(values);
    return nonfinite;
}

static int32_t d2s_stabilize_depth_range(D2SCoreMLIO *ctx, uint64_t frame_id,
                                         float *lo, float *hi) {
    const char *disabled = getenv("D2S_DEPTH_RANGE_TEMPORAL");
    if (disabled != NULL && (strcmp(disabled, "0") == 0 ||
                             strcasecmp(disabled, "false") == 0 ||
                             strcasecmp(disabled, "off") == 0)) {
        return 1;
    }
    if (!isfinite(*lo) || !isfinite(*hi) || *hi <= *lo) return 1;

    D2SLockGuard lock(&ctx->mutex);
    if (ctx->depth_range_initialized && frame_id <= ctx->depth_range_frame_id) {
        return 1;
    }
    if (!ctx->depth_range_initialized ||
        frame_id - ctx->depth_range_frame_id > 2) {
        ctx->depth_range_lo = *lo;
        ctx->depth_range_hi = *hi;
        ctx->depth_range_frame_id = frame_id;
        ctx->depth_range_initialized = 1;
        return 1;
    }

    const float previous_span = fmaxf(ctx->depth_range_hi - ctx->depth_range_lo, 1e-6f);
    const float current_span = fmaxf(*hi - *lo, 1e-6f);
    const float scale = fmaxf(previous_span, current_span);
    const float range_jump = fmaxf(fabsf(*lo - ctx->depth_range_lo),
                                   fabsf(*hi - ctx->depth_range_hi));
    if (range_jump > 0.50f * scale) {
        // Large model-range discontinuity is treated like a scene cut: reset
        // immediately rather than carrying stale normalization into the new scene.
        ctx->depth_range_lo = *lo;
        ctx->depth_range_hi = *hi;
    } else {
        // Light one-frame damping suppresses quantile shimmer without adding
        // image history or delaying the RGB frame.
        ctx->depth_range_lo = ctx->depth_range_lo * 0.35f + *lo * 0.65f;
        ctx->depth_range_hi = ctx->depth_range_hi * 0.35f + *hi * 0.65f;
    }
    ctx->depth_range_frame_id = frame_id;
    *lo = ctx->depth_range_lo;
    *hi = ctx->depth_range_hi;
    return range_jump > 0.50f * scale;
}

int32_t d2s_coreml_io_predict(void *handle, void *pixel_buffer,
                              uint64_t frame_id, D2SCoreMLIOResult *result) {
    @autoreleasepool {
        D2SCoreMLIO *ctx = (D2SCoreMLIO *)handle;
        if (ctx == NULL || pixel_buffer == NULL || result == NULL) return D2S_COREML_ERROR;
        int32_t slot_index;
        {
            D2SLockGuard lock(&ctx->mutex);
            slot_index = d2s_find_slot(ctx);
            if (slot_index < 0) {
                d2s_set_error(ctx, @"native CoreML resource ring is full");
                return D2S_COREML_BUSY;
            }
            // Reserve before touching Metal/Core ML so another prediction can
            // use a different ring slot while this one is in flight.
            ctx->slots[slot_index].state = 2;
        }
        D2SSlot *slot = &ctx->slots[slot_index];
        CVPixelBufferRef buffer = (CVPixelBufferRef)pixel_buffer;
        size_t width = CVPixelBufferGetWidth(buffer);
        size_t height = CVPixelBufferGetHeight(buffer);
        CVMetalTextureRef cv_texture = NULL;
        CVReturn texture_status = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, ctx->texture_cache, buffer, NULL,
            MTLPixelFormatBGRA8Unorm, width, height, 0, &cv_texture);
        if (texture_status != kCVReturnSuccess || cv_texture == NULL) {
            d2s_set_error(ctx, @"ScreenCaptureKit pixel buffer could not become a Metal texture");
            d2s_release_slot(slot);
            return D2S_COREML_ERROR;
        }
        slot->color_cv_texture = cv_texture;
        slot->color_texture = CVMetalTextureGetTexture(slot->color_cv_texture);
        slot->pixel_buffer = CVPixelBufferRetain(buffer);
        size_t input_bytes =
            (size_t)(3 * ctx->input_width * ctx->input_height * sizeof(float));
        if (slot->input_buffer == nil || slot->input_buffer.length < input_bytes) {
            slot->input_buffer = [ctx->device newBufferWithLength:(NSUInteger)input_bytes
                                                             options:MTLResourceStorageModeShared];
        }
        size_t output_bytes = ctx->depth_count *
            (ctx->output_type == MLMultiArrayDataTypeFloat16 ? sizeof(uint16_t) : sizeof(float));
        if (slot->raw_depth_buffer == nil || slot->raw_depth_buffer.length < output_bytes) {
            slot->raw_depth_buffer = [ctx->device newBufferWithLength:output_bytes
                                                                options:MTLResourceStorageModeShared];
        }
        size_t normalized_bytes = ctx->depth_count * sizeof(float);
        if (slot->normalized_depth_buffer == nil ||
            slot->normalized_depth_buffer.length < normalized_bytes) {
            slot->normalized_depth_buffer = [ctx->device newBufferWithLength:normalized_bytes
                                                                       options:MTLResourceStorageModeShared];
        }
        if (slot->status_buffer == nil || slot->status_buffer.length < sizeof(uint32_t)) {
            slot->status_buffer = [ctx->device newBufferWithLength:sizeof(uint32_t)
                                                               options:MTLResourceStorageModeShared];
        }
        slot->output_is_float16 = ctx->output_type == MLMultiArrayDataTypeFloat16;
        if (!slot->color_texture || !slot->input_buffer || !slot->raw_depth_buffer ||
            !slot->normalized_depth_buffer || !slot->status_buffer) {
            d2s_set_error(ctx, @"native CoreML Metal buffer allocation failed");
            d2s_release_slot(slot);
            return D2S_COREML_ERROR;
        }
        double preprocess_start = d2s_now_ms();
        if (!d2s_dispatch_preprocess(ctx, slot, (uint32_t)width, (uint32_t)height)) {
            d2s_set_error(ctx, @"Metal CoreML preprocessing command failed");
            d2s_release_slot(slot);
            return D2S_COREML_ERROR;
        }
        double preprocess_ms = d2s_now_ms() - preprocess_start;
        NSArray<NSNumber *> *input_strides = @[
            @(3 * ctx->input_width * ctx->input_height),
            @(ctx->input_width * ctx->input_height),
            @(ctx->input_width), @1];
        NSError *error = nil;
        MLMultiArray *input_array = [[MLMultiArray alloc]
            initWithDataPointer:slot->input_buffer.contents
            shape:ctx->input_shape dataType:MLMultiArrayDataTypeFloat32
            strides:input_strides deallocator:nil error:&error];
        if (input_array == nil) {
            d2s_set_error(ctx, error.localizedDescription ?: @"CoreML input binding failed");
            d2s_release_slot(slot);
            return D2S_COREML_ERROR;
        }
        slot->input_array = input_array;
        NSArray<NSNumber *> *output_strides = @[
            @(ctx->depth_width * ctx->depth_height), @(ctx->depth_width), @1];
        MLMultiArray *output_array = [[MLMultiArray alloc]
            initWithDataPointer:slot->raw_depth_buffer.contents
            shape:ctx->output_shape dataType:ctx->output_type
            strides:output_strides deallocator:nil error:&error];
        if (output_array == nil) {
            d2s_set_error(ctx, error.localizedDescription ?: @"CoreML output binding failed");
            d2s_release_slot(slot);
            return D2S_COREML_ERROR;
        }
        slot->output_array = output_array;
        MLFeatureValue *input_value = [MLFeatureValue featureValueWithMultiArray:input_array];
        MLDictionaryFeatureProvider *features = [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:@{ctx->input_name: input_value} error:&error];
        if (features == nil) {
            d2s_set_error(ctx, error.localizedDescription ?: @"CoreML feature provider creation failed");
            d2s_release_slot(slot);
            return D2S_COREML_ERROR;
        }
        slot->features = features;
        MLPredictionOptions *options = [MLPredictionOptions new];
        options.outputBackings = @{ctx->output_name: output_array};
        slot->prediction_options = options;
        double model_start = d2s_now_ms();
        id<MLFeatureProvider> prediction = [ctx->model predictionFromFeatures:features
                                                                       options:options
                                                                        error:&error];
        double model_ms = d2s_now_ms() - model_start;
        if (prediction == nil) {
            d2s_set_error(ctx, error.localizedDescription ?: @"CoreML prediction failed");
            d2s_release_slot(slot);
            return D2S_COREML_ERROR;
        }
        slot->prediction = prediction;
        MLFeatureValue *output_value = [prediction featureValueForName:ctx->output_name];
        MLMultiArray *returned_array = output_value.multiArrayValue;
        if (returned_array == nil ||
            (returned_array != output_array &&
             returned_array.dataPointer != output_array.dataPointer)) {
            d2s_set_error(ctx, @"CoreML ignored the requested shared output backing");
            d2s_release_slot(slot);
            return D2S_COREML_OUTPUT_BACKING;
        }
        double postprocess_start = d2s_now_ms();
        float lo = 0.0f, hi = 1.0f;
        uint32_t nonfinite = d2s_depth_range(ctx, slot, &lo, &hi);
        const float raw_lo = lo;
        const float raw_hi = hi;
        const int32_t range_history_reset =
            d2s_stabilize_depth_range(ctx, frame_id, &lo, &hi);
        slot->normalize_lo = lo;
        slot->normalize_hi = hi;
        double postprocess_ms = d2s_now_ms() - postprocess_start;
        {
            D2SLockGuard lock(&ctx->mutex);
            slot->state = 1;
            slot->frame_id = frame_id;
        }
        memset(result, 0, sizeof(*result));
        result->slot = slot_index;
        result->source_width = (int32_t)width;
        result->source_height = (int32_t)height;
        result->input_width = ctx->input_width;
        result->input_height = ctx->input_height;
        result->depth_width = ctx->depth_width;
        result->depth_height = ctx->depth_height;
        result->input_shared = 1;
        result->output_backing_used = 1;
        result->output_zero_copy = 0;
        // Normalization is encoded together with the final warp in pack().
        // This keeps the depth publish path from waiting on a separate Metal
        // command and still sanitizes every value before sampling.
        result->finite_depth = (nonfinite == 0);
        result->nonfinite_count = nonfinite;
        result->normalize_lo = lo;
        result->normalize_hi = hi;
        result->raw_normalize_lo = raw_lo;
        result->raw_normalize_hi = raw_hi;
        result->normalization_history_reset = range_history_reset;
        result->preprocess_ms = preprocess_ms;
        result->model_ms = model_ms;
        result->postprocess_ms = postprocess_ms;
        return D2S_COREML_OK;
    }
}

static int32_t d2s_coreml_io_pack_internal(
    void *handle, int32_t slot_index, void *destination,
    void *metal_destination, size_t destination_size,
    int32_t output_width, int32_t output_height, int32_t output_format,
    const D2SCoreMLIOWarpConfig *warp_config, float eye_offset,
    float depth_strength, float convergence, float smooth_texels,
    uint32_t output_channels, D2SCoreMLIOPackResult *result) {
    @autoreleasepool {
        D2SCoreMLIO *ctx = (D2SCoreMLIO *)handle;
        if (result != NULL) memset(result, 0, sizeof(*result));
        if (ctx == NULL || (destination == NULL && metal_destination == NULL) ||
            slot_index < 0 || slot_index >= 3 ||
            output_width <= 0 || output_height <= 0 || output_format < D2S_OUTPUT_HALF_SBS ||
            output_format > D2S_OUTPUT_LEIA ||
            (output_channels != 3u && output_channels != 4u)) {
            return D2S_COREML_ERROR;
        }
        D2SSlot *slot = &ctx->slots[slot_index];
        uint32_t source_width;
        uint32_t source_height;
        BOOL expected_size = NO;
        {
            D2SLockGuard lock(&ctx->mutex);
            if (slot->state != 1 || slot->pixel_buffer == NULL) {
                d2s_set_error(ctx, @"native CoreML pack slot is not ready");
                return D2S_COREML_ERROR;
            }
            source_width = (uint32_t)CVPixelBufferGetWidth(slot->pixel_buffer);
            source_height = (uint32_t)CVPixelBufferGetHeight(slot->pixel_buffer);
            switch (output_format) {
                case D2S_OUTPUT_FULL_SBS:
                    expected_size = (output_width == (int32_t)(source_width * 2u) &&
                                     output_height == (int32_t)source_height);
                    break;
                case D2S_OUTPUT_FULL_TAB:
                    expected_size = (output_width == (int32_t)source_width &&
                                     output_height == (int32_t)(source_height * 2u));
                    break;
                default:
                    expected_size = (output_width == (int32_t)source_width &&
                                     output_height == (int32_t)source_height);
                    break;
            }
            if (
                destination_size < (size_t)output_width * output_height * output_channels ||
                !expected_size ||
                ((output_format == D2S_OUTPUT_HALF_SBS ||
                  output_format == D2S_OUTPUT_FULL_SBS) && output_width % 2 != 0)) {
                d2s_set_error(ctx, @"native CoreML pack slot or destination is invalid");
                return D2S_COREML_ERROR;
            }
        }
        // Output-resolution depth buffers are shared by the pack queue. Keep
        // allocation and command use ordered if more than one caller packs.
        D2SLockGuard pack_lock(&ctx->pack_mutex);
        size_t output_bytes = (size_t)output_width * (size_t)output_height * output_channels;
        id<MTLBuffer> destination_buffer = nil;
        if (metal_destination != NULL) {
            destination_buffer = (__bridge id<MTLBuffer>)metal_destination;
            if (destination_buffer.length < output_bytes ||
                (destination_size > 0 && destination_size < output_bytes)) {
                d2s_set_error(ctx, @"exported Metal destination is smaller than the packed frame");
                return D2S_COREML_ERROR;
            }
            if (result != NULL) result->metal_buffer_alias = 1;
        } else {
            if (slot->packed_buffer == nil || slot->packed_buffer.length < output_bytes) {
                slot->packed_buffer = [ctx->device newBufferWithLength:(NSUInteger)output_bytes
                                                                   options:MTLResourceStorageModeShared];
            }
            destination_buffer = slot->packed_buffer;
        }
        if (destination_buffer == nil) {
            d2s_set_error(ctx, @"Metal native warp output buffer allocation failed");
            return D2S_COREML_ERROR;
        }
        D2SCoreMLIOWarpConfig default_stereo = {
            1.0f, 48.0f, 0.0f, 0.04f, 0.0f,
            0, 0, 1, 2, 0.08f,
            1.0f, 1.0f, 1.0f, 2, 0, 2, 1, 0.0f, 0.0f, 0, 1,
        };
        D2SWarpParams params = {
            source_width,
            source_height,
            (uint32_t)ctx->depth_width,
            (uint32_t)ctx->depth_height,
            (uint32_t)output_width,
            (uint32_t)output_height,
            (uint32_t)output_format,
            output_channels,
            warp_config != NULL ? *warp_config : default_stereo,
        };
        BOOL global_depth_gaussian = params.stereo.edge_aa_enabled != 0;
        BOOL model_depth_filtered = NO;
        // Depth is filtered before the warp; no pixel-space AA follows it.
        params.stereo.edge_aa_enabled = 0;
        id<MTLBuffer> warp_destination = destination_buffer;
        id<MTLCommandBuffer> command = [ctx->pack_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        d2s_encode_normalize(ctx, slot, encoder, slot->normalize_lo, slot->normalize_hi);
        [encoder endEncoding];

        id<MTLBuffer> warp_depth_buffer = slot->normalized_depth_buffer;
        if (params.stereo.antialias_strength > 0.0f &&
            ctx->depth_filter_pipeline != nil) {
            size_t depth_bytes = ctx->depth_count * sizeof(float);
            if (slot->filtered_depth_buffer == nil ||
                slot->filtered_depth_buffer.length < depth_bytes) {
                slot->filtered_depth_buffer = [ctx->device
                    newBufferWithLength:(NSUInteger)depth_bytes
                    options:MTLResourceStorageModePrivate];
            }
            if (slot->filter_intermediate_buffer == nil ||
                slot->filter_intermediate_buffer.length < depth_bytes) {
                slot->filter_intermediate_buffer = [ctx->device
                    newBufferWithLength:(NSUInteger)depth_bytes
                    options:MTLResourceStorageModePrivate];
            }
            if (slot->filtered_depth_buffer != nil &&
                slot->filter_intermediate_buffer != nil) {
                BOOL filter_complete = YES;
                for (int axis = 0; axis < 2; ++axis) {
                    D2SDepthFilterParams filter_params = {
                        (uint32_t)ctx->depth_width,
                        (uint32_t)ctx->depth_height,
                        fminf(4.0f, fmaxf(0.0f, 2.0f * params.stereo.antialias_strength)),
                        0.1f,
                        0.35f,
                        0.0f,
                        params.stereo.edge_threshold,
                        axis,
                    };
                    id<MTLBuffer> filter_source = axis == 0
                        ? slot->normalized_depth_buffer
                        : slot->filtered_depth_buffer;
                    id<MTLBuffer> filter_output = axis == 0
                        ? slot->filtered_depth_buffer
                        : slot->filter_intermediate_buffer;
                    id<MTLComputeCommandEncoder> filter_encoder =
                        [command computeCommandEncoder];
                    if (filter_encoder == nil) {
                        filter_complete = NO;
                        break;
                    }
                    [filter_encoder setComputePipelineState:ctx->depth_filter_pipeline];
                    [filter_encoder setBuffer:filter_source offset:0 atIndex:0];
                    [filter_encoder setBuffer:filter_output offset:0 atIndex:1];
                    [filter_encoder setBytes:&filter_params
                        length:sizeof(filter_params) atIndex:2];
                    MTLSize filter_grid = MTLSizeMake(
                        (NSUInteger)ctx->depth_width,
                        (NSUInteger)ctx->depth_height, 1);
                    [filter_encoder dispatchThreads:filter_grid
                        threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
                    [filter_encoder endEncoding];
                }
                if (filter_complete) {
                    warp_depth_buffer = slot->filter_intermediate_buffer;
                    model_depth_filtered = YES;
                    params.stereo.antialias_strength = -1.0f;
                }
            }
        }
        global_depth_gaussian = global_depth_gaussian && !model_depth_filtered;

        if (global_depth_gaussian) {
            if (ctx->depth_upsample_pipeline == nil ||
                ctx->depth_global_gaussian_pipeline == nil) {
                d2s_set_error(ctx, @"Metal output-resolution depth AA pipeline is unavailable");
                return D2S_COREML_ERROR;
            }
            size_t source_depth_bytes =
                (size_t)source_width * source_height * sizeof(float);
            if (ctx->upsampled_depth_buffer == nil ||
                ctx->upsampled_depth_buffer.length < source_depth_bytes) {
                ctx->upsampled_depth_buffer = [ctx->device
                    newBufferWithLength:(NSUInteger)source_depth_bytes
                    options:MTLResourceStorageModePrivate];
            }
            if (ctx->global_gaussian_depth_buffer == nil ||
                ctx->global_gaussian_depth_buffer.length < source_depth_bytes) {
                ctx->global_gaussian_depth_buffer = [ctx->device
                    newBufferWithLength:(NSUInteger)source_depth_bytes
                    options:MTLResourceStorageModePrivate];
            }
            if (ctx->upsampled_depth_buffer == nil ||
                ctx->global_gaussian_depth_buffer == nil) {
                d2s_set_error(ctx, @"Metal global depth Gaussian buffer allocation failed");
                return D2S_COREML_ERROR;
            }

            if (source_width != (uint32_t)ctx->depth_width ||
                source_height != (uint32_t)ctx->depth_height) {
                D2SDepthUpsampleParams upsample_params = {
                    source_width,
                    source_height,
                    (uint32_t)ctx->depth_width,
                    (uint32_t)ctx->depth_height,
                    params.stereo.edge_threshold,
                };
                id<MTLComputeCommandEncoder> upsample_encoder =
                    [command computeCommandEncoder];
                [upsample_encoder setComputePipelineState:ctx->depth_upsample_pipeline];
                [upsample_encoder setTexture:slot->color_texture atIndex:0];
                [upsample_encoder setBuffer:warp_depth_buffer offset:0 atIndex:0];
                [upsample_encoder setBuffer:ctx->upsampled_depth_buffer offset:0 atIndex:1];
                [upsample_encoder setBytes:&upsample_params
                    length:sizeof(upsample_params) atIndex:2];
                MTLSize upsample_grid = MTLSizeMake(source_width, source_height, 1);
                [upsample_encoder dispatchThreads:upsample_grid
                    threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
                [upsample_encoder endEncoding];
                warp_depth_buffer = ctx->upsampled_depth_buffer;
            }

            D2SDepthGaussianParams gaussian_params = {
                source_width,
                source_height,
                params.stereo.depth_pop,
            };
            id<MTLComputeCommandEncoder> gaussian_encoder =
                [command computeCommandEncoder];
            [gaussian_encoder setComputePipelineState:ctx->depth_global_gaussian_pipeline];
            [gaussian_encoder setBuffer:warp_depth_buffer offset:0 atIndex:0];
            [gaussian_encoder setBuffer:ctx->global_gaussian_depth_buffer offset:0 atIndex:1];
            [gaussian_encoder setBytes:&gaussian_params
                length:sizeof(gaussian_params) atIndex:2];
            MTLSize gaussian_grid = MTLSizeMake(source_width, source_height, 1);
            [gaussian_encoder dispatchThreads:gaussian_grid
                threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
            [gaussian_encoder endEncoding];
            warp_depth_buffer = ctx->global_gaussian_depth_buffer;
            params.depth_width = source_width;
            params.depth_height = source_height;
            params.stereo.depth_pop = 0.0f;
        }

        encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:ctx->warp_pipeline];
        [encoder setTexture:slot->color_texture atIndex:0];
        [encoder setBuffer:warp_depth_buffer offset:0 atIndex:0];
        [encoder setBuffer:warp_destination offset:0 atIndex:1];
        [encoder setBytes:&params length:sizeof(params) atIndex:2];
        NSUInteger threads = 64;
        const char *thread_override = getenv("D2S_METAL_WARP_THREADS");
        if (thread_override != NULL && thread_override[0] != '\0') {
            char *end = NULL;
            long requested = strtol(thread_override, &end, 10);
            if (end != thread_override && end != NULL && *end == '\0' &&
                (requested == 64 || requested == 128 || requested == 256) &&
                (NSUInteger)requested <=
                    ctx->warp_pipeline.maxTotalThreadsPerThreadgroup) {
                threads = (NSUInteger)requested;
            }
        }
        if (result != NULL) result->warp_threads = (int32_t)threads;
        MTLSize grid = MTLSizeMake((NSUInteger)output_width * output_height, 1, 1);
        [encoder dispatchThreads:grid
          threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            d2s_set_error(ctx, @"Metal native warp command failed");
            return D2S_COREML_ERROR;
        }
        if (result != NULL && command.GPUEndTime >= command.GPUStartTime &&
            command.GPUStartTime > 0.0) {
            result->metal_gpu_ms =
                (command.GPUEndTime - command.GPUStartTime) * 1000.0;
        }
        if (metal_destination == NULL) {
            // Keep the established mapped-memory handoff as a safe fallback
            // when the Vulkan allocation could not be exported to Metal.
            double copy_started = d2s_now_ms();
            memcpy(destination, destination_buffer.contents, output_bytes);
            if (result != NULL) {
                result->host_handoff_copy_ms = d2s_now_ms() - copy_started;
            }
        }
        return D2S_COREML_OK;
    }
}

int32_t d2s_coreml_io_pack(void *handle, int32_t slot_index, void *destination,
                           size_t destination_size, int32_t output_width,
                           int32_t output_height, int32_t output_format,
                           const D2SCoreMLIOWarpConfig *warp_config,
                           float eye_offset, float depth_strength,
                           float convergence, float smooth_texels,
                           D2SCoreMLIOPackResult *result) {
    return d2s_coreml_io_pack_internal(
        handle, slot_index, destination, NULL, destination_size,
        output_width, output_height, output_format, warp_config, eye_offset,
        depth_strength, convergence, smooth_texels, 4u, result);
}

int32_t d2s_coreml_io_pack_rgb(void *handle, int32_t slot_index, void *destination,
                               size_t destination_size, int32_t output_width,
                               int32_t output_height, int32_t output_format,
                               const D2SCoreMLIOWarpConfig *warp_config,
                               float eye_offset, float depth_strength,
                               float convergence, float smooth_texels,
                               D2SCoreMLIOPackResult *result) {
    return d2s_coreml_io_pack_internal(
        handle, slot_index, destination, NULL, destination_size,
        output_width, output_height, output_format, warp_config, eye_offset,
        depth_strength, convergence, smooth_texels, 3u, result);
}

int32_t d2s_coreml_io_pack_metal(
    void *handle, int32_t slot_index, void *metal_destination,
    size_t destination_size, int32_t output_width, int32_t output_height,
    int32_t output_format, const D2SCoreMLIOWarpConfig *warp_config,
    float eye_offset, float depth_strength, float convergence,
    float smooth_texels, D2SCoreMLIOPackResult *result) {
    return d2s_coreml_io_pack_internal(
        handle, slot_index, NULL, metal_destination, destination_size,
        output_width, output_height, output_format, warp_config, eye_offset,
        depth_strength, convergence, smooth_texels, 4u, result);
}

int32_t d2s_coreml_io_release(void *handle, int32_t slot_index) {
    D2SCoreMLIO *ctx = (D2SCoreMLIO *)handle;
    if (ctx == NULL || slot_index < 0 || slot_index >= 3) return D2S_COREML_ERROR;
    {
        D2SLockGuard lock(&ctx->mutex);
        d2s_release_slot(&ctx->slots[slot_index]);
    }
    return D2S_COREML_OK;
}

const char *d2s_coreml_io_last_error(void *handle) {
    D2SCoreMLIO *ctx = (D2SCoreMLIO *)handle;
    return ctx == NULL ? "native CoreML bridge is not initialized" : ctx->error;
}

void d2s_coreml_io_destroy(void *handle) {
    @autoreleasepool {
        D2SCoreMLIO *ctx = (D2SCoreMLIO *)handle;
        d2s_destroy_context(ctx);
    }
}
