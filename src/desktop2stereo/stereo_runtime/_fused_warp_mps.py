"""Fused warp+SBS-pack Metal kernel (torch.mps.compile_shader) for the
Vulkan local viewer path.

Replaces the torch node chain (permute/cat/contiguous + separate depth
quantize) with ONE Metal kernel that applies the exact Metal-warp math
(gaussian taps, asymmetric shaping, edge falloff) and writes final
Half-SBS RGBA8 bytes. The packer thread then only moves bytes into the
IOSurface stage / host frame — no per-frame SBS synthesis on the MPS
stream, minimal queue coupling.

Kernel validated pixel-exact against a numpy reference
(max abs diff 0 on random inputs); see tools/ notes in docs/33.

Platform: darwin + MPS only. Kill switch: D2S_VK_FUSED_WARP=0.
"""

from __future__ import annotations

import functools
import os
import sys

import numpy as np


WARP_MSL = r"""
#include <metal_stdlib>
using namespace metal;

static inline float sampf(device float* img, uint W, uint H, float u, float v) {
    float x = clamp(u * (float)W - 0.5f, 0.0f, (float)W - 1.001f);
    float y = clamp(v * (float)H - 0.5f, 0.0f, (float)H - 1.001f);
    int x0 = (int)floor(x), y0 = (int)floor(y);
    int x1 = min(x0 + 1, (int)W - 1), y1 = min(y0 + 1, (int)H - 1);
    float fx = x - (float)x0, fy = y - (float)y0;
    float a = img[y0 * W + x0], b_ = img[y0 * W + x1];
    float c = img[y1 * W + x0], d = img[y1 * W + x1];
    return mix(mix(a, b_, fx), mix(c, d, fx), fy);
}

static inline float sampf_class(
    device float* col, device float* dep, uint channel,
    uint W, uint H, float u, float v, float class_depth
) {
    float x = clamp(u * (float)W - 0.5f, 0.0f, (float)W - 1.001f);
    float y = clamp(v * (float)H - 0.5f, 0.0f, (float)H - 1.001f);
    uint x0 = (uint)floor(x), y0 = (uint)floor(y);
    uint x1 = min(x0 + 1u, W - 1u), y1 = min(y0 + 1u, H - 1u);
    float fx = x - (float)x0, fy = y - (float)y0;
    float w00 = (1.0f - fx) * (1.0f - fy);
    float w10 = fx * (1.0f - fy);
    float w01 = (1.0f - fx) * fy;
    float w11 = fx * fy;
    uint plane = W * H;
    float total = 0.0f;
    float weight = 0.0f;
    float linear = sampf(col + channel * plane, W, H, u, v);
    float d = dep[y0 * W + x0];
    float keep = 1.0f - smoothstep(0.02f, 0.06f, abs(d - class_depth));
    total += col[channel * plane + y0 * W + x0] * w00 * keep; weight += w00 * keep;
    d = dep[y0 * W + x1];
    keep = 1.0f - smoothstep(0.02f, 0.06f, abs(d - class_depth));
    total += col[channel * plane + y0 * W + x1] * w10 * keep; weight += w10 * keep;
    d = dep[y1 * W + x0];
    keep = 1.0f - smoothstep(0.02f, 0.06f, abs(d - class_depth));
    total += col[channel * plane + y1 * W + x0] * w01 * keep; weight += w01 * keep;
    d = dep[y1 * W + x1];
    keep = 1.0f - smoothstep(0.02f, 0.06f, abs(d - class_depth));
    total += col[channel * plane + y1 * W + x1] * w11 * keep; weight += w11 * keep;
    float edge = smoothstep(0.02f, 0.06f, max(max(dep[y0 * W + x0], dep[y0 * W + x1]),
        max(dep[y1 * W + x0], dep[y1 * W + x1])) -
        min(min(dep[y0 * W + x0], dep[y0 * W + x1]),
        min(dep[y1 * W + x0], dep[y1 * W + x1])));
    float support = min(weight, 1.0f);
    float class_value = total / max(weight, 1.0e-6f);
    return mix(linear, class_value, edge * support);
}

static inline float2 pack_depth_class(
    device float* col, device float* dep, uint srcW, uint srcH,
    float u, float v
) {
    float center = sampf(dep, srcW, srcH, u, v);
    float du = 1.0f / (float)srcW;
    float dv = 1.0f / (float)srcH;
    float dl = sampf(dep, srcW, srcH, u - du, v);
    float dr = sampf(dep, srcW, srcH, u + du, v);
    float duv = sampf(dep, srcW, srcH, u, v - dv);
    float dd = sampf(dep, srcW, srcH, u, v + dv);
    float low = min(min(dl, dr), min(duv, dd));
    float high = max(max(dl, dr), max(duv, dd));
    float edge_strength = smoothstep(0.0125f, 0.05f, high - low);
    float midpoint = (low + high) * 0.5f;
    uint plane = srcW * srcH;
    float center_luma = 0.0f;
    float low_luma = 0.0f;
    float high_luma = 0.0f;
    float low_count = 0.0f;
    float high_count = 0.0f;
    float samples_u[4] = {u - du, u + du, u, u};
    float samples_v[4] = {v, v, v - dv, v + dv};
    float samples_d[4] = {dl, dr, duv, dd};
    for (uint c = 0u; c < 3u; ++c) {
        center_luma += sampf(col + c * plane, srcW, srcH, u, v)
                     * (c == 0u ? 0.299f : (c == 1u ? 0.587f : 0.114f));
    }
    for (uint i = 0u; i < 4u; ++i) {
        float luma = 0.0f;
        for (uint c = 0u; c < 3u; ++c) {
            luma += sampf(col + c * plane, srcW, srcH, samples_u[i], samples_v[i])
                  * (c == 0u ? 0.299f : (c == 1u ? 0.587f : 0.114f));
        }
        float low_weight = 1.0f - smoothstep(midpoint - 0.02f, midpoint + 0.02f, samples_d[i]);
        float high_weight = 1.0f - low_weight;
        low_luma += luma * low_weight;
        high_luma += luma * high_weight;
        low_count += low_weight;
        high_count += high_weight;
    }
    low_luma /= max(low_count, 1.0e-6f);
    high_luma /= max(high_count, 1.0e-6f);
    float color_strength = smoothstep(0.01f, 0.05f, abs(high_luma - low_luma));
    float prefer_high = smoothstep(-0.02f, 0.02f,
        abs(center_luma - low_luma) - abs(center_luma - high_luma));
    return float2(prefer_high, edge_strength * color_strength);
}

static inline float warp_pack_sample(
    device float* col, device float* dep, uint channel, uint srcW, uint srcH,
    float eye, float depthStrength, float convergence, float smoothTexels,
    float u, float v
) {
    float du = smoothTexels / (float)srcW;
    float d0 = sampf(dep, srcW, srcH, u, v);
    float dm = sampf(dep, srcW, srcH, u - du, v);
    float dp_ = sampf(dep, srcW, srcH, u + du, v);
    float d = clamp(d0 * 0.7f + dm * 0.15f + dp_ * 0.15f, 0.0f, 1.0f);
    float2 depth_class = pack_depth_class(col, dep, srcW, srcH, u, v);
    float low_depth = min(dm, min(dp_, d0));
    float high_depth = max(dm, max(dp_, d0));
    float selected_depth = mix(low_depth, high_depth, depth_class.x);
    d = mix(d, selected_depth, depth_class.y);
    float d_shaped = d * (1.0f + 0.35f * (1.0f - d));
    float shift = (d_shaped - convergence) * depthStrength * eye;
    shift *= smoothstep(0.0f, 0.05f, u) * smoothstep(0.0f, 0.05f, 1.0f - u);
    float sample_u = clamp(u + shift, 0.0f, 1.0f);
    float linear = sampf(col + channel * (srcW * srcH), srcW, srcH, sample_u, v);
    float classified = sampf_class(col, dep, channel, srcW, srcH, sample_u, v, d);
    return mix(linear, classified, depth_class.y);
}

kernel void warp_pack(
    device uchar* out  [[buffer(0)]],
    device float* col  [[buffer(1)]],
    device float* dep  [[buffer(2)]],
    constant float& eyeOffset     [[buffer(3)]],
    constant float& depthStrength [[buffer(4)]],
    constant float& convergence   [[buffer(5)]],
    constant uint&  srcW          [[buffer(6)]],
    constant uint&  srcH          [[buffer(7)]],
    constant uint&  outW          [[buffer(8)]],
    constant uint&  outH          [[buffer(9)]],
    constant float& smoothTexels  [[buffer(10)]],
    uint idx [[thread_position_in_grid]])
{
    // HALF-SBS contract: output frame is outW x outH (runtime input
    // resolution per the NVIDIA-path contract), each eye squeezed into
    // outW/2; full_sbs passes outW = 2*srcW for unsqueezed eyes.
    uint pw = outW / 2u;           // per-eye width
    uint total = outW * outH * 4u;
    if (idx >= total) return;
    uint py = idx / (outW * 4u);
    uint rem = idx % (outW * 4u);
    uint px = rem / 4u;
    uint comp = rem % 4u;
    if (comp == 3u) { out[idx] = 255u; return; }

    bool left = px < pw;
    uint lx = left ? px : (px - pw);       // coordinate within the eye
    float eye = left ? -eyeOffset : eyeOffset;
    float u = ((float)lx + 0.5f) / (float)pw;   // normalized across the eye
    float v = ((float)py + 0.5f) / (float)outH;

    float value = warp_pack_sample(
        col, dep, comp, srcW, srcH, eye, depthStrength, convergence,
        smoothTexels, u, v);
    float cval = value * 255.0f + 0.5f;
    out[idx] = (uchar)clamp(cval, 0.0f, 255.0f);
}

static inline float sample_common(
    device const float* image,
    uint batch,
    uint channel,
    uint W,
    uint H,
    uint C,
    float x,
    float y
) {
    // Match grid_sample(..., align_corners=True, padding_mode=border).
    x = clamp(x, 0.0f, (float)W - 1.0f);
    y = clamp(y, 0.0f, (float)H - 1.0f);
    uint x0 = (uint)floor(x), y0 = (uint)floor(y);
    uint x1 = min(x0 + 1u, W - 1u), y1 = min(y0 + 1u, H - 1u);
    float fx = x - (float)x0, fy = y - (float)y0;
    uint plane = H * W;
    uint base = (batch * C + channel) * plane;
    float a = image[base + y0 * W + x0];
    float b = image[base + y0 * W + x1];
    float c = image[base + y1 * W + x0];
    float d = image[base + y1 * W + x1];
    return mix(mix(a, b, fx), mix(c, d, fx), fy);
}

static inline float luma_common(
    device const float* col,
    uint batch,
    uint x,
    uint y,
    uint W,
    uint H,
    uint C
) {
    uint pixel = y * W + x;
    float r = sample_common(col, batch, 0u, W, H, C, float(x), float(y));
    float g = sample_common(col, batch, 1u, W, H, C, float(x), float(y));
    float b = sample_common(col, batch, 2u, W, H, C, float(x), float(y));
    return r * 0.299f + g * 0.587f + b * 0.114f;
}

static inline float2 edge_depth_class_common(
    device const float* col,
    device const float* dep,
    uint batch,
    uint x,
    uint y,
    uint W,
    uint H,
    uint C,
    float center_depth
) {
    uint xl = x > 0u ? x - 1u : x;
    uint xr = min(x + 1u, W - 1u);
    uint yu = y > 0u ? y - 1u : y;
    uint yd = min(y + 1u, H - 1u);
    uint plane = W * H;
    uint base = batch * plane;
    float dl = dep[base + y * W + xl];
    float dr = dep[base + y * W + xr];
    float du = dep[base + yu * W + x];
    float dd = dep[base + yd * W + x];
    float low = min(min(dl, dr), min(du, dd));
    float high = max(max(dl, dr), max(du, dd));
    float midpoint = (low + high) * 0.5f;
    float center = luma_common(col, batch, x, y, W, H, C);
    float ll = luma_common(col, batch, xl, y, W, H, C);
    float lr = luma_common(col, batch, xr, y, W, H, C);
    float lu = luma_common(col, batch, x, yu, W, H, C);
    float ld = luma_common(col, batch, x, yd, W, H, C);
    float low_sum = 0.0f;
    float high_sum = 0.0f;
    float low_l = 1.0f - smoothstep(midpoint - 0.02f, midpoint + 0.02f, dl);
    float low_r = 1.0f - smoothstep(midpoint - 0.02f, midpoint + 0.02f, dr);
    float low_u = 1.0f - smoothstep(midpoint - 0.02f, midpoint + 0.02f, du);
    float low_d = 1.0f - smoothstep(midpoint - 0.02f, midpoint + 0.02f, dd);
    float high_l = 1.0f - low_l, high_r = 1.0f - low_r;
    float high_u = 1.0f - low_u, high_d = 1.0f - low_d;
    float low_count = low_l + low_r + low_u + low_d;
    float high_count = high_l + high_r + high_u + high_d;
    low_sum = ll * low_l + lr * low_r + lu * low_u + ld * low_d;
    high_sum = ll * high_l + lr * high_r + lu * high_u + ld * high_d;
    float low_luma = low_sum / max(low_count, 1.0e-6f);
    float high_luma = high_sum / max(high_count, 1.0e-6f);
    float depth_strength = smoothstep(0.0125f, 0.05f, high - low);
    float color_strength = smoothstep(0.01f, 0.05f, abs(high_luma - low_luma));
    float prefer_high = smoothstep(-0.02f, 0.02f,
        abs(center - low_luma) - abs(center - high_luma));
    return float2(prefer_high, depth_strength * color_strength);
}

static inline float sample_class_common(
    device const float* col,
    device const float* dep,
    uint batch,
    uint channel,
    uint W,
    uint H,
    uint C,
    float x,
    float y,
    float class_depth
) {
    x = clamp(x, 0.0f, (float)W - 1.0f);
    y = clamp(y, 0.0f, (float)H - 1.0f);
    uint x0 = (uint)floor(x), y0 = (uint)floor(y);
    uint x1 = min(x0 + 1u, W - 1u), y1 = min(y0 + 1u, H - 1u);
    float fx = x - (float)x0, fy = y - (float)y0;
    float w00 = (1.0f - fx) * (1.0f - fy);
    float w10 = fx * (1.0f - fy);
    float w01 = (1.0f - fx) * fy;
    float w11 = fx * fy;
    uint plane = H * W;
    uint depth_base = batch * plane;
    float total = 0.0f;
    float weight = 0.0f;
    float linear = sample_common(col, batch, channel, W, H, C, x, y);
    float d = dep[depth_base + y0 * W + x0];
    float keep = 1.0f - smoothstep(0.02f, 0.06f, abs(d - class_depth));
    total += sample_common(col, batch, channel, W, H, C, (float)x0, (float)y0) * w00 * keep;
    weight += w00 * keep;
    d = dep[depth_base + y0 * W + x1];
    keep = 1.0f - smoothstep(0.02f, 0.06f, abs(d - class_depth));
    total += sample_common(col, batch, channel, W, H, C, (float)x1, (float)y0) * w10 * keep;
    weight += w10 * keep;
    d = dep[depth_base + y1 * W + x0];
    keep = 1.0f - smoothstep(0.02f, 0.06f, abs(d - class_depth));
    total += sample_common(col, batch, channel, W, H, C, (float)x0, (float)y1) * w01 * keep;
    weight += w01 * keep;
    d = dep[depth_base + y1 * W + x1];
    keep = 1.0f - smoothstep(0.02f, 0.06f, abs(d - class_depth));
    total += sample_common(col, batch, channel, W, H, C, (float)x1, (float)y1) * w11 * keep;
    weight += w11 * keep;
    float d00 = dep[depth_base + y0 * W + x0];
    float d10 = dep[depth_base + y0 * W + x1];
    float d01 = dep[depth_base + y1 * W + x0];
    float d11 = dep[depth_base + y1 * W + x1];
    float edge = smoothstep(0.02f, 0.06f,
        max(max(d00, d10), max(d01, d11)) - min(min(d00, d10), min(d01, d11)));
    float class_value = total / max(weight, 1.0e-6f);
    return mix(linear, class_value, edge * min(weight, 1.0f));
}

static inline float projection_scale_common(float value) {
    return 0.875f + 0.125f / (1.0f + exp((1.0f - 2.0f * value) / 0.08f));
}

static inline float projected_common(
    device const float* col, device const float* dep, device const float* shift,
    uint batch, uint channel, uint W, uint H, uint C, uint x, uint y,
    float target_depth, float target_shift, float eye, constant float2& bounds
) {
    float source_x = clamp(float(x) + target_shift * projection_scale_common(target_depth) * eye,
                           0.0f, float(W) - 1.0f);
    float source_depth = sample_common(dep, batch, 0u, W, H, 1u, source_x, float(y));
    float source_shift = sample_common(shift, batch, 0u, W, H, 1u, source_x, float(y));
    float refined_x = clamp(float(x) + source_shift * projection_scale_common(source_depth) * eye,
                            0.0f, float(W) - 1.0f);
    float front_x = clamp(float(x) + bounds.x * eye, 0.0f, float(W) - 1.0f);
    float front_depth = sample_common(dep, batch, 0u, W, H, 1u, front_x, float(y));
    float confidence = max(smoothstep(0.02f, 0.04f, abs(source_depth - target_depth)),
                           smoothstep(0.125f, 0.5f, abs(refined_x - source_x)));
    confidence = max(confidence, smoothstep(0.02f, 0.04f, front_depth - source_depth));
    float linear = sample_common(col, batch, channel, W, H, C, refined_x, float(y));
    if (confidence <= 0.0f) return linear;
    int first = int(ceil(float(x) + min(bounds.x * eye, bounds.y * eye) - 1.0f));
    int count = int(bounds.y - bounds.x) + 4;
    float best_depth = -1.0f, best_coverage = 0.0f;
    int best_x = int(x), left_x = int(x), right_x = int(x);
    float left_error = 1.0e6f, right_error = 1.0e6f, left_depth = -1.0f, right_depth = -1.0f;
    uint base = batch * W * H + y * W;
    for (int offset = 0; offset < count; ++offset) {
        int sx = first + offset;
        if (sx < 0 || sx >= int(W)) continue;
        float sample_depth = dep[base + uint(sx)];
        float error = float(sx) - shift[base + uint(sx)] * projection_scale_common(sample_depth) * eye - float(x);
        float coverage = max(1.0f - abs(error), 0.0f);
        if (coverage > 0.0f && (sample_depth > best_depth ||
            (sample_depth == best_depth && coverage > best_coverage))) {
            best_depth = sample_depth; best_coverage = coverage; best_x = sx;
        }
        if (error <= 0.0f && abs(error) < left_error) {
            left_error = abs(error); left_depth = sample_depth; left_x = sx;
        }
        if (error >= 0.0f && error < right_error) {
            right_error = error; right_depth = sample_depth; right_x = sx;
        }
    }
    bool use_left = left_error < 1.0e5f && (right_error >= 1.0e5f || left_depth <= right_depth);
    int background_x = use_left ? left_x : right_x;
    float background_depth = use_left ? left_depth : right_depth;
    float left_color = sample_common(col, batch, channel, W, H, C, float(left_x), float(y));
    float right_color = sample_common(col, batch, channel, W, H, C, float(right_x), float(y));
    float side_fraction = left_error / max(left_error + right_error, 1.0e-6f);
    float background_linear = mix(left_color, right_color, side_fraction);
    float background_edge = smoothstep(0.02f, 0.04f, abs(left_depth - right_depth));
    float background = mix(background_linear, use_left ? left_color : right_color, background_edge);
    float foreground = 0.0f, foreground_coverage = 0.0f;
    for (int offset = -1; offset <= 1; ++offset) {
        int sx = best_x + offset;
        if (best_depth < 0.0f || sx < 0 || sx >= int(W)) continue;
        float sample_depth = dep[base + uint(sx)];
        float error = float(sx) - shift[base + uint(sx)] * projection_scale_common(sample_depth) * eye - float(x);
        float class_weight = 1.0f - smoothstep(0.02f, 0.04f, best_depth - sample_depth);
        float coverage = max(1.0f - abs(error), 0.0f) * class_weight;
        foreground += sample_common(col, batch, channel, W, H, C, float(sx), float(y)) * coverage;
        foreground_coverage += coverage;
    }
    foreground /= max(foreground_coverage, 1.0e-6f);
    float background_confidence = smoothstep(0.02f, 0.04f, best_depth - background_depth);
    float resolved = mix(foreground, background,
        (1.0f - min(foreground_coverage, 1.0f)) * background_confidence);
    if (best_depth < 0.0f) resolved = background;
    return mix(linear, resolved, confidence);
}

static inline float blend_common(
    device const float* col,
    device const float* dep,
    device const float* shift,
    uint batch,
    uint channel,
    uint W,
    uint H,
    uint C,
    uint x,
    uint y,
    float eye_sign,
    constant float2& projection_bounds
) {
    uint plane = H * W;
    uint depth_idx = batch * plane + y * W + x;
    float d = clamp(dep[depth_idx], 0.0f, 1.0f);
    float w0 = exp(-(d * d) / 0.08f);
    float w1 = exp(-((d - 1.0f) * (d - 1.0f)) / 0.08f);
    float weight_sum = max(w0 + w1, 1.0e-6f);
    float base = shift[depth_idx];
    if (projection_bounds.x != 0.0f || projection_bounds.y != 0.0f) {
        return projected_common(col, dep, shift, batch, channel, W, H, C,
                                x, y, d, base, eye_sign, projection_bounds);
    }
    float shift0 = base * 0.875f;
    float shift1 = base;
    float2 depth_class = edge_depth_class_common(
        col, dep, batch, x, y, W, H, C, d
    );
    w1 = mix(w1, depth_class.x, depth_class.y);
    w0 = 1.0f - w1;
    weight_sum = max(w0 + w1, 1.0e-6f);
    float sample0 = sample_class_common(col, dep, batch, channel, W, H, C,
        (float)x + shift0 * eye_sign, (float)y, d);
    float sample1 = sample_class_common(col, dep, batch, channel, W, H, C,
        (float)x + shift1 * eye_sign, (float)y, d);
    return (w0 * sample0 + w1 * sample1) / weight_sum;
}

static inline float downsample_common(
    device const float* col,
    device const float* dep,
    device const float* shift,
    uint batch,
    uint channel,
    uint W,
    uint H,
    uint C,
    uint x,
    uint y,
    bool horizontal,
    float eye_sign,
    constant float2& projection_bounds
) {
    uint center = 2u * (horizontal ? x : y);
    uint limit = (horizontal ? W : H) - 1u;
    uint x0 = horizontal ? min(center, limit) : x;
    uint y0 = horizontal ? y : min(center, limit);
    uint x1 = horizontal ? min(center + 1u, limit) : x;
    uint y1 = horizontal ? y : min(center + 1u, limit);
    return 0.5f * (
        blend_common(col, dep, shift, batch, channel, W, H, C, x0, y0, eye_sign, projection_bounds) +
        blend_common(col, dep, shift, batch, channel, W, H, C, x1, y1, eye_sign, projection_bounds)
    );
}

kernel void warp_composite2_u8(
    device uchar* out        [[buffer(0)]],
    device const float* col  [[buffer(1)]],
    device const float* dep  [[buffer(2)]],
    device const float* shift [[buffer(3)]],
    constant uint& B         [[buffer(4)]],
    constant uint& C         [[buffer(5)]],
    constant uint& W         [[buffer(6)]],
    constant uint& H         [[buffer(7)]],
    constant uint& outW      [[buffer(8)]],
    constant uint& outH      [[buffer(9)]],
    constant uint& format    [[buffer(10)]],
    constant float2& projection_bounds [[buffer(11)]],
    uint idx [[thread_position_in_grid]])
{
    uint total = B * C * outH * outW;
    if (idx >= total) return;
    uint out_plane = outH * outW;
    uint out_pixels = C * out_plane;
    uint batch = idx / out_pixels;
    uint rem = idx % out_pixels;
    uint channel = rem / out_plane;
    uint pixel = rem % out_plane;
    uint ox = pixel % outW;
    uint oy = pixel / outW;
    bool horizontal = (format == 0u || format == 1u);
    bool half_res = (format == 0u || format == 2u);
    uint eyeW = horizontal ? (half_res ? W / 2u : W) : W;
    uint eyeH = horizontal ? H : (half_res ? H / 2u : H);
    bool right_eye = horizontal ? ox >= eyeW : oy >= eyeH;
    uint x = horizontal ? (right_eye ? ox - eyeW : ox) : ox;
    uint y = horizontal ? oy : (right_eye ? oy - eyeH : oy);
    float eye_sign = right_eye ? -1.0f : 1.0f;
    float value;
    if (half_res) {
        value = downsample_common(
            col, dep, shift, batch, channel, W, H, C, x, y,
            horizontal, eye_sign, projection_bounds
        );
    } else {
        value = blend_common(
            col, dep, shift, batch, channel, W, H, C, x, y, eye_sign, projection_bounds
        );
    }
    out[idx] = (uchar)clamp(value * 255.0f + 0.5f, 0.0f, 255.0f);
}

kernel void warp_composite2(
    device float* left       [[buffer(0)]],
    device float* right      [[buffer(1)]],
    device const float* col  [[buffer(2)]],
    device const float* dep  [[buffer(3)]],
    device const float* shift [[buffer(4)]],
    constant uint& B         [[buffer(5)]],
    constant uint& C         [[buffer(6)]],
    constant uint& W         [[buffer(7)]],
    constant uint& H         [[buffer(8)]],
    constant float2& projection_bounds [[buffer(9)]],
    uint idx [[thread_position_in_grid]])
{
    uint total = B * C * H * W;
    if (idx >= total) return;
    uint plane = H * W;
    uint pixels = C * plane;
    uint batch = idx / pixels;
    uint rem = idx % pixels;
    uint channel = rem / plane;
    uint pixel = rem % plane;
    uint y = pixel / W;
    uint x = pixel % W;
    float left_value = blend_common(
        col, dep, shift, batch, channel, W, H, C, x, y, 1.0f, projection_bounds
    );
    float right_value = blend_common(
        col, dep, shift, batch, channel, W, H, C, x, y, -1.0f, projection_bounds
    );
    left[idx] = left_value;
    right[idx] = right_value;
}

static inline float read_eye(
    device const float* eye,
    uint batch,
    uint channel,
    uint W,
    uint H,
    uint C,
    uint x,
    uint y
) {
    return eye[(batch * C + channel) * H * W + y * W + x];
}

kernel void pack_eyes_u8(
    device uchar* out       [[buffer(0)]],
    device const float* left [[buffer(1)]],
    device const float* right [[buffer(2)]],
    constant uint& B        [[buffer(3)]],
    constant uint& C        [[buffer(4)]],
    constant uint& W        [[buffer(5)]],
    constant uint& H        [[buffer(6)]],
    constant uint& outW     [[buffer(7)]],
    constant uint& outH     [[buffer(8)]],
    constant uint& format   [[buffer(9)]],
    uint idx [[thread_position_in_grid]])
{
    uint total = B * C * outH * outW;
    if (idx >= total) return;
    uint out_plane = outH * outW;
    uint out_pixels = C * out_plane;
    uint batch = idx / out_pixels;
    uint rem = idx % out_pixels;
    uint channel = rem / out_plane;
    uint pixel = rem % out_plane;
    uint ox = pixel % outW;
    uint oy = pixel / outW;
    bool horizontal = (format == 0u || format == 1u);
    bool half_res = (format == 0u || format == 2u);
    uint eyeW = horizontal ? (half_res ? W / 2u : W) : W;
    uint eyeH = horizontal ? H : (half_res ? H / 2u : H);
    bool right_eye = horizontal ? ox >= eyeW : oy >= eyeH;
    uint x = horizontal ? (right_eye ? ox - eyeW : ox) : ox;
    uint y = horizontal ? oy : (right_eye ? oy - eyeH : oy);
    device const float* eye = right_eye ? right : left;
    float value;
    if (half_res && horizontal) {
        value = 0.5f * (read_eye(eye, batch, channel, W, H, C, 2u * x, y) +
                        read_eye(eye, batch, channel, W, H, C, min(2u * x + 1u, W - 1u), y));
    } else if (half_res) {
        uint center = 2u * y;
        float taps[4] = {-1.0f, 9.0f, 9.0f, -1.0f};
        value = 0.0f;
        for (uint tap = 0u; tap < 4u; ++tap) {
            int source = (int)center + (int)tap - 1;
            uint limit = (horizontal ? W : H) - 1u;
            uint coordinate = (uint)clamp(source, 0, (int)limit);
            uint sx = horizontal ? coordinate : x;
            uint sy = horizontal ? y : coordinate;
            value += taps[tap] * read_eye(
                eye, batch, channel, W, H, C, sx, sy
            );
        }
        value *= 1.0f / 16.0f;
    } else {
        value = read_eye(eye, batch, channel, W, H, C, x, y);
    }
    out[idx] = (uchar)clamp(value * 255.0f + 0.5f, 0.0f, 255.0f);
}

static inline float joint_depth_curvature(
    device const float* depth, uint base, int x, int y, uint height, uint width) {
    float curvatureX = 0.0f, curvatureY = 0.0f;
    if (width >= 3u) {
        uint mx = uint(clamp(x, 1, int(width) - 2));
        uint row = base + uint(y) * width;
        curvatureX = abs(depth[row + mx - 1u] - 2.0f * depth[row + mx] + depth[row + mx + 1u]);
    }
    if (height >= 3u) {
        uint my = uint(clamp(y, 1, int(height) - 2));
        uint column = base + uint(x);
        curvatureY = abs(depth[column + (my - 1u) * width] -
                         2.0f * depth[column + my * width] + depth[column + (my + 1u) * width]);
    }
    return max(curvatureX, curvatureY);
}

kernel void joint_bilateral_depth(
    device float* output [[buffer(0)]],
    device const float* depth [[buffer(1)]],
    device const float* rgb [[buffer(2)]],
    device const float* guide [[buffer(3)]],
    constant uint& depthH [[buffer(4)]],
    constant uint& depthW [[buffer(5)]],
    constant uint& height [[buffer(6)]],
    constant uint& width [[buffer(7)]],
    constant float& colorSigma [[buffer(8)]],
    constant float& edgeThreshold [[buffer(9)]],
    constant uint& batches [[buffer(10)]],
    uint index [[thread_position_in_grid]]) {
    uint pixels = width * height;
    uint depthPixels = depthW * depthH;
    uint total = pixels * batches;
    if (index >= total) return;
    uint batch = index / pixels;
    uint pixel = index - batch * pixels;
    uint x = pixel % width;
    uint y = pixel / width;
    float dx = clamp((float(x) + 0.5f) * float(depthW) / float(width) - 0.5f,
                     0.0f, float(depthW) - 1.0f);
    float dy = clamp((float(y) + 0.5f) * float(depthH) / float(height) - 0.5f,
                     0.0f, float(depthH) - 1.0f);
    uint x0 = uint(floor(dx));
    uint y0 = uint(floor(dy));
    uint x1 = min(x0 + 1u, depthW - 1u);
    uint y1 = min(y0 + 1u, depthH - 1u);
    float fx = dx - float(x0);
    float fy = dy - float(y0);
    uint i00 = y0 * depthW + x0;
    uint i10 = y0 * depthW + x1;
    uint i01 = y1 * depthW + x0;
    uint i11 = y1 * depthW + x1;
    uint depthBase = batch * depthPixels;
    float d00 = depth[depthBase + i00];
    float d10 = depth[depthBase + i10];
    float d01 = depth[depthBase + i01];
    float d11 = depth[depthBase + i11];
    float w00 = (1.0f - fy) * (1.0f - fx);
    float w10 = (1.0f - fy) * fx;
    float w01 = fy * (1.0f - fx);
    float w11 = fy * fx;
    float low = min(min(d00, d10), min(d01, d11));
    float high = max(max(d00, d10), max(d01, d11));
    float linear = w00 * d00 + w10 * d10 + w01 * d01 + w11 * d11;

    float edgeSignal =
        w00 * joint_depth_curvature(depth, depthBase, int(x0), int(y0), depthH, depthW) +
        w10 * joint_depth_curvature(depth, depthBase, int(x1), int(y0), depthH, depthW) +
        w01 * joint_depth_curvature(depth, depthBase, int(x0), int(y1), depthH, depthW) +
        w11 * joint_depth_curvature(depth, depthBase, int(x1), int(y1), depthH, depthW);
    float edgeWeight = smoothstep(max(edgeThreshold, 1.0e-6f) * 0.25f,
                                  max(edgeThreshold, 1.0e-6f) * 0.5f, edgeSignal);
    if (edgeWeight <= 0.0f) {
        output[index] = linear;
        return;
    }
    uint rgbBase = batch * 3u * pixels;
    uint guideBase = batch * 3u * depthPixels;
    float3 target = float3(rgb[rgbBase + pixel], rgb[rgbBase + pixels + pixel],
                           rgb[rgbBase + 2u * pixels + pixel]);
    int cx = int(floor(dx + 0.5f)), cy = int(floor(dy + 0.5f));
    float spatial_total = 0.0f, depth_sum = 0.0f, depth_square_sum = 0.0f;
    float3 color_sum = float3(0.0f);
    float color_square_sum = 0.0f, total_weight = 0.0f, guided_sum = 0.0f;
    float max_log_weight = -1.0e30f;
    float matching_distance_sum = 0.0f;
    float low_count = 0.0f, high_count = 0.0f, low_delta_sum = 0.0f, high_delta_sum = 0.0f;
    float low_total = 0.0f, high_total = 0.0f, low_sum = 0.0f, high_sum = 0.0f;
    float threshold = max(edgeThreshold, 1.0e-6f);
    float sigma = max(colorSigma / 3.0f, 1.0e-4f);
    for (int oy = -1; oy <= 1; ++oy) {
        for (int ox = -1; ox <= 1; ++ox) {
            uint sx = uint(clamp(cx + ox, 0, int(depthW) - 1));
            uint sy = uint(clamp(cy + oy, 0, int(depthH) - 1));
            float spatial = (1.0f - smoothstep(0.5f, 1.5f, abs(dx - float(cx + ox)))) *
                            (1.0f - smoothstep(0.5f, 1.5f, abs(dy - float(cy + oy))));
            uint sample_index = sy * depthW + sx;
            float sample = depth[depthBase + sample_index];
            float3 color = float3(guide[guideBase + sample_index], guide[guideBase + depthPixels + sample_index],
                                   guide[guideBase + 2u * depthPixels + sample_index]);
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
            float low = 1.0f - smoothstep(-0.5f * threshold, 0.5f * threshold, sample - linear);
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
    float guided = total_weight > 1.0e-8f ? guided_sum / total_weight : linear;
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
    output[index] =
        clamp(mix(linear, guided, edgeWeight * confidence), 0.0f, 1.0f);
}
"""


@functools.lru_cache(maxsize=1)
def _lib():
    import torch

    return torch.mps.compile_shader(WARP_MSL)


def mps_joint_bilateral_upsample(depth, rgb, height: int, width: int,
                                 color_sigma: float = 0.12,
                                 edge_threshold: float = 0.04):
    """Run guided depth reconstruction in one Metal dispatch."""
    import torch
    import torch.nn.functional as F

    depth = depth.contiguous().float()
    rgb = rgb.contiguous().float()
    batch, _, depth_height, depth_width = depth.shape
    guide = F.interpolate(rgb, size=(depth_height, depth_width), mode="bilinear",
                          align_corners=False).contiguous()
    output = torch.empty((batch, 1, height, width), device="mps", dtype=torch.float32)
    pixels = height * width
    _lib().joint_bilateral_depth(
        output, depth, rgb, guide,
        int(depth_height), int(depth_width), int(height), int(width),
        float(color_sigma), float(edge_threshold), int(batch),
        threads=pixels * batch, group_size=256,
    )
    return output


def warp_params_from_env() -> tuple[float, float, float]:
    """Mirror macos_metal_viewer's calibration knobs exactly."""
    ipd_uv = float(os.environ.get("D2S_METAL_WARP_IPD", "0.064") or 0.064)
    depth_strength = 0.1 * float(
        os.environ.get("D2S_METAL_WARP_DEPTH_STRENGTH", "4.0") or 4.0
    )
    convergence = float(os.environ.get("D2S_METAL_WARP_CONVERGENCE", "0.0") or 0.0)
    return ipd_uv / 2.0, depth_strength, convergence


def fused_enabled() -> bool:
    """Darwin + Vulkan viewer + not explicitly disabled."""
    return (
        sys.platform == "darwin"
        and os.environ.get("D2S_MAC_VIEWER") == "vulkan"
        and os.environ.get("D2S_VK_FUSED_WARP", "1")
        not in {"0", "false", "off"}
    )


@functools.lru_cache(maxsize=8)
def _smooth_texels_cached(key: tuple[int, int, int]) -> float:
    import os as _os

    raw = _os.environ.get("D2S_WARP_DEPTH_SMOOTH_TEXELS", "")
    try:
        return max(0.0, float(raw))
    except Exception:
        pass
    eye_w, src_w, base = key
    # Scale-invariant default: 1.5 texels at the reference where eye width
    # equals source width; grows proportionally when the packed frame is
    # larger than the source (native-res presentation).
    return 1.5 * (float(eye_w) / float(src_w)) if src_w else 1.5


def warp_smooth_texels(src_w: int, out_w: int) -> float:
    """Depth-smoothing aperture in source texels for the warp kernel."""
    eye_w = max(1, int(out_w) // 2)  # half-SBS per-eye width
    return _smooth_texels_cached((eye_w, int(src_w), int(out_w)))


def pack_target(src_w: int, src_h: int, output_format: str = "half_sbs") -> tuple[int, int]:
    """Frame dims for a given runtime input size and output format.

    Mirrors the NVIDIA local-mode contract: SBS geometry follows the
    RUNTIME input resolution and the selected format -- never the viewer
    window. 1080p in -> half_sbs/half_tab 1920x1080, full_sbs 3840x1080,
    and full_tab 1920x2160. Mono and diagnostic modes preserve source size.
    """
    sw, sh = int(src_w), int(src_h)
    output_format = str(output_format)
    if output_format == "full_sbs":
        return sw * 2, sh
    if output_format == "full_tab":
        return sw, sh * 2
    if output_format in {"half_sbs", "half_tab"}:
        # The packed half modes preserve the source frame dimensions while
        # requiring an even split for the local viewer's eye regions.
        return (sw - sw % 2, sh) if output_format == "half_sbs" else (sw, sh)
    return sw, sh


def fused_sbs_pack(rgb_f32_chw, depth_f32, host_out=None, out_size=None,
                   output_format: str = "half_sbs"):
    """Run the fused kernel; return (host_view|None, w, h) like
    _pack_sbs_host_frame, or None on any failure (caller falls back).

    Default frame dims follow the NVIDIA local-mode contract: derived from
    the runtime INPUT resolution and ``output_format`` (half_sbs WxH,
    full_sbs 2WxH). ``out_size`` remains as an explicit override."""
    try:
        import torch

        if rgb_f32_chw.dim() == 4 and int(rgb_f32_chw.shape[0]) == 1:
            rgb_f32_chw = rgb_f32_chw.squeeze(0)  # BCHW -> CHW
        if rgb_f32_chw.dim() != 3:
            if os.environ.get("D2S_FUSED_DEBUG"):
                print(f"[fused] skip: dim={rgb_f32_chw.dim()}", flush=True)
            return None
        channels, h, w = (
            int(rgb_f32_chw.shape[0]),
            int(rgb_f32_chw.shape[-2]),
            int(rgb_f32_chw.shape[-1]),
        )
        if channels != 3 or h <= 0 or w <= 0:
            if os.environ.get("D2S_FUSED_DEBUG"):
                print(f"[fused] skip: ch={channels} h={h} w={w}", flush=True)
            return None
        dep = depth_f32
        if dep.dim() == 3:
            dep = dep.squeeze(0)
        ow, oh = (
            (int(out_size[0]), int(out_size[1]))
            if out_size is not None
            else pack_target(w, h, output_format)
        )
        if ow % 2 != 0 or ow < 4 or oh < 4:
            return None  # half-SBS needs an even frame width
        render_width = ow
        out_t = torch.empty(render_width * oh * 4, dtype=torch.uint8, device="mps")
        eo, ds, cv = warp_params_from_env()
        stex = warp_smooth_texels(w, ow)
        _lib().warp_pack(
            out_t, rgb_f32_chw.contiguous(), dep.contiguous(),
            float(eo), float(ds), float(cv),
            int(w), int(h), int(render_width), int(oh), float(stex),
        )
        # Half-SBS: reported dims are the FRAME dims (ow x oh), matching the
        # synthesized half_sbs contract the viewer was built around.
        if host_out is not None:
            dst = torch.frombuffer(host_out, dtype=torch.uint8)
            dst.copy_(out_t)
            view = np.frombuffer(host_out, dtype=np.uint8).reshape(oh, ow, 4)
            return view, ow, oh
        host = out_t.cpu().numpy().reshape(oh, ow, 4)
        return host, ow, oh
    except Exception as exc:
        if os.environ.get("D2S_FUSED_DEBUG"):
            print(f"[fused] pack failed: {exc!r}", flush=True)
        return None


@functools.lru_cache(maxsize=16)
def _mps_projection_bounds(bounds: tuple[float, float]):
    import torch

    return torch.tensor(bounds, dtype=torch.float32, device="mps")


def mps_warp_composite2(rgb_f32, depth_f32, base_shift, edge_aa_enabled=None):
    """Run the canonical two-layer warp without MPS grid_sample launches.

    The kernel mirrors the common synthesis path for the streaming profile:
    two depth layers, symmetric eyes, bilinear border sampling, and the
    already-resolved pixel shift. It intentionally returns ``None`` for any
    unsupported shape so the caller can use the existing torch path.
    """
    try:
        import torch

        if sys.platform != "darwin" or rgb_f32.device.type != "mps":
            return None
        if rgb_f32.ndim == 3:
            rgb_f32 = rgb_f32.unsqueeze(0)
        if rgb_f32.ndim != 4 or rgb_f32.dtype != torch.float32:
            return None
        if depth_f32.ndim == 3:
            depth_f32 = depth_f32.unsqueeze(1)
        if base_shift.ndim == 3:
            base_shift = base_shift.unsqueeze(1)
        if depth_f32.ndim != 4 or base_shift.ndim != 4:
            return None
        batch, channels, height, width = map(int, rgb_f32.shape)
        if channels != 3 or tuple(depth_f32.shape) != (batch, 1, height, width):
            return None
        if tuple(base_shift.shape) != (batch, 1, height, width):
            return None
        left = torch.empty_like(rgb_f32)
        right = torch.empty_like(rgb_f32)
        _lib().warp_composite2(
            left,
            right,
            rgb_f32.contiguous(),
            depth_f32.contiguous(),
            base_shift.contiguous(),
            int(batch),
            int(channels),
            int(width),
            int(height),
            _mps_projection_bounds(getattr(base_shift, "_d2s_projection_bounds", (0.0, 0.0))),
        )
        return left, right
    except Exception as exc:
        if os.environ.get("D2S_FUSED_DEBUG"):
            print(f"[mps-warp] canonical kernel failed: {exc!r}", flush=True)
        return None


def mps_warp_composite2_u8(rgb_f32, depth_f32, base_shift, output_format: str):
    """Pack the canonical two-layer output directly into an MPS uint8 tensor."""
    try:
        import torch

        if sys.platform != "darwin" or rgb_f32.device.type != "mps":
            return None
        if rgb_f32.ndim == 3:
            rgb_f32 = rgb_f32.unsqueeze(0)
        if rgb_f32.ndim != 4 or rgb_f32.dtype != torch.float32:
            return None
        if depth_f32.ndim == 3:
            depth_f32 = depth_f32.unsqueeze(1)
        if base_shift.ndim == 3:
            base_shift = base_shift.unsqueeze(1)
        batch, channels, height, width = map(int, rgb_f32.shape)
        if channels != 3 or height % 2 or width % 2:
            return None
        if tuple(depth_f32.shape) != (batch, 1, height, width):
            return None
        if tuple(base_shift.shape) != (batch, 1, height, width):
            return None
        formats = {"half_sbs": 0, "full_sbs": 1, "half_tab": 2, "full_tab": 3}
        format_id = formats.get(str(output_format))
        if format_id is None:
            return None
        out_width = width * 2 if format_id == 1 else width
        out_height = height * 2 if format_id == 3 else height
        warped = mps_warp_composite2(rgb_f32, depth_f32, base_shift)
        if warped is None:
            return None
        left, right = warped
        out = torch.empty(
            (batch, channels, out_height, out_width),
            dtype=torch.uint8,
            device="mps",
        )
        _lib().pack_eyes_u8(
            out,
            left,
            right,
            int(batch),
            int(channels),
            int(width),
            int(height),
            int(out_width),
            int(out_height),
            int(format_id),
        )
        return out
    except Exception as exc:
        if os.environ.get("D2S_FUSED_DEBUG"):
            print(f"[mps-warp] packed canonical kernel failed: {exc!r}", flush=True)
        return None
