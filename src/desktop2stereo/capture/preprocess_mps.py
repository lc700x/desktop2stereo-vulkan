"""Fused full-resolution Metal preprocessing for HDR P010 media."""

from __future__ import annotations

import functools

_P010_PQ_BT2020_MSL = r"""
#include <metal_stdlib>
using namespace metal;

struct P010Params {
    uint width;
    uint height;
};

static inline float read_p010(device const uchar* source, uint offset) {
    const uint sample = uint(source[offset]) | (uint(source[offset + 1]) << 8);
    return float(sample >> 6);
}

static inline float2 read_uv(
    device const uchar* source, uint uv_offset, uint width, uint x, uint y
) {
    const uint offset = uv_offset + (y * width + x * 2u) * 2u;
    return float2(read_p010(source, offset), read_p010(source, offset + 2u));
}

static inline float2 sample_uv(
    device const uchar* source, uint uv_offset, uint width, uint height,
    uint x, uint y
) {
    const uint chroma_width = width / 2u;
    const uint chroma_height = height / 2u;
    const float source_x = max((float(x) + 0.5f) * 0.5f - 0.5f, 0.0f);
    const float source_y = max((float(y) + 0.5f) * 0.5f - 0.5f, 0.0f);
    const uint x0 = min(uint(floor(source_x)), chroma_width - 1u);
    const uint y0 = min(uint(floor(source_y)), chroma_height - 1u);
    const uint x1 = min(x0 + 1u, chroma_width - 1u);
    const uint y1 = min(y0 + 1u, chroma_height - 1u);
    const float fx = source_x - float(x0);
    const float fy = source_y - float(y0);
    const float2 top = mix(
        read_uv(source, uv_offset, width, x0, y0),
        read_uv(source, uv_offset, width, x1, y0),
        fx
    );
    const float2 bottom = mix(
        read_uv(source, uv_offset, width, x0, y1),
        read_uv(source, uv_offset, width, x1, y1),
        fx
    );
    return mix(top, bottom, fy);
}

static inline float pq_to_linear(float value) {
    const float encoded = powr(clamp(value, 0.0f, 1.0f), 32.0f / 2523.0f);
    const float numerator = max(encoded - 3424.0f / 4096.0f, 0.0f);
    const float denominator = max(
        2413.0f / 128.0f - (2392.0f / 128.0f) * encoded,
        1.0e-6f
    );
    return powr(numerator / denominator, 16384.0f / 2610.0f) * 100.0f;
}

static inline float hable(float value) {
    return ((value * (0.15f * value + 0.05f) + 0.004f)
        / (value * (0.15f * value + 0.50f) + 0.06f)) - 0.02f / 0.30f;
}

static inline float linear_to_srgb(float value) {
    value = clamp(value, 0.0f, 1.0f);
    return value < 0.0031308f
        ? value * 12.92f
        : 1.055f * powr(value, 1.0f / 2.4f) - 0.055f;
}

kernel void d2s_p010_pq_bt2020_to_rgb(
    device const uchar* source [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant P010Params& params [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    const uint width = params.width;
    const uint height = params.height;
    const uint pixels = width * height;
    if (index >= pixels) return;

    const uint x = index % width;
    const uint y = index / width;
    const uint y_offset = index * 2u;
    const uint uv_offset = pixels * 2u;
    const float luma = (read_p010(source, y_offset) - 64.0f) / 876.0f;
    const float2 chroma = (sample_uv(source, uv_offset, width, height, x, y) - 512.0f) / 896.0f;
    const float y_value = clamp(luma, 0.0f, 1.0f);
    const float3 rgb = clamp(
        float3(
            y_value + 1.4746f * chroma.y,
            y_value - 0.164553f * chroma.x - 0.571353f * chroma.y,
            y_value + 1.8814f * chroma.x
        ),
        0.0f,
        1.0f
    );
    const float3 hdr = float3(
        pq_to_linear(rgb.r), pq_to_linear(rgb.g), pq_to_linear(rgb.b)
    );
    const float3 converted = float3(
        1.660491f * hdr.r - 0.587641f * hdr.g - 0.072850f * hdr.b,
        -0.124550f * hdr.r + 1.132900f * hdr.g - 0.008349f * hdr.b,
        -0.018151f * hdr.r - 0.100579f * hdr.g + 1.118730f * hdr.b
    );
    const float luminance = dot(converted, float3(0.2126f, 0.7152f, 0.0722f));
    const float minimum = min(converted.r, min(converted.g, converted.b));
    const float compression = minimum < 0.0f
        ? clamp(-minimum / max(luminance - minimum, 1.0e-6f), 0.0f, 1.0f)
        : 0.0f;
    const float3 gamut_mapped = max(
        converted + compression * (luminance - converted), 0.0f
    );
    const float3 tone_mapped = float3(
        hable(gamut_mapped.r), hable(gamut_mapped.g), hable(gamut_mapped.b)
    ) / hable(11.2f);
    output[index] = linear_to_srgb(tone_mapped.r);
    output[pixels + index] = linear_to_srgb(tone_mapped.g);
    output[2u * pixels + index] = linear_to_srgb(tone_mapped.b);
}
"""

_RGB_AREA_RESIZE_MSL = r"""
#include <metal_stdlib>
using namespace metal;

struct AreaResizeParams {
    uint source_width;
    uint source_height;
    uint target_width;
    uint target_height;
};

kernel void d2s_resize_rgb_area(
    device const float* source [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant AreaResizeParams& params [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    const uint source_width = params.source_width;
    const uint source_height = params.source_height;
    const uint target_width = params.target_width;
    const uint target_height = params.target_height;
    const uint target_pixels = target_width * target_height;
    if (index >= target_pixels) return;

    const uint x = index % target_width;
    const uint y = index / target_width;
    const uint first_x = uint(floor(float(x) * float(source_width) / float(target_width)));
    const uint last_x = min(
        uint(ceil(float(x + 1u) * float(source_width) / float(target_width))),
        source_width
    );
    const uint first_y = uint(floor(float(y) * float(source_height) / float(target_height)));
    const uint last_y = min(
        uint(ceil(float(y + 1u) * float(source_height) / float(target_height))),
        source_height
    );
    float3 total = float3(0.0f);
    uint sample_count = 0u;

    for (uint source_y = first_y; source_y < last_y; ++source_y) {
        for (uint source_x = first_x; source_x < last_x; ++source_x) {
            const uint source_index = source_y * source_width + source_x;
            total.r += source[source_index];
            total.g += source[source_width * source_height + source_index];
            total.b += source[2u * source_width * source_height + source_index];
            ++sample_count;
        }
    }

    const float inverse_count = 1.0f / float(max(sample_count, 1u));
    output[index] = total.r * inverse_count;
    output[target_pixels + index] = total.g * inverse_count;
    output[2u * target_pixels + index] = total.b * inverse_count;
}
"""


@functools.lru_cache(maxsize=1)
def _p010_pq_bt2020_shader(torch_module):
    return torch_module.mps.compile_shader(_P010_PQ_BT2020_MSL)


@functools.lru_cache(maxsize=16)
def _p010_params(width: int, height: int, torch_module):
    return torch_module.tensor((width, height), dtype=torch_module.int32, device="mps")


@functools.lru_cache(maxsize=1)
def _rgb_area_resize_shader(torch_module):
    return torch_module.mps.compile_shader(_RGB_AREA_RESIZE_MSL)


@functools.lru_cache(maxsize=16)
def _rgb_area_resize_params(source_width: int, source_height: int, target_width: int,
                            target_height: int, torch_module):
    return torch_module.tensor(
        (source_width, source_height, target_width, target_height),
        dtype=torch_module.int32,
        device="mps",
    )


def _resize_rgb_area(frame_rgb, target_size, *, torch_module):
    target_height, target_width = (int(value) for value in target_size)
    source_height, source_width = (int(value) for value in frame_rgb.shape[-2:])
    output = torch_module.empty(
        (1, 3, target_height, target_width),
        dtype=torch_module.float32,
        device="mps",
    )
    params = _rgb_area_resize_params(
        source_width, source_height, target_width, target_height, torch_module
    )
    _rgb_area_resize_shader(torch_module).d2s_resize_rgb_area(
        frame_rgb,
        output,
        params,
        threads=target_width * target_height,
        group_size=256,
    )
    return output


def p010_pq_bt2020_to_rgb(frame_raw, *, torch_module, target_size=None):
    """Convert P010/PQ/BT.2020 on Metal, then area-resize on the GPU."""
    import torch.nn.functional as F

    width = int(frame_raw.shape[1])
    height = int(frame_raw.shape[0]) * 2 // 3
    raw_bytes = frame_raw.view(torch_module.uint8).reshape(height * 3 // 2, width, 2)
    raw_bytes = raw_bytes.to(device="mps", non_blocking=False)
    output = torch_module.empty((1, 3, height, width), dtype=torch_module.float32, device="mps")
    params = _p010_params(width, height, torch_module)
    _p010_pq_bt2020_shader(torch_module).d2s_p010_pq_bt2020_to_rgb(
        raw_bytes,
        output,
        params,
        threads=width * height,
        group_size=256,
    )
    if target_size is not None and tuple(int(value) for value in target_size) != (height, width):
        target_height, target_width = (int(value) for value in target_size)
        if target_height <= height and target_width <= width:
            if height % target_height == 0 and width % target_width == 0:
                output = F.interpolate(
                    output,
                    size=(target_height, target_width),
                    mode="area",
                )
            else:
                output = _resize_rgb_area(
                    output, (target_height, target_width), torch_module=torch_module
                )
        else:
            output = F.interpolate(
                output,
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
                antialias=target_height < height or target_width < width,
            )
    return output
