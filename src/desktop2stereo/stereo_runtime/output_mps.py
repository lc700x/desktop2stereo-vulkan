"""Small Metal kernels used by the Apple Silicon output path."""

from __future__ import annotations

import functools

import torch


_HALF_SBS_SRGB_MSL = r"""
#include <metal_stdlib>
using namespace metal;

struct HalfSbsParams {
    uint width;
    uint height;
    uint channels;
    uint batches;
};

inline float srgb_to_linear(float value) {
    value = clamp(value, 0.0f, 1.0f);
    return value <= 0.04045f ? value / 12.92f
        : powr((value + 0.055f) / 1.055f, 2.4f);
}

inline float linear_to_srgb(float value) {
    value = clamp(value, 0.0f, 1.0f);
    return value <= 0.0031308f ? value * 12.92f
        : 1.055f * powr(value, 1.0f / 2.4f) - 0.055f;
}

kernel void d2s_make_half_sbs_srgb(
    device const float* left [[buffer(0)]],
    device const float* right [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant HalfSbsParams& params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    const uint half_width = params.width / 2;
    const uint pixels = params.height * params.width;
    const uint plane_count = params.batches * params.channels;
    const uint total = plane_count * pixels;
    if (index >= total) return;

    const uint plane = index / pixels;
    const uint pixel = index - plane * pixels;
    const uint y = pixel / params.width;
    const uint x = pixel - y * params.width;
    const bool use_left = x < half_width;
    const uint local_x = use_left ? x : x - half_width;
    const uint src_pixel = y * params.width + local_x * 2;
    const uint src_index = plane * pixels + src_pixel;
    device const float* source = use_left ? left : right;
    const float a = source[src_index];
    const float b = source[src_index + 1];

    float result;
    if ((plane % params.channels) < 3) {
        result = linear_to_srgb(
            0.5f * (srgb_to_linear(a) + srgb_to_linear(b)));
    } else {
        result = 0.5f * (a + b);
    }
    output[index] = result;
}
"""


@functools.lru_cache(maxsize=1)
def _half_sbs_srgb_library():
    return torch.mps.compile_shader(_HALF_SBS_SRGB_MSL)


@functools.lru_cache(maxsize=32)
def _half_sbs_params(width: int, height: int, channels: int, batches: int) -> torch.Tensor:
    return torch.tensor((width, height, channels, batches), dtype=torch.int32, device="mps")


def make_half_sbs_srgb(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Pack two RGB/RGBA eye images using exact 2:1 linear-light area sampling."""
    if (
        left.device.type != "mps"
        or right.device != left.device
        or left.dtype != torch.float32
        or right.dtype != torch.float32
        or left.ndim != 4
        or left.shape != right.shape
        or left.shape[1] not in (3, 4)
        or left.shape[-1] % 2
    ):
        raise ValueError("Half-SBS Metal expects matching MPS float32 BCHW RGB/RGBA with even width")

    left = left.contiguous()
    right = right.contiguous()
    batches, channels, height, width = map(int, left.shape)
    output = torch.empty_like(left)
    params = _half_sbs_params(width, height, channels, batches)
    _half_sbs_srgb_library().d2s_make_half_sbs_srgb(
        left,
        right,
        output,
        params,
        threads=output.numel(),
        group_size=256,
    )
    return output
