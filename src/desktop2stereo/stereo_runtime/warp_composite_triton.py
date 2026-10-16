from __future__ import annotations

import torch
import triton
import triton.language as tl

from .triton_runtime import triton_runtime_available


@triton.jit
def _smoothstep(value, edge0, edge1):
    t = tl.minimum(tl.maximum((value - edge0) / (edge1 - edge0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


@triton.jit
def _edge_mask(depth, base_shift, pixel, x, y, width: tl.constexpr,
               height: tl.constexpr, depth_value, shift, active):
    left_pixel = y * width + tl.maximum(x - 1, 0)
    right_pixel = y * width + tl.minimum(x + 1, width - 1)
    up_pixel = tl.maximum(y - 1, 0) * width + x
    down_pixel = tl.minimum(y + 1, height - 1) * width + x
    depth_left = tl.load(depth + left_pixel, mask=active & (x > 0), other=depth_value)
    depth_right = tl.load(depth + right_pixel, mask=active & (x + 1 < width), other=depth_value)
    depth_up = tl.load(depth + up_pixel, mask=active & (y > 0), other=depth_value)
    depth_down = tl.load(depth + down_pixel, mask=active & (y + 1 < height), other=depth_value)
    shift_left = tl.load(base_shift + left_pixel, mask=active & (x > 0), other=shift)
    shift_right = tl.load(base_shift + right_pixel, mask=active & (x + 1 < width), other=shift)
    shift_up = tl.load(base_shift + up_pixel, mask=active & (y > 0), other=shift)
    shift_down = tl.load(base_shift + down_pixel, mask=active & (y + 1 < height), other=shift)
    depth_edge = tl.maximum(
        tl.maximum(tl.abs(depth_left - depth_value), tl.abs(depth_right - depth_value)),
        tl.maximum(tl.abs(depth_up - depth_value), tl.abs(depth_down - depth_value)),
    )
    shift_edge = tl.maximum(
        tl.maximum(tl.abs(shift_left - shift), tl.abs(shift_right - shift)),
        tl.maximum(tl.abs(shift_up - shift), tl.abs(shift_down - shift)),
    )
    depth_confidence = _smoothstep(depth_edge, 0.0125, 0.05)
    shift_confidence = _smoothstep(shift_edge, 0.125, 0.5)
    return tl.where(active, tl.maximum(depth_confidence, shift_confidence), 0.0)


@triton.jit
def _edge_depth_class(rgb, depth, pixel, x, y, width: tl.constexpr,
                      height: tl.constexpr, pixels: tl.constexpr,
                      depth_value, active):
    """Return -1 for a smooth/unreliable pixel, otherwise 0/1 for its RGB class."""
    left_pixel = y * width + tl.maximum(x - 1, 0)
    right_pixel = y * width + tl.minimum(x + 1, width - 1)
    up_pixel = tl.maximum(y - 1, 0) * width + x
    down_pixel = tl.minimum(y + 1, height - 1) * width + x
    left_depth = tl.load(depth + left_pixel, mask=active & (x > 0), other=depth_value)
    right_depth = tl.load(depth + right_pixel, mask=active & (x + 1 < width), other=depth_value)
    up_depth = tl.load(depth + up_pixel, mask=active & (y > 0), other=depth_value)
    down_depth = tl.load(depth + down_pixel, mask=active & (y + 1 < height), other=depth_value)
    low_depth = tl.minimum(tl.minimum(left_depth, right_depth),
                           tl.minimum(up_depth, down_depth))
    high_depth = tl.maximum(tl.maximum(left_depth, right_depth),
                            tl.maximum(up_depth, down_depth))
    midpoint = (low_depth + high_depth) * 0.5

    center_luma = _luma_at(rgb, pixel, pixels, active)
    left_luma = _luma_at(rgb, left_pixel, pixels, active)
    right_luma = _luma_at(rgb, right_pixel, pixels, active)
    up_luma = _luma_at(rgb, up_pixel, pixels, active)
    down_luma = _luma_at(rgb, down_pixel, pixels, active)
    left_low = 1.0 - _smoothstep(left_depth, midpoint - 0.02, midpoint + 0.02)
    right_low = 1.0 - _smoothstep(right_depth, midpoint - 0.02, midpoint + 0.02)
    up_low = 1.0 - _smoothstep(up_depth, midpoint - 0.02, midpoint + 0.02)
    down_low = 1.0 - _smoothstep(down_depth, midpoint - 0.02, midpoint + 0.02)
    left_high = 1.0 - left_low
    right_high = 1.0 - right_low
    up_high = 1.0 - up_low
    down_high = 1.0 - down_low
    low_count = left_low + right_low + up_low + down_low
    high_count = left_high + right_high + up_high + down_high
    low_luma = (
        left_luma * left_low
        + right_luma * right_low
        + up_luma * up_low
        + down_luma * down_low
    ) / tl.maximum(low_count, 1.0e-6)
    high_luma = (
        left_luma * left_high
        + right_luma * right_high
        + up_luma * up_high
        + down_luma * down_high
    ) / tl.maximum(high_count, 1.0e-6)
    color_contrast = tl.abs(high_luma - low_luma)
    depth_confidence = _smoothstep(high_depth - low_depth, 0.0125, 0.05)
    color_confidence = _smoothstep(color_contrast, 0.01, 0.05)
    low_distance = tl.abs(center_luma - low_luma)
    high_distance = tl.abs(center_luma - high_luma)
    prefer_high = _smoothstep(low_distance - high_distance, -0.02, 0.02)
    confidence = tl.where(active, depth_confidence * color_confidence, 0.0)
    return prefer_high, confidence


@triton.jit
def _luma_at(rgb, pixel, pixels: tl.constexpr, active):
    red = tl.load(rgb + pixel, mask=active, other=0.0)
    green = tl.load(rgb + pixels + pixel, mask=active, other=0.0)
    blue = tl.load(rgb + 2 * pixels + pixel, mask=active, other=0.0)
    return red * 0.299 + green * 0.587 + blue * 0.114


@triton.jit
def _edge_layer_weights(rgb, depth, pixel, x, y, width: tl.constexpr,
                        height: tl.constexpr, pixels: tl.constexpr,
                        depth_value, active, w0, w1):
    prefer_high, confidence = _edge_depth_class(
        rgb, depth, pixel, x, y, width, height, pixels, depth_value, active
    )
    return (
        w0 * (1.0 - confidence) + (1.0 - prefer_high) * confidence,
        w1 * (1.0 - confidence) + prefer_high * confidence,
    )


@triton.jit
def _warp_composite2_kernel(
    rgb,
    depth,
    base_shift,
    left,
    right,
    total: tl.constexpr,
    width: tl.constexpr,
    height: tl.constexpr,
    pixels: tl.constexpr,
    softness: tl.constexpr,
    projection_min: tl.constexpr,
    projection_max: tl.constexpr,
    edge_aa_enabled: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < total
    pixel = offsets % pixels
    y = pixel // width
    x = pixel - y * width
    channel = offsets // pixels

    depth_value = tl.load(depth + pixel, mask=active, other=0.0)
    w0_raw = tl.exp(-((depth_value - 0.0) * (depth_value - 0.0)) / softness)
    w1_raw = tl.exp(-((depth_value - 1.0) * (depth_value - 1.0)) / softness)
    wsum = w0_raw + w1_raw
    w0 = w0_raw / wsum
    w1 = w1_raw / wsum
    w0, w1 = _edge_layer_weights(
        rgb, depth, pixel, x, y, width, height, pixels, depth_value, active, w0, w1
    )

    shift = tl.load(base_shift + pixel, mask=active, other=0.0)
    left_value = _sample_two_layers(
        rgb, depth, base_shift, depth_value, channel, y, x, shift, 0.875, 1.0, width, pixels, active, w0, w1,
        projection_min, projection_max,
        )
    right_value = _sample_two_layers(
        rgb, depth, base_shift, depth_value, channel, y, x, shift, -0.875, -1.0, width, pixels, active, w0, w1,
        projection_min, projection_max,
        )
    if edge_aa_enabled:
        edge_confidence = _edge_mask(
            depth, base_shift, pixel, x, y, width, height, depth_value, shift, active
        )
        edge = active & (edge_confidence > 0.0)
        if tl.sum(edge.to(tl.int32), 0) > 0:
            left_filtered = _edge_supersample_two_layers(
                rgb, depth, depth_value, channel, x, y, shift, 0.875, 1.0,
                w0, w1, width, pixels, height, edge
            )
            right_filtered = _edge_supersample_two_layers(
                rgb, depth, depth_value, channel, x, y, shift, -0.875, -1.0,
                w0, w1, width, pixels, height, edge
            )
            left_value += (left_filtered - left_value) * edge_confidence
            right_value += (right_filtered - right_value) * edge_confidence
    tl.store(left + offsets, left_value, mask=active)
    tl.store(right + offsets, right_value, mask=active)


@triton.jit
def _warp_composite2_rgba_u8_kernel(
    rgb,
    depth,
    base_shift,
    left,
    right,
    pixels: tl.constexpr,
    width: tl.constexpr,
    softness: tl.constexpr,
    projection_min: tl.constexpr,
    projection_max: tl.constexpr,
    edge_aa_enabled: tl.constexpr,
    block: tl.constexpr,
):
    pixel = tl.program_id(0) * block + tl.arange(0, block)
    active = pixel < pixels
    y = pixel // width
    x = pixel - y * width

    depth_value = tl.load(depth + pixel, mask=active, other=0.0)
    w0_raw = tl.exp(-(depth_value * depth_value) / softness)
    depth_from_one = depth_value - 1.0
    w1_raw = tl.exp(-(depth_from_one * depth_from_one) / softness)
    weight_sum = w0_raw + w1_raw
    w0 = w0_raw / weight_sum
    w1 = w1_raw / weight_sum
    w0, w1 = _edge_layer_weights(
        rgb, depth, pixel, x, y, width, pixels // width, pixels,
        depth_value, active, w0, w1
    )
    shift = tl.load(base_shift + pixel, mask=active, other=0.0)

    if projection_min != 0.0 or projection_max != 0.0:
        left_visibility = _dibr_visibility_at(depth, base_shift, x, y, depth_value, shift,
                                               1.0, width, active, projection_min, projection_max)
        left_r = _sample_dibr_visibility(rgb, depth, base_shift, 0, x, y, 1.0,
                                                    width, pixels, active, left_visibility)
        left_g = _sample_dibr_visibility(rgb, depth, base_shift, 1, x, y, 1.0,
                                                    width, pixels, active, left_visibility)
        left_b = _sample_dibr_visibility(rgb, depth, base_shift, 2, x, y, 1.0,
                                                    width, pixels, active, left_visibility)
        right_visibility = _dibr_visibility_at(depth, base_shift, x, y, depth_value, shift,
                                               -1.0, width, active, projection_min, projection_max)
        right_r = _sample_dibr_visibility(rgb, depth, base_shift, 0, x, y, -1.0,
                                                    width, pixels, active, right_visibility)
        right_g = _sample_dibr_visibility(rgb, depth, base_shift, 1, x, y, -1.0,
                                                    width, pixels, active, right_visibility)
        right_b = _sample_dibr_visibility(rgb, depth, base_shift, 2, x, y, -1.0,
                                                    width, pixels, active, right_visibility)
    else:
        left_r = _sample_two_layers(
            rgb, depth, base_shift, depth_value, 0, y, x, shift, 0.875, 1.0, width, pixels, active, w0, w1,
            projection_min, projection_max,
            )
        left_g = _sample_two_layers(
            rgb, depth, base_shift, depth_value, 1, y, x, shift, 0.875, 1.0, width, pixels, active, w0, w1,
            projection_min, projection_max,
            )
        left_b = _sample_two_layers(
            rgb, depth, base_shift, depth_value, 2, y, x, shift, 0.875, 1.0, width, pixels, active, w0, w1,
            projection_min, projection_max,
            )
        right_r = _sample_two_layers(
            rgb, depth, base_shift, depth_value, 0, y, x, shift, -0.875, -1.0, width, pixels, active, w0, w1,
            projection_min, projection_max,
            )
        right_g = _sample_two_layers(
            rgb, depth, base_shift, depth_value, 1, y, x, shift, -0.875, -1.0, width, pixels, active, w0, w1,
            projection_min, projection_max,
            )
        right_b = _sample_two_layers(
            rgb, depth, base_shift, depth_value, 2, y, x, shift, -0.875, -1.0, width, pixels, active, w0, w1,
            projection_min, projection_max,
            )

    if edge_aa_enabled:
        edge_confidence = _edge_mask(
            depth, base_shift, pixel, x, y, width, pixels // width,
            depth_value, shift, active
        )
        edge = active & (edge_confidence > 0.0)
        if tl.sum(edge.to(tl.int32), 0) > 0:
            filtered_left_r = _edge_supersample_two_layers(rgb, depth, depth_value, 0, x, y, shift, 0.875, 1.0, w0, w1, width, pixels, pixels // width, edge)
            filtered_left_g = _edge_supersample_two_layers(rgb, depth, depth_value, 1, x, y, shift, 0.875, 1.0, w0, w1, width, pixels, pixels // width, edge)
            filtered_left_b = _edge_supersample_two_layers(rgb, depth, depth_value, 2, x, y, shift, 0.875, 1.0, w0, w1, width, pixels, pixels // width, edge)
            filtered_right_r = _edge_supersample_two_layers(rgb, depth, depth_value, 0, x, y, shift, -0.875, -1.0, w0, w1, width, pixels, pixels // width, edge)
            filtered_right_g = _edge_supersample_two_layers(rgb, depth, depth_value, 1, x, y, shift, -0.875, -1.0, w0, w1, width, pixels, pixels // width, edge)
            filtered_right_b = _edge_supersample_two_layers(rgb, depth, depth_value, 2, x, y, shift, -0.875, -1.0, w0, w1, width, pixels, pixels // width, edge)
            left_r += (filtered_left_r - left_r) * edge_confidence
            left_g += (filtered_left_g - left_g) * edge_confidence
            left_b += (filtered_left_b - left_b) * edge_confidence
            right_r += (filtered_right_r - right_r) * edge_confidence
            right_g += (filtered_right_g - right_g) * edge_confidence
            right_b += (filtered_right_b - right_b) * edge_confidence

    output_offset = pixel * 4
    tl.store(left + output_offset, _rgba_u8(left_r), mask=active)
    tl.store(left + output_offset + 1, _rgba_u8(left_g), mask=active)
    tl.store(left + output_offset + 2, _rgba_u8(left_b), mask=active)
    tl.store(left + output_offset + 3, 255, mask=active)
    tl.store(right + output_offset, _rgba_u8(right_r), mask=active)
    tl.store(right + output_offset + 1, _rgba_u8(right_g), mask=active)
    tl.store(right + output_offset + 2, _rgba_u8(right_b), mask=active)
    tl.store(right + output_offset + 3, 255, mask=active)


@triton.jit
def _warp_composite2_full_sbs_kernel(
    rgb,
    depth,
    base_shift,
    out,
    total: tl.constexpr,
    width: tl.constexpr,
    out_width: tl.constexpr,
    pixels: tl.constexpr,
    softness: tl.constexpr,
    projection_min: tl.constexpr,
    projection_max: tl.constexpr,
    edge_aa_enabled: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < total
    pixel = offsets % (pixels * 2)
    y = pixel // out_width
    x = pixel - y * out_width
    channel = offsets // (pixels * 2)

    use_left = x < width
    source_x = tl.where(use_left, x, x - width)
    source_pixel = y * width + source_x
    depth_value = tl.load(depth + source_pixel, mask=active, other=0.0)
    w0_raw = tl.exp(-((depth_value - 0.0) * (depth_value - 0.0)) / softness)
    w1_raw = tl.exp(-((depth_value - 1.0) * (depth_value - 1.0)) / softness)
    wsum = w0_raw + w1_raw
    w0 = w0_raw / wsum
    w1 = w1_raw / wsum
    w0, w1 = _edge_layer_weights(
        rgb, depth, source_pixel, source_x, y, width, pixels // width,
        pixels, depth_value, active, w0, w1
    )
    shift = tl.load(base_shift + source_pixel, mask=active, other=0.0)
    left_value = _sample_two_layers(
        rgb, depth, base_shift, depth_value, channel, y, source_x, shift, 0.875, 1.0, width, pixels, active, w0,
        w1, projection_min, projection_max,
        )
    right_value = _sample_two_layers(
        rgb, depth, base_shift, depth_value, channel, y, source_x, shift, -0.875, -1.0, width, pixels, active, w0,
        w1, projection_min, projection_max,
        )
    if edge_aa_enabled:
        edge = _edge_mask(depth, base_shift, source_pixel, source_x, y, width,
                          pixels // width, depth_value, shift, active)
        if tl.sum(edge.to(tl.int32), 0) > 0:
            left_value = tl.where(edge, _edge_supersample_two_layers(
                rgb, depth, depth_value, channel, source_x, y, shift,
                0.875, 1.0, w0, w1, width, pixels, pixels // width, edge), left_value)
            right_value = tl.where(edge, _edge_supersample_two_layers(
                rgb, depth, depth_value, channel, source_x, y, shift,
                -0.875, -1.0, w0, w1, width, pixels, pixels // width, edge), right_value)
    tl.store(out + offsets, tl.where(use_left, left_value, right_value), mask=active)


@triton.jit
def _composite_weights_at(
    rgb,
    depth,
    base_shift,
    y,
    x,
    width: tl.constexpr,
    pixels: tl.constexpr,
    softness: tl.constexpr,
    active,
):
    source_pixel = y * width + x
    depth_value = tl.load(depth + source_pixel, mask=active, other=0.0)
    w0_raw = tl.exp(-(depth_value * depth_value) / softness)
    depth_from_one = depth_value - 1.0
    w1_raw = tl.exp(-(depth_from_one * depth_from_one) / softness)
    weight_sum = w0_raw + w1_raw
    shift = tl.load(base_shift + source_pixel, mask=active, other=0.0)
    w0, w1 = _edge_layer_weights(
        rgb, depth, source_pixel, x, y, width, pixels // width, pixels,
        depth_value, active, w0_raw / weight_sum, w1_raw / weight_sum
    )
    return depth_value, shift, w0, w1


@triton.jit
def _warp_composite2_half_sbs_kernel(
    rgb,
    depth,
    base_shift,
    out,
    total: tl.constexpr,
    width: tl.constexpr,
    half_width: tl.constexpr,
    pixels: tl.constexpr,
    softness: tl.constexpr,
    projection_min: tl.constexpr,
    projection_max: tl.constexpr,
    output_uint8: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < total
    pixel = offsets % pixels
    y = pixel // width
    x = pixel - y * width

    use_left = x < half_width
    source_x = tl.where(use_left, x, x - half_width)
    x0 = source_x * 2
    x1 = x0 + 1

    scale0 = tl.where(use_left, 0.875, -0.875)
    scale1 = tl.where(use_left, 1.0, -1.0)
    depth0, shift0, w00, w01 = _composite_weights_at(
        rgb, depth, base_shift, y, x0, width, pixels, softness, active
    )
    depth1, shift1, w10, w11 = _composite_weights_at(
        rgb, depth, base_shift, y, x1, width, pixels, softness, active
    )
    if projection_min != 0.0 or projection_max != 0.0:
        visibility0 = _dibr_visibility_at(depth, base_shift, x0, y, depth0, shift0,
                                         scale1, width, active, projection_min, projection_max)
        visibility1 = _dibr_visibility_at(depth, base_shift, x1, y, depth1, shift1,
                                         scale1, width, active, projection_min, projection_max)
    # Depth classification, disparity and visibility are shared by RGB.
    for channel in tl.static_range(3):
        if projection_min != 0.0 or projection_max != 0.0:
            value0 = _sample_dibr_visibility(rgb, depth, base_shift, channel, x0, y,
                                            scale1, width, pixels, active, visibility0)
            value1 = _sample_dibr_visibility(rgb, depth, base_shift, channel, x1, y,
                                            scale1, width, pixels, active, visibility1)
        else:
            value0 = _sample_two_layers(
                rgb, depth, base_shift, depth0, channel, y, x0, shift0, scale0, scale1,
                width, pixels, active, w00, w01, projection_min, projection_max
            )
            value1 = _sample_two_layers(
                rgb, depth, base_shift, depth1, channel, y, x1, shift1, scale0, scale1,
                width, pixels, active, w10, w11, projection_min, projection_max
            )
        linear0 = tl.where(value0 <= 0.04045, value0 / 12.92,
            tl.exp2(2.4 * tl.log2(tl.maximum((value0 + 0.055) / 1.055, 1.0e-20))))
        linear1 = tl.where(value1 <= 0.04045, value1 / 12.92,
            tl.exp2(2.4 * tl.log2(tl.maximum((value1 + 0.055) / 1.055, 1.0e-20))))
        linear_mean = (linear0 + linear1) * 0.5
        value = tl.where(linear_mean <= 0.0031308, linear_mean * 12.92,
            1.055 * tl.exp2(tl.log2(tl.maximum(linear_mean, 1.0e-20)) / 2.4) - 0.055)
        if output_uint8:
            scaled = tl.minimum(tl.maximum(value, 0.0), 1.0) * 255.0
            lower = tl.floor(scaled)
            fraction = scaled - lower
            # Match torch.round's ties-to-even before storing the packed bytes.
            round_up = (fraction > 0.5) | ((fraction == 0.5) & ((lower.to(tl.int32) & 1) != 0))
            value = lower + round_up.to(tl.float32)
        tl.store(out + channel * pixels + offsets, value, mask=active)


@triton.jit
def _rgba_u8(value):
    return (tl.minimum(tl.maximum(value, 0.0), 1.0) * 255.0).to(tl.uint8)


@triton.jit
def _sample_two_layers(
    rgb, depth, base_shift, class_depth, channel, y, x, shift, scale0, scale1, width: tl.constexpr,
    pixels: tl.constexpr, active, w0, w1, projection_min: tl.constexpr, projection_max: tl.constexpr,
):
    if projection_min != 0.0 or projection_max != 0.0:
        return _dibr_warp_at(rgb, depth, base_shift, channel, x, y, class_depth, shift,
                             scale1, width, pixels, active, projection_min, projection_max)
    x0 = x + shift * scale0
    x1 = x + shift * scale1
    v0 = _sample_border_linear_class(rgb, depth, class_depth, channel, y, x0, width, pixels, active)
    v1 = _sample_border_linear_class(rgb, depth, class_depth, channel, y, x1, width, pixels, active)
    return v0 * w0 + v1 * w1


@triton.jit
def _projection_scale(value):
    return 0.875 + 0.125 / (1.0 + tl.exp((1.0 - 2.0 * value) / 0.08))


@triton.jit
def _depth_shift_at(depth, base_shift, x, y, width: tl.constexpr, active):
    sx = tl.minimum(tl.maximum(x, 0.0), width - 1.0)
    x0 = tl.floor(sx).to(tl.int32)
    x1 = tl.minimum(x0 + 1, width - 1)
    fraction = sx - x0
    index = y * width
    d0 = tl.load(depth + index + x0, active, 0.0)
    d1 = tl.load(depth + index + x1, active, 0.0)
    s0 = tl.load(base_shift + index + x0, active, 0.0)
    s1 = tl.load(base_shift + index + x1, active, 0.0)
    return d0 + fraction * (d1 - d0), s0 + fraction * (s1 - s0)


@triton.jit
def _dibr_visibility_at(depth, base_shift, x, y, target_depth, target_shift,
                        eye, width: tl.constexpr, active,
                        projection_min: tl.constexpr, projection_max: tl.constexpr):
    # Geometry is independent of color. Share the source visibility search
    # across RGB instead of traversing the entire ray once for each channel.
    source_x = tl.minimum(tl.maximum(x + target_shift * _projection_scale(target_depth) * eye, 0.0), width - 1.0)
    source_depth, source_shift = _depth_shift_at(depth, base_shift, source_x, y, width, active)
    refined_x = tl.minimum(tl.maximum(x + source_shift * _projection_scale(source_depth) * eye, 0.0), width - 1.0)
    front_x = tl.minimum(tl.maximum(x + projection_min * eye, 0.0), width - 1.0)
    front_depth, _ = _depth_shift_at(depth, base_shift, front_x, y, width, active)
    confidence = tl.maximum(_smoothstep(tl.abs(source_depth - target_depth), 0.02, 0.04),
                            _smoothstep(tl.abs(refined_x - source_x), 0.125, 0.5))
    confidence = tl.maximum(confidence, _smoothstep(front_depth - source_depth, 0.02, 0.04))
    needs_gather = active & (confidence > 0.0)
    first = tl.ceil(x + tl.minimum(projection_min * eye, projection_max * eye) - 1.0).to(tl.int32)
    best_depth = tl.full(x.shape, -1.0, tl.float32)
    best_coverage = tl.full(x.shape, 0.0, tl.float32)
    best_x = x.to(tl.int32)
    left_error = tl.full(x.shape, 1.0e6, tl.float32)
    right_error = tl.full(x.shape, 1.0e6, tl.float32)
    left_depth = tl.full(x.shape, -1.0, tl.float32)
    right_depth = tl.full(x.shape, -1.0, tl.float32)
    left_x = x.to(tl.int32)
    right_x = x.to(tl.int32)
    if tl.sum(needs_gather.to(tl.int32), 0) > 0:
        for offset in range(int(projection_max - projection_min) + 4):
            sx = first + offset
            valid = needs_gather & (sx >= 0) & (sx < width)
            index = y * width + sx
            sample_depth = tl.load(depth + index, valid, 0.0)
            sample_shift = tl.load(base_shift + index, valid, 0.0)
            error = sx - sample_shift * _projection_scale(sample_depth) * eye - x
            coverage = tl.maximum(1.0 - tl.abs(error), 0.0)
            best = valid & (coverage > 0.0) & ((sample_depth > best_depth)
                | ((sample_depth == best_depth) & (coverage > best_coverage)))
            best_depth = tl.where(best, sample_depth, best_depth)
            best_coverage = tl.where(best, coverage, best_coverage)
            best_x = tl.where(best, sx, best_x)
            left = valid & (error <= 0.0) & (tl.abs(error) < left_error)
            right = valid & (error >= 0.0) & (error < right_error)
            left_error = tl.where(left, tl.abs(error), left_error)
            right_error = tl.where(right, error, right_error)
            left_depth = tl.where(left, sample_depth, left_depth)
            right_depth = tl.where(right, sample_depth, right_depth)
            left_x = tl.where(left, sx, left_x)
            right_x = tl.where(right, sx, right_x)
    return (refined_x, confidence, needs_gather, best_depth, best_x, left_error, right_error, left_depth, right_depth, left_x, right_x)


@triton.jit
def _sample_dibr_visibility(rgb, depth, base_shift, channel, x, y, eye,
                            width: tl.constexpr, pixels: tl.constexpr, active, visibility):
    (refined_x, confidence, needs_gather, best_depth, best_x, left_error, right_error, left_depth, right_depth, left_x, right_x) = visibility
    value = _sample_border_linear(rgb, channel, y, refined_x, width, pixels, active)
    if tl.sum(needs_gather.to(tl.int32), 0) > 0:
        use_left = (left_error < 1.0e5) & ((right_error >= 1.0e5) | (left_depth <= right_depth))
        background_x = tl.where(use_left, left_x, right_x)
        background_depth = tl.where(use_left, left_depth, right_depth)
        left_color = _sample_border_linear(rgb, channel, y, left_x, width, pixels, needs_gather)
        right_color = _sample_border_linear(rgb, channel, y, right_x, width, pixels, needs_gather)
        # Interpolate a stretched continuous surface in projected space. A
        # binary lower-depth donor flips on tiny depth changes and stripes the
        # uncovered pixels; only a real depth edge needs one-sided extension.
        side_fraction = left_error / tl.maximum(left_error + right_error, 1.0e-6)
        background_linear = left_color + side_fraction * (right_color - left_color)
        background_edge = _smoothstep(tl.abs(left_depth - right_depth), 0.02, 0.04)
        background_near = tl.where(use_left, left_color, right_color)
        background = background_linear + background_edge * (background_near - background_linear)
        foreground = tl.full(x.shape, 0.0, tl.float32)
        foreground_coverage = tl.full(x.shape, 0.0, tl.float32)
        for offset in tl.static_range(-1, 2):
            sx = best_x + offset
            valid = needs_gather & (best_depth >= 0.0) & (sx >= 0) & (sx < width)
            sample_depth = tl.load(depth + y * width + sx, valid, 0.0)
            sample_shift = tl.load(base_shift + y * width + sx, valid, 0.0)
            error = sx - sample_shift * _projection_scale(sample_depth) * eye - x
            class_weight = 1.0 - _smoothstep(best_depth - sample_depth, 0.02, 0.04)
            coverage = tl.where(valid, tl.maximum(1.0 - tl.abs(error), 0.0) * class_weight, 0.0)
            sample = tl.load(rgb + channel * pixels + y * width + sx, valid, 0.0)
            foreground += sample * coverage
            foreground_coverage += coverage
        foreground /= tl.maximum(foreground_coverage, 1.0e-6)
        alpha = tl.minimum(foreground_coverage, 1.0)
        background_confidence = _smoothstep(best_depth - background_depth, 0.02, 0.04)
        resolved = foreground + (1.0 - alpha) * background_confidence * (background - foreground)
        resolved = tl.where(best_depth < 0.0, background, resolved)
        value += tl.where(needs_gather, confidence * (resolved - value), 0.0)
    return value


@triton.jit
def _dibr_warp_at(rgb, depth, base_shift, channel, x, y, target_depth, target_shift,
                  eye, width: tl.constexpr, pixels: tl.constexpr, active,
                  projection_min: tl.constexpr, projection_max: tl.constexpr):
    visibility = _dibr_visibility_at(depth, base_shift, x, y, target_depth, target_shift,
                                     eye, width, active, projection_min, projection_max)
    return _sample_dibr_visibility(rgb, depth, base_shift, channel, x, y, eye,
                                   width, pixels, active, visibility)


@triton.jit
def _warp_composite2_rgba_u8_tiled_kernel(
    rgb,
    depth,
    base_shift,
    left,
    right,
    width: tl.constexpr,
    height: tl.constexpr,
    projection_min: tl.constexpr,
    projection_max: tl.constexpr,
    tile_x: tl.constexpr,
    tile_y: tl.constexpr,
):
    lanes = tl.arange(0, tile_x * tile_y)
    x = tl.program_id(0) * tile_x + lanes % tile_x
    y = tl.program_id(1) * tile_y + lanes // tile_x
    active = (x < width) & (y < height)
    pixel = y * width + x
    depth_value = tl.load(depth + pixel, mask=active, other=0.0)
    shift = tl.load(base_shift + pixel, mask=active, other=0.0)
    left_visibility = _dibr_visibility_at(
        depth, base_shift, x, y, depth_value, shift, 1.0, width, active,
        projection_min, projection_max,
    )
    right_visibility = _dibr_visibility_at(
        depth, base_shift, x, y, depth_value, shift, -1.0, width, active,
        projection_min, projection_max,
    )
    for channel in tl.static_range(3):
        left_value = _sample_dibr_visibility(
            rgb, depth, base_shift, channel, x, y, 1.0, width,
            width * height, active, left_visibility,
        )
        right_value = _sample_dibr_visibility(
            rgb, depth, base_shift, channel, x, y, -1.0, width,
            width * height, active, right_visibility,
        )
        tl.store(left + pixel * 4 + channel, _rgba_u8(left_value), mask=active)
        tl.store(right + pixel * 4 + channel, _rgba_u8(right_value), mask=active)
    tl.store(left + pixel * 4 + 3, 255, mask=active)
    tl.store(right + pixel * 4 + 3, 255, mask=active)


@triton.jit
def _warp_composite2_half_sbs_tiled_kernel(
    rgb,
    depth,
    base_shift,
    out,
    width: tl.constexpr,
    height: tl.constexpr,
    half_width: tl.constexpr,
    pixels: tl.constexpr,
    softness: tl.constexpr,
    projection_min: tl.constexpr,
    projection_max: tl.constexpr,
    output_uint8: tl.constexpr,
    tile_x: tl.constexpr,
    tile_y: tl.constexpr,
):
    lanes = tl.arange(0, tile_x * tile_y)
    x = tl.program_id(0) * tile_x + lanes % tile_x
    y = tl.program_id(1) * tile_y + lanes // tile_x
    active = (x < width) & (y < height)
    pixel = y * width + x
    use_left = x < half_width
    source_x = tl.where(use_left, x, x - half_width)
    x0 = source_x * 2
    x1 = x0 + 1
    scale0 = tl.where(use_left, 0.875, -0.875)
    scale1 = tl.where(use_left, 1.0, -1.0)
    depth0, shift0, w00, w01 = _composite_weights_at(
        rgb, depth, base_shift, y, x0, width, pixels, softness, active
    )
    depth1, shift1, w10, w11 = _composite_weights_at(
        rgb, depth, base_shift, y, x1, width, pixels, softness, active
    )
    if projection_min != 0.0 or projection_max != 0.0:
        visibility0 = _dibr_visibility_at(
            depth, base_shift, x0, y, depth0, shift0, scale1, width, active,
            projection_min, projection_max,
        )
        visibility1 = _dibr_visibility_at(
            depth, base_shift, x1, y, depth1, shift1, scale1, width, active,
            projection_min, projection_max,
        )
    for channel in tl.static_range(3):
        if projection_min != 0.0 or projection_max != 0.0:
            value0 = _sample_dibr_visibility(
                rgb, depth, base_shift, channel, x0, y, scale1, width,
                pixels, active, visibility0,
            )
            value1 = _sample_dibr_visibility(
                rgb, depth, base_shift, channel, x1, y, scale1, width,
                pixels, active, visibility1,
            )
        else:
            value0 = _sample_two_layers(
                rgb, depth, base_shift, depth0, channel, y, x0, shift0,
                scale0, scale1, width, pixels, active, w00, w01,
                projection_min, projection_max,
            )
            value1 = _sample_two_layers(
                rgb, depth, base_shift, depth1, channel, y, x1, shift1,
                scale0, scale1, width, pixels, active, w10, w11,
                projection_min, projection_max,
            )
        linear0 = tl.where(
            value0 <= 0.04045,
            value0 / 12.92,
            tl.exp2(2.4 * tl.log2(tl.maximum((value0 + 0.055) / 1.055, 1.0e-20))),
        )
        linear1 = tl.where(
            value1 <= 0.04045,
            value1 / 12.92,
            tl.exp2(2.4 * tl.log2(tl.maximum((value1 + 0.055) / 1.055, 1.0e-20))),
        )
        linear_mean = (linear0 + linear1) * 0.5
        value = tl.where(
            linear_mean <= 0.0031308,
            linear_mean * 12.92,
            1.055 * tl.exp2(tl.log2(tl.maximum(linear_mean, 1.0e-20)) / 2.4) - 0.055,
        )
        if output_uint8:
            scaled = tl.minimum(tl.maximum(value, 0.0), 1.0) * 255.0
            lower = tl.floor(scaled)
            fraction = scaled - lower
            round_up = (fraction > 0.5) | (
                (fraction == 0.5) & ((lower.to(tl.int32) & 1) != 0)
            )
            value = lower + round_up.to(tl.float32)
        tl.store(out + channel * pixels + pixel, value, mask=active)

@triton.jit
def _sample_border_linear(rgb, channel, y, sample_x, width: tl.constexpr, pixels: tl.constexpr, active):
    # Reflection mirrors the source at both boundaries and turns a displaced
    # edge into a visible second edge.  Border sampling extends the nearest
    # real pixel without introducing that mirrored geometry.
    x_clamped = tl.minimum(tl.maximum(sample_x, 0.0), width - 1.0)
    x0_float = tl.floor(x_clamped)
    x0 = x0_float.to(tl.int64)
    x1 = tl.minimum(x0 + 1, width - 1)
    frac = x_clamped - x0_float
    base = channel * pixels + y * width
    v0 = tl.load(rgb + base + x0, mask=active, other=0.0)
    v1 = tl.load(rgb + base + x1, mask=active, other=0.0)
    return v0 + (v1 - v0) * frac


@triton.jit
def _sample_border_linear_class(rgb, depth, class_depth, channel, y, sample_x,
                                width: tl.constexpr, pixels: tl.constexpr,
                                active):
    x_clamped = tl.minimum(tl.maximum(sample_x, 0.0), width - 1.0)
    x0_float = tl.floor(x_clamped)
    x0 = x0_float.to(tl.int64)
    x1 = tl.minimum(x0 + 1, width - 1)
    frac = x_clamped - x0_float
    base = channel * pixels + y * width
    v0 = tl.load(rgb + base + x0, mask=active, other=0.0)
    v1 = tl.load(rgb + base + x1, mask=active, other=0.0)
    d0 = tl.load(depth + y * width + x0, mask=active, other=class_depth)
    d1 = tl.load(depth + y * width + x1, mask=active, other=class_depth)
    edge_confidence = _smoothstep(tl.abs(d0 - d1), 0.02, 0.06)
    keep0 = 1.0 - _smoothstep(tl.abs(d0 - class_depth), 0.02, 0.06)
    keep1 = 1.0 - _smoothstep(tl.abs(d1 - class_depth), 0.02, 0.06)
    class_weight = keep0 * (1.0 - frac) + keep1 * frac
    class_value = (
        v0 * keep0 * (1.0 - frac)
        + v1 * keep1 * frac
    ) / tl.maximum(class_weight, 1.0e-6)
    linear_value = v0 + (v1 - v0) * frac
    confidence = edge_confidence * tl.minimum(class_weight, 1.0)
    return linear_value + (class_value - linear_value) * confidence


@triton.jit
def _sample_rgb_bilinear(rgb, channel, x, y, width: tl.constexpr, height: tl.constexpr, pixels: tl.constexpr, active):
    clamped_x = tl.minimum(tl.maximum(x, 0.0), width - 1.0)
    clamped_y = tl.minimum(tl.maximum(y, 0.0), height - 1.0)
    x0f = tl.floor(clamped_x)
    y0f = tl.floor(clamped_y)
    x0 = x0f.to(tl.int64)
    y0 = y0f.to(tl.int64)
    x1 = tl.minimum(x0 + 1, width - 1)
    y1 = tl.minimum(y0 + 1, height - 1)
    tx = clamped_x - x0f
    ty = clamped_y - y0f
    base = channel * pixels
    p00 = tl.load(rgb + base + y0 * width + x0, mask=active, other=0.0)
    p10 = tl.load(rgb + base + y0 * width + x1, mask=active, other=0.0)
    p01 = tl.load(rgb + base + y1 * width + x0, mask=active, other=0.0)
    p11 = tl.load(rgb + base + y1 * width + x1, mask=active, other=0.0)
    top = p00 + (p10 - p00) * tx
    bottom = p01 + (p11 - p01) * tx
    return top + (bottom - top) * ty


@triton.jit
def _sample_rgb_bilinear_class(
    rgb, depth, class_depth, channel, x, y,
    width: tl.constexpr, height: tl.constexpr, pixels: tl.constexpr, active
):
    """Bilinear color sample restricted to the center depth class at edges."""
    clamped_x = tl.minimum(tl.maximum(x, 0.0), width - 1.0)
    clamped_y = tl.minimum(tl.maximum(y, 0.0), height - 1.0)
    x0f = tl.floor(clamped_x)
    y0f = tl.floor(clamped_y)
    x0 = x0f.to(tl.int64)
    y0 = y0f.to(tl.int64)
    x1 = tl.minimum(x0 + 1, width - 1)
    y1 = tl.minimum(y0 + 1, height - 1)
    tx = clamped_x - x0f
    ty = clamped_y - y0f
    base = channel * pixels
    p00 = tl.load(rgb + base + y0 * width + x0, mask=active, other=0.0)
    p10 = tl.load(rgb + base + y0 * width + x1, mask=active, other=0.0)
    p01 = tl.load(rgb + base + y1 * width + x0, mask=active, other=0.0)
    p11 = tl.load(rgb + base + y1 * width + x1, mask=active, other=0.0)
    d00 = tl.load(depth + y0 * width + x0, mask=active, other=class_depth)
    d10 = tl.load(depth + y0 * width + x1, mask=active, other=class_depth)
    d01 = tl.load(depth + y1 * width + x0, mask=active, other=class_depth)
    d11 = tl.load(depth + y1 * width + x1, mask=active, other=class_depth)
    depth_min = tl.minimum(tl.minimum(d00, d10), tl.minimum(d01, d11))
    depth_max = tl.maximum(tl.maximum(d00, d10), tl.maximum(d01, d11))
    edge_confidence = _smoothstep(depth_max - depth_min, 0.02, 0.06)
    w00 = (1.0 - tx) * (1.0 - ty)
    w10 = tx * (1.0 - ty)
    w01 = (1.0 - tx) * ty
    w11 = tx * ty
    keep00 = 1.0 - _smoothstep(tl.abs(d00 - class_depth), 0.02, 0.06)
    keep10 = 1.0 - _smoothstep(tl.abs(d10 - class_depth), 0.02, 0.06)
    keep01 = 1.0 - _smoothstep(tl.abs(d01 - class_depth), 0.02, 0.06)
    keep11 = 1.0 - _smoothstep(tl.abs(d11 - class_depth), 0.02, 0.06)
    class_weight = (
        keep00 * w00 + keep10 * w10 + keep01 * w01 + keep11 * w11
    )
    class_value = (
        p00 * keep00 * w00
        + p10 * keep10 * w10
        + p01 * keep01 * w01
        + p11 * keep11 * w11
    ) / tl.maximum(class_weight, 1.0e-6)
    top = p00 + (p10 - p00) * tx
    bottom = p01 + (p11 - p01) * tx
    linear_value = top + (bottom - top) * ty
    confidence = edge_confidence * tl.minimum(class_weight, 1.0)
    return linear_value + (class_value - linear_value) * confidence


@triton.jit
def _sample_subpixel_two_layers(
    rgb, depth, class_depth, channel, x, y, shift, scale0, scale1, w0, w1,
    width: tl.constexpr, height: tl.constexpr, pixels: tl.constexpr, active
):
    sample0 = _sample_rgb_bilinear_class(
        rgb, depth, class_depth, channel, x + shift * scale0, y,
        width, height, pixels, active
    )
    sample1 = _sample_rgb_bilinear_class(
        rgb, depth, class_depth, channel, x + shift * scale1, y,
        width, height, pixels, active
    )
    return sample0 * w0 + sample1 * w1


@triton.jit
def _edge_supersample_two_layers(
    rgb, depth, class_depth, channel, x, y, shift, scale0, scale1, w0, w1,
    width: tl.constexpr, pixels: tl.constexpr, height: tl.constexpr, active
):
    # Resolve all color taps with the center pixel's disparity and layer mix.
    # Every subpixel uses the center depth class, so a tap cannot reintroduce
    # the opposite side of a contour as a second displaced edge.
    total = _sample_subpixel_two_layers(
        rgb, depth, class_depth, channel, x - 0.25, y - 0.25, shift,
        scale0, scale1, w0, w1, width, height, pixels, active
    )
    total += _sample_subpixel_two_layers(
        rgb, depth, class_depth, channel, x + 0.25, y - 0.25, shift,
        scale0, scale1, w0, w1, width, height, pixels, active
    )
    total += _sample_subpixel_two_layers(
        rgb, depth, class_depth, channel, x - 0.25, y + 0.25, shift,
        scale0, scale1, w0, w1, width, height, pixels, active
    )
    total += _sample_subpixel_two_layers(
        rgb, depth, class_depth, channel, x + 0.25, y + 0.25, shift,
        scale0, scale1, w0, w1, width, height, pixels, active
    )
    return total * 0.25


def can_use_triton_warp_composite2(rgb: torch.Tensor, depth: torch.Tensor, base_shift: torch.Tensor, *, layers: int, symmetric: bool) -> bool:
    return (
        layers == 2
        and symmetric
        and triton_runtime_available(rgb.device)
        and rgb.dtype == torch.float32
        and depth.dtype == torch.float32
        and base_shift.dtype == torch.float32
        and rgb.ndim == 4
        and depth.ndim == 4
        and base_shift.ndim == 4
        and rgb.shape[0] == 1
        and rgb.shape[1] == 3
        and depth.shape[0] == 1
        and depth.shape[1] == 1
        and base_shift.shape == depth.shape
        and rgb.shape[-2:] == depth.shape[-2:]
    )


def warp_composite2(rgb: torch.Tensor, depth: torch.Tensor, base_shift: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    projection_min, projection_max = getattr(base_shift, "_d2s_projection_bounds", (0.0, 0.0))
    rgb = rgb.contiguous()
    depth = depth.contiguous()
    base_shift = base_shift.contiguous()
    left = torch.empty_like(rgb)
    right = torch.empty_like(rgb)
    _, _, height, width = rgb.shape
    pixels = height * width
    total = rgb.numel()
    block = 256
    grid = (triton.cdiv(total, block),)
    _warp_composite2_kernel[grid](rgb, depth, base_shift, left, right, total,
        width, height, pixels, 0.08, projection_min, projection_max, False, block)
    return left, right


def warp_composite2_half_sbs(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    base_shift: torch.Tensor,
    *,
    output_uint8: bool = False,
) -> torch.Tensor:
    projection_min, projection_max = getattr(base_shift, "_d2s_projection_bounds", (0.0, 0.0))
    rgb = rgb.contiguous()
    depth = depth.contiguous()
    base_shift = base_shift.contiguous()
    _, channels, height, width = rgb.shape
    if width % 2:
        raise ValueError("half-SBS direct output requires an even width")
    out = torch.empty_like(rgb, dtype=torch.uint8 if output_uint8 else rgb.dtype)
    pixels = height * width
    tile_x, tile_y = 32, 4
    grid = (triton.cdiv(width, tile_x), triton.cdiv(height, tile_y))
    _warp_composite2_half_sbs_tiled_kernel[grid](
        rgb,
        depth,
        base_shift,
        out,
        width,
        height,
        width // 2,
        pixels,
        0.08,
        projection_min,
        projection_max,
        output_uint8,
        tile_x,
        tile_y,
        num_warps=4,
    )
    return out


def warp_composite2_full_sbs(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    base_shift: torch.Tensor,
) -> torch.Tensor:
    projection_min, projection_max = getattr(base_shift, "_d2s_projection_bounds", (0.0, 0.0))
    rgb = rgb.contiguous()
    depth = depth.contiguous()
    base_shift = base_shift.contiguous()
    _, channels, height, width = rgb.shape
    out = torch.empty((1, channels, height, width * 2), device=rgb.device, dtype=rgb.dtype)
    pixels = height * width
    total = out.numel()
    block = 256
    grid = (triton.cdiv(total, block),)
    _warp_composite2_full_sbs_kernel[
        grid
    ](rgb, depth, base_shift, out, total, width, width * 2, pixels, 0.08, projection_min, projection_max,
       False, block)
    return out


def warp_composite2_rgba_u8(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    base_shift: torch.Tensor,
    *,
    edge_aa_enabled: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    projection_min, projection_max = getattr(base_shift, "_d2s_projection_bounds", (0.0, 0.0))
    rgb = rgb.contiguous()
    depth = depth.contiguous()
    base_shift = base_shift.contiguous()
    _, _, height, width = rgb.shape
    left = torch.empty((height, width, 4), device=rgb.device, dtype=torch.uint8)
    right = torch.empty_like(left)
    pixels = height * width
    if projection_min != 0.0 or projection_max != 0.0:
        tile_x, tile_y = 32, 4
        grid = (triton.cdiv(width, tile_x), triton.cdiv(height, tile_y))
        _warp_composite2_rgba_u8_tiled_kernel[grid](
            rgb,
            depth,
            base_shift,
            left,
            right,
            width,
            height,
            projection_min,
            projection_max,
            tile_x,
            tile_y,
            num_warps=4,
        )
        return left, right
    block = 256
    grid = (triton.cdiv(pixels, block),)
    _warp_composite2_rgba_u8_kernel[
        grid
    ](rgb, depth, base_shift, left, right, pixels, width, 0.08, projection_min, projection_max,
       bool(edge_aa_enabled and projection_min == projection_max == 0.0), block)
    return left, right
