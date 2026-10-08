from __future__ import annotations

import torch
import triton
import triton.language as tl

from .triton_runtime import triton_runtime_available


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
    return active & ((depth_edge > 0.025) | (shift_edge > 0.25))


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
    low_count = (
        (left_depth < midpoint).to(tl.float32)
        + (right_depth < midpoint).to(tl.float32)
        + (up_depth < midpoint).to(tl.float32)
        + (down_depth < midpoint).to(tl.float32)
    )
    high_count = 4.0 - low_count
    low_luma = (
        left_luma * (left_depth < midpoint).to(tl.float32)
        + right_luma * (right_depth < midpoint).to(tl.float32)
        + up_luma * (up_depth < midpoint).to(tl.float32)
        + down_luma * (down_depth < midpoint).to(tl.float32)
    ) / tl.maximum(low_count, 1.0)
    high_luma = (
        left_luma * (left_depth >= midpoint).to(tl.float32)
        + right_luma * (right_depth >= midpoint).to(tl.float32)
        + up_luma * (up_depth >= midpoint).to(tl.float32)
        + down_luma * (down_depth >= midpoint).to(tl.float32)
    ) / tl.maximum(high_count, 1.0)
    color_contrast = tl.abs(high_luma - low_luma)
    use_class = active & (high_depth - low_depth > 0.025) & (low_count > 0.0) \
        & (high_count > 0.0) & (color_contrast >= 0.02)
    choose_high = tl.abs(center_luma - high_luma) <= tl.abs(center_luma - low_luma)
    return tl.where(use_class, choose_high.to(tl.int32), tl.full_like(x, -1))


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
    classification = _edge_depth_class(
        rgb, depth, pixel, x, y, width, height, pixels, depth_value, active
    )
    hard_low = classification == 0
    hard_high = classification == 1
    return (
        tl.where(hard_low, 1.0, tl.where(hard_high, 0.0, w0)),
        tl.where(hard_low, 0.0, tl.where(hard_high, 1.0, w1)),
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
    left_value = _sample_two_layers(rgb, depth, depth_value, channel, y, x, shift, 0.875, 1.0, width, pixels, active, w0, w1)
    right_value = _sample_two_layers(rgb, depth, depth_value, channel, y, x, shift, -0.875, -1.0, width, pixels, active, w0, w1)
    if edge_aa_enabled:
        edge = _edge_mask(depth, base_shift, pixel, x, y, width, height, depth_value, shift, active)
        if tl.sum(edge.to(tl.int32), 0) > 0:
            left_value = tl.where(
                edge,
                _edge_supersample_two_layers(
                    rgb, depth, depth_value, channel, x, y, shift, 0.875, 1.0,
                    w0, w1, width, pixels, height, edge
                ),
                left_value,
            )
            right_value = tl.where(
                edge,
                _edge_supersample_two_layers(
                    rgb, depth, depth_value, channel, x, y, shift, -0.875, -1.0,
                    w0, w1, width, pixels, height, edge
                ),
                right_value,
            )
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

    left_r = _sample_two_layers(rgb, depth, depth_value, 0, y, x, shift, 0.875, 1.0, width, pixels, active, w0, w1)
    left_g = _sample_two_layers(rgb, depth, depth_value, 1, y, x, shift, 0.875, 1.0, width, pixels, active, w0, w1)
    left_b = _sample_two_layers(rgb, depth, depth_value, 2, y, x, shift, 0.875, 1.0, width, pixels, active, w0, w1)
    right_r = _sample_two_layers(rgb, depth, depth_value, 0, y, x, shift, -0.875, -1.0, width, pixels, active, w0, w1)
    right_g = _sample_two_layers(rgb, depth, depth_value, 1, y, x, shift, -0.875, -1.0, width, pixels, active, w0, w1)
    right_b = _sample_two_layers(rgb, depth, depth_value, 2, y, x, shift, -0.875, -1.0, width, pixels, active, w0, w1)

    if edge_aa_enabled:
        edge = _edge_mask(depth, base_shift, pixel, x, y, width, pixels // width,
                          depth_value, shift, active)
        if tl.sum(edge.to(tl.int32), 0) > 0:
            left_r = tl.where(edge, _edge_supersample_two_layers(rgb, depth, depth_value, 0, x, y, shift, 0.875, 1.0, w0, w1, width, pixels, pixels // width, edge), left_r)
            left_g = tl.where(edge, _edge_supersample_two_layers(rgb, depth, depth_value, 1, x, y, shift, 0.875, 1.0, w0, w1, width, pixels, pixels // width, edge), left_g)
            left_b = tl.where(edge, _edge_supersample_two_layers(rgb, depth, depth_value, 2, x, y, shift, 0.875, 1.0, w0, w1, width, pixels, pixels // width, edge), left_b)
            right_r = tl.where(edge, _edge_supersample_two_layers(rgb, depth, depth_value, 0, x, y, shift, -0.875, -1.0, w0, w1, width, pixels, pixels // width, edge), right_r)
            right_g = tl.where(edge, _edge_supersample_two_layers(rgb, depth, depth_value, 1, x, y, shift, -0.875, -1.0, w0, w1, width, pixels, pixels // width, edge), right_g)
            right_b = tl.where(edge, _edge_supersample_two_layers(rgb, depth, depth_value, 2, x, y, shift, -0.875, -1.0, w0, w1, width, pixels, pixels // width, edge), right_b)

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
    left_value = _sample_two_layers(rgb, depth, depth_value, channel, y, source_x, shift, 0.875, 1.0, width, pixels, active, w0, w1)
    right_value = _sample_two_layers(rgb, depth, depth_value, channel, y, source_x, shift, -0.875, -1.0, width, pixels, active, w0, w1)
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
def _sample_composite_at(
    rgb,
    depth,
    base_shift,
    channel,
    y,
    x,
    scale0,
    scale1,
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
    return _sample_two_layers(
        rgb,
        depth,
        depth_value,
        channel,
        y,
        x,
        shift,
        scale0,
        scale1,
        width,
        pixels,
        active,
        w0,
        w1,
    )


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
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < total
    pixel = offsets % pixels
    y = pixel // width
    x = pixel - y * width
    channel = offsets // pixels

    use_left = x < half_width
    source_x = tl.where(use_left, x, x - half_width)
    x0 = source_x * 2
    x1 = x0 + 1

    left0 = _sample_composite_at(rgb, depth, base_shift, channel, y, x0, 0.875, 1.0, width, pixels, softness, active)
    left1 = _sample_composite_at(rgb, depth, base_shift, channel, y, x1, 0.875, 1.0, width, pixels, softness, active)
    right0 = _sample_composite_at(rgb, depth, base_shift, channel, y, x0, -0.875, -1.0, width, pixels, softness, active)
    right1 = _sample_composite_at(rgb, depth, base_shift, channel, y, x1, -0.875, -1.0, width, pixels, softness, active)
    left_value = (left0 + left1) * 0.5
    right_value = (right0 + right1) * 0.5
    value = tl.where(use_left, left_value, right_value)
    tl.store(out + offsets, value, mask=active)


@triton.jit
def _rgba_u8(value):
    return (tl.minimum(tl.maximum(value, 0.0), 1.0) * 255.0).to(tl.uint8)


@triton.jit
def _sample_two_layers(rgb, depth, class_depth, channel, y, x, shift, scale0, scale1, width: tl.constexpr, pixels: tl.constexpr, active, w0, w1):
    x0 = x + shift * scale0
    x1 = x + shift * scale1
    v0 = _sample_border_linear_class(rgb, depth, class_depth, channel, y, x0, width, pixels, active)
    v1 = _sample_border_linear_class(rgb, depth, class_depth, channel, y, x1, width, pixels, active)
    return v0 * w0 + v1 * w1


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
    edge = tl.abs(d0 - d1) > 0.04
    keep0 = tl.abs(d0 - class_depth) <= 0.04
    keep1 = tl.abs(d1 - class_depth) <= 0.04
    class_weight = tl.where(keep0, 1.0 - frac, 0.0) + tl.where(keep1, frac, 0.0)
    class_value = (
        v0 * tl.where(keep0, 1.0 - frac, 0.0)
        + v1 * tl.where(keep1, frac, 0.0)
    ) / tl.maximum(class_weight, 1.0e-6)
    linear_value = v0 + (v1 - v0) * frac
    return tl.where(edge & (class_weight > 1.0e-6), class_value, linear_value)


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
    edge = depth_max - depth_min > 0.04
    w00 = (1.0 - tx) * (1.0 - ty)
    w10 = tx * (1.0 - ty)
    w01 = (1.0 - tx) * ty
    w11 = tx * ty
    keep00 = tl.abs(d00 - class_depth) <= 0.04
    keep10 = tl.abs(d10 - class_depth) <= 0.04
    keep01 = tl.abs(d01 - class_depth) <= 0.04
    keep11 = tl.abs(d11 - class_depth) <= 0.04
    class_weight = (
        tl.where(keep00, w00, 0.0) + tl.where(keep10, w10, 0.0)
        + tl.where(keep01, w01, 0.0) + tl.where(keep11, w11, 0.0)
    )
    class_value = (
        p00 * tl.where(keep00, w00, 0.0)
        + p10 * tl.where(keep10, w10, 0.0)
        + p01 * tl.where(keep01, w01, 0.0)
        + p11 * tl.where(keep11, w11, 0.0)
    ) / tl.maximum(class_weight, 1.0e-6)
    top = p00 + (p10 - p00) * tx
    bottom = p01 + (p11 - p01) * tx
    linear_value = top + (bottom - top) * ty
    return tl.where(edge & (class_weight > 1.0e-6), class_value, linear_value)


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
        width, height, pixels, 0.08, False, block)
    return left, right


def warp_composite2_half_sbs(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    base_shift: torch.Tensor,
) -> torch.Tensor:
    rgb = rgb.contiguous()
    depth = depth.contiguous()
    base_shift = base_shift.contiguous()
    _, channels, height, width = rgb.shape
    if width % 2:
        raise ValueError("half-SBS direct output requires an even width")
    out = torch.empty_like(rgb)
    pixels = height * width
    total = out.numel()
    block = 256
    grid = (triton.cdiv(total, block),)
    _warp_composite2_half_sbs_kernel[
        grid
    ](rgb, depth, base_shift, out, total, width, width // 2, pixels, 0.08, block)
    return out


def warp_composite2_full_sbs(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    base_shift: torch.Tensor,
) -> torch.Tensor:
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
    ](rgb, depth, base_shift, out, total, width, width * 2, pixels, 0.08,
       False, block)
    return out


def warp_composite2_rgba_u8(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    base_shift: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rgb = rgb.contiguous()
    depth = depth.contiguous()
    base_shift = base_shift.contiguous()
    _, _, height, width = rgb.shape
    left = torch.empty((height, width, 4), device=rgb.device, dtype=torch.uint8)
    right = torch.empty_like(left)
    pixels = height * width
    block = 256
    grid = (triton.cdiv(pixels, block),)
    _warp_composite2_rgba_u8_kernel[
        grid
    ](rgb, depth, base_shift, left, right, pixels, width, 0.08,
       False, block)
    return left, right
