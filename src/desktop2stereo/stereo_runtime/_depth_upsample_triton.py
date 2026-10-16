from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _smoothstep(value, edge0, edge1):
    t = tl.minimum(tl.maximum((value - edge0) / (edge1 - edge0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


@triton.jit
def _depth_curvature(depth, base, x, y, height: tl.constexpr, width: tl.constexpr, active):
    # One-sided second differences keep a plane flat at image boundaries.
    mx = tl.minimum(tl.maximum(x, 1), width - 2)
    my = tl.minimum(tl.maximum(y, 1), height - 2)
    left = tl.load(depth + base + y * width + tl.maximum(mx - 1, 0), active, 0.0)
    center_x = tl.load(depth + base + y * width + tl.maximum(mx, 0), active, 0.0)
    right = tl.load(depth + base + y * width + tl.minimum(mx + 1, width - 1), active, 0.0)
    above = tl.load(depth + base + tl.maximum(my - 1, 0) * width + x, active, 0.0)
    center_y = tl.load(depth + base + tl.maximum(my, 0) * width + x, active, 0.0)
    below = tl.load(depth + base + tl.minimum(my + 1, height - 1) * width + x, active, 0.0)
    return tl.maximum(
        tl.where(width >= 3, tl.abs(left - 2.0 * center_x + right), 0.0),
        tl.where(height >= 3, tl.abs(above - 2.0 * center_y + below), 0.0),
    )


@triton.jit
def _joint_bilateral_upsample_kernel(
    depth,
    rgb,
    guide,
    output,
    depth_height: tl.constexpr,
    depth_width: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    pixels: tl.constexpr,
    batches: tl.constexpr,
    color_sigma: tl.constexpr,
    edge_threshold: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    total_pixels = pixels * batches
    active = offsets < total_pixels
    batch = offsets // pixels
    pixel = offsets % pixels
    x = pixel % width
    y = pixel // width
    dx = tl.maximum(
        tl.minimum((x.to(tl.float32) + 0.5) * depth_width / width - 0.5,
                   depth_width - 1.0), 0.0
    )
    dy = tl.maximum(
        tl.minimum((y.to(tl.float32) + 0.5) * depth_height / height - 0.5,
                   depth_height - 1.0), 0.0
    )
    x0 = dx.to(tl.int32)
    y0 = dy.to(tl.int32)
    x1 = tl.minimum(x0 + 1, depth_width - 1)
    y1 = tl.minimum(y0 + 1, depth_height - 1)
    fx = dx - x0
    fy = dy - y0
    i00 = y0 * depth_width + x0
    i10 = y0 * depth_width + x1
    i01 = y1 * depth_width + x0
    i11 = y1 * depth_width + x1

    depth_base = batch * depth_height * depth_width
    d00 = tl.load(depth + depth_base + i00, active, 0.0)
    d10 = tl.load(depth + depth_base + i10, active, 0.0)
    d01 = tl.load(depth + depth_base + i01, active, 0.0)
    d11 = tl.load(depth + depth_base + i11, active, 0.0)
    w00 = (1.0 - fy) * (1.0 - fx)
    w10 = (1.0 - fy) * fx
    w01 = fy * (1.0 - fx)
    w11 = fy * fx
    bilinear = w00 * d00 + w10 * d10 + w01 * d01 + w11 * d11

    cx = tl.floor(dx + 0.5).to(tl.int32)
    cy = tl.floor(dy + 0.5).to(tl.int32)
    tl.store(output + offsets, bilinear, active)
    # Interpolate curvature too: a nearest-grid, binary decision makes small
    # frame-to-frame depth changes switch a whole model texel's reconstruction.
    edge_signal = (
        w00 * _depth_curvature(depth, depth_base, x0, y0, depth_height, depth_width, active)
        + w10 * _depth_curvature(depth, depth_base, x1, y0, depth_height, depth_width, active)
        + w01 * _depth_curvature(depth, depth_base, x0, y1, depth_height, depth_width, active)
        + w11 * _depth_curvature(depth, depth_base, x1, y1, depth_height, depth_width, active)
    )
    edge_weight = _smoothstep(edge_signal, max(edge_threshold, 1.0e-6) * 0.25,
                            max(edge_threshold, 1.0e-6) * 0.5)
    candidates = active & (edge_weight > 0.0)
    if tl.sum(candidates.to(tl.int32), 0) > 0:
        active = candidates
        # The nine model-grid taps have compact, continuous spatial support.
        # A tap reaches zero before the window moves to the next texel, so a
        # moving silhouette cannot acquire a model-grid step from class selection.
        rgb_base = batch * 3 * pixels
        guide_pixels = depth_height * depth_width
        guide_base = batch * 3 * guide_pixels
        tr = tl.load(rgb + rgb_base + pixel, active, 0.0)
        tg = tl.load(rgb + rgb_base + pixels + pixel, active, 0.0)
        tb = tl.load(rgb + rgb_base + 2 * pixels + pixel, active, 0.0)
        spatial_total = tl.full((block,), 0.0, tl.float32)
        depth_sum = tl.full((block,), 0.0, tl.float32)
        depth_square_sum = tl.full((block,), 0.0, tl.float32)
        red_sum = tl.full((block,), 0.0, tl.float32)
        green_sum = tl.full((block,), 0.0, tl.float32)
        blue_sum = tl.full((block,), 0.0, tl.float32)
        color_square_sum = tl.full((block,), 0.0, tl.float32)
        total = tl.full((block,), 0.0, tl.float32)
        guided_sum = tl.full((block,), 0.0, tl.float32)
        max_log_weight = tl.full((block,), -1.0e30, tl.float32)
        low_count = tl.full((block,), 0.0, tl.float32)
        high_count = tl.full((block,), 0.0, tl.float32)
        low_delta_sum = tl.full((block,), 0.0, tl.float32)
        high_delta_sum = tl.full((block,), 0.0, tl.float32)
        low_total = tl.full((block,), 0.0, tl.float32)
        high_total = tl.full((block,), 0.0, tl.float32)
        low_sum = tl.full((block,), 0.0, tl.float32)
        high_sum = tl.full((block,), 0.0, tl.float32)
        matching_distance_sum = tl.full((block,), 0.0, tl.float32)
        sigma = tl.maximum(color_sigma / 3.0, 1.0e-4)
        threshold = tl.maximum(edge_threshold, 1.0e-6)
        for oy in tl.static_range(-1, 2):
            for ox in tl.static_range(-1, 2):
                sx = tl.minimum(tl.maximum(cx + ox, 0), depth_width - 1)
                sy = tl.minimum(tl.maximum(cy + oy, 0), depth_height - 1)
                spatial = (1.0 - _smoothstep(tl.abs(dx - cx - ox), 0.5, 1.5)) * (
                    1.0 - _smoothstep(tl.abs(dy - cy - oy), 0.5, 1.5)
                )
                index = sy * depth_width + sx
                sample = tl.load(depth + depth_base + index, active, 0.0)
                red = tl.load(guide + guide_base + index, active, 0.0)
                green = tl.load(guide + guide_base + guide_pixels + index, active, 0.0)
                blue = tl.load(guide + guide_base + 2 * guide_pixels + index, active, 0.0)
                distance = ((tr - red) * (tr - red) + (tg - green) * (tg - green)
                            + (tb - blue) * (tb - blue)) / 3.0
                log_weight = tl.where(spatial > 0.0,
                    tl.log(tl.maximum(spatial, 1.0e-30)) - distance / (2.0 * sigma * sigma), -1.0e30)
                new_max = tl.maximum(max_log_weight, log_weight)
                rescale = tl.exp(max_log_weight - new_max)
                weight = tl.where(spatial > 0.0, tl.exp(log_weight - new_max), 0.0)
                total = total * rescale + weight
                guided_sum = guided_sum * rescale + weight * sample
                matching_distance_sum = matching_distance_sum * rescale + weight * distance
                max_log_weight = new_max
                delta = (tl.abs(tr - red) + tl.abs(tg - green) + tl.abs(tb - blue)) / 3.0
                low = 1.0 - _smoothstep(sample - bilinear, -0.5 * threshold, 0.5 * threshold)
                high = 1.0 - low
                low_count += spatial * low
                high_count += spatial * high
                low_delta_sum += spatial * low * delta
                high_delta_sum += spatial * high * delta
                low_total = low_total * rescale + weight * low
                high_total = high_total * rescale + weight * high
                low_sum = low_sum * rescale + weight * low * sample
                high_sum = high_sum * rescale + weight * high * sample
                spatial_total += spatial
                depth_sum += spatial * sample
                depth_square_sum += spatial * sample * sample
                red_sum += spatial * red
                green_sum += spatial * green
                blue_sum += spatial * blue
                color_square_sum += spatial * (red * red + green * green + blue * blue) / 3.0
        denominator = tl.maximum(spatial_total, 1.0e-8)
        mean_depth = depth_sum / denominator
        mean_red = red_sum / denominator
        mean_green = green_sum / denominator
        mean_blue = blue_sum / denominator
        depth_variance = tl.maximum(depth_square_sum / denominator - mean_depth * mean_depth, 0.0)
        color_variance = tl.maximum(color_square_sum / denominator - (
            mean_red * mean_red + mean_green * mean_green + mean_blue * mean_blue
        ) / 3.0, 0.0)
        confidence = _smoothstep(depth_variance, threshold * threshold * 0.0625, threshold * threshold * 0.25)
        confidence *= _smoothstep(color_variance, 0.000025, 0.0004)
        guided = tl.where(total > 1.0e-8, guided_sum / tl.maximum(total, 1.0e-8), bilinear)
        low_delta = low_delta_sum / tl.maximum(low_count, 1.0e-8)
        high_delta = high_delta_sum / tl.maximum(high_count, 1.0e-8)
        prefer_high = _smoothstep(low_delta - high_delta, -0.02, 0.02)
        selected_low = low_sum / tl.maximum(low_total, 1.0e-8)
        selected_high = high_sum / tl.maximum(high_total, 1.0e-8)
        selected = selected_low + prefer_high * (selected_high - selected_low)
        support = _smoothstep(tl.minimum(low_count, high_count), 0.001, 0.1)
        class_confidence = _smoothstep(tl.abs(low_delta - high_delta), 0.0025, 0.03) * support
        class_confidence *= 1.0 - _smoothstep(matching_distance_sum / tl.maximum(total, 1.0e-8),
                                             0.000025, 0.0004)
        guided += class_confidence * (selected - guided)
        value = bilinear + edge_weight * confidence * (guided - bilinear)
        tl.store(output + offsets, tl.minimum(tl.maximum(value, 0.0), 1.0), active)


def joint_bilateral_upsample(
    depth: torch.Tensor,
    rgb: torch.Tensor,
    guide: torch.Tensor,
    height: int,
    width: int,
    *,
    color_sigma: float = 0.12,
    depth_edge_threshold: float = 0.04,
) -> torch.Tensor:
    if depth.ndim != 4 or depth.shape[1] != 1 or rgb.ndim != 4 or rgb.shape[1] < 3:
        raise ValueError("expected B1HW depth and BCHW RGB tensors")
    batch, _, depth_height, depth_width = depth.shape
    if rgb.shape[0] != batch or guide.shape != (batch, 3, depth_height, depth_width):
        raise ValueError("RGB, guide, and depth tensor shapes do not match")
    if rgb.shape[-2:] != (height, width):
        raise ValueError("RGB tensor must already match output size")
    depth = depth.contiguous()
    rgb = rgb[:, :3].contiguous()
    guide = guide.contiguous()
    output = torch.empty((batch, 1, height, width), device=depth.device, dtype=torch.float32)
    pixels = height * width
    block = 256 if torch.version.hip else 64
    _joint_bilateral_upsample_kernel[(triton.cdiv(pixels * batch, block),)](
        depth,
        rgb,
        guide,
        output,
        depth_height,
        depth_width,
        height,
        width,
        pixels,
        batch,
        float(color_sigma),
        float(depth_edge_threshold),
        block,
        num_warps=4 if torch.version.hip else 2,
    )
    return output
