from __future__ import annotations

import torch
import triton
import triton.language as tl

from .triton_runtime import triton_runtime_available


@triton.jit
def _motion_luma_kernel(
    rgb,
    out,
    height: tl.constexpr,
    width: tl.constexpr,
    stride_c: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    flow_height: tl.constexpr,
    flow_width: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < flow_height * flow_width
    x = offsets % flow_width
    y = offsets // flow_width
    x0 = x * width // flow_width
    x1 = tl.maximum((x + 1) * width // flow_width, x0 + 1)
    y0 = y * height // flow_height
    y1 = tl.maximum((y + 1) * height // flow_height, y0 + 1)
    luma_sum = tl.full((block,), 0.0, tl.float32)
    for sample_y in tl.static_range(0, 8):
        iy = tl.minimum(y0 + ((2 * sample_y + 1) * (y1 - y0)) // 16, y1 - 1)
        for sample_x in tl.static_range(0, 8):
            ix = tl.minimum(x0 + ((2 * sample_x + 1) * (x1 - x0)) // 16, x1 - 1)
            pixel = iy * stride_h + ix * stride_w
            red = tl.load(rgb + pixel, mask=active, other=0.0)
            green = tl.load(rgb + stride_c + pixel, mask=active, other=0.0)
            blue = tl.load(rgb + 2 * stride_c + pixel, mask=active, other=0.0)
            luma_sum += red * 0.299 + green * 0.587 + blue * 0.114
    tl.store(out + offsets, luma_sum * (1.0 / 64.0), mask=active)


@triton.jit
def _motion_flow_kernel(
    current,
    previous,
    flow_x,
    flow_y,
    confidence,
    height: tl.constexpr,
    width: tl.constexpr,
    block: tl.constexpr,
    radius: tl.constexpr = 2,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < height * width
    x = offsets % width
    y = offsets // width
    best = tl.full((block,), 1.0e6, tl.float32)
    second = tl.full((block,), 1.0e6, tl.float32)
    best_x = tl.full((block,), 0, tl.int32)
    best_y = tl.full((block,), 0, tl.int32)
    for dy in tl.static_range(-radius, radius + 1):
        for dx in tl.static_range(-radius, radius + 1):
            cost = tl.full((block,), 0.0, tl.float32)
            for patch_y in tl.static_range(-1, 2):
                for patch_x in tl.static_range(-1, 2):
                    current_x = tl.minimum(tl.maximum(x + patch_x, 0), width - 1)
                    current_y = tl.minimum(tl.maximum(y + patch_y, 0), height - 1)
                    previous_x = tl.minimum(tl.maximum(x + dx + patch_x, 0), width - 1)
                    previous_y = tl.minimum(tl.maximum(y + dy + patch_y, 0), height - 1)
                    current_value = tl.load(
                        current + current_y * width + current_x,
                        mask=active,
                        other=0.0,
                    )
                    previous_value = tl.load(
                        previous + previous_y * width + previous_x,
                        mask=active,
                        other=0.0,
                    )
                    cost += tl.abs(current_value - previous_value)
            cost *= 1.0 / 9.0
            is_better = cost < best
            second = tl.minimum(second, tl.where(is_better, best, cost))
            best = tl.minimum(best, cost)
            best_x = tl.where(is_better, dx, best_x)
            best_y = tl.where(is_better, dy, best_y)
    uniqueness = tl.minimum(tl.maximum((second - best) / (second + 1.0e-4), 0.0), 1.0)
    match_confidence = tl.exp(-best * 30.0) * uniqueness
    match_confidence = tl.where((best < 0.08) & (uniqueness > 0.05), match_confidence, 0.0)
    tl.store(flow_x + offsets, best_x.to(tl.float32), mask=active)
    tl.store(flow_y + offsets, best_y.to(tl.float32), mask=active)
    tl.store(confidence + offsets, match_confidence, mask=active)


@triton.jit
def _motion_bilinear_sample(
    image,
    x,
    y,
    width: tl.constexpr,
    height: tl.constexpr,
):
    x = tl.minimum(tl.maximum(x, 0.0), width - 1.0)
    y = tl.minimum(tl.maximum(y, 0.0), height - 1.0)
    x0_float = tl.floor(x)
    y0_float = tl.floor(y)
    fraction_x = x - x0_float
    fraction_y = y - y0_float
    x0 = x0_float.to(tl.int32)
    y0 = y0_float.to(tl.int32)
    x1 = tl.minimum(x0 + 1, width - 1)
    y1 = tl.minimum(y0 + 1, height - 1)
    top_left = tl.load(image + y0 * width + x0)
    top_right = tl.load(image + y0 * width + x1)
    bottom_left = tl.load(image + y1 * width + x0)
    bottom_right = tl.load(image + y1 * width + x1)
    top = top_left * (1.0 - fraction_x) + top_right * fraction_x
    bottom = bottom_left * (1.0 - fraction_x) + bottom_right * fraction_x
    return top * (1.0 - fraction_y) + bottom * fraction_y


@triton.jit
def _motion_confidence_kernel(
    forward_x,
    forward_y,
    forward_confidence,
    reverse_x,
    reverse_y,
    reverse_confidence,
    out,
    height: tl.constexpr,
    width: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < height * width
    x = tl.minimum(offsets % width, width - 1).to(tl.float32)
    y = tl.minimum(offsets // width, height - 1).to(tl.float32)
    dx = tl.load(forward_x + offsets, mask=active, other=0.0)
    dy = tl.load(forward_y + offsets, mask=active, other=0.0)
    reverse_dx = _motion_bilinear_sample(reverse_x, x + dx, y + dy, width, height)
    reverse_dy = _motion_bilinear_sample(reverse_y, x + dx, y + dy, width, height)
    reverse_match = _motion_bilinear_sample(reverse_confidence, x + dx, y + dy, width, height)
    forward_backward_error = tl.sqrt((dx + reverse_dx) * (dx + reverse_dx) + (dy + reverse_dy) * (dy + reverse_dy))
    confidence = tl.load(forward_confidence + offsets, mask=active, other=0.0)
    confidence *= reverse_match * tl.exp(-forward_backward_error * 1.5)
    confidence = tl.where(forward_backward_error < 1.5, confidence, 0.0)
    tl.store(out + offsets, confidence, mask=active)


@triton.jit
def _motion_edge_aware_depth_sample(
    image,
    x,
    y,
    current_value,
    width: tl.constexpr,
    height: tl.constexpr,
    edge_threshold: tl.constexpr,
):
    x = tl.minimum(tl.maximum(x, 0.0), width - 1.0)
    y = tl.minimum(tl.maximum(y, 0.0), height - 1.0)
    x0_float = tl.floor(x)
    y0_float = tl.floor(y)
    fraction_x = x - x0_float
    fraction_y = y - y0_float
    x0 = x0_float.to(tl.int32)
    y0 = y0_float.to(tl.int32)
    x1 = tl.minimum(x0 + 1, width - 1)
    y1 = tl.minimum(y0 + 1, height - 1)
    top_left = tl.load(image + y0 * width + x0)
    top_right = tl.load(image + y0 * width + x1)
    bottom_left = tl.load(image + y1 * width + x0)
    bottom_right = tl.load(image + y1 * width + x1)
    weight_top_left = (1.0 - fraction_x) * (1.0 - fraction_y)
    weight_top_right = fraction_x * (1.0 - fraction_y)
    weight_bottom_left = (1.0 - fraction_x) * fraction_y
    weight_bottom_right = fraction_x * fraction_y
    weight_top_left = tl.where(tl.abs(top_left - current_value) <= edge_threshold, weight_top_left, 0.0)
    weight_top_right = tl.where(tl.abs(top_right - current_value) <= edge_threshold, weight_top_right, 0.0)
    weight_bottom_left = tl.where(
        tl.abs(bottom_left - current_value) <= edge_threshold,
        weight_bottom_left,
        0.0,
    )
    weight_bottom_right = tl.where(
        tl.abs(bottom_right - current_value) <= edge_threshold,
        weight_bottom_right,
        0.0,
    )
    weight_sum = weight_top_left + weight_top_right + weight_bottom_left + weight_bottom_right
    weighted_depth = (
        top_left * weight_top_left
        + top_right * weight_top_right
        + bottom_left * weight_bottom_left
        + bottom_right * weight_bottom_right
    )
    return tl.where(weight_sum > 1.0e-6, weighted_depth / tl.maximum(weight_sum, 1.0e-6), current_value)


@triton.jit
def _motion_stabilize_depth_kernel(
    current,
    previous,
    flow_x,
    flow_y,
    confidence,
    out,
    height: tl.constexpr,
    width: tl.constexpr,
    flow_height: tl.constexpr,
    flow_width: tl.constexpr,
    strength: tl.constexpr,
    edge_threshold: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < height * width
    x = tl.minimum(offsets % width, width - 1).to(tl.float32)
    y = tl.minimum(offsets // width, height - 1).to(tl.float32)
    flow_x_coord = (x + 0.5) * (flow_width / width) - 0.5
    flow_y_coord = (y + 0.5) * (flow_height / height) - 0.5
    dx = _motion_bilinear_sample(flow_x, flow_x_coord, flow_y_coord, flow_width, flow_height)
    dy = _motion_bilinear_sample(flow_y, flow_x_coord, flow_y_coord, flow_width, flow_height)
    confidence_value = _motion_bilinear_sample(
        confidence,
        flow_x_coord,
        flow_y_coord,
        flow_width,
        flow_height,
    )
    current_value = tl.load(current + offsets, mask=active, other=0.0)
    previous_value = _motion_edge_aware_depth_sample(
        previous,
        x + dx * (width / flow_width),
        y + dy * (height / flow_height),
        current_value,
        width,
        height,
        edge_threshold,
    )
    delta = previous_value - current_value
    result = current_value + delta * tl.minimum(tl.maximum(confidence_value, 0.0), 1.0) * strength
    tl.store(out + offsets, tl.minimum(tl.maximum(result, 0.0), 1.0), mask=active)


def can_use_motion_depth_stabilization(depth: torch.Tensor, rgb: torch.Tensor) -> bool:
    if rgb.ndim == 3:
        rgb = rgb.unsqueeze(0)
    return (
        depth.device.type == "cuda"
        and triton_runtime_available(depth.device)
        and depth.dtype == torch.float32
        and depth.ndim == 4
        and depth.shape[0] == 1
        and depth.shape[1] == 1
        and rgb.device == depth.device
        and rgb.dtype == torch.float32
        and rgb.ndim == 4
        and rgb.shape[0] == 1
        and rgb.shape[1] == 3
        and depth.shape[-2] > 0
        and depth.shape[-1] > 0
        and rgb.shape[-2] > 0
        and rgb.shape[-1] > 0
    )


def _downsample_motion_luma(rgb: torch.Tensor) -> torch.Tensor:
    if rgb.ndim == 3:
        rgb = rgb.unsqueeze(0)
    flow_height, flow_width = 135, 240
    out = torch.empty((flow_height, flow_width), device=rgb.device, dtype=torch.float32)
    block = 128
    _motion_luma_kernel[(triton.cdiv(flow_height * flow_width, block),)](
        rgb,
        out,
        int(rgb.shape[-2]),
        int(rgb.shape[-1]),
        int(rgb.stride(1)),
        int(rgb.stride(2)),
        int(rgb.stride(3)),
        flow_height,
        flow_width,
        block,
        num_warps=4,
    )
    return out


def stabilize_depth_motion_compensated(
    current: torch.Tensor,
    previous: torch.Tensor,
    rgb: torch.Tensor,
    previous_luma: torch.Tensor,
    *,
    strength: float = 0.65,
    edge_threshold: float = 0.04,
) -> tuple[torch.Tensor, torch.Tensor]:
    current = current.detach().contiguous()
    previous = previous.detach().contiguous()
    luma = _downsample_motion_luma(rgb)
    flow_height, flow_width = 135, 240
    small_count = flow_height * flow_width
    small_block = 128
    small_grid = (triton.cdiv(small_count, small_block),)
    flow_forward_x = torch.empty_like(luma)
    flow_forward_y = torch.empty_like(luma)
    flow_forward_confidence = torch.empty_like(luma)
    flow_reverse_x = torch.empty_like(luma)
    flow_reverse_y = torch.empty_like(luma)
    flow_reverse_confidence = torch.empty_like(luma)
    confidence = torch.empty_like(luma)
    _motion_flow_kernel[small_grid](
        luma,
        previous_luma,
        flow_forward_x,
        flow_forward_y,
        flow_forward_confidence,
        flow_height,
        flow_width,
        small_block,
        num_warps=4,
    )
    _motion_flow_kernel[small_grid](
        previous_luma,
        luma,
        flow_reverse_x,
        flow_reverse_y,
        flow_reverse_confidence,
        flow_height,
        flow_width,
        small_block,
        num_warps=4,
    )
    _motion_confidence_kernel[small_grid](
        flow_forward_x,
        flow_forward_y,
        flow_forward_confidence,
        flow_reverse_x,
        flow_reverse_y,
        flow_reverse_confidence,
        confidence,
        flow_height,
        flow_width,
        small_block,
        num_warps=4,
    )
    height, width = (int(current.shape[-2]), int(current.shape[-1]))
    out = torch.empty_like(current)
    block = 128
    _motion_stabilize_depth_kernel[(triton.cdiv(height * width, block),)](
        current,
        previous,
        flow_forward_x,
        flow_forward_y,
        confidence,
        out,
        height,
        width,
        flow_height,
        flow_width,
        float(max(0.0, min(0.65, strength))),
        float(max(0.0, min(1.0, edge_threshold))),
        block,
        num_warps=4,
    )
    return out, luma


def warmup_motion_depth_stabilization(rgb: torch.Tensor, depth: torch.Tensor) -> None:
    if rgb.ndim == 3:
        rgb = rgb.unsqueeze(0)
    if not can_use_motion_depth_stabilization(depth, rgb):
        return
    previous_luma = _downsample_motion_luma(rgb)
    stabilize_depth_motion_compensated(
        depth,
        depth,
        rgb,
        previous_luma,
    )


@triton.jit
def _temporal_masked_kernel(
    left,
    right,
    prev_left,
    prev_right,
    mask,
    out_left,
    out_right,
    alpha: tl.constexpr,
    total: tl.constexpr,
    pixels: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < total
    pixel = offsets % pixels
    blend = tl.load(mask + pixel, mask=active, other=0.0) * alpha

    left_value = tl.load(left + offsets, mask=active, other=0.0)
    right_value = tl.load(right + offsets, mask=active, other=0.0)
    prev_left_value = tl.load(prev_left + offsets, mask=active, other=0.0)
    prev_right_value = tl.load(prev_right + offsets, mask=active, other=0.0)

    tl.store(out_left + offsets, left_value * (1.0 - blend) + prev_left_value * blend, mask=active)
    tl.store(out_right + offsets, right_value * (1.0 - blend) + prev_right_value * blend, mask=active)


def can_use_triton_temporal_masked(
    left: torch.Tensor,
    right: torch.Tensor,
    prev_left: torch.Tensor,
    prev_right: torch.Tensor,
    mask: torch.Tensor,
) -> bool:
    return (
        triton_runtime_available(left.device)
        and left.dtype == torch.float32
        and right.dtype == torch.float32
        and prev_left.dtype == torch.float32
        and prev_right.dtype == torch.float32
        and mask.dtype == torch.float32
        and left.ndim == 4
        and right.shape == left.shape
        and prev_left.shape == left.shape
        and prev_right.shape == right.shape
        and mask.ndim == 4
        and mask.shape[0] == left.shape[0]
        and mask.shape[1] == 1
        and mask.shape[-2:] == left.shape[-2:]
    )


def apply_temporal_masked(
    left: torch.Tensor,
    right: torch.Tensor,
    prev_left: torch.Tensor,
    prev_right: torch.Tensor,
    mask: torch.Tensor,
    *,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    left = left.contiguous()
    right = right.contiguous()
    prev_left = prev_left.contiguous()
    prev_right = prev_right.contiguous()
    mask = mask.contiguous()
    out_left = torch.empty_like(left)
    out_right = torch.empty_like(right)
    pixels = left.shape[-2] * left.shape[-1]
    total = left.numel()
    block = 256
    grid = (triton.cdiv(total, block),)
    _temporal_masked_kernel[grid](
        left,
        right,
        prev_left,
        prev_right,
        mask,
        out_left,
        out_right,
        float(alpha),
        total,
        pixels,
        block,
    )
    return out_left, out_right
