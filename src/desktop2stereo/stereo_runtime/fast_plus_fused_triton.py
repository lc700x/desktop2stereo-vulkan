from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from .triton_runtime import triton_runtime_available


@triton.jit
def _clamp_i32(value, lo: tl.constexpr, hi: tl.constexpr):
    return tl.minimum(tl.maximum(value, lo), hi)


@triton.jit
def _shift_from_depth(
    depth_value,
    convergence: tl.constexpr,
    max_disparity_px: tl.constexpr,
    depth_strength: tl.constexpr,
):
    depth_value = tl.minimum(tl.maximum(depth_value, 0.0), 1.0)
    return -(depth_value - convergence) * max_disparity_px * depth_strength * 0.5


@triton.jit
def _load_depth_at(depth, y, x, width: tl.constexpr, height: tl.constexpr):
    yy = _clamp_i32(y, 0, height - 1)
    xx = _clamp_i32(x, 0, width - 1)
    return tl.load(depth + yy * width + xx)


@triton.jit
def _sample_eye(
    rgb,
    depth,
    channel,
    y,
    target_x,
    eye_sign,
    width: tl.constexpr,
    height: tl.constexpr,
    pixels: tl.constexpr,
    convergence: tl.constexpr,
    max_disparity_px: tl.constexpr,
    depth_strength: tl.constexpr,
):
    tx_i = _clamp_i32(target_x, 0, width - 1)
    depth_value = _load_depth_at(depth, y, tx_i, width, height)
    shift_px = _shift_from_depth(depth_value, convergence, max_disparity_px, depth_strength)
    src_x = target_x.to(tl.float32) + shift_px * eye_sign
    src_x = tl.minimum(tl.maximum(src_x, 0.0), (width - 1) * 1.0)
    x0_f = tl.floor(src_x)
    x0 = x0_f.to(tl.int32)
    x1 = _clamp_i32(x0 + 1, 0, width - 1)
    frac = src_x - x0_f
    base = channel * pixels + y * width
    v0 = tl.load(rgb + base + x0)
    v1 = tl.load(rgb + base + x1)
    # A bilinear tap straddling a low-resolution depth step contains both
    # sides of the source contour.  Reusing that mixed color after shifting
    # it creates the second, displaced edge seen in SBS.  Keep the source
    # pixel's depth class when the two taps are a real edge; flat regions use
    # the original linear sample.
    left_depth = _load_depth_at(depth, y, x0, width, height)
    right_depth = _load_depth_at(depth, y, x1, width, height)
    depth_edge = tl.abs(left_depth - right_depth) > 0.04
    keep_left = tl.abs(left_depth - depth_value) <= 0.04
    keep_right = tl.abs(right_depth - depth_value) <= 0.04
    class_weight = tl.where(keep_left, 1.0 - frac, 0.0) + tl.where(keep_right, frac, 0.0)
    class_value = (
        v0 * tl.where(keep_left, 1.0 - frac, 0.0)
        + v1 * tl.where(keep_right, frac, 0.0)
    ) / tl.maximum(class_weight, 1.0e-6)
    linear_value = v0 * (1.0 - frac) + v1 * frac
    return tl.where(depth_edge & (class_weight > 1.0e-6), class_value, linear_value)


@triton.jit
def _mask_at(
    depth,
    y,
    x,
    width: tl.constexpr,
    height: tl.constexpr,
    convergence: tl.constexpr,
    max_disparity_px: tl.constexpr,
    depth_strength: tl.constexpr,
    edge_threshold: tl.constexpr,
    shift_edge_threshold_px: tl.constexpr,
):
    found = x < 0
    for dy in tl.static_range(-1, 2):
        yy = y + dy
        valid_y = (yy >= 0) & (yy < height)
        for dx in tl.static_range(-1, 2):
            xx = x + dx
            valid = valid_y & (xx >= 0) & (xx < width)
            center_depth = _load_depth_at(depth, yy, xx, width, height)
            right_depth = _load_depth_at(depth, yy, xx + 1, width, height)
            down_depth = _load_depth_at(depth, yy + 1, xx, width, height)
            center_shift = tl.abs(_shift_from_depth(center_depth, convergence, max_disparity_px, depth_strength))
            right_shift = tl.abs(_shift_from_depth(right_depth, convergence, max_disparity_px, depth_strength))
            down_shift = tl.abs(_shift_from_depth(down_depth, convergence, max_disparity_px, depth_strength))
            depth_edge = (tl.abs(right_depth - center_depth) + tl.abs(down_depth - center_depth)) > edge_threshold
            shift_edge = (tl.abs(right_shift - center_shift) + tl.abs(down_shift - center_shift)) > shift_edge_threshold_px
            found = found | (valid & (depth_edge | shift_edge))
    return found


@triton.jit
def _filled_eye(
    rgb,
    depth,
    channel,
    y,
    target_x,
    eye_sign,
    width: tl.constexpr,
    height: tl.constexpr,
    pixels: tl.constexpr,
    convergence: tl.constexpr,
    max_disparity_px: tl.constexpr,
    depth_strength: tl.constexpr,
    edge_threshold: tl.constexpr,
    shift_edge_threshold_px: tl.constexpr,
):
    value = _sample_eye(
        rgb,
        depth,
        channel,
        y,
        target_x,
        eye_sign,
        width,
        height,
        pixels,
        convergence,
        max_disparity_px,
        depth_strength,
    )
    mask = _mask_at(
        depth,
        y,
        target_x,
        width,
        height,
        convergence,
        max_disparity_px,
        depth_strength,
        edge_threshold,
        shift_edge_threshold_px,
    )
    left_depth = _load_depth_at(depth, y, target_x - 1, width, height)
    right_depth = _load_depth_at(depth, y, target_x + 1, width, height)
    reliable_direction = tl.abs(right_depth - left_depth) > edge_threshold
    sample_right = right_depth < left_depth

    left1 = _sample_eye(rgb, depth, channel, y, target_x - 1, eye_sign, width, height, pixels, convergence, max_disparity_px, depth_strength)
    right1 = _sample_eye(rgb, depth, channel, y, target_x + 1, eye_sign, width, height, pixels, convergence, max_disparity_px, depth_strength)
    left2 = _sample_eye(rgb, depth, channel, y, target_x - 2, eye_sign, width, height, pixels, convergence, max_disparity_px, depth_strength)
    right2 = _sample_eye(rgb, depth, channel, y, target_x + 2, eye_sign, width, height, pixels, convergence, max_disparity_px, depth_strength)
    balanced = (left1 + right1 + left2 + right2) * 0.25
    background = tl.where(sample_right, right1 * 0.65 + right2 * 0.35, left1 * 0.65 + left2 * 0.35)
    filled = tl.where(reliable_direction, background, balanced)
    return tl.where(mask, value + (filled - value) * 0.60, value)


@triton.jit
def _fast_plus_half_sbs_uint8_kernel(
    rgb,
    depth,
    out,
    total: tl.constexpr,
    width: tl.constexpr,
    height: tl.constexpr,
    half_width: tl.constexpr,
    pixels: tl.constexpr,
    convergence: tl.constexpr,
    max_disparity_px: tl.constexpr,
    depth_strength: tl.constexpr,
    edge_threshold: tl.constexpr,
    shift_edge_threshold_px: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < total
    pixel = offsets % pixels
    y = pixel // width
    x = pixel - y * width
    channel = offsets // pixels

    use_left = x < half_width
    src_x_out = tl.where(use_left, x, x - half_width)
    target_x0 = src_x_out * 2
    target_x1 = _clamp_i32(target_x0 + 1, 0, width - 1)
    eye_sign = tl.where(use_left, 1.0, -1.0)

    v0 = _filled_eye(
        rgb,
        depth,
        channel,
        y,
        target_x0,
        eye_sign,
        width,
        height,
        pixels,
        convergence,
        max_disparity_px,
        depth_strength,
        edge_threshold,
        shift_edge_threshold_px,
    )
    v1 = _filled_eye(
        rgb,
        depth,
        channel,
        y,
        target_x1,
        eye_sign,
        width,
        height,
        pixels,
        convergence,
        max_disparity_px,
        depth_strength,
        edge_threshold,
        shift_edge_threshold_px,
    )
    value = tl.minimum(tl.maximum((v0 + v1) * 0.5, 0.0), 1.0) * 255.0
    tl.store(out + offsets, value.to(tl.uint8), mask=active)


@triton.jit
def _fast_plus_warp_only_half_sbs_uint8_kernel(
    rgb,
    depth,
    out,
    total: tl.constexpr,
    width: tl.constexpr,
    height: tl.constexpr,
    half_width: tl.constexpr,
    pixels: tl.constexpr,
    convergence: tl.constexpr,
    max_disparity_px: tl.constexpr,
    depth_strength: tl.constexpr,
    block: tl.constexpr,
):
    """Render the complete SBS image without the expensive hole-fill branch."""

    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < total
    pixel = offsets % pixels
    y = pixel // width
    x = pixel - y * width
    channel = offsets // pixels
    use_left = x < half_width
    src_x_out = tl.where(use_left, x, x - half_width)
    target_x0 = src_x_out * 2
    target_x1 = _clamp_i32(target_x0 + 1, 0, width - 1)
    eye_sign = tl.where(use_left, 1.0, -1.0)
    v0 = _sample_eye(
        rgb,
        depth,
        channel,
        y,
        target_x0,
        eye_sign,
        width,
        height,
        pixels,
        convergence,
        max_disparity_px,
        depth_strength,
    )
    v1 = _sample_eye(
        rgb,
        depth,
        channel,
        y,
        target_x1,
        eye_sign,
        width,
        height,
        pixels,
        convergence,
        max_disparity_px,
        depth_strength,
    )
    value = tl.minimum(tl.maximum((v0 + v1) * 0.5, 0.0), 1.0) * 255.0
    tl.store(out + offsets, value.to(tl.uint8), mask=active)


@triton.jit
def _fast_plus_hole_ratio_probe_kernel(
    depth,
    out,
    total: tl.constexpr,
    width: tl.constexpr,
    height: tl.constexpr,
    half_width: tl.constexpr,
    probe_width: tl.constexpr,
    probe_height: tl.constexpr,
    convergence: tl.constexpr,
    max_disparity_px: tl.constexpr,
    depth_strength: tl.constexpr,
    edge_threshold: tl.constexpr,
    shift_edge_threshold_px: tl.constexpr,
    block: tl.constexpr,
):
    """Estimate the hole ratio on a small regular grid before full compaction."""

    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < total
    probe_x = offsets % probe_width
    probe_y = offsets // probe_width
    x = (probe_x * half_width) // probe_width
    y = (probe_y * height) // probe_height
    target_x0 = x * 2
    target_x1 = _clamp_i32(target_x0 + 1, 0, width - 1)
    found = _mask_at(
        depth,
        y,
        target_x0,
        width,
        height,
        convergence,
        max_disparity_px,
        depth_strength,
        edge_threshold,
        shift_edge_threshold_px,
    )
    found = found | _mask_at(
        depth,
        y,
        target_x1,
        width,
        height,
        convergence,
        max_disparity_px,
        depth_strength,
        edge_threshold,
        shift_edge_threshold_px,
    )
    tl.store(out + offsets, found.to(tl.uint8), mask=active)


@triton.jit
def _fast_plus_active_mask_kernel(
    depth,
    out,
    total: tl.constexpr,
    width: tl.constexpr,
    height: tl.constexpr,
    half_width: tl.constexpr,
    convergence: tl.constexpr,
    max_disparity_px: tl.constexpr,
    depth_strength: tl.constexpr,
    edge_threshold: tl.constexpr,
    shift_edge_threshold_px: tl.constexpr,
    block: tl.constexpr,
):
    """Build one active-pixel flag per output eye pixel for sparse filling."""

    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < total
    y = offsets // half_width
    x = offsets - y * half_width
    target_x0 = x * 2
    target_x1 = _clamp_i32(target_x0 + 1, 0, width - 1)
    found = _mask_at(
        depth,
        y,
        target_x0,
        width,
        height,
        convergence,
        max_disparity_px,
        depth_strength,
        edge_threshold,
        shift_edge_threshold_px,
    )
    found = found | _mask_at(
        depth,
        y,
        target_x1,
        width,
        height,
        convergence,
        max_disparity_px,
        depth_strength,
        edge_threshold,
        shift_edge_threshold_px,
    )
    tl.store(out + offsets, found.to(tl.uint8), mask=active)


@triton.jit
def _fast_plus_sparse_fill_kernel(
    rgb,
    depth,
    active_pixels,
    out,
    total: tl.constexpr,
    active_count,
    channels: tl.constexpr,
    width: tl.constexpr,
    height: tl.constexpr,
    half_width: tl.constexpr,
    pixels: tl.constexpr,
    convergence: tl.constexpr,
    max_disparity_px: tl.constexpr,
    depth_strength: tl.constexpr,
    edge_threshold: tl.constexpr,
    shift_edge_threshold_px: tl.constexpr,
    block: tl.constexpr,
):
    """Apply the expensive directional fill only to compacted hole pixels."""

    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < total
    active_slot = offsets % active_count
    channel_eye = offsets // active_count
    eye = channel_eye // channels
    channel = channel_eye - eye * channels
    pixel = tl.load(active_pixels + active_slot, mask=active, other=0).to(tl.int32)
    y = pixel // half_width
    x = pixel - y * half_width
    target_x0 = x * 2
    target_x1 = _clamp_i32(target_x0 + 1, 0, width - 1)
    eye_sign = tl.where(eye == 0, 1.0, -1.0)

    v0 = _filled_eye(
        rgb,
        depth,
        channel,
        y,
        target_x0,
        eye_sign,
        width,
        height,
        pixels,
        convergence,
        max_disparity_px,
        depth_strength,
        edge_threshold,
        shift_edge_threshold_px,
    )
    v1 = _filled_eye(
        rgb,
        depth,
        channel,
        y,
        target_x1,
        eye_sign,
        width,
        height,
        pixels,
        convergence,
        max_disparity_px,
        depth_strength,
        edge_threshold,
        shift_edge_threshold_px,
    )
    value = tl.minimum(tl.maximum((v0 + v1) * 0.5, 0.0), 1.0) * 255.0
    out_pixel = eye * half_width + x
    out_offset = channel * pixels + y * width + out_pixel
    tl.store(out + out_offset, value.to(tl.uint8), mask=active)


def can_use_fast_plus_fused_half_sbs_uint8(rgb: torch.Tensor, depth: torch.Tensor) -> bool:
    return (
        triton_runtime_available(rgb.device)
        and rgb.dtype == torch.float32
        and depth.dtype == torch.float32
        and rgb.ndim == 4
        and depth.ndim == 4
        and rgb.shape[0] == 1
        and rgb.shape[1] == 3
        and depth.shape[0] == 1
        and depth.shape[1] == 1
        and rgb.shape[-2:] == depth.shape[-2:]
        and rgb.shape[-1] % 2 == 0
    )


def _sparse_hole_threshold() -> float:
    try:
        value = float(os.environ.get("D2S_TRITON_SPARSE_HOLE_THRESHOLD", "0.08"))
    except (TypeError, ValueError):
        value = 0.08
    return max(0.0, min(1.0, value))


def _sparse_probe_size() -> int:
    try:
        value = int(os.environ.get("D2S_TRITON_SPARSE_PROBE_SIZE", "64"))
    except (TypeError, ValueError):
        value = 64
    return max(16, min(128, value))


def _sparse_min_pixels() -> int:
    try:
        value = int(os.environ.get("D2S_TRITON_SPARSE_MIN_PIXELS", "1048576"))
    except (TypeError, ValueError):
        value = 1048576
    return max(0, value)


def _estimate_hole_ratio(
    depth: torch.Tensor,
    *,
    width: int,
    height: int,
    half_width: int,
    convergence: float,
    max_disparity_px: float,
    depth_strength: float,
    edge_threshold: float,
    shift_edge_threshold_px: float,
) -> float:
    probe_size = _sparse_probe_size()
    probe_width = min(probe_size, half_width)
    probe_height = min(probe_size, height)
    total = probe_height * probe_width
    probe = torch.empty((total,), device=depth.device, dtype=torch.uint8)
    block = 256
    _fast_plus_hole_ratio_probe_kernel[(triton.cdiv(total, block),)](
        depth,
        probe,
        total,
        width,
        height,
        half_width,
        probe_width,
        probe_height,
        float(convergence),
        float(max_disparity_px),
        float(depth_strength),
        float(edge_threshold),
        float(shift_edge_threshold_px),
        block,
    )
    return float(probe.sum().item()) / float(total)


def _compact_active_pixels(
    depth: torch.Tensor,
    *,
    width: int,
    height: int,
    half_width: int,
    convergence: float,
    max_disparity_px: float,
    depth_strength: float,
    edge_threshold: float,
    shift_edge_threshold_px: float,
) -> torch.Tensor:
    total = height * half_width
    mask = torch.empty((total,), device=depth.device, dtype=torch.uint8)
    block = 256
    _fast_plus_active_mask_kernel[(triton.cdiv(total, block),)](
        depth,
        mask,
        total,
        width,
        height,
        half_width,
        float(convergence),
        float(max_disparity_px),
        float(depth_strength),
        float(edge_threshold),
        float(shift_edge_threshold_px),
        block,
    )
    return torch.nonzero(mask, as_tuple=False).flatten()


def make_fast_plus_fused_half_sbs_uint8(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    *,
    convergence: float,
    max_disparity_px: float,
    depth_strength: float,
    edge_threshold: float = 0.03,
) -> torch.Tensor:
    rgb = rgb.contiguous()
    depth = depth.contiguous()
    _, channels, height, width = rgb.shape
    out = torch.empty((1, channels, height, width), device=rgb.device, dtype=torch.uint8)
    half_width = width // 2
    pixels = height * width
    total = out.numel()
    depth_strength = max(0.0, float(depth_strength))
    max_possible_shift = max(0.0, float(max_disparity_px)) * depth_strength * 0.5
    shift_edge_threshold_px = max(0.20, max_possible_shift * 0.05)
    block = 256
    # On small frames, the probe and compaction launches cost more than the
    # fill work they can save. Keep the original dense kernel for that case.
    if height * half_width < _sparse_min_pixels():
        grid = (triton.cdiv(total, block),)
        _fast_plus_half_sbs_uint8_kernel[grid](
            rgb,
            depth,
            out,
            total,
            width,
            height,
            half_width,
            pixels,
            float(convergence),
            float(max_disparity_px),
            float(depth_strength),
            float(edge_threshold),
            float(shift_edge_threshold_px),
            block,
        )
        return out
    probe_ratio = _estimate_hole_ratio(
        depth,
        width=width,
        height=height,
        half_width=half_width,
        convergence=convergence,
        max_disparity_px=max_disparity_px,
        depth_strength=depth_strength,
        edge_threshold=edge_threshold,
        shift_edge_threshold_px=shift_edge_threshold_px,
    )
    if probe_ratio < _sparse_hole_threshold():
        active_pixels = _compact_active_pixels(
            depth,
            width=width,
            height=height,
            half_width=half_width,
            convergence=convergence,
            max_disparity_px=max_disparity_px,
            depth_strength=depth_strength,
            edge_threshold=edge_threshold,
            shift_edge_threshold_px=shift_edge_threshold_px,
        )
        active_count = int(active_pixels.numel())
        actual_ratio = active_count / float(height * half_width)
        if active_count and actual_ratio < _sparse_hole_threshold():
            warp_total = out.numel()
            _fast_plus_warp_only_half_sbs_uint8_kernel[(triton.cdiv(warp_total, block),)](
                rgb,
                depth,
                out,
                warp_total,
                width,
                height,
                half_width,
                pixels,
                float(convergence),
                float(max_disparity_px),
                float(depth_strength),
                block,
            )
            sparse_total = active_count * channels * 2
            _fast_plus_sparse_fill_kernel[(triton.cdiv(sparse_total, block),)](
                rgb,
                depth,
                active_pixels,
                out,
                sparse_total,
                active_count,
                channels,
                width,
                height,
                half_width,
                pixels,
                float(convergence),
                float(max_disparity_px),
                float(depth_strength),
                float(edge_threshold),
                float(shift_edge_threshold_px),
                block,
            )
            return out

    grid = (triton.cdiv(total, block),)
    _fast_plus_half_sbs_uint8_kernel[grid](
        rgb,
        depth,
        out,
        total,
        width,
        height,
        half_width,
        pixels,
        float(convergence),
        float(max_disparity_px),
        float(depth_strength),
        float(edge_threshold),
        float(shift_edge_threshold_px),
        block,
    )
    return out
