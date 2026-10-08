from __future__ import annotations

import torch
import triton
import triton.language as tl

from .triton_runtime import triton_runtime_available


@triton.jit
def _area_half2(value_0, value_1):
    return (value_0 + value_1) * 0.5


@triton.jit
def _half_sbs_kernel(
    left,
    right,
    out,
    total: tl.constexpr,
    width: tl.constexpr,
    half_width: tl.constexpr,
    pixels: tl.constexpr,
    LINEAR_SRGB: tl.constexpr,
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
    x0 = src_x_out * 2
    x1 = x0 + 1
    base = channel * pixels + y * width

    left_v0 = tl.load(left + base + x0, mask=active & use_left, other=0.0)
    left_v1 = tl.load(left + base + x1, mask=active & use_left, other=0.0)
    right_v0 = tl.load(right + base + x0, mask=active & ~use_left, other=0.0)
    right_v1 = tl.load(right + base + x1, mask=active & ~use_left, other=0.0)
    if LINEAR_SRGB:
        left_linear0 = tl.where(left_v0 <= 0.04045, left_v0 / 12.92,
            tl.exp2(2.4 * tl.log2(tl.maximum((left_v0 + 0.055) / 1.055, 1.0e-20))))
        left_linear1 = tl.where(left_v1 <= 0.04045, left_v1 / 12.92,
            tl.exp2(2.4 * tl.log2(tl.maximum((left_v1 + 0.055) / 1.055, 1.0e-20))))
        right_linear0 = tl.where(right_v0 <= 0.04045, right_v0 / 12.92,
            tl.exp2(2.4 * tl.log2(tl.maximum((right_v0 + 0.055) / 1.055, 1.0e-20))))
        right_linear1 = tl.where(right_v1 <= 0.04045, right_v1 / 12.92,
            tl.exp2(2.4 * tl.log2(tl.maximum((right_v1 + 0.055) / 1.055, 1.0e-20))))
        left_mean = (left_linear0 + left_linear1) * 0.5
        right_mean = (right_linear0 + right_linear1) * 0.5
        left_value = tl.where(left_mean <= 0.0031308, left_mean * 12.92,
            1.055 * tl.exp2(tl.log2(tl.maximum(left_mean, 1.0e-20)) / 2.4) - 0.055)
        right_value = tl.where(right_mean <= 0.0031308, right_mean * 12.92,
            1.055 * tl.exp2(tl.log2(tl.maximum(right_mean, 1.0e-20)) / 2.4) - 0.055)
    else:
        left_value = _area_half2(left_v0, left_v1)
        right_value = _area_half2(right_v0, right_v1)
    value = tl.where(use_left, left_value, right_value)
    tl.store(out + offsets, value, mask=active)


def can_use_triton_half_sbs(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        triton_runtime_available(left.device)
        and left.dtype == torch.float32
        and right.dtype == torch.float32
        and left.ndim == 4
        and right.ndim == 4
        and left.shape == right.shape
        and left.shape[0] == 1
        and left.shape[1] == 3
        and left.shape[-1] % 2 == 0
    )


def can_use_triton_full_sbs(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        triton_runtime_available(left.device)
        and left.dtype == torch.float32
        and right.dtype == torch.float32
        and left.ndim == 4
        and right.ndim == 4
        and left.shape == right.shape
        and left.shape[0] == 1
        and left.shape[1] == 3
    )


def can_use_triton_half_tab(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        triton_runtime_available(left.device)
        and left.dtype == torch.float32
        and right.dtype == torch.float32
        and left.ndim == 4
        and right.ndim == 4
        and left.shape == right.shape
        and left.shape[0] == 1
        and left.shape[1] == 3
        and left.shape[-2] % 2 == 0
    )


def can_use_triton_full_tab(left: torch.Tensor, right: torch.Tensor) -> bool:
    return can_use_triton_full_sbs(left, right)


def can_use_triton_depth_map(depth: torch.Tensor, channels: int) -> bool:
    return (
        triton_runtime_available(depth.device)
        and depth.dtype == torch.float32
        and depth.ndim == 4
        and depth.shape[0] == 1
        and depth.shape[1] == 1
        and channels == 3
    )


def _can_use_triton_postprocess(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        triton_runtime_available(left.device)
        and left.dtype == torch.float32
        and right.dtype == torch.float32
        and left.ndim == 4
        and right.ndim == 4
        and left.shape == right.shape
        and left.shape[0] == 1
        and left.shape[1] == 3
    )


def can_use_triton_anaglyph(left: torch.Tensor, right: torch.Tensor) -> bool:
    return _can_use_triton_postprocess(left, right)


def can_use_triton_interleaved(left: torch.Tensor, right: torch.Tensor) -> bool:
    return _can_use_triton_postprocess(left, right)


def can_use_triton_leia(left: torch.Tensor, right: torch.Tensor) -> bool:
    return _can_use_triton_postprocess(left, right)


@triton.jit
def _half_sbs_uint8_kernel(
    left,
    right,
    out,
    total: tl.constexpr,
    width: tl.constexpr,
    half_width: tl.constexpr,
    pixels: tl.constexpr,
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
    x0 = src_x_out * 2
    x1 = x0 + 1
    base = channel * pixels + y * width

    left_v0 = tl.load(left + base + x0, mask=active & use_left, other=0.0)
    left_v1 = tl.load(left + base + x1, mask=active & use_left, other=0.0)
    right_v0 = tl.load(right + base + x0, mask=active & ~use_left, other=0.0)
    right_v1 = tl.load(right + base + x1, mask=active & ~use_left, other=0.0)
    left_value = _area_half2(left_v0, left_v1)
    right_value = _area_half2(right_v0, right_v1)
    value = tl.where(use_left, left_value, right_value)
    value = tl.minimum(tl.maximum(value, 0.0), 1.0) * 255.0
    tl.store(out + offsets, value.to(tl.uint8), mask=active)


def make_half_sbs_uint8(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.contiguous()
    right = right.contiguous()
    _, channels, height, width = left.shape
    out = torch.empty((1, channels, height, width), device=left.device, dtype=torch.uint8)
    half_width = width // 2
    pixels = height * width
    total = out.numel()
    block = 256
    grid = (triton.cdiv(total, block),)
    _half_sbs_uint8_kernel[grid](left, right, out, total, width, half_width, pixels, block)
    return out


@triton.jit
def _chw_rgb_to_hwc_rgba_u8_kernel(src, out, pixels: tl.constexpr, width: tl.constexpr, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < pixels
    y = offsets // width
    x = offsets - y * width
    src_offset = y * width + x
    r = tl.load(src + src_offset, mask=active, other=0.0)
    g = tl.load(src + pixels + src_offset, mask=active, other=0.0)
    b = tl.load(src + pixels * 2 + src_offset, mask=active, other=0.0)
    out_base = offsets * 4
    tl.store(out + out_base, (tl.minimum(tl.maximum(r, 0.0), 1.0) * 255.0).to(tl.uint8), mask=active)
    tl.store(out + out_base + 1, (tl.minimum(tl.maximum(g, 0.0), 1.0) * 255.0).to(tl.uint8), mask=active)
    tl.store(out + out_base + 2, (tl.minimum(tl.maximum(b, 0.0), 1.0) * 255.0).to(tl.uint8), mask=active)
    tl.store(out + out_base + 3, 255, mask=active)


def make_chw_rgb_to_hwc_rgba_u8(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.contiguous()
    _, channels, height, width = tensor.shape
    if channels != 3:
        raise ValueError("expected BCHW RGB tensor with 3 channels")
    out = torch.empty((height, width, 4), device=tensor.device, dtype=torch.uint8)
    pixels = height * width
    block = 256
    grid = (triton.cdiv(pixels, block),)
    _chw_rgb_to_hwc_rgba_u8_kernel[grid](tensor, out, pixels, width, block)
    return out


def make_half_sbs(left: torch.Tensor, right: torch.Tensor, *, linear_srgb: bool = False) -> torch.Tensor:
    left = left.contiguous()
    right = right.contiguous()
    out = torch.empty_like(left)
    _, _, height, width = left.shape
    half_width = width // 2
    pixels = height * width
    total = left.numel()
    block = 256
    grid = (triton.cdiv(total, block),)
    _half_sbs_kernel[grid](left, right, out, total, width, half_width, pixels,
                           bool(linear_srgb), block)
    return out


@triton.jit
def _full_sbs_kernel(
    left,
    right,
    out,
    total: tl.constexpr,
    width: tl.constexpr,
    out_width: tl.constexpr,
    out_pixels: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < total
    out_pixel = offsets % out_pixels
    y = out_pixel // out_width
    x = out_pixel - y * out_width
    channel = offsets // out_pixels

    use_left = x < width
    src_x = tl.where(use_left, x, x - width)
    src_offset = channel * (out_pixels // 2) + y * width + src_x
    value = tl.where(
        use_left,
        tl.load(left + src_offset, mask=active & use_left, other=0.0),
        tl.load(right + src_offset, mask=active & ~use_left, other=0.0),
    )
    tl.store(out + offsets, value, mask=active)


def make_full_sbs(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.contiguous()
    right = right.contiguous()
    _, channels, height, width = left.shape
    out = torch.empty((1, channels, height, width * 2), device=left.device, dtype=left.dtype)
    out_width = width * 2
    out_pixels = height * out_width
    total = out.numel()
    block = 256
    grid = (triton.cdiv(total, block),)
    _full_sbs_kernel[grid](left, right, out, total, width, out_width, out_pixels, block)
    return out


@triton.jit
def _half_tab_kernel(
    left,
    right,
    out,
    total: tl.constexpr,
    height: tl.constexpr,
    half_height: tl.constexpr,
    width: tl.constexpr,
    pixels: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < total
    pixel = offsets % pixels
    y = pixel // width
    x = pixel - y * width
    channel = offsets // pixels

    use_left = y < half_height
    src_y_out = tl.where(use_left, y, y - half_height)
    y0 = src_y_out * 2
    ym1 = tl.maximum(y0 - 1, 0)
    y1 = y0 + 1
    y2 = tl.minimum(y0 + 2, height - 1)
    channel_base = channel * pixels

    left_vm1 = tl.load(left + channel_base + ym1 * width + x, mask=active & use_left, other=0.0)
    left_v0 = tl.load(left + channel_base + y0 * width + x, mask=active & use_left, other=0.0)
    left_v1 = tl.load(left + channel_base + y1 * width + x, mask=active & use_left, other=0.0)
    left_v2 = tl.load(left + channel_base + y2 * width + x, mask=active & use_left, other=0.0)
    right_vm1 = tl.load(right + channel_base + ym1 * width + x, mask=active & ~use_left, other=0.0)
    right_v0 = tl.load(right + channel_base + y0 * width + x, mask=active & ~use_left, other=0.0)
    right_v1 = tl.load(right + channel_base + y1 * width + x, mask=active & ~use_left, other=0.0)
    right_v2 = tl.load(right + channel_base + y2 * width + x, mask=active & ~use_left, other=0.0)
    left_value = _lanczos2_half4(left_vm1, left_v0, left_v1, left_v2)
    right_value = _lanczos2_half4(right_vm1, right_v0, right_v1, right_v2)
    value = tl.where(use_left, left_value, right_value)
    tl.store(out + offsets, value, mask=active)


def make_half_tab(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.contiguous()
    right = right.contiguous()
    out = torch.empty_like(left)
    _, _, height, width = left.shape
    half_height = height // 2
    pixels = height * width
    total = left.numel()
    block = 256
    grid = (triton.cdiv(total, block),)
    _half_tab_kernel[grid](left, right, out, total, height, half_height, width, pixels, block)
    return out


@triton.jit
def _full_tab_kernel(
    left,
    right,
    out,
    total: tl.constexpr,
    height: tl.constexpr,
    out_height: tl.constexpr,
    width: tl.constexpr,
    out_pixels: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < total
    out_pixel = offsets % out_pixels
    y = out_pixel // width
    x = out_pixel - y * width
    channel = offsets // out_pixels

    use_left = y < height
    src_y = tl.where(use_left, y, y - height)
    src_offset = channel * (out_pixels // 2) + src_y * width + x
    value = tl.where(
        use_left,
        tl.load(left + src_offset, mask=active & use_left, other=0.0),
        tl.load(right + src_offset, mask=active & ~use_left, other=0.0),
    )
    tl.store(out + offsets, value, mask=active)


def make_full_tab(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.contiguous()
    right = right.contiguous()
    _, channels, height, width = left.shape
    out = torch.empty((1, channels, height * 2, width), device=left.device, dtype=left.dtype)
    out_height = height * 2
    out_pixels = out_height * width
    total = out.numel()
    block = 256
    grid = (triton.cdiv(total, block),)
    _full_tab_kernel[grid](left, right, out, total, height, out_height, width, out_pixels, block)
    return out


@triton.jit
def _depth_map_kernel(
    depth,
    out,
    total: tl.constexpr,
    pixels: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < total
    pixel = offsets % pixels
    value = tl.load(depth + pixel, mask=active, other=0.0)
    tl.store(out + offsets, value, mask=active)


def make_depth_map(depth: torch.Tensor, channels: int) -> torch.Tensor:
    depth = depth.contiguous()
    _, _, height, width = depth.shape
    out = torch.empty((1, channels, height, width), device=depth.device, dtype=depth.dtype)
    pixels = height * width
    total = out.numel()
    block = 256
    grid = (triton.cdiv(total, block),)
    _depth_map_kernel[grid](depth, out, total, pixels, block)
    return out


@triton.jit
def _stereo_postprocess_kernel(
    left,
    right,
    out,
    total: tl.constexpr,
    width: tl.constexpr,
    pixels: tl.constexpr,
    mode: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < total
    pixel = offsets % pixels
    y = pixel // width
    x = pixel - y * width
    channel = offsets // pixels

    use_left = channel == 0
    if mode == 1:
        use_left = (y % 2) == 0
    if mode == 2:
        use_left = (x % 2) == 0

    value = tl.where(
        use_left,
        tl.load(left + offsets, mask=active, other=0.0),
        tl.load(right + offsets, mask=active, other=0.0),
    )
    tl.store(out + offsets, value, mask=active)


def _make_stereo_postprocess(left: torch.Tensor, right: torch.Tensor, mode: int) -> torch.Tensor:
    left = left.contiguous()
    right = right.contiguous()
    out = torch.empty_like(left)
    _, _, height, width = left.shape
    pixels = height * width
    total = left.numel()
    block = 256
    grid = (triton.cdiv(total, block),)
    _stereo_postprocess_kernel[grid](left, right, out, total, width, pixels, mode, block)
    return out


def make_anaglyph(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return _make_stereo_postprocess(left, right, mode=0)


def make_interleaved(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return _make_stereo_postprocess(left, right, mode=1)


def make_leia(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return _make_stereo_postprocess(left, right, mode=2)
