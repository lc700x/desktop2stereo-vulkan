from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - depends on optional CUDA/Triton install
    triton = None
    tl = None


@triton.jit
def _bgr_to_rgb_resize_norm_kernel(
    src,
    out,
    total: tl.constexpr,
    in_h: tl.constexpr,
    in_w: tl.constexpr,
    channels: tl.constexpr,
    out_h: tl.constexpr,
    out_w: tl.constexpr,
    scale_y: tl.constexpr,
    scale_x: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < total
    pixels = out_h * out_w
    channel = offsets // pixels
    pixel = offsets - channel * pixels
    oy = pixel // out_w
    ox = pixel - oy * out_w

    src_y = (oy.to(tl.float32) + 0.5) * scale_y - 0.5
    src_x = (ox.to(tl.float32) + 0.5) * scale_x - 0.5

    y0f = tl.floor(src_y)
    x0f = tl.floor(src_x)
    wy = src_y - y0f
    wx = src_x - x0f

    y0 = y0f.to(tl.int32)
    x0 = x0f.to(tl.int32)
    y0 = tl.minimum(tl.maximum(y0, 0), in_h - 1)
    x0 = tl.minimum(tl.maximum(x0, 0), in_w - 1)
    y1 = tl.minimum(y0 + 1, in_h - 1)
    x1 = tl.minimum(x0 + 1, in_w - 1)

    src_channel = 2 - channel
    base00 = (y0 * in_w + x0) * channels + src_channel
    base01 = (y0 * in_w + x1) * channels + src_channel
    base10 = (y1 * in_w + x0) * channels + src_channel
    base11 = (y1 * in_w + x1) * channels + src_channel

    v00 = tl.load(src + base00, mask=active, other=0.0).to(tl.float32)
    v01 = tl.load(src + base01, mask=active, other=0.0).to(tl.float32)
    v10 = tl.load(src + base10, mask=active, other=0.0).to(tl.float32)
    v11 = tl.load(src + base11, mask=active, other=0.0).to(tl.float32)

    top = v00 + (v01 - v00) * wx
    bottom = v10 + (v11 - v10) * wx
    value = (top + (bottom - top) * wy) * (1.0 / 255.0)
    tl.store(out + offsets, value, mask=active)


@triton.jit
def _bgra_to_rgb_same_size_norm_kernel(
    src,
    out,
    total_pixels: tl.constexpr,
    block: tl.constexpr,
):
    pixel = tl.program_id(0) * block + tl.arange(0, block)
    active = pixel < total_pixels
    packed_src = src.to(tl.pointer_type(tl.uint32))
    bgra = tl.load(packed_src + pixel, mask=active, other=0)
    blue = (bgra & 0xFF).to(tl.float32) * (1.0 / 255.0)
    green = ((bgra >> 8) & 0xFF).to(tl.float32) * (1.0 / 255.0)
    red = ((bgra >> 16) & 0xFF).to(tl.float32) * (1.0 / 255.0)

    plane_size = total_pixels
    tl.store(out + pixel, red, mask=active)
    tl.store(out + plane_size + pixel, green, mask=active)
    tl.store(out + 2 * plane_size + pixel, blue, mask=active)


@triton.jit
def _bgr_to_rgb_area_resize_norm_kernel(
    src,
    out,
    total: tl.constexpr,
    in_h: tl.constexpr,
    in_w: tl.constexpr,
    channels: tl.constexpr,
    out_h: tl.constexpr,
    out_w: tl.constexpr,
    scale_y: tl.constexpr,
    scale_x: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < total
    pixels = out_h * out_w
    channel = offsets // pixels
    pixel = offsets - channel * pixels
    oy = pixel // out_w
    ox = pixel - oy * out_w

    y_start = oy.to(tl.float32) * scale_y
    y_end = y_start + scale_y
    x_start = ox.to(tl.float32) * scale_x
    x_end = x_start + scale_x
    y0 = tl.floor(y_start).to(tl.int32)
    x0 = tl.floor(x_start).to(tl.int32)
    src_channel = 2 - channel

    value = tl.zeros((block,), dtype=tl.float32)
    for ky in tl.static_range(0, 3):
        sy = y0 + ky
        wy = tl.maximum(
            0.0,
            tl.minimum(y_end, sy.to(tl.float32) + 1.0)
            - tl.maximum(y_start, sy.to(tl.float32)),
        )
        for kx in tl.static_range(0, 3):
            sx = x0 + kx
            wx = tl.maximum(
                0.0,
                tl.minimum(x_end, sx.to(tl.float32) + 1.0)
                - tl.maximum(x_start, sx.to(tl.float32)),
            )
            src_offset = (sy * in_w + sx) * channels + src_channel
            sample = tl.load(
                src + src_offset,
                mask=active & (sy < in_h) & (sx < in_w),
                other=0.0,
            ).to(tl.float32)
            value += sample * wx * wy

    value *= (1.0 / (scale_x * scale_y * 255.0))
    tl.store(out + offsets, value, mask=active)


def can_use_triton_preprocess(frame_raw: torch.Tensor) -> bool:
    return (
        triton is not None
        and isinstance(frame_raw, torch.Tensor)
        and frame_raw.is_cuda
        and frame_raw.ndim == 3
        and frame_raw.shape[-1] in (3, 4)
        and frame_raw.dtype == torch.uint8
    )


def bgr_to_rgb_resize_norm(frame_raw: torch.Tensor, out_height: int, out_width: int) -> torch.Tensor:
    if not can_use_triton_preprocess(frame_raw):
        raise TypeError("Triton preprocess requires a CUDA uint8 HWC tensor with 3 or 4 channels")
    if not frame_raw.is_contiguous():
        frame_raw = frame_raw.contiguous()

    in_h = int(frame_raw.shape[0])
    in_w = int(frame_raw.shape[1])
    channels = int(frame_raw.shape[2])
    out_height = int(out_height)
    out_width = int(out_width)
    out = torch.empty((3, out_height, out_width), device=frame_raw.device, dtype=torch.float32)

    total = out.numel()
    block = 256
    scale_y = float(in_h) / float(out_height)
    scale_x = float(in_w) / float(out_width)

    if (
        torch.version.cuda is not None
        and channels == 4
        and in_h == out_height
        and in_w == out_width
    ):
        pixels = in_h * in_w
        _bgra_to_rgb_same_size_norm_kernel[(triton.cdiv(pixels, block),)](
            frame_raw,
            out,
            pixels,
            block,
        )
        return out

    grid = (triton.cdiv(total, block),)
    kernel = _bgr_to_rgb_resize_norm_kernel
    if 1.0 < scale_y <= 2.0 and 1.0 < scale_x <= 2.0:
        kernel = _bgr_to_rgb_area_resize_norm_kernel
    kernel[grid](
        frame_raw,
        out,
        total,
        in_h,
        in_w,
        channels,
        out_height,
        out_width,
        scale_y,
        scale_x,
        block,
    )
    return out
