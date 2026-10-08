"""Optional CUDA/ROCm display FXAA; failed compilers use the tensor fallback."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .triton_runtime import triton_runtime_available


@triton.jit
def _read(image, x, y, batch, channel: tl.constexpr, W: tl.constexpr, H: tl.constexpr,
          C: tl.constexpr, first, last, active, U8: tl.constexpr):
    x = tl.minimum(tl.maximum(x, first), last).to(tl.int32)
    y = tl.minimum(tl.maximum(y, 0), H - 1).to(tl.int32)
    value = tl.load(image + (batch * C + channel) * W * H + y * W + x, mask=active, other=0).to(tl.float32)
    if U8:
        value *= 1.0 / 255.0
    return value


@triton.jit
def _luma(image, x, y, batch, W: tl.constexpr, H: tl.constexpr, C: tl.constexpr,
          first, last, active, U8: tl.constexpr):
    r = _read(image, x, y, batch, 0, W, H, C, first, last, active, U8)
    g = _read(image, x, y, batch, 1, W, H, C, first, last, active, U8)
    b = _read(image, x, y, batch, 2, W, H, C, first, last, active, U8)
    return r * 0.299 + g * 0.587 + b * 0.114


@triton.jit
def _sample_luma(image, x, y, batch, W: tl.constexpr, H: tl.constexpr, C: tl.constexpr,
                 first, last, active, U8: tl.constexpr):
    bx, by = tl.floor(x), tl.floor(y)
    fx, fy = x - bx, y - by
    a = _luma(image, bx, by, batch, W, H, C, first, last, active, U8)
    b = _luma(image, bx + 1, by, batch, W, H, C, first, last, active, U8)
    c = _luma(image, bx, by + 1, batch, W, H, C, first, last, active, U8)
    d = _luma(image, bx + 1, by + 1, batch, W, H, C, first, last, active, U8)
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


@triton.jit
def _decode(value):
    return tl.where(value <= 0.04045, value / 12.92,
                    tl.exp2(2.4 * tl.log2(tl.maximum((value + 0.055) / 1.055, 1e-20))))


@triton.jit
def _sample_rgb_channel(image, x, y, batch, channel: tl.constexpr, W: tl.constexpr,
                        H: tl.constexpr, C: tl.constexpr, first, last, active, U8: tl.constexpr):
    bx, by = tl.floor(x), tl.floor(y)
    fx, fy = x - bx, y - by
    a = _decode(_read(image, bx, by, batch, channel, W, H, C, first, last, active, U8))
    b = _decode(_read(image, bx + 1, by, batch, channel, W, H, C, first, last, active, U8))
    c = _decode(_read(image, bx, by + 1, batch, channel, W, H, C, first, last, active, U8))
    d = _decode(_read(image, bx + 1, by + 1, batch, channel, W, H, C, first, last, active, U8))
    linear = tl.minimum(tl.maximum((a * (1 - fx) + b * fx) * (1 - fy) +
                                   (c * (1 - fx) + d * fx) * fy, 0), 1)
    return tl.where(linear <= 0.0031308, 12.92 * linear,
                    1.055 * tl.exp2(tl.log2(tl.maximum(linear, 1e-20)) / 2.4) - 0.055)


@triton.jit
def _fxaa_kernel(image, output, W: tl.constexpr, H: tl.constexpr, C: tl.constexpr,
                 B: tl.constexpr, SPLIT: tl.constexpr, U8: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offset < B * W * H
    batch = offset // (W * H)
    pixel = offset % (W * H)
    x, y = pixel % W, pixel // W
    if SPLIT > 0 and SPLIT < W:
        first, last = tl.where(x < SPLIT, 0, SPLIT), tl.where(x < SPLIT, SPLIT - 1, W - 1)
    else:
        first, last = 0, W - 1
    m = _luma(image, x, y, batch, W, H, C, first, last, active, U8)
    n = _luma(image, x, y - 1, batch, W, H, C, first, last, active, U8)
    s = _luma(image, x, y + 1, batch, W, H, C, first, last, active, U8)
    w = _luma(image, x - 1, y, batch, W, H, C, first, last, active, U8)
    e = _luma(image, x + 1, y, batch, W, H, C, first, last, active, U8)
    minimum = tl.minimum(m, tl.minimum(tl.minimum(n, s), tl.minimum(w, e)))
    maximum = tl.maximum(m, tl.maximum(tl.maximum(n, s), tl.maximum(w, e)))
    contrast = maximum - minimum
    edge = active & (contrast >= tl.maximum(0.0312, maximum * 0.125) + 1e-5)
    normal_x, normal_y, final_offset = tl.full((BLOCK,), 0, tl.float32), tl.full((BLOCK,), 0, tl.float32), tl.full((BLOCK,), 0, tl.float32)
    if tl.sum(edge.to(tl.int32), 0) > 0:
        nw = _luma(image, x - 1, y - 1, batch, W, H, C, first, last, edge, U8)
        ne = _luma(image, x + 1, y - 1, batch, W, H, C, first, last, edge, U8)
        sw = _luma(image, x - 1, y + 1, batch, W, H, C, first, last, edge, U8)
        se = _luma(image, x + 1, y + 1, batch, W, H, C, first, last, edge, U8)
        horizontal = 2 * tl.abs(n + s - 2 * m) + tl.abs(nw + sw - 2 * w) + tl.abs(ne + se - 2 * e)
        vertical = 2 * tl.abs(w + e - 2 * m) + tl.abs(nw + ne - 2 * n) + tl.abs(sw + se - 2 * s)
        horizontal_edge = horizontal >= vertical - 1e-6
        first_neighbor, second_neighbor = tl.where(horizontal_edge, n, w), tl.where(horizontal_edge, s, e)
        g1, g2 = tl.abs(first_neighbor - m), tl.abs(second_neighbor - m)
        negative_side = g1 >= g2 - 1e-6
        sign = tl.where(negative_side, -1.0, 1.0)
        normal_x, normal_y = tl.where(horizontal_edge, 0.0, sign), tl.where(horizontal_edge, sign, 0.0)
        tangent_x, tangent_y = tl.where(horizontal_edge, 1.0, 0.0), tl.where(horizontal_edge, 0.0, 1.0)
        edge_x, edge_y = x + normal_x * 0.5, y + normal_y * 0.5
        mean = (m + tl.where(negative_side, first_neighbor, second_neighbor)) * 0.5
        threshold = tl.maximum(g1, g2) * 0.25 - 1.1e-4
        dn, dp = tl.full((BLOCK,), 1.0, tl.float32), tl.full((BLOCK,), 1.0, tl.float32)
        vn = _sample_luma(image, edge_x - tangent_x, edge_y - tangent_y, batch, W, H, C, first, last, edge, U8) - mean
        vp = _sample_luma(image, edge_x + tangent_x, edge_y + tangent_y, batch, W, H, C, first, last, edge, U8) - mean
        done_n, done_p = ~edge | (tl.abs(vn) >= threshold), ~edge | (tl.abs(vp) >= threshold)
        for i in tl.static_range(6):
            step = 1.5 if i == 0 else 4.0 if i == 4 else 8.0 if i == 5 else 2.0
            dn += tl.where(done_n, 0.0, step)
            dp += tl.where(done_p, 0.0, step)
            next_n = _sample_luma(image, edge_x - tangent_x * dn, edge_y - tangent_y * dn, batch, W, H, C, first, last, ~done_n & edge, U8) - mean
            next_p = _sample_luma(image, edge_x + tangent_x * dp, edge_y + tangent_y * dp, batch, W, H, C, first, last, ~done_p & edge, U8) - mean
            vn, vp = tl.where(done_n, vn, next_n), tl.where(done_p, vp, next_p)
            done_n, done_p = done_n | (tl.abs(vn) >= threshold), done_p | (tl.abs(vp) >= threshold)
        endpoint = tl.where(dn <= dp, vn, vp)
        span_valid = (endpoint < 0) != (m < mean)
        coverage = tl.where(span_valid, 0.5 - tl.minimum(dn, dp) / (dn + dp), 0.0)
        neighborhood = (2 * (n + s + w + e) + nw + ne + sw + se) / 12.0
        subpixel = tl.minimum(tl.maximum(tl.abs(neighborhood - m) / tl.maximum(contrast, 1e-6), 0), 1)
        subpixel = subpixel * subpixel * (3 - 2 * subpixel)
        final_offset = tl.maximum(coverage, subpixel * subpixel * 0.75)
    for channel in tl.static_range(C):
        original = tl.load(image + (batch * C + channel) * W * H + pixel, mask=active, other=0)
        value = original
        if channel < 3:
            filtered = _sample_rgb_channel(image, x + normal_x * final_offset, y + normal_y * final_offset,
                                           batch, channel, W, H, C, first, last, edge, U8)
            if U8:
                filtered = tl.floor(tl.minimum(tl.maximum(filtered * 255 + 0.5, 0), 255))
            value = tl.where(edge, filtered, original)
        tl.store(output + (batch * C + channel) * W * H + pixel, value, mask=active)


def fxaa(image: torch.Tensor, eye_width: int) -> torch.Tensor:
    if not triton_runtime_available(image.device):
        raise RuntimeError("Triton display compiler is unavailable")
    batch, channels, height, width = map(int, image.shape)
    output = torch.empty_like(image)
    _fxaa_kernel[(triton.cdiv(batch * height * width, 128),)](
        image, output, width, height, channels, batch, int(eye_width), image.dtype == torch.uint8, 128,
    )
    return output
