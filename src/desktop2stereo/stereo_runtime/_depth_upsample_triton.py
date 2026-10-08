from __future__ import annotations

import torch
import triton
import triton.language as tl


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
    total = pixels * batches
    active = offsets < total
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
    vmin = tl.minimum(tl.minimum(d00, d10), tl.minimum(d01, d11))
    vmax = tl.maximum(tl.maximum(d00, d10), tl.maximum(d01, d11))
    bilinear = w00 * d00 + w10 * d10 + w01 * d01 + w11 * d11

    rgb_base = batch * 3 * pixels
    guide_pixels = depth_height * depth_width
    guide_base = batch * 3 * guide_pixels
    tr = tl.load(rgb + rgb_base + pixel, active, 0.0)
    tg = tl.load(rgb + rgb_base + pixels + pixel, active, 0.0)
    tb = tl.load(rgb + rgb_base + 2 * pixels + pixel, active, 0.0)
    g00r = tl.load(guide + guide_base + i00, active, 0.0)
    g00g = tl.load(guide + guide_base + guide_pixels + i00, active, 0.0)
    g00b = tl.load(guide + guide_base + 2 * guide_pixels + i00, active, 0.0)
    g10r = tl.load(guide + guide_base + i10, active, 0.0)
    g10g = tl.load(guide + guide_base + guide_pixels + i10, active, 0.0)
    g10b = tl.load(guide + guide_base + 2 * guide_pixels + i10, active, 0.0)
    g01r = tl.load(guide + guide_base + i01, active, 0.0)
    g01g = tl.load(guide + guide_base + guide_pixels + i01, active, 0.0)
    g01b = tl.load(guide + guide_base + 2 * guide_pixels + i01, active, 0.0)
    g11r = tl.load(guide + guide_base + i11, active, 0.0)
    g11g = tl.load(guide + guide_base + guide_pixels + i11, active, 0.0)
    g11b = tl.load(guide + guide_base + 2 * guide_pixels + i11, active, 0.0)
    delta00 = (tl.abs(tr - g00r) + tl.abs(tg - g00g) + tl.abs(tb - g00b)) / 3.0
    delta10 = (tl.abs(tr - g10r) + tl.abs(tg - g10g) + tl.abs(tb - g10b)) / 3.0
    delta01 = (tl.abs(tr - g01r) + tl.abs(tg - g01g) + tl.abs(tb - g01b)) / 3.0
    delta11 = (tl.abs(tr - g11r) + tl.abs(tg - g11g) + tl.abs(tb - g11b)) / 3.0
    q00 = w00 * tl.exp(-delta00 / color_sigma)
    q10 = w10 * tl.exp(-delta10 / color_sigma)
    q01 = w01 * tl.exp(-delta01 / color_sigma)
    q11 = w11 * tl.exp(-delta11 / color_sigma)
    total = q00 + q10 + q01 + q11
    guided = (q00 * d00 + q10 * d10 + q01 * d01 + q11 * d11) / tl.maximum(total, 1.0e-8)
    midpoint = (vmin + vmax) * 0.5
    high00 = d00 >= midpoint
    high10 = d10 >= midpoint
    high01 = d01 >= midpoint
    high11 = d11 >= midpoint
    high_score = (
        q00 * high00.to(tl.float32)
        + q10 * high10.to(tl.float32)
        + q01 * high01.to(tl.float32)
        + q11 * high11.to(tl.float32)
    )
    low_score = total - high_score
    nearest_delta = tl.minimum(tl.minimum(delta00, delta10), tl.minimum(delta01, delta11))
    choose_high = tl.where(
        delta00 == nearest_delta,
        high00,
        tl.where(delta10 == nearest_delta, high10,
                 tl.where(delta01 == nearest_delta, high01, high11)),
    )
    s00 = q00 * (choose_high == high00).to(tl.float32)
    s10 = q10 * (choose_high == high10).to(tl.float32)
    s01 = q01 * (choose_high == high01).to(tl.float32)
    s11 = q11 * (choose_high == high11).to(tl.float32)
    selected_total = s00 + s10 + s01 + s11
    selected = (s00 * d00 + s10 * d10 + s01 * d01 + s11 * d11) / tl.maximum(
        selected_total, 1.0e-8
    )
    color_min = tl.minimum(tl.minimum(delta00, delta10), tl.minimum(delta01, delta11))
    color_max = tl.maximum(tl.maximum(delta00, delta10), tl.maximum(delta01, delta11))
    use_class = (
        (vmax - vmin >= edge_threshold)
        & (selected_total > 1.0e-8)
        & (color_max - color_min >= 0.005)
    )
    guided = tl.where(use_class, selected, guided)
    value = tl.where(vmax - vmin >= edge_threshold, guided, bilinear)
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
    block = 256
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
    )
    return output
