from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.errors import TritonError

from .triton_runtime import triton_runtime_available


@triton.jit
def _load_rgb(source, x, y, width: tl.constexpr, height: tl.constexpr,
              pixels: tl.constexpr, input_uint8: tl.constexpr, active):
    sx = tl.maximum(0, tl.minimum(x, width - 1))
    sy = tl.maximum(0, tl.minimum(y, height - 1))
    offset = sy * width + sx
    scale = 1.0 / 255.0 if input_uint8 else 1.0
    r = tl.load(source + offset, mask=active, other=0.0).to(tl.float32) * scale
    g = tl.load(source + pixels + offset, mask=active, other=0.0).to(tl.float32) * scale
    b = tl.load(source + pixels * 2 + offset, mask=active, other=0.0).to(tl.float32) * scale
    return r, g, b


@triton.jit
def _luma(r, g, b):
    return r * 0.299 + g * 0.587 + b * 0.114


@triton.jit
def _easu_set(direction_x, direction_y, length_value, weight, a, b, c, d, e):
    gradient_x = d - b
    inverse_x = 1.0 / tl.maximum(tl.maximum(tl.abs(d - c), tl.abs(c - b)), 1.0e-6)
    direction_x += gradient_x * weight
    normalized_x = tl.minimum(tl.maximum(tl.abs(gradient_x) * inverse_x, 0.0), 1.0)
    length_value += normalized_x * normalized_x * weight
    gradient_y = e - a
    inverse_y = 1.0 / tl.maximum(tl.maximum(tl.abs(e - c), tl.abs(c - a)), 1.0e-6)
    direction_y += gradient_y * weight
    normalized_y = tl.minimum(tl.maximum(tl.abs(gradient_y) * inverse_y, 0.0), 1.0)
    length_value += normalized_y * normalized_y * weight
    return direction_x, direction_y, length_value


@triton.jit
def _easu_weight(offset_x, offset_y, direction_x, direction_y,
                 length_x, length_y, lobe, clip_value):
    rotated_x = (offset_x * direction_x + offset_y * direction_y) * length_x
    rotated_y = (offset_x * -direction_y + offset_y * direction_x) * length_y
    distance_squared = tl.minimum(rotated_x * rotated_x + rotated_y * rotated_y, clip_value)
    weight_b = 0.4 * distance_squared - 1.0
    weight_a = lobe * distance_squared - 1.0
    weight_b = 1.5625 * weight_b * weight_b - 0.5625
    return weight_b * weight_a * weight_a


@triton.jit
def _easu_kernel(source, output, total: tl.constexpr,
                 width: tl.constexpr, height: tl.constexpr,
                 out_width: tl.constexpr, out_height: tl.constexpr,
                 pixels: tl.constexpr, out_pixels: tl.constexpr,
                 input_uint8: tl.constexpr, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < total
    output_y = offsets // out_width
    output_x = offsets - output_y * out_width
    source_x = (output_x.to(tl.float32) + 0.5) * (width / float(out_width)) - 0.5
    source_y = (output_y.to(tl.float32) + 0.5) * (height / float(out_height)) - 0.5
    base_x = tl.floor(source_x).to(tl.int32)
    base_y = tl.floor(source_y).to(tl.int32)
    pp_x = source_x - base_x.to(tl.float32)
    pp_y = source_y - base_y.to(tl.float32)

    br, bg, bb = _load_rgb(source, base_x, base_y - 1, width, height, pixels, input_uint8, active)
    cr, cg, cb = _load_rgb(source, base_x + 1, base_y - 1, width, height, pixels, input_uint8, active)
    er, eg, eb = _load_rgb(source, base_x - 1, base_y, width, height, pixels, input_uint8, active)
    fr, fg, fb = _load_rgb(source, base_x, base_y, width, height, pixels, input_uint8, active)
    gr, gg, gb = _load_rgb(source, base_x + 1, base_y, width, height, pixels, input_uint8, active)
    hr, hg, hb = _load_rgb(source, base_x + 2, base_y, width, height, pixels, input_uint8, active)
    ir, ig, ib = _load_rgb(source, base_x - 1, base_y + 1, width, height, pixels, input_uint8, active)
    jr, jg, jb = _load_rgb(source, base_x, base_y + 1, width, height, pixels, input_uint8, active)
    kr, kg, kb = _load_rgb(source, base_x + 1, base_y + 1, width, height, pixels, input_uint8, active)
    lr, lg, lb = _load_rgb(source, base_x + 2, base_y + 1, width, height, pixels, input_uint8, active)
    nr, ng, nb = _load_rgb(source, base_x, base_y + 2, width, height, pixels, input_uint8, active)
    ore, oge, obe = _load_rgb(source, base_x + 1, base_y + 2, width, height, pixels, input_uint8, active)

    bl = _luma(br, bg, bb); cl = _luma(cr, cg, cb)
    el = _luma(er, eg, eb); fl = _luma(fr, fg, fb)
    gl = _luma(gr, gg, gb); hl = _luma(hr, hg, hb)
    il = _luma(ir, ig, ib); jl = _luma(jr, jg, jb)
    kl = _luma(kr, kg, kb); ll = _luma(lr, lg, lb)
    nl = _luma(nr, ng, nb); ol = _luma(ore, oge, obe)
    direction_x = tl.zeros_like(source_x)
    direction_y = tl.zeros_like(source_y)
    length_value = tl.zeros_like(source_x)
    direction_x, direction_y, length_value = _easu_set(
        direction_x, direction_y, length_value,
        (1.0 - pp_x) * (1.0 - pp_y), bl, el, fl, gl, jl)
    direction_x, direction_y, length_value = _easu_set(
        direction_x, direction_y, length_value,
        pp_x * (1.0 - pp_y), cl, fl, gl, hl, kl)
    direction_x, direction_y, length_value = _easu_set(
        direction_x, direction_y, length_value,
        (1.0 - pp_x) * pp_y, fl, il, jl, kl, nl)
    direction_x, direction_y, length_value = _easu_set(
        direction_x, direction_y, length_value,
        pp_x * pp_y, gl, jl, kl, ll, ol)
    direction_length = direction_x * direction_x + direction_y * direction_y
    inverse_direction = tl.rsqrt(tl.maximum(direction_length, 1.0e-12))
    direction_x = tl.where(direction_length < 0.000030517578125, 1.0, direction_x * inverse_direction)
    direction_y = tl.where(direction_length < 0.000030517578125, 0.0, direction_y * inverse_direction)
    length_value = 0.25 * length_value * length_value
    stretch = 1.0 / tl.maximum(tl.maximum(tl.abs(direction_x), tl.abs(direction_y)), 1.0e-6)
    length_x = 1.0 + (stretch - 1.0) * length_value
    length_y = 1.0 - 0.5 * length_value
    lobe = 0.5 + (0.21 - 0.5) * length_value
    clip_value = 1.0 / tl.maximum(lobe, 1.0e-6)

    result_r = tl.zeros_like(source_x); result_g = tl.zeros_like(source_x)
    result_b = tl.zeros_like(source_x); weight_sum = tl.zeros_like(source_x)
    w = _easu_weight(-pp_x, -1.0 - pp_y, direction_x, direction_y, length_x, length_y, lobe, clip_value)
    result_r += br * w; result_g += bg * w; result_b += bb * w; weight_sum += w
    w = _easu_weight(1.0 - pp_x, -1.0 - pp_y, direction_x, direction_y, length_x, length_y, lobe, clip_value)
    result_r += cr * w; result_g += cg * w; result_b += cb * w; weight_sum += w
    w = _easu_weight(-1.0 - pp_x, 1.0 - pp_y, direction_x, direction_y, length_x, length_y, lobe, clip_value)
    result_r += ir * w; result_g += ig * w; result_b += ib * w; weight_sum += w
    w = _easu_weight(-pp_x, 1.0 - pp_y, direction_x, direction_y, length_x, length_y, lobe, clip_value)
    result_r += jr * w; result_g += jg * w; result_b += jb * w; weight_sum += w
    w = _easu_weight(-pp_x, -pp_y, direction_x, direction_y, length_x, length_y, lobe, clip_value)
    result_r += fr * w; result_g += fg * w; result_b += fb * w; weight_sum += w
    w = _easu_weight(-1.0 - pp_x, -pp_y, direction_x, direction_y, length_x, length_y, lobe, clip_value)
    result_r += er * w; result_g += eg * w; result_b += eb * w; weight_sum += w
    w = _easu_weight(1.0 - pp_x, 1.0 - pp_y, direction_x, direction_y, length_x, length_y, lobe, clip_value)
    result_r += kr * w; result_g += kg * w; result_b += kb * w; weight_sum += w
    w = _easu_weight(2.0 - pp_x, 1.0 - pp_y, direction_x, direction_y, length_x, length_y, lobe, clip_value)
    result_r += lr * w; result_g += lg * w; result_b += lb * w; weight_sum += w
    w = _easu_weight(2.0 - pp_x, -pp_y, direction_x, direction_y, length_x, length_y, lobe, clip_value)
    result_r += hr * w; result_g += hg * w; result_b += hb * w; weight_sum += w
    w = _easu_weight(1.0 - pp_x, -pp_y, direction_x, direction_y, length_x, length_y, lobe, clip_value)
    result_r += gr * w; result_g += gg * w; result_b += gb * w; weight_sum += w
    w = _easu_weight(1.0 - pp_x, 2.0 - pp_y, direction_x, direction_y, length_x, length_y, lobe, clip_value)
    result_r += ore * w; result_g += oge * w; result_b += obe * w; weight_sum += w

    safe_weight = tl.where(tl.abs(weight_sum) <= 1.0e-6, 1.0, weight_sum)
    result_r = tl.where(tl.abs(weight_sum) <= 1.0e-6, fr, result_r / safe_weight)
    result_g = tl.where(tl.abs(weight_sum) <= 1.0e-6, fg, result_g / safe_weight)
    result_b = tl.where(tl.abs(weight_sum) <= 1.0e-6, fb, result_b / safe_weight)
    min_r = tl.minimum(tl.minimum(fr, gr), tl.minimum(jr, kr)); max_r = tl.maximum(tl.maximum(fr, gr), tl.maximum(jr, kr))
    min_g = tl.minimum(tl.minimum(fg, gg), tl.minimum(jg, kg)); max_g = tl.maximum(tl.maximum(fg, gg), tl.maximum(jg, kg))
    min_b = tl.minimum(tl.minimum(fb, gb), tl.minimum(jb, kb)); max_b = tl.maximum(tl.maximum(fb, gb), tl.maximum(jb, kb))
    tl.store(output + offsets, tl.minimum(max_r, tl.maximum(min_r, result_r)), mask=active)
    tl.store(output + out_pixels + offsets, tl.minimum(max_g, tl.maximum(min_g, result_g)), mask=active)
    tl.store(output + out_pixels * 2 + offsets, tl.minimum(max_b, tl.maximum(min_b, result_b)), mask=active)


@triton.jit
def _rcas_kernel(source, output, total: tl.constexpr, width: tl.constexpr,
                 height: tl.constexpr, pixels: tl.constexpr,
                 sharpness: tl.constexpr, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < total
    y = offsets // width
    x = offsets - y * width
    br, bg, bb = _load_rgb(source, x, y - 1, width, height, pixels, False, active)
    dr, dg, db = _load_rgb(source, x - 1, y, width, height, pixels, False, active)
    er, eg, eb = _load_rgb(source, x, y, width, height, pixels, False, active)
    fr, fg, fb = _load_rgb(source, x + 1, y, width, height, pixels, False, active)
    hr, hg, hb = _load_rgb(source, x, y + 1, width, height, pixels, False, active)
    bl = _luma(br, bg, bb); dl = _luma(dr, dg, db); el = _luma(er, eg, eb)
    fl = _luma(fr, fg, fb); hl = _luma(hr, hg, hb)
    noise = tl.abs(0.25 * (bl + dl + fl + hl) - el)
    lmax = tl.maximum(tl.maximum(tl.maximum(bl, dl), tl.maximum(el, fl)), hl)
    lmin = tl.minimum(tl.minimum(tl.minimum(bl, dl), tl.minimum(el, fl)), hl)
    noise = 1.0 - 0.5 * tl.minimum(tl.maximum(noise / tl.maximum(tl.abs(lmax - lmin), 1.0e-6), 0.0), 1.0)
    min_r = tl.minimum(tl.minimum(br, dr), tl.minimum(fr, hr)); max_r = tl.maximum(tl.maximum(br, dr), tl.maximum(fr, hr))
    min_g = tl.minimum(tl.minimum(bg, dg), tl.minimum(fg, hg)); max_g = tl.maximum(tl.maximum(bg, dg), tl.maximum(fg, hg))
    min_b = tl.minimum(tl.minimum(bb, db), tl.minimum(fb, hb)); max_b = tl.maximum(tl.maximum(bb, db), tl.maximum(fb, hb))
    hit_min_r = tl.minimum(min_r, er) / tl.maximum(4.0 * max_r, 1.0e-6)
    hit_min_g = tl.minimum(min_g, eg) / tl.maximum(4.0 * max_g, 1.0e-6)
    hit_min_b = tl.minimum(min_b, eb) / tl.maximum(4.0 * max_b, 1.0e-6)
    hit_max_r = (1.0 - tl.maximum(max_r, er)) / tl.minimum(4.0 * min_r - 4.0, -1.0e-6)
    hit_max_g = (1.0 - tl.maximum(max_g, eg)) / tl.minimum(4.0 * min_g - 4.0, -1.0e-6)
    hit_max_b = (1.0 - tl.maximum(max_b, eb)) / tl.minimum(4.0 * min_b - 4.0, -1.0e-6)
    lobe = tl.maximum(tl.maximum(tl.maximum(-hit_min_r, hit_max_r), tl.maximum(-hit_min_g, hit_max_g)), tl.maximum(-hit_min_b, hit_max_b))
    contrast = tl.exp2(-2.0 * (1.0 - sharpness))
    lobe = tl.maximum(-0.1875, tl.minimum(lobe, 0.0)) * contrast * noise
    reciprocal = 1.0 / tl.maximum(tl.abs(4.0 * lobe + 1.0), 1.0e-6)
    out_r = tl.minimum(tl.maximum((lobe * (br + dr + fr + hr) + er) * reciprocal, 0.0), 1.0)
    out_g = tl.minimum(tl.maximum((lobe * (bg + dg + fg + hg) + eg) * reciprocal, 0.0), 1.0)
    out_b = tl.minimum(tl.maximum((lobe * (bb + db + fb + hb) + eb) * reciprocal, 0.0), 1.0)
    tl.store(output + offsets, out_r, mask=active)
    tl.store(output + pixels + offsets, out_g, mask=active)
    tl.store(output + pixels * 2 + offsets, out_b, mask=active)


def can_use_output_quality_triton(image: torch.Tensor) -> bool:
    return (
        triton_runtime_available(image.device)
        and image.ndim == 4
        and image.shape[0] == 1
        and image.shape[1] == 3
        and image.dtype in {torch.float16, torch.float32, torch.uint8}
    )


def easu_resize(image: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
    image = image.contiguous()
    _, _, height, width = image.shape
    output = torch.empty((1, 3, target_height, target_width), device=image.device, dtype=torch.float32)
    total = int(target_height) * int(target_width)
    block = 256
    _easu_kernel[(triton.cdiv(total, block),)](
        image, output, total, int(width), int(height), int(target_width), int(target_height),
        int(width) * int(height), total, image.dtype == torch.uint8, block,
    )
    return output


def apply_rcas(image: torch.Tensor, sharpness: float) -> torch.Tensor:
    image = image.contiguous().float()
    _, _, height, width = image.shape
    output = torch.empty_like(image)
    total = int(height) * int(width)
    block = 256
    _rcas_kernel[(triton.cdiv(total, block),)](
        image, output, total, int(width), int(height), total,
        max(0.0, min(1.0, float(sharpness))), block,
    )
    return output


@triton.jit
def _horizontal_area_srgb_kernel(
    source, output, total: tl.constexpr, width: tl.constexpr, height: tl.constexpr,
    target: tl.constexpr, channels: tl.constexpr, sb: tl.constexpr, sc: tl.constexpr,
    sy: tl.constexpr, sx: tl.constexpr, ratio: tl.constexpr, block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < total
    x = offsets % target
    row = offsets // target
    y = row % height
    plane = row // height
    channel = plane % channels
    batch = plane // channels
    base = batch * sb + channel * sc + y * sy + x * ratio * sx
    reduced = tl.full((block,), 0.0, tl.float32)
    for tap in range(ratio):
        value = tl.load(source + base + tap * sx, mask=active, other=0.0).to(tl.float32)
        value = tl.minimum(tl.maximum(value, 0.0), 1.0)
        linear = tl.where(value <= 0.04045, value / 12.92,
                          tl.exp2(2.4 * tl.log2(tl.maximum((value + 0.055) / 1.055, 1.0e-20))))
        reduced += linear / ratio
    reduced = tl.minimum(tl.maximum(reduced, 0.0), 1.0)
    encoded = tl.where(reduced <= 0.0031308, reduced * 12.92,
                       1.055 * tl.exp2(tl.log2(tl.maximum(reduced, 1.0e-20)) / 2.4) - 0.055)
    tl.store(output + offsets, tl.minimum(tl.maximum(encoded, 0.0), 1.0), mask=active)


def horizontal_area_srgb(image: torch.Tensor, target_width: int) -> torch.Tensor | None:
    """Fuse the existing integer box filter and exact sRGB transfer; no quality change."""
    width = int(image.shape[-1])
    if (not can_use_output_quality_triton(image) or image.dtype == torch.uint8
            or target_width <= 0 or width % target_width != 0 or width // target_width not in (2, 3, 4)):
        return None
    batches, channels, height, _ = image.shape
    output = torch.empty((batches, channels, height, target_width), device=image.device, dtype=image.dtype)
    total = output.numel()
    block = 256
    try:
        _horizontal_area_srgb_kernel[(triton.cdiv(total, block),)](
            image, output, total, width, height, target_width, channels,
            *image.stride(), width // target_width, block,
        )
    except TritonError:
        return None
    return output
