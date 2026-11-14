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
    weight = tl.zeros((block,), dtype=tl.float32)
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
            # fp32 subtraction of two ~4K coordinates loses precision, so the
            # theoretical scale_x*scale_y is not the sum of the tap weights.
            # Normalise by the weights actually used. This also keeps border
            # pixels from being darkened when taps are clipped at the edge.
            valid = active & (sy >= 0) & (sy < in_h) & (sx >= 0) & (sx < in_w)
            tap_weight = tl.where(valid, wx * wy, 0.0)
            sample = tl.load(
                src + src_offset,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            value += sample * tap_weight
            weight += tap_weight

    value /= tl.maximum(weight * 255.0, 1.0e-6)
    tl.store(out + offsets, value, mask=active)


@triton.jit
def _media_pow(value, exponent):
    return tl.exp2(tl.log2(tl.maximum(value, 1.0e-20)) * exponent)


@triton.jit
def _media_hable(value):
    a, b, c, d, e, f = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
    return ((value * (a * value + c * b) + d * e) / (value * (a * value + b) + d * f)) - e / f


@triton.jit
def _media_to_rgb_norm_kernel(
    src,
    out,
    pixels: tl.constexpr,
    width: tl.constexpr,
    channels: tl.constexpr,
    rgbp: tl.constexpr,
    source_uint8: tl.constexpr,
    p010: tl.constexpr,
    transfer: tl.constexpr,
    gamut: tl.constexpr,
    yuv_matrix: tl.constexpr,
    full_range: tl.constexpr,
    chroma_location: tl.constexpr,
    block: tl.constexpr,
):
    pixel = tl.program_id(0) * block + tl.arange(0, block)
    active = pixel < pixels
    if rgbp:
        red = tl.load(src + pixel, mask=active, other=0).to(tl.float32)
        green = tl.load(src + pixels + pixel, mask=active, other=0).to(tl.float32)
        blue = tl.load(src + 2 * pixels + pixel, mask=active, other=0).to(tl.float32)
    elif yuv_matrix != 0:
        x = pixel % width
        y = pixel // width
        luma_sample = tl.load(src + pixel, mask=active, other=64 if p010 else 16)
        if p010:
            luma = (luma_sample.to(tl.int32) >> 6).to(tl.float32)
        else:
            luma = luma_sample.to(tl.float32)
        chroma_x = (x.to(tl.float32) - (0.5 if chroma_location == 2 else 0.0)) * 0.5
        chroma_y = (y.to(tl.float32) - (0.0 if chroma_location == 3 else 0.5)) * 0.5
        chroma_x0_raw = tl.floor(chroma_x).to(tl.int32)
        chroma_y0_raw = tl.floor(chroma_y).to(tl.int32)
        wx = chroma_x - chroma_x0_raw.to(tl.float32)
        wy = chroma_y - chroma_y0_raw.to(tl.float32)
        chroma_width = width // 2
        chroma_height = (pixels // width) // 2
        x0 = tl.minimum(tl.maximum(chroma_x0_raw, 0), chroma_width - 1)
        x1 = tl.minimum(tl.maximum(chroma_x0_raw + 1, 0), chroma_width - 1)
        y0 = tl.minimum(tl.maximum(chroma_y0_raw, 0), chroma_height - 1)
        y1 = tl.minimum(tl.maximum(chroma_y0_raw + 1, 0), chroma_height - 1)
        uv_base = pixels + y0 * width + x0 * 2
        uv_right = pixels + y0 * width + x1 * 2
        uv_bottom = pixels + y1 * width + x0 * 2
        uv_bottom_right = pixels + y1 * width + x1 * 2
        chroma_neutral = 512 if p010 else 128
        u00_raw = tl.load(src + uv_base, mask=active, other=chroma_neutral)
        v00_raw = tl.load(src + uv_base + 1, mask=active, other=chroma_neutral)
        u01_raw = tl.load(src + uv_right, mask=active, other=chroma_neutral)
        v01_raw = tl.load(src + uv_right + 1, mask=active, other=chroma_neutral)
        u10_raw = tl.load(src + uv_bottom, mask=active, other=chroma_neutral)
        v10_raw = tl.load(src + uv_bottom + 1, mask=active, other=chroma_neutral)
        u11_raw = tl.load(src + uv_bottom_right, mask=active, other=chroma_neutral)
        v11_raw = tl.load(src + uv_bottom_right + 1, mask=active, other=chroma_neutral)
        if p010:
            u00 = (u00_raw.to(tl.int32) >> 6).to(tl.float32)
            v00 = (v00_raw.to(tl.int32) >> 6).to(tl.float32)
            u01 = (u01_raw.to(tl.int32) >> 6).to(tl.float32)
            v01 = (v01_raw.to(tl.int32) >> 6).to(tl.float32)
            u10 = (u10_raw.to(tl.int32) >> 6).to(tl.float32)
            v10 = (v10_raw.to(tl.int32) >> 6).to(tl.float32)
            u11 = (u11_raw.to(tl.int32) >> 6).to(tl.float32)
            v11 = (v11_raw.to(tl.int32) >> 6).to(tl.float32)
        else:
            u00 = u00_raw.to(tl.float32)
            v00 = v00_raw.to(tl.float32)
            u01 = u01_raw.to(tl.float32)
            v01 = v01_raw.to(tl.float32)
            u10 = u10_raw.to(tl.float32)
            v10 = v10_raw.to(tl.float32)
            u11 = u11_raw.to(tl.float32)
            v11 = v11_raw.to(tl.float32)
        cb = (u00 + (u01 - u00) * wx) * (1.0 - wy) + (u10 + (u11 - u10) * wx) * wy
        cr = (v00 + (v01 - v00) * wx) * (1.0 - wy) + (v10 + (v11 - v10) * wx) * wy
        if p010:
            if full_range:
                luma *= 1.0 / 1023.0
                cb = (cb - 512.0) * (1.0 / 1023.0)
                cr = (cr - 512.0) * (1.0 / 1023.0)
            else:
                luma = (luma - 64.0) * (1.0 / 876.0)
                cb = (cb - 512.0) * (1.0 / 896.0)
                cr = (cr - 512.0) * (1.0 / 896.0)
        elif full_range:
            luma *= 1.0 / 255.0
            cb = (cb - 128.0) * (1.0 / 255.0)
            cr = (cr - 128.0) * (1.0 / 255.0)
        else:
            luma = (luma - 16.0) * (1.0 / 219.0)
            cb = (cb - 128.0) * (1.0 / 224.0)
            cr = (cr - 128.0) * (1.0 / 224.0)
        if yuv_matrix == 1:
            red = luma + 1.5748 * cr
            green = luma - 0.187324 * cb - 0.468124 * cr
            blue = luma + 1.8556 * cb
        elif yuv_matrix == 2:
            red = luma + 1.4746 * cr
            green = luma - 0.164553 * cb - 0.571353 * cr
            blue = luma + 1.8814 * cb
        else:
            red = luma + 1.402 * cr
            green = luma - 0.344136 * cb - 0.714136 * cr
            blue = luma + 1.772 * cb
        red = tl.minimum(tl.maximum(red, 0.0), 1.0)
        green = tl.minimum(tl.maximum(green, 0.0), 1.0)
        blue = tl.minimum(tl.maximum(blue, 0.0), 1.0)
    else:
        base = pixel * channels
        blue = tl.load(src + base, mask=active, other=0).to(tl.float32)
        green = tl.load(src + base + 1, mask=active, other=0).to(tl.float32)
        red = tl.load(src + base + 2, mask=active, other=0).to(tl.float32)

    if source_uint8 and yuv_matrix == 0:
        red *= 1.0 / 255.0
        green *= 1.0 / 255.0
        blue *= 1.0 / 255.0

    if transfer == 1:
        red_enc = _media_pow(tl.minimum(tl.maximum(red, 0.0), 1.0), 32.0 / 2523.0)
        green_enc = _media_pow(tl.minimum(tl.maximum(green, 0.0), 1.0), 32.0 / 2523.0)
        blue_enc = _media_pow(tl.minimum(tl.maximum(blue, 0.0), 1.0), 32.0 / 2523.0)
        red_num = tl.maximum(red_enc - 3424.0 / 4096.0, 0.0)
        green_num = tl.maximum(green_enc - 3424.0 / 4096.0, 0.0)
        blue_num = tl.maximum(blue_enc - 3424.0 / 4096.0, 0.0)
        red_den = tl.maximum(2413.0 / 128.0 - (2392.0 / 128.0) * red_enc, 1.0e-6)
        green_den = tl.maximum(2413.0 / 128.0 - (2392.0 / 128.0) * green_enc, 1.0e-6)
        blue_den = tl.maximum(2413.0 / 128.0 - (2392.0 / 128.0) * blue_enc, 1.0e-6)
        red = _media_pow(red_num / red_den, 16384.0 / 2610.0) * 100.0
        green = _media_pow(green_num / green_den, 16384.0 / 2610.0) * 100.0
        blue = _media_pow(blue_num / blue_den, 16384.0 / 2610.0) * 100.0
    elif transfer == 2:
        red_enc = tl.minimum(tl.maximum(red, 0.0), 1.0)
        green_enc = tl.minimum(tl.maximum(green, 0.0), 1.0)
        blue_enc = tl.minimum(tl.maximum(blue, 0.0), 1.0)
        red_scene = tl.where(
            red_enc <= 0.5,
            red_enc * red_enc / 3.0,
            (tl.exp((red_enc - 0.55991073) / 0.17883277) + 0.28466892) / 12.0,
        )
        green_scene = tl.where(
            green_enc <= 0.5,
            green_enc * green_enc / 3.0,
            (tl.exp((green_enc - 0.55991073) / 0.17883277) + 0.28466892) / 12.0,
        )
        blue_scene = tl.where(
            blue_enc <= 0.5,
            blue_enc * blue_enc / 3.0,
            (tl.exp((blue_enc - 0.55991073) / 0.17883277) + 0.28466892) / 12.0,
        )
        red = _media_pow(tl.maximum(red_scene, 0.0), 1.2) * 10.0
        green = _media_pow(tl.maximum(green_scene, 0.0), 1.2) * 10.0
        blue = _media_pow(tl.maximum(blue_scene, 0.0), 1.2) * 10.0

    if gamut == 1 or gamut == 2 or gamut == 3:
        if transfer == 1 or transfer == 2:
            red = tl.maximum(red, 0.0)
            green = tl.maximum(green, 0.0)
            blue = tl.maximum(blue, 0.0)
        else:
            red_enc = tl.minimum(tl.maximum(red, 0.0), 1.0)
            green_enc = tl.minimum(tl.maximum(green, 0.0), 1.0)
            blue_enc = tl.minimum(tl.maximum(blue, 0.0), 1.0)
            red = tl.where(
                red_enc < 0.081,
                red_enc / 4.5,
                _media_pow(tl.maximum((red_enc + 0.099) / 1.099, 0.0), 1.0 / 0.45),
            )
            green = tl.where(
                green_enc < 0.081,
                green_enc / 4.5,
                _media_pow(tl.maximum((green_enc + 0.099) / 1.099, 0.0), 1.0 / 0.45),
            )
            blue = tl.where(
                blue_enc < 0.081,
                blue_enc / 4.5,
                _media_pow(tl.maximum((blue_enc + 0.099) / 1.099, 0.0), 1.0 / 0.45),
            )
        if gamut == 1:
            out_red = 1.660491 * red - 0.587641 * green - 0.072850 * blue
            out_green = -0.124550 * red + 1.132900 * green - 0.008349 * blue
            out_blue = -0.018151 * red - 0.100579 * green + 1.118730 * blue
        elif gamut == 2:
            out_red = 1.22494018 * red - 0.22494018 * green
            out_green = -0.04205696 * red + 1.04205696 * green
            out_blue = -0.01963756 * red - 0.07863605 * green + 1.09827360 * blue
        else:
            out_red = 1.15751641 * red - 0.15496238 * green - 0.00255403 * blue
            out_green = -0.04150007 * red + 1.04556792 * green - 0.00406785 * blue
            out_blue = -0.01805004 * red - 0.07857827 * green + 1.09662831 * blue
        out_min = tl.minimum(tl.minimum(out_red, out_green), out_blue)
        out_luma = 0.2126 * out_red + 0.7152 * out_green + 0.0722 * out_blue
        compression = tl.where(
            out_min < 0.0,
            tl.minimum(
                tl.maximum(-out_min / tl.maximum(out_luma - out_min, 1.0e-6), 0.0),
                1.0,
            ),
            0.0,
        )
        red = tl.maximum(out_red + compression * (out_luma - out_red), 0.0)
        green = tl.maximum(out_green + compression * (out_luma - out_green), 0.0)
        blue = tl.maximum(out_blue + compression * (out_luma - out_blue), 0.0)

    if transfer == 1 or transfer == 2:
        # Linear HDR units are 100 cd/m^2. Hable's standard 11.2 white
        # point is 1120 cd/m^2, not 10000 cd/m^2.
        peak = 11.2 if transfer == 1 else 10.0
        red = _media_hable(red) / _media_hable(peak)
        green = _media_hable(green) / _media_hable(peak)
        blue = _media_hable(blue) / _media_hable(peak)
        red = tl.where(
            red < 0.0031308,
            red * 12.92,
            1.055 * _media_pow(tl.maximum(red, 0.0), 1.0 / 2.4) - 0.055,
        )
        green = tl.where(
            green < 0.0031308,
            green * 12.92,
            1.055 * _media_pow(tl.maximum(green, 0.0), 1.0 / 2.4) - 0.055,
        )
        blue = tl.where(
            blue < 0.0031308,
            blue * 12.92,
            1.055 * _media_pow(tl.maximum(blue, 0.0), 1.0 / 2.4) - 0.055,
        )
    elif gamut == 1 or gamut == 2 or gamut == 3:
        red = tl.where(
            red < 0.018,
            red * 4.5,
            1.099 * _media_pow(tl.maximum(red, 0.0), 0.45) - 0.099,
        )
        green = tl.where(
            green < 0.018,
            green * 4.5,
            1.099 * _media_pow(tl.maximum(green, 0.0), 0.45) - 0.099,
        )
        blue = tl.where(
            blue < 0.018,
            blue * 4.5,
            1.099 * _media_pow(tl.maximum(blue, 0.0), 0.45) - 0.099,
        )

    plane = pixels
    tl.store(out + pixel, tl.minimum(tl.maximum(red, 0.0), 1.0), mask=active)
    tl.store(out + plane + pixel, tl.minimum(tl.maximum(green, 0.0), 1.0), mask=active)
    tl.store(out + 2 * plane + pixel, tl.minimum(tl.maximum(blue, 0.0), 1.0), mask=active)


_MEDIA_TRITON_DISABLED: set[tuple[str, int | None]] = set()


def can_use_triton_media_preprocess(frame_raw: torch.Tensor) -> bool:
    device = getattr(frame_raw, "device", None)
    device_key = (str(getattr(device, "type", "")), getattr(device, "index", None))
    return (
        triton is not None
        and isinstance(frame_raw, torch.Tensor)
        and frame_raw.is_cuda
        and frame_raw.ndim in (2, 3)
        and frame_raw.dtype in (torch.uint8, torch.uint16, torch.float16, torch.float32)
        and device_key not in _MEDIA_TRITON_DISABLED
    )


def media_to_rgb_norm(
    frame_raw: torch.Tensor,
    *,
    rgbp: bool,
    transfer: str,
    primaries: str,
    yuv_matrix: str = "",
    full_range: bool = False,
    chroma_location: str = "left",
    p010: bool = False,
) -> torch.Tensor:
    if not can_use_triton_media_preprocess(frame_raw):
        raise TypeError("Triton media preprocess requires a supported GPU tensor")
    if not frame_raw.is_contiguous():
        frame_raw = frame_raw.contiguous()
    yuv_matrix = str(yuv_matrix or "").casefold()
    if yuv_matrix:
        expected_dtype = torch.uint16 if p010 else torch.uint8
        if frame_raw.ndim != 2 or frame_raw.dtype != expected_dtype:
            pixel_format = "P010" if p010 else "NV12"
            raise ValueError(
                f"{pixel_format} media frames require a two-dimensional {expected_dtype} tensor"
            )
        width = int(frame_raw.shape[1])
        height = int(frame_raw.shape[0]) * 2 // 3
        if width % 2 or height % 2 or int(frame_raw.shape[0]) != height * 3 // 2:
            raise ValueError(f"Invalid NV12 frame shape: {tuple(frame_raw.shape)}")
        matrix_id = {
            "bt709": 1,
            "bt2020nc": 2,
            "smpte170m": 3,
            "bt470bg": 3,
        }.get(yuv_matrix)
        if matrix_id is None:
            raise ValueError(f"Unsupported NV12 matrix: {yuv_matrix}")
        channels = 1
        chroma_id = {"left": 1, "center": 2, "topleft": 3}.get(
            str(chroma_location or "left").casefold(), 1
        )
    elif rgbp:
        if frame_raw.ndim != 3 or frame_raw.shape[0] != 3:
            raise ValueError(f"RGBP media frames require three planes, got {tuple(frame_raw.shape)}")
        height, width = int(frame_raw.shape[1]), int(frame_raw.shape[2])
        channels = 3
        matrix_id = 0
        chroma_id = 0
    else:
        if frame_raw.ndim != 3 or frame_raw.shape[-1] != 3:
            raise ValueError(f"Packed BGR media frames require three channels, got {tuple(frame_raw.shape)}")
        height, width = int(frame_raw.shape[0]), int(frame_raw.shape[1])
        channels = int(frame_raw.shape[-1])
        matrix_id = 0
        chroma_id = 0

    transfer = str(transfer).casefold()
    primaries = str(primaries).casefold()
    transfer_id = {"pq": 1, "hlg": 2}.get(transfer, 3 if transfer == "sdr" else 0)
    gamut_id = {"bt2020": 1, "smpte432": 2, "smpte431": 3}.get(primaries, 0)
    pixels = height * width
    block = 256
    out = torch.empty((1, 3, height, width), device=frame_raw.device, dtype=torch.float32)
    device = frame_raw.device
    device_key = (str(device.type), device.index)
    try:
        _media_to_rgb_norm_kernel[(triton.cdiv(pixels, block),)](
            frame_raw,
            out,
            pixels,
            width,
            channels,
            bool(rgbp),
            frame_raw.dtype == torch.uint8,
            bool(p010),
            transfer_id,
            gamut_id,
            matrix_id,
            bool(full_range),
            chroma_id,
            block,
        )
    except Exception:
        _MEDIA_TRITON_DISABLED.add(device_key)
        raise
    return out


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
