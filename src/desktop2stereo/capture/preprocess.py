from __future__ import annotations

import numpy as np


_LINEAR_GAMUT_TO_BT709 = {
    "bt2020": (
        (1.660491, -0.587641, -0.072850),
        (-0.124550, 1.132900, -0.008349),
        (-0.018151, -0.100579, 1.118730),
    ),
    "smpte432": (
        (1.22494018, -0.22494018, 0.0),
        (-0.04205696, 1.04205696, 0.0),
        (-0.01963756, -0.07863605, 1.09827360),
    ),
    "smpte431": (
        (1.15751641, -0.15496238, -0.00255403),
        (-0.04150007, 1.04556792, -0.00406785),
        (-0.01805004, -0.07857827, 1.09662831),
    ),
}


def capture_frame_to_rgb(
    frame_raw,
    target_resolution=None,
    *,
    target_height=None,
    size=None,
    device=None,
    use_torch=None,
    output="auto",
    original_format="",
    frame_raw_device=None,
    capture_copy_mode=None,
    capture_zero_copy=None,
):
    """
    Convert a raw capture frame to the RGB frame consumed by stereo_runtime.

    This keeps capture-side responsibilities out of stereo runtime code:
    raw BGRA/BGR frame -> RGB frame, explicit device transfer, and the existing
    processing-resolution resize used by the Desktop2Stereo pipeline.
    """
    target_resolution = _resolve_target_resolution(
        target_resolution,
        target_height=target_height,
        size=size,
    )
    if use_torch is None:
        use_torch = _is_torch_tensor(frame_raw)
    if use_torch or output == "tensor":
        return _capture_frame_to_rgb_torch(
            frame_raw,
            target_resolution,
            device=device,
            frame_raw_device=frame_raw_device,
            original_format=original_format,
            capture_copy_mode=capture_copy_mode,
            capture_zero_copy=capture_zero_copy,
        )
    return _capture_frame_to_rgb_numpy(frame_raw, target_resolution)


def prepare_rgb_for_stereo_runtime(frame_rgb, *, device=None):
    """Prepare capture-owned RGB frame for stereo_runtime's strict RGB input contract."""
    import torch

    if device is None:
        from utils import DEVICE
        device = DEVICE

    scale_from_uint8 = False
    if isinstance(frame_rgb, np.ndarray):
        scale_from_uint8 = np.issubdtype(frame_rgb.dtype, np.integer)
        frame_rgb = torch.from_numpy(frame_rgb)
        if frame_rgb.ndim == 3 and frame_rgb.shape[-1] == 3:
            frame_rgb = frame_rgb.permute(2, 0, 1).contiguous()

    if not isinstance(frame_rgb, torch.Tensor):
        raise TypeError("frame_rgb must be a torch.Tensor or numpy.ndarray prepared by capture")

    source_frame = frame_rgb
    already_normalized = getattr(source_frame, "_d2s_preprocess_backend", None) in {
        "torch_bgr_norm",
        "torch_bgr_resize_norm",
        "triton_bgr_resize_norm",
        "triton_bgr_area_resize_norm",
        "triton_media_hdr_fused",
        "triton_media_rgb_fused",
        "triton_media_nv12_hdr_fused",
        "triton_media_nv12_fused",
        "triton_media_p010_hdr_fused",
        "triton_media_p010_fused",
        "triton_media_hdr_resize",
        "triton_media_rgb_resize",
        "triton_media_nv12_hdr_resize",
        "triton_media_nv12_resize",
        "triton_media_p010_hdr_resize",
        "triton_media_p010_resize",
        "torch_media_nv12",
    }
    if frame_rgb.ndim == 3 and frame_rgb.shape[0] == 3:
        tensor = frame_rgb.unsqueeze(0)
    elif frame_rgb.ndim == 4 and frame_rgb.shape[1] == 3:
        tensor = frame_rgb
    else:
        raise ValueError(f"frame_rgb must be RGB CHW/BCHW for stereo_runtime, got shape {tuple(frame_rgb.shape)}")

    scale_from_uint8 = scale_from_uint8 or not tensor.is_floating_point()
    tensor = tensor.to(device=device, dtype=torch.float32)
    if scale_from_uint8:
        tensor = tensor.mul_(1.0 / 255.0)
    if scale_from_uint8 or not already_normalized:
        tensor = tensor.clamp_(0.0, 1.0)
    _copy_preprocess_metadata(source_frame, tensor)
    return tensor


def _is_torch_tensor(value):
    torch_type = getattr(type(value), "__module__", "")
    return torch_type.startswith("torch") and hasattr(value, "permute")


def _same_torch_device(source_device, requested_device, torch_module) -> bool:
    source_device = torch_module.device(source_device)
    requested_device = torch_module.device(requested_device)
    if source_device.type != requested_device.type:
        return False

    def current_index(device_type):
        backend = getattr(torch_module, device_type, None)
        current_device = getattr(backend, "current_device", None)
        if callable(current_device):
            try:
                return int(current_device())
            except Exception:
                pass
        if device_type == "cuda" and torch_module.cuda.is_available():
            return int(torch_module.cuda.current_device())
        return 0

    def device_index(device):
        if device.index is not None:
            return device.index
        return current_index(device.type)

    return device_index(source_device) == device_index(requested_device)


def _capture_frame_to_rgb_torch(
    frame_raw,
    target_resolution,
    *,
    device=None,
    frame_raw_device=None,
    original_format="",
    capture_copy_mode=None,
    capture_zero_copy=None,
):
    import torch
    import torch.nn.functional as F

    if device is None:
        from utils import DEVICE
        device = DEVICE

    origin_device = str(frame_raw_device or _frame_device_text(frame_raw))
    input_kind = _frame_kind_text(frame_raw)
    if isinstance(frame_raw, np.ndarray):
        frame_raw = torch.from_numpy(frame_raw)

    if not isinstance(frame_raw, torch.Tensor):
        raise TypeError("frame_raw must be a torch.Tensor or numpy.ndarray for tensor preprocess")
    media_format = str(original_format or "").casefold()
    if media_format.startswith(("nv12_", "p010le_")):
        return _capture_nv12_to_rgb_torch(
            frame_raw,
            target_resolution,
            original_format=media_format,
            requested_device=device,
            origin_device=origin_device,
            input_kind=input_kind,
            capture_copy_mode=capture_copy_mode,
            capture_zero_copy=capture_zero_copy,
        )
    if media_format.startswith(("bgr24_", "rgbp")):
        if media_format.startswith("rgbp"):
            media_height, media_width = int(frame_raw.shape[-2]), int(frame_raw.shape[-1])
            media_rgbp = True
        else:
            media_height, media_width = int(frame_raw.shape[0]), int(frame_raw.shape[1])
            media_rgbp = False
        media_out_height, media_out_width = _resolve_target_size(
            target_resolution, media_height, media_width
        )
        media_input = frame_raw if isinstance(frame_raw, torch.Tensor) else torch.from_numpy(frame_raw)
        requested_device = torch.device(device)
        if not _same_torch_device(media_input.device, requested_device, torch):
            media_input = media_input.to(device=requested_device, non_blocking=True)
        try:
            from capture.preprocess_triton import (
                can_use_triton_media_preprocess,
                media_to_rgb_norm,
            )

            if can_use_triton_media_preprocess(media_input):
                parts = media_format.split("_")
                transfer = parts[1] if len(parts) > 1 else ""
                primaries = parts[2] if len(parts) > 2 else ""
                out = media_to_rgb_norm(
                    media_input,
                    rgbp=media_rgbp,
                    transfer=transfer,
                    primaries=primaries,
                )
                resized = (media_height, media_width) != (media_out_height, media_out_width)
                if resized:
                    out = F.interpolate(
                        out,
                        size=(media_out_height, media_out_width),
                        mode="bilinear",
                        align_corners=False,
                        antialias=media_out_height < media_height or media_out_width < media_width,
                    ).clamp_(0.0, 1.0)
                    backend = (
                        "triton_media_hdr_resize"
                        if transfer in {"pq", "hlg"}
                        else "triton_media_rgb_resize"
                    )
                else:
                    backend = (
                        "triton_media_hdr_fused"
                        if transfer in {"pq", "hlg"}
                        else "triton_media_rgb_fused"
                    )
                _mark_preprocess_metadata(
                    out,
                    backend=backend,
                    input_kind=input_kind,
                    origin_device=origin_device,
                    output_device=_frame_device_text(out),
                    capture_copy_mode=capture_copy_mode,
                    capture_zero_copy=capture_zero_copy,
                )
                return out
        except Exception:
            pass
    if media_format.startswith("bgr24_"):
        frame_rgbp = frame_raw[..., [2, 1, 0]].permute(2, 0, 1).contiguous()
        return _capture_rgbp_to_rgb_torch(
            frame_rgbp,
            target_resolution,
            original_format=media_format.replace("bgr24_", "rgbp_", 1),
            requested_device=device,
            origin_device=origin_device,
            input_kind=input_kind,
            capture_copy_mode=capture_copy_mode,
            capture_zero_copy=capture_zero_copy,
        )
    if media_format.startswith("rgbp"):
        return _capture_rgbp_to_rgb_torch(
            frame_raw,
            target_resolution,
            original_format=media_format,
            requested_device=device,
            origin_device=origin_device,
            input_kind=input_kind,
            capture_copy_mode=capture_copy_mode,
            capture_zero_copy=capture_zero_copy,
        )
    if frame_raw.ndim != 3 or frame_raw.shape[-1] not in (3, 4):
        raise ValueError(f"frame_raw must be HWC BGR/BGRA, got shape {tuple(frame_raw.shape)}")

    h0, w0 = frame_raw.shape[:2]
    new_height, new_width = _resolve_target_size(target_resolution, h0, w0)
    requested_device = torch.device(device)
    coreml_raw_bgr = None
    if requested_device.type == "mps" and frame_raw.device.type == "cpu":
        # Keep the capture-owned host bytes for CoreML. The provider converts
        # and resizes directly to model resolution, avoiding a second full
        # render-resolution CPU preprocessing pass.
        coreml_raw_bgr = frame_raw

    try:
        from capture.preprocess_triton import bgr_to_rgb_resize_norm, can_use_triton_preprocess

        if can_use_triton_preprocess(frame_raw) and _same_torch_device(frame_raw.device, requested_device, torch):
            out = bgr_to_rgb_resize_norm(frame_raw, new_height, new_width)
            scale_y = float(h0) / float(new_height)
            scale_x = float(w0) / float(new_width)
            backend = "triton_bgr_resize_norm"
            if 1.0 < scale_y <= 2.0 and 1.0 < scale_x <= 2.0:
                backend = "triton_bgr_area_resize_norm"
            _mark_preprocess_metadata(
                out,
                backend=backend,
                input_kind=input_kind,
                origin_device=origin_device,
                output_device=_frame_device_text(out),
                capture_copy_mode=capture_copy_mode,
                capture_zero_copy=capture_zero_copy,
            )
            _mark_coreml_raw_bgr(out, coreml_raw_bgr)
            return out
    except Exception:
        pass

    source_device = torch.device(frame_raw.device)
    if not _same_torch_device(source_device, requested_device, torch):
        frame_raw = frame_raw.to(device=requested_device, non_blocking=True)

    if media_format == "bgr24" and frame_raw.shape[-1] == 3:
        try:
            from capture.preprocess_triton import bgr_to_rgb_resize_norm, can_use_triton_preprocess

            if can_use_triton_preprocess(frame_raw):
                full_resolution_rgb = bgr_to_rgb_resize_norm(frame_raw, h0, w0).unsqueeze(0)
                if (new_height, new_width) != (h0, w0):
                    out = F.interpolate(
                        full_resolution_rgb,
                        size=(new_height, new_width),
                        mode="bilinear",
                        align_corners=False,
                        antialias=new_height < h0 or new_width < w0,
                    ).clamp_(0.0, 1.0)
                else:
                    out = full_resolution_rgb
                _mark_preprocess_metadata(
                    out,
                    backend="triton_bgr_resize_norm",
                    input_kind=input_kind,
                    origin_device=origin_device,
                    output_device=_frame_device_text(out),
                    capture_copy_mode=capture_copy_mode,
                    capture_zero_copy=capture_zero_copy,
                )
                _mark_coreml_raw_bgr(out, coreml_raw_bgr)
                return out
        except Exception:
            pass

    frame_rgb = frame_raw[..., [2, 1, 0]].permute(2, 0, 1).contiguous()
    with torch.no_grad():
        frame_float = frame_rgb.to(dtype=torch.float32).mul_(1.0 / 255.0)
        if new_height == h0 and new_width == w0:
            out = frame_float.unsqueeze(0).clamp_(0.0, 1.0)
            _mark_preprocess_metadata(
                out,
                backend="torch_bgr_norm",
                input_kind=input_kind,
                origin_device=origin_device,
                output_device=_frame_device_text(out),
                capture_copy_mode=capture_copy_mode,
                capture_zero_copy=capture_zero_copy,
            )
            _mark_coreml_raw_bgr(out, coreml_raw_bgr)
            return out
        out = F.interpolate(
            frame_float.unsqueeze(0),
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
            antialias=new_height < h0 or new_width < w0,
        ).clamp_(0.0, 1.0)
        _mark_preprocess_metadata(
            out,
            backend="torch_bgr_resize_norm",
            input_kind=input_kind,
            origin_device=origin_device,
            output_device=_frame_device_text(out),
            capture_copy_mode=capture_copy_mode,
            capture_zero_copy=capture_zero_copy,
        )
        _mark_coreml_raw_bgr(out, coreml_raw_bgr)
        return out


def _capture_nv12_to_rgb_torch(
    frame_raw,
    target_resolution,
    *,
    original_format,
    requested_device,
    origin_device,
    input_kind,
    capture_copy_mode=None,
    capture_zero_copy=None,
):
    import torch
    import torch.nn.functional as F

    p010 = original_format.startswith("p010le_")
    expected_dtype = torch.uint16 if p010 else torch.uint8
    if frame_raw.ndim != 2 or frame_raw.dtype != expected_dtype:
        pixel_format = "P010" if p010 else "NV12"
        raise ValueError(
            f"{pixel_format} media frames require a two-dimensional {expected_dtype} tensor, "
            f"got {tuple(frame_raw.shape)} {frame_raw.dtype}"
        )
    width = int(frame_raw.shape[1])
    height = int(frame_raw.shape[0]) * 2 // 3
    if width % 2 or height % 2 or int(frame_raw.shape[0]) != height * 3 // 2:
        raise ValueError(f"Invalid NV12 frame shape: {tuple(frame_raw.shape)}")
    requested_device = torch.device(requested_device)
    if not _same_torch_device(frame_raw.device, requested_device, torch):
        frame_raw = frame_raw.to(device=requested_device, non_blocking=True)
    if not frame_raw.is_contiguous():
        frame_raw = frame_raw.contiguous()

    parts = original_format.split("_")
    transfer = parts[1] if len(parts) > 1 else "sdr"
    primaries = parts[2] if len(parts) > 2 else "bt709"
    yuv_matrix = parts[3] if len(parts) > 3 else "bt709"
    full_range = len(parts) > 4 and parts[4] == "pc"
    chroma_location = parts[5] if len(parts) > 5 else "left"
    out_height, out_width = _resolve_target_size(target_resolution, height, width)
    out = None
    try:
        from capture.preprocess_triton import can_use_triton_media_preprocess, media_to_rgb_norm

        if can_use_triton_media_preprocess(frame_raw):
            out = media_to_rgb_norm(
                frame_raw,
                rgbp=False,
                transfer=transfer,
                primaries=primaries,
                yuv_matrix=yuv_matrix,
                full_range=full_range,
                chroma_location=chroma_location,
                p010=p010,
            )
            if (height, width) != (out_height, out_width):
                out = F.interpolate(
                    out,
                    size=(out_height, out_width),
                    mode="bilinear",
                    align_corners=False,
                    antialias=out_height < height or out_width < width,
                ).clamp_(0.0, 1.0)
    except Exception:
        out = None

    if out is None:
        uv = frame_raw[height:].reshape(height // 2, width // 2, 2)
        if p010:
            y = torch.bitwise_right_shift(frame_raw[:height].to(torch.int32), 6).to(torch.float32)
            uv = torch.bitwise_right_shift(uv.to(torch.int32), 6).to(torch.float32)
            chroma_center = 512.0
            if full_range:
                y = y * (1.0 / 1023.0)
                chroma_scale = 1.0 / 1023.0
            else:
                y = (y - 64.0) * (1.0 / 876.0)
                chroma_scale = 1.0 / 896.0
        else:
            y = frame_raw[:height].to(dtype=torch.float32)
            uv = uv.to(dtype=torch.float32)
            chroma_center = 128.0
            if full_range:
                y = y * (1.0 / 255.0)
                chroma_scale = 1.0 / 255.0
            else:
                y = (y - 16.0) * (1.0 / 219.0)
                chroma_scale = 1.0 / 224.0
        uv = uv.permute(2, 0, 1).unsqueeze(0)
        uv = F.interpolate(uv, size=(height, width), mode="bilinear", align_corners=False)[0]
        cb = (uv[0] - chroma_center) * chroma_scale
        cr = (uv[1] - chroma_center) * chroma_scale
        if yuv_matrix == "bt2020nc":
            red = y + 1.4746 * cr
            green = y - 0.164553 * cb - 0.571353 * cr
            blue = y + 1.8814 * cb
        elif yuv_matrix in {"smpte170m", "bt470bg"}:
            red = y + 1.402 * cr
            green = y - 0.344136 * cb - 0.714136 * cr
            blue = y + 1.772 * cb
        else:
            red = y + 1.5748 * cr
            green = y - 0.187324 * cb - 0.468124 * cr
            blue = y + 1.8556 * cb
        frame_rgb = torch.stack((red, green, blue), dim=0).clamp_(0.0, 1.0)
        if transfer in {"pq", "hlg", "sdr"}:
            frame_rgb = _convert_rgbp_to_bt709(frame_rgb, transfer, primaries, torch)
        if (height, width) != (out_height, out_width):
            frame_rgb = F.interpolate(
                frame_rgb.unsqueeze(0),
                size=(out_height, out_width),
                mode="bilinear",
                align_corners=False,
                antialias=out_height < height or out_width < width,
            )
        else:
            frame_rgb = frame_rgb.unsqueeze(0)
        out = frame_rgb.clamp_(0.0, 1.0)
        backend = "torch_media_p010" if p010 else "torch_media_nv12"
    else:
        resized = (height, width) != (out_height, out_width)
        prefix = "triton_media_p010" if p010 else "triton_media_nv12"
        if resized:
            backend = (
                f"{prefix}_hdr_resize"
                if transfer in {"pq", "hlg"}
                else f"{prefix}_resize"
            )
        else:
            backend = (
                f"{prefix}_hdr_fused"
                if transfer in {"pq", "hlg"}
                else f"{prefix}_fused"
            )
    _mark_preprocess_metadata(
        out,
        backend=backend,
        input_kind=input_kind,
        origin_device=origin_device,
        output_device=_frame_device_text(out),
        capture_copy_mode=capture_copy_mode,
        capture_zero_copy=capture_zero_copy,
    )
    return out


def _capture_rgbp_to_rgb_torch(
    frame_raw,
    target_resolution,
    *,
    original_format,
    requested_device,
    origin_device,
    input_kind,
    capture_copy_mode=None,
    capture_zero_copy=None,
):
    import torch
    import torch.nn.functional as F

    if frame_raw.ndim == 3 and frame_raw.shape[0] in (3, 4):
        frame_rgb = frame_raw[:3]
    elif frame_raw.ndim == 4 and frame_raw.shape[1] in (3, 4):
        if frame_raw.shape[0] != 1:
            raise ValueError("RGBP media frames must have a single frame batch")
        frame_rgb = frame_raw[0, :3]
    else:
        raise ValueError(f"RGBP frame must be CHW/BCHW, got shape {tuple(frame_raw.shape)}")

    requested_device = torch.device(requested_device)
    if not _same_torch_device(frame_rgb.device, requested_device, torch):
        frame_rgb = frame_rgb.to(device=requested_device, non_blocking=True)
    height, width = int(frame_rgb.shape[1]), int(frame_rgb.shape[2])
    new_height, new_width = _resolve_target_size(target_resolution, height, width)
    with torch.no_grad():
        if frame_rgb.is_floating_point():
            frame_float = frame_rgb.to(dtype=torch.float32)
        else:
            frame_float = frame_rgb.to(dtype=torch.float32).mul_(1.0 / 255.0)

        parts = original_format.split("_")
        transfer = parts[1] if len(parts) > 1 else ""
        primaries = parts[2] if len(parts) > 2 else ""
        if transfer in {"pq", "hlg", "sdr"}:
            frame_float = _convert_rgbp_to_bt709(frame_float, transfer, primaries, torch)

        if new_height != height or new_width != width:
            frame_float = F.interpolate(
                frame_float.unsqueeze(0),
                size=(new_height, new_width),
                mode="bilinear",
                align_corners=False,
                antialias=new_height < height or new_width < width,
            )
        else:
            frame_float = frame_float.unsqueeze(0)
        out = frame_float.clamp_(0.0, 1.0)
    _mark_preprocess_metadata(
        out,
        backend="torch_rgbp_hdr_resize_norm" if transfer in {"pq", "hlg"} else "torch_rgbp_resize_norm",
        input_kind=input_kind,
        origin_device=origin_device,
        output_device=_frame_device_text(out),
        capture_copy_mode=capture_copy_mode,
        capture_zero_copy=capture_zero_copy,
    )
    return out


def _convert_rgbp_to_bt709(frame_rgb, transfer, primaries, torch_module):
    def hable(value):
        a, b, c, d, e, f = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
        return ((value * (a * value + c * b) + d * e) / (value * (a * value + b) + d * f)) - e / f

    if transfer == "pq":
        encoded = frame_rgb.clamp(0.0, 1.0).pow(32.0 / 2523.0)
        numerator = (encoded - 3424.0 / 4096.0).clamp_min(0.0)
        denominator = (2413.0 / 128.0 - (2392.0 / 128.0) * encoded).clamp_min(1e-6)
        linear = (numerator / denominator).pow(16384.0 / 2610.0) * 100.0
        # Linear HDR units are 100 cd/m^2. Hable's standard 11.2 white
        # point is therefore 1120 cd/m^2; normalising to 10000 cd/m^2 made
        # SDR highlights and skin tones noticeably dark and oversaturated.
        peak = 11.2
    elif transfer == "hlg":
        encoded = frame_rgb.clamp(0.0, 1.0)
        linear = torch_module.where(
            encoded <= 0.5,
            encoded.square() / 3.0,
            (torch_module.exp((encoded - 0.55991073) / 0.17883277) + 0.28466892) / 12.0,
        ).clamp_min(0.0).pow(1.2) * 10.0
        peak = 10.0
    elif primaries in _LINEAR_GAMUT_TO_BT709:
        encoded = frame_rgb.clamp(0.0, 1.0)
        linear = torch_module.where(
            encoded < 0.081,
            encoded / 4.5,
            ((encoded + 0.099) / 1.099).clamp_min(0.0).pow(1.0 / 0.45),
        )
        peak = 1.0
    else:
        return frame_rgb.clamp(0.0, 1.0)

    gamut_matrix = _LINEAR_GAMUT_TO_BT709.get(primaries)
    if gamut_matrix is not None:
        matrix = torch_module.tensor(gamut_matrix, dtype=linear.dtype, device=linear.device)
        linear = torch_module.matmul(matrix, linear.reshape(3, -1)).reshape_as(linear)
        luminance = (
            linear
            * torch_module.tensor(
                (0.2126, 0.7152, 0.0722), dtype=linear.dtype, device=linear.device
            ).view(3, *([1] * (linear.ndim - 1)))
        ).sum(dim=0, keepdim=True)
        minimum = linear.min(dim=0, keepdim=True).values
        compression = torch_module.where(
            minimum < 0.0,
            (-minimum / (luminance - minimum).clamp_min(1e-6)).clamp(0.0, 1.0),
            torch_module.zeros_like(minimum),
        )
        # Desaturate out-of-gamut colours toward their luminance instead of
        # clipping each channel independently. Per-channel clipping shifts
        # hue in saturated BT.2020 highlights.
        linear = (linear + compression * (luminance - linear)).clamp_min_(0.0)

    if transfer in {"pq", "hlg"}:
        peak_value = torch_module.tensor(
            peak, dtype=linear.dtype, device=linear.device
        )
        linear = hable(linear.clamp_min(0.0)) / hable(peak_value)
        return torch_module.where(
            linear < 0.0031308,
            linear * 12.92,
            1.055 * linear.clamp_min(0.0).pow(1.0 / 2.4) - 0.055,
        ).clamp(0.0, 1.0)

    return torch_module.where(
        linear < 0.018,
        linear * 4.5,
        1.099 * linear.clamp_min(0.0).pow(0.45) - 0.099,
    ).clamp(0.0, 1.0)


def _capture_frame_to_rgb_numpy(frame_raw, target_resolution):
    import cv2

    is_umat = isinstance(frame_raw, cv2.UMat)
    shape = frame_raw.get().shape if is_umat else frame_raw.shape
    h0, w0 = shape[:2]

    new_height, new_width = _resolve_target_size(target_resolution, h0, w0)

    if shape[2] == 4:
        frame_rgb = cv2.cvtColor(frame_raw, cv2.COLOR_BGRA2RGB)
    elif shape[2] == 3:
        frame_rgb = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported capture frame channel count: {shape[2]}")

    if new_height == h0 and new_width == w0:
        return frame_rgb

    interpolation = cv2.INTER_AREA if new_height < h0 or new_width < w0 else cv2.INTER_CUBIC
    return cv2.resize(frame_rgb, (new_width, new_height), interpolation=interpolation)


def _resolve_target_resolution(target_resolution, *, target_height=None, size=None):
    values = [value for value in (target_resolution, target_height, size) if value is not None]
    if len(values) != 1:
        raise TypeError("Provide exactly one of target_resolution, target_height, or size")
    return values[0]


def _mark_preprocess_metadata(
    tensor,
    *,
    backend,
    input_kind,
    origin_device,
    output_device,
    capture_copy_mode=None,
    capture_zero_copy=None,
):
    setattr(tensor, "_d2s_preprocess_backend", backend)
    setattr(tensor, "_d2s_preprocess_input_kind", input_kind)
    setattr(tensor, "_d2s_preprocess_device_origin", origin_device)
    setattr(tensor, "_d2s_preprocess_device_output", output_device)
    setattr(tensor, "_d2s_preprocess_device_transfer", f"{origin_device}->{output_device}")
    if capture_copy_mode is not None:
        setattr(tensor, "_d2s_capture_copy_mode", str(capture_copy_mode))
    if capture_zero_copy is not None:
        setattr(tensor, "_d2s_capture_zero_copy", bool(capture_zero_copy))


def _copy_preprocess_metadata(source, target):
    for name in (
        "_d2s_preprocess_backend",
        "_d2s_preprocess_input_kind",
        "_d2s_preprocess_device_origin",
        "_d2s_preprocess_device_output",
        "_d2s_preprocess_device_transfer",
        "_d2s_capture_copy_mode",
        "_d2s_capture_zero_copy",
        "_d2s_coreml_raw_bgr",
    ):
        if hasattr(source, name):
            setattr(target, name, getattr(source, name))


def _mark_coreml_raw_bgr(tensor, value) -> None:
    if value is not None:
        setattr(tensor, "_d2s_coreml_raw_bgr", value)


def _frame_kind_text(value):
    if isinstance(value, np.ndarray):
        return "numpy"
    if _is_torch_tensor(value):
        return "torch.Tensor"
    return type(value).__name__


def _frame_device_text(value):
    if isinstance(value, np.ndarray):
        return "cpu"
    device = getattr(value, "device", None)
    if device is None:
        return "unknown"
    return str(device)


def _resolve_target_size(target_resolution, source_height, source_width):
    if isinstance(target_resolution, (tuple, list)) and len(target_resolution) == 2:
        target_width = _even_size(target_resolution[0])
        target_height = _even_size(target_resolution[1])
        return target_height, target_width

    target_height = _even_size(target_resolution)
    target_width = int(source_width * target_height / source_height)
    target_width = _even_size(target_width)
    return target_height, target_width


def _even_size(value):
    return max(2, (int(value) // 2) * 2)


def prepare_rgb_for_depth_runtime(frame_rgb, *, device=None):
    """Compatibility alias for older host code."""
    return prepare_rgb_for_stereo_runtime(frame_rgb, device=device)
