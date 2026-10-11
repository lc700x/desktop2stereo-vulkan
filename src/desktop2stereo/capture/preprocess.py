from __future__ import annotations

import numpy as np


def capture_frame_to_rgb(
    frame_raw,
    target_resolution=None,
    *,
    target_height=None,
    size=None,
    device=None,
    use_torch=None,
    output="auto",
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
