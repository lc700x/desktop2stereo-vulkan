from __future__ import annotations

from streaming.audio import STEREO_MIX_NAMES
from streaming.config import DEFAULT_PORT

from stereo_runtime.model_capabilities import (
    COMPILE_FIX_KEYWORDS,
    DISABLE_COREML_KEYWORDS,
    DISABLE_CUDNN_KEYWORDS,
    DISABLE_MIGRAPHX_KEYWORDS,
    DISABLE_OPENVINO_KEYWORDS,
    DISABLE_TRITON_KEYWORDS,
    DISABLE_TRT_KEYWORDS,
    FORCE_FP32_KEYWORDS,
    TRT_FIX_KEYWORDS,
)

from .app_info import DEBUG, OS_NAME, VERSION
from .bootstrap import bootstrap_settings
from .display import (
    _get_device_name_from_mss_monitor,
    get_monitor_size,
)
from .network import get_local_ip
from .runtime_state import shutdown_event
from .settings import read_yaml


_settings = None
_runtime_exports = None
_device_runtime = None


_RUNTIME_EXPORT_ATTRS = {
    "MODEL_MAPPING": "model_mapping",
    "STREAM_QUALITY": "stream_quality",
    "STREAM_PORT": "stream_port",
    "LOCAL_IP": "local_ip",
    "RUN_MODE": "run_mode",
    "STREAM_MODE": "stream_mode",
    "USE_3D_MONITOR": "use_3d_monitor",
    "LOSSLESS_SCALING_SUPPORT": "lossless_scaling_support",
    "MODEL": "model",
    "MODEL_ID": "model_id",
    "ALL_MODELS": "all_models",
    "CACHE_PATH": "cache_path",
    "DEPTH_RESOLUTION": "depth_resolution",
    "DEVICE_ID": "device_id",
    "FP16": "fp16",
    "MONITOR_INDEX": "monitor_index",
    "DISPLAY_MODE": "display_mode",
    "STEREO_DISPLAY_INDEX": "stereo_display_index",
    "STEREO_DISPLAY_SELECTION": "stereo_display_selection",
    "OUTPUT_RESOLUTION": "output_resolution",
    "RENDER_SIZE_CONFIG": "render_size_config",
    "SHOW_FPS": "show_fps",
    "DEPTH_STRENGTH": "depth_strength",
    "CONVERGENCE": "convergence",
    "CAPTURE_MODE": "capture_mode",
    "WINDOW_TITLE": "window_title",
    "TARGET_FPS": "target_fps",
    "FPS": "fps",
    "FOREGROUND_SCALE": "foreground_scale",
    "AA_STRENGTH": "aa_strength",
    "USE_TORCH_COMPILE": "use_torch_compile",
    "USE_TENSORRT": "use_tensorrt",
    "RECOMPILE_TRT": "recompile_trt",
    "USE_COREML": "use_coreml",
    "RECOMPILE_COREML": "recompile_coreml",
    "USE_MIGRAPHX": "use_migraphx",
    "RECOMPILE_MIGRAPHX": "recompile_migraphx",
    "USE_OPENVINO": "use_openvino",
    "RECOMPILE_OPENVINO": "recompile_openvino",
    "CAPTURE_TOOL": "capture_tool",
    "FILL_16_9": "fill_16_9",
    "LOCAL_VSYNC": "local_vsync",
    "UPSCALER": "upscaler",
    "UPSCALER_SHARPNESS": "upscaler_sharpness",
    "FIX_VIEWER_ASPECT": "fix_viewer_aspect",
    "STEREOMIX_DEVICE": "stereo_mix_device",
    "STREAM_KEY": "stream_key",
    "AUDIO_DELAY": "audio_delay",
    "CRF": "crf",
    "LANG": "language",
    "ROWS": "controller_help_rows",
    "ENV_ROWS": "environment_help_rows",
    "CONTROLLER_MODEL": "controller_model",
    "ENVIRONMENT_MODEL": "environment_model",
    "XR_HEADSET_MODEL": "xr_headset_model",
    "OPENXR_SCREEN_WIDTH": "openxr_screen_width",
    "OPENXR_SCREEN_DISTANCE": "openxr_screen_distance",
    "XR_PREVIEW_WINDOW": "xr_preview_window",
}


_VIEWER_ATTRS = {
    "crop_icon",
    "get_font_type",
    "hide_window_from_capture",
    "send_ctrl_cmd_f",
    "set_window_mouse_passthrough",
    "set_window_to_bottom",
    "show_window_in_capture",
}


def _get_settings():
    global _settings
    if _settings is None:
        _settings = bootstrap_settings("settings.yaml", os_name=OS_NAME)
    return _settings


def _get_runtime_exports():
    global _runtime_exports
    if _runtime_exports is None:
        from .runtime_exports import resolve_runtime_exports

        _runtime_exports = resolve_runtime_exports(_get_settings(), os_name=OS_NAME)
    return _runtime_exports


def _get_device_runtime():
    global _device_runtime
    if _device_runtime is None:
        from .device_runtime import resolve_device_runtime

        _device_runtime = resolve_device_runtime(__getattr__("DEVICE_ID"))
    return _device_runtime


def __getattr__(name):
    if name == "settings":
        value = _get_settings()
    elif name in _RUNTIME_EXPORT_ATTRS:
        value = getattr(_get_runtime_exports(), _RUNTIME_EXPORT_ATTRS[name])
    elif name == "DEVICE":
        value = _get_device_runtime().device
    elif name == "DEVICE_INFO":
        value = _get_device_runtime().device_info
    elif name in _VIEWER_ATTRS:
        import viewer

        value = getattr(viewer, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


__all__ = [
    "AA_STRENGTH",
    "ALL_MODELS",
    "AUDIO_DELAY",
    "CACHE_PATH",
    "CAPTURE_MODE",
    "CAPTURE_TOOL",
    "COMPILE_FIX_KEYWORDS",
    "CONVERGENCE",
    "CONTROLLER_MODEL",
    "CRF",
    "DEBUG",
    "DEFAULT_PORT",
    "DEPTH_RESOLUTION",
    "DEPTH_STRENGTH",
    "DEVICE",
    "DEVICE_ID",
    "DEVICE_INFO",
    "DISABLE_COREML_KEYWORDS",
    "DISABLE_CUDNN_KEYWORDS",
    "DISABLE_MIGRAPHX_KEYWORDS",
    "DISABLE_OPENVINO_KEYWORDS",
    "DISABLE_TRITON_KEYWORDS",
    "DISABLE_TRT_KEYWORDS",
    "DISPLAY_MODE",
    "ENVIRONMENT_MODEL",
    "ENV_ROWS",
    "FILL_16_9",
    "FIX_VIEWER_ASPECT",
    "FORCE_FP32_KEYWORDS",
    "FOREGROUND_SCALE",
    "FP16",
    "FPS",
    "LANG",
    "LOCAL_IP",
    "LOCAL_VSYNC",
    "LOSSLESS_SCALING_SUPPORT",
    "MODEL",
    "MODEL_ID",
    "MODEL_MAPPING",
    "MONITOR_INDEX",
    "OS_NAME",
    "OPENXR_SCREEN_DISTANCE",
    "OPENXR_SCREEN_WIDTH",
    "OUTPUT_RESOLUTION",
    "RECOMPILE_COREML",
    "RECOMPILE_MIGRAPHX",
    "RECOMPILE_OPENVINO",
    "RECOMPILE_TRT",
    "RENDER_SIZE_CONFIG",
    "ROWS",
    "RUN_MODE",
    "SHOW_FPS",
    "STEREOMIX_DEVICE",
    "STEREO_DISPLAY_INDEX",
    "STEREO_DISPLAY_SELECTION",
    "STEREO_MIX_NAMES",
    "STREAM_KEY",
    "STREAM_MODE",
    "STREAM_PORT",
    "STREAM_QUALITY",
    "TARGET_FPS",
    "TRT_FIX_KEYWORDS",
    "UPSCALER",
    "UPSCALER_SHARPNESS",
    "USE_3D_MONITOR",
    "USE_COREML",
    "USE_MIGRAPHX",
    "USE_OPENVINO",
    "USE_TENSORRT",
    "USE_TORCH_COMPILE",
    "VERSION",
    "WINDOW_TITLE",
    "XR_HEADSET_MODEL",
    "XR_PREVIEW_WINDOW",
    "_get_device_name_from_mss_monitor",
    "crop_icon",
    "get_font_type",
    "get_local_ip",
    "get_monitor_size",
    "hide_window_from_capture",
    "read_yaml",
    "send_ctrl_cmd_f",
    "set_window_mouse_passthrough",
    "set_window_to_bottom",
    "settings",
    "show_window_in_capture",
    "shutdown_event",
]
