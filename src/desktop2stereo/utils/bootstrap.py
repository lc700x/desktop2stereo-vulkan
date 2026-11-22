from __future__ import annotations

import os

from .network import configure_huggingface_endpoint
from .platform_env import configure_platform_environment
from .settings import load_settings


def _normalize_legacy_settings(settings: dict, *, os_name: str | None = None) -> dict:
    """Expose the flat settings contract while the new schema is being migrated."""
    if "Stream Quality" in settings:
        return settings

    from stereo_runtime.model_capabilities import model_name_mapping

    model_mapping = model_name_mapping()

    def model_list() -> dict:
        result = {}
        for name in model_mapping:
            if name.startswith("InfiniDepth-"):
                resolutions = [192, 240, 304, 336, 384, 448, 512]
            elif name.startswith("DA3"):
                resolutions = [182, 224, 280, 322, 378, 434, 504]
            elif name == "DepthPro-Large":
                resolutions = [1536]
            else:
                resolutions = [196, 238, 294, 336, 392, 448, 518]
            result[name] = {"resolutions": resolutions}
        return result

    graphics = settings.get("graphics", {}) or {}
    capture = settings.get("capture", {}) or {}
    inference = settings.get("inference", {}) or {}
    stereo = settings.get("stereo", {}) or {}
    openxr = settings.get("openxr", {}) or {}
    output = settings.get("output", {}) or {}
    model = str(inference.get("model", "Distill-Any-Depth-Base"))
    flat_defaults = {
        "Monitor Index": int(capture.get("monitor_index", 1)),
        "Monitor Identity": None,
        "Capture Mode": str(capture.get("mode", "Monitor")),
        "Window Title": None,
        "Depth Model": model,
        "Model List": model_list(),
        "Depth Resolution": 294,
        "Depth Strength": float(stereo.get("depth_strength", 0.15)),
        "Depth Pop": 0.0,
        "Anti-aliasing": 2,
        "FP16": True,
        "torch.compile": False,
        "TensorRT": False,
        "Recompile TensorRT": False,
        "MIGraphX": False,
        "Recompile MIGraphX": False,
        "CoreML": os_name == "Darwin",
        "Recompile CoreML": False,
        "OpenVINO": False,
        "Recompile OpenVINO": False,
        "Computing Device": 0,
        "Run Mode": "OpenXR Link" if bool(openxr.get("enabled", False)) else "Local Viewer",
        "Display Mode": "Half-SBS",
        "Show FPS": True,
        "Convergence": float(stereo.get("convergence", 0.5)),
        "Processing Resolution": "Auto",
        "Target FPS": 0,
        "Display Fit Mode": "contain",
        "Fill 16:9": True,
        "VSync": False,
        "Fix Viewer Aspect": False,
        "Language": "EN",
        "XR Headset Model": "PICO",
        "XR Preview Window": True,
        "Controller Model": "PICO",
        "Environment Model": "Default",
        "Stream Protocol": "WebRTC",
        "Streamer Port": 1122,
        "Stream Quality": 100,
        "Stream Key": "live",
        "Stereo Mix": None,
        "CRF": 23,
        "Audio Delay": -0.1,
        "Capture Tool": str(capture.get("tool", "none")),
        "NVIDIA Frame Generation": False,
        "Lossless Scaling Support": False,
        "LSFG Support": False,
        "Stereo Output": None,
        "Stereo Output Identity": None,
        "Render Size Policy": "scaled",
        "Render Scale": "100%",
        "XR Render": 1.0,
        "XR Render Mode": "auto",
        "Render Fixed Width": 1920,
        "Render Fixed Height": 1080,
        "Render Max Pixels": 3840 * 2160,
        "Render Min Dimension": 480,
        "Render Align": 1,
        "Upscaler": "Off",
        "Upscaler Sharpness": 0.0,
    }
    normalized = dict(settings)
    for key, value in flat_defaults.items():
        normalized.setdefault(key, value)
    return normalized


def bootstrap_settings(path: str, *, os_name: str) -> dict:
    settings = _normalize_legacy_settings(load_settings(path), os_name=os_name)
    configure_platform_environment(os_name)
    configure_huggingface_endpoint()
    if str(settings.get("Debug Mode", False) or False).strip().lower() in ("1", "true", "yes", "on"):
        os.environ["D2S_DEBUG"] = "1"
    return settings
