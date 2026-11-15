"""Platform capture and GPU-resource capability probes.

The probes report what can be established on the current machine. They never
turn API/module presence into a claim that a complete zero-copy pipeline works.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
from pathlib import Path
from typing import Any


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _linux_display_server() -> str:
    if os.environ.get("WAYLAND_DISPLAY"):
        return "wayland"
    if os.environ.get("DISPLAY"):
        return "x11"
    return "unknown"


def probe_linux_capture_capabilities() -> dict[str, Any]:
    """Describe the implemented MSS path and unimplemented native alternatives."""
    display_server = _linux_display_server()
    mss_available = _module_available("mss")
    xlib_available = _module_available("Xlib")
    pipewire_cli = shutil.which("pw-cli") is not None or shutil.which("pw-cat") is not None
    render_nodes = sorted(
        path for path in ("/dev/dri/renderD128", "/dev/dri/renderD129", "/dev/dri/renderD130")
        if os.path.exists(path)
    )
    x11_window = bool(mss_available and xlib_available and display_server == "x11")
    return {
        "platform": "Linux",
        "display_server": display_server,
        "implemented": {
            "monitor_capture": bool(mss_available),
            "window_capture": x11_window,
            "backend": "MSS + Xlib" if mss_available else None,
            "resource_type": "cpu_bgra_numpy" if mss_available else None,
            "gpu_to_cpu": False,
            "gpu_copy_count": 0,
            "zero_copy": False,
        },
        "wayland": {
            "detected": display_server == "wayland",
            "pipewire_cli": pipewire_cli,
            "capture_implemented": False,
            "status": "待验证/未实现",
            "reason": "项目没有 PipeWire screen-capture consumer",
        },
        "dmabuf": {
            "render_nodes": render_nodes,
            "capture_implemented": False,
            "explicit_sync": "未实现",
            "implicit_sync": "未验证",
            "status": "待验证/未实现",
        },
        "window_capture_boundary": (
            "X11 coordinate crop only; no window-surface zero-copy"
            if x11_window
            else "window capture unavailable without X11 + Xlib"
        ),
        "gpu_identity": {
            "drm_render_node": None,
            "pci_bdf": None,
            "vulkan_device_uuid": None,
            "status": "未实现",
        },
        "fallback_order": [
            "PipeWire/DMA-BUF（未来，待验证）",
            "MSS + Xlib CPU兼容路径",
            "CPU推理与CPU合成/输出",
        ],
    }


def probe_windows_capture_capabilities() -> dict[str, Any]:
    """Probe Windows capture tools and resource-interop implementation presence."""
    report: dict[str, Any] = {
        "platform": "Windows",
        "tools": {},
        "resource_contract": {
            "d3d11_texture": "optional_borrowed_resource",
            "adapter_luid": "required_for_cross_api_share",
            "ownership": "borrowed_until_frame_release",
            "mismatch_action": "reject_share_and_fallback",
        },
        "d3d11_vulkan_import": {
            "implementation_present": False,
            "hardware_verified": False,
            "status": "待真实硬件验证",
        },
        "gpu_media_decode": {
            "decoder_bridge": "ffmpeg-d3d11va",
            "bridge_present": False,
            "transport": "shared D3D11 BGRA texture + hardware fence -> HIP tensor",
            "hdr_output": "GPU D3D11 Video Processor -> SDR BT.709",
            "supported_input_transfers": ["PQ", "HLG", "BT.2020 SDR", "SDR full/studio range"],
            "cpu_pixel_readback": False,
            "hot_path_cpu_synchronize": False,
            "hardware_verified": False,
            "status": "bridge unavailable",
        },
        "directml": {
            "available": False,
            "model_operator_support": "not_probed",
            "d3d11_shared_resource": "not_implemented",
            "vulkan_external_memory": "not_probed",
            "hardware_verified": False,
        },
    }
    if platform.system() != "Windows":
        report["reason"] = "Windows-only probe skipped on this platform"
        return report

    decoder_bridge = (
        Path(__file__).resolve().parent
        / "native"
        / "windows"
        / "d2s_ffmpeg_d3d11_decoder.dll"
    )
    report["gpu_media_decode"].update({
        "bridge_present": decoder_bridge.is_file(),
        "bridge_path": str(decoder_bridge),
        "hardware_verified": False,
        "status": (
            "bundled; hardware capability is checked per source/codec"
            if decoder_bridge.is_file()
            else "bridge unavailable; AMF/software fallback only"
        ),
    })

    report["tools"] = {
        "WindowsCapture": {
            "module": "windows_capture",
            "available": _module_available("windows_capture"),
            "modes": ["Monitor", "Window"],
            "resource_output": "optional D3D11 texture; CPU compatibility copy otherwise",
        },
        "WindowsCaptureCUDA": {
            "module": "wc_cuda",
            "available": _module_available("wc_cuda"),
            "modes": ["Monitor", "Window"],
            "resource_output": "CUDA tensor",
        },
        "WindowsCaptureROCm": {
            "module": "wc_rocm",
            "available": _module_available("wc_rocm"),
            "modes": ["Monitor", "Window"],
            "resource_output": "ROCm tensor",
        },
    }
    try:
        from .backends.desktop_duplication_native import probe

        report["tools"]["DesktopDuplication"] = probe()
    except Exception as exc:
        report["tools"]["DesktopDuplication"] = {
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    try:
        from stereo_runtime.providers.directml import probe_directml_capabilities
        from stereo_runtime.providers.directml_resource import (
            probe_directml_resource_capabilities,
        )

        report["directml"] = probe_directml_capabilities()
        report["directml"].update(probe_directml_resource_capabilities())
    except Exception as exc:
        report["directml"]["reason"] = f"{type(exc).__name__}: {exc}"
    try:
        from viewer.vulkan_resources import VulkanD3D11ImportedImage

        report["d3d11_vulkan_import"] = {
            "implementation_present": VulkanD3D11ImportedImage is not None,
            "hardware_verified": False,
            "status": "待真实硬件验证",
            "requirements": [
                "non-zero shared handle",
                "matching Adapter LUID",
                "D3D11 external-memory handle type",
                "format and synchronization compatibility",
            ],
        }
    except Exception as exc:
        report["d3d11_vulkan_import"]["reason"] = f"{type(exc).__name__}: {exc}"
    report["fallback_order"] = [
        "厂商 WindowsCaptureCUDA/WindowsCaptureROCm（若模块与设备可用）",
        "WindowsCapture 原生资源（若暴露 D3D11 texture）",
        "WindowsCapture CPU兼容帧",
        "DesktopDuplication（仅显示器）/DXCamera",
    ]
    return report


def probe_capture_capabilities() -> dict[str, Any]:
    system = platform.system()
    if system == "Windows":
        return probe_windows_capture_capabilities()
    if system == "Linux":
        return probe_linux_capture_capabilities()
    return {
        "platform": system,
        "status": "本次目标不实现该平台原生捕获",
        "hardware_verified": False,
    }


__all__ = [
    "probe_capture_capabilities",
    "probe_linux_capture_capabilities",
    "probe_windows_capture_capabilities",
]
