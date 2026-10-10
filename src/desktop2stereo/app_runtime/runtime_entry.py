"""Assemble the Python capture and stereo pipeline for the Vulkan project."""

from __future__ import annotations

import json
import os
import platform
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from capture import capture_frame_to_rgb, prepare_rgb_for_stereo_runtime
from capture.adaptive_rate import AdaptiveCaptureRate, adaptive_capture_enabled_for_mode
from capture.session import CaptureSessionLoop
from stereo_runtime.pipeline import RuntimePipelineLoop
from stereo_runtime.render_size import RenderSizePolicy
from utils import (
    CAPTURE_MODE,
    CAPTURE_TOOL,
    CONVERGENCE,
    DEPTH_STRENGTH,
    DEVICE,
    DEVICE_INFO,
    DISPLAY_MODE,
    FPS,
    LOCAL_VSYNC,
    MONITOR_INDEX,
    OPENXR_SCREEN_DISTANCE,
    OPENXR_SCREEN_WIDTH,
    OS_NAME,
    OUTPUT_RESOLUTION,
    RENDER_SIZE_CONFIG,
    RUN_MODE,
    SHOW_FPS,
    STEREO_DISPLAY_INDEX,
    STEREO_DISPLAY_SELECTION,
    WINDOW_TITLE,
    _get_settings,
    shutdown_event,
)
from utils.display_info import resolve_windows_fullscreen_policy
from utils.run_mode import normalize_run_mode, target_fps_for_run_mode
from utils.xr_headset_presets import (
    DEFAULT_XR_HEADSET_MODEL,
    resolve_xr_headset_preset,
)
from xr_viewer.settings_menu import OPENXR_RENDER_SCALE_MAX, OPENXR_RENDER_SCALE_MIN
from streaming.stream_session import (
    CALIBRATABLE_STREAM_MODES,
    NetworkStreamSessionConfig,
    is_network_stream_mode,
    resolve_stream_audio_device,
    resolve_network_video_backend,
    supports_network_calibration,
)

from .runtime_callbacks import RuntimeCallbacks
from .runtime_context import (
    build_capture_callbacks,
    build_runtime_pipeline_context,
    create_runtime_context,
)
from .runtime_output import VulkanRuntimeOutputConsumer
from stereo_runtime.nvfruc import probe_nvfruc
from stereo_runtime.nvfruc_calibration import (
    NvFrucCalibrationCache,
    NvFrucCalibrationController,
    calibration_fingerprint,
    output_base_fps,
)
from stereo_runtime.nvfruc_stage import NvFrucStage


def _resolve_filament_environment_paths(
    settings: dict,
    src_root: Path,
) -> tuple[Path | None, Path | None, Path | None]:
    environment_name = str(settings.get("Environment Model", "")).strip()
    selected_name = (
        "Default"
        if not environment_name or environment_name.lower() == "none"
        else environment_name
    )
    environments_root = src_root / "xr_viewer" / "environments"

    def resolve(name: str) -> tuple[Path | None, Path, Path | None]:
        room_dir = environments_root / name
        profile_path = room_dir / "profile.json"
        if not profile_path.is_file():
            raise FileNotFoundError(
                f"OpenXR environment profile not found: {profile_path}"
            )
        try:
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"OpenXR environment profile is invalid: {profile_path}"
            ) from exc
        if not isinstance(profile, dict):
            raise ValueError(
                f"OpenXR environment profile root must be an object: {profile_path}"
            )

        glb_value = profile.get("glb", "environment.glb")
        background = profile.get("background")
        background = background if isinstance(background, dict) else {}
        image_value = (
            background.get("image")
            or background.get("path")
            or background.get("file")
            or profile.get("background_image")
        )
        panorama_path = None
        if image_value:
            candidate = room_dir / str(image_value)
            if not candidate.is_file():
                raise FileNotFoundError(f"OpenXR environment panorama not found: {candidate}")
            panorama_path = candidate
        if glb_value in (None, "", False):
            return None, profile_path, panorama_path
        glb_path = room_dir / str(glb_value)
        if not glb_path.is_file():
            raise FileNotFoundError(f"OpenXR environment GLB not found: {glb_path}")
        return glb_path, profile_path, panorama_path

    try:
        return resolve(selected_name)
    except (FileNotFoundError, ValueError) as exc:
        if selected_name.lower() == "default":
            raise
        print(
            f"[OpenXRViewer] Environment '{selected_name}' unavailable: {exc}; "
            "falling back to Default",
            flush=True,
        )
        return resolve("Default")


def _load_common_filament_defaults(src_root: Path) -> dict[str, object]:
    """Load shared environment defaults without making startup depend on them."""
    common_path = src_root / "xr_viewer" / "environments" / "common.json"
    try:
        common = json.loads(common_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(common, dict):
        return {}
    filament = common.get("filament", {})
    return filament if isinstance(filament, dict) else {}


def _resolve_openxr_render_scale(
    settings: dict,
    processing_size: int | tuple[int, int] | None = None,
) -> float:
    """Resolve the explicit OpenXR projection override.

    The GUI ``Render Scale`` belongs to capture/inference preprocessing.  It
    must not resize the OpenXR projection target, otherwise a 4K capture
    downscaled to 1K would take a different presentation path from native 1K.
    ``processing_size`` remains accepted for compatibility with callers and
    tests, but is intentionally not used for projection sizing.
    """
    # This is the OpenXR projection resolution multiplier, expressed as a
    # percentage in the GUI.  Keep the old name as a compatibility alias for
    # existing settings files and launch scripts.
    env_value = os.environ.get("D2S_OPENXR_RENDER_RESOLUTION")
    if not env_value:
        env_value = os.environ.get("D2S_OPENXR_RENDER_SCALE")
    if env_value:
        try:
            text = str(env_value).strip()
            value = float(text[:-1]) / 100.0 if text.endswith("%") else float(text)
            return max(OPENXR_RENDER_SCALE_MIN, min(OPENXR_RENDER_SCALE_MAX, value))
        except ValueError:
            pass
    if _openxr_render_scale_is_auto(settings):
        return max(
            OPENXR_RENDER_SCALE_MIN,
            min(
                OPENXR_RENDER_SCALE_MAX,
                float(resolve_xr_headset_preset(settings.get("XR Headset Model")).recommended_render_scale),
            ),
        )
    for key in ("XR Render", "OpenXR Render Resolution", "OpenXR Render Scale"):
        if key not in settings:
            continue
        try:
            raw = settings[key]
            text = str(raw).strip()
            value = float(text[:-1]) / 100.0 if text.endswith("%") else float(raw)
            return max(OPENXR_RENDER_SCALE_MIN, min(OPENXR_RENDER_SCALE_MAX, value))
        except (TypeError, ValueError):
            continue
    return 1.0


def _openxr_render_scale_is_auto(settings: dict) -> bool:
    mode = str(settings.get("XR Render Mode", "")).strip().lower()
    if mode in {"auto", "headset", "headset optimized", "headset optimized mode"}:
        return True
    for key in ("XR Render", "OpenXR Render Resolution", "OpenXR Render Scale"):
        value = settings.get(key)
        if isinstance(value, str) and value.strip().lower() in {
            "auto",
            "auto (headset)",
            "headset optimized",
        }:
            return True
    return False


def _openxr_projection_config(settings: dict) -> dict[str, object]:
    """Resolve OpenXR presentation settings independently of Filament."""
    return {
        "render_scale": _resolve_openxr_render_scale(settings),
        "render_scale_auto": _openxr_render_scale_is_auto(settings),
        "swapchain_color_mode": str(
            settings.get("OpenXR Color Mode", "sRGB")
        ).strip().lower(),
        "controller_model": str(settings.get("Controller Model", "PICO")),
        "headset_model": str(
            settings.get("XR Headset Model", DEFAULT_XR_HEADSET_MODEL)
        ),
        "monitor_index": max(1, int(settings.get("Monitor Index", 1) or 1)),
    }


def _openxr_glow_modes(settings: dict) -> dict[str, object]:
    """Load glow modes while handling YAML 1.1's ``off`` boolean alias."""
    modes = settings.get("OpenXR Glow Modes", {})
    if not isinstance(modes, dict):
        return {}
    return {
        str(environment): ("off" if value is False else value)
        for environment, value in modes.items()
    }


def _openxr_filament_config(
    settings: dict,
) -> dict[str, object]:
    """Resolve the selected packaged Filament scene for direct OpenXR runs."""
    src_root = Path(__file__).resolve().parents[1]
    platform_bridge = {
        "Windows": src_root / "xr_viewer" / "native" / "windows"
        / "filament_bridge.dll",
        "Linux": src_root / "xr_viewer" / "native" / "linux"
        / "libfilament_bridge.so",
        "Darwin": src_root / "xr_viewer" / "native" / "macos"
        / "libfilament_bridge.dylib",
    }.get(platform.system())
    glb_path, profile_path, panorama_path = _resolve_filament_environment_paths(
        settings,
        src_root,
    )
    common_filament = _load_common_filament_defaults(src_root)
    environment_name = str(settings.get("Environment Model", "Default")).strip()
    default_environment = not environment_name or environment_name.lower() in {"default", "none"}

    bridge_path = os.environ.get("D2S_FILAMENT_BRIDGE") or (
        str(platform_bridge) if platform_bridge and platform_bridge.is_file() else None
    )
    configured_glb = os.environ.get("D2S_FILAMENT_GLB")
    configured_profile = os.environ.get("D2S_FILAMENT_PROFILE")
    saved_transparency = settings.get("OpenXR Glow Transparency")
    if not isinstance(saved_transparency, dict):
        # Migrate the previous opacity-oriented setting without changing its
        # visual result: transparency is the inverse of opacity.
        saved_opacity = settings.get("OpenXR Glow Opacity")
        if isinstance(saved_opacity, dict):
            saved_transparency = {}
            for environment, opacity in saved_opacity.items():
                try:
                    saved_transparency[str(environment)] = 1.0 - min(
                        1.0, max(0.0, float(opacity))
                    )
                except (TypeError, ValueError):
                    continue
        else:
            saved_transparency = {}
    return {
        "filament_bridge_path": bridge_path,
        "filament_glb_path": configured_glb or (
            str(glb_path) if glb_path is not None else None
        ),
        "filament_profile_path": configured_profile
        or (str(profile_path) if profile_path is not None else None),
        "filament_panorama_path": os.environ.get("D2S_FILAMENT_PANORAMA")
        or (str(panorama_path) if panorama_path is not None else None),
        "filament_scene_exposure_ev": float(
            settings.get(
                "Filament Scene Exposure",
                common_filament.get("scene_exposure_ev", 2.0),
            )
        ),
        "filament_skybox_brightness": float(
            settings.get(
                "Filament Skybox Brightness",
                common_filament.get("skybox_brightness", 1.0),
            )
        ),
        "filament_ambient_light_color": tuple(
            common_filament.get("ambient_light_color", (0.14, 0.13, 0.15))
        ),
        "filament_ambient_light_intensity_lux": float(
            common_filament.get("ambient_light_intensity_lux", 30000.0)
        ),
        "filament_controller_ambient_light_intensity_lux": float(
            common_filament.get("controller_ambient_light_intensity_lux", 8000.0)
        ),
        "filament_controller_hdr_ambient_light_intensity_lux": float(
            common_filament.get("controller_hdr_ambient_light_intensity_lux", 8000.0)
        ),
        "filament_controller_light_intensity_candela": float(
            common_filament.get("controller_light_intensity_candela", 2000.0)
        ),
        "filament_fill_light_color": tuple(
            common_filament.get("controller_head_light_color", (0.55, 0.55, 0.58))
        ),
        "filament_controller_head_light_weight": float(
            common_filament.get("controller_head_light_weight", 0.85)
        ),
        "filament_controller_top_light_weight": float(
            common_filament.get("controller_top_light_weight", 0.6)
        ),
        "filament_controller_top_light_color": tuple(
            common_filament.get("controller_top_light_color", (0.95, 0.97, 1.0))
        ),
        "filament_controller_head_light_offset": tuple(
            common_filament.get("controller_head_light_offset", (0.0, 0.05, 0.0))
        ),
        "filament_controller_top_light_offset": tuple(
            common_filament.get("controller_top_light_offset", (0.0, 0.45, -0.18))
        ),
        "filament_controller_head_light_falloff": float(
            common_filament.get("controller_head_light_falloff", 2.0)
        ),
        "filament_controller_top_light_falloff": float(
            common_filament.get("controller_top_light_falloff", 2.0)
        ),
        "filament_controller_head_light_cast_shadows": bool(
            common_filament.get("controller_head_light_cast_shadows", False)
        ),
        "filament_controller_top_light_cast_shadows": bool(
            common_filament.get("controller_top_light_cast_shadows", False)
        ),
        "filament_controller_screen_light_enabled": bool(
            common_filament.get("controller_screen_light_enabled", True)
        ),
        "filament_controller_screen_light_intensity_lux": float(
            common_filament.get("controller_screen_light_intensity_lux", 500.0)
        ),
        "filament_controller_screen_light_saturation": float(
            common_filament.get("controller_screen_light_saturation", 0.65)
        ),
        "filament_controller_screen_light_max_luminance": float(
            common_filament.get("controller_screen_light_max_luminance", 0.40)
        ),
        "filament_controller_screen_light_smoothing_seconds": float(
            common_filament.get("controller_screen_light_smoothing_seconds", 0.18)
        ),
        "filament_controller_screen_light_sample_hz": float(
            common_filament.get("controller_screen_light_sample_hz", 12.0)
        ),
        "filament_controller_screen_light_cast_shadows": bool(
            common_filament.get("controller_screen_light_cast_shadows", False)
        ),
        "filament_environment_screen_light_enabled": bool(
            False
            if default_environment
            else common_filament.get("environment_screen_light_enabled", True)
        ),
        "filament_environment_screen_light_intensity_candela": float(
            common_filament.get("environment_screen_light_intensity_candela", 120.0)
        ),
        "filament_environment_screen_light_saturation": float(
            common_filament.get("environment_screen_light_saturation", 0.70)
        ),
        "filament_environment_screen_light_max_luminance": float(
            common_filament.get("environment_screen_light_max_luminance", 0.40)
        ),
        "filament_environment_screen_light_smoothing_seconds": float(
            common_filament.get("environment_screen_light_smoothing_seconds", 0.18)
        ),
        "filament_environment_screen_light_sample_hz": float(
            common_filament.get("environment_screen_light_sample_hz", 12.0)
        ),
        "filament_environment_screen_light_falloff": float(
            common_filament.get("environment_screen_light_falloff", 4.0)
        ),
        "filament_environment_screen_light_offset": float(
            common_filament.get("environment_screen_light_offset", 0.08)
        ),
        "filament_environment_screen_light_cast_shadows": bool(
            common_filament.get("environment_screen_light_cast_shadows", False)
        ),
        "filament_glow_sample_hz": float(
            common_filament.get("glow_sample_hz", 30.0)
        ),
        "filament_glow_smoothing_seconds": float(
            common_filament.get("glow_smoothing_seconds", 0.10)
        ),
        # These values are resolved by the legacy viewer-settings path and
        # exported through utils; keep the Vulkan entrypoint as a consumer.
        "filament_screen_width": float(OPENXR_SCREEN_WIDTH),
        "filament_screen_distance": float(OPENXR_SCREEN_DISTANCE),
        "filament_screen_states": (
            dict(settings.get("OpenXR Screen States", {}))
            if isinstance(settings.get("OpenXR Screen States", {}), dict)
            else {}
        ),
        "filament_glow_modes": _openxr_glow_modes(settings),
        "filament_glow_transparencies": dict(saved_transparency),
    }


def _exclude_local_output_from_capture(settings: dict, *, os_name: str) -> bool:
    return (
        str(os_name).strip().lower() == "windows"
        and settings.get("Run Mode", "Local Viewer") == "3D Monitor"
    )


def _queue_clear(queue) -> None:
    while True:
        try:
            queue.get_nowait()
        except Exception:
            return


def _consume_stop_request(
    stop_request_path: str | None,
    *,
    expected_pid: int,
) -> bool:
    """Consume a stop request only when it targets this runtime process."""
    if not stop_request_path:
        return False
    request_path = Path(stop_request_path)
    try:
        requested_pid = int(request_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False
    if requested_pid != expected_pid:
        return False
    try:
        request_path.unlink()
    except OSError:
        pass
    return True


def _openxr_runtime_exit_code(presenter: Any | None) -> int:
    """Prefer an explicit user stop over a racing OpenXR device-loss result."""
    return 77 if (
        presenter is not None
        and getattr(presenter, "fatal_device_loss", False)
        and not getattr(presenter, "_runtime_stop_requested", False)
    ) else 0


def _watch_stop_request(
    stop_request_path: str,
    *,
    expected_pid: int,
    stop_event: threading.Event,
    poll_interval: float = 0.05,
) -> None:
    """Bridge the GUI stop file into this child process's shutdown event."""
    while not stop_event.wait(poll_interval):
        if _consume_stop_request(
            stop_request_path,
            expected_pid=expected_pid,
        ):
            print("[Main] GUI stop request received", flush=True)
            stop_event.set()
            return


def _wait_for_runtime_ready(
    ready_event: threading.Event,
    pipeline_thread: threading.Thread,
) -> bool:
    print(
        "[Main] Waiting for inference load, first frame, and stereo warmup "
        "before OpenXR initialization...",
        flush=True,
    )
    while not shutdown_event.is_set():
        if ready_event.wait(0.05):
            print(
                "[Main] Inference pipeline ready; starting OpenXR "
                "Vulkan/Filament initialization",
                flush=True,
            )
            return True
        if not pipeline_thread.is_alive():
            raise RuntimeError(
                "Stereo pipeline stopped before inference startup completed"
            )
    return False


def _resolve_local_viewer_render_size_config(settings, run_mode, device):
    """Use the v2.5 local-viewer source size while keeping 4K I/O.

    The old local path captured the 4K monitor and presented to the 4K
    display, but ran the Half-SBS warp at one eye (1920x1080) and upscaled
    that packed result in the display shader.  The current scaled/native path
    accidentally moved depth and warp to 3840x2160.  Restore that contract
    only for the non-Darwin GPU local viewer; the Vulkan swapchain and capture
    target remain native 4K.  D2S_LOCAL_VIEWER_NATIVE_4K=1 is an explicit
    escape hatch for full-resolution processing.
    """
    if platform.system() == "Darwin" or run_mode not in {"Local Viewer", "Viewer"}:
        return RENDER_SIZE_CONFIG
    if str(getattr(device, "type", device)).strip().lower() != "cuda":
        return RENDER_SIZE_CONFIG
    if os.environ.get("D2S_LOCAL_VIEWER_NATIVE_4K", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }:
        return RENDER_SIZE_CONFIG
    if str(settings.get("Processing Resolution", "Auto")).strip().lower() != "auto":
        return RENDER_SIZE_CONFIG
    if str(settings.get("Display Mode", "")).strip().lower().replace("_", "-") != "half-sbs":
        return RENDER_SIZE_CONFIG
    if RENDER_SIZE_CONFIG.policy is not RenderSizePolicy.SCALED:
        return RENDER_SIZE_CONFIG
    if RENDER_SIZE_CONFIG.scale_factor != "4K / 100%":
        return RENDER_SIZE_CONFIG
    if not isinstance(OUTPUT_RESOLUTION, (tuple, list)) or len(OUTPUT_RESOLUTION) != 2:
        return RENDER_SIZE_CONFIG
    output_width, output_height = (int(OUTPUT_RESOLUTION[0]), int(OUTPUT_RESOLUTION[1]))
    if output_width < 2 or output_height < 2:
        return RENDER_SIZE_CONFIG
    return replace(
        RENDER_SIZE_CONFIG,
        policy=RenderSizePolicy.FIXED,
        fixed_width=output_width // 2,
        fixed_height=output_height // 2,
    )


def run_processing_runtime(*, max_seconds: float | None = None) -> int:
    """Run capture, inference, and pipeline threads until shutdown is requested."""

    if platform.system() == "Windows":
        # DPI awareness is first-wins per process: later calls silently
        # return E_ACCESSDENIED, so the window can be created unaware while
        # whatever raced ahead is already set. An unaware/system-aware
        # process on a scaled (e.g. 150%) monitor gets a DWM-virtualized
        # framebuffer (1920x1200 -> 1280x800), putting the fullscreen SBS
        # image in the monitor's top-left instead of covering it. main.py
        # already requests per-monitor v2 before any import; keep the
        # first-wins-safe guard here for direct/non-main callers.
        from windows_dpi import set_per_monitor_dpi_v2

        set_per_monitor_dpi_v2()

    shutdown_event.clear()
    stop_request_thread = None
    stop_request_path = os.environ.get("D2S_STOP_REQUEST_FILE", "").strip()
    if stop_request_path:
        stop_request_thread = threading.Thread(
            target=_watch_stop_request,
            kwargs={
                "stop_request_path": stop_request_path,
                "expected_pid": os.getpid(),
                "stop_event": shutdown_event,
            },
            name="GuiStopRequest",
            daemon=True,
        )
        stop_request_thread.start()
    settings = _get_settings()
    configured_run_mode = normalize_run_mode(
        settings.get("Run Mode", "Local Viewer")
    )
    if configured_run_mode in {"Local Viewer", "Viewer"} and platform.system() == "Darwin":
        # Local Viewer presents to a physical display; skip the headset-tier
        # upscale chain (see output_sampling_plan_for_config).
        os.environ.setdefault("D2S_CAP_OUTPUT_UPSCALE", "1")
        # Ship packed uint8 SBS so the viewer skips float32 transport
        # and re-quantization (pack once in the runtime, upload once in the
        # viewer). setdefault keeps an explicit user override authoritative.
        os.environ.setdefault("D2S_RUNTIME_OUTPUT_UINT8", "1")
        # Deferred Metal shader warp is armed for the Metal viewer (the
        # macOS default) and dropped if it falls back; see the Darwin viewer
        # selection below.
        # Vulkan fused warp-pack ships INPUT-RESOLUTION SBS (NVIDIA-path
        # contract: 1080p in -> half_sbs 1920x1080). Quarter-res synth-scale
        # preprocess would read as blur once packed at input res, so the
        # fused path runs the pipeline at full render size. Explicit user
        # overrides of either env stay authoritative.
        _mac_viewer = os.environ.get("D2S_MAC_VIEWER") or "vulkan"
        _vk_fused_on = os.environ.get("D2S_VK_FUSED_WARP", "1") not in {
            "0", "false", "off",
        }
        if _mac_viewer == "vulkan" and _vk_fused_on:
            os.environ.setdefault("D2S_HALF_RES_SYNTH", "0")
            os.environ.setdefault("D2S_PREPROCESS_AT_SYNTH_SCALE", "0")
        else:
            # Halve synthesis resolution and upscale: warp cost scales with
            # pixel count, so this cuts the dominant stage ~4x for the Metal
            # viewer (its warp path skips synthesis; sampling upscales).
            os.environ.setdefault("D2S_HALF_RES_SYNTH", "1")
            # Preprocess directly at synthesis scale: skips a display-size
            # upscale plus redundant downscale per frame.
            os.environ.setdefault("D2S_PREPROCESS_AT_SYNTH_SCALE", "1")
        # Depth input scale is OPT-IN (e.g. D2S_DEPTH_INPUT_SCALE=0.75 buys
        # ~2x faster M1 MPS depth at corr 0.95); no default anymore. The
        # softened 0.75x depth made object-edge parallax flicker vs v2.5.0,
        # so the Local Viewer runs full export resolution unless this is
        # explicitly set. "0"/"false"/"off" disable it explicitly as well.
        # Default macOS viewer is now Vulkan (fused Metal warp-pack
        # kernel beat the CAMetalLayer path in round-24 A/Bs:
        # 46.7 vs 45.5-45.9 fps). D2S_MAC_VIEWER=metal restores it.
        os.environ.setdefault("D2S_MAC_VIEWER", "vulkan")
        # Pack the presentation frame on the runtime thread (queue drained)
        # so viewers memcpy instead of syncing MPS mid-present.
        os.environ.setdefault("D2S_VIEWER_HOST_FRAME", "1")
        # Wrap every SCK frame as an owned CVPixelBuffer+CVMetalTexture so
        # the warp viewer can sample the capture directly (zero-copy).
        os.environ.setdefault("D2S_SCK_ZEROCOPY_TEX", "1")
        # Native Core ML consumes the IOSurface directly. The capture callback
        # therefore skips its CPU base-address read; it re-materializes only
        # if the bridge capability preflight fails.
        if bool(settings.get("CoreML", False)):
            # Keep Core ML depth inference off the Apple GPU by default. A
            # warmed 60 Hz Local Viewer comparison found CPU+ANE slightly
            # improved throughput and reduced pack/latency outliers while
            # leaving Metal available for stereo synthesis and presentation.
            # setdefault preserves an explicit user-selected compute unit.
            os.environ.setdefault("D2S_COREML_COMPUTE_UNITS", "ane")
            os.environ.setdefault("D2S_SCK_NATIVE_ONLY", "1")
    elif configured_run_mode in {"Local Viewer", "Viewer"}:
        # The Vulkan local viewer consumes GPU RGBA8 directly.  Pack on the
        # producer GPU so the external buffer transfers 1 byte/channel rather
        # than a float32 frame.  Keep the Darwin branch above unchanged: its
        # viewer has platform-specific output handling and owns this setting.
        os.environ.setdefault("D2S_RUNTIME_OUTPUT_UINT8", "1")
    direct_stream_mode = is_network_stream_mode(configured_run_mode) or configured_run_mode == "MJPEG Streamer"
    if direct_stream_mode:
        os.environ["D2S_RUNTIME_OUTPUT_UINT8"] = "1"
        if platform.system() == "Darwin":
            # The macOS network sink currently consumes host RGB frames. Keep
            # stereo generation on the canonical path so its geometry and
            # postprocessing match CUDA/ROCm instead of paying the Vulkan
            # layered pass's host readback and using different edge rules.
            os.environ.setdefault("D2S_MAC_STREAM_CANONICAL_SYNTHESIS", "1")
            # Prefer the existing IOSurface -> CoreML -> Metal pack bridge for
            # network streams. It is attempted only after native capability
            # preflight; the pipeline retains the Python CoreML fallback.
            os.environ.setdefault("D2S_MAC_STREAM_NATIVE_IO", "1")
            # Replace MPS grid_sample's two full-frame synchronization points
            # with the canonical two-layer Metal kernel for the eligible
            # realtime profile. Unsupported profiles still use torch.
            os.environ.setdefault("D2S_MAC_STREAM_MPS_FUSED", "1")
    configured_target_fps = target_fps_for_run_mode(settings)
    nvfruc_requested = bool(
        settings.get(
            "NVIDIA Frame Generation",
            settings.get("Lossless Scaling Support", False),
        )
    )
    base_runtime_fps = output_base_fps(FPS, enabled=nvfruc_requested)
    if nvfruc_requested:
        print(
            "[NvFRUC] FPS semantics: "
            f"output_target={int(FPS)} base_runtime={base_runtime_fps}",
            flush=True,
        )
    adaptive_capture_rate = AdaptiveCaptureRate(
        base_runtime_fps,
        enabled=adaptive_capture_enabled_for_mode(
            configured_run_mode, configured_target_fps
        ),
    )
    effective_render_size_config = _resolve_local_viewer_render_size_config(
        settings,
        configured_run_mode,
        DEVICE,
    )
    if effective_render_size_config is not RENDER_SIZE_CONFIG:
        print(
            "[Main] Local Viewer v2.5 source path: "
            f"capture={OUTPUT_RESOLUTION[0]}x{OUTPUT_RESOLUTION[1]} "
            f"processing={effective_render_size_config.fixed_width}x"
            f"{effective_render_size_config.fixed_height} "
            "presentation=native-4K; set D2S_LOCAL_VIEWER_NATIVE_4K=1 "
            "to force native processing",
            flush=True,
        )
    if is_network_stream_mode(configured_run_mode):
        probe_capture_fps = adaptive_capture_rate.begin_stream_probe(int(base_runtime_fps))
        print(
            "[DirectSbsStream] Stream-rate probe capture headroom: "
            f"requested={int(base_runtime_fps)} capture={probe_capture_fps} FPS",
            flush=True,
        )
    context = create_runtime_context(
        file_path=str(Path(__file__).resolve().parents[1] / "main.py"),
        settings=settings,
        cache_path=str(Path(__file__).resolve().parents[1] / "models"),
        device=DEVICE,
        device_info=DEVICE_INFO,
        output_resolution=OUTPUT_RESOLUTION,
        render_size_config=effective_render_size_config,
        fps=base_runtime_fps,
        window_title=WINDOW_TITLE,
        capture_mode=CAPTURE_MODE,
        monitor_index=MONITOR_INDEX,
        capture_tool=CAPTURE_TOOL,
        os_name=OS_NAME,
        run_mode=RUN_MODE,
        depth_strength=DEPTH_STRENGTH,
        convergence=CONVERGENCE,
        capture_fps_provider=adaptive_capture_rate.current_fps,
    )
    nvfruc_calibration = None
    if context.nvfruc_frame_generation:
        nvfruc_probe = probe_nvfruc()
        if not nvfruc_probe.available:
            print(
                "[NvFRUC] Frame generation requested but unavailable: "
                f"{nvfruc_probe.reason}. Continuing without NvFRUC.",
                flush=True,
            )
            # Disable NvFRUC for this session and continue
            context.nvfruc_frame_generation = False
            base_runtime_fps = output_base_fps(FPS, enabled=False)
            adaptive_capture_rate = AdaptiveCaptureRate(
                base_runtime_fps,
                enabled=adaptive_capture_enabled_for_mode(
                    configured_run_mode, configured_target_fps
                ),
            )
            if is_network_stream_mode(configured_run_mode):
                probe_capture_fps = adaptive_capture_rate.begin_stream_probe(int(base_runtime_fps))
                print(
                    "[DirectSbsStream] Stream-rate probe capture headroom: "
                    f"requested={int(base_runtime_fps)} capture={probe_capture_fps} FPS",
                    flush=True,
                )
        else:
            calibration_values = {
                "device": str(DEVICE_INFO),
                "depth_model": str(settings.get("Depth Model", "")),
                "inference_backend": str(settings.get("Inference Backend", "")),
                "precision": str(settings.get("Precision", "")),
                "input_resolution": str(OUTPUT_RESOLUTION),
                "run_mode": configured_run_mode,
                "output_format": str(settings.get("Output Format", "")),
                "encoder": str(settings.get("Video Encoder Backend", "auto")),
            }
            calibration_fingerprint_value = calibration_fingerprint(calibration_values)
            calibration_cache = NvFrucCalibrationCache(
                Path(context.base_dir) / "models" / "nvfruc"
            )

            def apply_nvfruc_limit(output_limit: int) -> None:
                base_limit = output_base_fps(output_limit, enabled=True)
                adaptive_capture_rate.set_calibration_limit(base_limit)
                print(
                    "[NvFRUC] calibrated output limit: "
                    f"output={int(output_limit)} base_runtime={int(base_limit)}",
                    flush=True,
                )

            nvfruc_calibration = NvFrucCalibrationController(
                output_target_fps=int(FPS),
                fingerprint=calibration_fingerprint_value,
                cache=calibration_cache,
                on_limit=apply_nvfruc_limit,
            )

    callbacks = RuntimeCallbacks(
        context,
        show_fps=bool(SHOW_FPS),
        display_fit_mode=settings.get("Display Fit Mode", "contain"),
    )

    low_sbs_report_count = 0

    def observe_sbs_fps(sbs_fps, frame_count=None):
        nonlocal low_sbs_report_count
        capture_fps = callbacks.capture_fps()
        capture_target = adaptive_capture_rate.observe_sbs_fps(
            sbs_fps,
            capture_fps=capture_fps,
            frame_count=frame_count,
        )
        callbacks.breakdown_set_latest(
            "adaptive_capture_target_fps", capture_target
        )
        if nvfruc_calibration is not None:
            previous_target = nvfruc_calibration.current_target_fps
            monitored_target = nvfruc_calibration.monitor_submission(sbs_fps)
            if monitored_target < previous_target:
                print(
                    "[NvFRUC] sustained output underrun; "
                    f"downshifted output target {previous_target} -> {monitored_target} FPS",
                    flush=True,
                )
        if float(sbs_fps) < 5.0:
            low_sbs_report_count += 1
            if low_sbs_report_count <= 5 or low_sbs_report_count % 6 == 0:
                latencies = context.thread_latencies
                print(
                    "[VulkanLocalViewer] Low SBS flow: "
                    f"sbs={float(sbs_fps):.1f} capture={capture_fps:.1f} "
                    f"runtime={callbacks.runtime_fps():.1f} target={capture_target} "
                    f"raw_q={context.raw_q.qsize()} runtime_q={context.runtime_q.qsize()} "
                    f"capture_latency={float(latencies.get('capture', 0.0)) * 1000.0:.1f}ms "
                    f"runtime_latency={float(latencies.get('runtime', 0.0)) * 1000.0:.1f}ms",
                    flush=True,
                )
        return capture_target

    def report_display_refresh_warning(refresh_hz: int, sbs_fps: float) -> None:
        if str(settings.get("Language", "EN")).strip().upper() == "CN":
            message = (
                f"SBS 输出显示器当前仅 {refresh_hz} Hz，低于实测 {sbs_fps:.1f} FPS "
                "或建议最低 60 Hz；请在 Windows 显示设置或显卡控制面板中提高刷新率。"
            )
        else:
            message = (
                f"The SBS output display is running at {refresh_hz} Hz, below the "
                f"measured {sbs_fps:.1f} FPS or the recommended 60 Hz minimum; "
                "increase its refresh rate in Windows or the GPU control panel."
            )
        print(f"[VulkanLocalViewer] WARNING: {message}", flush=True)
        print(f"[D2S_STATUS] {message}", flush=True)
        print(
            "[D2S_DISPLAY_REFRESH_WARNING] "
            + json.dumps(
                {
                    "kind": "output",
                    "refresh_hz": int(refresh_hz),
                    "sbs_fps": round(float(sbs_fps), 1),
                }
            ),
            flush=True,
        )

    def report_capture_refresh_warning(
        refresh_hz: int, capture_target: int
    ) -> None:
        if str(settings.get("Language", "EN")).strip().upper() == "CN":
            message = (
                f"输入显示器当前仅 {refresh_hz} Hz，低于动态捕获目标 "
                f"{capture_target} FPS；请提高输入显示器刷新率，或手动降低捕获帧率。"
            )
        else:
            message = (
                f"The input display is running at {refresh_hz} Hz, below the dynamic "
                f"capture target of {capture_target} FPS; increase the input display "
                "refresh rate or lower the capture target manually."
            )
        print(f"[VulkanLocalViewer] WARNING: {message}", flush=True)
        print(f"[D2S_STATUS] {message}", flush=True)
        print(
            "[D2S_DISPLAY_REFRESH_WARNING] "
            + json.dumps(
                {
                    "kind": "input_capture_target",
                    "refresh_hz": int(refresh_hz),
                    "capture_target": int(capture_target),
                }
            ),
            flush=True,
        )

    callbacks.breakdown_set_latest(
        "adaptive_capture_target_fps", adaptive_capture_rate.current_fps()
    )
    runtime_ready_event = threading.Event()

    if str(RUN_MODE).strip().lower() == "openxr":
        # Keep source inference alive during the headset wake-up grace period.
        # The presenter enters hard idle after the configured 60-second timeout.
        context.openxr_state.bootstrap_done.set()
        context.openxr_state.source_active.set()

    capture_callbacks = build_capture_callbacks(
        raw_q=context.raw_q,
        shutdown_event=shutdown_event,
        queue_clear=callbacks.queue_clear_nonblocking,
        inc_source_stat=callbacks.source_stat_inc,
        inc_breakdown=callbacks.breakdown_inc,
        put_raw_latest=callbacks.put_raw_latest,
        is_paused=callbacks.openxr_source_paused,
        is_hard_idle=callbacks.openxr_hard_idle_active,
        on_session_update=callbacks.capture_session_update,
        on_tick=callbacks.log_source_health,
    )

    presenter = None
    pipeline_context = build_runtime_pipeline_context(
        shutdown_event=shutdown_event,
        app_context=context,
        run_mode=RUN_MODE,
        device=DEVICE,
        capture_frame_to_rgb=capture_frame_to_rgb,
        prepare_rgb_for_stereo_runtime=prepare_rgb_for_stereo_runtime,
        current_openxr_render_config=callbacks.current_openxr_render_config,
        is_hard_idle=callbacks.openxr_hard_idle_active,
        is_source_paused=callbacks.openxr_source_paused,
        log_source_health=callbacks.log_source_health,
        source_stat_inc=callbacks.source_stat_inc,
        breakdown_inc=callbacks.breakdown_inc,
        breakdown_add_time=callbacks.breakdown_add_time,
        breakdown_add_runtime_timing=callbacks.breakdown_add_runtime_timing,
        set_preprocess_backend=callbacks.set_runtime_preprocess_backend,
        queue_clear=callbacks.queue_clear_nonblocking,
        queue_drain_latest=callbacks.queue_drain_latest,
        queue_put_latest=callbacks.queue_put_latest,
        log_stereo_runtime_mode_once=callbacks.log_stereo_runtime_mode_once,
        apply_stereo_hot_reload_if_needed=callbacks.apply_stereo_hot_reload_if_needed,
        warmup_stereo_once_for_frame=callbacks.warmup_stereo_once_for_frame,
        log_fast_plus_fused_runtime_state=callbacks.log_fast_plus_fused_runtime_state,
        runtime_ready_event=runtime_ready_event,
        openxr_presenter_pressure=lambda: bool(
            presenter is not None and presenter.inference_backpressure_active()
        ),
    )

    capture_thread = threading.Thread(
        target=CaptureSessionLoop(context.capture_config, capture_callbacks).run,
        args=(shutdown_event,),
        name="VulkanCapture",
        daemon=True,
    )
    pipeline = RuntimePipelineLoop(pipeline_context)
    print("[Main] Loading inference runtime before capture and OpenXR...", flush=True)
    pipeline.prepare()
    print("[Main] Inference runtime loaded", flush=True)
    pipeline_thread = threading.Thread(
        target=pipeline.run,
        name="VulkanStereoPipeline",
        daemon=True,
    )
    presenter_thread = None
    output_consumer = None
    output_thread = None
    local_viewer_thread = None
    network_output = None
    main_thread_job = None
    nvfruc_stage = None
    nvfruc_thread = None
    presentation_q = context.runtime_q
    if context.nvfruc_frame_generation:
        nvfruc_stage = NvFrucStage(
            input_q=context.runtime_q,
            output_q=context.presentation_q,
            shutdown_event=shutdown_event,
            output_format_provider=lambda: getattr(
                context.runtime_config, "output_format", "half_sbs"
            ),
            calibration_controller=nvfruc_calibration,
            on_status=lambda message: print(f"[NvFRUC] {message}", flush=True),
        )
        presentation_q = context.presentation_q
        nvfruc_thread = threading.Thread(
            target=nvfruc_stage.run,
            name="NvFRUCFrameGeneration",
            daemon=True,
        )
        nvfruc_thread.start()
    capture_thread.start()
    pipeline_thread.start()
    try:
        if str(RUN_MODE).strip().lower() == "openxr":
            if not _wait_for_runtime_ready(runtime_ready_event, pipeline_thread):
                return 0
            from xr_viewer.core_openxr_vulkan import (
                OpenXrVulkanConfig,
                OpenXrVulkanPresenter,
            )

            presenter_config = _openxr_projection_config(settings)
            presenter_config.update(_openxr_filament_config(settings))
            presenter = OpenXrVulkanPresenter(
                OpenXrVulkanConfig(**presenter_config),
                on_headset_state=callbacks.on_openxr_headset_state,
                on_controller_shortcut=callbacks.on_openxr_controller_shortcut,
                on_breakdown_inc=callbacks.breakdown_inc,
                on_breakdown_add_time=callbacks.breakdown_add_time,
                on_breakdown_set_latest=callbacks.breakdown_set_latest,
                on_runtime_fps=callbacks.runtime_fps,
                on_capture_fps=callbacks.capture_fps,
                on_sbs_fps=observe_sbs_fps if adaptive_capture_rate.enabled else None,
            )
            presenter_thread = threading.Thread(
                target=presenter.run_until,
                args=(shutdown_event,),
                name="VulkanOpenXRPresenter",
                daemon=True,
            )
            presenter_thread.start()
            output_consumer = VulkanRuntimeOutputConsumer(
                runtime_q=presentation_q,
                shutdown_event=shutdown_event,
                source_stat_inc=callbacks.source_stat_inc,
                sink=presenter,
            )
            output_thread = threading.Thread(
                target=output_consumer.run,
                name="VulkanOutputConsumer",
                daemon=True,
            )
            output_thread.start()
        elif configured_run_mode in {
            "RTMP Streamer",
            "MJPEG Streamer",
        }:
            from streaming.direct_sbs import (
                DirectSbsOutputConsumer,
                AmdAmfDirectSbsOutput,
                FfmpegDirectSbsOutput,
                IntelD3D11DirectSbsOutput,
                IntelQsvDirectSbsOutput,
                MjpegDirectSbsOutput,
                PyNvDirectSbsOutput,
                VulkanDirectSbsOutput,
            )
            from streaming.stream_calibration import build_calibration_fingerprint

            # Stream aspect: follow local viewer's keep-ratio logic to avoid distortion when not 16:9
            stream_fit_mode = str(settings.get("Stream Display Fit Mode", settings.get("Display Fit Mode", "contain"))).strip()
            # Shared input_size (tex_w,tex_h) for dynamic eye ratio like VulkanLocalViewer
            cap_mode_stream = str(settings.get("Capture Mode", "")).strip()
            stream_input_size: tuple[int, int] | None = None
            try:
                if cap_mode_stream.casefold() == "window":
                    title_stream = str(settings.get("Window Title", "")).strip()
                    if title_stream and OS_NAME == "Windows":
                        try:
                            import win32gui
                            hwnd_stream = win32gui.FindWindow(None, title_stream)
                            if hwnd_stream:
                                _, _, w_stream, h_stream = win32gui.GetClientRect(hwnd_stream)
                                if w_stream > 0 and h_stream > 0:
                                    stream_input_size = (int(w_stream), int(h_stream))
                        except Exception:
                            pass
                    if stream_input_size is None:
                        from utils.display import get_monitor_size
                        stream_input_size = get_monitor_size(int(MONITOR_INDEX))
                else:
                    from utils.display import get_monitor_size
                    stream_input_size = get_monitor_size(int(MONITOR_INDEX))
            except Exception:
                stream_input_size = None

            if configured_run_mode in CALIBRATABLE_STREAM_MODES:
                selected_audio = resolve_stream_audio_device(
                    settings,
                    os_name=OS_NAME,
                )
                stream_config = replace(
                    NetworkStreamSessionConfig.from_settings(settings, fps=int(FPS)),
                    stereo_mix_device=selected_audio,
                )
                output_kwargs = dict(
                    base_dir=context.base_dir,
                    protocol=stream_config.protocol,
                    port=stream_config.port,
                    stream_key=stream_config.stream_key,
                    fps=stream_config.fps,
                    crf=stream_config.crf,
                    stereo_mix_device=stream_config.stereo_mix_device,
                    audio_delay=stream_config.audio_delay,
                    os_name=OS_NAME,
                    prefer_nvenc=(
                        OS_NAME in {"Windows", "Linux"}
                        and "NVIDIA" in str(DEVICE_INFO).upper()
                    ),
                    display_mode=stream_config.display_mode,
                    fit_mode=stream_fit_mode,
                    input_size=stream_input_size,
                    # Auto calibration and manual mode share the same encoder
                    # path.  The calibration controller temporarily replaces
                    # these values during its probe; manual mode must retain
                    # the values selected in the GUI instead of clearing them.
                    target_bitrate_mbps=stream_config.target_bitrate_mbps,
                    peak_bitrate_mbps=stream_config.peak_bitrate_mbps,
                    auto_calibration=(
                        supports_network_calibration(
                            configured_run_mode,
                            settings.get("Stream Protocol", "WebRTC"),
                        )
                        and os.environ.get("D2S_STREAM_CALIBRATE", "0") == "1"
                    ),
                    calibration_port=int(
                        settings.get(
                            "Stream Calibration Port",
                            min(65535, stream_config.port + 1),
                        )
                    ),
                    on_calibration_fps=adaptive_capture_rate.set_calibration_limit,
                    calibration_fingerprint=build_calibration_fingerprint(settings),
                    on_stream_fps_selected=adaptive_capture_rate.finish_stream_probe,
                )
                backend_decision = resolve_network_video_backend(
                    configured_run_mode,
                    settings.get("Video Encoder Backend", "auto"),
                    device_info=DEVICE_INFO,
                )
                video_backend = backend_decision.backend
                has_nvidia_gpu = "NVIDIA" in str(DEVICE_INFO).upper()
                has_amd_gpu = any(
                    token in str(DEVICE_INFO).upper()
                    for token in ("AMD", "RADEON")
                )
                has_intel_gpu = "INTEL" in str(DEVICE_INFO).upper()
                print(
                    f"[DirectSbsStream] mode={configured_run_mode} "
                    f"encoder={video_backend} ({backend_decision.reason})",
                    flush=True,
                )
                network_output = FfmpegDirectSbsOutput(**output_kwargs)
                if video_backend == "auto":
                    network_output.close()
                    if has_intel_gpu:
                        # Intel owns its D3D11/oneVPL vendor path first; its
                        # native sink already reuses the Vulkan packed-SBS
                        # bridge when the shared-surface contract is available.
                        network_output = IntelD3D11DirectSbsOutput(**output_kwargs)
                    else:
                        # NVIDIA/AMD Auto enters the same lazy output at the
                        # vendor-native stage, then proceeds to Vulkan and the
                        # portable OpenGL/FFmpeg fallbacks.
                        network_output = VulkanDirectSbsOutput(
                            **output_kwargs, vendor_gpu_first=True
                        )
                elif video_backend in {"intel", "qsv"}:
                    network_output.close()
                    network_output = (
                        IntelD3D11DirectSbsOutput(**output_kwargs)
                        if video_backend == "intel"
                        else IntelQsvDirectSbsOutput(**output_kwargs)
                    )
                elif video_backend == "vulkan":
                    network_output.close()
                    network_output = VulkanDirectSbsOutput(**output_kwargs)
                elif video_backend == "pynv" and has_nvidia_gpu:
                    try:
                        network_output.close()
                        network_output = PyNvDirectSbsOutput(**output_kwargs)
                    except Exception as exc:
                        print(
                            f"[DirectSbsStream] PyNvVideoCodec startup unavailable: {exc}; "
                            "falling back to FFmpeg without changing MediaMTX settings",
                            flush=True,
                        )
                        network_output = FfmpegDirectSbsOutput(**output_kwargs)
                elif video_backend == "amd" and has_amd_gpu and not selected_audio:
                    try:
                        network_output.close()
                        network_output = AmdAmfDirectSbsOutput(**output_kwargs)
                    except Exception as exc:
                        print(
                            f"[DirectSbsStream] AMD AMF bridge unavailable: {exc}; "
                            "falling back to FFmpeg without changing MediaMTX settings",
                            flush=True,
                        )
                        network_output = FfmpegDirectSbsOutput(**output_kwargs)
                elif video_backend == "amd" and not has_amd_gpu:
                    print(
                        "[DirectSbsStream] AMD AMF backend requires an AMD/Radeon GPU; "
                        "falling back to FFmpeg hardware/software encoder",
                        flush=True,
                    )
                elif video_backend == "amd" and selected_audio:
                    print(
                        "[DirectSbsStream] native AMD AMF path requires audio disabled; "
                        "falling back to FFmpeg to preserve audio",
                        flush=True,
                    )
                elif video_backend == "pynv" and not has_nvidia_gpu:
                    print(
                        "[DirectSbsStream] PyNvVideoCodec is NVIDIA-only; "
                        "falling back to FFmpeg hardware/software encoder",
                        flush=True,
                    )
                if (
                    isinstance(network_output, IntelD3D11DirectSbsOutput)
                    and str(os.environ.get("D2S_ONEVPL_FINAL_SBS", "0")).strip().casefold()
                    in {"1", "true", "yes", "on"}
                ):
                    # StereoRuntime defers the supported Vulkan request to the
                    # Intel sink, which owns the Vulkan image ring and the
                    # D3D11/oneVPL lifetime.
                    os.environ.setdefault("D2S_INTEL_VULKAN_SBS", "1")
            else:
                network_output = MjpegDirectSbsOutput(
                    port=int(settings.get("Streamer Port", 1122)),
                    fps=int(FPS),
                    quality=int(settings.get("Stream Quality", 90)),
                    display_mode=str(settings.get("Display Mode", "Half-SBS")),
                    fit_mode=stream_fit_mode,
                    input_size=stream_input_size,
                    on_stream_fps_selected=adaptive_capture_rate.finish_stream_probe,
                )
            callbacks.set_stream_output(network_output)
            network_output.start()
            output_consumer = DirectSbsOutputConsumer(
                runtime_q=presentation_q,
                shutdown_event=shutdown_event,
                output=network_output,
                source_stat_inc=callbacks.source_stat_inc,
                show_fps_provider=callbacks.show_fps,
                on_sbs_fps=(
                    observe_sbs_fps if adaptive_capture_rate.enabled else None
                ),
                fps_report_interval=(
                    1.0
                    if str(getattr(network_output, "protocol", "")).strip().upper()
                    != "MJPEG"
                    else 5.0
                ),
            )
            output_thread = threading.Thread(
                target=output_consumer.run,
                name="DirectSbsOutputConsumer",
                daemon=True,
            )
            output_thread.start()
        elif str(RUN_MODE).strip().lower() == "viewer":
            # Local Viewer no longer falls through as an unconsumed runtime_q.
            # It owns a GLFW Vulkan surface and presents the already packed SBS.
            from viewer.vulkan_local_viewer import (
                VulkanLocalViewerConfig,
                run_vulkan_local_viewer,
            )

            selected_monitor = (
                int(STEREO_DISPLAY_INDEX)
                if bool(STEREO_DISPLAY_SELECTION)
                else int(MONITOR_INDEX)
            )
            fullscreen_policy = "exclusive"
            fullscreen_target = None
            if (
                OS_NAME == "Windows"
                and configured_run_mode == "Local Viewer"
                and bool(STEREO_DISPLAY_SELECTION)
            ):
                fullscreen_policy, fullscreen_target = (
                    resolve_windows_fullscreen_policy(selected_monitor)
                )
                target_kind = (
                    fullscreen_target.display_kind
                    if fullscreen_target is not None
                    else "unknown"
                )
                target_technology = (
                    fullscreen_target.output_technology
                    if fullscreen_target is not None
                    else None
                )
                target_source = (
                    fullscreen_target.display_kind_source
                    if fullscreen_target is not None
                    else "unmatched"
                )
                print(
                    "[VulkanLocalViewer] Automatic fullscreen policy: "
                    f"display_kind={target_kind} "
                    f"output_technology={target_technology} "
                    f"source={target_source} mode={fullscreen_policy}",
                    flush=True,
                )
            exclude_from_capture = _exclude_local_output_from_capture(
                settings,
                os_name=OS_NAME,
            )
            # Enable cursor passthrough when the SBS fullscreen window covers
            # the same display that is being captured. This applies to
            # 3D Monitor single-display (no second output) and to Window capture
            # when the selected window lives on the chosen stereo-output
            # monitor. The window is made click-through (WS_EX_TRANSPARENT /
            # GLFW_MOUSE_PASSTHROUGH) so the system cursor stays visible over
            # the stereo image and input reaches the underlying desktop.
            cursor_passthrough = False
            capture_mode = str(settings.get("Capture Mode", "")).strip()
            if exclude_from_capture:
                try:
                    no_second_output = not bool(STEREO_DISPLAY_SELECTION) or int(
                        STEREO_DISPLAY_INDEX
                    ) == int(MONITOR_INDEX)
                    if no_second_output:
                        cursor_passthrough = True
                    else:
                        try:
                            import mss

                            with mss.mss() as sct:
                                if len(sct.monitors) - 1 <= 1:
                                    cursor_passthrough = True
                        except Exception:
                            pass
                except Exception:
                    cursor_passthrough = True
            # Window capture on the same monitor as the stereo output also
            # needs passthrough even when a second display exists (Local Viewer
            # or 3 to 2 case). The monitor index for Window mode is the window's
            # display (via get_monitor_index_for_point), so equality means the
            # window and the SBS output share one screen.
            if not cursor_passthrough and capture_mode.casefold() == "window":
                try:
                    if bool(STEREO_DISPLAY_SELECTION) and int(
                        STEREO_DISPLAY_INDEX
                    ) == int(MONITOR_INDEX):
                        cursor_passthrough = True
                except Exception:
                    pass
            if cursor_passthrough:
                reason = (
                    "single-display 3D Monitor mode"
                    if exclude_from_capture
                    else "window capture on stereo-output display"
                )
                print(
                    f"[VulkanLocalViewer] Cursor passthrough enabled for {reason}",
                    flush=True,
                )
            # tex_w,tex_h in legacy viewer: original capture size before packing,
            # used so eye ratio stays dynamic with input (W/2 for HalfSBS etc.).
            input_size: tuple[int, int] | None = None
            try:
                cap_mode = str(settings.get("Capture Mode", "")).strip()
                if cap_mode.casefold() == "window":
                    title = str(settings.get("Window Title", "")).strip()
                    if title and OS_NAME == "Windows":
                        try:
                            import win32gui

                            hwnd = win32gui.FindWindow(None, title)
                            if hwnd:
                                _, _, w, h = win32gui.GetClientRect(hwnd)
                                if w > 0 and h > 0:
                                    input_size = (int(w), int(h))
                        except Exception:
                            pass
                    if input_size is None:
                        from utils.display import get_monitor_size

                        input_size = get_monitor_size(int(MONITOR_INDEX))
                else:
                    from utils.display import get_monitor_size

                    input_size = get_monitor_size(int(MONITOR_INDEX))
            except Exception:
                input_size = None
            local_viewer_config = VulkanLocalViewerConfig(
                title=f"{WINDOW_TITLE or 'Desktop2Stereo'} Vulkan Viewer",
                monitor_index=max(0, selected_monitor),
                fullscreen=bool(STEREO_DISPLAY_SELECTION) or bool(cursor_passthrough),
                capture_compatible_fullscreen=(
                    fullscreen_policy == "capture_compatible"
                ),
                window_preview=bool(settings.get("Window Preview", False)),
                preview_monitor_index=max(0, int(MONITOR_INDEX)),
                exclude_from_capture=exclude_from_capture,
                show_taskbar_button=(
                    OS_NAME == "Windows"
                    and configured_run_mode in {"Local Viewer", "3D Monitor"}
                    and bool(settings.get("LSFG Support", False))
                ),
                cursor_passthrough=cursor_passthrough,
                input_size=input_size,
                capture_mode=str(settings.get("Capture Mode", "Monitor")),
                window_title=str(settings.get("Window Title", "") or "") if str(settings.get("Capture Mode", "")).casefold() == "window" else None,
                vsync=bool(LOCAL_VSYNC),
                show_fps=bool(SHOW_FPS),
                show_fps_provider=callbacks.show_fps,
                display_mode=str(DISPLAY_MODE),
                display_fit_mode=settings.get("Display Fit Mode", "contain"),
                display_fit_mode_provider=callbacks.display_fit_mode,
                on_sbs_fps=observe_sbs_fps if adaptive_capture_rate.enabled else None,
                on_display_refresh_warning=report_display_refresh_warning,
                on_capture_refresh_warning=report_capture_refresh_warning,
                on_breakdown_inc=callbacks.breakdown_inc,
                on_breakdown_add_time=callbacks.breakdown_add_time,
            )
            local_viewer_kwargs = {
                "runtime_q": presentation_q,
                "shutdown_event": shutdown_event,
                "config": local_viewer_config,
            }

            def _packer_on_stat(name: str, value: float) -> None:
                # Durations aggregate as times; counters as increments.
                if name == "packer_ms":
                    # add_time appends _ms; store as packer_ms directly.
                    callbacks.breakdown_add_time("packer", value / 1000.0)
                elif name == "fused_pack_ms":
                    callbacks.breakdown_add_time("rt_fused_pack", value / 1000.0)
                else:
                    callbacks.breakdown_inc(name, int(value))

            if OS_NAME == "Darwin":
                from viewer.host_frame_packer import (
                    maybe_install_local_viewer_packer,
                )

                _maybe_install_local_viewer_packer = (
                    maybe_install_local_viewer_packer
                )
            else:
                def _maybe_install_local_viewer_packer(**_kw):
                    return False
            if _maybe_install_local_viewer_packer(
                pipeline_q=presentation_q,
                local_viewer_kwargs=local_viewer_kwargs,
                on_stat=_packer_on_stat,
                os_name=OS_NAME,
            ):
                # Pipeline still writes to presentation_q untouched; the
                # viewer consumes from the packer's output instead.
                pass
            if OS_NAME == "Darwin":
                # macOS GLFW/NSApp requires every windowing call on the
                # process main thread (glfw.init deadlocks off it). The idle
                # wait loop below is parked on a helper thread instead and the
                # viewer owns the main thread.
                #
                # Default viewer on macOS is Vulkan since round 24 (fused
                # Metal warp-pack kernel via torch.mps.compile_shader beats
                # the CAMetalLayer path). D2S_MAC_VIEWER=metal selects the
                # full CAMetalLayer deferred-warp path; on its failure this
                # still falls back to run_vulkan_local_viewer, which chains
                # Vulkan -> Metal stub -> OpenGL.
                use_metal_viewer = (
                    os.environ.get("D2S_MAC_VIEWER", "vulkan").strip().lower()
                    == "metal"
                )

                def _run_viewer_on_main_thread():
                    try:
                        if use_metal_viewer:
                            try:
                                os.environ.setdefault("D2S_METAL_SHADER_WARP", "1")
                                from viewer.macos_metal_viewer import (
                                    run_metal_local_viewer,
                                )

                                run_metal_local_viewer(**local_viewer_kwargs)
                                return
                            except Exception as exc:
                                # Fallback viewers consume synthesized SBS;
                                # drop the deferred-warp flag so the runtime
                                # stops shipping raw rgb+depth.
                                os.environ.pop("D2S_METAL_SHADER_WARP", None)
                                print(
                                    f"[MetalLocalViewer] Metal viewer failed "
                                    f"({type(exc).__name__}: {exc}); falling "
                                    "back to Vulkan",
                                    flush=True,
                                )
                        else:
                            # Primary/fallback Vulkan path: the fused
                            # warp-pack needs the runtime to ship raw
                            # rgb+depth tensors instead of synthesized SBS.
                            os.environ.setdefault("D2S_METAL_SHADER_WARP", "1")
                        # Vulkan first; internally falls back Metal -> OpenGL.
                        run_vulkan_local_viewer(**local_viewer_kwargs)
                    finally:
                        shutdown_event.set()

                main_thread_job = _run_viewer_on_main_thread
            else:
                local_viewer_thread = threading.Thread(
                    target=run_vulkan_local_viewer,
                    kwargs=local_viewer_kwargs,
                    name="VulkanLocalViewer",
                    daemon=True,
                )
                local_viewer_thread.start()
        print(
            f"Desktop2Stereo Vulkan runtime started: mode={RUN_MODE} device={DEVICE_INFO}",
            flush=True,
        )
        deadline = (
            None
            if max_seconds is None
            else time.monotonic() + max(0.0, max_seconds)
        )

        def _wait_until_shutdown():
            while not shutdown_event.is_set():
                if deadline is not None and time.monotonic() >= deadline:
                    # Also unblock a viewer that owns the main thread.
                    shutdown_event.set()
                    break
                time.sleep(0.05)

        if main_thread_job is not None:
            threading.Thread(
                target=_wait_until_shutdown,
                name="RuntimeWait",
                daemon=True,
            ).start()
            main_thread_job()
        else:
            _wait_until_shutdown()
    except KeyboardInterrupt:
        pass
    finally:
        shutdown_event.set()
        if stop_request_thread is not None:
            stop_request_thread.join(timeout=0.2)
        callbacks.stop_active_capture_session()
        _queue_clear(context.raw_q)
        _queue_clear(context.runtime_q)
        _queue_clear(context.presentation_q)
        if nvfruc_stage is not None:
            nvfruc_stage.close()
        if nvfruc_thread is not None:
            nvfruc_thread.join(timeout=2.0)
        pipeline_thread.join(timeout=2.0)
        capture_thread.join(timeout=2.0)
        if output_thread is not None:
            output_thread.join(timeout=2.0)
        if output_consumer is not None:
            close_output_consumer = getattr(output_consumer, "close", None)
            if callable(close_output_consumer):
                close_output_consumer()
        if network_output is not None:
            try:
                network_output.close()
            except Exception:
                pass
        if presenter_thread is not None:
            # run_until owns Filament/Vulkan teardown on the Presenter thread.
            # Do not let the main thread race that teardown after a timeout.
            presenter_thread.join()
        runtime_exit_code = _openxr_runtime_exit_code(presenter)
        if local_viewer_thread is not None:
            local_viewer_thread.join(timeout=2.0)
        if presenter is not None:
            presenter.close()
        close = getattr(context.stereo_runtime, "close", None)
        if callable(close):
            close()
    # rc=77 is handled by the GUI with a bounded fresh-process relaunch.  A
    # dead Vulkan device cannot be repaired by in-process OpenXR reconnects.
    return runtime_exit_code
