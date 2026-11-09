from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import platform

from stereo_runtime.render_size import RenderSizeConfig, render_size_config_from_settings
from utils import display_info
from utils.display import compute_output_resolution, get_fps
from utils.run_mode import is_local_same_monitor_3d_mode, target_fps_for_run_mode
from utils.xr_headset_presets import DEFAULT_XR_HEADSET_MODEL, resolve_xr_headset_preset

from viewer.controller_help import get_controller_help_rows
from viewer.upscaler import normalize_upscaler, normalize_upscaler_sharpness


@dataclass(frozen=True)
class ViewerSettings:
    monitor_index: int
    display_mode: str
    stereo_display_index: int | None
    stereo_display_selection: bool
    output_resolution: int | tuple[int, int]
    render_size_config: RenderSizeConfig
    show_fps: bool
    depth_strength: float
    convergence: float
    capture_mode: str
    window_title: str | None
    target_fps: int
    fps: int
    fill_16_9: bool
    local_vsync: bool
    upscaler: str
    upscaler_sharpness: float
    language: str
    controller_help_rows: list
    environment_help_rows: list
    controller_model: str
    environment_model: str
    xr_headset_model: str
    openxr_screen_width: float
    openxr_screen_distance: float
    xr_preview_window: bool
    _is_4k_default_applied: bool = False


def _resolve_available_display_index(
    saved_index,
    identity,
    displays,
    *,
    fallback_index: int | None = None,
) -> int | None:
    available_indices = {int(display.capture_index) for display in displays}
    if not available_indices:
        return None

    if isinstance(identity, Mapping) and identity:
        resolved_index = display_info.resolve_display_capture_index(
            saved_index,
            identity,
            displays,
        )
    else:
        try:
            resolved_index = int(saved_index)
        except (TypeError, ValueError):
            resolved_index = None

    if resolved_index in available_indices:
        return resolved_index

    if fallback_index in available_indices:
        return fallback_index

    from gui.capture_sources import get_primary_monitor_index

    primary_index = get_primary_monitor_index()
    if primary_index in available_indices:
        return primary_index
    return int(displays[0].capture_index)


def resolve_viewer_settings(settings: dict) -> ViewerSettings:
    displays = display_info.enumerate_displays()
    if not displays:
        raise RuntimeError(
            "no system displays are available; connect or enable a display and restart"
        )

    monitor_index = _resolve_available_display_index(
        settings["Monitor Index"],
        settings.get("Monitor Identity"),
        displays,
    )
    if monitor_index is None:
        raise RuntimeError(
            "no system displays are available; connect or enable a display and restart"
        )
    display_mode = settings["Display Mode"]
    capture_mode = settings["Capture Mode"]
    single_display_local_3d = len(displays) == 1 and is_local_same_monitor_3d_mode(
        settings.get("Run Mode", "Local Viewer"),
        os_name=platform.system(),
        monitor_count=len(displays),
        capture_mode=capture_mode,
        input_monitor_index=monitor_index,
        output_monitor_index=monitor_index,
    )
    if single_display_local_3d:
        stereo_display_index = monitor_index
        stereo_display_selection = True
    elif len(displays) == 1:
        # Ignore old output selections when single-display auto-3D does not
        # apply, such as window capture or platforms without same-monitor mode.
        stereo_display_index = None
        stereo_display_selection = False
    else:
        saved_stereo_display_index = settings.get("Stereo Output")
        stereo_display_index = (
            _resolve_available_display_index(
                saved_stereo_display_index,
                settings.get("Stereo Output Identity"),
                displays,
                fallback_index=monitor_index,
            )
            if saved_stereo_display_index is not None
            else None
        )
        stereo_display_selection = stereo_display_index is not None
    same_monitor_local_3d = is_local_same_monitor_3d_mode(
        settings.get("Run Mode", "Local Viewer"),
        os_name=platform.system(),
        monitor_count=len(displays),
        capture_mode=capture_mode,
        input_monitor_index=monitor_index,
        output_monitor_index=stereo_display_index,
    )
    if same_monitor_local_3d and display_mode not in {
        "Half-SBS", "Full-SBS", "Half-TAB", "Full-TAB"
    }:
        display_mode = "Half-SBS"
    use_stereo_monitor = same_monitor_local_3d
    output_resolution = compute_output_resolution(
        settings.get("Processing Resolution", "Auto"),
        display_mode,
        monitor_index,
        stereo_display_index,
        use_stereo_monitor=use_stereo_monitor,
    )
    render_size_config = render_size_config_from_settings(settings)
    window_title = settings["Window Title"] if capture_mode == "Window" else None
    target_fps = target_fps_for_run_mode(settings)
    # Auto-configured 4K streams follow the target display's actual refresh
    # rate instead of imposing a fixed cap. Explicit Target FPS still wins.
    is_4k = bool(output_resolution)
    if isinstance(output_resolution, (tuple, list)):
        is_4k = int(output_resolution[0]) >= 3840 and int(output_resolution[1]) >= 2160
    elif isinstance(output_resolution, int):
        is_4k = output_resolution >= 2160
    detected_fps = get_fps(window_title, monitor_index)
    if target_fps < 1 and is_4k:
        target_fps = detected_fps
    fps = target_fps if 1 <= target_fps <= 240 else detected_fps
    language = settings["Language"]
    controller_help_rows, environment_help_rows = get_controller_help_rows(language)
    xr_headset_model = settings.get("XR Headset Model", DEFAULT_XR_HEADSET_MODEL)
    xr_headset_preset = resolve_xr_headset_preset(xr_headset_model)

    return ViewerSettings(
        monitor_index=monitor_index,
        display_mode=display_mode,
        stereo_display_index=stereo_display_index,
        stereo_display_selection=stereo_display_selection,
        output_resolution=output_resolution,
        render_size_config=render_size_config,
        show_fps=settings["Show FPS"],
        depth_strength=settings["Depth Strength"],
        convergence=settings["Convergence"],
        capture_mode=capture_mode,
        window_title=window_title,
        target_fps=target_fps,
        fps=fps,
        _is_4k_default_applied=bool(is_4k and target_fps >= 40),
        fill_16_9=settings["Fill 16:9"],
        local_vsync=settings["VSync"],
        upscaler=normalize_upscaler(settings.get("Upscaler", "Off")),
        upscaler_sharpness=normalize_upscaler_sharpness(settings.get("Upscaler Sharpness", 0.35)),
        language=language,
        controller_help_rows=controller_help_rows,
        environment_help_rows=environment_help_rows,
        controller_model=settings["Controller Model"],
        environment_model=settings.get("Environment Model", "Default"),
        xr_headset_model=xr_headset_preset.key,
        openxr_screen_width=xr_headset_preset.width_m,
        openxr_screen_distance=xr_headset_preset.distance_m,
        xr_preview_window=settings.get("XR Preview Window", True),
    )
