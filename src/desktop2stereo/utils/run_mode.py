from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunModeConfig:
    run_mode: str
    stream_mode: str | None
    use_3d_monitor: bool
    lossless_scaling_support: bool
    fix_viewer_aspect: bool


def normalize_run_mode(raw_run_mode: str) -> str:
    run_mode = str(raw_run_mode or "").strip()
    if run_mode.casefold() == "legacy streamer":
        return "MJPEG Streamer"
    if run_mode.casefold() in {
        "nvidia streamer",
        "nvidia gpu streamer",
        "gpu streamer",
    }:
        # GPU streaming is now the vendor-accelerated backend policy of the
        # single advanced network-streaming mode.
        return "RTMP Streamer"
    return run_mode


def target_fps_setting_key(raw_run_mode: str) -> str:
    """Return the FPS setting owned by the selected run mode."""
    return (
        "Stream Target FPS"
        if normalize_run_mode(raw_run_mode) in {"RTMP Streamer", "Moonlight Streamer"}
        else "Target FPS"
    )


def target_fps_for_run_mode(settings: dict, default: int = 0) -> int:
    """Resolve mode-scoped FPS while accepting configurations from before the split."""
    key = target_fps_setting_key(settings.get("Run Mode", "Local Viewer"))
    fallback = settings.get("Target FPS", default)
    try:
        return int(settings.get(key, fallback) or 0)
    except (TypeError, ValueError):
        return int(default)


def resolve_run_mode(
    raw_run_mode: str,
    *,
    os_name: str,
    fix_viewer_aspect: bool,
    lossless_scaling_support: bool,
) -> RunModeConfig:
    raw_run_mode = normalize_run_mode(raw_run_mode)
    use_3d_monitor = False
    stream_mode = None
    resolved_lossless_scaling = False
    resolved_fix_viewer_aspect = True if raw_run_mode == "RTMP Streamer" else fix_viewer_aspect

    if raw_run_mode == "Local Viewer":
        run_mode = "Viewer"
    elif raw_run_mode == "3D Monitor" and os_name == "Windows":
        run_mode = "Viewer"
        use_3d_monitor = True
    elif raw_run_mode == "MJPEG Streamer":
        run_mode = "Viewer"
        stream_mode = "MJPEG"
    elif raw_run_mode == "RTMP Streamer":
        run_mode = "Viewer"
        stream_mode = "RTMP"
        if os_name == "Windows":
            resolved_lossless_scaling = lossless_scaling_support
    elif raw_run_mode == "Moonlight Streamer" and os_name == "Windows":
        run_mode = "Viewer"
        stream_mode = "MOONLIGHT"
    elif raw_run_mode == "OpenXR Link":
        run_mode = "OpenXR"
    else:
        run_mode = "Streamer"

    return RunModeConfig(
        run_mode=run_mode,
        stream_mode=stream_mode,
        use_3d_monitor=use_3d_monitor,
        lossless_scaling_support=resolved_lossless_scaling,
        fix_viewer_aspect=resolved_fix_viewer_aspect,
    )
