from __future__ import annotations

from dataclasses import replace

from .runners import PollingCaptureRunner
from .types import CaptureConfig

_WINDOWS_EVENT_TOOLS = {"WindowsCapture", "WindowsCaptureROCm", "WindowsCaptureCUDA"}
_WINDOWS_DESKTOP_DUPLICATION = "DesktopDuplication"


def _default_os_name():
    from utils import OS_NAME
    return OS_NAME


def _default_capture_tool():
    from utils import CAPTURE_TOOL
    return CAPTURE_TOOL


def _desktop_duplication_native_available() -> bool:
    try:
        from .backends.desktop_duplication_native import probe

        return bool(probe().get("available"))
    except Exception:
        return False


class DesktopDuplicationFallbackRunner:
    """Try WindowsCapture before Desktop Duplication's DXCamera fallback."""

    def __init__(self, config: CaptureConfig):
        self.config = config
        self._polling = PollingCaptureRunner(
            config, lambda: create_capture_source(config)
        )
        self._event = None

    def stop(self):
        if self._event is not None:
            self._event.stop()
        self._polling.stop()

    def run(self, **callbacks):
        if _desktop_duplication_native_available():
            return self._polling.run(**callbacks)
        try:
            from .backends.windows_capture_event import WindowsCaptureEventRunner

            event_config = replace(self.config, capture_tool="WindowsCapture")
            self._event = WindowsCaptureEventRunner(event_config)
            return self._event.run(**callbacks)
        except Exception as exc:
            print(
                "[capture] WindowsCapture fallback unavailable; "
                f"using Desktop Duplication/DXCamera fallback: {exc}",
                flush=True,
            )
            return self._polling.run(**callbacks)


def normalize_config(config: CaptureConfig | None = None) -> CaptureConfig:
    if config is None:
        return CaptureConfig(os_name=_default_os_name(), capture_tool=_default_capture_tool())
    os_name = config.os_name or _default_os_name()
    capture_tool = config.capture_tool or _default_capture_tool()
    return CaptureConfig(
        output_resolution=config.output_resolution,
        fps=config.fps,
        window_title=config.window_title,
        capture_mode=config.capture_mode,
        monitor_index=config.monitor_index,
        capture_tool=capture_tool,
        device_id=config.device_id,
        os_name=os_name,
        fps_provider=config.fps_provider,
        cursor_capture=bool(getattr(config, "cursor_capture", True)),
        excluded_window_handles_provider=config.excluded_window_handles_provider,
        on_window_exclusion_ready=config.on_window_exclusion_ready,
        on_window_exclusion_fallback=config.on_window_exclusion_fallback,
        gpu_required=bool(config.gpu_required),
        media_path=config.media_path,
        media_control_path=config.media_control_path,
        media_gui_control_path=config.media_gui_control_path,
        media_viewer_control_path=getattr(config, "media_viewer_control_path", None),
        media_state_path=config.media_state_path,
        media_session_id=config.media_session_id,
        media_resume_position=float(config.media_resume_position or 0.0),
        media_volume=float(getattr(config, "media_volume", 1.0) or 1.0),
        media_muted=bool(getattr(config, "media_muted", False)),
        media_browser_directory=getattr(config, "media_browser_directory", None),
        media_audio_enabled=bool(config.media_audio_enabled),
        media_presentation_sync=bool(getattr(config, "media_presentation_sync", False)),
        render_size_config=getattr(config, "render_size_config", None),
        window_handle=getattr(config, "window_handle", None),
        window_rect=getattr(config, "window_rect", None),
    )


def get_desktop_grabber_class(config: CaptureConfig | None = None):
    config = normalize_config(config)
    if config.os_name == "Windows":
        if config.capture_tool == _WINDOWS_DESKTOP_DUPLICATION:
            from .backends.windows_desktop_duplication import DesktopGrabber
            return DesktopGrabber
        if config.capture_tool in _WINDOWS_EVENT_TOOLS:
            raise RuntimeError(
                f"{config.capture_tool} is an event capture backend; use create_capture_runner instead"
            )
        from .backends.windows_dxcamera import DesktopGrabber
        return DesktopGrabber
    if config.os_name == "Darwin":
        if config.capture_tool == "ScreenCaptureKit":
            from .backends.macos_screencapturekit import DesktopGrabber
            return DesktopGrabber
        from .backends.macos_coregraphics import DesktopGrabber
        return DesktopGrabber
    if config.os_name and config.os_name.startswith("Linux"):
        from .backends.linux_mss import DesktopGrabber
        return DesktopGrabber
    from .backends.windows_dxcamera import DesktopGrabber
    return DesktopGrabber


class DesktopGrabber:
    def __new__(cls, *args, **kwargs):
        backend_cls = get_desktop_grabber_class()
        return backend_cls(*args, **kwargs)


def _is_media_mode(config: CaptureConfig) -> bool:
    return str(config.capture_mode or "").strip().casefold() in {
        "file", "video", "image",
    }


def create_capture_source(config: CaptureConfig | None = None):
    config = normalize_config(config)
    if _is_media_mode(config):
        from .media import MediaCaptureSource

        return MediaCaptureSource(config)
    source_cls = get_desktop_grabber_class(config)
    kwargs = {
        "output_resolution": config.output_resolution,
        "fps": config.fps,
        "window_title": config.window_title,
        "capture_mode": config.capture_mode,
        "monitor_index": config.monitor_index,
    }
    if config.os_name == "Darwin" and config.capture_tool == "ScreenCaptureKit":
        kwargs["fps_provider"] = config.fps_provider
        kwargs["gpu_required"] = bool(config.gpu_required)
    return source_cls(**kwargs)


def create_capture_runner(config: CaptureConfig | None = None):
    config = normalize_config(config)
    if _is_media_mode(config):
        return PollingCaptureRunner(config, lambda: create_capture_source(config))
    if config.gpu_required and config.os_name == "Windows":
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("GPU capture requires the selected CUDA or ROCm runtime") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("GPU capture is required, but no CUDA or ROCm device is available")
        capture_tool = "WindowsCaptureROCm" if torch.version.hip is not None else "WindowsCaptureCUDA"
        config = replace(config, capture_tool=capture_tool)
    elif config.gpu_required and config.os_name == "Darwin":
        config = replace(config, capture_tool="ScreenCaptureKit")
    if config.os_name == "Windows" and config.capture_tool == _WINDOWS_DESKTOP_DUPLICATION:
        if not _desktop_duplication_native_available():
            return DesktopDuplicationFallbackRunner(config)
        return PollingCaptureRunner(config, lambda: create_capture_source(config))
    if config.os_name == "Windows" and config.capture_tool in _WINDOWS_EVENT_TOOLS:
        from .backends.windows_capture_event import WindowsCaptureEventRunner
        return WindowsCaptureEventRunner(config)
    return PollingCaptureRunner(config, lambda: create_capture_source(config))
