import sys
import types

import pytest

from capture import CaptureConfig, create_capture_runner
from capture.factory import get_desktop_grabber_class, normalize_config
from capture.factory import DesktopDuplicationFallbackRunner
from capture.runners import PollingCaptureRunner


class FakeGrabber:
    pass


def _install_fake_backend(monkeypatch, module_name):
    module = types.ModuleType(module_name)
    module.DesktopGrabber = FakeGrabber
    monkeypatch.setitem(sys.modules, module_name, module)
    return module


def test_windows_dxcamera_selects_dxcamera_backend(monkeypatch):
    _install_fake_backend(monkeypatch, "capture.backends.windows_dxcamera")

    cls = get_desktop_grabber_class(CaptureConfig(os_name="Windows", capture_tool="DXCamera"))

    assert cls is FakeGrabber


def test_windows_capture_tools_select_event_runner():
    for capture_tool in ["WindowsCapture", "WindowsCaptureCUDA", "WindowsCaptureROCm"]:
        runner = create_capture_runner(CaptureConfig(os_name="Windows", capture_tool=capture_tool))

        assert runner.__class__.__name__ == "WindowsCaptureEventRunner"


def test_gpu_required_windows_capture_selects_cuda_or_rocm(monkeypatch):
    for hip, expected in ((None, "WindowsCaptureCUDA"), ("7.1", "WindowsCaptureROCm")):
        torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: True),
            version=types.SimpleNamespace(hip=hip),
        )
        monkeypatch.setitem(sys.modules, "torch", torch)

        runner = create_capture_runner(
            CaptureConfig(
                os_name="Windows",
                capture_tool="DXCamera",
                gpu_required=True,
            )
        )

        assert runner.capture_tool == expected


def test_gpu_required_capture_selects_the_installed_vendor_tool(monkeypatch):
    import types

    for hip, expected in ((None, "WindowsCaptureCUDA"), ("7.1", "WindowsCaptureROCm")):
        torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: True),
            version=types.SimpleNamespace(hip=hip),
        )
        monkeypatch.setitem(sys.modules, "torch", torch)

        runner = create_capture_runner(
            CaptureConfig(
                os_name="Windows",
                capture_tool="DXCamera",
                gpu_required=True,
            )
        )

        assert runner.capture_tool == expected


def test_macos_screencapturekit_selects_sck_backend(monkeypatch):
    _install_fake_backend(monkeypatch, "capture.backends.macos_screencapturekit")

    cls = get_desktop_grabber_class(CaptureConfig(os_name="Darwin", capture_tool="ScreenCaptureKit"))

    assert cls is FakeGrabber


def test_macos_non_sck_selects_coregraphics_backend(monkeypatch):
    _install_fake_backend(monkeypatch, "capture.backends.macos_coregraphics")

    cls = get_desktop_grabber_class(CaptureConfig(os_name="Darwin", capture_tool="CoreGraphics"))

    assert cls is FakeGrabber


def test_linux_selects_mss_backend(monkeypatch):
    _install_fake_backend(monkeypatch, "capture.backends.linux_mss")

    cls = get_desktop_grabber_class(CaptureConfig(os_name="Linux", capture_tool="DXCamera"))

    assert cls is FakeGrabber


def test_polling_tools_create_polling_runner(monkeypatch):
    _install_fake_backend(monkeypatch, "capture.backends.windows_dxcamera")

    for capture_tool in ["DXCamera"]:
        runner = create_capture_runner(CaptureConfig(os_name="Windows", capture_tool=capture_tool))

        assert isinstance(runner, PollingCaptureRunner)


@pytest.mark.skipif(sys.platform != "win32", reason="Desktop Duplication is Windows-only")
def test_desktop_duplication_selects_explicit_backend():
    cls = get_desktop_grabber_class(
        CaptureConfig(os_name="Windows", capture_tool="DesktopDuplication")
    )

    assert cls.__module__.endswith("windows_desktop_duplication")


@pytest.mark.skipif(sys.platform != "win32", reason="Desktop Duplication is Windows-only")
def test_desktop_duplication_is_polling_runner():
    runner = create_capture_runner(
        CaptureConfig(os_name="Windows", capture_tool="DesktopDuplication")
    )

    assert isinstance(runner, PollingCaptureRunner)


def test_desktop_duplication_tries_windows_capture_when_native_bridge_is_missing(monkeypatch):
    monkeypatch.setattr("capture.factory._desktop_duplication_native_available", lambda: False)

    runner = create_capture_runner(
        CaptureConfig(os_name="Windows", capture_tool="DesktopDuplication")
    )

    assert isinstance(runner, DesktopDuplicationFallbackRunner)


def test_normalize_config_preserves_media_runtime_fields():
    render_size_config = object()
    config = CaptureConfig(
        media_path="D:/media/movie.mkv",
        media_control_path="D:/runtime/control.json",
        media_gui_control_path="D:/runtime/control.json",
        media_state_path="D:/runtime/state.json",
        media_session_id="session-123",
        media_resume_position=12.5,
        media_audio_enabled=False,
        media_presentation_sync=True,
        render_size_config=render_size_config,
    )

    normalized = normalize_config(config)
    assert normalized.media_presentation_sync is True

    assert (
        normalized.media_path,
        normalized.media_control_path,
        normalized.media_gui_control_path,
        normalized.media_state_path,
        normalized.media_session_id,
        normalized.media_resume_position,
        normalized.media_audio_enabled,
        normalized.render_size_config,
    ) == (
        config.media_path,
        config.media_control_path,
        config.media_gui_control_path,
        config.media_state_path,
        config.media_session_id,
        config.media_resume_position,
        config.media_audio_enabled,
        render_size_config,
    )



def test_polling_runner_stops_source_once_after_normal_exit():
    import threading

    shutdown = threading.Event()
    stop_calls = []

    class Source:
        def grab(self):
            shutdown.set()
            return None, (1, 1)

        def stop(self):
            stop_calls.append("stop")

    runner = PollingCaptureRunner(CaptureConfig(), lambda: Source())
    runner.run(shutdown_event=shutdown, on_frame=lambda _frame: None)
    runner.stop()

    assert stop_calls == ["stop"]


def test_polling_runner_stops_source_when_grab_raises():
    import threading

    stop_calls = []

    class Source:
        def grab(self):
            raise RuntimeError("capture failed")

        def stop(self):
            stop_calls.append("stop")

    runner = PollingCaptureRunner(CaptureConfig(), lambda: Source())
    try:
        runner.run(shutdown_event=threading.Event(), on_frame=lambda _frame: None)
    except RuntimeError as exc:
        assert str(exc) == "capture failed"
    else:
        raise AssertionError("capture error was not propagated")

    assert stop_calls == ["stop"]


def test_polling_runner_marks_host_video_frames_as_media_for_gpu_preprocessing():
    import threading

    import numpy as np

    from stereo_runtime.pipeline import _gpu_capture_input_requires_device_residency

    shutdown = threading.Event()
    captured = []

    class Source:
        mode = "video"
        frame_format = "bgr24"

        def grab(self):
            return np.zeros((2, 2, 3), dtype=np.uint8), (2, 2)

        def stop(self):
            pass

    def on_frame(frame):
        captured.append(frame)
        shutdown.set()

    runner = PollingCaptureRunner(
        CaptureConfig(capture_mode="Video", gpu_required=True), lambda: Source()
    )
    runner.run(shutdown_event=shutdown, on_frame=on_frame)

    assert captured[0].capture_mode == "Video"
    assert not _gpu_capture_input_requires_device_residency(
        types.SimpleNamespace(gpu_required=True), captured[0]
    )


def test_polling_runner_uses_current_media_mode_after_source_switch():
    import threading

    import numpy as np

    shutdown = threading.Event()
    captured = []

    class Source:
        mode = "image"
        frame_format = "bgr24"

        def grab(self):
            return np.zeros((2, 2, 3), dtype=np.uint8), (2, 2)

        def stop(self):
            pass

    def on_frame(frame):
        captured.append(frame)
        shutdown.set()

    runner = PollingCaptureRunner(
        CaptureConfig(capture_mode="Video"), lambda: Source()
    )
    runner.run(shutdown_event=shutdown, on_frame=on_frame)

    assert captured[0].capture_mode == "Image"


def test_media_controls_progress_while_ordered_frame_publication_is_blocked():
    import threading
    import numpy as np

    shutdown = threading.Event()
    blocked = threading.Event()
    command_applied = threading.Event()
    source_stopped = []
    class Source:
        mode = "video"
        frame_format = "bgr24"
        def grab(self):
            return np.zeros((2, 2, 3), dtype=np.uint8), (2, 2)
        def poll_control(self):
            if blocked.is_set():
                command_applied.set()
        def stop(self):
            source_stopped.append(True)
    runner = PollingCaptureRunner(CaptureConfig(capture_mode="Video"), Source)
    def publish(_frame):
        blocked.set()
        assert command_applied.wait(1.0), "Resume must not depend on a paused FIFO consumer"
        shutdown.set()
    runner.run(shutdown_event=shutdown, on_frame=publish)
    assert source_stopped == [True]
