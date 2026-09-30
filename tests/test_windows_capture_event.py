import sys
import threading
import types

import pytest

from capture import CaptureConfig, CapturedFrame, FrameCopyMode
from capture.backends import windows_capture_event


class FakeControl:
    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


class FakeFrame:
    def __init__(self, buffer):
        self.frame_buffer = buffer


class CopyBuffer:
    def __init__(self):
        self.copied = False

    def copy(self):
        self.copied = True
        return "copied-buffer"


class CloneBuffer:
    def __init__(self):
        self.cloned = False

    def clone(self):
        self.cloned = True
        return "cloned-buffer"


class NativeTexture:
    resource_kind = "d3d11_texture"
    format = "BGRA8"
    width = 16
    height = 8
    adapter_luid = 0x1234

    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


class NativeBuffer:
    def __init__(self):
        self.d3d11_texture = NativeTexture()

    def copy(self):
        return "cpu-compat-copy"


def _install_capture_module(monkeypatch, module_name):
    module = types.ModuleType(module_name)

    class FakeWindowsCapture:
        last_instance = None

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.handlers = []
            FakeWindowsCapture.last_instance = self

        def event(self, handler):
            self.handlers.append(handler)
            return handler

        def start(self):
            raise RuntimeError("stop fake capture")

        def stop(self):
            self.stopped = True

    module.WindowsCapture = FakeWindowsCapture
    module.Frame = FakeFrame
    module.InternalCaptureControl = FakeControl
    monkeypatch.setitem(sys.modules, module_name, module)
    return module


def test_load_windows_capture_selects_cuda_and_rocm_modules(monkeypatch):
    cuda = _install_capture_module(monkeypatch, "wc_cuda")
    rocm = _install_capture_module(monkeypatch, "wc_rocm")
    base = _install_capture_module(monkeypatch, "windows_capture")

    assert windows_capture_event._load_windows_capture("WindowsCaptureCUDA")[0] is cuda.WindowsCapture
    assert windows_capture_event._load_windows_capture("WindowsCaptureROCm")[0] is rocm.WindowsCapture
    assert windows_capture_event._load_windows_capture("WindowsCapture")[0] is base.WindowsCapture


def test_windows_capture_runner_uses_copy_or_clone_buffers(monkeypatch):
    module = _install_capture_module(monkeypatch, "wc_cuda")
    monkeypatch.setenv("D2S_WGC_SOFTWARE_THROTTLE", "0")
    monkeypatch.setattr(windows_capture_event, "_setup_dpi_awareness", lambda: None)
    monkeypatch.setattr(windows_capture_event.WindowsCaptureEventRunner, "_start_keyboard_worker", lambda self, event: None)

    runner = windows_capture_event.WindowsCaptureEventRunner(
        CaptureConfig(
            os_name="Windows",
            capture_tool="WindowsCaptureCUDA",
            capture_mode="Monitor",
            monitor_index=3,
            output_resolution=(3840, 2160),
        )
    )
    received = []
    shutdown_event = threading.Event()

    def on_frame(captured_frame):
        received.append(captured_frame)

    def on_error(exc):
        shutdown_event.set()

    runner.run(shutdown_event=shutdown_event, on_frame=on_frame, on_error=on_error)

    capture = module.WindowsCapture.last_instance
    assert capture.kwargs == {"monitor_index": 3, "minimum_update_interval": 17}
    assert len(capture.handlers) == 2

    shutdown_event.clear()
    copy_buffer = CopyBuffer()
    capture.handlers[0](FakeFrame(copy_buffer), FakeControl())
    assert copy_buffer.copied is False
    assert isinstance(received[-1], CapturedFrame)
    assert received[-1].frame is copy_buffer
    assert received[-1].target_height == (3840, 2160)
    assert received[-1].copy_mode is FrameCopyMode.GPU_TENSOR
    assert received[-1].frame_raw_device == "cuda"
    assert received[-1].metadata["zero_copy"] is True
    assert received[-1].capture_tool == "WindowsCaptureCUDA"
    assert received[-1].capture_mode == "Monitor"
    assert received[-1].monitor_index == 3
    assert received[-1].original_format == "CopyBuffer"
    assert received[-1].metadata["backend"] == "windows_capture_event"

    shutdown_event.clear()
    clone_buffer = CloneBuffer()
    capture.handlers[0](FakeFrame(clone_buffer), FakeControl())
    assert clone_buffer.cloned is False
    assert received[-1].frame is clone_buffer
    assert received[-1].copy_mode is FrameCopyMode.GPU_TENSOR


def test_windows_capture_native_resource_is_primary_with_explicit_cpu_compat_copy(monkeypatch):
    module = _install_capture_module(monkeypatch, "windows_capture")
    monkeypatch.setenv("D2S_WGC_SOFTWARE_THROTTLE", "0")
    monkeypatch.setattr(windows_capture_event, "_setup_dpi_awareness", lambda: None)
    monkeypatch.setattr(
        windows_capture_event.WindowsCaptureEventRunner,
        "_start_keyboard_worker",
        lambda self, event: None,
    )

    runner = windows_capture_event.WindowsCaptureEventRunner(
        CaptureConfig(
            os_name="Windows",
            capture_tool="WindowsCapture",
            capture_mode="Monitor",
            monitor_index=1,
            output_resolution=(16, 8),
        )
    )
    received = []
    shutdown_event = threading.Event()
    runner.run(
        shutdown_event=shutdown_event,
        on_frame=received.append,
        on_error=lambda exc: shutdown_event.set(),
    )

    shutdown_event.clear()
    native_buffer = NativeBuffer()
    module.WindowsCapture.last_instance.handlers[0](
        FakeFrame(native_buffer),
        FakeControl(),
    )

    captured = received[-1]
    assert captured.frame is native_buffer.d3d11_texture
    assert captured.cpu_compat_frame == "cpu-compat-copy"
    assert captured.native_resource is captured.frame
    assert captured.copy_mode is FrameCopyMode.NONE
    assert captured.frame_raw_device == "d3d11"
    assert captured.metadata["native_resource_output"] is True
    assert captured.metadata["gpu_to_cpu"] is True
    assert captured.metadata["gpu_copy_count"] == 1
    assert captured.metadata["compatibility_copy_mode"] == "copy"
    assert captured.metadata["compatibility_frame_retained"] is True
    assert captured.metadata["zero_copy"] is False


def test_windows_capture_cuda_software_limiter_maps_120hz_callbacks_to_60fps(monkeypatch):
    monkeypatch.delenv("D2S_WGC_SOFTWARE_THROTTLE", raising=False)
    runner = windows_capture_event.WindowsCaptureEventRunner(
        CaptureConfig(capture_tool="WindowsCaptureCUDA", capture_mode="Monitor", monitor_index=1, fps=60)
    )

    accepted = [
        runner._accept_software_paced_frame(now)
        for now in (10.0, 10.0083, 10.0166, 10.0250, 10.0333)
    ]

    assert accepted == [True, False, True, False, True]
    assert runner._software_limited_frames == 2


def test_windows_capture_cuda_software_limiter_reads_runtime_fps_provider(monkeypatch):
    monkeypatch.delenv("D2S_WGC_SOFTWARE_THROTTLE", raising=False)
    target = {"fps": 60}
    runner = windows_capture_event.WindowsCaptureEventRunner(
        CaptureConfig(
            capture_tool="WindowsCaptureCUDA",
            capture_mode="Monitor",
            monitor_index=1,
            fps=60,
            fps_provider=lambda: target["fps"],
        )
    )

    assert runner._accept_software_paced_frame(10.0) is True
    assert runner._accept_software_paced_frame(10.0083) is False
    target["fps"] = 24
    assert runner._accept_software_paced_frame(10.0166) is True
    assert runner._software_pacing_fps == 24


def test_windows_capture_cuda_software_limiter_can_be_disabled(monkeypatch):
    monkeypatch.setenv("D2S_WGC_SOFTWARE_THROTTLE", "0")
    runner = windows_capture_event.WindowsCaptureEventRunner(
        CaptureConfig(capture_tool="WindowsCaptureCUDA", capture_mode="Monitor", monitor_index=1, fps=60)
    )

    assert all(runner._accept_software_paced_frame(now) for now in (10.0, 10.001, 10.002))


def test_windows_capture_replays_last_frame_at_runtime_target_fps():
    target = {"fps": 20}
    runner = windows_capture_event.WindowsCaptureEventRunner(
        CaptureConfig(
            capture_tool="WindowsCaptureCUDA",
            capture_mode="Monitor",
            monitor_index=1,
            fps=60,
            fps_provider=lambda: target["fps"],
        )
    )
    captured_frame = windows_capture_event.capture_frame_from_raw(
        "frame",
        1080,
        10.0,
        config=runner.config,
        metadata={"backend": "windows_capture_event"},
    )

    runner._remember_emitted_frame(captured_frame, 10.0)

    assert runner._replay_frame_if_due(10.049) is None
    replayed = runner._replay_frame_if_due(10.05)
    assert replayed is not None
    assert replayed.frame == "frame"
    assert replayed.timestamp == pytest.approx(10.05)
    assert replayed.metadata["replayed_static_frame"] is True
    assert captured_frame.metadata == {"backend": "windows_capture_event"}

    target["fps"] = 10
    assert runner._replay_frame_if_due(10.149) is None
    assert runner._replay_frame_if_due(10.15) is not None


def test_windows_capture_cuda_limiter_runs_before_gpu_buffer_delivery(monkeypatch):
    module = _install_capture_module(monkeypatch, "wc_cuda")
    monkeypatch.setattr(windows_capture_event, "_setup_dpi_awareness", lambda: None)
    monkeypatch.setattr(windows_capture_event.WindowsCaptureEventRunner, "_start_keyboard_worker", lambda self, event: None)
    runner = windows_capture_event.WindowsCaptureEventRunner(
        CaptureConfig(capture_tool="WindowsCaptureCUDA", capture_mode="Monitor", monitor_index=1, fps=60)
    )
    shutdown_event = threading.Event()
    received = []
    runner.run(
        shutdown_event=shutdown_event,
        on_frame=received.append,
        on_error=lambda exc: shutdown_event.set(),
    )

    decisions = iter((True, False))
    monkeypatch.setattr(runner, "_accept_software_paced_frame", lambda now: next(decisions))
    copied = []

    def record_copy(frame_buffer, capture_tool):
        copied.append(frame_buffer)
        return frame_buffer, FrameCopyMode.GPU_TENSOR, "cuda"

    monkeypatch.setattr(windows_capture_event, "_copy_frame_buffer", record_copy)
    shutdown_event.clear()
    first = CopyBuffer()
    second = CopyBuffer()
    handler = module.WindowsCapture.last_instance.handlers[0]
    handler(FakeFrame(first), FakeControl())
    handler(FakeFrame(second), FakeControl())

    assert copied == [first]
    assert len(received) == 1


def test_windows_capture_cuda_can_force_frame_copy(monkeypatch):
    monkeypatch.setenv("D2S_WGC_COPY_FRAME_BUFFER", "1")
    clone_buffer = CloneBuffer()

    raw, copy_mode, device = windows_capture_event._copy_frame_buffer(clone_buffer, "WindowsCaptureCUDA")

    assert clone_buffer.cloned is True
    assert raw == "cloned-buffer"
    assert copy_mode is FrameCopyMode.CLONE
    assert device == "cuda"


def test_windows_capture_runner_marks_rocm_direct_device(monkeypatch):
    module = _install_capture_module(monkeypatch, "wc_rocm")
    monkeypatch.setattr(windows_capture_event, "_setup_dpi_awareness", lambda: None)
    monkeypatch.setattr(windows_capture_event.WindowsCaptureEventRunner, "_start_keyboard_worker", lambda self, event: None)

    runner = windows_capture_event.WindowsCaptureEventRunner(
        CaptureConfig(
            os_name="Windows",
            capture_tool="WindowsCaptureROCm",
            capture_mode="Monitor",
            monitor_index=2,
            output_resolution=(1920, 1080),
        )
    )
    received = []
    shutdown_event = threading.Event()

    runner.run(
        shutdown_event=shutdown_event,
        on_frame=received.append,
        on_error=lambda exc: shutdown_event.set(),
    )

    capture = module.WindowsCapture.last_instance
    clone_buffer = CloneBuffer()
    shutdown_event.clear()
    capture.handlers[0](FakeFrame(clone_buffer), FakeControl())

    assert clone_buffer.cloned is False
    assert received[-1].frame is clone_buffer
    assert received[-1].copy_mode is FrameCopyMode.GPU_TENSOR
    assert received[-1].frame_raw_device == "rocm"
    assert received[-1].metadata["zero_copy"] is True


def test_windows_capture_cuda_source_fps_log_defaults_off(capsys):
    runner = windows_capture_event.WindowsCaptureEventRunner(
        CaptureConfig(capture_tool="WindowsCaptureCUDA", capture_mode="Monitor", monitor_index=1)
    )

    runner._log_capture_fps(10.0)
    runner._record_capture_timing(copy_seconds=0.002, enqueue_seconds=0.001, handler_seconds=0.004)
    runner._log_capture_fps(11.0)

    assert capsys.readouterr().out == ""


def test_windows_capture_cuda_logs_source_fps_when_enabled(monkeypatch, capsys):
    monkeypatch.setenv("D2S_WGC_CAPTURE_FPS_LOG", "1")
    runner = windows_capture_event.WindowsCaptureEventRunner(
        CaptureConfig(capture_tool="WindowsCaptureCUDA", capture_mode="Monitor", monitor_index=1)
    )

    runner._log_capture_fps(10.0)
    runner._record_capture_timing(copy_seconds=0.002, enqueue_seconds=0.001, handler_seconds=0.004)
    runner._log_capture_fps(10.5)
    runner._record_capture_timing(copy_seconds=0.004, enqueue_seconds=0.003, handler_seconds=0.006)
    runner._log_capture_fps(11.0)

    assert (
        "[WindowsCaptureCUDA] capture_fps=2.0 frames=2 monitor=1 mode=Monitor "
        "copy_ms=3.00 enqueue_ms=2.00 handler_ms=5.00"
    ) in capsys.readouterr().out


def _reset_capture_gap_defer_state():
    windows_capture_event._PENDING_CAPTURE_GAP_LOGS.clear()
    windows_capture_event._CAPTURE_GAP_DEFER_RELEASED = False


def test_windows_capture_cuda_suppresses_callback_gaps_by_default(monkeypatch, capsys):
    monkeypatch.delenv("D2S_WGC_CAPTURE_GAP_LOG", raising=False)
    runner = windows_capture_event.WindowsCaptureEventRunner(
        CaptureConfig(capture_tool="WindowsCaptureCUDA", capture_mode="Monitor", monitor_index=1)
    )

    runner._log_capture_gap(10.0, {"monitor_index": 1})
    runner._log_capture_gap(10.8, {"monitor_index": 1})

    assert capsys.readouterr().out == ""


def test_windows_capture_cuda_logs_callback_gap(monkeypatch, capsys):
    _reset_capture_gap_defer_state()
    monkeypatch.setenv("D2S_WGC_CAPTURE_GAP_LOG", "1")
    runner = windows_capture_event.WindowsCaptureEventRunner(
        CaptureConfig(capture_tool="WindowsCaptureCUDA", capture_mode="Monitor", monitor_index=1)
    )

    runner._log_capture_gap(10.0, {"monitor_index": 1, "minimum_update_interval": 8})
    runner._log_capture_gap(10.2, {"monitor_index": 1, "minimum_update_interval": 8})
    assert capsys.readouterr().out == ""

    runner._log_capture_gap(10.8, {"monitor_index": 1, "minimum_update_interval": 8})
    assert "[CaptureGap] tool=WindowsCaptureCUDA mode=Monitor monitor=1 gap=0.60s" in capsys.readouterr().out


def test_windows_capture_cuda_logs_at_most_three_callback_gaps(monkeypatch, capsys):
    monkeypatch.setenv("D2S_WGC_CAPTURE_GAP_LOG", "1")
    runner = windows_capture_event.WindowsCaptureEventRunner(
        CaptureConfig(capture_tool="WindowsCaptureCUDA", capture_mode="Monitor", monitor_index=1)
    )

    for now in (10.0, 10.8, 11.6, 12.4, 13.2):
        runner._log_capture_gap(now, {"monitor_index": 1})

    assert capsys.readouterr().out.count("[CaptureGap]") == 3


def test_windows_capture_cuda_defers_gap_until_openxr_projection(monkeypatch, capsys):
    _reset_capture_gap_defer_state()
    monkeypatch.setenv("D2S_WGC_CAPTURE_GAP_LOG", "1")
    monkeypatch.setenv("D2S_DEFER_CAPTURE_GAP_UNTIL_OPENXR_PROJECTION", "1")
    runner = windows_capture_event.WindowsCaptureEventRunner(
        CaptureConfig(capture_tool="WindowsCaptureCUDA", capture_mode="Monitor", monitor_index=1)
    )

    runner._log_capture_gap(10.0, {"monitor_index": 1, "minimum_update_interval": 8})
    runner._log_capture_gap(10.8, {"monitor_index": 1, "minimum_update_interval": 8})

    assert capsys.readouterr().out == ""
    windows_capture_event.flush_pending_capture_gap_logs()
    assert "[CaptureGap] tool=WindowsCaptureCUDA mode=Monitor monitor=1 gap=0.80s" in capsys.readouterr().out


@pytest.mark.parametrize("capture_tool", ["WindowsCapture", "WindowsCaptureROCm"])
def test_non_cuda_capture_does_not_log_source_fps(capture_tool, capsys):
    runner = windows_capture_event.WindowsCaptureEventRunner(
        CaptureConfig(capture_tool=capture_tool, capture_mode="Monitor", monitor_index=1)
    )

    runner._log_capture_fps(10.0)
    runner._log_capture_fps(11.0)

    assert capsys.readouterr().out == ""


def test_windows_capture_runner_uses_window_name_for_window_capture(monkeypatch):
    module = _install_capture_module(monkeypatch, "windows_capture")
    monkeypatch.setattr(windows_capture_event, "_setup_dpi_awareness", lambda: None)
    monkeypatch.setattr(windows_capture_event.WindowsCaptureEventRunner, "_start_keyboard_worker", lambda self, event: None)

    runner = windows_capture_event.WindowsCaptureEventRunner(
        CaptureConfig(
            os_name="Windows",
            capture_tool="WindowsCapture",
            capture_mode="Window",
            window_title="Stereo Viewer",
        )
    )
    shutdown_event = threading.Event()

    def on_error(exc):
        shutdown_event.set()

    runner.run(
        shutdown_event=shutdown_event,
        on_frame=lambda captured_frame: None,
        on_error=on_error,
    )

    assert module.WindowsCapture.last_instance.kwargs == {"window_name": "Stereo Viewer"}


def test_windows_capture_cuda_accepts_env_capture_options(monkeypatch):
    config = CaptureConfig(capture_tool="WindowsCaptureCUDA", capture_mode="Monitor", monitor_index=1)

    assert windows_capture_event._windows_capture_kwargs(config, "WindowsCaptureCUDA") == {
        "monitor_index": 1,
        "minimum_update_interval": 17,
    }

    monkeypatch.setenv("D2S_WGC_REUSE_OUTPUT_BUFFER", "1")
    monkeypatch.setenv("D2S_WGC_OUTPUT_BUFFER_COUNT", "6")

    assert windows_capture_event._windows_capture_kwargs(config, "WindowsCaptureCUDA") == {
        "monitor_index": 1,
        "minimum_update_interval": 17,
        "reuse_output_buffer": True,
        "output_buffer_count": 6,
    }
