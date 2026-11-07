import ctypes
import sys
import threading
import time
import types
from types import SimpleNamespace

import pytest
from capture import CaptureConfig, CapturedFrame, FrameCopyMode
from capture.backends import windows_capture_event


def test_bundled_cuda_capture_patch_paces_before_transfer():
    import runpy
    from pathlib import Path

    patch = runpy.run_path(str(Path(__file__).resolve().parents[1] / "src/env_install/apply-wc-cuda-patches.py"))
    source = '''class Capture:
    def __init__(self, minimum_update_interval):
        self._stream = None
        self._running = True
        self.transfers = []
    def start(self):
        try:
            while self._running:
                # Use a small timeout to keep Python responsive (Ctrl+C)
                self.transfers.append(time.perf_counter())
                time.sleep(0.003)
                self._running = len(self.transfers) < 3
        finally:
            pass
'''
    patched = patch["patched_source"](source)
    assert patch["patched_source"](patched) == patched
    now = [0.0]
    clock = types.SimpleNamespace(perf_counter=lambda: now[0], sleep=lambda delay: now.__setitem__(0, now[0] + delay))
    namespace = {"time": clock}
    exec(patched, namespace)  # noqa: S102 - execute the generated patch source under test.
    capture = namespace["Capture"](16)
    capture.start()
    assert capture.transfers == pytest.approx([0.0, 0.016, 0.032])


def test_bundled_cuda_capture_patch_rejects_unknown_source():
    import runpy
    from pathlib import Path

    patch = runpy.run_path(str(Path(__file__).resolve().parents[1] / "src/env_install/apply-wc-cuda-patches.py"))
    with pytest.raises(ValueError, match="Unsupported wc-cuda source layout"):
        patch["patched_source"]("class DifferentCapture: pass\n")


def test_bundled_rocm_capture_patch_uses_each_hip_device_luid():
    import runpy
    from pathlib import Path

    patch = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "src/env_install/apply-wc-rocm-patches.py")
    )
    source = '''class Capture:
    def __init__(self, minimum_update_interval):
        self._stream        = None
    def start(self):
        try:
            while self._running:
                res = self._inner.get_frame(self._last_id, timeout=0.1)
        finally:
            pass
'''
    patched = patch["patched_source"](source)
    assert patch["patched_source"](patched) == patched

    device_luids = ((0x1234, 5), (0x5678, 9))

    class FakeHipLuid:
        def __init__(self):
            self.argtypes = None
            self.restype = None

        def __call__(self, luid_ptr, mask_ptr, device_id):
            low, high = device_luids[device_id]
            packed = low.to_bytes(4, "little") + high.to_bytes(4, "little", signed=True)
            ctypes.memmove(luid_ptr, packed, len(packed))
            ctypes.cast(mask_ptr, ctypes.POINTER(ctypes.c_uint))[0] = 1
            return 0

    fake_luid = FakeHipLuid()
    released = []
    namespace = {
        "ctypes": ctypes,
        "_get_hip": lambda: SimpleNamespace(hipDeviceGetLuid=fake_luid),
    }
    exec(patched, namespace)  # noqa: S102 - execute the generated patch source under test.

    assert namespace["get_luid"](0) == (0x1234, 5)
    assert namespace["get_luid"](1) == (0x5678, 9)
    assert released == []


def test_bundled_rocm_capture_patch_returns_unknown_when_hip_identity_fails():
    import runpy
    from pathlib import Path

    patch = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "src/env_install/apply-wc-rocm-patches.py")
    )
    source = '''class Capture:
    def __init__(self, minimum_update_interval):
        self._stream        = None
    def start(self):
        try:
            while self._running:
                res = self._inner.get_frame(self._last_id, timeout=0.1)
        finally:
            pass
'''
    patched = patch["patched_source"](source)
    class FakeHipLuid:
        argtypes = None
        restype = None

        def __call__(self, _luid_ptr, _mask_ptr, _device_id):
            return 35

    namespace = {
        "ctypes": ctypes,
        "_get_hip": lambda: SimpleNamespace(hipDeviceGetLuid=FakeHipLuid()),
    }
    exec(patched, namespace)  # noqa: S102 - execute the generated patch source under test.

    assert namespace["get_luid"](0) == (0, 0)


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
    if module_name in {"wc_cuda", "wc_rocm"}:
        module.get_luid = lambda _device_id=0: (0x1234, 0)
    monkeypatch.setitem(sys.modules, module_name, module)
    return module


def test_load_windows_capture_selects_cuda_and_rocm_modules(monkeypatch):
    cuda = _install_capture_module(monkeypatch, "wc_cuda")
    rocm = _install_capture_module(monkeypatch, "wc_rocm")
    base = _install_capture_module(monkeypatch, "windows_capture")

    assert windows_capture_event._load_windows_capture("WindowsCaptureCUDA")[0] is cuda.WindowsCapture
    assert windows_capture_event._load_windows_capture("WindowsCaptureROCm")[0] is rocm.WindowsCapture
    assert windows_capture_event._load_windows_capture("WindowsCapture")[0] is base.WindowsCapture


def test_gpu_required_capture_rejects_cpu_compatibility_frames():
    with pytest.raises(RuntimeError, match="produces CPU frames"):
        windows_capture_event._copy_frame_buffer(
            object(), "WindowsCapture", gpu_required=True
        )


def test_gpu_required_capture_rejects_unverified_adapter_identity(monkeypatch):
    runner = windows_capture_event.WindowsCaptureEventRunner(
        CaptureConfig(capture_tool="WindowsCaptureCUDA", gpu_required=True)
    )
    monkeypatch.setattr(
        windows_capture_event,
        "_capture_adapter_luid",
        lambda *_args, **_kwargs: (0, "adapter identity missing"),
    )

    with pytest.raises(RuntimeError, match="cannot verify adapter identity"):
        runner.run(shutdown_event=threading.Event(), on_frame=lambda _frame: None)


def test_gpu_required_capture_rejects_cpu_compatibility_frames():
    with pytest.raises(RuntimeError, match="produces CPU frames"):
        windows_capture_event._copy_frame_buffer(
            object(), "WindowsCapture", gpu_required=True
        )


def test_gpu_required_capture_rejects_unknown_adapter_identity(monkeypatch):
    runner = windows_capture_event.WindowsCaptureEventRunner(
        CaptureConfig(capture_tool="WindowsCaptureCUDA", gpu_required=True)
    )
    monkeypatch.setattr(
        windows_capture_event,
        "_capture_adapter_luid",
        lambda *_args, **_kwargs: (0, "adapter identity missing"),
    )

    with pytest.raises(RuntimeError, match="cannot verify adapter identity"):
        runner.run(shutdown_event=threading.Event(), on_frame=lambda _frame: None)


def test_windows_capture_runner_uses_copy_or_clone_buffers(monkeypatch):
    module = _install_capture_module(monkeypatch, "wc_cuda")
    monkeypatch.setenv("D2S_WGC_SOFTWARE_THROTTLE", "0")
    monkeypatch.setattr(windows_capture_event, "_setup_dpi_awareness", lambda: None)
    monkeypatch.setattr(windows_capture_event, "_capture_monitor_rect", lambda _index: (0, 0, 1920, 1080))
    monkeypatch.setattr(windows_capture_event, "_cursor_position_uv", lambda _rect: (0.25, 0.75))
    monkeypatch.setattr(windows_capture_event.WindowsCaptureEventRunner, "_start_keyboard_worker", lambda self, event: None)

    runner = windows_capture_event.WindowsCaptureEventRunner(
        CaptureConfig(
            os_name="Windows",
            capture_tool="WindowsCaptureCUDA",
            device_id=1,
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
    assert capture.kwargs == {
        "monitor_index": 3,
        "cursor_capture": True,
        "minimum_update_interval": 16,
        "device_id": 1,
    }
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
    assert received[-1].metadata["resource_kind"] == "cuda_tensor"
    assert received[-1].capture_tool == "WindowsCaptureCUDA"
    assert received[-1].capture_mode == "Monitor"
    assert received[-1].monitor_index == 3
    assert received[-1].original_format == "CopyBuffer"
    assert received[-1].metadata["backend"] == "windows_capture_event"
    assert received[-1].metadata["cursor_capture"] is True
    assert received[-1].metadata["cursor_position_uv"] == (0.25, 0.75)

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

    assert runner._replay_frame_if_due(10.099) is None
    replayed = runner._replay_frame_if_due(10.1)
    assert replayed is not None
    assert replayed.frame == "frame"
    assert replayed.timestamp == pytest.approx(10.1)
    assert replayed.metadata["replayed_static_frame"] is True
    assert captured_frame.metadata == {"backend": "windows_capture_event"}

    target["fps"] = 10
    assert runner._replay_frame_if_due(10.199) is None
    assert runner._replay_frame_if_due(10.2) is not None


@pytest.mark.parametrize("capture_tool", ["WindowsCaptureCUDA", "WindowsCaptureROCm"])
def test_gpu_replay_waits_for_capture_stall_before_replaying(capture_tool):
    runner = windows_capture_event.WindowsCaptureEventRunner(
        CaptureConfig(capture_tool=capture_tool, fps=60)
    )
    captured_frame = windows_capture_event.capture_frame_from_raw(
        "frame", 1080, 10.0, config=runner.config
    )

    runner._remember_emitted_frame(captured_frame, 10.0)

    assert runner._replay_frame_if_due(10.032) is None
    replayed = runner._replay_frame_if_due(10.034)
    assert replayed is not None
    assert replayed.metadata["replayed_static_frame"] is True


def test_rocm_replay_cannot_be_published_after_a_new_capture(monkeypatch):
    runner = windows_capture_event.WindowsCaptureEventRunner(
        CaptureConfig(capture_tool="WindowsCaptureROCm", fps=60)
    )
    old_frame = windows_capture_event.capture_frame_from_raw(
        "old", 1080, 1.0, config=runner.config
    )
    new_frame = windows_capture_event.capture_frame_from_raw(
        "new", 1080, 2.0, config=runner.config
    )
    runner._remember_emitted_frame(old_frame, time.perf_counter() - 1.0)

    replay_selected = threading.Event()
    release_replay = threading.Event()
    capture_waiting = threading.Event()
    published = []
    original_replay = runner._replay_frame_if_due

    def pause_after_selecting_replay(now):
        replay_frame = original_replay(now)
        replay_selected.set()
        assert release_replay.wait(timeout=2.0)
        return replay_frame

    class ObservedLock:
        def __init__(self):
            self._lock = threading.Lock()

        def __enter__(self):
            if threading.current_thread().name == "fresh-capture":
                capture_waiting.set()
            self._lock.acquire()
            return self

        def __exit__(self, *_exc):
            self._lock.release()

    monkeypatch.setattr(runner, "_replay_frame_if_due", pause_after_selecting_replay)
    runner._frame_publish_lock = ObservedLock()

    def on_frame(frame):
        published.append(
            (frame.frame, bool(frame.metadata.get("replayed_static_frame")))
        )

    replay_thread = threading.Thread(
        target=runner._publish_replay_frame_if_due,
        args=(on_frame,),
        name="static-replay",
    )
    replay_thread.start()
    assert replay_selected.wait(timeout=2.0)

    capture_thread = threading.Thread(
        target=runner._publish_captured_frame,
        args=(new_frame, 2.0, on_frame),
        name="fresh-capture",
    )
    capture_thread.start()
    assert capture_waiting.wait(timeout=2.0)

    release_replay.set()
    replay_thread.join(timeout=2.0)
    capture_thread.join(timeout=2.0)

    assert not replay_thread.is_alive()
    assert not capture_thread.is_alive()
    assert published == [("old", True), ("new", False)]


def test_frame_publication_lock_serializes_gpu_capture_and_replay():
    rocm = windows_capture_event.WindowsCaptureEventRunner(
        CaptureConfig(capture_tool="WindowsCaptureROCm")
    )
    cuda = windows_capture_event.WindowsCaptureEventRunner(
        CaptureConfig(capture_tool="WindowsCaptureCUDA")
    )

    assert rocm._frame_publish_lock is not None
    assert cuda._frame_publish_lock is not None


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

    def record_copy(frame_buffer, capture_tool, *, gpu_required=False):
        assert gpu_required is False
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
            device_id=1,
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
    assert received[-1].metadata["resource_kind"] == "rocm_tensor"
    assert received[-1].metadata["capture_adapter_luid"] == 0x1234
    assert received[-1].metadata["capture_device_id"] == 1


@pytest.mark.parametrize(
    ("capture_tool", "module_name"),
    [
        ("WindowsCaptureCUDA", "wc_cuda"),
        ("WindowsCaptureROCm", "wc_rocm"),
    ],
)
def test_gpu_capture_unknown_adapter_falls_back_to_compatibility_capture(
    monkeypatch, capture_tool, module_name
):
    gpu_capture = _install_capture_module(monkeypatch, module_name)
    gpu_capture.get_luid = lambda _device_id=0: (0, 0)
    compatibility = _install_capture_module(monkeypatch, "windows_capture")
    monkeypatch.setattr(windows_capture_event, "_setup_dpi_awareness", lambda: None)
    monkeypatch.setattr(
        windows_capture_event.WindowsCaptureEventRunner,
        "_start_keyboard_worker",
        lambda self, _event: None,
    )

    runner = windows_capture_event.WindowsCaptureEventRunner(
        CaptureConfig(capture_tool=capture_tool, capture_mode="Monitor", monitor_index=1)
    )
    shutdown_event = threading.Event()
    runner.run(
        shutdown_event=shutdown_event,
        on_frame=lambda _frame: None,
        on_error=lambda _exc: shutdown_event.set(),
    )

    assert runner.capture_tool == "WindowsCapture"
    assert compatibility.WindowsCapture.last_instance is not None
    assert gpu_capture.WindowsCapture.last_instance is None


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

    assert module.WindowsCapture.last_instance.kwargs == {
        "window_name": "Stereo Viewer",
        "cursor_capture": True,
    }


def test_windows_capture_cuda_accepts_env_capture_options(monkeypatch):
    config = CaptureConfig(capture_tool="WindowsCaptureCUDA", capture_mode="Monitor", monitor_index=1)

    assert windows_capture_event._windows_capture_kwargs(config, "WindowsCaptureCUDA") == {
        "monitor_index": 1,
        "cursor_capture": True,
        "minimum_update_interval": 16,
    }

    monkeypatch.setenv("D2S_WGC_REUSE_OUTPUT_BUFFER", "1")
    monkeypatch.setenv("D2S_WGC_OUTPUT_BUFFER_COUNT", "6")

    assert windows_capture_event._windows_capture_kwargs(config, "WindowsCaptureCUDA") == {
        "monitor_index": 1,
        "cursor_capture": True,
        "minimum_update_interval": 16,
        "reuse_output_buffer": True,
        "output_buffer_count": 6,
    }


def test_windows_capture_can_disable_source_cursor_for_sbs_overlay():
    config = CaptureConfig(
        capture_tool="WindowsCaptureCUDA",
        capture_mode="Monitor",
        monitor_index=1,
        cursor_capture=False,
    )

    assert windows_capture_event._windows_capture_kwargs(
        config, "WindowsCaptureCUDA"
    )["cursor_capture"] is False


@pytest.mark.parametrize("crate", ("wc_cuda", "wc_rocm"))
def test_native_gpu_capture_maps_cursor_capture_false_to_without_cursor(crate):
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "native"
        / crate
        / "src"
        / "lib.rs"
    ).read_text(encoding="utf-8")
    assert "cursor_capture: bool" in source
    assert "CursorCaptureSettings::WithoutCursor" in source
    assert "cursor_capture=true" in source


def test_gpu_capture_kwargs_include_only_current_exclusion_handles():
    config = CaptureConfig(
        capture_tool="WindowsCaptureCUDA",
        capture_mode="Monitor",
        monitor_index=1,
        excluded_window_handles_provider=lambda: (123, 456),
    )

    active = windows_capture_event._windows_capture_kwargs(
        config,
        "WindowsCaptureCUDA",
    )
    fallback = windows_capture_event._windows_capture_kwargs(
        config,
        "WindowsCaptureCUDA",
        include_window_exclusion=False,
    )

    assert active["excluded_window_handles"] == [123, 456]
    assert "excluded_window_handles" not in fallback


def test_missing_gpu_session_exclusion_binding_falls_back_with_status(monkeypatch):
    module = _install_capture_module(monkeypatch, "wc_cuda")
    monkeypatch.setattr(windows_capture_event, "_setup_dpi_awareness", lambda: None)
    monkeypatch.setattr(
        windows_capture_event.WindowsCaptureEventRunner,
        "_start_keyboard_worker",
        lambda self, event: None,
    )
    fallback = []
    errors = []
    shutdown_event = threading.Event()
    runner = windows_capture_event.WindowsCaptureEventRunner(
        CaptureConfig(
            os_name="Windows",
            capture_tool="WindowsCaptureCUDA",
            device_id=0,
            excluded_window_handles_provider=lambda: (0x1234,),
            on_window_exclusion_fallback=fallback.append,
        )
    )
    monkeypatch.setattr(
        windows_capture_event,
        "_capture_adapter_luid",
        lambda *_args, **_kwargs: (0x1234, None),
    )

    def on_error(exc):
        errors.append(exc)
        shutdown_event.set()

    runner.run(
        shutdown_event=shutdown_event,
        on_frame=lambda _frame: None,
        on_error=on_error,
    )

    assert "excluded_window_handles" not in module.WindowsCapture.last_instance.kwargs
    assert fallback and "no session-exclusion method" in fallback[0]
    assert errors


def test_empty_exclusion_handles_fall_back_to_global_exclusion(monkeypatch):
    # A session opened with an empty exclusion list still reports a
    # configuration iteration but excludes nothing, which silently restores the
    # same-monitor feedback loop. It must fall back instead of starting it.
    module = _install_capture_module(monkeypatch, "wc_cuda")
    module.WindowsCapture.set_excluded_windows = lambda self, handles: 2
    monkeypatch.setattr(windows_capture_event, "_setup_dpi_awareness", lambda: None)
    monkeypatch.setattr(
        windows_capture_event.WindowsCaptureEventRunner,
        "_start_keyboard_worker",
        lambda self, event: None,
    )
    monkeypatch.setattr(
        windows_capture_event,
        "_capture_adapter_luid",
        lambda *_args, **_kwargs: (0x1234, None),
    )
    fallback = []
    shutdown_event = threading.Event()
    runner = windows_capture_event.WindowsCaptureEventRunner(
        CaptureConfig(
            os_name="Windows",
            capture_tool="WindowsCaptureCUDA",
            device_id=0,
            excluded_window_handles_provider=lambda: (),
            on_window_exclusion_fallback=fallback.append,
        )
    )

    def on_error(_exc):
        shutdown_event.set()

    runner.run(
        shutdown_event=shutdown_event,
        on_frame=lambda _frame: None,
        on_error=on_error,
    )

    assert "excluded_window_handles" not in module.WindowsCapture.last_instance.kwargs
    assert fallback and "no output window handles" in fallback[0]


def test_initial_exclusion_iteration_gates_stale_frames(monkeypatch, capsys):
    module = _install_capture_module(monkeypatch, "wc_cuda")
    monkeypatch.setenv("D2S_WGC_SOFTWARE_THROTTLE", "0")
    module.WindowsCapture.set_excluded_windows = lambda self, handles: 2
    module.WindowsCapture.window_exclusion_available = True
    module.WindowsCapture.window_exclusion_configuration_iteration = 2
    monkeypatch.setattr(windows_capture_event, "_setup_dpi_awareness", lambda: None)
    monkeypatch.setattr(
        windows_capture_event.WindowsCaptureEventRunner,
        "_start_keyboard_worker",
        lambda self, event: None,
    )
    monkeypatch.setattr(
        windows_capture_event,
        "_capture_adapter_luid",
        lambda *_args, **_kwargs: (0x1234, None),
    )

    ready = []
    received = []
    shutdown_event = threading.Event()
    runner = windows_capture_event.WindowsCaptureEventRunner(
        CaptureConfig(
            os_name="Windows",
            capture_tool="WindowsCaptureCUDA",
            capture_mode="Monitor",
            monitor_index=1,
            excluded_window_handles_provider=lambda: (0x1234,),
            on_window_exclusion_ready=lambda: ready.append(True),
        )
    )
    runner.run(
        shutdown_event=shutdown_event,
        on_frame=received.append,
        on_error=lambda _exc: shutdown_event.set(),
    )

    capture = module.WindowsCapture.last_instance
    handler = capture.handlers[0]
    shutdown_event.clear()

    stale = FakeFrame(CloneBuffer())
    stale.configuration_iteration = 1
    handler(stale, FakeControl())
    assert received == []

    active = FakeFrame(CloneBuffer())
    active.configuration_iteration = 2
    handler(active, FakeControl())

    assert len(received) == 1
    assert ready == [True]
    assert "configuration 2 (frame 2)" in capsys.readouterr().out


@pytest.mark.parametrize("ending", ["shutdown", "hard_idle", "closed", "error"])
def test_standard_capture_uses_free_threaded_lifecycle(monkeypatch, ending):
    module = _install_capture_module(monkeypatch, "windows_capture")
    shutdown = threading.Event()
    calls = []
    state = {"finished": False, "hard_idle": False}

    def stop():
        calls.append("stop")
        state["finished"] = True
        shutdown.set()

    def wait():
        calls.append("wait")
        if ending == "error":
            raise RuntimeError("capture worker failed")

    control = SimpleNamespace(is_finished=lambda: state["finished"], stop=stop, wait=wait)

    def start_free_threaded(_capture):
        calls.append("start_free_threaded")
        if ending == "shutdown":
            shutdown.set()
        elif ending == "hard_idle":
            state["hard_idle"] = True
        else:
            state["finished"] = True
            if ending == "closed":
                shutdown.set()
        return control

    monkeypatch.setattr(module.WindowsCapture, "start_free_threaded", start_free_threaded, raising=False)
    monkeypatch.setattr(windows_capture_event, "_setup_dpi_awareness", lambda: None)
    monkeypatch.setattr(windows_capture_event, "_capture_monitor_rect", lambda _: None)
    monkeypatch.setattr(windows_capture_event.WindowsCaptureEventRunner, "_start_keyboard_worker", lambda *args: None)
    monkeypatch.setattr(
        windows_capture_event.WindowsCaptureEventRunner,
        "_start_static_replay_worker",
        lambda *args, **kw: None,
    )
    monkeypatch.setattr(windows_capture_event.time, "sleep", lambda _: None)
    errors = []
    sessions = []

    def on_error(exc):
        errors.append(str(exc))
        shutdown.set()

    runner = windows_capture_event.WindowsCaptureEventRunner(CaptureConfig(capture_tool="WindowsCapture"))
    runner.run(
        shutdown_event=shutdown,
        on_frame=lambda _: None,
        on_error=on_error,
        is_hard_idle=lambda: state["hard_idle"],
        on_session_update=lambda session, handle: sessions.append((session, handle)),
    )

    assert calls == ["start_free_threaded", "stop" if ending in {"shutdown", "hard_idle"} else "wait"]
    assert errors == (["capture worker failed"] if ending == "error" else [])
    assert sessions[-1] == (None, None)
    assert runner.session is None and runner.control is None
