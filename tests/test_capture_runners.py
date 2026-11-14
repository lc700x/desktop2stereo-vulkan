from __future__ import annotations

import threading

import numpy as np

from capture.runners import PollingCaptureRunner
from capture.types import CaptureConfig, CapturedFrame, FrameCopyMode


class FakePollingSource:
    def __init__(self):
        self.stopped = False

    def grab(self):
        return np.zeros((4, 8, 3), dtype=np.uint8), (8, 4)

    def stop(self):
        self.stopped = True


class FakeNativePollingSource:
    def __init__(self):
        self.grab_called = False
        self.resource = object()

    def grab(self):
        self.grab_called = True
        raise AssertionError("native-only polling must not call CPU grab")

    def grab_native_zero_copy(self, timeout=0.0):
        assert timeout > 0.0
        return self.resource, (8, 4)

    def stop(self):
        pass


class FakeControlledPollingSource:
    def __init__(self):
        self.control_polls = 0
        self.frames = 0

    def poll_control(self):
        self.control_polls += 1

    def grab(self):
        self.frames += 1
        return np.zeros((4, 8, 3), dtype=np.uint8), (8, 4)

    def stop(self):
        pass


def test_polling_capture_runner_marks_explicit_non_zero_copy_metadata():
    source = FakePollingSource()
    shutdown = threading.Event()
    frames = []
    runner = PollingCaptureRunner(
        CaptureConfig(
            output_resolution=(8, 4),
            capture_tool="DXCamera",
            capture_mode="Monitor",
            monitor_index=1,
            os_name="Windows",
        ),
        lambda: source,
    )

    def on_frame(frame):
        frames.append(frame)
        shutdown.set()

    runner.run(shutdown_event=shutdown, on_frame=on_frame)

    assert len(frames) == 1
    captured = frames[0]
    assert isinstance(captured, CapturedFrame)
    assert captured.copy_mode is FrameCopyMode.COPY
    assert captured.metadata["backend"] == "FakePollingSource"
    assert captured.metadata["zero_copy"] is False
    assert captured.capture_tool == "DXCamera"
    assert captured.capture_size == (8, 4)
    assert captured.frame_raw_device == "cpu"
    assert captured.frame_raw_dtype == "uint8"


def test_polling_capture_runner_uses_atomic_native_zero_copy_poll():
    source = FakeNativePollingSource()
    shutdown = threading.Event()
    frames = []
    runner = PollingCaptureRunner(
        CaptureConfig(output_resolution=(8, 4), fps=60, os_name="Darwin"),
        lambda: source,
    )

    def on_frame(frame):
        frames.append(frame)
        shutdown.set()

    runner.run(shutdown_event=shutdown, on_frame=on_frame)

    assert not source.grab_called
    assert len(frames) == 1
    assert frames[0].frame is None
    assert frames[0].sck_zero_copy is source.resource
    assert frames[0].metadata["zero_copy"] is True


def test_polling_capture_runner_polls_media_controls_while_hard_idle():
    source = FakeControlledPollingSource()
    shutdown = threading.Event()
    idle_checks = iter((True, False))
    runner = PollingCaptureRunner(
        CaptureConfig(output_resolution=(8, 4), fps=60, os_name="Windows"),
        lambda: source,
    )

    def on_frame(_frame):
        shutdown.set()

    runner.run(
        shutdown_event=shutdown,
        on_frame=on_frame,
        is_hard_idle=lambda: next(idle_checks),
    )

    assert source.control_polls == 1
    assert source.frames == 1
