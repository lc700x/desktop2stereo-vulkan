from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

from . import CaptureConfig, create_capture_runner
from .types import CapturedFrame, ensure_captured_frame


@dataclass(frozen=True)
class CaptureSessionCallbacks:
    clear_raw_queue: Callable[[], None]
    inc_source_stat: Callable[..., None]
    inc_breakdown: Callable[..., None]
    put_raw_latest: Callable[[Any], None]
    is_shutdown: Callable[[], bool]
    is_paused: Callable[[], bool]
    is_hard_idle: Callable[[], bool]
    on_session_update: Callable[[Any, Any], None]
    on_tick: Callable[[], None]
    abort: Callable[[], None] | None = None



def _frame_size_text(frame) -> str:
    shape = tuple(getattr(frame, "shape", ()))
    if len(shape) >= 2:
        return f"{int(shape[1])}x{int(shape[0])}"
    width = getattr(frame, "width", None)
    height = getattr(frame, "height", None)
    if width is not None and height is not None:
        return f"{int(width)}x{int(height)}"
    return "unknown"
class CaptureSessionLoop:
    def __init__(self, config: CaptureConfig, callbacks: CaptureSessionCallbacks):
        self.config = config
        self.callbacks = callbacks
        self._logged_frame_shape = False

    def run(self, shutdown_event) -> None:
        runner = create_capture_runner(self.config)
        try:
            runner.run(
                shutdown_event=shutdown_event,
                on_frame=self._frame_arrived,
                on_error=self._capture_error,
                on_closed=self._capture_closed,
                is_paused=self.callbacks.is_paused,
                is_hard_idle=self.callbacks.is_hard_idle,
                on_paused=self._capture_paused,
                on_session_update=self.callbacks.on_session_update,
                on_tick=self.callbacks.on_tick,
            )
        except Exception as exc:
            self._capture_error(exc)
            if not self.config.gpu_required:
                raise

    def _capture_paused(self, reason: str) -> None:
        self.callbacks.clear_raw_queue()
        if reason == "paused":
            self.callbacks.inc_source_stat("capture_dropped_paused")

    def _frame_arrived(self, frame_or_capture, size=None, capture_start_time: float | None = None) -> None:
        if isinstance(frame_or_capture, CapturedFrame):
            captured_frame = frame_or_capture
        else:
            if capture_start_time is None:
                raise ValueError("capture_start_time is required for legacy frame callbacks")
            captured_frame = ensure_captured_frame(
                (frame_or_capture, size, capture_start_time),
                config=self.config,
            )
        if not self._logged_frame_shape and os.environ.get('D2S_DEBUG', '0') in ('1', 'true', 'yes', 'on'):
            self._logged_frame_shape = True
            print(
                f"[capture_loop] frame raw={_frame_size_text(captured_frame.frame)} "
                f"target={captured_frame.target_height}",
                flush=True,
            )
        self.callbacks.inc_source_stat("capture_frames", last_capture_ts=captured_frame.timestamp)
        self.callbacks.inc_breakdown("capture")
        if self.callbacks.is_shutdown():
            return
        if self.callbacks.put_raw_latest(captured_frame):
            self.callbacks.inc_source_stat("raw_overwritten")
            self.callbacks.inc_breakdown("raw_overwritten")
        self.callbacks.inc_source_stat("raw_put")

    def _capture_error(self, exc: Exception) -> None:
        self.callbacks.inc_source_stat(
            "capture_errors",
            last_error=f"capture_loop {type(exc).__name__}: {exc}",
        )
        print(f"[capture_loop] Capture session error: {type(exc).__name__}: {exc}", flush=True)
        if self.config.gpu_required and self.callbacks.abort is not None:
            self.callbacks.abort()

    def _capture_closed(self) -> None:
        if not self.callbacks.is_shutdown():
            print("[capture_loop] Capture session closed")
            if self.config.gpu_required and self.callbacks.abort is not None:
                self.callbacks.abort()
