from __future__ import annotations

import time
from typing import Callable

from .types import CaptureConfig, FrameCopyMode, capture_frame_from_raw


class PollingCaptureRunner:
    def __init__(self, config: CaptureConfig, source_factory: Callable[[], object]):
        self.config = config
        self._source_factory = source_factory
        self._source = None
        self._frame_id = 0

    def _target_fps(self) -> int:
        provider = self.config.fps_provider
        if callable(provider):
            try:
                return max(1, min(240, int(provider())))
            except (TypeError, ValueError, RuntimeError):
                pass
        return max(1, min(240, int(self.config.fps)))

    @property
    def source(self):
        return self._source

    def stop(self):
        if self._source is not None and hasattr(self._source, "stop"):
            self._source.stop()

    def run(
        self,
        *,
        shutdown_event,
        on_frame,
        on_error=None,
        on_closed=None,
        is_paused=None,
        is_hard_idle=None,
        on_paused=None,
        on_session_update=None,
        on_tick=None,
    ):
        self._source = self._source_factory()
        if on_session_update is not None:
            on_session_update(self._source, None)
        try:
            while not shutdown_event.is_set():
                try:
                    if on_tick is not None:
                        on_tick()
                    if is_hard_idle is not None and is_hard_idle():
                        if on_paused is not None:
                            on_paused("hard_idle")
                        time.sleep(0.1)
                        continue
                    if is_paused is not None and is_paused():
                        if on_paused is not None:
                            on_paused("paused")
                        time.sleep(0.05)
                        continue

                    capture_start_time = time.perf_counter()
                    native_grab = getattr(self._source, "grab_native_zero_copy", None)
                    if callable(native_grab):
                        # Native-only sources must wait and take under one
                        # producer lock; separate grab/take calls can miss a
                        # callback edge and halve the effective frame rate.
                        zero_copy, size = native_grab(
                            timeout=1.0 / self._target_fps()
                        )
                        frame_raw = None
                    else:
                        frame_raw, size = self._source.grab()
                        zero_copy = None
                    native_depth_profile = None
                    pop_native_depth = getattr(
                        self._source, "pop_native_depth_profile", None
                    )
                    if callable(pop_native_depth):
                        native_depth_profile = pop_native_depth()
                    if native_grab is None:
                        take_zero_copy = getattr(self._source, "take_latest_zero_copy", None)
                        if callable(take_zero_copy):
                            zero_copy = take_zero_copy()
                    if shutdown_event.is_set():
                        break
                    # Native-only ScreenCaptureKit intentionally has no CPU
                    # frame. Do not publish an empty poll between IOSurface
                    # callbacks; the next callback owns the real frame.
                    if frame_raw is None and zero_copy is None:
                        continue
                    on_frame(
                        capture_frame_from_raw(
                            frame_raw,
                            size,
                            capture_start_time,
                            config=self.config,
                            copy_mode=FrameCopyMode.COPY,
                            original_format=str(getattr(self._source, "frame_format", "") or ""),
                            capture_size=(
                                (int(size[0]), int(size[1]))
                                if isinstance(size, (tuple, list)) and len(size) >= 2
                                else None
                            ),
                            sck_zero_copy=zero_copy,
                            metadata={
                                "backend": type(self._source).__name__,
                                "capture_frame_status": getattr(
                                    zero_copy, "frame_status", None
                                ),
                                "zero_copy": zero_copy is not None,
                                "zero_copy_ready": zero_copy is not None,
                                "gpu_to_cpu": False if zero_copy is not None else None,
                                "gpu_copy_count": 0 if zero_copy is not None else None,
                                "resource_kind": (
                                    "iosurface_metal" if zero_copy is not None else None
                                ),
                                "resource_format": (
                                    "BGRA8Unorm" if zero_copy is not None else None
                                ),
                                "capture_frame_id": self._frame_id,
                                **({
                                    "native_depth_profile": native_depth_profile,
                                    "native_depth_backend": "openvino_d3d11_remote",
                                } if native_depth_profile is not None else {}),
                            },
                        )
                    )
                    self._frame_id += 1
                except Exception as exc:
                    if on_error is not None:
                        on_error(exc)
                    else:
                        raise
        finally:
            if on_session_update is not None:
                on_session_update(None, None)
            if on_closed is not None:
                on_closed()
