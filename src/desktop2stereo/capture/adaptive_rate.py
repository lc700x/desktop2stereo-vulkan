from __future__ import annotations

import math
import threading
import time


def adaptive_capture_enabled_for_mode(run_mode: str, target_fps: int) -> bool:
    mode = str(run_mode or "").strip().lower()
    return int(target_fps) <= 0 and mode in {
        "local viewer",
        "viewer",
        "openxr",
        "openxr link",
        "moonlight streamer",
        "rtmp streamer",
        "nvidia gpu streamer",
    }


class AdaptiveCaptureRate:
    """Keep capture slightly ahead of sustained SBS output capacity."""

    def __init__(
        self,
        base_fps: int,
        *,
        enabled: bool,
        evaluation_interval_s: float = 15.0,
        activity_guard_enabled: bool = False,
        minimum_sample_frames: int = 60,
    ) -> None:
        self.base_fps = max(1, int(base_fps))
        self.enabled = bool(enabled and self.base_fps > 24)
        self.evaluation_interval_s = max(1.0, float(evaluation_interval_s))
        self.activity_guard_enabled = bool(activity_guard_enabled)
        self.minimum_sample_frames = max(1, int(minimum_sample_frames))
        self._target_fps = self.base_fps
        self._calibration_limit: int | None = None
        self._stream_probe_active = False
        self._window_started: float | None = None
        self._window_samples: list[float] = []
        self._lock = threading.Lock()

    def current_fps(self) -> int:
        with self._lock:
            return int(self._target_fps)

    def begin_stream_probe(self, requested_fps: int, *, headroom: int = 5) -> int:
        """Temporarily capture above the requested network output rate."""
        with self._lock:
            requested = max(1, min(240, int(requested_fps)))
            self._stream_probe_active = True
            self._target_fps = min(240, requested + max(0, int(headroom)))
            self._window_started = None
            self._window_samples.clear()
            return int(self._target_fps)

    def finish_stream_probe(self, selected_fps: int) -> int:
        """Restore manual capture or retain +5 FPS headroom in auto mode."""
        with self._lock:
            self._stream_probe_active = False
            if self.enabled:
                self._target_fps = min(
                    self.base_fps,
                    max(1, int(selected_fps)) + 5,
                )
            else:
                self._target_fps = self.base_fps
            return int(self._target_fps)

    def set_calibration_limit(self, fps: int | None) -> int:
        with self._lock:
            self._calibration_limit = (
                None if fps is None else max(1, min(self.base_fps, int(fps)))
            )
            if self._calibration_limit is not None:
                self._target_fps = self._calibration_limit
            return int(self._target_fps)

    def observe_sbs_fps(
        self,
        sbs_fps: float,
        *,
        capture_fps: float | None = None,
        frame_count: int | None = None,
        now: float | None = None,
    ) -> int:
        try:
            measured = max(0.0, float(sbs_fps))
        except (TypeError, ValueError):
            return self.current_fps()
        timestamp = time.monotonic() if now is None else float(now)
        with self._lock:
            if self._stream_probe_active:
                return int(self._target_fps)
            if not self.enabled or measured <= 0.0:
                return int(self._target_fps)
            if self.activity_guard_enabled:
                recovery_floor = max(1.0, self._target_fps * 0.25)
                if (
                    self._target_fps < self.base_fps
                    and capture_fps is not None
                    and capture_fps < recovery_floor
                ):
                    self._target_fps = self.base_fps
                    self._window_started = timestamp
                    self._window_samples.clear()
                    return int(self._target_fps)
            if frame_count is not None and int(frame_count) < self.minimum_sample_frames:
                return int(self._target_fps)
            if self._window_started is None:
                self._window_started = timestamp
                self._window_samples = [measured]
                return int(self._target_fps)
            self._window_samples.append(measured)
            if timestamp - self._window_started < self.evaluation_interval_s:
                return int(self._target_fps)
            peak_fps = max(self._window_samples)
            ceiling = self.base_fps
            if self._calibration_limit is not None:
                ceiling = min(ceiling, self._calibration_limit)
            self._target_fps = min(
                ceiling,
                max(1, int(math.ceil(peak_fps + 5.0))),
            )
            self._window_started = timestamp
            self._window_samples.clear()
            return int(self._target_fps)
