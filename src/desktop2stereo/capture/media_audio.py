"""Bounded PCM playback clocked by successfully presented source video PTS."""

from __future__ import annotations

import math
import threading
from statistics import median
from collections import deque
from typing import Any

import numpy as np


class MediaAudioSync:
    """Keep source sample identity while withholding audio ahead of visible video.

    The audio callback never waits for decoding or rendering. DAC timestamps
    come from PortAudio, not the time a PCM chunk was read or queued. Only the
    first presentation discards seek/startup preroll. A measured late PCM cursor
    is advanced to the visible video timestamp, never by decoder wall time;
    samples otherwise play once at the original rate (no pitch/rate conversion).
    """

    def __init__(self, *, generation: int, start_pts: float, sample_rate: int = 48000) -> None:
        self.generation = int(generation)
        self.start_pts = float(start_pts)
        self.sample_rate = int(sample_rate)
        self._condition = threading.Condition()
        self._blocks: deque[tuple[int, np.ndarray]] = deque()
        self._read_frames = 0
        self._offset = 0
        self._video: tuple[int, float, float, float] | None = None
        self._presentation_epochs: deque[float] = deque(maxlen=9)
        self._clock_epoch = 0.0
        self._aligned = False
        self._audio_pts: float | None = None
        self._dac_at = 0.0
        self._audio_end_pts = 0.0
        self._sync_error_ms: float | None = None
        self._dac_blocks: deque[tuple[float, float, int]] = deque(maxlen=64)
        self._resync_frames = 0
        self._underruns = 0
        self._device_underflows = 0
        self._held_frames = 0
        self._preroll_frames = 0
        self._closed = False
        self._source_eof = False
        self._trailing_silence_frames = 0

    def observe_video(self, generation: int, index: int, pts: float, at: float, period: float) -> None:
        if generation != self.generation or not all(math.isfinite(value) for value in (pts, at, period)):
            return
        with self._condition:
            if self._closed or (self._video is not None and index <= self._video[0]):
                return
            if self._video is not None:
                _, previous_pts, previous_at, previous_period = self._video
                if abs((at - previous_at) - (pts - previous_pts)) > 2.0 * previous_period:
                    self._presentation_epochs.clear()
            self._video = (int(index), float(pts), float(at), max(0.001, float(period)))
            self._presentation_epochs.append(float(at) - float(pts))
            # A missed display tick must not modulate audio by a whole refresh.
            # Use the measured short-window phase, not a configured delay.
            self._clock_epoch = median(self._presentation_epochs)
            self._condition.notify_all()

    def enqueue(self, samples: np.ndarray, stop: threading.Event) -> bool:
        """Backpressure only the decoder; PCM storage stays bounded to 200 ms."""
        with self._condition:
            while len(self._blocks) >= 20 and not self._closed and not stop.is_set():
                self._condition.wait(0.01)
            if self._closed or stop.is_set():
                return False
            self._blocks.append((self._read_frames, samples))
            self._read_frames += len(samples)
            return True

    @staticmethod
    def _video_pts(video: tuple[int, float, float, float], at: float) -> float:
        _, pts, presented_at, period = video
        # A stalled/paused picture is not permission to continue audio forever.
        return pts + min(period, at - presented_at)

    def render(
        self,
        output: np.ndarray,
        *,
        dac_at: float,
        volume: float,
        muted: bool,
        callback_at: float | None = None,
    ) -> None:
        output.fill(0)
        with self._condition:
            video = self._video
            if self._closed or video is None:
                self._held_frames += len(output)
                return
            callback_at = dac_at if callback_at is None else callback_at
            fresh = callback_at - video[2] <= 1.5 * video[3]

            # The DAC buffer plays in the future. Predict that future source
            # time while fresh video is arriving, rather than inserting silence
            # every time a 10 ms audio block straddles a 24 fps video tick.
            def expected(at: float) -> float:
                return at - self._clock_epoch if fresh else self._video_pts(video, at)

            target_pts = expected(dac_at)
            written = 0
            resync_target = None
            while written < len(output) and self._blocks:
                start_frame, samples = self._blocks[0]
                sample_pts = self.start_pts + (start_frame + self._offset) / self.sample_rate
                if not self._aligned:
                    discard = min(
                        len(samples) - self._offset, max(0, int(round((target_pts - sample_pts) * self.sample_rate)))
                    )
                    self._offset += discard
                    self._preroll_frames += discard
                    if self._offset == len(samples):
                        self._blocks.popleft()
                        self._offset = 0
                        continue
                    sample_pts = self.start_pts + (start_frame + self._offset) / self.sample_rate
                    self._aligned = True
                sample_dac_at = dac_at + written / self.sample_rate
                expected_pts = expected(sample_dac_at)
                ahead = sample_pts - expected_pts
                block_period = len(output) / self.sample_rate
                if self._aligned and ahead < -2.0 * block_period:
                    resync_target = expected_pts
                if resync_target is not None and sample_pts < resync_target - 0.5 / self.sample_rate:
                    # Recover a measured startup/seek/underrun discontinuity
                    # in source-sample units; never let latency keep accumulating.
                    discard = min(
                        len(samples) - self._offset, int(round((resync_target - sample_pts) * self.sample_rate))
                    )
                    self._offset += discard
                    self._resync_frames += discard
                    if self._offset == len(samples):
                        self._blocks.popleft()
                        self._offset = 0
                    continue
                if ahead > 0.5 * block_period:
                    silence = min(len(output) - written, max(1, int(math.ceil(ahead * self.sample_rate))))
                    written += silence
                    self._held_frames += silence
                    continue
                # Only send source samples belonging to the latest visible
                # video's time interval, including the DAC buffer's end.
                allowed = (
                    len(output) if fresh else max(0, int(round((video[1] + video[3] - sample_pts) * self.sample_rate)))
                )
                count = min(len(samples) - self._offset, len(output) - written, allowed)
                if count <= 0:
                    self._held_frames += len(output) - written
                    written = len(output)
                    break
                gain = 0.0 if muted else max(0.0, min(1.0, float(volume)))
                np.multiply(samples[self._offset : self._offset + count], gain, out=output[written : written + count])
                self._audio_pts = sample_pts
                self._dac_blocks.append((sample_dac_at, sample_pts, count))
                self._dac_at = sample_dac_at
                self._audio_end_pts = sample_pts + count / self.sample_rate
                self._sync_error_ms = (sample_pts - expected_pts) * 1000.0
                written += count
                self._offset += count
                if self._offset == len(samples):
                    self._blocks.popleft()
                    self._offset = 0
            if written < len(output):
                if self._source_eof:
                    self._trailing_silence_frames += len(output) - written
                else:
                    self._underruns += 1
            self._condition.notify_all()

    def note_source_eof(self) -> None:
        """A decoded audio stream can end slightly before the video timeline."""
        with self._condition:
            self._source_eof = True
            self._condition.notify_all()

    def note_device_underflow(self) -> None:
        with self._condition:
            self._device_underflows += 1

    def snapshot(self, at: float) -> dict[str, Any]:
        with self._condition:
            audio_pts = None
            for dac_at, pts, frames in reversed(self._dac_blocks):
                if dac_at <= at < dac_at + frames / self.sample_rate:
                    audio_pts = pts + at - dac_at
                    break
            video_pts = self._video_pts(self._video, at) if self._video is not None else None
            return {
                "clock": "presented_video_pts_portaudio_dac",
                "sampled_at": at,
                "video_display_at": self._video[2] if self._video is not None else None,
                "last_audio_dac_at": self._dac_at,
                "audio_pts": audio_pts,
                "video_pts": video_pts,
                "offset_ms": (audio_pts - video_pts) * 1000.0
                if audio_pts is not None and video_pts is not None
                else None,
                "callback_offset_ms": self._sync_error_ms,
                "buffered_blocks": len(self._blocks),
                "underruns": self._underruns,
                "device_underflows": self._device_underflows,
                "source_eof": self._source_eof,
                "trailing_silence_frames": self._trailing_silence_frames,
                "held_frames": self._held_frames,
                "preroll_frames": self._preroll_frames,
                "resync_frames": self._resync_frames,
            }

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._blocks.clear()
            self._condition.notify_all()
