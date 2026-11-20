"""Measured media-rate selection; source time never changes with the chosen divisor."""
from __future__ import annotations

import math
from collections import deque
from fractions import Fraction

_RATE_EPSILON = 1e-6


def stable_media_cadence_candidates(
    source_fps: float,
    display_fps: float,
    max_fps: float,
    *,
    min_fps: float = 12.0,
    include_fps: float | None = None,
    align_display_ticks: bool = True,
) -> list[float]:
    """Return stable output rates, highest first.

    A decimated high-rate source should update on an integral number of
    compositor ticks. For example, 72 Hz with a 40 FPS ceiling yields 36 FPS
    (two ticks per frame), then 24 FPS (three ticks), rather than 30 FPS whose
    2.4-tick cadence alternates between two and three ticks.

    A source that already fits below the display rate is retained at its native
    rate, even when it does not divide the display period exactly.
    """
    try:
        source = float(source_fps)
        display = float(display_fps)
        ceiling = min(source, float(max_fps))
        minimum = float(min_fps)
    except (TypeError, ValueError):
        return []
    if not all(math.isfinite(value) and value > 0.0 for value in (source, display, ceiling, minimum)):
        return []

    candidates: set[float] = set()
    if source <= display + _RATE_EPSILON and source <= ceiling + _RATE_EPSILON:
        candidates.add(source)
    if include_fps is not None:
        try:
            current = float(include_fps)
        except (TypeError, ValueError):
            current = 0.0
        ticks = display / current if current > 0.0 else 0.0
        if (
            math.isfinite(ticks)
            and current <= ceiling + _RATE_EPSILON
            and (
                not align_display_ticks
                or abs(ticks - round(ticks)) <= 0.02
            )
        ):
            candidates.add(round(current, 9))
    if align_display_ticks:
        max_ticks = max(1, int(math.floor(display / minimum + _RATE_EPSILON)))
        for ticks in range(1, max_ticks + 1):
            candidate = display / float(ticks)
            if candidate > ceiling + _RATE_EPSILON or candidate < minimum - _RATE_EPSILON:
                continue
            candidates.add(round(candidate, 9))
    else:
        max_step = max(1, int(math.ceil(source / minimum - _RATE_EPSILON)))
        for step in range(1, max_step + 1):
            candidate = source / float(step)
            if candidate > ceiling + _RATE_EPSILON or candidate < minimum - _RATE_EPSILON:
                continue
            candidates.add(round(candidate, 9))
    return sorted(candidates, reverse=True)


def display_aligned_media_fps(
    source_fps: float,
    display_fps: float,
    max_fps: float,
    *,
    min_fps: float = 12.0,
) -> float | None:
    """Return the highest stable source/output rate under the ceiling."""
    candidates = stable_media_cadence_candidates(
        source_fps, display_fps, max_fps, min_fps=min_fps
    )
    return candidates[0] if candidates else None


class SourceFrameSchedule:
    """Deterministic integer source-frame advances for a rational output rate."""

    def __init__(self, source_fps: float, playback_fps: float, *, max_denominator: int = 240):
        self.source_fps = max(1.0, float(source_fps or 0.0))
        self.playback_fps = max(1.0, float(playback_fps or 0.0))
        ratio = Fraction(self.source_fps / self.playback_fps).limit_denominator(max_denominator)
        self._ratio = max(Fraction(1, 1), ratio)
        self._remainder = 0

    @property
    def rates(self) -> tuple[float, float]:
        return self.source_fps, self.playback_fps

    def reset(self) -> None:
        self._remainder = 0

    def next_advance(self) -> int:
        self._remainder += self._ratio.numerator
        advance, self._remainder = divmod(self._remainder, self._ratio.denominator)
        return max(1, int(advance))

    def ffmpeg_select_expression(self) -> str:
        """Return an FFmpeg select expression for this same source schedule."""
        numerator = self._ratio.numerator
        denominator = self._ratio.denominator
        if numerator <= denominator:
            return ""
        if numerator % denominator == 0:
            return f"not(mod(n\\,{numerator // denominator}))"
        remainder = 0
        selected = []
        offset = 0
        for _ in range(denominator):
            selected.append(offset)
            remainder += numerator
            advance, remainder = divmod(remainder, denominator)
            offset += max(1, int(advance))
        return "+".join(
            f"eq(mod(n\\,{numerator})\\,{position % numerator})"
            for position in selected
        )


def _candidate_index(candidates: list[float], rate: float) -> int | None:
    for index, candidate in enumerate(candidates):
        if abs(candidate - float(rate)) <= 0.01:
            return index
    return None


def _candidate_at_or_below(candidates: list[float], ceiling: float) -> float | None:
    for candidate in candidates:
        if candidate <= float(ceiling) + _RATE_EPSILON:
            return candidate
    return None


class MediaCadenceController:
    def __init__(self, source_fps: float, ceiling_fps: float):
        self.source_fps = float(source_fps)
        self.ceiling_fps = min(self.source_fps, float(ceiling_fps))
        self.samples = deque(maxlen=240)
        self._last_decision_at = 0.0
        self._last_change_at = 0.0
        self._recovery_since = None
        self._failed_until: dict[float, float] = {}
        self._failure_counts: dict[float, int] = {}
        self.reason = "warming_up"

    def reset_window(self) -> None:
        self.samples.clear()
        self._recovery_since = None

    def observe(
        self, *, pts: float, at: float, playback_fps: float,
        service_ms: float, decode_ms: float = 0.0, display_fps: float = 0.0,
        buffer_depth: int = 0, lateness_ms: float = 0.0,
        display_vsync: bool = True,
    ) -> float | None:
        if not all(math.isfinite(value) for value in (pts, at, playback_fps, service_ms, decode_ms, display_fps, lateness_ms)):
            return None
        if playback_fps <= 0.0 or service_ms <= 0.0:
            return None
        if self.samples and (pts <= self.samples[-1][1] or at <= self.samples[-1][0]):
            return None
        if self.samples and at - self.samples[-1][0] > max(0.5, 4.0 / playback_fps):
            self.reset_window()
        # A decoder fetching a source batch can wait for many originals. A
        # nonempty look-ahead queue proves that this is not the critical path;
        # treating its batch p95 as a per-output cost creates a downshift spiral.
        critical_ms = max(service_ms, decode_ms) if buffer_depth <= 1 else service_ms
        self.samples.append((at, pts, critical_ms, max(0.0, lateness_ms)))
        elapsed = at - self.samples[0][0]
        if len(self.samples) < 16 or elapsed < 1.0 or at - self._last_decision_at < 1.0:
            return None
        self._last_decision_at = at
        costs = sorted(sample[2] for sample in self.samples)
        p95_ms = costs[math.ceil(0.95 * len(costs)) - 1]
        # Decode and inference overlap. Use their measured critical-path cost,
        # not paced FPS, so lighter scenes can recover a higher rate.
        # Treat p95 as a candidate-rate estimate, then verify that candidate
        # against delivered cadence and missed display ticks. A second fixed
        # 5% penalty here compounded with the recovery gate and stranded rates
        # that already fit the measured service budget.
        capacity = min(self.ceiling_fps, 1000.0 / p95_ms)
        if display_fps > 0.0:
            capacity = min(capacity, display_fps)
        speed = (pts - self.samples[0][1]) / elapsed
        delivered_fps = (len(self.samples) - 1) / elapsed
        late = sorted(sample[3] for sample in self.samples)
        late_p95_ms = late[math.ceil(0.95 * len(late)) - 1]
        intervals = sorted(
            (self.samples[index][0] - self.samples[index - 1][0]) * 1000.0
            for index in range(1, len(self.samples))
        )
        interval_p95_ms = intervals[math.ceil(0.95 * len(intervals)) - 1]
        target_interval_ms = 1000.0 / playback_fps
        source_divisor = self.source_fps / playback_fps
        display_divisor = display_fps / playback_fps if display_fps > 0.0 else 0.0
        # A source that already fits inside the display refresh may use its
        # native/integer source divisor even when the display needs pulldown:
        # 30 FPS on 72 Hz is the common 2/3-tick pattern, not a missed frame.
        # For a higher-rate source (for example 120 -> 30 on 72 Hz), retain the
        # display-aligned 24 FPS choice unless the measured timer itself is
        # stable; this avoids turning decimated source motion into a 2/3-tick
        # pattern merely to gain nominal FPS.
        display_tick_aligned = bool(display_fps > 0.0 and display_vsync)
        structural_pulldown = (
            bool(display_fps > 0.0)
            and display_vsync
            and self.source_fps <= display_fps + _RATE_EPSILON
            and abs(source_divisor - round(source_divisor)) <= _RATE_EPSILON
            and abs(display_divisor - round(display_divisor)) > _RATE_EPSILON
        )
        # Only intervals beyond the expected ceil compositor tick are real
        # cadence pressure. This catches a 36 FPS stream that slips from its
        # expected 2 ticks to 3 ticks.
        if display_tick_aligned:
            expected_ticks = max(1, math.ceil(display_fps / playback_fps - 1e-6))
            structural_interval_limit_ms = (1000.0 / display_fps) * expected_ticks * 1.10
            outlier_ratio = sum(
                interval > structural_interval_limit_ms for interval in intervals
            ) / max(1, len(intervals))
            interval_jitter = outlier_ratio > 0.40
        else:
            interval_jitter = interval_p95_ms > target_interval_ms * 1.15
        over_budget = (speed < 0.98 or delivered_fps < playback_fps * 0.98
                       or late_p95_ms > 0.5 * target_interval_ms
                       or interval_jitter)
        if over_budget:
            capacity = min(capacity, delivered_fps * 0.92)
            self.reason = "presentation_over_budget"
            if interval_jitter:
                self.reason = "presentation_interval_jitter"
        else:
            self.reason = "measured_service_budget"
        display_candidates = (
            stable_media_cadence_candidates(
                self.source_fps,
                display_fps,
                min(self.ceiling_fps, capacity),
                min_fps=6.0,
                include_fps=playback_fps,
                align_display_ticks=display_vsync,
            )
            if display_fps > 0.0
            else []
        )
        # Do not force a rate change merely because a rare decode batch made
        # the service estimate lower than a proven 1.0x presentation cadence.
        # The alignment rule is actionable only when the current rate itself
        # fits the measured capacity.
        misaligned = (
            bool(display_vsync)
            and bool(display_candidates)
            and _candidate_index(display_candidates, playback_fps) is None
            and not structural_pulldown
            and (
                playback_fps <= capacity + _RATE_EPSILON
                or playback_fps < 12.0
            )
        )
        step = max(1, math.ceil(self.source_fps / max(1.0, capacity) - 1e-6))
        target = self.source_fps / step
        current_step = max(1, round(self.source_fps / playback_fps))
        # Once source time is stable, lock the proven rate. A long decode
        # batch or one high p95 must not lower a rate that already plays 1.0x.
        if over_budget or misaligned:
            self._recovery_since = None
            if misaligned:
                self.reason = "display_cadence_misaligned"
            if display_candidates:
                limit = capacity
                if over_budget:
                    limit = min(limit, delivered_fps * 0.92)
                aligned_target = _candidate_at_or_below(display_candidates, limit)
                if aligned_target is None:
                    self.reason = "floor_over_budget"
                    return None
                target = aligned_target
                if abs(target - playback_fps) <= 0.01:
                    if not misaligned:
                        self.reason = "floor_over_budget"
                    return None
            elif step <= current_step:
                return None
            if self._last_change_at and at - self._last_change_at < 2.0:
                return None
        else:
            # After a temporary large overload, one integer divisor at a
            # time can trap playback near 10 FPS for minutes. Recover a
            # bounded 50% rate step, still with measured 25% service headroom.
            # Sparse source divisors (for example 30 -> 60 for a 60 FPS
            # source) have no intermediate rung; permit that smallest step
            # only when it also passes the full headroom/recovery window.
            if display_candidates:
                current_index = _candidate_index(display_candidates, playback_fps)
                if current_index is None:
                    if (
                        structural_pulldown
                        and self.source_fps <= self.ceiling_fps + _RATE_EPSILON
                    ):
                        next_fps = self.source_fps
                        next_step = 1
                    else:
                        self._recovery_since = None
                        self.reason = "locked_stable"
                        return None
                elif current_index == 0:
                    self._recovery_since = None
                    self.reason = "locked_stable"
                    return None
                else:
                    next_fps = display_candidates[current_index - 1]
                    next_step = max(1, math.ceil(self.source_fps / max(1.0, next_fps) - 1e-6))
            else:
                smallest_next = self.source_fps / max(1, current_step - 1)
                recovery_ceiling = min(self.ceiling_fps, max(playback_fps * 1.5, smallest_next), 750.0 / p95_ms)
                if display_fps > 0.0:
                    recovery_ceiling = min(recovery_ceiling, display_fps)
                next_step = max(1, math.ceil(self.source_fps / max(1.0, recovery_ceiling) - 1e-6))
                next_fps = self.source_fps / next_step
            if at < self._failed_until.get(round(next_fps, 3), 0.0):
                self._recovery_since = None
                self.reason = "recovery_backoff"
                return None
            has_headroom = (
                next_fps > playback_fps + _RATE_EPSILON
                and (bool(display_candidates) or next_step < current_step)
                and next_fps <= self.ceiling_fps + 1e-6
                and (display_fps <= 0.0 or next_fps <= display_fps + 1e-6)
                and p95_ms <= (1000.0 / next_fps)
                and buffer_depth > 1
            )
            if not has_headroom:
                self._recovery_since = None
                self.reason = "locked_stable"
                return None
            if self._recovery_since is None:
                self._recovery_since = at
            if at - self._recovery_since < 15.0 or at - self._last_change_at < 15.0:
                return None
            target = next_fps
            self.reason = "sustained_headroom"
        self._last_change_at = at
        if over_budget and target < playback_fps - _RATE_EPSILON:
            failed_rate = round(float(playback_fps), 3)
            failures = self._failure_counts.get(failed_rate, 0) + 1
            self._failure_counts[failed_rate] = failures
            self._failed_until[failed_rate] = at + min(300.0, 30.0 * (2 ** (failures - 1)))
        self.reset_window()
        return target
