from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CadenceDecision:
    active_hz: int
    proposed_hz: int
    reason: str
    changed: bool
    over_budget_streak: int
    headroom_streak: int


class StableCadenceGovernor:
    """Pure cadence policy; it never changes depth or synthesis parameters.

    The governor starts in observe-only mode so proposals can be reviewed before
    presentation cadence is changed.
    """

    def __init__(
        self,
        requested_hz: int,
        *,
        ladder: tuple[int, ...] = (72, 60, 45, 30),
        observe_only: bool = True,
        step_down_after: int = 3,
        step_up_after: int = 5,
        queue_limit: int = 2,
        depth_age_limit_ms: float = 100.0,
        over_budget_ratio: float = 1.0,
        recovery_headroom_ratio: float = 0.75,
    ) -> None:
        requested = max(1, int(requested_hz))
        candidates = sorted(
            {int(value) for value in ladder if int(value) > 0 and int(value) <= requested},
            reverse=True,
        )
        if not candidates or candidates[0] != requested:
            candidates.insert(0, requested)
        self.requested_hz = requested
        self.observe_only = bool(observe_only)
        self._ladder = tuple(candidates)
        self._index = 0
        self._step_down_after = max(1, int(step_down_after))
        self._step_up_after = max(1, int(step_up_after))
        self._queue_limit = max(0, int(queue_limit))
        self._depth_age_limit_ms = max(0.0, float(depth_age_limit_ms))
        self._over_budget_ratio = max(0.1, float(over_budget_ratio))
        self._recovery_headroom_ratio = max(0.1, float(recovery_headroom_ratio))
        self._over_budget_streak = 0
        self._headroom_streak = 0
        self._last_reason = "startup"
        self._history: list[CadenceDecision] = []

    @property
    def active_hz(self) -> int:
        return self.requested_hz if self.observe_only else self.selected_hz

    @property
    def selected_hz(self) -> int:
        return int(self._ladder[self._index])

    @property
    def history(self) -> tuple[CadenceDecision, ...]:
        return tuple(self._history)

    def _decision(self, *, changed: bool) -> CadenceDecision:
        return CadenceDecision(
            active_hz=self.active_hz,
            proposed_hz=self.selected_hz,
            reason=self._last_reason,
            changed=bool(changed),
            over_budget_streak=self._over_budget_streak,
            headroom_streak=self._headroom_streak,
        )

    def observe(
        self,
        frame_time_ms: float,
        *,
        queue_depth: int = 0,
        depth_age_ms: float = 0.0,
        dropped_frames: int = 0,
        repeated_frames: int = 0,
    ) -> CadenceDecision:
        del dropped_frames, repeated_frames
        try:
            frame_time_ms = max(0.0, float(frame_time_ms))
            depth_age_ms = max(0.0, float(depth_age_ms))
            queue_depth = max(0, int(queue_depth))
        except (TypeError, ValueError):
            return self._decision(changed=False)

        budget_ms = 1000.0 / max(1, self.selected_hz)
        over_budget = frame_time_ms > budget_ms * self._over_budget_ratio
        if over_budget:
            self._over_budget_streak += 1
            self._headroom_streak = 0
        else:
            self._over_budget_streak = 0

        changed = False
        if (
            self._over_budget_streak >= self._step_down_after
            and self._index + 1 < len(self._ladder)
        ):
            self._index += 1
            self._over_budget_streak = 0
            self._headroom_streak = 0
            self._last_reason = "sustained_over_budget"
            changed = True
        elif queue_depth > self._queue_limit or depth_age_ms > self._depth_age_limit_ms:
            self._headroom_streak = 0
            self._last_reason = (
                "queue_depth_guard"
                if queue_depth > self._queue_limit
                else "depth_age_guard"
            )
        elif self._index > 0:
            next_budget_ms = 1000.0 / max(1, self._ladder[self._index - 1])
            has_headroom = (
                frame_time_ms
                <= next_budget_ms * self._recovery_headroom_ratio
            )
            self._headroom_streak = self._headroom_streak + 1 if has_headroom else 0
            if self._headroom_streak >= self._step_up_after:
                self._index -= 1
                self._headroom_streak = 0
                self._last_reason = "sustained_headroom"
                changed = True
            else:
                self._last_reason = (
                    "headroom_building" if has_headroom else "steady"
                )
        else:
            self._last_reason = "steady"

        decision = self._decision(changed=changed)
        self._history.append(decision)
        if len(self._history) > 256:
            del self._history[:-256]
        return decision
