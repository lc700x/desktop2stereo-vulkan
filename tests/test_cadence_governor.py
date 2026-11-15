from __future__ import annotations

from stereo_runtime.cadence_governor import StableCadenceGovernor


def test_single_over_budget_frame_does_not_change_cadence():
    governor = StableCadenceGovernor(72)

    decision = governor.observe(20.0)

    assert decision.proposed_hz == 72
    assert decision.changed is False
    assert decision.reason == "steady"


def test_sustained_over_budget_steps_down_one_ladder_rung():
    governor = StableCadenceGovernor(72)

    decisions = [governor.observe(20.0) for _ in range(3)]

    assert decisions[-1].proposed_hz == 60
    assert decisions[-1].active_hz == 72
    assert decisions[-1].changed is True
    assert decisions[-1].reason == "sustained_over_budget"


def test_recovery_requires_sustained_headroom():
    governor = StableCadenceGovernor(72)
    for _ in range(3):
        governor.observe(20.0)
    assert governor.selected_hz == 60

    decisions = [governor.observe(8.0) for _ in range(5)]

    assert decisions[-2].proposed_hz == 60
    assert decisions[-1].proposed_hz == 72
    assert decisions[-1].reason == "sustained_headroom"


def test_guards_block_upward_step_for_queue_or_depth_age():
    queue_governor = StableCadenceGovernor(72)
    for _ in range(3):
        queue_governor.observe(20.0)
    for _ in range(8):
        decision = queue_governor.observe(8.0, queue_depth=3)
    assert decision.proposed_hz == 60
    assert decision.reason == "queue_depth_guard"

    depth_governor = StableCadenceGovernor(72)
    for _ in range(3):
        depth_governor.observe(20.0)
    for _ in range(8):
        decision = depth_governor.observe(8.0, depth_age_ms=150.0)
    assert decision.proposed_hz == 60
    assert decision.reason == "depth_age_guard"


def test_control_mode_steps_active_cadence_without_touching_quality_inputs():
    governor = StableCadenceGovernor(72, observe_only=False)

    for _ in range(3):
        decision = governor.observe(20.0)

    assert decision.active_hz == 60
    assert decision.proposed_hz == 60


def test_floor_reports_over_budget_without_changing_quality_inputs():
    governor = StableCadenceGovernor(72, observe_only=False)

    for _ in range(12):
        decision = governor.observe(40.0, depth_age_ms=52.0)

    assert decision.active_hz == 30
    assert decision.proposed_hz == 30
    assert decision.reason == "floor_over_budget"
    assert decision.frame_time_ms == 40.0
    assert decision.budget_ms == 1000.0 / 30.0
    assert decision.depth_age_ms == 52.0
