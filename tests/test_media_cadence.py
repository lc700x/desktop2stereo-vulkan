import pytest
from capture.media_cadence import (
    MediaCadenceController,
    SourceFrameSchedule,
    stable_media_cadence_candidates,
)


def drive(controller, fps, service_ms, seconds=4.0, start=100.0, speed=1.0, decode_ms=0.0,
          display=60.0, buffer_depth=8, vsync=True):
    changes = []
    for i in range(int(seconds * fps)):
        at = start + i / fps
        target = controller.observe(pts=(at - start) * speed, at=at, playback_fps=fps,
                                    service_ms=service_ms, decode_ms=decode_ms, display_fps=display,
                                    buffer_depth=buffer_depth, display_vsync=vsync)
        if target is not None:
            changes.append(target)
    return changes


def test_different_hardware_costs_choose_different_rates_not_fixed_40():
    assert drive(MediaCadenceController(120, 120), 60, 27, speed=0.5)[0] == 30.0
    assert drive(MediaCadenceController(120, 120), 60, 7, display=120) == []
    assert drive(MediaCadenceController(120, 120), 60, 60, speed=0.25)[0] == 15.0


def test_light_24fps_video_keeps_every_original_frame():
    assert drive(MediaCadenceController(24, 60), 24, 8) == []


def test_vsync_off_keeps_measured_source_cadence_without_refresh_grid_downshift():
    assert drive(MediaCadenceController(120, 60), 24, 28, vsync=False) == []
    assert drive(MediaCadenceController(120, 60), 40, 23, vsync=False) == []
    assert drive(MediaCadenceController(120, 60), 24, 28, vsync=True)[0] == 30.0


def test_decode_bottleneck_and_slow_source_time_trigger_reduction():
    assert drive(MediaCadenceController(120, 60), 60, 5, decode_ms=60,
                 buffer_depth=0, speed=0.25)[0] == 15.0
    controller = MediaCadenceController(120, 60)
    changes = [controller.observe(pts=i / 30.0, at=100.0 + i / 20.0, playback_fps=30.0, service_ms=5.0)
               for i in range(80)]
    assert next(value for value in changes if value is not None) < 20.0


def test_recovery_requires_sustained_headroom_and_moves_to_next_display_rung():
    controller = MediaCadenceController(120, 60)
    assert drive(controller, 30, 6, seconds=7.0) == []
    changes = drive(MediaCadenceController(120, 60), 30, 6, seconds=19.0)
    assert changes[0] == 60.0


def test_recovery_probes_rate_that_fits_measured_p95_service_budget():
    controller = MediaCadenceController(120, 60)
    changes = drive(controller, 20, 30.6, seconds=19.0)
    assert changes == [30.0]

    # A failed probe is detected from the presented cadence, not assumed to
    # be sustainable merely because its service time fit the frame budget.
    stable = MediaCadenceController(120, 60)
    assert drive(stable, 30, 30.6, seconds=4.0) == []
    assert stable.reason == "locked_stable"


def test_recovery_does_not_probe_rate_over_measured_service_budget():
    changes = drive(MediaCadenceController(120, 60), 20, 34.0, seconds=19.0)
    assert changes == []


def test_invalid_and_duplicate_samples_do_not_change_cadence():
    controller = MediaCadenceController(120, 60)
    for _ in range(100):
        assert controller.observe(pts=1, at=100, playback_fps=60, service_ms=5) is None
    for value in [float("nan"), float("inf"), 0.0, -1.0]:
        assert controller.observe(pts=2, at=101, playback_fps=60, service_ms=value) is None
    assert len(controller.samples) == 1


def test_pause_reset_keeps_old_costs_out_of_the_new_window():
    controller = MediaCadenceController(120, 60)
    drive(controller, 30, 5, seconds=0.5)
    controller.reset_window()
    assert not controller.samples
    assert drive(controller, 30, 5, seconds=1.0, start=120) == []


def test_nonstarving_decode_batch_wait_does_not_cause_downshift_spiral():
    controller = MediaCadenceController(120, 60)
    for i in range(480):
        at = 100.0 + i / 30.0
        assert controller.observe(pts=i / 30.0, at=at, playback_fps=30.0,
                                  service_ms=27.0, decode_ms=97.0, buffer_depth=8, display_fps=60.0) is None
    assert controller.reason == "locked_stable"


def test_starved_decode_queue_remains_part_of_sustainable_budget():
    controller = MediaCadenceController(120, 60)
    changes = [controller.observe(pts=i / 30.0, at=100.0 + i / 15.0, playback_fps=30.0,
                                  service_ms=5.0, decode_ms=60.0, buffer_depth=0, display_fps=60.0)
               for i in range(80)]
    assert next(value for value in changes if value is not None) == 12.0


def test_proven_source_speed_locks_rate_despite_rare_slow_cost_samples():
    controller = MediaCadenceController(120, 60)
    for i in range(720):
        assert controller.observe(pts=i / 24.0, at=100.0 + i / 24.0, playback_fps=24.0,
                                  service_ms=80.0 if i % 10 == 0 else 30.0,
                                  decode_ms=97.0, buffer_depth=0, display_fps=60.0) is None
    assert controller.reason == "locked_stable"


def test_skipped_source_frames_are_over_budget_even_when_source_speed_is_one():
    controller = MediaCadenceController(60, 60)
    changes = [controller.observe(pts=i / 50.0, at=100.0 + i / 50.0, playback_fps=60.0,
                                  service_ms=8.0, buffer_depth=8, display_fps=60.0)
               for i in range(180)]
    assert next(value for value in changes if value is not None) == 30.0


def test_repeated_late_and_catch_up_frames_cannot_pass_on_average_fps_only():
    controller = MediaCadenceController(120, 60)
    changes = [controller.observe(pts=i / 24.0, at=100.0 + i / 24.0, playback_fps=24.0,
                                  service_ms=30.0, buffer_depth=8, display_fps=60.0,
                                  lateness_ms=60.0 if i % 4 == 0 else 0.0)
               for i in range(100)]
    assert next(value for value in changes if value is not None) == 20.0


def test_recovery_from_large_temporary_overload_is_bounded_but_not_one_divisor_only():
    controller = MediaCadenceController(120, 60)
    changes = [controller.observe(pts=i / 10.0, at=100.0 + i / 10.0, playback_fps=10.0,
                                  service_ms=30.0, buffer_depth=8, display_fps=60.0)
               for i in range(220)]
    assert next(value for value in changes if value is not None) == 12.0


def test_72hz_candidates_use_integral_compositor_ticks():
    assert stable_media_cadence_candidates(120, 72, 40)[:3] == [36.0, 24.0, 18.0]
    assert stable_media_cadence_candidates(24, 60, 60)[0] == 24.0


def test_vsync_off_candidates_follow_source_divisors_under_display_ceiling():
    assert stable_media_cadence_candidates(120, 60, 40, align_display_ticks=False)[:4] == [
        40.0, 30.0, 24.0, 20.0
    ]


@pytest.mark.parametrize("source_fps", [30.0, 60.0])
def test_integral_source_divisor_accepts_required_72hz_pulldown(source_fps: float) -> None:
    controller = MediaCadenceController(source_fps, 30.0)
    source_pts = 10.0
    at = 100.0
    changes = []
    # 30 FPS on 72 Hz uses the deterministic 2,3,2,3,2-tick pulldown.
    for index in range(90):
        target = controller.observe(
            pts=source_pts,
            at=at,
            playback_fps=30.0,
            service_ms=8.0,
            display_fps=72.0,
            buffer_depth=8,
        )
        if target is not None:
            changes.append(target)
        at += (3 if index % 5 in {1, 3} else 2) / 72.0
        source_pts += 1.0 / 30.0
    assert changes == []


def test_120_to_36_source_schedule_preserves_ordered_three_four_pattern():
    schedule = SourceFrameSchedule(120.0, 36.0)
    assert [schedule.next_advance() for _ in range(9)] == [3, 3, 4, 3, 3, 4, 3, 3, 4]
    assert schedule.ffmpeg_select_expression() == (
        "eq(mod(n\\,10)\\,0)+eq(mod(n\\,10)\\,3)+eq(mod(n\\,10)\\,6)"
    )


def test_misaligned_30fps_72hz_playback_downshifts_to_24_when_36_is_over_budget():
    changes = drive(MediaCadenceController(120, 120), 30, 28, display=72)
    assert changes[0] == 24.0


def test_misaligned_30fps_72hz_playback_advances_to_36_when_budget_allows():
    changes = drive(MediaCadenceController(120, 120), 30, 26, display=72)
    assert changes[0] == 36.0


def test_failed_presentation_rate_gets_recovery_backoff():
    controller = MediaCadenceController(60, 60)
    at = 100.0
    pts = 0.0
    changes = []
    for index in range(160):
        interval = (1.0 / 18.0, 1.0 / 18.0, 1.0 / 12.0, 1.0 / 12.0)[index % 4]
        at += interval
        pts += interval
        target = controller.observe(
            pts=pts,
            at=at,
            playback_fps=14.4,
            service_ms=5.0,
            display_fps=72.0,
            buffer_depth=8,
        )
        if target is not None:
            changes.append(target)
    assert changes[:1] == [12.0]

    stable_changes = []
    for index in range(12 * 44):
        at += 1.0 / 12.0
        pts += 1.0 / 12.0
        target = controller.observe(
            pts=pts,
            at=at,
            playback_fps=12.0,
            service_ms=5.0,
            display_fps=72.0,
            buffer_depth=8,
        )
        if target is not None:
            stable_changes.append(target)
    assert stable_changes == []


def test_crossed_display_ticks_downshift_even_when_source_speed_is_one() -> None:
    controller = MediaCadenceController(60, 60)
    at = 100.0
    pts = 0.0
    changes = []
    for index in range(160):
        interval = (1.0 / 18.0, 1.0 / 18.0, 1.0 / 12.0, 1.0 / 12.0)[index % 4]
        at += interval
        pts += interval
        target = controller.observe(
            pts=pts,
            at=at,
            playback_fps=14.4,
            service_ms=5.0,
            display_fps=72.0,
            buffer_depth=8,
        )
        if target is not None:
            changes.append(target)
    assert changes
    assert changes[0] == 12.0


def test_extreme_load_uses_integral_ten_tick_rate_below_twelve_fps():
    controller = MediaCadenceController(120.0, 120.0)
    changes = []
    for index in range(32):
        at = 100.0 + index / 7.5
        target = controller.observe(
            pts=at - 100.0,
            at=at,
            playback_fps=7.5,
            service_ms=127.0,
            display_fps=72.0,
            buffer_depth=8,
        )
        if target is not None:
            changes.append(target)
    assert changes
    assert changes[0] == 7.2


def test_sparse_source_divisor_can_recover_when_there_is_no_intermediate_rung():
    controller = MediaCadenceController(60, 60)
    changes = [controller.observe(pts=i / 30.0, at=100.0 + i / 30.0, playback_fps=30.0,
                                  service_ms=8.0, buffer_depth=8, display_fps=72.0)
               for i in range(600)]
    assert next(value for value in changes if value is not None) == 60.0
