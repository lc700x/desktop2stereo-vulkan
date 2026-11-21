from pathlib import Path

from path_config import APP_ROOT

from capture.adaptive_rate import (
    AdaptiveCaptureRate,
    adaptive_capture_enabled_for_mode,
)
from gui.localization import get_messages


ROOT = Path(__file__).resolve().parents[1]
BUILDERS_SOURCE = APP_ROOT / "gui" / "builders.py"
HANDLERS_SOURCE = APP_ROOT / "gui" / "handlers.py"


def test_auto_capture_calibrates_from_sustained_sbs_after_15_seconds() -> None:
    rate = AdaptiveCaptureRate(60, enabled=True)

    assert rate.observe_sbs_fps(20.0, frame_count=100, now=1.0) == 60
    assert rate.observe_sbs_fps(20.0, frame_count=100, now=10.0) == 60
    assert rate.observe_sbs_fps(20.0, frame_count=100, now=16.0) == 25


def test_auto_capture_uses_peak_sustained_sbs_sample() -> None:
    rate = AdaptiveCaptureRate(60, enabled=True)

    rate.observe_sbs_fps(20.0, frame_count=100, now=1.0)
    rate.observe_sbs_fps(30.0, frame_count=150, now=10.0)
    assert rate.observe_sbs_fps(20.0, frame_count=100, now=16.0) == 35


def test_auto_capture_tracks_improved_sustained_capacity() -> None:
    rate = AdaptiveCaptureRate(60, enabled=True)
    rate.observe_sbs_fps(20.0, frame_count=100, now=1.0)
    rate.observe_sbs_fps(20.0, frame_count=100, now=16.0)
    assert rate.current_fps() == 25

    assert rate.observe_sbs_fps(40.0, frame_count=200, now=17.0) == 25
    assert rate.observe_sbs_fps(40.0, frame_count=200, now=32.0) == 45


def test_auto_capture_is_capped_by_base_refresh_rate() -> None:
    rate = AdaptiveCaptureRate(60, enabled=True)

    rate.observe_sbs_fps(90.0, frame_count=450, now=1.0)
    assert rate.observe_sbs_fps(90.0, frame_count=450, now=16.0) == 60


def test_auto_capture_respects_a_runtime_minimum_fps() -> None:
    rate = AdaptiveCaptureRate(60, enabled=True, minimum_fps=45)

    rate.observe_sbs_fps(8.0, now=1.0)
    assert rate.observe_sbs_fps(8.0, now=16.0) == 45


def test_auto_capture_has_no_default_runtime_minimum_fps() -> None:
    rate = AdaptiveCaptureRate(60, enabled=True)

    rate.observe_sbs_fps(8.0, now=1.0)
    assert rate.observe_sbs_fps(8.0, now=16.0) == 13


def test_auto_capture_holds_target_when_no_frames_arrive() -> None:
    rate = AdaptiveCaptureRate(60, enabled=True)

    rate.observe_sbs_fps(20.0, frame_count=100, now=1.0)
    assert rate.current_fps() == 60


def test_sparse_sbs_windows_do_not_reduce_capture_target() -> None:
    rate = AdaptiveCaptureRate(60, enabled=True)

    rate.observe_sbs_fps(5.0, frame_count=25, now=1.0)
    assert rate.observe_sbs_fps(8.0, frame_count=40, now=20.0) == 60


def test_static_dynamic_activity_guard_is_disabled_by_default() -> None:
    rate = AdaptiveCaptureRate(60, enabled=True)

    rate.observe_sbs_fps(20.0, frame_count=100, now=1.0)
    rate.observe_sbs_fps(20.0, frame_count=100, now=16.0)
    assert rate.current_fps() == 25
    assert rate.observe_sbs_fps(1.0, capture_fps=0.0, now=21.0) == 25


def test_static_dynamic_activity_guard_can_be_reenabled() -> None:
    rate = AdaptiveCaptureRate(60, enabled=True, activity_guard_enabled=True)

    rate.observe_sbs_fps(20.0, frame_count=100, now=1.0)
    rate.observe_sbs_fps(20.0, frame_count=100, now=16.0)
    assert rate.observe_sbs_fps(1.0, capture_fps=0.5, now=21.0) == 60


def test_invalid_sbs_sample_is_ignored() -> None:
    rate = AdaptiveCaptureRate(60, enabled=True)

    assert rate.observe_sbs_fps("invalid", frame_count=100, now=1.0) == 60
    assert rate.observe_sbs_fps(0.0, frame_count=100, now=16.0) == 60


def test_manual_capture_rate_is_not_adapted() -> None:
    rate = AdaptiveCaptureRate(60, enabled=False)

    for now in range(10):
        assert rate.observe_sbs_fps(20.0, frame_count=100, now=float(now)) == 60


def test_manual_network_probe_temporarily_adds_five_fps_headroom() -> None:
    rate = AdaptiveCaptureRate(30, enabled=False)

    assert rate.begin_stream_probe(30) == 35
    assert rate.observe_sbs_fps(34.0, frame_count=100, now=1.0) == 35
    assert rate.finish_stream_probe(30) == 30


def test_auto_network_probe_retains_selected_rate_headroom() -> None:
    rate = AdaptiveCaptureRate(60, enabled=True)

    assert rate.begin_stream_probe(30) == 35
    assert rate.finish_stream_probe(30) == 35


def test_calibration_limit_caps_capture_during_tier_test() -> None:
    rate = AdaptiveCaptureRate(60, enabled=True, evaluation_interval_s=1.0)

    assert rate.set_calibration_limit(40) == 40
    rate.observe_sbs_fps(55.0, frame_count=100, now=0.0)
    rate.observe_sbs_fps(55.0, frame_count=100, now=1.1)

    assert rate.current_fps() == 40
    assert rate.set_calibration_limit(None) == 40


def test_auto_capture_is_enabled_for_local_viewer_but_not_3d_monitor() -> None:
    assert adaptive_capture_enabled_for_mode("Local Viewer", 0)
    assert adaptive_capture_enabled_for_mode("OpenXR Link", 0)
    assert adaptive_capture_enabled_for_mode("RTMP Streamer", 0)
    assert not adaptive_capture_enabled_for_mode("3D Monitor", 0)
    assert not adaptive_capture_enabled_for_mode("RTMP Streamer", 60)
    assert not adaptive_capture_enabled_for_mode("Local Viewer", 60)


def test_capture_fps_gui_exposes_24_and_30_with_adaptive_tooltip() -> None:
    source = BUILDERS_SOURCE.read_text(encoding="utf-8")
    assert 'options=["Auto"] + [str(fps) for fps in range(5, 95, 5)]' in source
    handlers_source = HANDLERS_SOURCE.read_text(encoding="utf-8")
    assert '[t["Auto"]] + [str(fps) for fps in range(5, 95, 5)]' in handlers_source
    tooltip = get_messages("ZH")["tooltip_target_fps"]
    assert "SBS" in tooltip
    assert "+ 5 FPS" in tooltip
    assert "每 15 秒" in tooltip
    assert "持续输出窗口" in tooltip
