"""Acceptance summaries must not hide drops, pause gaps, or missing runtime evidence."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

spec = spec_from_file_location("media_acceptance_analysis", Path(__file__).resolve().parents[1]
                              / "scripts" / "benchmark" / "analyze_media_acceptance.py")
analysis = module_from_spec(spec)
spec.loader.exec_module(analysis)


def frames(step=3, count=41, start=0.0):
    return [{"t": start + i / 40.0, "media_generation": 1, "source_frame_index": i * step,
             "source_pts": i * step / 120.0, "media_source_fps": 120.0,
             "media_playback_fps": 40.0} for i in range(count)]


def test_selected_source_divisor_is_not_reported_as_dropping():
    result = analysis.summarize_segment(frames())
    assert result["source_pts_speed"] == pytest.approx(1.0)
    assert result["fresh_fps"] == pytest.approx(40.0)
    assert result["expected_source_steps"] == {3: 40}
    assert not result["anomalies"]


def test_consistent_double_decimation_cannot_pass_by_using_modal_delta():
    result = analysis.summarize_segment(frames(step=6))
    assert len(result["anomalies"]) == 40
    assert result["anomalies"][0]["expected_step"] == 3
    assert result["source_pts_speed"] == pytest.approx(2.0)


def test_duplicate_and_out_of_order_frames_remain_visible():
    rows = frames(count=5)
    rows[2]["source_frame_index"] = rows[1]["source_frame_index"]
    rows[3]["source_frame_index"] = 0
    result = analysis.summarize_segment(rows)
    assert 0 in result["source_steps"]
    assert -3 in result["source_steps"]
    assert len(result["anomalies"]) == 3


def test_pause_is_excluded_from_speed_but_retention_is_measured():
    before = frames(count=801)
    after = frames(count=801, start=25.0)
    for row in after:
        row["source_frame_index"] += 2403
        row["source_pts"] += 20.025
    data = {"presents": before + after,
            "scenario_events": [{"t": 20.1, "action": "pause"}, {"t": 25.0, "action": "play"}],
            "observations": [{"t": 22.0, "playing": False, "source_identity": [1, 2400, 20.0], "layers": 2}]}
    result = analysis.summarize_run(data, warmup_s=1.0)
    assert len(result["segments"]) == 2
    assert all(row["source_pts_speed"] == pytest.approx(1.0) for row in result["segments"])
    assert result["pause"]["source_identities"] == [[1, 2400, 20.0]]
    assert result["pause"]["layers"] == {2: 1}


def test_no_presentation_and_missing_audio_are_not_proof_of_success():
    assert analysis.summarize_run({"presents": [], "rc": 0})["status"] == "no_presentation"
    result = analysis.summarize_run({"presents": frames()}, warmup_s=0.0)
    assert result["audio_clock_offset_ms"] == {}
    assert result["pcm_underruns"] is None


def test_predicted_clock_is_used_for_display_intervals_not_cpu_completion():
    rows = frames(count=4)
    for i, row in enumerate(rows):
        row["predicted_present_time_s"] = 100.0 + i / 72.0
        row["refresh_hz"] = 72.0
    result = analysis.summarize_segment(rows)
    assert result["presentation_interval_ms"]["p50"] == pytest.approx(1000.0 / 72.0)
    assert result["display_ticks"] == {1: 3}


def test_multiple_videos_are_split_by_generation_not_combined_into_a_false_speed():
    first = frames()
    second = frames(start=5.0)
    for row in second:
        row["media_generation"] = 2
        row["source_pts"] += 60.0
    data = {"presents": first + second, "media_sessions": {"1": {"path": "first.mp4"},
                                                         "2": {"path": "second.mp4"}}}
    result = analysis.summarize_run(data, warmup_s=0.0)
    assert result["status"] == "multi_video"
    assert result["videos"]["1"]["media"]["path"] == "first.mp4"
    assert result["videos"]["2"]["media"]["path"] == "second.mp4"
    assert all(video["segments"][0]["source_pts_speed"] == pytest.approx(1.0)
               for video in result["videos"].values())


def test_steady_audio_counter_deltas_do_not_mislabel_startup_underruns():
    data = {"presents": frames(), "observations": [
        {"t": 0.1, "playing": True, "audio_sync": {"underruns": 4, "device_underflows": 0}},
        {"t": 0.9, "playing": True, "audio_sync": {"underruns": 4, "device_underflows": 0}},
    ]}
    result = analysis.summarize_run(data, warmup_s=0.0)
    assert result["pcm_underruns"] == 4
    assert result["steady_audio_counter_deltas"] == [{"underruns": 0, "device_underflows": 0}]


def test_whole_playing_speed_retains_startup_slowdown_even_when_steady_speed_is_correct():
    rows = frames(count=801)
    for row in rows[1:]:
        row["t"] += 0.5
    result = analysis.summarize_run({"presents": rows}, warmup_s=1.0)
    assert result["segments"][0]["source_pts_speed"] == pytest.approx(1.0)
    assert result["whole_playing_source_speed"] == pytest.approx(20.0 / 20.5)


def test_eof_duration_uses_source_timeline_and_pause_exclusion():
    rows = frames(count=801)
    for row in rows:
        if row["t"] >= 5.0:
            row["t"] += 2.0
    data = {"presents": rows, "scenario_events": [
        {"t": 5.0, "action": "pause"}, {"t": 7.0, "action": "play"},
        {"t": 22.025, "action": "eof", "duration": 20.025},
    ]}
    result = analysis.summarize_run(data, warmup_s=0.0)
    assert result["duration_evidence"]["source_duration_s"] == 20.025
    assert result["duration_evidence"]["playing_duration_s"] == pytest.approx(20.025)
    assert result["duration_evidence"]["difference_ms"] == pytest.approx(0.0)


def test_announced_terminal_partial_is_visible_but_not_an_unplanned_drop():
    rows = frames(step=3, count=4)
    rows[-1]["source_frame_index"] = 7
    rows[-1]["media_terminal_partial"] = True
    result = analysis.summarize_segment(rows)
    assert result["terminal_partial_frames"] == 1
    assert result["source_steps"] == {3: 2, 1: 1}
    assert result["anomalies"] == []
    rows[-1]["source_frame_index"] = 6
    assert analysis.summarize_segment(rows)["anomalies"]  # EOF never excuses duplicates.
    rows[-1]["source_frame_index"] = 12
    assert analysis.summarize_segment(rows)["anomalies"]  # Nor skipped selected frames.


def test_modelled_indices_cannot_hide_real_decoder_pts_skips():
    rows = frames(count=41)
    for row in rows:
        row["source_pts"] *= 2.0
    result = analysis.summarize_segment(rows)
    assert not result["anomalies"]
    assert len(result["pts_anomalies"]) == 40


def test_native_millisecond_pts_quantization_is_not_a_dropped_frame():
    rows = frames(count=41)
    for row in rows:
        row["source_pts"] = round(row["source_pts"], 3)
    assert not analysis.summarize_segment(rows)["pts_anomalies"]


def test_duration_excludes_applied_pause_not_the_earlier_request_timestamp():
    data = {"presents": frames(count=801), "scenario_events": [
        {"t": 5.0, "action": "pause"}, {"t": 7.0, "action": "play"},
        {"t": 5.02, "action": "control_applied", "command": "pause"},
        {"t": 7.08, "action": "control_applied", "command": "play"},
        {"t": 22.085, "action": "eof", "duration": 20.025},
    ]}
    result = analysis.summarize_run(data, warmup_s=0.0)
    assert result["pause_clock"] == "control_applied"
    assert result["duration_evidence"]["difference_ms"] == pytest.approx(0.0)


def test_duration_uses_display_epoch_and_exact_frozen_source_clock_not_callback_delay():
    rows = frames(count=801)
    for row in rows:
        row["presentation_at_relative_s"] = row["t"]
        row["t"] += 0.06
    data = {"presents": rows, "scenario_events": [
        {"t": 5.08, "clock_t": 5.0, "action": "control_applied", "command": "pause"},
        {"t": 7.18, "clock_t": 7.0, "action": "control_applied", "command": "play"},
        {"t": 22.025, "action": "eof", "duration": 20.025},
    ]}
    result = analysis.summarize_run(data, warmup_s=0.0)
    assert result["pause_clock"] == "source_epoch"
    assert result["duration_evidence"]["difference_ms"] == pytest.approx(0.0)
