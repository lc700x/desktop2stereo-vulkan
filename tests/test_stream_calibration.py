import json
import socket
import urllib.request

import pytest

from streaming.stream_calibration import (
    CalibrationTier,
    StreamCalibrationController,
    calibration_tiers,
    evaluate_calibration_window,
    build_calibration_fingerprint,
    calibration_fingerprint_matches,
    minimum_bitrate_for_resolution,
    recommended_crf_for_bitrate,
)


def test_calibration_fingerprint_tracks_input_display_resolution_tier():
    base = {
        "Run Mode": "RTMP Streamer",
        "Input Display Resolution Tier": "1K",
        "Stream Protocol": "WebRTC",
    }
    changed = dict(base, **{"Input Display Resolution Tier": "2K"})

    assert build_calibration_fingerprint(base)["Input Display Resolution Tier"] == "1K"
    assert build_calibration_fingerprint(base) != build_calibration_fingerprint(changed)


def test_calibration_fingerprint_ignores_capture_monitor_index():
    base = {
        "Monitor Index": 1,
        "Input Display Resolution Tier": "4K",
        "Stream Protocol": "WebRTC",
    }
    changed = dict(base, **{"Monitor Index": 3})

    assert build_calibration_fingerprint(base) == build_calibration_fingerprint(changed)


def test_calibration_fingerprint_ignores_show_fps_overlay():
    base = {
        "Show FPS": False,
        "Input Display Resolution Tier": "4K",
        "Stream Protocol": "WebRTC",
    }
    changed = dict(base, **{"Show FPS": True})

    assert build_calibration_fingerprint(base) == build_calibration_fingerprint(changed)

    # Profiles written before this fix may still contain the old UI field.
    legacy_profile = dict(build_calibration_fingerprint(base), **{"Show FPS": "False"})
    assert calibration_fingerprint_matches(legacy_profile, changed)


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [(1920, 1080, "1K"), (2560, 1440, "2K"), (3840, 2160, "4K")],
)
def test_monitor_resolution_tier_uses_current_selected_display(
    monkeypatch, width, height, expected
):
    import gui.capture_sources as capture_sources

    monkeypatch.setattr(
        capture_sources,
        "list_monitors",
        lambda: [{"capture_index": 2, "width": width, "height": height}],
    )

    assert capture_sources.monitor_resolution_tier(2) == expected


def _receiver_report(fps=30.0, **overrides):
    report = {
        "decoded_fps": fps,
        "dropped_frames": 0,
        "freeze_count": 0,
        "packets_lost": 0,
        "bitrate_mbps": 40.0,
        "jitter_buffer_ms": 20.0,
    }
    report.update(overrides)
    return report


def test_calibration_tiers_are_ascending_and_bounded():
    tiers = calibration_tiers(50)

    assert [tier.fps for tier in tiers] == [30]
    assert [tier.target_mbps for tier in tiers] == [30]
    assert all(tier.target_mbps <= tier.peak_mbps for tier in tiers)


def test_default_probe_window_allows_headset_to_stabilize(tmp_path):
    controller = StreamCalibrationController(
        bind_port=12000,
        stream_port=1122,
        stream_key="live",
        maximum_fps=30,
        state_path=tmp_path / "state.json",
        profile_path=tmp_path / "profile.json",
    )

    assert controller.stage_seconds == 5.0
    assert controller.stability_seconds == 8.0
    assert controller.state()["overall_progress"] == 0.0


def test_probe_timer_starts_on_first_decoded_frame_after_bitrate_switch(tmp_path):
    now = [100.0]
    controller = StreamCalibrationController(
        bind_port=12000,
        stream_port=1122,
        stream_key="live",
        maximum_fps=30,
        state_path=tmp_path / "state.json",
        profile_path=tmp_path / "profile.json",
        stage_seconds=5.0,
        settle_seconds=2.0,
        clock=lambda: now[0],
    )

    controller.take_pending_tier()
    controller.observe_sender({"submitted_fps": 30.0})
    controller.add_receiver_report(_receiver_report(0.0))
    assert controller.state()["status"] == "waiting_receiver"
    assert controller.state()["stage_progress"] == 0.0

    now[0] = 109.0
    controller.add_receiver_report(_receiver_report(30.0))
    assert controller.state()["status"] == "testing"
    assert controller.state()["stage_progress"] == 0.0

    now[0] = 113.9
    controller.add_receiver_report(_receiver_report(30.0))
    assert controller.state()["status"] == "testing"
    now[0] = 114.0
    controller.add_receiver_report(_receiver_report(30.0))
    assert controller.state()["status"] != "testing"


def test_calibration_window_requires_sender_and_receiver_stability():
    tier = CalibrationTier(fps=40, target_mbps=40, peak_mbps=46)
    reports = [_receiver_report(39.5) for _ in range(8)]

    passed, metrics = evaluate_calibration_window(
        tier, reports, {"submitted_fps": 39.4}
    )
    assert passed
    assert metrics["decoded_fps"] == 39.5

    failed, _ = evaluate_calibration_window(
        tier,
        reports[:-1] + [_receiver_report(39.5, freeze_count=1)],
        {"submitted_fps": 39.4},
    )
    assert not failed


def test_controller_advances_then_rolls_back_to_highest_stable_tier(tmp_path):
    now = [0.0]
    controller = StreamCalibrationController(
        bind_port=12000,
        stream_port=1122,
        stream_key="live",
        maximum_fps=40,
        state_path=tmp_path / "state.json",
        profile_path=tmp_path / "profile.json",
        stage_seconds=2.0,
        stability_seconds=2.0,
        settle_seconds=0.0,
        clock=lambda: now[0],
    )

    assert controller.take_pending_tier().target_mbps == 30
    controller.observe_sender({"submitted_fps": 29.8})
    for _ in range(5):
        controller.add_receiver_report(_receiver_report(29.5, bitrate_mbps=100.0))
    assert controller.state()["receiver_latest"]["decoded_fps"] == 29.5
    now[0] = 2.1
    controller.add_receiver_report(_receiver_report(29.5, bitrate_mbps=100.0))

    assert controller.take_pending_tier().target_mbps == 60
    first_progress = controller.state()["overall_progress"]
    assert first_progress > 0.0
    controller.observe_sender({"submitted_fps": 29.8})
    for _ in range(5):
        controller.add_receiver_report(_receiver_report(29.5, bitrate_mbps=100.0))
    now[0] = 4.2
    controller.add_receiver_report(_receiver_report(29.5, bitrate_mbps=100.0))

    assert controller.take_pending_tier().target_mbps == 80
    assert controller.state()["overall_progress"] > first_progress
    controller.observe_sender({"submitted_fps": 29.8})
    for _ in range(5):
        controller.add_receiver_report(
            _receiver_report(29.5, bitrate_mbps=100.0, packets_lost=1)
        )
    now[0] = 6.3
    controller.add_receiver_report(
        _receiver_report(29.5, bitrate_mbps=100.0, packets_lost=1)
    )

    for expected_target in (70, 75, 77):
        assert controller.take_pending_tier().target_mbps == expected_target
        controller.observe_sender({"submitted_fps": 29.8})
        for _ in range(5):
            controller.add_receiver_report(
                _receiver_report(29.5, bitrate_mbps=100.0)
            )
        now[0] += 2.1
        controller.add_receiver_report(_receiver_report(29.5, bitrate_mbps=100.0))

    # The search bracketed the unstable ceiling at 80 Mbps and converged within
    # the 4 Mbps search resolution; confirm that candidate using the longer
    # stability window.
    assert controller.take_pending_tier() is None
    controller.observe_sender({"submitted_fps": 29.8})
    for _ in range(5):
        controller.add_receiver_report(_receiver_report(29.5, bitrate_mbps=100.0))
    now[0] += 2.1
    controller.add_receiver_report(_receiver_report(29.5, bitrate_mbps=100.0))

    assert controller.take_pending_tier() is None
    profile = json.loads((tmp_path / "profile.json").read_text(encoding="utf-8"))
    assert profile["fps"] == 30
    assert profile["network_max_mbps"] == 77
    assert profile["target_mbps"] == 61
    assert profile["peak_mbps"] == 69
    assert profile["measured_bitrate_mbps"] == 100.0
    assert controller.state()["status"] == "complete"


def test_controller_stops_at_first_unstable_bitrate_probe(tmp_path):
    now = [0.0]
    controller = StreamCalibrationController(
        bind_port=12000,
        stream_port=1122,
        stream_key="live",
        maximum_fps=30,
        state_path=tmp_path / "state.json",
        profile_path=tmp_path / "profile.json",
        stage_seconds=2.0,
        stability_seconds=2.0,
        settle_seconds=0.0,
        clock=lambda: now[0],
    )
    original = controller.take_pending_tier()
    controller.observe_sender({"submitted_fps": 29.8})
    for _ in range(5):
        controller.add_receiver_report(_receiver_report(20.0, dropped_frames=2, packets_lost=1))
    now[0] = 2.1
    controller.add_receiver_report(_receiver_report(20.0, dropped_frames=2, packets_lost=1))

    for expected_target in (18, 24, 27):
        assert controller.take_pending_tier().target_mbps == expected_target
        controller.observe_sender({"submitted_fps": 29.8})
        for _ in range(5):
            controller.add_receiver_report(_receiver_report(29.5))
        now[0] += 2.1
        controller.add_receiver_report(_receiver_report(29.5))

    assert controller.take_pending_tier() is None
    controller.observe_sender({"submitted_fps": 29.8})
    for _ in range(5):
        controller.add_receiver_report(_receiver_report(29.5))
    now[0] += 2.1
    controller.add_receiver_report(_receiver_report(29.5))
    assert controller.state()["status"] == "complete"
    profile = json.loads((tmp_path / "profile.json").read_text(encoding="utf-8"))
    assert profile["network_max_mbps"] == 27
    assert profile["target_mbps"] == 21
    assert profile["peak_mbps"] == 24


def test_calibration_tier_uses_selected_input_resolution():
    tier = calibration_tiers(
        60,
        input_width=2560,
        input_height=1440,
    )[0]

    assert tier.fps == 30
    assert tier.target_mbps == 13


@pytest.mark.parametrize(
    ("width", "height", "minimum", "maximum"),
    [
        (640, 480, 2, 3),
        (1280, 720, 3, 5),
        (1920, 1080, 5, 8),
        (3840, 2160, 20, 35),
    ],
)
def test_calibration_tier_starts_in_resolution_appropriate_range(
    width, height, minimum, maximum
):
    tier = calibration_tiers(
        60,
        input_width=width,
        input_height=height,
    )[0]

    assert minimum <= tier.target_mbps <= maximum


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [
        (640, 480, 2),
        (1280, 720, 3),
        (1920, 1080, 6),
        (2560, 1440, 10),
        (3840, 2160, 20),
    ],
)
def test_stable_profile_floor_follows_resolution(width, height, expected):
    assert minimum_bitrate_for_resolution(width, height) == expected


def test_stable_1080p_profile_allows_a_6_mbps_floor(tmp_path):
    controller = StreamCalibrationController(
        bind_port=12000,
        stream_port=1122,
        stream_key="live",
        maximum_fps=30,
        state_path=tmp_path / "state.json",
        profile_path=tmp_path / "profile.json",
    )
    controller.configure_input_resolution(1920, 1080)
    controller._complete_locked(
        CalibrationTier(fps=30, target_mbps=7, peak_mbps=8),
        {},
        "stable",
    )

    profile = json.loads((tmp_path / "profile.json").read_text(encoding="utf-8"))
    assert profile["network_max_mbps"] == 7
    assert profile["target_mbps"] == 6


def test_calibration_window_rejects_a_probe_that_did_not_reach_target_rate():
    tier = CalibrationTier(fps=30, target_mbps=30, peak_mbps=34)
    reports = [_receiver_report(29.5, bitrate_mbps=8.0) for _ in range(8)]

    passed, metrics = evaluate_calibration_window(
        tier,
        reports,
        {"submitted_fps": 30.0},
    )

    assert passed is False
    assert "insufficient_probe_bitrate" in metrics["failure_reasons"]


def test_headset_page_uses_whep_and_reports_browser_stats(tmp_path):
    controller = StreamCalibrationController(
        bind_port=12000,
        stream_port=1122,
        stream_key="live",
        maximum_fps=30,
        state_path=tmp_path / "state.json",
        profile_path=tmp_path / "profile.json",
    )

    page = controller._viewer_html()
    assert "/whep" in page
    assert "getStats()" in page
    assert "framesDecoded" in page
    assert "jitterBufferDelay" in page
    assert "addTransceiver('audio'" not in page
    assert "WHEP ${response.status} ${await response.text()}" in page
    assert "new MediaStream([e.track])" in page
    assert "pollCalibrationState" in page
    assert "当前测试：目标" in page
    assert "最终结果：稳定上限" in page


def test_calibration_fingerprint_tracks_runtime_and_receiver_choices():
    fingerprint = build_calibration_fingerprint(
        {
            "Computing Device": 0,
            "XR Headset Model": "Pico 4 / 4 Ultra",
            "Display Mode": "Half-SBS",
            "Depth Model": "Distill-Any-Depth-Base",
            "Stream Protocol": "WebRTC",
        }
    )

    assert fingerprint["Computing Device"] == "0"
    assert fingerprint["XR Headset Model"] == "Pico 4 / 4 Ultra"
    assert fingerprint["Stream Protocol"] == "WebRTC"


def test_calibration_fingerprint_excludes_result_fields_but_tracks_other_changes():
    base = {
        "Target FPS": 60,
        "Stream Target FPS": 30,
        "Use Stream Calibration": True,
        "Stream Target Bitrate Mbps": 30,
        "Stream Peak Bitrate Mbps": 34,
        "Show Log Panel": True,
        "Streamer Port": 1122,
        "CRF": 23,
    }
    changed = dict(base, **{"Streamer Port": 1123})

    first = build_calibration_fingerprint(base)
    second = build_calibration_fingerprint(changed)

    assert "Target FPS" not in first
    assert "Stream Target FPS" not in first
    assert "Stream Target Bitrate Mbps" not in first
    assert first["Streamer Port"] == "1122"
    assert first != second


def test_calibration_fingerprint_ignores_stream_key_and_audio_settings():
    base = {
        "Streamer Port": 1122,
        "Stream Key": "live",
        "Stereo Mix": "soundcard:Headphones",
        "Audio Delay": -0.1,
    }
    changed = dict(
        base,
        **{
            "Stream Key": "other",
            "Stereo Mix": "soundcard:Speakers",
            "Audio Delay": 0.2,
        },
    )

    assert build_calibration_fingerprint(base) == build_calibration_fingerprint(changed)


def test_calibration_fingerprint_ignores_local_stereo_output_monitor():
    base = {
        "Run Mode": "RTMP Streamer",
        "Stream Protocol": "WebRTC",
        "Input Display Resolution Tier": "4K",
        "Stereo Output": 2,
    }
    changed = dict(base, **{"Stereo Output": 3})

    assert build_calibration_fingerprint(base) == build_calibration_fingerprint(changed)


@pytest.mark.parametrize(
    ("bitrate", "expected"),
    [(35, 20), (30, 20), (29, 23), (25, 23), (24, 26), (21, 26), (20, 28), (19, 28), (18, 30)],
)
def test_recommended_crf_follows_calibrated_safe_bitrate(bitrate, expected):
    assert recommended_crf_for_bitrate(bitrate) == expected


def test_calibration_fingerprint_accepts_legacy_crf_field():
    settings = {"Run Mode": "RTMP Streamer", "Stream Protocol": "WebRTC"}
    saved = dict(build_calibration_fingerprint(settings), CRF="23")

    assert calibration_fingerprint_matches(saved, settings)


def test_feedback_http_server_serves_page_and_accepts_stats(tmp_path, capsys):
    reservation = socket.socket()
    reservation.bind(("127.0.0.1", 0))
    port = reservation.getsockname()[1]
    reservation.close()
    controller = StreamCalibrationController(
        bind_port=port,
        stream_port=1122,
        stream_key="live",
        maximum_fps=30,
        state_path=tmp_path / "state.json",
        profile_path=tmp_path / "profile.json",
    )
    controller.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as response:
            assert "RTCPeerConnection" in response.read().decode("utf-8")
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/stats",
            data=json.dumps(_receiver_report()).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            assert json.loads(response.read())["ok"] is True
        assert controller.state()["receiver_connected"] is True
        output = capsys.readouterr().out
        assert "HTTP GET client=127.0.0.1 path=/ status=200" in output
        assert "HTTP POST client=127.0.0.1 path=/stats status=200" in output
    finally:
        controller.close()
