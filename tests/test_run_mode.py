from __future__ import annotations

from streaming.stream_session import is_network_stream_mode, supports_network_calibration
from utils.run_mode import resolve_run_mode, target_fps_setting_key


def test_moonlight_mode_is_a_separate_windows_streamer():
    config = resolve_run_mode(
        "Moonlight Streamer",
        os_name="Windows",
        fix_viewer_aspect=False,
        lossless_scaling_support=False,
    )

    assert config.run_mode == "Viewer"
    assert config.stream_mode == "MOONLIGHT"
    assert not config.lossless_scaling_support
    assert target_fps_setting_key("Moonlight Streamer") == "Stream Target FPS"
    assert is_network_stream_mode("Moonlight Streamer")
    assert not supports_network_calibration("Moonlight Streamer", "WebRTC")


def test_moonlight_mode_is_not_enabled_on_macos():
    config = resolve_run_mode(
        "Moonlight Streamer",
        os_name="Darwin",
        fix_viewer_aspect=True,
        lossless_scaling_support=False,
    )

    assert config.run_mode == "Streamer"
    assert config.stream_mode is None
