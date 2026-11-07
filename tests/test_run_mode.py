from __future__ import annotations

from streaming.stream_session import is_network_stream_mode, supports_network_calibration
from utils.run_mode import (
    is_local_same_monitor_3d_mode,
    normalize_run_mode,
    resolve_run_mode,
    target_fps_setting_key,
)


def test_same_monitor_local_viewer_migrates_legacy_mode_and_enables_3d():
    assert normalize_run_mode("3D Monitor") == "Local Viewer"
    assert is_local_same_monitor_3d_mode(
        "Local Viewer",
        os_name="Windows",
        monitor_count=1,
        capture_mode="Monitor",
    )
    assert is_local_same_monitor_3d_mode(
        "3D Monitor",
        os_name="Windows",
        monitor_count=2,
        capture_mode="Monitor",
        input_monitor_index=2,
        output_monitor_index=2,
    )
    assert not is_local_same_monitor_3d_mode(
        "Local Viewer",
        os_name="Windows",
        monitor_count=2,
        capture_mode="Monitor",
        input_monitor_index=1,
        output_monitor_index=2,
    )
    assert not is_local_same_monitor_3d_mode(
        "Local Viewer",
        os_name="Windows",
        monitor_count=1,
        capture_mode="Window",
    )
    assert not is_local_same_monitor_3d_mode(
        "Local Viewer",
        os_name="Darwin",
        monitor_count=1,
        capture_mode="Monitor",
    )

    config = resolve_run_mode(
        "Local Viewer",
        os_name="Windows",
        fix_viewer_aspect=True,
        lossless_scaling_support=False,
        monitor_count=2,
        capture_mode="Monitor",
        input_monitor_index=2,
        output_monitor_index=2,
    )
    assert config.use_3d_monitor


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


def test_moonlight_mode_uses_the_viewer_runtime_on_macos():
    config = resolve_run_mode(
        "Moonlight Streamer",
        os_name="Darwin",
        fix_viewer_aspect=True,
        lossless_scaling_support=False,
    )

    assert config.run_mode == "Viewer"
    assert config.stream_mode == "MOONLIGHT"
