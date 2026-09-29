import asyncio
import json
from types import SimpleNamespace

import pytest

import gui.process as gui_process


def test_windows_process_tree_is_killed_before_parent(monkeypatch):
    events = []

    class FakeTreeKill:
        async def wait(self):
            events.append("taskkill.wait")
            return 0

    class FakeProcess:
        returncode = None

        def kill(self):
            events.append("parent.kill")

    async def fake_create_subprocess_exec(*args, **kwargs):
        events.append(tuple(args))
        return FakeTreeKill()

    monkeypatch.setattr(gui_process, "OS_NAME", "Windows")
    monkeypatch.setattr(
        gui_process.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    asyncio.run(
        gui_process.GUIProcessMixin._kill_process_tree(
            object(), FakeProcess(), 4242
        )
    )

    assert events == [
        ("taskkill", "/f", "/t", "/pid", "4242"),
        "taskkill.wait",
        "parent.kill",
    ]


def test_mediamtx_warning_is_not_promoted_by_error_text(monkeypatch):
    levels = []
    monkeypatch.setattr(
        gui_process.child_logger, "warning", lambda message: levels.append("warning")
    )
    monkeypatch.setattr(
        gui_process.child_logger, "error", lambda message: levels.append("error")
    )

    gui_process.GUIProcessMixin._log_child_line(
        object(),
        "[MediaMTX] 2026/08/19 19:05:54 WAR [HLS] segment changed - "
        "this will cause an error in iOS clients",
    )

    assert levels == ["warning"]


def test_graceful_stop_timeout_allows_runtime_cleanup():
    assert gui_process._GRACEFUL_PROCESS_STOP_TIMEOUT_S >= 8.0


def test_run_completion_flags_preserve_runtime_glow_selection(tmp_path):
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        "OpenXR Glow Modes:\n"
        "  Default: off\n"
        "OpenXR Glow Transparency:\n"
        "  Default: 0.35\n"
        "Recompile TensorRT: true\n",
        encoding="utf-8",
    )

    assert gui_process._save_run_completion_flags(str(settings_path)) == (True, "")

    settings = gui_process.read_yaml(str(settings_path))
    # PyYAML emits the YAML 1.1 spelling ``false`` for the string ``off``;
    # the runtime normalizes that value back to the explicit OFF mode.
    assert settings["OpenXR Glow Modes"] == {"Default": False}
    assert settings["OpenXR Glow Transparency"] == {"Default": 0.35}
    assert settings["Recompile TensorRT"] is False


def test_firewall_probe_parses_single_and_multiple_rules():
    single = gui_process._parse_firewall_block_output(json.dumps({"Protocol": "TCP"}))
    multiple = gui_process._parse_firewall_block_output(
        json.dumps([{"Protocol": "TCP"}, {"Protocol": "UDP"}])
    )

    assert single == [{"Protocol": "TCP"}]
    assert multiple == [{"Protocol": "TCP"}, {"Protocol": "UDP"}]


def test_firewall_probe_is_read_only_and_targets_bundled_python(monkeypatch):
    calls = []

    class Result:
        returncode = 0
        stdout = json.dumps({"Protocol": "TCP", "DisplayName": "Python"})
        stderr = ""

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return Result()

    monkeypatch.setattr(gui_process, "OS_NAME", "Windows")
    monkeypatch.setattr(gui_process.subprocess, "run", fake_run)

    result = gui_process._detect_windows_firewall_blocks(r"C:\D2S\python.exe")

    assert result[0]["DisplayName"] == "Python"
    assert calls[0][0][0:3] == ["powershell.exe", "-NoProfile", "-NonInteractive"]
    assert "Get-NetFirewallRule" in calls[0][0][-1]
    assert calls[0][1]["env"]["D2S_FIREWALL_EXE"].casefold().endswith(r"c:\d2s\python.exe")
    assert calls[0][1]["timeout"] == 20
    assert "Write-Output '[]'" in calls[0][0][-1]
    assert "$matches.Count -eq 0" in calls[0][0][-1]


def test_firewall_probe_accepts_empty_rule_result(monkeypatch):
    class Result:
        returncode = 0
        stdout = "[]\n"
        stderr = ""

    monkeypatch.setattr(gui_process, "OS_NAME", "Windows")
    monkeypatch.setattr(gui_process.subprocess, "run", lambda *_args, **_kwargs: Result())

    assert gui_process._detect_windows_firewall_blocks(r"C:\D2S\python.exe") == []


def test_firewall_probe_timeout_is_not_treated_as_no_block_rules(monkeypatch):
    def fake_run(*_args, **_kwargs):
        raise gui_process.subprocess.TimeoutExpired("powershell.exe", 20)

    monkeypatch.setattr(gui_process, "OS_NAME", "Windows")
    monkeypatch.setattr(gui_process.subprocess, "run", fake_run)

    with pytest.raises(gui_process.FirewallProbeError, match="timed out"):
        gui_process._detect_windows_firewall_blocks(r"C:\D2S\python.exe")


def test_manual_firewall_check_removes_matching_rules_in_background(monkeypatch):
    events = []

    class Harness(gui_process.GUIProcessMixin):
        def set_status(self, message, key=None):
            events.append(message)

        def _safe_update(self, *controls):
            pass

    gui = Harness()
    gui.locale = "EN"
    gui._calibration_firewall_btn = SimpleNamespace(disabled=False)
    gui._calibration_dialog_firewall_hint = SimpleNamespace(value="")
    monkeypatch.setattr(
        gui_process,
        "_detect_windows_firewall_blocks",
        lambda: [{"Protocol": "TCP", "DisplayName": "Python"}],
    )
    monkeypatch.setattr(
        gui_process,
        "_remove_windows_firewall_blocks",
        lambda: (True, ""),
    )

    asyncio.run(gui._check_stream_calibration_firewall_async())

    assert gui._calibration_firewall_btn.disabled is False
    assert "were removed" in gui._calibration_dialog_firewall_hint.value
    assert events[0] == "Checking Windows Firewall rules..."


def test_calibration_stops_runtime_before_showing_result(monkeypatch, tmp_path):
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps({"fps": 30, "target_mbps": 25, "peak_mbps": 29}),
        encoding="utf-8",
    )
    events = []

    class Harness(gui_process.GUIProcessMixin):
        async def _async_stop(self):
            events.append("stop")

        def set_status(self, message, key=None):
            events.append(message)

        def _refresh_stream_calibration_status(self):
            events.append("refresh")

        def _collect_config(self):
            events.append("collect")

        def _safe_update(self, *controls):
            pass

    gui = Harness()
    gui.locale = "EN"
    gui._config = {}
    gui.target_fps_dd = SimpleNamespace(value="0")
    gui.stream_calibration_mode_dd = SimpleNamespace(value="")
    gui._calibration_previous_target_value = "Auto"
    gui._calibration_active = True
    gui._calibration_dialog = None

    monkeypatch.setattr(gui_process, "STREAM_CALIBRATION_PROFILE_FILE", str(profile_path))
    monkeypatch.setattr(gui_process, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(gui_process, "save_yaml", lambda *args: (True, ""))

    asyncio.run(gui._apply_stream_calibration_profile())

    assert events.index("stop") < events.index("refresh")
    assert events.index("stop") < events.index(
        "Calibration applied: 30 FPS, 25 Mbps"
    )


def test_limited_7mbps_profile_applies_safe_1080p_rate(monkeypatch, tmp_path):
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps({
            "fps": 30,
            "target_mbps": 5,
            "peak_mbps": 6,
            "network_max_mbps": 7,
            "stability": "limited",
            "metrics": {"failure_reasons": ["insufficient_probe_bitrate"]},
        }),
        encoding="utf-8",
    )
    events = []

    class Harness(gui_process.GUIProcessMixin):
        async def _async_stop(self):
            events.append("stop")

        def _restore_precalibration_target(self):
            events.append("restore")

        def _set_stream_bitrate_controls(self, target, peak):
            events.append(("controls", target, peak))

        def _refresh_stream_calibration_status(self):
            events.append("refresh")

        def _collect_config(self):
            pass

        def set_status(self, message, key=None):
            events.append(message)

        def _safe_update(self, *controls):
            pass

    gui = Harness()
    gui.locale = "EN"
    gui._config = {"Monitor Identity": {"width": 1920, "height": 1080}}
    gui.target_fps_dd = SimpleNamespace(value="0")
    gui.stream_calibration_mode_dd = SimpleNamespace(value="")
    gui._calibration_previous_target_value = "Auto"
    gui._calibration_active = True
    gui._calibration_dialog = None

    monkeypatch.setattr(gui_process, "STREAM_CALIBRATION_PROFILE_FILE", str(profile_path))
    monkeypatch.setattr(gui_process, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(gui_process, "save_yaml", lambda *args: (True, ""))

    asyncio.run(gui._apply_stream_calibration_profile())

    assert "restore" not in events
    assert gui._config["Stream Target Bitrate Mbps"] == 5
    assert "Calibration applied: 30 FPS, 5 Mbps" in events


def test_page_close_applies_completed_calibration_before_stopping(monkeypatch, tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    events = []

    class Harness(gui_process.GUIProcessMixin):
        async def _apply_stream_calibration_profile(self):
            events.append("apply")
            self._calibration_active = False

        async def _async_stop(self):
            events.append("stop")

    gui = Harness()
    gui._closed = False
    gui._calibration_active = True
    gui._calibration_poll_task = None
    gui._esc_task = None
    gui._log_poll_task = None
    monkeypatch.setattr(gui_process, "STREAM_CALIBRATION_STATE_FILE", str(state_path))

    asyncio.run(gui._on_page_close())

    assert events == ["apply", "stop"]


def test_display_refresh_warning_uses_non_stopping_modal_dialog():
    shown = []
    popped = []

    class Harness(gui_process.GUIProcessMixin):
        pass

    gui = Harness()
    gui.locale = "CN"
    gui.page = SimpleNamespace(
        show_dialog=lambda dialog: shown.append(dialog),
        pop_dialog=lambda: popped.append(True),
    )
    gui._display_refresh_warning_dialog = None

    gui._show_display_refresh_warning({"refresh_hz": 30, "sbs_fps": 58.2})
    gui._show_display_refresh_warning({"refresh_hz": 30, "sbs_fps": 58.2})

    assert len(shown) == 1
    assert shown[0].modal is True
    assert "30 Hz" in shown[0].content.controls[0].value
    assert "继续运行" in shown[0].content.controls[1].value

    gui._close_display_refresh_warning()
    assert popped == [True]
    assert gui._display_refresh_warning_dialog is None


def test_input_refresh_warning_is_queued_behind_output_warning():
    shown = []
    popped = []

    class Harness(gui_process.GUIProcessMixin):
        pass

    gui = Harness()
    gui.locale = "CN"
    gui.page = SimpleNamespace(
        show_dialog=lambda dialog: shown.append(dialog),
        pop_dialog=lambda: popped.append(True),
    )
    gui._display_refresh_warning_dialog = None
    gui._display_refresh_warning_payload = None
    gui._pending_display_refresh_warnings = []

    gui._show_display_refresh_warning(
        {"kind": "output", "refresh_hz": 30, "sbs_fps": 58.2}
    )
    gui._show_display_refresh_warning(
        {
            "kind": "input_capture_target",
            "refresh_hz": 60,
            "capture_target": 64,
        }
    )

    assert len(shown) == 1
    gui._close_display_refresh_warning()
    assert len(shown) == 2
    assert "输入显示器" in shown[1].content.controls[0].value
    assert "60 Hz" in shown[1].content.controls[0].value
    assert "64 FPS" in shown[1].content.controls[0].value

    gui._close_display_refresh_warning()
    assert popped == [True, True]
    assert gui._display_refresh_warning_dialog is None


def test_child_refresh_warning_payload_opens_dialog():
    payloads = []

    class Harness(gui_process.GUIProcessMixin):
        def _show_display_refresh_warning(self, payload):
            payloads.append(payload)

    gui = Harness()
    gui._log_child_line(
        '[D2S_DISPLAY_REFRESH_WARNING] {"refresh_hz": 30, "sbs_fps": 58.2}'
    )

    assert payloads == [{"refresh_hz": 30, "sbs_fps": 58.2}]


def test_running_ui_does_not_redraw_enabled_stop_button():
    updates = []

    class Harness(gui_process.GUIProcessMixin):
        def _safe_update(self, *controls):
            updates.extend(controls)

    gui = Harness()
    gui.run_btn = SimpleNamespace(disabled=False)
    gui.stop_btn = SimpleNamespace(disabled=False)
    gui.stream_calibration_btn = SimpleNamespace(disabled=False)

    gui._set_running_ui(True)

    assert gui.run_btn.disabled is True
    assert gui.stop_btn.disabled is False
    assert gui.stream_calibration_btn.disabled is True
    assert gui.stop_btn not in updates

    updates.clear()
    gui._set_running_ui(True)
    assert updates == []


def test_startup_restores_profile_values_after_shutdown_race(monkeypatch, tmp_path):
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps({
        "fps": 30,
        "target_mbps": 25,
        "peak_mbps": 29,
        "stability": "stable",
        "fingerprint": {"Streamer Port": "1122"},
    }), encoding="utf-8")
    saved = []

    class Harness(gui_process.GUIProcessMixin):
        def _stream_calibration_auto_enabled(self):
            return True

        def _target_fps_to_display(self, fps):
            return str(fps)

    gui = Harness()
    gui._config = {
        "Streamer Port": 1122,
        "Target FPS": 0,
        "Stream Target FPS": 0,
        "Use Stream Calibration": True,
        "Stream Target Bitrate Mbps": 0,
        "Stream Peak Bitrate Mbps": 0,
    }
    gui.run_mode_key = "Local Viewer"
    gui.stream_calibration_mode_dd = SimpleNamespace(value="Auto Calibration")
    gui.target_fps_dd = SimpleNamespace(value="Auto")
    monkeypatch.setattr(gui_process, "STREAM_CALIBRATION_PROFILE_FILE", str(profile_path))
    monkeypatch.setattr(gui_process, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(gui_process, "save_yaml", lambda path, config: saved.append((path, dict(config))) or (True, ""))

    gui._persist_current_stream_calibration_profile()

    assert gui._config["Target FPS"] == 0
    assert gui._config["Stream Target FPS"] == 30
    assert gui._config["Stream Target Bitrate Mbps"] == 25
    assert gui._config["Stream Peak Bitrate Mbps"] == 29
    assert gui._config["CRF"] == 23
    assert gui.target_fps_dd.value == "Auto"
    assert saved


def test_calibration_profile_status_distinguishes_stale_from_missing(monkeypatch, tmp_path):
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps({
            "fps": 30,
            "target_mbps": 25,
            "peak_mbps": 29,
            "fingerprint": {"old": "settings"},
        }),
        encoding="utf-8",
    )

    class Harness(gui_process.GUIProcessMixin):
        pass

    gui = Harness()
    gui.stream_calibration_mode_dd = SimpleNamespace(value="Auto Calibration")
    gui._config = {"Streamer Port": 1122}
    monkeypatch.setattr(gui_process, "STREAM_CALIBRATION_PROFILE_FILE", str(profile_path))

    assert gui._stream_calibration_profile_status() == "stale"

    profile_path.unlink()
    assert gui._stream_calibration_profile_status() == "missing"


def test_refresh_does_not_compare_profile_until_run(monkeypatch, tmp_path):
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps({
            "fps": 30,
            "target_mbps": 25,
            "peak_mbps": 29,
            "fingerprint": {"old": "settings"},
        }),
        encoding="utf-8",
    )

    class Harness(gui_process.GUIProcessMixin):
        def _safe_update(self, *controls):
            pass

    gui = Harness()
    gui.locale = "CN"
    gui._config = {"Streamer Port": 1122}
    gui.stream_calibration_status = SimpleNamespace(value="", color=None)
    gui.stream_calibration_warning = SimpleNamespace(value="", visible=False)
    gui.stream_calibration_warning_row = SimpleNamespace(visible=False)
    gui.stream_calibration_result = SimpleNamespace(value="", color=None, visible=False)
    gui.stream_calibration_result_row = SimpleNamespace(visible=False)
    monkeypatch.setattr(gui_process, "STREAM_CALIBRATION_PROFILE_FILE", str(profile_path))

    gui._refresh_stream_calibration_status()

    assert "30 FPS" in gui.stream_calibration_status.value
    assert gui.stream_calibration_warning.visible is False


def test_limited_calibration_profile_is_not_treated_as_current(monkeypatch, tmp_path):
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps({
            "fps": 30,
            "target_mbps": 22,
            "peak_mbps": 25,
            "stability": "limited",
            "fingerprint": gui_process.build_calibration_fingerprint({"Streamer Port": 1122}),
        }),
        encoding="utf-8",
    )

    class Harness(gui_process.GUIProcessMixin):
        pass

    gui = Harness()
    gui.stream_calibration_mode_dd = SimpleNamespace(value="Auto Calibration")
    gui._config = {"Streamer Port": 1122}
    monkeypatch.setattr(gui_process, "STREAM_CALIBRATION_PROFILE_FILE", str(profile_path))

    assert gui._stream_calibration_profile_status() == "missing"


def test_limited_calibration_result_is_visible_below_transport_profile(monkeypatch, tmp_path):
    profile_path = tmp_path / "profile.json"
    state_path = tmp_path / "state.json"
    settings = {"Streamer Port": 1122}
    created_at = 123.0
    profile_path.write_text(
        json.dumps({
            "created_at": created_at,
            "fps": 30,
            "target_mbps": 24,
            "peak_mbps": 30,
            "measured_bitrate_mbps": 21.4,
            "network_max_mbps": 30,
            "stability": "limited",
            "fingerprint": gui_process.build_calibration_fingerprint(settings),
        }),
        encoding="utf-8",
    )
    state_path.write_text(
        json.dumps({
            "status": "complete",
            "result": {"created_at": created_at, "stability": "limited"},
        }),
        encoding="utf-8",
    )

    class Harness(gui_process.GUIProcessMixin):
        def _safe_update(self, *controls):
            pass

    gui = Harness()
    gui.locale = "CN"
    gui._config = settings
    gui.stream_calibration_status = SimpleNamespace(value="", color=None)
    gui.stream_calibration_warning = SimpleNamespace(value="", visible=False)
    gui.stream_calibration_warning_row = SimpleNamespace(visible=False)
    gui.stream_calibration_result = SimpleNamespace(value="", color=None, visible=False)
    gui.stream_calibration_result_row = SimpleNamespace(visible=False)
    monkeypatch.setattr(gui_process, "STREAM_CALIBRATION_PROFILE_FILE", str(profile_path))
    monkeypatch.setattr(gui_process, "STREAM_CALIBRATION_STATE_FILE", str(state_path))

    gui._refresh_stream_calibration_status()

    assert gui.stream_calibration_result.visible is True
    assert gui.stream_calibration_result_row.visible is True
    assert "网络校准在 30 Mbps 未通过" in gui.stream_calibration_result.value
    assert "降低分辨率后重新校准" in gui.stream_calibration_result.value


def test_limited_7mbps_profile_is_shown_as_sufficient_for_1080p(
    monkeypatch, tmp_path
):
    profile_path = tmp_path / "profile.json"
    settings = {
        "Streamer Port": 1122,
        "Monitor Identity": {"width": 1920, "height": 1080},
    }
    profile_path.write_text(
        json.dumps({
            "fps": 30,
            "target_mbps": 5,
            "peak_mbps": 6,
            "network_max_mbps": 7,
            "stability": "limited",
            "input_resolution": [1920, 1080],
            "minimum_bitrate_mbps": 5,
            "metrics": {"failure_reasons": ["insufficient_probe_bitrate"]},
            "fingerprint": gui_process.build_calibration_fingerprint(settings),
        }),
        encoding="utf-8",
    )

    class Harness(gui_process.GUIProcessMixin):
        def _safe_update(self, *controls):
            pass

        def _persist_current_stream_calibration_profile(self):
            pass

    gui = Harness()
    gui.locale = "EN"
    gui._config = settings
    gui.stream_calibration_mode_dd = SimpleNamespace(value="Auto Calibration")
    gui.stream_calibration_status = SimpleNamespace(value="", color=None)
    gui.stream_calibration_warning = SimpleNamespace(value="", visible=False)
    gui.stream_calibration_warning_row = SimpleNamespace(visible=False)
    gui.stream_calibration_result = SimpleNamespace(value="", color=None, visible=False)
    gui.stream_calibration_result_row = SimpleNamespace(visible=False)
    monkeypatch.setattr(gui_process, "STREAM_CALIBRATION_PROFILE_FILE", str(profile_path))

    assert gui._stream_calibration_profile_status() == "current"
    gui._refresh_stream_calibration_status()

    assert gui.stream_calibration_status.value == "30 FPS · 5 Mbps"
    assert gui.stream_calibration_warning.visible is False
    assert gui.stream_calibration_result.visible is True
    assert "Safe bitrate 5 Mbps meets this resolution" in gui.stream_calibration_result.value
    assert "lower the resolution" not in gui.stream_calibration_result.value


def test_stable_calibration_result_shows_network_limit_and_safe_rates(
    monkeypatch, tmp_path
):
    fit_calls = []
    profile_path = tmp_path / "profile.json"
    settings = {"Streamer Port": 1122}
    profile_path.write_text(
        json.dumps({
            "fps": 30,
            "target_mbps": 32,
            "peak_mbps": 36,
            "network_max_mbps": 40,
            "stability": "stable",
            "fingerprint": gui_process.build_calibration_fingerprint(settings),
        }),
        encoding="utf-8",
    )

    class Harness(gui_process.GUIProcessMixin):
        def _safe_update(self, *controls):
            pass

        def _fit_window_to_content(self, update=True, resize_window=False):
            fit_calls.append((update, resize_window))

    gui = Harness()
    gui.locale = "CN"
    gui._config = settings
    gui.stream_calibration_status = SimpleNamespace(value="", color=None)
    gui.stream_calibration_warning = SimpleNamespace(value="", visible=False)
    gui.stream_calibration_warning_row = SimpleNamespace(visible=False)
    gui.stream_calibration_result = SimpleNamespace(value="", color=None, visible=False)
    gui.stream_calibration_result_row = SimpleNamespace(visible=False)
    monkeypatch.setattr(gui_process, "STREAM_CALIBRATION_PROFILE_FILE", str(profile_path))

    gui._refresh_stream_calibration_status()

    assert "网络稳定上限：40 Mbps" in gui.stream_calibration_result.value
    assert "安全码率：32 Mbps" in gui.stream_calibration_result.value
    assert "峰值" not in gui.stream_calibration_result.value
    assert "帧率 30 FPS" in gui.stream_calibration_result.value
    assert fit_calls == [(False, True)]


def test_backend_status_payload_is_rendered_as_read_only_telemetry():
    control = SimpleNamespace(value="", visible=False)
    bar = SimpleNamespace(visible=False)
    updates = []

    class Harness(gui_process.GUIProcessMixin):
        backend_status_text = control
        _backend_status_bar = bar

        def _safe_update(self, *controls):
            updates.extend(controls)

    gui = Harness()
    gui._set_backend_status({
        "depth_backend": "pytorch_cuda",
        "stereo_backend": "vulkan",
        "fallback": True,
        "fallback_reasons": ["TensorRT unavailable"],
        "gpu_to_cpu": True,
        "gpu_copy_count": 1,
        "zero_copy": False,
    })

    assert control.visible is True
    assert bar.visible is True
    assert "深度=pytorch_cuda" in control.value
    assert "GPU复制=1" in control.value
    assert "TensorRT unavailable" in control.value
    assert updates == [control, bar]
