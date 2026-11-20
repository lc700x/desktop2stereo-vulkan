import inspect
import json
import logging
from string import Formatter
from types import SimpleNamespace

import flet as ft
import pytest
from gui.builders import GUIBuilderMixin, _main_menu_action_style
from gui.handlers import GUIHandlerMixin
from gui.localization import _STATUS_MESSAGES, MESSAGE_CATALOGS, translate_status
from gui.process import GUIProcessMixin


def test_media_controls_have_translations_for_every_locale():
    assert MESSAGE_CATALOGS["EN"]["OpenXR Controls"] == "XR Control"
    assert MESSAGE_CATALOGS["CN"]["OpenXR Controls"] == "XR 控件"
    assert MESSAGE_CATALOGS["EN"]["Media Player"] == "Media Player"
    assert MESSAGE_CATALOGS["CN"]["Media Player"] == "媒体播放"
    assert MESSAGE_CATALOGS["EN"]["Recent"] == "Recent"
    assert MESSAGE_CATALOGS["CN"]["Recent"] == "最近打开"
    assert MESSAGE_CATALOGS["EN"]["Open File"] == "Open File"
    assert MESSAGE_CATALOGS["CN"]["Open File"] == "打开文件"


def test_main_menu_group_titles_are_localized_and_refreshed():
    titles = {
        "Appearance & Language": ("Appearance & Language", "外观与语言"),
        "Depth & Stereo": ("Depth & Stereo", "深度与立体"),
        "Devices & Output": ("Devices & Output", "设备与输出"),
    }
    for key, expected in titles.items():
        assert tuple(MESSAGE_CATALOGS[locale][key] for locale in ("EN", "CN")) == expected

    builder_source = inspect.getsource(GUIBuilderMixin.build_ui)
    handler_source = inspect.getsource(GUIHandlerMixin.update_ui_texts)
    assert "self._appearance_group_heading, lang_row" in builder_source
    assert "self._depth_stereo_group_heading, row0" in builder_source
    assert "self._devices_output_group_heading, row5" in builder_source
    for field, key in (
        ("_appearance_group_heading", "Appearance & Language"),
        ("_depth_stereo_group_heading", "Depth & Stereo"),
        ("_devices_output_group_heading", "Devices & Output"),
    ):
        assert f"self.{field}.value = t[\"{key}\"]" in handler_source


def test_main_menu_actions_use_theme_colors_for_all_interaction_states():
    states = ft.ControlState
    primary = _main_menu_action_style("primary")
    danger = _main_menu_action_style("danger")
    secondary = _main_menu_action_style("secondary")
    expected_states = {
        states.DEFAULT,
        states.HOVERED,
        states.PRESSED,
        states.FOCUSED,
        states.DISABLED,
    }

    for style in (primary, danger, secondary):
        assert set(style.bgcolor) == expected_states
        assert set(style.color) == expected_states
        assert set(style.side) == expected_states
        assert style.elevation == 0

    assert primary.bgcolor[states.DEFAULT] == ft.Colors.PRIMARY
    assert primary.bgcolor[states.HOVERED] == ft.Colors.PRIMARY_CONTAINER
    assert primary.color[states.DEFAULT] == ft.Colors.ON_PRIMARY
    assert danger.bgcolor[states.DEFAULT] == ft.Colors.ERROR
    assert danger.bgcolor[states.HOVERED] == ft.Colors.ERROR_CONTAINER
    assert secondary.bgcolor[states.HOVERED] == ft.Colors.SURFACE_CONTAINER_HIGH
    assert secondary.side[states.FOCUSED].width == 2

    builder_source = inspect.getsource(GUIBuilderMixin.build_ui)
    for button_name, variant in (
        ("reset_btn", "secondary"),
        ("stop_btn", "danger"),
        ("run_btn", "primary"),
    ):
        start = builder_source.index(f"self.{button_name} = ft.Button(")
        end = builder_source.find("\n        self.", start + 1)
        button_source = builder_source[start:end if end >= 0 else None]
        assert f'_main_menu_action_style("{variant}")' in button_source


def test_capture_tool_change_keeps_capture_mode_locked_while_runtime_runs():
    gui = SimpleNamespace(
        process=SimpleNamespace(returncode=None),
        capture_mode_dd=SimpleNamespace(disabled=True, update=lambda: None),
        set_status=lambda *_args, **_kwargs: None,
        _sync_capture_mode_visibility=lambda: None,
    )

    GUIHandlerMixin.on_capture_tool_change(gui, None)
    assert gui.capture_mode_dd.disabled is True

    gui.process.returncode = 0
    GUIHandlerMixin.on_capture_tool_change(gui, None)
    assert gui.capture_mode_dd.disabled is False


def test_media_controls_refresh_labels_and_state_for_selected_locale():
    gui = SimpleNamespace(
        locale="CN",
        _media_runtime_state=None,
        media_open_file_label=SimpleNamespace(value="Open File"),
        media_open_folder_btn=SimpleNamespace(tooltip="Open Disc Folder"),
        media_play_btn=SimpleNamespace(tooltip="Play/Pause"),
        media_mute_btn=SimpleNamespace(tooltip="Mute"),
        media_volume_label=SimpleNamespace(value="Volume"),
        media_status_text=SimpleNamespace(value="No media loaded"),
    )

    GUIHandlerMixin._update_media_control_texts(gui)

    assert gui.media_open_file_label.value == "打开文件"
    assert gui.media_open_folder_btn.tooltip == "打开光盘文件夹"
    assert gui.media_play_btn.tooltip == "播放/暂停"
    assert gui.media_mute_btn.tooltip == "静音"
    assert gui.media_volume_label.value == "音量"
    assert gui.media_status_text.value == "尚未加载媒体"

    gui.locale = "EN"
    gui._media_runtime_state = {
        "status": "Playing",
        "playing": True,
        "muted": True,
    }
    GUIHandlerMixin._update_media_control_texts(gui)

    assert gui.media_status_text.value == "Playing"
    assert gui.media_play_btn.tooltip == "Pause"
    assert gui.media_mute_btn.tooltip == "Unmute"


def test_polled_media_status_uses_selected_locale(tmp_path):
    state_path = tmp_path / "media-state.json"
    state_path.write_text(json.dumps({
        "session": "test-session",
        "path": "C:/Videos/sample.mp4",
        "position": 1.0,
        "duration": 10.0,
        "mode": "video",
        "playing": True,
    }), encoding="utf-8")
    gui = GUIProcessMixin()
    gui._media_state_path = str(state_path)
    gui._media_session_id = "test-session"
    gui._media_path = ""
    gui._media_resume_position = 0.0
    gui._config = {}
    gui.locale = "CN"
    gui._safe_update = lambda *controls: None
    gui.media_status_text = SimpleNamespace(value="")
    gui.media_position_text = SimpleNamespace(value="")
    gui.media_seek_slider = SimpleNamespace(max=1.0, value=0.0)
    gui.media_play_btn = SimpleNamespace(icon=None, tooltip="")
    gui.media_path_text = SimpleNamespace(value="")

    assert gui._poll_media_state() is True
    assert gui.media_status_text.value == "正在播放"


@pytest.mark.parametrize("key", _STATUS_MESSAGES)
def test_status_messages_translate_both_ways(key):
    en = MESSAGE_CATALOGS["EN"][key]
    cn = MESSAGE_CATALOGS["CN"][key]
    fields = {field: "model_fp16_294x518.onnx" for _, field, _, _ in Formatter().parse(en)
              if field is not None}
    english = en.format(**fields)
    chinese = cn.format(**fields)
    assert translate_status(english, "CN") == chinese
    assert translate_status(chinese, "EN") == ("Download" if key == "download" else english)


def test_formatted_status_refresh_preserves_arguments_and_replaces_stale_key():
    class Harness(GUIProcessMixin):
        locale = "EN"
        status_text = SimpleNamespace(value="")

        def _safe_update(self, *controls):
            pass

    gui = Harness()
    gui.set_status(MESSAGE_CATALOGS["EN"]["Running"], key="Running")
    gui.set_status("Calibration result error: missing profile.json")
    gui.locale = "CN"
    gui._refresh_status_display()
    assert gui.status_text.value == "校准结果错误：missing profile.json"
    assert gui._status_key is None
    gui.set_status(MESSAGE_CATALOGS["CN"]["exited_with_code"].format(77))
    gui.locale = "EN"
    gui._refresh_status_display()
    assert gui.status_text.value == MESSAGE_CATALOGS["EN"]["exited_with_code"].format(77)
    gui.set_status("", key="")
    gui.locale = "CN"
    gui._refresh_status_display()
    assert gui.status_text.value == ""


def test_unknown_diagnostic_is_preserved():
    diagnostic = "VkStatus -4: driver detail C:/models/model.onnx"
    assert translate_status(diagnostic, "CN") == diagnostic


@pytest.mark.parametrize("label,detail", [
    ("Selected input monitor:", "1"),
    ("Selected input window:", "My video - 01"),
    ("Failed to load settings.yaml:", "C:/settings.yaml"),
])
def test_status_prefixes_refresh_without_translating_user_data(label, detail):
    en = MESSAGE_CATALOGS["EN"][label] + " " + detail
    cn = MESSAGE_CATALOGS["CN"][label] + " " + detail
    assert translate_status(en, "CN") == cn
    assert translate_status(cn, "EN") == en


def test_progress_refresh_translates_labels_and_preserves_measurements():
    gui = GUIProcessMixin()
    gui.locale = "CN"
    for name in ("panel", "title", "percent", "bar", "detail"):
        setattr(gui, f"download_progress_{name}", SimpleNamespace(value="", visible=False))
    gui._update_download_progress({
        "desc": "Exporting ONNX: model.onnx", "percent": 50,
        "downloaded": "5 steps", "size": "10 steps", "speed": "1.0 steps/s", "eta": "00:05",
    })
    assert gui.download_progress_title.value == "正在导出 ONNX：model.onnx"
    assert "5 步 / 10 步" in gui.download_progress_detail.value
    assert "预计剩余时间 00:05" in gui.download_progress_detail.value
    gui.locale = "EN"
    gui._update_download_progress(gui._download_progress_payload)
    assert gui.download_progress_title.value == "Exporting ONNX: model.onnx"
    assert "ETA 00:05" in gui.download_progress_detail.value


@pytest.mark.parametrize("state,percent,value,color", [
    ("running", None, None, ft.Colors.BLUE),
    ("complete", 100.0, 1.0, ft.Colors.GREEN),
    ("failed", None, 0.0, ft.Colors.RED),
])
def test_migraphx_activity_reaches_progress_panel_and_translates(state, percent, value, color, caplog):
    gui = GUIProcessMixin()
    gui.locale = "CN"
    for name in ("panel", "title", "percent", "bar", "detail"):
        setattr(gui, f"download_progress_{name}", SimpleNamespace(value="", visible=False))
    payload = {
        "desc": "Building MIGraphX graph: model.mgx", "percent": percent,
        "elapsed": "00:12", "unit": "activity", "state": state,
    }

    with caplog.at_level(logging.INFO, logger="child"):
        gui._log_child_line("[D2S_PROGRESS] " + json.dumps(payload))
    record = next(record for record in reversed(caplog.records) if record.name == "child")
    assert record.levelno == logging.INFO
    event = gui._progress_event((record.levelno, record.name, "", record.getMessage()))
    gui._update_download_progress(event)

    assert gui.download_progress_panel.visible is True
    assert gui.download_progress_bar.value == value
    assert gui.download_progress_bar.color == color
    assert gui.download_progress_title.value == "正在编译 MIGraphX 模型：model.mgx"
    assert gui.download_progress_detail.value == "已用时间 00:12"
    assert gui.download_progress_percent.value == (
        "操作失败" if state == "failed" else "100.0%" if state == "complete" else ""
    )
    gui.locale = "EN"
    gui._update_download_progress(gui._download_progress_payload)
    assert gui.download_progress_title.value == "Building MIGraphX graph: model.mgx"
    assert gui.download_progress_detail.value == "Elapsed 00:12"



def test_media_selection_persists_path_and_resumes_same_video(monkeypatch, tmp_path):
    import gui.process as gui_process

    class Harness(GUIHandlerMixin, GUIProcessMixin):
        def _fit_window_to_content(self):
            pass

        def _safe_update(self, *_controls):
            pass

        def set_status(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(gui_process, "BASE_DIR", str(tmp_path))
    first_video = tmp_path / "first.mp4"
    second_video = tmp_path / "second.mp4"
    first_video.write_bytes(b"video")
    second_video.write_bytes(b"video")
    gui = Harness()
    gui.locale = "EN"
    gui.capture_mode_key = "Video"
    gui.media_path = str(first_video)
    gui._media_path = str(first_video)
    gui.media_resume_position = 12.5
    gui._media_resume_position = 12.5
    gui._media_session_id = None
    gui._media_runtime_state = None
    gui._config = {"Media Path": str(first_video), "Media Playback Position": 12.5}
    gui.media_path_text = SimpleNamespace(value="")

    assert gui._set_media_source(str(first_video)) is True
    settings = gui_process.read_yaml(str(tmp_path / "settings.yaml"))
    assert settings["Media Path"] == str(first_video)
    assert settings["Media Playback Position"] == 12.5

    assert gui._set_media_source(str(second_video)) is True
    settings = gui_process.read_yaml(str(tmp_path / "settings.yaml"))
    assert settings["Media Path"] == str(second_video)
    assert settings["Media Playback Position"] == 0.0


def test_media_seek_and_pause_persist_latest_position(monkeypatch, tmp_path):
    import gui.process as gui_process

    class Harness(GUIHandlerMixin, GUIProcessMixin):
        pass

    monkeypatch.setattr(gui_process, "BASE_DIR", str(tmp_path))
    media_path = tmp_path / "movie.mp4"
    control_path = tmp_path / "media-control.json"
    gui = Harness()
    gui.locale = "EN"
    gui.capture_mode_key = "Video"
    gui.media_path = str(media_path)
    gui._media_path = str(media_path)
    gui.media_resume_position = 2.0
    gui._media_resume_position = 2.0
    gui._media_session_id = "media-session"
    gui._media_control_path = str(control_path)
    gui._media_command_session = None
    gui._media_command_revision = -1
    gui._media_runtime_state = {
        "session": "media-session",
        "path": str(media_path),
        "mode": "Video",
        "position": 2.0,
        "duration": 30.0,
        "playing": True,
    }
    gui._config = {"Media Path": str(media_path), "Media Playback Position": 2.0}
    gui.media_seek_slider = SimpleNamespace(value=18.0)

    assert gui.on_media_seek() is True
    settings = gui_process.read_yaml(str(tmp_path / "settings.yaml"))
    assert settings["Media Playback Position"] == 18.0

    assert gui.on_media_play_pause() is True
    settings = gui_process.read_yaml(str(tmp_path / "settings.yaml"))
    assert settings["Media Playback Position"] == 18.0
    assert json.loads(control_path.read_text(encoding="utf-8"))["action"] == "pause"


def test_media_play_pause_refreshes_runtime_state_before_deciding_action(tmp_path):
    class Harness(GUIHandlerMixin, GUIProcessMixin):
        pass

    media_path = tmp_path / "movie.mp4"
    control_path = tmp_path / "media-control.json"
    state_path = tmp_path / "media-state.json"
    state_path.write_text(
        json.dumps({
            "session": "media-session",
            "path": str(media_path),
            "mode": "Video",
            "position": 0.0,
            "duration": 30.0,
            "playing": False,
        }),
        encoding="utf-8",
    )
    gui = Harness()
    gui.locale = "EN"
    gui.capture_mode_key = "Video"
    gui.media_path = str(media_path)
    gui._media_path = str(media_path)
    gui._media_resume_position = 0.0
    gui._media_session_id = "media-session"
    gui._media_state_path = str(state_path)
    gui._media_control_path = str(control_path)
    gui._media_command_session = None
    gui._media_command_revision = -1
    gui._media_runtime_state = {
        "session": "media-session",
        "path": str(media_path),
        "mode": "Video",
        "position": 0.0,
        "duration": 30.0,
        "playing": True,
    }

    assert gui.on_media_play_pause() is True
    assert json.loads(control_path.read_text(encoding="utf-8"))["action"] == "play"
