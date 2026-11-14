from __future__ import annotations

from types import SimpleNamespace

import gui.handlers as handlers


def _control():
    return SimpleNamespace(visible=True, disabled=False, value="")


def test_file_source_hides_capture_tool_and_all_main_gui_media_controls(monkeypatch):
    monkeypatch.setattr(handlers, "OS_NAME", "Windows")
    gui = handlers.GUIHandlerMixin()
    gui.capture_mode_key = "File"
    gui.media_path = "C:/Videos/movie.mp4"
    for name in (
        "monitor_dd",
        "window_dd",
        "capture_tool_label",
        "capture_tool_dd",
        "_row8_spacer",
        "media_controls_row",
        "media_transport_row",
        "media_seek_row",
        "media_path_text",
        "media_open_group",
        "media_open_file_label",
        "media_open_file_btn",
        "media_open_disc_label",
        "media_open_folder_btn",
        "refresh_btn",
        "media_rewind_btn",
        "media_play_btn",
        "media_forward_btn",
        "media_seek_slider",
        "media_position_text",
        "media_mute_btn",
        "media_volume_label",
        "media_volume_slider",
        "media_status_text",
    ):
        setattr(gui, name, _control())
    gui._safe_update = lambda *_controls: None
    gui._fit_window_to_content = lambda **_kwargs: None

    gui._sync_capture_mode_visibility()

    assert gui.capture_tool_label.visible is True
    assert gui.capture_tool_dd.visible is True
    assert gui.capture_tool_dd.disabled is True
    assert gui.media_controls_row.visible is False
    assert gui.media_transport_row.visible is False
    assert gui.media_seek_row.visible is False
    assert gui.media_path_text.visible is False
    assert gui.media_open_group.visible is False
    assert gui.media_open_file_label.visible is False
    assert gui.media_open_file_btn.visible is False
    assert gui.media_open_disc_label.visible is False
    assert gui.media_open_folder_btn.visible is False
    assert gui.refresh_btn.visible is False
    assert gui.media_rewind_btn.visible is False
    assert gui.media_play_btn.visible is False
    assert gui.media_forward_btn.visible is False
    assert gui.media_seek_slider.visible is False
    assert gui.media_position_text.visible is False
    assert gui.media_mute_btn.visible is False
    assert gui.media_volume_label.visible is False
    assert gui.media_volume_slider.visible is False
    assert gui.media_status_text.visible is False

    gui.capture_mode_key = "Monitor"
    for control in (gui.capture_tool_label, gui.capture_tool_dd):
        control.visible = False
    gui._sync_capture_mode_visibility()
    assert gui.capture_tool_label.visible is True
    assert gui.capture_tool_dd.visible is True
    assert gui.capture_tool_dd.disabled is False
