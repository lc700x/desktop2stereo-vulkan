from __future__ import annotations

from types import SimpleNamespace

import gui.handlers as handlers


def _control():
    return SimpleNamespace(visible=True, disabled=False, value="")


def test_file_source_hides_capture_tool_and_all_main_gui_media_controls(monkeypatch):
    for os_name, capture_tool_supported in (
        ("Windows", True),
        ("Darwin", True),
        ("Linux", False),
    ):
        monkeypatch.setattr(handlers, "OS_NAME", os_name)
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

        assert gui.capture_tool_label.visible is False
        assert gui.capture_tool_dd.visible is False
        assert gui.capture_tool_dd.disabled is True
        for name in (
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
            assert getattr(gui, name).visible is False

        gui.capture_mode_key = "Monitor"
        gui._sync_capture_mode_visibility()
        assert gui.capture_tool_label.visible is capture_tool_supported
        assert gui.capture_tool_dd.visible is capture_tool_supported
        assert gui.capture_tool_dd.disabled is False
