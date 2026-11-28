from types import SimpleNamespace

import flet as ft
from gui.controls import S
from gui.process import GUIProcessMixin
from path_config import APP_ROOT


BUILDERS_SOURCE = APP_ROOT / "gui" / "builders.py"


def test_log_panel_matches_left_control_group_insets() -> None:
    source = BUILDERS_SOURCE.read_text(encoding="utf-8")
    device_start = source.index("device_group = ft.Container(")
    device_end = source.index("self.device_group = device_group", device_start)
    log_start = source.index("self.log_panel = ft.Container(")
    log_end = source.index("btn_row = ft.Row(", log_start)

    expected_padding = (
        "padding=ft.Padding(PANEL_PADDING_X, PANEL_PADDING_Y, "
        "PANEL_PADDING_X, PANEL_PADDING_Y)"
    )
    for group_source in (source[device_start:device_end], source[log_start:log_end]):
        assert expected_padding in group_source
        assert "margin=ft.Margin" not in group_source


def test_log_horizontal_scrollbar_is_pinned_to_the_panel_bottom() -> None:
    source = BUILDERS_SOURCE.read_text(encoding="utf-8")
    viewport_start = source.index("self.log_viewport = ft.Column(")
    scroll_end = source.index("self.log_panel = ft.Container(", viewport_start)
    scroll_source = source[viewport_start:scroll_end]

    assert "width=S(900)" in scroll_source
    assert "orientation=ft.ScrollbarOrientation.RIGHT" in scroll_source
    assert "thumb_visibility=False" in scroll_source
    assert "track_visibility=False" in scroll_source
    assert "thickness=0" in scroll_source
    assert "on_scroll=self.on_log_viewport_scroll" in scroll_source
    assert "self.log_scroll_row = ft.Row(" in scroll_source
    assert "expand=True" in scroll_source
    assert "orientation=ft.ScrollbarOrientation.BOTTOM" in scroll_source
    assert "vertical_alignment=ft.CrossAxisAlignment.STRETCH" in scroll_source
    assert "self.log_vertical_scrollbar = ft.GestureDetector(" in scroll_source
    assert "right=0" in scroll_source
    assert "bottom=S(6)" in scroll_source
    assert "ft.Colors.with_opacity(0.62, ft.Colors.WHITE)" in scroll_source
    assert "ft.Colors.with_opacity(0.14, ft.Colors.WHITE)" in scroll_source
    assert "on_hover=self.on_log_vertical_scrollbar_hover" in scroll_source
    assert "[self.log_scroll_row, self.log_scroll_overlay]" in scroll_source
    assert "on_size_change=self.on_log_body_size_change" in scroll_source


def test_log_panel_expands_to_the_available_window_width() -> None:
    source = BUILDERS_SOURCE.read_text(encoding="utf-8")
    panel_start = source.index("self.log_panel = ft.Container(")
    panel_end = source.index("btn_row = ft.Row(", panel_start)
    panel_source = source[panel_start:panel_end]

    assert "visible=False" in panel_source
    assert "expand=True" in panel_source
    assert "self.log_body_stack = ft.Stack(" in source


def test_vertical_scrollbar_geometry_tracks_overflow_and_position() -> None:
    max_extent, thumb_height, thumb_top, visible = GUIProcessMixin._log_vertical_scrollbar_geometry(
        2000,
        500,
        300,
        750,
    )

    assert max_extent == 1500
    assert thumb_height == 75
    assert thumb_top == 112.5
    assert visible is True

    max_extent, thumb_height, thumb_top, visible = GUIProcessMixin._log_vertical_scrollbar_geometry(
        400,
        500,
        300,
        0,
    )

    assert max_extent == 0
    assert thumb_height == 0
    assert thumb_top == 0
    assert visible is False


def test_vertical_scrollbar_hover_keeps_the_default_white_highlight() -> None:
    thumb = SimpleNamespace(bgcolor=None)
    updates = []
    gui = SimpleNamespace(
        log_vertical_thumb=thumb,
        _safe_update=lambda *controls: updates.extend(controls),
    )

    GUIProcessMixin.on_log_vertical_scrollbar_hover(gui, SimpleNamespace(data="true"))
    assert thumb.bgcolor == ft.Colors.with_opacity(0.92, ft.Colors.WHITE)
    assert updates == [thumb]

    GUIProcessMixin.on_log_vertical_scrollbar_hover(gui, SimpleNamespace(data="false"))
    assert thumb.bgcolor == ft.Colors.with_opacity(0.62, ft.Colors.WHITE)


def test_log_panel_internal_spacing_matches_left_control_groups() -> None:
    source = BUILDERS_SOURCE.read_text(encoding="utf-8")
    panel_start = source.index("self.log_panel = ft.Container(")
    panel_end = source.index("btn_row = ft.Row(", panel_start)
    panel_source = source[panel_start:panel_end]

    assert "spacing=ACTION_GAP, vertical_alignment=ft.CrossAxisAlignment.CENTER" in panel_source
    assert "spacing=ROW_GAP, expand=True" in panel_source


def test_group_and_section_spacing_uses_shared_values() -> None:
    source = BUILDERS_SOURCE.read_text(encoding="utf-8")

    for marker in (
        "self._scroll_area = ft.Column([",
        "main_col = ft.Column([",
        "self._root_row = ft.Row(",
        "self._content_with_buttons = ft.Column(",
    ):
        start = source.index(marker)
        end = source.find("\n        self.", start + 1)
        block = source[start:end if end >= 0 else None]
        assert "spacing=SECTION_GAP" in block

    assert "ft.Column([self._appearance_group_heading, lang_row], spacing=ROW_GAP)" in source
    assert "self.stream_url_row], spacing=ROW_GAP" in source
