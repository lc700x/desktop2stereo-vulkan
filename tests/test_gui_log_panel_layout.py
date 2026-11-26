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
    body_start = source.index("self.log_body = ft.Container(", viewport_start)
    scroll_source = source[viewport_start:body_start]

    assert "width=S(900)" in scroll_source
    assert "orientation=ft.ScrollbarOrientation.RIGHT" in scroll_source
    assert "self.log_scroll_row = ft.Row(" in scroll_source
    assert "expand=True" in scroll_source
    assert "orientation=ft.ScrollbarOrientation.BOTTOM" in scroll_source
    assert "vertical_alignment=ft.CrossAxisAlignment.STRETCH" in scroll_source
    assert "content=self.log_scroll_row" in source[body_start:]


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
