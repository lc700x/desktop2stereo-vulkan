import asyncio
from pathlib import Path

import flet as ft

from path_config import APP_ROOT

from gui.config import DEFAULTS
from gui.controls import CompactDisplayField, CompactDropdown, S
from gui.builders import GUIBuilderMixin


ROOT = Path(__file__).resolve().parents[1]
BUILDERS_SOURCE = APP_ROOT / "gui" / "builders.py"
HANDLERS_SOURCE = APP_ROOT / "gui" / "handlers.py"
GUI_SOURCE = APP_ROOT / "gui" / "gui.py"


def test_left_settings_area_uses_a_bounded_scroll_viewport() -> None:
    source = BUILDERS_SOURCE.read_text(encoding="utf-8")
    scroll_start = source.index("scroll_area = ft.Column([")
    scroll_end = source.index("self.log_level_dd", scroll_start)
    scroll_source = source[scroll_start:scroll_end]

    assert "scroll=ft.ScrollMode.AUTO" in scroll_source
    assert "expand=True" in scroll_source


def test_compact_dropdown_setters_tolerate_unattached_flet_controls() -> None:
    dropdown = CompactDropdown(options=["Local Viewer"], value="Local Viewer")

    def fail_update():
        raise AssertionError("Control must be added to the page first")

    dropdown.update = fail_update
    dropdown.value = "RTMP Streamer"
    dropdown.options = ["RTMP Streamer", "Local Viewer"]
    dropdown.set_tooltip("Run mode")

    assert dropdown.value == "RTMP Streamer"
    assert dropdown.options == ["RTMP Streamer", "Local Viewer"]


def test_compact_dropdown_update_is_safe_before_page_mount() -> None:
    dropdown = CompactDropdown(options=["Local Viewer"], value="Local Viewer")
    # Flet raises AssertionError for the inherited update() before mounting.
    dropdown.update()


def test_native_window_uses_a_visible_compact_startup_surface() -> None:
    source = GUI_SOURCE.read_text(encoding="utf-8")
    async_main_start = source.index("async def _async_main(page: ft.Page):")
    app_start = source.index("app = Desktop2StereoGUI(page)", async_main_start)
    bootstrap = source[async_main_start:app_start]

    assert "page.window.visible = False" not in bootstrap
    assert "page.window.width = S(520)" in bootstrap
    assert "page.window.height = S(300)" in bootstrap
    assert "ft.ProgressRing" in bootstrap
    assert "Desktop2Stereo is starting..." in bootstrap
    assert "page.update()" in bootstrap
    assert "await asyncio.sleep(0.1)" in bootstrap
    assert bootstrap.index("await asyncio.sleep(0.1)") < bootstrap.index(
        "_write_gui_ready_flag()"
    )


def test_startup_audio_detection_is_deferred_until_after_window_show() -> None:
    source = GUI_SOURCE.read_text(encoding="utf-8")
    setup_start = source.index("async def setup(self):")
    setup_end = source.index("def _signal_gui_ready", setup_start)
    setup_source = source[setup_start:setup_end]

    show_index = setup_source.index("self.page.window.visible = True")
    audio_index = setup_source.index("self.populate_audio_devices_after_startup()")
    assert show_index < audio_index

    handlers = HANDLERS_SOURCE.read_text(encoding="utf-8")
    async_start = handlers.index("async def populate_audio_devices_after_startup")
    async_end = handlers.index("def _populate_audio_generic", async_start)
    assert "await asyncio.to_thread" in handlers[async_start:async_end]


def test_torch_device_detection_is_deferred_until_after_window_show() -> None:
    source = GUI_SOURCE.read_text(encoding="utf-8")
    setup_start = source.index("async def setup(self):")
    setup_end = source.index("def _signal_gui_ready", setup_start)
    setup_source = source[setup_start:setup_end]

    assert "self.populate_devices()" not in setup_source
    show_index = setup_source.index("self.page.window.visible = True")
    device_index = setup_source.index("self.populate_devices_after_startup()")
    assert show_index < device_index

    builders = BUILDERS_SOURCE.read_text(encoding="utf-8")
    build_start = builders.index("def build_ui(self):")
    build_end = builders.index("def _build_streamer_rows", build_start)
    assert "DEVICES.values()" not in builders[build_start:build_end]
    assert 'options=["Detecting compute devices..."]' in builders[build_start:build_end]
    assert "async def populate_devices_after_startup" in builders


def test_left_scroll_area_contains_the_action_footer() -> None:
    source = BUILDERS_SOURCE.read_text(encoding="utf-8")

    assert "scroll_area.controls.append(footer)" in source
    assert "content=scroll_area" in source


def test_stop_and_run_buttons_are_inset_from_the_right_edge() -> None:
    source = BUILDERS_SOURCE.read_text(encoding="utf-8")
    row_start = source.index("btn_row = ft.Row(")
    row_end = source.index("self._btn_bar =", row_start)

    row_source = source[row_start:row_end]
    assert "padding=ft.Padding(0, 0, S(40), 0)" in row_source
    assert "spacing=S(20)" in row_source


def test_stop_and_run_buttons_have_matching_widths() -> None:
    source = BUILDERS_SOURCE.read_text(encoding="utf-8")
    controls_start = source.index("self.stop_btn = ft.Button(")
    controls_end = source.index("lang_row =", controls_start)
    controls_source = source[controls_start:controls_end]

    assert controls_source.count("width=S(130)") == 2


def test_main_panels_stretch_to_the_page_height() -> None:
    source = BUILDERS_SOURCE.read_text(encoding="utf-8")
    root_start = source.index("self._root_row = ft.Row(")
    root_end = source.index("page.add(self._root_row)", root_start)
    root_source = source[root_start:root_end]

    assert "vertical_alignment=ft.CrossAxisAlignment.STRETCH" in root_source


def test_startup_fit_is_not_cancelled_by_an_early_resize_event(monkeypatch) -> None:
    class WindowHarness:
        def __init__(self):
            self.width = 0
            self.update_calls = 0

        def update(self):
            self.update_calls += 1

    class PageHarness:
        def __init__(self):
            self.window = WindowHarness()

    class StartupFitHarness(GUIBuilderMixin):
        def __init__(self):
            self._closed = False
            self._startup_fit_armed = True
            self.fit_calls = 0
            self.page = PageHarness()
            self.run_mode_key = "RTMP Streamer"
            self.log_panel = None

        def _estimate_main_panel_width(self):
            return 513

        def _estimate_window_width(self, main_width=None):
            return 568

        async def _resize_window_after_log_visibility_change(self):
            self.fit_calls += 1
            self.page.window.width = 568

    async def no_wait(_delay):
        return None

    monkeypatch.setattr("gui.builders.asyncio.sleep", no_wait)
    gui = StartupFitHarness()

    asyncio.run(gui._apply_startup_fit())

    assert gui.fit_calls == 2
    assert gui.page.window.update_calls == 1
    assert gui.page.window.width == 568
    assert gui._startup_fit_armed is False


def test_stream_calibration_uses_the_shared_label_and_control_widths() -> None:
    source = BUILDERS_SOURCE.read_text(encoding="utf-8")
    align_start = source.index("left_labels = [")
    align_end = source.index("right_labels = [", align_start)
    row_start = source.index("self.stream_calibration_mode_dd = CompactDropdown(")
    row_end = source.index("self.crf_label =", row_start)

    assert "self.stream_calibration_label" in source[align_start:align_end]
    assert source[row_start:row_end].count("width=S(130)") == 2
    row_controls_start = source.index("self.stream_calibration_row = ft.Row(")
    row_controls_end = source.index("self.crf_label =", row_controls_start)
    row_controls = source[row_controls_start:row_controls_end]
    assert row_controls.index("self.stream_calibration_mode_dd") < row_controls.index(
        "self.stream_calibration_btn"
    )
    between = row_controls[
        row_controls.index("self.stream_calibration_mode_dd"):
        row_controls.index("self.stream_calibration_btn")
    ]
    assert "ft.Container(width=S(10))" in between
    assert "self.stream_calibration_status" not in row_controls

    warning_row_start = source.index("self.stream_calibration_warning_row = ft.Row(")
    warning_row_end = source.index("self.stream_calibration_result = ft.Text(", warning_row_start)
    warning_row = source[warning_row_start:warning_row_end]
    assert "[self.stream_calibration_warning]" in warning_row
    assert "ft.Container" not in warning_row

    result_row_start = source.index("self.stream_calibration_result_row = ft.Row(")
    result_row_end = source.index("self.stream_calibration_row = ft.Row(", result_row_start)
    result_row = source[result_row_start:result_row_end]
    assert "[self.stream_calibration_result]" in result_row
    assert "ft.Container" not in result_row


def test_stream_url_field_has_a_bounded_width() -> None:
    source = BUILDERS_SOURCE.read_text(encoding="utf-8")
    field_start = source.index("self.stream_url_tf = CompactDisplayField(")
    field_end = source.index("self.preview_btn =", field_start)
    field_source = source[field_start:field_end]

    assert "min_width=S(130)" in field_source
    assert "max_width=S(230)" in field_source


def test_stream_url_updates_while_stream_settings_are_collapsed() -> None:
    source = HANDLERS_SOURCE.read_text(encoding="utf-8")
    start = source.index("def update_stream_url")
    end = source.index("def _on_stream_protocol_change", start)
    handler = source[start:end]

    assert "if not self.stream_container.visible" not in handler
    assert 'self.run_mode_key not in {"MJPEG Streamer", "RTMP Streamer"}' in handler
    assert "self.stream_url_tf.value = self._format_stream_url" in handler


def test_advanced_streaming_exposes_shared_calibration_rows() -> None:
    source = BUILDERS_SOURCE.read_text(encoding="utf-8")
    start = source.index("def _get_streamer_row_map")
    end = source.index("\n    # ── data population ──", start)
    row_map = source[start:end]

    assert '"RTMP Streamer": [0, 1, 2, 3, 5, 6, 7, 8]' in row_map
    assert '"MJPEG Streamer": [0, 5]' in row_map
    assert '"GPU Streamer"' not in row_map


def test_compact_display_field_adapts_between_minimum_and_maximum_widths() -> None:
    field = CompactDisplayField("short", min_width=S(130), max_width=S(230))
    assert field.width == S(130)

    field.value = "https://example.test/" + "stream/" * 100
    assert field.width == S(230)
    estimator = object.__new__(GUIBuilderMixin)
    assert estimator._estimate_control_width(field) == S(230)


def test_window_height_reserves_the_complete_action_footer() -> None:
    source = BUILDERS_SOURCE.read_text(encoding="utf-8")
    estimator_start = source.index("def _estimate_window_height")
    estimator_end = source.index("# ── label alignment", estimator_start)
    estimator_source = source[estimator_start:estimator_end]

    assert "footer_height = S(104)" in estimator_source
    assert "safety_margin = S(0)" in estimator_source
    assert "max_height = S(1040)" in estimator_source


def test_group_height_counts_dynamic_calibration_result_row() -> None:
    estimator = object.__new__(GUIBuilderMixin)
    primary_row = ft.Row([ft.Text("Transmission Profile")])
    result_row = ft.Row([ft.Text("Calibration result")], visible=False)
    group = ft.Container(ft.Column([primary_row, result_row], spacing=S(8)), visible=True)

    height_without_result = estimator._estimate_group_height(group, include_margin=False)
    result_row.visible = True
    height_with_result = estimator._estimate_group_height(group, include_margin=False)

    assert height_with_result - height_without_result == S(34) + S(8)


def test_calibration_messages_are_kept_on_one_line() -> None:
    source = BUILDERS_SOURCE.read_text(encoding="utf-8")
    warning_start = source.index("self.stream_calibration_warning = ft.Text(")
    result_end = source.index("self.stream_calibration_result_row = ft.Row(", warning_start)
    message_controls = source[warning_start:result_end]

    assert message_controls.count("no_wrap=True") == 2


def test_window_preview_is_an_independent_advanced_checkbox_after_vsync() -> None:
    source = BUILDERS_SOURCE.read_text(encoding="utf-8")
    row_start = source.index("self.row6b = ft.Row(")
    row_end = source.index("self.render_policy_label", row_start)
    row_source = source[row_start:row_end]

    assert "self.local_vsync_cb" in row_source
    assert "self.window_preview_cb" in row_source
    assert row_source.index("self.local_vsync_cb") < row_source.index("self.window_preview_cb")


def test_reset_defaults_disable_depth_antialiasing() -> None:
    source = BUILDERS_SOURCE.read_text(encoding="utf-8")

    assert DEFAULTS["Anti-aliasing"] == 0
    assert DEFAULTS["Depth Antialias Strength"] == 0.0
    assert DEFAULTS["CRF"] == 23
    assert DEFAULTS["Stream Protocol"] == "WebRTC"
    assert 'options=[v for v in aa_options], value="0"' in source


def test_crf_tooltip_includes_the_recommended_range() -> None:
    source = (APP_ROOT / "gui" / "localization.py").read_text(encoding="utf-8")

    assert '30 Mbps or higher -> CRF 20' in source
    assert '25-29 Mbps -> CRF 23' in source
    assert '30 Mbps 及以上建议 CRF 20' in source
    assert '25-29 Mbps 建议 CRF 23' in source
    assert '低于 19 Mbps 不建议继续提高 CRF' in source


def test_every_stream_parameter_control_has_an_explanatory_tooltip() -> None:
    localization = (APP_ROOT / "gui" / "localization.py").read_text(encoding="utf-8")
    handlers = HANDLERS_SOURCE.read_text(encoding="utf-8")
    tooltip_bindings = {
        "stream_settings_cb": "tooltip_stream_settings",
        "stream_url_tf": "tooltip_stream_url",
        "preview_btn": "tooltip_stream_preview",
        "stream_port_tf": "tooltip_stream_port",
        "stream_quality_dd": "tooltip_stream_quality",
        "stream_proto_dd": "tooltip_stream_proto",
        "stream_key_tf": "tooltip_stream_key",
        "crf_tf": "tooltip_crf",
        "audio_delay_tf": "tooltip_audio_delay",
        "audio_dd": "tooltip_audio",
        "video_backend_dd": "tooltip_video_backend",
        "stream_calibration_mode_dd": "tooltip_stream_calibration_mode",
        "stream_calibration_btn": "tooltip_stream_calibration_start",
    }

    for control, tooltip_key in tooltip_bindings.items():
        assert f'(self.{control}, "{tooltip_key}")' in handlers
        assert localization.count(f'"{tooltip_key}":') == 2

    assert "高级网络推流使用 CRF 和码率控制" in localization
    assert "WebRTC 推荐用于低延迟头显浏览器" in localization
    assert "正数延后音频，负数提前音频" in localization


def test_model_and_mode_tooltips_explain_recommendations_and_tradeoffs() -> None:
    localization = (APP_ROOT / "gui" / "localization.py").read_text(encoding="utf-8")

    assert "Distill-Any-Depth first" in localization
    assert "InfiniDepth second" in localization
    assert "首先使用 Distill-Any-Depth" in localization
    assert "其次选择 InfiniDepth" in localization
    assert "Monitor captures the selected full display" in localization
    assert "屏幕模式捕获选定显示器的完整画面" in localization
    assert "Advanced Streaming publishes compressed H.264/H.265" in localization
    assert "高级网络推流：通过 WebRTC/RTSP/RTMP 发布 H.264/H.265" in localization
    assert "Half-SBS packs left/right views" in localization
    assert "Half-SBS：左右眼并排" in localization


def test_run_mode_change_refits_native_window_height() -> None:
    source = HANDLERS_SOURCE.read_text(encoding="utf-8")
    start = source.index("def on_run_mode_change")
    end = source.index("def on_advanced_device_change", start)
    handler = source[start:end]

    assert "self._fit_window_to_content(update=True, resize_window=True)" in handler


def test_advanced_sections_refit_native_window_height() -> None:
    source = HANDLERS_SOURCE.read_text(encoding="utf-8")
    device_start = source.index("def on_advanced_device_change")
    device_end = source.index("def on_render_policy_change", device_start)
    stereo_start = source.index("def on_advanced_stereo_change")
    stereo_end = source.index("def _sync_advanced_stereo_visibility", stereo_start)

    expected = "self._fit_window_to_content(update=True, resize_window=True)"
    assert expected in source[device_start:device_end]
    assert expected in source[stereo_start:stereo_end]


def test_stream_settings_checkbox_controls_stream_parameter_panel() -> None:
    builders = BUILDERS_SOURCE.read_text(encoding="utf-8")
    handlers = HANDLERS_SOURCE.read_text(encoding="utf-8")
    run_row_start = builders.index("self.row7a = ft.Row(")
    run_row_end = builders.index("self.xr_headset_row = ft.Row(", run_row_start)
    run_row = builders[run_row_start:run_row_end]
    mode_change_start = handlers.index("def on_run_mode_change")
    toggle_start = handlers.index("def on_stream_settings_change", mode_change_start)
    advanced_start = handlers.index("def on_advanced_device_change", toggle_start)
    mode_change = handlers[mode_change_start:toggle_start]
    toggle = handlers[toggle_start:advanced_start]
    visibility_start = handlers.index("def _sync_visibility(self):")
    visibility_end = handlers.index("def ", visibility_start + 4)
    visibility = handlers[visibility_start:visibility_end]

    assert run_row.index("self.run_mode_dd") < run_row.index("self.stream_settings_cb")
    assert 'label="Stream Settings"' in builders
    assert 'value=False' in builders
    assert "self.stream_settings_cb.value = False" in mode_change
    assert "self.stream_settings_cb.visible = is_streamer" in visibility
    assert "self.stream_settings_cb.value" in visibility
    assert "self._show_streamer_rows(*row_indices)" in toggle
    assert "resize_window=True" in toggle


def test_reset_defaults_refits_window_to_left_gui_content() -> None:
    source = (APP_ROOT / "gui" / "process.py").read_text(encoding="utf-8")
    reset_start = source.index("def reset_defaults")
    reset_end = source.index("# ── URL actions ──", reset_start)
    reset_source = source[reset_start:reset_end]

    assert "self._fit_window_to_content(update=True, resize_window=True)" in reset_source


def test_display_mode_is_to_the_right_of_headset_model() -> None:
    source = BUILDERS_SOURCE.read_text(encoding="utf-8")
    run_row_start = source.index("self.row7a = ft.Row(")
    headset_row_start = source.index("self.xr_headset_row = ft.Row(")
    headset_row_end = source.index("self.row7b = ft.Row(", headset_row_start)
    assembly_start = source.index("device_group = ft.Container(")
    assembly_end = source.index("lang_group = ft.Container(", assembly_start)
    assembly = source[assembly_start:assembly_end]

    run_row = source[run_row_start:headset_row_start]
    headset_row = source[headset_row_start:headset_row_end]
    assert "self.display_mode_label" not in run_row
    assert headset_row.index("self.xr_headset_label") < headset_row.index("self.display_mode_label")
    assert headset_row.index("self.xr_headset_dd") < headset_row.index("self.display_mode_dd")
    assert assembly.index("self.row7a") < assembly.index("self.xr_headset_row")
    assert assembly.index("self.xr_headset_row") < assembly.index("self.row7b")


def test_headset_model_remains_visible_in_every_run_mode() -> None:
    source = HANDLERS_SOURCE.read_text(encoding="utf-8")
    start = source.index("def _sync_visibility(self):")
    end = source.index("def ", start + 4)
    visibility = source[start:end]

    assert "self.xr_headset_label.visible = True" in visibility
    assert "self.xr_headset_dd.visible = True" in visibility


def test_refresh_button_follows_the_capture_source_dropdown() -> None:
    source = BUILDERS_SOURCE.read_text(encoding="utf-8")
    row_start = source.index("row8 = ft.Row(")
    row_end = source.index("# Row 10:", row_start)
    row_source = source[row_start:row_end]

    assert "ft.Container(expand=True)" not in row_source
    assert row_source.index("self.monitor_dd") < row_source.index("self.refresh_btn")
    assert row_source.index("self.window_dd") < row_source.index("self.refresh_btn")
