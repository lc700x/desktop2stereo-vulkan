from types import SimpleNamespace

import pytest

from gui.config import DEFAULTS
from gui.handlers import GUIHandlerMixin
from gui.localization import UI_MESSAGES
from stereo_runtime.render_size import render_size_config_from_settings
from utils.bootstrap import _normalize_legacy_settings


@pytest.mark.parametrize("locale", ["EN", "CN"])
def test_quality_scale_options_and_legacy_values_are_locale_independent(locale):
    target = GUIHandlerMixin()
    target.locale = locale
    assert target._render_scale_options() == ["100%", "85%", "75%", "50%"]
    for old, new in [("4K / 100%", "100%"), ("3K / 85%", "85%"), ("2K / 75%", "75%"), ("1K / 50%", "50%")]:
        assert target._render_scale_to_display(old) == new
        assert target._display_to_render_scale(new) == new
        assert render_size_config_from_settings({"Render Scale": new}).scale_factor == new
    assert target._render_scale_to_display("invalid") == "100%"


@pytest.mark.parametrize("mode", [
    "Local Viewer", "OpenXR Link", "MJPEG Streamer", "RTMP Streamer", "Moonlight Streamer",
])
@pytest.mark.parametrize("advanced", [False, True])
def test_quality_scale_visibility_preserves_value_for_all_output_modes(mode, advanced):
    target = GUIHandlerMixin()
    target.advanced_device_cb = SimpleNamespace(value=advanced)
    target._advanced_device_rows = []
    names = [
        "row6b", "row6c", "row6d", "row6e", "row6f", "row6g",
        "render_policy_label", "render_policy_dd", "render_scale_label", "render_scale_dd",
        "render_fixed_label", "render_fixed_dd", "render_max_pixels_label", "render_max_pixels_dd",
        "render_min_dimension_label", "render_min_dimension_dd", "render_align_label", "render_align_dd",
        "openxr_render_resolution_label", "openxr_render_resolution_dd", "target_fps_label", "target_fps_dd",
        "xr_preview_cb", "local_vsync_cb", "window_preview_cb", "upscaler_label", "upscaler_dd",
        "upscaler_sharpness_label", "upscaler_sharpness_dd",
    ]
    for name in names:
        setattr(target, name, SimpleNamespace(visible=False, value="75%"))
    target._sync_device_advanced_visibility(mode)
    assert target.render_scale_dd.visible is advanced
    assert target.render_scale_label.visible is advanced
    assert target.row6d.visible is advanced
    assert target.render_scale_dd.value == "75%"
    assert target.local_vsync_cb.visible == (advanced and mode == "Local Viewer")
    assert target.openxr_render_resolution_dd.visible == (advanced and mode == "OpenXR Link")


def test_quality_scale_labels_tooltips_and_defaults():
    assert UI_MESSAGES["EN"]["Render Scale:"] == "Quality Scale:"
    assert UI_MESSAGES["CN"]["Render Scale:"] == "画质缩放:"
    assert "Stop and run again" in UI_MESSAGES["EN"]["tooltip_render_scale"]
    assert "停止后重新运行" in UI_MESSAGES["CN"]["tooltip_render_scale"]
    assert DEFAULTS["Render Scale"] == "100%"
    assert _normalize_legacy_settings({})["Render Scale"] == "100%"


def test_stream_quality_control_is_replaced_by_single_quality_scale():
    from path_config import APP_ROOT

    builders = (APP_ROOT / "gui" / "builders.py").read_text(encoding="utf-8")
    handlers = (APP_ROOT / "gui" / "handlers.py").read_text(encoding="utf-8")
    config = (APP_ROOT / "gui" / "config_mgr.py").read_text(encoding="utf-8")
    assert builders.count("self.render_scale_dd =") == 1
    assert "self.stream_quality_dd" not in builders + handlers + config
    assert "self.stream_quality_label" not in builders + handlers + config
    assert 'self._config.get("Stream Quality")' in config


def test_stream_fit_translation_is_short_and_stream_specific():
    assert UI_MESSAGES["CN"]["Stream Fit:"] == "画面适配:"
    assert UI_MESSAGES["CN"]["MJPEG Streamer"] == "低级网络推流"
    assert "推流输出画布" in UI_MESSAGES["CN"]["tooltip_stream_fit"]
    assert "stream output canvas" in UI_MESSAGES["EN"]["tooltip_stream_fit"]


@pytest.mark.parametrize("locale", ["EN", "CN"])
def test_stream_fit_shares_port_row_without_duplicate_mount(locale):
    from gui.gui import Desktop2StereoGUI
    from gui.controls import S

    app = Desktop2StereoGUI(SimpleNamespace())
    app.locale = locale
    app._build_streamer_rows()
    port_row = app.stream_port_quality_row
    assert port_row.controls[:2] == [app.stream_port_label, app.stream_port_tf]
    assert port_row.controls[2].width == S(40)
    assert port_row.controls[3:] == [app.stream_display_fit_label, app.stream_display_fit_dd]
    assert app.stream_display_fit_label.width == app.stream_key_label.width
    assert not hasattr(app, "stream_fit_row")
    assert app._streamer_rows.count(port_row) == 1
    row_map = app._get_streamer_row_map()
    assert row_map["MJPEG Streamer"] == [0]
    assert app._streamer_rows[row_map["RTMP Streamer"][4]] is app.stream_calibration_row
    assert app._streamer_rows[row_map["RTMP Streamer"][-1]] is app.stream_bitrate_row
    assert "does not prove" in UI_MESSAGES["EN"]["tooltip_show_fps"]
    assert "不代表播放提速" in UI_MESSAGES["CN"]["tooltip_show_fps"]
