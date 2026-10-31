import pytest
from stereo_runtime.render_size import RenderSizePolicy
from utils.display_info import DisplayInfo
from viewer.settings import resolve_viewer_settings

BASE_SETTINGS = {
    "Run Mode": "Local Viewer",
    "Monitor Index": 1,
    "Display Mode": "Half-SBS",
    "Stereo Output": None,
    "Processing Resolution": "Auto",
    "Capture Mode": "Monitor",
    "Window Title": "",
    "Target FPS": 60,
    "Stream Target FPS": 30,
    "Language": "EN",
    "Show FPS": False,
    "Depth Strength": 2.0,
    "Convergence": 0.0,
    "Fill 16:9": True,
    "Upscaler": "Off",
    "Upscaler Sharpness": 0.35,
    "Controller Model": "PICO",
    "Environment Model": "None",
    "XR Preview Window": False,
}


@pytest.fixture(autouse=True)
def mock_display_enumeration(monkeypatch):
    from gui import capture_sources
    from utils import display_info

    displays = [
        DisplayInfo(1, 1, 0, 0, 1920, 1080, stable_id="display-1", is_primary=True),
        DisplayInfo(2, 2, 1920, 0, 1920, 1080, stable_id="display-2"),
    ]
    monkeypatch.setattr(display_info, "enumerate_displays", lambda: displays)
    monkeypatch.setattr(capture_sources, "get_primary_monitor_index", lambda: 1)
    return displays


def test_local_viewer_fps_is_not_limited_by_stream_calibration():
    resolved = resolve_viewer_settings(dict(BASE_SETTINGS, VSync=False))

    assert resolved.target_fps == 60
    assert resolved.fps == 60


def test_network_stream_uses_its_mode_scoped_target_fps():
    settings = dict(BASE_SETTINGS, VSync=False, **{"Run Mode": "RTMP Streamer"})

    resolved = resolve_viewer_settings(settings)

    assert resolved.target_fps == 30
    assert resolved.fps == 30


def test_resolve_viewer_settings_reads_vsync_key():
    settings = dict(BASE_SETTINGS, VSync=False)

    resolved = resolve_viewer_settings(settings)

    assert resolved.local_vsync is False
    assert not hasattr(resolved, "ipd")


def test_resolve_viewer_settings_requires_vsync_key():
    settings = dict(BASE_SETTINGS)

    try:
        resolve_viewer_settings(settings)
    except KeyError as exc:
        assert exc.args == ("VSync",)
    else:
        raise AssertionError("resolve_viewer_settings should require VSync")


def test_resolve_viewer_settings_resolves_xr_headset_screen_preset():
    settings = dict(BASE_SETTINGS, VSync=False, **{"XR Headset Model": "XREAL Air / Air 2 / Pro"})

    resolved = resolve_viewer_settings(settings)

    assert resolved.xr_headset_model == "XREAL Air / Air 2 / Pro"
    assert resolved.openxr_screen_distance == 4.0
    assert resolved.openxr_screen_width == 4.62


def test_resolve_viewer_settings_rebinds_saved_display_identities(monkeypatch):
    from utils import display_info

    current_displays = [
        DisplayInfo(1, 1, 0, 0, 1920, 1080, stable_id="output-id", model="Output"),
        DisplayInfo(2, 2, 1920, 0, 1920, 1080, stable_id="other-id", model="Other"),
        DisplayInfo(3, 3, 3840, 0, 3840, 2160, stable_id="input-id", model="Input"),
    ]
    monkeypatch.setattr(display_info, "enumerate_displays", lambda: current_displays)
    settings = dict(
        BASE_SETTINGS,
        VSync=False,
        **{
            "Monitor Index": 1,
            "Monitor Identity": {"stable_id": "input-id", "model": "Input"},
            "Stereo Output": 3,
            "Stereo Output Identity": {"stable_id": "output-id", "model": "Output"},
        },
    )

    resolved = resolve_viewer_settings(settings)

    assert resolved.monitor_index == 3
    assert resolved.stereo_display_index == 1
    assert resolved.stereo_display_selection is True


def test_resolve_viewer_settings_falls_back_to_primary_for_disconnected_input(monkeypatch):
    from utils import display_info

    current_displays = [
        DisplayInfo(1, 1, 0, 0, 1920, 1080, stable_id="connected-id"),
        DisplayInfo(2, 2, 1920, 0, 1920, 1080, stable_id="other-id"),
    ]
    monkeypatch.setattr(
        display_info,
        "enumerate_displays",
        lambda: current_displays,
    )
    settings = dict(
        BASE_SETTINGS,
        VSync=False,
        **{
            "Monitor Identity": {"stable_id": "disconnected-id"},
        },
    )

    resolved = resolve_viewer_settings(settings)

    assert resolved.monitor_index == 1


def test_resolve_viewer_settings_uses_first_display_when_primary_is_unavailable(monkeypatch):
    from gui import capture_sources
    from utils import display_info

    current_displays = [
        DisplayInfo(2, 1, 0, 0, 1920, 1080, stable_id="connected-id"),
        DisplayInfo(3, 2, 1920, 0, 1920, 1080, stable_id="other-id"),
    ]
    monkeypatch.setattr(display_info, "enumerate_displays", lambda: current_displays)
    monkeypatch.setattr(capture_sources, "get_primary_monitor_index", lambda: 1)
    settings = dict(
        BASE_SETTINGS,
        VSync=False,
        **{
            "Monitor Index": 99,
            "Monitor Identity": {"stable_id": "disconnected-id"},
        },
    )

    resolved = resolve_viewer_settings(settings)

    assert resolved.monitor_index == 2


def test_resolve_viewer_settings_falls_back_to_input_for_disconnected_output(monkeypatch):
    from utils import display_info

    current_displays = [
        DisplayInfo(1, 1, 0, 0, 1920, 1080, stable_id="primary-id"),
        DisplayInfo(2, 2, 1920, 0, 1920, 1080, stable_id="input-id"),
    ]
    monkeypatch.setattr(display_info, "enumerate_displays", lambda: current_displays)
    settings = dict(
        BASE_SETTINGS,
        VSync=False,
        **{
            "Monitor Index": 1,
            "Monitor Identity": {"stable_id": "input-id"},
            "Stereo Output": 3,
            "Stereo Output Identity": {"stable_id": "disconnected-output"},
        },
    )

    resolved = resolve_viewer_settings(settings)

    assert resolved.monitor_index == 2
    assert resolved.stereo_display_index == 2
    assert resolved.stereo_display_selection is True


def test_resolve_viewer_settings_falls_back_for_invalid_numeric_indices(monkeypatch):
    settings = dict(
        BASE_SETTINGS,
        VSync=False,
        **{"Monitor Index": 99, "Stereo Output": 99},
    )

    resolved = resolve_viewer_settings(settings)

    assert resolved.monitor_index == 1
    assert resolved.stereo_display_index == 1


def test_resolve_viewer_settings_reports_when_no_system_displays_exist(monkeypatch):
    from utils import display_info

    monkeypatch.setattr(display_info, "enumerate_displays", list)

    with pytest.raises(RuntimeError, match="no system displays are available"):
        resolve_viewer_settings(dict(BASE_SETTINGS, VSync=False))


def test_resolve_viewer_settings_keeps_unset_stereo_output_unset():
    resolved = resolve_viewer_settings(dict(BASE_SETTINGS, VSync=False))

    assert resolved.stereo_display_index is None
    assert resolved.stereo_display_selection is False


def test_resolve_viewer_settings_reads_render_size_config():
    settings = dict(
        BASE_SETTINGS,
        VSync=False,
        **{
            "Render Size Policy": "scaled",
            "Render Scale": "1K / 50%",
            "Render Fixed Width": 1600,
            "Render Fixed Height": 900,
            "Render Max Pixels": 2073600,
            "Render Min Dimension": 540,
            "Render Align": 8,
        },
    )

    resolved = resolve_viewer_settings(settings)

    assert resolved.render_size_config.policy is RenderSizePolicy.SCALED
    assert resolved.render_size_config.scale_factor == "1K / 50%"
    assert resolved.render_size_config.fixed_width == 1600
    assert resolved.render_size_config.fixed_height == 900
    assert resolved.render_size_config.max_pixels == 2073600
    assert resolved.render_size_config.min_dimension == 540
    assert resolved.render_size_config.align == 8

