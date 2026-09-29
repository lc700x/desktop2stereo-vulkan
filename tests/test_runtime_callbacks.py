from pathlib import Path

from path_config import APP_ROOT
from types import SimpleNamespace
import sys
import threading

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from app_runtime.runtime_callbacks import RuntimeCallbacks
from stereo_runtime.settings_snapshot import RuntimeSettingsSnapshot


class FakeOpenXrState:
    def __init__(self, depth_strength: float) -> None:
        self.runtime_settings_snapshot = SimpleNamespace(
            depth_strength=depth_strength
        )
        self.updates: list[dict[str, float | None]] = []

    def update_runtime_config(self, **values) -> None:
        self.updates.append(values)
        if values.get("depth_strength") is not None:
            self.runtime_settings_snapshot.depth_strength = values[
                "depth_strength"
            ]


def _callbacks(depth_strength: float = 0.75) -> RuntimeCallbacks:
    context = SimpleNamespace(
        stereo_runtime=SimpleNamespace(
            stereo_config=SimpleNamespace(depth_strength=depth_strength)
        ),
        openxr_state=FakeOpenXrState(depth_strength),
        fps_breakdown=SimpleNamespace(
            inc=lambda *_args, **_kwargs: None,
            add_runtime_timing=lambda *_args, **_kwargs: None,
        ),
    )
    return RuntimeCallbacks(context)


def test_local_breakdown_is_not_gated_by_openxr_render_state() -> None:
    callbacks = _callbacks()
    callbacks.context.run_mode = "Viewer"
    callbacks.context.openxr_state.render_active = threading.Event()

    assert callbacks._render_active_for_breakdown() is True

    callbacks.context.run_mode = "OpenXR"
    assert callbacks._render_active_for_breakdown() is False
    callbacks.context.openxr_state.render_active.set()
    assert callbacks._render_active_for_breakdown() is True


def test_capture_fps_reports_accepted_capture_rate_and_expires(monkeypatch) -> None:
    callbacks = _callbacks()
    times = iter((10.0, 10.5, 11.0))
    monkeypatch.setattr(
        "app_runtime.runtime_callbacks.time.perf_counter",
        lambda: next(times),
    )

    callbacks.breakdown_inc("capture")
    callbacks.breakdown_inc("capture")

    assert callbacks.capture_fps() == pytest.approx(2.0)

    monkeypatch.setattr(
        "app_runtime.runtime_callbacks.time.perf_counter", lambda: 12.0
    )
    assert callbacks.capture_fps() == 0.0


def test_show_fps_hot_reload_updates_viewer_provider() -> None:
    callbacks = _callbacks()
    callbacks.context.settings_update_q = SimpleNamespace(
        put_nowait=lambda _snapshot: None,
        get_nowait=lambda: (_ for _ in ()).throw(Exception()),
    )
    assert callbacks.show_fps() is False
    assert callbacks.display_fit_mode() == "contain"

    callbacks.send_settings_snapshot(
        RuntimeSettingsSnapshot(
            version=1,
            timestamp=1.0,
            presentation_flags={
                "show_fps": True,
                "display_fit_mode": "cover",
            },
        )
    )

    assert callbacks.show_fps() is True
    assert callbacks.display_fit_mode() == "cover"


def test_audio_delay_hot_reload_is_forwarded_to_stream_output() -> None:
    callbacks = _callbacks()
    callbacks.context.stereo_active_preset = None
    seen = []
    snapshot = RuntimeSettingsSnapshot(version=2, timestamp=1.0)
    callbacks.context.stereo_hot_reloader = SimpleNamespace(
        poll_settings_snapshot_if_needed=lambda **_kwargs: (
            snapshot,
            None,
            {"audio_delay": 0.35},
        ),
        log_settings_snapshot=lambda *_args, **_kwargs: None,
    )
    callbacks.set_stream_output(
        SimpleNamespace(request_audio_delay=seen.append)
    )
    callbacks.send_settings_snapshot = lambda _snapshot: None
    callbacks.update_openxr_runtime_config = lambda **_kwargs: None
    callbacks.log_stereo_runtime_mode_once = lambda *_args, **_kwargs: None

    assert callbacks.apply_stereo_hot_reload_if_needed() is True
    assert seen == [0.35]


def test_controller_shortcut_toggles_stereo_and_restores_depth() -> None:
    callbacks = _callbacks(0.75)

    assert callbacks.on_openxr_controller_shortcut("toggle_stereo") is True
    assert callbacks.context.openxr_state.runtime_settings_snapshot.depth_strength == 0.0

    assert callbacks.on_openxr_controller_shortcut("toggle_stereo") is True
    assert (
        callbacks.context.openxr_state.runtime_settings_snapshot.depth_strength
        == pytest.approx(0.75)
    )


def test_controller_shortcut_resets_depth_and_rejects_renderer_action() -> None:
    callbacks = _callbacks(0.6)
    callbacks.context.openxr_state.runtime_settings_snapshot.depth_strength = 0.2

    assert callbacks.on_openxr_controller_shortcut("reset_depth") is True
    assert (
        callbacks.context.openxr_state.runtime_settings_snapshot.depth_strength
        == pytest.approx(0.6)
    )
    assert callbacks.on_openxr_controller_shortcut("toggle_screen_shape") is False


def test_controller_shortcut_adjusts_depth_continuously_with_clamp() -> None:
    callbacks = _callbacks(0.6)

    assert callbacks.on_openxr_controller_shortcut(
        "adjust_depth_strength", delta=0.25
    ) is True
    assert (
        callbacks.context.openxr_state.runtime_settings_snapshot.depth_strength
        == pytest.approx(0.85)
    )
    callbacks.on_openxr_controller_shortcut(
        "adjust_depth_strength", delta=-20.0
    )
    assert callbacks.context.openxr_state.runtime_settings_snapshot.depth_strength == 0.0


def test_settings_menu_runtime_value_updates_snapshot_without_persisting() -> None:
    callbacks = _callbacks(0.6)
    callbacks.context.openxr_state.runtime_settings_snapshot = RuntimeSettingsSnapshot(
        version=4, timestamp=1.0, depth_strength=0.6
    )
    snapshots = []
    callbacks.context.settings_update_q = SimpleNamespace(
        put_nowait=lambda snapshot: snapshots.append(snapshot),
        get_nowait=lambda: (_ for _ in ()).throw(Exception()),
    )
    callbacks.send_settings_snapshot = snapshots.append

    assert callbacks.on_openxr_controller_shortcut(
        "set_runtime_setting", name="color_brightness", value=1.4,
        persist=False,
    ) is True
    snapshot = callbacks.context.openxr_state.updates[-1]["snapshot"]
    assert snapshot.version == 5
    assert snapshot.color_brightness == pytest.approx(1.4)
    assert snapshots[-1] is snapshot


def test_settings_menu_can_reset_all_picture_values_in_one_snapshot() -> None:
    callbacks = _callbacks(0.6)
    callbacks.context.openxr_state.runtime_settings_snapshot = RuntimeSettingsSnapshot(
        version=7, timestamp=1.0, depth_strength=0.6
    )
    snapshots = []
    callbacks.send_settings_snapshot = snapshots.append
    values = {"color_brightness": 1.0, "vulkan_projection_rcas_sharpness": 0.5}

    assert callbacks.on_openxr_controller_shortcut(
        "set_runtime_settings", settings=values, persist=False
    ) is True
    snapshot = callbacks.context.openxr_state.updates[-1]["snapshot"]
    assert snapshot.version == 8
    assert snapshot.color_brightness == 1.0
    assert snapshot.vulkan_projection_rcas_sharpness == 0.5
    assert snapshots == [snapshot]


def test_settings_menu_updates_depth_and_cross_eyed_together() -> None:
    callbacks = _callbacks(0.6)
    callbacks.context.openxr_state.runtime_settings_snapshot = RuntimeSettingsSnapshot(
        version=9, timestamp=1.0, depth_strength=0.6, cross_eyed=True
    )
    snapshots = []
    callbacks.send_settings_snapshot = snapshots.append

    assert callbacks.on_openxr_controller_shortcut(
        "set_runtime_settings",
        settings={"depth_strength": 0.25, "cross_eyed": False},
        persist=False,
    ) is True
    snapshot = callbacks.context.openxr_state.updates[-1]["snapshot"]
    assert snapshot.depth_strength == pytest.approx(0.25)
    assert snapshot.cross_eyed is False


def test_settings_menu_persists_dedicated_openxr_render_scale(
    tmp_path, monkeypatch
) -> None:
    callbacks = _callbacks(0.6)
    callbacks.context.base_dir = str(tmp_path)
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text("OpenXR Render Scale: 1.0\n", encoding="utf-8")

    assert callbacks.on_openxr_controller_shortcut(
        "persist_openxr_render_scale", value=1.75
    ) is True
    assert "OpenXR Render Scale: 1.75" in settings_path.read_text(encoding="utf-8")
    assert "XR Render: 1.75" in settings_path.read_text(encoding="utf-8")


def test_settings_menu_persists_effective_automatic_openxr_render_scale(
    tmp_path,
) -> None:
    callbacks = _callbacks(0.6)
    callbacks.context.base_dir = str(tmp_path)
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        "XR Headset Model: Meta Quest 2\nOpenXR Render Scale: 1.24\n",
        encoding="utf-8",
    )

    assert callbacks.on_openxr_controller_shortcut(
        "persist_openxr_render_auto", value=1.0
    ) is True

    from stereo_runtime.hot_reload import read_yaml

    settings = read_yaml(str(settings_path))
    assert settings["XR Render Mode"] == "auto"
    assert settings["XR Render"] == pytest.approx(1.0)
    assert settings["OpenXR Render Scale"] == pytest.approx(1.0)


def test_settings_menu_persists_and_resets_openxr_screen_state(tmp_path) -> None:
    callbacks = _callbacks(0.6)
    callbacks.context.base_dir = str(tmp_path)
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        "Environment Model: Default\nOpenXR Render Scale: 1.0\n",
        encoding="utf-8",
    )
    state = {
        "position": [1.0, 1.6, -2.5],
        "width": 3.2,
        "height": 1.8,
        "rotation_deg": [-90.0, 5.0, 0.0],
        "curved": True,
        "curve_half_angle_rad": 0.72,
    }

    assert callbacks.on_openxr_controller_shortcut(
        "persist_openxr_screen_state",
        environment="Default",
        state=state,
    ) is True

    from stereo_runtime.hot_reload import read_yaml

    settings = read_yaml(str(settings_path))
    assert settings["OpenXR Render Scale"] == 1.0
    assert settings["OpenXR Screen States"]["Default"] == state

    assert callbacks.on_openxr_controller_shortcut(
        "reset_openxr_screen_state", environment="Default"
    ) is True
    settings = read_yaml(str(settings_path))
    assert settings["OpenXR Screen States"] == {}


def test_settings_menu_persists_default_environment_glow_mode(tmp_path) -> None:
    callbacks = _callbacks(0.6)
    callbacks.context.base_dir = str(tmp_path)
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        "Environment Model: Default\nOpenXR Glow Modes:\n  Cinema: veil\n",
        encoding="utf-8",
    )

    assert callbacks.on_openxr_controller_shortcut(
        "persist_openxr_glow_mode",
        environment="Default",
        mode="off",
    ) is True

    from stereo_runtime.hot_reload import read_yaml

    settings = read_yaml(str(settings_path))
    assert settings["OpenXR Glow Modes"] == {
        "Cinema": "veil",
        "Default": "off",
    }


def test_settings_menu_persists_default_environment_glow_transparency(tmp_path) -> None:
    callbacks = _callbacks(0.6)
    callbacks.context.base_dir = str(tmp_path)
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        "OpenXR Glow Transparency:\n  Cinema: 0.4\n",
        encoding="utf-8",
    )

    assert callbacks.on_openxr_controller_shortcut(
        "persist_openxr_glow_transparency",
        environment="Default",
        transparency=0.65,
    ) is True

    from stereo_runtime.hot_reload import read_yaml

    settings = read_yaml(str(settings_path))
    assert settings["OpenXR Glow Transparency"] == {
        "Cinema": 0.4,
        "Default": 0.65,
    }
