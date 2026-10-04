from types import SimpleNamespace
import multiprocessing
import os
import queue
import subprocess
import sys
import threading
import time


def _control(
    key: str,
    *,
    label: str = "Control",
    kind: str = "button",
    minimum: float = 0.0,
    maximum: float = 1.0,
    step: float = 0.05,
    enabled: bool = True,
):
    return SimpleNamespace(
        key=key,
        label=label,
        kind=kind,
        minimum=minimum,
        maximum=maximum,
        step=step,
        enabled=enabled,
    )


def _spawn_waiting_flet_descendant(start_gate, child_pipe) -> None:
    if not start_gate.wait(timeout=10):
        return
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    child_pipe.send(child.pid)
    child.wait()


def test_desktop_settings_icon_uses_transparent_70_percent_style():
    from xr_viewer.desktop_settings_menu import (
        DESKTOP_SETTINGS_ICON_OPACITY,
        DESKTOP_SETTINGS_ICON_IMAGE_SIZE,
        DESKTOP_SETTINGS_ICON_SIZE,
        DESKTOP_SETTINGS_ICON_TRANSPARENT_COLOR,
    )

    assert DESKTOP_SETTINGS_ICON_OPACITY == 0.40
    assert DESKTOP_SETTINGS_ICON_TRANSPARENT_COLOR == "#010101"
    assert DESKTOP_SETTINGS_ICON_SIZE == (51, 57)
    assert DESKTOP_SETTINGS_ICON_IMAGE_SIZE == (42, 42)


def test_floating_icon_geometry_is_relative_to_input_monitor():
    from xr_viewer.desktop_settings_menu import _icon_geometry_for_monitor

    assert _icon_geometry_for_monitor(
        (3840, 0, 1920, 1200), fallback_size=(1920, 1080)
    ) == (5685, 571, 51, 57)


def test_floating_icon_geometry_supports_negative_monitor_coordinates():
    from xr_viewer.desktop_settings_menu import _icon_geometry_for_monitor

    x, y, width, height = _icon_geometry_for_monitor(
        (-1920, -100, 1920, 1080), fallback_size=(1920, 1080)
    )
    assert (x, y) == (-75, 411)
    assert (width, height) == (51, 57)


def test_flet_panel_is_centered_on_the_selected_input_monitor():
    from xr_viewer.desktop_settings_menu import _flet_panel_position_for_monitor

    assert _flet_panel_position_for_monitor((3840, 0, 1920, 1200)) == (4420, 275)


def test_flet_panel_position_preserves_negative_monitor_coordinates():
    from xr_viewer.desktop_settings_menu import _flet_panel_position_for_monitor

    assert _flet_panel_position_for_monitor((-1920, -100, 1920, 1080)) == (-1340, 115)


def test_snapshot_layout_signature_ignores_live_value_changes():
    from xr_viewer.desktop_settings_menu import _snapshot_layout_signature

    controls = (
        _control("tab:picture", label="Picture"),
        _control("color_brightness", label="Brightness", kind="slider"),
    )
    first = {"tab": "picture", "controls": controls, "values": {"color_brightness": 1.0}}
    second = {"tab": "picture", "controls": controls, "values": {"color_brightness": 1.4}}

    assert _snapshot_layout_signature(first) == _snapshot_layout_signature(second)


def test_snapshot_layout_signature_tracks_discrete_selection_changes():
    from xr_viewer.desktop_settings_menu import _snapshot_layout_signature

    controls = (
        _control("openxr:render_auto", label="Headset optimized"),
        _control("openxr_render_scale", label="Render Resolution", kind="slider"),
    )
    manual = {
        "tab": "picture",
        "controls": controls,
        "values": {"openxr_render_auto": False, "openxr_render_scale": 1.0},
    }
    automatic = {
        "tab": "picture",
        "controls": controls,
        "values": {"openxr_render_auto": True, "openxr_render_scale": 1.0},
    }

    assert _snapshot_layout_signature(manual) != _snapshot_layout_signature(automatic)


def test_snapshot_layout_signature_tracks_language_changes():
    from xr_viewer.desktop_settings_menu import _snapshot_layout_signature

    controls = (_control("tab:screen", label="Screen"),)
    english = {"lang": "EN", "tab": "screen", "controls": controls}
    chinese = {"lang": "CN", "tab": "screen", "controls": controls}

    assert _snapshot_layout_signature(english) != _snapshot_layout_signature(chinese)


def test_flet_slider_snapshot_values_are_bounded_to_the_control_range():
    from xr_viewer.desktop_settings_menu import _bounded_slider_value

    assert _bounded_slider_value(2.066, 0.25, 20.0) == 2.066
    assert _bounded_slider_value(25.0, 0.25, 20.0) == 20.0
    assert _bounded_slider_value(float("nan"), 0.25, 20.0) == 0.25


def test_flet_static_openxr_labels_have_english_and_chinese_translations():
    from gui.localization import gettext_for

    messages = (
        "OpenXR Settings",
        "Desktop2Stereo OpenXR Settings",
        "Physical mouse controls are synchronized with the in-headset menu.",
        "Waiting for OpenXR settings...",
        "Reset picture",
        "Reset depth",
        "Reset placement",
    )
    assert all(gettext_for("EN", message) == message for message in messages)
    assert gettext_for("CN", "Reset picture") == "\u91cd\u7f6e\u753b\u9762"
    assert gettext_for("CN", "Reset depth") == "\u91cd\u7f6e\u666f\u6df1"
    assert gettext_for("CN", "Reset placement") == "\u91cd\u7f6e\u5c4f\u5e55\u4f4d\u7f6e"
    assert gettext_for("CN", "OpenXR Settings") == "OpenXR 设置"
    assert gettext_for("CN", "Desktop2Stereo OpenXR Settings") == "Desktop2Stereo OpenXR 设置"
    assert gettext_for("CN", messages[2]) == "物理鼠标控制与头显内菜单同步。"
    assert gettext_for("CN", messages[3]) == "正在等待 OpenXR 设置..."


def test_screen_curveness_and_rotation_use_separate_flet_rows():
    from xr_viewer.desktop_settings_menu import (
        _button_row_group,
        _screen_button_row_group,
    )

    assert _screen_button_row_group("screen:type:flat") == "screen_curveness"
    assert _screen_button_row_group("screen:type:deep") == "screen_curveness"
    assert _screen_button_row_group("screen:rotate:-90") == "screen_rotation"
    assert _screen_button_row_group("screen:rotate:+90") == "screen_rotation"
    assert _screen_button_row_group("screen:section:crop") == "screen_section"
    assert _screen_button_row_group("section:reset_defaults") is None
    assert _button_row_group("depth:toggle_stereo") == "depth_stereo"
    assert _button_row_group("depth:toggle_cross_eyed") == "depth_stereo"
    assert _button_row_group("glow:surround") == "glow_modes"
    assert _button_row_group("glow:off") == "glow_modes"
    assert _button_row_group("room:model:Default") == "room_models"
    assert _button_row_group("room:seat:middle") == "room_seats"
    assert _button_row_group("room:toggle_screen_reflection") == "room_scene"
    assert _button_row_group("reset:color_adjustment") == "color_adjustment"
    assert _button_row_group("reset:screen_placement") == "screen_placement"
    assert _button_row_group("screen:reset_crop") == "screen_crop"
    assert _button_row_group("section:reset_defaults") == "screen_placement"


def test_flet_controls_use_one_compact_dimension_step():
    from xr_viewer.desktop_settings_menu import (
        _FLET_BUTTON_HEIGHT,
        _FLET_PANEL_SIZE,
        _FLET_SECONDARY_BUTTON_HEIGHT,
        _FLET_SIDEBAR_WIDTH,
        _FLET_SPACING,
        _FLET_STEP_BUTTON_SIZE,
        _compact_flet_dimension,
    )

    assert _FLET_PANEL_SIZE == (760, 650)
    assert _FLET_SIDEBAR_WIDTH == 144
    assert _FLET_SPACING == 8
    assert _FLET_BUTTON_HEIGHT == 40
    assert _FLET_SECONDARY_BUTTON_HEIGHT == 32
    assert _FLET_STEP_BUTTON_SIZE == 40
    assert _compact_flet_dimension(64) == 48
    assert _compact_flet_dimension(48) == 40
    assert _compact_flet_dimension(136) == 104
    assert _compact_flet_dimension(216) == 160
    assert _compact_flet_dimension(236) == 176


def test_flet_step_button_symbols_use_centered_zero_padding_style():
    import flet as ft

    from xr_viewer.desktop_settings_menu import (
        _make_step_button_label,
        _make_step_button_style,
    )

    style = _make_step_button_style(ft)
    minus = _make_step_button_label(ft, "step:minus:color_brightness")
    plus = _make_step_button_label(ft, "step:plus:color_brightness")

    assert style.alignment == ft.Alignment.CENTER
    assert style.padding == ft.Padding.symmetric(horizontal=0, vertical=0)
    assert minus.value == "-" and plus.value == "+"
    assert minus.text_align == plus.text_align == ft.TextAlign.CENTER


def test_flet_reset_actions_are_rendered_as_card_footer_controls():
    from xr_viewer.desktop_settings_menu import _is_reset_control

    assert _is_reset_control("reset:room_scene")
    assert _is_reset_control("screen:reset_crop")
    assert _is_reset_control("section:reset_defaults")
    assert not _is_reset_control("room:exposure")


def test_flet_formats_symmetric_crop_values_for_the_shared_snapshot():
    from xr_viewer.desktop_settings_menu import _format_value

    assert _format_value(12.0, 1.0, "screen:crop_width") == "12% each"
    assert _format_value(7.0, 1.0, "screen:crop_height") == "7% each"


def test_flet_selectors_reflect_shared_menu_values():
    from xr_viewer.desktop_settings_menu import (
        _control_is_selected,
        _toggle_value_for,
    )

    values = {
        "screen:section": "crop",
        "screen:curve_half_angle": 0.72,
        "depth_strength": 0.25,
        "cross_eyed": True,
        "room:screen_reflection_enabled": True,
        "screen:dynamic_crop": False,
        "glow:mode": "veil",
        "room:model": "3d_b",
        "room:seat_index": 2,
    }

    assert _control_is_selected("screen:section:crop", values)
    assert _control_is_selected("screen:type:deep", values)
    assert _control_is_selected("glow:veil", values)
    assert _control_is_selected("room:model:3d_b", values)
    assert _control_is_selected("room:seat:back", values)
    assert not _control_is_selected("screen:type:flat", values)
    assert _toggle_value_for("depth:toggle_stereo", values)
    assert _toggle_value_for("depth:toggle_cross_eyed", values)
    assert _toggle_value_for("room:toggle_screen_reflection", values)
    assert not _toggle_value_for("screen:dynamic_crop", values)


def test_openxr_option_labels_have_chinese_translations():
    from gui.localization import gettext_for
    from xr_viewer.settings_menu import OpenXrSettingsMenu

    menu = OpenXrSettingsMenu()
    labels = set()
    for tab in ("picture", "depth", "glow", "room", "screen"):
        menu.set_tab(tab)
        labels.update(
            control.label
            for control in menu.controls(show_glow=True, lang="EN")
            if control.kind != "slider_step"
        )

    assert all(
        gettext_for("CN", label) != label
        for label in labels
        if label not in {
            "-90°", "+90°", "2D / 3D", "OFF", "RCAS", "Gamma"
        }
    )


def test_flet_control_labels_are_translated_before_rendering():
    from gui.localization import gettext_for

    assert gettext_for("CN", "Layout") == "布局"
    assert gettext_for("CN", "Dynamic Crop") == "动态裁剪"


def test_flet_tab_group_uses_shared_snapshot_tab_order():
    controls = (
        _control("tab:screen", label="Screen"),
        _control("tab:depth", label="Depth"),
        _control("tab:glow", label="Glow"),
        _control("tab:picture", label="Picture"),
    )

    tab_keys = tuple(
        str(control.key)
        for control in controls
        if str(control.key).startswith("tab:")
    )

    assert tab_keys == (
        "tab:screen", "tab:depth", "tab:glow", "tab:picture",
    )


def test_clicking_icon_toggles_the_flet_panel_once():
    from xr_viewer.desktop_settings_menu import DesktopOpenXrSettingsWindow

    window = DesktopOpenXrSettingsWindow()
    calls = []
    window._toggle_panel = lambda: calls.append("toggle")

    assert window._on_icon_click(None) == "break"
    assert calls == ["toggle"]


def test_flet_settings_child_closes_when_runtime_parent_exits():
    from xr_viewer.desktop_settings_menu import _quit_flet_when_parent_exits

    parent_exited = threading.Event()
    command_queue = queue.Queue()

    class ParentProcess:
        def join(self):
            parent_exited.wait(timeout=1.0)

    worker = threading.Thread(
        target=_quit_flet_when_parent_exits,
        args=(ParentProcess(), command_queue),
        daemon=True,
    )
    worker.start()
    assert command_queue.empty()
    parent_exited.set()

    worker.join(timeout=1.0)
    assert not worker.is_alive()
    assert command_queue.get_nowait() == "__quit__"


def test_settings_stop_waits_for_flet_process_to_exit():
    from xr_viewer.desktop_settings_menu import DesktopOpenXrSettingsWindow

    class StoppedProcess:
        pid = 123

        def __init__(self):
            self.join_calls = []
            self.terminated = False

        def join(self, timeout=None):
            self.join_calls.append(timeout)

        def is_alive(self):
            return False

        def terminate(self):
            self.terminated = True

    window = DesktopOpenXrSettingsWindow()
    process = StoppedProcess()
    window._flet_process = process

    window.stop()

    assert process.join_calls == [2.0]
    assert not process.terminated


def test_settings_stop_kills_flet_process_tree_after_timeout(monkeypatch):
    from types import SimpleNamespace
    import xr_viewer.desktop_settings_menu as settings_menu

    class SlowProcess:
        pid = 321

        def __init__(self):
            self.alive = True
            self.join_calls = []
            self.terminated = False

        def join(self, timeout=None):
            self.join_calls.append(timeout)

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.terminated = True
            self.alive = False

    process = SlowProcess()
    commands = []

    def run_taskkill(args, **kwargs):
        commands.append((args, kwargs))
        process.alive = False

    monkeypatch.setattr(settings_menu, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(settings_menu.subprocess, "run", run_taskkill)

    settings_menu._stop_flet_process(process)

    assert commands[0][0] == ["taskkill", "/f", "/t", "/pid", "321"]
    assert commands[0][1]["timeout"] == 2.0
    assert process.join_calls == [2.0, 1.0]
    assert not process.terminated


def test_windows_settings_job_kills_worker_and_flet_descendants_on_owner_close():
    import pytest
    from gui.flet_runtime import _windows_process_snapshot
    from xr_viewer.desktop_settings_menu import _WindowsKillOnCloseJob

    if os.name != "nt":
        pytest.skip("Windows Job Objects are only available on Windows")

    context = multiprocessing.get_context("spawn")
    start_gate = context.Event()
    parent_pipe, child_pipe = context.Pipe(duplex=False)
    worker = context.Process(
        target=_spawn_waiting_flet_descendant,
        args=(start_gate, child_pipe),
        name="test-flet-job-worker",
    )
    job = _WindowsKillOnCloseJob()
    descendant_pid = None
    try:
        worker.start()
        assert job.assign(worker.pid), "worker could not be assigned to its cleanup job"
        start_gate.set()
        assert parent_pipe.poll(5.0), "worker did not start its Flet-like child"
        descendant_pid = parent_pipe.recv()

        # Closing the final job handle models the owning Desktop2Stereo process
        # crashing before it can run the normal stop callback.
        job._kernel32.CloseHandle(job._handle)
        job._handle = None
        worker.join(timeout=5.0)
        assert not worker.is_alive()

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            running = {pid for pid, _, _ in _windows_process_snapshot()}
            if descendant_pid not in running:
                break
            time.sleep(0.05)
        assert descendant_pid not in {pid for pid, _, _ in _windows_process_snapshot()}
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=2.0)
        job.close()
        parent_pipe.close()
        child_pipe.close()


def test_physical_mouse_control_uses_the_existing_openxr_action_queue():
    from xr_viewer.desktop_settings_menu import DesktopOpenXrSettingsWindow

    window = DesktopOpenXrSettingsWindow()
    window.actions.put(("depth_strength", 0.75))

    assert window.actions.get(timeout=1.0) == ("depth_strength", 0.75)
