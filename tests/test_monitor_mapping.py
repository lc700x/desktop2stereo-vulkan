import ast
import sys
import types
from pathlib import Path

from path_config import APP_ROOT


ROOT = Path(__file__).resolve().parents[1]
BUILDERS_SOURCE = APP_ROOT / "gui" / "builders.py"


def _load_monitor_methods(monitors, primary_index):
    source = BUILDERS_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GUIBuilderMixin")
    wanted = {"populate_monitors", "_apply_stereo_output", "update_stereo_monitor_menu"}
    selected = [node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    module = ast.Module(body=[ast.ClassDef(name="MonitorMixin", bases=[], keywords=[], body=selected, decorator_list=[])], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "list_monitors": lambda: monitors,
        "get_primary_monitor_index": lambda: primary_index,
        "PRIMARY_MONITOR_SUFFIX": " (Primary)",
        "UI_MESSAGES": {
            "EN": {"Window Preview": "Window Preview"},
            "CN": {"Window Preview": "窗口预览"},
        },
    }
    exec(compile(module, str(BUILDERS_SOURCE), "exec"), namespace)
    return namespace["MonitorMixin"]


class FakeDropdown:
    def __init__(self, value=""):
        self.options = []
        self.value = value
        self.update_count = 0

    def update(self):
        self.update_count += 1


class FakeMonitorGui:
    def __init__(
        self,
        monitor_value="",
        stereo_value="Window Preview",
        run_mode_key="Local Viewer",
    ):
        self.monitor_dd = FakeDropdown(monitor_value)
        self.stereo_monitor_dd = FakeDropdown(stereo_value)
        self.monitor_label_to_index = {}
        self.capture_mode_key = "Monitor"
        self.run_mode_key = run_mode_key
        self.locale = "EN"
        self.fit_count = 0

    def _fit_window_to_content(self):
        self.fit_count += 1


FAKE_MSS_MONITORS = [
    {"left": 0, "top": 0, "width": 9600, "height": 2160},
    {"left": 5760, "top": 0, "width": 3840, "height": 2160},
    {"left": 3840, "top": 0, "width": 1920, "height": 1080},
    {"left": 0, "top": 0, "width": 3840, "height": 2160},
]


class FakeMssContext:
    monitors = FAKE_MSS_MONITORS

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeMssModule:
    @staticmethod
    def mss():
        return FakeMssContext()


class FakeWin32Api:
    @staticmethod
    def EnumDisplayMonitors():
        return [
            ("h5", None, (5760, 0, 9600, 2160)),
            ("h2", None, (3840, 0, 5760, 1080)),
            ("h1", None, (0, 0, 3840, 2160)),
        ]

    @staticmethod
    def GetMonitorInfo(handle):
        return {
            "h5": {"Device": r"\\.\DISPLAY5"},
            "h2": {"Device": r"\\.\DISPLAY2"},
            "h1": {"Device": r"\\.\DISPLAY1"},
        }[handle]


def test_list_monitors_uses_windows_display_numbers(monkeypatch):
    from gui import capture_sources

    monkeypatch.setattr(capture_sources, "OS_NAME", "Windows")
    monkeypatch.setitem(sys.modules, "mss", FakeMssModule)
    monkeypatch.setitem(sys.modules, "win32api", FakeWin32Api)

    monitors = capture_sources.list_monitors()

    assert [mon["display_number"] for mon in monitors] == [1, 2, 3]
    assert [mon["device_display_number"] for mon in monitors] == [1, 2, 5]
    assert [mon["capture_index"] for mon in monitors] == [3, 2, 1]


def test_two_monitors_allow_same_monitor_local_output():
    monitors = [
        {"capture_index": 1, "display_number": 1, "left": 0, "top": 0, "width": 1920, "height": 1080},
        {"capture_index": 2, "display_number": 2, "left": 1920, "top": 0, "width": 2560, "height": 1440},
    ]
    current = "2: 2560x1440 @ (1920,0)"
    gui = FakeMonitorGui(monitor_value=current)
    monitor_mixin = _load_monitor_methods(monitors, primary_index=1)
    gui.update_stereo_monitor_menu = monitor_mixin.update_stereo_monitor_menu.__get__(gui, FakeMonitorGui)

    monitor_mixin.populate_monitors(gui)

    assert gui.monitor_dd.value == current
    assert gui.monitor_label_to_index[current] == 2
    # The captured input monitor stays selectable as the same-monitor output.
    assert gui.stereo_monitor_dd.options == [
        "1: 1920x1080 @ (0,0) (Primary)",
        "2: 2560x1440 @ (1920,0)",
    ]
    assert gui.stereo_monitor_dd.value == "1: 1920x1080 @ (0,0) (Primary)"
    assert gui.fit_count == 1


def test_populate_monitors_falls_back_to_primary_when_current_missing():
    monitors = [
        {"capture_index": 1, "display_number": 1, "left": 0, "top": 0, "width": 1920, "height": 1080},
        {"capture_index": 2, "display_number": 2, "left": 1920, "top": 0, "width": 2560, "height": 1440},
    ]
    gui = FakeMonitorGui(monitor_value="9: stale")
    monitor_mixin = _load_monitor_methods(monitors, primary_index=2)
    gui.update_stereo_monitor_menu = monitor_mixin.update_stereo_monitor_menu.__get__(gui, FakeMonitorGui)

    monitor_mixin.populate_monitors(gui)

    primary = "2: 2560x1440 @ (1920,0) (Primary)"
    assert gui.monitor_dd.value == primary
    assert gui.stereo_monitor_dd.options == [
        "1: 1920x1080 @ (0,0)",
        "2: 2560x1440 @ (1920,0) (Primary)",
    ]
    assert gui.stereo_monitor_dd.value == "1: 1920x1080 @ (0,0)"


def test_populate_monitors_includes_detected_model_name():
    monitors = [
        {
            "capture_index": 1,
            "display_number": 1,
            "left": 0,
            "top": 0,
            "width": 3840,
            "height": 2160,
            "model": "VITURE",
            "name": "VITURE",
        },
    ]
    gui = FakeMonitorGui()
    monitor_mixin = _load_monitor_methods(monitors, primary_index=1)
    gui.update_stereo_monitor_menu = monitor_mixin.update_stereo_monitor_menu.__get__(gui, FakeMonitorGui)

    monitor_mixin.populate_monitors(gui)

    assert gui.monitor_dd.value == "1: VITURE 3840x2160 (Primary)"


def test_stereo_output_has_no_window_preview_option():
    gui = FakeMonitorGui()
    gui.locale = "CN"
    monitor_mixin = _load_monitor_methods([], primary_index=1)

    monitor_mixin.update_stereo_monitor_menu(gui)

    assert gui.stereo_monitor_dd.options == []
    assert gui.stereo_monitor_dd.value == ""


def test_three_monitors_default_output_to_last_available_monitor():
    monitors = [
        {"capture_index": 1, "display_number": 1, "left": 0, "top": 0, "width": 1920, "height": 1080},
        {"capture_index": 2, "display_number": 2, "left": 1920, "top": 0, "width": 2560, "height": 1440},
        {"capture_index": 3, "display_number": 3, "left": 4480, "top": 0, "width": 3840, "height": 2160},
    ]
    gui = FakeMonitorGui(monitor_value="1: 1920x1080 @ (0,0) (Primary)")
    monitor_mixin = _load_monitor_methods(monitors, primary_index=1)
    gui.update_stereo_monitor_menu = monitor_mixin.update_stereo_monitor_menu.__get__(gui, FakeMonitorGui)

    monitor_mixin.populate_monitors(gui)

    assert gui.stereo_monitor_dd.options == [
        "1: 1920x1080 @ (0,0) (Primary)",
        "2: 2560x1440 @ (1920,0)",
        "3: 3840x2160 @ (4480,0)",
    ]
    assert gui.stereo_monitor_dd.value == "2: 2560x1440 @ (1920,0)"


def test_switching_local_input_to_output_monitor_selects_last_other_monitor():
    gui = FakeMonitorGui(
        monitor_value="3: 3840x2160 @ (4480,0)",
        stereo_value="3: 3840x2160 @ (4480,0)",
    )
    gui.monitor_label_to_index = {
        "1: 1920x1080 @ (0,0)": 1,
        "2: 2560x1440 @ (1920,0)": 2,
        "3: 3840x2160 @ (4480,0)": 3,
    }
    monitor_mixin = _load_monitor_methods([], primary_index=1)

    monitor_mixin.update_stereo_monitor_menu(gui)

    assert gui.stereo_monitor_dd.options == [
        "1: 1920x1080 @ (0,0)",
        "2: 2560x1440 @ (1920,0)",
        "3: 3840x2160 @ (4480,0)",
    ]
    assert gui.stereo_monitor_dd.value == "3: 3840x2160 @ (4480,0)"


def test_saved_local_output_rejects_unknown_monitor():
    gui = FakeMonitorGui(monitor_value="2: 2560x1440 @ (1920,0)")
    gui.monitor_label_to_index = {
        "1: 1920x1080 @ (0,0)": 1,
        "2: 2560x1440 @ (1920,0)": 2,
    }
    gui._get_monitor_count = lambda: 2
    monitor_mixin = _load_monitor_methods([], primary_index=1)

    monitor_mixin._apply_stereo_output(gui, {"Stereo Output": 99})

    assert gui.stereo_monitor_dd.value == ""


def test_saved_local_output_selects_same_monitor_when_it_matches_input():
    shared = "2: 2560x1440 @ (1920,0)"
    gui = FakeMonitorGui(monitor_value=shared)
    gui.monitor_label_to_index = {
        "1: 1920x1080 @ (0,0)": 1,
        shared: 2,
    }
    gui._get_monitor_count = lambda: 2
    monitor_mixin = _load_monitor_methods([], primary_index=1)

    monitor_mixin._apply_stereo_output(gui, {"Stereo Output": 2})

    assert gui.stereo_monitor_dd.value == shared


def test_3d_monitor_output_can_match_input_monitor():
    shared = "2: 2560x1440 @ (1920,0)"
    gui = FakeMonitorGui(
        monitor_value=shared,
        run_mode_key="3D Monitor",
    )
    gui.monitor_label_to_index = {
        "1: 1920x1080 @ (0,0)": 1,
        shared: 2,
    }
    gui._get_monitor_count = lambda: 2
    monitor_mixin = _load_monitor_methods([], primary_index=1)

    monitor_mixin._apply_stereo_output(gui, {"Stereo Output": 2})

    assert gui.stereo_monitor_dd.value == shared


def test_single_monitor_local_viewer_uses_same_monitor_3d_mode():
    # With a single display, Local Viewer + Monitor capture auto-enters the
    # same-monitor 3D mode (SBS replaces the captured frame in-place).
    from gui.process import GUIProcessMixin

    class Gui:
        run_mode_key = "Local Viewer"
        locale = "CN"
        capture_mode_key = "Monitor"
        monitor_label_to_index = {"1: 1920x1080 @ (0,0)": 1}
        stereo_monitor_dd = FakeDropdown("1: 1920x1080 @ (0,0)")
        monitor_dd = FakeDropdown("1: 1920x1080 @ (0,0)")
        stream_port_tf = FakeDropdown("1122")
        crf_tf = FakeDropdown("23")
        audio_delay_tf = FakeDropdown("0.0")
        stream_key_tf = FakeDropdown("live")

        @staticmethod
        def _get_monitor_count():
            return 1

    ok, message = GUIProcessMixin._validate_config_before_run(Gui())

    assert ok
