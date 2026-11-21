from __future__ import annotations

from xr_viewer.core_controller_guide_input import CoreControllerGuideInputMixin
from xr_viewer.core_controller_input import CoreControllerInputMixin
from xr_viewer.core_controller_shortcuts import CoreControllerShortcutsMixin
from viewer.controller_help import get_controller_help_rows


class GuideHost(CoreControllerGuideInputMixin, CoreControllerShortcutsMixin):
    def __init__(self) -> None:
        self._frame_now = 1.0
        self._controller_inputs = ({}, {})
        self._keyboard_visible = False
        self._grip_target_l = None
        self._grip_target_r = None
        self._controller_calibration_mode = False
        self._vulkan_controller_proxy_enabled = False
        self.actions: list[tuple[str, dict]] = []
        self.environment_mode = False
        self._init_controller_shortcuts()
        self._init_controller_guide_input()

    def _operation_guide_environment_mode(self) -> bool:
        return self.environment_mode

    def update_shortcuts(self, *, left=None, right=None, after=0.0, environment=False) -> None:
        self._frame_now += float(after)
        self._controller_inputs = (left or {}, right or {})
        self.environment_mode = environment
        self._handle_controller_shortcuts(controller_toggle_checked=True)

    def _dispatch_controller_shortcut(self, action: str, **values) -> None:
        self.actions.append((action, values))

    def update(self, *, left=None, right=None, after=0.0) -> None:
        self._frame_now += float(after)
        self._controller_inputs = (left or {}, right or {})
        self._handle_controller_guide_input(after)


def test_ab_chord_switches_brand_then_enters_calibration() -> None:
    host = GuideHost()
    buttons = {"a_button": 1.0, "b_button": 1.0}

    host.update(right=buttons)
    host.update(right=buttons, after=0.51)
    host.update(right=buttons, after=4.5)

    assert [action for action, _values in host.actions] == [
        "switch_controller_brand",
        "toggle_controller_calibration",
    ]


def test_ab_chord_does_not_switch_brand_when_controller_model_is_none() -> None:
    host = GuideHost()
    host._vulkan_controller_proxy_enabled = True
    buttons = {"a_button": 1.0, "b_button": 1.0}

    host.update(right=buttons)
    host.update(right=buttons, after=0.51)

    assert host.actions == []


def test_grip_sticks_match_screen_and_depth_guide_rows() -> None:
    host = GuideHost()

    host.update(left={"grip": 1.0, "joystick_x": 0.5}, after=0.1)
    host.update(
        left={"joystick_y": 0.8},
        right={"grip": 1.0, "joystick_x": 0.6, "joystick_y": -0.4},
        after=0.1,
    )

    assert [action for action, _values in host.actions] == [
        "rotate_screen",
        "adjust_depth_strength",
        "resize_screen",
    ]


def test_no_grip_axes_and_keyboard_axes_are_exclusive() -> None:
    host = GuideHost()
    host.update(
        left={"joystick_x": 0.4, "joystick_y": -0.5},
        right={"joystick_y": 0.7},
        after=0.1,
    )
    host._keyboard_visible = True
    host._grip_target_l = "keyboard"
    host.update(
        left={"grip": 1.0, "stick_click": 1.0},
        right={"joystick_x": 0.5, "joystick_y": -0.5},
        after=0.1,
    )

    assert [action for action, _values in host.actions] == [
        "arrow_axes",
        "scroll_axes",
        "rotate_keyboard",
    ]


def test_keyboard_grip_controls_require_laser_latched_keyboard_target() -> None:
    host = GuideHost()
    host._keyboard_visible = True
    host._grip_target_l = "screen"
    host.update(
        left={"grip": 1.0, "stick_click": 1.0, "joystick_x": 0.5},
        after=0.1,
    )
    host._grip_target_l = "keyboard"
    host.update(
        left={"grip": 1.0, "stick_click": 1.0, "joystick_x": 0.5},
        after=0.1,
    )

    assert [action for action, _values in host.actions] == [
        "rotate_screen",
        "orbit_keyboard",
    ]


def test_calibration_axes_and_b_save_suppress_normal_controls() -> None:
    host = GuideHost()
    host._controller_calibration_mode = True

    host.update(
        left={"joystick_y": 0.5},
        right={"joystick_x": 0.4, "joystick_y": -0.3},
        after=0.1,
    )
    host.update(right={"b_button": 1.0}, after=0.1)

    assert [action for action, _values in host.actions] == [
        "adjust_controller_calibration",
        "save_controller_calibration",
    ]


def test_operation_guide_matches_b_long_press_product_contract() -> None:
    cn_rows, cn_environment_rows = get_controller_help_rows("CN")
    en_rows, en_environment_rows = get_controller_help_rows("EN")

    assert ("右 B 键", "长按 1s", "显示/隐藏操作指南", False) in cn_rows
    assert ("右 B 键", "长按 1s", "显示/隐藏操作指南", False) in cn_environment_rows
    assert (
        "Right B button",
        "Long press 1s",
        "Show/hide operation guide",
        False,
    ) in en_rows
    assert (
        "Right B button",
        "Long press 1s",
        "Show/hide operation guide",
        False,
    ) in en_environment_rows
    assert any("同时长按 5s" in row[1] for row in cn_rows)
    assert any("同时长按 5s" in row[1] for row in cn_environment_rows)
    assert any("Hold together 5s" in row[1] for row in en_rows)
    assert any("Hold together 5s" in row[1] for row in en_environment_rows)


def test_abxy_short_and_long_presses_match_the_controller_action_map() -> None:
    cases = (
        ("a_button", "right", "toggle_screen_shape", "cycle_status_panel", False),
        ("b_button", "right", "toggle_passthrough", "toggle_operation_guide", False),
        ("x_button", "left", "toggle_keyboard", "toggle_environment_background", False),
        ("y_button", "left", "reset_screen", "cycle_screen_preset", False),
        ("y_button", "left", "reset_screen", "cycle_room_environment", True),
    )
    for button, hand, short_action, long_action, environment in cases:
        host = GuideHost()
        pressed = {button: 1.0}
        kwargs = {hand: pressed}
        host.update_shortcuts(**kwargs, environment=environment)
        host.update_shortcuts(**kwargs, after=1.01, environment=environment)
        host.update_shortcuts(environment=environment)
        assert [action for action, _values in host.actions] == [long_action]

        host = GuideHost()
        host.update_shortcuts(**kwargs, environment=environment)
        host.update_shortcuts(after=0.2, environment=environment)
        assert [action for action, _values in host.actions] == [short_action]


def test_face_button_rows_document_short_long_and_contextual_behavior_bilingually() -> None:
    cn_rows, cn_environment_rows = get_controller_help_rows("CN")
    en_rows, en_environment_rows = get_controller_help_rows("EN")
    assert ("右 A 键", "短按", "屏幕曲面/平面切换", False) in cn_rows
    assert ("右 A 键", "长按 1s", "循环显示: 隐藏/FPS左下角/FPS+快捷键指南", False) in cn_rows
    assert ("右 A 键", "短按", "屏幕曲面/平面切换", False) in cn_environment_rows
    assert ("右 A 键", "长按 1s", "循环状态面板: 隐藏/FPS左下角/FPS+快捷键指南", False) in cn_environment_rows
    for rows in (cn_rows, cn_environment_rows):
        assert ("左 X 键", "短按", "显示/隐藏虚拟键盘", False) in rows
        assert ("左 X 键", "长按 1s", "切换环境天空盒/黑色背景", False) in rows
        assert ("右 B 键", "短按", "切换VDXR绿屏/背景色", False) in rows
        assert ("右 B 键", "长按 1s", "显示/隐藏操作指南", False) in rows
    assert ("左 Y 键", "长按 1s", "循环屏幕预设", False) in cn_rows
    assert ("左 Y 键", "长按 1s", "切换房间环境", False) in cn_environment_rows
    assert ("Right A button", "Short press", "Toggle curved/flat screen", False) in en_rows
    assert ("Right A button", "Long press 1s", "Cycle panel: hidden/lower-left FPS/FPS + shortcut guide", False) in en_rows
    assert ("Right A button", "Short press", "Toggle curved/flat screen", False) in en_environment_rows
    assert ("Right A button", "Long press 1s", "Cycle status panel: hidden/lower-left FPS/FPS + shortcut guide", False) in en_environment_rows
    for rows in (en_rows, en_environment_rows):
        assert ("Left X button", "Short press", "Show/hide virtual keyboard", False) in rows
        assert ("Left X button", "Long press 1s", "Toggle environment skybox / black", False) in rows
        assert ("Right B button", "Short press", "Toggle VDXR green / background", False) in rows
        assert ("Right B button", "Long press 1s", "Show/hide operation guide", False) in rows
    assert ("Left Y button", "Long press 1s", "Cycle screen presets", False) in en_rows
    assert ("Left Y button", "Long press 1s", "Switch room environment", False) in en_environment_rows


def test_operation_guide_matches_keyboard_orbit_axis_mapping_and_reset_scope() -> None:
    cn_rows, _ = get_controller_help_rows("CN")
    en_rows, _ = get_controller_help_rows("EN")

    assert (
        "左握持 + 左摇杆 左右/前后",
        "按住推动",
        "键盘球面环绕",
        False,
    ) in cn_rows
    assert (
        "Left Grip + left stick L/R or U/D",
        "Hold & push",
        "Orbit keyboard around head",
        False,
    ) in en_rows
    assert (
        "左 Y 键",
        "短按",
        "重置屏幕位置和尺寸",
        False,
    ) in cn_rows
    assert (
        "Left Y button",
        "Short press",
        "Reset screen position and size",
        False,
    ) in en_rows
    assert not any("摇杆按下" in row[0] and "键盘球面环绕" in row[2] for row in cn_rows)
    assert not any("stick press" in row[0] and "Orbit keyboard" in row[2] for row in en_rows)


def test_left_stick_radial_deadzone_and_response_curve_are_bounded() -> None:
    normalize = CoreControllerInputMixin._normalize_left_stick

    assert normalize(0.05, 0.0) == (0.0, 0.0)
    assert normalize(0.08, 0.0) == (0.0, 0.0)
    low_x, low_y = normalize(0.10, 0.0)
    mid_x, _mid_y = normalize(0.5, 0.0)
    full_x, _full_y = normalize(1.0, 0.0)
    diagonal_x, diagonal_y = normalize(1.0, 1.0)

    assert low_x > 0.0 and low_y == 0.0
    assert low_x < mid_x < full_x == 1.0
    assert diagonal_x == diagonal_y
    diagonal_magnitude = (diagonal_x**2 + diagonal_y**2) ** 0.5
    assert 0.99 < diagonal_magnitude <= 1.0
    assert normalize(-0.10, 0.0)[0] == -low_x


def test_left_stick_activation_is_lower_without_changing_right_stick_deadzone() -> None:
    host = GuideHost()

    assert host._guide_axis_active(0.10, left=True)
    assert not host._guide_axis_active(0.10)


def test_disabled_controller_sampling_reads_only_both_trigger_actions() -> None:
    class Runtime:
        def __init__(self) -> None:
            self.synced = False

        def sync_actions(self, _session, _sync_info) -> None:
            self.synced = True

    class InputHost(CoreControllerInputMixin):
        def __init__(self) -> None:
            self.xr = Runtime()
            self.session = object()
            self._xr_actions_sync_info = object()
            self._act_left_trigger = "left-trigger"
            self._act_right_trigger = "right-trigger"
            self._controller_input_disabled = True
            self._controller_toggle_wait_for_release = False
            self._controller_inputs = ({}, {})
            self.read_actions: list[tuple[str, str]] = []

        def _read_float_action(self, action, hand_path: str) -> float:
            self.read_actions.append((action, hand_path))
            return 0.8 if action == "left-trigger" else 0.9

    host = InputHost()
    host._sync_controller_inputs(1.0 / 90.0)

    assert host.xr.synced
    assert host.read_actions == [
        ("left-trigger", "/user/hand/left"),
        ("right-trigger", "/user/hand/right"),
    ]
    assert host._controller_inputs == (
        {"trigger": 0.8},
        {"trigger": 0.9},
    )
