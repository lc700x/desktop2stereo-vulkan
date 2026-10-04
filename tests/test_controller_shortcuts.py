from __future__ import annotations

from xr_viewer.core_controller_shortcuts import CoreControllerShortcutsMixin


class ShortcutHost(CoreControllerShortcutsMixin):
    def __init__(self) -> None:
        self._frame_now = 1.0
        self._controller_inputs = ({}, {})
        self._keyboard_visible = False
        self._grip_target_l = None
        self.room_mode = False
        self.actions: list[str] = []
        self._init_controller_shortcuts()

    def _dispatch_controller_shortcut(self, action: str, **_values) -> None:
        self.actions.append(action)

    def _operation_guide_environment_mode(self) -> bool:
        return self.room_mode

    def update(self, *, left=None, right=None, after=0.0) -> None:
        self._frame_now += float(after)
        self._controller_inputs = (left or {}, right or {})
        self._handle_controller_shortcuts()


def test_face_buttons_match_legacy_short_and_long_actions() -> None:
    host = ShortcutHost()

    host.update(right={"a_button": 1.0})
    host.update(right={}, after=0.2)
    host.update(right={"a_button": 1.0}, after=0.1)
    host.update(right={"a_button": 1.0}, after=1.01)
    host.update(right={}, after=0.1)
    host.update(right={"b_button": 1.0}, after=0.1)
    host.update(right={}, after=0.2)
    host.update(left={"y_button": 1.0}, after=0.1)
    host.update(left={"y_button": 1.0}, after=1.01)
    host.update(left={}, after=0.1)

    assert host.actions == [
        "toggle_screen_shape",
        "cycle_status_panel",
        "toggle_passthrough",
        "cycle_screen_preset",
    ]


def test_x_press_toggles_keyboard_short_and_skybox_black_after_one_second() -> None:
    host = ShortcutHost()

    host.update(left={"x_button": 1.0})
    host.update(left={}, after=0.2)
    host.update(left={"x_button": 1.0}, after=0.1)
    host.update(left={"x_button": 1.0}, after=1.0)
    host.update(left={"x_button": 1.0}, after=3.01)
    host.update(left={}, after=0.1)

    assert host.actions == [
        "toggle_keyboard",
        "toggle_environment_background",
    ]


def test_menu_and_stick_clicks_emit_shared_shortcuts() -> None:
    host = ShortcutHost()

    host.update(left={"menu_button": 1.0})
    host.update(left={}, after=0.2)
    host.update(left={"stick_click": 1.0}, after=0.1)
    host.update(left={}, after=0.2)
    host.update(left={"stick_click": 1.0}, after=0.1)
    host.update(left={"stick_click": 1.0}, after=1.01)
    host.update(left={}, after=0.1)
    host.update(right={"stick_click": 1.0}, after=0.1)
    host.update(right={}, after=0.2)
    host.update(right={"stick_click": 1.0}, after=0.1)
    host.update(right={"stick_click": 1.0}, after=1.01)

    assert host.actions == [
        "cycle_status_panel",
        "copy",
        "cut",
        "paste",
        "enter",
    ]


def test_grip_stick_actions_and_ab_combo_suppress_normal_buttons() -> None:
    host = ShortcutHost()

    host.update(left={"grip": 1.0, "stick_click": 1.0})
    host.update(left={"grip": 1.0})
    host.update(right={"grip": 1.0, "stick_click": 1.0})
    host.update(right={"grip": 1.0})
    host.update(right={"a_button": 1.0, "b_button": 1.0})
    host.update(right={})

    assert host.actions == ["toggle_stereo", "reset_depth"]


def test_keyboard_orbit_stick_click_does_not_toggle_stereo_or_copy() -> None:
    host = ShortcutHost()
    host._keyboard_visible = True
    host._grip_target_l = "keyboard"

    host.update(left={"grip": 1.0, "stick_click": 1.0})
    host.update(left={"grip": 1.0})
    host._grip_target_l = None
    host.update(left={})

    assert host.actions == []


def test_room_y_long_press_cycles_environment_instead_of_screen_preset() -> None:
    host = ShortcutHost()
    host.room_mode = True

    host.update(left={"y_button": 1.0})
    host.update(left={"y_button": 1.0}, after=1.0)
    host.update(left={})

    assert host.actions == ["cycle_room_environment"]


def test_b_long_press_toggles_operation_guide() -> None:
    host = ShortcutHost()

    host.update(right={"b_button": 1.0})
    host.update(right={"b_button": 1.0}, after=1.0)
    host.update(right={})

    assert host.actions == ["toggle_operation_guide"]


def test_both_triggers_toggle_controller_input_and_are_the_only_disabled_input() -> None:
    host = ShortcutHost()

    host.update(left={"trigger": 1.0}, right={"trigger": 1.0})
    host.update(left={"trigger": 1.0}, right={"trigger": 1.0}, after=5.0)
    assert host._controller_input_disabled is True
    assert host.actions == ["controller_input_mode_changed"]

    host.update(right={"a_button": 1.0, "stick_click": 1.0})
    host.update(left={"menu_button": 1.0}, right={"b_button": 1.0})
    assert host.actions == ["controller_input_mode_changed"]

    # Release the activation chord, then use it again to restore input.
    host.update()
    host.update(left={"trigger": 1.0}, right={"trigger": 1.0})
    host.update(left={"trigger": 1.0}, right={"trigger": 1.0}, after=5.0)
    assert host._controller_input_disabled is False
    assert host.actions == [
        "controller_input_mode_changed",
        "controller_input_mode_changed",
    ]


def test_both_trigger_toggle_resets_if_either_trigger_releases_before_five_seconds() -> None:
    host = ShortcutHost()

    host.update(left={"trigger": 1.0}, right={"trigger": 1.0})
    host.update(left={"trigger": 1.0}, right={"trigger": 1.0}, after=4.9)
    host.update(left={}, right={"trigger": 1.0}, after=0.1)

    assert host._controller_input_disabled is False
    assert host.actions == []
