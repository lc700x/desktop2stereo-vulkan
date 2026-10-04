"""OpenXR controller input values used by Filament controller animations."""

from __future__ import annotations

import ctypes


_VIVE_TB_Y = 0.5
_LEFT_STICK_DEADZONE = 0.08
_LEFT_STICK_RESPONSE_EXPONENT = 0.75

class CoreControllerInputMixin:
    @staticmethod
    def _normalize_left_stick(x: float, y: float) -> tuple[float, float]:
        """Apply a small radial deadzone and responsive, bounded stick curve."""
        x = max(-1.0, min(1.0, float(x or 0.0)))
        y = max(-1.0, min(1.0, float(y or 0.0)))
        magnitude = (x * x + y * y) ** 0.5
        if magnitude > 1.0:
            x /= magnitude
            y /= magnitude
            magnitude = 1.0
        if magnitude <= _LEFT_STICK_DEADZONE:
            return 0.0, 0.0
        normalized = (magnitude - _LEFT_STICK_DEADZONE) / (
            1.0 - _LEFT_STICK_DEADZONE
        )
        response = normalized ** _LEFT_STICK_RESPONSE_EXPONENT
        scale = response / magnitude
        return x * scale, y * scale

    def _read_bool_action_raw(self, action, hand_path):
        if action is None:
            return False
        try:
            path = self._path_left if hand_path == "/user/hand/left" else self._path_right
            state = self.xr.get_action_state_boolean(
                self.session,
                self.xr.ActionStateGetInfo(action=action, subaction_path=path),
            )
            return bool(state.is_active and state.current_state)
        except Exception:
            return False

    def _read_bool_action(self, action, hand_path):
        """Read a button, including the legacy Vive/WMR trackpad emulation."""
        pressed = self._read_bool_action_raw(action, hand_path)
        if action is self._act_y_btn and hand_path == "/user/hand/left":
            pressed = pressed or getattr(self, "_emu_y", False)
        elif action is self._act_x_btn and hand_path == "/user/hand/left":
            pressed = pressed or getattr(self, "_emu_x", False)
        elif action is self._act_b_btn and hand_path == "/user/hand/right":
            pressed = pressed or getattr(self, "_emu_b", False)
        elif action is self._act_a_btn and hand_path == "/user/hand/right":
            pressed = pressed or getattr(self, "_emu_a", False)
        elif action is self._act_left_stick_click and hand_path == "/user/hand/left":
            pressed = False if (
                getattr(self, "_emu_x", False) or getattr(self, "_emu_y", False)
            ) else (pressed or getattr(self, "_emu_lsc", False))
        elif action is self._act_right_stick_click and hand_path == "/user/hand/right":
            pressed = False if (
                getattr(self, "_emu_a", False) or getattr(self, "_emu_b", False)
            ) else (pressed or getattr(self, "_emu_rsc", False))
        return pressed

    def _read_bool_edge(self, action, hand_path, previous_state):
        """Return a rising edge using runtime change state with a safe fallback."""
        if action is None:
            return False
        try:
            path = self._path_left if hand_path == "/user/hand/left" else self._path_right
            state = self.xr.get_action_state_boolean(
                self.session,
                self.xr.ActionStateGetInfo(action=action, subaction_path=path),
            )
            pressed = self._read_bool_action(action, hand_path)
            changed = bool(getattr(state, "changed", False))
            if not changed:
                try:
                    ptr = ctypes.cast(ctypes.byref(state), ctypes.POINTER(ctypes.c_int32))
                    changed = bool(ptr[2])
                except Exception:
                    pass
            return pressed if changed else pressed and not previous_state
        except Exception:
            return False

    def _update_trackpad_button_emu(self) -> None:
        """Preserve legacy trackpad top/center/bottom button emulation."""
        for hand, stick_act, click_act, top_attr, bottom_attr, center_attr in (
            (
                "/user/hand/left",
                self._act_left_stick,
                self._act_left_stick_click,
                "_emu_y",
                "_emu_x",
                "_emu_lsc",
            ),
            (
                "/user/hand/right",
                self._act_right_stick,
                self._act_right_stick_click,
                "_emu_b",
                "_emu_a",
                "_emu_rsc",
            ),
        ):
            if not self._read_bool_action_raw(click_act, hand):
                setattr(self, top_attr, False)
                setattr(self, bottom_attr, False)
                setattr(self, center_attr, False)
                continue
            try:
                path = self._path_left if hand == "/user/hand/left" else self._path_right
                state = self.xr.get_action_state_vector2f(
                    self.session,
                    self.xr.ActionStateGetInfo(action=stick_act, subaction_path=path),
                )
                value_y = float(state.current_state.y) if state.is_active else 0.0
            except Exception:
                value_y = 0.0
            setattr(self, top_attr, value_y > _VIVE_TB_Y)
            setattr(self, bottom_attr, value_y < -_VIVE_TB_Y)
            setattr(self, center_attr, abs(value_y) <= _VIVE_TB_Y)

    def _read_float_action(self, action, hand_path):
        if action is None:
            return 0.0
        try:
            path = self._path_left if hand_path == "/user/hand/left" else self._path_right
            state = self.xr.get_action_state_float(
                self.session,
                self.xr.ActionStateGetInfo(action=action, subaction_path=path),
            )
            return float(state.current_state) if state.is_active else 0.0
        except Exception:
            return 0.0

    def _update_controller_press_animation_state(
        self, dt: float = 0.0, lx: float = 0.0, ly: float = 0.0,
        rx: float = 0.0, ry: float = 0.0,
    ) -> None:
        """Keep legacy input smoothing and touch/click animation semantics."""
        lx = max(-1.0, min(1.0, float(lx or 0.0)))
        ly = max(-1.0, min(1.0, float(ly or 0.0)))
        rx = max(-1.0, min(1.0, float(rx or 0.0)))
        ry = max(-1.0, min(1.0, float(ry or 0.0)))

        left_touched = (
            self._read_bool_action(getattr(self, "_act_left_stick_touch", None), "/user/hand/left")
            or abs(lx) > 0.02 or abs(ly) > 0.02
            or self._read_bool_action(self._act_left_stick_click, "/user/hand/left")
        )
        right_touched = (
            self._read_bool_action(getattr(self, "_act_right_stick_touch", None), "/user/hand/right")
            or abs(rx) > 0.02 or abs(ry) > 0.02
            or self._read_bool_action(self._act_right_stick_click, "/user/hand/right")
        )

        def button(action, hand):
            return 1.0 if self._read_bool_action(action, hand) else 0.0

        left = {
            "trigger": self._read_float_action(self._act_left_trigger, "/user/hand/left"),
            "grip": button(self._act_left_grip, "/user/hand/left"),
            "x_button": button(self._act_x_btn, "/user/hand/left"),
            "y_button": button(self._act_y_btn, "/user/hand/left"),
            "joystick": button(self._act_left_stick_click, "/user/hand/left"),
            "joystick_x": lx, "joystick_y": -ly,
            "joystick_touched": 1.0 if left_touched else 0.0,
            "touchpad": button(self._act_left_stick_click, "/user/hand/left"),
            "touchpad_x": lx, "touchpad_y": -ly,
            "touchpad_touched": 1.0 if left_touched else 0.0,
            "menu_button": button(self._act_menu_btn, "/user/hand/left"),
        }
        right = {
            "trigger": self._read_float_action(self._act_right_trigger, "/user/hand/right"),
            "grip": button(self._act_right_grip, "/user/hand/right"),
            "a_button": button(self._act_a_btn, "/user/hand/right"),
            "b_button": button(self._act_b_btn, "/user/hand/right"),
            "joystick": button(self._act_right_stick_click, "/user/hand/right"),
            "joystick_x": rx, "joystick_y": -ry,
            "joystick_touched": 1.0 if right_touched else 0.0,
            "touchpad": button(self._act_right_stick_click, "/user/hand/right"),
            "touchpad_x": rx, "touchpad_y": -ry,
            "touchpad_touched": 1.0 if right_touched else 0.0,
        }
        dt = max(0.0, min(0.050, float(dt or 0.0)))
        alpha = 1.0 if dt <= 0.0 else min(1.0, dt * 24.0)

        def smooth(current, target):
            return {
                key: float(current.get(key, 0.0) or 0.0)
                + (float(value) - float(current.get(key, 0.0) or 0.0)) * alpha
                for key, value in target.items()
            }

        self._ctrl_press_l = smooth(getattr(self, "_ctrl_press_l", {}), left)
        self._ctrl_press_r = smooth(getattr(self, "_ctrl_press_r", {}), right)

    def _read_stick_action(self, action, hand_path):
        if action is None:
            return 0.0, 0.0
        try:
            path = self._path_left if hand_path == "/user/hand/left" else self._path_right
            state = self.xr.get_action_state_vector2f(
                self.session,
                self.xr.ActionStateGetInfo(action=action, subaction_path=path),
            )
            if not state.is_active:
                return 0.0, 0.0
            return float(state.current_state.x), -float(state.current_state.y)
        except Exception:
            return 0.0, 0.0

    def _sync_controller_inputs(self, delta_seconds: float) -> None:
        self.xr.sync_actions(self.session, self._xr_actions_sync_info)
        if (
            getattr(self, "_controller_input_disabled", False)
            or getattr(self, "_controller_toggle_wait_for_release", False)
        ):
            left_trigger = self._read_float_action(
                self._act_left_trigger, "/user/hand/left"
            )
            right_trigger = self._read_float_action(
                self._act_right_trigger, "/user/hand/right"
            )
            self._controller_inputs = (
                {"trigger": left_trigger},
                {"trigger": right_trigger},
            )
            return
        self._update_trackpad_button_emu()
        lx, ly = self._read_stick_action(self._act_left_stick, "/user/hand/left")
        rx, ry = self._read_stick_action(self._act_right_stick, "/user/hand/right")
        lx, ly = self._normalize_left_stick(lx, ly)

        def values(hand: str, left: bool) -> dict[str, float]:
            prefix = "left_" if left else "right_"
            trigger = self._read_float_action(
                self._act_left_trigger if left else self._act_right_trigger, hand
            )
            grip = 1.0 if self._read_bool_action(
                self._act_left_grip if left else self._act_right_grip, hand
            ) else 0.0
            buttons = {
                "a_button": self._act_a_btn,
                "b_button": self._act_b_btn,
                "x_button": self._act_x_btn,
                "y_button": self._act_y_btn,
                "menu_button": self._act_menu_btn,
            }
            result = {"trigger": trigger, "grip": grip}
            for name, action in buttons.items():
                result[name] = 1.0 if self._read_bool_action(action, hand) else 0.0
            return result

        left = values("/user/hand/left", True)
        right = values("/user/hand/right", False)
        left["stick_click"] = 1.0 if self._read_bool_action(self._act_left_stick_click, "/user/hand/left") else 0.0
        right["stick_click"] = 1.0 if self._read_bool_action(self._act_right_stick_click, "/user/hand/right") else 0.0
        left_touched = self._read_bool_action(
            self._act_left_stick_touch, "/user/hand/left"
        ) or bool(left.get("stick_click", 0.0))
        right_touched = self._read_bool_action(
            self._act_right_stick_touch, "/user/hand/right"
        ) or bool(right.get("stick_click", 0.0))
        left.update({
            "joystick_x": lx,
            "joystick_y": ly,
            "joystick_touched": 1.0 if left_touched else 0.0,
            "touchpad_x": lx,
            "touchpad_y": ly,
            "touchpad_touched": 1.0 if left_touched else 0.0,
        })
        right.update({
            "joystick_x": rx,
            "joystick_y": ry,
            "joystick_touched": 1.0 if right_touched else 0.0,
            "touchpad_x": rx,
            "touchpad_y": ry,
            "touchpad_touched": 1.0 if right_touched else 0.0,
        })
        self._controller_inputs = (left, right)
        self._update_controller_press_animation_state(delta_seconds, lx, ly, rx, ry)

    def _controller_input(self, hand: int) -> dict[str, float]:
        values = getattr(self, "_controller_inputs", ({}, {}))
        return values[0 if int(hand) == 0 else 1]
