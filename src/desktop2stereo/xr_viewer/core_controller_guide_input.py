from __future__ import annotations


class CoreControllerGuideInputMixin:
    """Resolve continuous and chorded controls listed in the operation guide."""

    _GUIDE_LEFT_DEADZONE = 0.03
    _GUIDE_RIGHT_DEADZONE = 0.15
    _BRAND_SWITCH_SECONDS = 0.5
    _CALIBRATION_SECONDS = 5.0
    _SCREEN_ROTATION_SPEED = 45.0
    _SCREEN_SIZE_SPEED = 1.2
    _SCREEN_DISTANCE_SPEED = 1.5
    _DEPTH_STRENGTH_SPEED = 0.5

    def _init_controller_guide_input(self) -> None:
        self._guide_ab_started_at = 0.0
        self._guide_brand_switch_fired = False
        self._guide_calibration_fired = False
        self._guide_calibration_b_last = False

    def _emit_guide_action(self, action: str, **values) -> None:
        dispatcher = getattr(self, "_dispatch_controller_shortcut", None)
        if callable(dispatcher):
            dispatcher(action, **values)

    def _guide_axis_active(self, value: float, *, left: bool = False) -> bool:
        deadzone = (
            self._GUIDE_LEFT_DEADZONE if left else self._GUIDE_RIGHT_DEADZONE
        )
        return abs(float(value)) > deadzone

    def _handle_controller_guide_input(self, delta_seconds: float) -> None:
        """Resolve one frame without applying renderer-specific mutations."""
        left, right = self._controller_inputs
        dt = max(0.0, min(0.1, float(delta_seconds)))
        now = self._shortcut_time()
        pressed = self._shortcut_pressed
        a = pressed(right, "a_button")
        b = pressed(right, "b_button")
        ab = a and b
        calibration = bool(getattr(self, "_controller_calibration_mode", False))

        if ab:
            if self._guide_ab_started_at <= 0.0:
                self._guide_ab_started_at = now
            held = now - self._guide_ab_started_at
            controller_model_available = not bool(
                getattr(self, "_vulkan_controller_proxy_enabled", False)
            )
            if (
                controller_model_available
                and held >= self._BRAND_SWITCH_SECONDS
                and not self._guide_brand_switch_fired
            ):
                self._emit_guide_action("switch_controller_brand")
                self._guide_brand_switch_fired = True
            if held >= self._CALIBRATION_SECONDS and not self._guide_calibration_fired:
                self._emit_guide_action("toggle_controller_calibration")
                self._guide_calibration_fired = True
        else:
            self._guide_ab_started_at = 0.0
            self._guide_brand_switch_fired = False
            self._guide_calibration_fired = False

        if ab:
            return

        lx = float(left.get("joystick_x", 0.0) or 0.0)
        ly = float(left.get("joystick_y", 0.0) or 0.0)
        rx = float(right.get("joystick_x", 0.0) or 0.0)
        ry = float(right.get("joystick_y", 0.0) or 0.0)

        if calibration:
            if (
                self._guide_axis_active(ly, left=True)
                or self._guide_axis_active(rx)
                or self._guide_axis_active(ry)
            ):
                self._emit_guide_action(
                    "adjust_controller_calibration",
                    offset_y=ly * 0.15 * dt,
                    offset_z=ry * 0.15 * dt,
                    rotation_deg=rx * 45.0 * dt,
                )
            if b and not a and not self._guide_calibration_b_last:
                self._emit_guide_action("save_controller_calibration")
            self._guide_calibration_b_last = b and not a
            return
        self._guide_calibration_b_last = False

        grip_l = pressed(left, "grip")
        grip_r = pressed(right, "grip")
        keyboard = bool(getattr(self, "_keyboard_visible", False))
        left_target = getattr(self, "_grip_target_l", None)
        right_target = getattr(self, "_grip_target_r", None)
        if grip_l and not grip_r:
            if keyboard and left_target == "keyboard":
                # The legacy viewer used the left stick axes directly for
                # keyboard orbit. Stick click remains a separate shortcut.
                if (
                    self._guide_axis_active(lx, left=True)
                    or self._guide_axis_active(ly, left=True)
                ):
                    self._emit_guide_action(
                        "orbit_keyboard", horizontal=lx * dt, vertical=ly * dt
                    )
                if self._guide_axis_active(rx) or self._guide_axis_active(ry):
                    self._emit_guide_action(
                        "rotate_keyboard",
                        yaw_delta=-rx * self._SCREEN_ROTATION_SPEED * dt,
                        pitch_delta=ry * self._SCREEN_ROTATION_SPEED * dt,
                    )
            else:
                # Preserve both legacy rotation mappings: left stick handles
                # local fine rotation and right stick handles independent yaw
                # and pitch rotation while the left grip is held.
                if (
                    self._guide_axis_active(lx, left=True)
                    or self._guide_axis_active(ly, left=True)
                ):
                    self._emit_guide_action(
                        "rotate_screen",
                        yaw_delta=-lx * self._SCREEN_ROTATION_SPEED * dt,
                        pitch_delta=ly * self._SCREEN_ROTATION_SPEED * dt,
                    )
                if self._guide_axis_active(rx) or self._guide_axis_active(ry):
                    self._emit_guide_action(
                        "rotate_screen",
                        yaw_delta=-rx * self._SCREEN_ROTATION_SPEED * dt,
                        pitch_delta=ry * self._SCREEN_ROTATION_SPEED * dt,
                    )
            return

        if grip_r and not grip_l:
            if keyboard and right_target == "keyboard":
                if (
                    self._guide_axis_active(lx, left=True)
                    or self._guide_axis_active(ly, left=True)
                ):
                    self._emit_guide_action(
                        "resize_keyboard",
                        width_delta=lx * self._SCREEN_SIZE_SPEED * dt,
                        distance_delta=ly * self._SCREEN_DISTANCE_SPEED * dt,
                    )
            else:
                if self._guide_axis_active(ly, left=True):
                    self._emit_guide_action(
                        "adjust_depth_strength",
                        delta=ly * self._DEPTH_STRENGTH_SPEED * dt,
                    )
                if self._guide_axis_active(rx) or self._guide_axis_active(ry):
                    self._emit_guide_action(
                        "resize_screen",
                        width_delta=rx * self._SCREEN_SIZE_SPEED * dt,
                        distance_delta=ry * self._SCREEN_DISTANCE_SPEED * dt,
                    )
            return

        if not grip_l and not grip_r and not keyboard:
            self._emit_guide_action("arrow_axes", horizontal=lx, vertical=ly)
            self._emit_guide_action("scroll_axes", horizontal=rx, vertical=ry, dt=dt)
