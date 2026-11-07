from __future__ import annotations

from pathlib import Path

from path_config import APP_ROOT

import numpy as np

from xr_viewer.core_openxr_vulkan import OpenXrVulkanPresenter


ROOT = Path(__file__).resolve().parents[1]


def test_behavior_matrix_tracks_old_and_vulkan_functions() -> None:
    matrix = (
        ROOT / "docs" / "05-openxr-behavior-migration-matrix.md"
    ).read_text(encoding="utf-8")

    required_pairs = (
        ("CoreOpenXRInputMixin._poll_xr_events", "OpenXrVulkanPresenter.poll_events"),
        ("CoreWindowInputMixin._cycle_a_panel", "_set_shortcut_panel"),
        ("CoreWindowInputMixin._cycle_b_panel", "_set_hand_shortcut_panel"),
        ("implementation._laser_screen_hit_uv", "_screen_ray_hit"),
        ("CoreLaserRenderMixin._laser_beam_setup", "bridge_laser.cpp"),
        ("core_screen_quality._prepare_screen_quality_texture", "Filament external screen image path"),
    )
    for old_symbol, vulkan_symbol in required_pairs:
        assert old_symbol in matrix
        assert vulkan_symbol in matrix


def test_laser_hit_rings_use_the_vulkan_quad_contract() -> None:
    source = (
        APP_ROOT / "xr_viewer" / "core_openxr_vulkan.py"
    ).read_text(encoding="utf-8")

    assert "def _cursor_overlay_specs" in source
    assert "specs.extend(self._cursor_overlay_specs" in source
    assert "layers = []" in source
    assert "layer = self._upload_tool_quad(*spec)" in source
    assert "layers.extend(item for item in layer if item is not None)" in source
    assert "CompositionLayerQuad" in source
    assert "_render_laser_hit_circles" not in source


def test_left_grip_guide_preserves_both_legacy_rotation_sticks() -> None:
    presenter = OpenXrVulkanPresenter()
    actions: list[tuple[str, dict]] = []
    presenter._dispatch_controller_shortcut = (
        lambda action, **values: actions.append((action, values))
    )
    presenter._controller_inputs = (
        {
            "grip": 1.0,
            "joystick_x": 0.4,
            "joystick_y": 0.0,
        },
        {
            "joystick_x": -0.3,
            "joystick_y": 0.2,
        },
    )
    presenter._grip_target_l = "screen"

    presenter._handle_controller_guide_input(1.0 / 90.0)

    rotations = [values for action, values in actions if action == "rotate_screen"]
    assert len(rotations) == 2
    assert rotations[0]["yaw_delta"] < 0.0
    assert rotations[1]["yaw_delta"] > 0.0
    assert rotations[1]["pitch_delta"] > 0.0


def test_keyboard_left_grip_orbit_does_not_require_stick_click() -> None:
    presenter = OpenXrVulkanPresenter()
    actions: list[tuple[str, dict]] = []
    presenter._dispatch_controller_shortcut = (
        lambda action, **values: actions.append((action, values))
    )
    presenter._controller_inputs = (
        {
            "grip": 1.0,
            "joystick_x": 0.25,
            "joystick_y": -0.2,
            "stick_click": 0.0,
        },
        {},
    )
    presenter._keyboard_visible = True
    presenter._grip_target_l = "keyboard"

    presenter._handle_controller_guide_input(1.0 / 90.0)

    assert [action for action, _values in actions] == ["orbit_keyboard"]


def test_reference_space_recreation_restarts_profile_application() -> None:
    class FakeXR:
        ReferenceSpaceCreateInfo = staticmethod(
            lambda **kwargs: ("create_info", kwargs)
        )

        def __init__(self) -> None:
            self.destroyed: list[object] = []

        def create_reference_space(self, session, info):
            assert session == "session"
            assert info[1]["reference_space_type"] == "stage"
            return "new-space"

        def destroy_space(self, space) -> None:
            self.destroyed.append(space)

    presenter = OpenXrVulkanPresenter()
    presenter.xr = FakeXR()
    presenter.session = "session"
    presenter._reference_space_type = "stage"
    presenter.reference_space = "old-space"
    presenter._xr_space = "old-space"
    presenter._profile_space_applied = True
    presenter._profile_initial_head = np.eye(4, dtype=np.float32)
    presenter._head_position_w = np.ones(3, dtype=np.float64)
    presenter._head_forward_w = np.ones(3, dtype=np.float64)

    presenter._recreate_reference_space_after_runtime_change()

    assert presenter.reference_space == "new-space"
    assert presenter._xr_space == "new-space"
    assert presenter._profile_space_applied is False
    assert presenter._profile_initial_head is None
    assert presenter._head_position_w is None
    assert presenter._head_forward_w is None
    assert presenter.xr.destroyed == ["old-space"]


def test_curved_screen_ray_hit_matches_legacy_cylinder_uv() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = (
        (0.0, 0.0, -2.0), 2.4, 1.35, (0.0, 0.0, 0.0)
    )
    presenter._screen_curved = True
    ray_matrix = np.eye(4, dtype=np.float32)

    center_hit = presenter._screen_ray_hit(
        ray_matrix,
        ray_origin=np.asarray((0.0, 0.0, 0.0)),
        ray_direction=np.asarray((0.0, 0.0, -1.0)),
    )
    assert center_hit == (0.5, 0.5)

    radius = 2.4 / 2.0 / 0.72
    endpoint = np.asarray(
        (radius * np.sin(0.72), 0.0, -2.0 + radius * (1.0 - np.cos(0.72)))
    )
    edge_hit = presenter._screen_ray_hit(
        ray_matrix,
        ray_origin=np.asarray((0.0, 0.0, 0.0)),
        ray_direction=endpoint / np.linalg.norm(endpoint),
    )
    assert edge_hit is not None
    assert edge_hit[0] == 1.0
    assert edge_hit[1] == 0.5
    np.testing.assert_allclose(
        presenter._screen_uv_to_world(*edge_hit), endpoint, atol=1e-5
    )


def test_flat_screen_ray_hit_keeps_legacy_bottom_to_top_v() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = (
        (0.0, 0.0, -2.0), 2.4, 1.35, (0.0, 0.0, 0.0)
    )
    ray_matrix = np.eye(4, dtype=np.float32)

    bottom = presenter._screen_ray_hit(
        ray_matrix,
        ray_origin=np.asarray((0.0, -0.6, 0.0)),
        ray_direction=np.asarray((0.0, 0.0, -1.0)),
    )
    top = presenter._screen_ray_hit(
        ray_matrix,
        ray_origin=np.asarray((0.0, 0.6, 0.0)),
        ray_direction=np.asarray((0.0, 0.0, -1.0)),
    )

    assert bottom is not None and top is not None
    assert bottom[1] < 0.5
    assert top[1] > 0.5


def test_adjust_depth_strength_remains_runtime_owned() -> None:
    actions: list[tuple[str, dict]] = []
    presenter = OpenXrVulkanPresenter(
        on_controller_shortcut=lambda action, **values: (
            actions.append((action, values)) or True
        )
    )

    presenter._dispatch_controller_shortcut(
        "adjust_depth_strength", delta=0.25
    )

    assert actions == [("adjust_depth_strength", {"delta": 0.25})]
    assert presenter._unsupported_shortcut_actions == set()
