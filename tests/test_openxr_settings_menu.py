import pytest

from gui.localization import gettext_for
from xr_viewer.settings_menu import (
    OpenXrSettingsMenu,
    _CONTENT_LEFT,
    _CONTENT_PANEL_LEFT,
    _CONTENT_PANEL_RIGHT,
    _CONTENT_RIGHT,
    _MENU_GROUP_GAP,
    _MENU_GROUP_PADDING,
    _MENU_TITLE_CARD_GAP,
    _MENU_TITLE_HEIGHT,
    _SIDEBAR_PANEL_RIGHT,
    PICTURE_DEFAULTS,
    SETTINGS_MENU_TEXTURE_SIZE,
    clamp_picture_values,
)


def test_picture_layout_exposes_all_planned_controls():
    menu = OpenXrSettingsMenu()
    menu.set_tab("picture")
    keys = {control.key for control in menu.controls()}
    assert {
        "openxr_render_scale", "color_brightness", "color_contrast", "color_saturation", "color_gamma",
        "color_temperature", "color_tint", "vulkan_projection_min_lod",
        "vulkan_projection_max_lod", "vulkan_projection_mip_lod_bias",
        "vulkan_projection_rcas_sharpness",
    } <= keys


def test_screen_tab_precedes_picture_tab_in_shared_layout():
    menu = OpenXrSettingsMenu()
    tabs = [
        control.key for control in menu.controls(show_glow=True)
        if control.key.startswith("tab:")
    ]

    assert tabs == [
        "tab:screen", "tab:depth", "tab:glow", "tab:room", "tab:picture",
    ]


def test_screen_is_the_default_openxr_settings_tab():
    menu = OpenXrSettingsMenu()

    assert menu.tab == "screen"


def test_screen_rotation_actions_have_localized_ascii_labels():
    menu = OpenXrSettingsMenu()
    controls = {
        control.key: control for control in menu.controls(lang="CN")
        if control.key.startswith("screen:rotate:")
    }

    assert gettext_for("EN", controls["screen:rotate:-90"].label) == "Rotate -90"
    assert gettext_for("EN", controls["screen:rotate:+90"].label) == "Rotate +90"
    assert gettext_for("CN", controls["screen:rotate:-90"].label) == "\u5de6\u8f6c 90 \u5ea6"
    assert gettext_for("CN", controls["screen:rotate:+90"].label) == "\u53f3\u8f6c 90 \u5ea6"


def test_slider_hit_and_quantization():
    menu = OpenXrSettingsMenu()
    menu.set_tab("picture")
    brightness = next(control for control in menu.controls() if control.key == "color_brightness")
    x0, y0, x1, y1 = brightness.rect
    control = menu.hit_test(((x0 + x1) * 0.5, (y0 + y1) * 0.5))
    assert control is not None and control.key == "color_brightness"
    assert control.value_from_u((x0 + x1) * 0.5) == 1.1


def test_outside_click_opens_only_after_release():
    menu = OpenXrSettingsMenu()
    assert menu.sample_trigger(0, 0.8, outside_targets=True) is False
    assert menu.sample_trigger(0, 0.2, outside_targets=True) is True
    assert menu.sample_trigger(1, 0.8, outside_targets=False) is False
    assert menu.sample_trigger(1, 0.2, outside_targets=False) is False


def test_disabled_curve_control_does_not_hit():
    menu = OpenXrSettingsMenu()
    menu.set_tab("screen")
    subtle = next(
        control for control in menu.controls(allow_curve=False)
        if control.key == "screen:type:subtle"
    )
    x0, y0, x1, y1 = subtle.rect
    assert menu.hit_test(
        ((x0 + x1) * 0.5, (y0 + y1) * 0.5), allow_curve=False
    ) is None


def test_min_lod_never_exceeds_max_lod():
    values = clamp_picture_values({
        "vulkan_projection_min_lod": 1.5,
        "vulkan_projection_max_lod": 0.5,
    })
    assert values["vulkan_projection_min_lod"] == 0.5


def test_tab_switch_rebuilds_page_controls():
    menu = OpenXrSettingsMenu()
    assert menu.set_tab("picture") is True
    assert menu.set_tab("screen") is True
    keys = {control.key for control in menu.controls()}
    assert {
        "screen:width", "screen:height", "screen:type:flat",
        "screen:type:subtle", "screen:type:medium", "screen:type:deep",
    } <= keys
    assert "color_brightness" not in keys


def test_depth_tab_exposes_runtime_depth_controls():
    menu = OpenXrSettingsMenu()
    assert menu.set_tab("depth") is True
    controls = {control.key: control for control in menu.controls()}
    keys = set(controls)
    assert {
        "depth_strength", "depth:toggle_stereo",
        "depth:toggle_cross_eyed", "reset:depth_stereo",
    } <= keys
    assert {controls[key].group for key in (
        "depth_strength", "depth:toggle_stereo",
        "depth:toggle_cross_eyed", "reset:depth_stereo",
    )} == {"depth_stereo"}
    assert controls["depth:toggle_stereo"].kind == "toggle"
    assert controls["depth:toggle_cross_eyed"].kind == "toggle"
    depth = next(control for control in menu.controls() if control.key == "depth_strength")
    assert (depth.minimum, depth.maximum, depth.step) == (0.0, 1.0, 0.05)


def test_glow_tab_is_visible_only_for_default_environment():
    menu = OpenXrSettingsMenu()
    assert "tab:glow" not in {control.key for control in menu.controls()}
    assert "tab:glow" in {
        control.key for control in menu.controls(show_glow=True)
    }
    menu.set_tab("glow")
    assert {
        "glow:surround", "glow:glow", "glow:veil", "glow:off",
        "glow:transparency", "reset:glow_modes", "reset:glow_transparency",
    } <= {control.key for control in menu.controls(show_glow=True)}
    transparency = next(
        control for control in menu.controls(show_glow=True)
        if control.key == "glow:transparency"
    )
    assert (
        transparency.minimum,
        transparency.maximum,
        transparency.step,
    ) == (0.0, 1.0, 0.05)
    assert not any(
        control.key.startswith("glow:") for control in menu.controls()
    )


def test_sidebar_navigation_has_fixed_aligned_targets_for_every_locale():
    menu = OpenXrSettingsMenu()
    english = [
        control for control in menu.controls(show_glow=True, lang="EN")
        if control.key.startswith("tab:")
    ]
    chinese = [
        control for control in menu.controls(show_glow=True, lang="CN")
        if control.key.startswith("tab:")
    ]
    assert len(english) == 5
    assert all(
        tuple(round(value * size) for value, size in zip(
            control.rect, SETTINGS_MENU_TEXTURE_SIZE * 2
        )) == (48, 64 + index * 80, 240, 128 + index * 80)
        for index, control in enumerate(english)
    )
    assert all(
        (right.rect[1] - left.rect[3]) * SETTINGS_MENU_TEXTURE_SIZE[1] == pytest.approx(16)
        for left, right in zip(english, english[1:])
    )
    assert all(
        round((control.rect[2] - control.rect[0]) * SETTINGS_MENU_TEXTURE_SIZE[0]) % 8 == 0
        for control in chinese
    )
    assert {
        tuple(round(value * size) for value, size in zip(
            control.rect, SETTINGS_MENU_TEXTURE_SIZE * 2
        ))
        for control in chinese
    } == {
        tuple(round(value * size) for value, size in zip(
            control.rect, SETTINGS_MENU_TEXTURE_SIZE * 2
        ))
        for control in english
    }


def test_stop_target_is_fixed_on_every_page_and_disabled_while_stopping():
    menu = OpenXrSettingsMenu()
    for tab in menu.tabs:
        menu.set_tab(tab)
        stop = next(item for item in menu.controls(show_glow=True) if item.key == "runtime:stop")
        assert stop.label == "Stop"
        assert stop.enabled
        assert tuple(round(value * size) for value, size in zip(
            stop.rect, (1024, 832, 1024, 832)
        )) == (48, 704, 240, 768)
        assert menu.hit_test(
            ((48 + 240) / (2 * 1024), (704 + 768) / (2 * 832)),
            show_glow=True,
        ).key == "runtime:stop"

    assert menu.set_stopping()
    stopping = next(item for item in menu.controls(show_glow=True) if item.key == "runtime:stop")
    assert stopping.label == "Stop"
    assert not stopping.enabled
    assert not menu.set_stopping()


def test_screen_layout_has_shape_and_placement_resets_in_their_cards():
    menu = OpenXrSettingsMenu()
    menu.set_tab("screen")
    controls = {control.key: control for control in menu.controls()}
    keys = set(controls)
    assert {
        "screen:distance", "screen:rotate:-90",
        "screen:rotate:+90",
    } <= keys
    assert controls["reset:screen_shape"].group == "screen_shape"
    assert controls["reset:screen_placement"].group == "screen_placement"
    height = next(control for control in menu.controls() if control.key == "screen:height")
    assert (height.minimum, height.maximum, height.step) == (-10.0, 10.0, 0.05)
    distance = controls["screen:distance"]
    assert (distance.minimum, distance.maximum, distance.step) == (0.25, 20.0, 0.05)
    rotations = [
        controls[key] for key in ("screen:rotate:-90", "screen:rotate:+90")
    ]
    assert max(control.rect[3] for control in rotations) < (
        controls["screen:width"].rect[1] - 0.05
    )


def test_screen_crop_section_has_symmetric_crop_controls_and_navigation():
    menu = OpenXrSettingsMenu()
    assert menu.screen_section == "layout"
    assert menu.set_screen_section("crop") is True
    controls = {control.key: control for control in menu.controls()}

    assert {
        "screen:section:layout", "screen:section:crop", "screen:auto_crop",
        "screen:dynamic_crop", "screen:reset_crop", "screen:crop_width",
        "screen:crop_height",
    } <= controls.keys()
    assert controls["screen:dynamic_crop"].kind == "toggle"
    assert controls["screen:reset_crop"].group == "screen_crop"
    assert controls["screen:crop_height"].group == "screen_crop"
    assert (
        controls["screen:crop_width"].minimum,
        controls["screen:crop_width"].maximum,
        controls["screen:crop_width"].step,
    ) == (0.0, 45.0, 1.0)
    assert "screen:width" not in controls

    assert menu.set_screen_section("layout") is True
    assert "screen:width" in {control.key for control in menu.controls()}


def test_screen_subsection_and_page_content_use_the_shared_top_alignment():
    menu = OpenXrSettingsMenu()
    controls = {control.key: control for control in menu.controls()}
    section_tabs = [
        controls["screen:section:layout"],
        controls["screen:section:crop"],
    ]
    type_controls = [
        controls[f"screen:type:{name}"]
        for name in ("flat", "subtle", "medium", "deep")
    ]

    assert min(control.rect[1] for control in section_tabs) == pytest.approx(56 / 832)
    assert max(control.rect[3] for control in section_tabs) < min(
        control.rect[1] for control in type_controls
    )
    assert min(control.rect[1] for control in type_controls) == pytest.approx(176 / 832)
    assert min(
        right.rect[0] - left.rect[2]
        for left, right in zip(type_controls, type_controls[1:])
    ) == pytest.approx(16 / SETTINGS_MENU_TEXTURE_SIZE[0])
    rotations = [
        controls["screen:rotate:-90"],
        controls["screen:rotate:+90"],
    ]
    assert max(control.rect[3] for control in type_controls) < min(
        control.rect[1] for control in rotations
    )
    assert rotations[0].rect[2] < rotations[1].rect[0]


def test_room_tab_exposes_models_three_seats_and_live_sliders():
    menu = OpenXrSettingsMenu()
    menu.room_models = (("3d_a", "Room A"), ("3d_b", "Room B"))
    menu.set_tab("room")
    controls = {control.key: control for control in menu.controls()}
    assert {
        "room:model:3d_a", "room:model:3d_b",
        "room:seat:front", "room:seat:middle", "room:seat:back",
        "room:seat_height", "room:exposure",
        "room:toggle_screen_reflection", "reset:room_seats",
        "reset:room_scene",
    } <= controls.keys()
    assert (controls["room:seat_height"].minimum, controls["room:seat_height"].maximum) == (-3.0, 3.0)
    assert (controls["room:exposure"].minimum, controls["room:exposure"].maximum) == (-8.0, 8.0)
    assert [
        controls[f"room:seat:{seat}"].label
        for seat in ("front", "middle", "back")
    ] == ["Front", "Middle", "Back"]
    assert controls["room:toggle_screen_reflection"].kind == "toggle"


def test_room_tab_remains_available_before_an_environment_is_selected():
    menu = OpenXrSettingsMenu()

    assert "tab:room" in {control.key for control in menu.controls()}
    assert menu.set_tab("room") is True


def test_room_tab_keeps_three_model_rows_above_seat_and_live_controls():
    menu = OpenXrSettingsMenu()
    menu.room_models = tuple(
        (f"room_{index}", f"Room {index}") for index in range(15)
    )
    menu.set_tab("room")
    controls = {control.key: control for control in menu.controls()}
    model_bottom = max(
        control.rect[3] for key, control in controls.items()
        if key.startswith("room:model:")
    )
    seat_top = min(
        controls[f"room:seat:{seat}"].rect[1]
        for seat in ("front", "middle", "back")
    )
    reflection = controls["room:toggle_screen_reflection"]
    seat_height = controls["room:seat_height"]
    exposure = controls["room:exposure"]

    assert model_bottom < seat_top
    assert reflection.rect[2] - reflection.rect[0] == pytest.approx(688 / 1024)
    assert seat_height.rect[3] < controls["reset:room_seats"].rect[1]
    assert controls["reset:room_seats"].rect[3] < reflection.rect[1]
    assert reflection.rect[3] < exposure.rect[1]
    assert exposure.rect[3] < controls["reset:room_scene"].rect[1]


def test_picture_resets_are_scoped_to_quality_and_color_cards():
    menu = OpenXrSettingsMenu()
    menu.set_tab("picture")
    control_list = menu.controls()
    controls = {control.key: control for control in control_list}
    assert all(
        control.rect[0] * SETTINGS_MENU_TEXTURE_SIZE[0] >= 256
        for control in control_list
        if not control.key.startswith("tab:") and control.key != "runtime:stop"
    )
    assert controls["reset:render_quality"].group == "render_quality"
    assert controls["reset:color_adjustment"].group == "color_adjustment"
    assert "section:reset_defaults" not in controls
    assert set(PICTURE_DEFAULTS) == {
        key for key, control in controls.items() if control.kind == "slider"
    }
    assert controls["openxr_render_scale"].group == "render_resolution"
    assert controls["openxr:render_auto"].group == "render_resolution"
    assert "close" not in controls


def test_picture_resolution_card_has_separate_non_overlapping_controls():
    menu = OpenXrSettingsMenu()
    menu.set_tab("picture")
    controls = {control.key: control for control in menu.controls()}
    pixels = lambda control: tuple(round(value * size) for value, size in zip(
        control.rect, (1024, 832, 1024, 832)
    ))

    auto = pixels(controls["openxr:render_auto"])
    resolution_steps = [
        pixels(controls[key]) for key in (
            "step:minus:openxr_render_scale", "step:plus:openxr_render_scale",
        )
    ]
    resolution_group = next(
        group for group in menu.layout().groups
        if group.key == "render_resolution"
    )
    quality_group = next(
        group for group in menu.layout().groups
        if group.key == "render_quality"
    )
    resolution_group_pixels = tuple(round(value * size) for value, size in zip(
        resolution_group.rect, (1024, 832, 1024, 832)
    ))
    quality_group_pixels = tuple(round(value * size) for value, size in zip(
        quality_group.rect, (1024, 832, 1024, 832)
    ))

    assert auto[3] < min(step[1] for step in resolution_steps)
    assert max(step[3] for step in resolution_steps) < resolution_group_pixels[3]
    assert resolution_group_pixels[3] < quality_group_pixels[1]


def test_each_page_uses_the_shared_top_edge_without_a_redundant_global_header():
    menu = OpenXrSettingsMenu()
    for tab in menu.tabs:
        menu.set_tab(tab)
        layout = menu.layout(show_glow=True)
        content_groups = [group for group in layout.groups if not group.fixed]
        assert content_groups
        expected_top = 120 if tab == "screen" else 32
        assert min(round(group.rect[1] * 832) for group in content_groups) == expected_top
        if tab == "screen":
            heading = next(group for group in layout.groups if group.key == "screen_page_heading")
            assert heading.fixed
            assert round(heading.rect[1] * 832) == 32


def test_screen_reset_stays_in_placement_card_below_secondary_navigation():
    menu = OpenXrSettingsMenu()
    controls = {control.key: control for control in menu.controls()}
    toolbar = [
        controls[key] for key in (
            "screen:section:layout", "screen:section:crop",
        )
    ]
    boxes = [tuple(round(value * size) for value, size in zip(
        control.rect, (1024, 832, 1024, 832)
    )) for control in toolbar]
    assert boxes == [
        (288, 56, 632, 104),
        (632, 56, 976, 104),
    ]
    assert controls["reset:screen_placement"].group == "screen_placement"
    assert controls["reset:screen_placement"].rect[1] > toolbar[1].rect[3]


@pytest.mark.parametrize("tab", OpenXrSettingsMenu.tabs)
def test_all_page_controls_follow_the_shared_8px_grid(tab):
    menu = OpenXrSettingsMenu()
    menu.set_tab(tab)
    controls = menu.controls(show_glow=True, lang="CN")
    for control in controls:
        pixels = tuple(round(value * size) for value, size in zip(
            control.rect, (1024, 832, 1024, 832)
        ))
        # The shared 8 px grid allows half-grid edges only where four equal
        # shape tiles must divide the fixed 688 px content width.
        x_pixels = (pixels[0], pixels[2])
        y_pixels = (pixels[1], pixels[3])
        allow_centered_column_rounding = control.key.startswith(("room:model:", "room:seat:"))
        assert all(
            value % 2 == 0 or allow_centered_column_rounding
            for value in x_pixels
        ), (tab, control.key, pixels)
        assert all(value % 8 == 0 for value in y_pixels), (tab, control.key, pixels)


def test_shared_page_layout_uses_groups_without_repeating_sidebar_page_titles():
    menu = OpenXrSettingsMenu()
    repeated_titles = {
        "Video appearance", "Stereo depth", "Glow effects",
        "Screen geometry", "Screen crop",
    }
    for tab in menu.tabs:
        menu.set_tab(tab)
        layout = menu.layout(show_glow=True, lang="EN")
        assert not repeated_titles.intersection(
            control.label for control in layout.controls
        )
        assert layout.groups
        for group in layout.groups:
            pixels = tuple(round(value * size) for value, size in zip(
                group.rect, (1024, 832, 1024, 832)
            ))
            x_pixels = (pixels[0], pixels[2])
            y_pixels = (pixels[1], pixels[3])
            assert all(value % 4 == 0 for value in x_pixels), (tab, group.key, pixels)
            assert all(value % 8 == 0 for value in y_pixels), (tab, group.key, pixels)


def test_layout_cache_reuses_geometry_until_a_layout_input_changes():
    menu = OpenXrSettingsMenu()
    initial = menu.layout()
    assert menu.layout() is initial

    slider_key = "screen:width"
    initial_slider = next(
        item for item in initial.controls if item.key == slider_key
    )
    minimum = menu.layout(values={slider_key: initial_slider.minimum})
    minimum_controls = {item.key: item for item in minimum.controls}
    assert minimum_controls[slider_key] is initial_slider
    assert not minimum_controls[f"step:minus:{slider_key}"].enabled

    maximum = menu.layout(values={slider_key: initial_slider.maximum})
    maximum_controls = {item.key: item for item in maximum.controls}
    assert maximum_controls[slider_key] is initial_slider
    assert maximum_controls[f"step:minus:{slider_key}"].enabled
    assert not maximum_controls[f"step:plus:{slider_key}"].enabled
    assert menu.layout(values={slider_key: initial_slider.maximum}) is maximum

    menu.set_tab("picture")
    picture = menu.layout()
    assert picture is not initial
    assert menu.layout() is picture

    menu.room_models = (("studio", "Studio"),)
    menu.set_tab("room")
    room = menu.layout()
    menu.room_models = (("studio", "Studio"), ("office", "Office"))
    assert menu.layout() is not room


def test_content_groups_share_equal_left_and_right_edges():
    assert _CONTENT_PANEL_LEFT - _SIDEBAR_PANEL_RIGHT == _MENU_GROUP_GAP
    assert _CONTENT_LEFT - _CONTENT_PANEL_LEFT == _MENU_GROUP_PADDING
    assert _CONTENT_PANEL_RIGHT - _CONTENT_RIGHT == _MENU_GROUP_PADDING

    menu = OpenXrSettingsMenu()
    for tab in menu.tabs:
        menu.set_tab(tab)
        layout = menu.layout(show_glow=True, lang="EN")
        if tab == "picture":
            picture_groups = {group.key: group for group in layout.groups}
            render_edges = tuple(
                round(value * 1024)
                for value in picture_groups["render_quality"].rect
            )[::2]
            color_edges = tuple(
                round(value * 1024)
                for value in picture_groups["color_adjustment"].rect
            )[::2]
            gap = round(
                (picture_groups["color_adjustment"].rect[0]
                 - picture_groups["render_quality"].rect[2]) * 1024
            )
            assert render_edges == (272, 624)
            assert color_edges == (640, 992)
            assert gap == 16
            continue
        assert layout.groups
        for group in layout.groups:
            left, _top, right, _bottom = tuple(round(value * 1024) for value in group.rect)
            assert left == 272, (tab, group.key, left)
            assert right == 992, (tab, group.key, right)


@pytest.mark.parametrize("model_count", (0, 1, 5, 15, 20))
def test_room_model_grid_wraps_to_three_columns_and_computes_scroll_extent(model_count):
    menu = OpenXrSettingsMenu()
    menu.room_models = tuple(
        (f"room_{index}", f"Environment {index}") for index in range(model_count)
    )
    menu.set_tab("room")
    layout = menu.layout()
    models = [
        item for item in layout.controls if item.key.startswith("room:model:")
    ]

    assert len(models) == model_count
    assert ("room_models" in {group.key for group in layout.groups}) == (model_count > 0)
    columns = min(3, model_count) if model_count else 0
    max_column_width = {1: 688, 2: 336, 3: 218}.get(columns, 0)
    assert all(
        (item.rect[2] - item.rect[0]) * 1024 <= max_column_width
        for item in models
    )
    if model_count <= 1:
        assert layout.scroll_max == 0
    elif model_count <= 5:
        assert layout.scroll_max > 0
        assert layout.scroll_max <= 40
    else:
        assert layout.scroll_max > 0

    pixel_boxes = [
        tuple(round(value * size) for value, size in zip(
            item.rect, (1024, 832, 1024, 832)
        ))
        for item in models
    ]
    for index, first in enumerate(pixel_boxes):
        for second in pixel_boxes[index + 1:]:
            assert (
                first[2] <= second[0] or second[2] <= first[0]
                or first[3] <= second[1] or second[3] <= first[1]
            )


def test_room_scroll_tracks_the_content_and_clamps_at_both_ends():
    menu = OpenXrSettingsMenu()
    menu.room_models = tuple(
        (f"room_{index}", f"Environment {index}") for index in range(20)
    )
    menu.set_tab("room")
    before = menu.layout()
    before_y = next(
        item.rect[1] for item in before.controls if item.key == "room:seat:front"
    )

    assert menu.scroll_by_wheel_axis(-1.0, 0.05, 0.15)
    after = menu.layout()
    after_y = next(
        item.rect[1] for item in after.controls if item.key == "room:seat:front"
    )
    assert after_y < before_y
    for _ in range(100):
        menu.scroll_by_wheel_axis(-1.0, 0.1, 0.15)
    assert menu.scroll_offset == pytest.approx(menu.layout().scroll_max)
    assert menu.scroll_viewport_contains((0.5, 0.9))
    assert not menu.scroll_viewport_contains((0.1, 0.5))
    for _ in range(100):
        menu.scroll_by_wheel_axis(1.0, 0.1, 0.15)
    assert menu.scroll_offset == pytest.approx(0.0)


def test_scrolled_room_reset_buttons_remain_inside_the_viewport_and_hit_test():
    menu = OpenXrSettingsMenu()
    menu.room_models = tuple(
        (f"room_{index}", f"Environment {index}") for index in range(20)
    )
    menu.set_tab("room")
    menu.scroll_offset = menu.layout().scroll_max
    controls = {control.key: control for control in menu.controls()}

    for key in ("reset:room_seats", "reset:room_scene"):
        control = controls[key]
        center = (
            (control.rect[0] + control.rect[2]) * 0.5,
            (control.rect[1] + control.rect[3]) * 0.5,
        )
        assert menu.scroll_viewport_contains(center)
        assert menu.hit_test(center).key == key


@pytest.mark.parametrize(
    ("tab", "section", "reset_keys"),
    (
        ("picture", None, ("reset:render_quality", "reset:color_adjustment")),
        ("depth", None, ("reset:depth_stereo",)),
        ("glow", None, ("reset:glow_modes", "reset:glow_transparency")),
        ("room", None, ("reset:room_seats", "reset:room_scene")),
        ("screen", "layout", ("reset:screen_shape", "reset:screen_placement")),
        ("screen", "crop", ("screen:reset_crop",)),
    ),
)
def test_card_resets_stay_in_their_group_and_hit_test(tab, section, reset_keys):
    menu = OpenXrSettingsMenu()
    menu.set_tab(tab)
    if section is not None:
        menu.set_screen_section(section)
    controls = {control.key: control for control in menu.controls(show_glow=True)}
    groups = {group.key: group for group in menu.layout(show_glow=True).groups}

    for key in reset_keys:
        control = controls[key]
        group = groups[control.group]
        width, height = SETTINGS_MENU_TEXTURE_SIZE
        control_box = tuple(round(value * size) for value, size in zip(
            control.rect, (width, height, width, height)
        ))
        group_box = tuple(round(value * size) for value, size in zip(
            group.rect, (width, height, width, height)
        ))
        card_top = group_box[1]
        if group.title:
            card_top += _MENU_TITLE_HEIGHT + _MENU_TITLE_CARD_GAP

        assert control_box[0] >= group_box[0] + 16
        assert control_box[2] <= group_box[2] - 16
        assert control_box[1] >= card_top + 16
        assert control_box[3] <= group_box[3] - 16
        center = (
            (control.rect[0] + control.rect[2]) * 0.5,
            (control.rect[1] + control.rect[3]) * 0.5,
        )
        assert group.rect[0] <= center[0] <= group.rect[2]
        assert group.rect[1] <= center[1] <= group.rect[3]
        assert menu.hit_test(center, show_glow=True).key == key


def test_group_cards_use_balanced_insets_and_consistent_vertical_gaps():
    menu = OpenXrSettingsMenu()
    cases = (
        ("screen", "screen_shape", "screen_placement", 120, 384, 400, 736),
        ("glow", "glow_modes", "glow_transparency", 32, 304, 320, 488),
    )
    for tab, first_key, second_key, first_top, first_bottom, second_top, second_bottom in cases:
        menu.set_tab(tab)
        layout = menu.layout(show_glow=True)
        groups = {group.key: group for group in layout.groups}
        first = tuple(round(value * 832) for value in groups[first_key].rect)
        second = tuple(round(value * 832) for value in groups[second_key].rect)
        assert (first[1], first[3]) == (first_top, first_bottom)
        assert (second[1], second[3]) == (second_top, second_bottom)
        assert second[1] - first[3] == 16

    menu.set_tab("depth")
    depth_groups = {group.key: group for group in menu.layout().groups}
    assert tuple(round(value * 832) for value in depth_groups["depth_stereo"].rect)[1::2] == (
        32, 328,
    )


def test_glow_mode_resets_have_consistent_spacing_from_controls_and_card_edges():
    menu = OpenXrSettingsMenu()
    menu.set_tab("glow")
    layout = menu.layout(show_glow=True)
    groups = {group.key: group for group in layout.groups}
    controls = {control.key: control for control in layout.controls}

    def pixel_rect(rect):
        return tuple(round(value * size) for value, size in zip(
            rect,
            (SETTINGS_MENU_TEXTURE_SIZE[0], SETTINGS_MENU_TEXTURE_SIZE[1]) * 2,
        ))

    modes = [
        pixel_rect(controls[key].rect)
        for key in ("glow:surround", "glow:glow", "glow:veil", "glow:off")
    ]
    mode_reset = pixel_rect(controls["reset:glow_modes"].rect)
    mode_card = pixel_rect(groups["glow_modes"].rect)
    transparency_steps = [
        pixel_rect(controls[key].rect)
        for key in (
            "step:minus:glow:transparency",
            "step:plus:glow:transparency",
        )
    ]
    transparency_reset = pixel_rect(controls["reset:glow_transparency"].rect)
    transparency_card = pixel_rect(groups["glow_transparency"].rect)

    assert mode_reset[1] - max(rect[3] for rect in modes) == 16
    assert mode_card[3] - mode_reset[3] == 16
    assert transparency_card[1] - mode_card[3] == 16
    assert transparency_reset[1] - max(rect[3] for rect in transparency_steps) == 16
    assert transparency_card[3] - transparency_reset[3] == 16


@pytest.mark.parametrize("tab", OpenXrSettingsMenu.tabs)
def test_controls_respect_their_cards_content_safe_area(tab):
    menu = OpenXrSettingsMenu()
    menu.set_tab(tab)
    if tab == "room":
        menu.room_models = tuple(
            (f"room_{index}", f"Environment {index}") for index in range(5)
        )
    if tab == "screen":
        sections = ("layout", "crop")
    else:
        sections = (None,)

    for section in sections:
        if section is not None:
            menu.set_screen_section(section)
        layout = menu.layout(show_glow=True)
        groups = {group.key: group for group in layout.groups}
        for control in layout.controls:
            group = groups.get(control.group)
            if group is None or control.fixed:
                continue
            width, height = SETTINGS_MENU_TEXTURE_SIZE
            control_box = tuple(round(value * size) for value, size in zip(
                control.rect, (width, height, width, height)
            ))
            group_box = tuple(round(value * size) for value, size in zip(
                group.rect, (width, height, width, height)
            ))
            card_top = group_box[1]
            if group.title:
                card_top += _MENU_TITLE_HEIGHT + _MENU_TITLE_CARD_GAP

            assert control_box[0] >= group_box[0] + 16, control.key
            assert control_box[2] <= group_box[2] - 16, control.key
            assert control_box[1] >= card_top + 16, control.key
            assert control_box[3] <= group_box[3] - 16, control.key


def test_room_groups_keep_symmetric_grid_spacing_and_content_padding():
    menu = OpenXrSettingsMenu()
    menu.room_models = tuple((f"room_{index}", f"Room {index}") for index in range(5))
    menu.set_tab("room")
    layout = menu.layout()
    groups = {group.key: group for group in layout.groups}
    controls = {control.key: control for control in layout.controls}
    models = [control for control in layout.controls if control.key.startswith("room:model:")]
    boxes = [tuple(round(value * size) for value, size in zip(
        control.rect, (1024, 832, 1024, 832)
    )) for control in models]

    assert [box[2] - box[0] for box in boxes if box[1] == boxes[0][1]] == [218, 218, 218]
    assert [boxes[index + 1][0] - boxes[index][2] for index in range(2)] == [16, 16]
    assert groups["room_seats"].rect[1] * 832 - groups["room_models"].rect[3] * 832 == pytest.approx(16)
    seat = controls["room:seat:front"]
    seat_group = groups["room_seats"]
    assert seat.rect[1] * 832 - (seat_group.rect[1] * 832 + 40) == pytest.approx(16)


def test_openxr_render_scale_uses_half_to_quadruple_range():
    menu = OpenXrSettingsMenu()
    menu.set_tab("picture")
    control = next(
        item for item in menu.controls() if item.key == "openxr_render_scale"
    )
    assert (control.minimum, control.maximum, control.step) == (0.5, 4.0, 0.05)
    assert control.value_from_u(control.rect[0]) == 0.5
    assert control.value_from_u(control.rect[2]) == 4.0


def test_slider_minus_and_plus_are_independent_hit_targets():
    menu = OpenXrSettingsMenu()
    menu.set_tab("picture")
    slider = next(
        item for item in menu.controls() if item.key == "color_brightness"
    )
    minus = next(
        item for item in menu.controls()
        if item.key == "step:minus:color_brightness"
    )
    plus = next(
        item for item in menu.controls()
        if item.key == "step:plus:color_brightness"
    )
    for expected, control in ((minus.key, minus), (plus.key, plus)):
        x0, y0, x1, y1 = control.rect
        hit = menu.hit_test(((x0 + x1) * 0.5, (y0 + y1) * 0.5))
        assert hit is not None and hit.key == expected
        assert (x1 - x0) * SETTINGS_MENU_TEXTURE_SIZE[0] >= 48
        assert (y1 - y0) * SETTINGS_MENU_TEXTURE_SIZE[1] >= 48
    assert minus.rect[2] < slider.rect[0]
    assert plus.rect[0] > slider.rect[2]


def test_every_slider_uses_a_48_pixel_ray_target_and_separate_icon_button_slop():
    menu = OpenXrSettingsMenu()
    for tab in menu.tabs:
        menu.set_tab(tab)
        controls = menu.controls(show_glow=True)
        for control in controls:
            if control.kind == "slider":
                assert (control.rect[3] - control.rect[1]) * 832 == pytest.approx(48)
            if control.kind == "slider_step":
                assert (control.rect[3] - control.rect[1]) * 832 == pytest.approx(64)
                assert (control.rect[2] - control.rect[0]) * 1024 >= 48


def test_slider_step_icon_buttons_disable_at_their_value_limits():
    menu = OpenXrSettingsMenu()
    slider_key = "screen:width"
    slider = next(item for item in menu.controls() if item.key == slider_key)
    minus = next(
        item for item in menu.controls()
        if item.key == f"step:minus:{slider_key}"
    )
    plus = next(
        item for item in menu.controls()
        if item.key == f"step:plus:{slider_key}"
    )

    minimum_controls = {
        item.key: item for item in menu.controls(values={slider_key: slider.minimum})
    }
    maximum_controls = {
        item.key: item for item in menu.controls(values={slider_key: slider.maximum})
    }

    assert not minimum_controls[minus.key].enabled
    assert minimum_controls[plus.key].enabled
    assert maximum_controls[minus.key].enabled
    assert not maximum_controls[plus.key].enabled
    assert menu.hit_test(
        ((minus.rect[0] + minus.rect[2]) / 2, (minus.rect[1] + minus.rect[3]) / 2),
        values={slider_key: slider.minimum},
    ) is None
