from __future__ import annotations

import numpy as np
import pytest

from gui.localization import MESSAGE_CATALOGS, gettext_for, normalize_locale
from xr_viewer.overlay_textures import (
    build_controller_callout_rgba,
    build_settings_menu_rgba,
    build_screen_adjust_osd_rgba,
    build_screen_preset_osd_rgba,
    load_overlay_font,
)
from xr_viewer.settings_menu import (
    _CONTENT_LEFT,
    _CONTENT_RIGHT,
    OPENXR_MENU_COLORS,
    OpenXrSettingsMenu,
    SETTINGS_MENU_TEXTURE_SIZE,
)


def test_openxr_settings_menu_text_uses_every_gui_locale_catalog():
    keys = (
        "Picture", "Depth", "Glow", "Room", "Screen",
        "Surround Glow", "Veil", "OFF", "Glow effects",
        "Screen reflection light", "Reset picture", "Reset depth",
        "Reset placement",
        "B button", "Hold: show operation guide",
    )
    for locale, catalog in MESSAGE_CATALOGS.items():
        assert all(key in catalog for key in keys), locale
        assert all(gettext_for(locale, key) == catalog[key] for key in keys)
    assert normalize_locale("zh-CN") == "CN"
    for catalog in MESSAGE_CATALOGS.values():
        assert {"Stop", "Stopping...", "Stopped"} <= catalog.keys()


def test_openxr_uses_bundled_inter_for_latin_text():
    font = load_overlay_font(18, bold=True)

    assert font.path.endswith("InterVariable.ttf")
    assert load_overlay_font(18, bold=True) is font


def test_menu_cursor_position_is_not_rasterized_into_the_full_panel_texture():
    menu = OpenXrSettingsMenu()
    first = build_settings_menu_rgba(menu, {}, cursor_uv=(0.2, 0.3))
    second = build_settings_menu_rgba(menu, {}, cursor_uv=(0.8, 0.7))

    assert np.array_equal(first, second)


@pytest.mark.parametrize("locale", tuple(MESSAGE_CATALOGS))
@pytest.mark.parametrize("size", ((1024, 768), (2048, 1536)))
def test_controller_callout_copy_is_localized_and_fits_inside_its_box(locale, size):
    rgba = build_controller_callout_rgba(lang=locale, size=size)
    width, height = size
    scale_x, scale_y = width / 1024, height / 768
    body_top = int(260 * scale_y)
    body_bottom = int(310 * scale_y)
    box_right = int(950 * scale_x)

    assert rgba.shape == (height, width, 4)
    assert not np.any(rgba[body_top:body_bottom, box_right + 1:, 3])


def test_settings_menu_canvas_has_room_below_bottom_controls():
    menu = OpenXrSettingsMenu()
    menu.set_tab("screen")
    rgba = build_settings_menu_rgba(menu, {}, lang="CN")
    assert rgba.shape[:2] == (
        SETTINGS_MENU_TEXTURE_SIZE[1], SETTINGS_MENU_TEXTURE_SIZE[0]
    )
    controls = {control.key: control for control in menu.controls()}
    assert "section:reset_defaults" not in controls
    stop = controls["runtime:stop"]
    assert int(stop.rect[3] * rgba.shape[0]) == 768
    assert max(
        group.rect[3] * rgba.shape[0]
        for group in menu.layout().groups
        if not group.fixed
    ) <= 768


def test_settings_menu_omits_the_global_title_and_left_aligns_section_headings():
    menu = OpenXrSettingsMenu()
    rgba = build_settings_menu_rgba(menu, {})

    shell = (36, 36, 36, 250)
    assert tuple(rgba[64, 636]) == shell
    assert tuple(rgba[20, SETTINGS_MENU_TEXTURE_SIZE[0] // 2]) == (
        36, 36, 36, 250
    )
    title_pixels = rgba[32:80, 272:500, :3]
    title_mask = np.all(title_pixels == (190, 196, 204), axis=2)
    xs = np.where(title_mask.any(axis=0))[0]
    assert xs.size
    assert 15 <= xs.min() <= 19
    assert xs.max() < 190


@pytest.mark.parametrize(
    ("tab", "screen_section", "show_glow"),
    (
        ("picture", "layout", False),
        ("depth", "layout", False),
        ("glow", "layout", True),
        ("room", "layout", False),
        ("screen", "layout", False),
        ("screen", "crop", False),
    ),
)
def test_each_settings_group_heading_aligns_to_its_own_card(
    tab, screen_section, show_glow,
):
    menu = OpenXrSettingsMenu()
    menu.room_models = (("studio", "Studio"), ("office", "Office"))
    menu.set_tab(tab)
    if screen_section != "layout":
        menu.set_screen_section(screen_section)
    values = {"show_glow_tab": show_glow}
    layout = menu.layout(show_glow=show_glow, lang="EN", values=values)
    rgba = build_settings_menu_rgba(menu, values, lang="EN")
    muted = (190, 196, 204)
    shell = (36, 36, 36, 250)
    card = (48, 48, 48, 255)
    titled_groups = [group for group in layout.groups if group.title]

    assert titled_groups
    assert all(group.title != "OpenXR Settings" for group in titled_groups)
    for group in titled_groups:
        x0, y0, x1, y1 = tuple(
            round(value * size)
            for value, size in zip(group.rect, (1024, 832, 1024, 832))
        )
        title_area = rgba[y0:y0 + 36, x0:x1, :3]
        title_mask = np.all(title_area == muted, axis=2)
        ys, xs = np.where(title_mask)
        assert xs.size, (tab, group.key, group.title)
        assert 15 <= xs.min() <= 19, (tab, group.key, group.title, int(xs.min()))
        assert y0 <= y0 + int(ys.min()) < y0 + 24
        if not group.fixed:
            center_x = (x0 + x1) // 2
            assert tuple(rgba[y0 + 36, center_x]) == shell
            assert tuple(rgba[y0 + 40, center_x]) == card


def test_settings_menu_content_card_has_balanced_bottom_inset():
    menu = OpenXrSettingsMenu()
    rgba = build_settings_menu_rgba(menu, {})
    center_x = SETTINGS_MENU_TEXTURE_SIZE[0] // 2
    height = SETTINGS_MENU_TEXTURE_SIZE[1]

    assert tuple(rgba[height - 37, center_x]) == (36, 36, 36, 250)
    assert tuple(rgba[height - 25, center_x]) == (36, 36, 36, 250)


def test_settings_menu_shell_has_equal_texture_edge_margins():
    menu = OpenXrSettingsMenu()
    rgba = build_settings_menu_rgba(menu, {})
    shell = (36, 36, 36, 250)
    transparent = (0, 0, 0, 0)
    center_x = 636  # Centered in the 8 px gap between the fixed Layout/Crop tabs.
    center_y = SETTINGS_MENU_TEXTURE_SIZE[1] // 2
    assert tuple(rgba[100, center_x]) == shell
    assert tuple(rgba[110, center_x]) == shell
    assert tuple(rgba[115, center_x]) == shell
    assert tuple(rgba[15, center_x]) == transparent
    assert tuple(rgba[center_y, 20]) == shell
    assert tuple(rgba[center_y, 15]) == transparent
    assert tuple(rgba[811, center_x]) == shell
    assert tuple(rgba[816, center_x]) == transparent
    assert tuple(rgba[817, center_x]) == transparent
    assert tuple(rgba[center_y, 1003]) == shell
    assert tuple(rgba[center_y, 1008]) == transparent


@pytest.mark.parametrize("tab", OpenXrSettingsMenu.tabs)
def test_settings_menu_has_no_decorative_borders_or_dividers(tab) -> None:
    menu = OpenXrSettingsMenu()
    menu.room_models = tuple(
        (f"room_{index}", f"Room {index}") for index in range(15)
    )
    menu.set_tab(tab)
    rgba = build_settings_menu_rgba(menu, {}, lang="EN")
    outline = tuple(
        int(OPENXR_MENU_COLORS["outline"][index:index + 2], 16)
        for index in (1, 3, 5)
    )
    assert not np.any(np.all(rgba[:, :, :3] == outline, axis=2))


def test_openxr_settings_menu_text_is_centered_and_active_selection_has_no_underline():
    menu = OpenXrSettingsMenu()
    rgba = build_settings_menu_rgba(menu, {}, lang="EN")

    def visible_text_center(color, box):
        x0, y0, x1, y1 = box
        region = rgba[y0:y1, x0:x1, :3]
        mask = np.all(region == color, axis=2)
        ys, xs = np.where(mask)
        assert xs.size
        return (xs.min() + xs.max()) / 2 + x0

    text_primary = (235, 237, 240)
    selection_text = (235, 237, 240)
    assert visible_text_center(selection_text, (100, 64, 232, 128)) == pytest.approx(130, abs=6)
    assert tuple(rgba[72, 56, :3]) == (22, 95, 194)

    controls = {control.key: control for control in menu.controls()}
    minus = controls["step:minus:screen:width"]
    plus = controls["step:plus:screen:width"]
    text_left = max(_CONTENT_LEFT, round(minus.rect[0] * rgba.shape[1]))
    text_right = min(_CONTENT_RIGHT, round(plus.rect[2] * rgba.shape[1]))
    text_region = rgba[408:440, text_left:text_right, :3]
    text_mask = np.any(
        np.all(text_region == text_primary, axis=2)
        | np.all(text_region == (190, 196, 204), axis=2),
        axis=0,
    )
    text_x = np.where(text_mask)[0]
    assert text_x.size
    slider_text_center = (text_x.min() + text_x.max()) / 2 + text_left
    assert slider_text_center == pytest.approx((text_left + text_right) / 2, abs=4)

    flat = next(control for control in menu.controls() if control.key == "screen:type:flat")
    center_x = round((flat.rect[0] + flat.rect[2]) * 0.5 * rgba.shape[1])
    bottom_y = round(flat.rect[3] * rgba.shape[0]) - 2
    below_y = round(flat.rect[3] * rgba.shape[0]) + 8
    assert tuple(rgba[bottom_y, center_x]) == (22, 95, 194, 255)
    assert tuple(rgba[below_y, center_x]) == (48, 48, 48, 255)
    stop = controls["runtime:stop"]
    stop_box = tuple(round(value * size) for value, size in zip(
        stop.rect, (rgba.shape[1], rgba.shape[0], rgba.shape[1], rgba.shape[0])
    ))
    stop_region = rgba[stop_box[1]:stop_box[3], stop_box[0]:stop_box[2], :3]
    assert tuple(rgba[stop_box[1] + 10, (stop_box[0] + stop_box[2]) // 2]) == (179, 50, 72, 255)
    assert np.any(np.all(stop_region == text_primary, axis=2))


def test_screen_shape_choices_have_equal_width_and_symmetric_spacing():
    menu = OpenXrSettingsMenu()
    controls = [
        item for item in menu.controls()
        if item.key.startswith("screen:type:")
    ]
    boxes = [
        tuple(round(value * size) for value, size in zip(
            control.rect, (1024, 832, 1024, 832)
        ))
        for control in controls
    ]
    widths = [right - left for left, _top, right, _bottom in boxes]
    gaps = [
        next_left - right
        for (_, _, right, _), (next_left, _, _, _) in zip(boxes, boxes[1:])
    ]

    assert widths == [160, 160, 160, 160]
    assert gaps == [16, 16, 16]
    assert boxes[0][0] - _CONTENT_LEFT == 0
    assert _CONTENT_RIGHT - boxes[-1][2] == 0


@pytest.mark.parametrize(("value", "at_left"), ((0.25, True), (2.0, False)))
def test_openxr_slider_handle_stays_inside_the_rail_at_both_limits(value, at_left):
    menu = OpenXrSettingsMenu()
    controls = {control.key: control for control in menu.controls()}
    slider = controls["screen:width"]
    rgba = build_settings_menu_rgba(menu, {"screen:width": value}, lang="EN")
    x0, y0, x1, y1 = slider.rect
    x0, x1 = round(x0 * 1024), round(x1 * 1024)
    center_y = round((y0 + y1) * 832 / 2)
    handle_x = x0 + 10 if at_left else x1 - 10

    assert tuple(rgba[center_y, handle_x, :3]) == (235, 237, 240)


def test_openxr_slider_icon_buttons_show_meta_hover_press_and_disabled_states():
    menu = OpenXrSettingsMenu()
    menu.set_tab("picture")
    controls = {control.key: control for control in menu.controls()}
    plus_key = "step:plus:color_brightness"
    plus = controls[plus_key]
    center_x = round((plus.rect[0] + plus.rect[2]) * 1024 / 2)
    center_y = round((plus.rect[1] + plus.rect[3]) * 832 / 2)

    hovered = build_settings_menu_rgba(
        menu,
        {"color_brightness": 1.1},
        hover_key=plus_key,
        lang="EN",
    )
    menu.active_key = plus_key
    pressed = build_settings_menu_rgba(
        menu,
        {"color_brightness": 1.1},
        hover_key=plus_key,
        lang="EN",
    )
    at_limit = build_settings_menu_rgba(
        menu,
        {"color_brightness": 2.0},
        lang="EN",
    )

    assert tuple(hovered[center_y - 12, center_x, :3]) == (28, 103, 195)
    assert tuple(pressed[center_y - 12, center_x, :3]) == (22, 95, 194)
    assert tuple(at_limit[center_y - 12, center_x, :3]) == (48, 48, 48)


def test_openxr_boolean_settings_render_as_borderless_meta_switches():
    menu = OpenXrSettingsMenu()
    menu.set_tab("depth")
    controls = {control.key: control for control in menu.controls()}
    toggle = controls["depth:toggle_stereo"]
    box = tuple(round(value * size) for value, size in zip(
        toggle.rect, (1024, 832, 1024, 832)
    ))
    switch_x0 = box[2] - 18 - 72
    switch_y = (box[1] + box[3]) // 2

    inactive = build_settings_menu_rgba(
        menu, {"depth_strength": 0.0}, lang="EN"
    )
    active = build_settings_menu_rgba(
        menu, {"depth_strength": 0.25}, lang="EN"
    )

    assert tuple(inactive[switch_y, switch_x0 + 40, :3]) == (132, 139, 149)
    assert tuple(active[switch_y, switch_x0 + 40, :3]) == (22, 184, 95)
    assert controls["depth:toggle_stereo"].kind == "toggle"


def test_settings_menu_palette_keeps_light_text_legible_in_control_states():
    def luminance(color):
        channels = []
        for channel in color:
            value = channel / 255
            channels.append(
                value / 12.92 if value <= 0.04045
                else ((value + 0.055) / 1.055) ** 2.4
            )
        return sum(weight * value for weight, value in zip((0.2126, 0.7152, 0.0722), channels))

    def contrast(first, second):
        bright, dark = sorted((luminance(first), luminance(second)), reverse=True)
        return (bright + 0.05) / (dark + 0.05)

    surface = (48, 48, 48)
    light_text = (235, 237, 240)
    selected = (22, 95, 194)
    primary = (22, 95, 194)
    hover = (28, 103, 195)
    green = (22, 184, 95)
    destructive = (179, 50, 72)
    destructive_hover = (184, 59, 80)
    assert contrast((235, 237, 240), surface) >= 4.5
    assert contrast((190, 196, 204), surface) >= 4.5
    assert contrast((163, 168, 176), (72, 72, 72)) >= 3.0
    assert contrast(light_text, selected) >= 4.5
    assert contrast(light_text, primary) >= 4.5
    assert contrast(light_text, hover) >= 4.5
    assert contrast(light_text, destructive) >= 4.5
    assert contrast(light_text, destructive_hover) >= 4.5
    assert contrast(green, surface) >= 3.0


def test_screen_preset_osd_matches_legacy_colors_and_centering():
    rgba = build_screen_preset_osd_rgba("1000\" IMAX")

    assert rgba.shape == (78, 768, 4)
    assert np.any(np.all(rgba[:, :, :3] == (32, 32, 36), axis=2))
    assert np.any(np.all(rgba[:, :, :3] == (150, 158, 185), axis=2))
    assert np.any(np.all(rgba[:, :, :3] == (0, 210, 230), axis=2))


def test_screen_adjust_osd_matches_legacy_centered_style():
    rgba = build_screen_adjust_osd_rgba(2.4, 3.5)

    assert rgba.shape == (78, 512, 4)
    assert rgba.dtype == np.uint8
    assert rgba.flags.c_contiguous
    assert tuple(rgba[0, 0]) == (0, 0, 0, 0)
    assert np.any(np.all(rgba[:, :, :3] == (32, 32, 36), axis=2))
    assert np.any(np.all(rgba[:, :, :3] == (150, 158, 185), axis=2))
    assert np.any(np.all(rgba[:, :, :3] == (0, 210, 230), axis=2))

    alpha = rgba[:, :, 3]
    occupied = np.where(alpha > 0)
    assert occupied[1].min() < 32
    assert occupied[1].max() > 480
