from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

SETTINGS_MENU_TEXTURE_SIZE = (1024, 832)
SETTINGS_MENU_WORLD_SIZE = (0.95, 0.77)
OPENXR_RENDER_SCALE_MIN = 0.5
OPENXR_RENDER_SCALE_MAX = 4.0
# Shared semantic tokens keep the in-headset and desktop OpenXR panels aligned.
OPENXR_MENU_COLORS = {
    "surface": "#242424",
    "surface_header": "#2C2C2C",
    "surface_container": "#303030",
    "surface_container_high": "#484848",
    "outline": "#828993",
    "track": "#848B95",
    "primary": "#165FC2",
    "primary_hover": "#1C67C3",
    "primary_state_text": "#EBEDF0",
    "primary_container": "#1766C7",
    "switch_track_active": "#16B85F",
    "primary_hover_container": "#1C67C3",
    "selection_container": "#165FC2",
    "selection_hover_container": "#1C67C3",
    "selection_text": "#EBEDF0",
    "text_primary": "#EBEDF0",
    "text_secondary": "#BEC4CC",
    "text_disabled": "#A3A8B0",
    "destructive": "#B33248",
    "destructive_hover": "#B83B50",
    "destructive_text": "#EBEDF0",
    "destructive_outline": "#C68289",
}
OPENXR_MENU_RADII = {
    "panel": 24,
    "group": 16,
    "control": 8,
}
_GRID = 8
_MENU_GROUP_PADDING = 16
_MENU_GROUP_GAP = 16
_MENU_TITLE_HEIGHT = 24
_MENU_TITLE_CARD_GAP = 16
_MENU_CONTROL_GAP = 16
_MENU_SLIDER_ROW_STEP = 80
_SIDEBAR_PANEL_LEFT = 32
_SIDEBAR_PANEL_RIGHT = 256
_SIDEBAR_CONTROL_PADDING = 16
_SIDEBAR_CONTROL_LEFT = _SIDEBAR_PANEL_LEFT + _SIDEBAR_CONTROL_PADDING
_SIDEBAR_CONTROL_WIDTH = (
    _SIDEBAR_PANEL_RIGHT - _SIDEBAR_PANEL_LEFT
    - 2 * _SIDEBAR_CONTROL_PADDING
)
_CONTENT_PANEL_LEFT = _SIDEBAR_PANEL_RIGHT + _MENU_GROUP_GAP
_CONTENT_PANEL_RIGHT = SETTINGS_MENU_TEXTURE_SIZE[0] - 32
_CONTENT_PANEL_WIDTH = _CONTENT_PANEL_RIGHT - _CONTENT_PANEL_LEFT
_CONTENT_LEFT = _CONTENT_PANEL_LEFT + _MENU_GROUP_PADDING
_CONTENT_RIGHT = _CONTENT_PANEL_RIGHT - _MENU_GROUP_PADDING
_CONTENT_WIDTH = _CONTENT_RIGHT - _CONTENT_LEFT
_CONTENT_TOP = 32
_CONTENT_BOTTOM = 768

OPENXR_MENU_GROUP_LABELS = {
    "render_quality": "Render quality",
    "color_adjustment": "Color adjustment",
    "screen_shape": "Screen shape",
    "screen_placement": "Screen placement",
    "screen_crop": "Crop settings",
    "depth_modes": "Stereo mode",
    "glow_modes": "Glow mode",
    "room_models": "Environment",
    "room_seats": "Seat position",
    "room_scene": "Scene controls",
}


def _rect_px(x0: int, y0: int, x1: int, y1: int) -> tuple[float, float, float, float]:
    width, height = SETTINGS_MENU_TEXTURE_SIZE
    return x0 / width, y0 / height, x1 / width, y1 / height


@dataclass(frozen=True, slots=True)
class MenuControl:
    key: str
    label: str
    rect: tuple[float, float, float, float]
    kind: str = "button"
    minimum: float = 0.0
    maximum: float = 1.0
    step: float = 0.05
    enabled: bool = True
    group: str = ""
    fixed: bool = False

    def contains(self, u: float, v: float) -> bool:
        x0, y0, x1, y1 = self.rect
        return x0 <= u <= x1 and y0 <= v <= y1

    def value_from_u(self, u: float) -> float:
        x0, _y0, x1, _y1 = self.rect
        fraction = max(0.0, min(1.0, (float(u) - x0) / max(x1 - x0, 1e-9)))
        raw = self.minimum + fraction * (self.maximum - self.minimum)
        steps = round((raw - self.minimum) / max(self.step, 1e-9))
        return max(self.minimum, min(self.maximum, self.minimum + steps * self.step))


@dataclass(frozen=True, slots=True)
class MenuGroup:
    key: str
    title: str
    rect: tuple[float, float, float, float]
    fixed: bool = False


@dataclass(frozen=True, slots=True)
class MenuLayout:
    controls: tuple[MenuControl, ...]
    groups: tuple[MenuGroup, ...]
    content_viewport: tuple[float, float, float, float]
    scroll_max: float


PICTURE_CONTROLS = (
    # Percentage of the OpenXR runtime's recommended eye resolution.  The
    # runtime still owns its Graphics Quality setting and max extent.
    ("openxr_render_scale", "Render Resolution", OPENXR_RENDER_SCALE_MIN, OPENXR_RENDER_SCALE_MAX, 0.05),
    ("color_brightness", "Brightness", 0.2, 2.0, 0.1),
    ("color_contrast", "Contrast", 0.5, 2.0, 0.1),
    ("color_saturation", "Saturation", 0.0, 2.0, 0.1),
    ("color_gamma", "Gamma", 0.5, 2.0, 0.1),
    ("color_temperature", "Temperature", -100.0, 100.0, 10.0),
    ("color_tint", "Tint", -100.0, 100.0, 10.0),
    ("vulkan_projection_min_lod", "Min LOD", 0.0, 2.0, 0.05),
    ("vulkan_projection_max_lod", "Max LOD", 0.0, 2.0, 0.05),
    ("vulkan_projection_mip_lod_bias", "MIP Bias", -1.5, 0.0, 0.05),
    ("vulkan_projection_rcas_sharpness", "RCAS", 0.0, 1.0, 0.05),
)

PICTURE_DEFAULTS = {
    "openxr_render_scale": 1.0,
    "color_brightness": 1.0,
    "color_contrast": 1.0,
    "color_saturation": 1.0,
    "color_gamma": 1.0,
    "color_temperature": 0.0,
    "color_tint": 0.0,
    "vulkan_projection_min_lod": 0.0,
    "vulkan_projection_max_lod": 0.35,
    "vulkan_projection_mip_lod_bias": -0.35,
    "vulkan_projection_rcas_sharpness": 0.5,
}


class OpenXrSettingsMenu:
    """Renderer-independent sidebar, hit-test, and trigger state for the XR menu."""

    tabs = ("picture", "depth", "glow", "room", "screen")

    def __init__(self) -> None:
        self.visible = False
        self.tab = "screen"
        # The in-headset texture has room for either the geometry controls or
        # the crop controls, but not both without making laser targets too
        # small.  The desktop Flet mirror reads this same state.
        self.screen_section = "layout"
        self.hover_key: str | None = None
        self.active_hand: int | None = None
        self.active_key: str | None = None
        self.dirty = True
        self.revision = 0
        self._trigger_down = [False, False]
        self._outside_down = [False, False]
        self.stopping = False
        self.room_models: tuple[tuple[str, str], ...] = ()
        self.room_tab_visible = True
        self.scroll_offset = 0.0
        self._layout_cache_key: tuple[object, ...] | None = None
        self._layout_cache_base: MenuLayout | None = None
        self._layout_cache_view_key: tuple[object, ...] | None = None
        self._layout_cache_value: MenuLayout | None = None

    def open(self) -> None:
        self.visible = True
        self.active_hand = None
        self.active_key = None
        self.mark_dirty()

    def close(self) -> None:
        self.visible = False
        self.hover_key = None
        self.active_hand = None
        self.active_key = None
        self.mark_dirty()

    def set_stopping(self, stopping: bool = True) -> bool:
        stopping = bool(stopping)
        if self.stopping == stopping:
            return False
        self.stopping = stopping
        self.mark_dirty()
        return True

    def mark_dirty(self) -> None:
        self.dirty = True
        self.revision += 1

    def set_tab(self, tab: str) -> bool:
        if tab not in self.tabs or tab == self.tab:
            return False
        self.tab = tab
        self.hover_key = None
        self.scroll_offset = 0.0
        self.mark_dirty()
        return True

    def set_screen_section(self, section: str) -> bool:
        if section not in {"layout", "crop"} or section == self.screen_section:
            return False
        self.screen_section = section
        self.hover_key = None
        self.scroll_offset = 0.0
        self.mark_dirty()
        return True

    def layout(
        self, *, allow_curve: bool = True, show_glow: bool = False,
        lang: str = "EN", values: Mapping[str, object] | None = None,
    ) -> MenuLayout:
        """Build one grid-aligned layout shared by rendering, ray hits and Flet."""
        cache_key = (
            self.tab, self.screen_section, bool(allow_curve), bool(show_glow),
            str(lang), self.room_models, self.stopping,
        )
        if cache_key == self._layout_cache_key and self._layout_cache_base is not None:
            return self._materialize_layout(self._layout_cache_base, values)
        controls: list[MenuControl] = []
        groups: list[MenuGroup] = []

        def rect(x: int, y: int, width: int, height: int) -> tuple[float, float, float, float]:
            return _rect_px(x, y, x + width, y + height)

        def add(
            key: str,
            label: str,
            box: tuple[int, int, int, int],
            *,
            kind: str = "button",
            minimum: float = 0.0,
            maximum: float = 1.0,
            step: float = 0.05,
            enabled: bool = True,
            group: str = "",
            fixed: bool = False,
        ) -> None:
            x, y, width, height = box
            controls.append(MenuControl(
                key, label, rect(x, y, width, height), kind, minimum, maximum,
                step, enabled, group, fixed,
            ))

        def add_group(
            key: str,
            x: int,
            y: int,
            width: int,
            height: int,
            *,
            title: str | None = None,
            fixed: bool = False,
        ) -> None:
            groups.append(MenuGroup(
                key,
                title if title is not None else OPENXR_MENU_GROUP_LABELS.get(key, ""),
                rect(x, y, width, height),
                fixed,
            ))

        def add_slider(
            key: str,
            label: str,
            x: int,
            row_y: int,
            row_width: int,
            minimum: float,
            maximum: float,
            step: float,
            group: str,
        ) -> None:
            # +/- retain 64 px ray targets. The slider and both targets share a
            # center line, with 8 px between neighboring hit regions.
            center_y = row_y + 48
            slider_left = x + 72
            slider_right = x + row_width - 72
            add(
                key, label,
                (slider_left, center_y - 24, slider_right - slider_left, 48),
                kind="slider", minimum=minimum, maximum=maximum, step=step,
                group=group,
            )
            add(
                f"step:minus:{key}", "-",
                (x, center_y - 32, 64, 64),
                kind="slider_step", minimum=minimum, maximum=maximum,
                step=step, group=group,
            )
            add(
                f"step:plus:{key}", "+",
                (x + row_width - 64, center_y - 32, 64, 64),
                kind="slider_step", minimum=minimum, maximum=maximum,
                step=step, group=group,
            )

        def add_button(
            key: str,
            label: str,
            box: tuple[int, int, int, int],
            *,
            group: str,
            kind: str = "button",
            enabled: bool = True,
            fixed: bool = False,
        ) -> None:
            add(key, label, box, kind=kind, group=group, enabled=enabled, fixed=fixed)

        # Keep fixed navigation anchored to the top of the panel. Group labels
        # carry page context, so the redundant global title bar is omitted.
        visible_tabs = ["screen", "depth"]
        if show_glow:
            visible_tabs.append("glow")
        visible_tabs.extend(("room", "picture"))
        for index, tab in enumerate(visible_tabs):
            y0 = 64 + index * 80
            add_button(
                f"tab:{tab}", tab.title(),
                (_SIDEBAR_CONTROL_LEFT, y0, _SIDEBAR_CONTROL_WIDTH, 64),
                group="navigation", fixed=True,
            )
        add_button(
            "runtime:stop", "Stop",
            (_SIDEBAR_CONTROL_LEFT, 704, _SIDEBAR_CONTROL_WIDTH, 64),
            group="navigation_actions", enabled=not self.stopping, fixed=True,
        )
        # Screen Layout/Crop is true secondary navigation and stays pinned
        # while its longer settings page scrolls.
        if self.tab == "screen":
            add_group(
                "screen_page_heading", _CONTENT_PANEL_LEFT, 32,
                _CONTENT_PANEL_WIDTH, 48,
                title="Screen geometry", fixed=True,
            )
            add_group(
                "screen_navigation", _CONTENT_PANEL_LEFT, 88,
                _CONTENT_PANEL_WIDTH, 48, title="", fixed=True,
            )
            navigation_width = (_CONTENT_WIDTH - _MENU_CONTROL_GAP) // 2
            add_button(
                "screen:section:layout", "Layout",
                (_CONTENT_LEFT, 88, navigation_width, 48),
                group="screen_navigation", fixed=True,
            )
            add_button(
                "screen:section:crop", "Crop",
                (
                    _CONTENT_LEFT + navigation_width + _MENU_CONTROL_GAP,
                    88, navigation_width, 48,
                ),
                group="screen_navigation", fixed=True,
            )

        if self.tab == "picture":
            render_keys = {
                "openxr_render_scale", "vulkan_projection_min_lod",
                "vulkan_projection_max_lod", "vulkan_projection_mip_lod_bias",
                "vulkan_projection_rcas_sharpness",
            }
            color_controls = tuple(item for item in PICTURE_CONTROLS if item[0] not in render_keys)
            render_controls = tuple(item for item in PICTURE_CONTROLS if item[0] in render_keys)
            render_height = 536
            color_height = 544
            picture_gap = _MENU_CONTROL_GAP
            picture_column_width = (_CONTENT_PANEL_WIDTH - picture_gap) // 2
            render_group_x = _CONTENT_PANEL_LEFT
            color_group_x = render_group_x + picture_column_width + picture_gap
            render_control_x = render_group_x + _MENU_GROUP_PADDING
            color_control_x = color_group_x + _MENU_GROUP_PADDING
            picture_control_width = picture_column_width - 2 * _MENU_GROUP_PADDING
            add_group(
                "render_quality", render_group_x, 32,
                picture_column_width, render_height,
            )
            add_group(
                "color_adjustment", color_group_x, 32,
                picture_column_width, color_height,
            )
            add_button(
                "openxr:render_auto", "Headset optimized",
                (render_control_x, 88, picture_control_width, 48),
                group="render_quality",
            )
            for index, (key, label, minimum, maximum, step) in enumerate(render_controls):
                add_slider(
                    key, label, render_control_x,
                    152 + index * _MENU_SLIDER_ROW_STEP, picture_control_width,
                    minimum, maximum, step, "render_quality",
                )
            for index, (key, label, minimum, maximum, step) in enumerate(color_controls):
                add_slider(
                    key, label, color_control_x,
                    80 + index * _MENU_SLIDER_ROW_STEP, picture_control_width,
                    minimum, maximum, step, "color_adjustment",
                )
        elif self.tab == "depth":
            add_group(
                "depth_strength", _CONTENT_PANEL_LEFT, 32,
                _CONTENT_PANEL_WIDTH, 112, title="",
            )
            add_slider(
                "depth_strength", "Depth strength", _CONTENT_LEFT, 40, _CONTENT_WIDTH,
                0.0, 1.0, 0.05, "depth_strength",
            )
            add_group(
                "depth_modes", _CONTENT_PANEL_LEFT, 160,
                _CONTENT_PANEL_WIDTH, 136,
            )
            mode_width = (_CONTENT_WIDTH - _MENU_CONTROL_GAP) // 2
            add_button(
                "depth:toggle_stereo", "2D / 3D",
                (_CONTENT_LEFT, 216, mode_width, 64),
                group="depth_modes", kind="toggle",
            )
            add_button(
                "depth:toggle_cross_eyed", "Cross eyed",
                (
                    _CONTENT_LEFT + mode_width + _MENU_CONTROL_GAP,
                    216, mode_width, 64,
                ), group="depth_modes", kind="toggle",
            )
        elif self.tab == "glow" and show_glow:
            add_group(
                "glow_modes", _CONTENT_PANEL_LEFT, 32,
                _CONTENT_PANEL_WIDTH, 216,
            )
            glow_column_width = (_CONTENT_WIDTH - _MENU_CONTROL_GAP) // 2
            for key, label, box in (
                (
                    "glow:surround", "Surround Glow",
                    (_CONTENT_LEFT, 88, glow_column_width, 64),
                ),
                (
                    "glow:glow", "Glow",
                    (
                        _CONTENT_LEFT + glow_column_width + _MENU_CONTROL_GAP,
                        88, glow_column_width, 64,
                    ),
                ),
                (
                    "glow:veil", "Veil",
                    (_CONTENT_LEFT, 168, glow_column_width, 64),
                ),
                (
                    "glow:off", "OFF",
                    (
                        _CONTENT_LEFT + glow_column_width + _MENU_CONTROL_GAP,
                        168, glow_column_width, 64,
                    ),
                ),
            ):
                add_button(key, label, box, group="glow_modes")
            add_group(
                "glow_transparency", _CONTENT_PANEL_LEFT, 264,
                _CONTENT_PANEL_WIDTH, 112, title="",
            )
            add_slider(
                "glow:transparency", "Glow transparency",
                _CONTENT_LEFT, 272, _CONTENT_WIDTH,
                0.0, 1.0, 0.05, "glow_transparency",
            )
        elif self.tab == "room":
            y = _CONTENT_TOP
            if self.room_models:
                columns = min(3, len(self.room_models))
                rows = (len(self.room_models) + columns - 1) // columns
                row_height = 64
                group_height = (
                    _MENU_TITLE_HEIGHT + _MENU_TITLE_CARD_GAP
                    + _MENU_GROUP_PADDING + rows * row_height
                    + max(0, rows - 1) * _MENU_CONTROL_GAP
                    + _MENU_GROUP_PADDING
                )
                add_group(
                    "room_models", _CONTENT_PANEL_LEFT, y,
                    _CONTENT_PANEL_WIDTH, group_height,
                )
                horizontal_gap = _MENU_CONTROL_GAP
                content_width = _CONTENT_WIDTH
                column_width = (
                    content_width - (columns - 1) * horizontal_gap
                ) // columns
                total_width = columns * column_width + (columns - 1) * horizontal_gap
                start_x = _CONTENT_LEFT + (content_width - total_width) // 2
                for index, (model_key, model_label) in enumerate(self.room_models):
                    column, row = index % columns, index // columns
                    x0 = start_x + column * (column_width + horizontal_gap)
                    y0 = (
                        y + _MENU_TITLE_HEIGHT + _MENU_TITLE_CARD_GAP
                        + _MENU_GROUP_PADDING + row * _MENU_SLIDER_ROW_STEP
                    )
                    add_button(
                        f"room:model:{model_key}", model_label,
                        (x0, y0, column_width, row_height),
                        group="room_models",
                    )
                y += group_height + _MENU_GROUP_GAP
            add_group(
                "room_seats", _CONTENT_PANEL_LEFT, y,
                _CONTENT_PANEL_WIDTH, 136,
            )
            seat_width = (_CONTENT_WIDTH - 2 * _MENU_CONTROL_GAP) // 3
            seat_gap = _MENU_CONTROL_GAP
            content_width = _CONTENT_WIDTH
            seat_start = _CONTENT_LEFT + (
                content_width - (seat_width * 3 + seat_gap * 2)
            ) // 2
            for index, (seat, label) in enumerate((
                ("front", "Front"), ("middle", "Middle"), ("back", "Back"),
            )):
                add_button(
                    f"room:seat:{seat}", label,
                    (seat_start + index * (seat_width + seat_gap), y + 56, seat_width, 64),
                    group="room_seats",
                )
            y += 136 + _MENU_GROUP_GAP
            add_group(
                "room_scene", _CONTENT_PANEL_LEFT, y,
                _CONTENT_PANEL_WIDTH, 320,
            )
            add_button(
                "room:toggle_screen_reflection", "Screen reflection light",
                (_CONTENT_LEFT, y + 56, _CONTENT_WIDTH, 64),
                group="room_scene", kind="toggle",
            )
            add_slider(
                "room:seat_height", "Seat height", _CONTENT_LEFT,
                y + 128, _CONTENT_WIDTH,
                -3.0, 3.0, 0.05, "room_scene",
            )
            add_slider(
                "room:exposure", "Scene brightness", _CONTENT_LEFT,
                y + 224, _CONTENT_WIDTH,
                -8.0, 8.0, 0.1, "room_scene",
            )
        elif self.tab == "screen" and self.screen_section == "crop":
            add_group(
                "screen_crop", _CONTENT_PANEL_LEFT, 152,
                _CONTENT_PANEL_WIDTH, 160, title="",
            )
            crop_button_width = (_CONTENT_WIDTH - _MENU_CONTROL_GAP) // 2
            crop_reset_width = 168
            crop_reset_x = _CONTENT_PANEL_LEFT + (
                _CONTENT_PANEL_WIDTH - crop_reset_width
            ) // 2
            for key, label, box, kind in (
                (
                    "screen:auto_crop", "Auto Crop",
                    (_CONTENT_LEFT, 168, crop_button_width, 64), "button",
                ),
                (
                    "screen:dynamic_crop", "Dynamic Crop",
                    (
                        _CONTENT_LEFT + crop_button_width + _MENU_CONTROL_GAP,
                        168, crop_button_width, 64,
                    ), "toggle",
                ),
                (
                    "screen:reset_crop", "Reset Crop",
                    (crop_reset_x, 248, crop_reset_width, 48), "button",
                ),
            ):
                add_button(key, label, box, group="screen_crop", kind=kind)
            add_group(
                "screen_crop_ranges", _CONTENT_PANEL_LEFT, 328,
                _CONTENT_PANEL_WIDTH, 224, title="Crop range",
            )
            add_slider(
                "screen:crop_width", "Width crop (Left / Right)",
                _CONTENT_LEFT, 376, _CONTENT_WIDTH,
                0.0, 45.0, 1.0, "screen_crop_ranges",
            )
            add_slider(
                "screen:crop_height", "Height crop (Top / Bottom)",
                _CONTENT_LEFT, 456, _CONTENT_WIDTH,
                0.0, 45.0, 1.0, "screen_crop_ranges",
            )
        elif self.tab == "screen":
            add_group(
                "screen_shape", _CONTENT_PANEL_LEFT, 152,
                _CONTENT_PANEL_WIDTH, 200, title="",
            )
            shape_items = (
                ("screen:type:flat", "Flat", True),
                ("screen:type:subtle", "Subtle", allow_curve),
                ("screen:type:medium", "Medium", allow_curve),
                ("screen:type:deep", "Deep", allow_curve),
            )
            shape_width = (_CONTENT_WIDTH - 3 * _MENU_CONTROL_GAP) // 4
            for index, (key, label, enabled) in enumerate(shape_items):
                x0 = _CONTENT_LEFT + index * (shape_width + _MENU_CONTROL_GAP)
                add_button(
                    key, label, (x0, 168, shape_width, 104),
                    group="screen_shape", enabled=enabled,
                )
            for key, label, x0 in (
                (
                    "screen:rotate:-90", "Rotate -90",
                    _CONTENT_PANEL_LEFT + (_CONTENT_PANEL_WIDTH - 336) // 2,
                ),
                (
                    "screen:rotate:+90", "Rotate +90",
                    _CONTENT_PANEL_LEFT + (_CONTENT_PANEL_WIDTH - 336) // 2 + 176,
                ),
            ):
                add_button(
                    key, label, (x0, 288, 160, 48), group="screen_shape",
                )
            add_group(
                "screen_placement", _CONTENT_PANEL_LEFT, 368,
                _CONTENT_PANEL_WIDTH, 304,
            )
            for index, (key, label, minimum, maximum, step) in enumerate((
                ("screen:width", "Screen size", 0.25, 2.0, 0.01),
                ("screen:height", "Screen height", -10.0, 10.0, 0.05),
                ("screen:distance", "Screen distance", 0.25, 20.0, 0.05),
            )):
                add_slider(
                    key, label, _CONTENT_LEFT,
                    416 + index * _MENU_SLIDER_ROW_STEP, _CONTENT_WIDTH,
                    minimum, maximum, step, "screen_placement",
                )

        viewport_y = 152 if self.tab == "screen" else _CONTENT_TOP
        viewport = rect(
            _CONTENT_PANEL_LEFT, viewport_y,
            _CONTENT_PANEL_RIGHT - _CONTENT_PANEL_LEFT,
            _CONTENT_BOTTOM - viewport_y,
        )
        content_bottom = max(
            [viewport_y, *(int(round(group.rect[3] * SETTINGS_MENU_TEXTURE_SIZE[1]))
                            for group in groups if not group.fixed)],
        )
        scroll_max = max(0.0, float(content_bottom - _CONTENT_BOTTOM))
        self._layout_cache_key = cache_key
        self._layout_cache_base = MenuLayout(
            tuple(controls), tuple(groups), viewport, scroll_max,
        )
        self._layout_cache_view_key = None
        self._layout_cache_value = None
        return self._materialize_layout(self._layout_cache_base, values)

    def _materialize_layout(
        self, base: MenuLayout, values: Mapping[str, object] | None,
    ) -> MenuLayout:
        self.scroll_offset = min(max(0.0, float(self.scroll_offset)), base.scroll_max)
        slider_keys = tuple(
            control.key for control in base.controls if control.kind == "slider"
        )
        values_key = None if values is None else tuple(
            (key, repr(values.get(key))) for key in slider_keys
        )
        view_key = (float(self.scroll_offset), values_key)
        if view_key == self._layout_cache_view_key and self._layout_cache_value is not None:
            return self._layout_cache_value

        offset = self.scroll_offset / SETTINGS_MENU_TEXTURE_SIZE[1]

        def shifted_rect(
            value: tuple[float, float, float, float], fixed: bool,
        ) -> tuple[float, float, float, float]:
            if fixed or offset <= 0.0:
                return value
            x0, y0, x1, y1 = value
            return x0, y0 - offset, x1, y1 - offset

        controls = tuple(
            item if shifted_rect(item.rect, item.fixed) is item.rect else MenuControl(
                item.key, item.label, shifted_rect(item.rect, item.fixed),
                item.kind, item.minimum, item.maximum, item.step, item.enabled,
                item.group, item.fixed,
            )
            for item in base.controls
        )
        groups = tuple(
            item if shifted_rect(item.rect, item.fixed) is item.rect else MenuGroup(
                item.key, item.title, shifted_rect(item.rect, item.fixed), item.fixed,
            )
            for item in base.groups
        )
        controls = self._apply_slider_step_states(controls, values)
        result = MenuLayout(controls, groups, base.content_viewport, base.scroll_max)
        self._layout_cache_view_key = view_key
        self._layout_cache_value = result
        return result

    def controls(
        self, *, allow_curve: bool = True, show_glow: bool = False,
        lang: str = "EN", values: Mapping[str, object] | None = None,
    ) -> tuple[MenuControl, ...]:
        return self.layout(
            allow_curve=allow_curve, show_glow=show_glow, lang=lang, values=values,
        ).controls

    def scroll_viewport_contains(self, uv: tuple[float, float] | None) -> bool:
        if uv is None:
            return False
        viewport = self.layout().content_viewport
        u, v = (float(value) for value in uv)
        return viewport[0] <= u <= viewport[2] and viewport[1] <= v <= viewport[3]

    def scroll_by_wheel_axis(
        self, axis: float, delta_seconds: float, deadzone: float,
    ) -> bool:
        amount = float(axis)
        magnitude = abs(amount)
        if magnitude <= float(deadzone):
            return False
        layout = self.layout()
        if layout.scroll_max <= 0.0:
            return False
        normalized = (magnitude - float(deadzone)) / max(1.0 - float(deadzone), 1e-6)
        speed = 2.0 + (35.0 - 2.0) * normalized ** 2.8
        wheel_pixels = amount * speed * max(0.0, min(0.1, float(delta_seconds))) * 48.0
        next_offset = min(
            layout.scroll_max,
            max(0.0, self.scroll_offset - wheel_pixels),
        )
        if abs(next_offset - self.scroll_offset) < 0.5:
            return False
        self.scroll_offset = next_offset
        self.mark_dirty()
        return True

    @staticmethod
    def _apply_slider_step_states(
        controls: tuple[MenuControl, ...],
        values: Mapping[str, object] | None,
    ) -> tuple[MenuControl, ...]:
        if values is None:
            return controls
        sliders = {
            control.key: control for control in controls
            if control.kind == "slider"
        }
        updated = []
        for control in controls:
            if control.kind != "slider_step":
                updated.append(control)
                continue
            _prefix, operation, slider_key = control.key.split(":", 2)
            slider = sliders.get(slider_key)
            if slider is None:
                updated.append(control)
                continue
            try:
                value = float(values.get(slider_key, slider.minimum))
            except (TypeError, ValueError):
                updated.append(control)
                continue
            tolerance = max(abs(slider.step) * 1e-6, 1e-9)
            at_limit = (
                value <= slider.minimum + tolerance
                if operation == "minus"
                else value >= slider.maximum - tolerance
            )
            updated.append(MenuControl(
                control.key,
                control.label,
                control.rect,
                control.kind,
                control.minimum,
                control.maximum,
                control.step,
                control.enabled and not at_limit,
                control.group,
                control.fixed,
            ))
        return tuple(updated)

    def hit_test(
        self, uv: tuple[float, float] | None, *, allow_curve: bool = True,
        show_glow: bool = False, lang: str = "EN",
        values: Mapping[str, object] | None = None,
    ) -> MenuControl | None:
        if uv is None:
            return None
        u, v = uv
        layout = self.layout(
            allow_curve=allow_curve, show_glow=show_glow, lang=lang, values=values,
        )
        for control in reversed(layout.controls):
            if (
                not control.fixed
                and not (
                    layout.content_viewport[0] <= float(u) <= layout.content_viewport[2]
                    and layout.content_viewport[1] <= float(v) <= layout.content_viewport[3]
                )
            ):
                continue
            if control.enabled and control.contains(float(u), float(v)):
                return control
        return None

    def sample_trigger(self, hand: int, value: float, *, outside_targets: bool) -> bool:
        """Return True once a short outside click should open the menu."""
        hand = int(hand)
        pressed = float(value) >= 0.7
        released = float(value) <= 0.3
        if not self._trigger_down[hand] and pressed:
            self._trigger_down[hand] = True
            self._outside_down[hand] = bool(outside_targets)
        elif self._trigger_down[hand] and released:
            should_open = self._outside_down[hand] and bool(outside_targets)
            self._trigger_down[hand] = False
            self._outside_down[hand] = False
            return should_open
        return False


def clamp_picture_values(values: dict[str, float]) -> dict[str, float]:
    result = dict(values)
    if "vulkan_projection_min_lod" in result and "vulkan_projection_max_lod" in result:
        result["vulkan_projection_min_lod"] = min(
            float(result["vulkan_projection_min_lod"]),
            float(result["vulkan_projection_max_lod"]),
        )
    return result


def control_by_key(controls: Iterable[MenuControl], key: str) -> MenuControl | None:
    return next((control for control in controls if control.key == key), None)
