# Desktop2Stereo OpenXR viewer: shared overlay RGBA texture builders.

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

from gui.localization import gettext_for, normalize_locale
from .keyboard_layout import _KB_ROWS, _KB_TEX_H, _KB_TEX_W, _KB_UNITS_WIDE, _KeyEntry
from .laser_params import CURSOR_RING_INNER_RATIO
from .settings_menu import (
    OPENXR_MENU_COLORS,
    OPENXR_MENU_RADII,
    SETTINGS_MENU_TEXTURE_SIZE,
    _MENU_TITLE_HEIGHT,
    _MENU_TITLE_CARD_GAP,
    _CONTENT_PANEL_LEFT,
    _CONTENT_PANEL_RIGHT,
    _SIDEBAR_PANEL_LEFT,
    _SIDEBAR_PANEL_RIGHT,
)
from viewer.controller_help import get_controller_help_rows


_INTER_FONT_PATH = Path(__file__).resolve().parent / "fonts" / "InterVariable.ttf"


def build_msdf_text_osd_rgba(
    atlas,
    *,
    size=(512, 78),
    runs=(),
    background=(32, 32, 36, 210),
    radius=12,
):
    """Decode the offline MSDF atlas into one complete Quad Layer texture."""
    ow, oh = int(size[0]), int(size[1])
    if ow <= 0 or oh <= 0:
        raise ValueError("MSDF OSD dimensions must be positive")
    image = Image.new("RGBA", (ow, oh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        [0, 0, ow - 1, oh - 1], radius=int(radius), fill=tuple(background)
    )
    rgba = np.array(image, dtype=np.uint8, copy=True, order="C")
    distance_range = max(1.0, float(getattr(atlas, "distance_range", 4.0)))
    page_cache = {}

    def page_pixels(page):
        if page not in page_cache:
            page_cache[page] = np.asarray(atlas.page_rgba(page), dtype=np.uint8)
        return page_cache[page]

    def blend_glyph(instance, color):
        x0 = int(round(instance.position[0]))
        y0 = int(round(instance.position[1]))
        dst_w = max(1, int(round(instance.size[0])))
        dst_h = max(1, int(round(instance.size[1])))
        if instance.size[0] <= 0.0 or instance.size[1] <= 0.0:
            return
        page = page_pixels(int(instance.page))
        src_x0 = max(0, int(round(instance.uv_min[0] * atlas.page_width)))
        src_y0 = max(0, int(round(instance.uv_min[1] * atlas.page_height)))
        src_x1 = min(atlas.page_width, int(round(instance.uv_max[0] * atlas.page_width)))
        src_y1 = min(atlas.page_height, int(round(instance.uv_max[1] * atlas.page_height)))
        if src_x1 <= src_x0 or src_y1 <= src_y0:
            return
        crop = Image.fromarray(page[src_y0:src_y1, src_x0:src_x1, :3], mode="RGB")
        crop = crop.resize((dst_w, dst_h), Image.Resampling.BILINEAR)
        msdf = np.asarray(crop, dtype=np.float32) / 255.0
        median = np.median(msdf, axis=2)
        screen_px_range = max(1.0, min(16.0, dst_h / distance_range))
        coverage = np.clip((median - 0.5) * screen_px_range + 0.5, 0.0, 1.0)
        color_rgba = np.asarray(color, dtype=np.float32)
        if color_rgba.shape != (4,):
            raise ValueError("MSDF text color must contain four components")
        source_alpha = coverage * (color_rgba[3] / 255.0)
        dst_x0, dst_y0 = max(0, x0), max(0, y0)
        dst_x1, dst_y1 = min(ow, x0 + dst_w), min(oh, y0 + dst_h)
        if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
            return
        sx0, sy0 = dst_x0 - x0, dst_y0 - y0
        sx1, sy1 = sx0 + dst_x1 - dst_x0, sy0 + dst_y1 - dst_y0
        source_alpha = source_alpha[sy0:sy1, sx0:sx1]
        destination = rgba[dst_y0:dst_y1, dst_x0:dst_x1].astype(np.float32) / 255.0
        destination_alpha = destination[..., 3]
        output_alpha = source_alpha + destination_alpha * (1.0 - source_alpha)
        safe_alpha = np.maximum(output_alpha, 1e-6)
        source_rgb = color_rgba[:3] / 255.0
        destination[..., :3] = (
            source_rgb[None, None, :] * source_alpha[..., None]
            + destination[..., :3] * destination_alpha[..., None] * (1.0 - source_alpha[..., None])
        ) / safe_alpha[..., None]
        destination[..., 3] = output_alpha
        rgba[dst_y0:dst_y1, dst_x0:dst_x1] = np.clip(
            destination * 255.0 + 0.5, 0.0, 255.0
        ).astype(np.uint8)

    for run in runs:
        instances = atlas.layout(
            str(run.get("text", "")),
            origin=(float(run.get("x", 0.0)), float(run.get("y", 0.0))),
            scale=float(run.get("scale", 1.0)),
        )
        for instance in instances:
            blend_glyph(instance, run.get("color", (255, 255, 255, 255)))
    return np.ascontiguousarray(rgba)


@lru_cache(maxsize=64)
def _load_overlay_font_cached(size, font_type, prefer_cjk, bold):
    candidates = []
    if prefer_cjk:
        candidates.append(
            r"C:\Windows\Fonts\msyhbd.ttc" if bold
            else r"C:\Windows\Fonts\msyh.ttc"
        )
    elif _INTER_FONT_PATH.is_file():
        candidates.append(str(_INTER_FONT_PATH))
    if bold:
        candidates.extend((
            r"C:\Windows\Fonts\seguisb.ttf",
            r"C:\Windows\Fonts\segoeuib.ttf",
        ))
    candidates.extend((
        r"C:\Windows\Fonts\seguisym.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        font_type,
    ))
    for candidate in candidates:
        if not candidate:
            continue
        try:
            font = ImageFont.truetype(candidate, size)
            if Path(candidate) == _INTER_FONT_PATH:
                font.set_variation_by_name("SemiBold" if bold else "Regular")
            return font
        except Exception:
            continue
    return ImageFont.load_default()


def load_overlay_font(size, font_type=None, *, prefer_cjk=False, bold=False):
    font_key = None if font_type is None else str(font_type)
    return _load_overlay_font_cached(
        int(size), font_key, bool(prefer_cjk), bool(bold)
    )


def _fit_overlay_font(draw, text, size, max_width, *, prefer_cjk=True, bold=False, minimum=14):
    for font_size in range(int(size), int(minimum) - 1, -1):
        font = load_overlay_font(font_size, prefer_cjk=prefer_cjk, bold=bold)
        if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
            return font
    return load_overlay_font(minimum, prefer_cjk=prefer_cjk, bold=bold)


def build_settings_menu_rgba(menu, values, *, hover_key=None, cursor_uv=None, lang="EN"):
    """Rasterize the shared, grid-aligned XR settings layout."""
    width, height = SETTINGS_MENU_TEXTURE_SIZE
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    def rgba(token, alpha=255):
        value = OPENXR_MENU_COLORS[token].lstrip("#")
        return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4)) + (alpha,)

    colors = {
        "shell": rgba("surface", 250),
        "header": rgba("surface_header"),
        "card": rgba("surface_container"),
        "card_alt": rgba("surface_container_high"),
        "track": rgba("track"),
        "blue": rgba("primary"),
        "blue_hover": rgba("primary_hover"),
        "blue_state_text": rgba("primary_state_text"),
        "blue_container": rgba("primary_container"),
        "blue_hover_container": rgba("primary_hover_container"),
        "selection_container": rgba("selection_container"),
        "selection_text": rgba("selection_text"),
        "switch_track_active": rgba("switch_track_active"),
        "text": rgba("text_primary"),
        "muted": rgba("text_secondary"),
        "disabled": rgba("text_disabled"),
        "stop": rgba("destructive"),
        "stop_hover": rgba("destructive_hover"),
        "stop_text": rgba("destructive_text"),
    }
    locale = normalize_locale(lang)
    prefer_cjk = locale == "CN"
    layout = menu.layout(
        allow_curve=bool(values.get("screen_allow_curve", True)),
        show_glow=bool(values.get("show_glow_tab", False)),
        lang=locale,
        values=values,
    )
    controls = layout.controls
    controls_by_key = {control.key: control for control in controls}
    translate = lambda message: gettext_for(locale, message)
    outer_inset = 16
    draw.rounded_rectangle(
        # Use an explicit inclusive endpoint so the transparent margin is
        # exactly the same on all four sides of the texture.
        (
            outer_inset,
            outer_inset,
            width - 1 - outer_inset,
            height - 1 - outer_inset,
        ),
        radius=OPENXR_MENU_RADII["panel"],
        fill=colors["shell"],
    )
    draw.rounded_rectangle(
        (
            _SIDEBAR_PANEL_LEFT, 32,
            _SIDEBAR_PANEL_RIGHT, height - 32,
        ),
        radius=OPENXR_MENU_RADII["group"],
        fill=colors["card"],
    )
    draw.rounded_rectangle(
        (
            _CONTENT_PANEL_LEFT, 32,
            _CONTENT_PANEL_RIGHT, height - 32,
        ),
        radius=OPENXR_MENU_RADII["group"],
        fill=colors["shell"],
    )
    content_viewport_px = tuple(
        int(round(value * size))
        for value, size in zip(
            layout.content_viewport, (width, height, width, height)
        )
    )
    content_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    content_draw = ImageDraw.Draw(content_layer)

    def pixel_rect(control):
        return tuple(
            int(round(value * size))
            for value, size in zip(control.rect, (width, height, width, height))
        )

    def wrap_text(target, text, font, max_width):
        """Wrap translated labels at word boundaries, then at glyph boundaries."""
        lines = []
        for paragraph in str(text).splitlines() or [""]:
            current = ""
            for character in paragraph:
                candidate = current + character
                if not current or target.textlength(candidate, font=font) <= max_width:
                    current = candidate
                    continue
                break_at = current.rfind(" ")
                if break_at > 0:
                    lines.append(current[:break_at].rstrip())
                    current = current[break_at + 1:].lstrip() + character
                else:
                    lines.append(current.rstrip())
                    current = character.lstrip()
            lines.append(current.rstrip())
        return "\n".join(lines)

    def centered_text(target, text, box, font_size, *, bold=False, minimum=18, color=None):
        x0, y0, x1, y1 = box
        font = _fit_overlay_font(
            target, text, font_size, max(1, x1 - x0 - 16),
            prefer_cjk=prefer_cjk, bold=bold, minimum=minimum,
        )
        wrapped = wrap_text(target, text, font, max(1, x1 - x0 - 16))
        bbox = target.multiline_textbbox((0, 0), wrapped, font=font, spacing=2, align="center")
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        target.multiline_text(
            (
                (x0 + x1 - text_width) / 2 - bbox[0],
                (y0 + y1 - text_height) / 2 - bbox[1],
            ),
            wrapped, font=font, fill=color or colors["text"], spacing=2,
            align="center",
        )

    def selected_for(control):
        key = control.key
        if key.startswith("tab:"):
            return key[4:] == menu.tab
        if key.startswith("screen:type:"):
            target_angles = {
                "screen:type:flat": 0.0,
                "screen:type:subtle": np.deg2rad(20.0),
                "screen:type:medium": np.deg2rad(30.0),
                "screen:type:deep": 0.72,
            }
            return abs(
                float(values.get("screen:curve_half_angle", 0.0))
                - float(target_angles[key])
            ) < 1e-3
        if key == "depth:toggle_stereo":
            return float(values.get("depth_strength", 0.0)) > 0.0
        if key == "depth:toggle_cross_eyed":
            return bool(values.get("cross_eyed", False))
        if key.startswith("glow:"):
            return key == f"glow:{values.get('glow:mode', 'off')}"
        if key.startswith("room:model:"):
            return key == f"room:model:{values.get('room:model', 'Default')}"
        if key.startswith("room:seat:"):
            seats = ("front", "middle", "back")
            index = int(values.get("room:seat_index", 0)) % len(seats)
            return key == f"room:seat:{seats[index]}"
        if key == "room:toggle_screen_reflection":
            return bool(values.get("room:screen_reflection_enabled", True))
        if key == "openxr:render_auto":
            return bool(values.get("openxr_render_auto", False))
        if key.startswith("screen:section:"):
            return key.rsplit(":", 1)[1] == getattr(menu, "screen_section", "layout")
        if key == "screen:dynamic_crop":
            return bool(values.get("screen:dynamic_crop", False))
        return False

    def draw_group(target, group):
        if not group.title:
            return
        x0, y0, x1, _y1 = pixel_rect(group)
        group_title = translate(group.title)
        font = _fit_overlay_font(
            target, group_title, 20, x1 - x0 - 32,
            prefer_cjk=prefer_cjk, bold=True, minimum=18,
        )
        bounds = target.textbbox((0, 0), group_title, font=font)
        title_height = bounds[3] - bounds[1]
        target.text(
            (
                x0 + 16 - bounds[0],
                y0 + (_MENU_TITLE_HEIGHT - title_height) / 2 - bounds[1],
            ),
            group_title,
            font=font,
            fill=colors["muted"],
        )

    def draw_navigation_icon(target, tab, left, top, color):
        right, bottom = left + 24, top + 24
        if tab == "screen":
            target.rounded_rectangle(
                (left + 2, top + 3, right - 2, bottom - 6),
                radius=3, outline=color, width=2,
            )
            target.line(
                (left + 7, bottom - 2, right - 7, bottom - 2),
                fill=color, width=2,
            )
            target.line(
                ((left + right) // 2, bottom - 6, (left + right) // 2, bottom - 2),
                fill=color, width=2,
            )
        elif tab == "depth":
            for offset in (0, 6, 12):
                target.line(
                    (
                        left + 3, top + 5 + offset,
                        left + 12, top + 1 + offset,
                        left + 21, top + 5 + offset,
                    ),
                    fill=color, width=2, joint="curve",
                )
                target.line(
                    (
                        left + 3, top + 5 + offset,
                        left + 3, top + 8 + offset,
                        left + 12, top + 12 + offset,
                        left + 21, top + 8 + offset,
                        left + 21, top + 5 + offset,
                    ),
                    fill=color, width=2, joint="curve",
                )
        elif tab == "glow":
            center = ((left + right) // 2, (top + bottom) // 2)
            target.ellipse(
                (center[0] - 4, center[1] - 4, center[0] + 4, center[1] + 4),
                outline=color, width=2,
            )
            for start, end in (
                ((center[0], top + 1), (center[0], top + 5)),
                ((center[0], bottom - 5), (center[0], bottom - 1)),
                ((left + 1, center[1]), (left + 5, center[1])),
                ((right - 5, center[1]), (right - 1, center[1])),
                ((left + 4, top + 4), (left + 7, top + 7)),
                ((right - 7, bottom - 7), (right - 4, bottom - 4)),
                ((right - 4, top + 4), (right - 7, top + 7)),
                ((left + 7, bottom - 7), (left + 4, bottom - 4)),
            ):
                target.line((*start, *end), fill=color, width=2)
        elif tab == "room":
            target.line(
                (
                    left + 2, top + 11, left + 12, top + 3,
                    right - 2, top + 11,
                ),
                fill=color, width=2, joint="curve",
            )
            target.rectangle(
                (left + 5, top + 10, right - 5, bottom - 2),
                outline=color, width=2,
            )
            target.rectangle(
                (left + 10, top + 15, left + 14, bottom - 2),
                outline=color, width=1,
            )
        else:
            target.rounded_rectangle(
                (left + 2, top + 3, right - 2, bottom - 3),
                radius=3, outline=color, width=2,
            )
            target.ellipse(
                (left + 6, top + 7, left + 10, top + 11),
                outline=color, width=2,
            )
            target.line(
                (left + 4, bottom - 6, left + 9, top + 14,
                 left + 13, bottom - 8, left + 17, top + 15,
                 right - 4, top + 12),
                fill=color, width=2, joint="curve",
            )

    def draw_slider(target, control):
        box = pixel_rect(control)
        x0, y0, x1, y1 = box
        center_y = (y0 + y1) // 2
        current = float(values.get(control.key, control.minimum))
        fraction = max(
            0.0,
            min(1.0, (current - control.minimum) / max(control.maximum - control.minimum, 1e-9)),
        )
        track_left = x0 + 10
        track_right = x1 - 10
        target.rounded_rectangle(
            (track_left, center_y - 4, track_right, center_y + 4),
            radius=4, fill=colors["track"],
        )
        fill_x = int(track_left + (track_right - track_left) * fraction)
        if fill_x > track_left:
            target.rounded_rectangle(
                (track_left, center_y - 4, fill_x, center_y + 4),
                radius=4, fill=colors["blue_hover"] if control.key == hover_key else colors["blue"],
            )
        target.ellipse(
            (fill_x - 10, center_y - 10, fill_x + 10, center_y + 10),
            fill=colors["text"],
        )
        minus = controls_by_key[f"step:minus:{control.key}"]
        plus = controls_by_key[f"step:plus:{control.key}"]
        for step_control, is_plus in ((minus, False), (plus, True)):
            sx0, sy0, sx1, sy1 = pixel_rect(step_control)
            step_x = (sx0 + sx1) // 2
            step_hovered = step_control.key == hover_key
            step_pressed = step_control.key == getattr(menu, "active_key", None)
            if not step_control.enabled:
                button_fill, glyph_color = colors["card"], colors["disabled"]
            elif step_pressed:
                button_fill, glyph_color = colors["blue"], colors["blue_state_text"]
            elif step_hovered:
                button_fill, glyph_color = colors["blue_hover"], colors["blue_state_text"]
            else:
                button_fill, glyph_color = colors["card_alt"], colors["text"]
            target.ellipse(
                (step_x - 16, center_y - 16, step_x + 16, center_y + 16),
                fill=button_fill,
            )
            target.line(
                (step_x - 5, center_y, step_x + 5, center_y),
                fill=glyph_color, width=3,
            )
            if is_plus:
                target.line(
                    (step_x, center_y - 5, step_x, center_y + 5),
                    fill=glyph_color, width=3,
                )
        label = translate(control.label)
        value_text = (
            f"{round(current * 100):.0f}%"
            if control.key == "openxr_render_scale"
            else f"{current:.0f}% each"
            if control.key in {"screen:crop_width", "screen:crop_height"}
            else f"{current:.2f}"
        )
        row_left = int(minus.rect[0] * width)
        row_right = int(plus.rect[2] * width)
        value_column = min(80, max(64, row_right - row_left - 128))
        value_left = row_right - value_column
        label_font = _fit_overlay_font(
            target, label, 18, max(1, value_left - row_left - 8),
            prefer_cjk=prefer_cjk, minimum=18,
        )
        value_font = _fit_overlay_font(
            target, value_text, 16, max(1, row_right - value_left),
            prefer_cjk=prefer_cjk, minimum=14,
        )
        label_box = target.textbbox((0, 0), label, font=label_font)
        value_box = target.textbbox((0, 0), value_text, font=value_font)
        label_top = center_y - 43
        target.text(
            (row_left - label_box[0], label_top - label_box[1]),
            label, font=label_font, fill=colors["text"],
        )
        value_width = value_box[2] - value_box[0]
        target.text(
            (row_right - value_width - value_box[0], label_top - value_box[1]),
            value_text, font=value_font,
            fill=colors["text"] if control.key == hover_key else colors["muted"],
        )

    def draw_control(target, control):
        if control.kind == "slider_step":
            return
        box = pixel_rect(control)
        x0, y0, x1, y1 = box
        hovered = control.key == hover_key
        pressed = control.key == getattr(menu, "active_key", None)
        active = selected_for(control)
        label = translate(control.label)
        if control.kind == "slider":
            draw_slider(target, control)
            return

        if control.key == "runtime:stop":
            fill = (
                colors["card_alt"] if not control.enabled
                else colors["stop_hover"] if hovered or pressed else colors["stop"]
            )
        elif not control.enabled:
            fill = colors["card"]
        elif pressed:
            fill = colors["blue"]
        elif hovered:
            fill = colors["blue_hover"]
        elif active:
            fill = colors["selection_container"]
        else:
            fill = colors["card_alt"]
        if control.kind == "toggle":
            fill = colors["blue"] if pressed else colors["blue_hover"] if hovered else colors["card"]
        target.rounded_rectangle(
            box, radius=OPENXR_MENU_RADII["control"], fill=fill,
        )
        text_color = (
            colors["disabled"] if not control.enabled
            else colors["stop_text"] if control.key == "runtime:stop"
            else colors["selection_text"] if active or hovered or pressed
            else colors["text"]
        )
        if control.kind == "toggle":
            switch_width = min(72, max(48, (x1 - x0) // 4))
            switch_height = 32
            switch_x1 = x1 - 16
            switch_x0 = switch_x1 - switch_width
            switch_y0 = (y0 + y1 - switch_height) // 2
            switch_y1 = switch_y0 + switch_height
            target.rounded_rectangle(
                (switch_x0, switch_y0, switch_x1, switch_y1),
                radius=switch_height // 2,
                fill=colors["switch_track_active"] if active else colors["track"],
            )
            knob_x = switch_x1 - switch_height // 2 if active else switch_x0 + switch_height // 2
            target.ellipse(
                (knob_x - switch_height // 2 + 3, switch_y0 + 3,
                 knob_x + switch_height // 2 - 3, switch_y1 - 3),
                fill=colors["text"],
            )
            button_font = _fit_overlay_font(
                target, label, 18, max(1, switch_x0 - x0 - 32),
                prefer_cjk=prefer_cjk, minimum=18,
            )
            bbox = target.textbbox((0, 0), label, font=button_font)
            target.text(
                (x0 + 16 - bbox[0], (y0 + y1 - (bbox[3] - bbox[1])) / 2 - bbox[1]),
                label, font=button_font, fill=text_color,
            )
            return

        if control.key.startswith("tab:"):
            icon_color = (
                colors["selection_text"] if active or hovered or pressed
                else colors["muted"]
            )
            icon_left = x0 + 16
            icon_top = (y0 + y1 - 24) // 2
            draw_navigation_icon(
                target, control.key.split(":", 1)[1],
                icon_left, icon_top, icon_color,
            )
            font = _fit_overlay_font(
                target, label, 18, max(1, x1 - icon_left - 48),
                prefer_cjk=prefer_cjk, bold=True, minimum=18,
            )
            bbox = target.textbbox((0, 0), label, font=font)
            text_top = (y0 + y1 - (bbox[3] - bbox[1])) / 2 - bbox[1]
            target.text(
                (icon_left + 36 - bbox[0], text_top),
                label, font=font, fill=text_color,
            )
            return

        if control.key.startswith("screen:type:"):
            cx = (x0 + x1) // 2
            arc_y = y0 + 22
            half_span = min(35, max(24, (x1 - x0) // 3))
            depth = {
                "screen:type:flat": 0,
                "screen:type:subtle": 7,
                "screen:type:medium": 13,
                "screen:type:deep": 20,
            }[control.key]
            arc_color = (
                colors["disabled"] if not control.enabled
                else colors["blue_state_text"] if active
                else colors["muted"]
            )
            points = []
            for index in range(25):
                t = index / 24.0
                x = cx - half_span + 2.0 * half_span * t
                y = arc_y + depth * (1.0 - (2.0 * t - 1.0) ** 2)
                points.append((x, y))
            target.line(points, fill=arc_color, width=4)
            text_box = (x0 + 4, y0 + 48, x1 - 4, y1 - 4)
            centered_text(target, label, text_box, 18, bold=True, color=text_color)
            return

        button_font_size = 20 if control.key in {"runtime:stop"} else 18
        centered_text(
            target, label, (x0 + 8, y0 + 4, x1 - 8, y1 - 4),
            button_font_size,
            bold=control.key.startswith(("tab:", "runtime:stop")),
            minimum=18,
            color=text_color,
        )

    for group in layout.groups:
        if group.fixed:
            if group.title:
                draw_group(draw, group)
            continue
        box = pixel_rect(group)
        if group.title:
            x0, y0, x1, y1 = box
            card_box = (
                x0, y0 + _MENU_TITLE_HEIGHT + _MENU_TITLE_CARD_GAP,
                x1, y1,
            )
        else:
            card_box = box
        content_draw.rounded_rectangle(
            card_box,
            radius=OPENXR_MENU_RADII["group"],
            fill=colors["card"],
        )
        draw_group(content_draw, group)
    for control in controls:
        target = draw if control.fixed else content_draw
        if control.fixed:
            draw_control(target, control)
        else:
            draw_control(target, control)
    clip_x0, clip_y0, clip_x1, clip_y1 = content_viewport_px
    image.alpha_composite(
        content_layer.crop((clip_x0, clip_y0, clip_x1, clip_y1)),
        dest=(clip_x0, clip_y0),
    )
    return np.ascontiguousarray(np.asarray(image, dtype=np.uint8))


def build_keyboard_rgba(show_shifted, keyboard_width, keyboard_height, font_type=None, *, hover_indices=(), held_indices=(), locked_indices=(), hover_points=()):
    """Build the validated keyboard texture content for any renderer."""
    hover_indices = set(i for i in hover_indices if i is not None)
    held_indices = set(i for i in held_indices if i is not None)
    locked_indices = set(i for i in locked_indices if i is not None)
    hover_points = tuple(p for p in hover_points if p is not None)
    tw, th = _KB_TEX_W, _KB_TEX_H
    row_h = th / len(_KB_ROWS)
    unit_w = tw / float(_KB_UNITS_WIDE)
    unit_m = float(keyboard_width) / float(_KB_UNITS_WIDE)
    pad = 3

    img = Image.new("RGBA", (tw, th), (30, 30, 35, 230))
    draw = ImageDraw.Draw(img)
    font = load_overlay_font(16, font_type)

    keys = []
    kw_half = float(keyboard_width) / 2.0
    kh_half = float(keyboard_height) / 2.0
    row_h_m = float(keyboard_height) / len(_KB_ROWS)

    for row_i, row in enumerate(_KB_ROWS):
        py0 = int(row_i * row_h)
        py1 = int((row_i + 1) * row_h)
        ly1 = kh_half - row_i * row_h_m
        ly0 = ly1 - row_h_m
        px = 0.0
        lx = -kw_half
        for label, vk_normal, shifted_label, vk_shifted, width_units in row:
            px_end = px + width_units * unit_w
            lx_end = lx + width_units * unit_m

            if vk_normal == -1:
                px = px_end
                lx = lx_end
                continue

            key_index = len(keys)
            is_held = key_index in held_indices
            is_locked = key_index in locked_indices
            is_hover = key_index in hover_indices
            if is_locked:
                key_fill = (240, 145, 35, 255)
                key_outline = (255, 220, 130, 255)
            elif is_held:
                key_fill = (92, 122, 170, 255)
                key_outline = (245, 248, 255, 255)
            elif is_hover:
                key_fill = (72, 92, 125, 255)
                key_outline = (245, 248, 255, 255)
            else:
                key_fill = (60, 62, 70, 255)
                key_outline = (130, 132, 140, 255)
            draw.rectangle(
                [px + pad, py0 + pad, px_end - pad, py1 - pad],
                fill=key_fill,
                outline=key_outline,
            )

            display_label = shifted_label if show_shifted and shifted_label is not None else label
            if font:
                draw.text(
                    ((px + px_end) / 2.0, (py0 + py1) / 2.0),
                    display_label,
                    font=font,
                    fill=(220, 220, 225, 255),
                    anchor="mm",
                )
            else:
                draw.text((int(px + pad + 2), int(py0 + pad + 2)), display_label, fill=(220, 220, 225, 255))

            keys.append(
                _KeyEntry(
                    label=label,
                    shifted_label=shifted_label,
                    vk=vk_normal,
                    shifted_vk=vk_shifted if vk_shifted is not None else vk_normal,
                    rect_uv=(px / tw, py0 / th, px_end / tw, py1 / th),
                    rect_local=(lx, ly0, lx_end, ly1),
                )
            )

            px = px_end
            lx = lx_end

    for lx, ly in hover_points:
        cx = int((float(lx) + kw_half) / max(float(keyboard_width), 1e-6) * tw)
        cy = int((kh_half - float(ly)) / max(float(keyboard_height), 1e-6) * th)
        if 0 <= cx < tw and 0 <= cy < th:
            r_outer = 11
            r_inner = int(round(r_outer * CURSOR_RING_INNER_RATIO))
            draw.ellipse([cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer], fill=(80, 180, 255, 235))
            draw.ellipse([cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner], fill=(255, 255, 255, 220))

    return np.ascontiguousarray(np.asarray(img, dtype=np.uint8)), keys


def build_cursor_rgba(size=64):
    size = max(8, int(size))
    yy, xx = np.ogrid[:size, :size]
    c = (size - 1) * 0.5
    d = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
    outer = size * 0.45
    inner = outer * CURSOR_RING_INNER_RATIO
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[d <= outer] = (80, 180, 255, 235)
    rgba[d <= inner] = (255, 255, 255, 235)
    return np.ascontiguousarray(rgba)


def build_laser_rgba(width=64, height=512):
    """Build a transparent vertical beam texture for an OpenXR quad layer."""
    width = max(4, int(width))
    height = max(16, int(height))
    x = np.abs(np.arange(width, dtype=np.float32) - (width - 1) * 0.5)
    core = np.clip(1.0 - x / max(width * 0.5, 1.0), 0.0, 1.0)
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., 0] = 80
    rgba[..., 1] = 190
    rgba[..., 2] = 255
    rgba[..., 3] = np.maximum(0.0, core[None, :] * 255.0).astype(np.uint8)
    return np.ascontiguousarray(rgba)


def build_short_osd_rgba(lines, font_type=None, *, width=768, height=96):
    img = Image.new("RGBA", (int(width), int(height)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0, 0, int(width) - 1, int(height) - 1], radius=14, fill=(32, 32, 36, 210))
    draw.text(
        (18, 16),
        "  ".join(str(line) for line in lines[:2]),
        font=load_overlay_font(24, font_type, prefer_cjk=True),
        fill=(220, 235, 255, 255),
    )
    return np.ascontiguousarray(np.asarray(img, dtype=np.uint8))


def build_screen_adjust_osd_rgba(
    screen_width,
    screen_distance,
    font_type=None,
    *,
    size=(512, 78),
    draw_text=True,
):
    """Build the legacy centered screen size/distance OSD."""
    ow, oh = int(size[0]), int(size[1])
    img = Image.new("RGBA", (ow, oh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        [0, 0, ow - 1, oh - 1],
        radius=12,
        fill=(32, 32, 36, 210),
    )
    if not draw_text:
        return np.ascontiguousarray(np.asarray(img, dtype=np.uint8))
    font = load_overlay_font(24, font_type, prefer_cjk=True)
    bold_font = font
    label_color = (150, 158, 185, 255)
    value_color = (0, 210, 230, 255)
    size_value = f"{float(screen_width):.2f} x {float(screen_width) * 9.0 / 16.0:.2f} m"
    distance_value = f"{float(screen_distance):.2f} m"
    parts = (
        ("Size", label_color, bold_font),
        (size_value, value_color, font),
        ("Dist", label_color, bold_font),
        (distance_value, value_color, font),
    )
    gap = 8
    widths = [_text_width(draw, text, part_font) for text, _color, part_font in parts]
    total_width = sum(widths) + gap * (len(parts) - 1)
    x = max(0, (ow - total_width) // 2)
    y = (oh - 32) // 2
    for (text, color, part_font), part_width in zip(parts, widths):
        draw.text((x, y), text, font=part_font, fill=color)
        x += part_width + gap
    return np.ascontiguousarray(np.asarray(img, dtype=np.uint8))


def build_screen_preset_osd_rgba(
    preset_label,
    font_type=None,
    *,
    size=(768, 78),
    draw_text=True,
):
    """Build the legacy centered screen preset OSD."""
    ow, oh = int(size[0]), int(size[1])
    img = Image.new("RGBA", (ow, oh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        [0, 0, ow - 1, oh - 1],
        radius=12,
        fill=(32, 32, 36, 210),
    )
    if not draw_text:
        return np.ascontiguousarray(np.asarray(img, dtype=np.uint8))
    label_font = load_overlay_font(24, font_type, prefer_cjk=True)
    value_font = label_font
    label_color = (150, 158, 185, 255)
    value_color = (0, 210, 230, 255)
    label = "Preset"
    gap = 8
    label_width = _text_width(draw, label, label_font)
    value_width = _text_width(draw, str(preset_label), value_font)
    x = max(0, (ow - label_width - gap - value_width) // 2)
    y = (oh - 32) // 2
    draw.text((x, y), label, font=label_font, fill=label_color)
    draw.text((x + label_width + gap, y), str(preset_label), font=value_font, fill=value_color)
    return np.ascontiguousarray(np.asarray(img, dtype=np.uint8))


def _text_width(draw, text, font):
    try:
        return int(draw.textlength(text, font=font))
    except AttributeError:
        return int(font.getsize(text)[0]) if hasattr(font, "getsize") else len(str(text)) * 10


def _draw_status_row(draw, y, label, value, *, label_font, value_font, label_color, value_color, x, val_x):
    def _ascent(font):
        try:
            return font.getmetrics()[0]
        except Exception:
            return 0

    label_dy = max(0, _ascent(value_font) - _ascent(label_font))
    value_dy = max(0, _ascent(label_font) - _ascent(value_font))
    draw.text((x, y + label_dy), label, font=label_font, fill=label_color)
    draw.text((val_x, y + value_dy), value, font=value_font, fill=value_color)


def build_fps_overlay_rgba(
    *,
    actual_fps,
    sbs_fps,
    capture_fps,
    latency_ms,
    screen_width,
    screen_height,
    screen_distance,
    depth_strength,
    vr_res,
    sbs_res,
    controller_brand,
    environment_visible,
    font_type=None,
    size=(896, 224),
):
    ow, oh = int(size[0]), int(size[1])
    img = Image.new("RGBA", (ow, oh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0, 0, ow - 1, oh - 1], radius=14, fill=(32, 32, 36, 210))

    font = load_overlay_font(24, font_type, prefer_cjk=True)
    label_font = load_overlay_font(24, font_type, prefer_cjk=True)
    c_label = (150, 158, 185, 255)
    c_green = (0, 230, 90, 255)
    c_cyan = (0, 210, 230, 255)
    c_amber = (255, 190, 40, 255)
    pad = 14
    labels = ["[Performance]", "[3D Display]", "[Resolution]", "[Controller]", "[Environment]"]
    val_x = pad + max(_text_width(draw, label, label_font) for label in labels) + 10

    lat_str = f"{float(latency_ms):.0f}ms" if float(latency_ms or 0.0) > 0 else "N/A"
    fps_str = (
        f"XR {float(actual_fps):.0f} FPS   SBS {float(sbs_fps):.0f} FPS   "
        f"Capture {float(capture_fps):.0f} FPS   Latency {lat_str}"
    )
    _draw_status_row(
        draw,
        22,
        "[Performance]",
        fps_str,
        label_font=label_font,
        value_font=font,
        label_color=c_label,
        value_color=c_green,
        x=pad,
        val_x=val_x,
    )
    scr_str = (
        f"{float(screen_width):.2f} x {float(screen_height):.2f} m"
        f"  @  {float(screen_distance):.2f} m"
        f"   Depth Strength {float(depth_strength):.2f}"
    )
    _draw_status_row(
        draw,
        56,
        "[3D Display]",
        scr_str,
        label_font=label_font,
        value_font=font,
        label_color=c_label,
        value_color=c_cyan,
        x=pad,
        val_x=val_x,
    )
    vw, vh = vr_res
    sw, sh = sbs_res
    _draw_status_row(
        draw,
        90,
        "[Resolution]",
        f"XR {int(vw)}x{int(vh)}/eye   Screen {int(sw)}x{int(sh)}",
        label_font=label_font,
        value_font=font,
        label_color=c_label,
        value_color=c_amber,
        x=pad,
        val_x=val_x,
    )
    if controller_brand:
        _draw_status_row(
            draw,
            124,
            "[Controller]",
            f"Model: {controller_brand}",
            label_font=label_font,
            value_font=font,
            label_color=c_label,
            value_color=c_cyan,
            x=pad,
            val_x=val_x,
        )
    _draw_status_row(
        draw,
        158,
        "[Environment]",
        "ON" if environment_visible else "OFF",
        label_font=label_font,
        value_font=font,
        label_color=c_label,
        value_color=c_cyan,
        x=pad,
        val_x=val_x,
    )
    return np.ascontiguousarray(np.asarray(img, dtype=np.uint8))


def build_team_status_rgba(
    *,
    actual_fps,
    sbs_fps,
    latency_ms,
    screen_width,
    screen_height,
    screen_distance,
    depth_strength,
    vr_res,
    sbs_res,
    environment_name,
    controller_brand,
    shortcuts_visible,
    font_type=None,
    size=(768, 224),
):
    ow, oh = int(size[0]), int(size[1])
    img = Image.new("RGBA", (ow, oh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0, 0, ow - 1, oh - 1], radius=14, fill=(32, 32, 36, 210))

    font = load_overlay_font(24, font_type, prefer_cjk=True)
    label_font = load_overlay_font(24, font_type, prefer_cjk=True)
    c_label = (150, 158, 185, 255)
    c_green = (0, 230, 90, 255)
    c_cyan = (0, 210, 230, 255)
    c_amber = (255, 190, 40, 255)
    pad = 14
    labels = ["[Performance]", "[3D Display]", "[Resolution]", "[Show Shortcuts]", "[Models]"]
    val_x = pad + max(_text_width(draw, label, label_font) for label in labels) + 10

    lat_str = f"{float(latency_ms):.0f}ms" if float(latency_ms or 0.0) > 0 else "--"
    _draw_status_row(
        draw,
        22,
        "[Performance]",
        f"XR {float(actual_fps):.0f} FPS   SBS {float(sbs_fps):.0f} FPS   Latency {lat_str}",
        label_font=label_font,
        value_font=font,
        label_color=c_label,
        value_color=c_green,
        x=pad,
        val_x=val_x,
    )
    _draw_status_row(
        draw,
        56,
        "[3D Display]",
        (
            f"{float(screen_width):.2f} x {float(screen_height):.2f} m"
            f"  @  {float(screen_distance):.2f} m   Depth Strength {float(depth_strength):.2f}"
        ),
        label_font=label_font,
        value_font=font,
        label_color=c_label,
        value_color=c_cyan,
        x=pad,
        val_x=val_x,
    )
    vw, vh = vr_res
    sw, sh = sbs_res
    _draw_status_row(
        draw,
        90,
        "[Resolution]",
        f"XR {int(vw)}x{int(vh)}/eye   Screen {int(sw)}x{int(sh)}",
        label_font=label_font,
        value_font=font,
        label_color=c_label,
        value_color=c_amber,
        x=pad,
        val_x=val_x,
    )
    model_str = f"Environment: {environment_name or 'Default'}"
    if controller_brand:
        model_str += f"   Controller: {controller_brand}"
    _draw_status_row(
        draw,
        124,
        "[Models]",
        model_str,
        label_font=label_font,
        value_font=font,
        label_color=c_label,
        value_color=c_cyan,
        x=pad,
        val_x=val_x,
    )

    draw.text((pad, 158), "[Show Shortcuts]", font=label_font, fill=c_label)
    sw_w, sw_h = 52, 26
    sw_x = val_x
    sw_y = 158 + (34 - sw_h) // 2
    track_col = (0, 200, 80, 255) if shortcuts_visible else (80, 84, 100, 255)
    draw.rounded_rectangle([sw_x, sw_y, sw_x + sw_w, sw_y + sw_h], radius=sw_h // 2, fill=track_col)
    kr = sw_h // 2 - 2
    kx = (sw_x + sw_w - kr - 3) if shortcuts_visible else (sw_x + kr + 3)
    ky = sw_y + sw_h // 2
    draw.ellipse([kx - kr, ky - kr, kx + kr, ky + kr], fill=(255, 255, 255, 255))
    return np.ascontiguousarray(np.asarray(img, dtype=np.uint8))


def build_help_rgba(*, environment_mode=False, font_type=None, lang="EN"):
    rows, env_rows = get_controller_help_rows(lang)
    rows = env_rows if environment_mode else rows
    return _build_help_rows_rgba(rows, font_type=font_type, two_columns=True)


def build_controller_callout_rgba(*, font_type=None, lang="CN", size=(2048, 1536)):
    """Build a transparent, controller-centered operation callout texture."""
    width, height = int(size[0]), int(size[1])
    scale_x = width / 1024.0
    scale_y = height / 768.0
    scale = min(scale_x, scale_y)

    def point(value):
        return (int(round(value[0] * scale_x)), int(round(value[1] * scale_y)))

    def rectangle(value):
        return tuple(
            int(round(component * (scale_x if index % 2 == 0 else scale_y)))
            for index, component in enumerate(value)
        )

    # Transparent texels deliberately keep white RGB. This prevents bilinear
    # filtering at thin opaque edges from sampling transparent black and
    # producing dark borders in the OpenXR compositor.
    img = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    border = (255, 255, 255, 255)
    title_color = (255, 255, 255, 255)
    body_color = (255, 255, 255, 255)
    fill = (255, 255, 255, 0)
    callouts = ((
        (700, 210, 950, 330),
        gettext_for(lang, "B button"),
        (gettext_for(lang, "Hold: show operation guide"),),
        (540, 300),
    ),)

    for rect, title, lines, target in callouts:
        x0, y0, x1, y1 = rect
        scaled_rect = rectangle(rect)
        text_width = max(
            1,
            scaled_rect[2] - scaled_rect[0] - int(round(32 * scale_x)),
        )

        def fit_font(text, nominal_size):
            minimum_size = max(12, int(round(14 * scale)))
            for font_size in range(int(nominal_size), minimum_size - 1, -1):
                font = load_overlay_font(
                    font_size, font_type, prefer_cjk=True
                )
                if draw.textbbox((0, 0), text, font=font)[2] <= text_width:
                    return font
            return load_overlay_font(
                minimum_size, font_type, prefer_cjk=True
            )

        title_font = fit_font(title, int(round(28 * scale)))
        body_fonts = tuple(
            fit_font(line, int(round(20 * scale)))
            for line in lines
        )
        draw.rounded_rectangle(
            scaled_rect,
            radius=max(1, int(round(8 * scale))),
            fill=fill,
            outline=border,
            width=max(1, int(round(3 * scale))),
        )
        draw.text(point((x0 + 16, y0 + 10)), title, font=title_font, fill=title_color)
        for index, (line, body_font) in enumerate(zip(lines, body_fonts)):
            draw.text(
                point((x0 + 16, y0 + 52 + index * 30)),
                line,
                font=body_font,
                fill=body_color,
            )
        edge = (x1, (y0 + y1) // 2) if x1 < 512 else (x0, (y0 + y1) // 2)
        elbow_x = edge[0] + (50 if edge[0] < 512 else -50)
        scaled_edge = point(edge)
        scaled_elbow = point((elbow_x, edge[1]))
        scaled_target = point(target)
        draw.line(
            (scaled_edge, scaled_elbow, scaled_target),
            fill=border,
            width=max(1, int(round(3 * scale))),
        )
        tx, ty = scaled_target
        radius = max(2, int(round(5 * scale)))
        draw.ellipse((tx - radius, ty - radius, tx + radius, ty + radius), fill=border)

    return np.ascontiguousarray(np.asarray(img, dtype=np.uint8))

def build_team_help_rgba(*, font_type=None, lang="EN"):
    rows, _env_rows = get_controller_help_rows(lang)
    return _build_help_rows_rgba(rows, font_type=font_type, two_columns=False)


def _build_help_rows_rgba(rows, *, font_type=None, two_columns):
    font_size = 16 if two_columns else 21
    title_size = 18 if two_columns else 21
    font = load_overlay_font(font_size, font_type, prefer_cjk=True)
    title_font = load_overlay_font(title_size, font_type, prefer_cjk=True)
    draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    col_w = [0, 0, 0]
    for row in rows:
        is_title = bool(row[3]) if len(row) >= 4 else False
        for ci in range(3):
            col_w[ci] = max(col_w[ci], _text_width(draw, row[ci], title_font if is_title else font))

    gap = 20
    mid_gap = 50
    pad_x = 30
    pad_y = 20
    line_h = font_size + 6
    inner_w = col_w[0] + gap + col_w[1] + gap + col_w[2]
    if two_columns:
        title_indices = [i for i, row in enumerate(rows) if len(row) >= 4 and row[3]]
        mid_idx = title_indices[4] if len(title_indices) > 4 else len(rows)
        left_rows = rows[:mid_idx]
        right_rows = rows[mid_idx:]
        tw = inner_w * 2 + mid_gap + pad_x * 2
        th = max(len(left_rows), len(right_rows)) * line_h + pad_y * 2
    else:
        left_rows = rows
        right_rows = []
        tw = inner_w + pad_x * 2
        th = len(rows) * line_h + pad_y * 2

    img = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0, 0, tw - 1, th - 1], radius=14, fill=(18, 18, 28, 210))
    col_x = [pad_x, pad_x + col_w[0] + gap, pad_x + col_w[0] + gap + col_w[1] + gap]
    col_x2 = [pad_x + inner_w + mid_gap, pad_x + inner_w + mid_gap + col_w[0] + gap, pad_x + inner_w + mid_gap + col_w[0] + gap + col_w[1] + gap]

    def _draw_rows(group_rows, xs):
        for ri, row in enumerate(group_rows):
            is_title = bool(row[3]) if len(row) >= 4 else False
            y = pad_y + ri * line_h
            row_font = title_font if is_title else font
            color = (90, 190, 255, 255) if is_title else (200, 210, 235, 255)
            for ci in range(3):
                if row[ci]:
                    draw.text((xs[ci], y), row[ci], font=row_font, fill=color)

    _draw_rows(left_rows, col_x)
    if two_columns:
        _draw_rows(right_rows, col_x2)
    return np.ascontiguousarray(np.asarray(img, dtype=np.uint8))
