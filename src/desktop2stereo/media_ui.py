from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from gui.localization import gettext_for, normalize_locale
from xr_viewer.overlay_textures import load_overlay_font
from xr_viewer.settings_menu import OPENXR_MENU_COLORS, OPENXR_MENU_RADII

_FONT_PATH = Path(__file__).resolve().parent / "xr_viewer" / "fonts" / "InterVariable.ttf"
_MEDIA_BAR_SIZE = (1024, 160)
MEDIA_BROWSER_TEXTURE_SIZE = (1600, 1200)
MEDIA_BROWSER_TEXTURE_ASPECT = (
    MEDIA_BROWSER_TEXTURE_SIZE[0] / MEDIA_BROWSER_TEXTURE_SIZE[1]
)
_MEDIA_BROWSER_SIZE = MEDIA_BROWSER_TEXTURE_SIZE
MEDIA_BROWSER_VISIBLE_ROWS = 10
_BROWSER_VISIBLE_ROWS = MEDIA_BROWSER_VISIBLE_ROWS
MEDIA_HOVER_DURATION_S = 0.16
_MEDIA_HOVER_BUTTON_SCALE = 0.035
_MEDIA_HOVER_ROW_SCALE = 0.010


@dataclass
class MediaHoverAnimation:
    """Shared hover timing for Local Viewer and OpenXR media surfaces."""

    duration_s: float = MEDIA_HOVER_DURATION_S
    target_hit: str | None = None
    visual_hit: str | None = None
    progress: float = 0.0
    last_at: float | None = None

    def update(self, hover: str | None, now: float) -> tuple[str | None, float]:
        hover = str(hover) if hover else None
        previous_identity = _hover_animation_identity(self.target_hit)
        next_identity = _hover_animation_identity(hover)
        if hover != self.target_hit:
            self.target_hit = hover
            if hover is not None:
                self.visual_hit = hover
                if next_identity != previous_identity:
                    self.progress = 0.0
        if self.last_at is None:
            self.last_at = float(now)
        dt = max(0.0, min(0.1, float(now) - self.last_at))
        self.last_at = float(now)
        target = 1.0 if self.target_hit is not None else 0.0
        duration = max(0.001, float(self.duration_s))
        if target > self.progress:
            self.progress = min(target, self.progress + dt / duration)
        elif target < self.progress:
            self.progress = max(target, self.progress - dt / (duration * 0.82))
        if self.progress <= 0.0 and self.target_hit is None:
            self.visual_hit = None
        return self.visual_hit, max(0.0, min(1.0, self.progress))


def _rgba(token: str, alpha: int = 255) -> tuple[int, int, int, int]:
    value = OPENXR_MENU_COLORS[token].lstrip("#")
    return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4)) + (int(alpha),)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _ease_out_cubic(value: float) -> float:
    value = _clamp01(value)
    return 1.0 - (1.0 - value) ** 3


def _mix_rgba(
    start: tuple[int, int, int, int],
    end: tuple[int, int, int, int],
    amount: float,
) -> tuple[int, int, int, int]:
    amount = _ease_out_cubic(amount)
    return tuple(
        int(round(a + (b - a) * amount)) for a, b in zip(start, end)
    )


def _scaled_rect(
    rect: tuple[int, int, int, int],
    progress: float,
    *,
    scale: float,
    lift: float = 0.0,
) -> tuple[int, int, int, int]:
    amount = _ease_out_cubic(progress)
    x0, y0, x1, y1 = (float(value) for value in rect)
    cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5 - lift * amount
    half_w = (x1 - x0) * 0.5 * (1.0 + float(scale) * amount)
    half_h = (y1 - y0) * 0.5 * (1.0 + float(scale) * amount)
    return (
        int(round(cx - half_w)),
        int(round(cy - half_h)),
        int(round(cx + half_w)),
        int(round(cy + half_h)),
    )


def _hover_parts(hover: str | None) -> tuple[str, str]:
    value = str(hover or "")
    action, separator, target = value.partition(":")
    return (action.casefold(), target if separator else "")


def _hover_animation_identity(hover: str | None) -> str | None:
    if not hover:
        return None
    action, _target = _hover_parts(hover)
    if action in {"seek", "volume"}:
        return action
    return str(hover)


def _hover_amount(hover: str | None, action: str, progress: float) -> float:
    return _clamp01(progress) if _hover_parts(hover)[0] == str(action).casefold() else 0.0


def _draw_control_surface(
    draw,
    rect: tuple[int, int, int, int],
    *,
    enabled: bool,
    hovered: bool,
    progress: float,
    primary: bool = False,
    idle_alpha: int = 220,
    disabled_alpha: int = 105,
    radius: int = OPENXR_MENU_RADII["control"],
) -> tuple[int, int, int, int]:
    amount = _clamp01(progress) if hovered and enabled else 0.0
    target = _scaled_rect(
        rect,
        amount,
        scale=_MEDIA_HOVER_BUTTON_SCALE,
        lift=1.5,
    )
    base = _rgba(
        "primary_container" if primary else "surface_container_high",
        idle_alpha if enabled else disabled_alpha,
    )
    hover = _rgba("primary_hover", 230 if enabled else disabled_alpha)
    fill = _mix_rgba(base, hover, amount)
    outline = _rgba(
        "primary_hover" if primary else "outline",
        int(28 + 72 * _ease_out_cubic(amount)) if enabled else 20,
    )
    if primary and amount > 0.0 and enabled:
        halo = (
            target[0] - 2,
            target[1] - 2,
            target[2] + 2,
            target[3] + 2,
        )
        draw.rounded_rectangle(
            halo,
            radius=radius + 2,
            outline=_rgba("primary_hover", int(35 + 95 * _ease_out_cubic(amount))),
            width=2,
        )
    draw.rounded_rectangle(target, radius=radius, fill=fill, outline=outline, width=1)
    return target


def _font(size: int, *, bold: bool = False):
    try:
        return load_overlay_font(
            int(size),
            prefer_cjk=True,
            bold=bool(bold),
        )
    except Exception:
        from PIL import ImageFont

        try:
            return ImageFont.truetype(str(_FONT_PATH), int(size))
        except Exception:
            return ImageFont.load_default()


def _text(draw, xy, text, size=20, *, fill=None, bold=False, anchor="la"):
    draw.text(
        xy,
        str(text),
        font=_font(size, bold=bold),
        fill=fill or _rgba("text_primary"),
        anchor=anchor,
    )


def _fit_text(draw, text: str, size: int, max_width: float, *, bold: bool = False) -> str:
    value = str(text or "")
    if max_width <= 0:
        return ""
    font = _font(size, bold=bold)
    if draw.textlength(value, font=font) <= max_width:
        return value
    low, high = 0, len(value)
    while low < high:
        middle = (low + high + 1) // 2
        candidate = value[:middle].rstrip() + "…"
        if draw.textlength(candidate, font=font) <= max_width:
            low = middle
        else:
            high = middle - 1
    return value[:low].rstrip() + "…" if low else "…"


def _draw_file_icon(draw, cx: int, cy: int, color) -> None:
    left, top, right, bottom = cx - 9, cy - 11, cx + 9, cy + 11
    draw.rounded_rectangle((left, top, right, bottom), radius=2, outline=color, width=2)
    draw.line((cx + 2, top, right, top + 7, cx + 2, top + 7), fill=color, width=2)
    draw.line((left + 4, cy + 2, right - 4, cy + 2), fill=color, width=2)
    draw.line((left + 4, cy + 7, right - 4, cy + 7), fill=color, width=2)


def _draw_disc_icon(draw, cx: int, cy: int, color) -> None:
    draw.ellipse((cx - 11, cy - 11, cx + 11, cy + 11), outline=color, width=2)
    draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), outline=color, width=2)
    draw.ellipse((cx - 1, cy - 1, cx + 1, cy + 1), fill=color)


def _draw_folder_icon(draw, cx: int, cy: int, color) -> None:
    draw.rounded_rectangle(
        (cx - 11, cy - 8, cx + 11, cy + 9), radius=2, outline=color, width=2
    )
    draw.line((cx - 10, cy - 8, cx - 5, cy - 12, cx + 2, cy - 12), fill=color, width=2)
    draw.line((cx + 2, cy - 12, cx + 5, cy - 8), fill=color, width=2)


def _draw_video_icon(draw, cx: int, cy: int, color) -> None:
    draw.rounded_rectangle(
        (cx - 12, cy - 8, cx + 12, cy + 9), radius=2, outline=color, width=2
    )
    draw.polygon(((cx - 3, cy - 5), (cx + 6, cy), (cx - 3, cy + 5)), fill=color)


def _draw_image_icon(draw, cx: int, cy: int, color) -> None:
    draw.rounded_rectangle(
        (cx - 12, cy - 9, cx + 12, cy + 9), radius=2, outline=color, width=2
    )
    draw.ellipse((cx + 3, cy - 5, cx + 7, cy - 1), fill=color)
    draw.line((cx - 9, cy + 6, cx - 2, cy - 1, cx + 4, cy + 5), fill=color, width=2)
    draw.line((cx + 4, cy + 5, cx + 8, cy + 1, cx + 11, cy + 5), fill=color, width=2)


def _draw_drive_icon(draw, cx: int, cy: int, color) -> None:
    draw.rounded_rectangle(
        (cx - 12, cy - 8, cx + 12, cy + 8), radius=3, outline=color, width=2
    )
    draw.line((cx - 7, cy + 4, cx + 7, cy + 4), fill=color, width=2)
    draw.ellipse((cx + 5, cy + 1, cx + 8, cy + 4), fill=color)


def _draw_close_x(draw, cx: int, cy: int, color) -> None:
    draw.line((cx - 5, cy - 5, cx + 5, cy + 5), fill=color, width=2)
    draw.line((cx + 5, cy - 5, cx - 5, cy + 5), fill=color, width=2)


def _draw_kind_icon(draw, kind: str, cx: int, cy: int, color) -> None:
    if kind in {"directory", "disc"}:
        if kind == "disc":
            _draw_disc_icon(draw, cx, cy, color)
        else:
            _draw_folder_icon(draw, cx, cy, color)
    elif kind == "video":
        _draw_video_icon(draw, cx, cy, color)
    elif kind == "image":
        _draw_image_icon(draw, cx, cy, color)
    elif kind == "drive":
        _draw_drive_icon(draw, cx, cy, color)
    else:
        _draw_file_icon(draw, cx, cy, color)


def _draw_play_icon(draw, cx: int, cy: int, color, *, pause: bool) -> None:
    if pause:
        draw.rounded_rectangle((cx - 8, cy - 11, cx - 3, cy + 11), radius=2, fill=color)
        draw.rounded_rectangle((cx + 3, cy - 11, cx + 8, cy + 11), radius=2, fill=color)
        return
    draw.polygon(((cx - 7, cy - 12), (cx + 11, cy), (cx - 7, cy + 12)), fill=color)


def _draw_skip_icon(draw, cx: int, cy: int, color, *, previous: bool) -> None:
    # Previous is |◀; next is ▶|.
    sign = 1 if previous else -1
    bar_left = cx - sign * 10 - 2
    draw.rectangle((bar_left, cy - 10, bar_left + 4, cy + 10), fill=color)
    draw.polygon(
        (
            (cx + sign * 8, cy - 10),
            (cx - sign * 3, cy),
            (cx + sign * 8, cy + 10),
        ),
        fill=color,
    )


def _draw_speaker_icon(draw, cx: int, cy: int, color, *, muted: bool) -> None:
    draw.polygon(
        (
            (cx - 11, cy - 4),
            (cx - 5, cy - 4),
            (cx + 2, cy - 10),
            (cx + 2, cy + 10),
            (cx - 5, cy + 4),
            (cx - 11, cy + 4),
        ),
        fill=color,
    )
    if muted:
        draw.line((cx + 5, cy - 7, cx + 15, cy + 7), fill=color, width=2)
        draw.line((cx + 15, cy - 7, cx + 5, cy + 7), fill=color, width=2)
    else:
        draw.arc((cx - 2, cy - 10, cx + 15, cy + 10), -55, 55, fill=color, width=2)
        draw.arc((cx + 1, cy - 14, cx + 20, cy + 14), -50, 50, fill=color, width=2)


def _format_time(seconds: float) -> str:
    total = max(0, int(float(seconds or 0.0)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def media_transport_icon(state: dict) -> str:
    """Return the control glyph for the next available transport action.

    A playing video exposes Pause; a paused video exposes Play. Keeping this
    decision in one helper prevents the Local Viewer and XR renderers from
    drifting into opposite icon semantics.
    """
    return "pause" if bool(state.get("playing", False)) else "play"


def media_control_enabled(action: str, state: dict) -> bool:
    """Return whether the current media state allows a shared player action."""
    if not state:
        return False
    ready = bool(state.get("ready", True))
    mode = str(state.get("mode") or "Video")
    video = ready and mode != "Image" and float(state.get("duration", 0.0) or 0.0) > 0.0
    audio = video and bool(state.get("audio_enabled", True))
    playlist_size = len(state.get("playlist_entries") or [])
    action = str(action or "").casefold()
    if action in {"browse", "browse_disc", "browser_close"}:
        return ready
    if action == "browser_up":
        return ready and bool(state.get("browser_parent"))
    if action == "browser_refresh":
        return ready and bool(state.get("browser_path"))
    if action == "browse_open":
        return ready and bool(state.get("browser_selection"))
    if action in {"recent", "recent_remove"}:
        return ready and bool(state.get("recent_entries"))
    if action == "play":
        return video
    if action in {"seek-10", "seek+10", "seek", "seek_relative"}:
        return video
    if action in {"mute", "volume"}:
        return audio
    if action in {"playlist_previous", "playlist_next"}:
        return ready and playlist_size > 1
    if action == "playlist_clear":
        return ready and playlist_size > 0
    if action in {"playlist_select", "playlist_remove", "playlist_move_up", "playlist_move_down"}:
        return ready and playlist_size > 0
    if action == "playlist_sort":
        return ready and playlist_size > 1
    if action == "playlist_add":
        return ready and bool(state.get("browser_selection"))
    if action == "playlist_page_previous":
        return ready and int(state.get("playlist_offset", 0) or 0) > 0
    if action == "playlist_page_next":
        return ready and int(state.get("playlist_offset", 0) or 0) + 4 < playlist_size
    if action in {"entry", "mount"}:
        return ready
    if action == "wake":
        return True
    return ready


def _seek_track_rect(width: int, height: int) -> tuple[int, int, int, int]:
    track_y = max(26, int(height * 0.20))
    return (24, track_y - 10, width - 24, track_y + 16)


def _browser_layout(state: dict, width: int, height: int) -> dict[str, int]:
    """Keep browser painting and hit testing on one shared geometry."""
    mounts = list(state.get("browser_mounts") or [])
    recent = list(state.get("recent_entries") or [])
    playlist = list(state.get("playlist_entries") or [])
    sidebar_right = max(480, min(520, int(width * 0.31)))
    mount_count = min(3, len(mounts))
    recent_count = min(2, len(recent))
    playlist_offset = max(0, min(int(state.get("playlist_offset", 0) or 0), max(0, len(playlist) - 1)))
    playlist_count = min(4, len(playlist) - playlist_offset)
    mount_top = max(150, int(height * 0.125))
    mount_row_height = 72
    recent_heading_y = mount_top + mount_count * mount_row_height + 34
    recent_rows_y = recent_heading_y + 36
    recent_row_height = 72
    playlist_heading_y = (
        recent_rows_y + max(1, recent_count) * recent_row_height + 34
    )
    playlist_rows_y = playlist_heading_y + 36
    playlist_row_height = 72
    file_rows_y = max(164, int(height * 0.14))
    return {
        "sidebar_right": sidebar_right,
        "mount_top": mount_top,
        "mount_count": mount_count,
        "mount_row_height": mount_row_height,
        "recent_heading_y": recent_heading_y,
        "recent_rows_y": recent_rows_y,
        "recent_count": recent_count,
        "recent_row_height": recent_row_height,
        "playlist_heading_y": playlist_heading_y,
        "playlist_rows_y": playlist_rows_y,
        "playlist_count": playlist_count,
        "playlist_offset": playlist_offset,
        "playlist_row_height": playlist_row_height,
        "file_rows_y": file_rows_y,
        "file_row_height": max(
            72, (height - file_rows_y - 120) // _BROWSER_VISIBLE_ROWS
        ),
    }


def _browser_button_rects(
    width: int, height: int
) -> dict[str, tuple[int, int, int, int]]:
    """Return MetaXR-sized browser actions with at least 48 px targets."""
    top_margin = max(24, int(width * 0.018))
    footer_margin = max(48, int(width * 0.030))
    top_gap = max(10, int(width * 0.008))
    footer_gap = max(24, int(width * 0.015))
    button_w = 112
    footer_button_w = 112
    button_h = 56
    footer_y = height - footer_margin - button_h
    refresh_x = width - top_margin - button_w
    up_x = refresh_x - top_gap - button_w
    close_x = up_x - top_gap - button_w
    previous_x = footer_margin
    next_x = previous_x + footer_button_w + footer_gap
    clear_x = next_x + footer_button_w + footer_gap
    return {
        "browser_close": (
            close_x,
            top_margin,
            close_x + button_w,
            top_margin + button_h,
        ),
        "browser_up": (
            up_x,
            top_margin,
            up_x + button_w,
            top_margin + button_h,
        ),
        "browser_refresh": (
            refresh_x,
            top_margin,
            refresh_x + button_w,
            top_margin + button_h,
        ),
        "browse_open": (
            width - footer_margin - 176,
            footer_y,
            width - footer_margin,
            footer_y + button_h,
        ),
        "playlist_previous": (
            previous_x,
            footer_y,
            previous_x + footer_button_w,
            footer_y + button_h,
        ),
        "playlist_next": (
            next_x,
            footer_y,
            next_x + footer_button_w,
            footer_y + button_h,
        ),
        "playlist_clear": (clear_x, footer_y, clear_x + footer_button_w, footer_y + button_h),
        "playlist_add": (clear_x + footer_button_w + footer_gap, footer_y,
                         clear_x + 2 * footer_button_w + footer_gap, footer_y + button_h),
        "playlist_sort": (clear_x + 2 * (footer_button_w + footer_gap), footer_y,
                          clear_x + 3 * footer_button_w + 2 * footer_gap, footer_y + button_h),
    }


def _button_rects(width: int, height: int) -> dict[str, tuple[int, int, int, int]]:
    # Spacing follows one rule: equal outer padding, one gutter inside each
    # group, and equal flexible gutters between the three logical groups.
    margin = max(18, int(width * 0.018))
    gap = max(10, int(width * 0.012))
    control_center = 108
    open_h = 72
    small_h = 64
    play_size = 80
    y0 = control_center - small_h // 2
    bottom = control_center + small_h // 2
    open_top = control_center - open_h // 2
    open_bottom = control_center + open_h // 2
    play_top = control_center - play_size // 2
    play_bottom = control_center + play_size // 2
    browser_button_y = margin
    browser_button_bottom = margin + 48
    browser_button_w = 96
    browser_gap = 8
    browser_refresh_x = width - margin - browser_button_w
    browser_up_x = browser_refresh_x - browser_gap - browser_button_w
    browser_close_x = browser_up_x - browser_gap - browser_button_w
    footer_y = height - margin - 48

    open_file_w = max(136, min(164, int(width * 0.148)))
    open_disc_w = max(132, min(156, int(width * 0.142)))
    browse_right = margin + open_file_w
    disc_left = browse_right + gap
    disc_right = disc_left + open_disc_w

    previous_w = 64
    step_w = 64
    play_w = play_size
    transport_w = previous_w * 2 + step_w * 2 + play_w + gap * 4

    mute_w = 64
    volume_w = max(132, min(150, int(width * 0.146)))
    volume_left = width - margin - volume_w
    mute_left = volume_left - gap - mute_w
    transport_left = disc_right + max(
        gap,
        (mute_left - disc_right - transport_w) // 2,
    )
    previous_left = transport_left
    seek_back_left = previous_left + previous_w + gap
    play_left = seek_back_left + step_w + gap
    seek_forward_left = play_left + play_w + gap
    next_left = seek_forward_left + step_w + gap
    return {
        "browse": (margin, open_top, browse_right, open_bottom),
        "browse_disc": (disc_left, open_top, disc_right, open_bottom),
        "previous": (previous_left, y0, previous_left + previous_w, bottom),
        "seek-10": (seek_back_left, y0, seek_back_left + step_w, bottom),
        "play": (play_left, play_top, play_left + play_w, play_bottom),
        "seek+10": (seek_forward_left, y0, seek_forward_left + step_w, bottom),
        "next": (next_left, y0, next_left + previous_w, bottom),
        "mute": (mute_left, y0, mute_left + mute_w, bottom),
        "volume": (volume_left, y0, volume_left + volume_w, bottom),
        "browser_up": (browser_up_x, browser_button_y, browser_up_x + browser_button_w, browser_button_bottom),
        "browser_refresh": (
            browser_refresh_x,
            browser_button_y,
            browser_refresh_x + browser_button_w,
            browser_button_bottom,
        ),
        "browser_close": (browser_close_x, browser_button_y, browser_close_x + browser_button_w, browser_button_bottom),
        "browse_open": (width - margin - 132, footer_y, width - margin, footer_y + 48),
        "playlist_previous": (margin, footer_y, margin + 82, footer_y + 48),
        "playlist_next": (margin + 88, footer_y, margin + 170, footer_y + 48),
        "playlist_clear": (margin + 176, footer_y, margin + 272, footer_y + 48),
    }


def build_media_bar_rgba(
    state: dict,
    *,
    width: int = 1024,
    height: int = 160,
    hover: str | None = None,
    hover_progress: float = 1.0,
    alpha: int = 188,
    lang: str = "EN",
    collapsed: bool = False,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    def translate(message):
        return gettext_for(normalize_locale(lang), message)

    image = Image.new("RGBA", (int(width), int(height)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    width, height = image.size
    if collapsed:
        # The controls fully disappear when idle. Pointer/ray movement still
        # reveals them through the invisible wake zone below.
        return np.asarray(image, dtype=np.uint8)
    outer = (6, 6, width - 7, height - 7)
    draw.rounded_rectangle(
        outer,
        radius=OPENXR_MENU_RADII["panel"],
        fill=_rgba("surface", alpha),
        outline=_rgba("text_secondary", 48),
        width=2,
    )
    rects = _button_rects(width, height)
    hover_action, hover_target = _hover_parts(hover)
    hover_amount = _clamp01(hover_progress)
    duration = max(0.0, float(state.get("duration", 0.0) or 0.0))
    position = max(0.0, min(float(state.get("position", 0.0) or 0.0), duration or 1e12))
    fraction = position / duration if duration > 0 else 0.0
    track_left, track_right = 24, width - 24
    track_y = max(26, int(height * 0.20))
    # Keep endpoint thumb/glow inside a label gutter so hover scaling never
    # crowds the elapsed/duration timecodes at either end of the timeline.
    time_label_inset = 22
    seek_enabled = media_control_enabled("seek", state)
    _text(draw, (track_left + time_label_inset, 10), _format_time(position), 16, fill=_rgba("text_secondary"))
    _text(
        draw,
        (track_right - time_label_inset, 10),
        _format_time(duration),
        16,
        fill=_rgba("text_secondary"),
        anchor="ra",
    )
    seek_hovered = bool(seek_enabled and hover_action == "seek")
    seek_progress = hover_amount if seek_hovered else 0.0
    track_height = 8 + int(round(4 * _ease_out_cubic(seek_progress)))
    track_top = track_y - (track_height - 8) // 2
    draw.rounded_rectangle(
        (track_left, track_top, track_right, track_top + track_height),
        radius=max(4, track_height // 2),
        fill=_mix_rgba(
            _rgba("track", 210 if seek_enabled else 120),
            _rgba("primary", 230),
            seek_progress * 0.35,
        ),
    )
    fill_x = int(track_left + (track_right - track_left) * fraction)
    if seek_enabled and fill_x > track_left:
        draw.rounded_rectangle(
            (track_left, track_top, fill_x, track_top + track_height),
            radius=max(4, track_height // 2),
            fill=_mix_rgba(_rgba("primary"), _rgba("primary_hover"), seek_progress),
        )
    if duration > 0 and seek_enabled:
        draw.ellipse(
            (fill_x - 8, track_y - 4, fill_x + 8, track_y + 12),
            fill=_rgba("text_primary"),
        )
    if seek_hovered:
        try:
            preview_fraction = _clamp01(float(hover_target))
        except (TypeError, ValueError):
            preview_fraction = fraction
        preview_x = int(track_left + (track_right - track_left) * preview_fraction)
        preview_radius = 8 + int(round(4 * _ease_out_cubic(seek_progress)))
        glow_radius = preview_radius + 5
        draw.ellipse(
            (
                preview_x - glow_radius,
                track_y + 4 - glow_radius,
                preview_x + glow_radius,
                track_y + 4 + glow_radius,
            ),
            outline=_rgba("primary_hover", int(70 + 80 * seek_progress)),
            width=2,
        )
        draw.ellipse(
            (
                preview_x - preview_radius,
                track_y + 4 - preview_radius,
                preview_x + preview_radius,
                track_y + 4 + preview_radius,
            ),
            fill=_rgba("text_primary"),
            outline=_rgba("primary_hover"),
            width=2,
        )

    def button(
        key: str,
        label: str,
        *,
        icon: str | None = None,
        icon_kind: str | None = None,
    ):
        source = rects[key]
        enabled = media_control_enabled(key, state)
        mapped_hover = key
        if key == "seek-10":
            active_hover = hover_action == "seek_relative" and hover_target == "-10"
        elif key == "seek+10":
            active_hover = hover_action == "seek_relative" and hover_target == "10"
        elif key == "previous":
            active_hover = hover_action == "playlist_previous"
        elif key == "next":
            active_hover = hover_action == "playlist_next"
        else:
            active_hover = hover_action == mapped_hover.casefold()
        active = bool(enabled and active_hover)
        progress = hover_amount if active else 0.0
        color = _rgba("text_primary", 255 if enabled else 104)
        if key == "play":
            draw_rect = _scaled_rect(
                source,
                progress,
                scale=_MEDIA_HOVER_BUTTON_SCALE,
                lift=1.5,
            )
            x0, y0, x1, y1 = draw_rect
            ring = _mix_rgba(
                _rgba("primary", 255 if enabled else 105),
                _rgba("primary_hover", 255 if enabled else 105),
                progress,
            )
            if progress > 0.0 and enabled:
                halo = (x0 - 2, y0 - 2, x1 + 2, y1 + 2)
                draw.ellipse(
                    halo,
                    outline=_rgba("primary_hover", int(35 + 95 * _ease_out_cubic(progress))),
                    width=2,
                )
            draw.ellipse(draw_rect, fill=ring)
            inner = _scaled_rect(
                (x0 + 4, y0 + 4, x1 - 4, y1 - 4),
                progress,
                scale=_MEDIA_HOVER_BUTTON_SCALE,
            )
            draw.ellipse(
                inner,
                fill=_rgba("surface_container_high", 245 if enabled else 130),
            )
        else:
            draw_rect = _draw_control_surface(
                draw,
                source,
                enabled=enabled,
                hovered=active,
                progress=progress,
                idle_alpha=220 if key in {"browse", "browse_disc"} else 74,
                disabled_alpha=34,
                radius=OPENXR_MENU_RADII["control"],
            )
            x0, y0, x1, y1 = draw_rect
        x0, y0, x1, y1 = draw_rect
        center_y = (y0 + y1) // 2
        if icon_kind == "file":
            _draw_file_icon(draw, x0 + 22, center_y, color)
            _text(draw, (x0 + 44, center_y), label, 14, fill=color, anchor="lm")
        elif icon_kind == "disc":
            _draw_disc_icon(draw, x0 + 22, center_y, color)
            _text(draw, (x0 + 44, center_y), label, 14, fill=color, anchor="lm")
        elif icon_kind in {"previous", "next"}:
            _draw_skip_icon(
                draw,
                (x0 + x1) // 2,
                center_y,
                color,
                previous=icon_kind == "previous",
            )
        elif icon_kind in {"play", "pause"}:
            _draw_play_icon(
                draw,
                (x0 + x1) // 2,
                center_y,
                color,
                pause=icon_kind == "pause",
            )
        elif icon_kind in {"speaker", "muted"}:
            _draw_speaker_icon(
                draw,
                (x0 + x1) // 2,
                center_y,
                color,
                muted=icon_kind == "muted",
            )
        elif icon:
            _text(draw, ((x0 + x1) // 2, center_y), icon, 25, fill=color, anchor="mm")
        elif label:
            size = 18 if (x1 - x0) >= 90 else 16
            _text(
                draw,
                ((x0 + x1) // 2, (y0 + y1) // 2),
                label,
                size,
                fill=color,
                anchor="mm",
                bold=key == "play",
            )

    button("browse", translate("Open File"), icon_kind="file")
    button("browse_disc", translate("Open Disc"), icon_kind="disc")
    button("previous", "", icon_kind="previous")
    button("seek-10", "-10s")
    transport_icon = media_transport_icon(state)
    button(
        "play",
        translate("Pause") if transport_icon == "pause" else translate("Play"),
        icon_kind=transport_icon,
    )
    button("seek+10", "+10s")
    button("next", "", icon_kind="next")
    button(
        "mute",
        "",
        icon_kind="muted" if state.get("muted") else "speaker",
    )
    volume = max(0.0, min(1.0, float(state.get("volume", 1.0) or 0.0)))
    volume_enabled = media_control_enabled("volume", state)
    volume_active = bool(hover_action == "volume" and volume_enabled)
    volume_progress = hover_amount if volume_active else 0.0
    x0, y0, x1, y1 = _draw_control_surface(
        draw,
        rects["volume"],
        enabled=volume_enabled,
        hovered=volume_active,
        progress=volume_progress,
        idle_alpha=74,
        disabled_alpha=34,
        radius=OPENXR_MENU_RADII["control"],
    )
    center_y = (y0 + y1) // 2
    track_height = 8 + int(round(2 * _ease_out_cubic(volume_progress)))
    track_x0, track_x1 = x0 + 12, x1 - 12
    draw.rounded_rectangle(
        (track_x0, center_y - track_height // 2, track_x1, center_y + track_height // 2),
        radius=max(4, track_height // 2),
        fill=_mix_rgba(
            _rgba("track", 220 if volume_enabled else 105),
            _rgba("primary", 230),
            volume_progress * 0.28,
        ),
    )
    vx = int(track_x0 + (track_x1 - track_x0) * volume)
    knob_radius = 7 + int(round(2 * _ease_out_cubic(volume_progress)))
    if volume_enabled:
        draw.rounded_rectangle(
            (track_x0, center_y - track_height // 2, vx, center_y + track_height // 2),
            radius=max(4, track_height // 2),
            fill=_mix_rgba(_rgba("primary"), _rgba("primary_hover"), volume_progress),
        )
    draw.ellipse(
        (vx - knob_radius, center_y - knob_radius, vx + knob_radius, center_y + knob_radius),
        fill=_rgba("text_primary", 255 if volume_enabled else 104),
        outline=_rgba("primary_hover", int(255 * volume_progress)) if volume_active else None,
        width=2 if volume_active else 1,
    )
    return np.asarray(image, dtype=np.uint8)


def media_bar_hit(
    u: float,
    v: float,
    state: dict | None = None,
    *,
    collapsed: bool = False,
) -> str | None:
    width, height = _MEDIA_BAR_SIZE
    x, y = float(u) * width, float(v) * height
    state = state or {}
    if collapsed:
        # Keep a generous invisible wake target for XR rays and local pointer
        # movement, but never render a persistent bottom bar or handle.
        wake_zone = (
            int(width * 0.18),
            int(height * 0.82),
            int(width * 0.82),
            height,
        )
        x0, y0, x1, y1 = wake_zone
        return "wake" if x0 <= x <= x1 and y0 <= y <= y1 else None
    track_x0, track_y0, track_x1, track_y1 = _seek_track_rect(width, height)
    if (
        media_control_enabled("seek", state)
        and track_x0 <= x <= track_x1
        and track_y0 <= y <= track_y1
    ):
        fraction = media_bar_value_fraction(u, action="seek")
        return f"seek:{fraction:.6f}"
    rects = _button_rects(width, height)
    for key in (
        "browse",
        "browse_disc",
        "seek-10",
        "play",
        "seek+10",
        "previous",
        "next",
        "mute",
        "volume",
    ):
        x0, y0, x1, y1 = rects[key]
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            continue
        if key == "volume":
            if not media_control_enabled(key, state):
                return None
            fraction = media_bar_value_fraction(u, action="volume")
            return f"volume:{fraction:.6f}"
        if not media_control_enabled(key, state):
            return None
        if key in {"seek-10", "seek+10"}:
            return f"seek_relative:{10 if key.endswith('+10') else -10}"
        if key == "previous":
            return "playlist_previous"
        if key == "next":
            return "playlist_next"
        return key
    return None


def media_bar_value_fraction(u: float, *, action: str) -> float | None:
    """Map a bar-local horizontal coordinate to a control's value.

    The volume control covers only a small part of the bar, so drag/click
    handling must not treat the whole-bar coordinate as the volume value.
    """
    width, height = _MEDIA_BAR_SIZE
    try:
        x = max(0.0, min(1.0, float(u))) * width
    except (TypeError, ValueError):
        return None
    action = str(action or "").casefold()
    if action == "seek":
        start, end = 24.0, float(width - 24)
    elif action == "volume":
        start, _, end, _ = _button_rects(width, height)["volume"]
        start, end = float(start + 12), float(end - 12)
    else:
        return None
    return max(0.0, min(1.0, (x - start) / max(1.0, end - start)))


def build_pointer_rgba(size: int = 48) -> np.ndarray:
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    center = size // 2
    draw.ellipse(
        (center - 12, center - 12, center + 12, center + 12),
        outline=_rgba("text_primary", 245),
        width=3,
    )
    draw.line((center, center - 20, center, center - 6), fill=_rgba("primary", 255), width=3)
    draw.line((center, center + 6, center, center + 20), fill=_rgba("primary", 255), width=3)
    draw.line((center - 20, center, center - 6, center), fill=_rgba("primary", 255), width=3)
    draw.line((center + 6, center, center + 20, center), fill=_rgba("primary", 255), width=3)
    return np.asarray(image, dtype=np.uint8)


def build_media_browser_rgba(
    state: dict,
    *,
    width: int = MEDIA_BROWSER_TEXTURE_SIZE[0],
    height: int = MEDIA_BROWSER_TEXTURE_SIZE[1],
    hover: str | None = None,
    hover_progress: float = 1.0,
    scroll_offset: int = 0,
    lang: str = "EN",
) -> np.ndarray:
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (int(width), int(height)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    width, height = image.size
    layout = _browser_layout(state, width, height)
    rects = _browser_button_rects(width, height)
    shell = (12, 12, width - 13, height - 13)
    draw.rounded_rectangle(shell, radius=OPENXR_MENU_RADII["panel"], fill=_rgba("surface", 238))
    sidebar_right = layout["sidebar_right"]
    draw.rounded_rectangle(
        (28, 28, sidebar_right, height - 28),
        radius=OPENXR_MENU_RADII["group"],
        fill=_rgba("surface_container", 235),
    )
    def translate(message):
        return gettext_for(normalize_locale(lang), message)

    draw.text(
        (48, 84),
        translate("Locations"),
        font=_font(22, bold=True),
        fill=_rgba("text_secondary"),
    )
    mounts = list(state.get("browser_mounts") or [])
    hover_action, hover_target = _hover_parts(hover)
    hover_amount = _clamp01(hover_progress)
    hover_path = hover_target
    for index, mount in enumerate(mounts[:layout["mount_count"]]):
        center_y = layout["mount_top"] + index * layout["mount_row_height"] + layout["mount_row_height"] // 2
        path = str(mount.get("path", ""))
        selected = path == str(state.get("browser_path", ""))
        active = hover_action == "mount" and path == hover_path
        if selected or active:
            row = _scaled_rect(
                (40, center_y - 32, sidebar_right - 14, center_y + 32),
                hover_amount if active else 0.0,
                scale=_MEDIA_HOVER_ROW_SCALE,
                lift=0.8,
            )
            draw.rounded_rectangle(
                row,
                radius=8,
                fill=_mix_rgba(
                    _rgba("primary_container", 225),
                    _rgba("primary_hover", 235),
                    hover_amount if active else 0.0,
                ),
            )
        _draw_kind_icon(
            draw,
            str(mount.get("kind") or "drive"),
            58,
            center_y,
            _rgba("text_primary"),
        )
        _text(
            draw,
            (84, center_y),
            _fit_text(
                draw,
                str(mount.get("name") or mount.get("path") or ""),
                21,
                sidebar_right - 120,
            ),
            21,
            fill=_rgba("text_primary"),
            anchor="lm",
        )

    recent = list(state.get("recent_entries") or [])
    draw.text(
        (48, layout["recent_heading_y"]),
        translate("Recent"),
        font=_font(22, bold=True),
        fill=_rgba("text_secondary"),
    )
    if recent:
        for index, entry in enumerate(recent[:layout["recent_count"]]):
            center_y = (
                layout["recent_rows_y"]
                + index * layout["recent_row_height"]
                + layout["recent_row_height"] // 2
            )
            path = str(entry.get("path", ""))
            hovered = hover_action == "recent" and path == hover_path
            selected = path == str(state.get("browser_selection", ""))
            if hovered or selected:
                row = _scaled_rect(
                    (40, center_y - 18, sidebar_right - 14, center_y + 18),
                    hover_amount if hovered else 0.0,
                    scale=_MEDIA_HOVER_ROW_SCALE,
                    lift=0.8,
                )
                draw.rounded_rectangle(
                    row,
                    radius=8,
                    fill=_mix_rgba(
                        _rgba("primary_container", 190),
                        _rgba("primary_hover", 220),
                        hover_amount if hovered else 0.0,
                    ),
                )
            _draw_kind_icon(
                draw,
                str(entry.get("kind") or "video"),
                58,
                center_y,
                _rgba("text_primary"),
            )
            _text(
                draw,
                (84, center_y),
                _fit_text(
                    draw,
                    str(entry.get("name") or Path(path).name or path),
                    21,
                    sidebar_right - 164,
                ),
                21,
                fill=_rgba("text_primary"),
                anchor="lm",
            )
            remove_hover = hover_action == "recent_remove" and path == hover_path
            if remove_hover:
                radius = 16 + int(round(3 * _ease_out_cubic(hover_amount)))
                draw.ellipse(
                    (
                        sidebar_right - 30 - radius,
                        center_y - radius,
                        sidebar_right - 30 + radius,
                        center_y + radius,
                    ),
                    fill=_rgba("primary_hover", 190),
                )
            _draw_close_x(
                draw,
                sidebar_right - 30,
                center_y,
                _rgba("text_primary" if remove_hover else "text_secondary"),
            )
    else:
        _text(
            draw,
            (84, layout["recent_rows_y"] + 18),
            translate("No recent media"),
            21,
            fill=_rgba("text_secondary"),
            anchor="lm",
        )

    playlist = list(state.get("playlist_entries") or [])
    draw.text(
        (48, layout["playlist_heading_y"]),
        translate("Playlist"),
        font=_font(22, bold=True),
        fill=_rgba("text_secondary"),
    )
    playlist_index = int(state.get("playlist_index", -1) if state.get("playlist_index") is not None else -1)
    playlist_offset = layout["playlist_offset"]
    for index, item in enumerate(playlist[playlist_offset:playlist_offset + layout["playlist_count"]]):
        center_y = (
            layout["playlist_rows_y"]
            + index * layout["playlist_row_height"]
            + layout["playlist_row_height"] // 2
        )
        item_path = str(item.get("path", ""))
        row_hovered = hover_action == "playlist_select" and hover_target == item_path
        if index + playlist_offset == playlist_index or row_hovered:
            row = _scaled_rect(
                (40, center_y - 32, sidebar_right - 14, center_y + 32),
                hover_amount if row_hovered else 0.0,
                scale=_MEDIA_HOVER_ROW_SCALE,
                lift=0.8,
            )
            draw.rounded_rectangle(
                row,
                radius=6,
                fill=_mix_rgba(
                    _rgba("primary_container", 190),
                    _rgba("primary_hover", 225),
                    hover_amount if row_hovered else 0.0,
                ),
            )
        _text(
            draw,
            (52, center_y),
            f"{index + playlist_offset + 1:02d}",
            18,
            fill=_rgba("text_secondary"),
            anchor="lm",
        )
        _text(
            draw,
            (82, center_y),
            _fit_text(
                draw,
                Path(item.get("path", "")).name,
                21,
                sidebar_right - 242,
            ),
            21,
            fill=_rgba("text_primary"),
            anchor="lm",
        )

        for action, label, left in (
            ("playlist_move_up", "↑", sidebar_right - 164),
            ("playlist_move_down", "↓", sidebar_right - 112),
            ("playlist_remove", "×", sidebar_right - 60),
        ):
            enabled = action == "playlist_remove" or (
                index + playlist_offset > 0 if action == "playlist_move_up"
                else index + playlist_offset + 1 < len(playlist)
            )
            action_hovered = hover_action == action and hover_target == item_path
            action_rect = _draw_control_surface(
                draw,
                (left, center_y - 24, left + 48, center_y + 24),
                enabled=enabled,
                hovered=action_hovered,
                progress=hover_amount,
                idle_alpha=220,
                disabled_alpha=100,
                radius=OPENXR_MENU_RADII["control"],
            )
            _text(draw, ((action_rect[0] + action_rect[2]) // 2, (action_rect[1] + action_rect[3]) // 2),
                label, 22, fill=_rgba("text_primary", 255 if enabled else 104), anchor="mm")
    for action, label, left in (
        ("playlist_page_previous", "‹", sidebar_right - 112),
        ("playlist_page_next", "›", sidebar_right - 60),
    ):
        y = layout["playlist_heading_y"] - 12
        enabled = media_control_enabled(action, state)
        action_hovered = hover_action == action
        action_rect = _draw_control_surface(
            draw,
            (left, y, left + 48, y + 48),
            enabled=enabled,
            hovered=action_hovered,
            progress=hover_amount,
            idle_alpha=220,
            disabled_alpha=100,
            radius=OPENXR_MENU_RADII["control"],
        )
        _text(draw, ((action_rect[0] + action_rect[2]) // 2, (action_rect[1] + action_rect[3]) // 2),
            label, 22, fill=_rgba("text_primary", 255 if enabled else 104), anchor="mm")

    playlist_rects = rects
    for key, label in (
        ("playlist_previous", translate("Previous")),
        ("playlist_next", translate("Next")),
        ("playlist_clear", translate("Clear")),
        ("playlist_add", translate("Add")),
        ("playlist_sort", translate("Sort Z–A" if state.get("playlist_sort_descending") is False else "Sort A–Z")),
    ):
        enabled = media_control_enabled(key, state)
        action_hovered = hover_action == key
        action_rect = _draw_control_surface(
            draw,
            playlist_rects[key],
            enabled=enabled,
            hovered=action_hovered,
            progress=hover_amount,
            idle_alpha=220,
            disabled_alpha=105,
            radius=8,
        )
        _text(
            draw,
            ((action_rect[0] + action_rect[2]) // 2, (action_rect[1] + action_rect[3]) // 2),
            label,
            21,
            fill=_rgba("text_primary", 255 if enabled else 104),
            anchor="mm",
        )

    open_enabled = media_control_enabled("browse_open", state)
    open_hovered = hover_action == "browse_open"
    open_rect = _draw_control_surface(
        draw,
        playlist_rects["browse_open"],
        enabled=open_enabled,
        hovered=open_hovered,
        progress=hover_amount,
        primary=True,
        idle_alpha=230,
        disabled_alpha=105,
        radius=8,
    )
    _text(
        draw,
        ((open_rect[0] + open_rect[2]) // 2, (open_rect[1] + open_rect[3]) // 2),
        translate("Open"),
        17,
        fill=_rgba("text_primary", 255 if open_enabled else 104),
        anchor="mm",
        bold=True,
    )

    path_text = str(state.get("browser_path") or "")
    header_left = sidebar_right + 34
    draw.text(
        (header_left, 42),
        translate("Files"),
        font=_font(28, bold=True),
        fill=_rgba("text_primary"),
    )
    header_actions_left = rects["browser_close"][0] - 24
    _text(
        draw,
        (header_left, 94),
        _fit_text(draw, path_text, 19, header_actions_left - header_left),
        19,
        fill=_rgba("text_secondary"),
    )
    for key, label in (
        ("browser_close", translate("Close")),
        ("browser_up", translate("Up")),
        ("browser_refresh", translate("Refresh")),
    ):
        enabled = media_control_enabled(key, state)
        action_hovered = hover_action == key
        action_rect = _draw_control_surface(
            draw,
            rects[key],
            enabled=enabled,
            hovered=action_hovered,
            progress=hover_amount,
            idle_alpha=220,
            disabled_alpha=105,
            radius=8,
        )
        _text(
            draw,
            ((action_rect[0] + action_rect[2]) // 2, (action_rect[1] + action_rect[3]) // 2),
            label,
            18,
            fill=_rgba("text_primary", 255 if enabled else 104),
            anchor="mm",
        )
    entries = list(state.get("browser_entries") or [])
    visible_rows = _BROWSER_VISIBLE_ROWS
    selected_path = str(state.get("browser_selection") or "")
    row_top = layout["file_rows_y"]
    row_height = layout["file_row_height"]
    visible = entries[int(scroll_offset): int(scroll_offset) + visible_rows]
    for index, entry in enumerate(visible):
        y0 = row_top + index * row_height
        y1 = y0 + row_height - 4
        path = str(entry.get("path", ""))
        hovered = hover_action == "entry" and path == hover_path
        if path == selected_path or hovered:
            row = _scaled_rect(
                (sidebar_right + 18, y0, width - 38, y1),
                hover_amount if hovered else 0.0,
                scale=_MEDIA_HOVER_ROW_SCALE,
                lift=0.8,
            )
            draw.rounded_rectangle(
                row,
                radius=8,
                fill=_mix_rgba(
                    _rgba("primary_container", 180),
                    _rgba("primary_hover", 220),
                    hover_amount if hovered else 0.0,
                ),
            )
        center_y = (y0 + y1) // 2
        _draw_kind_icon(
            draw,
            str(entry.get("kind") or "other"),
            sidebar_right + 42,
            center_y,
            _rgba("text_primary"),
        )
        _text(
            draw,
            (sidebar_right + 76, center_y),
            _fit_text(
                draw,
                str(entry.get("name") or path),
                21,
                width - sidebar_right - 204,
            ),
            21,
            fill=_rgba("text_primary"),
            anchor="lm",
        )
        if entry.get("kind") in {"video", "image", "disc"}:
            left = width - 94
            add_hovered = hover_action == "playlist_add" and hover_target == path
            add_rect = _draw_control_surface(
                draw,
                (left, center_y - 24, left + 48, center_y + 24),
                enabled=True,
                hovered=add_hovered,
                progress=hover_amount,
                idle_alpha=220,
                disabled_alpha=105,
                radius=OPENXR_MENU_RADII["control"],
            )
            _text(
                draw,
                ((add_rect[0] + add_rect[2]) // 2, (add_rect[1] + add_rect[3]) // 2),
                "+",
                24,
                fill=_rgba("text_primary"),
                anchor="mm",
            )
    if not entries:
        _text(
            draw,
            ((sidebar_right + width) // 2, height // 2),
            translate("No playable media in this folder"),
            22,
            fill=_rgba("text_secondary"),
            anchor="mm",
        )
    error = str(state.get("browser_error") or state.get("error") or "")
    if error:
        _text(
            draw,
            (sidebar_right + 36, height - 88),
            _fit_text(draw, error, 16, width - sidebar_right - 190),
            16,
            fill=_rgba("destructive_outline"),
        )
    open_x0, open_y0, _open_x1, _open_y1 = rects["browse_open"]
    _text(
        draw,
        (open_x0 - 18, open_y0 + 28),
        translate("Click to select · Double-click to open"),
        18,
        fill=_rgba("text_secondary"),
        anchor="rm",
    )
    return np.asarray(image, dtype=np.uint8)


def media_browser_hit(
    u: float,
    v: float,
    state: dict,
    *,
    scroll_offset: int = 0,
) -> tuple[str, object] | None:
    width, height = _MEDIA_BROWSER_SIZE
    x, y = float(u) * width, float(v) * height
    rects = _browser_button_rects(width, height)
    for key in (
        "browser_up",
        "browser_refresh",
        "browser_close",
        "browse_open",
        "playlist_previous",
        "playlist_next",
        "playlist_clear",
        "playlist_add",
        "playlist_sort",
    ):
        x0, y0, x1, y1 = rects[key]
        if x0 <= x <= x1 and y0 <= y <= y1:
            if key in {"browse_open", "playlist_add"}:
                selected = str(state.get("browser_selection") or "")
                return (
                    (key, selected)
                    if media_control_enabled(key, state)
                    else None
                )
            return (key, None) if media_control_enabled(key, state) else None
    layout = _browser_layout(state, width, height)
    sidebar_right = layout["sidebar_right"]
    playlist = list(state.get("playlist_entries") or [])
    mounts = list(state.get("browser_mounts") or [])
    recent = list(state.get("recent_entries") or [])
    if 28 <= x <= sidebar_right - 14:
        for index, mount in enumerate(mounts[:layout["mount_count"]]):
            center_y = layout["mount_top"] + index * layout["mount_row_height"] + layout["mount_row_height"] // 2
            if abs(y - center_y) <= layout["mount_row_height"] // 2:
                return "mount", mount.get("path")
        for index, entry in enumerate(recent[:layout["recent_count"]]):
            center_y = (
                layout["recent_rows_y"]
                + index * layout["recent_row_height"]
                + layout["recent_row_height"] // 2
            )
            if abs(y - center_y) <= layout["recent_row_height"] // 2:
                path = str(entry.get("path") or "")
                if x >= sidebar_right - 56:
                    return (
                        ("recent_remove", path)
                        if media_control_enabled("recent_remove", state)
                        else None
                    )
                return (
                    ("recent", path)
                    if media_control_enabled("recent", state)
                    else None
                )
        for action, left in (("playlist_page_previous", sidebar_right - 112),
                             ("playlist_page_next", sidebar_right - 60)):
            if left <= x <= left + 48 and layout["playlist_heading_y"] - 12 <= y <= layout["playlist_heading_y"] + 36:
                return (action, None) if media_control_enabled(action, state) else None
        offset = layout["playlist_offset"]
        for index, item in enumerate(playlist[offset:offset + layout["playlist_count"]]):
            center_y = (
                layout["playlist_rows_y"]
                + index * layout["playlist_row_height"]
                + layout["playlist_row_height"] // 2
            )
            if abs(y - center_y) <= 24:
                for action, left in (("playlist_move_up", sidebar_right - 164),
                                     ("playlist_move_down", sidebar_right - 112),
                                     ("playlist_remove", sidebar_right - 60)):
                    if left <= x <= left + 48:
                        enabled = action == "playlist_remove" or (
                            index + offset > 0 if action == "playlist_move_up" else index + offset + 1 < len(playlist)
                        )
                        return (action, item.get("path")) if enabled else None
                return (
                    ("playlist_select", item.get("path"))
                    if media_control_enabled("playlist_select", state)
                    else None
                )
    row_top = layout["file_rows_y"]
    row_height = layout["file_row_height"]
    row_bottom = row_top + _BROWSER_VISIBLE_ROWS * row_height
    if sidebar_right + 18 <= x <= width - 38 and row_top <= y < row_bottom:
        index = int((y - row_top) // row_height) + int(scroll_offset)
        entries = list(state.get("browser_entries") or [])
        if 0 <= index < len(entries):
            if width - 94 <= x <= width - 46 and entries[index].get("kind") in {"video", "image", "disc"}:
                center_y = row_top + (index - int(scroll_offset)) * row_height + (row_height - 4) // 2
                if abs(y - center_y) <= 24:
                    return "playlist_add", entries[index].get("path")
            return ("entry", entries[index].get("path"))
    return None
