from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np
from gui.localization import gettext_for, normalize_locale
from xr_viewer.overlay_textures import load_overlay_font_for_text
from xr_viewer.settings_menu import OPENXR_MENU_COLORS, OPENXR_MENU_RADII

_FONT_PATH = Path(__file__).resolve().parent / "xr_viewer" / "fonts" / "InterVariable.ttf"
MEDIA_BAR_LAYOUT_SIZE = (1024, 160)
MEDIA_BAR_TEXTURE_SIZE = (2048, 320)
_MEDIA_BAR_SIZE = MEDIA_BAR_LAYOUT_SIZE
MEDIA_BROWSER_LAYOUT_SIZE = (1600, 1200)
MEDIA_BROWSER_TEXTURE_SIZE = (2048, 1536)
MEDIA_BROWSER_TEXTURE_ASPECT = (
    MEDIA_BROWSER_TEXTURE_SIZE[0] / MEDIA_BROWSER_TEXTURE_SIZE[1]
)
MEDIA_BROWSER_XR_MAX_WIDTH_FRACTION = 0.94
MEDIA_BROWSER_XR_MAX_HEIGHT_FRACTION = 0.90
MEDIA_BROWSER_FONT_SCALE_MIN = 1.0
MEDIA_BROWSER_FONT_SCALE_MAX = 2.6
# A 705px panel is the 1080p Half-SBS eye layout; keep its text at 1.6x.
MEDIA_BROWSER_FONT_BOOST = 0.94
_MEDIA_BROWSER_SIZE = MEDIA_BROWSER_LAYOUT_SIZE
MEDIA_BROWSER_VISIBLE_ROWS = 10
_BROWSER_VISIBLE_ROWS = MEDIA_BROWSER_VISIBLE_ROWS
_BROWSER_GRID_COLUMNS = 3
_BROWSER_GRID_ROWS = 3
_BROWSER_GRID_GAP = 18
_BROWSER_THUMBNAIL_SIZE = (384, 216)
_BROWSER_THUMBNAIL_CACHE_ITEMS = 32
_BROWSER_THUMBNAIL_MAX_PENDING = 8
MEDIA_HOVER_DURATION_S = 0.16
MEDIA_POINTER_IDLE_SECONDS = 3.0
_MEDIA_HOVER_BUTTON_SCALE = 0.035
_MEDIA_HOVER_ROW_SCALE = 0.010


def media_browser_font_scale(panel_height: int | float) -> float:
    """Keep browser text legible as the GPU panel is fit into each eye."""
    try:
        height = max(1.0, float(panel_height))
    except (TypeError, ValueError):
        height = float(MEDIA_BROWSER_LAYOUT_SIZE[1])
    scale = max(
        MEDIA_BROWSER_FONT_SCALE_MIN,
        MEDIA_BROWSER_LAYOUT_SIZE[1] / height,
    )
    return max(
        MEDIA_BROWSER_FONT_SCALE_MIN,
        min(MEDIA_BROWSER_FONT_SCALE_MAX, scale * MEDIA_BROWSER_FONT_BOOST),
    )


def media_browser_font_scale_for_eye(eye_width: int, eye_height: int) -> float:
    """Scale text for a browser quad fitted into one XR eye viewport."""
    width = max(1.0, float(eye_width))
    height = max(1.0, float(eye_height))
    panel_width = min(
        width * MEDIA_BROWSER_XR_MAX_WIDTH_FRACTION,
        height * MEDIA_BROWSER_XR_MAX_HEIGHT_FRACTION * MEDIA_BROWSER_TEXTURE_ASPECT,
    )
    return media_browser_font_scale(panel_width / MEDIA_BROWSER_TEXTURE_ASPECT)


class _ScaledImageDraw:
    """Draw logical browser coordinates onto a denser texture."""

    def __init__(self, draw, scale_x: float, scale_y: float) -> None:
        self._draw = draw
        self._scale_x = float(scale_x)
        self._scale_y = float(scale_y)
        self._stroke_scale = min(self._scale_x, self._scale_y)

    def _point(self, point):
        return (
            round(float(point[0]) * self._scale_x),
            round(float(point[1]) * self._scale_y),
        )

    def _rect(self, rect):
        left, top, right, bottom = rect
        return (
            round(float(left) * self._scale_x),
            round(float(top) * self._scale_y),
            round(float(right) * self._scale_x),
            round(float(bottom) * self._scale_y),
        )

    def rounded_rectangle(self, xy, **kwargs):
        kwargs["radius"] = max(
            1, round(float(kwargs.get("radius", 0)) * self._stroke_scale)
        )
        kwargs["width"] = max(
            1, round(float(kwargs.get("width", 1)) * self._stroke_scale)
        )
        return self._draw.rounded_rectangle(self._rect(xy), **kwargs)

    def rectangle(self, xy, **kwargs):
        kwargs["width"] = max(
            1, round(float(kwargs.get("width", 1)) * self._stroke_scale)
        )
        return self._draw.rectangle(self._rect(xy), **kwargs)

    def ellipse(self, xy, **kwargs):
        if "width" in kwargs:
            kwargs["width"] = max(
                1, round(float(kwargs["width"]) * self._stroke_scale)
            )
        return self._draw.ellipse(self._rect(xy), **kwargs)

    def arc(self, xy, start, end, **kwargs):
        if "width" in kwargs:
            kwargs["width"] = max(
                1, round(float(kwargs["width"]) * self._stroke_scale)
            )
        return self._draw.arc(self._rect(xy), start, end, **kwargs)

    def line(self, xy, **kwargs):
        if xy and isinstance(xy[0], (tuple, list)):
            points = [self._point(point) for point in xy]
        else:
            points = [
                coordinate
                for index in range(0, len(xy), 2)
                for coordinate in self._point((xy[index], xy[index + 1]))
            ]
        kwargs["width"] = max(
            1, round(float(kwargs.get("width", 1)) * self._stroke_scale)
        )
        return self._draw.line(points, **kwargs)

    def polygon(self, xy, **kwargs):
        return self._draw.polygon([self._point(point) for point in xy], **kwargs)

    def text(self, xy, text, **kwargs):
        return self._draw.text(self._point(xy), text, **kwargs)

    def textbbox(self, xy, text, **kwargs):
        return self._draw.textbbox(self._point(xy), text, **kwargs)

    def textlength(self, text, **kwargs):
        return self._draw.textlength(text, **kwargs)


def _browser_layout_dimensions(width: int, height: int) -> tuple[int, int]:
    """Use logical coordinates for the shared 2K browser texture."""
    if (int(width), int(height)) == MEDIA_BROWSER_TEXTURE_SIZE:
        return MEDIA_BROWSER_LAYOUT_SIZE
    return int(width), int(height)


def _media_bar_layout_dimensions(width: int, height: int) -> tuple[int, int]:
    if (int(width), int(height)) == MEDIA_BAR_TEXTURE_SIZE:
        return MEDIA_BAR_LAYOUT_SIZE
    return int(width), int(height)


def _load_media_browser_thumbnail(
    path: str,
    kind: str,
    size: tuple[int, int] = _BROWSER_THUMBNAIL_SIZE,
) -> np.ndarray | None:
    """Decode one small preview on a worker thread; unsupported files use icons."""
    try:
        if kind == "image":
            from PIL import Image, ImageOps

            with Image.open(path) as source:
                if source.width * source.height > 80_000_000:
                    return None
                source.draft("RGB", size)
                preview = ImageOps.exif_transpose(source)
                preview.thumbnail(size, Image.Resampling.LANCZOS)
                return np.asarray(preview.convert("RGB"), dtype=np.uint8).copy()

        if kind == "video":
            import cv2
            from PIL import Image

            timeout_key = getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None)
            read_timeout_key = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)
            if timeout_key is not None and read_timeout_key is not None:
                capture = cv2.VideoCapture(
                    path,
                    cv2.CAP_FFMPEG,
                    [int(timeout_key), 2500, int(read_timeout_key), 2500],
                )
            else:
                capture = cv2.VideoCapture(path)
            try:
                if not capture.isOpened():
                    return None
                ok, frame = capture.read()
                if not ok or frame is None:
                    return None
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                preview = Image.fromarray(frame, mode="RGB")
                preview.thumbnail(size, Image.Resampling.LANCZOS)
                return np.asarray(preview, dtype=np.uint8).copy()
            finally:
                capture.release()
    except Exception:
        return None
    return None


class MediaBrowserThumbnailCache:
    """Bounded, asynchronous thumbnail cache for the OpenXR browser surface."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="d2s-media-thumbnail"
        )
        self._cache: OrderedDict[str, np.ndarray | None] = OrderedDict()
        self._pending: dict[str, Future[np.ndarray | None]] = {}
        self._revision = 0
        self._closed = False
        self._lock = Lock()

    def visible_thumbnails(
        self,
        entries: list[dict],
        *,
        scroll_offset: int,
        view_mode: str,
    ) -> tuple[dict[str, np.ndarray], int]:
        state = {"browser_entries": entries, "browser_view_mode": view_mode}
        layout = _browser_layout(state, *MEDIA_BROWSER_TEXTURE_SIZE)
        visible_count = layout["visible_entry_count"]
        offset = max(0, min(int(scroll_offset), max(0, len(entries) - visible_count)))
        visible = entries[offset : offset + visible_count]
        candidates = visible + entries[
            offset + visible_count : offset + visible_count + layout["scroll_step"]
        ]

        with self._lock:
            if self._closed:
                return {}, self._revision
            for path, future in tuple(self._pending.items()):
                if not future.done():
                    continue
                self._pending.pop(path, None)
                try:
                    thumbnail = future.result()
                except Exception:
                    thumbnail = None
                self._cache[path] = thumbnail
                self._cache.move_to_end(path)
                while len(self._cache) > _BROWSER_THUMBNAIL_CACHE_ITEMS:
                    self._cache.popitem(last=False)
                self._revision += 1

            for entry in candidates:
                path = str(entry.get("path") or "")
                kind = str(entry.get("kind") or "")
                if not path or kind not in {"image", "video"}:
                    continue
                if path in self._cache:
                    self._cache.move_to_end(path)
                    continue
                if path in self._pending:
                    continue
                if len(self._pending) >= _BROWSER_THUMBNAIL_MAX_PENDING:
                    break
                self._pending[path] = self._executor.submit(
                    _load_media_browser_thumbnail, path, kind
                )

            return (
                {path: self._cache[path] for path in (str(item.get("path") or "") for item in visible)
                 if path in self._cache and self._cache[path] is not None},
                self._revision,
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)


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
    hover_alpha: int = 230,
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
    hover = _rgba("primary_hover", hover_alpha if enabled else disabled_alpha)
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


def _font(size: int, *, text: str = "", bold: bool = False):
    try:
        return load_overlay_font_for_text(
            int(size),
            str(text or ""),
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
        font=_font(size, text=str(text), bold=bold),
        fill=fill or _rgba("text_primary"),
        anchor=anchor,
    )


def _fit_text(draw, text: str, size: int, max_width: float, *, bold: bool = False) -> str:
    value = str(text or "")
    if max_width <= 0:
        return ""
    font = _font(size, text=value, bold=bold)
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


def _draw_browser_preview_icon(draw, kind: str, cx: int, cy: int, color) -> None:
    if kind == "directory":
        draw.rounded_rectangle(
            (cx - 40, cy - 22, cx + 40, cy + 30),
            radius=8,
            outline=color,
            width=5,
        )
        draw.line((cx - 38, cy - 22, cx - 20, cy - 38, cx + 4, cy - 38), fill=color, width=5)
        draw.line((cx + 4, cy - 38, cx + 16, cy - 22), fill=color, width=5)
    elif kind == "video":
        draw.rounded_rectangle(
            (cx - 42, cy - 30, cx + 42, cy + 30),
            radius=8,
            outline=color,
            width=5,
        )
        draw.polygon(
            ((cx - 10, cy - 18), (cx + 20, cy), (cx - 10, cy + 18)),
            fill=color,
        )
    elif kind == "image":
        draw.rounded_rectangle(
            (cx - 42, cy - 30, cx + 42, cy + 30),
            radius=8,
            outline=color,
            width=5,
        )
        draw.ellipse((cx + 12, cy - 20, cx + 26, cy - 6), fill=color)
        draw.line(
            (cx - 30, cy + 20, cx - 4, cy - 6, cx + 14, cy + 14, cx + 30, cy - 2),
            fill=color,
            width=5,
        )
    else:
        draw.rounded_rectangle(
            (cx - 32, cy - 40, cx + 32, cy + 40),
            radius=7,
            outline=color,
            width=5,
        )
        draw.line((cx - 18, cy - 10, cx + 18, cy - 10), fill=color, width=4)
        draw.line((cx - 18, cy + 4, cx + 18, cy + 4), fill=color, width=4)
        draw.line((cx - 18, cy + 18, cx + 10, cy + 18), fill=color, width=4)


def _paste_browser_thumbnail(
    image,
    draw,
    rect: tuple[int, int, int, int],
    thumbnail: np.ndarray | None,
    kind: str,
    *,
    radius: int,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> None:
    from PIL import Image, ImageDraw as PillowImageDraw, ImageOps

    left, top, right, bottom = rect
    pixel_left = round(left * scale_x)
    pixel_top = round(top * scale_y)
    width = max(1, round((right - left) * scale_x))
    height = max(1, round((bottom - top) * scale_y))
    pixel_radius = max(1, round(radius * min(scale_x, scale_y)))
    draw.rounded_rectangle(
        rect,
        radius=radius,
        fill=_rgba("surface_container_high", 230),
    )
    if thumbnail is not None:
        try:
            source = Image.fromarray(np.asarray(thumbnail, dtype=np.uint8)).convert("RGB")
            resampling = getattr(Image, "Resampling", Image).LANCZOS
            source = ImageOps.fit(source, (width, height), method=resampling)
            mask = Image.new("L", (width, height), 0)
            PillowImageDraw.Draw(mask).rounded_rectangle(
                (0, 0, width - 1, height - 1), radius=pixel_radius, fill=255
            )
            image.paste(source, (pixel_left, pixel_top), mask)
            draw.rounded_rectangle(
                rect,
                radius=radius,
                outline=_rgba("outline", 92),
                width=1,
            )
            return
        except Exception:
            pass
    _draw_browser_preview_icon(
        draw,
        kind,
        (left + right) // 2,
        (top + bottom) // 2,
        _rgba("text_secondary"),
    )


def _draw_browser_view_icon(draw, rect, view_mode: str, color) -> None:
    left, top, right, bottom = rect
    cx, cy = (left + right) // 2, (top + bottom) // 2
    if view_mode == "grid":
        for row in range(2):
            for column in range(2):
                x = cx - 12 + column * 14
                y = cy - 12 + row * 14
                draw.rounded_rectangle((x, y, x + 10, y + 10), radius=2, fill=color)
        return
    for offset in (-10, 0, 10):
        draw.ellipse((cx - 14, cy + offset - 2, cx - 10, cy + offset + 2), fill=color)
        draw.rounded_rectangle(
            (cx - 5, cy + offset - 2, cx + 14, cy + offset + 2),
            radius=2,
            fill=color,
        )


def _draw_browser_panel_background(image, draw, rect) -> None:
    from PIL import Image, ImageDraw

    top_rgb = np.asarray(_rgba("surface_header")[:3], dtype=np.float32)
    bottom_rgb = np.asarray(_rgba("surface")[:3], dtype=np.float32)
    blend = np.linspace(0.0, 1.0, image.height, dtype=np.float32)[:, None, None]
    rows = np.rint(top_rgb[None, None, :] * (1.0 - blend) + bottom_rgb[None, None, :] * blend)
    pixels = np.empty((image.height, image.width, 4), dtype=np.uint8)
    pixels[:, :, :3] = np.repeat(rows.astype(np.uint8), image.width, axis=1)
    pixels[:, :, 3] = 238
    gradient = Image.fromarray(pixels, mode="RGBA")
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        rect,
        radius=OPENXR_MENU_RADII["panel"],
        fill=255,
    )
    image.paste(gradient, (0, 0), mask)
    draw.rounded_rectangle(
        rect,
        radius=OPENXR_MENU_RADII["panel"],
        outline=_rgba("outline", 52),
        width=1,
    )


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
    if action == "viewer_stop":
        return True
    if action in {"browse", "browse_disc", "browser_close"}:
        return ready
    if action == "media_exit":
        return ready and mode != "Browser"
    if action == "browser_up":
        return ready and bool(state.get("browser_parent"))
    if action == "browser_refresh":
        return ready and bool(state.get("browser_path"))
    if action == "browser_sort":
        return ready and len(state.get("browser_entries") or []) > 1
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
    return (24, track_y - 10, width - 78, track_y + 16)


def _browser_layout(state: dict, width: int, height: int) -> dict[str, int | str]:
    """Keep browser painting and hit testing on one shared geometry."""
    width, height = _browser_layout_dimensions(width, height)
    mounts = list(state.get("browser_mounts") or [])
    recent = list(state.get("recent_entries") or [])
    playlist = list(state.get("playlist_entries") or [])
    sidebar_right = max(480, min(520, int(width * 0.31)))
    mount_count = min(4, len(mounts))
    recent_count = min(2, len(recent))
    playlist_offset = max(0, min(int(state.get("playlist_offset", 0) or 0), max(0, len(playlist) - 1)))
    playlist_count = min(4, len(playlist) - playlist_offset)
    locations_heading_y = max(
        132,
        max(32, int(width * 0.018)) + 64 + 20,
    )
    mount_top = max(150, int(height * 0.125), locations_heading_y + 38)
    mount_row_height = 72
    recent_heading_y = mount_top + mount_count * mount_row_height + 34
    recent_rows_y = recent_heading_y + 36
    recent_row_height = 72
    playlist_heading_y = (
        recent_rows_y + max(1, recent_count) * recent_row_height + 34
    )
    playlist_rows_y = playlist_heading_y + 36
    playlist_row_height = 72
    view_mode = str(state.get("browser_view_mode") or "grid").casefold()
    if view_mode not in {"grid", "list"}:
        view_mode = "grid"
    file_rows_y = max(168, int(height * 0.14))
    file_row_height = max(84, (height - file_rows_y - 120) // _BROWSER_VISIBLE_ROWS)
    grid_left = sidebar_right + 18
    grid_right = width - 48
    grid_width = max(_BROWSER_GRID_COLUMNS, grid_right - grid_left)
    grid_card_width = (
        grid_width - _BROWSER_GRID_GAP * (_BROWSER_GRID_COLUMNS - 1)
    ) // _BROWSER_GRID_COLUMNS
    grid_bottom = max(file_rows_y + 1, height - 132)
    grid_card_height = max(
        220,
        (grid_bottom - file_rows_y - _BROWSER_GRID_GAP * (_BROWSER_GRID_ROWS - 1))
        // _BROWSER_GRID_ROWS,
    )
    grid_preview_height = min(grid_card_height - 96, (grid_card_width - 24) * 9 // 16)
    visible_entry_count = (
        _BROWSER_GRID_COLUMNS * _BROWSER_GRID_ROWS
        if view_mode == "grid"
        else _BROWSER_VISIBLE_ROWS
    )
    return {
        "sidebar_right": sidebar_right,
        "locations_heading_y": locations_heading_y,
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
        "file_row_height": file_row_height,
        "view_mode": view_mode,
        "visible_entry_count": visible_entry_count,
        "scroll_step": _BROWSER_GRID_COLUMNS if view_mode == "grid" else 1,
        "grid_columns": _BROWSER_GRID_COLUMNS,
        "grid_rows": _BROWSER_GRID_ROWS,
        "grid_gap": _BROWSER_GRID_GAP,
        "grid_left": grid_left,
        "grid_top": file_rows_y,
        "grid_card_width": grid_card_width,
        "grid_card_height": grid_card_height,
        "grid_preview_height": grid_preview_height,
        "scrollbar_left": width - 44,
        "scrollbar_right": width - 16,
        "scrollbar_top": file_rows_y,
        "scrollbar_bottom": max(file_rows_y + 1, height - 132),
        "file_content_right": width - 48,
    }


def media_browser_scroll_metrics(
    state: dict,
    *,
    width: int = MEDIA_BROWSER_TEXTURE_SIZE[0],
    height: int = MEDIA_BROWSER_TEXTURE_SIZE[1],
) -> tuple[int, int, int]:
    """Return the current scroll step, visible count, and maximum item offset."""
    layout = _browser_layout(state, width, height)
    entries = list(state.get("browser_entries") or [])
    visible_count = layout["visible_entry_count"]
    if layout["view_mode"] == "grid":
        total_rows = (len(entries) + layout["grid_columns"] - 1) // layout["grid_columns"]
        max_offset = max(0, total_rows - layout["grid_rows"]) * layout["grid_columns"]
    else:
        max_offset = max(0, len(entries) - visible_count)
    return (
        layout["scroll_step"],
        visible_count,
        max_offset,
    )


def media_browser_scrollbar_metrics(
    state: dict,
    scroll_offset: int = 0,
    *,
    width: int = MEDIA_BROWSER_TEXTURE_SIZE[0],
    height: int = MEDIA_BROWSER_TEXTURE_SIZE[1],
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int] | None]:
    """Return scrollbar geometry in logical browser pixels."""
    layout = _browser_layout(state, width, height)
    _step, visible_count, max_offset = media_browser_scroll_metrics(
        state, width=width, height=height
    )
    track = (
        layout["scrollbar_left"],
        layout["scrollbar_top"],
        layout["scrollbar_right"],
        layout["scrollbar_bottom"],
    )
    entries = list(state.get("browser_entries") or [])
    if max_offset <= 0 or not entries:
        return track, None
    track_height = max(1, track[3] - track[1])
    thumb_height = min(
        track_height,
        max(52, int(round(track_height * min(1.0, visible_count / len(entries))))),
    )
    travel = max(0, track_height - thumb_height)
    offset = max(0, min(int(scroll_offset), max_offset))
    thumb_top = track[1] + int(round(travel * offset / max_offset))
    return track, (track[0], thumb_top, track[2], thumb_top + thumb_height)


def media_browser_scrollbar_offset(
    state: dict,
    y: float,
    *,
    width: int = MEDIA_BROWSER_TEXTURE_SIZE[0],
    height: int = MEDIA_BROWSER_TEXTURE_SIZE[1],
) -> int:
    """Map a pointer y coordinate to the nearest valid browser offset."""
    track, thumb = media_browser_scrollbar_metrics(
        state, width=width, height=height
    )
    if thumb is None:
        return 0
    step, _visible, max_offset = media_browser_scroll_metrics(
        state, width=width, height=height
    )
    thumb_height = thumb[3] - thumb[1]
    travel = max(1, (track[3] - track[1]) - thumb_height)
    center = max(
        track[1] + thumb_height / 2,
        min(track[3] - thumb_height / 2, float(y)),
    )
    fraction = (center - track[1] - thumb_height / 2) / travel
    aligned_step = max(1, step)
    offset = int(round(fraction * max_offset / aligned_step)) * aligned_step
    return max(0, min(max_offset, offset))


def _browser_grid_card_rect(
    layout: dict[str, int | str], index: int
) -> tuple[int, int, int, int]:
    row, column = divmod(index, layout["grid_columns"])
    left = layout["grid_left"] + column * (
        layout["grid_card_width"] + layout["grid_gap"]
    )
    top = layout["grid_top"] + row * (
        layout["grid_card_height"] + layout["grid_gap"]
    )
    return (
        left,
        top,
        left + layout["grid_card_width"],
        top + layout["grid_card_height"],
    )


def _browser_grid_add_rect(card: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return (card[2] - 56, card[3] - 56, card[2] - 8, card[3] - 8)


def _browser_list_add_rect(center_y: int, width: int) -> tuple[int, int, int, int]:
    return (width - 94, center_y - 24, width - 46, center_y + 24)


def _browser_button_rects(
    width: int, height: int
) -> dict[str, tuple[int, int, int, int]]:
    """Return MetaXR-sized browser actions with at least 48 px targets."""
    width, height = _browser_layout_dimensions(width, height)
    top_margin = max(32, int(width * 0.018))
    footer_margin = max(48, int(width * 0.030))
    top_gap = max(10, int(width * 0.008))
    footer_gap = max(24, int(width * 0.015))
    button_w = 168
    footer_button_w = 112
    button_h = 64
    footer_y = height - footer_margin - button_h
    refresh_x = width - top_margin - button_w
    up_x = refresh_x - top_gap - button_w
    close_x = up_x - top_gap - button_w
    view_size = 56
    view_gap = 8
    view_list_x = close_x - view_gap - view_size
    view_grid_x = view_list_x - view_gap - view_size
    sort_w = 96
    sort_x = view_grid_x - top_gap - sort_w
    previous_x = footer_margin
    next_x = previous_x + footer_button_w + footer_gap
    clear_x = next_x + footer_button_w + footer_gap
    return {
        "viewer_stop": (
            top_margin,
            top_margin,
            top_margin + button_w,
            top_margin + button_h,
        ),
        "browser_view_grid": (
            view_grid_x,
            top_margin,
            view_grid_x + view_size,
            top_margin + view_size,
        ),
        "browser_view_list": (
            view_list_x,
            top_margin,
            view_list_x + view_size,
            top_margin + view_size,
        ),
        "browser_sort": (
            sort_x,
            top_margin,
            sort_x + sort_w,
            top_margin + button_h,
        ),
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
        "media_exit": (width - 60, 12, width - 12, 60),
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
    width: int = MEDIA_BAR_TEXTURE_SIZE[0],
    height: int = MEDIA_BAR_TEXTURE_SIZE[1],
    hover: str | None = None,
    hover_progress: float = 1.0,
    alpha: int = 188,
    lang: str = "EN",
    collapsed: bool = False,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    def translate(message):
        return gettext_for(normalize_locale(lang), message)

    render_width, render_height = int(width), int(height)
    width, height = _media_bar_layout_dimensions(render_width, render_height)
    scale_x = render_width / max(1, width)
    scale_y = render_height / max(1, height)
    raster_scale = min(scale_x, scale_y)
    image = Image.new("RGBA", (render_width, render_height), (0, 0, 0, 0))
    draw = _ScaledImageDraw(ImageDraw.Draw(image), scale_x, scale_y)

    def bar_text(draw, xy, text, size=20, *, fill=None, bold=False, anchor="la"):
        _text(
            draw,
            xy,
            text,
            max(1, round(float(size) * raster_scale)),
            fill=fill,
            bold=bold,
            anchor=anchor,
        )
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
    track_left, track_right = 24, width - 78
    track_y = max(26, int(height * 0.20))
    # Keep endpoint thumb/glow inside a label gutter so hover scaling never
    # crowds the elapsed/duration timecodes at either end of the timeline.
    time_label_inset = 22
    seek_enabled = media_control_enabled("seek", state)
    bar_text(draw, (track_left + time_label_inset, 10), _format_time(position), 18, fill=_rgba("text_secondary"))
    bar_text(
        draw,
        (track_right - time_label_inset, 10),
        _format_time(duration),
        18,
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
        color = _rgba("primary_state_text" if enabled else "text_secondary")
        if key == "play":
            draw_rect = _scaled_rect(
                source,
                progress,
                scale=_MEDIA_HOVER_BUTTON_SCALE,
                lift=1.5,
            )
            x0, y0, x1, y1 = draw_rect
            ring = _mix_rgba(
                _rgba("primary" if enabled else "surface_container_high"),
                _rgba("primary_hover"),
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
                fill=_rgba("surface_container_high"),
            )
        else:
            draw_rect = _draw_control_surface(
                draw,
                source,
                enabled=enabled,
                hovered=active,
                progress=progress,
                idle_alpha=255,
                disabled_alpha=255,
                hover_alpha=255,
                radius=OPENXR_MENU_RADII["control"],
            )
            x0, y0, x1, y1 = draw_rect
        x0, y0, x1, y1 = draw_rect
        center_y = (y0 + y1) // 2
        if icon_kind == "file":
            _draw_file_icon(draw, x0 + 22, center_y, color)
            bar_text(draw, (x0 + 44, center_y), label, 16, fill=color, bold=True, anchor="lm")
        elif icon_kind == "disc":
            _draw_disc_icon(draw, x0 + 22, center_y, color)
            bar_text(draw, (x0 + 44, center_y), label, 16, fill=color, bold=True, anchor="lm")
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
            bar_text(draw, ((x0 + x1) // 2, center_y), icon, 25, fill=color, anchor="mm")
        elif label:
            size = 20 if (x1 - x0) >= 90 else 18
            bar_text(
                draw,
                ((x0 + x1) // 2, (y0 + y1) // 2),
                label,
                size,
                fill=color,
                anchor="mm",
                bold=True,
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
        idle_alpha=255,
        disabled_alpha=255,
        hover_alpha=255,
        radius=OPENXR_MENU_RADII["control"],
    )
    center_y = (y0 + y1) // 2
    track_height = 8 + int(round(2 * _ease_out_cubic(volume_progress)))
    track_x0, track_x1 = x0 + 12, x1 - 12
    draw.rounded_rectangle(
        (track_x0, center_y - track_height // 2, track_x1, center_y + track_height // 2),
        radius=max(4, track_height // 2),
        fill=_mix_rgba(
            _rgba("media_volume_track", 220 if volume_enabled else 105),
            _rgba("media_volume_track", 255),
            volume_progress,
        ),
    )
    vx = int(track_x0 + (track_x1 - track_x0) * volume)
    knob_radius = 7 + int(round(2 * _ease_out_cubic(volume_progress)))
    if volume_enabled:
        draw.rounded_rectangle(
            (track_x0, center_y - track_height // 2, vx, center_y + track_height // 2),
            radius=max(4, track_height // 2),
            fill=_mix_rgba(
                _rgba("media_volume_fill"),
                _rgba("media_volume_fill_hover"),
                volume_progress,
            ),
        )
    draw.ellipse(
        (vx - knob_radius, center_y - knob_radius, vx + knob_radius, center_y + knob_radius),
        fill=_rgba(
            "media_volume_fill_hover" if volume_active else "media_volume_fill",
            255 if volume_enabled else 104,
        ),
        outline=_rgba("primary_hover", int(255 * volume_progress)) if volume_active else None,
        width=2 if volume_active else 1,
    )
    exit_enabled = media_control_enabled("media_exit", state)
    exit_hovered = bool(exit_enabled and hover_action == "media_exit")
    exit_progress = hover_amount if exit_hovered else 0.0
    exit_rect = rects["media_exit"]
    draw.rounded_rectangle(
        exit_rect,
        radius=OPENXR_MENU_RADII["control"],
        fill=_rgba("destructive", 255 if exit_enabled else 96),
        outline=_rgba("destructive_hover", int(220 * exit_progress)) if exit_hovered else None,
        width=2 if exit_hovered else 1,
    )
    center_x = (exit_rect[0] + exit_rect[2]) // 2
    center_y = (exit_rect[1] + exit_rect[3]) // 2
    arm = 9
    draw.line(
        (center_x - arm, center_y - arm, center_x + arm, center_y + arm),
        fill=_rgba("destructive_text", 255 if exit_enabled else 140),
        width=4,
    )
    draw.line(
        (center_x + arm, center_y - arm, center_x - arm, center_y + arm),
        fill=_rgba("destructive_text", 255 if exit_enabled else 140),
        width=4,
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
    exit_left, exit_top, exit_right, exit_bottom = _button_rects(width, height)["media_exit"]
    if exit_left <= x <= exit_right and exit_top <= y <= exit_bottom:
        return "media_exit" if media_control_enabled("media_exit", state) else None
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
        start, end = 24.0, float(width - 78)
    elif action == "volume":
        start, _, end, _ = _button_rects(width, height)["volume"]
        start, end = float(start + 12), float(end - 12)
    else:
        return None
    return max(0.0, min(1.0, (x - start) / max(1.0, end - start)))


def media_bar_rect_for_eye(
    eye_rect: tuple[int, int, int, int], display_mode: str = ""
) -> tuple[int, int, int, int]:
    """Return the shared playback bar bounds for one eye in drawable pixels."""
    left, top, right, bottom = (int(value) for value in eye_rect)
    eye_width = max(1, right - left)
    eye_height = max(1, bottom - top)
    panel_width = min(int(eye_width * 0.92), _MEDIA_BAR_SIZE[0])
    panel_height = max(
        70,
        int(panel_width * _MEDIA_BAR_SIZE[1] / _MEDIA_BAR_SIZE[0]),
    )
    normalized_mode = str(display_mode or "").strip().casefold().replace("_", "-")
    if normalized_mode == "half-sbs":
        panel_width = max(1, panel_width // 2)
    x = left + (eye_width - panel_width) // 2
    y = top + eye_height - panel_height - max(10, int(eye_height * 0.035))
    return (x, y, x + panel_width, y + panel_height)


def media_browser_rect_for_eye(
    eye_rect: tuple[int, int, int, int], display_mode: str = ""
) -> tuple[int, int, int, int]:
    """Return the shared browser panel bounds for one eye in drawable pixels."""
    left, top, right, bottom = (int(value) for value in eye_rect)
    eye_width = max(1, right - left)
    eye_height = max(1, bottom - top)
    max_width = max(1, int(eye_width * 0.98))
    max_height = max(1, int(eye_height * 0.88))
    if max_width / max_height > MEDIA_BROWSER_TEXTURE_ASPECT:
        panel_height = max_height
        panel_width = max(1, int(panel_height * MEDIA_BROWSER_TEXTURE_ASPECT))
    else:
        panel_width = max_width
        panel_height = max(1, int(panel_width / MEDIA_BROWSER_TEXTURE_ASPECT))
    normalized_mode = str(display_mode or "").strip().casefold().replace("_", "-")
    if normalized_mode == "half-sbs":
        panel_width = max(1, panel_width // 2)
    x = left + (eye_width - panel_width) // 2
    y = top + (eye_height - panel_height) // 2
    return (x, y, x + panel_width, y + panel_height)


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
    thumbnails: dict[str, np.ndarray] | None = None,
    lang: str = "EN",
    font_scale: float = 1.0,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    try:
        font_scale = max(0.75, min(MEDIA_BROWSER_FONT_SCALE_MAX, float(font_scale)))
    except (TypeError, ValueError):
        font_scale = 1.0

    render_width, render_height = int(width), int(height)
    width, height = _browser_layout_dimensions(render_width, render_height)
    scale_x = render_width / max(1, width)
    scale_y = render_height / max(1, height)
    raster_scale = min(scale_x, scale_y)

    def browser_font(size: int, *, text: str = "", bold: bool = False):
        return _font(
            max(1, round(float(size) * font_scale * raster_scale)),
            text=text,
            bold=bold,
        )

    def browser_text(draw, xy, text, size=20, *, fill=None, bold=False, anchor="la"):
        _text(
            draw,
            xy,
            text,
            max(1, round(float(size) * font_scale * raster_scale)),
            fill=fill,
            bold=bold,
            anchor=anchor,
        )

    def browser_fit(draw, text: str, size: int, max_width: float, *, bold: bool = False) -> str:
        return _fit_text(
            draw,
            text,
            max(1, round(float(size) * font_scale * raster_scale)),
            max_width * raster_scale,
            bold=bold,
        )

    image = Image.new("RGBA", (render_width, render_height), (0, 0, 0, 0))
    draw = _ScaledImageDraw(ImageDraw.Draw(image), scale_x, scale_y)
    layout = _browser_layout(state, width, height)
    rects = _browser_button_rects(width, height)
    thumbnails = thumbnails or {}
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
        (48, layout["locations_heading_y"]),
        translate("Locations"),
        font=browser_font(22, text=translate("Locations"), bold=True),
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
        mount_kind = str(mount.get("kind") or "drive")
        mount_name = (
            translate("User folder")
            if mount_kind == "home"
            else str(mount.get("name") or mount.get("path") or "")
        )
        _draw_kind_icon(
            draw,
            "directory" if mount_kind == "home" else mount_kind,
            58,
            center_y,
            _rgba("text_primary"),
        )
        browser_text(
            draw,
            (84, center_y),
            browser_fit(
                draw,
                mount_name,
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
        font=browser_font(22, text=translate("Recent"), bold=True),
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
            browser_text(
                draw,
                (84, center_y),
                browser_fit(
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
        browser_text(
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
        font=browser_font(22, text=translate("Playlist"), bold=True),
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
        browser_text(
            draw,
            (52, center_y),
            f"{index + playlist_offset + 1:02d}",
            18,
            fill=_rgba("text_secondary"),
            anchor="lm",
        )
        browser_text(
            draw,
            (82, center_y),
            browser_fit(
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
            browser_text(draw, ((action_rect[0] + action_rect[2]) // 2, (action_rect[1] + action_rect[3]) // 2),
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
        browser_text(draw, ((action_rect[0] + action_rect[2]) // 2, (action_rect[1] + action_rect[3]) // 2),
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
        browser_text(
            draw,
            ((action_rect[0] + action_rect[2]) // 2, (action_rect[1] + action_rect[3]) // 2),
            label,
            14,
            fill=_rgba("text_primary", 255 if enabled else 104),
            anchor="mm",
        )

    stop_progress = hover_amount if hover_action == "viewer_stop" else 0.0
    stop_rect = _scaled_rect(
        rects["viewer_stop"],
        stop_progress,
        scale=_MEDIA_HOVER_BUTTON_SCALE,
        lift=1.0,
    )
    stop_fill = _mix_rgba(
        _rgba("destructive"),
        _rgba("destructive_hover"),
        stop_progress,
    )
    draw.rounded_rectangle(
        stop_rect,
        radius=10,
        fill=stop_fill,
        outline=_rgba("destructive_hover", 220 if stop_progress else 145),
        width=2 if stop_progress else 1,
    )
    stop_center_x = stop_rect[0] + 36
    stop_center_y = (stop_rect[1] + stop_rect[3]) // 2
    draw.rounded_rectangle(
        (
            stop_center_x - 8,
            stop_center_y - 8,
            stop_center_x + 8,
            stop_center_y + 8,
        ),
        radius=3,
        fill=_rgba("destructive_text"),
    )
    browser_text(
        draw,
        (stop_rect[0] + 62, stop_center_y),
        translate("Stop"),
        17,
        fill=_rgba("destructive_text"),
        bold=True,
        anchor="lm",
    )

    path_text = str(state.get("browser_path") or "")
    header_left = sidebar_right + 34
    draw.text(
        (header_left, 42),
        translate("Files"),
        font=browser_font(28, text=translate("Files"), bold=True),
        fill=_rgba("text_primary"),
    )
    header_actions_left = rects["browser_view_grid"][0] - 24
    browser_text(
        draw,
        (header_left, 94),
        browser_fit(draw, path_text, 19, header_actions_left - header_left),
        19,
        fill=_rgba("text_secondary"),
    )
    for key, label in (
        (
            "browser_sort",
            translate(
                "Sort A–Z"
                if state.get("browser_sort_descending") is True
                else "Sort Z–A"
            ),
        ),
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
        browser_text(
            draw,
            ((action_rect[0] + action_rect[2]) // 2, (action_rect[1] + action_rect[3]) // 2),
            label,
            15,
            fill=_rgba("text_primary", 255 if enabled else 104),
            anchor="mm",
        )
    for key, mode in (("browser_view_grid", "grid"), ("browser_view_list", "list")):
        enabled = media_control_enabled(key, state)
        action_rect = _draw_control_surface(
            draw,
            rects[key],
            enabled=enabled,
            hovered=hover_action == key,
            progress=hover_amount,
            primary=layout["view_mode"] == mode,
            idle_alpha=230,
            disabled_alpha=105,
            radius=OPENXR_MENU_RADII["control"],
        )
        _draw_browser_view_icon(
            draw,
            action_rect,
            mode,
            _rgba("text_primary", 255 if enabled else 104),
        )
    entries = list(state.get("browser_entries") or [])
    selected_path = str(state.get("browser_selection") or "")
    row_top = layout["file_rows_y"]
    visible_count = layout["visible_entry_count"]
    _scroll_step, _visible_count, max_offset = media_browser_scroll_metrics(
        state, width=width, height=height
    )
    scroll_offset = max(0, min(int(scroll_offset), max_offset))
    visible = entries[scroll_offset : scroll_offset + visible_count]
    if layout["view_mode"] == "grid":
        for index, entry in enumerate(visible):
            card = _browser_grid_card_rect(layout, index)
            path = str(entry.get("path", ""))
            kind = str(entry.get("kind") or "other")
            selected = path == selected_path
            hovered = hover_action == "entry" and path == hover_path
            card_fill = _mix_rgba(
                _rgba("surface_container", 225),
                _rgba("primary_container", 220),
                0.58 if selected else 0.0,
            )
            draw.rounded_rectangle(
                card,
                radius=OPENXR_MENU_RADII["control"],
                fill=card_fill,
                outline=(
                    _rgba("primary_hover", 220)
                    if hovered
                    else _rgba("outline", 150 if selected else 52)
                ),
                width=2 if selected or hovered else 1,
            )
            preview_top = card[1] + 12
            preview_height = min(
                layout["grid_preview_height"], card[3] - card[1] - 94
            )
            preview_rect = (
                card[0] + 12,
                preview_top,
                card[2] - 12,
                preview_top + preview_height,
            )
            _paste_browser_thumbnail(
                image,
                draw,
                preview_rect,
                thumbnails.get(path),
                kind,
                radius=OPENXR_MENU_RADII["control"],
                scale_x=scale_x,
                scale_y=scale_y,
            )
            browser_text(
                draw,
                (card[0] + 14, preview_rect[3] + 14),
                browser_fit(
                    draw,
                    str(entry.get("name") or path),
                    24,
                    card[2] - card[0] - 28,
                ),
                24,
                fill=_rgba("text_primary"),
            )
            if kind in {"video", "image", "disc"}:
                add_hovered = hover_action == "playlist_add" and hover_target == path
                add_rect = _draw_control_surface(
                    draw,
                    _browser_grid_add_rect(card),
                    enabled=True,
                    hovered=add_hovered,
                    progress=hover_amount,
                    idle_alpha=220,
                    radius=OPENXR_MENU_RADII["control"],
                )
                browser_text(
                    draw,
                    (
                        (add_rect[0] + add_rect[2]) // 2,
                        (add_rect[1] + add_rect[3]) // 2,
                    ),
                    "+",
                    24,
                    fill=_rgba("text_primary"),
                    anchor="mm",
                )
    else:
        row_height = layout["file_row_height"]
        for index, entry in enumerate(visible):
            y0 = row_top + index * row_height
            y1 = y0 + row_height - 4
            path = str(entry.get("path", ""))
            hovered = hover_action == "entry" and path == hover_path
            if path == selected_path or hovered:
                row = _scaled_rect(
                    (sidebar_right + 18, y0, layout["file_content_right"], y1),
                    hover_amount if hovered else 0.0,
                    scale=_MEDIA_HOVER_ROW_SCALE,
                    lift=0.8,
                )
                draw.rounded_rectangle(
                    row,
                    radius=OPENXR_MENU_RADII["control"],
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
            browser_text(
                draw,
                (sidebar_right + 76, center_y),
                browser_fit(
                    draw,
                    str(entry.get("name") or path),
                    24,
                    layout["file_content_right"] - sidebar_right - 124,
                ),
                24,
                fill=_rgba("text_primary"),
                anchor="lm",
            )
            if entry.get("kind") in {"video", "image", "disc"}:
                add_hovered = hover_action == "playlist_add" and hover_target == path
                add_rect = _draw_control_surface(
                    draw,
                    _browser_list_add_rect(center_y, width),
                    enabled=True,
                    hovered=add_hovered,
                    progress=hover_amount,
                    idle_alpha=220,
                    radius=OPENXR_MENU_RADII["control"],
                )
                browser_text(
                    draw,
                    (
                        (add_rect[0] + add_rect[2]) // 2,
                        (add_rect[1] + add_rect[3]) // 2,
                    ),
                    "+",
                    24,
                    fill=_rgba("text_primary"),
                    anchor="mm",
                )
    scrollbar_track, scrollbar_thumb = media_browser_scrollbar_metrics(
        state,
        scroll_offset,
        width=width,
        height=height,
    )
    if scrollbar_thumb is not None:
        draw.rounded_rectangle(
            scrollbar_track,
            radius=max(4, (scrollbar_track[2] - scrollbar_track[0]) // 2),
            fill=_rgba("surface_container_high", 190),
        )
        scrollbar_hovered = hover_action == "browser_scroll"
        draw.rounded_rectangle(
            scrollbar_thumb,
            radius=max(4, (scrollbar_thumb[2] - scrollbar_thumb[0]) // 2),
            fill=_mix_rgba(
                _rgba("text_secondary", 210),
                _rgba("primary_hover", 255),
                hover_amount if scrollbar_hovered else 0.0,
            ),
        )
    if not entries:
        browser_text(
            draw,
            ((sidebar_right + width) // 2, height // 2),
            translate("No playable media in this folder"),
            22,
            fill=_rgba("text_secondary"),
            anchor="mm",
        )
    error = str(state.get("browser_error") or state.get("error") or "")
    footer_y = rects["playlist_previous"][1]
    footer_left = sidebar_right + 36
    footer_right = width - 24
    if error:
        browser_text(
            draw,
            (footer_left, footer_y - 20),
            browser_fit(draw, error, 16, footer_right - footer_left),
            16,
            fill=_rgba("destructive_outline"),
        )
    instruction_right = footer_right
    browser_text(
        draw,
        (instruction_right, footer_y + 28),
        browser_fit(
            draw,
            translate("Click to open"),
            18,
            max(80, instruction_right - footer_left),
        ),
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
        "viewer_stop",
        "browser_up",
        "browser_refresh",
        "browser_close",
        "browser_view_grid",
        "browser_view_list",
        "browser_sort",
        "playlist_previous",
        "playlist_next",
        "playlist_clear",
        "playlist_add",
        "playlist_sort",
    ):
        x0, y0, x1, y1 = rects[key]
        if x0 <= x <= x1 and y0 <= y <= y1:
            if key == "playlist_add":
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
    entries = list(state.get("browser_entries") or [])
    _scroll_step, visible_count, max_offset = media_browser_scroll_metrics(
        state, width=width, height=height
    )
    track, thumb = media_browser_scrollbar_metrics(
        state,
        scroll_offset,
        width=width,
        height=height,
    )
    if thumb is not None and track[0] <= x <= track[2] and track[1] <= y <= track[3]:
        return (
            "browser_scroll",
            media_browser_scrollbar_offset(
                state,
                y,
                width=width,
                height=height,
            ),
        )
    offset = max(0, min(int(scroll_offset), max_offset))
    visible = entries[offset : offset + visible_count]
    if layout["view_mode"] == "grid":
        for index, entry in enumerate(visible):
            left, top, right, bottom = _browser_grid_card_rect(layout, index)
            if left <= x <= right and top <= y <= bottom:
                if entry.get("kind") in {"video", "image", "disc"}:
                    add_left, add_top, add_right, add_bottom = _browser_grid_add_rect(
                        (left, top, right, bottom)
                    )
                    if add_left <= x <= add_right and add_top <= y <= add_bottom:
                        return "playlist_add", entry.get("path")
                return "entry", entry.get("path")
    else:
        row_top = layout["file_rows_y"]
        row_height = layout["file_row_height"]
        row_bottom = row_top + visible_count * row_height
        if sidebar_right + 18 <= x <= layout["file_content_right"] and row_top <= y < row_bottom:
            index = int((y - row_top) // row_height)
            if 0 <= index < len(visible):
                entry = visible[index]
                center_y = row_top + index * row_height + (row_height - 4) // 2
                if entry.get("kind") in {"video", "image", "disc"}:
                    add_left, add_top, add_right, add_bottom = _browser_list_add_rect(
                        center_y, width
                    )
                    if add_left <= x <= add_right and add_top <= y <= add_bottom:
                        return "playlist_add", entry.get("path")
                return "entry", entry.get("path")
    return None
