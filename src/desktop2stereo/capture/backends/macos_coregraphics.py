from __future__ import annotations

from collections import OrderedDict
import cv2
import numpy as np
from PIL import Image

import objc
import Quartz as QZ
import Quartz.CoreGraphics as CG
from AppKit import NSCursor, NSBitmapImageRep, NSPNGFileType, NSScreen
from Quartz import CGCursorIsVisible, NSEvent

# Keep the base cursor cache small and stable
_cursor_cache = {
    "bgra": None,
    "hotspot": None,
    "alpha_f32": None,
    "premultiplied_bgr_f32": None,
    "last_cursor": None,
}

# Small bounded cache for resized cursor variants
_CURSOR_RESIZE_CACHE_MAX = 4

def _find_window(matcher):
    windows = QZ.CGWindowListCopyWindowInfo(
        QZ.kCGWindowListOptionAll, QZ.kCGNullWindowID
    ) or []
    return [w for w in windows if matcher(w)]

def get_window_info_mac(window_title):
    """
    Return a dict with window_id + bounds for a unique window match.
    """
    matches = _find_window(
        lambda w: w.get("kCGWindowName") == window_title
        or w.get("kCGWindowOwnerName") == window_title
    )

    if len(matches) == 0:
        return None
    if len(matches) > 1:
        raise ValueError(f"Found multiple windows with name: {window_title}")

    win = matches[0]
    bounds = win.get("kCGWindowBounds", {}) or {}

    return {
        "window_id": int(win["kCGWindowNumber"]),
        "left": int(bounds.get("X", 0)),
        "top": int(bounds.get("Y", 0)),
        "width": int(bounds.get("Width", 0)),
        "height": int(bounds.get("Height", 0)),
    }

def get_window_client_bounds_mac(window_title):
    """
    Return (x, y, w, h) for a window by title; None if not found.
    """
    info = get_window_info_mac(window_title)
    if info is None:
        return None, None, None, None
    return info["left"], info["top"], info["width"], info["height"]

def _cg_capture_region_as_bgra(
    region: tuple[int, int, int, int] | None = None,
    window_id: int | None = None,
) -> np.ndarray:
    """
    Capture a region or window using CoreGraphics and return BGRA uint8.
    """
    if window_id is not None and region is not None:
        raise ValueError("Only one of region or window_id must be specified")

    if window_id is not None:
        image = CG.CGWindowListCreateImage(
            CG.CGRectNull,
            CG.kCGWindowListOptionIncludingWindow,
            window_id,
            CG.kCGWindowImageBoundsIgnoreFraming | CG.kCGWindowImageNominalResolution,
        )
    else:
        cg_region = CG.CGRectInfinite if region is None else CG.CGRectMake(*region)
        image = CG.CGWindowListCreateImage(
            cg_region,
            CG.kCGWindowListOptionOnScreenOnly,
            CG.kCGNullWindowID,
            CG.kCGWindowImageDefault,
        )

    if image is None:
        raise RuntimeError("Could not capture image with CoreGraphics")

    width = CG.CGImageGetWidth(image)
    height = CG.CGImageGetHeight(image)
    bpr = CG.CGImageGetBytesPerRow(image)

    provider = CG.CGImageGetDataProvider(image)
    data = CG.CGDataProviderCopyData(provider)
    raw = np.frombuffer(data, dtype=np.uint8)

    # On macOS this is typically BGRA-compatible for OpenCV use.
    frame = np.lib.stride_tricks.as_strided(
        raw,
        shape=(height, width, 4),
        strides=(bpr, 4, 1),
        writeable=True,
    ).copy()

    return frame

def get_cursor_image_and_hotspot():
    """
    Return cursor image in BGRA, hotspot, alpha and premultiplied BGR.
    Cache is refreshed only when the actual system cursor changes.
    """
    try:
        with objc.autorelease_pool():
            cursor = NSCursor.currentSystemCursor()
            if cursor is None:
                return None, None, None, None

            if cursor == _cursor_cache["last_cursor"] and _cursor_cache["bgra"] is not None:
                return (
                    _cursor_cache["bgra"],
                    _cursor_cache["hotspot"],
                    _cursor_cache["alpha_f32"],
                    _cursor_cache["premultiplied_bgr_f32"],
                )

            _cursor_cache["last_cursor"] = cursor

            ns_image = cursor.image()
            if ns_image is None:
                return None, None, None, None

            hot_pt = cursor.hotSpot()
            hotspot = (int(hot_pt.x), int(hot_pt.y))

            tiff_data = ns_image.TIFFRepresentation()
            if tiff_data is None:
                return None, None, None, None

            bitmap = NSBitmapImageRep.imageRepWithData_(tiff_data)
            if bitmap is None:
                return None, None, None, None

            png_data = bitmap.representationUsingType_properties_(NSPNGFileType, None)
            if png_data is None:
                return None, None, None, None

            rgba = np.asarray(Image.open(io.BytesIO(png_data)).convert("RGBA"), dtype=np.uint8)
            bgra = rgba[:, :, [2, 1, 0, 3]].copy()

            alpha = bgra[:, :, 3].astype(np.float32) / 255.0
            premultiplied_bgr = bgra[:, :, :3].astype(np.float32) * alpha[:, :, None]

            _cursor_cache["bgra"] = bgra
            _cursor_cache["hotspot"] = hotspot
            _cursor_cache["alpha_f32"] = alpha
            _cursor_cache["premultiplied_bgr_f32"] = premultiplied_bgr

            return bgra, hotspot, alpha, premultiplied_bgr

    except Exception:
        return None, None, None, None

def get_cursor_position():
    """Return current cursor (x, y) in macOS display coordinates (origin bottom-left)."""
    ev = CG.CGEventCreate(None)
    loc = CG.CGEventGetLocation(ev)
    return loc.x, loc.y

def is_cursor_visible():
    """Check if cursor is visible (cached check for performance)."""
    return CGCursorIsVisible()

def overlay_cursor_on_frame(frame_bgr, cursor_bgra, hotspot, cursor_pos,
                            alpha_f32=None, premultiplied_bgr_f32=None):
    """
    Overlay cursor onto a frame that is either BGR or BGRA.
    Only the first 3 channels are blended; alpha is preserved if present.
    """
    if cursor_bgra is None:
        x_cv, y_cv = cursor_pos
        cv2.circle(frame_bgr, (int(round(x_cv)), int(round(y_cv))), 8, (0, 0, 255), -1)
        return frame_bgr

    h_frame, w_frame = frame_bgr.shape[:2]
    x_cv, y_cv = cursor_pos
    cur_h, cur_w = cursor_bgra.shape[:2]
    hot_x, hot_y = hotspot

    top_left_x = int(round(x_cv - hot_x))
    top_left_y = int(round(y_cv - hot_y))

    x0 = max(top_left_x, 0)
    y0 = max(top_left_y, 0)
    x1 = min(top_left_x + cur_w, w_frame)
    y1 = min(top_left_y + cur_h, h_frame)

    if x0 >= x1 or y0 >= y1:
        return frame_bgr

    src_x0 = x0 - top_left_x
    src_y0 = y0 - top_left_y
    src_x1 = src_x0 + (x1 - x0)
    src_y1 = src_y0 + (y1 - y0)

    dst_region = frame_bgr[y0:y1, x0:x1]
    dst_rgb = dst_region[:, :, :3] if dst_region.ndim == 3 and dst_region.shape[2] == 4 else dst_region

    if premultiplied_bgr_f32 is not None and alpha_f32 is not None:
        src_premult = premultiplied_bgr_f32[src_y0:src_y1, src_x0:src_x1]
        alpha_roi = alpha_f32[src_y0:src_y1, src_x0:src_x1]
        src_region = None
    else:
        src_region = cursor_bgra[src_y0:src_y1, src_x0:src_x1]
        alpha_roi = src_region[:, :, 3].astype(np.float32) / 255.0
        src_premult = src_region[:, :, :3].astype(np.float32) * alpha_roi[:, :, None]

    a_min = float(alpha_roi.min())
    a_max = float(alpha_roi.max())

    if a_max <= 1e-6:
        return frame_bgr

    if a_min >= 0.999:
        if src_region is not None:
            dst_rgb[:, :, :] = src_region[:, :, :3]
        else:
            np.copyto(dst_rgb, np.clip(src_premult + 0.5, 0, 255).astype(np.uint8))
        return frame_bgr

    alpha_3ch = alpha_roi[:, :, None]
    dst_f32 = dst_rgb.astype(np.float32, copy=False)
    blended = src_premult + dst_f32 * (1.0 - alpha_3ch)

    np.clip(blended, 0, 255, out=blended)
    res_uint8 = blended.astype(np.uint8, copy=False)

    if a_max >= 0.999:
        mask_opaque = (alpha_roi >= 0.999)
        if mask_opaque.any():
            if src_region is not None:
                res_uint8[mask_opaque] = src_region[:, :, :3][mask_opaque]
            else:
                res_uint8[mask_opaque] = np.clip(src_premult + 0.5, 0, 255).astype(np.uint8)[mask_opaque]

    np.copyto(dst_rgb, res_uint8)
    return frame_bgr

class DesktopGrabber:
    def __init__(self, output_resolution=1080, fps=60, window_title=None,
                capture_mode="Monitor", monitor_index=1, with_cursor=True):
        # Normalize (width, height) tuples to a target height; see
        # macos_screencapturekit for details.
        if isinstance(output_resolution, (tuple, list)):
            output_resolution = int(output_resolution[1])
        self.scaled_height = int(output_resolution)
        self.fps = fps
        self.with_cursor = with_cursor
        self.window_title = window_title
        self.capture_mode = capture_mode
        self.prev_rect = None
        self.window_id = None

        # bounded per-instance resize cache
        self._cursor_cache = OrderedDict()

        if self.capture_mode == "Monitor":
            screens = list(NSScreen.screens())
            if not screens:
                raise RuntimeError("No screens found")

            mon_index = max(1, min(monitor_index, len(screens)))
            screen = screens[mon_index - 1]
            frame = screen.frame()

            self.left = int(frame.origin.x)
            self.top = int(frame.origin.y)
            self.width = int(frame.size.width)
            self.height = int(frame.size.height)
        else:
            info = get_window_info_mac(self.window_title)
            if info is None:
                raise RuntimeError(f"Window '{self.window_title}' not found")

            self.window_id = info["window_id"]
            self.left = info["left"]
            self.top = info["top"]
            self.width = info["width"]
            self.height = info["height"]

    def _ensure_rect(self):
        if self.capture_mode != "Monitor":
            info = get_window_info_mac(self.window_title)
            if info is None:
                return
            current = (info["left"], info["top"], info["width"], info["height"])
            if current == self.prev_rect:
                return

            self.prev_rect = current
            self.window_id = info["window_id"]
            self.left, self.top, self.width, self.height = current

    def get_scale(self):
        mouse_location = NSEvent.mouseLocation()
        screens = NSScreen.screens()

        for screen in screens:
            frame = screen.frame()
            if frame.origin.x <= mouse_location.x <= frame.origin.x + frame.size.width and \
            frame.origin.y <= mouse_location.y <= frame.origin.y + frame.size.height:
                return screen.backingScaleFactor()

        raise RuntimeError("No screen found under cursor")

    def _get_resized_cursor(self, cursor_bgra, hotspot, system_scale):
        """
        Keep your original cursor sizing logic unchanged.
        Bounded cache prevents memory growth.
        """
        scale_factor = max(1, 16 // max(1, int(system_scale)))

        cache_key = (id(cursor_bgra), cursor_bgra.shape, scale_factor)
        cached = self._cursor_cache.get(cache_key)
        if cached is not None:
            self._cursor_cache.move_to_end(cache_key)
            return cached

        h, w = cursor_bgra.shape[:2]
        if h > scale_factor and w > scale_factor:
            new_w, new_h = max(1, w // scale_factor), max(1, h // scale_factor)
            resized_bgra = cv2.resize(cursor_bgra, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized_bgra = cursor_bgra

        alpha_f32 = resized_bgra[:, :, 3].astype(np.float32) / 255.0
        premultiplied = resized_bgra[:, :, :3].astype(np.float32) * alpha_f32[:, :, None]

        result = (resized_bgra, hotspot, alpha_f32, premultiplied)
        self._cursor_cache[cache_key] = result
        self._cursor_cache.move_to_end(cache_key)

        while len(self._cursor_cache) > _CURSOR_RESIZE_CACHE_MAX:
            self._cursor_cache.popitem(last=False)

        return result

    def grab(self, output_format="bgr"):
        """
        output_format:
            - "bgra": no full-frame color conversion
            - "bgr" : convert once at the end
        """
        self._ensure_rect()

        if self.capture_mode == "Monitor":
            frame = _cg_capture_region_as_bgra(
                region=(self.left, self.top, self.width, self.height)
            )
        else:
            frame = _cg_capture_region_as_bgra(window_id=self.window_id)

        if self.with_cursor and CGCursorIsVisible():
            x, y = get_cursor_position()
            system_scale = self.get_scale()

            if 0 <= x - self.left <= self.width and 0 <= y - self.top <= self.height:
                cursor_x = (x - self.left) * system_scale
                cursor_y = (y - self.top) * system_scale

                cursor_bgra, hotspot, alpha_f32, premultiplied = get_cursor_image_and_hotspot()
                if cursor_bgra is not None:
                    cursor_bgra, hotspot, alpha_f32, premultiplied = self._get_resized_cursor(
                        cursor_bgra, hotspot, system_scale
                    )

                    overlay_cursor_on_frame(
                        frame,
                        cursor_bgra,
                        hotspot,
                        (cursor_x, cursor_y),
                        alpha_f32=alpha_f32,
                        premultiplied_bgr_f32=premultiplied,
                    )

        if output_format == "bgra":
            return frame, self.scaled_height
        elif output_format == "bgr":
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR), self.scaled_height
        else:
            raise ValueError("output_format must be 'bgr' or 'bgra'")

    def stop(self):
        if hasattr(self, "_cursor_cache"):
            self._cursor_cache.clear()
