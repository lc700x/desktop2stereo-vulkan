import atexit
import ctypes
import platform
import time
from dataclasses import dataclass
from ctypes import wintypes


OS_NAME = platform.system()


if OS_NAME == "Darwin":
    try:
        import Quartz
    except ImportError:
        Quartz = None

    KEY_F = 3
    MODIFY_FLAGS = (
        Quartz.kCGEventFlagMaskControl | Quartz.kCGEventFlagMaskCommand
        if Quartz is not None else 0
    )

    def send_ctrl_cmd_f(key=KEY_F, flags=MODIFY_FLAGS):
        if Quartz is None:
            return
        for down in (True, False):
            event = Quartz.CGEventCreateKeyboardEvent(None, key, down)
            Quartz.CGEventSetFlags(event, flags)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
            if down:
                time.sleep(0.02)
else:
    def send_ctrl_cmd_f(*args, **kwargs):
        return None


if OS_NAME == "Windows":
    import glfw
    import win32con
    import win32gui
    from windows_dpi import set_per_monitor_dpi_v2

    set_per_monitor_dpi_v2()

    def configure_macos_viewer_window(
        window, *, fullscreen=False, show_in_app_switcher=False
    ):
        return None

    def pin_macos_viewer_window(
        window, *, fullscreen=False, show_in_app_switcher=False
    ):
        return None

    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    SetWindowDisplayAffinity = user32.SetWindowDisplayAffinity
    SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
    SetWindowDisplayAffinity.restype = wintypes.BOOL
    ShowCursor = user32.ShowCursor
    ShowCursor.argtypes = [wintypes.BOOL]
    ShowCursor.restype = ctypes.c_int
    SetSystemCursor = user32.SetSystemCursor
    SetSystemCursor.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    SetSystemCursor.restype = wintypes.BOOL
    CreateCursor = user32.CreateCursor
    CreateCursor.argtypes = [
        wintypes.HINSTANCE, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
    ]
    CreateCursor.restype = wintypes.HANDLE
    LoadCursorW = user32.LoadCursorW
    LoadCursorW.argtypes = [wintypes.HINSTANCE, ctypes.c_void_p]
    LoadCursorW.restype = wintypes.HANDLE
    WDA_NONE = 0
    WDA_EXCLUDEFROMCAPTURE = 0x11
    WS_EX_TRANSPARENT = 0x20
    GWL_EXSTYLE = -20
    SYSTEM_CURSOR_IDS = (
        32512, 32513, 32514, 32515, 32516,
        32642, 32643, 32644, 32645, 32646,
        32648, 32649, 32650,
    )
    _cursor_hidden = False
    _hide_calls = 0
    _cursor_originals: dict[int, int] = {}
    _cursor_clones: dict[int, int] = {}
    _cursor_handle_map: dict[int, int] = {}
    _cursor_suppression_windows: set[int] = set()

    class CursorInfo(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("flags", wintypes.DWORD),
            ("hCursor", wintypes.HANDLE),
            ("ptScreenPos", wintypes.POINT),
        ]

    class IconInfo(ctypes.Structure):
        _fields_ = [
            ("fIcon", wintypes.BOOL),
            ("xHotspot", wintypes.DWORD),
            ("yHotspot", wintypes.DWORD),
            ("hbmMask", wintypes.HBITMAP),
            ("hbmColor", wintypes.HBITMAP),
        ]

    class Bitmap(ctypes.Structure):
        _fields_ = [
            ("bmType", ctypes.c_long),
            ("bmWidth", ctypes.c_long),
            ("bmHeight", ctypes.c_long),
            ("bmWidthBytes", ctypes.c_long),
            ("bmPlanes", wintypes.WORD),
            ("bmBitsPixel", wintypes.WORD),
            ("bmBits", ctypes.c_void_p),
        ]

    class BitmapInfoHeader(ctypes.Structure):
        _fields_ = [
            ("biSize", wintypes.DWORD),
            ("biWidth", ctypes.c_long),
            ("biHeight", ctypes.c_long),
            ("biPlanes", wintypes.WORD),
            ("biBitCount", wintypes.WORD),
            ("biCompression", wintypes.DWORD),
            ("biSizeImage", wintypes.DWORD),
            ("biXPelsPerMeter", ctypes.c_long),
            ("biYPelsPerMeter", ctypes.c_long),
            ("biClrUsed", wintypes.DWORD),
            ("biClrImportant", wintypes.DWORD),
        ]

    class BitmapInfo(ctypes.Structure):
        _fields_ = [("bmiHeader", BitmapInfoHeader), ("bmiColors", wintypes.DWORD * 3)]

    class BlendFunction(ctypes.Structure):
        _fields_ = [
            ("BlendOp", ctypes.c_ubyte),
            ("BlendFlags", ctypes.c_ubyte),
            ("SourceConstantAlpha", ctypes.c_ubyte),
            ("AlphaFormat", ctypes.c_ubyte),
        ]

    user32.GetCursorInfo.argtypes = [ctypes.POINTER(CursorInfo)]
    user32.GetCursorInfo.restype = wintypes.BOOL
    user32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
    user32.GetCursorPos.restype = wintypes.BOOL
    user32.MonitorFromPoint.argtypes = [wintypes.POINT, wintypes.DWORD]
    user32.MonitorFromPoint.restype = wintypes.HANDLE
    user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
    user32.MonitorFromWindow.restype = wintypes.HANDLE
    user32.GetIconInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(IconInfo)]
    user32.GetIconInfo.restype = wintypes.BOOL
    user32.DrawIconEx.argtypes = [
        wintypes.HDC, ctypes.c_int, ctypes.c_int, wintypes.HICON,
        ctypes.c_int, ctypes.c_int, wintypes.UINT, wintypes.HBRUSH, wintypes.DWORD,
    ]
    user32.DrawIconEx.restype = wintypes.BOOL
    user32.CopyIcon.argtypes = [wintypes.HANDLE]
    user32.CopyIcon.restype = wintypes.HANDLE
    user32.DestroyIcon.argtypes = [wintypes.HANDLE]
    user32.DestroyIcon.restype = wintypes.BOOL
    user32.GetDC.argtypes = [wintypes.HWND]
    user32.GetDC.restype = wintypes.HDC
    user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    user32.ReleaseDC.restype = ctypes.c_int
    user32.CreateWindowExW.argtypes = [
        wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID,
    ]
    user32.CreateWindowExW.restype = wintypes.HWND
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.ShowWindow.restype = wintypes.BOOL
    user32.SetWindowPos.argtypes = [
        wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, wintypes.UINT,
    ]
    user32.SetWindowPos.restype = wintypes.BOOL
    user32.UpdateLayeredWindow.argtypes = [
        wintypes.HWND, wintypes.HDC, ctypes.POINTER(wintypes.POINT),
        ctypes.POINTER(wintypes.SIZE), wintypes.HDC,
        ctypes.POINTER(wintypes.POINT), wintypes.COLORREF,
        ctypes.POINTER(BlendFunction), wintypes.DWORD,
    ]
    user32.UpdateLayeredWindow.restype = wintypes.BOOL
    user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    user32.ClientToScreen.restype = wintypes.BOOL
    user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.GetClientRect.restype = wintypes.BOOL
    user32.DestroyWindow.argtypes = [wintypes.HWND]
    user32.DestroyWindow.restype = wintypes.BOOL

    gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
    gdi32.CreateCompatibleDC.restype = wintypes.HDC
    gdi32.DeleteDC.argtypes = [wintypes.HDC]
    gdi32.DeleteDC.restype = wintypes.BOOL
    gdi32.CreateDIBSection.argtypes = [
        wintypes.HDC, ctypes.POINTER(BitmapInfo), wintypes.UINT,
        ctypes.POINTER(ctypes.c_void_p), wintypes.HANDLE, wintypes.DWORD,
    ]
    gdi32.CreateDIBSection.restype = wintypes.HBITMAP
    gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
    gdi32.SelectObject.restype = wintypes.HGDIOBJ
    gdi32.GetObjectW.argtypes = [wintypes.HGDIOBJ, ctypes.c_int, ctypes.c_void_p]
    gdi32.GetObjectW.restype = ctypes.c_int
    gdi32.GetDIBits.argtypes = [
        wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT,
        ctypes.c_void_p, ctypes.POINTER(BitmapInfo), wintypes.UINT,
    ]
    gdi32.GetDIBits.restype = ctypes.c_int
    gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
    gdi32.DeleteObject.restype = wintypes.BOOL

    def _cursor_bitmap_size(icon, bitmap) -> tuple[int, int]:
        width = int(bitmap.bmWidth)
        height = abs(int(bitmap.bmHeight))
        if not icon.hbmColor:
            height //= 2
        return width, height

    def _cursor_eye_size(width: int, height: int, mode: str) -> tuple[int, int]:
        normalized_mode = str(mode or "").strip().casefold().replace("_", "-")
        if normalized_mode == "half-sbs":
            width = max(1, round(width * 0.5))
        elif normalized_mode == "half-tab":
            height = max(1, round(height * 0.5))
        return width, height

    @dataclass(frozen=True)
    class CursorSnapshot:
        key: tuple[int, str]
        rgba: bytes
        width: int
        height: int
        hotspot_x: int
        hotspot_y: int

    def cursor_snapshot_key(display_mode: str) -> tuple[int, str] | None:
        current = CursorInfo()
        current.cbSize = ctypes.sizeof(current)
        if not user32.GetCursorInfo(ctypes.byref(current)) or not current.hCursor:
            return None
        if not (int(current.flags) & 0x00000001):
            return None
        mode = str(display_mode or "Half-SBS").strip().casefold().replace("_", "-")
        return (int(current.hCursor), mode)

    def cursor_screen_position() -> tuple[int, int] | None:
        current = CursorInfo()
        current.cbSize = ctypes.sizeof(current)
        if not user32.GetCursorInfo(ctypes.byref(current)):
            return None
        if not (int(current.flags) & 0x00000001):
            return None
        return (int(current.ptScreenPos.x), int(current.ptScreenPos.y))

    def capture_cursor_snapshot(display_mode: str) -> CursorSnapshot | None:
        current = CursorInfo()
        current.cbSize = ctypes.sizeof(current)
        if not user32.GetCursorInfo(ctypes.byref(current)) or not current.hCursor:
            return None
        if not (int(current.flags) & 0x00000001):
            return None

        cursor_key = int(current.hCursor)
        copied_cursor = 0
        cursor_handle = _cursor_handle_map.get(cursor_key)
        if cursor_handle is None:
            copied_cursor = int(user32.CopyIcon(current.hCursor) or 0)
            cursor_handle = copied_cursor
        if not cursor_handle:
            return None

        cursor = wintypes.HANDLE(cursor_handle)
        icon = IconInfo()
        source_width = source_height = 0
        width = height = 0
        hotspot_x = hotspot_y = 0
        screen = None
        dc = None
        bitmap = None
        old_bitmap = None
        bits = ctypes.c_void_p()
        try:
            if not user32.GetIconInfo(cursor, ctypes.byref(icon)):
                return None
            bitmap_handle = icon.hbmColor or icon.hbmMask
            dimensions = Bitmap()
            if not bitmap_handle or not gdi32.GetObjectW(
                bitmap_handle,
                ctypes.sizeof(dimensions),
                ctypes.byref(dimensions),
            ):
                return None
            source_width, source_height = _cursor_bitmap_size(icon, dimensions)
            if (
                source_width <= 0
                or source_height <= 0
                or source_width > 384
                or source_height > 384
            ):
                return None
            width, height = _cursor_eye_size(
                source_width, source_height, display_mode
            )
            screen = user32.GetDC(None)
            dc = gdi32.CreateCompatibleDC(screen)
            if not dc:
                return None
            bitmap_info = BitmapInfo()
            bitmap_info.bmiHeader.biSize = ctypes.sizeof(BitmapInfoHeader)
            bitmap_info.bmiHeader.biWidth = width
            bitmap_info.bmiHeader.biHeight = -height
            bitmap_info.bmiHeader.biPlanes = 1
            bitmap_info.bmiHeader.biBitCount = 32
            bitmap = gdi32.CreateDIBSection(
                dc,
                ctypes.byref(bitmap_info),
                0,
                ctypes.byref(bits),
                None,
                0,
            )
            if not bitmap or not bits:
                return None
            old_bitmap = gdi32.SelectObject(dc, bitmap)
            ctypes.memset(bits, 0, width * height * 4)
            if not user32.DrawIconEx(
                dc,
                0,
                0,
                cursor,
                width,
                height,
                0,
                None,
                0x0001 | 0x0002,
            ):
                return None

            raw = ctypes.string_at(bits, width * height * 4)
            rgba = bytearray(len(raw))
            for offset in range(0, len(raw), 4):
                blue, green, red, alpha = raw[offset : offset + 4]
                if alpha:
                    red = min(255, (red * 255 + alpha // 2) // alpha)
                    green = min(255, (green * 255 + alpha // 2) // alpha)
                    blue = min(255, (blue * 255 + alpha // 2) // alpha)
                else:
                    blue = green = red = 0
                rgba[offset : offset + 4] = bytes((red, green, blue, alpha))

            if not any(rgba[3::4]):
                for offset in range(0, len(rgba), 4):
                    if any(rgba[offset : offset + 3]):
                        rgba[offset + 3] = 255
            hotspot_x = round(int(icon.xHotspot) * width / source_width)
            hotspot_y = round(int(icon.yHotspot) * height / source_height)
            mode = str(display_mode or "Half-SBS").strip().casefold().replace("_", "-")
            return CursorSnapshot(
                key=(cursor_key, mode),
                rgba=bytes(rgba),
                width=width,
                height=height,
                hotspot_x=hotspot_x,
                hotspot_y=hotspot_y,
            )
        finally:
            if icon.hbmMask:
                gdi32.DeleteObject(icon.hbmMask)
            if icon.hbmColor:
                gdi32.DeleteObject(icon.hbmColor)
            if copied_cursor:
                user32.DestroyIcon(copied_cursor)
            if dc and old_bitmap:
                gdi32.SelectObject(dc, old_bitmap)
            if bitmap:
                gdi32.DeleteObject(bitmap)
            if dc:
                gdi32.DeleteDC(dc)
            if screen:
                user32.ReleaseDC(None, screen)

    def cursor_overlay_rects(
        owner_hwnd: int,
        eye_rects,
        snapshot: CursorSnapshot,
        position: tuple[int, int] | None,
    ) -> tuple[tuple[int, int, int, int], ...]:
        if position is None:
            return ()
        origin = wintypes.POINT(0, 0)
        rect = wintypes.RECT()
        if not (
            user32.ClientToScreen(owner_hwnd, ctypes.byref(origin))
            and user32.GetClientRect(owner_hwnd, ctypes.byref(rect))
        ):
            return ()
        client_width = int(rect.right - rect.left)
        client_height = int(rect.bottom - rect.top)
        local_x = int(position[0]) - origin.x
        local_y = int(position[1]) - origin.y
        if (
            client_width <= 0
            or client_height <= 0
            or not (0 <= local_x < client_width and 0 <= local_y < client_height)
        ):
            return ()
        u = local_x / client_width
        v = local_y / client_height
        result = []
        for left, top, right, bottom in eye_rects:
            region_width = int(right - left)
            region_height = int(bottom - top)
            if region_width <= 0 or region_height <= 0:
                continue
            x = int(left) + round(u * region_width) - snapshot.hotspot_x
            y = int(top) + round(v * region_height) - snapshot.hotspot_y
            result.append((x, y, snapshot.width, snapshot.height))
        return tuple(result)

    def _restore_cursor_scheme() -> None:
        global _hide_calls, _cursor_hidden
        for _ in range(_hide_calls):
            ShowCursor(True)
        _hide_calls = 0
        for cursor_id, original in _cursor_originals.items():
            SetSystemCursor(original, cursor_id)
        _cursor_originals.clear()
        for clone in _cursor_clones.values():
            user32.DestroyIcon(clone)
        _cursor_clones.clear()
        _cursor_handle_map.clear()
        _cursor_hidden = False

    atexit.register(_restore_cursor_scheme)

    def _set_native_cursor_suppressed(hidden: bool) -> None:
        global _cursor_hidden, _hide_calls
        if bool(hidden) == _cursor_hidden:
            return
        if not hidden:
            _restore_cursor_scheme()
            return
        originals: dict[int, int] = {}
        clones: dict[int, int] = {}
        handle_map: dict[int, int] = {}
        for cursor_id in SYSTEM_CURSOR_IDS:
            shared = LoadCursorW(None, ctypes.c_void_p(cursor_id))
            if not shared:
                continue
            original = user32.CopyIcon(shared)
            clone = user32.CopyIcon(shared)
            mask_size = 32 * 32 // 8
            invisible = CreateCursor(
                None, 0, 0, 32, 32,
                ctypes.create_string_buffer(b"\xff" * mask_size),
                ctypes.create_string_buffer(b"\x00" * mask_size),
            )
            if not original or not clone or not invisible or not SetSystemCursor(invisible, cursor_id):
                if original:
                    user32.DestroyIcon(original)
                if clone:
                    user32.DestroyIcon(clone)
                break
            originals[cursor_id] = original
            clones[cursor_id] = clone
            handle_map[int(shared)] = clone
        if len(originals) != len(SYSTEM_CURSOR_IDS):
            for cursor_id, original in originals.items():
                SetSystemCursor(original, cursor_id)
            for clone in clones.values():
                user32.DestroyIcon(clone)
            return
        _cursor_originals.update(originals)
        _cursor_clones.update(clones)
        _cursor_handle_map.update(handle_map)
        count = 0
        while count >= 0 and _hide_calls < 32:
            count = int(ShowCursor(False))
            _hide_calls += 1
        if count >= 0:
            _restore_cursor_scheme()
            return
        _cursor_hidden = True

    def native_cursor_suppressed() -> bool:
        return _cursor_hidden

    def update_native_cursor_suppression() -> None:
        """Hide the system cursor only while it is on a stereo-output monitor."""
        if not _cursor_suppression_windows:
            _set_native_cursor_suppressed(False)
            return
        position = wintypes.POINT()
        if not user32.GetCursorPos(ctypes.byref(position)):
            return
        monitor = user32.MonitorFromPoint(position, 0)
        hidden = bool(monitor) and any(
            user32.MonitorFromWindow(hwnd, 0) == monitor
            for hwnd in _cursor_suppression_windows
        )
        # The cursor scheme changes only when crossing an involved monitor's
        # boundary. Keep the existing cursor resource and two-eye overlay handling.
        _set_native_cursor_suppressed(hidden)

    def _set_window_capture_affinity(glfw_window, affinity):
        hwnd = glfw.get_win32_window(glfw_window)
        if not hwnd:
            print("[WindowCapture] Failed to resolve the native HWND.", flush=True)
            return False
        ctypes.windll.kernel32.SetLastError(0)
        if not SetWindowDisplayAffinity(hwnd, affinity):
            error = int(ctypes.windll.kernel32.GetLastError())
            print(
                f"[WindowCapture] SetWindowDisplayAffinity failed: hwnd=0x{int(hwnd):X} "
                f"affinity=0x{int(affinity):X} winerror={error}",
                flush=True,
            )
            return False
        return True

    def set_hwnd_capture_affinity(hwnd, affinity):
        if not hwnd:
            return False
        ctypes.windll.kernel32.SetLastError(0)
        return bool(SetWindowDisplayAffinity(hwnd, affinity))

    def hide_window_from_capture(window):
        result = _set_window_capture_affinity(window, WDA_EXCLUDEFROMCAPTURE)
        if result:
            print("[WindowCapture] SBS output is excluded from screen capture.", flush=True)
        return result

    def show_window_in_capture(window):
        result = _set_window_capture_affinity(window, WDA_NONE)
        if result:
            print("[WindowCapture] SBS output is visible to screen capture.", flush=True)
        return result

    def set_window_to_bottom(window):
        hwnd = glfw.get_win32_window(window)
        if hwnd:
            win32gui.SetWindowPos(
                hwnd, win32con.HWND_BOTTOM, 0, 0, 0, 0,
                win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE,
            )

    def set_window_mouse_passthrough(window, enabled, *, hide_cursor_overlay=False):
        try:
            hwnd = glfw.get_win32_window(window)
        except Exception:
            return False
        if not hwnd:
            return False
        hidden = bool(enabled and hide_cursor_overlay)
        try:
            passthrough = getattr(glfw, "MOUSE_PASSTHROUGH", None)
            if passthrough is not None:
                glfw.set_window_attrib(window, passthrough, int(enabled))
            else:
                style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
                style = style | WS_EX_TRANSPARENT if enabled else style & ~WS_EX_TRANSPARENT
                user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
        except Exception as exc:
            print(f"[WindowCapture] Cursor passthrough failed: {type(exc).__name__}: {exc}", flush=True)
            return False
        try:
            attr = getattr(glfw, "CURSOR", None)
            if attr is not None:
                mode = getattr(
                    glfw,
                    "CURSOR_DISABLED" if hidden else "CURSOR_NORMAL",
                    0x00034003 if hidden else 0x00034001,
                )
                glfw.set_input_mode(window, attr, mode)
        except Exception:
            pass
        if hidden:
            _cursor_suppression_windows.add(int(hwnd))
        else:
            _cursor_suppression_windows.discard(int(hwnd))
        update_native_cursor_suppression()
        return True

    class StereoCursorOverlay:
        def __init__(self, owner_hwnd, display_mode):
            self.owner_hwnd = int(owner_hwnd)
            self.mode = str(display_mode or "Half-SBS").casefold()
            self.window_handles = []
            self.cursor_handles = []
            self.icon_size = (32, 32)
            self.cursor_dc = None
            self.cursor_bitmap = None
            self.cursor_bits = ctypes.c_void_p()
            self.cursor_old_bitmap = None
            self.info = CursorInfo()
            self.info.cbSize = ctypes.sizeof(CursorInfo)
            self._ensure_cursor_buffer(32, 32)
            try:
                for _ in range(2):
                    # Independent top-level HWNDs are visible to Remote Desktop
                    # while the WGC exclusion list can keep them out of the source.
                    hwnd = user32.CreateWindowExW(
                        0x00080000 | 0x08000000 | 0x80 | WS_EX_TRANSPARENT,
                        "STATIC", None, 0x80000000, 0, 0, 384, 384,
                        None, None, None, None,
                    )
                    if not hwnd:
                        raise RuntimeError("native cursor overlay window creation failed")
                    self.window_handles.append(int(hwnd))
                    user32.ShowWindow(hwnd, 0)
            except Exception:
                self.close()
                raise

        def capture_window_handles(self):
            return tuple(self.window_handles)

        def _ensure_cursor_buffer(self, width: int, height: int) -> None:
            current_size = (width, height)
            if self.icon_size == current_size and self.cursor_dc:
                return
            if self.cursor_old_bitmap and self.cursor_dc:
                gdi32.SelectObject(self.cursor_dc, self.cursor_old_bitmap)
                gdi32.DeleteObject(self.cursor_bitmap)
                gdi32.DeleteDC(self.cursor_dc)
            self.cursor_dc = None
            self.cursor_bitmap = None
            self.cursor_bits = ctypes.c_void_p()
            screen = user32.GetDC(None)
            self.cursor_dc = gdi32.CreateCompatibleDC(screen)
            user32.ReleaseDC(None, screen)
            cursor_info = BitmapInfo()
            cursor_info.bmiHeader.biSize = ctypes.sizeof(BitmapInfoHeader)
            cursor_info.bmiHeader.biWidth, cursor_info.bmiHeader.biHeight = width, -height
            cursor_info.bmiHeader.biPlanes, cursor_info.bmiHeader.biBitCount = 1, 32
            self.cursor_bitmap = gdi32.CreateDIBSection(self.cursor_dc, ctypes.byref(cursor_info), 0, ctypes.byref(self.cursor_bits), None, 0)
            if not self.cursor_dc or not self.cursor_bitmap or not self.cursor_bits:
                raise RuntimeError("native cursor overlay allocation failed")
            self.cursor_old_bitmap = gdi32.SelectObject(self.cursor_dc, self.cursor_bitmap)
            self.icon_size = current_size

        def update_capture_policy(self):
            return None

        def update(self, eye_rects):
            current = CursorInfo()
            current.cbSize = ctypes.sizeof(current)
            if not user32.GetCursorInfo(ctypes.byref(current)) or not current.hCursor:
                self.hide()
                return
            cursor_key = int(current.hCursor)
            copied_cursor = 0
            cursor_handle = _cursor_handle_map.get(cursor_key)
            if cursor_handle is None:
                copied_cursor = int(user32.CopyIcon(current.hCursor) or 0)
                cursor_handle = copied_cursor
            if not cursor_handle:
                self.hide()
                return
            cursor = wintypes.HANDLE(cursor_handle)
            icon = IconInfo()
            try:
                if not user32.GetIconInfo(cursor, ctypes.byref(icon)):
                    self.hide()
                    return
                bitmap = icon.hbmColor or icon.hbmMask
                dimensions = Bitmap()
                if not bitmap or not gdi32.GetObjectW(bitmap, ctypes.sizeof(dimensions), ctypes.byref(dimensions)):
                    self.hide()
                    return
                source_width, source_height = _cursor_bitmap_size(icon, dimensions)
                if source_width <= 0 or source_height <= 0 or source_width > 384 or source_height > 384:
                    self.hide()
                    return
                width, height = _cursor_eye_size(
                    source_width, source_height, self.mode
                )
                self._ensure_cursor_buffer(width, height)
                ctypes.memset(self.cursor_bits, 0, width * height * 4)
                if not user32.DrawIconEx(
                    self.cursor_dc, 0, 0, cursor, width, height, 0,
                    None, 0x0001 | 0x0002,
                ):
                    self.hide()
                    return
                hotspot_x = round(int(icon.xHotspot) * width / source_width)
                hotspot_y = round(int(icon.yHotspot) * height / source_height)
                eye_width, eye_height = width, height
            finally:
                if icon.hbmMask:
                    gdi32.DeleteObject(icon.hbmMask)
                if icon.hbmColor:
                    gdi32.DeleteObject(icon.hbmColor)
                if copied_cursor:
                    user32.DestroyIcon(copied_cursor)

            origin, rect = wintypes.POINT(0, 0), wintypes.RECT()
            if not (user32.ClientToScreen(self.owner_hwnd, ctypes.byref(origin)) and user32.GetClientRect(self.owner_hwnd, ctypes.byref(rect))):
                self.hide()
                return
            client_width, client_height = rect.right - rect.left, rect.bottom - rect.top
            local_x, local_y = current.ptScreenPos.x - origin.x, current.ptScreenPos.y - origin.y
            if client_width <= 0 or client_height <= 0 or not (0 <= local_x < client_width and 0 <= local_y < client_height):
                self.hide()
                return
            u, v = local_x / client_width, local_y / client_height
            screen = user32.GetDC(None)
            try:
                blend = BlendFunction(0, 0, 255, 1)
                for index, hwnd in enumerate(self.window_handles):
                    if index >= len(eye_rects):
                        user32.ShowWindow(hwnd, 0)
                        continue
                    left, top, right, bottom = eye_rects[index]
                    region_width, region_height = right - left, bottom - top
                    eye_u = u
                    eye_v = v
                    x = origin.x + left + round(eye_u * region_width) - hotspot_x
                    y = origin.y + top + round(eye_v * region_height) - hotspot_y
                    user32.SetWindowPos(hwnd, ctypes.c_void_p(-1), x, y, eye_width, eye_height, 0x0010|0x0040|0x0200)
                    dest, src, size = wintypes.POINT(x,y), wintypes.POINT(0,0), wintypes.SIZE(eye_width,eye_height)
                    user32.UpdateLayeredWindow(hwnd, screen, ctypes.byref(dest), ctypes.byref(size), self.cursor_dc, ctypes.byref(src), 0, ctypes.byref(blend), 2)
            finally:
                user32.ReleaseDC(None, screen)

        def hide(self):
            for hwnd in self.window_handles:
                user32.ShowWindow(hwnd, 0)

        def close(self):
            for hwnd in self.window_handles:
                user32.ShowWindow(hwnd, 0)
                user32.DestroyWindow(hwnd)
            self.window_handles = []
            if self.cursor_old_bitmap and self.cursor_dc:
                gdi32.SelectObject(self.cursor_dc, self.cursor_old_bitmap)
            if self.cursor_bitmap:
                gdi32.DeleteObject(self.cursor_bitmap)
            if self.cursor_dc:
                gdi32.DeleteDC(self.cursor_dc)

elif OS_NAME == "Darwin":
    import AppKit
    from AppKit import NSCursor, NSEvent, NSScreen

    @dataclass(frozen=True)
    class CursorSnapshot:
        key: tuple[int, str, float]
        rgba: bytes
        width: int
        height: int
        hotspot_x: int
        hotspot_y: int

    _cursor_passthrough_windows: dict[int, tuple[object, bool]] = {}
    _macos_window_pin_times: dict[int, float] = {}
    _macos_activation_policy: int | None = None
    _hidden_cursor_display_id: int | None = None
    _cursor_suppression_target_display_id: int | None = None
    _cursor_suppression_cursor_position: tuple[int, int] | None = None
    _appkit_cursor_hidden = False
    _cursor_background_control_attempted = False
    _cursor_background_control_configured = False
    _cursor_snapshot_error_logged = False

    def _cocoa_window(window):
        import glfw

        ptr = glfw.get_cocoa_window(window)
        if not ptr:
            raise RuntimeError("glfwGetCocoaWindow returned null")
        import objc

        return objc.objc_object(c_void_p=ptr)

    def configure_macos_viewer_window(
        window, *, fullscreen=False, show_in_app_switcher=False
    ):
        """Configure output visibility while preserving fullscreen overlay behavior."""
        global _macos_activation_policy
        try:
            application = AppKit.NSApplication.sharedApplication()
            policy_name = (
                "NSApplicationActivationPolicyRegular"
                if show_in_app_switcher
                else "NSApplicationActivationPolicyAccessory"
            )
            activation_policy = getattr(AppKit, policy_name, None)
            if (
                activation_policy is not None
                and hasattr(application, "setActivationPolicy_")
                and _macos_activation_policy != int(activation_policy)
            ):
                changed = application.setActivationPolicy_(activation_policy)
                if changed is not False:
                    _macos_activation_policy = int(activation_policy)
        except Exception:
            # The window can still be pinned if AppKit rejects policy changes.
            pass
        ns_window = _cocoa_window(window)
        # A normal Cocoa window belongs to the Space in which it was created.
        # Native video fullscreen creates a different Space, so ordering the
        # same window again is not enough. The output must join the active
        # Space without activating or stealing focus from the video player.
        behavior = getattr(AppKit, "NSWindowCollectionBehaviorDefault", 0)
        # Keep the explicit full-screen auxiliary bit even when this viewer is
        # itself borderless fullscreen. CanJoinAllApplications covers another
        # app's Space; FullScreenAuxiliary keeps AppKit from dropping the
        # window during the Space transition. They are distinct from the
        # Stage Manager auxiliary behavior and can be combined here.
        behavior_names = [
            "NSWindowCollectionBehaviorCanJoinAllSpaces",
            "NSWindowCollectionBehaviorFullScreenAuxiliary",
            "NSWindowCollectionBehaviorCanJoinAllApplications",
        ]
        if not show_in_app_switcher:
            behavior_names.append("NSWindowCollectionBehaviorIgnoresCycle")
        for name in behavior_names:
            behavior |= int(getattr(AppKit, name, 0))
        if fullscreen:
            # Stationary prevents Mission Control from moving the output to a
            # different desktop while the source app changes fullscreen state.
            behavior |= int(
                getattr(AppKit, "NSWindowCollectionBehaviorStationary", 0)
            )
        ns_window.setCollectionBehavior_(behavior)
        if hasattr(ns_window, "setCanHide_"):
            ns_window.setCanHide_(False)
        ns_window.setHidesOnDeactivate_(False)
        level = getattr(
            AppKit,
            "NSStatusWindowLevel",
            getattr(AppKit, "NSFloatingWindowLevel", AppKit.NSNormalWindowLevel),
        ) if fullscreen else getattr(
            AppKit,
            "NSFloatingWindowLevel",
            AppKit.NSNormalWindowLevel,
        )
        ns_window.setLevel_(level)
        return ns_window

    def pin_macos_viewer_window(
        window, *, fullscreen=False, show_in_app_switcher=False
    ):
        """Keep SBS output above native fullscreen apps without activating it."""
        ns_window = configure_macos_viewer_window(
            window,
            fullscreen=fullscreen,
            show_in_app_switcher=show_in_app_switcher,
        )
        window_id = int(ns_window.windowNumber())
        now = time.monotonic()
        last_pin = _macos_window_pin_times.get(window_id, 0.0)
        # During a Space transition AppKit can report the window as hidden
        # even though it still exists. Re-order that state explicitly so it
        # can join the newly active fullscreen Space.
        if not bool(ns_window.isVisible()) or now - last_pin >= 0.10:
            ns_window.orderFrontRegardless()
            _macos_window_pin_times[window_id] = now
        return ns_window

    def _cursor_scale(position=None) -> float:
        point = NSEvent.mouseLocation() if position is None else position
        for screen in NSScreen.screens():
            # PyObjC returns CoreFoundation.CGRect for NSScreen.frame() on
            # recent macOS releases; AppKit.NSPointInRect expects an NSRect
            # object and raises ``CGRect has no attribute containsPoint_``.
            frame = screen.frame()
            if (
                float(frame.origin.x) <= float(point.x) <
                float(frame.origin.x + frame.size.width)
                and float(frame.origin.y) <= float(point.y) <
                float(frame.origin.y + frame.size.height)
            ):
                return float(screen.backingScaleFactor())
        return 1.0

    def _system_cursor():
        try:
            import objc

            cursor = NSCursor.currentSystemCursor()
            if cursor is None:
                cursor = NSCursor.arrowCursor()
            return cursor, int(objc.pyobjc_id(cursor)) if cursor is not None else 0
        except Exception:
            return None, 0

    def cursor_snapshot_key(display_mode: str):
        cursor, cursor_id = _system_cursor()
        if cursor is None:
            return None
        mode = str(display_mode or "Half-SBS").strip().casefold().replace("_", "-")
        position = NSEvent.mouseLocation()
        return cursor_id, mode, _cursor_scale(position)

    def cursor_screen_position():
        point = NSEvent.mouseLocation()
        return int(round(point.x)), int(round(point.y))

    def capture_cursor_snapshot(display_mode: str):
        global _cursor_snapshot_error_logged
        try:
            from PIL import Image

            cursor, cursor_id = _system_cursor()
            if cursor is None:
                return None
            image = cursor.image()
            if image is None:
                return None
            position = NSEvent.mouseLocation()
            scale = _cursor_scale(position)
            mode = str(display_mode or "Half-SBS").strip().casefold().replace("_", "-")
            tiff = image.TIFFRepresentation()
            if tiff is None:
                return None
            import io

            rgba_image = Image.open(io.BytesIO(bytes(tiff))).convert("RGBA")
            target_width = max(1, round(float(image.size().width) * scale))
            target_height = max(1, round(float(image.size().height) * scale))
            if rgba_image.size != (target_width, target_height):
                rgba_image = rgba_image.resize(
                    (target_width, target_height), Image.Resampling.LANCZOS
                )
            hotspot = cursor.hotSpot()
            hotspot_x = round(float(hotspot.x) * scale)
            # AppKit's NSCursor hotspot uses the same top-left image origin
            # used by the TIFF pixels. Do not invert Y here; the compositor
            # converts the screen point into the Metal/Vulkan top-left output
            # coordinates when it builds each eye rectangle.
            hotspot_y = round(float(hotspot.y) * scale)
            if mode == "half-sbs":
                half_width = max(1, round(rgba_image.width * 0.5))
                rgba_image = rgba_image.resize(
                    (half_width, rgba_image.height), Image.Resampling.LANCZOS
                )
                hotspot_x = round(hotspot_x * half_width / target_width)
            hotspot_x = max(0, min(rgba_image.width - 1, hotspot_x))
            hotspot_y = max(0, min(rgba_image.height - 1, hotspot_y))
            rgba = bytearray(rgba_image.tobytes())
            if not any(rgba[3::4]):
                for offset in range(0, len(rgba), 4):
                    if any(rgba[offset : offset + 3]):
                        rgba[offset + 3] = 255
            return CursorSnapshot(
                key=(cursor_id, mode, scale),
                rgba=bytes(rgba),
                width=rgba_image.width,
                height=rgba_image.height,
                hotspot_x=hotspot_x,
                hotspot_y=hotspot_y,
            )
        except Exception as exc:
            if not _cursor_snapshot_error_logged:
                _cursor_snapshot_error_logged = True
                print(
                    "[WindowCapture] macOS cursor snapshot failed: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
            return None

    def cursor_overlay_rects(owner_window, eye_rects, snapshot, position):
        if position is None or not eye_rects:
            return ()
        try:
            point = AppKit.NSPoint(float(position[0]), float(position[1]))
            window_point = owner_window.convertPointFromScreen_(point)
            content = owner_window.contentView()
            local = content.convertPoint_fromView_(window_point, None)
            bounds = content.bounds()
            width, height = float(bounds.size.width), float(bounds.size.height)
            if width <= 0.0 or height <= 0.0:
                return ()
            if not (0.0 <= local.x < width and 0.0 <= local.y < height):
                return ()
            u = local.x / width
            v = 1.0 - local.y / height
            result = []
            for left, top, right, bottom in eye_rects:
                region_width = int(right) - int(left)
                region_height = int(bottom) - int(top)
                if region_width <= 0 or region_height <= 0:
                    continue
                x = int(left) + round(u * region_width) - snapshot.hotspot_x
                y = int(top) + round(v * region_height) - snapshot.hotspot_y
                result.append((x, y, snapshot.width, snapshot.height))
            return tuple(result)
        except Exception:
            return ()

    def _restore_hidden_cursor() -> None:
        global _appkit_cursor_hidden, _cursor_suppression_target_display_id
        global _hidden_cursor_display_id, _cursor_suppression_cursor_position
        display_id = _hidden_cursor_display_id
        _hidden_cursor_display_id = None
        _cursor_suppression_target_display_id = None
        _cursor_suppression_cursor_position = None
        if display_id is not None:
            try:
                Quartz.CGDisplayShowCursor(display_id)
            except Exception:
                pass
        if _appkit_cursor_hidden:
            _appkit_cursor_hidden = False
            try:
                NSCursor.setHiddenUntilMouseMoves_(False)
                NSCursor.unhide()
            except Exception:
                pass

    def _configure_background_cursor_control() -> bool:
        """Allow this accessory app to hide the WindowServer cursor in the background.

        ``CGDisplayHideCursor`` is a real public API, but macOS ignores its
        hide request when the calling process is inactive unless this
        WindowServer connection property is enabled. The SBS window must stay
        inactive so mouse events can pass through to the source application.
        This private connection property is the same bridge used by native
        macOS cursor utilities; if it is unavailable, the public API remains
        the best fallback.
        """
        global _cursor_background_control_attempted
        global _cursor_background_control_configured
        if _cursor_background_control_attempted:
            return _cursor_background_control_configured
        _cursor_background_control_attempted = True
        try:
            core_graphics = ctypes.CDLL(
                "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"
            )
            core_foundation = ctypes.CDLL(
                "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
            )
            main_connection = core_graphics.CGSMainConnectionID
            main_connection.restype = ctypes.c_uint32
            connection_id = int(main_connection())
            if connection_id == 0:
                return False

            set_property = core_graphics.CGSSetConnectionProperty
            set_property.argtypes = [
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            set_property.restype = ctypes.c_int32

            create_string = core_foundation.CFStringCreateWithCString
            create_string.argtypes = [
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.c_uint32,
            ]
            create_string.restype = ctypes.c_void_p
            release = core_foundation.CFRelease
            release.argtypes = [ctypes.c_void_p]
            key = create_string(
                None,
                b"SetsCursorInBackground",
                0x08000100,  # kCFStringEncodingUTF8
            )
            true_value = ctypes.c_void_p.in_dll(
                core_foundation, "kCFBooleanTrue"
            ).value
            if not key or not true_value:
                return False
            try:
                result = int(
                    set_property(
                        connection_id,
                        connection_id,
                        key,
                        true_value,
                    )
                )
            finally:
                release(key)
            _cursor_background_control_configured = result == 0
        except Exception:
            _cursor_background_control_configured = False
        return _cursor_background_control_configured

    def _hide_display_cursor(display_id: int) -> bool:
        try:
            _configure_background_cursor_control()
            result = Quartz.CGDisplayHideCursor(display_id)
        except Exception:
            return False
        return result in (None, 0)

    def _reassert_display_cursor_hidden(display_id: int) -> bool:
        """Refresh one owned Quartz hide count after a pointer movement.

        Quartz tracks display cursor hiding as a reference count. Pairing a
        show/hide here keeps exactly one count owned by this process instead
        of adding a count on every polling tick.
        """
        try:
            Quartz.CGDisplayShowCursor(display_id)
        except Exception:
            pass
        return _hide_display_cursor(display_id)

    def _cursor_display_id() -> int | None:
        """Return the display currently under the pointer when available."""
        try:
            point = NSEvent.mouseLocation()
            for screen in NSScreen.screens():
                frame = screen.frame()
                if (
                    float(frame.origin.x) <= float(point.x) <
                    float(frame.origin.x + frame.size.width)
                    and float(frame.origin.y) <= float(point.y) <
                    float(frame.origin.y + frame.size.height)
                ):
                    return int(
                        screen.deviceDescription().get("NSScreenNumber", 0)
                    ) or None
        except Exception:
            pass
        return None

    def _cursor_is_visible() -> bool | None:
        try:
            return bool(Quartz.CGCursorIsVisible())
        except Exception:
            return None

    def _update_native_cursor_suppression() -> None:
        global _appkit_cursor_hidden, _cursor_suppression_target_display_id
        global _hidden_cursor_display_id, _cursor_suppression_cursor_position
        # The hide request is global on macOS (the CGDisplay argument is
        # ignored). Keep the window display as the logical owner so the
        # reference-counted show/hide pair remains stable across Spaces.
        target_display = None
        for window, should_hide in tuple(_cursor_passthrough_windows.values()):
            if not should_hide:
                continue
            screen = window.screen()
            if screen is not None:
                target_display = int(
                    screen.deviceDescription().get("NSScreenNumber", 0)
                ) or None
                break
        if target_display is None:
            target_display = _cursor_display_id()
        if (
            target_display is None
            and Quartz is not None
            and hasattr(Quartz, "CGMainDisplayID")
        ):
            target_display = int(Quartz.CGMainDisplayID())
        if (
            target_display == _cursor_suppression_target_display_id
            and (target_display is None or native_cursor_suppressed())
        ):
            position = None
            try:
                position = cursor_screen_position()
            except Exception:
                pass
            moved = (
                position is not None
                and position != _cursor_suppression_cursor_position
            )
            _cursor_suppression_cursor_position = position
            visible = _cursor_is_visible()
            if (moved or visible is True) and target_display is not None and (
                _hidden_cursor_display_id == target_display
            ):
                if _reassert_display_cursor_hidden(target_display):
                    _hidden_cursor_display_id = target_display
            return
        _restore_hidden_cursor()
        _cursor_suppression_target_display_id = target_display
        try:
            _cursor_suppression_cursor_position = cursor_screen_position()
        except Exception:
            _cursor_suppression_cursor_position = None
        if target_display is not None:
            if _hide_display_cursor(target_display):
                _hidden_cursor_display_id = target_display
            try:
                # Do not use setHiddenUntilMouseMoves_(True): macOS reveals
                # that temporary state on the next pointer event. The Quartz
                # display hide above is persistent; AppKit hide covers cursors
                # managed by the active application without auto-showing them.
                NSCursor.setHiddenUntilMouseMoves_(False)
                NSCursor.hide()
                _appkit_cursor_hidden = True
            except Exception:
                pass

    def native_cursor_suppressed():
        return _hidden_cursor_display_id is not None or _appkit_cursor_hidden

    def update_native_cursor_suppression():
        try:
            _update_native_cursor_suppression()
        except Exception:
            _restore_hidden_cursor()

    atexit.register(_restore_hidden_cursor)

    def set_window_mouse_passthrough(window, enabled, *, hide_cursor_overlay=False):
        try:
            ns_window = _cocoa_window(window)
            window_id = int(ns_window.windowNumber())
            ns_window.setIgnoresMouseEvents_(bool(enabled))
            if enabled:
                _cursor_passthrough_windows[window_id] = (
                    ns_window, bool(hide_cursor_overlay)
                )
            else:
                _cursor_passthrough_windows.pop(window_id, None)
            update_native_cursor_suppression()
            return True
        except Exception as exc:
            print(
                f"[WindowCapture] macOS cursor passthrough failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
            return False

    def hide_window_from_capture(*args, **kwargs):
        return False

    def show_window_in_capture(*args, **kwargs):
        return True

    def set_window_to_bottom(*args, **kwargs):
        return None

    def set_hwnd_capture_affinity(*args, **kwargs):
        return False

    class StereoCursorOverlay:
        def __init__(self, *args, **kwargs):
            self.window_handles = []

        def capture_window_handles(self):
            return ()

        def update_capture_policy(self):
            return None

        def update(self, *args, **kwargs):
            return None

        def hide(self):
            return None

        def close(self):
            return None

else:
    CursorSnapshot = None

    def cursor_snapshot_key(*args, **kwargs):
        return None

    def cursor_screen_position(*args, **kwargs):
        return None

    def capture_cursor_snapshot(*args, **kwargs):
        return None

    def cursor_overlay_rects(*args, **kwargs):
        return ()

    def native_cursor_suppressed():
        return False

    def update_native_cursor_suppression():
        return None

    def configure_macos_viewer_window(
        window, *, fullscreen=False, show_in_app_switcher=False
    ):
        return None

    def pin_macos_viewer_window(
        window, *, fullscreen=False, show_in_app_switcher=False
    ):
        return None

    class StereoCursorOverlay:
        def __init__(self, *args, **kwargs):
            self.window_handles = []
        def capture_window_handles(self):
            return ()
        def update_capture_policy(self):
            return None
        def update(self, *args, **kwargs):
            return None
        def hide(self):
            return None
        def close(self):
            return None
    def set_hwnd_capture_affinity(*args, **kwargs):
        return None
    def hide_window_from_capture(*args, **kwargs):
        return None
    def show_window_in_capture(*args, **kwargs):
        return None
    def set_window_to_bottom(*args, **kwargs):
        return None
    def set_window_mouse_passthrough(*args, **kwargs):
        return None
