import platform
import time


OS_NAME = platform.system()


if OS_NAME == "Darwin":
    try:
        import Quartz
    except ImportError:
        Quartz = None

    KEY_F = 3
    MODIFY_FLAGS = (
        Quartz.kCGEventFlagMaskControl | Quartz.kCGEventFlagMaskCommand
        if Quartz is not None
        else 0
    )

    def send_ctrl_cmd_f(key=KEY_F, flags=MODIFY_FLAGS):
        if Quartz is None:
            return
        ev_down = Quartz.CGEventCreateKeyboardEvent(None, key, True)
        Quartz.CGEventSetFlags(ev_down, flags)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev_down)

        time.sleep(0.02)

        ev_up = Quartz.CGEventCreateKeyboardEvent(None, key, False)
        Quartz.CGEventSetFlags(ev_up, flags)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev_up)

else:
    def send_ctrl_cmd_f(*args, **kwargs):
        return None


if OS_NAME == "Windows":
    import ctypes
    from ctypes import wintypes

    import glfw
    import win32con
    import win32gui

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        ctypes.windll.user32.SetProcessDPIAware()

    user32 = ctypes.windll.user32
    SetWindowDisplayAffinity = user32.SetWindowDisplayAffinity
    SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
    SetWindowDisplayAffinity.restype = wintypes.BOOL
    WDA_NONE = 0x00000000
    WDA_EXCLUDEFROMCAPTURE = 0x00000011
    WS_EX_TRANSPARENT = 0x00000020
    GWL_EXSTYLE = -20

    def _set_window_capture_affinity(glfw_window, affinity):
        hwnd = glfw.get_win32_window(glfw_window)
        if not hwnd:
            print(
                "[WindowCapture] Failed to resolve the native HWND.",
                flush=True,
            )
            return False
        ctypes.windll.kernel32.SetLastError(0)
        if not bool(SetWindowDisplayAffinity(hwnd, affinity)):
            error_code = int(ctypes.windll.kernel32.GetLastError())
            print(
                "[WindowCapture] SetWindowDisplayAffinity failed: "
                f"hwnd=0x{int(hwnd):X} affinity=0x{int(affinity):X} "
                f"winerror={error_code}",
                flush=True,
            )
            return False
        return True

    def hide_window_from_capture(glfw_window):
        hidden = _set_window_capture_affinity(
            glfw_window,
            WDA_EXCLUDEFROMCAPTURE,
        )
        if hidden:
            print(
                "[WindowCapture] SBS output is excluded from screen capture.",
                flush=True,
            )
        return hidden

    def show_window_in_capture(glfw_window):
        visible = _set_window_capture_affinity(glfw_window, WDA_NONE)
        if visible:
            print(
                "[WindowCapture] SBS output is visible to screen capture.",
                flush=True,
            )
        return visible

    def set_window_to_bottom(glfw_window):
        hwnd = glfw.get_win32_window(glfw_window)
        if hwnd:
            win32gui.SetWindowPos(
                hwnd,
                win32con.HWND_BOTTOM,
                0,
                0,
                0,
                0,
                win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE,
            )

    def set_window_mouse_passthrough(glfw_window, enabled: bool) -> bool:
        """Enable or disable mouse click-through for the Vulkan viewer window.

        When 3D Monitor mode runs on a single display without a second output
        monitor, the fullscreen SBS window covers the captured desktop. Enabling
        WS_EX_TRANSPARENT lets the system cursor remain visible over the SBS
        image and lets mouse input pass through to the underlying desktop (GLFW
        ``GLFW_MOUSE_PASSTHROUGH`` equivalence on Win32). This is a no-op on
        non-Windows platforms.
        """
        try:
            hwnd = glfw.get_win32_window(glfw_window)
        except Exception:
            hwnd = 0
        if not hwnd:
            return False
        try:
            # Prefer GLFW 3.4+ native passthrough attribute when available.
            passthrough_attr = getattr(glfw, "MOUSE_PASSTHROUGH", None)
            if passthrough_attr is not None:
                try:
                    glfw.set_window_attrib(glfw_window, passthrough_attr, 1 if enabled else 0)
                    # Ensure cursor remains visible over the stereo image.
                    try:
                        cursor_attr = getattr(glfw, "CURSOR", None)
                        cursor_normal = getattr(glfw, "CURSOR_NORMAL", 0x00034001)
                        if cursor_attr is not None:
                            glfw.set_input_mode(glfw_window, cursor_attr, cursor_normal)
                    except Exception:
                        pass
                    print(
                        f"[WindowCapture] Cursor passthrough {'enabled' if enabled else 'disabled'} "
                        "(GLFW_MOUSE_PASSTHROUGH).",
                        flush=True,
                    )
                    return True
                except Exception:
                    pass
            # Fallback: toggle WS_EX_TRANSPARENT directly.
            style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            if enabled:
                style |= WS_EX_TRANSPARENT
            else:
                style &= ~WS_EX_TRANSPARENT
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
            # Ensure cursor is visible when passthrough is active.
            if enabled:
                try:
                    cursor_attr = getattr(glfw, "CURSOR", None)
                    cursor_normal = getattr(glfw, "CURSOR_NORMAL", 0x00034001)
                    if cursor_attr is not None:
                        glfw.set_input_mode(glfw_window, cursor_attr, cursor_normal)
                except Exception:
                    pass
            print(
                f"[WindowCapture] Cursor passthrough {'enabled' if enabled else 'disabled'} "
                f"(WS_EX_TRANSPARENT=0x{WS_EX_TRANSPARENT:X}).",
                flush=True,
            )
            return True
        except Exception as exc:
            print(
                f"[WindowCapture] Cursor passthrough failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
            return False

else:
    def hide_window_from_capture(*args, **kwargs):
        return None

    def show_window_in_capture(*args, **kwargs):
        return None

    def set_window_to_bottom(*args, **kwargs):
        return None

    def set_window_mouse_passthrough(*args, **kwargs):
        return None
