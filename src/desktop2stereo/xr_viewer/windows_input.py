# Desktop2Stereo OpenXR viewer: OS mouse, keyboard, and scroll injection helpers.

import ctypes
import sys
import threading
import time

__all__ = [
    '_set_cursor_pos',
    '_send_mouse_flags',
    '_send_key',
    '_get_desktop_size',
    '_send_vscroll',
    '_send_hscroll',
    '_U32',
    '_MOUSEEVENTF_LEFTDOWN',
    '_MOUSEEVENTF_LEFTUP',
    '_MOUSEEVENTF_RIGHTDOWN',
    '_MOUSEEVENTF_RIGHTUP',
    '_start_physical_input_monitor',
    '_physical_input_generation',
    '_physical_mouse_active',
    '_physical_keyboard_active',
    '_physical_key_down',
    '_set_alt_long_press_callback',
    '_clear_alt_long_press_callback',
]


class _LongPressDetector:
    """Report one activation after a key has been held for a threshold."""

    def __init__(self, duration: float) -> None:
        self.duration = float(duration)
        self._down_since: float | None = None
        self._triggered = False

    def update(self, pressed: bool, now: float) -> bool:
        if not pressed:
            self._down_since = None
            self._triggered = False
            return False
        if self._down_since is None:
            self._down_since = float(now)
            return False
        if not self._triggered and float(now) - self._down_since + 1e-9 >= self.duration:
            self._triggered = True
            return True
        return False


# Windows input helpers (no-op on non-Windows)

if sys.platform == "win32":
    _U32 = ctypes.windll.user32

    class _MOUSEINPUT(ctypes.Structure):
        _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long),
                    ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
                    ("time", ctypes.c_ulong), ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

    class _INPUT(ctypes.Structure):
        class _I(ctypes.Union):
            _fields_ = [("mi", _MOUSEINPUT)]
        _anonymous_ = ("_i",)
        _fields_ = [("type", ctypes.c_ulong), ("_i", _I)]

    _MOUSEEVENTF_MOVE     = 0x0001
    _MOUSEEVENTF_LEFTDOWN = 0x0002
    _MOUSEEVENTF_LEFTUP   = 0x0004
    _MOUSEEVENTF_RIGHTDOWN= 0x0008
    _MOUSEEVENTF_RIGHTUP  = 0x0010
    _MOUSEEVENTF_ABSOLUTE = 0x8000
    _MOUSEEVENTF_WHEEL    = 0x0800
    _MOUSEEVENTF_HWHEEL   = 0x1000
    _KEYEVENTF_KEYUP      = 0x0002

    def _set_cursor_pos(x, y):
        # Use SetCursorPos with virtual-desktop pixel coordinates -works across all
        # monitors.  The old SendInput+MOVE+ABSOLUTE approach required manual
        # normalisation against the primary-monitor size and was fragile for
        # multi-monitor setups where the primary monitor isn't at (0,0).
        ctypes.windll.user32.SetCursorPos(int(x), int(y))

    def _send_mouse_flags(flags):
        inp = _INPUT(type=0)
        inp.mi.dwFlags = flags
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

    def _send_key(vk, shift=False, ctrl=False, alt=False, win=False):
        kbd = ctypes.windll.user32.keybd_event
        # Press modifiers (chord support: Ctrl+C, Alt+Tab, Win+R, etc.)
        if ctrl:  kbd(0x11, 0, 0, 0)             # VK_CONTROL down
        if shift: kbd(0x10, 0, 0, 0)             # VK_SHIFT down
        if alt:   kbd(0x12, 0, 0, 0)             # VK_MENU (Alt) down
        if win:   kbd(0x5B, 0, 0, 0)             # VK_LWIN down
        kbd(vk, 0, 0, 0)                          # key down
        kbd(vk, 0, _KEYEVENTF_KEYUP, 0)           # key up
        # Release modifiers in reverse
        if win:   kbd(0x5B, 0, _KEYEVENTF_KEYUP, 0)
        if alt:   kbd(0x12, 0, _KEYEVENTF_KEYUP, 0)
        if shift: kbd(0x10, 0, _KEYEVENTF_KEYUP, 0)
        if ctrl:  kbd(0x11, 0, _KEYEVENTF_KEYUP, 0)

    def _get_desktop_size():
        return _U32.GetSystemMetrics(0), _U32.GetSystemMetrics(1)

    def _send_vscroll(amount):
        inp = _INPUT(type=0)
        inp.mi.dwFlags = _MOUSEEVENTF_WHEEL
        inp.mi.mouseData = ctypes.c_ulong(int(amount) & 0xFFFFFFFF)
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

    def _send_hscroll(amount):
        inp = _INPUT(type=0)
        inp.mi.dwFlags = _MOUSEEVENTF_HWHEEL
        inp.mi.mouseData = ctypes.c_ulong(int(amount) & 0xFFFFFFFF)
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

    # ---- Physical-input priority monitor -------------------------------------
    # Low-level hooks detect PHYSICAL mouse/keyboard activity only: events the
    # app itself injects (SendInput) carry the LLMHF_INJECTED / LLKHF_INJECTED
    # flag and are ignored, and the controller beam's SetCursorPos does not
    # generate a hook event at all. The frame producer can then give the real
    # mouse/keyboard priority over the controller beam and the virtual keyboard.
    _LLMHF_INJECTED = 0x0001
    _LLKHF_INJECTED = 0x0010
    _WH_MOUSE_LL = 14
    _WH_KEYBOARD_LL = 13
    _WM_QUIT = 0x0012

    class _POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    class _MSLLHOOKSTRUCT(ctypes.Structure):
        _fields_ = [
            ("pt", _POINT),
            ("mouseData", ctypes.c_ulong),
            ("flags", ctypes.c_ulong),
            ("time", ctypes.c_ulong),
            ("dwExtraInfo", ctypes.c_ulonglong),
        ]

    class _KBDLLHOOKSTRUCT(ctypes.Structure):
        _fields_ = [
            ("vkCode", ctypes.c_ulong),
            ("scanCode", ctypes.c_ulong),
            ("flags", ctypes.c_ulong),
            ("time", ctypes.c_ulong),
            ("dwExtraInfo", ctypes.c_ulonglong),
        ]

    _physical_lock = threading.Lock()
    _physical_input_generation_value = 0
    _last_physical_mouse = 0.0
    _last_physical_keyboard = 0.0
    _physical_keys_down: set[int] = set()
    _physical_mouse_buttons_down: set[int] = set()
    _PHYSICAL_MOUSE_BUTTON_MESSAGES = {
        0x0201: 0x01, 0x0202: 0x01,  # Left button down/up.
        0x0204: 0x02, 0x0205: 0x02,  # Right button down/up.
        0x0207: 0x04, 0x0208: 0x04,  # Middle button down/up.
    }
    _PHYSICAL_MODIFIER_KEYS = {
        0x10: (0xA0, 0xA1),  # Shift.
        0x11: (0xA2, 0xA3),  # Control.
        0x12: (0xA4, 0xA5),  # Alt.
    }
    _VK_MENU = 0x12
    _VK_LMENU = 0xA4
    _VK_RMENU = 0xA5
    _ALT_LONG_PRESS_SECONDS = 0.6
    _alt_callback = None
    _alt_callback_lock = threading.Lock()
    _hook_thread = None
    _alt_poll_thread = None
    _hook_mouse = None
    _hook_keyboard = None
    _MouseProc = ctypes.WINFUNCTYPE(
        ctypes.c_ssize_t, ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p
    )
    _KeyboardProc = ctypes.WINFUNCTYPE(
        ctypes.c_ssize_t, ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p
    )

    @_MouseProc
    def _physical_mouse_proc(nCode, wParam, lParam):
        global _physical_input_generation_value, _last_physical_mouse
        if nCode >= 0 and lParam:
            try:
                data = ctypes.cast(lParam, ctypes.POINTER(_MSLLHOOKSTRUCT)).contents
                if not (int(data.flags) & _LLMHF_INJECTED):
                    with _physical_lock:
                        _physical_input_generation_value += 1
                        _last_physical_mouse = time.monotonic()
                        message = int(wParam)
                        button = _PHYSICAL_MOUSE_BUTTON_MESSAGES.get(message)
                        if message in (0x020B, 0x020C):
                            button = 0x05 if (int(data.mouseData) >> 16) == 1 else 0x06
                        if button is not None:
                            if message in (0x0201, 0x0204, 0x0207, 0x020B):
                                _physical_mouse_buttons_down.add(button)
                            else:
                                _physical_mouse_buttons_down.discard(button)
            except Exception:
                pass
        return ctypes.windll.user32.CallNextHookEx(
            _hook_mouse, nCode, wParam, lParam
        )

    @_KeyboardProc
    def _physical_keyboard_proc(nCode, wParam, lParam):
        global _physical_input_generation_value, _last_physical_keyboard
        if nCode >= 0 and lParam:
            try:
                data = ctypes.cast(lParam, ctypes.POINTER(_KBDLLHOOKSTRUCT)).contents
                if not (int(data.flags) & _LLKHF_INJECTED):
                    with _physical_lock:
                        _physical_input_generation_value += 1
                        _last_physical_keyboard = time.monotonic()
                        if int(data.flags) & 0x0080:
                            _physical_keys_down.discard(int(data.vkCode))
                        else:
                            _physical_keys_down.add(int(data.vkCode))
            except Exception:
                pass
        return ctypes.windll.user32.CallNextHookEx(
            _hook_keyboard, nCode, wParam, lParam
        )

    def _physical_hook_loop():
        global _hook_mouse, _hook_keyboard
        user32 = ctypes.windll.user32
        # Passing the callback as a void pointer avoids ctypes rejecting the
        # two distinct but ABI-identical WINFUNCTYPE instances on some Python
        # builds ("expected WinFunctionType instead of WinFunctionType").
        # A null module handle is required here: the callback lives in the
        # Python interpreter rather than a separate hook DLL. Passing the
        # interpreter's GetModuleHandleW(None) result makes Windows fail with
        # ERROR_MOD_NOT_FOUND (126), leaving physical input unmonitored.
        set_hook = user32.SetWindowsHookExW
        set_hook.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_ulong,
        ]
        set_hook.restype = ctypes.c_void_p
        call_next = user32.CallNextHookEx
        call_next.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        call_next.restype = ctypes.c_ssize_t
        _hook_mouse = set_hook(
            _WH_MOUSE_LL, ctypes.cast(_physical_mouse_proc, ctypes.c_void_p), None, 0
        )
        _hook_keyboard = set_hook(
            _WH_KEYBOARD_LL,
            ctypes.cast(_physical_keyboard_proc, ctypes.c_void_p),
            None,
            0,
        )
        msg = ctypes.wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
        if _hook_mouse:
            user32.UnhookWindowsHookEx(_hook_mouse)
        if _hook_keyboard:
            user32.UnhookWindowsHookEx(_hook_keyboard)

    def _alt_long_press_poll_loop():
        """Poll Alt like the GUI polls Escape, independent of hook delivery."""
        detector = _LongPressDetector(_ALT_LONG_PRESS_SECONDS)
        user32 = ctypes.windll.user32
        while True:
            try:
                pressed = any(
                    user32.GetAsyncKeyState(key) & 0x8000
                    for key in (_VK_MENU, _VK_LMENU, _VK_RMENU)
                )
                if detector.update(bool(pressed), time.monotonic()):
                    with _alt_callback_lock:
                        callback = _alt_callback
                    if callback is not None:
                        try:
                            callback()
                        except Exception:
                            pass
            except Exception:
                pass
            time.sleep(0.05)

    def _start_physical_input_monitor():
        """Start the low-level physical input hooks (idempotent)."""
        global _hook_thread, _alt_poll_thread
        if _hook_thread is not None:
            return
        _hook_thread = threading.Thread(
            target=_physical_hook_loop, name="d2s-physical-input", daemon=True
        )
        _hook_thread.start()
        _alt_poll_thread = threading.Thread(
            target=_alt_long_press_poll_loop,
            name="d2s-alt-long-press",
            daemon=True,
        )
        _alt_poll_thread.start()

    def _physical_mouse_active(timeout: float = 0.0) -> bool:
        with _physical_lock:
            return bool(_physical_mouse_buttons_down) or (
                time.monotonic() - _last_physical_mouse
            ) < float(timeout)

    def _physical_input_generation() -> int:
        with _physical_lock:
            return int(_physical_input_generation_value)

    def _physical_keyboard_active(timeout: float = 0.0) -> bool:
        with _physical_lock:
            return bool(_physical_keys_down) or (
                time.monotonic() - _last_physical_keyboard
            ) < float(timeout)

    def _physical_key_down(virtual_key: int) -> bool:
        with _physical_lock:
            key = int(virtual_key)
            return (
                key in _physical_keys_down
                or key in _physical_mouse_buttons_down
                or any(side in _physical_keys_down for side in _PHYSICAL_MODIFIER_KEYS.get(key, ()))
            )

    def _set_alt_long_press_callback(callback):
        global _alt_callback
        with _alt_callback_lock:
            _alt_callback = callback

    def _clear_alt_long_press_callback():
        global _alt_callback
        with _alt_callback_lock:
            _alt_callback = None

else:
    def _set_cursor_pos(x, y): pass
    def _send_mouse_flags(flags): pass
    def _send_key(vk, shift=False, ctrl=False, alt=False, win=False): pass
    def _send_vscroll(amount): pass
    def _send_hscroll(amount): pass
    def _get_desktop_size(): return (1920, 1080)
    _MOUSEEVENTF_LEFTDOWN  = 0x0002
    _MOUSEEVENTF_LEFTUP    = 0x0004
    _MOUSEEVENTF_RIGHTDOWN = 0x0008
    _MOUSEEVENTF_RIGHTUP   = 0x0010
    _KEYEVENTF_KEYUP       = 0x0002

    def _start_physical_input_monitor(): pass
    def _physical_input_generation() -> int: return 0
    def _physical_mouse_active(timeout: float = 0.0) -> bool: return False
    def _physical_keyboard_active(timeout: float = 0.0) -> bool: return False
    def _physical_key_down(virtual_key: int) -> bool: return False
    def _set_alt_long_press_callback(callback): return None
    def _clear_alt_long_press_callback(): return None
