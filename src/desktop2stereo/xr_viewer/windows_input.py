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
    '_physical_mouse_active',
    '_physical_keyboard_active',
]

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
    _LLKHF_INJECTED = 0x0001
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
    _last_physical_mouse = 0.0
    _last_physical_keyboard = 0.0
    _hook_thread = None
    _hook_mouse = None
    _hook_keyboard = None
    _MouseProc = ctypes.WINFUNCTYPE(
        ctypes.c_long, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p
    )
    _KeyboardProc = ctypes.WINFUNCTYPE(
        ctypes.c_long, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p
    )

    @_MouseProc
    def _physical_mouse_proc(nCode, wParam, lParam):
        global _last_physical_mouse
        if nCode >= 0 and lParam:
            try:
                data = ctypes.cast(lParam, ctypes.POINTER(_MSLLHOOKSTRUCT)).contents
                if not (int(data.flags) & _LLMHF_INJECTED):
                    with _physical_lock:
                        _last_physical_mouse = time.monotonic()
            except Exception:
                pass
        return ctypes.windll.user32.CallNextHookEx(
            _hook_mouse, nCode, wParam, lParam
        )

    @_KeyboardProc
    def _physical_keyboard_proc(nCode, wParam, lParam):
        global _last_physical_keyboard
        if nCode >= 0 and lParam:
            try:
                data = ctypes.cast(lParam, ctypes.POINTER(_KBDLLHOOKSTRUCT)).contents
                if not (int(data.flags) & _LLKHF_INJECTED):
                    with _physical_lock:
                        _last_physical_keyboard = time.monotonic()
            except Exception:
                pass
        return ctypes.windll.user32.CallNextHookEx(
            _hook_keyboard, nCode, wParam, lParam
        )

    def _physical_hook_loop():
        global _hook_mouse, _hook_keyboard
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        module = kernel32.GetModuleHandleW(None)
        _hook_mouse = user32.SetWindowsHookExW(
            _WH_MOUSE_LL, _physical_mouse_proc, module, 0
        )
        _hook_keyboard = user32.SetWindowsHookExW(
            _WH_KEYBOARD_LL, _physical_keyboard_proc, module, 0
        )
        msg = ctypes.wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
        if _hook_mouse:
            user32.UnhookWindowsHookEx(_hook_mouse)
        if _hook_keyboard:
            user32.UnhookWindowsHookEx(_hook_keyboard)

    def _start_physical_input_monitor():
        """Start the low-level physical input hooks (idempotent)."""
        global _hook_thread
        if _hook_thread is not None:
            return
        _hook_thread = threading.Thread(
            target=_physical_hook_loop, name="d2s-physical-input", daemon=True
        )
        _hook_thread.start()

    def _physical_mouse_active(timeout: float = 2.0) -> bool:
        with _physical_lock:
            return (time.monotonic() - _last_physical_mouse) < float(timeout)

    def _physical_keyboard_active(timeout: float = 2.0) -> bool:
        with _physical_lock:
            return (time.monotonic() - _last_physical_keyboard) < float(timeout)

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
    def _physical_mouse_active(timeout: float = 2.0) -> bool: return False
    def _physical_keyboard_active(timeout: float = 2.0) -> bool: return False
