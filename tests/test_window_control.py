import ctypes
from types import SimpleNamespace

import pytest

from viewer import window_control


pytestmark = pytest.mark.skipif(
    window_control.OS_NAME != "Windows",
    reason="Windows display-affinity API",
)


def test_hide_window_from_capture_uses_full_native_hwnd(monkeypatch):
    hwnd = 0x123456789ABC
    calls = []

    monkeypatch.setattr(
        window_control.glfw,
        "get_win32_window",
        lambda _window: hwnd,
    )
    monkeypatch.setattr(
        window_control,
        "SetWindowDisplayAffinity",
        lambda native_hwnd, affinity: calls.append(
            (native_hwnd, affinity)
        )
        or True,
    )

    assert window_control.hide_window_from_capture(object()) is True
    assert calls == [
        (hwnd, window_control.WDA_EXCLUDEFROMCAPTURE),
    ]


def test_set_hwnd_capture_affinity_accepts_raw_hwnd(monkeypatch):
    hwnd = 0x123456789ABC
    calls = []
    monkeypatch.setattr(
        window_control,
        "SetWindowDisplayAffinity",
        lambda native_hwnd, affinity: calls.append((native_hwnd, affinity)) or True,
    )

    assert window_control.set_hwnd_capture_affinity(
        hwnd,
        window_control.WDA_EXCLUDEFROMCAPTURE,
    )
    assert calls == [(hwnd, window_control.WDA_EXCLUDEFROMCAPTURE)]


def test_capture_affinity_api_declares_windows_types():
    from ctypes import wintypes

    assert window_control.SetWindowDisplayAffinity.argtypes == [
        wintypes.HWND,
        wintypes.DWORD,
    ]
    assert window_control.SetWindowDisplayAffinity.restype is wintypes.BOOL


def test_cursor_bitmap_size_uses_full_color_height_and_half_monochrome_mask():
    color = SimpleNamespace(hbmColor=1)
    monochrome = SimpleNamespace(hbmColor=None)
    bitmap = SimpleNamespace(bmWidth=48, bmHeight=64)

    assert window_control._cursor_bitmap_size(color, bitmap) == (48, 64)
    assert window_control._cursor_bitmap_size(monochrome, bitmap) == (48, 32)


@pytest.mark.parametrize(
    ("mode", "expected_size"),
    [
        ("Half-SBS", (24, 32)),
        ("Full-SBS", (48, 32)),
        ("Half-TAB", (48, 16)),
        ("Full-TAB", (48, 32)),
    ],
)
def test_cursor_eye_size_matches_stereo_packing(mode, expected_size):
    assert window_control._cursor_eye_size(48, 32, mode) == expected_size


def test_draw_icon_ex_declares_user32_types():
    from ctypes import wintypes

    assert window_control.user32.DrawIconEx.argtypes == [
        wintypes.HDC,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.HICON,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.UINT,
        wintypes.HBRUSH,
        wintypes.DWORD,
    ]
    assert window_control.user32.DrawIconEx.restype is wintypes.BOOL


@pytest.mark.parametrize(
    ("hide_cursor_overlay", "expected_cursor_mode"),
    [
        (True, 0x00034003),
        (False, 0x00034001),
    ],
)
def test_mouse_passthrough_hides_and_restores_the_window_cursor(
    monkeypatch,
    hide_cursor_overlay,
    expected_cursor_mode,
):
    window = object()
    calls = []
    monkeypatch.setattr(window_control.glfw, "get_win32_window", lambda _window: 123)
    monkeypatch.setattr(window_control.glfw, "MOUSE_PASSTHROUGH", 7, raising=False)
    monkeypatch.setattr(window_control.glfw, "CURSOR", 8, raising=False)
    monkeypatch.setattr(window_control.glfw, "CURSOR_HIDDEN", 0x00034002, raising=False)
    monkeypatch.setattr(window_control.glfw, "CURSOR_NORMAL", 0x00034001, raising=False)
    monkeypatch.setattr(
        window_control,
        "_set_native_cursor_suppressed",
        lambda value: calls.append(("system", bool(value))),
    )
    monkeypatch.setattr(
        window_control.glfw,
        "set_input_mode",
        lambda target, attr, mode: calls.append(("cursor", target, attr, mode)),
    )
    monkeypatch.setattr(
        window_control.glfw,
        "set_window_attrib",
        lambda target, attr, value: calls.append(("passthrough", target, attr, value)),
    )

    assert window_control.set_window_mouse_passthrough(
        window,
        True,
        hide_cursor_overlay=hide_cursor_overlay,
    )
    expected = [
        ("passthrough", window, 7, 1),
        ("cursor", window, 8, expected_cursor_mode),
        ("system", bool(hide_cursor_overlay)),
    ]
    assert calls == expected


def test_mouse_passthrough_hides_and_restores_the_system_cursor_scheme(monkeypatch):
    window = object()
    calls = []
    monkeypatch.setattr(window_control.glfw, "get_win32_window", lambda _window: 123)
    monkeypatch.setattr(window_control.glfw, "MOUSE_PASSTHROUGH", 7, raising=False)
    monkeypatch.setattr(window_control.glfw, "CURSOR", 8, raising=False)
    monkeypatch.setattr(
        window_control,
        "_set_native_cursor_suppressed",
        lambda value: calls.append(("system_sized", bool(value))),
    )
    monkeypatch.setattr(window_control.glfw, "set_input_mode", lambda *_args: None)
    monkeypatch.setattr(window_control.glfw, "set_window_attrib", lambda *_args: None)

    assert window_control.set_window_mouse_passthrough(
        window, True, hide_cursor_overlay=True
    )
    assert window_control.set_window_mouse_passthrough(
        window, False, hide_cursor_overlay=False
    )
    assert calls == [
        ("system_sized", True),
        ("system_sized", False),
    ]


def test_cursor_suppression_replaces_system_cursor_images_for_overlay(monkeypatch):
    calls = []
    monkeypatch.setattr(window_control, "_cursor_hidden", False)
    monkeypatch.setattr(window_control, "_hide_calls", 0)
    monkeypatch.setattr(window_control, "_cursor_originals", {})
    monkeypatch.setattr(window_control, "_cursor_clones", {})
    monkeypatch.setattr(window_control, "_cursor_handle_map", {})
    monkeypatch.setattr(
        window_control,
        "LoadCursorW",
        lambda _instance, cursor_id: int(cursor_id.value),
    )
    monkeypatch.setattr(window_control, "CreateCursor", lambda *_args: 999)
    monkeypatch.setattr(
        window_control,
        "SetSystemCursor",
        lambda handle, cursor_id: calls.append(("set", handle, int(cursor_id))) or True,
    )
    monkeypatch.setattr(
        window_control,
        "ShowCursor",
        lambda visible: calls.append(("show", bool(visible))) or (-1 if not visible else 0),
    )
    monkeypatch.setattr(
        window_control,
        "user32",
        SimpleNamespace(
            CopyIcon=lambda handle: int(handle) + 1000,
            DestroyIcon=lambda handle: calls.append(("destroy", int(handle))) or True,
        ),
    )

    window_control._set_native_cursor_suppressed(True)
    assert window_control.native_cursor_suppressed() is True
    assert sum(call[0] == "set" for call in calls) == len(window_control.SYSTEM_CURSOR_IDS)
    assert all(call[1] == 999 for call in calls if call[0] == "set")
    window_control._set_native_cursor_suppressed(False)
    assert window_control.native_cursor_suppressed() is False
    assert sum(call[0] == "set" for call in calls) == 2 * len(window_control.SYSTEM_CURSOR_IDS)


def test_overlay_registers_only_its_two_capture_excluded_windows(monkeypatch):
    overlay = object.__new__(window_control.StereoCursorOverlay)
    overlay.window_handles = [11, 22]
    calls = []
    monkeypatch.setattr(
        window_control,
        "set_hwnd_capture_affinity",
        lambda hwnd, affinity: calls.append((hwnd, affinity)) or True,
    )

    overlay.update_capture_policy()

    assert calls == []
