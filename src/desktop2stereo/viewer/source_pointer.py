"""Absolute pointer mapping for the Local Viewer's captured source."""

from __future__ import annotations

import ctypes
import os
from ctypes import wintypes


def source_pointer_point(
    u: float,
    v: float,
    *,
    monitor_rect: tuple[int, int, int, int] | None = None,
    window_client_rect: tuple[int, int, int, int] | None = None,
) -> tuple[int, int] | None:
    rect = window_client_rect or monitor_rect
    if rect is None or len(rect) != 4:
        return None
    left, top, width, height = (int(value) for value in rect)
    if width <= 0 or height <= 0:
        return None
    return (
        left + round(max(0.0, min(1.0, float(u))) * max(0, width - 1)),
        top + round(max(0.0, min(1.0, float(v))) * max(0, height - 1)),
    )


def send_window_pointer_move(hwnd: int, u: float, v: float) -> bool:
    if os.name != "nt" or not hwnd:
        return False
    try:
        import win32gui

        left, top, right, bottom = win32gui.GetClientRect(hwnd)
        point = source_pointer_point(
            u, v, window_client_rect=(left, top, right - left, bottom - top)
        )
        if point is None:
            return False
        x, y = point
        lparam = (int(y) << 16) | (int(x) & 0xFFFF)
        win32gui.PostMessage(hwnd, 0x0200, 0, lparam)
        return True
    except Exception:
        return False


def send_screen_pointer_move(screen_point: tuple[int, int] | None) -> bool:
    if os.name != "nt" or screen_point is None:
        return False
    try:
        user32 = ctypes.windll.user32
        user32.WindowFromPoint.argtypes = [wintypes.POINT]
        user32.WindowFromPoint.restype = wintypes.HWND
        user32.ScreenToClient.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
        user32.ScreenToClient.restype = wintypes.BOOL
        point = wintypes.POINT(int(screen_point[0]), int(screen_point[1]))
        hwnd = int(user32.WindowFromPoint(point) or 0)
        if not hwnd:
            return False
        client = wintypes.POINT(point.x, point.y)
        user32.ScreenToClient(hwnd, ctypes.byref(client))
        lparam = (int(client.y) << 16) | (int(client.x) & 0xFFFF)
        user32.PostMessageW(hwnd, 0x0200, 0, lparam)
        return True
    except Exception:
        return False


def send_window_pointer_click(
    hwnd: int,
    u: float,
    v: float,
    *,
    button: int = 0,
) -> bool:
    if os.name != "nt" or not hwnd:
        return False
    try:
        import win32gui
    except Exception:
        return False
    try:
        left, top, right, bottom = win32gui.GetClientRect(hwnd)
        point = source_pointer_point(
            u,
            v,
            window_client_rect=(left, top, right - left, bottom - top),
        )
        if point is None:
            return False
        x, y = point
        lparam = (int(y) << 16) | (int(x) & 0xFFFF)
        move = 0x0200
        down, up = (0x0201, 0x0202) if int(button) == 0 else (0x0204, 0x0205)
        win32gui.PostMessage(hwnd, move, 0, lparam)
        win32gui.PostMessage(hwnd, down, 1 if button == 0 else 2, lparam)
        win32gui.PostMessage(hwnd, up, 0, lparam)
        return True
    except Exception:
        return False


def send_screen_pointer_click(
    screen_point: tuple[int, int] | None,
    *,
    button: int = 0,
) -> bool:
    if os.name != "nt" or screen_point is None:
        return False
    try:
        user32 = ctypes.windll.user32
        current = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(current))
        user32.SetCursorPos(int(screen_point[0]), int(screen_point[1]))
        down, up = (0x0002, 0x0004) if int(button) == 0 else (0x0008, 0x0010)
        user32.mouse_event(down, 0, 0, 0, 0)
        user32.mouse_event(up, 0, 0, 0, 0)
        user32.SetCursorPos(int(current.x), int(current.y))
        return True
    except Exception:
        return False
