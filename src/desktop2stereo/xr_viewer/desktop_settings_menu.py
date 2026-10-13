"""Flet desktop mirror for the in-headset OpenXR settings menu.

The in-headset menu remains the source of truth. This module only mirrors its
snapshots in a Flet desktop window and returns physical mouse actions through
the existing action queue. The floating launcher uses the same Flet window.
"""

from __future__ import annotations

import asyncio
import ctypes
from ctypes import wintypes
from io import BytesIO
import math
import multiprocessing
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from gui.localization import gettext_for, normalize_locale
from .settings_menu import (
    OPENXR_MENU_COLORS,
    OPENXR_MENU_GROUP_LABELS,
    OPENXR_MENU_RADII,
    _MENU_TITLE_CARD_GAP,
    _MENU_TITLE_HEIGHT,
)


DESKTOP_SETTINGS_ICON_TRANSPARENT_COLOR = "#010101"
DESKTOP_SETTINGS_ICON_OPACITY = 0.40
DESKTOP_SETTINGS_ICON_SIZE = (51, 57)
DESKTOP_SETTINGS_ICON_IMAGE_SIZE = (42, 42)
_FLET_PANEL_SIZE = (760, 650)
_FLET_PANEL_POLL_SECONDS = 0.08
_FLET_DRAWER_WIDTH = 192
_FLET_DRAWER_TAB_WIDTH = 12
_FLET_DRAWER_HIDE_DELAY_SECONDS = 0.3
_FLET_SIDEBAR_WIDTH = 144
_FLET_SPACING = 8
_FLET_BUTTON_HEIGHT = 40
_FLET_SECONDARY_BUTTON_HEIGHT = 32
_FLET_STEP_BUTTON_SIZE = 40
_FLET_FONT_ASSETS_DIR = Path(__file__).resolve().parent / "fonts"


class _JobObjectBasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _JobObjectIoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JobObjectExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JobObjectBasicLimitInformation),
        ("IoInfo", _JobObjectIoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _JobObjectBasicAccountingInformation(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", ctypes.c_longlong),
        ("TotalKernelTime", ctypes.c_longlong),
        ("ThisPeriodTotalUserTime", ctypes.c_longlong),
        ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
        ("TotalPageFaultCount", wintypes.DWORD),
        ("TotalProcesses", wintypes.DWORD),
        ("ActiveProcesses", wintypes.DWORD),
        ("TotalTerminatedProcesses", wintypes.DWORD),
    ]


class _WindowsKillOnCloseJob:
    """Own the Flet worker and all of its descendants as one Windows job."""

    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
    _JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION = 1
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _PROCESS_SET_QUOTA = 0x0100
    _PROCESS_TERMINATE = 0x0001

    def __init__(self) -> None:
        if os.name != "nt":
            raise OSError("Windows process jobs are only available on Windows")
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self._kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        self._kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            wintypes.INT,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        self._kernel32.SetInformationJobObject.restype = wintypes.BOOL
        self._kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self._kernel32.OpenProcess.restype = wintypes.HANDLE
        self._kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self._kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        self._kernel32.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE,
            wintypes.INT,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self._kernel32.QueryInformationJobObject.restype = wintypes.BOOL
        self._kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self._kernel32.TerminateJobObject.restype = wintypes.BOOL
        self._kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self._kernel32.CloseHandle.restype = wintypes.BOOL

        self._handle = self._kernel32.CreateJobObjectW(None, None)
        if not self._handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = _JobObjectExtendedLimitInformation()
        limits.BasicLimitInformation.LimitFlags = self._JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self._kernel32.SetInformationJobObject(
            self._handle,
            self._JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ):
            error = ctypes.WinError(ctypes.get_last_error())
            self._kernel32.CloseHandle(self._handle)
            self._handle = None
            raise error

    def assign(self, process_id: int) -> bool:
        process_handle = self._kernel32.OpenProcess(
            self._PROCESS_SET_QUOTA | self._PROCESS_TERMINATE,
            False,
            int(process_id),
        )
        if not process_handle:
            return False
        try:
            return bool(
                self._kernel32.AssignProcessToJobObject(
                    self._handle,
                    process_handle,
                )
            )
        finally:
            self._kernel32.CloseHandle(process_handle)

    def _active_process_count(self) -> int | None:
        info = _JobObjectBasicAccountingInformation()
        if not self._kernel32.QueryInformationJobObject(
            self._handle,
            self._JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
            None,
        ):
            return None
        return int(info.ActiveProcesses)

    def close(self) -> None:
        handle = self._handle
        if not handle:
            return
        try:
            active = self._active_process_count()
            if active:
                self._kernel32.TerminateJobObject(handle, 1)
                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline:
                    active = self._active_process_count()
                    if active is None or active == 0:
                        break
                    time.sleep(0.05)
        finally:
            self._kernel32.CloseHandle(handle)
            self._handle = None


def _menu_color(token: str) -> str:
    return OPENXR_MENU_COLORS[token]


def _compact_flet_dimension(value: int) -> int:
    """Fit desktop menu controls into the next compact Flet size step."""
    scaled = float(value) * 0.75
    return max(32, int(math.floor(scaled / 8.0 + 0.5) * 8))


def _is_reset_control(key: str) -> bool:
    return (
        key.startswith("reset:")
        or key in {"screen:reset_crop", "section:reset_defaults"}
    )


def _make_step_button_style(ft: Any) -> Any:
    state = ft.ControlState
    return ft.ButtonStyle(
        bgcolor={
            state.DEFAULT: _menu_color("surface_container_high"),
            state.HOVERED: _menu_color("primary_hover"),
            state.PRESSED: _menu_color("primary"),
            state.DISABLED: _menu_color("surface_container"),
        },
        color={
            state.DEFAULT: _menu_color("text_primary"),
            state.HOVERED: _menu_color("primary_state_text"),
            state.PRESSED: _menu_color("primary_state_text"),
            state.DISABLED: _menu_color("text_disabled"),
        },
        padding=ft.Padding.symmetric(horizontal=0, vertical=0),
        elevation=0,
        text_style=ft.TextStyle(size=16, weight=ft.FontWeight.W_600),
        shape=ft.RoundedRectangleBorder(radius=20),
        alignment=ft.Alignment.CENTER,
    )


def _make_step_button_label(ft: Any, key: str) -> Any:
    symbol = "-" if ":minus:" in key else "+"
    return ft.Text(symbol, size=16, text_align=ft.TextAlign.CENTER)


def _quit_flet_when_parent_exits(parent_process: Any, commands: Any) -> None:
    """Close the separate Flet process if a native runtime crash skips stop()."""
    parent_process.join()
    try:
        commands.put_nowait("__quit__")
    except Exception:
        pass


def _watch_flet_parent(commands: Any) -> None:
    parent_process = multiprocessing.parent_process()
    if parent_process is None:
        return
    threading.Thread(
        target=_quit_flet_when_parent_exits,
        args=(parent_process, commands),
        name="desktop-settings-parent-watch",
        daemon=True,
    ).start()


def _stop_flet_process(process: Any) -> None:
    if process is None or process.pid is None:
        return
    process.join(timeout=2.0)
    if not process.is_alive():
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/f", "/t", "/pid", str(process.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=2.0,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError):
            pass
        process.join(timeout=1.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)


def _icon_geometry_for_monitor(
    monitor_rect: tuple[int, int, int, int] | None,
    *,
    fallback_size: tuple[int, int],
    pixel_ratio: float = 1.0,
) -> tuple[int, int, int, int]:
    """Return the right-edge drawer ``(x, y, width, height)`` inside the input monitor."""
    icon_width, icon_height = _FLET_DRAWER_WIDTH, DESKTOP_SETTINGS_ICON_SIZE[1]
    if monitor_rect is None:
        monitor_left, monitor_top = 0, 0
        monitor_width, monitor_height = fallback_size
    else:
        monitor_left, monitor_top, monitor_width, monitor_height = monitor_rect
    ratio = float(pixel_ratio or 1.0)
    monitor_left, monitor_top, monitor_width, monitor_height = (
        round(value / ratio)
        for value in (monitor_left, monitor_top, monitor_width, monitor_height)
    )
    icon_x = monitor_left + max(0, int(monitor_width) - icon_width)
    icon_y = monitor_top + max(0, (int(monitor_height) - icon_height) // 2)
    return icon_x, icon_y, icon_width, icon_height


def _windows_flet_window_handle() -> int | None:
    """Find this worker's Flet client window without targeting another runtime."""
    if os.name != "nt":
        return None
    from gui.flet_runtime import _flet_process_ids, _windows_process_snapshot

    client_pids = set(_flet_process_ids(_windows_process_snapshot(), os.getpid()))
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    found = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def visit(hwnd: int, _data: int) -> bool:
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value in client_pids:
            name = ctypes.create_unicode_buffer(128)
            user32.GetClassNameW(hwnd, name, len(name))
            if name.value == "FLUTTER_RUNNER_WIN32_WINDOW":
                found.append(hwnd)
                return False
        return True

    user32.EnumWindows(visit, 0)
    return found[0] if found else None


def _clip_windows_drawer(hwnd: int | None, *, expanded: bool, pixel_ratio: float) -> None:
    """Restrict the collapsed native window's drawing and hit area to its tab."""
    if hwnd is None:
        return
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.SetWindowRgn.argtypes = [wintypes.HWND, wintypes.HANDLE, wintypes.BOOL]
    region = None
    if not expanded:
        gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
        gdi32.CreateRectRgn.argtypes = [ctypes.c_int] * 4
        gdi32.CreateRectRgn.restype = wintypes.HANDLE
        gdi32.DeleteObject.argtypes = [wintypes.HANDLE]
        region = gdi32.CreateRectRgn(
            round((_FLET_DRAWER_WIDTH - _FLET_DRAWER_TAB_WIDTH) * pixel_ratio),
            0,
            round(_FLET_DRAWER_WIDTH * pixel_ratio),
            round(DESKTOP_SETTINGS_ICON_SIZE[1] * pixel_ratio),
        )
        if not region:
            raise ctypes.WinError(ctypes.get_last_error())
    if not user32.SetWindowRgn(hwnd, region, True):
        error = ctypes.get_last_error()
        if region is not None:
            gdi32.DeleteObject(region)
        raise ctypes.WinError(error)
    # Windows owns a successfully assigned region and releases it on replacement.


def _primary_screen_size() -> tuple[int, int]:
    """Return the primary screen size for launcher placement without Tk."""
    if os.name == "nt":
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.GetSystemMetrics.argtypes = [ctypes.c_int]
        user32.GetSystemMetrics.restype = ctypes.c_int
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    return 1920, 1080


def _flet_panel_position_for_monitor(
    monitor_rect: tuple[int, int, int, int] | None,
    *,
    pixel_ratio: float = 1.0,
) -> tuple[int, int] | None:
    """Return the centered Flet panel position for the selected input monitor."""
    if monitor_rect is None:
        return None
    try:
        monitor_left, monitor_top, monitor_width, monitor_height = (
            int(value) for value in monitor_rect
        )
    except (TypeError, ValueError):
        return None
    if monitor_width <= 0 or monitor_height <= 0:
        return None
    ratio = float(pixel_ratio or 1.0)
    monitor_left, monitor_top, monitor_width, monitor_height = (
        round(value / ratio)
        for value in (monitor_left, monitor_top, monitor_width, monitor_height)
    )
    panel_width, panel_height = _FLET_PANEL_SIZE
    return (
        monitor_left + max(0, (monitor_width - panel_width) // 2),
        monitor_top + max(0, (monitor_height - panel_height) // 2),
    )


def desktop_settings_menu_enabled() -> bool:
    value = os.environ.get("D2S_DESKTOP_SETTINGS_MENU", "1")
    return value.strip().lower() not in {"0", "false", "off", "no", "disabled"}


def _drain_latest(source: Any) -> Any | None:
    latest = None
    while True:
        try:
            latest = source.get_nowait()
        except queue.Empty:
            return latest


def _snapshot_layout_signature(snapshot: dict[str, Any]) -> tuple[Any, ...]:
    """Identify structural changes without treating live values as a rebuild."""
    controls = tuple(snapshot.get("controls") or ())
    values = dict(snapshot.get("values") or {})
    return (
        str(snapshot.get("tab") or "picture"),
        normalize_locale(snapshot.get("lang", "EN")),
        tuple(
            (
                str(control.key),
                str(control.label),
                str(control.kind),
                float(control.minimum),
                float(control.maximum),
                float(control.step),
                bool(control.enabled),
                str(getattr(control, "group", "")),
                bool(getattr(control, "fixed", False)),
            )
            for control in controls
        ),
        tuple(
            (str(group.key), str(group.title))
            for group in snapshot.get("groups", ())
        ),
        tuple(
            (str(control.key), _control_is_selected(str(control.key), values))
            for control in controls
            if str(control.kind) in {"button", "toggle"}
        ),
    )


def _slider_divisions(control: Any) -> int | None:
    span = float(control.maximum) - float(control.minimum)
    step = float(control.step)
    if span <= 0.0 or step <= 0.0:
        return None
    divisions = int(round(span / step))
    return divisions if 0 < divisions <= 500 else None


def _bounded_slider_value(value: float, minimum: float, maximum: float) -> float:
    """Keep live snapshots valid for Flet's strict Slider range validation."""
    if not math.isfinite(value):
        return float(minimum)
    return min(max(value, float(minimum)), float(maximum))


def _format_value(value: float, step: float, key: str = "") -> str:
    if key in {"screen:crop_width", "screen:crop_height"}:
        return f"{value:.0f}% each"
    step = abs(float(step))
    if step >= 1.0:
        return f"{value:.0f}"
    if step >= 0.1:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _toggle_value_for(key: str, values: dict[str, Any]) -> bool:
    """Map shared menu actions to the setting value shown by their switches."""
    if key == "depth:toggle_stereo":
        try:
            return float(values.get("depth_strength", 0.0)) > 0.0
        except (TypeError, ValueError):
            return False
    if key == "depth:toggle_cross_eyed":
        return bool(values.get("cross_eyed", False))
    if key == "room:toggle_screen_reflection":
        return bool(values.get("room:screen_reflection_enabled", True))
    if key == "screen:dynamic_crop":
        return bool(values.get("screen:dynamic_crop", False))
    return bool(values.get(key, False))


def _control_is_selected(key: str, values: dict[str, Any]) -> bool:
    """Return the active state for Meta selectable buttons and mode tiles."""
    if key == "openxr:render_auto":
        return bool(values.get("openxr_render_auto", False))
    if key.startswith("screen:section:"):
        return key.rsplit(":", 1)[1] == str(values.get("screen:section", "layout"))
    if key.startswith("screen:type:"):
        target_angles = {
            "screen:type:flat": 0.0,
            "screen:type:subtle": math.radians(20.0),
            "screen:type:medium": math.radians(30.0),
            "screen:type:deep": 0.72,
        }
        try:
            angle = float(values.get("screen:curve_half_angle", 0.0))
        except (TypeError, ValueError):
            return False
        return abs(angle - target_angles[key]) < 1e-3
    if key.startswith("glow:"):
        return key == f"glow:{values.get('glow:mode', 'off')}"
    if key.startswith("room:model:"):
        return key == f"room:model:{values.get('room:model', 'Default')}"
    if key.startswith("room:seat:"):
        try:
            seat_index = int(values.get("room:seat_index", 0)) % 3
        except (TypeError, ValueError):
            seat_index = 0
        return key == f"room:seat:{('front', 'middle', 'back')[seat_index]}"
    return False


def _screen_button_row_group(key: str) -> str | None:
    """Return the Flet row group for a screen-tab button, if applicable."""
    if key.startswith("screen:type:"):
        return "screen_curveness"
    if key.startswith("screen:rotate:"):
        return "screen_rotation"
    if key.startswith("screen:section:"):
        return "screen_section"
    return None


def _button_row_group(key: str) -> str | None:
    """Return the compact Flet row group matching the headset menu layout."""
    if key.startswith("reset:"):
        return key.split(":", 1)[1]
    if key == "section:reset_defaults":
        return "screen_placement"
    if key == "screen:reset_crop":
        return "screen_crop"
    screen_group = _screen_button_row_group(key)
    if screen_group is not None:
        return screen_group
    if key in {"depth:toggle_stereo", "depth:toggle_cross_eyed"}:
        return "depth_stereo"
    if key in {"glow:surround", "glow:glow", "glow:veil", "glow:off"}:
        return "glow_modes"
    if key.startswith("room:model:"):
        return "room_models"
    if key.startswith("room:seat:"):
        return "room_seats"
    if key == "room:toggle_screen_reflection":
        return "room_scene"
    return None


def _run_flet_desktop_settings_app(
    snapshots: Any,
    actions: Any,
    commands: Any,
    input_monitor_rect: tuple[int, int, int, int] | None,
    start_gate: Any | None = None,
) -> None:
    """Run Flet in its own process because Flet owns the main-thread signals."""
    if start_gate is not None and not start_gate.wait(timeout=5.0):
        print(
            "[DesktopSettings] Flet process ownership was not established; "
            "the settings window will not start.",
            flush=True,
        )
        return
    _watch_flet_parent(commands)
    try:
        from gui.flet_runtime import ensure_vendored_flet_view

        ensure_vendored_flet_view()
        import flet as ft
    except Exception as exc:
        print(
            "[DesktopSettings] Flet settings window failed to start: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        return

    async def main(page: Any) -> None:
        locale = "EN"

        def translate(message: str) -> str:
            return gettext_for(locale, message)

        page.title = translate("Desktop2Stereo OpenXR Settings")
        page.padding = 0
        page.spacing = 0
        page.bgcolor = ft.Colors.TRANSPARENT
        page.fonts = {"Inter": "InterVariable.ttf"}
        page.theme = ft.Theme(
            color_scheme_seed=_menu_color("primary"),
            font_family="Inter",
        )
        page.theme_mode = ft.ThemeMode.DARK
        icon_path = Path(__file__).resolve().parents[1] / "icon2.ico"
        if icon_path.is_file():
            page.window.icon = str(icon_path)

        def window_pixel_ratio() -> float:
            # Windows capture rectangles are physical pixels; Flet places
            # native windows in logical pixels. macOS monitor bounds use points.
            return float(page.media.device_pixel_ratio or 1.0) if os.name == "nt" else 1.0

        icon_geometry = _icon_geometry_for_monitor(
            input_monitor_rect,
            fallback_size=_primary_screen_size(),
            pixel_ratio=window_pixel_ratio(),
        )
        page.window.width = _FLET_DRAWER_WIDTH
        page.window.height = DESKTOP_SETTINGS_ICON_SIZE[1]
        page.window.left, page.window.top = icon_geometry[:2]
        page.window.min_width = 1
        page.window.min_height = 1
        page.window.maximizable = False
        page.window.resizable = False
        # The bundled Windows window_manager has no setMovable implementation.
        # Sending it aborts the remaining Window updates, including frameless
        # and prevent_close, leaving the icon's drawing and hit area offset.
        page.window.movable = None if os.name == "nt" else False
        page.window.always_on_top = True
        page.window.skip_task_bar = True
        page.window.frameless = True
        page.window.title_bar_hidden = True
        page.window.title_bar_buttons_hidden = True
        page.window.opacity = DESKTOP_SETTINGS_ICON_OPACITY
        page.window.bgcolor = ft.Colors.TRANSPARENT
        page.window.prevent_close = True
        page.window.visible = True

        body_content = ft.Column(
            controls=[
                ft.Text(
                    translate("Waiting for OpenXR settings..."),
                    color=_menu_color("text_secondary"),
                    size=14,
                )
            ],
            expand=True,
            scroll=ft.ScrollMode.AUTO,
            spacing=_FLET_SPACING,
        )
        body = ft.Container(
            content=body_content,
            expand=True,
            padding=ft.Padding.only(top=8),
        )
        sidebar = ft.Column(expand=True, spacing=_FLET_SPACING)
        section_toolbar = ft.Container(height=0)
        main_column = ft.Column(
            controls=[section_toolbar, body],
            expand=True,
            spacing=0,
        )
        main_area = ft.Container(
            content=main_column,
            expand=True,
            padding=ft.Padding.only(right=8, top=8, bottom=8),
        )
        layout = ft.Row(
            controls=[
                ft.Container(
                    width=_FLET_SIDEBAR_WIDTH,
                    expand=False,
                    bgcolor=_menu_color("surface_container"),
                    border_radius=OPENXR_MENU_RADII["group"],
                    padding=8,
                    content=sidebar,
                ),
                main_area,
            ],
            spacing=_FLET_SPACING,
            expand=True,
            vertical_alignment=ft.CrossAxisAlignment.STRETCH,
        )
        panel_view = ft.Container(
            expand=True,
            bgcolor=_menu_color("surface"),
            padding=8,
            content=layout,
        )

        active_monitor_rect = input_monitor_rect
        visible = False
        drawer_expanded = False
        drawer_hide_task: asyncio.Task | None = None
        native_window_handle = None

        def cancel_drawer_hide() -> None:
            nonlocal drawer_hide_task
            if drawer_hide_task is not None:
                drawer_hide_task.cancel()
                drawer_hide_task = None

        def apply_drawer_region() -> None:
            nonlocal native_window_handle
            if os.name == "nt":
                if native_window_handle is None:
                    native_window_handle = _windows_flet_window_handle()
                _clip_windows_drawer(
                    native_window_handle,
                    expanded=visible or drawer_expanded,
                    pixel_ratio=window_pixel_ratio(),
                )
            elif not visible:
                geometry = _icon_geometry_for_monitor(
                    active_monitor_rect,
                    fallback_size=_primary_screen_size(),
                    pixel_ratio=window_pixel_ratio(),
                )
                width = _FLET_DRAWER_WIDTH if drawer_expanded else _FLET_DRAWER_TAB_WIDTH
                page.window.width = width
                page.window.left = geometry[0] + _FLET_DRAWER_WIDTH - width
                launcher_view.width = width
                launcher_view.content.width = width

        def set_drawer_expanded(expanded: bool) -> None:
            nonlocal drawer_expanded
            drawer_expanded = expanded
            drawer_body.visible = expanded
            drawer_handle.visible = not expanded
            page.window.opacity = 1.0 if expanded else DESKTOP_SETTINGS_ICON_OPACITY
            apply_drawer_region()
            page.window.update()
            page.update()

        async def hide_drawer_after_leave() -> None:
            await asyncio.sleep(_FLET_DRAWER_HIDE_DELAY_SECONDS)
            if not visible and page.window.visible and drawer_expanded:
                set_drawer_expanded(False)

        async def on_launcher_hover(event: Any) -> None:
            nonlocal drawer_hide_task
            if visible:
                return
            cancel_drawer_hide()
            if event.data is True or event.data == "true":
                if not drawer_expanded:
                    set_drawer_expanded(True)
            elif drawer_expanded:
                drawer_hide_task = asyncio.create_task(hide_drawer_after_leave())

        def request_panel_toggle(_event: Any) -> None:
            try:
                commands.put_nowait("__toggle_flet__")
            except Exception:
                pass

        # Flutter does not decode ICO; convert the bundled app icon once in memory.
        from PIL import Image

        icon_bytes = BytesIO()
        with Image.open(icon_path) as icon_image:
            icon_image.save(icon_bytes, format="PNG")
        launcher_label = ft.Text(
            translate("OpenXR Settings"),
            size=12,
            color=_menu_color("text_primary"),
            no_wrap=True,
        )
        drawer_body = ft.Container(
            width=_FLET_DRAWER_WIDTH,
            height=DESKTOP_SETTINGS_ICON_SIZE[1],
            bgcolor=_menu_color("surface_container_high"),
            border_radius=ft.BorderRadius.only(top_left=12, bottom_left=12),
            padding=ft.Padding.symmetric(horizontal=8, vertical=6),
            visible=False,
            content=ft.Row(
                controls=[
                    ft.Image(
                        src=icon_bytes.getvalue(),
                        width=DESKTOP_SETTINGS_ICON_IMAGE_SIZE[0],
                        height=DESKTOP_SETTINGS_ICON_IMAGE_SIZE[1],
                        fit=ft.BoxFit.CONTAIN,
                    ),
                    launcher_label,
                ],
                spacing=8,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        )
        drawer_handle = ft.Container(
            right=0,
            width=_FLET_DRAWER_TAB_WIDTH,
            height=DESKTOP_SETTINGS_ICON_SIZE[1],
            bgcolor=_menu_color("primary"),
            border_radius=ft.BorderRadius.only(top_left=8, bottom_left=8),
            alignment=ft.Alignment.CENTER,
            content=ft.Container(width=3, height=20, border_radius=2, bgcolor=_menu_color("text_primary")),
        )
        launcher_view = ft.Container(
            width=_FLET_DRAWER_WIDTH,
            height=DESKTOP_SETTINGS_ICON_SIZE[1],
            bgcolor=ft.Colors.TRANSPARENT,
            on_hover=on_launcher_hover,
            on_click=request_panel_toggle,
            content=ft.Stack(
                controls=[drawer_body, drawer_handle],
                width=_FLET_DRAWER_WIDTH,
                height=DESKTOP_SETTINGS_ICON_SIZE[1],
                clip_behavior=ft.ClipBehavior.HARD_EDGE,
            ),
        )
        page.add(launcher_view)

        slider_widgets: dict[str, tuple[Any, Any, float]] = {}
        step_widgets: dict[str, tuple[Any, Any]] = {}
        toggle_widgets: dict[str, Any] = {}
        layout_signature: tuple[Any, ...] | None = None

        async def set_window_mode(show_panel: bool) -> None:
            nonlocal visible, drawer_expanded
            cancel_drawer_hide()
            visible = show_panel
            drawer_expanded = False
            drawer_body.visible = False
            drawer_handle.visible = True
            if show_panel:
                apply_drawer_region()
            # Apply minimum constraints before shrinking the native window.
            # Flet's Window service otherwise resizes against the old minimum.
            page.window.min_width = 580 if show_panel else 1
            page.window.min_height = 420 if show_panel else 1
            page.window.update()
            await page.window.wait_until_ready_to_show()
            page.controls.clear()
            if show_panel:
                page.add(panel_view)
                page.bgcolor = _menu_color("surface")
                page.window.width = _FLET_PANEL_SIZE[0]
                page.window.height = _FLET_PANEL_SIZE[1]
                page.window.resizable = True
                page.window.movable = None if os.name == "nt" else True
                page.window.skip_task_bar = False
                page.window.frameless = False
                page.window.title_bar_hidden = False
                page.window.title_bar_buttons_hidden = False
                page.window.opacity = 1.0
                page.window.bgcolor = _menu_color("surface")
                monitor_rect = active_monitor_rect or (
                    0,
                    0,
                    *_primary_screen_size(),
                )
                panel_position = _flet_panel_position_for_monitor(
                    monitor_rect, pixel_ratio=window_pixel_ratio()
                )
                if panel_position is not None:
                    page.window.left, page.window.top = panel_position
                page.window.visible = True
                page.window.focused = True
            else:
                page.add(launcher_view)
                page.bgcolor = ft.Colors.TRANSPARENT
                page.window.width = _FLET_DRAWER_WIDTH
                launcher_view.width = _FLET_DRAWER_WIDTH
                page.window.height = DESKTOP_SETTINGS_ICON_SIZE[1]
                page.window.resizable = False
                page.window.movable = None if os.name == "nt" else False
                page.window.skip_task_bar = True
                page.window.frameless = True
                page.window.title_bar_hidden = True
                page.window.title_bar_buttons_hidden = True
                page.window.opacity = DESKTOP_SETTINGS_ICON_OPACITY
                page.window.bgcolor = ft.Colors.TRANSPARENT
                geometry = _icon_geometry_for_monitor(
                    active_monitor_rect,
                    fallback_size=_primary_screen_size(),
                    pixel_ratio=window_pixel_ratio(),
                )
                page.window.left, page.window.top = geometry[:2]
                page.window.visible = True
                apply_drawer_region()

        def queue_action(key: str, value: float | None = None) -> None:
            try:
                actions.put_nowait((key, value))
            except Exception:
                pass

        def queue_button(key: str) -> Any:
            return lambda _event: queue_action(key)

        def queue_slider(key: str, value_label: Any, step: float) -> Any:
            def on_change(event: Any) -> None:
                try:
                    value = float(event.control.value)
                except (TypeError, ValueError):
                    return
                value_label.value = _format_value(value, step, key)
                value_label.update()
                queue_action(key, value)

            return on_change

        def rebuild(snapshot: dict[str, Any]) -> None:
            nonlocal layout_signature, locale
            slider_widgets.clear()
            step_widgets.clear()
            toggle_widgets.clear()
            locale = normalize_locale(snapshot.get("lang", "EN"))
            launcher_label.value = translate("OpenXR Settings")
            page.title = translate("Desktop2Stereo OpenXR Settings")
            controls = tuple(snapshot.get("controls") or ())
            tab = str(snapshot.get("tab") or "picture")
            values = dict(snapshot.get("values") or {})
            groups = tuple(snapshot.get("groups") or ())
            group_titles = {
                str(group.key): str(group.title)
                for group in groups
            }
            controls_by_key = {str(control.key): control for control in controls}
            nav_controls = [
                item for item in controls if str(item.key).startswith("tab:")
            ]
            stop_control = next(
                (item for item in controls if str(item.key) == "runtime:stop"),
                None,
            )
            section_controls = [
                item for item in controls if str(item.key).startswith("screen:section:")
            ]

            def make_button(control: Any, *, width: int | None = None) -> Any:
                key = str(control.key)
                active = _control_is_selected(key, values)
                destructive = key == "runtime:stop"
                enabled = bool(control.enabled)
                state = ft.ControlState
                background = (
                    _menu_color("destructive") if destructive and enabled
                    else _menu_color("surface_container_high") if destructive or not active
                    else _menu_color("selection_container")
                )
                text_color = (
                    _menu_color("text_disabled") if not enabled
                    else _menu_color("destructive_text") if destructive
                    else _menu_color("selection_text") if active
                    else _menu_color("text_primary")
                )
                hover_color = (
                    _menu_color("destructive_hover") if destructive
                    else _menu_color("selection_hover_container") if active
                    else _menu_color("primary_hover")
                )
                button_height = (
                    _FLET_SECONDARY_BUTTON_HEIGHT if _is_reset_control(key) or key.startswith(
                        ("screen:section:", "screen:rotate:")
                    ) else _FLET_BUTTON_HEIGHT
                )
                button_width = (
                    width if width is not None
                    else _FLET_SIDEBAR_WIDTH - 24 if destructive
                    else None
                )
                return ft.ElevatedButton(
                    content=ft.Text(
                        translate(str(control.label)),
                        size=14,
                        weight=ft.FontWeight.W_600,
                        text_align=ft.TextAlign.CENTER,
                    ),
                    on_click=queue_button(key),
                    disabled=not enabled,
                    width=button_width,
                    height=button_height,
                    style=ft.ButtonStyle(
                        bgcolor={
                            state.DEFAULT: background,
                            state.HOVERED: hover_color,
                            state.FOCUSED: hover_color,
                            state.PRESSED: (
                                _menu_color("destructive_hover")
                                if destructive else _menu_color("primary")
                            ),
                            state.DISABLED: _menu_color("surface_container_high"),
                        },
                        color={
                            state.DEFAULT: text_color,
                            state.HOVERED: (
                                _menu_color("destructive_text")
                                if destructive else _menu_color("primary_state_text")
                            ),
                            state.FOCUSED: (
                                _menu_color("destructive_text")
                                if destructive else _menu_color("primary_state_text")
                            ),
                            state.PRESSED: (
                                _menu_color("destructive_text")
                                if destructive else _menu_color("primary_state_text")
                            ),
                            state.DISABLED: _menu_color("text_disabled"),
                        },
                        padding=ft.Padding.symmetric(horizontal=6, vertical=4),
                        elevation=0,
                        text_style=ft.TextStyle(
                            size=14,
                            weight=ft.FontWeight.W_600,
                        ),
                        shape=ft.RoundedRectangleBorder(
                            radius=OPENXR_MENU_RADII["control"],
                        ),
                        alignment=ft.Alignment.CENTER,
                    ),
                )

            def make_step_button(control: Any) -> Any:
                return ft.ElevatedButton(
                    content=_make_step_button_label(ft, str(control.key)),
                    on_click=queue_button(str(control.key)),
                    disabled=not bool(control.enabled),
                    width=_FLET_STEP_BUTTON_SIZE,
                    height=_FLET_STEP_BUTTON_SIZE,
                    style=_make_step_button_style(ft),
                )

            def nav_button(control: Any) -> Any:
                key = str(control.key)
                active = key == f"tab:{tab}"
                state = ft.ControlState
                nav_icons = {
                    "tab:screen": ft.Icons.CROP_LANDSCAPE,
                    "tab:depth": ft.Icons.LAYERS,
                    "tab:glow": ft.Icons.AUTO_AWESOME,
                    "tab:room": ft.Icons.HOME_WORK,
                    "tab:picture": ft.Icons.IMAGE,
                }
                return ft.ElevatedButton(
                    content=ft.Row(
                        controls=[
                            ft.Icon(
                                nav_icons.get(key, ft.Icons.SETTINGS),
                                size=16,
                                color=(
                                    _menu_color("selection_text")
                                    if active else _menu_color("text_secondary")
                                ),
                            ),
                            ft.Text(
                                translate(str(control.label)),
                                size=14,
                                weight=ft.FontWeight.W_600,
                                text_align=ft.TextAlign.LEFT,
                                expand=True,
                            ),
                        ],
                        spacing=6,
                        expand=True,
                        alignment=ft.MainAxisAlignment.START,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    on_click=queue_button(key),
                    width=_FLET_SIDEBAR_WIDTH - 24,
                    height=_FLET_BUTTON_HEIGHT,
                    style=ft.ButtonStyle(
                        bgcolor={
                            state.DEFAULT: (
                                _menu_color("selection_container")
                                if active else _menu_color("surface_container_high")
                            ),
                            state.HOVERED: _menu_color("selection_hover_container"),
                            state.FOCUSED: _menu_color("selection_hover_container"),
                            state.PRESSED: _menu_color("primary"),
                            state.DISABLED: _menu_color("surface_container"),
                        },
                        color={
                            state.DEFAULT: (
                                _menu_color("selection_text")
                                if active else _menu_color("text_primary")
                            ),
                            state.HOVERED: _menu_color("selection_text"),
                            state.FOCUSED: _menu_color("selection_text"),
                            state.PRESSED: _menu_color("primary_state_text"),
                            state.DISABLED: _menu_color("text_disabled"),
                        },
                        padding=ft.Padding.symmetric(horizontal=8, vertical=4),
                        elevation=0,
                        text_style=ft.TextStyle(size=14, weight=ft.FontWeight.W_600),
                        shape=ft.RoundedRectangleBorder(
                            radius=OPENXR_MENU_RADII["control"],
                        ),
                        alignment=ft.Alignment.CENTER_LEFT,
                    ),
                )

            def step_controls_for(slider_key: str) -> tuple[Any | None, Any | None]:
                minus = controls_by_key.get(f"step:minus:{slider_key}")
                plus = controls_by_key.get(f"step:plus:{slider_key}")
                return minus, plus

            slider_key_groups: dict[str, list[Any]] = {}
            for item in controls:
                key = str(item.key)
                if key.startswith(("tab:", "step:")) or key in {
                    "runtime:stop",
                } or key.startswith("screen:section:"):
                    continue
                group_key = str(getattr(item, "group", "") or "settings")
                slider_key_groups.setdefault(group_key, []).append(item)

            def render_slider(control: Any) -> Any:
                key = str(control.key)
                try:
                    current = float(values.get(key, float(control.minimum)))
                except (TypeError, ValueError):
                    current = float(control.minimum)
                current = _bounded_slider_value(
                    current, float(control.minimum), float(control.maximum)
                )
                value_label = ft.Text(
                    _format_value(current, float(control.step), key),
                    width=56,
                    size=12,
                    text_align=ft.TextAlign.RIGHT,
                    color=_menu_color("text_secondary"),
                )
                slider = ft.Slider(
                    value=current,
                    min=float(control.minimum),
                    max=float(control.maximum),
                    divisions=_slider_divisions(control),
                    active_color=_menu_color("primary"),
                    inactive_color=_menu_color("track"),
                    thumb_color=_menu_color("text_primary"),
                    overlay_color=_menu_color("primary_hover_container"),
                    on_change=queue_slider(
                        key, value_label, float(control.step)
                    ),
                    disabled=not bool(control.enabled),
                    expand=True,
                )
                minus_control, plus_control = step_controls_for(key)
                minus_button = (
                    make_step_button(minus_control)
                    if minus_control is not None else ft.Container(width=_FLET_STEP_BUTTON_SIZE)
                )
                plus_button = (
                    make_step_button(plus_control)
                    if plus_control is not None else ft.Container(width=_FLET_STEP_BUTTON_SIZE)
                )
                step_widgets[key] = (minus_button, plus_button)
                slider_widgets[key] = (
                    slider, value_label, float(control.step)
                )
                return ft.Column(
                    controls=[
                        ft.Row(
                            controls=[
                                ft.Text(
                                    translate(str(control.label)),
                                    size=16,
                                    color=_menu_color("text_primary"),
                                    text_align=ft.TextAlign.LEFT,
                                    expand=True,
                                ),
                                value_label,
                            ],
                            spacing=4,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                        ft.Row(
                            controls=[
                                minus_button,
                                slider,
                                plus_button,
                            ],
                            spacing=4,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                    ],
                    spacing=4,
                )

            content: list[Any] = []
            for group_key, group_items in slider_key_groups.items():
                reset_items = [
                    item for item in group_items
                    if _is_reset_control(str(item.key))
                ]
                items = [
                    item for item in group_items
                    if not _is_reset_control(str(item.key))
                ]
                children: list[Any] = []
                pending_buttons: list[Any] = []

                def flush_buttons() -> None:
                    if not pending_buttons:
                        return
                    buttons = list(pending_buttons)
                    pending_buttons.clear()
                    if group_key == "screen_shape":
                        content_row = ft.Row(
                            controls=buttons,
                            alignment=ft.MainAxisAlignment.CENTER,
                            spacing=_FLET_SPACING,
                            wrap=True,
                            run_spacing=_FLET_SPACING,
                        )
                    else:
                        content_row = ft.Row(
                            controls=buttons,
                            alignment=ft.MainAxisAlignment.CENTER,
                            spacing=_FLET_SPACING,
                            wrap=True,
                            run_spacing=_FLET_SPACING,
                        )
                    children.append(content_row)

                shape_controls = (
                    [item for item in items if str(item.key).startswith("screen:type:")],
                    [item for item in items if str(item.key).startswith("screen:rotate:")],
                ) if group_key == "screen_shape" else None
                if shape_controls is not None:
                    for shape_row in shape_controls:
                        row_buttons = [
                            make_button(
                                item,
                                width=_compact_flet_dimension(
                                    160 if str(item.key).startswith("screen:rotate:") else 96
                                ),
                            )
                            for item in shape_row
                        ]
                        children.append(
                            ft.Row(
                                controls=row_buttons,
                                alignment=ft.MainAxisAlignment.CENTER,
                                spacing=_FLET_SPACING,
                                wrap=True,
                                run_spacing=_FLET_SPACING,
                            )
                        )
                else:
                    button_items = [
                        item for item in items
                        if str(item.kind) not in {"slider", "toggle"}
                    ]
                    button_items_by_key = {str(item.key): item for item in button_items}
                    ordered_items: list[Any] = []
                    for item in items:
                        key = str(item.key)
                        if key in button_items_by_key:
                            ordered_items.append(item)
                        elif str(item.kind) in {"slider", "toggle"}:
                            ordered_items.append(item)
                    for item in ordered_items:
                        if str(item.kind) == "slider":
                            flush_buttons()
                            children.append(render_slider(item))
                        elif str(item.kind) == "toggle":
                            flush_buttons()
                            toggle = ft.Switch(
                                label=translate(str(item.label)),
                                label_position=ft.LabelPosition.LEFT,
                                label_text_style=ft.TextStyle(
                                    size=14,
                                    color=_menu_color("text_primary"),
                                ),
                                value=_toggle_value_for(str(item.key), values),
                                active_color=_menu_color("text_primary"),
                                active_track_color=_menu_color("switch_track_active"),
                                inactive_thumb_color=_menu_color("text_primary"),
                                inactive_track_color=_menu_color("track"),
                                focus_color=_menu_color("primary_hover"),
                                hover_color=_menu_color("primary_hover_container"),
                                track_outline_color=_menu_color("surface_container"),
                                disabled=not bool(item.enabled),
                                on_change=lambda _event, toggle_key=str(item.key): queue_action(toggle_key),
                            )
                            toggle_widgets[str(item.key)] = toggle
                            children.append(
                                ft.Row(
                                    controls=[toggle],
                                    alignment=ft.MainAxisAlignment.START,
                                )
                            )
                        else:
                            width = None
                            key = str(item.key)
                            if group_key == "room_models":
                                width = _compact_flet_dimension(136)
                            elif group_key == "room_seats":
                                width = _compact_flet_dimension(136)
                            elif group_key == "screen_crop":
                                width = _compact_flet_dimension(136)
                            elif group_key in {"glow_modes", "depth_stereo"}:
                                width = _compact_flet_dimension(216)
                            pending_buttons.append(make_button(item, width=width))
                    flush_buttons()

                if reset_items:
                    children.append(
                        ft.Row(
                            controls=[
                                make_button(
                                    item,
                                    width=_compact_flet_dimension(160),
                                )
                                for item in reset_items
                            ],
                            alignment=ft.MainAxisAlignment.END,
                        )
                    )

                group_title = translate(
                    group_titles.get(group_key)
                    or OPENXR_MENU_GROUP_LABELS.get(group_key, "")
                )
                group_card = ft.Container(
                    content=ft.Column(
                        controls=children,
                        spacing=_FLET_SPACING,
                        horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                    ),
                    bgcolor=_menu_color("surface_container"),
                    border_radius=OPENXR_MENU_RADII["group"],
                    padding=8,
                )
                group_content: Any = group_card
                if group_title:
                    group_content = ft.Column(
                        controls=[
                            ft.Container(
                                height=_MENU_TITLE_HEIGHT,
                                alignment=ft.Alignment.CENTER_LEFT,
                                padding=ft.Padding.only(left=8, right=8),
                                content=ft.Text(
                                    group_title,
                                    size=16,
                                    weight=ft.FontWeight.W_600,
                                    color=_menu_color("text_secondary"),
                                    text_align=ft.TextAlign.LEFT,
                                ),
                            ),
                            group_card,
                        ],
                        spacing=_FLET_SPACING,
                        horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                    )
                content.append(group_content)

            if section_controls:
                toolbar_controls = [
                    ft.Container(
                        content=make_button(
                            item, width=_compact_flet_dimension(236)
                        ),
                        expand=True,
                    )
                    for item in section_controls
                ]
                section_toolbar.height = 72
                section_toolbar.content = ft.Column(
                    controls=[
                        ft.Container(
                            height=24,
                            alignment=ft.Alignment.CENTER_LEFT,
                            content=ft.Text(
                                translate(group_titles.get("screen_page_heading") or "Screen geometry"),
                                size=14,
                                weight=ft.FontWeight.W_600,
                                color=_menu_color("text_primary"),
                                text_align=ft.TextAlign.LEFT,
                            ),
                        ),
                        ft.Row(
                            controls=toolbar_controls,
                            spacing=_FLET_SPACING,
                            expand=True,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                    ],
                    spacing=_FLET_SPACING,
                    expand=True,
                )
            else:
                section_toolbar.height = 0
                section_toolbar.content = ft.Container()

            sidebar.controls = [
                *(nav_button(control) for control in nav_controls),
                ft.Container(expand=True),
                make_button(stop_control) if stop_control is not None else ft.Container(height=_FLET_BUTTON_HEIGHT),
            ]
            body_content.controls = content
            layout_signature = _snapshot_layout_signature(snapshot)

        def apply_values(snapshot: dict[str, Any]) -> bool:
            changed = False
            values = dict(snapshot.get("values") or {})
            for key, (slider, value_label, step) in slider_widgets.items():
                if key not in values:
                    continue
                try:
                    value = float(values[key])
                except (TypeError, ValueError):
                    continue
                value = _bounded_slider_value(
                    value,
                    float(slider.min),
                    float(slider.max),
                )
                if slider.value != value:
                    slider.value = value
                    value_label.value = _format_value(value, step, key)
                    changed = True
            for key, toggle in toggle_widgets.items():
                value = _toggle_value_for(key, values)
                if bool(toggle.value) != value:
                    toggle.value = value
                    changed = True
            controls_by_key = {
                str(control.key): control
                for control in snapshot.get("controls", ())
            }
            for slider_key, (minus_button, plus_button) in step_widgets.items():
                minus = controls_by_key.get(f"step:minus:{slider_key}")
                plus = controls_by_key.get(f"step:plus:{slider_key}")
                if minus is not None and minus_button.disabled != (not bool(minus.enabled)):
                    minus_button.disabled = not bool(minus.enabled)
                    changed = True
                if plus is not None and plus_button.disabled != (not bool(plus.enabled)):
                    plus_button.disabled = not bool(plus.enabled)
                    changed = True
            return changed

        async def on_window_event(event: Any) -> None:
            if event.type == ft.WindowEventType.CLOSE:
                await set_window_mode(False)
                page.window.update()
                page.update()
            elif event.type == ft.WindowEventType.SHOW and not visible:
                apply_drawer_region()

        page.window.on_event = on_window_event
        page.window.update()
        page.update()
        apply_drawer_region()

        while True:
            changed = False
            command = _drain_latest(commands)
            if command == "__quit__":
                break
            if command == "__toggle_flet__":
                await set_window_mode(not visible)
                changed = True
            elif command == "__show_flet__":
                await set_window_mode(True)
                changed = True
            elif command == "__hide_flet__":
                cancel_drawer_hide()
                visible = False
                page.window.visible = False
                changed = True
            elif command == "__show_icon__":
                await set_window_mode(False)
                changed = True

            snapshot = _drain_latest(snapshots)
            if snapshot is not None:
                monitor_rect = snapshot.get("input_monitor_rect")
                if isinstance(monitor_rect, (tuple, list)) and len(monitor_rect) == 4:
                    try:
                        normalized_rect = tuple(int(value) for value in monitor_rect)
                    except (TypeError, ValueError):
                        normalized_rect = None
                    if (
                        normalized_rect is not None
                        and normalized_rect[2] > 0
                        and normalized_rect[3] > 0
                        and normalized_rect != active_monitor_rect
                    ):
                        active_monitor_rect = normalized_rect
                        if not visible and page.window.visible:
                            geometry = _icon_geometry_for_monitor(
                                active_monitor_rect,
                                fallback_size=_primary_screen_size(),
                                pixel_ratio=window_pixel_ratio(),
                            )
                            page.window.left, page.window.top = geometry[:2]
                            apply_drawer_region()
                            changed = True
                signature = _snapshot_layout_signature(snapshot)
                if signature != layout_signature:
                    rebuild(snapshot)
                    changed = True
                elif apply_values(snapshot):
                    changed = True
            if changed:
                page.window.update()
                page.update()
            await asyncio.sleep(_FLET_PANEL_POLL_SECONDS)

        cancel_drawer_hide()
        page.window.prevent_close = False
        await page.window.destroy()

    try:
        ft.run(
            main,
            name="desktop2stereo-openxr-settings",
            view=ft.AppView.FLET_APP_HIDDEN,
            assets_dir=str(_FLET_FONT_ASSETS_DIR),
        )
    except Exception as exc:
        print(
            "[DesktopSettings] Flet settings window failed: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )


class DesktopOpenXrSettingsWindow:
    """Threaded launcher and queue bridge for the external Flet mirror."""

    def __init__(
        self,
        monitor_rect: tuple[int, int, int, int] | None = None,
    ) -> None:
        context = multiprocessing.get_context("spawn")
        self._snapshots = context.Queue(maxsize=1)
        self._actions = context.Queue()
        self._flet_commands = context.Queue()
        self._started = threading.Event()
        self._closed = threading.Event()
        self._icon_visible = True
        self._input_monitor_rect = monitor_rect
        self._flet_process: multiprocessing.Process | None = None
        self._flet_job: _WindowsKillOnCloseJob | None = None

    @property
    def actions(self) -> Any:
        return self._actions

    def start(self) -> None:
        if not desktop_settings_menu_enabled() or self._started.is_set():
            return
        self._started.set()
        context = multiprocessing.get_context("spawn")
        flet_job: _WindowsKillOnCloseJob | None = None
        start_gate = None
        if os.name == "nt":
            try:
                flet_job = _WindowsKillOnCloseJob()
                start_gate = context.Event()
            except Exception as exc:
                self._started.clear()
                print(
                    "[DesktopSettings] Could not establish Flet process ownership; "
                    f"settings window not started ({type(exc).__name__}: {exc}).",
                    flush=True,
                )
                return

        self._flet_process = context.Process(
            target=_run_flet_desktop_settings_app,
            args=(
                self._snapshots,
                self._actions,
                self._flet_commands,
                self._input_monitor_rect,
                start_gate,
            ),
            name="desktop2stereo-openxr-settings",
            daemon=True,
        )
        try:
            self._flet_process.start()
        except Exception as exc:
            _stop_flet_process(self._flet_process)
            if flet_job is not None:
                flet_job.close()
            self._flet_process = None
            self._started.clear()
            print(
                "[DesktopSettings] Could not start Flet settings process; "
                f"settings window not started ({type(exc).__name__}: {exc}).",
                flush=True,
            )
            return

        if flet_job is not None:
            if not flet_job.assign(self._flet_process.pid):
                _stop_flet_process(self._flet_process)
                flet_job.close()
                self._flet_process = None
                self._started.clear()
                print(
                    "[DesktopSettings] Could not assign the Flet process to its "
                    "cleanup job; settings window not started.",
                    flush=True,
                )
                return
            self._flet_job = flet_job
            start_gate.set()

    def stop(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        try:
            self._actions.put_nowait(("__close__", None))
        except Exception:
            pass
        try:
            self._flet_commands.put_nowait("__quit__")
        except Exception:
            pass
        try:
            _stop_flet_process(self._flet_process)
        finally:
            if self._flet_job is not None:
                self._flet_job.close()
                self._flet_job = None

    def toggle_icon_visibility(self) -> None:
        if self._closed.is_set():
            return
        self._icon_visible = not self._icon_visible
        try:
            self._flet_commands.put_nowait(
                "__show_icon__" if self._icon_visible else "__hide_flet__"
            )
        except Exception:
            pass

    def publish_snapshot(self, snapshot: dict[str, Any]) -> None:
        if not self._started.is_set() or self._closed.is_set():
            return
        monitor_rect = snapshot.get("input_monitor_rect")
        if isinstance(monitor_rect, (tuple, list)) and len(monitor_rect) == 4:
            try:
                normalized_rect = tuple(int(value) for value in monitor_rect)
            except (TypeError, ValueError):
                normalized_rect = None
            if normalized_rect is not None and normalized_rect[2] > 0 and normalized_rect[3] > 0:
                self._input_monitor_rect = normalized_rect
        try:
            self._snapshots.put_nowait(snapshot)
        except queue.Full:
            _drain_latest(self._snapshots)
            try:
                self._snapshots.put_nowait(snapshot)
            except queue.Full:
                pass

    def _on_icon_click(self, _event: Any) -> str:
        self._toggle_panel()
        return "break"

    def _toggle_panel(self) -> None:
        if self._closed.is_set():
            return
        try:
            self._flet_commands.put_nowait("__toggle_flet__")
        except Exception:
            pass
