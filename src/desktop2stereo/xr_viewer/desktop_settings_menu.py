"""Flet desktop mirror for the in-headset OpenXR settings menu.

The in-headset menu remains the source of truth. This module only mirrors its
snapshots in a normal Flet desktop window and returns physical mouse actions
through the existing action queue. A tiny Tk gear remains solely as the
transparent floating launcher requested for the OpenXR desktop view.
"""

from __future__ import annotations

import asyncio
import math
import multiprocessing
import os
import queue
import subprocess
import threading
import tkinter as tk
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
_FLET_SIDEBAR_WIDTH = 176
_FLET_FONT_ASSETS_DIR = Path(__file__).resolve().parent / "fonts"


def _menu_color(token: str) -> str:
    return OPENXR_MENU_COLORS[token]


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
) -> tuple[int, int, int, int]:
    """Return icon ``(x, y, width, height)`` inside the input monitor."""
    icon_width, icon_height = DESKTOP_SETTINGS_ICON_SIZE
    if monitor_rect is None:
        monitor_left, monitor_top = 0, 0
        monitor_width, monitor_height = fallback_size
    else:
        monitor_left, monitor_top, monitor_width, monitor_height = monitor_rect
    icon_x = monitor_left + max(0, int(monitor_width) - icon_width - 24)
    icon_y = monitor_top + max(0, (int(monitor_height) - icon_height) // 2)
    return icon_x, icon_y, icon_width, icon_height


def _flet_panel_position_for_monitor(
    monitor_rect: tuple[int, int, int, int] | None,
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
    screen_group = _screen_button_row_group(key)
    if screen_group is not None:
        return screen_group
    if key in {"depth:toggle_stereo", "depth:toggle_cross_eyed"}:
        return "depth_modes"
    if key in {"glow:surround", "glow:glow", "glow:veil", "glow:off"}:
        return "glow_modes"
    if key.startswith("room:model:"):
        return "room_models"
    if key.startswith("room:seat:"):
        return "room_seats"
    if key == "room:toggle_screen_reflection":
        return "room_reflection"
    return None


def _run_flet_desktop_settings_app(
    snapshots: Any,
    actions: Any,
    commands: Any,
    input_monitor_rect: tuple[int, int, int, int] | None,
) -> None:
    """Run Flet in its own process because Flet owns the main-thread signals."""
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
        page.bgcolor = _menu_color("surface")
        page.fonts = {"Inter": "InterVariable.ttf"}
        page.theme = ft.Theme(
            color_scheme_seed=_menu_color("primary"),
            font_family="Inter",
        )
        page.theme_mode = ft.ThemeMode.DARK
        icon_path = Path(__file__).resolve().parents[1] / "icon2.ico"
        if icon_path.is_file():
            page.window.icon = str(icon_path)
        page.window.width = _FLET_PANEL_SIZE[0]
        page.window.height = _FLET_PANEL_SIZE[1]
        panel_position = _flet_panel_position_for_monitor(input_monitor_rect)
        if panel_position is not None:
            page.window.left, page.window.top = panel_position
        page.window.min_width = 580
        page.window.min_height = 420
        page.window.maximizable = False
        page.window.always_on_top = True
        page.window.prevent_close = True
        page.window.visible = False

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
            spacing=16,
        )
        body = ft.Container(
            content=body_content,
            expand=True,
            padding=ft.Padding.only(top=16),
        )
        sidebar = ft.Column(expand=True, spacing=16)
        section_toolbar = ft.Container(height=0)
        main_column = ft.Column(
            controls=[section_toolbar, body],
            expand=True,
            spacing=0,
        )
        main_area = ft.Container(
            content=main_column,
            expand=True,
            padding=ft.Padding.only(right=16, top=16, bottom=16),
        )
        layout = ft.Row(
            controls=[
                ft.Container(
                    width=_FLET_SIDEBAR_WIDTH,
                    expand=False,
                    bgcolor=_menu_color("surface_container"),
                    border_radius=OPENXR_MENU_RADII["group"],
                    padding=16,
                    content=sidebar,
                ),
                main_area,
            ],
            spacing=16,
            expand=True,
            vertical_alignment=ft.CrossAxisAlignment.STRETCH,
        )
        page.add(
            ft.Container(
                expand=True,
                bgcolor=_menu_color("surface"),
                padding=16,
                content=layout,
            )
        )

        slider_widgets: dict[str, tuple[Any, Any, float]] = {}
        step_widgets: dict[str, tuple[Any, Any]] = {}
        toggle_widgets: dict[str, Any] = {}
        layout_signature: tuple[Any, ...] | None = None
        visible = False

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
                    48 if key.startswith(("screen:section:", "screen:rotate:"))
                    else 64
                )
                button_width = (
                    width if width is not None
                    else _FLET_SIDEBAR_WIDTH - 32 if destructive
                    else None
                )
                return ft.ElevatedButton(
                    content=ft.Text(
                        translate(str(control.label)),
                        size=18,
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
                        padding=ft.Padding.symmetric(horizontal=8, vertical=8),
                        elevation=0,
                        text_style=ft.TextStyle(size=18, weight=ft.FontWeight.W_600),
                        shape=ft.RoundedRectangleBorder(
                            radius=OPENXR_MENU_RADII["control"],
                        ),
                        alignment=ft.Alignment.CENTER,
                    ),
                )

            def make_step_button(control: Any) -> Any:
                return ft.ElevatedButton(
                    content="-" if ":minus:" in str(control.key) else "+",
                    on_click=queue_button(str(control.key)),
                    disabled=not bool(control.enabled),
                    width=64,
                    height=64,
                    style=ft.ButtonStyle(
                        bgcolor={
                            ft.ControlState.DEFAULT: _menu_color("surface_container_high"),
                            ft.ControlState.HOVERED: _menu_color("primary_hover"),
                            ft.ControlState.PRESSED: _menu_color("primary"),
                            ft.ControlState.DISABLED: _menu_color("surface_container"),
                        },
                        color={
                            ft.ControlState.DEFAULT: _menu_color("text_primary"),
                            ft.ControlState.HOVERED: _menu_color("primary_state_text"),
                            ft.ControlState.PRESSED: _menu_color("primary_state_text"),
                            ft.ControlState.DISABLED: _menu_color("text_disabled"),
                        },
                        elevation=0,
                        text_style=ft.TextStyle(size=24, weight=ft.FontWeight.W_600),
                        shape=ft.RoundedRectangleBorder(radius=24),
                    ),
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
                                size=20,
                                color=(
                                    _menu_color("selection_text")
                                    if active else _menu_color("text_secondary")
                                ),
                            ),
                            ft.Text(
                                translate(str(control.label)),
                                size=18,
                                weight=ft.FontWeight.W_600,
                                text_align=ft.TextAlign.LEFT,
                                expand=True,
                            ),
                        ],
                        spacing=12,
                        expand=True,
                        alignment=ft.MainAxisAlignment.START,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    on_click=queue_button(key),
                    width=_FLET_SIDEBAR_WIDTH - 32,
                    height=64,
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
                        padding=ft.Padding.symmetric(horizontal=12, vertical=8),
                        elevation=0,
                        text_style=ft.TextStyle(size=18, weight=ft.FontWeight.W_600),
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
                    "runtime:stop", "section:reset_defaults",
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
                    width=80,
                    size=16,
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
                    if minus_control is not None else ft.Container(width=64)
                )
                plus_button = (
                    make_step_button(plus_control)
                    if plus_control is not None else ft.Container(width=64)
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
                                    size=18,
                                    color=_menu_color("text_primary"),
                                    text_align=ft.TextAlign.LEFT,
                                    expand=True,
                                ),
                                value_label,
                            ],
                            spacing=8,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                        ft.Row(
                            controls=[
                                minus_button,
                                slider,
                                plus_button,
                            ],
                            spacing=8,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                    ],
                    spacing=8,
                )

            content: list[Any] = []
            for group_key, items in slider_key_groups.items():
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
                            spacing=16,
                            wrap=True,
                            run_spacing=16,
                        )
                    else:
                        content_row = ft.Row(
                            controls=buttons,
                            alignment=ft.MainAxisAlignment.CENTER,
                            spacing=16,
                            wrap=True,
                            run_spacing=16,
                        )
                    children.append(content_row)

                shape_controls = (
                    [item for item in items if str(item.key).startswith("screen:type:")],
                    [item for item in items if str(item.key).startswith("screen:rotate:")],
                ) if group_key == "screen_shape" else None
                if shape_controls is not None:
                    for shape_row in shape_controls:
                        row_buttons = [
                            make_button(item, width=160 if str(item.key).startswith("screen:rotate:") else 96)
                            for item in shape_row
                        ]
                        children.append(
                            ft.Row(
                                controls=row_buttons,
                                alignment=ft.MainAxisAlignment.CENTER,
                                spacing=16,
                                wrap=True,
                                run_spacing=16,
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
                                    size=18,
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
                                width = 136
                            elif group_key == "room_seats":
                                width = 136
                            elif group_key == "screen_crop":
                                width = 136
                            elif group_key == "glow_modes":
                                width = 216
                            elif group_key == "depth_modes":
                                width = 216
                            pending_buttons.append(make_button(item, width=width))
                    flush_buttons()

                group_title = translate(
                    group_titles.get(group_key)
                    or OPENXR_MENU_GROUP_LABELS.get(group_key, "")
                )
                group_card = ft.Container(
                    content=ft.Column(
                        controls=children,
                        spacing=16,
                        horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                    ),
                    bgcolor=_menu_color("surface_container"),
                    border_radius=OPENXR_MENU_RADII["group"],
                    padding=16,
                )
                group_content: Any = group_card
                if group_title:
                    group_content = ft.Column(
                        controls=[
                            ft.Container(
                                height=_MENU_TITLE_HEIGHT,
                                alignment=ft.Alignment.CENTER_LEFT,
                                padding=ft.Padding.only(left=16, right=16),
                                content=ft.Text(
                                    group_title,
                                    size=20,
                                    weight=ft.FontWeight.W_600,
                                    color=_menu_color("text_secondary"),
                                    text_align=ft.TextAlign.LEFT,
                                ),
                            ),
                            group_card,
                        ],
                        spacing=_MENU_TITLE_CARD_GAP,
                        horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                    )
                content.append(group_content)

            if section_controls:
                toolbar_controls = [
                    ft.Container(
                        content=make_button(item, width=236),
                        expand=True,
                    )
                    for item in section_controls
                ]
                section_toolbar.height = 88
                section_toolbar.content = ft.Column(
                    controls=[
                        ft.Container(
                            height=24,
                            alignment=ft.Alignment.CENTER_LEFT,
                            content=ft.Text(
                                translate(group_titles.get("screen_page_heading") or "Screen geometry"),
                                size=18,
                                weight=ft.FontWeight.W_600,
                                color=_menu_color("text_primary"),
                                text_align=ft.TextAlign.LEFT,
                            ),
                        ),
                        ft.Row(
                            controls=toolbar_controls,
                            spacing=16,
                            expand=True,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                    ],
                    spacing=16,
                    expand=True,
                )
            else:
                section_toolbar.height = 0
                section_toolbar.content = ft.Container()

            sidebar.controls = [
                *(nav_button(control) for control in nav_controls),
                ft.Container(expand=True),
                make_button(stop_control) if stop_control is not None else ft.Container(height=64),
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

        def on_window_event(event: Any) -> None:
            nonlocal visible
            if event.type == ft.WindowEventType.CLOSE:
                visible = False
                page.window.visible = False
                page.update()

        page.window.on_event = on_window_event
        page.update()

        while True:
            changed = False
            command = _drain_latest(commands)
            if command == "__quit__":
                break
            if command == "__toggle_flet__":
                visible = not visible
                page.window.visible = visible
                page.window.focused = visible
                changed = True
            elif command == "__show_flet__":
                visible = True
                page.window.visible = True
                page.window.focused = True
                changed = True
            elif command == "__hide_flet__":
                visible = False
                page.window.visible = False
                changed = True

            snapshot = _drain_latest(snapshots)
            if snapshot is not None:
                signature = _snapshot_layout_signature(snapshot)
                if signature != layout_signature:
                    rebuild(snapshot)
                    changed = True
                elif apply_values(snapshot):
                    changed = True
            if changed:
                page.update()
            await asyncio.sleep(_FLET_PANEL_POLL_SECONDS)

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
        self._root: Any | None = None
        self._icon: Any | None = None
        self._icon_visible = True
        self._icon_photo: Any = None
        self._input_monitor_rect = monitor_rect
        self._icon_geometry: tuple[int, int, int, int] | None = None
        self._flet_process: multiprocessing.Process | None = None
        self._icon_thread: threading.Thread | None = None
        self._icon_stopped = threading.Event()

    @property
    def actions(self) -> Any:
        return self._actions

    def start(self) -> None:
        if not desktop_settings_menu_enabled() or self._started.is_set():
            return
        self._started.set()
        self._flet_process = multiprocessing.get_context("spawn").Process(
            target=_run_flet_desktop_settings_app,
            args=(
                self._snapshots,
                self._actions,
                self._flet_commands,
                self._input_monitor_rect,
            ),
            name="desktop2stereo-openxr-settings",
            daemon=True,
        )
        self._flet_process.start()
        self._icon_thread = threading.Thread(
            target=self._run_icon,
            name="desktop-settings-menu-icon",
            daemon=True,
        )
        self._icon_thread.start()

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
        _stop_flet_process(self._flet_process)
        icon_thread = self._icon_thread
        if (
            icon_thread is not None
            and icon_thread is not threading.current_thread()
        ):
            # Tk must destroy its interpreter and ImageTk references on the
            # thread that created them.  Waiting here prevents Python from
            # finalizing those objects on the runtime thread during exit.
            self._icon_stopped.wait(timeout=2.0)

    def toggle_icon_visibility(self) -> None:
        if self._closed.is_set():
            return
        self._icon_visible = not self._icon_visible

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

    def _run_icon(self) -> None:
        root = None
        try:
            root = tk.Tk()
            root.overrideredirect(True)
            root.attributes("-topmost", True)
            root.attributes("-alpha", DESKTOP_SETTINGS_ICON_OPACITY)
            root.configure(bg=DESKTOP_SETTINGS_ICON_TRANSPARENT_COLOR)
            geometry = _icon_geometry_for_monitor(
                self._input_monitor_rect,
                fallback_size=(
                    max(0, int(root.winfo_screenwidth())),
                    max(0, int(root.winfo_screenheight())),
                ),
            )
            self._set_icon_geometry(root, geometry)
            root.protocol("WM_DELETE_WINDOW", self.stop)
            self._root = root
            self._icon = root
            self._build_icon_content(root)
            root.deiconify()
            root.lift()
            root.after(100, self._poll_icon)
            root.mainloop()
        except Exception as exc:
            print(
                "[DesktopSettings] Floating icon failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        finally:
            if root is not None:
                try:
                    root.destroy()
                except (RuntimeError, tk.TclError):
                    pass
            # Drop all Tk-owned references before this thread exits.  This is
            # deliberately done here, never by the runtime thread.
            self._icon_photo = None
            self._icon = None
            self._root = None
            self._icon_stopped.set()

    def _build_icon_content(self, icon: Any) -> None:
        try:
            icon.attributes(
                "-transparentcolor",
                DESKTOP_SETTINGS_ICON_TRANSPARENT_COLOR,
            )
        except tk.TclError:
            pass
        icon.bind("<Button-1>", self._on_icon_click)
        label = tk.Label(
            icon,
            bg=DESKTOP_SETTINGS_ICON_TRANSPARENT_COLOR,
            cursor="hand2",
            bd=0,
            highlightthickness=0,
        )
        try:
            from PIL import Image, ImageTk

            icon_path = Path(__file__).resolve().parents[1] / "icon2.ico"
            image = Image.open(icon_path).resize(DESKTOP_SETTINGS_ICON_IMAGE_SIZE)
            self._icon_photo = ImageTk.PhotoImage(image)
            label.configure(image=self._icon_photo)
        except Exception:
            label.configure(
                text="⚙",
                fg="white",
                font=("Segoe UI", 15, "bold"),
                width=2,
                height=1,
            )
        label.pack(padx=4, pady=4)
        label.bind("<Button-1>", self._on_icon_click)
        icon.update_idletasks()

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

    def _poll_icon(self) -> None:
        root = self._root
        if root is None:
            return
        if self._closed.is_set():
            root.destroy()
            return
        try:
            geometry = _icon_geometry_for_monitor(
                self._input_monitor_rect,
                fallback_size=(
                    max(0, int(root.winfo_screenwidth())),
                    max(0, int(root.winfo_screenheight())),
                ),
            )
            self._set_icon_geometry(root, geometry)
            if self._icon_visible:
                root.deiconify()
                root.attributes("-topmost", True)
            else:
                root.withdraw()
                self._flet_commands.put_nowait("__hide_flet__")
        except (queue.Full, tk.TclError):
            pass
        root.after(100, self._poll_icon)

    def _set_icon_geometry(
        self,
        root: Any,
        geometry: tuple[int, int, int, int],
    ) -> None:
        if geometry == self._icon_geometry:
            return
        x, y, width, height = geometry
        x_offset = f"+{x}" if x >= 0 else str(x)
        y_offset = f"+{y}" if y >= 0 else str(y)
        root.geometry(f"{width}x{height}{x_offset}{y_offset}")
        self._icon_geometry = geometry
