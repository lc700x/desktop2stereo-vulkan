import inspect
import re
from types import SimpleNamespace

import flet as ft

from .theme import color

SCALE = 0.9


def S(v):
    """Scale a dimension literal by the global UI SCALE factor."""
    return round(v * SCALE)


def _supports_parameter(control_type, parameter):
    """Feature-detect Flet control properties across the bundled runtime versions."""
    try:
        return parameter in inspect.signature(control_type.__init__).parameters
    except (TypeError, ValueError):
        return False


FONT_SIZE = S(14)
LABEL_ALIGN_WIDTH = 0
TRANSPARENT_CONTROL_STATES = {
    state: ft.Colors.TRANSPARENT for state in ft.ControlState
}


def _refresh_if_attached(*controls):
    """Refresh controls only when Flet has attached them to a page."""
    for control in controls:
        try:
            control.update()
        except (RuntimeError, AssertionError):
            # Flet raises AssertionError when update() is called before the
            # control is attached to a page. Property setters are also used
            # while the GUI tree is being assembled, so this is expected.
            pass


def set_label_align_width(value):
    global LABEL_ALIGN_WIDTH
    LABEL_ALIGN_WIDTH = value


class CompactTextField(ft.Container):
    """Compact text input, 32px height, visually consistent with CompactDropdown."""

    def __init__(self, value="", width=S(100), read_only=False, on_change=None, tooltip=None, filter=None, max_length=None):
        super().__init__()
        self.height = S(32)
        self.width = width if width else None
        self.padding = 0
        self.bgcolor = None
        self.border = None
        self.border_radius = 0
        self._read_only = read_only
        self._on_change = on_change
        self._value = value
        self._label = ft.Text(value or "", size=FONT_SIZE, color=color("text_primary"))
        self._committed = False
        self._tooltip = tooltip or ""
        self._filter = filter
        self._max_length = max_length
        self._build_display()

    def _build_display(self):
        self.content = ft.Container(
            height=S(32), padding=ft.Padding(S(8), 0, S(8), 0),
            bgcolor=color("surface_container_high"),
            border=ft.Border(
                ft.BorderSide(1, color("outline")),
                ft.BorderSide(1, color("outline")),
                ft.BorderSide(1, color("outline")),
                ft.BorderSide(1, color("outline")),
            ),
            border_radius=8,
            tooltip=self._tooltip,
            on_click=None if self._read_only else self._on_click,
            content=ft.Row([self._label], spacing=2, vertical_alignment=ft.CrossAxisAlignment.CENTER),
        )

    def _on_click(self, e):
        self._committed = False
        tf = ft.TextField(
            value=self._value, text_size=FONT_SIZE, dense=True,
            filled=False, border=ft.InputBorder.NONE,
            content_padding=ft.Padding(0, 0, 0, 0), height=S(28),
            autofocus=True, on_submit=self._on_submit, on_blur=self._on_submit,
            max_length=self._max_length,
            input_filter=ft.InputFilter(regex_string=self._filter, allow=True) if self._filter else None,
        )
        self.content = ft.Container(
            height=S(32), padding=ft.Padding(S(4), 0, S(4), 0),
            bgcolor=color("surface_container_high"),
            border=ft.Border(
                ft.BorderSide(1, color("outline")),
                ft.BorderSide(1, color("outline")),
                ft.BorderSide(1, color("outline")),
                ft.BorderSide(1, color("outline")),
            ),
            border_radius=8, content=tf,
        )
        _refresh_if_attached(self)

    def set_tooltip(self, text):
        self._tooltip = text

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self._value = val
        self._label.value = val
        _refresh_if_attached(self._label)

    def _on_submit(self, e):
        if self._committed:
            return
        self._committed = True
        raw = e.control.value
        if self._filter:
            raw = ''.join(c for c in raw if re.match(self._filter, c))
        self._value = raw
        if self._on_change:
            self._on_change(SimpleNamespace(control=SimpleNamespace(value=self._value)))
        self._label.value = self._value
        self._build_display()
        self.update()


class CompactDisplayField(ft.Container):
    """Read-only compact field with content-driven min/max width constraints."""

    def __init__(self, value="", min_width=None, max_width=None, on_click=None):
        super().__init__()
        self._value = str(value or "")
        self._min = min_width or 0
        self._max = max_width or 0
        self.height = S(32)
        self.padding = ft.Padding(S(8), 0, S(8), 0)
        self.border = ft.Border(
            ft.BorderSide(1, color("outline")),
            ft.BorderSide(1, color("outline")),
            ft.BorderSide(1, color("outline")),
            ft.BorderSide(1, color("outline")),
        )
        self.bgcolor = color("surface_container_high")
        self.border_radius = 8
        self.clip_behavior = ft.ClipBehavior.HARD_EDGE
        self.on_click = on_click
        self._label = ft.Text(
            self._value,
            size=FONT_SIZE,
            color=color("text_primary"),
            no_wrap=True,
            overflow=ft.TextOverflow.ELLIPSIS,
        )
        self.content = ft.Row(
            [self._label],
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
        self._apply_width()

    def _calc_auto_width(self):
        text_width = sum(
            FONT_SIZE * (1.0 if ord(char) > 127 else 0.5)
            for char in self._value
        )
        return int(text_width) + S(24)

    def _apply_width(self):
        width = self._calc_auto_width()
        if self._min:
            width = max(width, self._min)
        if self._max:
            width = min(width, self._max)
        self.width = width

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = str(value or "")
        self._label.value = self._value
        self._apply_width()
        _refresh_if_attached(self._label, self)


class CompactDropdown(ft.Container):
    """Compact dropdown, PopupMenuButton with controllable width."""

    _instances = None
    _CORNER_RADIUS = 14

    def __init__(self, options=None, value="", on_select=None, expand=False,
                 dyna_width=None, width=None, min_width=None, max_width=None,
                 tooltip=None, _instances_list=None):
        super().__init__()
        self._options = options or []
        self._on_select_cb = on_select
        self._dyna = dyna_width
        self._fixed = width
        self._min = min_width or 0
        self._max = max_width or 0
        self._tooltip = tooltip
        self._trigger_hovered = False
        self.height = S(32)
        self.padding = 0
        # Clip the child trigger so rounded hover paint cannot spill out.
        self.bgcolor = ft.Colors.TRANSPARENT
        self.border = None
        self.border_radius = self._CORNER_RADIUS
        self.clip_behavior = ft.ClipBehavior.HARD_EDGE
        self.expand_loose = expand

        self._label = ft.Text(value or "", size=FONT_SIZE, color=color("text_primary"))
        self._build_menu()
        self._apply_width()
        if _instances_list is not None:
            _instances_list.append(self)
        elif CompactDropdown._instances is not None:
            CompactDropdown._instances.append(self)

    def update(self):
        """Refresh when mounted; ignore Flet's pre-mount assertion."""
        try:
            return super().update()
        except (RuntimeError, AssertionError):
            return None

    def reapply_width(self):
        self._apply_width()

    def _calc_auto_width(self):
        txt = self._label.value or ""
        if not txt:
            return S(100)
        w = sum(FONT_SIZE * (1.2 if ord(c) > 127 else 0.6) for c in txt)
        return int(w) + S(34)

    def _apply_width(self):
        if self._dyna:
            self.width = LABEL_ALIGN_WIDTH or self._calc_auto_width()
        elif self._fixed is not None:
            self.width = self._fixed
        else:
            auto = self._calc_auto_width()
            if self._min and auto < self._min:
                self.width = self._min
            elif self._max and auto > self._max:
                self.width = self._max
            else:
                self.width = None

    def _build_menu(self):
        def on_item_click(e):
            val = e.control.data
            self._value = val
            self._label.value = val
            self._apply_width()
            _refresh_if_attached(self._label, self)
            if self._on_select_cb:
                ev = SimpleNamespace(control=SimpleNamespace(value=val))
                self._on_select_cb(ev)

        def build_item(option):
            option_surface = ft.Container(
                content=ft.Text(
                    option,
                    size=FONT_SIZE,
                    color=color("text_primary"),
                ),
                height=S(32),
                padding=ft.Padding(S(12), 0, S(12), 0),
                bgcolor=ft.Colors.TRANSPARENT,
                border_radius=S(10),
                alignment=ft.Alignment.CENTER_LEFT,
                expand=True,
            )

            def on_option_hover(e):
                hovered = str(getattr(e, "data", "false")).casefold() == "true"
                option_surface.bgcolor = (
                    color("primary_container")
                    if hovered
                    else ft.Colors.TRANSPARENT
                )
                _refresh_if_attached(option_surface)

            option_surface.on_hover = on_option_hover
            return ft.PopupMenuItem(
                content=option_surface,
                data=option,
                height=S(36),
                padding=0,
                on_click=on_item_click,
            )

        items = [build_item(option) for option in self._options]

        has_limit = self._min or self._max
        align = ft.MainAxisAlignment.SPACE_BETWEEN if (self._fixed is not None or self._dyna or has_limit) else ft.MainAxisAlignment.START
        rounded_shape = ft.RoundedRectangleBorder(radius=self._CORNER_RADIUS)
        self._trigger_surface = ft.Container(
            height=S(32),
            padding=ft.Padding(S(8), 0, S(8), 0),
            bgcolor=color(
                "primary_hover_container"
                if self._trigger_hovered
                else "surface_container_high"
            ),
            border=ft.Border(
                ft.BorderSide(1, color("outline")),
                ft.BorderSide(1, color("outline")),
                ft.BorderSide(1, color("outline")),
                ft.BorderSide(1, color("outline")),
            ),
            border_radius=self._CORNER_RADIUS,
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
            on_hover=self._on_trigger_hover,
            content=ft.Row([
                self._label,
                ft.Icon(ft.Icons.ARROW_DROP_DOWN, size=S(16), color=color("text_secondary")),
            ], spacing=2, alignment=align,
               vertical_alignment=ft.CrossAxisAlignment.CENTER),
        )

        popup_kwargs = {
            "items": items,
            "menu_position": ft.PopupMenuPosition.UNDER,
            "enable_feedback": False,
            "padding": 0,
            "menu_padding": S(4),
            "splash_radius": 0,
            # Keep hover paint on the rounded trigger surface, not the Material anchor.
            "shape": rounded_shape,
            "clip_behavior": ft.ClipBehavior.HARD_EDGE,
            "style": ft.ButtonStyle(
                shape=rounded_shape,
                padding=0,
                elevation=0,
                bgcolor=ft.Colors.TRANSPARENT,
                overlay_color=TRANSPARENT_CONTROL_STATES,
            ),
            # Match the log-level dropdown: an empty tooltip disables the
            # PopupMenuButton's built-in rectangular "Show menu" overlay.
            "tooltip": "",
            "content": self._trigger_surface,
        }
        # border_radius was added to PopupMenuButton after Flet 0.86.5.
        # Older installed/portable runtimes still get the clipped rounded child
        # surface and transparent Material overlay, without failing at startup.
        if _supports_parameter(ft.PopupMenuButton, "border_radius"):
            popup_kwargs["border_radius"] = self._CORNER_RADIUS
        self.content = ft.PopupMenuButton(**popup_kwargs)
        self.content.theme = ft.Theme(
            hover_color=ft.Colors.TRANSPARENT,
            splash_color=ft.Colors.TRANSPARENT,
            highlight_color=ft.Colors.TRANSPARENT,
            focus_color=ft.Colors.TRANSPARENT,
        )

    def _on_trigger_hover(self, e):
        self._trigger_hovered = str(getattr(e, "data", "false")).casefold() == "true"
        self._trigger_surface.bgcolor = color(
            "primary_hover_container"
            if self._trigger_hovered
            else "surface_container_high"
        )
        _refresh_if_attached(self._trigger_surface)

    def set_tooltip(self, text):
        self._tooltip = text
        self._build_menu()
        _refresh_if_attached(self)

    @property
    def value(self):
        return self._label.value

    @value.setter
    def value(self, val):
        self._label.value = val
        self._apply_width()
        self._build_menu()
        _refresh_if_attached(self._label, self)

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, opts):
        self._options = opts
        self._build_menu()
        _refresh_if_attached(self)
