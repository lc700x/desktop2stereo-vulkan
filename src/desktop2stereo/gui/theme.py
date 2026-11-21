"""Flet theme helpers matching the OpenXR Moonlight Blue surface tokens."""

from __future__ import annotations

import flet as ft
from theme_tokens import MOONLIGHT_BLUE_COLORS


def _color(name: str) -> str:
    return MOONLIGHT_BLUE_COLORS[name]


def build_app_theme(font_family: str, color: str = "blue") -> ft.Theme:
    """Build the desktop theme while keeping non-blue user themes intact."""
    if color.casefold() != "blue":
        return ft.Theme(
            color_scheme_seed=color,
            font_family=font_family,
            use_material3=True,
            visual_density=ft.VisualDensity.COMPACT,
        )

    scheme = ft.ColorScheme(
        primary=_color("primary"),
        on_primary=_color("primary_state_text"),
        primary_container=_color("primary_container"),
        on_primary_container=_color("text_primary"),
        secondary=_color("primary_hover"),
        on_secondary=_color("surface"),
        secondary_container=_color("surface_container_high"),
        on_secondary_container=_color("text_primary"),
        error=_color("destructive"),
        on_error=_color("destructive_text"),
        error_container="#7E4A52",
        on_error_container=_color("text_primary"),
        surface=_color("surface"),
        on_surface=_color("text_primary"),
        on_surface_variant=_color("text_secondary"),
        outline=_color("outline"),
        outline_variant="#365D7F",
        surface_container_lowest=_color("surface"),
        surface_container_low=_color("surface"),
        surface_container=_color("surface_header"),
        surface_container_high=_color("surface_container"),
        surface_container_highest=_color("surface_container_high"),
        surface_dim=_color("surface"),
        surface_bright=_color("surface_container_high"),
        scrim="#000000",
        shadow="#000000",
    )
    return ft.Theme(
        color_scheme=scheme,
        font_family=font_family,
        use_material3=True,
        visual_density=ft.VisualDensity.COMPACT,
        hover_color=ft.Colors.TRANSPARENT,
        focus_color=ft.Colors.TRANSPARENT,
        splash_color=ft.Colors.TRANSPARENT,
        highlight_color=ft.Colors.TRANSPARENT,
        disabled_color=_color("text_disabled"),
        canvas_color=_color("surface"),
        scaffold_bgcolor=_color("surface"),
        card_bgcolor=_color("surface_header"),
        divider_color="#365D7F",
    )


def color(name: str) -> str:
    """Return one shared token for controls that need an explicit color."""
    return _color(name)
