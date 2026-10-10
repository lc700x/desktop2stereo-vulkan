from __future__ import annotations

import pytest

from viewer.vulkan_fps_overlay import VulkanFpsOverlay, build_fps_panel_rgba


def test_fps_panel_uses_a_stable_texture_extent():
    pytest.importorskip("PIL")

    small_values = build_fps_panel_rgba(
        present_fps=5.2,
        capture_target=30,
        latency_ms=9,
        avg_latency_ms=28,
        content_fps=5.2,
        reuse_ratio=0.91,
    )
    large_values = build_fps_panel_rgba(
        present_fps=60.0,
        capture_target=120,
        latency_ms=1000,
        avg_latency_ms=450,
        content_fps=60.0,
        reuse_ratio=0.1,
    )

    assert small_values is not None
    assert large_values is not None
    assert small_values.size == large_values.size == (560, 256)
    assert small_values.mode == large_values.mode == "RGBA"
    assert small_values.getpixel((559, 255)) == (0, 0, 0, 0)


def test_set_panel_converts_before_command_recording():
    pytest.importorskip("PIL")
    from PIL import Image

    overlay = object.__new__(VulkanFpsOverlay)
    overlay._pending_panel = None
    overlay._panel_dirty = False
    panel = Image.new("RGB", (32, 24), (12, 34, 56))

    overlay.set_panel(panel)

    pixels, size = overlay._pending_panel
    assert size == panel.size
    assert pixels == panel.convert("RGBA").tobytes()
    assert overlay._panel_dirty is True
