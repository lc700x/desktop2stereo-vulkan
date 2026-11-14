from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
from media_ui import (
    _button_rects,
    build_media_bar_rgba,
    build_media_browser_rgba,
    build_pointer_rgba,
    media_bar_hit,
    media_bar_value_fraction,
    media_browser_hit,
    media_control_enabled,
    media_transport_icon,
)
from viewer.source_pointer import source_pointer_point
from viewer.vulkan_local_viewer import VulkanLocalViewer


def _state():
    return {
        "duration": 120.0,
        "position": 15.0,
        "playing": False,
        "volume": 0.5,
        "muted": False,
        "ready": True,
        "mode": "Video",
        "audio_enabled": True,
        "browser_visible": True,
        "browser_path": "C:/Videos",
        "browser_parent": "C:/",
        "browser_selection": "C:/Videos/movie.mp4",
        "browser_entries": [
            {"name": "movie.mp4", "path": "C:/Videos/movie.mp4", "kind": "video"},
        ],
        "browser_mounts": [],
        "playlist_entries": [
            {"name": "movie.mp4", "path": "C:/Videos/movie.mp4", "kind": "video"},
        ],
        "playlist_index": 0,
        "recent_entries": [
            {"name": "recent.mp4", "path": "F:/Videos/recent.mp4", "kind": "video"},
        ],
    }


def test_media_bar_controls_keep_large_touch_targets():
    rects = _button_rects(1024, 160)
    for key in ("previous", "seek-10", "seek+10", "next", "mute"):
        left, top, right, bottom = rects[key]
        assert right - left >= 64
        assert bottom - top >= 64
    left, top, right, bottom = rects["play"]
    assert right - left >= 80
    assert bottom - top >= 80


def test_media_bar_texture_and_hit_targets_are_stable():
    rgba = build_media_bar_rgba(_state(), width=1024, height=160)
    assert rgba.shape == (160, 1024, 4)
    assert rgba.dtype == np.uint8
    state = _state()
    assert media_bar_hit(0.546, 0.72, state) == "play"
    assert media_bar_hit(0.08, 0.72, state) == "browse"
    assert media_bar_hit(0.22, 0.72, state) == "browse_disc"
    assert media_bar_hit(0.5, 0.25, state).startswith("seek:")


def test_media_bar_previous_next_and_volume_hit_targets():
    state = {
        **_state(),
        "playlist_entries": [
            {"name": "first.mp4", "path": "C:/Videos/first.mp4", "kind": "video"},
            {"name": "second.mp4", "path": "C:/Videos/second.mp4", "kind": "video"},
        ],
        "playlist_index": 1,
    }
    assert media_bar_hit(0.384, 0.72, state) == "playlist_previous"
    assert media_bar_hit(0.69, 0.72, state) == "playlist_next"
    volume = media_bar_hit(0.90, 0.72, state)
    assert volume is not None and volume.startswith("volume:")
    assert abs(float(volume.split(":", 1)[1]) - 0.421) < 0.01
    assert media_bar_value_fraction(0.90, action="volume") is not None
    assert abs(media_bar_value_fraction(0.90, action="volume") - 0.421) < 0.01
    assert media_bar_value_fraction(0.90, action="seek") > 0.9
    assert media_bar_hit(0.5, 0.96, state, collapsed=True) == "wake"
    assert media_bar_hit(0.5, 0.50, state, collapsed=True) is None


def test_disabled_media_controls_are_not_actionable():
    image_state = {**_state(), "mode": "Image", "duration": 0.0}
    assert media_control_enabled("browse", image_state)
    assert not media_control_enabled("play", image_state)
    assert media_bar_hit(0.5, 0.72, image_state) is None
    assert media_control_enabled("recent", image_state)

    stopped = {**_state(), "ready": False}
    assert not media_control_enabled("browse", stopped)
    assert media_bar_hit(0.08, 0.72, stopped) is None


def test_browser_texture_and_playlist_hit_targets():
    rgba = build_media_browser_rgba(_state(), width=1400, height=900)
    assert rgba.shape == (900, 1400, 4)
    assert media_browser_hit(0.87, 0.05, _state()) == ("browser_up", None)
    entry_hit = media_browser_hit(0.1, 0.322, _state())
    assert entry_hit is not None
    assert entry_hit[0] == "playlist_select"
    assert media_browser_hit(0.07, 0.216, _state()) == (
        "recent",
        "F:/Videos/recent.mp4",
    )
    assert media_browser_hit(0.20, 0.216, _state()) == (
        "recent_remove",
        "F:/Videos/recent.mp4",
    )
    assert media_browser_hit(0.96, 0.96, _state()) == (
        "browse_open",
        "C:/Videos/movie.mp4",
    )


def test_browser_mounts_and_row_bounds_are_clickable_only_inside_geometry():
    state = {
        **_state(),
        "browser_mounts": [
            {"name": "媒体 (F:)", "path": "F:/", "kind": "drive"},
        ],
        "browser_entries": [
            {
                "name": f"movie-{index}.mp4",
                "path": f"F:/Videos/movie-{index}.mp4",
                "kind": "video",
            }
            for index in range(20)
        ],
    }
    assert media_browser_hit(0.1, 0.14, state) == ("mount", "F:/")
    assert media_browser_hit(0.72, 0.05, state) is None


def test_pointer_texture_and_source_coordinate_mapping():
    rgba = build_pointer_rgba()
    assert rgba.shape[2] == 4
    assert source_pointer_point(0.5, 0.5, monitor_rect=(100, 200, 1920, 1080)) == (
        1060,
        740,
    )


def test_local_viewer_media_hit_dispatches_shared_actions():
    actions = []
    viewer = object.__new__(VulkanLocalViewer)
    viewer.config = SimpleNamespace(
        media_action_callback=lambda action, value: actions.append((action, value)) or True,
        locale="EN",
    )
    viewer._media_state = {**_state(), "browser_visible": False}
    viewer._media_browser_scroll = 0
    viewer._media_bar_hover = None
    viewer._media_last_activity = time.monotonic()

    assert viewer._media_hit(0.546, 0.72) == "play"
    viewer._dispatch_media_hit("browse")
    viewer._dispatch_media_hit("browse_disc")
    viewer._dispatch_media_hit("browse_open:F:/Videos/movie.mp4")
    viewer._dispatch_media_hit("recent:F:/Videos/recent.mp4")
    viewer._dispatch_media_hit("recent_remove:F:/Videos/recent.mp4")
    viewer._dispatch_media_hit("seek_relative:-10")
    viewer._dispatch_media_hit("seek_relative:10")
    viewer._dispatch_media_hit("playlist_next")

    assert actions == [
        ("browse", None),
        ("browse", {"mode": "disc"}),
        ("browse_open", {"path": "F:/Videos/movie.mp4"}),
        ("browse_open", {"path": "F:/Videos/recent.mp4"}),
        ("recent_remove", {"path": "F:/Videos/recent.mp4"}),
        ("seek_relative", -10.0),
        ("seek_relative", 10.0),
        ("playlist_next", None),
    ]


def test_local_viewer_dispatches_absolute_seek_volume_and_browser_select():
    actions = []
    viewer = object.__new__(VulkanLocalViewer)
    viewer.config = SimpleNamespace(
        media_action_callback=lambda action, value: actions.append((action, value)) or True,
        locale="EN",
    )
    viewer._media_state = {**_state(), "browser_visible": True}

    viewer._dispatch_media_hit("seek:0.5")
    viewer._dispatch_media_hit("volume:0.25")
    viewer._dispatch_media_hit("entry:F:/Videos/movie.mp4")
    viewer._dispatch_media_hit("entry:F:/Videos/movie.mp4", double_click=True)

    assert actions == [
        ("seek", 60.0),
        ("volume", 0.25),
        ("browse_select", {"path": "F:/Videos/movie.mp4"}),
        ("browse_open", {"path": "F:/Videos/movie.mp4"}),
    ]


def test_local_viewer_places_media_overlays_inside_each_eye():
    viewer = object.__new__(VulkanLocalViewer)
    left = viewer._media_rect_for_eye((0, 0, 960, 1080))
    right = viewer._media_rect_for_eye((960, 0, 1920, 1080))
    assert left[0] >= 0 and left[2] <= 960
    assert right[0] >= 960 and right[2] <= 1920
    assert left[3] <= 1080 and right[3] <= 1080


def test_media_transport_icon_matches_available_action():
    assert media_transport_icon({"playing": True}) == "pause"
    assert media_transport_icon({"playing": False}) == "play"


def test_local_viewer_media_overlay_viewports_use_width_and_height():
    assert VulkanLocalViewer._media_overlay_viewport_rects(
        ((58, 912, 902, 1043), (1018, 912, 1862, 1043))
    ) == ((58, 912, 844, 131), (1018, 912, 844, 131))


def test_local_viewer_records_media_overlay_viewports_not_absolute_bounds():
    recorded = []

    class Overlay:
        def record_rects(self, _cmd, _image, _index, rects, **_kwargs):
            recorded.extend(rects)
            return True

    viewer = object.__new__(VulkanLocalViewer)
    viewer.config = SimpleNamespace(pointer_mapping_enabled=False)
    viewer.vk = SimpleNamespace(VK_IMAGE_LAYOUT_PRESENT_SRC_KHR=1000001002)
    viewer._media_bar_overlay = Overlay()
    viewer._media_browser_overlay = None
    viewer._media_pointer_overlay = None
    viewer._media_state = {"playing": True, "ready": True, "duration": 120.0}
    viewer._media_browser_visible = lambda: False
    viewer._media_bar_visible = lambda: True

    eye_rect = (0, 0, 960, 1080)
    absolute = viewer._media_rect_for_eye(eye_rect)
    drawn, layout = viewer._record_media_overlays(
        None,
        None,
        0,
        (eye_rect,),
        source_layout=0,
    )

    assert drawn is True
    assert layout == viewer.vk.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
    assert recorded == [
        (absolute[0], absolute[1], absolute[2] - absolute[0], absolute[3] - absolute[1])
    ]


def test_local_viewer_dispatches_playlist_controls_from_bar_hits():
    actions = []
    viewer = object.__new__(VulkanLocalViewer)
    viewer.config = SimpleNamespace(
        media_action_callback=lambda action, value: actions.append((action, value)) or True,
        locale="EN",
    )
    viewer._media_state = {
        **_state(),
        "playlist_entries": [
            {"name": "first.mp4", "path": "C:/Videos/first.mp4", "kind": "video"},
            {"name": "second.mp4", "path": "C:/Videos/second.mp4", "kind": "video"},
        ],
        "playlist_index": 1,
        "browser_visible": False,
    }
    viewer._media_browser_scroll = 0
    viewer._media_bar_hover = None
    viewer._media_last_activity = time.monotonic()

    viewer._dispatch_media_hit(viewer._media_hit(0.384, 0.72))
    viewer._dispatch_media_hit(viewer._media_hit(0.69, 0.72))

    assert actions == [("playlist_previous", None), ("playlist_next", None)]


def test_local_viewer_polls_cursor_to_reveal_controls_through_passthrough():
    calls = []
    viewer = object.__new__(VulkanLocalViewer)
    viewer.window = object()
    viewer.glfw = SimpleNamespace(
        get_cursor_pos=lambda _window: (480.0, 1145.0),
        get_window_size=lambda _window: (960, 1200),
    )
    viewer._media_last_cursor_position = None
    viewer._on_cursor_pos = lambda _window, x, y: calls.append((x, y))

    viewer._poll_media_pointer()
    viewer._poll_media_pointer()

    assert calls == [(480.0, 1145.0)]


def test_local_viewer_volume_drag_uses_volume_track_mapping():
    actions = []
    viewer = object.__new__(VulkanLocalViewer)
    viewer.config = SimpleNamespace(
        media_action_callback=lambda action, value: actions.append((action, value)) or True,
        locale="EN",
    )
    viewer._media_state = {**_state(), "browser_visible": False}
    viewer._media_drag_action = "volume"
    viewer._media_drag_last_sent_at = 0.0
    viewer._refresh_media_textures = lambda: None

    viewer._update_media_drag(0.90)

    assert len(actions) == 1
    assert actions[0][0] == "volume"
    assert abs(actions[0][1] - 0.421) < 0.01


def test_local_viewer_play_toggle_uses_pause_while_playing_and_updates_icon_state():
    actions = []
    viewer = object.__new__(VulkanLocalViewer)
    viewer.config = SimpleNamespace(
        media_action_callback=lambda action, value: actions.append((action, value)) or True,
    )
    viewer._media_state = {**_state(), "playing": True}
    viewer._media_bar_texture_key = ("stale",)

    viewer._dispatch_media_hit("play")
    assert actions == [("pause", None)]
    assert viewer._media_state["playing"] is False
    assert viewer._media_bar_texture_key is None

    viewer._dispatch_media_hit("play")
    assert actions == [("pause", None), ("play", None)]
    assert viewer._media_state["playing"] is True


def test_local_viewer_registers_media_overlays_in_swapchain_lifecycle():
    from inspect import getsource

    viewer = object.__new__(VulkanLocalViewer)
    bar, browser, pointer = object(), object(), object()
    viewer._media_bar_overlay = bar
    viewer._media_browser_overlay = browser
    viewer._media_pointer_overlay = pointer

    assert viewer._media_overlay_instances() == (bar, browser, pointer)
    create_source = getsource(VulkanLocalViewer._create_swapchain)
    destroy_source = getsource(VulkanLocalViewer._destroy_swapchain)
    assert "*self._media_overlay_instances()" in create_source
    assert "media_overlay.on_swapchain_destroyed()" in destroy_source
