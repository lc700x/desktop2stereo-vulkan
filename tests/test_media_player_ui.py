from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
from media_ui import (
    MEDIA_BROWSER_TEXTURE_SIZE,
    MediaHoverAnimation,
    _browser_button_rects,
    _browser_layout,
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


def test_media_hover_animation_has_enter_and_exit_transitions():
    animation = MediaHoverAnimation(duration_s=0.16)

    assert animation.update("play", 100.0) == ("play", 0.0)
    _, progress = animation.update("play", 100.08)
    assert 0.4 < progress < 0.7
    visual_hit, progress = animation.update(None, 100.14)
    assert visual_hit == "play"
    assert progress > 0.0
    assert animation.update(None, 100.40) == (None, 0.0)

    scrub = MediaHoverAnimation(duration_s=0.16)
    assert scrub.update("seek:0.10", 200.0) == ("seek:0.10", 0.0)
    _, progress = scrub.update("seek:0.10", 200.08)
    visual_hit, moved_progress = scrub.update("seek:0.30", 200.12)
    assert visual_hit == "seek:0.30"
    assert moved_progress > progress


def test_media_bar_controls_keep_large_touch_targets():
    rects = _button_rects(1024, 160)
    for key in ("previous", "seek-10", "seek+10", "next", "mute"):
        left, top, right, bottom = rects[key]
        assert right - left >= 64
        assert bottom - top >= 64
    left, top, right, bottom = rects["play"]
    assert right - left >= 80
    assert bottom - top >= 80


def test_media_bar_hover_changes_pixels_and_keeps_timeline_end_knob_visible():
    idle = build_media_bar_rgba(_state())
    hovered = build_media_bar_rgba(
        _state(),
        hover="seek:1.000000",
        hover_progress=1.0,
    )
    assert np.any(idle != hovered)
    # The preview knob at the far end must stay inside the texture and alter
    # the right-hand track region rather than being clipped by its edge.
    right_edge = hovered[28:52, 940:1020, 3]
    assert right_edge.max() > 0
    assert np.any(right_edge != idle[28:52, 940:1020, 3])


def test_media_bar_texture_and_hit_targets_are_stable():
    rgba = build_media_bar_rgba(_state(), width=1024, height=160)
    assert rgba.shape == (160, 1024, 4)
    assert rgba.dtype == np.uint8
    state = _state()
    assert media_bar_hit(0.546, 0.72, state) == "play"
    assert media_bar_hit(0.08, 0.72, state) == "browse"
    assert media_bar_hit(0.22, 0.72, state) == "browse_disc"
    assert media_bar_hit(0.5, 0.25, state).startswith("seek:")
    assert media_bar_hit(1000 / 1024, 0.25, state) == "seek:1.000000"


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


def _browser_uv(x: float, y: float) -> tuple[float, float]:
    width, height = MEDIA_BROWSER_TEXTURE_SIZE
    return x / width, y / height


def test_browser_hover_progress_changes_hovered_row_and_button_surfaces():
    idle = build_media_browser_rgba(_state())
    hovered = build_media_browser_rgba(
        _state(),
        hover="entry:C:/Videos/movie.mp4",
        hover_progress=1.0,
    )
    assert hovered.shape == idle.shape
    assert np.any(hovered != idle)
    open_hovered = build_media_browser_rgba(
        _state(),
        hover="browse_open:C:/Videos/movie.mp4",
        hover_progress=0.5,
    )
    assert np.any(open_hovered != idle)


def test_browser_texture_and_playlist_hit_targets():
    width, height = MEDIA_BROWSER_TEXTURE_SIZE
    rgba = build_media_browser_rgba(_state())
    assert rgba.shape == (height, width, 4)
    rects = _browser_button_rects(width, height)
    layout = _browser_layout(_state(), width, height)

    up = rects["browser_up"]
    assert media_browser_hit(
        *_browser_uv((up[0] + up[2]) / 2, (up[1] + up[3]) / 2), _state()
    ) == ("browser_up", None)

    playlist_y = (
        layout["playlist_rows_y"] + layout["playlist_row_height"] / 2
    )
    assert media_browser_hit(
        *_browser_uv(layout["sidebar_right"] / 2, playlist_y), _state()
    ) == ("playlist_select", "C:/Videos/movie.mp4")

    recent_y = layout["recent_rows_y"] + layout["recent_row_height"] / 2
    assert media_browser_hit(
        *_browser_uv(layout["sidebar_right"] / 2, recent_y), _state()
    ) == ("recent", "F:/Videos/recent.mp4")
    assert media_browser_hit(
        *_browser_uv(layout["sidebar_right"] - 24, recent_y), _state()
    ) == ("recent_remove", "F:/Videos/recent.mp4")

    open_rect = rects["browse_open"]
    assert media_browser_hit(
        *_browser_uv(
            (open_rect[0] + open_rect[2]) / 2,
            (open_rect[1] + open_rect[3]) / 2,
        ),
        _state(),
    ) == ("browse_open", "C:/Videos/movie.mp4")


def test_browser_footer_actions_use_content_inset_and_stable_group_gap():
    width, height = MEDIA_BROWSER_TEXTURE_SIZE
    rects = _browser_button_rects(width, height)
    previous = rects["playlist_previous"]
    next_button = rects["playlist_next"]
    clear = rects["playlist_clear"]
    open_button = rects["browse_open"]

    assert previous[0] == 48
    assert height - max(previous[3], next_button[3], clear[3]) == 48
    assert next_button[0] - previous[2] == 24
    assert clear[0] - next_button[2] == 24
    assert open_button[2] == width - 48
    assert open_button[3] == height - 48


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
    width, height = MEDIA_BROWSER_TEXTURE_SIZE
    layout = _browser_layout(state, width, height)
    mount_y = layout["mount_top"] + layout["mount_row_height"] / 2
    assert media_browser_hit(
        *_browser_uv(layout["sidebar_right"] / 2, mount_y), state
    ) == ("mount", "F:/")
    assert layout["file_row_height"] >= 72
    row_x = (layout["sidebar_right"] + width) / 2
    row_y = layout["file_rows_y"] + layout["file_row_height"] / 2
    assert media_browser_hit(*_browser_uv(row_x, row_y), state) == (
        "entry",
        "F:/Videos/movie-0.mp4",
    )
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

    browser = viewer._media_rect_for_eye((0, 0, 960, 1080), browser=True)
    assert browser[2] - browser[0] >= int(960 * 0.95)
    assert browser[3] - browser[1] >= int(1080 * 0.60)


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


def test_playlist_controls_share_painted_geometry_and_page_offset():
    state = {**_state(), "playlist_entries": [
        {"name": f"movie-{index}.mp4", "path": f"F:/movie-{index}.mp4", "kind": "video"}
        for index in range(8)
    ], "playlist_offset": 4, "playlist_index": 5}
    width, height = MEDIA_BROWSER_TEXTURE_SIZE
    layout = _browser_layout(state, width, height)
    y = layout["playlist_rows_y"] + layout["playlist_row_height"] / 2
    right = layout["sidebar_right"]
    for action, x in (("playlist_move_up", right - 140), ("playlist_move_down", right - 88),
                      ("playlist_remove", right - 36), ("playlist_select", 160)):
        assert media_browser_hit(*_browser_uv(x, y), state) == (action, "F:/movie-4.mp4")
    for action in ("playlist_add", "playlist_sort", "browse_open", "playlist_clear"):
        rect = _browser_button_rects(width, height)[action]
        u, v = _browser_uv((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2)
        hit = media_browser_hit(u, v, state)
        assert hit is not None and hit[0] == action
    assert layout["playlist_row_height"] >= 72
    image = build_media_browser_rgba(state, lang="CN")
    assert image.shape == (height, width, 4)


def test_add_button_does_not_open_selected_file():
    state = {**_state(), "browser_entries": [
        {"name": "movie.mp4", "path": "F:/movie.mp4", "kind": "video"}
    ]}
    width, height = MEDIA_BROWSER_TEXTURE_SIZE
    layout = _browser_layout(state, width, height)
    x = width - 70
    y = layout["file_rows_y"] + (layout["file_row_height"] - 4) / 2
    assert media_browser_hit(*_browser_uv(x, y), state) == ("playlist_add", "F:/movie.mp4")


def test_local_playlist_controls_dispatch_shared_commands():
    actions = []
    viewer = object.__new__(VulkanLocalViewer)
    viewer._media_state = {"playlist_sort_descending": False}
    viewer.config = SimpleNamespace(media_action_callback=lambda action, value: actions.append((action, value)) or True)
    viewer._dispatch_media_hit("playlist_move_up:F:/movie.mp4")
    viewer._dispatch_media_hit("playlist_remove:F:/movie.mp4")
    viewer._dispatch_media_hit("playlist_add:F:/new.mp4")
    viewer._dispatch_media_hit("playlist_sort")
    viewer._dispatch_media_hit("playlist_page_next")
    assert actions == [
        ("playlist_move", {"path": "F:/movie.mp4", "offset": -1}),
        ("playlist_remove", {"path": "F:/movie.mp4"}),
        ("playlist_add", {"path": "F:/new.mp4"}),
        ("playlist_sort", {"descending": True}),
        ("playlist_page", 4),
    ]
