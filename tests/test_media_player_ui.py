from __future__ import annotations

import sys
import time
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import media_ui
from media_ui import (
    MEDIA_BAR_TEXTURE_SIZE,
    MEDIA_BROWSER_LAYOUT_SIZE,
    MEDIA_BROWSER_TEXTURE_SIZE,
    MediaBrowserThumbnailCache,
    MediaHoverAnimation,
    _browser_button_rects,
    _browser_grid_card_rect,
    _browser_layout,
    _button_rects,
    build_media_bar_rgba,
    build_media_browser_rgba,
    build_pointer_rgba,
    media_bar_hit,
    media_bar_value_fraction,
    media_browser_hit,
    media_browser_font_scale_for_eye,
    media_browser_scroll_metrics,
    media_browser_scrollbar_metrics,
    media_browser_scrollbar_offset,
    media_control_enabled,
    media_transport_icon,
)
from viewer.source_pointer import source_pointer_point
import viewer.vulkan_local_viewer as vulkan_local_viewer
from viewer.vulkan_local_viewer import VulkanLocalViewer
from xr_viewer import overlay_textures
from xr_viewer.settings_menu import OPENXR_MENU_COLORS, OPENXR_MENU_RADII


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


def _rgb(token):
    value = OPENXR_MENU_COLORS[token].lstrip("#")
    return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4))


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
    right_edge = hovered[56:104, 1880:2040, 3]
    assert right_edge.max() > 0
    assert np.any(right_edge != idle[56:104, 1880:2040, 3])


def test_media_bar_default_uses_a_2x_texture_with_the_same_hit_geometry():
    rgba = build_media_bar_rgba(_state())

    assert rgba.shape == (MEDIA_BAR_TEXTURE_SIZE[1], MEDIA_BAR_TEXTURE_SIZE[0], 4)
    assert media_bar_hit(0.546, 0.72, _state()) == "play"


def test_media_bar_uses_larger_bolder_action_text(monkeypatch):
    font_calls = []
    load_font = media_ui._font

    def record_font(size, *, text="", bold=False):
        font_calls.append((size, text, bold))
        return load_font(size, text=text, bold=bold)

    monkeypatch.setattr(media_ui, "_font", record_font)

    build_media_bar_rgba({**_state(), "browser_visible": False})

    assert (32, "Open File", True) in font_calls
    assert (36, "00:15", False) in font_calls
    assert (36, "02:00", False) in font_calls


def test_vulkan_media_bar_overlay_upload_uses_hidpi_texture(monkeypatch):
    uploads = []

    class Overlay:
        def set_rgba(self, _pixels, size):
            uploads.append(size)

    viewer = object.__new__(VulkanLocalViewer)
    viewer.config = SimpleNamespace(locale="EN")
    viewer._media_state = {**_state(), "browser_visible": False}
    viewer._media_bar_overlay = Overlay()
    viewer._media_browser_overlay = None
    viewer._media_bar_hover = None
    viewer._media_bar_texture_key = None
    viewer._media_hover_animation = None
    viewer._media_bar_visible = lambda: True
    viewer._media_browser_visible = lambda: False
    viewer._media_bar_expanded = lambda: True
    monkeypatch.setattr(
        vulkan_local_viewer,
        "build_media_bar_rgba",
        lambda *_args, **_kwargs: np.zeros((320, 2048, 4), dtype=np.uint8),
    )

    viewer._refresh_media_textures()

    assert uploads == [MEDIA_BAR_TEXTURE_SIZE]


def test_media_browser_font_scale_tracks_eye_output_size():
    assert media_browser_font_scale_for_eye(1200, 1200) == pytest.approx(1.33, abs=0.01)
    assert media_browser_font_scale_for_eye(2048, 2048) == 1.0


def test_overlay_font_selection_uses_script_and_weight(monkeypatch):
    calls = []

    def fake_load(size, font_type=None, *, prefer_cjk=False, prefer_inter=False, bold=False):
        calls.append((size, prefer_cjk, prefer_inter, bold))
        return object()

    monkeypatch.setattr(overlay_textures, "load_overlay_font", fake_load)

    overlay_textures.load_overlay_font_for_text(24, "Media controls", bold=True)
    overlay_textures.load_overlay_font_for_text(24, "媒体控制", bold=False)

    assert calls == [(24, False, True, True), (24, True, False, False)]


def test_macos_cjk_font_uses_regular_and_semibold_collection_faces(monkeypatch):
    selected = []

    class FakeFont:
        pass

    def fake_truetype(path, size, *, index=0):
        if path == "/System/Library/Fonts/Hiragino Sans GB.ttc":
            selected.append((size, index))
            return FakeFont()
        raise OSError(path)

    monkeypatch.setattr(overlay_textures.ImageFont, "truetype", fake_truetype)

    loader = overlay_textures._load_overlay_font_cached.__wrapped__
    loader(24, None, True, False, False)
    loader(24, None, True, True, False)

    assert selected == [(24, 0), (24, 2)]


def test_media_volume_fill_uses_a_light_blue_hover_ramp():
    state = _state()
    idle = build_media_bar_rgba(state, width=1024, height=160)
    hovered = build_media_bar_rgba(
        state,
        width=1024,
        height=160,
        hover="volume:0.500000",
        hover_progress=1.0,
    )
    left, top, _right, bottom = _button_rects(1024, 160)["volume"]
    center_y = (top + bottom) // 2
    track_x = left + 14

    assert tuple(idle[center_y, track_x, :3]) == _rgb("media_volume_fill")
    assert tuple(hovered[center_y, track_x, :3]) == _rgb("media_volume_fill_hover")


def test_secondary_playback_controls_share_the_same_idle_background():
    rgba = build_media_bar_rgba(_state(), width=1024, height=160)
    rects = _button_rects(1024, 160)
    samples = []
    for key in (
        "browse",
        "browse_disc",
        "previous",
        "seek-10",
        "seek+10",
        "next",
        "mute",
        "volume",
    ):
        left, top, _right, bottom = rects[key]
        samples.append(tuple(rgba[(top + bottom) // 2, left + 8, :3]))

    assert len(set(samples)) == 1


def test_media_bar_idle_button_backgrounds_are_opaque():
    rgba = build_media_bar_rgba(_state(), width=1024, height=160)
    rects = _button_rects(1024, 160)
    points = []
    for key in (
        "browse",
        "browse_disc",
        "previous",
        "seek-10",
        "seek+10",
        "next",
        "mute",
        "volume",
        "media_exit",
    ):
        left, top, _right, bottom = rects[key]
        points.append((left + 8, (top + bottom) // 2))
    play_left, play_top, _play_right, play_bottom = rects["play"]
    points.append((play_left + 20, (play_top + play_bottom) // 2))

    assert all(int(rgba[y, x, 3]) == 255 for x, y in points)


def test_media_bar_texture_and_hit_targets_are_stable():
    rgba = build_media_bar_rgba(_state(), width=1024, height=160)
    assert rgba.shape == (160, 1024, 4)
    assert rgba.dtype == np.uint8
    state = _state()
    assert media_bar_hit(0.546, 0.72, state) == "play"
    assert media_bar_hit(0.08, 0.72, state) == "browse"
    assert media_bar_hit(0.22, 0.72, state) == "browse_disc"
    assert media_bar_hit(0.5, 0.25, state).startswith("seek:")
    assert media_bar_hit(1000 / 1024, 0.25, state) == "media_exit"
    assert media_bar_hit(930 / 1024, 0.25, state).startswith("seek:")


def test_media_exit_is_enabled_only_outside_the_file_browser():
    state = _state()
    exit_rect = _button_rects(1024, 160)["media_exit"]
    assert exit_rect == (964, 12, 1012, 60)
    exit_u = (exit_rect[0] + exit_rect[2]) / 2 / 1024
    exit_v = (exit_rect[1] + exit_rect[3]) / 2 / 160

    assert media_control_enabled("media_exit", state)
    assert media_bar_hit(exit_u, exit_v, state) == "media_exit"
    assert not media_control_enabled("media_exit", {**state, "mode": "Browser"})
    assert media_bar_hit(exit_u, exit_v, {**state, "ready": False}) is None


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
    width, height = MEDIA_BROWSER_LAYOUT_SIZE
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
    close_hovered = build_media_browser_rgba(
        _state(),
        hover="browser_close",
        hover_progress=0.5,
    )
    assert np.any(close_hovered != idle)


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

    assert "browse_open" not in rects


def test_media_browser_uses_spatial_theme_radii_and_grid_list_controls():
    assert OPENXR_MENU_RADII["panel"] == 28
    assert OPENXR_MENU_RADII["control"] == 14
    state = {
        **_state(),
        "browser_entries": [
            {"name": f"movie-{index}.mp4", "path": f"F:/movie-{index}.mp4", "kind": "video"}
            for index in range(20)
        ],
    }
    width, height = MEDIA_BROWSER_TEXTURE_SIZE
    buttons = _browser_button_rects(width, height)

    for mode, action in (("grid", "browser_view_grid"), ("list", "browser_view_list")):
        view_state = {**state, "browser_view_mode": mode}
        layout = _browser_layout(view_state, width, height)
        step, visible_count, max_offset = media_browser_scroll_metrics(view_state)
        assert layout["view_mode"] == mode
        assert step == (3 if mode == "grid" else 1)
        assert visible_count == (9 if mode == "grid" else 10)
        assert max_offset == (12 if mode == "grid" else 10)
        rect = buttons[action]
        assert media_browser_hit(
            *_browser_uv((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2),
            view_state,
        ) == (action, None)
        assert build_media_browser_rgba(view_state).shape == (height, width, 4)


def test_media_browser_scrollbar_tracks_grid_offsets_and_is_hittable():
    state = {
        **_state(),
        "browser_entries": [
            {"name": f"movie-{index}.mp4", "path": f"F:/movie-{index}.mp4", "kind": "video"}
            for index in range(30)
        ],
        "browser_view_mode": "grid",
    }
    track, top_thumb = media_browser_scrollbar_metrics(state, 0)
    _step, _visible, maximum = media_browser_scroll_metrics(state)
    _end_track, bottom_thumb = media_browser_scrollbar_metrics(state, maximum)

    assert top_thumb is not None and bottom_thumb is not None
    assert track[2] - track[0] == 28
    assert top_thumb[1] == track[1]
    assert bottom_thumb[3] == track[3]
    assert media_browser_scrollbar_offset(state, track[1]) == 0
    assert media_browser_scrollbar_offset(state, track[3]) == maximum

    hit = media_browser_hit(
        (track[0] + track[2]) / 2 / MEDIA_BROWSER_LAYOUT_SIZE[0],
        (track[3] - track[1]) / 2 / MEDIA_BROWSER_LAYOUT_SIZE[1],
        state,
    )
    assert hit is not None and hit[0] == "browser_scroll"


def test_media_browser_file_sort_button_is_hittable():
    state = {
        **_state(),
        "browser_entries": [
            {"name": f"movie-{index}.mp4", "path": f"F:/movie-{index}.mp4", "kind": "video"}
            for index in range(3)
        ],
    }
    width, height = MEDIA_BROWSER_TEXTURE_SIZE
    rect = _browser_button_rects(width, height)["browser_sort"]
    assert media_browser_hit(
        *_browser_uv((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2),
        state,
    ) == ("browser_sort", None)


def test_media_browser_stop_hit_requests_local_viewer_shutdown():
    state = {**_state(), "mode": "Browser", "browser_visible": True}
    width, height = MEDIA_BROWSER_TEXTURE_SIZE
    rect = _browser_button_rects(width, height)["viewer_stop"]
    layout = _browser_layout(state, width, height)
    assert rect[2] - rect[0] == 168
    assert rect[3] - rect[1] == 64
    assert rect[0] >= 32 and rect[1] >= 32
    assert layout["locations_heading_y"] >= rect[3] + 18
    assert layout["mount_top"] >= layout["locations_heading_y"] + 38
    hit = media_browser_hit(
        *_browser_uv((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2),
        state,
    )
    assert hit == ("viewer_stop", None)

    shutdown = Event()
    viewer = object.__new__(VulkanLocalViewer)
    viewer.config = SimpleNamespace(on_stop_requested=shutdown.set)
    viewer._dispatch_media_hit(hit[0])

    assert shutdown.is_set()
    rendered = build_media_browser_rgba(state)
    assert rendered.shape == (height, width, 4)


def test_media_browser_grid_preview_and_add_button_share_card_geometry():
    entry = {"name": "movie.mp4", "path": "F:/movie.mp4", "kind": "video"}
    state = {**_state(), "browser_entries": [entry]}
    layout = _browser_layout(state, *MEDIA_BROWSER_TEXTURE_SIZE)
    card = _browser_grid_card_rect(layout, 0)
    preview = np.empty((8, 8, 3), dtype=np.uint8)
    preview[:] = (210, 32, 48)
    rgba = build_media_browser_rgba(state, thumbnails={entry["path"]: preview})
    scale_x = MEDIA_BROWSER_TEXTURE_SIZE[0] / MEDIA_BROWSER_LAYOUT_SIZE[0]
    scale_y = MEDIA_BROWSER_TEXTURE_SIZE[1] / MEDIA_BROWSER_LAYOUT_SIZE[1]
    sample_y = round((card[1] + 40) * scale_y)
    sample_x = round((card[0] + 40) * scale_x)
    assert tuple(rgba[sample_y, sample_x, :3]) == (210, 32, 48)

    add_x, add_y = card[2] - 32, card[3] - 32
    assert media_browser_hit(*_browser_uv(add_x, add_y), state) == (
        "playlist_add",
        entry["path"],
    )

    list_state = {**state, "browser_view_mode": "list"}
    list_layout = _browser_layout(list_state, *MEDIA_BROWSER_TEXTURE_SIZE)
    row_y = list_layout["file_rows_y"] + list_layout["file_row_height"] / 2
    assert media_browser_hit(
        *_browser_uv(MEDIA_BROWSER_LAYOUT_SIZE[0] - 70, row_y), list_state
    ) == ("playlist_add", entry["path"])


def test_media_browser_thumbnail_cache_remembers_failed_previews(monkeypatch):
    path = "F:/unreadable.mp4"
    completed = Event()
    calls = []

    def failed_thumbnail(request_path, _kind, _size=(384, 216)):
        calls.append(request_path)
        completed.set()
        return None

    monkeypatch.setattr(media_ui, "_load_media_browser_thumbnail", failed_thumbnail)
    cache = MediaBrowserThumbnailCache()
    entries = [{"name": "unreadable.mp4", "path": path, "kind": "video"}]
    try:
        cache.visible_thumbnails(entries, scroll_offset=0, view_mode="grid")
        assert completed.wait(1.0)
        deadline = time.monotonic() + 1.0
        revision = 0
        while revision == 0 and time.monotonic() < deadline:
            _thumbnails, revision = cache.visible_thumbnails(
                entries, scroll_offset=0, view_mode="grid"
            )
            if revision == 0:
                time.sleep(0.005)
        assert revision > 0
        cache.visible_thumbnails(entries, scroll_offset=0, view_mode="grid")
        assert calls == [path]
    finally:
        cache.close()


def test_browser_footer_actions_use_content_inset_and_stable_group_gap():
    width, height = MEDIA_BROWSER_LAYOUT_SIZE
    rects = _browser_button_rects(width, height)
    previous = rects["playlist_previous"]
    next_button = rects["playlist_next"]
    clear = rects["playlist_clear"]

    assert previous[0] == 48
    assert height - max(previous[3], next_button[3], clear[3]) == 48
    assert next_button[0] - previous[2] == 24
    assert clear[0] - next_button[2] == 24
    assert "browse_open" not in rects


def test_browser_mounts_and_row_bounds_are_clickable_only_inside_geometry():
    state = {
        **_state(),
        "browser_view_mode": "list",
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
    assert media_browser_hit(0.72, 0.12, state) is None


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


def test_local_viewer_dispatches_absolute_seek_volume_and_browser_open():
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

    assert actions == [
        ("seek", 60.0),
        ("volume", 0.25),
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


def test_half_sbs_browser_fills_both_eyes_and_keeps_click_targets_aligned():
    viewer = object.__new__(VulkanLocalViewer)
    viewer.config = SimpleNamespace(display_mode="Half-SBS")
    eye_rects = ((0, 0, 960, 1080), (960, 0, 1920, 1080))
    browser_rects = tuple(
        viewer._media_rect_for_eye(rect, browser=True) for rect in eye_rects
    )
    state = {
        **_state(),
        "browser_mounts": [
            {"name": "User folder", "path": "C:/Users/test", "kind": "home"},
        ],
    }

    for rect in browser_rects:
        left, top, right, bottom = rect
        width, height = right - left, bottom - top
        assert width == 470
        assert height == 705
        click_x = left + (250 / 1600) * width
        click_y = top + (186 / 1200) * height
        hit_u = (click_x - left) / width
        hit_v = (click_y - top) / height
        assert media_browser_hit(hit_u, hit_v, state) == (
            "mount",
            "C:/Users/test",
        )
    assert browser_rects[1][0] > browser_rects[0][0]


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


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS Local Viewer cursor policy")
def test_local_viewer_virtual_cursor_is_rendered_for_media_only():
    pointer_rects = []

    class PointerOverlay:
        def record_rects(self, _cmd, _image, _index, rects, **_kwargs):
            pointer_rects.append(rects)
            return True

    viewer = object.__new__(VulkanLocalViewer)
    viewer.config = SimpleNamespace(
        capture_mode="Monitor",
        pointer_mapping_enabled=True,
        display_mode="Half-SBS",
    )
    viewer.vk = SimpleNamespace(VK_IMAGE_LAYOUT_PRESENT_SRC_KHR=1000001002)
    viewer._media_bar_overlay = None
    viewer._media_browser_overlay = None
    viewer._media_pointer_overlay = PointerOverlay()
    viewer._media_pointer_visible = lambda: True
    viewer._media_pointer_uv = (0.5, 0.5)
    viewer._media_browser_visible = lambda: False
    viewer._media_bar_visible = lambda: False

    eyes = ((0, 0, 960, 1080), (960, 0, 1920, 1080))
    drawn, _layout = viewer._record_media_overlays(
        None, None, 0, eyes, source_layout=0
    )
    assert drawn is False
    assert pointer_rects == []

    viewer.config.capture_mode = "Video"
    drawn, layout = viewer._record_media_overlays(
        None, None, 0, eyes, source_layout=0
    )

    assert drawn is True
    assert layout == viewer.vk.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
    assert pointer_rects == [((468, 516, 24, 48), (1428, 516, 24, 48))]


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
    for action in ("playlist_add", "playlist_sort", "playlist_clear"):
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
    ], "browser_view_mode": "list"}
    width, height = MEDIA_BROWSER_TEXTURE_SIZE
    layout = _browser_layout(state, width, height)
    x = MEDIA_BROWSER_LAYOUT_SIZE[0] - 70
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


def test_local_file_sort_control_dispatches_shared_command():
    actions = []
    viewer = object.__new__(VulkanLocalViewer)
    viewer._media_state = {"browser_sort_descending": False}
    viewer.config = SimpleNamespace(
        media_action_callback=lambda action, value: actions.append((action, value)) or True
    )

    viewer._dispatch_media_hit("browser_sort")

    assert actions == [("browser_sort", {"descending": True})]
