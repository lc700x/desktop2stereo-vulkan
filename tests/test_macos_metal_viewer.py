import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from media_ui import MEDIA_BAR_TEXTURE_SIZE, media_browser_hit
from viewer.vulkan_local_viewer import VulkanLocalViewer

import viewer.macos_metal_viewer as metal_viewer


_VIEWER_SOURCE = Path(metal_viewer.__file__).read_text()


def test_metal_viewer_preserves_display_referred_capture_bytes() -> None:
    attach_start = _VIEWER_SOURCE.index("def _attach_metal_layer")
    attach_end = _VIEWER_SOURCE.index("def _register_capture_window", attach_start)
    attach = _VIEWER_SOURCE[attach_start:attach_end]

    assert "self._drawable_pixel_format = Metal.MTLPixelFormatBGRA8Unorm" in attach
    assert "self._source_pixel_format = Metal.MTLPixelFormatRGBA8Unorm" in attach
    assert "layer.setColorspace_(Quartz.CGColorSpaceCreateWithName(Quartz.kCGColorSpaceSRGB))" in attach


def test_metal_viewer_keeps_media_browser_on_gpu_overlay_path() -> None:
    assert "build_media_browser_rgba" in _VIEWER_SOURCE
    assert "texture2d<float> mediaTex [[texture(3)]]" in _VIEWER_SOURCE
    assert "texture2d<float> mediaTex [[texture(4)]]" in _VIEWER_SOURCE
    assert "color = blend_overlay(color, frag, u.mediaRect, mediaTex, s);" in _VIEWER_SOURCE


def test_metal_viewer_allows_prepacked_media_frame() -> None:
    assert "_has_gpu_present_source(frame, self.config)" in _VIEWER_SOURCE
    assert "viewer_frame_np" in _VIEWER_SOURCE


def test_metal_viewer_shows_virtual_cursor_only_for_media_capture() -> None:
    viewer = object.__new__(metal_viewer.MetalLocalViewer)
    viewer.config = SimpleNamespace(
        capture_mode="Monitor",
        display_mode="Half-SBS",
        pointer_mapping_enabled=True,
        hide_cursor_overlay=True,
    )
    viewer._virtual_pointer_texture = object()
    viewer._virtual_pointer_uv = (0.5, 0.5)
    viewer._media_last_activity = 0.0
    eyes = ((0, 0, 960, 1080), (960, 0, 1920, 1080))

    assert viewer._cursor_overlay_rects(eyes) == ((0.0, 0.0, 0.0, 0.0),) * 2

    viewer.config.capture_mode = "Video"
    rects = viewer._cursor_overlay_rects(eyes)

    assert rects[0] != (0.0, 0.0, 0.0, 0.0)
    assert rects[1] != (0.0, 0.0, 0.0, 0.0)


def test_metal_viewer_stops_host_packer_before_closing(monkeypatch) -> None:
    events = []

    class FakePacker:
        def stop(self):
            events.append("packer_stop")

    class FakeViewer:
        def __init__(self, _config):
            pass

        def initialize(self):
            events.append("viewer_initialize")

        def present_blank(self):
            pass

        def close(self):
            events.append("viewer_close")

    runtime_q = queue.Queue()
    runtime_q._d2s_packer = FakePacker()
    monkeypatch.setattr(metal_viewer, "MetalLocalViewer", FakeViewer)
    shutdown_event = threading.Event()
    shutdown_event.set()

    metal_viewer.run_metal_local_viewer(
        runtime_q=runtime_q,
        shutdown_event=shutdown_event,
        config=SimpleNamespace(on_capture_windows_initialized=None),
    )

    assert events == ["viewer_initialize", "packer_stop", "viewer_close"]


def test_metal_viewer_notifies_media_presentation_after_frame(monkeypatch) -> None:
    events = []
    shutdown_event = threading.Event()
    runtime_q = queue.Queue()
    runtime_q.put(
        (
            SimpleNamespace(
                sbs=object(),
                debug_info={
                    "media_generation": 3,
                    "source_frame_index": 0,
                    "source_pts": 0.0,
                },
            ),
            10.0,
        )
    )
    runtime_q.put((SimpleNamespace(sbs=object(), debug_info={}), 11.0))
    runtime_q._d2s_ordered = True

    class FakeViewer:
        def __init__(self, _config):
            pass

        def initialize(self):
            pass

        def present_blank(self):
            pass

        def present(self, _frame):
            shutdown_event.set()
            return True

        def observe_media_presentation(self, result):
            events.append(("observe", result.debug_info))

        def record_frame_latency(self, *_args, **_kwargs):
            pass

        def record_present(self, _reused):
            pass

        def close(self):
            pass

    monkeypatch.setattr(metal_viewer, "MetalLocalViewer", FakeViewer)
    config = SimpleNamespace(
        capture_mode="Video",
        on_capture_windows_initialized=None,
        on_media_presented=lambda metadata, at: events.append(
            ("callback", metadata, at)
        ),
        on_breakdown_add_time=None,
        on_breakdown_inc=None,
        window_preview=False,
    )

    metal_viewer.run_metal_local_viewer(
        runtime_q=runtime_q,
        shutdown_event=shutdown_event,
        config=config,
    )

    assert events[0] == (
        "observe",
        {
            "media_generation": 3,
            "source_frame_index": 0,
            "source_pts": 0.0,
        },
    )
    assert events[1][0] == "callback"
    assert events[1][1]["source_frame_index"] == 0
    assert runtime_q.qsize() == 1


def test_metal_viewer_half_sbs_browser_panel_compresses_width() -> None:
    rect = metal_viewer.MetalLocalViewer._media_browser_rect_for_eye((0, 0, 960, 1080), "Half-SBS")
    width = rect[2] - rect[0]
    height = rect[3] - rect[1]
    assert width == 470
    assert height == 705


def test_metal_browser_shader_rects_keep_both_eye_origins_and_sizes() -> None:
    viewer = object.__new__(metal_viewer.MetalLocalViewer)
    viewer.config = SimpleNamespace(
        display_mode="Half-SBS",
        media_state_path="",
        media_session_id="",
        cursor_passthrough=False,
    )
    viewer.window = None
    viewer._media_state = {
        "mode": "browser",
        "ready": True,
        "browser_visible": True,
        "browser_mounts": [
            {"name": "User folder", "path": "/Users/test", "kind": "home"},
        ],
    }
    viewer._media_overlay_texture = object()

    eye_rects = ((0, 0, 960, 1080), (960, 0, 1920, 1080))
    shader_rects = viewer._media_overlay_rects(eye_rects)

    assert len(shader_rects) == 2
    assert len(viewer._media_browser_rects) == 2
    for shader_rect, hit_rect in zip(shader_rects, viewer._media_browser_rects):
        left, top, width, height = shader_rect
        hit_left, hit_top, hit_right, hit_bottom = hit_rect
        assert (left, top, width, height) == pytest.approx(
            (hit_left, hit_top, hit_right - hit_left, hit_bottom - hit_top)
        )
        assert width == pytest.approx(470)
        assert height == pytest.approx(705)
        mount_u, mount_v = 250 / 1600, 186 / 1200
        click_x = left + mount_u * width
        click_y = top + mount_v * height
        hit_u = (click_x - hit_left) / (hit_right - hit_left)
        hit_v = (click_y - hit_top) / (hit_bottom - hit_top)
        assert media_browser_hit(hit_u, hit_v, viewer._media_state) == (
            "mount",
            "/Users/test",
        )
    assert shader_rects[1][0] > shader_rects[0][0]


def test_metal_viewer_refreshes_and_places_media_controls(monkeypatch) -> None:
    class FakeTexture:
        def __init__(self):
            self.uploads = []

        def replaceRegion_mipmapLevel_withBytes_bytesPerRow_(self, *args):
            self.uploads.append(args)

    texture = FakeTexture()
    rendered = []
    viewer = object.__new__(metal_viewer.MetalLocalViewer)
    viewer.config = SimpleNamespace(
        display_mode="Half-SBS",
        locale="EN",
        media_state_path="",
        media_session_id="",
        cursor_passthrough=False,
    )
    viewer.window = None
    viewer._media_state = {
        "mode": "Video",
        "ready": True,
        "playing": True,
        "duration": 60.0,
        "position": 12.0,
        "volume": 1.0,
    }
    viewer._media_state_read_at = 0.0
    viewer._media_last_activity = time.monotonic()
    viewer._media_overlay_texture = None
    viewer._media_overlay_size = (0, 0)
    viewer._media_overlay_key = None
    viewer._media_browser_scroll = 0
    viewer._media_browser_hover = None
    viewer._media_bar_hover = None
    viewer._media_bar_hit_uv = None
    viewer._media_drag_action = None
    viewer._media_drag_eye = None
    viewer._media_panel_hovered = False
    viewer._media_browser_rects = ()
    viewer._media_bar_rects = ()
    viewer._media_bar_hover = None
    viewer._last_eye_rects = ()
    viewer._source_pixel_format = 1
    viewer._make_texture = lambda *_args: texture
    monkeypatch.setattr(
        metal_viewer,
        "Metal",
        SimpleNamespace(MTLRegionMake2D=lambda *_args: object()),
    )
    monkeypatch.setattr(
        metal_viewer,
        "build_media_bar_rgba",
        lambda *_args, **kwargs: rendered.append(kwargs)
        or SimpleNamespace(shape=(320, 2048, 4), tobytes=lambda: b"media-bar"),
        raising=False,
    )

    eye_rects = ((0, 0, 960, 1080), (960, 0, 1920, 1080))
    viewer._last_eye_rects = eye_rects
    viewer._refresh_media_overlay(eye_rects)
    overlay_rects = viewer._media_overlay_rects(eye_rects)

    assert rendered
    assert (rendered[0]["width"], rendered[0]["height"]) == MEDIA_BAR_TEXTURE_SIZE
    assert len(texture.uploads) == 1
    assert all(width > 0 and height > 0 for _, _, width, height in overlay_rects)
    assert viewer._media_bar_rects[0][3] > viewer._media_bar_rects[0][1]


def test_vulkan_and_metal_use_the_same_media_bar_geometry() -> None:
    eye_rect = (960, 0, 1920, 1080)
    vulkan = object.__new__(VulkanLocalViewer)
    vulkan.config = SimpleNamespace(display_mode="Half-SBS")
    metal = object.__new__(metal_viewer.MetalLocalViewer)
    metal.config = vulkan.config

    assert metal._media_bar_rect_for_eye(eye_rect) == vulkan._media_rect_for_eye(eye_rect)
    assert metal._media_bar_rect_for_eye(eye_rect) == (1219, 906, 1660, 1043)

    browser_rect = metal._media_browser_rect_for_eye(eye_rect, "Half-SBS")
    assert browser_rect == vulkan._media_rect_for_eye(eye_rect, browser=True)
    assert browser_rect == (1205, 187, 1675, 892)


def test_metal_media_controls_dispatch_and_drag_through_shared_actions(monkeypatch) -> None:
    actions = []
    viewer = object.__new__(metal_viewer.MetalLocalViewer)
    viewer.config = SimpleNamespace(
        media_action_callback=lambda action, value=None: actions.append((action, value)) or True,
    )
    viewer._media_state = {
        "mode": "Video",
        "ready": True,
        "playing": True,
        "duration": 100.0,
        "position": 10.0,
        "volume": 0.5,
        "audio_enabled": True,
    }
    viewer._media_overlay_key = None
    viewer._media_drag_action = "seek"
    viewer._media_drag_last_sent_at = 0.0
    monkeypatch.setattr(metal_viewer.time, "monotonic", lambda: 10.0)

    viewer._dispatch_media_bar_hit("play")
    viewer._update_media_drag(0.5)

    expected_seek = 100.0 * metal_viewer.media_bar_value_fraction(
        0.5, action="seek"
    )
    assert actions == [("pause", None), ("seek", expected_seek)]


def test_metal_browser_single_click_opens() -> None:
    actions = []
    viewer = object.__new__(metal_viewer.MetalLocalViewer)
    viewer.config = SimpleNamespace(
        media_action_callback=lambda action, value=None: actions.append((action, value)) or True,
    )
    viewer._media_state = {"browser_selection": "", "browser_revision": 4}
    viewer._media_overlay_key = None
    hit = ("entry", "/Users/test/movie.mp4")

    viewer._dispatch_media_hit(hit)

    assert actions == [("browse_open", {"path": "/Users/test/movie.mp4"})]
    assert viewer._media_state["browser_selection"] == "/Users/test/movie.mp4"


def test_metal_browser_stop_requests_runtime_shutdown() -> None:
    shutdown = threading.Event()
    viewer = object.__new__(metal_viewer.MetalLocalViewer)
    viewer.config = SimpleNamespace(on_stop_requested=shutdown.set)

    viewer._dispatch_media_hit(("viewer_stop", None))

    assert shutdown.is_set()


def test_metal_fps_overlay_visibility_tracks_live_show_fps_provider() -> None:
    state = [False]
    viewer = object.__new__(metal_viewer.MetalLocalViewer)
    viewer.config = SimpleNamespace(
        show_fps=False,
        show_fps_provider=lambda: state[0],
    )
    viewer._fps_overlay_texture = object()
    viewer._fps_overlay_visible = False

    assert viewer._sync_fps_overlay_visibility() is False

    state[0] = True
    assert viewer._sync_fps_overlay_visibility() is True
    assert viewer._fps_overlay_visible is True

    state[0] = False
    assert viewer._sync_fps_overlay_visibility() is False
    assert viewer._fps_overlay_visible is False


def test_metal_fps_report_uses_live_show_fps_provider(monkeypatch) -> None:
    state = [False]
    uploaded = []
    viewer = object.__new__(metal_viewer.MetalLocalViewer)
    viewer.config = SimpleNamespace(
        show_fps=False,
        show_fps_provider=lambda: state[0],
        on_sbs_fps=None,
    )
    viewer._fps_overlay_texture = object()
    viewer._fps_overlay_visible = False

    def upload(panel):
        uploaded.append(panel)
        return True

    viewer._upload_fps_overlay = upload
    monkeypatch.setattr(
        metal_viewer,
        "build_fps_panel_rgba",
        lambda **kwargs: kwargs,
    )

    viewer._report_present_fps(60.0, 300)
    assert uploaded == []

    state[0] = True
    viewer._report_present_fps(60.0, 300)
    assert uploaded == [{"present_fps": 60.0, "capture_target": None}]
    assert viewer._fps_overlay_visible is True

    state[0] = False
    viewer._report_present_fps(60.0, 300)
    assert viewer._fps_overlay_visible is False


def test_metal_cached_present_reuses_last_gpu_texture_for_cursor_redraw(monkeypatch) -> None:
    texture = object()
    calls = []
    viewer = object.__new__(metal_viewer.MetalLocalViewer)
    viewer.window = object()
    viewer.pipeline = object()
    viewer.config = SimpleNamespace()
    viewer._last_present_texture = texture
    viewer._last_present_size = (1920, 1080)
    viewer._sync_fps_overlay_visibility = lambda: calls.append("fps")
    viewer.poll_events = lambda: calls.append("events")
    viewer._present_packed_texture = (
        lambda value, width, height: calls.append((value, width, height)) or True
    )

    assert viewer.present_cached() is True
    assert calls == ["events", "fps", (texture, 1920, 1080)]


def test_metal_cached_present_returns_false_without_a_gpu_frame() -> None:
    viewer = object.__new__(metal_viewer.MetalLocalViewer)
    viewer.window = object()
    viewer.pipeline = object()
    viewer.config = SimpleNamespace()
    viewer._last_present_texture = None
    viewer._last_present_size = (0, 0)

    assert viewer.present_cached() is False


def test_metal_cached_present_accepts_deferred_native_frame() -> None:
    viewer = object.__new__(metal_viewer.MetalLocalViewer)
    viewer._last_present_texture = None
    viewer._last_native_frame = object()

    # The production loop's native deferred path must wake for cached redraws
    # even though it never creates a packed SBS texture.
    assert metal_viewer._has_cached_frame(viewer) is True

    viewer._last_native_frame = None
    viewer._last_present_texture = object()
    assert metal_viewer._has_cached_frame(viewer) is True

    viewer._last_present_texture = None
    assert metal_viewer._has_cached_frame(viewer) is False


def test_metal_native_frame_lease_waits_for_all_cached_commands() -> None:
    released = []

    class Frame:
        def release(self):
            released.append(True)

    viewer = object.__new__(metal_viewer.MetalLocalViewer)
    entry = {
        "frame": Frame(),
        "pending": 2,
        "completed": False,
        "drop": True,
        "released": False,
    }

    viewer._complete_native_entry(entry)
    assert released == []
    viewer._complete_native_entry(entry)
    assert released == [True]
    viewer._complete_native_entry(entry)
    assert released == [True]
