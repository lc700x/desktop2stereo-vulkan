from types import SimpleNamespace

import viewer.macos_metal_viewer as metal_viewer


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
