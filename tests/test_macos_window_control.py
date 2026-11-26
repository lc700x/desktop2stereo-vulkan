from __future__ import annotations

from types import SimpleNamespace
import io

import pytest

from viewer import window_control


pytestmark = pytest.mark.skipif(
    window_control.OS_NAME != "Darwin",
    reason="macOS Cocoa window policy",
)


class _FakeWindow:
    def __init__(self, *, visible: bool = True) -> None:
        self.behavior = None
        self.level = None
        self.front_count = 0
        self.visible = visible

    def windowNumber(self):
        return 90210

    def isVisible(self):
        return self.visible

    def setCollectionBehavior_(self, value):
        self.behavior = value

    def setCanHide_(self, _value):
        pass

    def setHidesOnDeactivate_(self, _value):
        pass

    def setLevel_(self, value):
        self.level = value

    def orderFrontRegardless(self):
        self.front_count += 1


@pytest.mark.parametrize("viewer_fullscreen", [False, True])
def test_sbs_window_joins_and_stays_above_another_apps_fullscreen_space(
    monkeypatch: pytest.MonkeyPatch,
    viewer_fullscreen: bool,
) -> None:
    fake_window = _FakeWindow()
    appkit = SimpleNamespace(
        NSApplicationActivationPolicyRegular=0,
        NSApplicationActivationPolicyAccessory=1,
        NSWindowCollectionBehaviorDefault=0,
        NSWindowCollectionBehaviorCanJoinAllSpaces=1 << 0,
        NSWindowCollectionBehaviorFullScreenAuxiliary=1 << 1,
        NSWindowCollectionBehaviorCanJoinAllApplications=1 << 2,
        NSWindowCollectionBehaviorIgnoresCycle=1 << 3,
        NSWindowCollectionBehaviorStationary=1 << 4,
        NSFloatingWindowLevel=20,
        NSStatusWindowLevel=25,
        NSNormalWindowLevel=0,
    )
    application = SimpleNamespace(policy=None)
    application.setActivationPolicy_ = lambda value: setattr(application, "policy", value)
    appkit.NSApplication = SimpleNamespace(sharedApplication=lambda: application)
    monkeypatch.setattr(window_control, "_macos_activation_policy", None)
    monkeypatch.setattr(window_control, "AppKit", appkit)
    monkeypatch.setattr(window_control, "_cocoa_window", lambda _window: fake_window)
    monkeypatch.setattr(window_control, "_macos_window_pin_times", {})

    window_control.pin_macos_viewer_window(object(), fullscreen=viewer_fullscreen)

    assert fake_window.behavior & appkit.NSWindowCollectionBehaviorCanJoinAllSpaces
    assert fake_window.behavior & appkit.NSWindowCollectionBehaviorFullScreenAuxiliary
    assert fake_window.level >= appkit.NSFloatingWindowLevel
    assert fake_window.front_count == 1
    assert application.policy == appkit.NSApplicationActivationPolicyAccessory


def test_lsfg_window_switcher_uses_regular_app_policy_without_dropping_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_window = _FakeWindow()
    appkit = SimpleNamespace(
        NSApplicationActivationPolicyRegular=0,
        NSApplicationActivationPolicyAccessory=1,
        NSWindowCollectionBehaviorDefault=0,
        NSWindowCollectionBehaviorCanJoinAllSpaces=1 << 0,
        NSWindowCollectionBehaviorFullScreenAuxiliary=1 << 1,
        NSWindowCollectionBehaviorCanJoinAllApplications=1 << 2,
        NSWindowCollectionBehaviorIgnoresCycle=1 << 3,
        NSWindowCollectionBehaviorStationary=1 << 4,
        NSFloatingWindowLevel=20,
        NSStatusWindowLevel=25,
        NSNormalWindowLevel=0,
    )
    application = SimpleNamespace(policy=None)

    def set_activation_policy(value):
        application.policy = value
        return True

    application.setActivationPolicy_ = set_activation_policy
    appkit.NSApplication = SimpleNamespace(sharedApplication=lambda: application)
    monkeypatch.setattr(window_control, "_macos_activation_policy", None)
    monkeypatch.setattr(window_control, "AppKit", appkit)
    monkeypatch.setattr(window_control, "_cocoa_window", lambda _window: fake_window)
    monkeypatch.setattr(window_control, "_macos_window_pin_times", {})

    window_control.pin_macos_viewer_window(
        object(), fullscreen=True, show_in_app_switcher=True
    )

    assert application.policy == appkit.NSApplicationActivationPolicyRegular
    assert fake_window.behavior & appkit.NSWindowCollectionBehaviorCanJoinAllSpaces
    assert fake_window.behavior & appkit.NSWindowCollectionBehaviorFullScreenAuxiliary
    assert not fake_window.behavior & appkit.NSWindowCollectionBehaviorIgnoresCycle
    assert fake_window.level == appkit.NSStatusWindowLevel
    assert fake_window.front_count == 1


def test_sbs_window_is_reordered_when_appkit_reports_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_window = _FakeWindow(visible=False)
    appkit = SimpleNamespace(
        NSApplicationActivationPolicyRegular=0,
        NSApplicationActivationPolicyAccessory=1,
        NSWindowCollectionBehaviorDefault=0,
        NSWindowCollectionBehaviorCanJoinAllSpaces=1 << 0,
        NSWindowCollectionBehaviorFullScreenAuxiliary=1 << 1,
        NSWindowCollectionBehaviorCanJoinAllApplications=1 << 2,
        NSWindowCollectionBehaviorIgnoresCycle=1 << 3,
        NSWindowCollectionBehaviorStationary=1 << 4,
        NSFloatingWindowLevel=20,
        NSStatusWindowLevel=25,
        NSNormalWindowLevel=0,
    )
    application = SimpleNamespace(policy=None)
    application.setActivationPolicy_ = lambda value: setattr(application, "policy", value)
    appkit.NSApplication = SimpleNamespace(sharedApplication=lambda: application)
    monkeypatch.setattr(window_control, "_macos_activation_policy", None)
    monkeypatch.setattr(window_control, "AppKit", appkit)
    monkeypatch.setattr(window_control, "_cocoa_window", lambda _window: fake_window)
    monkeypatch.setattr(window_control, "_macos_window_pin_times", {})

    window_control.pin_macos_viewer_window(object(), fullscreen=True)

    assert fake_window.front_count == 1


def test_cursor_snapshot_preserves_appkit_hotspot_y_origin(monkeypatch) -> None:
    from PIL import Image

    image = Image.new("RGBA", (17, 23), (255, 255, 255, 255))
    tiff = io.BytesIO()
    image.save(tiff, format="TIFF")

    class _Size:
        width = 17.0
        height = 23.0

    class _Point:
        x = 4.0
        y = 4.0

    class _Image:
        def size(self):
            return _Size()

        def TIFFRepresentation(self):
            return tiff.getvalue()

    class _Cursor:
        def image(self):
            return _Image()

        def hotSpot(self):
            return _Point()

    monkeypatch.setattr(
        window_control,
        "_system_cursor",
        lambda: (_Cursor(), 17),
    )
    monkeypatch.setattr(window_control, "_cursor_scale", lambda _point: 1.0)
    monkeypatch.setattr(
        window_control,
        "NSEvent",
        SimpleNamespace(mouseLocation=lambda: _Point()),
    )

    snapshot = window_control.capture_cursor_snapshot("Full-SBS")

    assert snapshot is not None
    assert snapshot.hotspot_x == 4
    assert snapshot.hotspot_y == 4


def test_cursor_overlay_maps_one_screen_point_into_both_sbs_eyes() -> None:
    class _Content:
        def bounds(self):
            return SimpleNamespace(size=SimpleNamespace(width=100.0, height=50.0))

        def convertPoint_fromView_(self, point, _view):
            return point

    class _Window:
        def convertPointFromScreen_(self, _point):
            # AppKit window coordinates use a bottom-left origin.
            return SimpleNamespace(x=25.0, y=10.0)

        def contentView(self):
            return _Content()

    snapshot = window_control.CursorSnapshot(
        key=(1, "full-sbs", 1.0),
        rgba=b"\x00" * (10 * 20 * 4),
        width=10,
        height=20,
        hotspot_x=2,
        hotspot_y=3,
    )

    rects = window_control.cursor_overlay_rects(
        _Window(),
        ((0, 0, 100, 50), (100, 0, 200, 50)),
        snapshot,
        (25, 10),
    )

    assert rects == ((23, 37, 10, 20), (123, 37, 10, 20))


def test_native_cursor_suppression_reasserts_after_pointer_motion_without_leaking_counts(
    monkeypatch,
) -> None:
    display_id = 77
    cg_calls = []
    appkit_calls = []

    class _Quartz:
        @staticmethod
        def CGDisplayHideCursor(value):
            cg_calls.append(("hide", value))
            return 0

        @staticmethod
        def CGDisplayShowCursor(value):
            cg_calls.append(("show", value))
            return 0

        @staticmethod
        def CGMainDisplayID():
            return display_id

    class _Cursor:
        @staticmethod
        def setHiddenUntilMouseMoves_(value):
            appkit_calls.append(("until_move", bool(value)))

        @staticmethod
        def hide():
            appkit_calls.append(("hide",))

        @staticmethod
        def unhide():
            appkit_calls.append(("unhide",))

    screen = SimpleNamespace(
        deviceDescription=lambda: {"NSScreenNumber": display_id},
    )
    ns_window = SimpleNamespace(screen=lambda: screen)
    positions = iter(((10, 10), (11, 10)))

    monkeypatch.setattr(window_control, "Quartz", _Quartz)
    monkeypatch.setattr(window_control, "NSCursor", _Cursor)
    monkeypatch.setattr(
        window_control,
        "_cursor_passthrough_windows",
        {1: (ns_window, True)},
    )
    monkeypatch.setattr(window_control, "cursor_screen_position", lambda: next(positions))
    monkeypatch.setattr(window_control, "_hidden_cursor_display_id", None)
    monkeypatch.setattr(window_control, "_cursor_suppression_target_display_id", None)
    monkeypatch.setattr(window_control, "_cursor_suppression_cursor_position", None)
    monkeypatch.setattr(window_control, "_appkit_cursor_hidden", False)

    window_control._update_native_cursor_suppression()
    window_control._update_native_cursor_suppression()
    window_control._restore_hidden_cursor()

    assert cg_calls == [
        ("hide", display_id),
        ("show", display_id),
        ("hide", display_id),
        ("show", display_id),
    ]
    assert ("until_move", True) not in appkit_calls
    assert appkit_calls.count(("hide",)) == 1
    assert appkit_calls.count(("unhide",)) == 1
