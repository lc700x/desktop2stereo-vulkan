import pytest
from types import SimpleNamespace

pytest.importorskip("ScreenCaptureKit")
from CoreMedia import CMTimeGetSeconds

from capture.backends import macos_screencapturekit as sck


@pytest.mark.parametrize(
    "fps,refresh,expected",
    [
        (60, 60, 0.0),
        (60, 59.94, 0.0),
        (120, 60, 0.0),
        (30, 60, 1 / 30),
        (60, 120, 1 / 60),
        (59, 59.94, 1 / 59),
        (60, 0, 1 / 60),
    ],
)
def test_sck_uses_native_clock_only_when_target_covers_display(fps, refresh, expected):
    source = sck.DesktopGrabber.__new__(sck.DesktopGrabber)
    source._source_refresh_hz = lambda: refresh

    assert CMTimeGetSeconds(source._minimum_frame_interval(fps)) == pytest.approx(expected)


def test_sck_rate_update_switches_between_native_and_capped_clock(monkeypatch):
    settings = {}

    class Configuration:
        @classmethod
        def alloc(cls):
            return cls()

        def init(self):
            return self

        def setWidth_(self, value):
            settings["width"] = value

        def setHeight_(self, value):
            settings["height"] = value

        def setShowsCursor_(self, value):
            settings["cursor"] = value

        def setPixelFormat_(self, value):
            settings["format"] = value

        def setQueueDepth_(self, value):
            settings["queue"] = value

        def setMinimumFrameInterval_(self, value):
            settings["interval"] = CMTimeGetSeconds(value)

    monkeypatch.setattr(sck.SCK, "SCStreamConfiguration", Configuration)
    source = sck.DesktopGrabber.__new__(sck.DesktopGrabber)
    source.width, source.height, source.with_cursor = 1920, 1080, True
    source.fps = source._last_fps_update_request = 30
    target = [60]
    source._fps_provider = lambda: target[0]
    source._source_refresh_hz = lambda: 60
    source._fps_update_error_logged = False
    source._stream = type(
        "Stream",
        (),
        {
            "updateConfiguration_completionHandler_": (
                lambda self, config, done: done(None)
            ),
        },
    )()

    source._sync_capture_fps()
    assert source.fps == 60
    assert settings["interval"] == 0.0
    target[0] = 30
    source._sync_capture_fps()
    assert source.fps == 30
    assert settings["interval"] == pytest.approx(1 / 30)


def test_sck_uses_display_ceiling_for_variable_refresh_screens(monkeypatch):
    import Quartz

    source = sck.DesktopGrabber.__new__(sck.DesktopGrabber)
    source._display = SimpleNamespace(displayID=lambda: 1)
    monkeypatch.setattr(Quartz, "CGDisplayCopyDisplayMode", lambda _: object())
    monkeypatch.setattr(Quartz, "CGDisplayModeGetRefreshRate", lambda _: 60.0)
    screen = SimpleNamespace(
        deviceDescription=lambda: {"NSScreenNumber": 1},
        maximumFramesPerSecond=lambda: 120,
    )
    monkeypatch.setattr(sck, "NSScreen", SimpleNamespace(screens=lambda: [screen]))

    assert source._source_refresh_hz() == 120
    assert CMTimeGetSeconds(source._minimum_frame_interval(60)) == pytest.approx(1 / 60)
