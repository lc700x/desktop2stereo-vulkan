import sys
import types

from capture import CaptureConfig, create_capture_runner
from capture.factory import get_desktop_grabber_class
from capture.factory import DesktopDuplicationFallbackRunner
from capture.runners import PollingCaptureRunner


class FakeGrabber:
    pass


def _install_fake_backend(monkeypatch, module_name):
    module = types.ModuleType(module_name)
    module.DesktopGrabber = FakeGrabber
    monkeypatch.setitem(sys.modules, module_name, module)
    return module


def test_windows_dxcamera_selects_dxcamera_backend(monkeypatch):
    _install_fake_backend(monkeypatch, "capture.backends.windows_dxcamera")

    cls = get_desktop_grabber_class(CaptureConfig(os_name="Windows", capture_tool="DXCamera"))

    assert cls is FakeGrabber


def test_windows_capture_tools_select_event_runner():
    for capture_tool in ["WindowsCapture", "WindowsCaptureCUDA", "WindowsCaptureROCm"]:
        runner = create_capture_runner(CaptureConfig(os_name="Windows", capture_tool=capture_tool))

        assert runner.__class__.__name__ == "WindowsCaptureEventRunner"


def test_gpu_required_windows_capture_selects_cuda_or_rocm(monkeypatch):
    for hip, expected in ((None, "WindowsCaptureCUDA"), ("7.1", "WindowsCaptureROCm")):
        torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: True),
            version=types.SimpleNamespace(hip=hip),
        )
        monkeypatch.setitem(sys.modules, "torch", torch)

        runner = create_capture_runner(
            CaptureConfig(
                os_name="Windows",
                capture_tool="DXCamera",
                gpu_required=True,
            )
        )

        assert runner.capture_tool == expected


def test_gpu_required_capture_selects_the_installed_vendor_tool(monkeypatch):
    import types

    for hip, expected in ((None, "WindowsCaptureCUDA"), ("7.1", "WindowsCaptureROCm")):
        torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: True),
            version=types.SimpleNamespace(hip=hip),
        )
        monkeypatch.setitem(sys.modules, "torch", torch)

        runner = create_capture_runner(
            CaptureConfig(
                os_name="Windows",
                capture_tool="DXCamera",
                gpu_required=True,
            )
        )

        assert runner.capture_tool == expected


def test_macos_screencapturekit_selects_sck_backend(monkeypatch):
    _install_fake_backend(monkeypatch, "capture.backends.macos_screencapturekit")

    cls = get_desktop_grabber_class(CaptureConfig(os_name="Darwin", capture_tool="ScreenCaptureKit"))

    assert cls is FakeGrabber


def test_macos_non_sck_selects_coregraphics_backend(monkeypatch):
    _install_fake_backend(monkeypatch, "capture.backends.macos_coregraphics")

    cls = get_desktop_grabber_class(CaptureConfig(os_name="Darwin", capture_tool="CoreGraphics"))

    assert cls is FakeGrabber


def test_linux_selects_mss_backend(monkeypatch):
    _install_fake_backend(monkeypatch, "capture.backends.linux_mss")

    cls = get_desktop_grabber_class(CaptureConfig(os_name="Linux", capture_tool="DXCamera"))

    assert cls is FakeGrabber


def test_polling_tools_create_polling_runner(monkeypatch):
    _install_fake_backend(monkeypatch, "capture.backends.windows_dxcamera")

    for capture_tool in ["DXCamera"]:
        runner = create_capture_runner(CaptureConfig(os_name="Windows", capture_tool=capture_tool))

        assert isinstance(runner, PollingCaptureRunner)


def test_desktop_duplication_selects_explicit_backend():
    cls = get_desktop_grabber_class(
        CaptureConfig(os_name="Windows", capture_tool="DesktopDuplication")
    )

    assert cls.__module__.endswith("windows_desktop_duplication")


def test_desktop_duplication_is_polling_runner():
    runner = create_capture_runner(
        CaptureConfig(os_name="Windows", capture_tool="DesktopDuplication")
    )

    assert isinstance(runner, PollingCaptureRunner)


def test_desktop_duplication_tries_windows_capture_when_native_bridge_is_missing(monkeypatch):
    monkeypatch.setattr("capture.factory._desktop_duplication_native_available", lambda: False)

    runner = create_capture_runner(
        CaptureConfig(os_name="Windows", capture_tool="DesktopDuplication")
    )

    assert isinstance(runner, DesktopDuplicationFallbackRunner)
