"""macOS capture output-resolution normalization tests (ported)."""

import pytest

pytest.importorskip("Quartz")  # macOS pyobjc only


def test_coregraphics_accepts_tuple_output_resolution() -> None:
    from capture.backends.macos_coregraphics import DesktopGrabber

    grabber = DesktopGrabber(output_resolution=(1280, 720), fps=30)
    assert grabber.scaled_height == 720

    grabber2 = DesktopGrabber(output_resolution=1080, fps=30)
    assert grabber2.scaled_height == 1080


def test_screencapturekit_accepts_tuple_output_resolution() -> None:
    from capture.backends.macos_screencapturekit import DesktopGrabber

    grabber = DesktopGrabber(output_resolution=(1920, 1080), fps=30)
    assert grabber.scaled_height == 1080
    assert grabber.frame_format == "bgra"
