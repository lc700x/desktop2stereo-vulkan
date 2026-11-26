from __future__ import annotations

import ctypes

import pytest

pytest.importorskip("win32gui", reason="Windows native capture is not installed on this host")

import desktop2stereo.capture.backends.windows_desktop_duplication as desktop_duplication
from desktop2stereo.capture.backends.desktop_duplication_native import (
    NativeDesktopDuplication,
    NativeD3D11TextureFrame,
)
from desktop2stereo.capture.backends.windows_desktop_duplication import DesktopGrabber
from desktop2stereo.stereo_runtime.providers.intel.openvino_remote import (
    OpenVINORemoteTensorCapability,
)


class _Frame:
    def __init__(self) -> None:
        self.released = False

    def release(self) -> None:
        self.released = True


class _Provider:
    def __init__(self, frame: _Frame, error: Exception | None = None) -> None:
        self.frame = frame
        self.error = error

    def predict_native(self, frame):
        assert frame is self.frame
        if self.error is not None:
            raise self.error
        return "depth-result"


def _grabber_with_frame(frame: _Frame) -> DesktopGrabber:
    grabber = DesktopGrabber.__new__(DesktopGrabber)
    grabber.acquire_native_frame = lambda: frame
    return grabber


def test_probe_does_not_promote_staging_readback_to_end_to_end_zero_copy(monkeypatch):
    monkeypatch.setattr(
        desktop_duplication,
        "probe_native",
        lambda: {"available": True, "texture_consumer": True},
    )
    monkeypatch.setattr(
        desktop_duplication,
        "probe_openvino_remote_tensor",
        lambda: OpenVINORemoteTensorCapability(
            runtime_available=True,
            directx_remote_tensor=True,
            native_bridge_required=False,
            native_bridge_available=True,
        ),
    )
    monkeypatch.setattr(
        desktop_duplication,
        "probe_onevpl_d3d11",
        lambda: {"available": True, "backend": "onevpl_d3d11_surface"},
    )
    monkeypatch.setattr(
        desktop_duplication,
        "probe_d3d11_sbs_surface",
        lambda: {"available": True, "backend": "d3d11_sbs_surface"},
    )

    result = desktop_duplication.probe()

    assert result["native_inference_zero_copy_ready"] is True
    assert result["zero_copy_ready"] is False
    assert "staging readback" in result["zero_copy_reason"]


class _AccessLostLibrary:
    def __init__(self) -> None:
        self.calls = 0

    def d2s_desktop_duplication_acquire(self, _handle, _texture, width, height, luid):
        self.calls += 1
        if self.calls == 1:
            return -2
        ctypes.cast(width, ctypes.POINTER(ctypes.c_int))[0] = 1920
        ctypes.cast(height, ctypes.POINTER(ctypes.c_int))[0] = 1080
        ctypes.cast(luid, ctypes.POINTER(ctypes.c_uint64))[0] = 0x1234
        return 1


class _ReadbackLibrary:
    def d2s_desktop_duplication_copy_frame(self, _handle, output, capacity, stride, width, height):
        assert capacity == 2 * 2 * 4
        ctypes.cast(stride, ctypes.POINTER(ctypes.c_int))[0] = 8
        ctypes.cast(width, ctypes.POINTER(ctypes.c_int))[0] = 2
        ctypes.cast(height, ctypes.POINTER(ctypes.c_int))[0] = 2
        ctypes.memmove(output, bytes(range(16)), 16)
        return 16


def test_native_frame_readback_uses_tightly_packed_bgra_buffer() -> None:
    owner = NativeDesktopDuplication.__new__(NativeDesktopDuplication)
    owner._lib = _ReadbackLibrary()
    owner._handle = ctypes.c_void_p(1)
    frame = NativeD3D11TextureFrame(owner, ctypes.c_void_p(2), 2, 2, 0)
    result = frame.readback_bgra()
    assert result.shape == (2, 2, 4)
    assert result[1, 1, 3] == 15


def test_native_acquire_retries_after_access_lost_recreation() -> None:
    owner = NativeDesktopDuplication.__new__(NativeDesktopDuplication)
    owner._lib = _AccessLostLibrary()
    owner._handle = ctypes.c_void_p(1)
    result = owner.acquire()
    assert result["width"] == 1920
    assert result["height"] == 1080
    assert result["adapter_luid"] == 0x1234
    assert owner._lib.calls == 2


def test_native_inference_releases_borrowed_frame() -> None:
    frame = _Frame()
    result = _grabber_with_frame(frame).infer_native_frame(_Provider(frame))
    assert result == "depth-result"
    assert frame.released is True


def test_native_inference_releases_frame_after_provider_error() -> None:
    frame = _Frame()
    with pytest.raises(RuntimeError, match="inference failed"):
        _grabber_with_frame(frame).infer_native_frame(
            _Provider(frame, RuntimeError("inference failed"))
        )
    assert frame.released is True
