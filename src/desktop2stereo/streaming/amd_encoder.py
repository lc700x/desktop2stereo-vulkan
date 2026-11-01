"""Optional Windows AMD AMF bridge loader.

The native bridge is intentionally optional. On ROCm it imports a HIP RGBA
device tensor into a shared D3D11 texture and submits that texture to AMF;
unsupported systems fall back to the regular FFmpeg path.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path


def _library_candidates(library_path: str | os.PathLike[str] | None = None) -> list[Path]:
    if library_path is not None:
        return [Path(library_path)]
    root = Path(__file__).resolve().parents[1]
    override = str(os.environ.get("D2S_AMD_ENCODER_DLL", "") or "").strip()
    candidates = [
        Path(__file__).resolve().with_name("amd_encoder") / "d2s_amd_encoder.dll",
        root / "native" / "windows" / "d2s_amd_encoder.dll",
        root / "native" / "amd_encoder" / "d2s_amd_encoder.dll",
    ]
    if override:
        candidates.insert(0, Path(override))
    return candidates


def probe_amd_amf(
    library_path: str | os.PathLike[str] | None = None,
) -> tuple[bool, str]:
    """Return (available, diagnostic) without importing any GPU Python package."""

    if os.name != "nt":
        return False, "AMD AMF bridge is Windows-only"
    for candidate in _library_candidates(library_path):
        if not candidate or not candidate.exists():
            continue
        try:
            bridge = ctypes.WinDLL(str(candidate))
            bridge.d2s_amd_encoder_probe.restype = ctypes.c_int
            bridge.d2s_amd_encoder_last_error.argtypes = [ctypes.c_char_p, ctypes.c_int]
            bridge.d2s_amd_encoder_last_error.restype = ctypes.c_int
            if bridge.d2s_amd_encoder_probe():
                return True, "AMD AMF runtime and DXGI adapter detected"
            buffer = ctypes.create_string_buffer(512)
            bridge.d2s_amd_encoder_last_error(buffer, len(buffer))
            return False, buffer.value.decode("utf-8", errors="replace")
        except OSError as exc:
            return False, f"AMD bridge load failed: {exc}"
    return False, "d2s_amd_encoder.dll is not installed"


class AmdAmfSurfaceEncoder:
    """Thin wrapper for the native D3D11-surface AMF encoder.

    The texture argument must be an existing ``ID3D11Texture2D`` pointer. The
    wrapper deliberately does not accept NumPy or CPU buffers: callers that
    cannot provide a GPU surface must use the normal FFmpeg fallback.
    """

    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        bitrate: int,
        *,
        hevc: bool = False,
        device_index: int = 0,
        library_path: str | os.PathLike[str] | None = None,
    ):
        available, detail = probe_amd_amf(library_path)
        if not available:
            raise RuntimeError(detail)
        dll = next(path for path in _library_candidates(library_path) if path and path.exists())
        self._dll = ctypes.WinDLL(str(dll))
        self._packet_capacity = 2 * 1024 * 1024
        self._packet_buffer = ctypes.create_string_buffer(self._packet_capacity)
        self._abi_version = 1
        abi_probe = getattr(self._dll, "d2s_amd_encoder_abi_version", None)
        if abi_probe is not None:
            abi_probe.restype = ctypes.c_int
            self._abi_version = int(abi_probe())
        self._dll.d2s_amd_encoder_create.restype = ctypes.c_void_p
        self._dll.d2s_amd_encoder_create.argtypes = [ctypes.c_int] * 4 + [ctypes.c_int]
        create_for_device = getattr(self._dll, "d2s_amd_encoder_create_for_hip_device", None)
        if create_for_device is not None:
            create_for_device.restype = ctypes.c_void_p
            create_for_device.argtypes = [ctypes.c_int] * 4 + [ctypes.c_int, ctypes.c_int]
        self._dll.d2s_amd_encoder_submit_texture.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._dll.d2s_amd_encoder_submit_texture.restype = ctypes.c_int
        submit_hip_argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        if self._abi_version >= 2:
            submit_hip_argtypes.append(ctypes.c_int)
        self._dll.d2s_amd_encoder_submit_hip_rgba.argtypes = submit_hip_argtypes
        self._dll.d2s_amd_encoder_submit_hip_rgba.restype = ctypes.c_int
        self._dll.d2s_amd_encoder_read_packet.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        self._dll.d2s_amd_encoder_read_packet.restype = ctypes.c_int
        self._dll.d2s_amd_encoder_destroy.argtypes = [ctypes.c_void_p]
        if self._abi_version >= 2 and create_for_device is not None:
            self._handle = create_for_device(
                width, height, fps, bitrate, int(hevc), max(0, int(device_index))
            )
        else:
            self._handle = self._dll.d2s_amd_encoder_create(
                width, height, fps, bitrate, int(hevc)
            )
        if not self._handle:
            raise RuntimeError("AMF surface encoder initialization failed")

    def submit_d3d11_texture(self, texture_pointer: int) -> bool:
        result = self._dll.d2s_amd_encoder_submit_texture(
            self._handle, ctypes.c_void_p(texture_pointer)
        )
        if result < 0:
            raise RuntimeError("AMF rejected the D3D11 texture")
        return result > 0

    def submit_hip_rgba(
        self,
        device_pointer: int,
        pitch_bytes: int,
        stream: int,
        *,
        force_idr: bool = False,
    ) -> bool:
        if force_idr and self._abi_version < 2:
            raise RuntimeError(
                "AMD AMF bridge is too old for Moonlight IDR requests; rebuild d2s_amd_encoder.dll"
            )
        args = (
            self._handle,
            ctypes.c_void_p(int(device_pointer)),
            int(pitch_bytes),
            ctypes.c_void_p(int(stream)),
        )
        if self._abi_version >= 2:
            args = (*args, int(force_idr))
        result = self._dll.d2s_amd_encoder_submit_hip_rgba(*args)
        if result == 0:
            return False
        if result < 0:
            raise RuntimeError("AMF rejected the HIP RGBA surface")
        return True

    def read_packet(self, capacity: int = 2 * 1024 * 1024) -> bytes | None:
        capacity = max(1, int(capacity))
        for _ in range(2):
            if capacity != self._packet_capacity:
                self._packet_capacity = capacity
                self._packet_buffer = ctypes.create_string_buffer(capacity)
            size = self._dll.d2s_amd_encoder_read_packet(
                self._handle, self._packet_buffer, capacity
            )
            if size > 0:
                return ctypes.string_at(self._packet_buffer, size)
            if size == 0:
                return None
            required = -size
            if required <= capacity or required > 32 * 1024 * 1024:
                raise RuntimeError(
                    f"AMF returned an invalid access-unit size: {required}"
                )
            capacity = required
        return None

    def close(self) -> None:
        if self._handle:
            self._dll.d2s_amd_encoder_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.close()
