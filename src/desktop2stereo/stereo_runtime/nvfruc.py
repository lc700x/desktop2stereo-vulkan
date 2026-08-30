"""NVIDIA NvFRUC runtime bridge and capability probing.

The bridge keeps CUDA image ownership in the native layer. Python only passes
borrowed CUDA device pointers and pitches; no CPU image readback is performed.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
from pathlib import Path
import platform
import sys
from typing import Any


_ABI_VERSION = 1
_DLL_NAME = "d2s_nvfruc_bridge.dll"
_DLL_DIRECTORY_HANDLES: list[Any] = []


@dataclass(frozen=True)
class NvFrucProbeResult:
    available: bool
    reason: str | None = None
    library_path: str | None = None
    abi_version: int | None = None
    capabilities: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "library_path": self.library_path,
            "abi_version": self.abi_version,
            "capabilities": self.capabilities,
        }


class NvFrucUnavailable(RuntimeError):
    """Raised when the packaged NvFRUC bridge cannot be used."""


def _candidate_libraries() -> list[Path]:
    root = Path(__file__).resolve().parents[3]
    candidates: list[Path] = []
    override = os.environ.get("D2S_NVFRUC_BRIDGE", "").strip()
    if override:
        candidates.append(Path(override))
    candidates.extend(
        (
            Path(__file__).with_name("nvfruc") / _DLL_NAME,
            root / "native" / "nvfruc_bridge" / "build" / "Release" / _DLL_NAME,
            root / "native" / "nvfruc_bridge" / _DLL_NAME,
        )
    )
    return candidates


def _prepare_cuda_dll_search_path() -> None:
    if platform.system() != "Windows" or not hasattr(os, "add_dll_directory"):
        return
    runtime_dirs = [
        Path(sys.prefix) / "Lib" / "site-packages" / "nvidia" / "cuda_runtime" / "bin",
        Path(__file__).resolve().parents[2]
        / "python3"
        / "Lib"
        / "site-packages"
        / "nvidia"
        / "cuda_runtime"
        / "bin",
    ]
    # Add the nvfruc directory so the bridge can find NvOFFRUC.dll
    nvfruc_dir = Path(__file__).resolve().parent / "nvfruc"
    if nvfruc_dir.is_dir():
        runtime_dirs.insert(0, nvfruc_dir)
    cuda_path = os.environ.get("CUDA_PATH", "").strip()
    if cuda_path:
        runtime_dirs.insert(0, Path(cuda_path) / "bin")
    nvfruc_runtime = os.environ.get("D2S_NVFRUC_RUNTIME_DIR", "").strip()
    if nvfruc_runtime:
        runtime_dirs.insert(0, Path(nvfruc_runtime))
    for runtime_dir in runtime_dirs:
        if not runtime_dir.is_dir():
            continue
        try:
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(runtime_dir)))
        except OSError:
            continue


def _configure_api(library: Any) -> None:
    library.d2s_nvfruc_abi_version.argtypes = []
    library.d2s_nvfruc_abi_version.restype = ctypes.c_uint32
    library.d2s_nvfruc_probe.argtypes = []
    library.d2s_nvfruc_probe.restype = ctypes.c_int32
    library.d2s_nvfruc_last_error.argtypes = []
    library.d2s_nvfruc_last_error.restype = ctypes.c_char_p
    library.d2s_nvfruc_create.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_int32,
        ctypes.c_uint64,
    ]
    library.d2s_nvfruc_create.restype = ctypes.c_void_p
    library.d2s_nvfruc_process.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_size_t,
        ctypes.c_double,
        ctypes.c_uint64,
        ctypes.c_size_t,
        ctypes.c_double,
        ctypes.c_uint64,
        ctypes.c_size_t,
        ctypes.c_double,
        ctypes.c_uint64,
    ]
    library.d2s_nvfruc_process.restype = ctypes.c_int32
    library.d2s_nvfruc_reset.argtypes = [ctypes.c_void_p]
    library.d2s_nvfruc_reset.restype = ctypes.c_int32
    library.d2s_nvfruc_destroy.argtypes = [ctypes.c_void_p]
    library.d2s_nvfruc_destroy.restype = None


def _last_error(library: Any) -> str:
    try:
        value = library.d2s_nvfruc_last_error()
    except Exception:
        return "native NvFRUC bridge operation failed"
    if not value:
        return "native NvFRUC bridge operation failed"
    return value.decode("utf-8", errors="replace")


def load_nvfruc_bridge() -> Any:
    if platform.system() != "Windows":
        raise NvFrucUnavailable("NvFRUC bridge is currently supported on Windows only")
    _prepare_cuda_dll_search_path()
    errors: list[str] = []
    for candidate in _candidate_libraries():
        if not candidate.is_file():
            errors.append(f"{candidate}: file not found")
            continue
        try:
            library = ctypes.WinDLL(str(candidate))
            _configure_api(library)
            abi = int(library.d2s_nvfruc_abi_version())
            if abi != _ABI_VERSION:
                raise NvFrucUnavailable(
                    f"ABI mismatch: expected {_ABI_VERSION}, bridge reports {abi}"
                )
            if int(library.d2s_nvfruc_probe()) != 0:
                raise NvFrucUnavailable(_last_error(library))
            return library
        except (OSError, AttributeError, NvFrucUnavailable) as exc:
            errors.append(f"{candidate}: {exc}")
    detail = "; ".join(errors) if errors else "no candidate library"
    raise NvFrucUnavailable(f"{_DLL_NAME} unavailable: {detail}")


def probe_nvfruc() -> NvFrucProbeResult:
    try:
        library = load_nvfruc_bridge()
    except NvFrucUnavailable as exc:
        return NvFrucProbeResult(available=False, reason=str(exc))
    return NvFrucProbeResult(
        available=True,
        library_path=str(getattr(library, "_name", "") or ""),
        abi_version=_ABI_VERSION,
        capabilities=1,
    )


def _cuda_tensor_view(tensor: Any) -> tuple[int, int]:
    if not getattr(tensor, "is_cuda", False):
        raise ValueError("NvFRUC requires CUDA tensors")
    if getattr(tensor, "dtype", None) is None or str(tensor.dtype) != "torch.uint8":
        raise ValueError("NvFRUC bridge resources must be contiguous uint8 CUDA tensors")
    if getattr(tensor, "ndim", 0) != 3 or int(tensor.shape[-1]) != 4:
        raise ValueError("NvFRUC bridge expects an HWC ARGB tensor")
    if not bool(getattr(tensor, "is_contiguous", lambda: False)()):
        raise ValueError("NvFRUC input/output tensors must be contiguous")
    pointer = int(tensor.data_ptr())
    pitch = int(tensor.stride(0) * tensor.element_size())
    return pointer, pitch


def _to_argb_hwc(rgb: Any) -> Any:
    """Convert a CUDA RGB tensor (CHW or HWC) to contiguous uint8 ARGB."""
    import torch

    if not isinstance(rgb, torch.Tensor) or not rgb.is_cuda:
        raise ValueError("NvFRUC requires a CUDA torch.Tensor")
    if rgb.ndim != 3:
        raise ValueError("NvFRUC expects a three-dimensional eye tensor")
    channels_last = int(rgb.shape[-1]) in (3, 4)
    if channels_last:
        hwc = rgb
    elif int(rgb.shape[0]) in (3, 4):
        hwc = rgb.permute(1, 2, 0)
    else:
        raise ValueError("NvFRUC expects an RGB/RGBA CHW or HWC tensor")
    if rgb.dtype != torch.uint8:
        hwc = (hwc.float().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    else:
        hwc = hwc.contiguous()
    if int(hwc.shape[-1]) == 3:
        argb = torch.empty(
            (int(hwc.shape[0]), int(hwc.shape[1]), 4),
            dtype=torch.uint8,
            device=hwc.device,
        )
        argb[..., 0] = 255
        argb[..., 1:] = hwc
        return argb
    return hwc.contiguous()


def _from_argb_hwc(argb: Any, reference: Any) -> Any:
    import torch

    rgb = argb[..., 1:]
    if getattr(reference, "dtype", None) != torch.uint8:
        rgb = rgb.float().div(255.0).to(reference.dtype)
    if getattr(reference, "ndim", 0) == 3 and int(reference.shape[0]) in (3, 4):
        return rgb.permute(2, 0, 1).contiguous()
    return rgb.contiguous()


class NvFrucSession:
    """One native NvFRUC context for one eye and one fixed frame size."""

    def __init__(
        self,
        width: int,
        height: int,
        *,
        device_index: int = 0,
        cuda_stream: int = 0,
        library: Any | None = None,
    ) -> None:
        self._library = library or load_nvfruc_bridge()
        self.width = int(width)
        self.height = int(height)
        self._handle = self._library.d2s_nvfruc_create(
            self.width,
            self.height,
            int(device_index),
            int(cuda_stream),
        )
        if not self._handle:
            raise NvFrucUnavailable(_last_error(self._library))
        self._closed = False

    def process(
        self,
        previous: Any,
        next_frame: Any,
        output: Any,
        *,
        previous_timestamp: float,
        next_timestamp: float,
        output_timestamp: float,
        cuda_stream: int = 0,
    ) -> None:
        if self._closed or not self._handle:
            raise NvFrucUnavailable("NvFRUC session is closed")
        previous_pointer, previous_pitch = _cuda_tensor_view(previous)
        next_pointer, next_pitch = _cuda_tensor_view(next_frame)
        output_pointer, output_pitch = _cuda_tensor_view(output)
        shape = tuple(int(value) for value in output.shape)
        if shape != tuple(int(value) for value in previous.shape):
            raise ValueError("NvFRUC input and output tensor shapes must match")
        status = int(
            self._library.d2s_nvfruc_process(
                self._handle,
                previous_pointer,
                previous_pitch,
                float(previous_timestamp),
                next_pointer,
                next_pitch,
                float(next_timestamp),
                output_pointer,
                output_pitch,
                float(output_timestamp),
                int(cuda_stream),
            )
        )
        if status != 0:
            raise NvFrucUnavailable(_last_error(self._library))

    def reset(self) -> None:
        if self._closed or not self._handle:
            return
        status = int(self._library.d2s_nvfruc_reset(self._handle))
        if status != 0:
            raise NvFrucUnavailable(_last_error(self._library))

    def close(self) -> None:
        if self._closed:
            return
        handle, self._handle = self._handle, None
        self._closed = True
        if handle:
            self._library.d2s_nvfruc_destroy(handle)

    def __enter__(self) -> "NvFrucSession":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class NvFrucStereoGenerator:
    """Generate one synchronized intermediate frame for a stereo eye pair."""

    def __init__(
        self,
        left_eye: Any,
        right_eye: Any,
        *,
        device_index: int = 0,
        cuda_stream: int = 0,
        library: Any | None = None,
    ) -> None:
        left_argb = _to_argb_hwc(left_eye)
        right_argb = _to_argb_hwc(right_eye)
        if tuple(left_argb.shape) != tuple(right_argb.shape):
            raise ValueError("NvFRUC requires equal left/right eye dimensions")
        height, width, _channels = (int(value) for value in left_argb.shape)
        self._library = library or load_nvfruc_bridge()
        self._left = NvFrucSession(
            width,
            height,
            device_index=device_index,
            cuda_stream=cuda_stream,
            library=self._library,
        )
        self._right = NvFrucSession(
            width,
            height,
            device_index=device_index,
            cuda_stream=cuda_stream,
            library=self._library,
        )
        self._cuda_stream = int(cuda_stream)
        self._closed = False
        del left_argb, right_argb

    def interpolate(
        self,
        previous: tuple[Any, Any],
        next_pair: tuple[Any, Any],
        *,
        previous_timestamp: float,
        next_timestamp: float,
        output_timestamp: float,
    ) -> tuple[Any, Any]:
        if self._closed:
            raise NvFrucUnavailable("stereo NvFRUC generator is closed")
        prev_left, prev_right = previous
        next_left, next_right = next_pair
        left_prev = _to_argb_hwc(prev_left)
        left_next = _to_argb_hwc(next_left)
        right_prev = _to_argb_hwc(prev_right)
        right_next = _to_argb_hwc(next_right)
        if (
            tuple(left_prev.shape) != tuple(left_next.shape)
            or tuple(right_prev.shape) != tuple(right_next.shape)
            or tuple(left_prev.shape) != tuple(right_prev.shape)
        ):
            raise ValueError("NvFRUC requires matching stereo frame dimensions")
        self._rebuild_for_shape(int(left_prev.shape[0]), int(left_prev.shape[1]))
        import torch

        left_output = torch.empty_like(left_prev)
        right_output = torch.empty_like(right_prev)
        self._left.process(
            left_prev,
            left_next,
            left_output,
            previous_timestamp=previous_timestamp,
            next_timestamp=next_timestamp,
            output_timestamp=output_timestamp,
            cuda_stream=self._cuda_stream,
        )
        self._right.process(
            right_prev,
            right_next,
            right_output,
            previous_timestamp=previous_timestamp,
            next_timestamp=next_timestamp,
            output_timestamp=output_timestamp,
            cuda_stream=self._cuda_stream,
        )
        return _from_argb_hwc(left_output, prev_left), _from_argb_hwc(
            right_output, prev_right
        )

    def reset(self) -> None:
        if not self._closed:
            self._left.reset()
            self._right.reset()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._left.close()
        self._right.close()

    def __enter__(self) -> "NvFrucStereoGenerator":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
