from __future__ import annotations

# Native destructors intentionally swallow teardown errors.
# ruff: noqa: BLE001, S110
import ctypes
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from wc_rocm import (
    _BGRA8_FORMAT,
    _HIP_EXT_MEM_D3D11_KMT,
    _HIP_EXT_SEM_D3D_FENCE,
    _HIP_MEMCPY_D2D,
    _get_hip,
    _HipExternalMemoryHandleDesc,
    _HipExternalMemoryMipmappedArrayDesc,
    _HipExternalSemaphoreHandleDesc,
    _HipExtSemWaitParams,
)


class D2SFFmpegD3D11Frame(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("slot", ctypes.c_uint32),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("dxgi_format", ctypes.c_uint32),
        ("adapter_luid_low", ctypes.c_uint32),
        ("adapter_luid_high", ctypes.c_int32),
        ("texture_shared_handle", ctypes.c_uint64),
        ("ready_fence_handle", ctypes.c_uint64),
        ("consumed_fence_handle", ctypes.c_uint64),
        ("ready_fence_value", ctypes.c_uint64),
        ("pts_ns", ctypes.c_int64),
        ("duration_ns", ctypes.c_int64),
        ("frame_index", ctypes.c_int64),
        ("color_primaries", ctypes.c_uint32),
        ("color_transfer", ctypes.c_uint32),
        ("color_matrix", ctypes.c_uint32),
        ("full_range", ctypes.c_uint32),
    ]


@dataclass(frozen=True)
class D3D11DecodedFrame:
    slot: int
    width: int
    height: int
    dxgi_format: int
    texture_shared_handle: int
    ready_fence_handle: int
    consumed_fence_handle: int
    ready_fence_value: int
    pts_seconds: float
    duration_seconds: float
    frame_index: int


def _decoder_dll_path() -> Path:
    override = os.environ.get("D2S_FFMPEG_D3D11_DECODER_DLL", "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parent / "native" / "windows" / (
        "d2s_ffmpeg_d3d11_decoder.dll"
    )


def _ffmpeg_bin_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "streaming"
        / "rtmp"
        / "ffmpeg"
        / "bin"
    )


class FFmpegD3D11Decoder:
    def __init__(self, path: str, *, start_seconds: float = 0.0, ring_size: int = 8):
        if os.name != "nt":
            raise RuntimeError("FFmpegD3D11Decoder is Windows-only")
        dll_path = _decoder_dll_path()
        if not dll_path.is_file():
            raise FileNotFoundError(f"D3D11 decoder bridge is unavailable: {dll_path}")
        ffmpeg_bin = _ffmpeg_bin_path()
        if not ffmpeg_bin.is_dir():
            raise FileNotFoundError(f"FFmpeg runtime directory is unavailable: {ffmpeg_bin}")
        self._dll_dir_handle = os.add_dll_directory(str(ffmpeg_bin))
        self._library = ctypes.WinDLL(str(dll_path))
        self._configure_signatures()
        error = ctypes.create_string_buffer(4096)
        self._handle = self._library.d2s_ffmpeg_d3d11_create(
            ctypes.c_wchar_p(str(path)),
            ctypes.c_double(max(0.0, float(start_seconds))),
            ctypes.c_int(max(2, min(16, int(ring_size)))),
            error,
            len(error),
        )
        if not self._handle:
            raise RuntimeError(
                error.value.decode("utf-8", errors="replace").strip()
                or "D3D11 decoder create failed"
            )

    def _configure_signatures(self) -> None:
        library = self._library
        library.d2s_ffmpeg_d3d11_probe.argtypes = [ctypes.c_char_p, ctypes.c_int]
        library.d2s_ffmpeg_d3d11_probe.restype = ctypes.c_int
        library.d2s_ffmpeg_d3d11_create.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        library.d2s_ffmpeg_d3d11_create.restype = ctypes.c_void_p
        library.d2s_ffmpeg_d3d11_next.argtypes = [
            ctypes.c_void_p,
            ctypes.c_double,
            ctypes.POINTER(D2SFFmpegD3D11Frame),
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        library.d2s_ffmpeg_d3d11_next.restype = ctypes.c_int
        library.d2s_ffmpeg_d3d11_seek.argtypes = [
            ctypes.c_void_p,
            ctypes.c_double,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        library.d2s_ffmpeg_d3d11_seek.restype = ctypes.c_int
        library.d2s_ffmpeg_d3d11_position.argtypes = [ctypes.c_void_p]
        library.d2s_ffmpeg_d3d11_position.restype = ctypes.c_double
        library.d2s_ffmpeg_d3d11_destroy.argtypes = [ctypes.c_void_p]
        library.d2s_ffmpeg_d3d11_destroy.restype = None

    def next_frame(self, target_seconds: float) -> D3D11DecodedFrame | None:
        if not self._handle:
            return None
        frame = D2SFFmpegD3D11Frame()
        frame.struct_size = ctypes.sizeof(D2SFFmpegD3D11Frame)
        error = ctypes.create_string_buffer(4096)
        result = self._library.d2s_ffmpeg_d3d11_next(
            self._handle,
            ctypes.c_double(max(0.0, float(target_seconds))),
            ctypes.byref(frame),
            0,
            error,
            len(error),
        )
        if result == 0:
            return None
        if result < 0:
            raise RuntimeError(
                error.value.decode("utf-8", errors="replace").strip()
                or "D3D11 decoder failed"
            )
        return D3D11DecodedFrame(
            slot=int(frame.slot),
            width=int(frame.width),
            height=int(frame.height),
            dxgi_format=int(frame.dxgi_format),
            texture_shared_handle=int(frame.texture_shared_handle),
            ready_fence_handle=int(frame.ready_fence_handle),
            consumed_fence_handle=int(frame.consumed_fence_handle),
            ready_fence_value=int(frame.ready_fence_value),
            pts_seconds=float(frame.pts_ns) / 1_000_000_000.0,
            duration_seconds=float(frame.duration_ns) / 1_000_000_000.0,
            frame_index=int(frame.frame_index),
        )

    def seek(self, seconds: float) -> None:
        if not self._handle:
            return
        error = ctypes.create_string_buffer(4096)
        result = self._library.d2s_ffmpeg_d3d11_seek(
            self._handle,
            ctypes.c_double(max(0.0, float(seconds))),
            error,
            len(error),
        )
        if result < 0:
            raise RuntimeError(
                error.value.decode("utf-8", errors="replace").strip()
                or "D3D11 decoder seek failed"
            )

    def position(self) -> float:
        if not self._handle:
            return 0.0
        return float(self._library.d2s_ffmpeg_d3d11_position(self._handle))

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle:
            self._library.d2s_ffmpeg_d3d11_destroy(handle)
        dll_dir_handle, self._dll_dir_handle = self._dll_dir_handle, None
        if dll_dir_handle is not None:
            dll_dir_handle.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class HipExternalSemaphore:
    def __init__(self, handle: int, device_id: int = 0):
        if not handle:
            raise RuntimeError("D3D11 fence handle is null")
        self._hip = _get_hip()
        self._hip.hipSetDevice(device_id)
        desc = _HipExternalSemaphoreHandleDesc()
        desc.type = _HIP_EXT_SEM_D3D_FENCE
        desc.handle.win32.handle = ctypes.c_void_p(handle)
        desc.handle.win32.name = None
        desc.flags = 0
        self._semaphore = ctypes.c_void_p()
        error = self._hip.hipImportExternalSemaphore(
            ctypes.byref(self._semaphore), ctypes.byref(desc)
        )
        if error != 0:
            raise RuntimeError(f"hipImportExternalSemaphore failed: {error}")
        self._params = _HipExtSemWaitParams()
        if not hasattr(self._hip, "hipSignalExternalSemaphoresAsync"):
            raise RuntimeError("HIP runtime lacks external semaphore signaling")

    def _submit(self, operation, value: int, stream) -> None:
        self._params.params.fence.value = int(value)
        error = operation(
            ctypes.byref(self._semaphore),
            ctypes.byref(self._params),
            1,
            ctypes.c_void_p(stream.cuda_stream),
        )
        if error != 0:
            raise RuntimeError(f"HIP external semaphore operation failed: {error}")

    def wait(self, value: int, stream) -> None:
        self._submit(
            self._hip.hipWaitExternalSemaphoresAsync,
            value,
            stream,
        )

    def signal(self, value: int, stream) -> None:
        self._submit(
            self._hip.hipSignalExternalSemaphoresAsync,
            value,
            stream,
        )

    def close(self) -> None:
        semaphore, self._semaphore = getattr(self, "_semaphore", None), None
        if semaphore:
            self._hip.hipDestroyExternalSemaphore(semaphore)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class HipD3D11BgraTexture:
    def __init__(self, shared_handle: int, width: int, height: int, device_id: int = 0):
        if not shared_handle:
            raise RuntimeError("D3D11 texture shared handle is null")
        self.width = int(width)
        self.height = int(height)
        self._hip = _get_hip()
        self._hip.hipSetDevice(device_id)
        handle_desc = _HipExternalMemoryHandleDesc()
        handle_desc.type = _HIP_EXT_MEM_D3D11_KMT
        handle_desc.handle.win32.handle = ctypes.c_void_p(int(shared_handle))
        handle_desc.handle.win32.name = None
        handle_desc.size = self.width * self.height * 4
        handle_desc.flags = 0
        self._external_memory = ctypes.c_void_p()
        error = self._hip.hipImportExternalMemory(
            ctypes.byref(self._external_memory), ctypes.byref(handle_desc)
        )
        if error != 0:
            raise RuntimeError(f"hipImportExternalMemory failed: {error}")
        mip_desc = _HipExternalMemoryMipmappedArrayDesc()
        mip_desc.formatDesc = _BGRA8_FORMAT
        mip_desc.extent.width = self.width
        mip_desc.extent.height = self.height
        mip_desc.extent.depth = 0
        mip_desc.numLevels = 1
        self._mip_array = ctypes.c_void_p()
        error = self._hip.hipExternalMemoryGetMappedMipmappedArray(
            ctypes.byref(self._mip_array),
            self._external_memory,
            ctypes.byref(mip_desc),
        )
        if error != 0:
            self.close()
            raise RuntimeError(
                f"hipExternalMemoryGetMappedMipmappedArray failed: {error}"
            )
        self._level = ctypes.c_void_p()
        error = self._hip.hipGetMipmappedArrayLevel(
            ctypes.byref(self._level), self._mip_array, 0
        )
        if error != 0:
            self.close()
            raise RuntimeError(f"hipGetMipmappedArrayLevel failed: {error}")

    def copy_to_tensor(self, tensor: torch.Tensor, stream) -> None:
        error = self._hip.hipMemcpy2DFromArrayAsync(
            ctypes.c_void_p(tensor.data_ptr()),
            self.width * 4,
            self._level,
            0,
            0,
            self.width * 4,
            self.height,
            _HIP_MEMCPY_D2D,
            ctypes.c_void_p(stream.cuda_stream),
        )
        if error != 0:
            raise RuntimeError(f"hipMemcpy2DFromArrayAsync failed: {error}")

    def close(self) -> None:
        mip_array, self._mip_array = getattr(self, "_mip_array", None), None
        external_memory, self._external_memory = (
            getattr(self, "_external_memory", None),
            None,
        )
        if mip_array:
            self._hip.hipFreeMipmappedArray(mip_array)
        if external_memory:
            self._hip.hipDestroyExternalMemory(external_memory)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class D3D11VideoFrameImporter:
    """Import decoder textures into a bounded GPU-resident tensor pool.

    The D3D11 decoder owns a ring of shared output textures.  We mirror that
    ring with one preallocated ROCm tensor per slot, which removes the large
    per-frame allocation and keeps the shared texture reusable as soon as the
    HIP copy is queued.  A consumer stream wait provides backpressure when a
    slot wraps around; there is intentionally no CPU-side synchronize on the
    hot path.
    """

    def __init__(self, device_id: int = 0):
        self.device_id = int(device_id)
        self._stream = torch.cuda.Stream(device=f"cuda:{self.device_id}")
        self._textures: dict[int, HipD3D11BgraTexture] = {}
        self._tensors: dict[int, torch.Tensor] = {}
        self._texture_keys: dict[int, tuple[int, int, int]] = {}
        self._ready_semaphore: HipExternalSemaphore | None = None
        self._consumed_semaphore: HipExternalSemaphore | None = None

    def _allocate_tensor(self, height: int, width: int) -> torch.Tensor:
        return torch.empty(
            (height, width, 4),
            dtype=torch.uint8,
            device=f"cuda:{self.device_id}",
        )

    def _tensor_for_frame(self, frame: D3D11DecodedFrame) -> torch.Tensor:
        key = (int(frame.width), int(frame.height), int(frame.texture_shared_handle))
        tensor = self._tensors.get(frame.slot)
        previous_key = self._texture_keys.get(frame.slot)
        if tensor is not None and previous_key != key:
            # A new decoder texture can reuse a numeric slot after a source
            # restart. Never read through a stale tensor shape or handle.
            del self._tensors[frame.slot]
            tensor = None
        if tensor is None:
            tensor = self._allocate_tensor(frame.height, frame.width)
            self._tensors[frame.slot] = tensor
            self._texture_keys[frame.slot] = key
            return tensor

        # The slot is about to be overwritten.  Wait on the consumer stream
        # before the copy stream writes it, rather than stalling the decoder
        # thread with stream.synchronize().  This is the same GPU-side
        # backpressure contract used by the CUDA WGC bridge.
        consumer_stream = torch.cuda.current_stream(self.device_id)
        self._stream.wait_stream(consumer_stream)
        return tensor

    def import_frame(self, frame: D3D11DecodedFrame) -> torch.Tensor:
        if self._ready_semaphore is None:
            self._ready_semaphore = HipExternalSemaphore(
                frame.ready_fence_handle, self.device_id
            )
            self._consumed_semaphore = HipExternalSemaphore(
                frame.consumed_fence_handle, self.device_id
            )
        texture_key = (
            int(frame.width),
            int(frame.height),
            int(frame.texture_shared_handle),
        )
        texture = self._textures.get(frame.slot)
        if texture is None or self._texture_keys.get(frame.slot) != texture_key:
            if texture is not None:
                texture.close()
            texture = HipD3D11BgraTexture(
                frame.texture_shared_handle,
                frame.width,
                frame.height,
                self.device_id,
            )
            self._textures[frame.slot] = texture
        tensor = self._tensor_for_frame(frame)
        assert self._ready_semaphore is not None
        assert self._consumed_semaphore is not None
        with torch.cuda.stream(self._stream):
            self._ready_semaphore.wait(frame.ready_fence_value, self._stream)
            texture.copy_to_tensor(tensor, self._stream)
            self._consumed_semaphore.signal(
                frame.ready_fence_value, self._stream
            )
        # Make the consumer see the copy without blocking the decode thread.
        # The event is enqueued on the consumer stream, so the next GPU op
        # automatically waits for the HIP DMA completion.
        torch.cuda.current_stream(self.device_id).wait_stream(self._stream)
        return tensor

    def close(self) -> None:
        # Teardown is the only place where a CPU synchronization is needed.
        try:
            self._stream.synchronize()
        except Exception:
            pass
        for texture in self._textures.values():
            texture.close()
        self._textures.clear()
        self._tensors.clear()
        self._texture_keys.clear()
        if self._ready_semaphore is not None:
            self._ready_semaphore.close()
            self._ready_semaphore = None
        if self._consumed_semaphore is not None:
            self._consumed_semaphore.close()
            self._consumed_semaphore = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def d3d11_decoder_available() -> bool:
    if os.name != "nt":
        return False
    return _decoder_dll_path().is_file()


__all__ = [
    "D3D11DecodedFrame",
    "D3D11VideoFrameImporter",
    "FFmpegD3D11Decoder",
    "d3d11_decoder_available",
]
