"""NVIDIA CUDA Runtime interop for exportable Vulkan image and buffer slots."""

from __future__ import annotations

import ctypes
import glob
import os
from pathlib import Path
from typing import Any

from runtime_paths import python_site_packages

from .vulkan_interop import VulkanInteropCapabilities, VulkanInteropMode
from .vulkan_resources import (
    VulkanExportableImage,
    VulkanExportableSemaphore,
    VulkanImageResource,
)


class CudaVulkanInteropError(RuntimeError):
    pass


class _Win32Handle(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_void_p), ("name", ctypes.c_void_p)]


class _ExternalHandleUnion(ctypes.Union):
    _fields_ = [("fd", ctypes.c_int), ("win32", _Win32Handle)]


class _ExternalMemoryHandleDesc(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("handle", _ExternalHandleUnion),
        ("size", ctypes.c_uint64),
        ("flags", ctypes.c_uint),
    ]


class _ChannelFormatDesc(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
        ("z", ctypes.c_int),
        ("w", ctypes.c_int),
        ("f", ctypes.c_int),
    ]


class _Extent(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_size_t),
        ("height", ctypes.c_size_t),
        ("depth", ctypes.c_size_t),
    ]


class _ExternalMipmappedArrayDesc(ctypes.Structure):
    _fields_ = [
        ("offset", ctypes.c_uint64),
        ("format_desc", _ChannelFormatDesc),
        ("extent", _Extent),
        ("flags", ctypes.c_uint),
        ("num_levels", ctypes.c_uint),
    ]


class _ExternalMemoryBufferDesc(ctypes.Structure):
    _fields_ = [
        ("offset", ctypes.c_size_t),
        ("size", ctypes.c_size_t),
        ("flags", ctypes.c_uint),
    ]


class _ExternalSemaphoreHandleDesc(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("handle", _ExternalHandleUnion),
        ("flags", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]


class _SemaphoreNvSciSync(ctypes.Union):
    _fields_ = [
        ("fence", ctypes.c_void_p),
        ("reserved", ctypes.c_uint64),
    ]


class _SemaphoreSignalParams(ctypes.Structure):
    _fields_ = [
        ("fence_value", ctypes.c_uint64),
        ("nv_sci_sync", _SemaphoreNvSciSync),
        ("keyed_mutex_key", ctypes.c_uint64),
        ("reserved", ctypes.c_uint * 12),
    ]


class _ExternalSemaphoreSignalParams(ctypes.Structure):
    _fields_ = [
        ("params", _SemaphoreSignalParams),
        ("flags", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]


class _ExternalSemaphoreWaitParams(ctypes.Structure):
    _fields_ = [
        ("params", _SemaphoreSignalParams),
        ("flags", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]


class _CudaSlot:
    def __init__(self, target: VulkanExportableImage, external_memory: ctypes.c_void_p, array: ctypes.c_void_p):
        self.target = target
        self.external_memory = external_memory
        self.array = array


class _CudaBufferSlot:
    def __init__(self, target: Any, external_memory: ctypes.c_void_p, pointer: ctypes.c_void_p):
        self.target = target
        self.external_memory = external_memory
        self.pointer = pointer


class _CudaSemaphore:
    def __init__(self, target: VulkanExportableSemaphore, external: ctypes.c_void_p):
        self.target = target
        self.external = external


class CudaVulkanImageImporter:
    """Import Vulkan-exported memory once and copy CUDA RGBA tensors into it."""

    _CUDA_OPAQUE_FD = 1
    _CUDA_OPAQUE_WIN32 = 2
    _CUDA_TIMELINE_SEMAPHORE_FD = 9
    _CUDA_TIMELINE_SEMAPHORE_WIN32 = 10
    _CUDA_D3D12_FENCE = 4
    _CUDA_D3D12_HEAP = 4
    _CUDA_ARRAY_COLOR_ATTACHMENT = 0x20
    _CUDA_MEMCPY_DEVICE_TO_DEVICE = 3

    def __init__(self, *, cudart_path: str | None = None) -> None:
        self._cudart = self._load_cudart(cudart_path)
        self._slots: dict[int, _CudaSlot] = {}
        self._buffer_slots: dict[int, _CudaBufferSlot] = {}
        self._semaphores: dict[int, _CudaSemaphore] = {}
        self._nv12_slots: dict[int, tuple[list[_CudaSlot], list[ctypes.c_void_p]]] = {}
        self._rgba_slots: dict[int, tuple[_CudaSlot, ctypes.c_void_p]] = {}

    def write_ffmpeg_rgba_frame(
        self, tensor: Any, frame: Any, *, stream: int | None = None
    ) -> int:
        """Write one CUDA RGBA tensor into an FFmpeg-owned Vulkan image.

        The native bridge exports a single ``R8G8B8A8`` image and a timeline
        semaphore for each reusable slot.  CUDA imports both objects once,
        waits for FFmpeg's previous value, performs a device-to-device copy,
        and signals the next value.  No RGB24 staging buffer or CPU readback
        is involved.
        """
        self._validate_ffmpeg_rgba_frame(tensor, frame)
        if stream is None:
            import torch

            stream = int(torch.cuda.current_stream(device=tensor.device).cuda_stream)
        key = int(frame.slot_id)
        if key not in self._rgba_slots:
            memory = ctypes.c_void_p()
            semaphore = ctypes.c_void_p()
            try:
                memory_desc = _ExternalMemoryHandleDesc(
                    type=self._CUDA_OPAQUE_WIN32 if os.name == "nt" else self._CUDA_OPAQUE_FD,
                    size=int(frame.memory_size[0]),
                    flags=0,
                )
                if os.name == "nt":
                    memory_desc.handle.win32.handle = ctypes.c_void_p(
                        int(frame.external_memory_handle[0])
                    )
                else:
                    memory_desc.handle.fd = int(frame.external_memory_handle[0])
                self._check(
                    self._cudart.cudaImportExternalMemory(
                        ctypes.byref(memory), ctypes.byref(memory_desc)
                    ),
                    "cudaImportExternalMemory(FFmpeg RGBA)",
                )
                mapped_desc = _ExternalMipmappedArrayDesc(
                    offset=int(frame.memory_offset[0]),
                    format_desc=_ChannelFormatDesc(8, 8, 8, 8, 0),
                    extent=_Extent(int(frame.width), int(frame.height), 0),
                    flags=self._CUDA_ARRAY_COLOR_ATTACHMENT,
                    num_levels=1,
                )
                mipmap = ctypes.c_void_p()
                self._check(
                    self._cudart.cudaExternalMemoryGetMappedMipmappedArray(
                        ctypes.byref(mipmap), memory, ctypes.byref(mapped_desc)
                    ),
                    "cudaExternalMemoryGetMappedMipmappedArray(FFmpeg RGBA)",
                )
                array = ctypes.c_void_p()
                self._check(
                    self._cudart.cudaGetMipmappedArrayLevel(
                        ctypes.byref(array), mipmap, 0
                    ),
                    "cudaGetMipmappedArrayLevel(FFmpeg RGBA)",
                )
                semaphore_desc = _ExternalSemaphoreHandleDesc(
                    type=self._CUDA_TIMELINE_SEMAPHORE_WIN32
                    if os.name == "nt"
                    else self._CUDA_TIMELINE_SEMAPHORE_FD,
                    flags=0,
                )
                if os.name == "nt":
                    semaphore_desc.handle.win32.handle = ctypes.c_void_p(
                        int(frame.external_semaphore_handle[0])
                    )
                else:
                    semaphore_desc.handle.fd = int(frame.external_semaphore_handle[0])
                self._check(
                    self._cudart.cudaImportExternalSemaphore(
                        ctypes.byref(semaphore), ctypes.byref(semaphore_desc)
                    ),
                    "cudaImportExternalSemaphore(FFmpeg RGBA)",
                )
                self._rgba_slots[key] = (
                    _CudaSlot(None, memory, array),
                    semaphore,
                )
            except Exception:
                if semaphore:
                    self._cudart.cudaDestroyExternalSemaphore(semaphore)
                if memory:
                    self._cudart.cudaDestroyExternalMemory(memory)
                raise
            finally:
                self._close_ffmpeg_frame_handles(frame)
        else:
            self._close_ffmpeg_frame_handles(frame)

        slot, semaphore = self._rgba_slots[key]
        wait_value = int(frame.semaphore_value[0])
        ready_value = wait_value + 1
        wait = _ExternalSemaphoreWaitParams(
            params=_SemaphoreSignalParams(fence_value=wait_value, keyed_mutex_key=0),
            flags=0,
        )
        self._check(
            self._cudart.cudaWaitExternalSemaphoresAsync(
                (ctypes.c_void_p * 1)(semaphore),
                ctypes.byref(wait),
                1,
                ctypes.c_void_p(int(stream)),
            ),
            "cudaWaitExternalSemaphoresAsync(FFmpeg RGBA)",
        )
        row_bytes = int(frame.width) * 4
        self._check(
            self._cudart.cudaMemcpy2DToArrayAsync(
                slot.array,
                0,
                0,
                ctypes.c_void_p(int(tensor.data_ptr())),
                row_bytes,
                row_bytes,
                int(frame.height),
                self._CUDA_MEMCPY_DEVICE_TO_DEVICE,
                ctypes.c_void_p(int(stream)),
            ),
            "cudaMemcpy2DToArrayAsync(FFmpeg RGBA)",
        )
        signal = _ExternalSemaphoreSignalParams(
            params=_SemaphoreSignalParams(
                fence_value=ready_value, keyed_mutex_key=0
            ),
            flags=0,
        )
        self._check(
            self._cudart.cudaSignalExternalSemaphoresAsync(
                (ctypes.c_void_p * 1)(semaphore),
                ctypes.byref(signal),
                1,
                ctypes.c_void_p(int(stream)),
            ),
            "cudaSignalExternalSemaphoresAsync(FFmpeg RGBA)",
        )
        return ready_value

    def write_ffmpeg_nv12_frame(self, y: Any, uv: Any, frame: Any, *, stream: int | None = None) -> int:
        """Write CUDA NV12 planes into one FFmpeg-exported Vulkan frame.

        ``frame`` is ``streaming.vulkan_bridge.VulkanVideoFrame``.  The
        producer waits on FFmpeg's current timeline value, copies only on GPU,
        then signals the next value returned to native ``submit_frame``.
        """
        self._validate_ffmpeg_nv12_planes(y, uv, frame)
        if stream is None:
            import torch
            stream = int(torch.cuda.current_stream(device=y.device).cuda_stream)
        key = int(frame.slot_id)
        if key not in self._nv12_slots:
            slots, semaphores = [], []
            try:
                for index, (_tensor, channels, width, height) in enumerate(((y, 1, frame.width, frame.height), (uv, 2, frame.width // 2, frame.height // 2))):
                    desc = _ExternalMemoryHandleDesc(type=self._CUDA_OPAQUE_WIN32 if os.name == "nt" else self._CUDA_OPAQUE_FD, size=int(frame.memory_size[index]), flags=0)
                    if os.name == "nt": desc.handle.win32.handle = ctypes.c_void_p(int(frame.external_memory_handle[index]))
                    else: desc.handle.fd = int(frame.external_memory_handle[index])
                    memory = ctypes.c_void_p(); self._check(self._cudart.cudaImportExternalMemory(ctypes.byref(memory), ctypes.byref(desc)), "cudaImportExternalMemory(FFmpeg NV12)")
                    mapped = _ExternalMipmappedArrayDesc(offset=int(frame.memory_offset[index]), format_desc=_ChannelFormatDesc(8, 8 if channels == 2 else 0, 0, 0, 0), extent=_Extent(width, height, 0), flags=self._CUDA_ARRAY_COLOR_ATTACHMENT, num_levels=1)
                    mip = ctypes.c_void_p(); self._check(self._cudart.cudaExternalMemoryGetMappedMipmappedArray(ctypes.byref(mip), memory, ctypes.byref(mapped)), "cudaExternalMemoryGetMappedMipmappedArray(FFmpeg NV12)")
                    array = ctypes.c_void_p(); self._check(self._cudart.cudaGetMipmappedArrayLevel(ctypes.byref(array), mip, 0), "cudaGetMipmappedArrayLevel(FFmpeg NV12)")
                    slots.append(_CudaSlot(None, memory, array))
                    sem_desc = _ExternalSemaphoreHandleDesc(type=self._CUDA_TIMELINE_SEMAPHORE_WIN32 if os.name == "nt" else self._CUDA_TIMELINE_SEMAPHORE_FD, flags=0)
                    if os.name == "nt": sem_desc.handle.win32.handle = ctypes.c_void_p(int(frame.external_semaphore_handle[index]))
                    else: sem_desc.handle.fd = int(frame.external_semaphore_handle[index])
                    sem = ctypes.c_void_p(); self._check(self._cudart.cudaImportExternalSemaphore(ctypes.byref(sem), ctypes.byref(sem_desc)), "cudaImportExternalSemaphore(FFmpeg NV12)"); semaphores.append(sem)
            except Exception:
                for semaphore in semaphores:
                    self._cudart.cudaDestroyExternalSemaphore(semaphore)
                for slot in slots:
                    self._cudart.cudaDestroyExternalMemory(slot.external_memory)
                raise
            finally:
                # FFmpeg duplicates these OS handles on every acquire. CUDA keeps
                # the imported object; the raw exported handles are no longer used.
                self._close_ffmpeg_frame_handles(frame)
            self._nv12_slots[key] = (slots, semaphores)
        else:
            self._close_ffmpeg_frame_handles(frame)
        slots, semaphores = self._nv12_slots[key]
        ready_value = max(int(value) for value in frame.semaphore_value) + 1
        for index, sem in enumerate(semaphores):
            wait = _ExternalSemaphoreWaitParams(params=_SemaphoreSignalParams(fence_value=int(frame.semaphore_value[index]), keyed_mutex_key=0), flags=0)
            self._check(self._cudart.cudaWaitExternalSemaphoresAsync((ctypes.c_void_p * 1)(sem), ctypes.byref(wait), 1, ctypes.c_void_p(int(stream))), "cudaWaitExternalSemaphoresAsync(FFmpeg NV12)")
            tensor = y if index == 0 else uv; row = int(frame.width)
            self._check(self._cudart.cudaMemcpy2DToArrayAsync(slots[index].array, 0, 0, ctypes.c_void_p(int(tensor.data_ptr())), row, row, int(frame.height if index == 0 else frame.height // 2), self._CUDA_MEMCPY_DEVICE_TO_DEVICE, ctypes.c_void_p(int(stream))), "cudaMemcpy2DToArrayAsync(FFmpeg NV12)")
            signal = _ExternalSemaphoreSignalParams(params=_SemaphoreSignalParams(fence_value=ready_value, keyed_mutex_key=0), flags=0)
            self._check(self._cudart.cudaSignalExternalSemaphoresAsync((ctypes.c_void_p * 1)(sem), ctypes.byref(signal), 1, ctypes.c_void_p(int(stream))), "cudaSignalExternalSemaphoresAsync(FFmpeg NV12)")
        return ready_value

    @staticmethod
    def _close_external_handle(handle: int) -> None:
        if not int(handle):
            return
        if os.name == "nt":
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
            kernel32.CloseHandle.restype = ctypes.c_int
            kernel32.CloseHandle(ctypes.c_void_p(int(handle)))
        else:
            os.close(int(handle))

    @classmethod
    def _close_ffmpeg_frame_handles(cls, frame: Any) -> None:
        for index in range(int(getattr(frame, "plane_count", 0))):
            cls._close_external_handle(int(frame.external_memory_handle[index]))
            cls._close_external_handle(int(frame.external_semaphore_handle[index]))

    @staticmethod
    def _validate_ffmpeg_rgba_frame(tensor: Any, frame: Any) -> None:
        if int(getattr(frame, "plane_count", 0)) != 1:
            raise CudaVulkanInteropError(
                "FFmpeg Vulkan RGBA frame must contain one plane"
            )
        if int(getattr(frame, "format", [0])[0]) not in (37, 43):
            raise CudaVulkanInteropError(
                "FFmpeg Vulkan RGBA frame must use R8G8B8A8_UNORM or SRGB"
            )
        if getattr(tensor, "device", None) is None or str(tensor.device.type) != "cuda":
            raise CudaVulkanInteropError("FFmpeg RGBA frame requires a CUDA tensor")
        if str(getattr(tensor, "dtype", "")) != "torch.uint8":
            raise CudaVulkanInteropError("FFmpeg RGBA frame requires torch.uint8")
        shape = (int(frame.height), int(frame.width), 4)
        if tuple(getattr(tensor, "shape", ())) != shape or not bool(tensor.is_contiguous()):
            raise CudaVulkanInteropError(
                f"FFmpeg RGBA frame requires contiguous tensor {shape!r}"
            )
        if int(frame.memory_size[0]) <= 0:
            raise CudaVulkanInteropError("FFmpeg Vulkan RGBA frame has invalid memory size")

    @staticmethod
    def _validate_ffmpeg_nv12_planes(y: Any, uv: Any, frame: Any) -> None:
        if int(getattr(frame, "plane_count", 0)) != 2:
            raise CudaVulkanInteropError("FFmpeg Vulkan frame must contain two NV12 planes")
        for name, tensor, shape in (
            ("Y", y, (int(frame.height), int(frame.width), 1)),
            ("UV", uv, (int(frame.height) // 2, int(frame.width) // 2, 2)),
        ):
            if getattr(tensor, "device", None) is None or str(tensor.device.type) != "cuda":
                raise CudaVulkanInteropError(f"FFmpeg NV12 {name} plane must be a CUDA tensor")
            if str(getattr(tensor, "dtype", "")) != "torch.uint8":
                raise CudaVulkanInteropError(f"FFmpeg NV12 {name} plane must be torch.uint8")
            if tuple(getattr(tensor, "shape", ())) != shape or not bool(tensor.is_contiguous()):
                raise CudaVulkanInteropError(f"FFmpeg NV12 {name} plane must be contiguous {shape!r}")
        if not all(int(frame.memory_size[index]) > 0 for index in range(2)):
            raise CudaVulkanInteropError("FFmpeg Vulkan frame has invalid external memory sizes")

    @property
    def capabilities(self) -> VulkanInteropCapabilities:
        return VulkanInteropCapabilities(
            producer="nvidia-cuda-runtime",
            mode=VulkanInteropMode.GPU_COPY,
            external_memory=True,
            external_semaphore=all(
                hasattr(self._cudart, name)
                for name in (
                    "cudaImportExternalSemaphore",
                    "cudaSignalExternalSemaphoresAsync",
                    "cudaWaitExternalSemaphoresAsync",
                    "cudaDestroyExternalSemaphore",
                )
            ),
            zero_copy=False,
        )

    @staticmethod
    def _load_cudart(cudart_path: str | None):
        candidates = []
        if cudart_path:
            candidates.append(str(cudart_path))
        env_path = os.environ.get("D2S_CUDART_PATH")
        if env_path:
            candidates.append(env_path)
        site_packages = python_site_packages()
        candidates.extend(
            glob.glob(str(site_packages / "nvidia" / "cuda_runtime" / "bin" / "cudart64_*.dll"))
        )
        candidates.extend(glob.glob(str(site_packages / "torch" / "lib" / "cudart64_*.dll")))
        candidates.append("cudart64_12.dll")
        for candidate in candidates:
            try:
                lib = ctypes.WinDLL(candidate) if os.name == "nt" else ctypes.CDLL(candidate)
                return CudaVulkanImageImporter._configure_functions(lib)
            except (OSError, AttributeError):
                continue
        raise CudaVulkanInteropError("CUDA Runtime library with external-memory API was not found")

    @staticmethod
    def _configure_functions(lib):
        lib.cudaImportExternalMemory.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(_ExternalMemoryHandleDesc)]
        lib.cudaImportExternalMemory.restype = ctypes.c_int
        lib.cudaExternalMemoryGetMappedMipmappedArray.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.POINTER(_ExternalMipmappedArrayDesc)]
        lib.cudaExternalMemoryGetMappedMipmappedArray.restype = ctypes.c_int
        lib.cudaGetMipmappedArrayLevel.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint]
        lib.cudaGetMipmappedArrayLevel.restype = ctypes.c_int
        lib.cudaExternalMemoryGetMappedBuffer.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.POINTER(_ExternalMemoryBufferDesc),
        ]
        lib.cudaExternalMemoryGetMappedBuffer.restype = ctypes.c_int
        lib.cudaMemcpy2DToArrayAsync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
        lib.cudaMemcpy2DToArrayAsync.restype = ctypes.c_int
        lib.cudaMemcpy2DAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.cudaMemcpy2DAsync.restype = ctypes.c_int
        lib.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.cudaMemcpyAsync.restype = ctypes.c_int
        lib.cudaDestroyExternalMemory.argtypes = [ctypes.c_void_p]
        lib.cudaDestroyExternalMemory.restype = ctypes.c_int
        import_external = getattr(lib, "cudaImportExternalSemaphore", None)
        signal_external = getattr(lib, "cudaSignalExternalSemaphoresAsync", None)
        wait_external = getattr(lib, "cudaWaitExternalSemaphoresAsync", None)
        destroy_external = getattr(lib, "cudaDestroyExternalSemaphore", None)
        if import_external is not None:
            import_external.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(_ExternalSemaphoreHandleDesc),
            ]
            import_external.restype = ctypes.c_int
        if signal_external is not None:
            signal_external.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(_ExternalSemaphoreSignalParams),
                ctypes.c_uint,
                ctypes.c_void_p,
            ]
            signal_external.restype = ctypes.c_int
        if wait_external is not None:
            wait_external.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(_ExternalSemaphoreWaitParams),
                ctypes.c_uint,
                ctypes.c_void_p,
            ]
            wait_external.restype = ctypes.c_int
        if destroy_external is not None:
            destroy_external.argtypes = [ctypes.c_void_p]
            destroy_external.restype = ctypes.c_int
        lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        lib.cudaStreamSynchronize.restype = ctypes.c_int
        return lib

    @staticmethod
    def _check(result: int, operation: str) -> None:
        if int(result) != 0:
            raise CudaVulkanInteropError(f"{operation} failed with CUDA error {int(result)}")

    def register_slot(self, target: VulkanExportableImage) -> VulkanImageResource:
        key = id(target)
        if key in self._slots:
            return target.resource
        if target.resource is None:
            raise CudaVulkanInteropError("exportable target has no Vulkan resource")
        prepare = getattr(target.context, "prepare_external_image_for_cuda", None)
        if not callable(prepare):
            raise CudaVulkanInteropError(
                "Vulkan context cannot establish an external CUDA image layout"
            )
        prepare(target.resource)
        handle = target.export_handle
        desc = _ExternalMemoryHandleDesc(
            type=self._CUDA_OPAQUE_WIN32 if os.name == "nt" else self._CUDA_OPAQUE_FD,
            size=int(target.allocation_size),
            flags=0,
        )
        if os.name == "nt":
            desc.handle.win32.handle = ctypes.c_void_p(int(handle))
        else:
            desc.handle.fd = int(handle)
        external_memory = ctypes.c_void_p()
        self._check(
            self._cudart.cudaImportExternalMemory(ctypes.byref(external_memory), ctypes.byref(desc)),
            "cudaImportExternalMemory",
        )
        # CUDA receives raw RGBA bytes. Vulkan's sRGB/UNORM interpretation is
        # carried by the image format and does not change this channel layout.
        mapped_desc = _ExternalMipmappedArrayDesc(
            offset=0,
            format_desc=_ChannelFormatDesc(8, 8, 8, 8, 0),
            extent=_Extent(target.width, target.height, 0),
            flags=self._CUDA_ARRAY_COLOR_ATTACHMENT,
            num_levels=1,
        )
        mipmap = ctypes.c_void_p()
        try:
            self._check(
                self._cudart.cudaExternalMemoryGetMappedMipmappedArray(
                    ctypes.byref(mipmap), external_memory, ctypes.byref(mapped_desc)
                ),
                "cudaExternalMemoryGetMappedMipmappedArray",
            )
            array = ctypes.c_void_p()
            self._check(
                self._cudart.cudaGetMipmappedArrayLevel(ctypes.byref(array), mipmap, 0),
                "cudaGetMipmappedArrayLevel",
            )
        except Exception:
            self._cudart.cudaDestroyExternalMemory(external_memory)
            raise
        self._slots[key] = _CudaSlot(target, external_memory, array)
        target.close_export_handle()
        return target.resource

    def copy_tensor(self, tensor: Any, target: VulkanExportableImage, *, stream: int | None = None) -> VulkanImageResource:
        resource = self.register_slot(target)
        if getattr(tensor, "device", None) is None or str(tensor.device.type) != "cuda":
            raise CudaVulkanInteropError("CUDA Vulkan copy requires a CUDA tensor")
        if getattr(tensor, "dtype", None) is None or str(tensor.dtype) != "torch.uint8":
            raise CudaVulkanInteropError("CUDA Vulkan copy requires torch.uint8 RGBA tensor")
        if getattr(tensor, "ndim", 0) != 3 or tuple(tensor.shape) != (target.height, target.width, 4):
            raise CudaVulkanInteropError("CUDA Vulkan copy requires HxWx4 tensor matching target")
        if not bool(tensor.is_contiguous()):
            raise CudaVulkanInteropError("CUDA Vulkan copy requires a contiguous tensor")
        if stream is None:
            import torch

            stream = int(torch.cuda.current_stream(device=tensor.device).cuda_stream)
        slot = self._slots[id(target)]
        self._check(
            self._cudart.cudaMemcpy2DToArrayAsync(
                slot.array,
                0,
                0,
                ctypes.c_void_p(int(tensor.data_ptr())),
                target.width * 4,
                target.width * 4,
                target.height,
                self._CUDA_MEMCPY_DEVICE_TO_DEVICE,
                ctypes.c_void_p(int(stream)),
            ),
            "cudaMemcpy2DToArrayAsync",
        )
        return resource

    def register_buffer(self, target: Any) -> None:
        """Import an exportable Vulkan storage buffer into CUDA once."""
        key = id(target)
        if key in self._buffer_slots:
            return
        handle = getattr(target, "export_handle", None)
        allocation_size = int(getattr(target, "allocation_size", 0))
        if handle is None or allocation_size < 1:
            raise CudaVulkanInteropError("exportable Vulkan buffer has no memory handle")
        desc = _ExternalMemoryHandleDesc(
            type=int(
                getattr(
                    target,
                    "cuda_external_memory_handle_type",
                    self._CUDA_OPAQUE_WIN32 if os.name == "nt" else self._CUDA_OPAQUE_FD,
                )
            ),
            size=allocation_size,
            flags=0,
        )
        if os.name == "nt":
            desc.handle.win32.handle = ctypes.c_void_p(int(handle))
        else:
            desc.handle.fd = int(handle)
        external_memory = ctypes.c_void_p()
        self._check(
            self._cudart.cudaImportExternalMemory(
                ctypes.byref(external_memory), ctypes.byref(desc)
            ),
            "cudaImportExternalMemory(buffer)",
        )
        mapped_desc = _ExternalMemoryBufferDesc(
            offset=0,
            size=allocation_size,
            flags=0,
        )
        pointer = ctypes.c_void_p()
        try:
            self._check(
                self._cudart.cudaExternalMemoryGetMappedBuffer(
                    ctypes.byref(pointer), external_memory, ctypes.byref(mapped_desc)
                ),
                "cudaExternalMemoryGetMappedBuffer",
            )
        except Exception:
            self._cudart.cudaDestroyExternalMemory(external_memory)
            raise
        target.close_export_handle()
        self._buffer_slots[key] = _CudaBufferSlot(target, external_memory, pointer)

    def copy_tensor_to_buffer(
        self, tensor: Any, target: Any, *, stream: int | None = None
    ) -> None:
        """Copy a contiguous CUDA tensor directly into an imported buffer."""
        self.register_buffer(target)
        if getattr(tensor, "device", None) is None or str(tensor.device.type) != "cuda":
            raise CudaVulkanInteropError("CUDA Vulkan buffer copy requires a CUDA tensor")
        if str(getattr(tensor, "dtype", "")) not in {
            "torch.float32",
            "torch.float16",
            "torch.uint8",
        }:
            raise CudaVulkanInteropError(
                "CUDA Vulkan buffer copy requires torch.float32, torch.float16, or torch.uint8"
            )
        if not bool(tensor.is_contiguous()):
            raise CudaVulkanInteropError("CUDA Vulkan buffer copy requires a contiguous tensor")
        byte_count = int(tensor.numel()) * int(tensor.element_size())
        if byte_count > int(getattr(target, "size", 0)):
            raise CudaVulkanInteropError("CUDA tensor does not fit in the Vulkan buffer")
        if stream is None:
            import torch

            stream = int(torch.cuda.current_stream(device=tensor.device).cuda_stream)
        slot = self._buffer_slots[id(target)]
        self._check(
            self._cudart.cudaMemcpyAsync(
                slot.pointer,
                ctypes.c_void_p(int(tensor.data_ptr())),
                byte_count,
                self._CUDA_MEMCPY_DEVICE_TO_DEVICE,
                ctypes.c_void_p(int(stream)),
            ),
            "cudaMemcpyAsync(buffer)",
        )

    def copy_tensor_to_pitched_buffer(
        self,
        tensor: Any,
        target: Any,
        *,
        row_pitch: int,
        stream: int | None = None,
    ) -> None:
        """Copy an HxWx4 CUDA byte tensor to a pitched D3D12 shared buffer."""
        self.register_buffer(target)
        if getattr(tensor, "device", None) is None or str(tensor.device.type) != "cuda":
            raise CudaVulkanInteropError("CUDA D3D12 buffer copy requires a CUDA tensor")
        if str(getattr(tensor, "dtype", "")) != "torch.uint8":
            raise CudaVulkanInteropError("CUDA D3D12 buffer copy requires a torch.uint8 RGBA tensor")
        shape = tuple(int(value) for value in getattr(tensor, "shape", ()))
        if len(shape) != 3 or shape[-1] != 4:
            raise CudaVulkanInteropError("CUDA D3D12 buffer copy requires an HxWx4 tensor")
        height, width, _channels = shape
        source_pitch = width * 4
        if int(row_pitch) < source_pitch:
            raise CudaVulkanInteropError("D3D12 destination row pitch is smaller than the image row")
        byte_count = int(row_pitch) * height
        if byte_count > int(getattr(target, "allocation_size", 0)):
            raise CudaVulkanInteropError("CUDA RGBA tensor does not fit in the D3D12 shared buffer")
        if not bool(tensor.is_contiguous()):
            raise CudaVulkanInteropError("CUDA D3D12 buffer copy requires a contiguous tensor")
        if stream is None:
            import torch

            stream = int(torch.cuda.current_stream(device=tensor.device).cuda_stream)
        slot = self._buffer_slots[id(target)]
        self._check(
            self._cudart.cudaMemcpy2DAsync(
                slot.pointer,
                int(row_pitch),
                ctypes.c_void_p(int(tensor.data_ptr())),
                source_pitch,
                source_pitch,
                height,
                self._CUDA_MEMCPY_DEVICE_TO_DEVICE,
                ctypes.c_void_p(int(stream)),
            ),
            "cudaMemcpy2DAsync(D3D12 buffer)",
        )

    def synchronize(self, *, stream: int | None = None) -> None:
        if stream is None:
            import torch

            stream = int(torch.cuda.current_stream().cuda_stream)
        self._check(self._cudart.cudaStreamSynchronize(ctypes.c_void_p(int(stream))), "cudaStreamSynchronize")

    def register_semaphore(self, target: VulkanExportableSemaphore) -> None:
        if not self.capabilities.external_semaphore:
            raise CudaVulkanInteropError("CUDA external semaphore API is unavailable")
        key = id(target)
        if key in self._semaphores:
            return
        handle = target.export_handle
        if bool(getattr(target, "timeline", False)):
            handle_type = (
                self._CUDA_TIMELINE_SEMAPHORE_WIN32
                if os.name == "nt"
                else self._CUDA_TIMELINE_SEMAPHORE_FD
            )
        else:
            handle_type = (
                self._CUDA_OPAQUE_WIN32 if os.name == "nt" else self._CUDA_OPAQUE_FD
            )
        desc = _ExternalSemaphoreHandleDesc(
            type=int(
                getattr(
                    target,
                    "cuda_external_semaphore_handle_type",
                    handle_type,
                )
            ),
            flags=0,
        )
        if os.name == "nt":
            desc.handle.win32.handle = ctypes.c_void_p(int(handle))
        else:
            desc.handle.fd = int(handle)
        external = ctypes.c_void_p()
        self._check(
            self._cudart.cudaImportExternalSemaphore(
                ctypes.byref(external), ctypes.byref(desc)
            ),
            "cudaImportExternalSemaphore",
        )
        target.close_export_handle()
        self._semaphores[key] = _CudaSemaphore(target, external)

    @staticmethod
    def _semaphore_value(target: VulkanExportableSemaphore, value: int) -> int:
        result = int(value)
        if bool(getattr(target, "timeline", False)):
            if result < 1:
                raise CudaVulkanInteropError(
                    "CUDA timeline semaphore value must be positive"
                )
        elif result != 0:
            raise CudaVulkanInteropError(
                "CUDA binary semaphore value must be zero"
            )
        return result

    def signal_semaphore(
        self,
        target: VulkanExportableSemaphore,
        *,
        value: int = 0,
        stream: int | None = None,
    ) -> None:
        if stream is None:
            import torch

            stream = int(torch.cuda.current_stream().cuda_stream)
        semaphore = self._semaphores.get(id(target))
        if semaphore is None:
            raise CudaVulkanInteropError("CUDA external semaphore is not registered")
        params = _ExternalSemaphoreSignalParams(
            params=_SemaphoreSignalParams(
                fence_value=self._semaphore_value(target, value),
                keyed_mutex_key=0,
            ),
            flags=0,
        )
        semaphore_array = (ctypes.c_void_p * 1)(semaphore.external)
        self._check(
            self._cudart.cudaSignalExternalSemaphoresAsync(
                semaphore_array,
                ctypes.byref(params),
                1,
                ctypes.c_void_p(int(stream)),
            ),
            "cudaSignalExternalSemaphoresAsync",
        )

    def wait_semaphore(
        self,
        target: VulkanExportableSemaphore,
        *,
        value: int = 0,
        stream: int | None = None,
    ) -> None:
        if stream is None:
            import torch

            stream = int(torch.cuda.current_stream().cuda_stream)
        wait_external = getattr(self._cudart, "cudaWaitExternalSemaphoresAsync", None)
        if wait_external is None:
            raise CudaVulkanInteropError(
                "CUDA external semaphore wait API is unavailable"
            )
        semaphore = self._semaphores.get(id(target))
        if semaphore is None:
            raise CudaVulkanInteropError("CUDA external semaphore is not registered")
        params = _ExternalSemaphoreWaitParams(
            params=_SemaphoreSignalParams(
                fence_value=self._semaphore_value(target, value),
                keyed_mutex_key=0,
            ),
            flags=0,
        )
        semaphore_array = (ctypes.c_void_p * 1)(semaphore.external)
        self._check(
            wait_external(
                semaphore_array,
                ctypes.byref(params),
                1,
                ctypes.c_void_p(int(stream)),
            ),
            "cudaWaitExternalSemaphoresAsync",
        )

    def release_slot(self, target: VulkanExportableImage) -> None:
        slot = self._slots.pop(id(target), None)
        if slot is not None:
            self._check(self._cudart.cudaDestroyExternalMemory(slot.external_memory), "cudaDestroyExternalMemory")

    def close(self) -> None:
        for slot, semaphore in tuple(self._rgba_slots.values()):
            self._check(
                self._cudart.cudaDestroyExternalSemaphore(semaphore),
                "cudaDestroyExternalSemaphore(FFmpeg RGBA)",
            )
            self._check(
                self._cudart.cudaDestroyExternalMemory(slot.external_memory),
                "cudaDestroyExternalMemory(FFmpeg RGBA)",
            )
        self._rgba_slots.clear()
        for slots, semaphores in tuple(self._nv12_slots.values()):
            for semaphore in semaphores:
                self._check(
                    self._cudart.cudaDestroyExternalSemaphore(semaphore),
                    "cudaDestroyExternalSemaphore(FFmpeg NV12)",
                )
            for slot in slots:
                self._check(
                    self._cudart.cudaDestroyExternalMemory(slot.external_memory),
                    "cudaDestroyExternalMemory(FFmpeg NV12)",
                )
        self._nv12_slots.clear()
        for semaphore in tuple(self._semaphores.values()):
            self._check(
                self._cudart.cudaDestroyExternalSemaphore(semaphore.external),
                "cudaDestroyExternalSemaphore",
            )
        self._semaphores.clear()
        for slot in tuple(self._buffer_slots.values()):
            self._check(
                self._cudart.cudaDestroyExternalMemory(slot.external_memory),
                "cudaDestroyExternalMemory(buffer)",
            )
        self._buffer_slots.clear()
        for slot in tuple(self._slots.values()):
            self._check(self._cudart.cudaDestroyExternalMemory(slot.external_memory), "cudaDestroyExternalMemory")
        self._slots.clear()
