"""AMD HIP runtime interop for exportable Vulkan image slots."""

from __future__ import annotations

import ctypes
import glob
import os
from pathlib import Path
from typing import Any

from runtime_paths import python_site_packages

from .vulkan_interop import VulkanInteropCapabilities, VulkanInteropMode
from .vulkan_resources import VulkanExportableImage, VulkanExportableSemaphore


class RocmVulkanInteropError(RuntimeError):
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


class _ExternalMemoryBufferDesc(ctypes.Structure):
    _fields_ = [
        ("offset", ctypes.c_uint64),
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


class _ExternalSemaphoreHandleDesc(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("handle", _ExternalHandleUnion),
        ("flags", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]


class _SemaphoreSignalParams(ctypes.Structure):
    _fields_ = [
        ("fence_value", ctypes.c_uint64),
        ("reserved", ctypes.c_uint64 * 8),
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


class _HipSlot:
    def __init__(self, target, external_memory, array):
        self.target = target
        self.external_memory = external_memory
        self.array = array


class _HipSemaphore:
    def __init__(self, target, external):
        self.target = target
        self.external = external


class _HipBufferSlot:
    def __init__(self, target, external_memory, pointer):
        self.target = target
        self.external_memory = external_memory
        self.pointer = pointer


class RocmVulkanImageImporter:
    """Import Vulkan memory once and copy HIP RGBA tensors into it."""

    _HIP_MEM_HANDLE_OPAQUE_FD = 1
    _HIP_MEM_HANDLE_OPAQUE_WIN32 = 2
    _HIP_ARRAY_COLOR_ATTACHMENT = 0x20
    _HIP_MEMCPY_DEVICE_TO_DEVICE = 3

    def __init__(self, *, hip_runtime_path: str | None = None) -> None:
        self._hip = self._load_hip_runtime(hip_runtime_path)
        self._slots: dict[int, _HipSlot] = {}
        self._buffer_slots: dict[int, _HipBufferSlot] = {}
        self._semaphores: dict[int, _HipSemaphore] = {}

    @property
    def capabilities(self) -> VulkanInteropCapabilities:
        return VulkanInteropCapabilities(
            producer="amd-rocm-hip",
            mode=VulkanInteropMode.GPU_COPY,
            external_memory=True,
            external_semaphore=all(
                hasattr(self._hip, name)
                for name in (
                    "hipImportExternalSemaphore",
                    "hipSignalExternalSemaphoresAsync",
                    "hipWaitExternalSemaphoresAsync",
                    "hipDestroyExternalSemaphore",
                )
            ),
            zero_copy=False,
        )

    @staticmethod
    def _load_hip_runtime(path: str | None):
        candidates = []
        if path:
            candidates.append(str(path))
        env_path = os.environ.get("D2S_HIP_RUNTIME_PATH")
        if env_path:
            candidates.append(env_path)
        site_packages = python_site_packages()
        candidates.extend(
            glob.glob(str(site_packages / "torch" / "lib" / "amdhip64*.dll"))
        )
        # ROCm SDK wheels ship the versioned HIP runtime (amdhip64_7.dll) under
        # their bin directories; mirror Triton's rocm_sdk discovery so the
        # external-memory API is found without D2S_HIP_RUNTIME_PATH.
        try:
            import rocm_sdk

            candidates.extend(
                str(path) for path in rocm_sdk.find_libraries("amdhip64")
            )
        except Exception:  # pragma: no cover - optional discovery path
            pass
        for sdk_dir in ("_rocm_sdk_core", "_rocm_sdk_devel"):
            candidates.extend(
                glob.glob(
                    str(site_packages / sdk_dir / "bin" / "amdhip64*.dll")
                )
            )
        candidates.extend(("amdhip64.dll", "libamdhip64.so", "libamdhip64.so.6"))
        for candidate in candidates:
            try:
                lib = ctypes.WinDLL(candidate) if os.name == "nt" else ctypes.CDLL(candidate)
                return RocmVulkanImageImporter._configure_functions(lib)
            except (OSError, AttributeError):
                continue
        raise RocmVulkanInteropError(
            "ROCm HIP runtime with external-memory API was not found"
        )

    @staticmethod
    def _configure_functions(lib):
        required = (
            "hipImportExternalMemory",
            "hipExternalMemoryGetMappedMipmappedArray",
            "hipGetMipmappedArrayLevel",
            "hipMemcpy2DToArrayAsync",
            "hipDestroyExternalMemory",
            "hipStreamSynchronize",
        )
        for name in required:
            if not hasattr(lib, name):
                raise AttributeError(name)
        lib.hipImportExternalMemory.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(_ExternalMemoryHandleDesc),
        ]
        lib.hipImportExternalMemory.restype = ctypes.c_int
        lib.hipExternalMemoryGetMappedMipmappedArray.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.POINTER(_ExternalMipmappedArrayDesc),
        ]
        lib.hipExternalMemoryGetMappedMipmappedArray.restype = ctypes.c_int
        lib.hipGetMipmappedArrayLevel.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint
        ]
        lib.hipGetMipmappedArrayLevel.restype = ctypes.c_int
        lib.hipMemcpy2DToArrayAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.hipMemcpy2DToArrayAsync.restype = ctypes.c_int
        lib.hipDestroyExternalMemory.argtypes = [ctypes.c_void_p]
        lib.hipDestroyExternalMemory.restype = ctypes.c_int
        lib.hipStreamSynchronize.argtypes = [ctypes.c_void_p]
        lib.hipStreamSynchronize.restype = ctypes.c_int
        for name in (
            "hipImportExternalSemaphore",
            "hipSignalExternalSemaphoresAsync",
            "hipDestroyExternalSemaphore",
        ):
            function = getattr(lib, name, None)
            if function is not None:
                if name == "hipImportExternalSemaphore":
                    function.argtypes = [
                        ctypes.POINTER(ctypes.c_void_p),
                        ctypes.POINTER(_ExternalSemaphoreHandleDesc),
                    ]
                elif name == "hipSignalExternalSemaphoresAsync":
                    function.argtypes = [
                        ctypes.POINTER(ctypes.c_void_p),
                        ctypes.POINTER(_ExternalSemaphoreSignalParams),
                        ctypes.c_uint,
                        ctypes.c_void_p,
                    ]
                else:
                    function.argtypes = [ctypes.c_void_p]
                function.restype = ctypes.c_int
        for name, argtypes, restype in (
            (
                "hipExternalMemoryGetMappedBuffer",
                [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.POINTER(_ExternalMemoryBufferDesc)],
                ctypes.c_int,
            ),
            (
                "hipMemcpyAsync",
                [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p],
                ctypes.c_int,
            ),
            (
                "hipMemcpy",
                [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int],
                ctypes.c_int,
            ),
        ):
            function = getattr(lib, name, None)
            if function is not None:
                function.argtypes = argtypes
                function.restype = restype
        return lib

    @staticmethod
    def _check(result: int, operation: str) -> None:
        if int(result) != 0:
            raise RocmVulkanInteropError(f"{operation} failed with HIP error {int(result)}")

    def register_slot(self, target: VulkanExportableImage):
        key = id(target)
        if key in self._slots:
            return target.resource
        prepare = getattr(target.context, "prepare_external_image_for_producer", None)
        if not callable(prepare):
            raise RocmVulkanInteropError(
                "Vulkan context cannot establish an external HIP image layout"
            )
        prepare(target.resource)
        handle = target.export_handle
        desc = _ExternalMemoryHandleDesc(
            type=(
                self._HIP_MEM_HANDLE_OPAQUE_WIN32
                if os.name == "nt"
                else self._HIP_MEM_HANDLE_OPAQUE_FD
            ),
            size=int(target.allocation_size),
            flags=0,
        )
        if os.name == "nt":
            desc.handle.win32.handle = ctypes.c_void_p(int(handle))
        else:
            desc.handle.fd = int(handle)
        external_memory = ctypes.c_void_p()
        self._check(
            self._hip.hipImportExternalMemory(
                ctypes.byref(external_memory), ctypes.byref(desc)
            ),
            "hipImportExternalMemory",
        )
        mapped_desc = _ExternalMipmappedArrayDesc(
            offset=0,
            format_desc=_ChannelFormatDesc(8, 8, 8, 8, 1),
            extent=_Extent(target.width, target.height, 0),
            flags=0,
            num_levels=1,
        )
        mipmap = ctypes.c_void_p()
        try:
            self._check(
                self._hip.hipExternalMemoryGetMappedMipmappedArray(
                    ctypes.byref(mipmap), external_memory, ctypes.byref(mapped_desc)
                ),
                "hipExternalMemoryGetMappedMipmappedArray",
            )
            array = ctypes.c_void_p()
            self._check(
                self._hip.hipGetMipmappedArrayLevel(
                    ctypes.byref(array), mipmap, 0
                ),
                "hipGetMipmappedArrayLevel",
            )
        except Exception:
            self._hip.hipDestroyExternalMemory(external_memory)
            raise
        self._slots[key] = _HipSlot(target, external_memory, array)
        target.close_export_handle()
        return target.resource

    def copy_tensor(self, tensor: Any, target: VulkanExportableImage, *, stream=None):
        resource = self.register_slot(target)
        if getattr(tensor, "device", None) is None or str(tensor.device.type) != "cuda":
            raise RocmVulkanInteropError("ROCm Vulkan copy requires a HIP tensor")
        if getattr(tensor, "dtype", None) is None or str(tensor.dtype) != "torch.uint8":
            raise RocmVulkanInteropError("ROCm Vulkan copy requires torch.uint8 RGBA tensor")
        if getattr(tensor, "ndim", 0) != 3 or tuple(tensor.shape) != (
            target.height,
            target.width,
            4,
        ):
            raise RocmVulkanInteropError("ROCm Vulkan copy requires HxWx4 tensor matching target")
        if not bool(tensor.is_contiguous()):
            raise RocmVulkanInteropError("ROCm Vulkan copy requires a contiguous tensor")
        if stream is None:
            import torch

            stream = int(torch.cuda.current_stream(device=tensor.device).cuda_stream)
        slot = self._slots[id(target)]
        self._check(
            self._hip.hipMemcpy2DToArrayAsync(
                slot.array,
                0,
                0,
                ctypes.c_void_p(int(tensor.data_ptr())),
                target.width * 4,
                target.width * 4,
                target.height,
                self._HIP_MEMCPY_DEVICE_TO_DEVICE,
                ctypes.c_void_p(int(stream)),
            ),
            "hipMemcpy2DToArrayAsync",
        )
        return resource

    def register_buffer(self, target: Any) -> None:
        """Import an exportable Vulkan storage buffer into HIP once."""
        key = id(target)
        if key in self._buffer_slots:
            return
        handle = getattr(target, "export_handle", None)
        allocation_size = int(getattr(target, "allocation_size", 0))
        if handle is None or allocation_size < 1:
            raise RocmVulkanInteropError(
                "exportable Vulkan buffer has no memory handle"
            )
        desc = _ExternalMemoryHandleDesc(
            type=(
                self._HIP_MEM_HANDLE_OPAQUE_WIN32
                if os.name == "nt"
                else self._HIP_MEM_HANDLE_OPAQUE_FD
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
            self._hip.hipImportExternalMemory(
                ctypes.byref(external_memory), ctypes.byref(desc)
            ),
            "hipImportExternalMemory(buffer)",
        )
        mapped_desc = _ExternalMemoryBufferDesc(
            offset=0,
            size=allocation_size,
            flags=0,
        )
        pointer = ctypes.c_void_p()
        try:
            self._check(
                self._hip.hipExternalMemoryGetMappedBuffer(
                    ctypes.byref(pointer), external_memory, ctypes.byref(mapped_desc)
                ),
                "hipExternalMemoryGetMappedBuffer",
            )
        except Exception:
            self._hip.hipDestroyExternalMemory(external_memory)
            raise
        target.close_export_handle()
        self._buffer_slots[key] = _HipBufferSlot(
            target, external_memory, pointer
        )

    def copy_tensor_to_buffer(
        self, tensor: Any, target: Any, *, stream=None
    ) -> None:
        """Copy a contiguous HIP float tensor into an imported Vulkan buffer."""
        self.register_buffer(target)
        if (
            getattr(tensor, "device", None) is None
            or str(tensor.device.type) != "cuda"
        ):
            raise RocmVulkanInteropError(
                "HIP Vulkan buffer copy requires a HIP tensor"
            )
        if str(getattr(tensor, "dtype", "")) != "torch.float32":
            raise RocmVulkanInteropError(
                "HIP Vulkan buffer copy requires torch.float32"
            )
        if not bool(tensor.is_contiguous()):
            raise RocmVulkanInteropError(
                "HIP Vulkan buffer copy requires a contiguous tensor"
            )
        byte_count = int(tensor.numel()) * 4
        if byte_count > int(getattr(target, "size", 0)):
            raise RocmVulkanInteropError(
                "HIP tensor does not fit in the Vulkan buffer"
            )
        if stream is None:
            import torch

            stream = int(torch.cuda.current_stream(device=tensor.device).cuda_stream)
        slot = self._buffer_slots[id(target)]
        sync_memcpy = getattr(self._hip, "hipMemcpy", None)
        if sync_memcpy is not None:
            # Synchronous hipMemcpy: completes before returning, so the glow
            # compute submit can read the buffer without a stream synchronize.
            # (hipMemcpyAsync + hipStreamSynchronize in the pipeline thread could
            # hang intermittently and stall the OpenXR compositor.)
            self._check(
                sync_memcpy(
                    slot.pointer,
                    ctypes.c_void_p(int(tensor.data_ptr())),
                    byte_count,
                    self._HIP_MEMCPY_DEVICE_TO_DEVICE,
                ),
                "hipMemcpy(buffer)",
            )
        else:
            self._check(
                self._hip.hipMemcpyAsync(
                    slot.pointer,
                    ctypes.c_void_p(int(tensor.data_ptr())),
                    byte_count,
                    self._HIP_MEMCPY_DEVICE_TO_DEVICE,
                    ctypes.c_void_p(int(stream)),
                ),
                "hipMemcpyAsync(buffer)",
            )

    def synchronize(self, *, stream=None) -> None:
        if stream is None:
            import torch

            stream = int(torch.cuda.current_stream().cuda_stream)
        self._check(
            self._hip.hipStreamSynchronize(ctypes.c_void_p(int(stream))),
            "hipStreamSynchronize",
        )

    def register_semaphore(self, target: VulkanExportableSemaphore) -> None:
        if not self.capabilities.external_semaphore:
            raise RocmVulkanInteropError("HIP external semaphore API is unavailable")
        if bool(getattr(target, "timeline", False)):
            raise RocmVulkanInteropError(
                "HIP timeline external semaphore is not implemented"
            )
        key = id(target)
        if key in self._semaphores:
            return
        handle = target.export_handle
        desc = _ExternalSemaphoreHandleDesc(
            type=(
                self._HIP_MEM_HANDLE_OPAQUE_WIN32
                if os.name == "nt"
                else self._HIP_MEM_HANDLE_OPAQUE_FD
            ),
            flags=0,
        )
        if os.name == "nt":
            desc.handle.win32.handle = ctypes.c_void_p(int(handle))
        else:
            desc.handle.fd = int(handle)
        external = ctypes.c_void_p()
        self._check(
            self._hip.hipImportExternalSemaphore(
                ctypes.byref(external), ctypes.byref(desc)
            ),
            "hipImportExternalSemaphore",
        )
        target.close_export_handle()
        self._semaphores[key] = _HipSemaphore(target, external)

    def signal_semaphore(self, target: VulkanExportableSemaphore, *, stream=None) -> None:
        if stream is None:
            import torch

            stream = int(torch.cuda.current_stream().cuda_stream)
        semaphore = self._semaphores.get(id(target))
        if semaphore is None:
            raise RocmVulkanInteropError("HIP external semaphore is not registered")
        params = _ExternalSemaphoreSignalParams(
            params=_SemaphoreSignalParams(fence_value=0), flags=0
        )
        handles = (ctypes.c_void_p * 1)(semaphore.external)
        self._check(
            self._hip.hipSignalExternalSemaphoresAsync(
                handles, ctypes.byref(params), 1, ctypes.c_void_p(int(stream))
            ),
            "hipSignalExternalSemaphoresAsync",
        )

    def wait_semaphore(self, target: VulkanExportableSemaphore, *, stream=None) -> None:
        if stream is None:
            import torch

            stream = int(torch.cuda.current_stream().cuda_stream)
        wait_external = getattr(self._hip, "hipWaitExternalSemaphoresAsync", None)
        if wait_external is None:
            raise RocmVulkanInteropError(
                "HIP external semaphore wait API is unavailable"
            )
        semaphore = self._semaphores.get(id(target))
        if semaphore is None:
            raise RocmVulkanInteropError("HIP external semaphore is not registered")
        params = _ExternalSemaphoreWaitParams(
            params=_SemaphoreSignalParams(fence_value=0), flags=0
        )
        handles = (ctypes.c_void_p * 1)(semaphore.external)
        self._check(
            wait_external(
                handles,
                ctypes.byref(params),
                1,
                ctypes.c_void_p(int(stream)),
            ),
            "hipWaitExternalSemaphoresAsync",
        )

    def close(self) -> None:
        destroy_semaphore = getattr(self._hip, "hipDestroyExternalSemaphore", None)
        if destroy_semaphore is not None:
            for semaphore in tuple(self._semaphores.values()):
                self._check(
                    destroy_semaphore(semaphore.external),
                    "hipDestroyExternalSemaphore",
                )
        self._semaphores.clear()
        for slot in tuple(self._slots.values()):
            self._check(
                self._hip.hipDestroyExternalMemory(slot.external_memory),
                "hipDestroyExternalMemory",
            )
        self._slots.clear()
        for slot in tuple(self._buffer_slots.values()):
            self._check(
                self._hip.hipDestroyExternalMemory(slot.external_memory),
                "hipDestroyExternalMemory",
            )
        self._buffer_slots.clear()
