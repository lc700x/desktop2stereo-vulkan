"""CUDA-to-D3D12 cross-adapter transfer into a Vulkan-owned display adapter."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

from .vulkan_resources import VulkanD3D12ImportedImage


class CudaD3D12BridgeError(RuntimeError):
    pass


class CudaD3D12BridgeBusyError(CudaD3D12BridgeError):
    """A ring slot's D3D12 command allocator is still in flight."""


class _CudaImportHandle:
    def __init__(self, handle: int, allocation_size: int, handle_type: int) -> None:
        self.export_handle: int | None = int(handle)
        self.allocation_size = int(allocation_size)
        self.size = int(allocation_size)
        self.cuda_external_memory_handle_type = int(handle_type)

    def close_export_handle(self) -> None:
        if self.export_handle is None:
            return
        if not ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(
            ctypes.c_void_p(self.export_handle)
        ):
            raise OSError(ctypes.get_last_error(), "CloseHandle failed")
        self.export_handle = None


class _CudaD3D12Fence(_CudaImportHandle):
    def __init__(self, handle: int) -> None:
        super().__init__(handle, 0, 4)  # cudaExternalSemaphoreHandleTypeD3D12Fence
        self.timeline = True


class _VulkanD3D12FenceSemaphore:
    """Import one shared D3D12 fence as a Vulkan timeline semaphore."""

    _HANDLE_TYPE = 0x00000008  # VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT
    _IMPORTABLE = 0x00000002  # VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT

    def __init__(self, context, handle: int, *, label: str) -> None:
        self.context = context
        self.vk = context.vk
        self.label = str(label)
        self.timeline = True
        self.semaphore = None
        self._handle_closed = False
        try:
            self._require_import_support()
            self._import_handle(int(handle))
        finally:
            if not self._handle_closed:
                self._close_handle(int(handle))

    @staticmethod
    def _close_handle(handle: int) -> None:
        if not ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(
            ctypes.c_void_p(handle)
        ):
            raise OSError(ctypes.get_last_error(), "CloseHandle failed")

    def _require_import_support(self) -> None:
        vk = self.vk
        features_by_type = getattr(
            self.context, "_d2s_d3d12_fence_import_features", None
        )
        if features_by_type is None:
            features_by_type = {}
            for name, semaphore_type in (
                ("binary", vk.VK_SEMAPHORE_TYPE_BINARY),
                ("timeline", vk.VK_SEMAPHORE_TYPE_TIMELINE),
            ):
                type_info = vk.VkSemaphoreTypeCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                    semaphoreType=semaphore_type,
                    initialValue=0,
                )
                info = vk.VkPhysicalDeviceExternalSemaphoreInfo(
                    sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO,
                    pNext=type_info,
                    handleType=self._HANDLE_TYPE,
                )
                properties = vk.vkGetPhysicalDeviceExternalSemaphoreProperties(
                    self.context.physical_device,
                    info,
                    vk.VkExternalSemaphoreProperties(
                        sType=vk.VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES
                    ),
                )
                features_by_type[name] = int(properties.externalSemaphoreFeatures)
            self.context._d2s_d3d12_fence_import_features = features_by_type
            print(
                "[VulkanOutput] D3D12 fence import capabilities: "
                f"binary=0x{features_by_type['binary']:x} "
                f"timeline=0x{features_by_type['timeline']:x}",
                flush=True,
            )
        if not features_by_type["timeline"] & self._IMPORTABLE:
            raise CudaD3D12BridgeError(
                "Vulkan device cannot import D3D12 fence timelines "
                f"(binary=0x{features_by_type['binary']:x}, "
                f"timeline=0x{features_by_type['timeline']:x})"
            )

    def _import_handle(self, handle: int) -> None:
        vk = self.vk
        type_info = vk.VkSemaphoreTypeCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
            pNext=None,
            semaphoreType=vk.VK_SEMAPHORE_TYPE_TIMELINE,
            initialValue=0,
        )
        self.semaphore = vk.vkCreateSemaphore(
            self.context.device,
            vk.VkSemaphoreCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                pNext=type_info,
            ),
            None,
        )
        try:
            proc = vk.lib.vkGetDeviceProcAddr(
                self.context.device, b"vkImportSemaphoreWin32HandleKHR"
            )
            if proc == vk.ffi.NULL:
                raise CudaD3D12BridgeError(
                    "vkImportSemaphoreWin32HandleKHR is unavailable"
                )
            import_function = vk.ffi.cast(
                "VkResult(*)(VkDevice, const VkImportSemaphoreWin32HandleInfoKHR*)",
                proc,
            )
            import_info = vk.VkImportSemaphoreWin32HandleInfoKHR(
                sType=vk.VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR,
                pNext=None,
                semaphore=self.semaphore,
                flags=0,
                handleType=self._HANDLE_TYPE,
                handle=vk.ffi.cast("HANDLE", handle),
                name=None,
            )
            result = int(
                import_function(self.context.device, vk.ffi.addressof(import_info))
            )
            if result != int(vk.VK_SUCCESS):
                raise CudaD3D12BridgeError(
                    f"Vulkan import of D3D12 fence failed: {result}"
                )
        except Exception:
            vk.vkDestroySemaphore(self.context.device, self.semaphore, None)
            self.semaphore = None
            raise
        finally:
            self._handle_closed = True
            self._close_handle(handle)

    def close(self) -> None:
        if self.semaphore is not None and self.context.device is not None:
            self.vk.vkDestroySemaphore(self.context.device, self.semaphore, None)
        self.semaphore = None


class CudaD3D12OutputBridge:
    """Stage full-resolution CUDA eyes through cross-adapter D3D12 resources.

    Pixel copies and cross-API handoff waits stay on the GPUs. CUDA writes
    cross-adapter buffers, D3D12 copies them into local textures, and Vulkan
    imports D3D12 timeline fences for image readiness and release.
    """

    _CUDA_D3D12_HEAP = 4

    def __init__(
        self,
        context,
        importer,
        *,
        source_luid: int,
        target_luid: int,
        width: int,
        height: int,
        ring_size: int,
    ) -> None:
        if os.name != "nt":
            raise CudaD3D12BridgeError("the CUDA/D3D12 bridge is available only on Windows")
        self.context = context
        self.importer = importer
        self.width = int(width)
        self.height = int(height)
        self.ring_size = int(ring_size)
        self._library = self._load_library()
        self._bridge = ctypes.c_void_p()
        self.left_images: list[VulkanD3D12ImportedImage] = []
        self.right_images: list[VulkanD3D12ImportedImage] = []
        self.left_buffers: list[_CudaImportHandle] = []
        self.right_buffers: list[_CudaImportHandle] = []
        self._source_fence: _CudaD3D12Fence | None = None
        self._source_consumed_fence: _CudaD3D12Fence | None = None
        self.ready_semaphore: _VulkanD3D12FenceSemaphore | None = None
        self.left_release_semaphores: list[_VulkanD3D12FenceSemaphore] = []
        self.right_release_semaphores: list[_VulkanD3D12FenceSemaphore] = []
        self._source_fence_value = 0
        self._source_consumed_values = [0] * self.ring_size
        self._last_stream: int | None = None
        self.cpu_wait_count = 0
        self._slot_completions: list[tuple[int, int, bool] | None] = [None] * self.ring_size
        self.row_pitch = 0
        self.buffer_size = 0
        self._configure_functions()
        self._create_native(source_luid, target_luid)
        try:
            self.row_pitch = int(self._library.d2s_d3d12_bridge_row_pitch(self._bridge))
            self.buffer_size = int(self._library.d2s_d3d12_bridge_buffer_size(self._bridge))
            if self.row_pitch < self.width * 4 or self.buffer_size < self.row_pitch * self.height:
                raise CudaD3D12BridgeError("D3D12 bridge returned invalid image pitches")
            for slot in range(self.ring_size):
                for eye, buffers, images in (
                    (0, self.left_buffers, self.left_images),
                    (1, self.right_buffers, self.right_images),
                ):
                    handle = self._native_handle(
                        self._library.d2s_d3d12_bridge_cuda_buffer_handle,
                        slot,
                        eye,
                    )
                    buffer = _CudaImportHandle(
                        handle, self.buffer_size, self._CUDA_D3D12_HEAP
                    )
                    buffers.append(buffer)
                    importer.register_buffer(buffer)

                    image_handle = self._native_handle(
                        self._library.d2s_d3d12_bridge_vulkan_image_handle,
                        slot,
                        eye,
                    )
                    image = VulkanD3D12ImportedImage(
                        context,
                        self.width,
                        self.height,
                        image_handle,
                        adapter_luid=int(target_luid),
                        label=f"cuda-d3d12-cross-eye-{eye}-{slot}",
                    )
                    images.append(image)
                    context.prepare_external_image_for_cuda(image.resource, defer=True)
            fence_handle = self._native_handle(
                self._library.d2s_d3d12_bridge_cuda_fence_handle
            )
            self._source_fence = _CudaD3D12Fence(fence_handle)
            importer.register_semaphore(self._source_fence)
            consumed_handle = self._native_handle(
                self._library.d2s_d3d12_bridge_cuda_consumed_fence_handle
            )
            self._source_consumed_fence = _CudaD3D12Fence(consumed_handle)
            importer.register_semaphore(self._source_consumed_fence)

            ready_handle = self._native_handle(
                self._library.d2s_d3d12_bridge_vulkan_ready_fence_handle
            )
            self.ready_semaphore = _VulkanD3D12FenceSemaphore(
                context, ready_handle, label="cuda-d3d12-vulkan-ready"
            )
            for slot in range(self.ring_size):
                for eye, semaphores in (
                    (0, self.left_release_semaphores),
                    (1, self.right_release_semaphores),
                ):
                    handle = self._native_handle(
                        self._library.d2s_d3d12_bridge_vulkan_release_fence_handle,
                        slot,
                        eye,
                    )
                    semaphores.append(
                        _VulkanD3D12FenceSemaphore(
                            context,
                            handle,
                            label=f"cuda-d3d12-vulkan-release-{eye}-{slot}",
                        )
                    )
        except Exception:
            self.close()
            raise

    @staticmethod
    def _load_library():
        path = Path(__file__).resolve().parent / "native" / "windows" / "d2s_cuda_d3d12_bridge.dll"
        if not path.is_file():
            raise CudaD3D12BridgeError(
                "bundled d2s_cuda_d3d12_bridge.dll is missing; cross-adapter GPU output is unavailable"
            )
        try:
            return ctypes.CDLL(str(path))
        except OSError as exc:
            raise CudaD3D12BridgeError(f"could not load D3D12 bridge: {exc}") from exc

    def _configure_functions(self) -> None:
        library = self._library
        library.d2s_d3d12_bridge_create.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_char_p,
            ctypes.c_uint32,
        ]
        library.d2s_d3d12_bridge_create.restype = ctypes.c_int
        library.d2s_d3d12_bridge_destroy.argtypes = [ctypes.c_void_p]
        library.d2s_d3d12_bridge_destroy.restype = None
        for name in (
            "d2s_d3d12_bridge_row_pitch",
            "d2s_d3d12_bridge_buffer_size",
        ):
            function = getattr(library, name)
            function.argtypes = [ctypes.c_void_p]
            function.restype = ctypes.c_uint64
        for name, arguments in (
            ("d2s_d3d12_bridge_cuda_buffer_handle", [ctypes.c_uint32, ctypes.c_uint32]),
            ("d2s_d3d12_bridge_vulkan_image_handle", [ctypes.c_uint32, ctypes.c_uint32]),
            ("d2s_d3d12_bridge_cuda_fence_handle", []),
            ("d2s_d3d12_bridge_cuda_consumed_fence_handle", []),
            ("d2s_d3d12_bridge_vulkan_ready_fence_handle", []),
            (
                "d2s_d3d12_bridge_vulkan_release_fence_handle",
                [ctypes.c_uint32, ctypes.c_uint32],
            ),
        ):
            function = getattr(library, name)
            function.argtypes = [ctypes.c_void_p, *arguments, ctypes.c_char_p, ctypes.c_uint32]
            function.restype = ctypes.c_int64
        library.d2s_d3d12_bridge_copy_to_vulkan.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_char_p,
            ctypes.c_uint32,
        ]
        library.d2s_d3d12_bridge_copy_to_vulkan.restype = ctypes.c_int
        library.d2s_d3d12_bridge_check_slot_ready.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_char_p,
            ctypes.c_uint32,
        ]
        library.d2s_d3d12_bridge_check_slot_ready.restype = ctypes.c_int
        library.d2s_d3d12_bridge_wait.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_char_p,
            ctypes.c_uint32,
        ]
        library.d2s_d3d12_bridge_wait.restype = ctypes.c_int
        library.d2s_d3d12_bridge_wait_slot.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_char_p,
            ctypes.c_uint32,
        ]
        library.d2s_d3d12_bridge_wait_slot.restype = ctypes.c_int

    def _create_native(self, source_luid: int, target_luid: int) -> None:
        error = ctypes.create_string_buffer(512)
        result = self._library.d2s_d3d12_bridge_create(
            int(source_luid),
            int(target_luid),
            self.width,
            self.height,
            self.ring_size,
            ctypes.byref(self._bridge),
            error,
            len(error),
        )
        if int(result) != 0 or not self._bridge.value:
            message = error.value.decode("utf-8", errors="replace") or f"HRESULT 0x{int(result) & 0xFFFFFFFF:08x}"
            raise CudaD3D12BridgeError(f"could not initialize cross-adapter bridge: {message}")

    def _native_handle(self, function, *arguments) -> int:
        error = ctypes.create_string_buffer(512)
        handle = int(function(self._bridge, *arguments, error, len(error))) & 0xFFFFFFFFFFFFFFFF
        if handle == 0:
            message = error.value.decode("utf-8", errors="replace") or "unknown Win32 error"
            raise CudaD3D12BridgeError(message)
        return handle

    def copy(
        self,
        left,
        right,
        *,
        frame_id: int,
        stream: int,
        release_values: tuple[int, int] = (0, 0),
    ) -> tuple[object, object, int, int]:
        if tuple(left.shape) != (self.height, self.width, 4) or tuple(right.shape) != (
            self.height,
            self.width,
            4,
        ):
            raise CudaD3D12BridgeError("CUDA eye dimensions changed during D3D12 bridge use")
        if getattr(left, "dtype", None) != getattr(right, "dtype", None):
            raise CudaD3D12BridgeError("left/right CUDA eye types differ")
        slot = int(frame_id) % self.ring_size
        error = ctypes.create_string_buffer(512)
        result = self._library.d2s_d3d12_bridge_check_slot_ready(
            self._bridge, slot, error, len(error)
        )
        if int(result) != 0:
            message = error.value.decode("utf-8", errors="replace") or f"HRESULT 0x{int(result) & 0xFFFFFFFF:08x}"
            if (int(result) & 0xFFFFFFFF) == 0x8000000A:
                raise CudaD3D12BridgeBusyError(message)
            raise CudaD3D12BridgeError(f"D3D12 slot readiness check failed: {message}")
        self._last_stream = int(stream)
        consumed_value = int(self._source_consumed_values[slot])
        if consumed_value:
            self.importer.wait_semaphore(
                self._source_consumed_fence,
                value=consumed_value,
                stream=stream,
            )
        self.importer.copy_tensor_to_pitched_buffer(
            left, self.left_buffers[slot], row_pitch=self.row_pitch, stream=stream
        )
        self.importer.copy_tensor_to_pitched_buffer(
            right, self.right_buffers[slot], row_pitch=self.row_pitch, stream=stream
        )
        self._source_fence_value += 1
        self.importer.signal_semaphore(
            self._source_fence,
            value=self._source_fence_value,
            stream=stream,
        )
        completion = ctypes.c_uint64()
        error = ctypes.create_string_buffer(512)
        result = self._library.d2s_d3d12_bridge_copy_to_vulkan(
            self._bridge,
            slot,
            self._source_fence_value,
            int(release_values[0]),
            int(release_values[1]),
            ctypes.byref(completion),
            error,
            len(error),
        )
        if int(result) != 0:
            message = error.value.decode("utf-8", errors="replace") or f"HRESULT 0x{int(result) & 0xFFFFFFFF:08x}"
            if (int(result) & 0xFFFFFFFF) == 0x8000000A:
                raise CudaD3D12BridgeBusyError(message)
            raise CudaD3D12BridgeError(f"could not queue D3D12 eye copy: {message}")
        self._source_consumed_values[slot] = int(completion.value)
        self._slot_completions[slot] = (int(frame_id), int(completion.value), False)
        return self.left_images[slot], self.right_images[slot], slot, int(completion.value)

    def completion_for_frame(self, frame_id: int) -> int:
        frame_key = int(frame_id)
        for entry in self._slot_completions:
            if entry is not None and entry[0] == frame_key:
                return int(entry[1])
        raise CudaD3D12BridgeError(
            f"D3D12 completion was not retained for frame {frame_key}"
        )

    def wait_frame(self, frame_id: int) -> None:
        frame_key = int(frame_id)
        slot = next(
            (
                index
                for index, entry in enumerate(self._slot_completions)
                if entry is not None and entry[0] == frame_key
            ),
            None,
        )
        if slot is None:
            raise CudaD3D12BridgeError(
                f"D3D12 completion was not retained for frame {frame_key}"
            )
        self.wait_slot(slot)

    def wait_slot(self, slot: int) -> None:
        index = int(slot)
        if index < 0 or index >= self.ring_size:
            raise CudaD3D12BridgeError("invalid D3D12 bridge output slot")
        entry = self._slot_completions[index]
        if entry is None or entry[2]:
            return
        error = ctypes.create_string_buffer(512)
        result = self._library.d2s_d3d12_bridge_wait_slot(
            self._bridge, index, 5000, error, len(error)
        )
        if int(result) != 0:
            message = error.value.decode("utf-8", errors="replace") or f"HRESULT 0x{int(result) & 0xFFFFFFFF:08x}"
            raise CudaD3D12BridgeError(
                f"D3D12 eye copy for slot {index} did not complete: {message}"
            )
        self.cpu_wait_count += 1
        self._slot_completions[index] = (entry[0], entry[1], True)

    def wait_all(self) -> None:
        for slot in range(self.ring_size):
            self.wait_slot(slot)

    def synchronize_cuda(self) -> None:
        """Finish queued source-buffer writes before releasing imported memory."""
        if self._last_stream is not None:
            self.importer.synchronize(stream=self._last_stream)
            self._last_stream = None

    def close(self) -> None:
        try:
            self.wait_all()
        except Exception as exc:  # noqa: BLE001
            print(f"[VulkanOutput] D3D12 bridge drain failed: {exc}", flush=True)
        try:
            self.synchronize_cuda()
        except Exception as exc:  # noqa: BLE001
            print(f"[VulkanOutput] CUDA bridge stream cleanup failed: {exc}", flush=True)
        try:
            self.context.wait_idle()
        except Exception as exc:  # noqa: BLE001
            print(f"[VulkanOutput] Vulkan bridge cleanup wait failed: {exc}", flush=True)
        for image in (*self.left_images, *self.right_images):
            try:
                image.close()
            except Exception as exc:  # noqa: BLE001
                print(f"[VulkanOutput] D3D12 image cleanup failed: {exc}", flush=True)
        for semaphore in (
            self.ready_semaphore,
            *self.left_release_semaphores,
            *self.right_release_semaphores,
        ):
            if semaphore is not None:
                try:
                    semaphore.close()
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[VulkanOutput] D3D12 semaphore cleanup failed: {exc}",
                        flush=True,
                    )
        self.ready_semaphore = None
        self.left_release_semaphores.clear()
        self.right_release_semaphores.clear()
        self.left_images.clear()
        self.right_images.clear()
        for buffer in (*self.left_buffers, *self.right_buffers):
            buffer.close_export_handle()
        self.left_buffers.clear()
        self.right_buffers.clear()
        if self._source_fence is not None:
            self._source_fence.close_export_handle()
            self._source_fence = None
        if self._bridge and self._bridge.value:
            self._library.d2s_d3d12_bridge_destroy(self._bridge)
            self._bridge = ctypes.c_void_p()
