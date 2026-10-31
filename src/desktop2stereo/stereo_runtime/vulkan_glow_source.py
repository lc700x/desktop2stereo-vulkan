from __future__ import annotations

from dataclasses import dataclass
import math
import os
import struct
import threading
import time
from typing import Any

from viewer.cuda_vulkan_interop import CudaVulkanImageImporter
from viewer.vulkan_context import ImageState, is_vulkan_device_lost_error
from viewer.vulkan_resources import (
    VulkanBinarySemaphore,
    VulkanExportableBuffer,
    VulkanExportableImage,
    VulkanExportableSemaphore,
)
from viewer.vulkan_descriptors import VulkanStorageBuffer

from .vulkan_glow_source_pass import VulkanGlowSourcePass


def _create_interop_importer():
    """Pick the CUDA or ROCm/HIP external-memory importer for the glow source."""
    try:
        import torch

        is_rocm = bool(getattr(torch.version, "hip", None))
    except Exception:
        is_rocm = False
    if is_rocm:
        from viewer.rocm_vulkan_interop import RocmVulkanImageImporter

        return RocmVulkanImageImporter()
    return CudaVulkanImageImporter()


class VulkanGlowSourceUnavailable(RuntimeError):
    pass


@dataclass(slots=True)
class _GlowSlot:
    index: int
    image: VulkanExportableImage
    compute_done: VulkanBinarySemaphore
    compute_command: Any
    graphics_command: Any
    compute_fence: Any
    graphics_fence: Any
    screen_light_buffer: VulkanStorageBuffer
    edge_light_buffer: VulkanStorageBuffer
    crop_detection_buffer: VulkanStorageBuffer
    input_buffer: VulkanExportableBuffer | None = None
    input_ready: VulkanExportableSemaphore | None = None
    state: str = "free"
    generation: int = 0
    lease_count: int = 0
    crop_detection_requested: bool = False


class VulkanGlowSourceComputeBackend:
    """Produce a reusable Glow texture on a dedicated Vulkan compute queue.

    The graphics queue never waits for unfinished Glow work. A completed
    compute slot is transitioned on the graphics queue before it is published;
    otherwise the last published slot remains active.
    """

    TARGET_WIDTH = 320
    TARGET_HEIGHT = 180
    MAX_INPUT_SIDE = 512
    INPUT_BUFFER_CAPACITY = MAX_INPUT_SIDE * MAX_INPUT_SIDE * 3 * 4

    def __init__(
        self, context: Any, *, slot_count: int = 3, host_input: bool = False
    ) -> None:
        self.context = context
        self.vk = context.vk
        self.slot_count = max(3, int(slot_count))
        self._host_input = bool(host_input)
        if int(context.compute_queue_family_index) != int(context.queue_family_index):
            raise VulkanGlowSourceUnavailable(
                "Glow compute and Filament graphics queues must share one queue family"
            )
        if int(getattr(context, "compute_queue_index", 0)) == 0:
            raise VulkanGlowSourceUnavailable(
                "a second Vulkan queue is required for non-blocking Glow compute"
            )
        self.compute_queue = context.compute_queue
        self.graphics_queue = context.graphics_queue
        self.importer = None if self._host_input else _create_interop_importer()
        # ROCm's stable path synchronizes HIP before Vulkan submission and
        # intentionally does not consume the HIP external semaphore here.
        # CUDA still needs the producer-ready semaphore: the CUDA copy is
        # asynchronous and the compute queue must not read the imported buffer
        # before that copy completes.
        self._rocm_interop = bool(
            self.importer is not None
            and "rocm_vulkan_interop" in type(self.importer).__module__
        )
        self._input_memory_handle_type = (
            None if self._host_input else self._rocm_input_memory_handle_type()
        )
        if self.importer is not None:
            capabilities = self.importer.capabilities
            if not capabilities.external_memory or not capabilities.external_semaphore:
                self.importer.close()
                raise VulkanGlowSourceUnavailable(
                    "CUDA external buffer/semaphore interop is unavailable"
                )
        self.effect_pass = VulkanGlowSourcePass(
            context,
            target_width=self.TARGET_WIDTH,
            target_height=self.TARGET_HEIGHT,
            slot_count=self.slot_count,
        )
        self.history_buffer = VulkanStorageBuffer(
            context, self.TARGET_WIDTH * self.TARGET_HEIGHT * 16
        )
        self.command_pool = self._create_command_pool()
        commands = self._allocate_commands(self.slot_count * 2)
        self.slots: list[_GlowSlot] = []
        output_format = self.vk.VK_FORMAT_R8G8B8A8_UNORM
        try:
            for index in range(self.slot_count):
                image = VulkanExportableImage(
                    context,
                    self.TARGET_WIDTH,
                    self.TARGET_HEIGHT,
                    format=output_format,
                    label=f"glow-source-{index}",
                )
                context.prepare_external_image_for_producer(image.resource)
                self.slots.append(
                    _GlowSlot(
                        index=index,
                        image=image,
                        compute_done=VulkanBinarySemaphore(
                            context, label=f"glow-compute-done-{index}"
                        ),
                        compute_command=commands[index * 2],
                        graphics_command=commands[index * 2 + 1],
                        compute_fence=self._create_fence(signaled=True),
                        graphics_fence=self._create_fence(signaled=True),
                        screen_light_buffer=VulkanStorageBuffer(context, 16),
                        edge_light_buffer=VulkanStorageBuffer(context, 24 * 16),
                        crop_detection_buffer=VulkanStorageBuffer(context, 16),
                    )
                )
        except Exception:
            self.close()
            raise
        self._input_capacity = 0
        self._generation = 0
        self._serial = 0
        self._current_slot: _GlowSlot | None = None
        self._frame_slots: dict[int, _GlowSlot] = {}
        self._last_submit_ms = 0.0
        self._reuse_count = 0
        self._budget_skip_count = 0
        self._screen_light_rgb = (0.18, 0.18, 0.18)
        self._edge_light_rgb = tuple((0.0, 0.0, 0.0) for _ in range(24))
        self._crop_detection_uv: tuple[float, float, float, float] | None = None
        self._crop_detection_serial = -1
        self._history_key: tuple[object, ...] | None = None
        self._history_last_submit = 0.0
        self._closed = False
        # Glow compute submits happen on the dedicated compute queue and must
        # NOT wait on the device-wide context._lock used by the graphics-queue
        # compositor. Contending on that lock serialized the glow compute with
        # the compositor and stalled OpenXR (compositor present dropped to ~9
        # FPS / headset idled). A per-glow lock keeps the glow submissions
        # mutually exclusive with themselves while the compositor runs free.
        self._submit_lock = threading.Lock()

    def _rocm_input_memory_handle_type(self) -> int | None:
        """Use AMD's KMT handle for HIP-imported Windows input buffers."""
        if os.name != "nt" or "rocm_vulkan_interop" not in type(self.importer).__module__:
            return None
        value = os.environ.get("D2S_ROCM_KMT_EXTERNAL_MEMORY", "auto")
        if value.strip().lower() in {"0", "false", "off", "no", "disabled"}:
            return None
        return getattr(
            self.vk,
            "VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT",
            None,
        )

    def _create_command_pool(self):
        return self.vk.vkCreateCommandPool(
            self.context.device,
            self.vk.VkCommandPoolCreateInfo(
                sType=self.vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                flags=self.vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                queueFamilyIndex=int(self.context.compute_queue_family_index),
            ),
            None,
        )

    def _allocate_commands(self, count: int):
        return self.vk.vkAllocateCommandBuffers(
            self.context.device,
            self.vk.VkCommandBufferAllocateInfo(
                sType=self.vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                commandPool=self.command_pool,
                level=self.vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount=int(count),
            ),
        )

    def _create_fence(self, *, signaled: bool):
        return self.vk.vkCreateFence(
            self.context.device,
            self.vk.VkFenceCreateInfo(
                sType=self.vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                flags=self.vk.VK_FENCE_CREATE_SIGNALED_BIT if signaled else 0,
            ),
            None,
        )

    def _fence_complete(self, fence: Any) -> bool:
        try:
            # PyVulkan returns None on VK_SUCCESS and raises VkNotReady while
            # the fence is pending.
            self.vk.vkGetFenceStatus(self.context.device, fence)
            return True
        except Exception as exc:
            not_ready = getattr(self.vk, "VkNotReady", None)
            if not_ready is not None and isinstance(exc, not_ready):
                return False
            if is_vulkan_device_lost_error(exc):
                mark_device_lost = getattr(self.context, "mark_device_lost", None)
                if callable(mark_device_lost):
                    mark_device_lost(exc)
            raise

    def _reset_command(self, command: Any, fence: Any) -> None:
        self.vk.vkResetFences(self.context.device, 1, [fence])
        self.vk.vkResetCommandBuffer(command, 0)

    def _begin_command(self, command: Any) -> None:
        self.vk.vkBeginCommandBuffer(
            command,
            self.vk.VkCommandBufferBeginInfo(
                sType=self.vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                flags=self.vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            ),
        )

    def _submit_queue(self, *args) -> None:
        # Use a dedicated lock (not the device-wide context._lock, which the
        # graphics-queue compositor contends on). Swapchains/queues are
        # submitted from distinct queues, so exclusivity against the compositor
        # is unnecessary and only introduced stalls.
        with self._submit_lock:
            self.vk.vkQueueSubmit(*args)

    def _input_ready_wait(self, semaphore: Any) -> tuple[tuple[Any, ...], tuple[int, ...]]:
        """Return the producer wait required by this interop backend."""
        if bool(getattr(self, "_rocm_interop", False)):
            # HIP is synchronized before this submission on the stable ROCm
            # path. Do not reintroduce the AMD external-semaphore wait.
            return (), ()
        # CUDA copy_tensor_to_buffer() and signal_semaphore() are asynchronous;
        # Vulkan compute must consume the CUDA-produced ready point.
        return (
            (semaphore,),
            (self.vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,),
        )

    def _ensure_inputs(self, required_size: int) -> None:
        required = int(required_size)
        if required <= self._input_capacity and all(
            slot.input_buffer is not None
            and (self._host_input or slot.input_ready is not None)
            for slot in self.slots
        ):
            return
        if any(slot.state != "free" for slot in self.slots):
            raise VulkanGlowSourceUnavailable(
                "Glow input resolution changed while GPU slots are active"
            )
        if self.importer is not None:
            self.importer.close()
        for slot in self.slots:
            if slot.input_ready is not None:
                slot.input_ready.close()
            if slot.input_buffer is not None:
                slot.input_buffer.close()
            slot.input_ready = None
            slot.input_buffer = None
        if self._host_input:
            self.importer = None
        else:
            self.importer = _create_interop_importer()
        self._rocm_interop = bool(
            self.importer is not None
            and "rocm_vulkan_interop" in type(self.importer).__module__
        )
        # Monitor switching can change the source aspect ratio while previous
        # glow slots are still leased by the compositor. Reserve the complete
        # capped source envelope once, so that a 16:9 -> 16:10 change does not
        # tear down/import buffers mid-flight and permanently lose glow.
        allocation_size = max(required, self.INPUT_BUFFER_CAPACITY)
        self._input_capacity = allocation_size
        for slot in self.slots:
            if self._host_input:
                slot.input_buffer = VulkanStorageBuffer(self.context, allocation_size)
                slot.input_ready = None
            else:
                slot.input_buffer = VulkanExportableBuffer(
                    self.context,
                    allocation_size,
                    label=f"glow-source-input-{slot.index}",
                    memory_handle_type=self._input_memory_handle_type,
                )
                slot.input_ready = VulkanExportableSemaphore(
                    self.context, label=f"glow-source-ready-{slot.index}"
                )
                self.importer.register_buffer(slot.input_buffer)
                self.importer.register_semaphore(slot.input_ready)

    @staticmethod
    def prefilter_scale(mode: str) -> float:
        """Return the old MIP footprint measured in source pixels."""
        normalized = str(mode or "off").strip().lower()
        if normalized in {"glow", "screen", "surround"}:
            return 256.0
        return 1.0

    def _prepare_source(self, source: Any):
        if self._host_input:
            return self._prepare_host_source(source)
        import torch

        if not isinstance(source, torch.Tensor) or not source.is_cuda:
            raise VulkanGlowSourceUnavailable("Glow GPU source requires a CUDA tensor")
        value = source
        if value.ndim == 3:
            if int(value.shape[-1]) in (3, 4):
                value = value[..., :3].permute(2, 0, 1).unsqueeze(0)
            elif int(value.shape[0]) in (3, 4):
                value = value[:3].unsqueeze(0)
        if value.ndim != 4 or int(value.shape[0]) != 1 or int(value.shape[1]) < 3:
            raise VulkanGlowSourceUnavailable(
                f"Glow GPU source requires [1,3,H,W], got {tuple(value.shape)}"
            )
        value = value[:, :3]
        if value.dtype == torch.uint8:
            value = value.to(dtype=torch.float32).div_(255.0)
        elif value.dtype != torch.float32:
            value = value.to(dtype=torch.float32)
        # Downsample so the HIP buffer copy + glow compute are cheap. The glow
        # target is only 320x180, so a ~512px source is ample; copying the full
        # 4K source every sample saturated the GPU and dropped the OpenXR
        # compositor to ~8 FPS on AMD. Capping the longest side keeps the glow
        # visually identical on NVIDIA and ROCm.
        hist, wid = int(value.shape[-2]), int(value.shape[-1])
        longest = max(hist, wid)
        max_side = self.MAX_INPUT_SIDE
        if longest > max_side:
            scale = max_side / float(longest)
            nh = max(1, int(round(hist * scale)))
            nw = max(1, int(round(wid * scale)))
            import torch.nn.functional as _F

            value = _F.interpolate(
                value, size=(nh, nw), mode="bilinear", align_corners=False
            )
        return value.contiguous()

    def _prepare_host_source(self, source: Any):
        """Downsample a CPU RGBA image for the safe cross-adapter glow path."""
        import numpy as np

        value = np.asarray(source)
        if value.ndim == 4:
            if value.shape[0] != 1:
                raise VulkanGlowSourceUnavailable(
                    f"Glow host source requires one image, got {value.shape[0]}"
                )
            value = value[0]
        if value.ndim != 3 or value.shape[-1] not in (3, 4):
            raise VulkanGlowSourceUnavailable(
                f"Glow host source requires [H,W,3|4], got {value.shape}"
            )
        value = value[..., :3]
        source_height, source_width = map(int, value.shape[:2])
        if source_height < 1 or source_width < 1:
            raise VulkanGlowSourceUnavailable("Glow host source is empty")
        longest = max(source_height, source_width)
        if longest > self.MAX_INPUT_SIDE:
            scale = self.MAX_INPUT_SIDE / float(longest)
            height = max(1, int(round(source_height * scale)))
            width = max(1, int(round(source_width * scale)))
            try:
                import cv2

                value = cv2.resize(value, (width, height), interpolation=cv2.INTER_AREA)
            except ImportError:
                y = np.minimum(
                    ((np.arange(height, dtype=np.float32) + 0.5)
                     * source_height / height).astype(np.int32),
                    source_height - 1,
                )
                x = np.minimum(
                    ((np.arange(width, dtype=np.float32) + 0.5)
                     * source_width / width).astype(np.int32),
                    source_width - 1,
                )
                value = value[y[:, None], x[None, :]]
        value = np.asarray(value, dtype=np.float32)
        if np.asarray(source).dtype == np.uint8:
            value *= 1.0 / 255.0
        return np.ascontiguousarray(value.transpose(2, 0, 1)[None])

    @staticmethod
    def _normalize_source_crop_uv(value: Any) -> tuple[float, float, float, float]:
        """Return a finite, in-bounds source crop rectangle in top-left UVs."""
        try:
            x, y, width, height = (float(item) for item in value)
        except (TypeError, ValueError):
            return (0.0, 0.0, 1.0, 1.0)
        if not all(math.isfinite(item) for item in (x, y, width, height)):
            return (0.0, 0.0, 1.0, 1.0)
        x = max(0.0, min(0.9, x))
        y = max(0.0, min(0.9, y))
        width = max(0.1, min(1.0 - x, width))
        height = max(0.1, min(1.0 - y, height))
        return (x, y, width, height)

    def submit(
        self,
        source: Any,
        *,
        mode: str,
        screen_light_only: bool = False,
        temporal_smoothing_seconds: float = 0.10,
        source_crop_uv: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
        detect_crop: bool = False,
    ) -> bool:
        if self._closed:
            return False
        self.poll()
        slot = next((item for item in self.slots if item.state == "free"), None)
        if slot is None:
            self._budget_skip_count += 1
            return False
        value = self._prepare_source(source)
        source_height, source_width = int(value.shape[-2]), int(value.shape[-1])
        self._ensure_inputs(
            self.effect_pass.input_buffer_size(source_width, source_height)
        )
        if slot.input_buffer is None or (
            not self._host_input and slot.input_ready is None
        ):
            raise VulkanGlowSourceUnavailable("Glow input slot is unavailable")
        start = time.perf_counter()
        prefilter_scale = self.prefilter_scale(mode)
        surround_region_average = str(mode or "").strip().lower() == "surround"
        source_crop_uv = self._normalize_source_crop_uv(source_crop_uv)
        history_key = (
            str(mode or "").strip().lower(), source_width, source_height,
            round(prefilter_scale, 4), surround_region_average,
            *(round(float(value), 5) for value in source_crop_uv),
        )
        temporal_alpha = 1.0
        if not screen_light_only and history_key == self._history_key:
            elapsed = max(0.0, start - self._history_last_submit)
            smoothing = max(0.0, float(temporal_smoothing_seconds))
            if smoothing > 0.0:
                temporal_alpha = 1.0 - math.exp(-elapsed / smoothing)
        self._reset_command(slot.compute_command, slot.compute_fence)
        if self._host_input:
            slot.input_buffer.write_bytes(memoryview(value).cast("B"))
            wait_semaphores, wait_stages = (), ()
        else:
            self.importer.copy_tensor_to_buffer(value, slot.input_buffer)
            self.importer.signal_semaphore(slot.input_ready)
            wait_semaphores, wait_stages = self._input_ready_wait(
                slot.input_ready.semaphore
            )
        self._begin_command(slot.compute_command)
        if self._host_input:
            input_barrier = self.vk.VkBufferMemoryBarrier(
                sType=self.vk.VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                srcAccessMask=self.vk.VK_ACCESS_HOST_WRITE_BIT,
                dstAccessMask=self.vk.VK_ACCESS_SHADER_READ_BIT,
                srcQueueFamilyIndex=self.vk.VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex=self.vk.VK_QUEUE_FAMILY_IGNORED,
                buffer=slot.input_buffer.buffer,
                offset=0,
                size=int(value.nbytes),
            )
            self.vk.vkCmdPipelineBarrier(
                slot.compute_command,
                self.vk.VK_PIPELINE_STAGE_HOST_BIT,
                self.vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0,
                None,
                1,
                [input_barrier],
                0,
                None,
            )
        self._record_history_barrier(slot.compute_command)
        self.effect_pass.record(
            slot.compute_command,
            slot_index=slot.index,
            source_buffer=slot.input_buffer,
            output_image=slot.image.resource,
            screen_light_buffer=slot.screen_light_buffer,
            edge_light_buffer=slot.edge_light_buffer,
            crop_detection_buffer=slot.crop_detection_buffer,
            history_buffer=self.history_buffer,
            source_width=source_width,
            source_height=source_height,
            prefilter_scale=prefilter_scale,
            surround_region_average=surround_region_average,
            screen_light_only=screen_light_only,
            temporal_alpha=temporal_alpha,
            source_crop_uv=source_crop_uv,
            detect_crop=detect_crop,
        )
        self._record_screen_light_host_barrier(
            slot.compute_command, slot.screen_light_buffer
        )
        self._record_screen_light_host_barrier(
            slot.compute_command, slot.edge_light_buffer,
            size=slot.edge_light_buffer.size,
        )
        if detect_crop:
            self._record_screen_light_host_barrier(
                slot.compute_command, slot.crop_detection_buffer,
            )
        self.vk.vkEndCommandBuffer(slot.compute_command)
        self._submit_queue(
            self.compute_queue,
            1,
            [
                self.vk.VkSubmitInfo(
                    sType=self.vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
                    waitSemaphoreCount=len(wait_semaphores),
                    pWaitSemaphores=list(wait_semaphores),
                    pWaitDstStageMask=list(wait_stages),
                    commandBufferCount=1,
                    pCommandBuffers=[slot.compute_command],
                    signalSemaphoreCount=1,
                    pSignalSemaphores=[slot.compute_done.semaphore],
                )
            ],
            slot.compute_fence,
        )
        self._generation += 1
        slot.generation = self._generation
        slot.crop_detection_requested = bool(detect_crop)
        slot.state = "computing"
        if not screen_light_only:
            self._history_key = history_key
            self._history_last_submit = start
        self._last_submit_ms = (time.perf_counter() - start) * 1000.0
        return True

    def _record_history_barrier(self, command: Any) -> None:
        vk = self.vk
        barrier = vk.VkBufferMemoryBarrier(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask=(
                vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT
            ),
            srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            buffer=self.history_buffer.buffer,
            offset=0,
            size=self.history_buffer.size,
        )
        vk.vkCmdPipelineBarrier(
            command,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, None, 1, [barrier], 0, None,
        )

    def _record_screen_light_host_barrier(
        self, command: Any, buffer: VulkanStorageBuffer, *, size: int = 16
    ) -> None:
        vk = self.vk
        barrier = vk.VkBufferMemoryBarrier(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_HOST_READ_BIT,
            srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            buffer=buffer.buffer,
            offset=0,
            size=int(size),
        )
        vk.vkCmdPipelineBarrier(
            command,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_HOST_BIT,
            0,
            0,
            None,
            1,
            [barrier],
            0,
            None,
        )

    def _read_screen_light(self, slot: _GlowSlot) -> None:
        values = struct.unpack("<4f", slot.screen_light_buffer.read_bytes(16))
        if all(value == value and abs(value) != float("inf") for value in values[:3]):
            self._screen_light_rgb = tuple(
                max(0.0, min(8.0, float(value))) for value in values[:3]
            )
        edge_values = struct.unpack("<96f", slot.edge_light_buffer.read_bytes(24 * 16))
        self._edge_light_rgb = tuple(
            tuple(max(0.0, min(8.0, float(edge_values[index * 4 + channel])))
                  for channel in range(3))
            for index in range(24)
        )
        if slot.crop_detection_requested:
            values = struct.unpack("<4f", slot.crop_detection_buffer.read_bytes(16))
            if all(value == value and abs(value) != float("inf") for value in values):
                self._crop_detection_uv = tuple(
                    max(0.0, min(1.0, float(value))) for value in values
                )
                self._crop_detection_serial = int(slot.generation)
            slot.crop_detection_requested = False

    def _record_image_barrier(
        self, command: Any, slot: _GlowSlot, *, to_sampling: bool
    ) -> None:
        vk = self.vk
        if to_sampling:
            source_access = vk.VK_ACCESS_SHADER_WRITE_BIT
            destination_access = vk.VK_ACCESS_SHADER_READ_BIT
            old_layout = vk.VK_IMAGE_LAYOUT_GENERAL
            new_layout = vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            source_stage = vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
            destination_stage = vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
        else:
            source_access = vk.VK_ACCESS_SHADER_READ_BIT
            destination_access = vk.VK_ACCESS_SHADER_WRITE_BIT
            old_layout = vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            new_layout = vk.VK_IMAGE_LAYOUT_GENERAL
            source_stage = vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
            destination_stage = vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
        barrier = vk.VkImageMemoryBarrier(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            srcAccessMask=source_access,
            dstAccessMask=destination_access,
            oldLayout=old_layout,
            newLayout=new_layout,
            srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            image=slot.image.resource.image,
            subresourceRange=vk.VkImageSubresourceRange(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0,
                levelCount=1,
                baseArrayLayer=0,
                layerCount=1,
            ),
        )
        vk.vkCmdPipelineBarrier(
            command,
            source_stage,
            destination_stage,
            0,
            0,
            None,
            0,
            None,
            1,
            [barrier],
        )

    def _publish(self, slot: _GlowSlot) -> None:
        if not self._fence_complete(slot.graphics_fence):
            return
        self._reset_command(slot.graphics_command, slot.graphics_fence)
        self._begin_command(slot.graphics_command)
        self._record_image_barrier(slot.graphics_command, slot, to_sampling=True)
        self.vk.vkEndCommandBuffer(slot.graphics_command)
        self._submit_queue(
            self.graphics_queue,
            1,
            [
                self.vk.VkSubmitInfo(
                    sType=self.vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
                    waitSemaphoreCount=1,
                    pWaitSemaphores=[slot.compute_done.semaphore],
                    pWaitDstStageMask=[self.vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT],
                    commandBufferCount=1,
                    pCommandBuffers=[slot.graphics_command],
                )
            ],
            slot.graphics_fence,
        )
        # Filament submits later work to this exact graphics queue. Vulkan
        # queue order therefore guarantees that the transition executes before
        # any fragment sample, without stalling the CPU on graphics_fence.
        self.context.register_image_state(
            slot.image.resource.image,
            ImageState(
                layout=self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                access_mask=self.vk.VK_ACCESS_SHADER_READ_BIT,
                stage_mask=self.vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                queue_family_index=self.context.queue_family_index,
            ),
        )
        previous = self._current_slot
        self._current_slot = slot
        slot.state = "current"
        self._serial += 1
        if previous is not None and previous is not slot:
            previous.state = "retired"

    def _submit_release(self, slot: _GlowSlot) -> None:
        self._reset_command(slot.graphics_command, slot.graphics_fence)
        self._begin_command(slot.graphics_command)
        self._record_image_barrier(slot.graphics_command, slot, to_sampling=False)
        self.vk.vkEndCommandBuffer(slot.graphics_command)
        self._submit_queue(
            self.graphics_queue,
            1,
            [
                self.vk.VkSubmitInfo(
                    sType=self.vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
                    commandBufferCount=1,
                    pCommandBuffers=[slot.graphics_command],
                )
            ],
            slot.graphics_fence,
        )
        # The Presenter invokes the lease callback only after the Filament
        # submissions for that frame. This release is queued behind those
        # reads on the same graphics queue.
        self.context.register_image_state(
            slot.image.resource.image,
            ImageState(
                layout=self.vk.VK_IMAGE_LAYOUT_GENERAL,
                access_mask=self.vk.VK_ACCESS_SHADER_WRITE_BIT,
                stage_mask=self.vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                queue_family_index=self.context.compute_queue_family_index,
            ),
        )
        slot.state = "releasing"

    def poll(self) -> None:
        if self._closed or bool(getattr(self.context, "device_lost", False)):
            return
        completed = sorted(
            (
                slot
                for slot in self.slots
                if slot.state == "computing" and self._fence_complete(slot.compute_fence)
            ),
            key=lambda item: item.generation,
        )
        for slot in completed:
            self._read_screen_light(slot)
            self._publish(slot)
        for slot in self.slots:
            if (
                slot.state == "retired"
                and slot.lease_count == 0
                and self._fence_complete(slot.graphics_fence)
            ):
                self._submit_release(slot)
            elif slot.state == "releasing" and self._fence_complete(slot.graphics_fence):
                slot.state = "free"

    def acquire(self, frame_id: int) -> dict[str, object]:
        self.poll()
        slot = self._current_slot
        if slot is None:
            return {}
        frame_key = int(frame_id)
        existing = self._frame_slots.get(frame_key)
        if existing is None:
            self._frame_slots[frame_key] = slot
            slot.lease_count += 1
        elif existing is not slot:
            raise RuntimeError("Glow frame was acquired from two different slots")
        self._reuse_count += 1
        return {
            "glow_vulkan_image": slot.image.resource,
            "glow_vulkan_serial": self._serial,
            "glow_source_path": (
                "vulkan_compute_host_input"
                if self._host_input
                else "vulkan_compute_external_image"
            ),
            "glow_gpu_submit_ms": self._last_submit_ms,
            "glow_reuse": self._reuse_count,
            "glow_budget_skip": self._budget_skip_count,
            "screen_light_linear_rgb": self._screen_light_rgb,
            "screen_light_sample_path": "vulkan_compute_reduction",
            "screen_edge_light_linear_rgb": self._edge_light_rgb,
            "crop_detection_uv": getattr(self, "_crop_detection_uv", None),
            "crop_detection_serial": getattr(self, "_crop_detection_serial", -1),
            "_vulkan_glow_release": self.release_frame,
        }

    def release_frame(self, frame_id: int) -> None:
        slot = self._frame_slots.pop(int(frame_id), None)
        if slot is None:
            return
        slot.lease_count = max(0, slot.lease_count - 1)
        self.poll()

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            device_lost = bool(getattr(self.context, "device_lost", False))
            if not device_lost and getattr(self, "compute_queue", None) is not None:
                self.vk.vkQueueWaitIdle(self.compute_queue)
            if not device_lost and getattr(self, "graphics_queue", None) is not None:
                self.vk.vkQueueWaitIdle(self.graphics_queue)
        finally:
            importer = getattr(self, "importer", None)
            if importer is not None:
                importer.close()
            for slot in getattr(self, "slots", ()):
                if slot.input_ready is not None:
                    slot.input_ready.close()
                if slot.input_buffer is not None:
                    slot.input_buffer.close()
                slot.screen_light_buffer.close()
                slot.edge_light_buffer.close()
                slot.crop_detection_buffer.close()
                slot.compute_done.close()
                slot.image.close()
                self.vk.vkDestroyFence(self.context.device, slot.compute_fence, None)
                self.vk.vkDestroyFence(self.context.device, slot.graphics_fence, None)
            effect_pass = getattr(self, "effect_pass", None)
            if effect_pass is not None:
                effect_pass.close()
            history_buffer = getattr(self, "history_buffer", None)
            if history_buffer is not None:
                history_buffer.close()
            command_pool = getattr(self, "command_pool", None)
            if command_pool is not None:
                self.vk.vkDestroyCommandPool(self.context.device, command_pool, None)
            self.slots = []
            self._frame_slots = {}
            self._current_slot = None
