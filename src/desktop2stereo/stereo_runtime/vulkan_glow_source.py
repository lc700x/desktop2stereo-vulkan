from __future__ import annotations

from dataclasses import dataclass
import math
import struct
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
    input_buffer: VulkanExportableBuffer | None = None
    input_ready: VulkanExportableSemaphore | None = None
    state: str = "free"
    generation: int = 0
    lease_count: int = 0


class VulkanGlowSourceComputeBackend:
    """Produce a reusable Glow texture on a dedicated Vulkan compute queue.

    The graphics queue never waits for unfinished Glow work. A completed
    compute slot is transitioned on the graphics queue before it is published;
    otherwise the last published slot remains active.
    """

    TARGET_WIDTH = 320
    TARGET_HEIGHT = 180

    def __init__(self, context: Any, *, slot_count: int = 3) -> None:
        self.context = context
        self.vk = context.vk
        self.slot_count = max(3, int(slot_count))
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
        self.importer = _create_interop_importer()
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
        self._history_key: tuple[object, ...] | None = None
        self._history_last_submit = 0.0
        self._closed = False

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
        # ponytail: device-wide lock; split by VkQueue handle if contention matters.
        with self.context._lock:
            self.vk.vkQueueSubmit(*args)

    def _ensure_inputs(self, required_size: int) -> None:
        required = int(required_size)
        if required <= self._input_capacity and all(
            slot.input_buffer is not None and slot.input_ready is not None
            for slot in self.slots
        ):
            return
        if any(slot.state != "free" for slot in self.slots):
            raise VulkanGlowSourceUnavailable(
                "Glow input resolution changed while GPU slots are active"
            )
        self.importer.close()
        for slot in self.slots:
            if slot.input_ready is not None:
                slot.input_ready.close()
            if slot.input_buffer is not None:
                slot.input_buffer.close()
            slot.input_ready = None
            slot.input_buffer = None
        self.importer = _create_interop_importer()
        self._input_capacity = required
        for slot in self.slots:
            slot.input_buffer = VulkanExportableBuffer(
                self.context,
                required,
                label=f"glow-source-input-{slot.index}",
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
        return value.contiguous()

    def submit(
        self,
        source: Any,
        *,
        mode: str,
        screen_light_only: bool = False,
        temporal_smoothing_seconds: float = 0.10,
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
        if slot.input_buffer is None or slot.input_ready is None:
            raise VulkanGlowSourceUnavailable("Glow input slot is unavailable")
        start = time.perf_counter()
        prefilter_scale = self.prefilter_scale(mode)
        surround_region_average = str(mode or "").strip().lower() == "surround"
        history_key = (
            str(mode or "").strip().lower(), source_width, source_height,
            round(prefilter_scale, 4), surround_region_average,
        )
        temporal_alpha = 1.0
        if not screen_light_only and history_key == self._history_key:
            elapsed = max(0.0, start - self._history_last_submit)
            smoothing = max(0.0, float(temporal_smoothing_seconds))
            if smoothing > 0.0:
                temporal_alpha = 1.0 - math.exp(-elapsed / smoothing)
        self._reset_command(slot.compute_command, slot.compute_fence)
        self.importer.copy_tensor_to_buffer(value, slot.input_buffer)
        self.importer.signal_semaphore(slot.input_ready)
        # AMD LLPC returns VK_ERROR_UNKNOWN when a compute-queue submit waits on
        # the HIP-signaled external binary semaphore (the graphics-queue path in
        # the local viewer is unaffected). Synchronize the HIP stream instead so
        # the buffer copy is complete before the compute submit, and let the
        # submit run without the external-semaphore wait. Ordering is preserved
        # by hipStreamSynchronize; the input_ready signal is then advisory only.
        sync = getattr(self.importer, "synchronize", None)
        if callable(sync):
            sync()
        self._begin_command(slot.compute_command)
        self._record_history_barrier(slot.compute_command)
        self.effect_pass.record(
            slot.compute_command,
            slot_index=slot.index,
            source_buffer=slot.input_buffer,
            output_image=slot.image.resource,
            screen_light_buffer=slot.screen_light_buffer,
            edge_light_buffer=slot.edge_light_buffer,
            history_buffer=self.history_buffer,
            source_width=source_width,
            source_height=source_height,
            prefilter_scale=prefilter_scale,
            surround_region_average=surround_region_average,
            screen_light_only=screen_light_only,
            temporal_alpha=temporal_alpha,
        )
        self._record_screen_light_host_barrier(
            slot.compute_command, slot.screen_light_buffer
        )
        self._record_screen_light_host_barrier(
            slot.compute_command, slot.edge_light_buffer,
            size=slot.edge_light_buffer.size,
        )
        self.vk.vkEndCommandBuffer(slot.compute_command)
        self._submit_queue(
            self.compute_queue,
            1,
            [
                self.vk.VkSubmitInfo(
                    sType=self.vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
                    waitSemaphoreCount=0,
                    pWaitSemaphores=[],
                    pWaitDstStageMask=[],
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
            "glow_source_path": "vulkan_compute_external_image",
            "glow_gpu_submit_ms": self._last_submit_ms,
            "glow_reuse": self._reuse_count,
            "glow_budget_skip": self._budget_skip_count,
            "screen_light_linear_rgb": self._screen_light_rgb,
            "screen_light_sample_path": "vulkan_compute_reduction",
            "screen_edge_light_linear_rgb": self._edge_light_rgb,
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
