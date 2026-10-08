"""GPU eye-image anti-aliasing after stereo visibility has been resolved."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

from viewer.vulkan_compute_pipeline import VulkanComputePipeline
from viewer.vulkan_descriptors import (
    DescriptorBinding,
    DescriptorBudget,
    VulkanDescriptorArena,
    VulkanStorageBuffer,
    VulkanStorageImage,
)


class _DeviceStorageBuffer(VulkanStorageBuffer):
    """Scratch only: device-local memory, never mapped by the CPU."""

    def _create(self) -> None:
        vk = self.vk
        self.buffer = vk.vkCreateBuffer(
            self.context.device,
            vk.VkBufferCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                size=self.size,
                usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            ),
            None,
        )
        requirements = vk.vkGetBufferMemoryRequirements(self.context.device, self.buffer)
        properties = vk.vkGetPhysicalDeviceMemoryProperties(self.context.physical_device)
        memory_type = next((
            index for index, item in enumerate(properties.memoryTypes)
            if requirements.memoryTypeBits & (1 << index)
            and item.propertyFlags & vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        ), None)
        if memory_type is None:
            self.close()
            raise RuntimeError("no device-local memory for edge-AA scratch")
        try:
            self.memory = vk.vkAllocateMemory(
                self.context.device,
                vk.VkMemoryAllocateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                    allocationSize=requirements.size,
                    memoryTypeIndex=memory_type,
                ),
                None,
            )
            vk.vkBindBufferMemory(self.context.device, self.buffer, self.memory, 0)
        except Exception:
            self.close()
            raise


class VulkanEyeEdgeAA:
    """Reuse one scratch pair per frame slot; record inside the stereo submit.

    Source buffer RGB is encoded sRGB; source image RGB is linear UNORM.
    The two shaders use the same edge search and positive linear-light mix.
    """

    def __init__(self, context: Any, width: int, height: int, *, images: bool,
                 packed_output: bool = False) -> None:
        self.context = context
        self.width, self.height = int(width), int(height)
        self.images, self.packed_output = bool(images), bool(packed_output)
        self.pipeline = None
        self.arena = None
        self.scratch: list[tuple[Any, Any]] = []
        self.descriptor_sets: list[Any] = []
        self.active_descriptor_set = None
        count = max(1, int(getattr(context, "frame_context_count", 3)))
        vk = context.vk
        kind = vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE if images else vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
        shader = "d2s_edge_aa_image.spv" if images else "d2s_edge_aa_buffer.spv"
        try:
            self.pipeline = VulkanComputePipeline(
                context, Path(__file__).resolve().parents[1] / "shaders" / shader,
                descriptor_bindings=[DescriptorBinding(binding=i, descriptor_type=kind) for i in range(4)],
                push_constants_size=12,
            )
            self.arena = VulkanDescriptorArena(
                context, DescriptorBudget(max_sets=count,
                    storage_buffers_per_set=0 if images else 4,
                    storage_images_per_set=4 if images else 0),
            )
            for _ in range(count):
                pair = []
                self.scratch.append(pair)
                for _eye in range(1 if self.packed_output else 2):
                    if images:
                        resource = VulkanStorageImage(context,
                            self.width * (2 if packed_output else 1), self.height)
                        pair.append(resource)
                        resource.transition_to_general()
                    else:
                        pair.append(_DeviceStorageBuffer(context, self.width * self.height * 12))
                if self.packed_output:
                    pair.append(pair[0])
                self.scratch[-1] = tuple(pair)
                self.descriptor_sets.append(self.arena.allocate(self.pipeline.descriptor_set_layout))
        except Exception:
            self.close()
            raise

    def prepare(self, slot: int, left: Any, right: Any) -> tuple[Any, Any]:
        pair = self.scratch[slot]
        self.active_descriptor_set = self.descriptor_sets[slot]
        update = self.arena.update_storage_image if self.images else self.arena.update_storage_buffer
        for binding, resource in enumerate((*pair, left, right)):
            update(self.active_descriptor_set, binding, resource)
        return pair

    def barrier(self, command_buffer: Any, *, before_render: bool = False) -> None:
        vk = self.context.vk
        barrier = vk.VkMemoryBarrier(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            srcAccessMask=vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT if before_render else vk.VK_ACCESS_SHADER_READ_BIT,
        )
        vk.vkCmdPipelineBarrier(
            command_buffer, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, [barrier], 0, None, 0, None,
        )

    def record(self, command_buffer: Any) -> None:
        self.barrier(command_buffer)
        self.pipeline.record_dispatch(command_buffer,
            group_count_x=(self.width + 15) // 16,
            group_count_y=(self.height + 15) // 16,
            group_count_z=2,
            descriptor_set=self.active_descriptor_set,
            push_constants=struct.pack("<III", self.width, self.height, int(self.packed_output)),
        )

    def close(self) -> None:
        if self.pipeline is not None:
            self.pipeline.close()
        if self.arena is not None:
            self.arena.close()
        for pair in self.scratch:
            for index, resource in enumerate(pair):
                if index == 0 or resource is not pair[0]:
                    resource.close()
        self.scratch = []
        self.descriptor_sets = []
        self.active_descriptor_set = None
        self.pipeline = self.arena = None
