from __future__ import annotations

from pathlib import Path
from typing import Any

from viewer.vulkan_compute_pipeline import VulkanComputePipeline
from viewer.vulkan_descriptors import (
    DescriptorBinding,
    DescriptorBudget,
    VulkanDescriptorArena,
)

from .output import output_edge_aa_enabled
from .vulkan_edge_aa import VulkanEyeEdgeAA
from .vulkan_stereo_pass import VulkanLayeredStereoParams


class VulkanStereoImagePass:
    """Write stereo eyes directly into presenter-owned storage images."""

    WORKGROUP_SIZE = 16
    PUSH_CONSTANTS_SIZE = 84
    BUFFER_COUNT = 4

    def __init__(
        self,
        context: Any,
        *,
        width: int,
        height: int,
        shader_path: str | Path = Path(__file__).resolve().parents[1] / "shaders" / "d2s_stereo_layered_output.spv",
        packed_output: bool = False,
        edge_aa_enabled: bool | None = None,
    ) -> None:
        if int(width) < 1 or int(height) < 1:
            raise ValueError("Vulkan stereo image dimensions must be positive")
        self.context = context
        self.width = int(width)
        self.height = int(height)
        self.packed_output = bool(packed_output)
        self.edge_aa_enabled = output_edge_aa_enabled() if edge_aa_enabled is None else bool(edge_aa_enabled)
        self.pipeline: VulkanComputePipeline | None = None
        self.descriptor_arena: VulkanDescriptorArena | None = None
        self.descriptor_sets: list[Any] = []
        self._descriptor_index = 0
        self._active_descriptor_set: Any | None = None
        self._active_push_constants: bytes | None = None
        self._edge_aa: VulkanEyeEdgeAA | None = None
        try:
            vk = context.vk
            self.pipeline = VulkanComputePipeline(
                context,
                shader_path,
                descriptor_bindings=[
                    DescriptorBinding(binding=0, descriptor_type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
                    DescriptorBinding(binding=1, descriptor_type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
                    DescriptorBinding(binding=2, descriptor_type=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE),
                    DescriptorBinding(binding=3, descriptor_type=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE),
                ],
                push_constants_size=self.PUSH_CONSTANTS_SIZE,
            )
            frame_count = max(1, int(getattr(context, "frame_context_count", 3)))
            self.descriptor_arena = VulkanDescriptorArena(
                context,
                DescriptorBudget(
                    max_sets=frame_count,
                    storage_buffers_per_set=2,
                    storage_images_per_set=2,
                ),
            )
            self.descriptor_sets = [
                self.descriptor_arena.allocate(self.pipeline.descriptor_set_layout)
                for _ in range(frame_count)
            ]
        except Exception:
            self.close()
            raise

    @property
    def group_counts(self) -> tuple[int, int, int]:
        return (
            (self.width + self.WORKGROUP_SIZE - 1) // self.WORKGROUP_SIZE,
            (self.height + self.WORKGROUP_SIZE - 1) // self.WORKGROUP_SIZE,
            1,
        )

    @property
    def input_buffer_sizes(self) -> dict[str, int]:
        pixels = self.width * self.height
        return {"rgb": pixels * 3 * 4, "depth": pixels * 4}

    def _record_active(self, command_buffer: Any) -> None:
        if self.pipeline is None or self._active_descriptor_set is None:
            raise RuntimeError("Vulkan stereo image pass is not ready")
        if self._edge_aa is not None:
            self._edge_aa.barrier(command_buffer, before_render=True)
        self.pipeline.record_dispatch(
            command_buffer,
            group_count_x=self.group_counts[0],
            group_count_y=self.group_counts[1],
            group_count_z=1,
            descriptor_set=self._active_descriptor_set,
            push_constants=self._active_push_constants,
        )
        if self._edge_aa is not None:
            self._edge_aa.record(command_buffer)

    def submit(
        self,
        rgb: Any,
        depth: Any,
        left_eye: Any,
        right_eye: Any,
        *,
        params: VulkanLayeredStereoParams,
        frame_id: int,
        config_version: int,
        ready_timeline: int | None = None,
        wait_semaphore: Any | None = None,
        signal_semaphore: Any | None = None,
    ) -> int:
        if self.pipeline is None or self.descriptor_arena is None:
            raise RuntimeError("Vulkan stereo image pass is closed")
        buffers = (rgb, depth)
        images = (left_eye, right_eye)
        expected = self.input_buffer_sizes
        for name, buffer in zip(("rgb", "depth"), buffers):
            if getattr(buffer, "context", None) is not self.context:
                raise ValueError(f"{name} buffer belongs to a different Vulkan context")
            if int(getattr(buffer, "size", 0)) < expected[name]:
                raise ValueError(f"{name} buffer is too small")
        for image in images:
            if getattr(image, "context", None) is not self.context:
                raise ValueError("stereo output image belongs to a different Vulkan context")
            expected_width = self.width * 2 if self.packed_output else self.width
            if int(getattr(image, "width", 0)) != expected_width or int(getattr(image, "height", 0)) != self.height:
                raise ValueError("stereo output image dimensions do not match")
            state = self.context.image_state(image.image)
            if state.layout != self.context.vk.VK_IMAGE_LAYOUT_GENERAL:
                raise ValueError("stereo output image must be in GENERAL layout before dispatch")

        slot = self._descriptor_index
        descriptor_set = self.descriptor_sets[slot]
        self._descriptor_index = (self._descriptor_index + 1) % len(self.descriptor_sets)
        if self.edge_aa_enabled:
            if self._edge_aa is None:
                self._edge_aa = VulkanEyeEdgeAA(
                    self.context, self.width, self.height,
                    images=True, packed_output=self.packed_output,
                )
            render_images = self._edge_aa.prepare(slot, *images)
        else:
            render_images = images
        self.descriptor_arena.update_storage_buffer(descriptor_set, 0, buffers[0])
        self.descriptor_arena.update_storage_buffer(descriptor_set, 1, buffers[1])
        self.descriptor_arena.update_storage_image(descriptor_set, 2, render_images[0])
        self.descriptor_arena.update_storage_image(descriptor_set, 3, render_images[1])
        self._active_descriptor_set = descriptor_set
        self._active_push_constants = params.pack_image(
            self.width,
            self.height,
            packed_output=self.packed_output,
            # Visibility is resolved once. Anti-alias the finished eye pixels.
            edge_aa_enabled=False,
        )
        submit_kwargs = {}
        if ready_timeline is not None:
            submit_kwargs["wait_for_timeline"] = int(ready_timeline)
        if wait_semaphore is not None:
            submit_kwargs["wait_semaphore"] = wait_semaphore
        if signal_semaphore is not None:
            submit_kwargs["signal_semaphore"] = signal_semaphore
        return self.context.submit_on("compute", self._record_active, **submit_kwargs)

    def close(self) -> None:
        if self._edge_aa is not None:
            self._edge_aa.close()
        self._edge_aa = None
        if self.pipeline is not None:
            self.pipeline.close()
        if self.descriptor_arena is not None:
            self.descriptor_arena.close()
        self.pipeline = None
        self.descriptor_arena = None
        self.descriptor_sets = []
        self._active_descriptor_set = None
        self._active_push_constants = None

    def __enter__(self) -> "VulkanStereoImagePass":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
