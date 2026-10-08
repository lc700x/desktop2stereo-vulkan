from __future__ import annotations

import struct
from dataclasses import dataclass
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

VULKAN_HOLE_FILL_BALANCED = 0
VULKAN_HOLE_FILL_QUALITY = 1
VULKAN_HOLE_FILL_NONE = 2


def resolve_vulkan_hole_fill_mode(hole_fill: object, hole_fill_mode: object) -> int:
    normalized_fill = str(hole_fill or "edge_aware").strip().lower()
    normalized_mode = str(hole_fill_mode or "balanced").strip().lower()
    if normalized_fill in {"none", "off", "disabled"} or normalized_mode in {
        "none",
        "off",
        "disabled",
    }:
        return VULKAN_HOLE_FILL_NONE
    if normalized_mode in {
        "quality",
        "content_aware",
        "directional",
    }:
        return VULKAN_HOLE_FILL_QUALITY
    return VULKAN_HOLE_FILL_BALANCED


def vulkan_hole_fill_backend_name(mode: int) -> str:
    if int(mode) == VULKAN_HOLE_FILL_NONE:
        return "none"
    if int(mode) == VULKAN_HOLE_FILL_QUALITY:
        return "vulkan_directional_content_aware_radius3"
    return "vulkan_balanced"


def resolve_vulkan_hole_fill_parameters(
    mode: int,
    *,
    fill_radius: object,
    fill_strength: object,
) -> tuple[int, float]:
    if int(mode) == VULKAN_HOLE_FILL_NONE:
        return 0, 0.0
    if int(mode) == VULKAN_HOLE_FILL_QUALITY:
        return 3, 1.0
    return max(0, min(3, int(fill_radius))), max(0.0, float(fill_strength))


@dataclass(frozen=True, slots=True)
class VulkanStereoFusedParams:
    """Push constants matching d2s_stereo_fused.comp."""

    depth_strength: float = 2.0
    max_disparity_px: float = 96.0
    convergence: float = 0.0
    edge_threshold: float = 0.03
    fill_strength: float = 0.60
    fill_radius: int = 1
    mask_feather_radius: int = 3
    symmetric: bool = True
    hole_fill_mode: int = VULKAN_HOLE_FILL_BALANCED

    def pack(self, width: int, height: int) -> bytes:
        if int(width) < 1 or int(height) < 1:
            raise ValueError("Vulkan stereo dimensions must be positive")
        if int(self.fill_radius) < 0 or int(self.mask_feather_radius) < 0:
            raise ValueError("Vulkan stereo fill radii must be non-negative")
        return struct.pack(
            "<IIfffffIIII",
            int(width),
            int(height),
            float(self.depth_strength),
            float(self.max_disparity_px),
            float(self.convergence),
            float(self.edge_threshold),
            float(self.fill_strength),
            int(self.fill_radius),
            int(self.mask_feather_radius),
            1 if self.symmetric else 0,
            int(self.hole_fill_mode),
        )


@dataclass(frozen=True, slots=True)
class VulkanLayeredStereoParams:
    """Push constants matching d2s_stereo_layered.comp."""

    depth_strength: float = 2.0
    max_disparity_px: float = 96.0
    convergence: float = 0.0
    edge_threshold: float = 0.04
    fill_strength: float = 1.0
    fill_radius: int = 3
    mask_feather_radius: int = 3
    symmetric: bool = True
    layers: int = 2
    softness: float = 0.08
    foreground_scale: float = 1.0
    midground_scale: float = 1.0
    background_scale: float = 1.0
    edge_dilation: int = 2
    screen_edge_suppression: int = 0
    hole_fill_mode: int = 0
    occlusion_enabled: bool = True

    def _pack_base(self, width: int, height: int) -> bytes:
        if int(width) < 1 or int(height) < 1:
            raise ValueError("Vulkan stereo dimensions must be positive")
        if int(self.layers) < 1 or int(self.layers) > 4:
            raise ValueError("Vulkan layered stereo supports one to four layers")
        if min(int(self.fill_radius), int(self.mask_feather_radius), int(self.edge_dilation)) < 0:
            raise ValueError("Vulkan layered stereo radii must be non-negative")
        return struct.pack(
            "<IIfffffIIIIffffIIII",
            int(width),
            int(height),
            float(self.depth_strength),
            float(self.max_disparity_px),
            float(self.convergence),
            float(self.edge_threshold),
            float(self.fill_strength),
            int(self.fill_radius),
            int(self.mask_feather_radius),
            1 if self.symmetric else 0,
            int(self.layers),
            float(self.softness),
            float(self.foreground_scale),
            float(self.midground_scale),
            float(self.background_scale),
            int(self.edge_dilation),
            int(self.screen_edge_suppression),
            int(self.hole_fill_mode),
            1 if self.occlusion_enabled else 0,
        )

    def pack(self, width: int, height: int, *, edge_aa_enabled: bool | None = None) -> bytes:
        if edge_aa_enabled is None:
            edge_aa_enabled = output_edge_aa_enabled()
        return self._pack_base(width, height) + struct.pack(
            "<I", 1 if edge_aa_enabled else 0
        )

    def pack_image(
        self,
        width: int,
        height: int,
        *,
        packed_output: bool = False,
        edge_aa_enabled: bool | None = None,
    ) -> bytes:
        """Pack image output flags after the shared layered parameters."""
        if edge_aa_enabled is None:
            edge_aa_enabled = output_edge_aa_enabled()
        return self._pack_base(width, height) + struct.pack(
            "<II", 1 if packed_output else 0, 1 if edge_aa_enabled else 0
        )


class VulkanStereoFusedPass:
    """Single-dispatch baseline stereo synthesis for Vulkan Compute.

    The pass consumes planar RGB and depth float buffers and writes planar left/right
    eyes plus an occlusion mask. It intentionally does not run depth inference.
    """

    WORKGROUP_SIZE = 16
    PUSH_CONSTANTS_SIZE = 44
    BUFFER_COUNT = 5

    def __init__(
        self,
        context: Any,
        *,
        width: int,
        height: int,
        shader_path: str | Path = Path(__file__).resolve().parents[1] / "shaders" / "d2s_stereo_fused.spv",
    ) -> None:
        if int(width) < 1 or int(height) < 1:
            raise ValueError("Vulkan stereo dimensions must be positive")
        self.context = context
        self.width = int(width)
        self.height = int(height)
        self.shader_path = Path(shader_path)
        self.pipeline: VulkanComputePipeline | None = None
        self.descriptor_arena: VulkanDescriptorArena | None = None
        self.descriptor_sets: list[Any] = []
        self._descriptor_index = 0
        self._active_descriptor_set: Any | None = None
        self._active_push_constants: bytes | None = None
        self._edge_aa: VulkanEyeEdgeAA | None = None
        self._active_edge_aa = False
        storage_buffer = context.vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
        try:
            bindings = [
                DescriptorBinding(binding=index, descriptor_type=storage_buffer)
                for index in range(self.BUFFER_COUNT)
            ]
            self.pipeline = VulkanComputePipeline(
                context,
                self.shader_path,
                descriptor_bindings=bindings,
                push_constants_size=self.PUSH_CONSTANTS_SIZE,
            )
            frame_count = max(1, int(getattr(context, "frame_context_count", 3)))
            self.descriptor_arena = VulkanDescriptorArena(
                context,
                DescriptorBudget(
                    max_sets=frame_count,
                    storage_buffers_per_set=self.BUFFER_COUNT,
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
    def buffer_sizes(self) -> dict[str, int]:
        pixels = self.width * self.height
        return {
            "rgb": pixels * 3 * 4,
            "depth": pixels * 4,
            "left_eye": pixels * 3 * 4,
            "right_eye": pixels * 3 * 4,
            "occlusion_mask": pixels * 4,
        }

    def _validate_buffers(self, buffers: tuple[Any, ...]) -> None:
        expected = self.buffer_sizes
        names = tuple(expected)
        if len(buffers) != len(names):
            raise ValueError(f"Vulkan stereo pass requires {len(names)} storage buffers")
        for name, buffer in zip(names, buffers):
            if getattr(buffer, "context", None) is not self.context:
                raise ValueError(f"{name} buffer belongs to a different Vulkan context")
            if int(getattr(buffer, "size", 0)) < expected[name]:
                raise ValueError(
                    f"{name} buffer is too small: {buffer.size} < {expected[name]}"
                )

    def _record_active(self, command_buffer: Any) -> None:
        if self.pipeline is None or self._active_descriptor_set is None:
            raise RuntimeError("Vulkan stereo pass is not ready")
        if self._active_edge_aa:
            self._edge_aa.barrier(command_buffer, before_render=True)
        self.pipeline.record_dispatch(
            command_buffer,
            group_count_x=self.group_counts[0],
            group_count_y=self.group_counts[1],
            group_count_z=1,
            descriptor_set=self._active_descriptor_set,
            push_constants=self._active_push_constants,
        )
        if self._active_edge_aa:
            self._edge_aa.record(command_buffer)

    def submit(
        self,
        rgb: Any,
        depth: Any,
        left_eye: Any,
        right_eye: Any,
        occlusion_mask: Any,
        *,
        params: VulkanStereoFusedParams | None = None,
        frame_id: int,
        config_version: int,
        ready_timeline: int | None = None,
        apply_edge_aa: bool = False,
    ) -> int:
        if self.descriptor_arena is None or self.pipeline is None:
            raise RuntimeError("Vulkan stereo pass is closed")
        buffers = (rgb, depth, left_eye, right_eye, occlusion_mask)
        self._validate_buffers(buffers)
        slot = self._descriptor_index
        descriptor_set = self.descriptor_sets[slot]
        self._descriptor_index = (self._descriptor_index + 1) % len(self.descriptor_sets)
        self._active_edge_aa = bool(apply_edge_aa)
        if self._active_edge_aa:
            if self._edge_aa is None:
                self._edge_aa = VulkanEyeEdgeAA(self.context, self.width, self.height, images=False)
            scratch = self._edge_aa.prepare(slot, left_eye, right_eye)
            render_buffers = (rgb, depth, *scratch, occlusion_mask)
        else:
            render_buffers = buffers
        for binding, buffer in enumerate(render_buffers):
            self.descriptor_arena.update_storage_buffer(descriptor_set, binding, buffer)
        self._active_descriptor_set = descriptor_set
        effective_params = params or VulkanStereoFusedParams()
        # Generic synthesis anti-aliases completed full eyes before SBS
        # packing. Standalone eye consumers can opt into this GPU pass.
        if isinstance(effective_params, VulkanLayeredStereoParams):
            self._active_push_constants = effective_params.pack(
                self.width, self.height, edge_aa_enabled=False)
        else:
            self._active_push_constants = effective_params.pack(self.width, self.height)
        submit_kwargs = {}
        if ready_timeline is not None:
            submit_kwargs["wait_for_timeline"] = int(ready_timeline)
        return self.context.submit_on("compute", self._record_active, **submit_kwargs)

    def close(self) -> None:
        if self._edge_aa is not None:
            self._edge_aa.close()
        self._edge_aa = None
        self._active_edge_aa = False
        if self.pipeline is not None:
            self.pipeline.close()
        if self.descriptor_arena is not None:
            self.descriptor_arena.close()
        self.pipeline = None
        self.descriptor_arena = None
        self.descriptor_sets = []
        self._active_descriptor_set = None
        self._active_push_constants = None

    def __enter__(self) -> "VulkanStereoFusedPass":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class VulkanLayeredStereoPass(VulkanStereoFusedPass):
    """Layered stereo synthesis pass for quality_4k and hq_4k modes."""

    PUSH_CONSTANTS_SIZE = 80

    def __init__(
        self,
        context: Any,
        *,
        width: int,
        height: int,
        shader_path: str | Path = Path(__file__).resolve().parents[1] / "shaders" / "d2s_stereo_layered.spv",
    ) -> None:
        super().__init__(context, width=width, height=height, shader_path=shader_path)

    def submit(
        self,
        rgb: Any,
        depth: Any,
        left_eye: Any,
        right_eye: Any,
        occlusion_mask: Any,
        *,
        params: VulkanLayeredStereoParams,
        frame_id: int,
        config_version: int,
        ready_timeline: int | None = None,
        apply_edge_aa: bool = False,
    ) -> int:
        return super().submit(
            rgb,
            depth,
            left_eye,
            right_eye,
            occlusion_mask,
            params=params,  # type: ignore[arg-type]
            frame_id=frame_id,
            config_version=config_version,
            ready_timeline=ready_timeline,
            apply_edge_aa=apply_edge_aa,
        )
