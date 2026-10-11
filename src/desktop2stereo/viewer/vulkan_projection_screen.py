from __future__ import annotations

import os
from pathlib import Path
import math
import struct
from typing import Any

from viewer.vulkan_compute_pipeline import read_spirv_words
from viewer.vulkan_context import ImageState
from viewer.vulkan_descriptors import (
    DescriptorBinding,
    VulkanStorageBuffer,
    create_descriptor_set_layout,
)
from viewer.vulkan_resources import VulkanTransientImage


class VulkanProjectionScreenPass:
    """Rasterize the world-space screen directly into one projection layer."""

    _SEGMENTS = 48
    _VERTEX_COUNT = (_SEGMENTS + 1) * 2
    _GLOW_SEGMENTS = 64
    _GLOW_SHELL_SEGMENTS = 48
    _GLOW_SHELL_RADIAL_SEGMENTS = 24
    _GLOW_VERTEX_COUNT = (_GLOW_SEGMENTS + 1) * 2
    _VEIL_FLAT_VERTEX_COUNT = 4 * 8 * 8 * 6
    _VEIL_CURVED_VERTEX_COUNT = (_GLOW_SEGMENTS * 2 + 2) * 6
    _SURROUND_VERTEX_COUNT = (
        4 * _GLOW_SHELL_RADIAL_SEGMENTS * _GLOW_SHELL_SEGMENTS * 6
    )
    _LASER_VERTEX_COUNT = 12
    _LASER_PARAM_SIZE = 80
    _CONTROLLER_PROXY_VERTEX_COUNT = 96
    _CONTROLLER_OVERLAY_PARAM_SIZE = 288
    _PUSH_CONSTANT_SIZE = 128
    _DESCRIPTOR_COUNT = 6
    _QUALITY_SLOT_COUNT = 3
    _DEFAULT_MIP_LOD_BIAS = -0.35
    _DEFAULT_MAX_MIP_LOD = 0.35
    _DEFAULT_RCAS_SHARPNESS = 0.5

    def __init__(
        self,
        context: Any,
        target_format: int,
        *,
        enable_panorama: bool = False,
        panorama_required: bool = False,
    ) -> None:
        self.context = context
        self.vk = context.vk
        self.target_format = int(target_format)
        self.panorama_enabled = bool(enable_panorama)
        self.panorama_required = bool(panorama_required and enable_panorama)
        self.min_mip_lod = self._min_mip_lod_from_env()
        self.mip_lod_bias = self._mip_lod_bias_from_env()
        self.max_mip_lod = self._max_mip_lod_from_env()
        self.rcas_sharpness = self._rcas_sharpness_from_env()
        self.anisotropy_enabled = bool(
            getattr(
                getattr(context, "device_info", None),
                "sampler_anisotropy_enabled",
                False,
            )
        )
        self.max_anisotropy = self._max_supported_anisotropy()
        (
            self.min_mip_lod,
            self.max_mip_lod,
            self.mip_lod_bias,
            self.rcas_sharpness,
        ) = self._normalize_sampling_config(
            self.min_mip_lod,
            self.max_mip_lod,
            self.mip_lod_bias,
            self.rcas_sharpness,
        )
        self.shader_modules: list[Any] = []
        self.descriptor_set_layout = None
        self.descriptor_pool = None
        self.descriptor_sets: list[Any] = []
        self.descriptor_timelines = [0] * self._DESCRIPTOR_COUNT
        self.rcas_descriptor_sets: list[Any] = []
        self.rcas_descriptor_timelines = [0] * self._DESCRIPTOR_COUNT
        self.quality_descriptor_sets: list[Any] = []
        self.quality_descriptor_timelines = [0] * self._DESCRIPTOR_COUNT
        self.hdr_descriptor_sets: list[Any] = []
        self.hdr_descriptor_timelines = [0] * self._DESCRIPTOR_COUNT
        self.glow_descriptor_sets: list[Any] = []
        self.glow_descriptor_timelines = [0] * self._DESCRIPTOR_COUNT
        self.glow_param_buffers: list[VulkanStorageBuffer] = []
        self.screen_crop_buffers: list[VulkanStorageBuffer] = []
        self.laser_descriptor_set_layout = None
        self.laser_descriptor_pool = None
        self.laser_descriptor_sets: list[Any] = []
        self.laser_descriptor_timelines = [0] * self._DESCRIPTOR_COUNT
        self.laser_param_buffers: list[VulkanStorageBuffer] = []
        self.quality_slot_timelines = [0] * self._QUALITY_SLOT_COUNT
        self.quality_images: dict[tuple[int, int, int], list[VulkanTransientImage]] = {}
        self.mip_slot_timelines = [0] * self._QUALITY_SLOT_COUNT
        self.mip_images: dict[tuple[int, int, int], list[VulkanTransientImage]] = {}
        self._quality_chain_oom_active: bool = False
        self._max_quality_pixels: int = 16 * 1024 * 1024
        self._mip_recording_templates: dict[
            tuple[int, int, int, int, int], dict[str, Any]
        ] = {}
        self._mip_template_hits = 0
        self._mip_template_misses = 0
        self._render_pass_recording_templates: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._render_pass_template_hits = 0
        self._render_pass_template_misses = 0
        self._image_barrier_templates: dict[tuple[int, ...], list[Any]] = {}
        self._image_barrier_template_hits = 0
        self._image_barrier_template_misses = 0
        self.last_submit_profile: dict[str, float] = {}
        self.sampler = None
        self._retired_samplers: list[tuple[Any, int]] = []
        self._last_submit_timeline = 0
        self.render_pass = None
        self.overlay_render_pass = None
        self.pipeline_layout = None
        self.pipeline = None
        self.glow_pipeline = None
        self.veil_pipeline = None
        self.surround_pipeline = None
        self.laser_pipeline = None
        self.controller_proxy_pipeline = None
        self.glow_pipeline_layout = None
        self.laser_pipeline_layout = None
        self.glow_descriptor_set_layout = None
        self.glow_descriptor_pool = None
        self.rcas_pipeline_layout = None
        self.rcas_pipeline = None
        self.copy_pipeline_layout = None
        self.copy_pipeline = None
        self.quality_pipeline_layout = None
        self.quality_pipeline = None
        self.hdr_pipeline_layout = None
        self.hdr_pipeline = None
        self.panorama_pipeline_layout = None
        self.panorama_pipeline = None
        self.image_views: dict[tuple[int, int], Any] = {}
        self.framebuffers: dict[tuple[int, int, int, int], Any] = {}
        self.overlay_framebuffers: dict[tuple[int, int, int, int], Any] = {}
        self.creation_stage = "initializing"
        try:
            self._create()
            self.creation_stage = "ready"
        except Exception as exc:
            failed_stage = self.creation_stage
            self.close()
            raise RuntimeError(
                "Vulkan projection pass creation failed: "
                f"stage={failed_stage} error={type(exc).__name__}: {exc}"
            ) from exc

    @classmethod
    def _min_mip_lod_from_env(cls) -> float:
        raw_value = os.environ.get("D2S_VULKAN_PROJECTION_MIN_LOD", "")
        if not raw_value.strip():
            return 0.0
        try:
            return max(0.0, min(16.0, float(raw_value)))
        except ValueError:
            return 0.0

    @classmethod
    def _mip_lod_bias_from_env(cls) -> float:
        raw_value = os.environ.get("D2S_VULKAN_PROJECTION_MIP_LOD_BIAS", "")
        if not raw_value.strip():
            return cls._DEFAULT_MIP_LOD_BIAS
        try:
            return max(-1.5, min(0.0, float(raw_value)))
        except ValueError:
            return cls._DEFAULT_MIP_LOD_BIAS

    @classmethod
    def _rcas_sharpness_from_env(cls) -> float:
        raw_value = os.environ.get("D2S_VULKAN_PROJECTION_RCAS_SHARPNESS", "")
        if not raw_value.strip():
            return cls._DEFAULT_RCAS_SHARPNESS
        try:
            return max(0.0, min(1.0, float(raw_value)))
        except ValueError:
            return cls._DEFAULT_RCAS_SHARPNESS

    @classmethod
    def _max_mip_lod_from_env(cls) -> float:
        raw_value = os.environ.get("D2S_VULKAN_PROJECTION_MAX_LOD", "")
        if not raw_value.strip():
            return cls._DEFAULT_MAX_MIP_LOD
        try:
            return max(0.0, min(16.0, float(raw_value)))
        except ValueError:
            return cls._DEFAULT_MAX_MIP_LOD

    @staticmethod
    def _normalize_sampling_config(min_lod, max_lod, mip_lod_bias, rcas_sharpness) -> tuple[float, float, float, float]:
        minimum = max(0.0, min(16.0, float(min_lod)))
        maximum = max(0.0, min(16.0, float(max_lod)))
        if maximum < minimum:
            maximum = minimum
        bias = max(-1.5, min(0.0, float(mip_lod_bias)))
        sharpness = max(0.0, min(1.0, float(rcas_sharpness)))
        return minimum, maximum, bias, sharpness

    def set_sampling_config(self, *, min_lod, max_lod, mip_lod_bias, rcas_sharpness) -> bool:
        """Apply projection sampling on the presenter thread."""
        values = self._normalize_sampling_config(min_lod, max_lod, mip_lod_bias, rcas_sharpness)
        minimum, maximum, bias, sharpness = values
        changed = values != (self.min_mip_lod, self.max_mip_lod, self.mip_lod_bias, self.rcas_sharpness)
        self.rcas_sharpness = sharpness
        if not changed:
            self._collect_retired_samplers()
            return False
        if (minimum, maximum, bias) != (self.min_mip_lod, self.max_mip_lod, self.mip_lod_bias):
            old_sampler = self.sampler
            self.min_mip_lod, self.max_mip_lod, self.mip_lod_bias = minimum, maximum, bias
            self.sampler = self._create_sampler()
            if old_sampler is not None:
                self._retired_samplers.append((old_sampler, int(getattr(self, "_last_submit_timeline", 0))))
        self._collect_retired_samplers()
        return True

    def _create_sampler(self):
        vk = self.vk
        return vk.vkCreateSampler(
            self.context.device,
            vk.VkSamplerCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                magFilter=vk.VK_FILTER_LINEAR,
                minFilter=vk.VK_FILTER_LINEAR,
                mipmapMode=vk.VK_SAMPLER_MIPMAP_MODE_LINEAR,
                addressModeU=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                addressModeV=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                addressModeW=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                anisotropyEnable=vk.VK_TRUE if self.anisotropy_enabled else vk.VK_FALSE,
                maxAnisotropy=self.max_anisotropy if self.anisotropy_enabled else 1.0,
                mipLodBias=self.mip_lod_bias,
                minLod=self.min_mip_lod,
                maxLod=self.max_mip_lod,
            ),
            None,
        )

    def _max_supported_anisotropy(self) -> float:
        if not self.anisotropy_enabled:
            return 1.0
        try:
            properties = self.vk.vkGetPhysicalDeviceProperties(self.context.physical_device)
            limits = getattr(properties, "limits", None)
            supported = float(getattr(limits, "maxSamplerAnisotropy", 1.0))
            return max(1.0, min(8.0, supported))
        except Exception:
            self.anisotropy_enabled = False
            return 1.0

    def _collect_retired_samplers(self) -> None:
        completed = self.context.completed_timeline_value()
        if completed is None:
            return
        pending = []
        for sampler, timeline in self._retired_samplers:
            if int(timeline) <= int(completed):
                self.vk.vkDestroySampler(self.context.device, sampler, None)
            else:
                pending.append((sampler, timeline))
        self._retired_samplers = pending

    def _create_shader_module(self, path: Path) -> Any:
        words = read_spirv_words(path)
        payload = struct.pack(f"<{len(words)}I", *words)
        module = self.vk.vkCreateShaderModule(
            self.context.device,
            self.vk.VkShaderModuleCreateInfo(
                sType=self.vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                codeSize=len(payload),
                pCode=payload,
            ),
            None,
        )
        self.shader_modules.append(module)
        return module

    def _create(self) -> None:
        vk = self.vk
        shader_root = Path(__file__).resolve().parents[1] / "shaders"
        self.creation_stage = "create_shader_modules"
        vertex_module = self._create_shader_module(
            shader_root / "d2s_projection_screen_vert.spv"
        )
        fragment_module = self._create_shader_module(
            shader_root / "d2s_projection_screen_frag.spv"
        )
        glow_fragment_module = self._create_shader_module(
            shader_root / "d2s_projection_glow_frag.spv"
        )
        glow_vertex_module = self._create_shader_module(
            shader_root / "d2s_projection_glow_vert.spv"
        )
        laser_vertex_module = self._create_shader_module(
            shader_root / "d2s_projection_laser_vert.spv"
        )
        laser_fragment_module = self._create_shader_module(
            shader_root / "d2s_projection_laser_frag.spv"
        )
        controller_proxy_vertex_module = self._create_shader_module(
            shader_root / "d2s_projection_controller_proxy_vert.spv"
        )
        controller_proxy_fragment_module = self._create_shader_module(
            shader_root / "d2s_projection_controller_proxy_frag.spv"
        )
        rcas_vertex_module = self._create_shader_module(
            shader_root / "d2s_projection_rcas_vert.spv"
        )
        rcas_fragment_module = self._create_shader_module(
            shader_root / "d2s_projection_rcas_frag.spv"
        )
        copy_fragment_module = self._create_shader_module(
            shader_root / "d2s_projection_copy_frag.spv"
        )
        quality_fragment_module = self._create_shader_module(
            shader_root / "d2s_projection_quality_frag.spv"
        )
        hdr_fragment_module = self._create_shader_module(
            shader_root / "d2s_projection_hdr_frag.spv"
        )
        panorama_vertex_module = None
        panorama_fragment_module = None
        if self.panorama_enabled:
            panorama_vertex_module = self._create_shader_module(
                shader_root / "d2s_projection_panorama_vert.spv"
            )
            panorama_fragment_module = self._create_shader_module(
                shader_root / "d2s_projection_panorama_frag.spv"
            )
        self.creation_stage = "create_sampled_descriptor_layout"
        self.descriptor_set_layout = create_descriptor_set_layout(
            self.context,
            [
                DescriptorBinding(
                    0,
                    vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    stage_flags=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                ),
                DescriptorBinding(
                    1,
                    vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    stage_flags=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                ),
            ],
        )
        self.creation_stage = "create_sampled_descriptor_pool"
        self.descriptor_pool = vk.vkCreateDescriptorPool(
            self.context.device,
            vk.VkDescriptorPoolCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                maxSets=self._DESCRIPTOR_COUNT * 4,
                poolSizeCount=2,
                pPoolSizes=[
                    vk.VkDescriptorPoolSize(
                        type=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        descriptorCount=self._DESCRIPTOR_COUNT * 4,
                    ),
                    vk.VkDescriptorPoolSize(
                        type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        descriptorCount=self._DESCRIPTOR_COUNT * 4,
                    ),
                ],
            ),
            None,
        )
        self.creation_stage = "allocate_sampled_descriptor_sets"
        allocated_descriptor_sets = list(
            vk.vkAllocateDescriptorSets(
                self.context.device,
                vk.VkDescriptorSetAllocateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                    descriptorPool=self.descriptor_pool,
                    descriptorSetCount=self._DESCRIPTOR_COUNT * 4,
                    pSetLayouts=[self.descriptor_set_layout] * (self._DESCRIPTOR_COUNT * 4),
                ),
            )
        )
        self.descriptor_sets = allocated_descriptor_sets[:self._DESCRIPTOR_COUNT]
        self.rcas_descriptor_sets = allocated_descriptor_sets[
            self._DESCRIPTOR_COUNT:self._DESCRIPTOR_COUNT * 2
        ]
        self.quality_descriptor_sets = allocated_descriptor_sets[
            self._DESCRIPTOR_COUNT * 2:self._DESCRIPTOR_COUNT * 3
        ]
        self.hdr_descriptor_sets = allocated_descriptor_sets[
            self._DESCRIPTOR_COUNT * 3:self._DESCRIPTOR_COUNT * 4
        ]
        self.screen_crop_buffers = [
            VulkanStorageBuffer(self.context, 16)
            for _ in range(self._DESCRIPTOR_COUNT)
        ]
        full_crop = struct.pack("<4f", 0.0, 0.0, 1.0, 1.0)
        for descriptor_set, buffer in zip(self.descriptor_sets, self.screen_crop_buffers):
            buffer.write_bytes(full_crop)
            vk.vkUpdateDescriptorSets(
                self.context.device,
                1,
                [
                    vk.VkWriteDescriptorSet(
                        sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                        dstSet=descriptor_set,
                        dstBinding=1,
                        descriptorCount=1,
                        descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        pBufferInfo=[
                            vk.VkDescriptorBufferInfo(
                                buffer=buffer.buffer, offset=0, range=16,
                            )
                        ],
                    )
                ],
                0,
                None,
            )
        self.creation_stage = "create_glow_descriptors"
        self.glow_descriptor_set_layout = create_descriptor_set_layout(
            self.context,
            [
                DescriptorBinding(
                    0,
                    vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    stage_flags=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                ),
                DescriptorBinding(
                    1,
                    vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    stage_flags=(
                        vk.VK_SHADER_STAGE_VERTEX_BIT
                        | vk.VK_SHADER_STAGE_FRAGMENT_BIT
                    ),
                ),
            ],
        )
        self.glow_descriptor_pool = vk.vkCreateDescriptorPool(
            self.context.device,
            vk.VkDescriptorPoolCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                maxSets=self._DESCRIPTOR_COUNT,
                poolSizeCount=2,
                pPoolSizes=[
                    vk.VkDescriptorPoolSize(
                        type=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        descriptorCount=self._DESCRIPTOR_COUNT,
                    ),
                    vk.VkDescriptorPoolSize(
                        type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        descriptorCount=self._DESCRIPTOR_COUNT,
                    ),
                ],
            ),
            None,
        )
        self.glow_descriptor_sets = list(
            vk.vkAllocateDescriptorSets(
                self.context.device,
                vk.VkDescriptorSetAllocateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                    descriptorPool=self.glow_descriptor_pool,
                    descriptorSetCount=self._DESCRIPTOR_COUNT,
                    pSetLayouts=[self.glow_descriptor_set_layout]
                    * self._DESCRIPTOR_COUNT,
                ),
            )
        )
        self.glow_param_buffers = [
            VulkanStorageBuffer(self.context, 96)
            for _ in range(self._DESCRIPTOR_COUNT)
        ]
        for descriptor_set, buffer in zip(
            self.glow_descriptor_sets, self.glow_param_buffers
        ):
            vk.vkUpdateDescriptorSets(
                self.context.device,
                1,
                [
                    vk.VkWriteDescriptorSet(
                        sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                        dstSet=descriptor_set,
                        dstBinding=1,
                        descriptorCount=1,
                        descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        pBufferInfo=[
                            vk.VkDescriptorBufferInfo(
                                buffer=buffer.buffer,
                                offset=0,
                                range=96,
                            )
                        ],
                    )
                ],
                0,
                None,
            )
        self.creation_stage = "create_laser_descriptors"
        self.laser_descriptor_set_layout = create_descriptor_set_layout(
            self.context,
            [
                DescriptorBinding(
                    0,
                    vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    stage_flags=(
                        vk.VK_SHADER_STAGE_VERTEX_BIT
                        | vk.VK_SHADER_STAGE_FRAGMENT_BIT
                    ),
                ),
            ],
        )
        self.laser_descriptor_pool = vk.vkCreateDescriptorPool(
            self.context.device,
            vk.VkDescriptorPoolCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                maxSets=self._DESCRIPTOR_COUNT,
                poolSizeCount=1,
                pPoolSizes=[
                    vk.VkDescriptorPoolSize(
                        type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        descriptorCount=self._DESCRIPTOR_COUNT,
                    )
                ],
            ),
            None,
        )
        self.laser_descriptor_sets = list(
            vk.vkAllocateDescriptorSets(
                self.context.device,
                vk.VkDescriptorSetAllocateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                    descriptorPool=self.laser_descriptor_pool,
                    descriptorSetCount=self._DESCRIPTOR_COUNT,
                    pSetLayouts=[self.laser_descriptor_set_layout]
                    * self._DESCRIPTOR_COUNT,
                ),
            )
        )
        self.laser_param_buffers = [
            VulkanStorageBuffer(self.context, self._CONTROLLER_OVERLAY_PARAM_SIZE)
            for _ in range(self._DESCRIPTOR_COUNT)
        ]
        for descriptor_set, buffer in zip(
            self.laser_descriptor_sets, self.laser_param_buffers
        ):
            vk.vkUpdateDescriptorSets(
                self.context.device,
                1,
                [
                    vk.VkWriteDescriptorSet(
                        sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                        dstSet=descriptor_set,
                        dstBinding=0,
                        descriptorCount=1,
                        descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        pBufferInfo=[
                            vk.VkDescriptorBufferInfo(
                                buffer=buffer.buffer,
                                offset=0,
                                range=self._CONTROLLER_OVERLAY_PARAM_SIZE,
                            )
                        ],
                    )
                ],
                0,
                None,
            )
        self.creation_stage = "create_sampler"
        self.sampler = self._create_sampler()
        attachment = vk.VkAttachmentDescription(
            format=self.target_format,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            loadOp=vk.VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=vk.VK_ATTACHMENT_STORE_OP_STORE,
            stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            finalLayout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        )
        color_reference = vk.VkAttachmentReference(
            attachment=0,
            layout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        )
        # Keep the subpass object alive across vkCreateRenderPass.  The Python
        # Vulkan binding stores pColorAttachments in an auxiliary CFFI array
        # owned by the VkSubpassDescription object; an inline temporary can be
        # collected before the native call reads that nested pointer.
        main_subpass = vk.VkSubpassDescription(
            pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
            colorAttachmentCount=1,
            pColorAttachments=[color_reference],
        )
        self.creation_stage = "create_clear_render_pass"
        self.render_pass = vk.vkCreateRenderPass(
            self.context.device,
            vk.VkRenderPassCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                attachmentCount=1,
                pAttachments=[attachment],
                subpassCount=1,
                pSubpasses=[main_subpass],
            ),
            None,
        )
        overlay_attachment = vk.VkAttachmentDescription(
            format=self.target_format,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            loadOp=vk.VK_ATTACHMENT_LOAD_OP_LOAD,
            storeOp=vk.VK_ATTACHMENT_STORE_OP_STORE,
            stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            finalLayout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        )
        overlay_subpass = vk.VkSubpassDescription(
            pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
            colorAttachmentCount=1,
            pColorAttachments=[color_reference],
        )
        self.creation_stage = "create_load_render_pass"
        self.overlay_render_pass = vk.vkCreateRenderPass(
            self.context.device,
            vk.VkRenderPassCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                attachmentCount=1, pAttachments=[overlay_attachment],
                subpassCount=1,
                pSubpasses=[overlay_subpass],
            ),
            None,
        )
        self.creation_stage = "create_screen_pipeline_layout"
        self.pipeline_layout = vk.vkCreatePipelineLayout(
            self.context.device,
            vk.VkPipelineLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                setLayoutCount=1,
                pSetLayouts=[self.descriptor_set_layout],
                pushConstantRangeCount=1,
                pPushConstantRanges=[
                    vk.VkPushConstantRange(
                        stageFlags=(
                            vk.VK_SHADER_STAGE_VERTEX_BIT
                            | vk.VK_SHADER_STAGE_FRAGMENT_BIT
                        ),
                        offset=0,
                        size=self._PUSH_CONSTANT_SIZE,
                    ),
                ],
            ),
            None,
        )
        self.creation_stage = "create_glow_pipeline_layout"
        self.glow_pipeline_layout = vk.vkCreatePipelineLayout(
            self.context.device,
            vk.VkPipelineLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                setLayoutCount=1,
                pSetLayouts=[self.glow_descriptor_set_layout],
                pushConstantRangeCount=1,
                pPushConstantRanges=[
                    vk.VkPushConstantRange(
                        stageFlags=(
                            vk.VK_SHADER_STAGE_VERTEX_BIT
                            | vk.VK_SHADER_STAGE_FRAGMENT_BIT
                        ),
                        offset=0,
                        size=self._PUSH_CONSTANT_SIZE,
                    ),
                ],
            ),
            None,
        )
        stages = [
            vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_VERTEX_BIT,
                module=vertex_module,
                pName="main",
            ),
            vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                module=fragment_module,
                pName="main",
            ),
        ]
        self.creation_stage = "create_screen_pipeline"
        self.pipeline = vk.vkCreateGraphicsPipelines(
            self.context.device,
            None,
            1,
            [
                vk.VkGraphicsPipelineCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                    stageCount=len(stages),
                    pStages=stages,
                    pVertexInputState=vk.VkPipelineVertexInputStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                    ),
                    pInputAssemblyState=vk.VkPipelineInputAssemblyStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                        topology=vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
                    ),
                    pViewportState=vk.VkPipelineViewportStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                        viewportCount=1,
                        scissorCount=1,
                    ),
                    pRasterizationState=vk.VkPipelineRasterizationStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                        polygonMode=vk.VK_POLYGON_MODE_FILL,
                        cullMode=vk.VK_CULL_MODE_NONE,
                        frontFace=vk.VK_FRONT_FACE_COUNTER_CLOCKWISE,
                        lineWidth=1.0,
                    ),
                    pMultisampleState=vk.VkPipelineMultisampleStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                        rasterizationSamples=vk.VK_SAMPLE_COUNT_1_BIT,
                    ),
                    pColorBlendState=vk.VkPipelineColorBlendStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                        attachmentCount=1,
                        pAttachments=[
                            vk.VkPipelineColorBlendAttachmentState(
                                blendEnable=vk.VK_TRUE,
                                srcColorBlendFactor=vk.VK_BLEND_FACTOR_SRC_ALPHA,
                                dstColorBlendFactor=vk.VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                                colorBlendOp=vk.VK_BLEND_OP_ADD,
                                srcAlphaBlendFactor=vk.VK_BLEND_FACTOR_ONE,
                                dstAlphaBlendFactor=vk.VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                                alphaBlendOp=vk.VK_BLEND_OP_ADD,
                                colorWriteMask=(
                                    vk.VK_COLOR_COMPONENT_R_BIT
                                    | vk.VK_COLOR_COMPONENT_G_BIT
                                    | vk.VK_COLOR_COMPONENT_B_BIT
                                    | vk.VK_COLOR_COMPONENT_A_BIT
                                ),
                            )
                        ],
                    ),
                    pDynamicState=vk.VkPipelineDynamicStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                        dynamicStateCount=2,
                        pDynamicStates=[
                            vk.VK_DYNAMIC_STATE_VIEWPORT,
                            vk.VK_DYNAMIC_STATE_SCISSOR,
                        ],
                    ),
                    layout=self.pipeline_layout,
                    renderPass=self.render_pass,
                    subpass=0,
                    basePipelineIndex=-1,
                )
            ],
            None,
        )[0]
        glow_stages = [
            vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_VERTEX_BIT,
                module=glow_vertex_module,
                pName="main",
            ),
            vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                module=glow_fragment_module,
                pName="main",
            ),
        ]
        def create_glow_pipeline(
            topology: int, *, additive: bool, maximum: bool = False
        ) -> Any:
            blend = vk.VkPipelineColorBlendAttachmentState(
                blendEnable=vk.VK_TRUE,
                srcColorBlendFactor=(
                    vk.VK_BLEND_FACTOR_ONE
                    if additive or maximum
                    else vk.VK_BLEND_FACTOR_SRC_ALPHA
                ),
                dstColorBlendFactor=(
                    vk.VK_BLEND_FACTOR_ONE
                    if additive or maximum
                    else vk.VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA
                ),
                colorBlendOp=(
                    vk.VK_BLEND_OP_MAX if maximum else vk.VK_BLEND_OP_ADD
                ),
                srcAlphaBlendFactor=(
                    vk.VK_BLEND_FACTOR_ZERO
                    if additive
                    else vk.VK_BLEND_FACTOR_ONE
                ),
                dstAlphaBlendFactor=(
                    vk.VK_BLEND_FACTOR_ONE
                    if additive
                    else vk.VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA
                ),
                alphaBlendOp=(
                    vk.VK_BLEND_OP_MAX if maximum else vk.VK_BLEND_OP_ADD
                ),
                colorWriteMask=(
                    vk.VK_COLOR_COMPONENT_R_BIT
                    | vk.VK_COLOR_COMPONENT_G_BIT
                    | vk.VK_COLOR_COMPONENT_B_BIT
                    | (0 if additive else vk.VK_COLOR_COMPONENT_A_BIT)
                ),
            )
            return vk.vkCreateGraphicsPipelines(
                self.context.device,
                None,
                1,
                [
                    vk.VkGraphicsPipelineCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                        stageCount=len(glow_stages),
                        pStages=glow_stages,
                        pVertexInputState=vk.VkPipelineVertexInputStateCreateInfo(
                            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                        ),
                        pInputAssemblyState=vk.VkPipelineInputAssemblyStateCreateInfo(
                            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                            topology=topology,
                        ),
                        pViewportState=vk.VkPipelineViewportStateCreateInfo(
                            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                            viewportCount=1,
                            scissorCount=1,
                        ),
                        pRasterizationState=vk.VkPipelineRasterizationStateCreateInfo(
                            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                            polygonMode=vk.VK_POLYGON_MODE_FILL,
                            cullMode=vk.VK_CULL_MODE_NONE,
                            frontFace=vk.VK_FRONT_FACE_COUNTER_CLOCKWISE,
                            lineWidth=1.0,
                        ),
                        pMultisampleState=vk.VkPipelineMultisampleStateCreateInfo(
                            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                            rasterizationSamples=vk.VK_SAMPLE_COUNT_1_BIT,
                        ),
                        pColorBlendState=vk.VkPipelineColorBlendStateCreateInfo(
                            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                            attachmentCount=1,
                            pAttachments=[blend],
                        ),
                        pDynamicState=vk.VkPipelineDynamicStateCreateInfo(
                            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                            dynamicStateCount=2,
                            pDynamicStates=[
                                vk.VK_DYNAMIC_STATE_VIEWPORT,
                                vk.VK_DYNAMIC_STATE_SCISSOR,
                            ],
                        ),
                        layout=self.glow_pipeline_layout,
                        renderPass=self.overlay_render_pass,
                        subpass=0,
                        basePipelineIndex=-1,
                    )
                ],
                None,
            )[0]

        self.glow_pipeline = create_glow_pipeline(
            vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, additive=False
        )
        self.veil_pipeline = create_glow_pipeline(
            vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, additive=False
        )
        self.surround_pipeline = create_glow_pipeline(
            # The four shell patches share their corner boundaries. Additive
            # blending counts those pixels twice and turns tiny coverage
            # changes into bright corner flashes as the head moves, while
            # normal replacement makes the last edge overwrite the others.
            # MAX keeps every shell opaque and selects the brighter component
            # in overlaps, avoiding both double brightness and exposed clear.
            vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            additive=False,
            maximum=True,
        )
        self.creation_stage = "create_laser_pipeline_layout"
        self.laser_pipeline_layout = vk.vkCreatePipelineLayout(
            self.context.device,
            vk.VkPipelineLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                setLayoutCount=1,
                pSetLayouts=[self.laser_descriptor_set_layout],
                pushConstantRangeCount=1,
                pPushConstantRanges=[
                    vk.VkPushConstantRange(
                        stageFlags=(
                            vk.VK_SHADER_STAGE_VERTEX_BIT
                            | vk.VK_SHADER_STAGE_FRAGMENT_BIT
                        ),
                        offset=0,
                        size=64,
                    )
                ],
            ),
            None,
        )
        laser_stages = [
            vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_VERTEX_BIT,
                module=laser_vertex_module,
                pName="main",
            ),
            vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                module=laser_fragment_module,
                pName="main",
            ),
        ]
        self.creation_stage = "create_laser_pipeline"
        self.laser_pipeline = vk.vkCreateGraphicsPipelines(
            self.context.device,
            None,
            1,
            [
                vk.VkGraphicsPipelineCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                    stageCount=len(laser_stages),
                    pStages=laser_stages,
                    pVertexInputState=vk.VkPipelineVertexInputStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                    ),
                    pInputAssemblyState=vk.VkPipelineInputAssemblyStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                        topology=vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                    ),
                    pViewportState=vk.VkPipelineViewportStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                        viewportCount=1,
                        scissorCount=1,
                    ),
                    pRasterizationState=vk.VkPipelineRasterizationStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                        polygonMode=vk.VK_POLYGON_MODE_FILL,
                        cullMode=vk.VK_CULL_MODE_NONE,
                        frontFace=vk.VK_FRONT_FACE_COUNTER_CLOCKWISE,
                        lineWidth=1.0,
                    ),
                    pMultisampleState=vk.VkPipelineMultisampleStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                        rasterizationSamples=vk.VK_SAMPLE_COUNT_1_BIT,
                    ),
                    pColorBlendState=vk.VkPipelineColorBlendStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                        attachmentCount=1,
                        pAttachments=[
                            vk.VkPipelineColorBlendAttachmentState(
                                blendEnable=vk.VK_FALSE,
                                colorWriteMask=(
                                    vk.VK_COLOR_COMPONENT_R_BIT
                                    | vk.VK_COLOR_COMPONENT_G_BIT
                                    | vk.VK_COLOR_COMPONENT_B_BIT
                                    | vk.VK_COLOR_COMPONENT_A_BIT
                                ),
                            )
                        ],
                    ),
                    pDynamicState=vk.VkPipelineDynamicStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                        dynamicStateCount=2,
                        pDynamicStates=[
                            vk.VK_DYNAMIC_STATE_VIEWPORT,
                            vk.VK_DYNAMIC_STATE_SCISSOR,
                        ],
                    ),
                    layout=self.laser_pipeline_layout,
                    renderPass=self.overlay_render_pass,
                    subpass=0,
                    basePipelineIndex=-1,
                )
            ],
            None,
        )[0]
        controller_proxy_stages = [
            vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_VERTEX_BIT,
                module=controller_proxy_vertex_module,
                pName="main",
            ),
            vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                module=controller_proxy_fragment_module,
                pName="main",
            ),
        ]
        self.creation_stage = "create_controller_proxy_pipeline"
        self.controller_proxy_pipeline = vk.vkCreateGraphicsPipelines(
            self.context.device,
            None,
            1,
            [
                vk.VkGraphicsPipelineCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                    stageCount=len(controller_proxy_stages),
                    pStages=controller_proxy_stages,
                    pVertexInputState=vk.VkPipelineVertexInputStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                    ),
                    pInputAssemblyState=vk.VkPipelineInputAssemblyStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                        topology=vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                    ),
                    pViewportState=vk.VkPipelineViewportStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                        viewportCount=1,
                        scissorCount=1,
                    ),
                    pRasterizationState=vk.VkPipelineRasterizationStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                        polygonMode=vk.VK_POLYGON_MODE_FILL,
                        cullMode=vk.VK_CULL_MODE_NONE,
                        frontFace=vk.VK_FRONT_FACE_COUNTER_CLOCKWISE,
                        lineWidth=1.0,
                    ),
                    pMultisampleState=vk.VkPipelineMultisampleStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                        rasterizationSamples=vk.VK_SAMPLE_COUNT_1_BIT,
                    ),
                    pColorBlendState=vk.VkPipelineColorBlendStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                        attachmentCount=1,
                        pAttachments=[
                            vk.VkPipelineColorBlendAttachmentState(
                                blendEnable=vk.VK_FALSE,
                                colorWriteMask=(
                                    vk.VK_COLOR_COMPONENT_R_BIT
                                    | vk.VK_COLOR_COMPONENT_G_BIT
                                    | vk.VK_COLOR_COMPONENT_B_BIT
                                    | vk.VK_COLOR_COMPONENT_A_BIT
                                ),
                            )
                        ],
                    ),
                    pDynamicState=vk.VkPipelineDynamicStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                        dynamicStateCount=2,
                        pDynamicStates=[
                            vk.VK_DYNAMIC_STATE_VIEWPORT,
                            vk.VK_DYNAMIC_STATE_SCISSOR,
                        ],
                    ),
                    layout=self.laser_pipeline_layout,
                    renderPass=self.overlay_render_pass,
                    subpass=0,
                    basePipelineIndex=-1,
                )
            ],
            None,
        )[0]
        self.creation_stage = "create_rcas_pipeline_layout"
        self.rcas_pipeline_layout = vk.vkCreatePipelineLayout(
            self.context.device,
            vk.VkPipelineLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                setLayoutCount=1,
                pSetLayouts=[self.descriptor_set_layout],
                pushConstantRangeCount=1,
                pPushConstantRanges=[
                    vk.VkPushConstantRange(
                        stageFlags=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                        offset=0,
                        size=16,
                    )
                ],
            ),
            None,
        )
        rcas_stages = [
            vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_VERTEX_BIT,
                module=rcas_vertex_module,
                pName="main",
            ),
            vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                module=rcas_fragment_module,
                pName="main",
            ),
        ]
        self.creation_stage = "create_rcas_pipeline"
        self.rcas_pipeline = vk.vkCreateGraphicsPipelines(
            self.context.device,
            None,
            1,
            [
                vk.VkGraphicsPipelineCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                    stageCount=2,
                    pStages=rcas_stages,
                    pVertexInputState=vk.VkPipelineVertexInputStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                    ),
                    pInputAssemblyState=vk.VkPipelineInputAssemblyStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                        topology=vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                    ),
                    pViewportState=vk.VkPipelineViewportStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                        viewportCount=1,
                        scissorCount=1,
                    ),
                    pRasterizationState=vk.VkPipelineRasterizationStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                        polygonMode=vk.VK_POLYGON_MODE_FILL,
                        cullMode=vk.VK_CULL_MODE_NONE,
                        frontFace=vk.VK_FRONT_FACE_COUNTER_CLOCKWISE,
                        lineWidth=1.0,
                    ),
                    pMultisampleState=vk.VkPipelineMultisampleStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                        rasterizationSamples=vk.VK_SAMPLE_COUNT_1_BIT,
                    ),
                    pColorBlendState=vk.VkPipelineColorBlendStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                        attachmentCount=1,
                        pAttachments=[
                            vk.VkPipelineColorBlendAttachmentState(
                                blendEnable=vk.VK_FALSE,
                                colorWriteMask=(
                                    vk.VK_COLOR_COMPONENT_R_BIT
                                    | vk.VK_COLOR_COMPONENT_G_BIT
                                    | vk.VK_COLOR_COMPONENT_B_BIT
                                    | vk.VK_COLOR_COMPONENT_A_BIT
                                ),
                            )
                        ],
                    ),
                    pDynamicState=vk.VkPipelineDynamicStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                        dynamicStateCount=2,
                        pDynamicStates=[
                            vk.VK_DYNAMIC_STATE_VIEWPORT,
                            vk.VK_DYNAMIC_STATE_SCISSOR,
                        ],
                    ),
                    layout=self.rcas_pipeline_layout,
                    renderPass=self.render_pass,
                    subpass=0,
                    basePipelineIndex=-1,
                )
            ],
            None,
        )[0]
        self.creation_stage = "create_copy_pipeline_layout"
        self.copy_pipeline_layout = vk.vkCreatePipelineLayout(
            self.context.device,
            vk.VkPipelineLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                setLayoutCount=1,
                pSetLayouts=[self.descriptor_set_layout],
            ),
            None,
        )
        copy_stages = [
            vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_VERTEX_BIT,
                module=rcas_vertex_module,
                pName="main",
            ),
            vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                module=copy_fragment_module,
                pName="main",
            ),
        ]
        self.creation_stage = "create_copy_pipeline"
        self.copy_pipeline = vk.vkCreateGraphicsPipelines(
            self.context.device,
            None,
            1,
            [
                vk.VkGraphicsPipelineCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                    stageCount=2,
                    pStages=copy_stages,
                    pVertexInputState=vk.VkPipelineVertexInputStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                    ),
                    pInputAssemblyState=vk.VkPipelineInputAssemblyStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                        topology=vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                    ),
                    pViewportState=vk.VkPipelineViewportStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                        viewportCount=1,
                        scissorCount=1,
                    ),
                    pRasterizationState=vk.VkPipelineRasterizationStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                        polygonMode=vk.VK_POLYGON_MODE_FILL,
                        cullMode=vk.VK_CULL_MODE_NONE,
                        frontFace=vk.VK_FRONT_FACE_COUNTER_CLOCKWISE,
                        lineWidth=1.0,
                    ),
                    pMultisampleState=vk.VkPipelineMultisampleStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                        rasterizationSamples=vk.VK_SAMPLE_COUNT_1_BIT,
                    ),
                    pColorBlendState=vk.VkPipelineColorBlendStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                        attachmentCount=1,
                        pAttachments=[
                            vk.VkPipelineColorBlendAttachmentState(
                                blendEnable=vk.VK_FALSE,
                                colorWriteMask=(
                                    vk.VK_COLOR_COMPONENT_R_BIT
                                    | vk.VK_COLOR_COMPONENT_G_BIT
                                    | vk.VK_COLOR_COMPONENT_B_BIT
                                    | vk.VK_COLOR_COMPONENT_A_BIT
                                ),
                            )
                        ],
                    ),
                    pDynamicState=vk.VkPipelineDynamicStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                        dynamicStateCount=2,
                        pDynamicStates=[
                            vk.VK_DYNAMIC_STATE_VIEWPORT,
                            vk.VK_DYNAMIC_STATE_SCISSOR,
                        ],
                    ),
                    layout=self.copy_pipeline_layout,
                    renderPass=self.render_pass,
                    subpass=0,
                    basePipelineIndex=-1,
                )
            ],
            None,
        )[0]
        self.creation_stage = "create_quality_pipeline_layout"
        self.quality_pipeline_layout = vk.vkCreatePipelineLayout(
            self.context.device,
            vk.VkPipelineLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                setLayoutCount=1,
                pSetLayouts=[self.descriptor_set_layout],
                pushConstantRangeCount=1,
                pPushConstantRanges=[vk.VkPushConstantRange(
                    stageFlags=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                    offset=0,
                    size=16,
                )],
            ),
            None,
        )
        quality_stages = [
            vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_VERTEX_BIT,
                module=rcas_vertex_module,
                pName="main",
            ),
            vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                module=quality_fragment_module,
                pName="main",
            ),
        ]
        self.creation_stage = "create_quality_pipeline"
        self.quality_pipeline = vk.vkCreateGraphicsPipelines(
            self.context.device,
            None,
            1,
            [
                vk.VkGraphicsPipelineCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                    stageCount=2,
                    pStages=quality_stages,
                    pVertexInputState=vk.VkPipelineVertexInputStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                    ),
                    pInputAssemblyState=vk.VkPipelineInputAssemblyStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                        topology=vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                    ),
                    pViewportState=vk.VkPipelineViewportStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                        viewportCount=1,
                        scissorCount=1,
                    ),
                    pRasterizationState=vk.VkPipelineRasterizationStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                        polygonMode=vk.VK_POLYGON_MODE_FILL,
                        cullMode=vk.VK_CULL_MODE_NONE,
                        frontFace=vk.VK_FRONT_FACE_COUNTER_CLOCKWISE,
                        lineWidth=1.0,
                    ),
                    pMultisampleState=vk.VkPipelineMultisampleStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                        rasterizationSamples=vk.VK_SAMPLE_COUNT_1_BIT,
                    ),
                    pColorBlendState=vk.VkPipelineColorBlendStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                        attachmentCount=1,
                        pAttachments=[vk.VkPipelineColorBlendAttachmentState(
                            blendEnable=vk.VK_FALSE,
                            colorWriteMask=(vk.VK_COLOR_COMPONENT_R_BIT | vk.VK_COLOR_COMPONENT_G_BIT | vk.VK_COLOR_COMPONENT_B_BIT | vk.VK_COLOR_COMPONENT_A_BIT),
                        )],
                    ),
                    pDynamicState=vk.VkPipelineDynamicStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                        dynamicStateCount=2,
                        pDynamicStates=[vk.VK_DYNAMIC_STATE_VIEWPORT, vk.VK_DYNAMIC_STATE_SCISSOR],
                    ),
                    layout=self.quality_pipeline_layout,
                    renderPass=self.render_pass,
                    subpass=0,
                    basePipelineIndex=-1,
                )
            ],
            None,
        )[0]
        self.creation_stage = "create_hdr_pipeline_layout"
        self.hdr_pipeline_layout = vk.vkCreatePipelineLayout(
            self.context.device,
            vk.VkPipelineLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                setLayoutCount=1,
                pSetLayouts=[self.descriptor_set_layout],
                pushConstantRangeCount=1,
                pPushConstantRanges=[vk.VkPushConstantRange(
                    stageFlags=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                    offset=0,
                    size=16,
                )],
            ),
            None,
        )
        hdr_stages = [
            vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_VERTEX_BIT,
                module=rcas_vertex_module,
                pName="main",
            ),
            vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                module=hdr_fragment_module,
                pName="main",
            ),
        ]
        self.creation_stage = "create_hdr_pipeline"
        self.hdr_pipeline = vk.vkCreateGraphicsPipelines(
            self.context.device,
            None,
            1,
            [
                vk.VkGraphicsPipelineCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                    stageCount=2,
                    pStages=hdr_stages,
                    pVertexInputState=vk.VkPipelineVertexInputStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                    ),
                    pInputAssemblyState=vk.VkPipelineInputAssemblyStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                        topology=vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                    ),
                    pViewportState=vk.VkPipelineViewportStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                        viewportCount=1,
                        scissorCount=1,
                    ),
                    pRasterizationState=vk.VkPipelineRasterizationStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                        polygonMode=vk.VK_POLYGON_MODE_FILL,
                        cullMode=vk.VK_CULL_MODE_NONE,
                        frontFace=vk.VK_FRONT_FACE_COUNTER_CLOCKWISE,
                        lineWidth=1.0,
                    ),
                    pMultisampleState=vk.VkPipelineMultisampleStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                        rasterizationSamples=vk.VK_SAMPLE_COUNT_1_BIT,
                    ),
                    pColorBlendState=vk.VkPipelineColorBlendStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                        attachmentCount=1,
                        pAttachments=[vk.VkPipelineColorBlendAttachmentState(
                            blendEnable=vk.VK_TRUE,
                            srcColorBlendFactor=vk.VK_BLEND_FACTOR_SRC_ALPHA,
                            dstColorBlendFactor=vk.VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                            colorBlendOp=vk.VK_BLEND_OP_ADD,
                            srcAlphaBlendFactor=vk.VK_BLEND_FACTOR_ONE,
                            dstAlphaBlendFactor=vk.VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                            alphaBlendOp=vk.VK_BLEND_OP_ADD,
                            colorWriteMask=(
                                vk.VK_COLOR_COMPONENT_R_BIT
                                | vk.VK_COLOR_COMPONENT_G_BIT
                                | vk.VK_COLOR_COMPONENT_B_BIT
                                | vk.VK_COLOR_COMPONENT_A_BIT
                            ),
                        )],
                    ),
                    pDynamicState=vk.VkPipelineDynamicStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                        dynamicStateCount=2,
                        pDynamicStates=[
                            vk.VK_DYNAMIC_STATE_VIEWPORT,
                            vk.VK_DYNAMIC_STATE_SCISSOR,
                        ],
                    ),
                    layout=self.hdr_pipeline_layout,
                    renderPass=self.render_pass,
                    subpass=0,
                    basePipelineIndex=-1,
                )
            ],
            None,
        )[0]
        if self.panorama_enabled:
            self._create_panorama_pipeline(
                panorama_vertex_module,
                panorama_fragment_module,
            )

    def _create_panorama_pipeline(self, vertex_module: Any, fragment_module: Any) -> None:
        """Create optional GPU panorama pass without taking down screen output."""
        vk = self.vk
        self.creation_stage = "create_panorama_pipeline_layout"
        try:
            self.panorama_pipeline_layout = vk.vkCreatePipelineLayout(
                self.context.device,
                vk.VkPipelineLayoutCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                    setLayoutCount=1,
                    pSetLayouts=[self.descriptor_set_layout],
                    pushConstantRangeCount=1,
                    pPushConstantRanges=[vk.VkPushConstantRange(
                        stageFlags=(
                            vk.VK_SHADER_STAGE_VERTEX_BIT
                            | vk.VK_SHADER_STAGE_FRAGMENT_BIT
                        ),
                        offset=0,
                        size=32,
                    )],
                ),
                None,
            )
            self.creation_stage = "create_panorama_pipeline"
            self.panorama_pipeline = vk.vkCreateGraphicsPipelines(
                self.context.device,
                None,
                1,
                [vk.VkGraphicsPipelineCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                    stageCount=2,
                    pStages=[
                        vk.VkPipelineShaderStageCreateInfo(
                            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                            stage=vk.VK_SHADER_STAGE_VERTEX_BIT,
                            module=vertex_module,
                            pName="main",
                        ),
                        vk.VkPipelineShaderStageCreateInfo(
                            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                            stage=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                            module=fragment_module,
                            pName="main",
                        ),
                    ],
                    pVertexInputState=vk.VkPipelineVertexInputStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO
                    ),
                    pInputAssemblyState=vk.VkPipelineInputAssemblyStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                        topology=vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                    ),
                    pViewportState=vk.VkPipelineViewportStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                        viewportCount=1,
                        scissorCount=1,
                    ),
                    pRasterizationState=vk.VkPipelineRasterizationStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                        polygonMode=vk.VK_POLYGON_MODE_FILL,
                        cullMode=vk.VK_CULL_MODE_NONE,
                        frontFace=vk.VK_FRONT_FACE_COUNTER_CLOCKWISE,
                        lineWidth=1.0,
                    ),
                    pMultisampleState=vk.VkPipelineMultisampleStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                        rasterizationSamples=vk.VK_SAMPLE_COUNT_1_BIT,
                    ),
                    pColorBlendState=vk.VkPipelineColorBlendStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                        attachmentCount=1,
                        pAttachments=[vk.VkPipelineColorBlendAttachmentState(
                            blendEnable=vk.VK_FALSE,
                            colorWriteMask=(
                                vk.VK_COLOR_COMPONENT_R_BIT
                                | vk.VK_COLOR_COMPONENT_G_BIT
                                | vk.VK_COLOR_COMPONENT_B_BIT
                                | vk.VK_COLOR_COMPONENT_A_BIT
                            ),
                        )],
                    ),
                    pDynamicState=vk.VkPipelineDynamicStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                        dynamicStateCount=2,
                        pDynamicStates=[
                            vk.VK_DYNAMIC_STATE_VIEWPORT,
                            vk.VK_DYNAMIC_STATE_SCISSOR,
                        ],
                    ),
                    layout=self.panorama_pipeline_layout,
                    renderPass=self.render_pass,
                    subpass=0,
                    basePipelineIndex=-1,
                )],
                None,
            )[0]
        except Exception as exc:
            if (
                bool(getattr(self.context, "device_lost", False))
                or self.panorama_required
            ):
                raise
            if self.panorama_pipeline is not None:
                try:
                    vk.vkDestroyPipeline(
                        self.context.device, self.panorama_pipeline, None
                    )
                except Exception:
                    pass
            if self.panorama_pipeline_layout is not None:
                try:
                    vk.vkDestroyPipelineLayout(
                        self.context.device,
                        self.panorama_pipeline_layout,
                        None,
                    )
                except Exception:
                    pass
            self.panorama_pipeline = None
            self.panorama_pipeline_layout = None
            self.panorama_enabled = False
            print(
                "[VulkanProjection] Optional panorama pipeline unavailable; "
                "keeping GPU screen composition active: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

    def _target_view_and_framebuffer(
        self, target: Any, array_layer: int, *, overlay: bool = False
    ) -> tuple[Any, Any]:
        vk = self.vk
        view_key = (id(target.image), int(array_layer))
        view = self.image_views.get(view_key)
        if view is None:
            view = vk.vkCreateImageView(
                self.context.device,
                vk.VkImageViewCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                    image=target.image,
                    viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
                    format=self.target_format,
                    subresourceRange=vk.VkImageSubresourceRange(
                        aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                        baseMipLevel=0,
                        levelCount=1,
                        baseArrayLayer=int(array_layer),
                        layerCount=1,
                    ),
                ),
                None,
            )
            self.image_views[view_key] = view
        framebuffer_key = (
            view_key[0], view_key[1], int(target.width), int(target.height)
        )
        framebuffer_cache = self.overlay_framebuffers if overlay else self.framebuffers
        framebuffer = framebuffer_cache.get(framebuffer_key)
        if framebuffer is None:
            framebuffer = vk.vkCreateFramebuffer(
                self.context.device,
                vk.VkFramebufferCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                    renderPass=self.overlay_render_pass if overlay else self.render_pass,
                    attachmentCount=1,
                    pAttachments=[view],
                    width=int(target.width),
                    height=int(target.height),
                    layers=1,
                ),
                None,
            )
            framebuffer_cache[framebuffer_key] = framebuffer
        return view, framebuffer

    def submit(
        self,
        source: Any,
        target: Any,
        *,
        array_layer: int,
        eye_index: int,
        frame_slot: int,
        push_constants: bytes,
        clear_color: tuple[float, float, float, float],
        wait_semaphore: Any | None,
        source_crop_uv: tuple[float, float, float, float] | None = None,
    ) -> int:
        draw = self._prepare_draw(
            source,
            target,
            array_layer=array_layer,
            eye_index=eye_index,
            frame_slot=frame_slot,
            push_constants=push_constants,
            clear_color=clear_color,
            source_crop_uv=source_crop_uv,
        )
        timeline = self.context.submit_on(
            "graphics",
            lambda command_buffer: self._record_draw(command_buffer, draw),
            wait_semaphore=wait_semaphore,
        )
        self._complete_draw(draw, timeline)
        self._last_submit_timeline = int(timeline)
        return int(timeline)

    def submit_stereo(
        self,
        draws: list[dict[str, Any]],
        *,
        load_target: bool = False,
        wait_for_timeline: int = 0,
        extra_wait_semaphores: list[Any] | tuple[Any, ...] = (),
    ) -> int:
        """Submit both eye projection draws in one graphics queue submission."""
        if len(draws) != 2:
            raise ValueError("stereo projection requires exactly two draws")
        prepared = []
        for item in draws:
            draw = self._prepare_draw(
                item["source"],
                item["target"],
                array_layer=int(item["array_layer"]),
                eye_index=int(item["eye_index"]),
                frame_slot=int(item["frame_slot"]),
                push_constants=item["push_constants"],
                clear_color=item["clear_color"],
                source_crop_uv=item.get("source_crop_uv"),
                overlay=bool(load_target),
            )
            if load_target:
                draw["target_old_layout"] = self.vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
            prepared.append(draw)
        wait_semaphores = [item["wait_semaphore"] for item in draws]
        wait_semaphores = [item for item in wait_semaphores if item is not None]
        wait_semaphores.extend(
            item for item in extra_wait_semaphores if item is not None
        )
        submit_profile: dict[str, float] = {}
        timeline = self.context.submit_on(
            "graphics",
            lambda command_buffer: [
                self._record_draw(command_buffer, draw) for draw in prepared
            ],
            wait_semaphore=wait_semaphores,
            wait_for_timeline=int(wait_for_timeline),
            on_submit_profile=submit_profile.update,
        )
        self.last_submit_profile = submit_profile
        for draw in prepared:
            self._complete_draw(draw, timeline)
        self._last_submit_timeline = int(timeline)
        return int(timeline)

    def submit_stereo_glow(
        self,
        draws: list[dict[str, Any]],
        *,
        wait_for_timeline: int,
        clear_target: bool = False,
    ) -> int:
        """Draw the legacy screen-edge Glow around the opaque SBS surface."""
        if len(draws) != 2 or self.glow_pipeline is None:
            return int(wait_for_timeline)
        prepared = []
        for item in draws:
            source = item.get("glow_source")
            push_constants = item.get("glow_push_constants")
            params = item.get("glow_params")
            mode = int(item.get("glow_mode", 0))
            if (
                source is None
                or not isinstance(push_constants, (bytes, bytearray))
                or not isinstance(params, (bytes, bytearray))
                or len(params) != 96
                or mode not in {1, 2, 3}
            ):
                return int(wait_for_timeline)
            eye_index = int(item["eye_index"])
            descriptor_index = (
                eye_index * 3 + int(item["frame_slot"])
            ) % self._DESCRIPTOR_COUNT
            glow_draw = self._prepare_draw(
                source,
                item["target"],
                array_layer=int(item["array_layer"]),
                eye_index=eye_index,
                frame_slot=int(item["frame_slot"]),
                push_constants=bytes(push_constants),
                clear_color=item["clear_color"],
                descriptor_set=self.glow_descriptor_sets[descriptor_index],
                descriptor_timelines=self.glow_descriptor_timelines,
                overlay=not clear_target,
            )
            self.glow_param_buffers[descriptor_index].write_bytes(params)
            if not clear_target:
                glow_draw["target_old_layout"] = self.vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
            glow_draw["pipeline_layout"] = self.glow_pipeline_layout
            if mode == 1:
                glow_draw["pipeline"] = self.glow_pipeline
                glow_draw["vertex_count"] = self._GLOW_VERTEX_COUNT
            elif mode == 2:
                glow_draw["pipeline"] = self.veil_pipeline
                glow_draw["vertex_count"] = (
                    self._VEIL_CURVED_VERTEX_COUNT
                    if bool(item.get("glow_curved", False))
                    else self._VEIL_FLAT_VERTEX_COUNT
                )
            else:
                glow_draw["pipeline"] = self.surround_pipeline
                glow_draw["vertex_count"] = self._SURROUND_VERTEX_COUNT
            prepared.append(glow_draw)
        timeline = self.context.submit_on(
            "graphics",
            lambda command_buffer: [
                self._record_draw(command_buffer, draw) for draw in prepared
            ],
            wait_for_timeline=int(wait_for_timeline),
        )
        for draw in prepared:
            self.glow_descriptor_timelines[draw["descriptor_index"]] = int(timeline)
        self._last_submit_timeline = int(timeline)
        return int(timeline)

    def _prepare_laser_draw(
        self,
        target: Any,
        *,
        array_layer: int,
        eye_index: int,
        frame_slot: int,
        push_constants: bytes,
        clear_color: tuple[float, float, float, float],
        clear_target: bool = False,
    ) -> dict[str, Any]:
        if self.laser_pipeline is None or len(push_constants) != 64:
            raise RuntimeError("Vulkan projection laser pass is unavailable")
        if int(target.format) != self.target_format:
            raise ValueError("projection target format changed after pipeline creation")
        descriptor_index = (
            int(eye_index) * 3 + int(frame_slot)
        ) % self._DESCRIPTOR_COUNT
        last_use = self.laser_descriptor_timelines[descriptor_index]
        if last_use:
            self.context.wait_for_timeline(last_use)
        _view, framebuffer = self._target_view_and_framebuffer(
            target, array_layer, overlay=not bool(clear_target)
        )
        return {
            "target": target,
            "array_layer": int(array_layer),
            "framebuffer": framebuffer,
            "descriptor_set": self.laser_descriptor_sets[descriptor_index],
            "payload": self.vk.ffi.new("char[]", push_constants),
            "push_constant_size": 64,
            "clear_color": clear_color,
            "descriptor_index": descriptor_index,
            "render_pass": (
                self.render_pass if clear_target else self.overlay_render_pass
            ),
            "pipeline": self.laser_pipeline,
            "pipeline_layout": self.laser_pipeline_layout,
            "vertex_count": self._LASER_VERTEX_COUNT,
            "target_old_layout": (
                self.vk.VK_IMAGE_LAYOUT_UNDEFINED
                if clear_target
                else self.vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
            ),
        }

    def submit_stereo_laser(
        self,
        draws: list[dict[str, Any]],
        *,
        wait_for_timeline: int,
    ) -> int:
        """Draw visible legacy controller laser beams over the projection screen."""
        if len(draws) != 2 or self.laser_pipeline is None:
            return int(wait_for_timeline)
        prepared = []
        for item in draws:
            params = item.get("laser_params")
            if not isinstance(params, (bytes, bytearray)) or len(params) != self._LASER_PARAM_SIZE:
                continue
            laser_draw = self._prepare_laser_draw(
                item["target"],
                array_layer=int(item["array_layer"]),
                eye_index=int(item["eye_index"]),
                frame_slot=int(item["frame_slot"]),
                push_constants=bytes(item["laser_push_constants"]),
                clear_color=item["clear_color"],
            )
            self.laser_param_buffers[laser_draw["descriptor_index"]].write_bytes(params)
            prepared.append(laser_draw)
        if not prepared:
            return int(wait_for_timeline)
        timeline = self.context.submit_on(
            "graphics",
            lambda command_buffer: [
                self._record_draw(command_buffer, draw) for draw in prepared
            ],
            wait_for_timeline=int(wait_for_timeline),
        )
        for draw in prepared:
            self.context.register_image_state(
                draw["target"].image,
                ImageState(
                    layout=self.vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    access_mask=self.vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    stage_mask=self.vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                    queue_family_index=self.context.queue_family_index,
                ),
            )
            self.laser_descriptor_timelines[draw["descriptor_index"]] = int(timeline)
        self._last_submit_timeline = int(timeline)
        return int(timeline)

    def submit_stereo_controller_proxy(
        self,
        draws: list[dict[str, Any]],
        *,
        wait_for_timeline: int,
        clear_target: bool = False,
    ) -> int:
        """Draw two local Vulkan controller cubes and beams without Filament."""
        if len(draws) != 2 or self.controller_proxy_pipeline is None:
            return int(wait_for_timeline)
        prepared = []
        for item in draws:
            params = item.get("controller_proxy_params")
            if (
                not isinstance(params, (bytes, bytearray))
                or len(params) != self._CONTROLLER_OVERLAY_PARAM_SIZE
            ):
                continue
            proxy_draw = self._prepare_laser_draw(
                item["target"],
                array_layer=int(item["array_layer"]),
                eye_index=int(item["eye_index"]),
                frame_slot=int(item["frame_slot"]),
                push_constants=bytes(item["controller_proxy_push_constants"]),
                clear_color=item["clear_color"],
                clear_target=bool(clear_target),
            )
            proxy_draw["pipeline"] = self.controller_proxy_pipeline
            proxy_draw["vertex_count"] = self._CONTROLLER_PROXY_VERTEX_COUNT
            self.laser_param_buffers[proxy_draw["descriptor_index"]].write_bytes(params)
            prepared.append(proxy_draw)
        if not prepared:
            return int(wait_for_timeline)
        timeline = self.context.submit_on(
            "graphics",
            lambda command_buffer: [
                self._record_draw(command_buffer, draw) for draw in prepared
            ],
            wait_for_timeline=int(wait_for_timeline),
        )
        for draw in prepared:
            self.context.register_image_state(
                draw["target"].image,
                ImageState(
                    layout=self.vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    access_mask=self.vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    stage_mask=self.vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                    queue_family_index=self.context.queue_family_index,
                ),
            )
            self.laser_descriptor_timelines[draw["descriptor_index"]] = int(timeline)
        self._last_submit_timeline = int(timeline)
        return int(timeline)

    def try_submit_stereo_rcas(self, draws: list[dict[str, Any]]) -> int | None:
        """Submit filter and RCAS in one queue submission, or skip without waiting."""
        if len(draws) != 2 or self.rcas_sharpness <= 0.0 or self.rcas_pipeline is None:
            return None
        frame_slot = int(draws[0]["frame_slot"]) % self._QUALITY_SLOT_COUNT
        completed = self.context.completed_timeline_value()
        if completed is None:
            return None
        descriptor_indices = [
            (int(item["eye_index"]) * 3 + int(item["frame_slot"]))
            % self._DESCRIPTOR_COUNT
            for item in draws
        ]
        in_use = [self.quality_slot_timelines[frame_slot]]
        in_use.extend(self.descriptor_timelines[index] for index in descriptor_indices)
        in_use.extend(self.rcas_descriptor_timelines[index] for index in descriptor_indices)
        if any(int(value) > int(completed) for value in in_use):
            return None
        target = draws[0]["target"]
        key = (int(target.width), int(target.height), int(target.format))
        images = self.quality_images.get(key)
        if images is None:
            images = [
                VulkanTransientImage(
                    self.context,
                    key[0],
                    key[1],
                    format=key[2],
                    label=f"projection-rcas-eye{eye}-slot{slot}",
                )
                for eye in range(2)
                for slot in range(self._QUALITY_SLOT_COUNT)
            ]
            self.quality_images[key] = images
        filtered_draws = []
        rcas_draws = []
        for item in draws:
            eye_index = int(item["eye_index"])
            scratch = images[eye_index * self._QUALITY_SLOT_COUNT + frame_slot].resource
            if scratch is None:
                return None
            filtered = self._prepare_draw(
                item["source"],
                scratch,
                array_layer=0,
                eye_index=eye_index,
                frame_slot=int(item["frame_slot"]),
                push_constants=item["push_constants"],
                clear_color=item["clear_color"],
                source_crop_uv=item.get("source_crop_uv"),
            )
            filtered["target_old_layout"] = self.context.image_state(scratch.image).layout
            filtered_draws.append(filtered)
            rcas_draws.append(
                self._prepare_rcas_draw(
                    scratch,
                    item["target"],
                    array_layer=int(item["array_layer"]),
                    eye_index=eye_index,
                    frame_slot=int(item["frame_slot"]),
                    sharpness=self.rcas_sharpness,
                )
            )
        wait_semaphores = [item["wait_semaphore"] for item in draws]
        wait_semaphores = [item for item in wait_semaphores if item is not None]
        submit_profile: dict[str, float] = {}
        timeline = self.context.submit_on(
            "graphics",
            lambda command_buffer: [
                self._record_draw(command_buffer, filtered_draws[index])
                or self._record_rcas_draw(command_buffer, rcas_draws[index])
                for index in range(2)
            ],
            wait_semaphore=wait_semaphores,
            on_submit_profile=submit_profile.update,
        )
        self.last_submit_profile = submit_profile
        for filtered, rcas in zip(filtered_draws, rcas_draws):
            self._complete_draw(filtered, timeline)
            self._complete_rcas_draw(rcas, timeline)
        self.quality_slot_timelines[frame_slot] = int(timeline)
        self._last_submit_timeline = int(timeline)
        return int(timeline)

    def try_submit_stereo_mip(
        self,
        draws: list[dict[str, Any]],
        *,
        load_target: bool = False,
        wait_for_timeline: int = 0,
        extra_wait_semaphores: list[Any] | tuple[Any, ...] = (),
    ) -> int | None:
        """Generate the final per-eye mip chain, with optional RCAS on mip zero."""
        use_rcas = self.rcas_sharpness > 0.0
        if (
            len(draws) != 2
            or self.copy_pipeline is None
            or (use_rcas and self.rcas_pipeline is None)
        ):
            return None
        frame_slot = int(draws[0]["frame_slot"]) % self._QUALITY_SLOT_COUNT
        completed = self.context.completed_timeline_value()
        if completed is None:
            return None
        descriptor_indices = [
            (int(item["eye_index"]) * 3 + int(item["frame_slot"]))
            % self._DESCRIPTOR_COUNT
            for item in draws
        ]
        in_use = [self.quality_slot_timelines[frame_slot], self.mip_slot_timelines[frame_slot]]
        in_use.extend(self.descriptor_timelines[index] for index in descriptor_indices)
        in_use.extend(self.rcas_descriptor_timelines[index] for index in descriptor_indices)
        in_use.extend(self.quality_descriptor_timelines[index] for index in descriptor_indices)
        if any(int(value) > int(completed) for value in in_use):
            return None
        source = draws[0]["source"]
        if any(
            int(item["source"].width) != int(source.width)
            or int(item["source"].height) != int(source.height)
            or int(item["target"].format) != int(draws[0]["target"].format)
            for item in draws[1:]
        ):
            return None
        if not self._supports_linear_blit(int(draws[0]["target"].format)):
            return None
        key = (int(source.width), int(source.height), int(draws[0]["target"].format))
        quality_images = None
        if use_rcas:
            quality_images = self.quality_images.get(key)
            if quality_images is None:
                quality_images = [
                    VulkanTransientImage(
                        self.context,
                        key[0],
                        key[1],
                        format=key[2],
                        label=f"projection-native-rcas-eye{eye}-slot{slot}",
                    )
                    for eye in range(2)
                    for slot in range(self._QUALITY_SLOT_COUNT)
                ]
                self.quality_images[key] = quality_images
        images = self.mip_images.get(key)
        if images is None:
            if self.mip_images:
                if any(int(value) > int(completed) for value in self.mip_slot_timelines):
                    return None
                for stale_images in self.mip_images.values():
                    for stale_image in stale_images:
                        stale_image.close()
                self.mip_images.clear()
            mip_levels = int(math.floor(math.log2(max(key[0], key[1])))) + 1
            images = [
                VulkanTransientImage(
                    self.context,
                    key[0],
                    key[1],
                    format=key[2],
                    label=f"projection-mip-eye{eye}-slot{slot}",
                    mip_levels=mip_levels,
                )
                for eye in range(2)
                for slot in range(self._QUALITY_SLOT_COUNT)
            ]
            self.mip_images[key] = images
        copy_draws = []
        rcas_draws = []
        screen_draws = []
        for item, descriptor_index in zip(draws, descriptor_indices):
            eye_index = int(item["eye_index"])
            mip = images[eye_index * self._QUALITY_SLOT_COUNT + frame_slot].resource
            if mip is None:
                return None
            copy_target = mip
            if use_rcas:
                quality = quality_images[eye_index * self._QUALITY_SLOT_COUNT + frame_slot].resource
                if quality is None:
                    return None
                copy_target = quality
            copy_draw = self._prepare_copy_draw(item["source"], copy_target, descriptor_index)
            copy_draws.append(copy_draw)
            if use_rcas:
                rcas = self._prepare_rcas_draw(
                    copy_target,
                    mip,
                    array_layer=0,
                    eye_index=eye_index,
                    frame_slot=int(item["frame_slot"]),
                    sharpness=self.rcas_sharpness,
                )
                rcas["target_old_layout"] = self.context.image_state(mip.image).layout
                rcas_draws.append(rcas)
            screen_draw = self._prepare_draw(
                    mip,
                    item["target"],
                    array_layer=int(item["array_layer"]),
                    eye_index=eye_index,
                    frame_slot=int(item["frame_slot"]),
                    push_constants=item["push_constants"],
                    clear_color=item["clear_color"],
                    source_crop_uv=item.get("source_crop_uv"),
                    source_ready_in_submission=True,
                    overlay=bool(load_target),
            )
            if load_target:
                screen_draw["target_old_layout"] = (
                    self.vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
                )
            screen_draws.append(screen_draw)
        wait_semaphores = [item["wait_semaphore"] for item in draws]
        wait_semaphores = [item for item in wait_semaphores if item is not None]
        wait_semaphores.extend(
            item for item in extra_wait_semaphores if item is not None
        )
        submit_profile: dict[str, float] = {}
        mip_template_hits = self._mip_template_hits
        mip_template_misses = self._mip_template_misses
        render_pass_template_hits = self._render_pass_template_hits
        render_pass_template_misses = self._render_pass_template_misses
        image_barrier_template_hits = self._image_barrier_template_hits
        image_barrier_template_misses = self._image_barrier_template_misses
        def record_stereo(command_buffer: Any) -> None:
            for index, copy_draw in enumerate(copy_draws):
                self._record_copy_draw(command_buffer, copy_draw)
                mip_draw = copy_draw
                if use_rcas:
                    mip_draw = rcas_draws[index]
                    self._record_rcas_draw(command_buffer, mip_draw)
                self._record_generate_mips(
                    command_buffer,
                    mip_draw["target"],
                    int(mip_draw["target_old_layout"]),
                )
                self._record_draw(command_buffer, screen_draws[index])

        timeline = self.context.submit_on(
            "graphics",
            record_stereo,
            wait_semaphore=wait_semaphores,
            wait_for_timeline=int(wait_for_timeline),
            on_submit_profile=submit_profile.update,
        )
        submit_profile["mip_template_hit"] = float(
            self._mip_template_hits - mip_template_hits
        )
        submit_profile["mip_template_new"] = float(
            self._mip_template_misses - mip_template_misses
        )
        submit_profile["render_pass_template_hit"] = float(
            self._render_pass_template_hits - render_pass_template_hits
        )
        submit_profile["render_pass_template_new"] = float(
            self._render_pass_template_misses - render_pass_template_misses
        )
        submit_profile["image_barrier_template_hit"] = float(
            self._image_barrier_template_hits - image_barrier_template_hits
        )
        submit_profile["image_barrier_template_new"] = float(
            self._image_barrier_template_misses - image_barrier_template_misses
        )
        self.last_submit_profile = submit_profile
        if use_rcas:
            for copy_draw, rcas_draw, screen_draw in zip(copy_draws, rcas_draws, screen_draws):
                self._complete_quality_mip_draw(copy_draw, rcas_draw, screen_draw, timeline)
        else:
            for copy_draw, screen_draw in zip(copy_draws, screen_draws):
                self._complete_mip_draw(copy_draw, screen_draw, timeline)
        self.quality_slot_timelines[frame_slot] = int(timeline)
        self.mip_slot_timelines[frame_slot] = int(timeline)
        self._last_submit_timeline = int(timeline)
        return int(timeline)

    def try_submit_stereo_quality_mip(
        self,
        draws: list[dict[str, Any]],
        *,
        mode: str,
        filter_scale: float,
        upscale_scale: float,
        quality_width: int | None = None,
        quality_height: int | None = None,
        load_target: bool = False,
        wait_for_timeline: int = 0,
        extra_wait_semaphores: list[Any] | tuple[Any, ...] = (),
    ) -> int | None:
        """Match Filament's quality chain before the final curved-screen draw."""
        if mode == "native_mip":
            return self.try_submit_stereo_mip(
                draws,
                load_target=load_target,
                wait_for_timeline=wait_for_timeline,
                extra_wait_semaphores=extra_wait_semaphores,
            )
        use_rcas = self.rcas_sharpness > 0.0
        if (
            len(draws) != 2
            or mode not in {"downsample_lanczos_rcas", "upscale_easu"}
            or self.quality_pipeline is None
            or self.copy_pipeline is None
            or (use_rcas and self.rcas_pipeline is None)
        ):
            return None
        frame_slot = int(draws[0]["frame_slot"]) % self._QUALITY_SLOT_COUNT
        completed = self.context.completed_timeline_value()
        if completed is None:
            return None
        source = draws[0]["source"]
        target_format = int(draws[0]["target"].format)
        if any(
            int(item["source"].width) != int(source.width)
            or int(item["source"].height) != int(source.height)
            or int(item["target"].format) != target_format
            for item in draws[1:]
        ) or not self._supports_linear_blit(target_format):
            return None
        scale = float(upscale_scale) if mode == "upscale_easu" else 1.0 / float(filter_scale)
        quality_width = max(
            16,
            (
                int(quality_width)
                if quality_width is not None
                else int(round(int(source.width) * scale))
            )
            & ~1,
        )
        quality_height = max(
            16,
            (
                int(quality_height)
                if quality_height is not None
                else int(round(int(source.height) * scale))
            )
            & ~1,
        )
        if self._quality_chain_oom_active:
            return None
        if (self._max_quality_pixels > 0
                and quality_width * quality_height > self._max_quality_pixels):
            self._quality_chain_oom_active = True
            return None
        key = (quality_width, quality_height, target_format)
        descriptor_indices = [
            (int(item["eye_index"]) * 3 + int(item["frame_slot"])) % self._DESCRIPTOR_COUNT
            for item in draws
        ]
        in_use = [self.quality_slot_timelines[frame_slot], self.mip_slot_timelines[frame_slot]]
        in_use.extend(self.descriptor_timelines[index] for index in descriptor_indices)
        in_use.extend(self.rcas_descriptor_timelines[index] for index in descriptor_indices)
        in_use.extend(self.quality_descriptor_timelines[index] for index in descriptor_indices)
        if any(int(value) > int(completed) for value in in_use):
            return None
        if key not in self.quality_images or key not in self.mip_images:
            # The projected footprint can move by a few pixels every frame.
            # Keep one completed extent alive instead of accumulating a full
            # pair of six-image eye/slot caches for every jittered extent.
            if self.quality_images or self.mip_images:
                self._close_quality_image_cache()
        quality_images = self.quality_images.get(key)
        if quality_images is None:
            created_quality_images: list[VulkanTransientImage] = []
            try:
                for eye in range(2):
                    for slot in range(self._QUALITY_SLOT_COUNT):
                        created_quality_images.append(
                            VulkanTransientImage(
                                self.context, quality_width, quality_height,
                                format=target_format,
                                label=f"projection-quality-eye{eye}-slot{slot}",
                            )
                        )
            except Exception:
                for image in created_quality_images:
                    image.close()
                self._quality_chain_oom_active = True
                return None
            quality_images = created_quality_images
            self.quality_images[key] = quality_images
        mip_images = self.mip_images.get(key)
        if mip_images is None:
            # The quality pass already converts to the projected screen
            # footprint and RCAS follows that conversion. A second mip chain
            # would only blur the final sample; native-mip mode owns mipmaps.
            mip_levels = 1
            created_mip_images: list[VulkanTransientImage] = []
            try:
                for eye in range(2):
                    for slot in range(self._QUALITY_SLOT_COUNT):
                        created_mip_images.append(
                            VulkanTransientImage(
                                self.context, quality_width, quality_height,
                                format=target_format,
                                label=f"projection-quality-mip-eye{eye}-slot{slot}",
                                mip_levels=mip_levels,
                            )
                        )
            except Exception:
                for image in created_mip_images:
                    image.close()
                self._quality_chain_oom_active = True
                # The quality images were allocated for this failed pair and
                # cannot be useful without their matching mip images.
                for image in quality_images:
                    image.close()
                self.quality_images.pop(key, None)
                return None
            mip_images = created_mip_images
            self.mip_images[key] = mip_images
        quality_draws = []
        rcas_draws = []
        screen_draws = []
        for item, descriptor_index in zip(draws, descriptor_indices):
            eye_index = int(item["eye_index"])
            quality = quality_images[eye_index * self._QUALITY_SLOT_COUNT + frame_slot].resource
            mip = mip_images[eye_index * self._QUALITY_SLOT_COUNT + frame_slot].resource
            if quality is None or mip is None:
                return None
            quality_draws.append(self._prepare_quality_draw(
                item["source"], quality, descriptor_index, mode=mode,
                scale=float(upscale_scale) if mode == "upscale_easu" else 1.0,
            ))
            if use_rcas:
                rcas = self._prepare_rcas_draw(
                    quality, mip, array_layer=0, eye_index=eye_index,
                    frame_slot=int(item["frame_slot"]), sharpness=self.rcas_sharpness,
                )
                rcas["target_old_layout"] = self.context.image_state(mip.image).layout
                rcas_draws.append(rcas)
            else:
                rcas_draws.append(
                    self._prepare_copy_draw(
                        quality,
                        mip,
                        descriptor_index,
                        source_ready_in_submission=True,
                        # Keep the EASU descriptor separate from the
                        # quality-to-MIP copy descriptor. Otherwise the copy
                        # update overwrites the EASU source before recording.
                        descriptor_sets=self.rcas_descriptor_sets,
                    )
                )
            screen_draw = self._prepare_draw(
                mip, item["target"], array_layer=int(item["array_layer"]),
                eye_index=eye_index, frame_slot=int(item["frame_slot"]),
                push_constants=item["push_constants"], clear_color=item["clear_color"],
                source_crop_uv=item.get("source_crop_uv"),
                source_ready_in_submission=True,
                overlay=bool(load_target),
            )
            if load_target:
                screen_draw["target_old_layout"] = (
                    self.vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
                )
            screen_draws.append(screen_draw)
        wait_semaphores = [item["wait_semaphore"] for item in draws if item["wait_semaphore"] is not None]
        wait_semaphores.extend(
            item for item in extra_wait_semaphores if item is not None
        )
        submit_profile: dict[str, float] = {}
        mip_template_hits = self._mip_template_hits
        mip_template_misses = self._mip_template_misses
        render_pass_template_hits = self._render_pass_template_hits
        render_pass_template_misses = self._render_pass_template_misses
        image_barrier_template_hits = self._image_barrier_template_hits
        image_barrier_template_misses = self._image_barrier_template_misses
        def record_stereo(command_buffer: Any) -> None:
            for index, quality_draw in enumerate(quality_draws):
                self._record_copy_draw(command_buffer, quality_draw)
                mip_draw = rcas_draws[index]
                if use_rcas:
                    self._record_rcas_draw(command_buffer, mip_draw)
                else:
                    self._record_copy_draw(command_buffer, mip_draw)
                # With one mip level this is only the color-attachment to
                # shader-read barrier; no mip generation is performed.
                self._record_generate_mips(
                    command_buffer,
                    mip_draw["target"],
                    int(mip_draw["target_old_layout"]),
                )
                self._record_draw(command_buffer, screen_draws[index])

        timeline = self.context.submit_on(
            "graphics",
            record_stereo,
            wait_semaphore=wait_semaphores,
            wait_for_timeline=int(wait_for_timeline),
            on_submit_profile=submit_profile.update,
        )
        submit_profile["mip_template_hit"] = float(
            self._mip_template_hits - mip_template_hits
        )
        submit_profile["mip_template_new"] = float(
            self._mip_template_misses - mip_template_misses
        )
        submit_profile["render_pass_template_hit"] = float(
            self._render_pass_template_hits - render_pass_template_hits
        )
        submit_profile["render_pass_template_new"] = float(
            self._render_pass_template_misses - render_pass_template_misses
        )
        submit_profile["image_barrier_template_hit"] = float(
            self._image_barrier_template_hits - image_barrier_template_hits
        )
        submit_profile["image_barrier_template_new"] = float(
            self._image_barrier_template_misses - image_barrier_template_misses
        )
        self.last_submit_profile = submit_profile
        if use_rcas:
            for quality, rcas, screen in zip(quality_draws, rcas_draws, screen_draws):
                self._complete_quality_mip_draw(quality, rcas, screen, timeline)
        else:
            for quality, copy_draw, screen in zip(quality_draws, rcas_draws, screen_draws):
                self._complete_quality_mip_draw(quality, copy_draw, screen, timeline)
        self.quality_slot_timelines[frame_slot] = int(timeline)
        self.mip_slot_timelines[frame_slot] = int(timeline)
        self._last_submit_timeline = int(timeline)
        return int(timeline)

    def submit_filament_hdr(
        self,
        draws: list[dict[str, Any]],
        sources: list[Any] | tuple[Any, ...],
        *,
        exposure_ev: float,
        wait_semaphores: list[Any] | tuple[Any, ...],
        load_target: bool = False,
    ) -> int:
        """Resolve a layered Filament HDR frame into the two OpenXR targets."""
        if len(draws) != 2 or len(sources) != 2:
            raise ValueError("Filament HDR resolve requires exactly two eyes")
        if self.hdr_pipeline is None or self.hdr_pipeline_layout is None:
            raise RuntimeError("Filament HDR resolve pipeline is unavailable")
        prepared = []
        frame_slot = int(draws[0]["frame_slot"]) % 3
        for eye_index, (item, source) in enumerate(zip(draws, sources)):
            descriptor_index = (
                int(eye_index) * 3 + frame_slot
            ) % self._DESCRIPTOR_COUNT
            descriptor_set = self.hdr_descriptor_sets[descriptor_index]
            self.vk.vkUpdateDescriptorSets(
                self.context.device,
                1,
                [self.vk.VkWriteDescriptorSet(
                    sType=self.vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=descriptor_set,
                    dstBinding=0,
                    descriptorCount=1,
                    descriptorType=self.vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    pImageInfo=[self.vk.VkDescriptorImageInfo(
                        sampler=self.sampler,
                        imageView=source.require_view(),
                        imageLayout=self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    )],
                )],
                0,
                None,
            )
            target = item["target"]
            target_layer = int(item["array_layer"])
            _view, framebuffer = self._target_view_and_framebuffer(
                target, target_layer, overlay=bool(load_target)
            )
            prepared.append({
                "source": source,
                "source_array_layer": int(eye_index),
                "target": target,
                "target_array_layer": target_layer,
                "framebuffer": framebuffer,
                "descriptor_set": descriptor_set,
                "descriptor_index": descriptor_index,
                "target_old_layout": self.context.image_state(target.image).layout,
                "source_ready_in_submission": True,
                "render_pass": (
                    self.overlay_render_pass if load_target else self.render_pass
                ),
                "pipeline": self.hdr_pipeline,
                "pipeline_layout": self.hdr_pipeline_layout,
                "payload": self.vk.ffi.new(
                    "char[]", struct.pack("<4f", float(exposure_ev), 0.0, 0.0, 0.0)
                ),
            })
        timeline = self.context.submit_on(
            "graphics",
            lambda command_buffer: [
                self._record_copy_draw(command_buffer, draw) for draw in prepared
            ],
            wait_semaphore=[item for item in wait_semaphores if item is not None],
            wait_for_timeline=max(
                self.hdr_descriptor_timelines[draw["descriptor_index"]]
                for draw in prepared
            ),
        )
        registered_sources: set[int] = set()
        for draw in prepared:
            source = draw["source"]
            source_key = id(source.image)
            if source_key not in registered_sources:
                registered_sources.add(source_key)
                self.context.register_image_state(
                    source.image,
                    ImageState(
                        layout=self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        access_mask=self.vk.VK_ACCESS_SHADER_READ_BIT,
                        stage_mask=self.vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                        queue_family_index=self.context.queue_family_index,
                    ),
                )
            self.context.register_image_state(
                draw["target"].image,
                ImageState(
                    layout=self.vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    access_mask=self.vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    stage_mask=self.vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                    queue_family_index=self.context.queue_family_index,
                ),
            )
            self.hdr_descriptor_timelines[draw["descriptor_index"]] = int(timeline)
        self._last_submit_timeline = int(timeline)
        return int(timeline)

    def submit_panorama(
        self, draws: list[dict[str, Any]], source: Any,
        *, wait_for_timeline: int = 0,
    ) -> int:
        """Render an equirectangular source into projection targets first."""
        if len(draws) != 2 or self.panorama_pipeline is None:
            return int(wait_for_timeline)
        prepared = []
        for index, item in enumerate(draws):
            descriptor_index = (index * 3 + int(item.get("frame_slot", 0))) % self._DESCRIPTOR_COUNT
            descriptor = self.descriptor_sets[descriptor_index]
            self.vk.vkUpdateDescriptorSets(self.context.device, 1, [self.vk.VkWriteDescriptorSet(
                sType=self.vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=descriptor, dstBinding=0, descriptorCount=1,
                descriptorType=self.vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                pImageInfo=[self.vk.VkDescriptorImageInfo(
                    sampler=self.sampler, imageView=source.require_view(),
                    imageLayout=self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                )],
            )], 0, None)
            target = item["target"]
            _, framebuffer = self._target_view_and_framebuffer(target, int(item["array_layer"]))
            prepared.append({
                "source": source, "target": target,
                "array_layer": int(item["array_layer"]),
                "framebuffer": framebuffer, "descriptor_set": descriptor,
                "descriptor_index": descriptor_index,
                "pipeline": self.panorama_pipeline,
                "pipeline_layout": self.panorama_pipeline_layout,
                "vertex_count": 3, "push_constant_size": 32,
                "payload": self.vk.ffi.new("char[]", item["panorama_push_constants"]),
                "clear_color": tuple(item.get("clear_color", (0.0, 0.0, 0.0, 1.0))),
                "target_old_layout": self.context.image_state(target.image).layout,
                "source_ready_in_submission": True,
            })
        timeline = self.context.submit_on(
            "graphics",
            lambda command_buffer: [self._record_draw(command_buffer, draw) for draw in prepared],
            wait_for_timeline=int(wait_for_timeline),
        )
        # Panorama currently uses the shared sampler descriptor pool. Wait for
        # this one-time background draw before later screen/Glow submissions
        # update those descriptors for the same frame.
        self.context.wait_for_timeline(timeline)
        for draw in prepared:
            self.descriptor_timelines[draw["descriptor_index"]] = int(timeline)
            self.context.register_image_state(draw["target"].image, ImageState(
                layout=self.vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                access_mask=self.vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                stage_mask=self.vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                queue_family_index=self.context.queue_family_index,
            ))
        self._last_submit_timeline = int(timeline)
        return int(timeline)

    def _supports_linear_blit(self, format_value: int) -> bool:
        properties = self.vk.vkGetPhysicalDeviceFormatProperties(
            self.context.physical_device, int(format_value)
        )
        required = (
            self.vk.VK_FORMAT_FEATURE_BLIT_SRC_BIT
            | self.vk.VK_FORMAT_FEATURE_BLIT_DST_BIT
            | self.vk.VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT
        )
        return bool(int(properties.optimalTilingFeatures) & int(required) == int(required))

    def _close_quality_image_cache(self) -> None:
        """Release cached quality intermediates after their timeline completes."""
        for images in self.quality_images.values():
            for image in images:
                image.close()
        for images in self.mip_images.values():
            for image in images:
                image.close()
        self.quality_images.clear()
        self.mip_images.clear()

    def _prepare_copy_draw(
        self,
        source: Any,
        target: Any,
        descriptor_index: int,
        *,
        source_ready_in_submission: bool = False,
        descriptor_sets: list[Any] | None = None,
    ) -> dict[str, Any]:
        if self.copy_pipeline is None or self.copy_pipeline_layout is None:
            raise RuntimeError("Vulkan projection mip copy pass is unavailable")
        source_view = source.require_view()
        source_state = self.context.image_state(source.image)
        if (
            not source_ready_in_submission
            and source_state.layout != self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        ):
            raise ValueError("projection mip source must be shader-readable")
        # Native mip now records copy and RCAS in the same command buffer.
        # Keep their image bindings separate so the RCAS update cannot replace
        # the copy source before that draw executes.
        descriptor_set = (descriptor_sets or self.quality_descriptor_sets)[descriptor_index]
        self.vk.vkUpdateDescriptorSets(
            self.context.device,
            1,
            [
                self.vk.VkWriteDescriptorSet(
                    sType=self.vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=descriptor_set,
                    dstBinding=0,
                    descriptorCount=1,
                    descriptorType=self.vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    pImageInfo=[
                        self.vk.VkDescriptorImageInfo(
                            sampler=self.sampler,
                            imageView=source_view,
                            imageLayout=self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        )
                    ],
                )
            ],
            0,
            None,
        )
        _view, framebuffer = self._target_view_and_framebuffer(target, 0)
        return {
            "source": source,
            "target": target,
            "framebuffer": framebuffer,
            "descriptor_set": descriptor_set,
            "descriptor_index": descriptor_index,
            "target_old_layout": self.context.image_state(target.image).layout,
            "source_ready_in_submission": bool(source_ready_in_submission),
        }

    def _prepare_quality_draw(
        self, source: Any, target: Any, descriptor_index: int, *, mode: str, scale: float
    ) -> dict[str, Any]:
        if self.quality_pipeline is None or self.quality_pipeline_layout is None:
            raise RuntimeError("Vulkan projection quality pass is unavailable")
        source_state = self.context.image_state(source.image)
        if source_state.layout != self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            raise ValueError("projection quality source must be shader-readable")
        descriptor_set = self.quality_descriptor_sets[descriptor_index]
        self.vk.vkUpdateDescriptorSets(
            self.context.device, 1, [self.vk.VkWriteDescriptorSet(
                sType=self.vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=descriptor_set, dstBinding=0, descriptorCount=1,
                descriptorType=self.vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                pImageInfo=[self.vk.VkDescriptorImageInfo(
                    sampler=self.sampler, imageView=source.require_view(),
                    imageLayout=self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                )],
            )], 0, None,
        )
        _view, framebuffer = self._target_view_and_framebuffer(target, 0)
        mode_value = 2.0 if mode == "upscale_easu" else 1.0
        payload = struct.pack(
            "<4f", 1.0 / float(source.width), 1.0 / float(source.height),
            mode_value, float(scale),
        )
        return {
            "source": source, "target": target, "framebuffer": framebuffer,
            "descriptor_set": descriptor_set, "descriptor_index": descriptor_index,
            "target_old_layout": self.context.image_state(target.image).layout,
            "pipeline": self.quality_pipeline,
            "pipeline_layout": self.quality_pipeline_layout,
            "payload": self.vk.ffi.new("char[]", payload),
        }

    def _image_barrier_template(
        self,
        image: Any,
        *,
        src_access: int,
        dst_access: int,
        old_layout: int,
        new_layout: int,
        base_mip_level: int = 0,
        level_count: int = 1,
        base_array_layer: int = 0,
        layer_count: int = 1,
    ) -> list[Any]:
        image_handle = int(self.vk.ffi.cast("uintptr_t", image))
        key = (
            image_handle,
            int(src_access),
            int(dst_access),
            int(old_layout),
            int(new_layout),
            int(base_mip_level),
            int(level_count),
            int(base_array_layer),
            int(layer_count),
        )
        cached = self._image_barrier_templates.get(key)
        if cached is not None:
            self._image_barrier_template_hits += 1
            return cached
        self._image_barrier_template_misses += 1
        barrier = [self.vk.VkImageMemoryBarrier(
            sType=self.vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            srcAccessMask=int(src_access),
            dstAccessMask=int(dst_access),
            oldLayout=int(old_layout),
            newLayout=int(new_layout),
            srcQueueFamilyIndex=self.vk.VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex=self.vk.VK_QUEUE_FAMILY_IGNORED,
            image=image,
            subresourceRange=self.vk.VkImageSubresourceRange(
                aspectMask=self.vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=int(base_mip_level),
                levelCount=int(level_count),
                baseArrayLayer=int(base_array_layer),
                layerCount=int(layer_count),
            ),
        )]
        self._image_barrier_templates[key] = barrier
        return barrier

    def _render_pass_recording_template(
        self,
        render_pass: Any,
        framebuffer: Any,
        width: int,
        height: int,
        clear_color: tuple[float, float, float, float],
    ) -> dict[str, Any]:
        normalized_color = tuple(float(value) for value in clear_color)
        key = (
            int(self.vk.ffi.cast("uintptr_t", render_pass)),
            int(self.vk.ffi.cast("uintptr_t", framebuffer)),
            int(width),
            int(height),
            normalized_color,
        )
        cached = self._render_pass_recording_templates.get(key)
        if cached is not None:
            self._render_pass_template_hits += 1
            return cached
        self._render_pass_template_misses += 1
        render_area = self.vk.VkRect2D(
            offset=self.vk.VkOffset2D(x=0, y=0),
            extent=self.vk.VkExtent2D(width=int(width), height=int(height)),
        )
        template = {
            "begin_info": self.vk.VkRenderPassBeginInfo(
                sType=self.vk.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                renderPass=render_pass,
                framebuffer=framebuffer,
                renderArea=render_area,
                clearValueCount=1,
                pClearValues=[self.vk.VkClearValue(
                    color=self.vk.VkClearColorValue(float32=list(normalized_color))
                )],
            ),
            "viewport": [self.vk.VkViewport(
                x=0.0,
                y=0.0,
                width=float(width),
                height=float(height),
                minDepth=0.0,
                maxDepth=1.0,
            )],
            "scissor": [render_area],
        }
        self._render_pass_recording_templates[key] = template
        return template

    def _record_copy_draw(self, command_buffer: Any, draw: dict[str, Any]) -> None:
        source = draw["source"]
        target = draw["target"]
        if draw.get("source_ready_in_submission", False):
            source_transition = self._image_barrier_template(
                source.image,
                src_access=self.vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                dst_access=self.vk.VK_ACCESS_SHADER_READ_BIT,
                old_layout=self.vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                new_layout=self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                base_array_layer=int(draw.get("source_array_layer", 0)),
            )
            self.vk.vkCmdPipelineBarrier(
                command_buffer,
                self.vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                self.vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0, 0, None, 0, None, 1, source_transition,
            )
        old_layout = int(draw["target_old_layout"])
        transition = self._image_barrier_template(
            target.image,
            src_access=(
                self.vk.VK_ACCESS_SHADER_READ_BIT
                if old_layout == self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL else 0
            ),
            dst_access=self.vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            old_layout=old_layout,
            new_layout=self.vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            base_array_layer=int(draw.get("target_array_layer", 0)),
        )
        self.vk.vkCmdPipelineBarrier(
            command_buffer,
            self.vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
            if old_layout == self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            else self.vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            self.vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            0, 0, None, 0, None, 1, transition,
        )
        render_template = self._render_pass_recording_template(
            draw.get("render_pass", self.render_pass),
            draw["framebuffer"],
            int(target.width),
            int(target.height),
            (0.0, 0.0, 0.0, 1.0),
        )
        self.vk.vkCmdBeginRenderPass(
            command_buffer,
            render_template["begin_info"],
            self.vk.VK_SUBPASS_CONTENTS_INLINE,
        )
        self.vk.vkCmdSetViewport(command_buffer, 0, 1, render_template["viewport"])
        self.vk.vkCmdSetScissor(command_buffer, 0, 1, render_template["scissor"])
        pipeline = draw.get("pipeline", self.copy_pipeline)
        pipeline_layout = draw.get("pipeline_layout", self.copy_pipeline_layout)
        self.vk.vkCmdBindPipeline(command_buffer, self.vk.VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline)
        self.vk.vkCmdBindDescriptorSets(command_buffer, self.vk.VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, [draw["descriptor_set"]], 0, None)
        if "payload" in draw:
            self.vk.vkCmdPushConstants(
                command_buffer, pipeline_layout, self.vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                0, 16, draw["payload"],
            )
        self.vk.vkCmdDraw(command_buffer, 3, 1, 0, 0)
        self.vk.vkCmdEndRenderPass(command_buffer)

    def _record_generate_mips(
        self, command_buffer: Any, image: Any, previous_layout: int
    ) -> None:
        template = self._mip_recording_template(image, previous_layout)
        if template["single_barrier"] is not None:
            self.vk.vkCmdPipelineBarrier(
                command_buffer,
                self.vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                self.vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0, 0, None, 0, None, 1, template["single_barrier"],
            )
            return
        self.vk.vkCmdPipelineBarrier(
            command_buffer,
            self.vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            self.vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, None, 0, None, 1, template["base_barriers"],
        )
        for src_stage, dst_transition, blit, src_transition in template["levels"]:
            self.vk.vkCmdPipelineBarrier(
                command_buffer,
                src_stage,
                self.vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0, None, 0, None, 1, dst_transition,
            )
            self.vk.vkCmdBlitImage(
                command_buffer,
                image.image,
                self.vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                image.image,
                self.vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                blit,
                self.vk.VK_FILTER_LINEAR,
            )
            if src_transition is not None:
                self.vk.vkCmdPipelineBarrier(
                    command_buffer,
                    self.vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                    self.vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                    0, 0, None, 0, None, 1, src_transition,
                )
        self.vk.vkCmdPipelineBarrier(
            command_buffer,
            self.vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            self.vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, None, 0, None,
            len(template["shader_barriers"]),
            template["shader_barriers"],
        )

    def _mip_recording_template(
        self, image: Any, previous_layout: int
    ) -> dict[str, Any]:
        mip_levels = int(getattr(image, "mip_levels", 1))
        image_handle = int(self.vk.ffi.cast("uintptr_t", image.image))
        key = (
            image_handle,
            int(image.width),
            int(image.height),
            mip_levels,
            int(previous_layout),
        )
        cached = self._mip_recording_templates.get(key)
        if cached is not None:
            self._mip_template_hits += 1
            return cached
        self._mip_template_misses += 1
        if mip_levels <= 1:
            template = {"single_barrier": [self.vk.VkImageMemoryBarrier(
                sType=self.vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                srcAccessMask=self.vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                dstAccessMask=self.vk.VK_ACCESS_SHADER_READ_BIT,
                oldLayout=self.vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                newLayout=self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                srcQueueFamilyIndex=self.vk.VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex=self.vk.VK_QUEUE_FAMILY_IGNORED,
                image=image.image,
                subresourceRange=self.vk.VkImageSubresourceRange(
                    aspectMask=self.vk.VK_IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel=0, levelCount=1, baseArrayLayer=0, layerCount=1,
                ),
            )], "base_barriers": (), "levels": (), "shader_barriers": ()}
            self._mip_recording_templates[key] = template
            return template
        barriers = [self.vk.VkImageMemoryBarrier(
            sType=self.vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            srcAccessMask=self.vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            dstAccessMask=self.vk.VK_ACCESS_TRANSFER_READ_BIT,
            oldLayout=self.vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            newLayout=self.vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            srcQueueFamilyIndex=self.vk.VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex=self.vk.VK_QUEUE_FAMILY_IGNORED,
            image=image.image,
            subresourceRange=self.vk.VkImageSubresourceRange(
                aspectMask=self.vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0, levelCount=1, baseArrayLayer=0, layerCount=1,
            ),
        )]
        levels = []
        for level in range(1, mip_levels):
            width = max(1, int(image.width) >> level)
            height = max(1, int(image.height) >> level)
            dst_transition = self.vk.VkImageMemoryBarrier(
                sType=self.vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                srcAccessMask=(
                    self.vk.VK_ACCESS_SHADER_READ_BIT
                    if previous_layout == self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                    else 0
                ),
                dstAccessMask=self.vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                oldLayout=previous_layout,
                newLayout=self.vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                srcQueueFamilyIndex=self.vk.VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex=self.vk.VK_QUEUE_FAMILY_IGNORED,
                image=image.image,
                subresourceRange=self.vk.VkImageSubresourceRange(
                    aspectMask=self.vk.VK_IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel=level, levelCount=1, baseArrayLayer=0, layerCount=1,
                ),
            )
            src_stage = (
                self.vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                if previous_layout == self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                else self.vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
            )
            blit = [self.vk.VkImageBlit(
                    srcSubresource=self.vk.VkImageSubresourceLayers(aspectMask=self.vk.VK_IMAGE_ASPECT_COLOR_BIT, mipLevel=level - 1, baseArrayLayer=0, layerCount=1),
                    srcOffsets=[self.vk.VkOffset3D(x=0, y=0, z=0), self.vk.VkOffset3D(x=max(1, int(image.width) >> (level - 1)), y=max(1, int(image.height) >> (level - 1)), z=1)],
                    dstSubresource=self.vk.VkImageSubresourceLayers(aspectMask=self.vk.VK_IMAGE_ASPECT_COLOR_BIT, mipLevel=level, baseArrayLayer=0, layerCount=1),
                    dstOffsets=[self.vk.VkOffset3D(x=0, y=0, z=0), self.vk.VkOffset3D(x=width, y=height, z=1)],
                )]
            src_transition = None
            if level + 1 < mip_levels:
                src_transition = self.vk.VkImageMemoryBarrier(
                    sType=self.vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    srcAccessMask=self.vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                    dstAccessMask=self.vk.VK_ACCESS_TRANSFER_READ_BIT,
                    oldLayout=self.vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    newLayout=self.vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    srcQueueFamilyIndex=self.vk.VK_QUEUE_FAMILY_IGNORED,
                    dstQueueFamilyIndex=self.vk.VK_QUEUE_FAMILY_IGNORED,
                    image=image.image,
                    subresourceRange=self.vk.VkImageSubresourceRange(aspectMask=self.vk.VK_IMAGE_ASPECT_COLOR_BIT, baseMipLevel=level, levelCount=1, baseArrayLayer=0, layerCount=1),
                )
            levels.append((
                src_stage,
                [dst_transition],
                blit,
                None if src_transition is None else [src_transition],
            ))
        shader_barriers = []
        for level in range(mip_levels):
            is_last_level = level + 1 == mip_levels
            shader_barriers.append(self.vk.VkImageMemoryBarrier(
                sType=self.vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                srcAccessMask=(
                    self.vk.VK_ACCESS_TRANSFER_WRITE_BIT
                    if is_last_level else self.vk.VK_ACCESS_TRANSFER_READ_BIT
                ),
                dstAccessMask=self.vk.VK_ACCESS_SHADER_READ_BIT,
                oldLayout=(
                    self.vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
                    if is_last_level else self.vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                ),
                newLayout=self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                srcQueueFamilyIndex=self.vk.VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex=self.vk.VK_QUEUE_FAMILY_IGNORED,
                image=image.image,
                subresourceRange=self.vk.VkImageSubresourceRange(
                    aspectMask=self.vk.VK_IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel=level, levelCount=1, baseArrayLayer=0, layerCount=1,
                ),
            ))
        template = {
            "single_barrier": None,
            "base_barriers": barriers,
            "levels": tuple(levels),
            "shader_barriers": shader_barriers,
        }
        self._mip_recording_templates[key] = template
        return template

    def _prepare_draw(
        self,
        source: Any,
        target: Any,
        *,
        array_layer: int,
        eye_index: int,
        frame_slot: int,
        push_constants: bytes,
        clear_color: tuple[float, float, float, float],
        source_crop_uv: tuple[float, float, float, float] | None = None,
        source_ready_in_submission: bool = False,
        descriptor_set: Any | None = None,
        descriptor_timelines: list[int] | None = None,
        overlay: bool = False,
    ) -> dict[str, Any]:
        if (
            self.pipeline is None
            or len(push_constants) != self._PUSH_CONSTANT_SIZE
        ):
            raise RuntimeError("Vulkan projection screen pass is unavailable")
        if int(target.format) != self.target_format:
            raise ValueError("projection target format changed after pipeline creation")
        source_view = source.require_view()
        source_state = self.context.image_state(source.image)
        if (
            not source_ready_in_submission
            and source_state.layout != self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        ):
            raise ValueError("projection screen source must be shader-readable")
        descriptor_index = (int(eye_index) * 3 + int(frame_slot)) % self._DESCRIPTOR_COUNT
        timeline_slots = descriptor_timelines or self.descriptor_timelines
        last_use = timeline_slots[descriptor_index]
        if last_use:
            self.context.wait_for_timeline(last_use)
        use_screen_crop = descriptor_set is None
        descriptor_set = descriptor_set or self.descriptor_sets[descriptor_index]
        if use_screen_crop:
            try:
                crop_values = tuple(
                    float(value) for value in (source_crop_uv or (0.0, 0.0, 1.0, 1.0))
                )
            except (TypeError, ValueError):
                crop_values = (0.0, 0.0, 1.0, 1.0)
            if len(crop_values) != 4 or not all(math.isfinite(value) for value in crop_values):
                crop_values = (0.0, 0.0, 1.0, 1.0)
            crop_x = max(0.0, min(0.9, crop_values[0]))
            crop_y = max(0.0, min(0.9, crop_values[1]))
            crop_width = max(0.1, min(1.0 - crop_x, crop_values[2]))
            crop_height = max(0.1, min(1.0 - crop_y, crop_values[3]))
            self.screen_crop_buffers[descriptor_index].write_bytes(
                struct.pack("<4f", crop_x, crop_y, crop_width, crop_height)
            )
        self.vk.vkUpdateDescriptorSets(
            self.context.device,
            1,
            [
                self.vk.VkWriteDescriptorSet(
                    sType=self.vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=descriptor_set,
                    dstBinding=0,
                    descriptorCount=1,
                    descriptorType=self.vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    pImageInfo=[
                        self.vk.VkDescriptorImageInfo(
                            sampler=self.sampler,
                            imageView=source_view,
                            imageLayout=self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        )
                    ],
                )
            ],
            0,
            None,
        )
        _view, framebuffer = self._target_view_and_framebuffer(
            target, array_layer, overlay=overlay
        )
        return {
            "target": target,
            "array_layer": int(array_layer),
            "framebuffer": framebuffer,
            "descriptor_set": descriptor_set,
            "payload": self.vk.ffi.new("char[]", push_constants),
            "clear_color": clear_color,
            "descriptor_index": descriptor_index,
            "render_pass": self.overlay_render_pass if overlay else self.render_pass,
        }

    def _record_draw(self, command_buffer: Any, draw: dict[str, Any]) -> None:
        target = draw["target"]
        source = draw.get("source")
        if draw.get("panorama_source_ready") and source is not None:
            source_state = self.context.image_state(source.image)
            barrier = self._image_barrier_template(
                source.image,
                src_access=source_state.access_mask,
                dst_access=self.vk.VK_ACCESS_SHADER_READ_BIT,
                old_layout=source_state.layout,
                new_layout=self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            )
            self.vk.vkCmdPipelineBarrier(
                command_buffer,
                source_state.stage_mask or self.vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                self.vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0, 0, None, 0, None, 1, barrier,
            )
        old_layout = int(draw.get("target_old_layout", self.vk.VK_IMAGE_LAYOUT_UNDEFINED))
        old_access = (
            self.vk.VK_ACCESS_SHADER_READ_BIT
            if old_layout == self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            else self.vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
            if old_layout == self.vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
            else 0
        )
        old_stage = (
            self.vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
            if old_layout == self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            else self.vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
            if old_layout == self.vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
            else self.vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
        )
        transition = self._image_barrier_template(
            target.image,
            src_access=old_access,
            dst_access=self.vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            old_layout=old_layout,
            new_layout=self.vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            base_array_layer=int(draw["array_layer"]),
        )
        self.vk.vkCmdPipelineBarrier(
            command_buffer,
            old_stage,
            self.vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            0, 0, None, 0, None, 1, transition,
        )
        render_template = self._render_pass_recording_template(
            draw.get("render_pass", self.render_pass),
            draw["framebuffer"],
            int(target.width),
            int(target.height),
            tuple(float(value) for value in draw["clear_color"]),
        )
        self.vk.vkCmdBeginRenderPass(
            command_buffer,
            render_template["begin_info"],
            self.vk.VK_SUBPASS_CONTENTS_INLINE,
        )
        self.vk.vkCmdSetViewport(command_buffer, 0, 1, render_template["viewport"])
        self.vk.vkCmdSetScissor(command_buffer, 0, 1, render_template["scissor"])
        self.vk.vkCmdBindPipeline(
            command_buffer,
            self.vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
            draw.get("pipeline", self.pipeline),
        )
        self.vk.vkCmdBindDescriptorSets(
            command_buffer,
            self.vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
            draw.get("pipeline_layout", self.pipeline_layout),
            0,
            1,
            [draw["descriptor_set"]],
            0,
            None,
        )
        self.vk.vkCmdPushConstants(
            command_buffer,
            draw.get("pipeline_layout", self.pipeline_layout),
            (
                self.vk.VK_SHADER_STAGE_VERTEX_BIT
                | self.vk.VK_SHADER_STAGE_FRAGMENT_BIT
            ),
            0,
            int(draw.get("push_constant_size", self._PUSH_CONSTANT_SIZE)),
            draw["payload"],
        )
        self.vk.vkCmdDraw(
            command_buffer,
            int(draw.get("vertex_count", self._VERTEX_COUNT)),
            1,
            0,
            0,
        )
        self.vk.vkCmdEndRenderPass(command_buffer)

    def _prepare_rcas_draw(
        self,
        source: Any,
        target: Any,
        *,
        array_layer: int,
        eye_index: int,
        frame_slot: int,
        sharpness: float,
    ) -> dict[str, Any]:
        if self.rcas_pipeline is None or self.rcas_pipeline_layout is None:
            raise RuntimeError("Vulkan projection RCAS pass is unavailable")
        descriptor_index = (int(eye_index) * 3 + int(frame_slot)) % self._DESCRIPTOR_COUNT
        descriptor_set = self.rcas_descriptor_sets[descriptor_index]
        source_view = source.require_view()
        self.vk.vkUpdateDescriptorSets(
            self.context.device,
            1,
            [
                self.vk.VkWriteDescriptorSet(
                    sType=self.vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=descriptor_set,
                    dstBinding=0,
                    descriptorCount=1,
                    descriptorType=self.vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    pImageInfo=[
                        self.vk.VkDescriptorImageInfo(
                            sampler=self.sampler,
                            imageView=source_view,
                            imageLayout=self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        )
                    ],
                )
            ],
            0,
            None,
        )
        _view, framebuffer = self._target_view_and_framebuffer(target, array_layer)
        payload = struct.pack(
            "<4f",
            1.0 / float(source.width),
            1.0 / float(source.height),
            float(sharpness),
            0.0,
        )
        return {
            "source": source,
            "target": target,
            "array_layer": int(array_layer),
            "framebuffer": framebuffer,
            "descriptor_set": descriptor_set,
            "payload": self.vk.ffi.new("char[]", payload),
            "descriptor_index": descriptor_index,
        }

    def _record_rcas_draw(self, command_buffer: Any, draw: dict[str, Any]) -> None:
        source = draw["source"]
        target = draw["target"]
        source_transition = self._image_barrier_template(
            source.image,
            src_access=self.vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            dst_access=self.vk.VK_ACCESS_SHADER_READ_BIT,
            old_layout=self.vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            new_layout=self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        )
        target_old_layout = int(draw.get("target_old_layout", self.vk.VK_IMAGE_LAYOUT_UNDEFINED))
        target_transition = self._image_barrier_template(
            target.image,
            src_access=(
                self.vk.VK_ACCESS_SHADER_READ_BIT
                if target_old_layout == self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL else 0
            ),
            dst_access=self.vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            old_layout=target_old_layout,
            new_layout=self.vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            base_array_layer=int(draw["array_layer"]),
        )
        self.vk.vkCmdPipelineBarrier(
            command_buffer,
            self.vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            self.vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
            | self.vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            0, 0, None, 0, None, 2, source_transition + target_transition,
        )
        render_template = self._render_pass_recording_template(
            self.render_pass,
            draw["framebuffer"],
            int(target.width),
            int(target.height),
            (0.0, 0.0, 0.0, 1.0),
        )
        self.vk.vkCmdBeginRenderPass(
            command_buffer,
            render_template["begin_info"],
            self.vk.VK_SUBPASS_CONTENTS_INLINE,
        )
        self.vk.vkCmdSetViewport(command_buffer, 0, 1, render_template["viewport"])
        self.vk.vkCmdSetScissor(command_buffer, 0, 1, render_template["scissor"])
        self.vk.vkCmdBindPipeline(command_buffer, self.vk.VK_PIPELINE_BIND_POINT_GRAPHICS, self.rcas_pipeline)
        self.vk.vkCmdBindDescriptorSets(command_buffer, self.vk.VK_PIPELINE_BIND_POINT_GRAPHICS, self.rcas_pipeline_layout, 0, 1, [draw["descriptor_set"]], 0, None)
        self.vk.vkCmdPushConstants(command_buffer, self.rcas_pipeline_layout, self.vk.VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16, draw["payload"])
        self.vk.vkCmdDraw(command_buffer, 3, 1, 0, 0)
        self.vk.vkCmdEndRenderPass(command_buffer)

    def _complete_draw(self, draw: dict[str, Any], timeline: int) -> None:
        target = draw["target"]
        self.context.register_image_state(
            target.image,
            ImageState(
                layout=self.vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                access_mask=self.vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                stage_mask=self.vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                queue_family_index=self.context.queue_family_index,
            ),
        )
        self.descriptor_timelines[draw["descriptor_index"]] = int(timeline)

    def _complete_rcas_draw(self, draw: dict[str, Any], timeline: int) -> None:
        source = draw["source"]
        target = draw["target"]
        self.context.register_image_state(
            source.image,
            ImageState(
                layout=self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                access_mask=self.vk.VK_ACCESS_SHADER_READ_BIT,
                stage_mask=self.vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                queue_family_index=self.context.queue_family_index,
            ),
        )
        self.context.register_image_state(
            target.image,
            ImageState(
                layout=self.vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                access_mask=self.vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                stage_mask=self.vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                queue_family_index=self.context.queue_family_index,
            ),
        )
        self.rcas_descriptor_timelines[draw["descriptor_index"]] = int(timeline)

    def _complete_mip_draw(
        self, copy_draw: dict[str, Any], screen_draw: dict[str, Any], timeline: int
    ) -> None:
        mip = copy_draw["target"]
        self.context.register_image_state(
            mip.image,
            ImageState(
                layout=self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                access_mask=self.vk.VK_ACCESS_SHADER_READ_BIT,
                stage_mask=self.vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                queue_family_index=self.context.queue_family_index,
            ),
        )
        self._complete_draw(screen_draw, timeline)
        self.quality_descriptor_timelines[copy_draw["descriptor_index"]] = int(timeline)

    def _complete_quality_mip_draw(
        self,
        quality_draw: dict[str, Any],
        rcas_draw: dict[str, Any],
        screen_draw: dict[str, Any],
        timeline: int,
    ) -> None:
        self.context.register_image_state(
            quality_draw["target"].image,
            ImageState(
                layout=self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                access_mask=self.vk.VK_ACCESS_SHADER_READ_BIT,
                stage_mask=self.vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                queue_family_index=self.context.queue_family_index,
            ),
        )
        self.context.register_image_state(
            rcas_draw["target"].image,
            ImageState(
                layout=self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                access_mask=self.vk.VK_ACCESS_SHADER_READ_BIT,
                stage_mask=self.vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                queue_family_index=self.context.queue_family_index,
            ),
        )
        self._complete_draw(screen_draw, timeline)
        self.quality_descriptor_timelines[quality_draw["descriptor_index"]] = int(timeline)
        self.rcas_descriptor_timelines[rcas_draw["descriptor_index"]] = int(timeline)

    def close(self) -> None:
        self._quality_chain_oom_active = False
        self._mip_recording_templates.clear()
        self._render_pass_recording_templates.clear()
        self._image_barrier_templates.clear()
        if self.context.device is not None:
            self._close_quality_image_cache()
            for framebuffer in self.framebuffers.values():
                self.vk.vkDestroyFramebuffer(self.context.device, framebuffer, None)
            for framebuffer in self.overlay_framebuffers.values():
                self.vk.vkDestroyFramebuffer(self.context.device, framebuffer, None)
            for view in self.image_views.values():
                self.vk.vkDestroyImageView(self.context.device, view, None)
            if self.pipeline is not None:
                self.vk.vkDestroyPipeline(self.context.device, self.pipeline, None)
            if self.glow_pipeline is not None:
                self.vk.vkDestroyPipeline(self.context.device, self.glow_pipeline, None)
            if self.veil_pipeline is not None:
                self.vk.vkDestroyPipeline(self.context.device, self.veil_pipeline, None)
            if self.surround_pipeline is not None:
                self.vk.vkDestroyPipeline(
                    self.context.device, self.surround_pipeline, None
                )
            if self.laser_pipeline is not None:
                self.vk.vkDestroyPipeline(
                    self.context.device, self.laser_pipeline, None
                )
            if self.controller_proxy_pipeline is not None:
                self.vk.vkDestroyPipeline(
                    self.context.device, self.controller_proxy_pipeline, None
                )
            if self.rcas_pipeline is not None:
                self.vk.vkDestroyPipeline(self.context.device, self.rcas_pipeline, None)
            if self.copy_pipeline is not None:
                self.vk.vkDestroyPipeline(self.context.device, self.copy_pipeline, None)
            if self.quality_pipeline is not None:
                self.vk.vkDestroyPipeline(self.context.device, self.quality_pipeline, None)
            if self.hdr_pipeline is not None:
                self.vk.vkDestroyPipeline(self.context.device, self.hdr_pipeline, None)
            if self.panorama_pipeline is not None:
                self.vk.vkDestroyPipeline(self.context.device, self.panorama_pipeline, None)
            if self.pipeline_layout is not None:
                self.vk.vkDestroyPipelineLayout(
                    self.context.device, self.pipeline_layout, None
                )
            if self.glow_pipeline_layout is not None:
                self.vk.vkDestroyPipelineLayout(
                    self.context.device, self.glow_pipeline_layout, None
                )
            if self.laser_pipeline_layout is not None:
                self.vk.vkDestroyPipelineLayout(
                    self.context.device, self.laser_pipeline_layout, None
                )
            if self.rcas_pipeline_layout is not None:
                self.vk.vkDestroyPipelineLayout(
                    self.context.device, self.rcas_pipeline_layout, None
                )
            if self.copy_pipeline_layout is not None:
                self.vk.vkDestroyPipelineLayout(
                    self.context.device, self.copy_pipeline_layout, None
                )
            if self.quality_pipeline_layout is not None:
                self.vk.vkDestroyPipelineLayout(
                    self.context.device, self.quality_pipeline_layout, None
                )
            if self.hdr_pipeline_layout is not None:
                self.vk.vkDestroyPipelineLayout(
                    self.context.device, self.hdr_pipeline_layout, None
                )
            if self.panorama_pipeline_layout is not None:
                self.vk.vkDestroyPipelineLayout(
                    self.context.device, self.panorama_pipeline_layout, None
                )
            if self.render_pass is not None:
                self.vk.vkDestroyRenderPass(self.context.device, self.render_pass, None)
            if self.overlay_render_pass is not None:
                self.vk.vkDestroyRenderPass(
                    self.context.device, self.overlay_render_pass, None
                )
            if self.sampler is not None:
                self.vk.vkDestroySampler(self.context.device, self.sampler, None)
            for sampler, _timeline in self._retired_samplers:
                self.vk.vkDestroySampler(self.context.device, sampler, None)
            if self.descriptor_pool is not None:
                self.vk.vkDestroyDescriptorPool(
                    self.context.device, self.descriptor_pool, None
                )
            for buffer in self.glow_param_buffers:
                buffer.close()
            for buffer in self.screen_crop_buffers:
                buffer.close()
            for buffer in self.laser_param_buffers:
                buffer.close()
            if self.glow_descriptor_pool is not None:
                self.vk.vkDestroyDescriptorPool(
                    self.context.device, self.glow_descriptor_pool, None
                )
            if self.descriptor_set_layout is not None:
                self.vk.vkDestroyDescriptorSetLayout(
                    self.context.device, self.descriptor_set_layout, None
                )
            if self.glow_descriptor_set_layout is not None:
                self.vk.vkDestroyDescriptorSetLayout(
                    self.context.device, self.glow_descriptor_set_layout, None
                )
            if self.laser_descriptor_pool is not None:
                self.vk.vkDestroyDescriptorPool(
                    self.context.device, self.laser_descriptor_pool, None
                )
            if self.laser_descriptor_set_layout is not None:
                self.vk.vkDestroyDescriptorSetLayout(
                    self.context.device, self.laser_descriptor_set_layout, None
                )
            for module in self.shader_modules:
                self.vk.vkDestroyShaderModule(self.context.device, module, None)
        self.framebuffers.clear()
        self.overlay_framebuffers.clear()
        self.image_views.clear()
        self.quality_images.clear()
        self.mip_images.clear()
        self.shader_modules.clear()
        self.descriptor_sets.clear()
        self.rcas_descriptor_sets.clear()
        self.quality_descriptor_sets.clear()
        self.hdr_descriptor_sets.clear()
        self.panorama_pipeline = None
        self.panorama_pipeline_layout = None
        self.glow_descriptor_sets.clear()
        self.glow_param_buffers.clear()
        self.screen_crop_buffers.clear()
        self.laser_descriptor_sets.clear()
        self.laser_param_buffers.clear()
        self.pipeline = None
        self.glow_pipeline = None
        self.veil_pipeline = None
        self.surround_pipeline = None
        self.laser_pipeline = None
        self.controller_proxy_pipeline = None
        self.pipeline_layout = None
        self.glow_pipeline_layout = None
        self.laser_pipeline_layout = None
        self.rcas_pipeline = None
        self.rcas_pipeline_layout = None
        self.copy_pipeline = None
        self.copy_pipeline_layout = None
        self.quality_pipeline = None
        self.quality_pipeline_layout = None
        self.hdr_pipeline = None
        self.hdr_pipeline_layout = None
        self.laser_descriptor_set_layout = None
        self.laser_descriptor_pool = None
        self.render_pass = None
        self.overlay_render_pass = None
        self.sampler = None
        self._retired_samplers.clear()
        self.descriptor_pool = None
        self.descriptor_set_layout = None
        self.glow_descriptor_pool = None
        self.glow_descriptor_set_layout = None
