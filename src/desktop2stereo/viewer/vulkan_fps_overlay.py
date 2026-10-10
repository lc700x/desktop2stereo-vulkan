"""On-screen FPS overlay for the Vulkan local viewer.

The local viewer presents with pure ``vkCmdBlitImage`` transfers, so it has no
render pass of its own.  This module borrows the projection pass machinery to
add one: the swapchain image is used as a color attachment, the already-blitted
frame is preserved with ``loadOp=LOAD``, and a textured quad is blended on top.

Two details matter for correctness:

* The overlay must be drawn **once per eye** in packed SBS/TAB display modes.
  Drawing it a single time would leave it visible to only one eye through a
  stereo headset, which reads as "the FPS counter is broken".
* Positioning uses viewport + scissor rather than vertex data.  The reused
  ``d2s_projection_rcas_vert`` shader emits a full-viewport triangle and
  derives UV from clip space, so restricting the viewport to the overlay rect
  maps the texture onto exactly that rect. This lets us reuse already-compiled
  SPIR-V without shipping a new shader.
"""

from __future__ import annotations

from pathlib import Path
import struct
import time
from typing import Any, Callable

from viewer.vulkan_compute_pipeline import read_spirv_words

_OVERLAY_SHADER_ROOT = Path(__file__).resolve().parents[1] / "shaders"
_VERTEX_SHADER = "d2s_projection_rcas_vert.spv"
_FRAGMENT_SHADER = "d2s_projection_copy_frag.spv"

# Layout of the generated RGBA panel, kept small so the blend is cheap.
_PANEL_PADDING = 14
_PANEL_RADIUS = 10
_FONT_SIZE = 26
_MAX_PANEL_WIDTH = 560
_PANEL_TEXTURE_HEIGHT = 256


def _load_panel_font(size: int):
    """Resolve a UI font, preferring the bundled Inter face."""
    from PIL import ImageFont

    bundled = Path(__file__).resolve().parents[1] / "xr_viewer" / "fonts" / "InterVariable.ttf"
    candidates = [bundled, Path("/System/Library/Fonts/Supplemental/Arial.ttf")]
    for path in candidates:
        try:
            if path.is_file():
                return ImageFont.truetype(str(path), int(size))
        except Exception:
            continue
    return ImageFont.load_default()


def build_fps_panel_rgba(
    *,
    present_fps: float,
    capture_target: int | None = None,
    latency_ms: float | None = None,
    avg_latency_ms: float | None = None,
    content_fps: float | None = None,
    reuse_ratio: float | None = None,
):
    """Rasterize the FPS panel as one RGBA image.

    Returns ``None`` when PIL is unavailable so the caller can silently keep
    the previous behaviour instead of failing the frame.
    """
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return None

    rows: list[tuple[str, tuple[int, int, int, int]]] = []
    rows.append((f"FPS  {float(present_fps):.1f}", (0, 230, 90, 255)))
    if capture_target is not None:
        rows.append((f"Target  {int(capture_target)}", (0, 210, 230, 255)))
    if content_fps is not None:
        rows.append((f"Content  {float(content_fps):.1f} FPS", (100, 210, 255, 255)))
    if reuse_ratio is not None:
        ratio = max(0.0, min(1.0, float(reuse_ratio)))
        color = (255, 135, 90, 255) if ratio >= 0.30 else (190, 190, 190, 255)
        rows.append((f"Reused  {ratio:.0%}", color))
    # Latency is the frame age at present time; avg keeps a rolling mean so a
    # single slow frame does not dominate the reading.
    if latency_ms is not None and float(latency_ms) > 0:
        rows.append((f"Latency  {float(latency_ms):.0f} ms", (255, 190, 40, 255)))
    if avg_latency_ms is not None and float(avg_latency_ms) > 0:
        rows.append((f"Avg  {float(avg_latency_ms):.0f} ms", (255, 190, 40, 255)))

    font = _load_panel_font(_FONT_SIZE)
    measure = ImageDraw.Draw(Image.new("RGBA", (1, 1), (0, 0, 0, 0)))
    widths: list[int] = []
    heights: list[int] = []
    for text, _ in rows:
        try:
            left, top, right, bottom = measure.textbbox((0, 0), text, font=font)
            widths.append(int(right - left))
            heights.append(int(bottom - top) + 4)
        except Exception:
            widths.append(len(text) * 14)
            heights.append(_FONT_SIZE + 4)

    spacing = 6
    panel_w = min(_MAX_PANEL_WIDTH, max(widths) + _PANEL_PADDING * 2)
    panel_h = sum(heights) + spacing * (len(rows) - 1) + _PANEL_PADDING * 2
    # Keep the sampled image extent stable as FPS/latency values gain or lose
    # digits. A transparent canvas preserves the visible panel size while
    # avoiding a device-local image allocation on every metrics refresh.
    texture_h = max(_PANEL_TEXTURE_HEIGHT, panel_h)
    panel = Image.new("RGBA", (_MAX_PANEL_WIDTH, texture_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(panel)
    draw.rounded_rectangle(
        [0, 0, panel_w - 1, panel_h - 1],
        radius=_PANEL_RADIUS,
        fill=(0, 0, 0, 150),
    )
    y = _PANEL_PADDING
    for index, (text, color) in enumerate(rows):
        draw.text((_PANEL_PADDING, y), text, font=font, fill=color)
        y += heights[index] + spacing
    return panel


def overlay_rect_for_eye(
    eye_rect: tuple[int, int, int, int],
    panel_size: tuple[int, int],
    *,
    margin: int = 28,
) -> tuple[int, int, int, int]:
    """Place the panel at the top-left of one eye's destination rectangle.

    ``eye_rect`` is the blit destination for a single eye in swapchain
    coordinates, so the panel follows the real presented geometry instead of
    assuming the window is split evenly. Anchoring to the corner (rather than
    centering) keeps the counter out of the way of the stereo image and matches
    where the previous Metal viewer drew it.
    """
    x0, y0, x1, y1 = (int(v) for v in eye_rect)
    panel_w, panel_h = (int(v) for v in panel_size)
    width = x1 - x0
    height = y1 - y0
    available_w = max(1, width - margin)
    available_h = max(1, height - margin)
    scale = min(
        1.0,
        available_w / max(panel_w, 1),
        available_h / max(panel_h, 1),
    )
    panel_w = max(1, int(round(panel_w * scale)))
    panel_h = max(1, int(round(panel_h * scale)))
    px = min(x0 + margin, max(x0, x1 - panel_w))
    py = min(y0 + margin, max(y0, y1 - panel_h))
    return px, py, panel_w, panel_h


def overlay_size_for_display_mode(
    panel_size: tuple[int, int], display_mode: str
) -> tuple[int, int]:
    """Compensate overlay pixels for half-resolution stereo packing.

    Half-SBS expands each eye horizontally when decoded, so the overlay must
    be encoded at half width to keep glyphs square in the displayed eye.
    Half-TAB has the same issue vertically. Cross-eyed/reversed variants retain
    the same packing and are covered by matching the layout token anywhere.
    """
    width, height = (max(1, int(value)) for value in panel_size)
    mode = str(display_mode or "").strip().casefold().replace("_", "-")
    if "half-sbs" in mode and "full-sbs" not in mode:
        width = max(1, (width + 1) // 2)
    elif "half-tab" in mode and "full-tab" not in mode:
        height = max(1, (height + 1) // 2)
    return width, height


class VulkanFpsOverlay:
    """Blit-preserving overlay pass drawn on top of the presented frame."""

    def __init__(self, viewer: Any) -> None:
        self.viewer = viewer
        self.vk = viewer.vk
        self.device = viewer.device
        self.enabled = False
        self.available = False
        self.reason = ""
        self._panel_size: tuple[int, int] = (0, 0)
        self._panel_dirty = False
        self._pending_panel = None
        self.last_upload_ms = 0.0
        self._staging = None
        self._staging_capacity = 0
        self._staging_mapped = None
        self._image = None
        self._image_memory = None
        self._image_view = None
        self._image_layout = None
        self._sampler = None
        self._descriptor_pool = None
        self._descriptor_layout = None
        self._descriptor_set = None
        self._render_pass = None
        self._render_pass_format: int | None = None
        self._framebuffers: list[Any] = []
        self._pipeline_layout = None
        self._pipeline = None
        self._pipeline_format: int | None = None
        self._shader_modules: list[Any] = []
        self._swap_format: int | None = None
        self._create()

    # ── construction ──

    def _memory_type(self, bits: int, required: int) -> int:
        props = self.vk.vkGetPhysicalDeviceMemoryProperties(self.viewer.physical_device)
        for index, item in enumerate(props.memoryTypes):
            if bits & (1 << index) and (item.propertyFlags & required) == required:
                return index
        raise RuntimeError("no compatible Vulkan memory type for FPS overlay")

    def _create_shader_module(self, path: Path) -> Any:
        words = read_spirv_words(path)
        payload = struct.pack(f"<{len(words)}I", *words)
        module = self.vk.vkCreateShaderModule(
            self.device,
            self.vk.VkShaderModuleCreateInfo(
                sType=self.vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                codeSize=len(payload),
                pCode=payload,
            ),
            None,
        )
        self._shader_modules.append(module)
        return module

    def _create(self) -> None:
        vk = self.vk
        try:
            self._sampler = vk.vkCreateSampler(
                self.device,
                vk.VkSamplerCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                    magFilter=vk.VK_FILTER_LINEAR,
                    minFilter=vk.VK_FILTER_LINEAR,
                    mipmapMode=vk.VK_SAMPLER_MIPMAP_MODE_NEAREST,
                    addressModeU=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                    addressModeV=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                    addressModeW=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                ),
                None,
            )
            self._descriptor_layout = vk.vkCreateDescriptorSetLayout(
                self.device,
                vk.VkDescriptorSetLayoutCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                    bindingCount=1,
                    pBindings=[
                        vk.VkDescriptorSetLayoutBinding(
                            binding=0,
                            descriptorType=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                            descriptorCount=1,
                            stageFlags=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                        )
                    ],
                ),
                None,
            )
            self._descriptor_pool = vk.vkCreateDescriptorPool(
                self.device,
                vk.VkDescriptorPoolCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                    maxSets=1,
                    poolSizeCount=1,
                    pPoolSizes=[
                        vk.VkDescriptorPoolSize(
                            type=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                            descriptorCount=1,
                        )
                    ],
                ),
                None,
            )
            allocated = vk.vkAllocateDescriptorSets(
                self.device,
                vk.VkDescriptorSetAllocateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                    descriptorPool=self._descriptor_pool,
                    descriptorSetCount=1,
                    pSetLayouts=[self._descriptor_layout],
                ),
            )
            self._descriptor_set = allocated[0]
            self._pipeline_layout = vk.vkCreatePipelineLayout(
                self.device,
                vk.VkPipelineLayoutCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                    setLayoutCount=1,
                    pSetLayouts=[self._descriptor_layout],
                ),
                None,
            )
            self.available = True
        except Exception as exc:
            self.available = False
            self.reason = f"{type(exc).__name__}: {exc}"

    def _ensure_render_pass(self, surface_format: int) -> bool:
        """Create the blit-preserving render pass for the swapchain format.

        The attachment format must match the swapchain image exactly, and
        macOS surfaces are typically sRGB, so this is built lazily per format
        instead of being hardcoded.
        """
        if self._render_pass is not None and self._render_pass_format == int(surface_format):
            return True
        self._destroy_render_pass()
        vk = self.vk
        try:
            # loadOp/LOAD keeps the blitted frame; final layout hands the
            # swapchain image straight back to presentation.
            attachment = vk.VkAttachmentDescription(
                format=surface_format,
                samples=vk.VK_SAMPLE_COUNT_1_BIT,
                loadOp=vk.VK_ATTACHMENT_LOAD_OP_LOAD,
                storeOp=vk.VK_ATTACHMENT_STORE_OP_STORE,
                stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
                initialLayout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                finalLayout=vk.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            )
            reference = vk.VkAttachmentReference(
                attachment=0,
                layout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            )
            subpass = vk.VkSubpassDescription(
                pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
                colorAttachmentCount=1,
                pColorAttachments=[reference],
            )
            self._render_pass = vk.vkCreateRenderPass(
                self.device,
                vk.VkRenderPassCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                    attachmentCount=1,
                    pAttachments=[attachment],
                    subpassCount=1,
                    pSubpasses=[subpass],
                ),
                None,
            )
            self._render_pass_format = int(surface_format)
            return True
        except Exception as exc:
            self.reason = f"render pass: {type(exc).__name__}: {exc}"
            return False

    def _ensure_pipeline(self, surface_format: int) -> bool:
        if self._pipeline is not None and self._pipeline_format == int(surface_format):
            return True
        self._destroy_pipeline()
        vk = self.vk
        try:
            vertex_module = self._create_shader_module(_OVERLAY_SHADER_ROOT / _VERTEX_SHADER)
            fragment_module = self._create_shader_module(
                _OVERLAY_SHADER_ROOT / _FRAGMENT_SHADER
            )
            blend = vk.VkPipelineColorBlendAttachmentState(
                blendEnable=vk.VK_TRUE,
                srcColorBlendFactor=vk.VK_BLEND_FACTOR_SRC_ALPHA,
                dstColorBlendFactor=vk.VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                colorBlendOp=vk.VK_BLEND_OP_ADD,
                srcAlphaBlendFactor=vk.VK_BLEND_FACTOR_ONE,
                dstAlphaBlendFactor=vk.VK_BLEND_FACTOR_ZERO,
                alphaBlendOp=vk.VK_BLEND_OP_ADD,
                colorWriteMask=(
                    vk.VK_COLOR_COMPONENT_R_BIT
                    | vk.VK_COLOR_COMPONENT_G_BIT
                    | vk.VK_COLOR_COMPONENT_B_BIT
                    | vk.VK_COLOR_COMPONENT_A_BIT
                ),
            )
            # Keep the stage array alive across vkCreateGraphicsPipelines.
            # The Python Vulkan binding stores pStages in an auxiliary CFFI
            # array; an inline list can be collected before MoltenVK converts
            # the SPIR-V entry points.
            shader_stages = [
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
            pipeline_info = vk.VkGraphicsPipelineCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                stageCount=len(shader_stages),
                pStages=shader_stages,
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
                pDepthStencilState=None,
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
                layout=self._pipeline_layout,
                renderPass=self._render_pass,
                subpass=0,
                basePipelineIndex=-1,
            )
            self._pipeline = vk.vkCreateGraphicsPipelines(
                self.device,
                None,
                1,
                [pipeline_info],
                None,
            )[0]
            self._pipeline_format = int(surface_format)
            return True
        except Exception as exc:
            self.reason = f"pipeline: {type(exc).__name__}: {exc}"
            return False

    # ── swapchain integration ──

    def on_swapchain_recreated(self, swap_images: list[Any], surface_format: int) -> None:
        """Rebuild per-image framebuffers after a swapchain (re)creation."""
        if not self.available:
            return
        self._release_framebuffers()
        if not self._ensure_render_pass(surface_format):
            return
        if not self._ensure_pipeline(surface_format):
            return
        try:
            vk = self.vk
            for image in swap_images:
                view = vk.vkCreateImageView(
                    self.device,
                    vk.VkImageViewCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                        image=image,
                        viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
                        format=surface_format,
                        subresourceRange=vk.VkImageSubresourceRange(
                            aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                            baseMipLevel=0,
                            levelCount=1,
                            baseArrayLayer=0,
                            layerCount=1,
                        ),
                    ),
                    None,
                )
                framebuffer = vk.vkCreateFramebuffer(
                    self.device,
                    vk.VkFramebufferCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                        renderPass=self._render_pass,
                        attachmentCount=1,
                        pAttachments=[view],
                        width=self.viewer.extent[0],
                        height=self.viewer.extent[1],
                        layers=1,
                    ),
                    None,
                )
                self._framebuffers.append((view, framebuffer))
            self._swap_format = int(surface_format)
        except Exception as exc:
            self._release_framebuffers()
            self.available = False
            self.reason = f"framebuffer: {type(exc).__name__}: {exc}"

    def on_swapchain_destroyed(self) -> None:
        """Release swapchain views before their images are destroyed."""
        self._release_framebuffers()
        self._swap_format = None

    def _release_framebuffers(self) -> None:
        for view, framebuffer in self._framebuffers:
            if framebuffer is not None:
                self.vk.vkDestroyFramebuffer(self.device, framebuffer, None)
            if view is not None:
                self.vk.vkDestroyImageView(self.device, view, None)
        self._framebuffers = []

    # ── panel texture ──

    def set_panel(self, panel: Any) -> None:
        """Queue preconverted RGBA bytes for upload on the present thread."""
        if panel is None:
            return
        width, height = panel.size
        if width <= 0 or height <= 0:
            return
        try:
            rgba = panel if getattr(panel, "mode", None) == "RGBA" else panel.convert("RGBA")
            self._pending_panel = (rgba.tobytes(), (int(width), int(height)))
        except Exception as exc:
            print(
                "[VulkanLocalViewer] FPS overlay raster conversion failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return
        self._panel_dirty = True

    def _upload_panel(self, cmd: Any) -> bool:
        self.last_upload_ms = 0.0
        pending_panel = self._pending_panel
        if pending_panel is None or not self._panel_dirty:
            return False
        upload_started = time.perf_counter()
        data, (width, height) = pending_panel
        vk = self.vk
        try:
            capacity = width * height * 4
            if self._staging is None or self._staging_capacity < capacity:
                self._destroy_staging()
                self._staging = vk.vkCreateBuffer(
                    self.device,
                    vk.VkBufferCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                        size=capacity,
                        usage=vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
                    ),
                    None,
                )
                requirements = vk.vkGetBufferMemoryRequirements(self.device, self._staging)
                self._staging_memory = vk.vkAllocateMemory(
                    self.device,
                    vk.VkMemoryAllocateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                        allocationSize=requirements.size,
                        memoryTypeIndex=self._memory_type(
                            requirements.memoryTypeBits,
                            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                            | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        ),
                    ),
                    None,
                )
                vk.vkBindBufferMemory(self.device, self._staging, self._staging_memory, 0)
                self._staging_mapped = vk.vkMapMemory(
                    self.device, self._staging_memory, 0, capacity, 0
                )
                self._staging_capacity = capacity
                self._destroy_image()
            payload = memoryview(data)
            if payload.format != "B":
                payload = payload.cast("B")
            self._staging_mapped[0 : payload.nbytes] = payload
            if self._image is None or self._panel_size != (width, height):
                self._destroy_image()
                self._image = vk.vkCreateImage(
                    self.device,
                    vk.VkImageCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                        imageType=vk.VK_IMAGE_TYPE_2D,
                        format=vk.VK_FORMAT_R8G8B8A8_UNORM,
                        extent=vk.VkExtent3D(width=width, height=height, depth=1),
                        mipLevels=1,
                        arrayLayers=1,
                        samples=vk.VK_SAMPLE_COUNT_1_BIT,
                        tiling=vk.VK_IMAGE_TILING_OPTIMAL,
                        usage=vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT
                        | vk.VK_IMAGE_USAGE_SAMPLED_BIT,
                        sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
                        initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
                    ),
                    None,
                )
                requirements = vk.vkGetImageMemoryRequirements(self.device, self._image)
                self._image_memory = vk.vkAllocateMemory(
                    self.device,
                    vk.VkMemoryAllocateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                        allocationSize=requirements.size,
                        memoryTypeIndex=self._memory_type(
                            requirements.memoryTypeBits,
                            vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        ),
                    ),
                    None,
                )
                vk.vkBindImageMemory(self.device, self._image, self._image_memory, 0)
                self._image_view = vk.vkCreateImageView(
                    self.device,
                    vk.VkImageViewCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                        image=self._image,
                        viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
                        format=vk.VK_FORMAT_R8G8B8A8_UNORM,
                        subresourceRange=vk.VkImageSubresourceRange(
                            aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                            baseMipLevel=0,
                            levelCount=1,
                            baseArrayLayer=0,
                            layerCount=1,
                        ),
                    ),
                    None,
                )
                self._image_layout = None
                self._panel_size = (width, height)
            source_layout = (
                self._image_layout
                if self._image_layout is not None
                else vk.VK_IMAGE_LAYOUT_UNDEFINED
            )
            self._transition(cmd, self._image, source_layout, vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
            vk.vkCmdCopyBufferToImage(
                cmd,
                self._staging,
                self._image,
                vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                [
                    vk.VkBufferImageCopy(
                        bufferOffset=0,
                        bufferRowLength=0,
                        bufferImageHeight=0,
                        imageSubresource=vk.VkImageSubresourceLayers(
                            aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                            mipLevel=0,
                            baseArrayLayer=0,
                            layerCount=1,
                        ),
                        imageOffset=vk.VkOffset3D(x=0, y=0, z=0),
                        imageExtent=vk.VkExtent3D(
                            width=width, height=height, depth=1
                        ),
                    )
                ],
            )
            self._transition(
                cmd,
                self._image,
                vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            )
            self._image_layout = vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            vk.vkUpdateDescriptorSets(
                self.device,
                1,
                [
                    vk.VkWriteDescriptorSet(
                        sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                        dstSet=self._descriptor_set,
                        dstBinding=0,
                        descriptorCount=1,
                        descriptorType=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        pImageInfo=[
                            vk.VkDescriptorImageInfo(
                                sampler=self._sampler,
                                imageView=self._image_view,
                                imageLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            )
                        ],
                    )
                ],
                0,
                None,
            )
            self._pending_panel = None
            self._panel_dirty = False
            return True
        except Exception as exc:
            print(
                "[VulkanLocalViewer] FPS overlay upload failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            self._pending_panel = None
            self._panel_dirty = False
            return False
        finally:
            self.last_upload_ms = (time.perf_counter() - upload_started) * 1000.0

    def _transition(self, cmd: Any, image: Any, old: int, new: int) -> None:
        if old == new:
            return
        barrier = self.vk.VkImageMemoryBarrier(
            sType=self.vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            oldLayout=old,
            newLayout=new,
            srcQueueFamilyIndex=self.vk.VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex=self.vk.VK_QUEUE_FAMILY_IGNORED,
            image=image,
            subresourceRange=self.vk.VkImageSubresourceRange(
                aspectMask=self.vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0,
                levelCount=1,
                baseArrayLayer=0,
                layerCount=1,
            ),
            srcAccessMask=0,
            dstAccessMask=self.vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
            if new == self.vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
            else self.vk.VK_ACCESS_SHADER_READ_BIT,
        )
        self.vk.vkCmdPipelineBarrier(
            cmd,
            self.vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            self.vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0,
            0,
            None,
            0,
            None,
            1,
            [barrier],
        )

    # ── per-frame draw ──

    def record(
        self,
        cmd: Any,
        swapchain_image: Any,
        swap_image_index: int,
        eye_rects: tuple[tuple[int, int, int, int], ...],
    ) -> bool:
        """Blend the panel onto one swapchain image. Returns True when drawn."""
        if not self.available or not self._framebuffers:
            return False
        # Upload a freshly rasterized panel; once it lands, keep drawing the
        # same texture every frame until a new one is queued.
        self._upload_panel(cmd)
        if self._image_view is None or self._panel_size == (0, 0):
            return False
        panel_w, panel_h = self._panel_size
        if panel_w <= 0 or panel_h <= 0:
            return False
        if swap_image_index >= len(self._framebuffers):
            return False
        display_size = overlay_size_for_display_mode(
            self._panel_size,
            getattr(getattr(self.viewer, "config", None), "display_mode", "Half-SBS"),
        )
        rects = [
            overlay_rect_for_eye(eye_rect, display_size)
            for eye_rect in eye_rects
        ] or [overlay_rect_for_eye((0, 0) + self.viewer.extent, display_size)]
        try:
            vk = self.vk
            self._transition(
                cmd,
                swapchain_image,
                vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            )
            _, framebuffer = self._framebuffers[swap_image_index]
            pass_begin = vk.VkRenderPassBeginInfo(
                sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                renderPass=self._render_pass,
                framebuffer=framebuffer,
                renderArea=vk.VkRect2D(
                    offset=vk.VkOffset2D(x=0, y=0),
                    extent=vk.VkExtent2D(
                        width=self.viewer.extent[0], height=self.viewer.extent[1]
                    ),
                ),
                clearValueCount=0,
            )
            vk.vkCmdBeginRenderPass(cmd, pass_begin, vk.VK_SUBPASS_CONTENTS_INLINE)
            vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, self._pipeline)
            vk.vkCmdBindDescriptorSets(
                cmd,
                vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
                self._pipeline_layout,
                0,
                1,
                [self._descriptor_set],
                0,
                None,
            )
            # One draw per eye: the fullscreen triangle emitted by the reused
            # vertex shader is mapped onto each panel rect via the viewport.
            for x, y, width, height in rects:
                vk.vkCmdSetViewport(
                    cmd,
                    0,
                    1,
                    [
                        vk.VkViewport(
                            x=float(x),
                            y=float(y),
                            width=float(width),
                            height=float(height),
                            minDepth=0.0,
                            maxDepth=1.0,
                        )
                    ],
                )
                vk.vkCmdSetScissor(
                    cmd,
                    0,
                    1,
                    [
                        vk.VkRect2D(
                            offset=vk.VkOffset2D(x=x, y=y),
                            extent=vk.VkExtent2D(width=width, height=height),
                        )
                    ],
                )
                vk.vkCmdDraw(cmd, 3, 1, 0, 0)
            vk.vkCmdEndRenderPass(cmd)
            return True
        except Exception as exc:
            print(
                "[VulkanLocalViewer] FPS overlay draw skipped: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            self.available = False
            self.reason = f"draw: {type(exc).__name__}: {exc}"
            return False

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    # ── teardown ──

    def _destroy_staging(self) -> None:
        vk = self.vk
        if self._staging_mapped is not None and getattr(self, "_staging_memory", None) is not None:
            try:
                vk.vkUnmapMemory(self.device, self._staging_memory)
            except Exception:
                pass
        self._staging_mapped = None
        if getattr(self, "_staging_memory", None) is not None:
            vk.vkFreeMemory(self.device, self._staging_memory, None)
            self._staging_memory = None
        if self._staging is not None:
            vk.vkDestroyBuffer(self.device, self._staging, None)
            self._staging = None
        self._staging_capacity = 0

    def _destroy_image(self) -> None:
        vk = self.vk
        if self._image_view is not None:
            vk.vkDestroyImageView(self.device, self._image_view, None)
            self._image_view = None
        if self._image is not None:
            vk.vkDestroyImage(self.device, self._image, None)
            self._image = None
        if getattr(self, "_image_memory", None) is not None:
            vk.vkFreeMemory(self.device, self._image_memory, None)
            self._image_memory = None
        self._image_layout = None
        self._panel_size = (0, 0)

    def _destroy_render_pass(self) -> None:
        if getattr(self, "_render_pass", None) is not None:
            try:
                self.vk.vkDestroyRenderPass(self.device, self._render_pass, None)
            except Exception:
                pass
        self._render_pass = None
        self._render_pass_format = None

    def _destroy_pipeline(self) -> None:
        if getattr(self, "_pipeline", None) is not None:
            try:
                self.vk.vkDestroyPipeline(self.device, self._pipeline, None)
            except Exception:
                pass
        self._pipeline = None
        self._pipeline_format = None

    def close(self) -> None:
        if self.vk is None or self.device is None:
            return
        self._release_framebuffers()
        self._destroy_image()
        self._destroy_staging()
        self._destroy_pipeline()
        self._destroy_render_pass()
        vk = self.vk
        for handle in (
            self._pipeline_layout,
            self._descriptor_pool,
            self._descriptor_layout,
            self._sampler,
        ):
            if handle is None:
                continue
            try:
                if handle == self._pipeline_layout:
                    vk.vkDestroyPipelineLayout(self.device, handle, None)
                elif handle == self._descriptor_pool:
                    vk.vkDestroyDescriptorPool(self.device, handle, None)
                elif handle == self._descriptor_layout:
                    vk.vkDestroyDescriptorSetLayout(self.device, handle, None)
                elif handle == self._sampler:
                    vk.vkDestroySampler(self.device, handle, None)
            except Exception:
                pass
        for module in self._shader_modules:
            try:
                vk.vkDestroyShaderModule(self.device, module, None)
            except Exception:
                pass
        self._shader_modules = []
        self.available = False
