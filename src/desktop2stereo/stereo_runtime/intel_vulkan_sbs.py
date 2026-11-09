"""GPU-only bridge from Vulkan eye images to Intel D3D11 final-SBS input."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from stereo_runtime.providers.intel.d3d11_sbs_surface import D3D11SbsSurface


@dataclass(frozen=True, slots=True)
class IntelVulkanSbsFrame:
    """Borrowed final BGRA texture contract consumed by the native encoder."""

    texture: int
    device: int
    width: int
    height: int
    adapter_luid: int
    gpu_to_cpu: bool = False
    # Strict end-to-end zero-copy remains false until a real Intel target
    # validates the complete Vulkan -> D3D11 -> VideoProcessor -> oneVPL path.
    zero_copy: bool = False
    # Vulkan writes the imported BGRA image, then VideoProcessor converts it
    # to the NV12 surface consumed by oneVPL: one GPU conversion is real.
    gpu_copy_count: int = 1
    producer_ready: bool = False
    ready_timeline: int = 0
    vulkan_resource: Any = None


def _eye_blit_regions(
    eye_size: tuple[int, int], target_size: tuple[int, int], display_mode: str,
    fit_mode: str, input_size: tuple[int, int] | None, *, packed_source: bool = False,
) -> tuple:
    """Map the stream's per-eye fit rectangles back to full-resolution source eyes."""
    from streaming.aspect import presentation_blit_regions

    width, height = eye_size
    mode = str(display_mode).casefold().replace("_", "-")
    is_tab = mode.endswith("tab")
    is_half = mode.startswith("half-")
    encoded_width = width // 2 if is_half and not is_tab else width
    encoded_height = height // 2 if is_half and is_tab else height
    encoded_width, encoded_height = max(1, encoded_width), max(1, encoded_height)
    source_size = (encoded_width, encoded_height * 2) if is_tab else (encoded_width * 2, encoded_height)
    regions = presentation_blit_regions(source_size, target_size, fit_mode, display_mode, input_size)
    mapped = []
    for index, (source, destination) in enumerate(regions):
        origin_x, origin_y = (0, index * encoded_height) if is_tab else (index * encoded_width, 0)
        crop = tuple(int(round(value)) for value in (
            (source[0] - origin_x) * width / encoded_width,
            (source[1] - origin_y) * height / encoded_height,
            (source[2] - origin_x) * width / encoded_width,
            (source[3] - origin_y) * height / encoded_height,
        ))
        if packed_source:
            crop = (crop[0] + index * width, crop[1], crop[2] + index * width, crop[3])
        mapped.append((crop, destination))
    return tuple(mapped)


class IntelVulkanSbsComposer:
    """Compose Vulkan left/right images into a D3D11-owned shared texture.

    The current bridge uses two Vulkan blits because the stereo shader produces
    separate eye images. There is no CPU readback, but the two GPU blits mean
    this is explicitly a GPU-only copy path, not strict zero-copy yet.
    """

    def __init__(
        self, context: Any, eye_width: int, eye_height: int, *, target_size: tuple[int, int] | None = None,
    ) -> None:
        self.context = context
        self.eye_width = int(eye_width)
        self.eye_height = int(eye_height)
        if self.eye_width < 1 or self.eye_height < 1:
            raise ValueError("Intel Vulkan SBS eye dimensions must be positive")
        adapter_luid = int(getattr(context.device_info, "adapter_luid", 0))
        if adapter_luid == 0:
            raise RuntimeError("Vulkan Adapter LUID is unavailable for Intel SBS composition")
        self.adapter_luid = adapter_luid
        self.target_size = target_size
        output_width, output_height = target_size or (self.eye_width * 2, self.eye_height)
        self.surface = D3D11SbsSurface(
            width=output_width,
            height=output_height,
            adapter_luid=adapter_luid,
        )
        if self.surface.adapter_luid != adapter_luid:
            self.surface.close()
            raise RuntimeError("D3D11 SBS surface Adapter LUID does not match Vulkan")
        self.imported = None
        try:
            self.imported = self.surface.import_bgra_into_vulkan(context)
        except Exception:
            self.surface.close()
            raise

    def compose(
        self,
        left_resource: Any,
        right_resource: Any,
        *,
        ready_timeline: int | None = None,
        display_mode: str = "Full-SBS",
        fit_mode: str = "stretch",
        input_size: tuple[int, int] | None = None,
        packed_source: bool = False,
    ) -> IntelVulkanSbsFrame:
        if self.imported is None or self.imported.resource is None:
            raise RuntimeError("Intel Vulkan SBS imported destination is unavailable")
        if getattr(left_resource, "context", None) is not self.context or getattr(
            right_resource, "context", None
        ) is not self.context:
            raise RuntimeError("Vulkan SBS eye resources belong to a different context")
        kwargs = {}
        if self.target_size is not None:
            regions = _eye_blit_regions(
                (self.eye_width, self.eye_height), self.target_size,
                display_mode, fit_mode, input_size, packed_source=packed_source,
            )
            native_regions = (
                ((0, 0, self.eye_width, self.eye_height), (0, 0, self.eye_width, self.eye_height)),
                ((0, 0, self.eye_width, self.eye_height),
                 (self.eye_width, 0, self.eye_width * 2, self.eye_height)),
            )
            if regions != native_regions:
                kwargs["blit_regions"] = regions
        timeline = self.context.compose_sbs_images(
            left_resource,
            right_resource,
            self.imported.resource,
            wait_for_timeline=ready_timeline,
            **kwargs,
        )
        # D3D11 has no Vulkan semaphore object in this bridge yet. Waiting for
        # the submitted Vulkan timeline is still GPU synchronization; it does
        # not map or copy image data through the CPU. The encoder consumer will
        # issue the D3D11 VideoProcessor conversion exactly once.
        self.context.wait_for_timeline(timeline)
        return IntelVulkanSbsFrame(
            texture=self.surface.bgra_texture,
            device=self.surface.device,
            width=self.surface.width,
            height=self.surface.height,
            adapter_luid=self.adapter_luid,
            producer_ready=True,
            ready_timeline=int(timeline),
            vulkan_resource=self.imported.resource,
        )

    def frame_after_vulkan_submit(self, timeline: int) -> IntelVulkanSbsFrame:
        """Return the D3D11 surface after a packed shader write completes."""
        if self.imported is None or self.imported.resource is None:
            raise RuntimeError("Intel Vulkan SBS imported destination is unavailable")
        self.context.wait_for_timeline(int(timeline))
        return IntelVulkanSbsFrame(
            texture=self.surface.bgra_texture,
            device=self.surface.device,
            width=self.surface.width,
            height=self.surface.height,
            adapter_luid=self.adapter_luid,
            producer_ready=True,
            ready_timeline=int(timeline),
            gpu_copy_count=1,
            vulkan_resource=self.imported.resource,
        )

    def close(self) -> None:
        if self.imported is not None:
            self.imported.close()
            self.imported = None
        if self.surface is not None:
            self.surface.close()
            self.surface = None


class IntelVulkanSbsRuntimeBridge:
    """Run a deferred Vulkan stereo request for the Intel network sink.

    Network mode has no OpenXR presenter that can provide image slots. This
    bridge owns a Vulkan context and dispatches the existing layered image pass
    directly into the D3D11-imported SBS image. The bridge is deliberately
    opt-in and does not perform an intermediate eye-image blit.
    """

    def __init__(
        self, eye_width: int, eye_height: int, *, ring_size: int = 3,
        target_size: tuple[int, int] | None = None, display_mode: str = "Full-SBS",
        fit_mode: str = "stretch", input_size: tuple[int, int] | None = None,
    ) -> None:
        from viewer.vulkan_context import VulkanContext, VulkanContextConfig
        from stereo_runtime.vulkan_backend import VulkanStereoImageComputeBackend

        self.eye_width = int(eye_width)
        self.eye_height = int(eye_height)
        self.ring_size = max(2, int(ring_size))
        self.context = VulkanContext.create(
            VulkanContextConfig(frame_context_count=self.ring_size)
        )
        self.backend = VulkanStereoImageComputeBackend(self.context)
        self.composer = IntelVulkanSbsComposer(
            self.context, self.eye_width, self.eye_height
        )
        destination = getattr(self.composer.imported, "resource", None)
        if destination is None:
            raise RuntimeError("Intel Vulkan SBS imported destination is unavailable")
        self.context.prepare_external_image_for_producer(destination)
        self.output_composer = (
            IntelVulkanSbsComposer(self.context, self.eye_width, self.eye_height, target_size=target_size)
            if target_size is not None and (
                target_size != (self.eye_width * 2, self.eye_height)
                or str(display_mode).casefold().endswith("tab")
            ) else None
        )
        self.display_mode, self.fit_mode, self.input_size = display_mode, fit_mode, input_size
        self._frame_id = 0

    def submit(self, request: Any) -> IntelVulkanSbsFrame:
        rgb = getattr(request, "rgb", None)
        depth = getattr(request, "depth", None)
        params = getattr(request, "params", None)
        if rgb is None or depth is None or params is None:
            raise ValueError("Vulkan Intel SBS request is incomplete")
        height = int(getattr(rgb, "shape", (0, 0, 0, 0))[-2])
        width = int(getattr(rgb, "shape", (0, 0, 0, 0))[-1])
        if (width, height) != (self.eye_width, self.eye_height):
            raise RuntimeError(
                "Vulkan Intel SBS request dimensions changed: "
                f"expected={self.eye_width}x{self.eye_height} actual={width}x{height}"
            )
        destination = getattr(self.composer.imported, "resource", None)
        if destination is None:
            raise RuntimeError("Vulkan Intel SBS imported destination is unavailable")
        timeline, _debug = self.backend.submit_to_images(
            rgb,
            depth,
            destination,
            destination,
            params=params,
            packed_output=True,
        )
        frame = (
            self.output_composer.compose(
                destination, destination, ready_timeline=timeline,
                display_mode=self.display_mode, fit_mode=self.fit_mode,
                input_size=self.input_size, packed_source=True,
            )
            if self.output_composer is not None
            else self.composer.frame_after_vulkan_submit(timeline)
        )
        self._frame_id += 1
        return frame

    def close(self) -> None:
        backend = getattr(self, "backend", None)
        if backend is not None:
            try:
                backend.close()
            except Exception:
                pass
            self.backend = None
        output_composer = getattr(self, "output_composer", None)
        if output_composer is not None:
            output_composer.close()
            self.output_composer = None
        composer = getattr(self, "composer", None)
        if composer is not None:
            try:
                composer.close()
            except Exception:
                pass
            self.composer = None
        context = getattr(self, "context", None)
        if context is not None:
            try:
                context.close()
            except Exception:
                pass
            self.context = None


__all__ = [
    "IntelVulkanSbsComposer",
    "IntelVulkanSbsFrame",
    "IntelVulkanSbsRuntimeBridge",
]
