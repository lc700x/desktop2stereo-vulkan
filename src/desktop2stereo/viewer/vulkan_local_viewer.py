"""Native Vulkan local viewer for packed SBS frames.

The OpenXR presenter has its own projection swapchain.  Local Viewer uses this
separate GLFW/Vulkan swapchain and never creates an OpenGL context.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import queue
import sys
import time
from typing import Any, Callable

from utils.display_info import resolve_glfw_monitor_index
from viewer.cuda_vulkan_interop import CudaVulkanImageImporter
from viewer.vulkan_resources import VulkanExportableImage, VulkanExportableSemaphore
from viewer.window_control import hide_window_from_capture, set_window_mouse_passthrough


LOCAL_VIEWER_SOURCE_FORMAT = "VK_FORMAT_R8G8B8A8_SRGB"
LOCAL_VIEWER_SRGB_SURFACE_FORMATS = (
    "VK_FORMAT_B8G8R8A8_SRGB",
    "VK_FORMAT_R8G8B8A8_SRGB",
)
DIRECT_DISPLAY_INSTANCE_EXTENSIONS = (
    "VK_KHR_display",
    "VK_EXT_direct_mode_display",
)
DIRECT_DISPLAY_WIN32_DEVICE_EXTENSION = "VK_NV_acquire_winrt_display"
FULL_SCREEN_EXCLUSIVE_INSTANCE_EXTENSION = "VK_KHR_get_surface_capabilities2"
FULL_SCREEN_EXCLUSIVE_DEVICE_EXTENSION = "VK_EXT_full_screen_exclusive"


def direct_display_capability(
    instance_extensions: Any,
    device_extensions: Any = (),
    *,
    platform: str = sys.platform,
) -> tuple[bool, tuple[str, ...]]:
    """Check raw-display requirements in the correct Vulkan extension scopes."""
    available_instance = set(instance_extensions)
    available_device = set(device_extensions)
    missing = [
        name
        for name in DIRECT_DISPLAY_INSTANCE_EXTENSIONS
        if name not in available_instance
    ]
    if (
        platform == "win32"
        and DIRECT_DISPLAY_WIN32_DEVICE_EXTENSION not in available_device
    ):
        missing.append(DIRECT_DISPLAY_WIN32_DEVICE_EXTENSION)
    return not missing, tuple(missing)


def full_screen_exclusive_capability(
    instance_extensions: Any,
    device_extensions: Any,
    *,
    platform: str = sys.platform,
) -> tuple[bool, tuple[str, ...]]:
    """Check the Win32 swapchain-exclusive prerequisites."""
    if platform != "win32":
        return False, ("Windows",)
    missing = []
    if FULL_SCREEN_EXCLUSIVE_INSTANCE_EXTENSION not in set(instance_extensions):
        missing.append(FULL_SCREEN_EXCLUSIVE_INSTANCE_EXTENSION)
    if FULL_SCREEN_EXCLUSIVE_DEVICE_EXTENSION not in set(device_extensions):
        missing.append(FULL_SCREEN_EXCLUSIVE_DEVICE_EXTENSION)
    return not missing, tuple(missing)


def should_request_full_screen_exclusive(
    fullscreen: bool,
    capture_compatible_fullscreen: bool,
) -> bool:
    return bool(fullscreen and not capture_compatible_fullscreen)


def choose_srgb_surface_format(formats: Any, vk: Any) -> tuple[Any, bool]:
    """Prefer an sRGB surface format for display-referred runtime frames."""
    for name in LOCAL_VIEWER_SRGB_SURFACE_FORMATS:
        value = getattr(vk, name)
        selected = next((item for item in formats if item.format == value), None)
        if selected is not None:
            return selected, True
    return formats[0], False


def choose_present_mode(
    modes: Any,
    vk: Any,
    *,
    vsync: bool,
    full_screen_exclusive: bool,
) -> Any:
    """Choose a non-blocking Windows-exclusive mode without changing windowed WSI."""
    available = set(modes)
    if vsync:
        return vk.VK_PRESENT_MODE_FIFO_KHR
    preferred = (
        vk.VK_PRESENT_MODE_IMMEDIATE_KHR
        if full_screen_exclusive
        else vk.VK_PRESENT_MODE_MAILBOX_KHR
    )
    return preferred if preferred in available else vk.VK_PRESENT_MODE_FIFO_KHR


def is_exclusive_fullscreen_toggle(key: int, action: int, mods: int, glfw: Any) -> bool:
    return (
        key == glfw.KEY_ENTER
        and action == glfw.PRESS
        and bool(mods & glfw.MOD_ALT)
    )


def should_restore_persistent_fullscreen(
    fullscreen: bool, visible: bool, iconic: bool, topmost: bool
) -> bool:
    return bool(fullscreen and (not visible or iconic or not topmost))


def configure_glfw_window_hints(glfw: Any, *, fullscreen: bool) -> None:
    """Reset process-global GLFW hints before creating each viewer window."""
    glfw.default_window_hints()
    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
    glfw.window_hint(glfw.AUTO_ICONIFY, glfw.FALSE)
    glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
    glfw.window_hint(glfw.DECORATED, glfw.TRUE)
    glfw.window_hint(glfw.FLOATING, glfw.FALSE)
    glfw.window_hint(glfw.FOCUS_ON_SHOW, glfw.TRUE)
    if fullscreen:
        # Create the output hidden so Windows cannot register a taskbar button
        # before WS_EX_TOOLWINDOW is applied by _configure_persistent_fullscreen.
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.FLOATING, glfw.TRUE)
        glfw.window_hint(glfw.FOCUS_ON_SHOW, glfw.FALSE)


def present_fps_if_due(frames: int, elapsed: float, interval: float = 5.0) -> float | None:
    if frames <= 0 or elapsed < interval:
        return None
    return frames / max(elapsed, 1e-6)


def display_refresh_warning_needed(
    refresh_hz: float,
    sbs_fps: float,
    minimum_refresh_hz: float = 60.0,
) -> bool:
    """Return whether the selected SBS display cannot show the produced rate."""
    refresh = float(refresh_hz)
    produced = float(sbs_fps)
    if refresh <= 0.0 or produced <= 0.0:
        return False
    return refresh + 3 < produced or refresh + 3 < minimum_refresh_hz


def capture_refresh_warning_needed(refresh_hz: float, capture_target: int) -> bool:
    """Return whether the input display is slower than adaptive capture."""
    refresh = float(refresh_hz)
    target = int(capture_target)
    if refresh <= 0.0 or target <= 0:
        return False
    return refresh + 0.5 < target


def fit_rect(source: tuple[int, int], target: tuple[int, int]) -> tuple[int, int, int, int]:
    """Return an aspect-correct, centered destination rectangle."""
    sw, sh = source
    tw, th = target
    if min(sw, sh, tw, th) <= 0:
        return 0, 0, 0, 0
    scale = min(tw / sw, th / sh)
    width, height = max(1, round(sw * scale)), max(1, round(sh * scale))
    return (tw - width) // 2, (th - height) // 2, width, height


def normalize_display_fit_mode(value: Any) -> str:
    normalized = str(value or "contain").strip().casefold().replace("-", "_")
    aliases = {
        "contain": "contain",
        "complete": "contain",
        "fit": "contain",
        "keep ratio (complete)": "contain",
        "保持比例（完整）": "contain",
        "保持比例(完整)": "contain",
        "cover": "cover",
        "fill": "cover",
        "keep ratio (fill)": "cover",
        "保持比例（铺满）": "cover",
        "保持比例(铺满)": "cover",
        "stretch": "stretch",
        "stretch to fill": "stretch",
        "拉伸铺满": "stretch",
    }
    return aliases.get(normalized, "contain")


def _cover_crop_rect(
    source: tuple[int, int], target_aspect: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Return a centered source crop matching the requested aspect ratio."""
    sw, sh = source
    tw, th = target_aspect
    if min(sw, sh, tw, th) <= 0:
        return 0, 0, max(0, sw), max(0, sh)
    if sw * th > sh * tw:
        width = max(1, min(sw, round(sh * tw / th)))
        return (sw - width) // 2, 0, width, sh
    height = max(1, min(sh, round(sw * th / tw)))
    return 0, (sh - height) // 2, sw, height


def presentation_blit_regions(
    source: tuple[int, int],
    target: tuple[int, int],
    fit_mode: str,
    display_mode: str = "Half-SBS",
    input_size: tuple[int, int] | None = None,
) -> tuple[tuple[tuple[int, int, int, int], tuple[int, int, int, int]], ...]:
    """Resolve source/destination blits without mixing the two packed eyes.

    ``input_size`` is the original capture WxH (monitor or window) before
    packing, used so the left/right eye ratio stays dynamic with input
    (tex_w,tex_h in legacy viewer). When not provided, falls back to
    deriving eye size from packed ``source``.
    """
    sw, sh = source
    tw, th = target
    if min(sw, sh, tw, th) <= 0:
        return ()
    mode = normalize_display_fit_mode(fit_mode)
    full_source = (0, 0, sw, sh)
    full_target = (0, 0, tw, th)
    packed_mode = str(display_mode or "").strip().casefold().replace("_", "-")

    if packed_mode in {"half-sbs", "full-sbs", "half-tab", "full-tab"}:
        # Keep packed eyes separate; do not mix them in one blit.
        is_sbs = packed_mode.endswith("sbs")
        is_half = packed_mode.startswith("half-")
        if is_sbs:
            source_split = sw // 2
            encoded_eye_size = (source_split, sh)
            logical_eye_size = (source_split * (2 if is_half else 1), sh)
            source_origins = ((0, 0), (source_split, 0))
        else:
            source_split = sh // 2
            encoded_eye_size = (sw, source_split)
            logical_eye_size = (sw, source_split * (2 if is_half else 1))
            source_origins = ((0, 0), (0, source_split))

        # Local-mode SBS/TAB presentation. Per the updated spec, "contain"
        # (keep ratio complete) must center the capture view in each half and
        # expand its long side to fill that half's height (or width when the
        # capture is narrower). Stretch fills each half; cover crops.
        if mode == "stretch":
            half_w, half_h = (tw // 2, th) if is_sbs else (tw, th // 2)
            if is_sbs:
                destinations = ((0, 0, half_w, th), (half_w, 0, tw, th))
            else:
                destinations = ((0, 0, tw, half_h), (0, half_h, tw, th))
            regions = []
            for (ox, oy), dest in zip(source_origins, destinations):
                ew, eh = encoded_eye_size
                regions.append(((ox, oy, ox + ew, oy + eh), dest))
            return tuple(regions)
        if mode == "contain":
            half_w, half_h = (tw // 2, th) if is_sbs else (tw, th // 2)
            # Dynamic eye ratio from original capture (tex_w,tex_h).
            # Keep original W/2 etc. for Half, then fit long side to half
            # with black bars (limit to half complete, avoid zoom/crop).
            if input_size is not None:
                iw, ih = input_size
                if is_sbs:
                    eye_fit_size = (max(1, iw // 2), ih) if is_half else (iw, ih)
                else:
                    eye_fit_size = (iw, max(1, ih // 2)) if is_half else (iw, ih)
            else:
                eye_fit_size = encoded_eye_size
            x, y, w, h = fit_rect(eye_fit_size, (half_w, half_h))
            if is_sbs:
                left_dest = (x, y, x + w, y + h)
                right_dest = (half_w + x, y, half_w + x + w, y + h)
                destinations = (left_dest, right_dest)
            else:
                top_dest = (x, y, x + w, y + h)
                bottom_dest = (x, half_h + y, x + w, half_h + y + h)
                destinations = (top_dest, bottom_dest)
            regions = []
            for (ox, oy), dest in zip(source_origins, destinations):
                ew, eh = encoded_eye_size
                regions.append(((ox, oy, ox + ew, oy + eh), dest))
            return tuple(regions)

        # cover ("铺满"): short side expands to half, cropping long side,
        # keeping fixed eye ratios: FullSBS WxH, HalfSBS W/2xH, etc., dynamic
        # with original input (tex_w,tex_h). Map eye crop to packed eye coords.
        if mode == "cover":
            half_w, half_h = (tw // 2, th) if is_sbs else (tw, th // 2)
            if input_size is not None:
                iw, ih = input_size
                if is_sbs:
                    eye_size = (max(1, iw // 2), ih) if is_half else (iw, ih)
                else:
                    eye_size = (iw, max(1, ih // 2)) if is_half else (iw, ih)
            else:
                eye_size = encoded_eye_size
            cx_eye, cy_eye, cw_eye, ch_eye = _cover_crop_rect(
                eye_size, (half_w, half_h)
            )
            # Map eye crop to packed eye coords (encoded is stretched eye)
            if eye_size != encoded_eye_size:
                sx = encoded_eye_size[0] / eye_size[0] if eye_size[0] else 1.0
                sy = encoded_eye_size[1] / eye_size[1] if eye_size[1] else 1.0
                crop_x = int(round(cx_eye * sx))
                crop_y = int(round(cy_eye * sy))
                crop_w = int(round(cw_eye * sx))
                crop_h = int(round(ch_eye * sy))
            else:
                crop_x, crop_y, crop_w, crop_h = cx_eye, cy_eye, cw_eye, ch_eye
            if is_sbs:
                destinations = ((0, 0, half_w, th), (half_w, 0, tw, th))
            else:
                destinations = ((0, 0, tw, half_h), (0, half_h, tw, th))
            regions = []
            for (ox, oy), dest in zip(source_origins, destinations):
                regions.append((
                    (ox + crop_x, oy + crop_y, ox + crop_x + crop_w, oy + crop_y + crop_h),
                    dest,
                ))
            return tuple(regions)

        # fallback (should not reach for packed modes)
        crop_x, crop_y = 0, 0
        crop_w, crop_h = encoded_eye_size
        target_box = full_target
        tx0, ty0, tx1, ty1 = target_box
        if is_sbs:
            target_split = tx0 + (tx1 - tx0) // 2
            destination_regions = (
                (tx0, ty0, target_split, ty1),
                (target_split, ty0, tx1, ty1),
            )
        else:
            target_split = ty0 + (ty1 - ty0) // 2
            destination_regions = (
                (tx0, ty0, tx1, target_split),
                (tx0, target_split, tx1, ty1),
            )
        regions = []
        for (origin_x, origin_y), destination_rect in zip(
            source_origins, destination_regions
        ):
            regions.append((
                (
                    origin_x + crop_x,
                    origin_y + crop_y,
                    origin_x + crop_x + crop_w,
                    origin_y + crop_y + crop_h,
                ),
                destination_rect,
            ))
        return tuple(regions)

    if mode == "stretch":
        return ((full_source, full_target),)
    if mode == "contain":
        x, y, width, height = fit_rect(source, target)
        return ((full_source, (x, y, x + width, y + height)),)
    crop_x, crop_y, crop_w, crop_h = _cover_crop_rect(source, target)
    return (((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h), full_target),)


def frame_to_rgba_bytes(frame: Any) -> tuple[bytes, int, int]:
    """Convert a CHW/HWC torch or numpy frame to tightly packed RGBA8."""
    import numpy as np

    image = frame.detach() if hasattr(frame, "detach") else frame
    if bool(getattr(image, "is_cuda", False)):
        image = image.cpu()
    if hasattr(image, "numpy"):
        image = image.numpy()
    image = np.asarray(image)
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"unsupported local viewer frame shape: {image.shape!r}")
    # Prefer HWC when its trailing channel dimension is explicit.  Runtime
    # tensors are normally CHW with a wide final image dimension.
    if image.shape[-1] not in (1, 3, 4) and image.shape[0] in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    height, width, channels = image.shape
    if channels not in (1, 3, 4):
        raise ValueError(f"unsupported local viewer channels: {channels}")
    if image.dtype != np.uint8:
        image = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    if channels == 1:
        image = np.repeat(image, 3, axis=2)
    if channels == 3:
        image = np.concatenate((image, np.full((height, width, 1), 255, dtype=np.uint8)), axis=2)
    return np.ascontiguousarray(image).tobytes(), int(width), int(height)


def frame_to_cuda_rgba(frame: Any) -> Any | None:
    """Return an HxWx4 CUDA uint8 tensor, without a host round trip."""
    image = frame.detach() if hasattr(frame, "detach") else frame
    if not bool(getattr(image, "is_cuda", False)):
        return None
    try:
        import torch

        if image.ndim == 4:
            image = image[0]
        if image.ndim != 3:
            return None
        if (
            image.dtype == torch.float32
            and image.shape[0] == 3
            and image.shape[-1] not in (1, 3, 4)
        ):
            try:
                from stereo_runtime.output_triton import make_chw_rgb_to_hwc_rgba_u8

                return make_chw_rgb_to_hwc_rgba_u8(image.unsqueeze(0))
            except Exception:
                pass
        if image.shape[-1] not in (1, 3, 4) and image.shape[0] in (1, 3, 4):
            image = image.permute(1, 2, 0)
        if int(image.shape[-1]) not in (1, 3, 4):
            return None
        if image.dtype != torch.uint8:
            image = image.clamp(0.0, 1.0).mul(255.0).to(torch.uint8)
        if int(image.shape[-1]) == 1:
            image = image.expand(-1, -1, 3)
        if int(image.shape[-1]) == 3:
            alpha = torch.full(
                (*image.shape[:2], 1), 255, dtype=torch.uint8, device=image.device
            )
            image = torch.cat((image, alpha), dim=2)
        return image.contiguous()
    except Exception:
        return None


def depth_to_red_blue_rgb(depth: Any) -> Any:
    """Map normalized inverse depth to blue/cyan/green/yellow/red RGB."""
    import torch

    value = depth.float().clamp(0.0, 1.0)
    red = (4.0 * value - 2.0).clamp(0.0, 1.0)
    green = torch.minimum(4.0 * value, 4.0 - 4.0 * value).clamp(0.0, 1.0)
    blue = (2.0 - 4.0 * value).clamp(0.0, 1.0)
    return torch.cat((red, green, blue), dim=1)


def depth_to_effective_disparity(depth: Any, depth_strength: float) -> Any:
    """Center depth colors on strength while retaining maximum detail at 0.25."""
    center = max(0.0, min(0.5, float(depth_strength))) / 0.5
    depth_contrast = 4.0 * center * (1.0 - center)
    normalized_depth = depth.float().clamp(0.0, 1.0)
    return (center + (normalized_depth - 0.5) * depth_contrast).clamp(0.0, 1.0)


def effective_disparity_to_red_blue_rgb(disparity: Any) -> Any:
    """Linearly mix blue and red so the midpoint is purple."""
    import torch

    value = disparity.float().clamp(0.0, 1.0)
    return torch.cat((value, torch.zeros_like(value), 1.0 - value), dim=1)


def depth_preview_frame(result: Any) -> Any | None:
    """Return hot-reloaded depth strength on the effective-disparity color scale."""
    debug_info = getattr(result, "debug_info", None) or {}
    depth = debug_info.get("output_depth")
    if depth is None:
        depth = getattr(result, "depth", None)
    if depth is None:
        return None

    size = getattr(result, "output_eye_size", None)
    if isinstance(size, (tuple, list)) and len(size) == 2:
        width, height = int(size[0]), int(size[1])
    else:
        shape = tuple(int(value) for value in getattr(depth, "shape", ()))
        if len(shape) < 2:
            return None
        height, width = shape[-2], shape[-1]

    from stereo_runtime.output import match_depth

    matched = match_depth(depth, height, width)
    if int(matched.shape[1]) != 1:
        matched = matched[:, :1]
    depth_strength = float(debug_info.get("depth_strength", 1.0))
    effective_disparity = depth_to_effective_disparity(matched, depth_strength)
    return effective_disparity_to_red_blue_rgb(effective_disparity)


@dataclass(frozen=True, slots=True)
class VulkanLocalViewerConfig:
    title: str = "Desktop2Stereo Vulkan Viewer"
    monitor_index: int = 0
    fullscreen: bool = False
    capture_compatible_fullscreen: bool = False
    window_preview: bool = False
    preview_monitor_index: int | None = None
    manage_glfw_lifecycle: bool = True
    exclude_from_capture: bool = False
    cursor_passthrough: bool = False
    # Original capture size (tex_w,tex_h in legacy viewer) for dynamic eye ratio.
    # When set, HalfSBS uses W/2×H etc. based on this, not packed sw/sh.
    # Kept for startup fallback; per-frame size is queried dynamically.
    input_size: tuple[int, int] | None = None
    capture_mode: str = "Monitor"
    window_title: str | None = None
    vsync: bool = True
    show_fps: bool = False
    show_fps_provider: Callable[[], bool] | None = None
    display_mode: str = "Half-SBS"
    display_fit_mode: str = "contain"
    display_fit_mode_provider: Callable[[], str] | None = None
    display_fit_enabled: bool = True
    on_sbs_fps: Callable[[float, int], int] | None = None
    on_display_refresh_warning: Callable[[int, float], None] | None = None
    on_capture_refresh_warning: Callable[[int, int], None] | None = None
    on_breakdown_inc: Callable[[str, int | float], None] | None = None
    on_breakdown_add_time: Callable[[str, float], None] | None = None
    window_width: int = 1280
    window_height: int = 720


class _LocalInteropContext:
    """Small VulkanContext-compatible surface for CUDA external images.

    Local presentation needs a GLFW present queue, unlike OpenXR's presentation
    context.  This adapter deliberately exposes only the shared exporter API.
    """

    def __init__(self, owner: "VulkanLocalViewer") -> None:
        self.owner = owner
        self.vk = owner.vk
        self.device = owner.device
        self.physical_device = owner.physical_device
        self.queue_family_index = owner.queue_family
        self.compute_queue_family_index = owner.queue_family
        self._resources: set[int] = set()

    def register_external_image(self, resource: Any) -> None:
        self._resources.add(id(resource))

    def unregister_external_image(self, resource: Any) -> None:
        self._resources.discard(id(resource))

    def prepare_external_image_for_producer(self, resource: Any) -> int:
        """Prepare an exportable image for any GPU producer backend (HIP too)."""
        return self.prepare_external_image_for_cuda(resource)

    def prepare_external_image_for_cuda(self, resource: Any) -> int:
        """Transition once into GENERAL, the layout CUDA owns between frames."""
        vk, owner = self.vk, self.owner
        vk.vkWaitForFences(owner.device, 1, [owner.fence], True, 1_000_000_000)
        vk.vkResetFences(owner.device, 1, [owner.fence])
        cmd = owner.command_buffer
        vk.vkResetCommandBuffer(cmd, 0)
        vk.vkBeginCommandBuffer(
            cmd,
            vk.VkCommandBufferBeginInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
            ),
        )
        _TransferSource.transition_image(
            vk,
            cmd,
            resource.image,
            vk.VK_IMAGE_LAYOUT_UNDEFINED,
            vk.VK_IMAGE_LAYOUT_GENERAL,
        )
        vk.vkEndCommandBuffer(cmd)
        vk.vkQueueSubmit(
            owner.queue,
            1,
            [
                vk.VkSubmitInfo(
                    sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
                    commandBufferCount=1,
                    pCommandBuffers=[cmd],
                )
            ],
            owner.fence,
        )
        vk.vkWaitForFences(owner.device, 1, [owner.fence], True, 1_000_000_000)
        return 0


class VulkanLocalViewer:
    """Transfer-only Vulkan renderer for a GLFW surface."""

    def __init__(self, config: VulkanLocalViewerConfig) -> None:
        self.config = config
        self.glfw = self.vk = None
        self.window = self.instance = self.surface = None
        self.physical_device = self.device = self.queue = None
        self.queue_family = None
        self.command_pool = self.command_buffer = None
        self.swapchain = None
        self.swap_images: list[Any] = []
        self._swap_image_initialized: list[bool] = []
        self.extent = (0, 0)
        self.source_format: int | None = None
        self.image_available = self.render_finished = self.fence = None
        self._source: _TransferSource | None = None
        self._interop_extensions: tuple[str, ...] = ()
        self._target_monitor = None
        self._exclusive_fullscreen = False
        self._full_screen_exclusive_enabled = False
        self._full_screen_exclusive_requested = False
        self._full_screen_exclusive_acquired = False
        self._full_screen_exclusive_notice_reported = False
        self._surface_capabilities_owner = None
        self._win32_hwnd = 0
        self._win32_hmonitor = 0
        self._win32_user32 = None
        self._last_visibility_enforce = 0.0
        self._windowed_rect = (40, 40, config.window_width, config.window_height)
        self._fps_frames = 0
        self._fps_started = time.perf_counter()
        self._output_refresh_hz = 0
        self._input_refresh_hz = 0
        self._display_refresh_warning_reported = False
        self._capture_refresh_warning_reported = False
        self._presentation_geometry_reported = False
        self._last_presentation_geometry = None

    def initialize(self) -> None:
        import glfw
        import vulkan as vk

        self.glfw, self.vk = glfw, vk
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed for Vulkan local viewer")
        configure_glfw_window_hints(glfw, fullscreen=self.config.fullscreen)
        monitors = glfw.get_monitors() or []
        monitor = (
            monitors[resolve_glfw_monitor_index(self.config.monitor_index, glfw)]
            if monitors
            else None
        )
        self._target_monitor = monitor
        input_capture_index = (
            self.config.preview_monitor_index
            if self.config.preview_monitor_index is not None
            else self.config.monitor_index
        )
        if monitors:
            input_monitor = monitors[
                resolve_glfw_monitor_index(input_capture_index, glfw)
            ]
            input_mode = glfw.get_video_mode(input_monitor)
            self._input_refresh_hz = int(input_mode.refresh_rate)
        width, height = self.config.window_width, self.config.window_height
        x, y = 40, 40
        if monitor is not None:
            mx, my = glfw.get_monitor_pos(monitor)
            x, y = int(mx) + 40, int(my) + 40
            self._windowed_rect = (mx + 40, my + 40, width, height)
        if self.config.fullscreen and monitor is not None:
            mode = glfw.get_video_mode(monitor)
            self._output_refresh_hz = int(mode.refresh_rate)
            width, height = int(mode.size.width), int(mode.size.height)
            x, y = int(mx), int(my)
            self._exclusive_fullscreen = True
            glfw.window_hint(glfw.DECORATED, glfw.FALSE)
        self.window = glfw.create_window(
            width,
            height,
            self.config.title,
            None,
            None,
        )
        if not self.window:
            raise RuntimeError("could not create Vulkan local viewer window")
        glfw.set_window_pos(self.window, x, y)
        if self._exclusive_fullscreen:
            self._configure_persistent_fullscreen()
            glfw.poll_events()
        if self.config.exclude_from_capture:
            hide_window_from_capture(self.window)
        if self.config.cursor_passthrough:
            set_window_mouse_passthrough(self.window, True)
        glfw.set_key_callback(self.window, self._on_key)
        self._create_device()
        self._create_swapchain()
        self._create_sync()
        if self._exclusive_fullscreen:
            mode = glfw.get_video_mode(self._target_monitor)
            print(
                "[VulkanLocalViewer] Display fullscreen active on 2D display: "
                f"{int(mode.size.width)}x{int(mode.size.height)}@{int(mode.refresh_rate)}Hz "
                "(Alt+Enter)",
                flush=True,
            )

    def _on_key(self, _window: Any, key: int, _scancode: int, action: int, mods: int) -> None:
        if is_exclusive_fullscreen_toggle(key, action, mods, self.glfw):
            self._set_exclusive_fullscreen(not self._exclusive_fullscreen)

    def poll_events(self) -> None:
        self.glfw.poll_events()
        if self.glfw.window_should_close(self.window):
            raise StopIteration
        self._keep_fullscreen_visible()

    def _configure_persistent_fullscreen(self) -> None:
        if sys.platform != "win32" or self.window is None:
            return
        try:
            import ctypes
            from ctypes import wintypes

            self._win32_hwnd = int(self.glfw.get_win32_window(self.window))
            self._win32_user32 = ctypes.windll.user32
            self._win32_user32.SetWindowPos.argtypes = [
                wintypes.HWND,
                wintypes.HWND,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                wintypes.UINT,
            ]
            self._win32_user32.SetWindowPos.restype = wintypes.BOOL
            self._win32_user32.ShowWindowAsync.argtypes = [wintypes.HWND, ctypes.c_int]
            self._win32_user32.ShowWindowAsync.restype = wintypes.BOOL
            self._win32_user32.IsIconic.argtypes = [wintypes.HWND]
            self._win32_user32.IsIconic.restype = wintypes.BOOL
            self._win32_user32.IsWindowVisible.argtypes = [wintypes.HWND]
            self._win32_user32.IsWindowVisible.restype = wintypes.BOOL
            self._win32_user32.MonitorFromWindow.argtypes = [
                wintypes.HWND,
                wintypes.DWORD,
            ]
            self._win32_user32.MonitorFromWindow.restype = wintypes.HANDLE
            self._win32_user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
            self._win32_user32.GetWindowLongW.restype = ctypes.c_long
            self._win32_user32.SetWindowLongW.argtypes = [
                wintypes.HWND,
                ctypes.c_int,
                ctypes.c_long,
            ]
            self._win32_user32.SetWindowLongW.restype = ctypes.c_long
            style = self._win32_user32.GetWindowLongW(self._win32_hwnd, -16)
            style &= ~(0x00020000 | 0x00010000)
            self._win32_user32.SetWindowLongW(self._win32_hwnd, -16, style)
            extended_style = self._win32_user32.GetWindowLongW(
                self._win32_hwnd, -20
            )
            extended_style |= 0x00000008 | 0x00000080 | 0x08000000
            extended_style &= ~0x00040000
            if self.config.cursor_passthrough:
                extended_style |= 0x00000020  # WS_EX_TRANSPARENT for cursor passthrough
            self._win32_user32.SetWindowLongW(
                self._win32_hwnd, -20, extended_style
            )
            self._refresh_win32_monitor()
            self._set_win32_topmost(self._exclusive_fullscreen)
            if self.config.cursor_passthrough:
                set_window_mouse_passthrough(self.window, True)
        except Exception as exc:
            self._win32_hwnd = 0
            self._win32_hmonitor = 0
            self._win32_user32 = None
            print(
                "[VulkanLocalViewer] Persistent fullscreen unavailable: "
                f"{type(exc).__name__}",
                flush=True,
            )

    def _refresh_win32_monitor(self) -> int:
        if not self._win32_hwnd or self._win32_user32 is None:
            self._win32_hmonitor = 0
            return 0
        handle = self._win32_user32.MonitorFromWindow(self._win32_hwnd, 2)
        self._win32_hmonitor = int(handle or 0)
        return self._win32_hmonitor

    def _set_win32_topmost(self, enabled: bool) -> None:
        if not self._win32_hwnd or self._win32_user32 is None:
            return
        import ctypes

        insert_after = -1 if enabled else -2
        self._win32_user32.SetWindowPos(
            ctypes.c_void_p(self._win32_hwnd),
            ctypes.c_void_p(insert_after),
            0,
            0,
            0,
            0,
            0x0001 | 0x0002 | 0x0010 | 0x0020 | 0x0040 | 0x0200,
        )

    def _keep_fullscreen_visible(self) -> None:
        if not self._exclusive_fullscreen or not self._win32_hwnd:
            return
        now = time.perf_counter()
        if now - self._last_visibility_enforce < 0.1:
            return
        self._last_visibility_enforce = now
        user32 = self._win32_user32
        hwnd = self._win32_hwnd
        visible = bool(user32.IsWindowVisible(hwnd))
        iconic = bool(user32.IsIconic(hwnd))
        topmost = bool(user32.GetWindowLongW(hwnd, -20) & 0x00000008)
        if should_restore_persistent_fullscreen(
            self._exclusive_fullscreen, visible, iconic, topmost
        ):
            if iconic or not visible:
                user32.ShowWindowAsync(hwnd, 4)
            self._set_win32_topmost(True)

    def _set_exclusive_fullscreen(self, enabled: bool) -> None:
        glfw = self.glfw
        if self.window is None or self._target_monitor is None:
            return
        if enabled:
            if self._exclusive_fullscreen:
                return
            x, y = glfw.get_window_pos(self.window)
            width, height = glfw.get_window_size(self.window)
            self._windowed_rect = (int(x), int(y), int(width), int(height))
            mode = glfw.get_video_mode(self._target_monitor)
            monitor_x, monitor_y = glfw.get_monitor_pos(self._target_monitor)
            glfw.set_window_attrib(self.window, glfw.DECORATED, glfw.FALSE)
            glfw.set_window_pos(self.window, int(monitor_x), int(monitor_y))
            glfw.set_window_size(
                self.window, int(mode.size.width), int(mode.size.height)
            )
            self._exclusive_fullscreen = True
            if not self._win32_hwnd:
                self._configure_persistent_fullscreen()
            self._refresh_win32_monitor()
            self._set_win32_topmost(True)
            if self.config.cursor_passthrough:
                set_window_mouse_passthrough(self.window, True)
        else:
            if not self._exclusive_fullscreen:
                return
            self._exclusive_fullscreen = False
            self._set_win32_topmost(False)
            x, y, width, height = self._windowed_rect
            glfw.set_window_attrib(self.window, glfw.DECORATED, glfw.TRUE)
            glfw.set_window_pos(self.window, x, y)
            glfw.set_window_size(self.window, width, height)
            if self.config.cursor_passthrough:
                set_window_mouse_passthrough(self.window, True)
        if self.device is not None:
            self.recreate_swapchain()
        print(
            f"[VulkanLocalViewer] Display fullscreen: {'on' if enabled else 'off'} "
            "(Alt+Enter)",
            flush=True,
        )

    def _report_present_fps(self, fps: float, frame_count: int) -> int | None:
        capture_target = (
            self.config.on_sbs_fps(fps, frame_count)
            if self.config.on_sbs_fps is not None
            else None
        )
        if (
            not self._display_refresh_warning_reported
            and self._exclusive_fullscreen
            and display_refresh_warning_needed(self._output_refresh_hz, fps)
        ):
            self._display_refresh_warning_reported = True
            callback = self.config.on_display_refresh_warning
            if callback is not None:
                callback(self._output_refresh_hz, fps)
        if (
            not self._capture_refresh_warning_reported
            and capture_target is not None
            and capture_refresh_warning_needed(
                self._input_refresh_hz, capture_target
            )
        ):
            self._capture_refresh_warning_reported = True
            callback = self.config.on_capture_refresh_warning
            if callback is not None:
                callback(self._input_refresh_hz, int(capture_target))
        show_fps = (
            bool(self.config.show_fps_provider())
            if self.config.show_fps_provider is not None
            else self.config.show_fps
        )
        if not show_fps:
            return capture_target
        target_text = (
            f" capture_target={capture_target}" if capture_target is not None else ""
        )
        print(
            f"[VulkanLocalViewer] Present FPS: {fps:.1f}{target_text}",
            flush=True,
        )
        if not self._exclusive_fullscreen:
            self.glfw.set_window_title(
                self.window, f"{self.config.title} | {fps:.1f} FPS"
            )
        return capture_target

    def _create_device(self) -> None:
        vk, glfw = self.vk, self.glfw
        extensions = list(glfw.get_required_instance_extensions() or [])
        available_instance_extensions = {
            prop.extensionName for prop in vk.vkEnumerateInstanceExtensionProperties(None)
        }
        request_full_screen_exclusive = should_request_full_screen_exclusive(
            self.config.fullscreen,
            self.config.capture_compatible_fullscreen,
        )
        if (
            request_full_screen_exclusive
            and sys.platform == "win32"
            and FULL_SCREEN_EXCLUSIVE_INSTANCE_EXTENSION
            in available_instance_extensions
            and FULL_SCREEN_EXCLUSIVE_INSTANCE_EXTENSION not in extensions
        ):
            extensions.append(FULL_SCREEN_EXCLUSIVE_INSTANCE_EXTENSION)
        app_info = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName=self.config.title,
            applicationVersion=1,
            pEngineName="Desktop2Stereo",
            engineVersion=1,
            apiVersion=vk.VK_API_VERSION_1_0,
        )
        self.instance = vk.vkCreateInstance(vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app_info,
            enabledExtensionCount=len(extensions),
            ppEnabledExtensionNames=extensions,
        ), None)
        surface_ptr = vk.ffi.new("VkSurfaceKHR *")
        result = glfw.create_window_surface(self.instance, self.window, None, surface_ptr)
        if result != vk.VK_SUCCESS:
            raise RuntimeError(f"glfwCreateWindowSurface failed ({result})")
        self.surface = surface_ptr[0]
        for candidate in vk.vkEnumeratePhysicalDevices(self.instance):
            available = {prop.extensionName for prop in vk.vkEnumerateDeviceExtensionProperties(candidate, None)}
            if "VK_KHR_swapchain" not in available:
                continue
            for index, props in enumerate(vk.vkGetPhysicalDeviceQueueFamilyProperties(candidate)):
                if props.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT and self._surface_support(candidate, index):
                    self.physical_device, self.queue_family = candidate, index
                    break
            if self.physical_device is not None:
                break
        if self.physical_device is None:
            raise RuntimeError("no Vulkan graphics/present queue available for local viewer")
        available = {
            prop.extensionName
            for prop in vk.vkEnumerateDeviceExtensionProperties(self.physical_device, None)
        }
        interop = (
            *VulkanExportableImage.required_device_extensions(),
            *VulkanExportableSemaphore.required_device_extensions(),
        )
        self._interop_extensions = tuple(name for name in interop if name in available)
        exclusive_capable, exclusive_missing = full_screen_exclusive_capability(
            available_instance_extensions,
            available,
        )
        self._full_screen_exclusive_enabled = bool(
            request_full_screen_exclusive and exclusive_capable
        )
        direct_display_supported, direct_display_missing = direct_display_capability(
            available_instance_extensions,
            available,
        )
        if not direct_display_supported:
            if self.config.fullscreen and self._full_screen_exclusive_enabled:
                fallback_mode = "Win32 Vulkan full-screen exclusive"
            elif self.config.fullscreen:
                fallback_mode = "persistent borderless Vulkan fullscreen"
            else:
                fallback_mode = "Vulkan window preview"
            print(
                "[VulkanLocalViewer] Vulkan raw direct-display unavailable: missing "
                f"{', '.join(direct_display_missing)}; using {fallback_mode}",
                flush=True,
            )
        if self.config.fullscreen and self.config.capture_compatible_fullscreen:
            print(
                "[VulkanLocalViewer] Capture-compatible DWM borderless fullscreen "
                "active; VK_EXT_full_screen_exclusive disabled",
                flush=True,
            )
        elif self.config.fullscreen and not self._full_screen_exclusive_enabled:
            print(
                "[VulkanLocalViewer] Win32 Vulkan full-screen exclusive unavailable: "
                f"missing {', '.join(exclusive_missing)}; using persistent borderless "
                "fullscreen",
                flush=True,
            )
        enabled_extensions = ["VK_KHR_swapchain", *self._interop_extensions]
        if self._full_screen_exclusive_enabled:
            enabled_extensions.append(FULL_SCREEN_EXCLUSIVE_DEVICE_EXTENSION)
        queue_info = vk.VkDeviceQueueCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=self.queue_family,
            queueCount=1,
            pQueuePriorities=[1.0],
        )
        self.device = vk.vkCreateDevice(self.physical_device, vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[queue_info],
            enabledExtensionCount=len(enabled_extensions),
            ppEnabledExtensionNames=enabled_extensions,
        ), None)
        self.queue = vk.vkGetDeviceQueue(self.device, self.queue_family, 0)
        self.command_pool = vk.vkCreateCommandPool(self.device, vk.VkCommandPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            queueFamilyIndex=self.queue_family,
            flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        ), None)
        self.command_buffer = vk.vkAllocateCommandBuffers(self.device, vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self.command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        ))[0]

    def _instance_function(self, name: bytes, signature: str) -> Any:
        proc = self.vk.lib.vkGetInstanceProcAddr(self.instance, name)
        if proc == self.vk.ffi.NULL:
            raise RuntimeError(f"Vulkan instance function unavailable: {name.decode()}")
        return self.vk.ffi.cast(signature, proc)

    def _device_function(self, name: bytes, signature: str) -> Any:
        proc = self.vk.lib.vkGetDeviceProcAddr(self.device, name)
        if proc == self.vk.ffi.NULL:
            raise RuntimeError(f"Vulkan device function unavailable: {name.decode()}")
        return self.vk.ffi.cast(signature, proc)

    def _surface_support(self, physical_device: Any, queue_family: int) -> bool:
        output = self.vk.ffi.new("VkBool32 *")
        function = self._instance_function(
            b"vkGetPhysicalDeviceSurfaceSupportKHR",
            "VkResult(*)(VkPhysicalDevice, uint32_t, VkSurfaceKHR, VkBool32 *)",
        )
        if int(function(physical_device, queue_family, self.surface, output)) != int(self.vk.VK_SUCCESS):
            return False
        return bool(output[0])

    def _full_screen_exclusive_chain(self) -> Any | None:
        if (
            not self._full_screen_exclusive_enabled
            or not self._exclusive_fullscreen
            or not self._refresh_win32_monitor()
        ):
            return None
        vk = self.vk
        win32_info = vk.VkSurfaceFullScreenExclusiveWin32InfoEXT(
            hmonitor=vk.ffi.cast("HMONITOR", self._win32_hmonitor),
        )
        return vk.VkSurfaceFullScreenExclusiveInfoEXT(
            pNext=win32_info,
            fullScreenExclusive=vk.VK_FULL_SCREEN_EXCLUSIVE_APPLICATION_CONTROLLED_EXT,
        )

    def _query_surface_capabilities(self) -> tuple[Any, Any | None]:
        vk = self.vk
        exclusive_chain = self._full_screen_exclusive_chain()
        if exclusive_chain is not None:
            support = vk.VkSurfaceCapabilitiesFullScreenExclusiveEXT()
            caps2 = vk.VkSurfaceCapabilities2KHR(pNext=support)
            surface_info = vk.VkPhysicalDeviceSurfaceInfo2KHR(
                pNext=exclusive_chain,
                surface=self.surface,
            )
            try:
                get_caps2 = self._instance_function(
                    b"vkGetPhysicalDeviceSurfaceCapabilities2KHR",
                    "VkResult(*)(VkPhysicalDevice, const VkPhysicalDeviceSurfaceInfo2KHR *, VkSurfaceCapabilities2KHR *)",
                )
                result = int(
                    get_caps2(
                        self.physical_device,
                        vk.ffi.addressof(surface_info),
                        vk.ffi.addressof(caps2),
                    )
                )
                if result == int(vk.VK_SUCCESS) and bool(
                    support.fullScreenExclusiveSupported
                ):
                    self._full_screen_exclusive_requested = True
                    selected_caps = caps2.surfaceCapabilities
                    if (
                        int(selected_caps.currentExtent.width) > 0
                        and int(selected_caps.currentExtent.height) > 0
                    ):
                        self._surface_capabilities_owner = (
                            caps2,
                            support,
                            surface_info,
                            exclusive_chain,
                        )
                        return selected_caps, exclusive_chain
                    # Some Win32 drivers briefly return a zero extent while the
                    # newly shown fullscreen window is settling. Keep the
                    # exclusive policy but use the legacy surface extent.
                    self.glfw.poll_events()
                    caps = vk.ffi.new("VkSurfaceCapabilitiesKHR *")
                    get_caps = self._instance_function(
                        b"vkGetPhysicalDeviceSurfaceCapabilitiesKHR",
                        "VkResult(*)(VkPhysicalDevice, VkSurfaceKHR, VkSurfaceCapabilitiesKHR *)",
                    )
                    if int(
                        get_caps(self.physical_device, self.surface, caps)
                    ) == int(vk.VK_SUCCESS):
                        self._surface_capabilities_owner = (
                            caps,
                            caps2,
                            support,
                            surface_info,
                            exclusive_chain,
                        )
                        return caps[0], exclusive_chain
                reason = (
                    f"surface query failed ({result})"
                    if result != int(vk.VK_SUCCESS)
                    else "selected display does not support exclusive access"
                )
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
            if not self._full_screen_exclusive_notice_reported:
                self._full_screen_exclusive_notice_reported = True
                print(
                    "[VulkanLocalViewer] Win32 Vulkan full-screen exclusive "
                    f"unavailable for this surface: {reason}; using persistent "
                    "borderless fullscreen",
                    flush=True,
                )

        self._full_screen_exclusive_requested = False
        caps = vk.ffi.new("VkSurfaceCapabilitiesKHR *")
        get_caps = self._instance_function(
            b"vkGetPhysicalDeviceSurfaceCapabilitiesKHR",
            "VkResult(*)(VkPhysicalDevice, VkSurfaceKHR, VkSurfaceCapabilitiesKHR *)",
        )
        if int(get_caps(self.physical_device, self.surface, caps)) != int(
            vk.VK_SUCCESS
        ):
            raise RuntimeError("could not query Vulkan surface capabilities")
        self._surface_capabilities_owner = caps
        return caps[0], None

    def _create_swapchain(self) -> None:
        vk = self.vk
        caps, exclusive_chain = self._query_surface_capabilities()
        get_formats = self._instance_function(b"vkGetPhysicalDeviceSurfaceFormatsKHR", "VkResult(*)(VkPhysicalDevice, VkSurfaceKHR, uint32_t *, VkSurfaceFormatKHR *)")
        format_count = vk.ffi.new("uint32_t *")
        get_formats(self.physical_device, self.surface, format_count, vk.ffi.NULL)
        formats = vk.ffi.new("VkSurfaceFormatKHR[]", int(format_count[0]))
        get_formats(self.physical_device, self.surface, format_count, formats)
        get_modes = self._instance_function(b"vkGetPhysicalDeviceSurfacePresentModesKHR", "VkResult(*)(VkPhysicalDevice, VkSurfaceKHR, uint32_t *, VkPresentModeKHR *)")
        mode_count = vk.ffi.new("uint32_t *")
        get_modes(self.physical_device, self.surface, mode_count, vk.ffi.NULL)
        modes = vk.ffi.new("VkPresentModeKHR[]", int(mode_count[0]))
        get_modes(self.physical_device, self.surface, mode_count, modes)
        fmt, is_srgb = choose_srgb_surface_format(formats, vk)
        self.source_format = getattr(
            vk,
            LOCAL_VIEWER_SOURCE_FORMAT if is_srgb else "VK_FORMAT_R8G8B8A8_UNORM",
        )
        if not is_srgb:
            print(
                "[VulkanLocalViewer] Surface has no sRGB format; using UNORM "
                "source fallback to preserve display-referred bytes",
                flush=True,
            )
        present_mode = choose_present_mode(
            modes,
            vk,
            vsync=self.config.vsync,
            full_screen_exclusive=self._full_screen_exclusive_requested,
        )
        if caps.currentExtent.width != 0xFFFFFFFF:
            extent = caps.currentExtent
        else:
            width, height = self.glfw.get_framebuffer_size(self.window)
            extent = vk.VkExtent2D(
                width=max(caps.minImageExtent.width, min(caps.maxImageExtent.width, width)),
                height=max(caps.minImageExtent.height, min(caps.maxImageExtent.height, height)),
            )
        count = max(2, caps.minImageCount + 1)
        if caps.maxImageCount:
            count = min(count, caps.maxImageCount)
        info = vk.VkSwapchainCreateInfoKHR(
            sType=vk.VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            pNext=exclusive_chain,
            surface=self.surface,
            minImageCount=count,
            imageFormat=fmt.format,
            imageColorSpace=fmt.colorSpace,
            imageExtent=extent,
            imageArrayLayers=1,
            imageUsage=vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            imageSharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            preTransform=caps.currentTransform,
            compositeAlpha=vk.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            presentMode=present_mode,
            clipped=True,
            oldSwapchain=vk.ffi.NULL,
        )
        output = vk.ffi.new("VkSwapchainKHR *")
        create = self._device_function(b"vkCreateSwapchainKHR", "VkResult(*)(VkDevice, const VkSwapchainCreateInfoKHR *, const VkAllocationCallbacks *, VkSwapchainKHR *)")
        result = int(create(self.device, vk.ffi.addressof(info), vk.ffi.NULL, output))
        if result != int(vk.VK_SUCCESS):
            raise RuntimeError(
                f"could not create Vulkan local-viewer swapchain ({result})"
            )
        self.swapchain = output[0]
        get_images = self._device_function(b"vkGetSwapchainImagesKHR", "VkResult(*)(VkDevice, VkSwapchainKHR, uint32_t *, VkImage *)")
        image_count = vk.ffi.new("uint32_t *")
        get_images(self.device, self.swapchain, image_count, vk.ffi.NULL)
        images = vk.ffi.new("VkImage[]", int(image_count[0]))
        get_images(self.device, self.swapchain, image_count, images)
        self.swap_images = list(images)
        self._swap_image_initialized = [False] * len(self.swap_images)
        self.extent = int(extent.width), int(extent.height)
        self._acquire_full_screen_exclusive()

    def _acquire_full_screen_exclusive(self) -> bool:
        if not self._full_screen_exclusive_requested or self.swapchain is None:
            self._full_screen_exclusive_acquired = False
            return False
        try:
            acquire = self._device_function(
                b"vkAcquireFullScreenExclusiveModeEXT",
                "VkResult(*)(VkDevice, VkSwapchainKHR)",
            )
            result = int(acquire(self.device, self.swapchain))
            if result == int(self.vk.VK_SUCCESS):
                self._full_screen_exclusive_acquired = True
                print(
                    "[VulkanLocalViewer] Win32 Vulkan full-screen exclusive active",
                    flush=True,
                )
                return True
            reason = f"Vulkan result {result}"
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
        self._full_screen_exclusive_acquired = False
        print(
            "[VulkanLocalViewer] Win32 Vulkan full-screen exclusive acquisition "
            f"failed: {reason}; continuing with persistent borderless fullscreen",
            flush=True,
        )
        return False

    def _release_full_screen_exclusive(self) -> None:
        if not self._full_screen_exclusive_acquired or self.swapchain is None:
            self._full_screen_exclusive_acquired = False
            return
        try:
            release = self._device_function(
                b"vkReleaseFullScreenExclusiveModeEXT",
                "VkResult(*)(VkDevice, VkSwapchainKHR)",
            )
            result = int(release(self.device, self.swapchain))
            if result != int(self.vk.VK_SUCCESS):
                print(
                    "[VulkanLocalViewer] Win32 Vulkan full-screen exclusive release "
                    f"returned {result}",
                    flush=True,
                )
        except Exception as exc:
            print(
                "[VulkanLocalViewer] Win32 Vulkan full-screen exclusive release "
                f"failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
        finally:
            self._full_screen_exclusive_acquired = False

    def _destroy_swapchain(self) -> None:
        if self.swapchain is not None:
            self._release_full_screen_exclusive()
            self._device_function(
                b"vkDestroySwapchainKHR",
                "void(*)(VkDevice, VkSwapchainKHR, const VkAllocationCallbacks *)",
            )(self.device, self.swapchain, self.vk.ffi.NULL)
        self.swapchain = None
        self.swap_images = []
        self._swap_image_initialized = []
        self._full_screen_exclusive_requested = False

    def recreate_swapchain(self) -> bool:
        """Rebuild an out-of-date window swapchain without touching CUDA images."""
        width, height = self.glfw.get_framebuffer_size(self.window)
        if int(width) <= 0 or int(height) <= 0:
            return False
        self.vk.vkDeviceWaitIdle(self.device)
        self._destroy_swapchain()
        self._create_swapchain()
        print(
            f"[VulkanLocalViewer] Swapchain recreated: {self.extent[0]}x{self.extent[1]}",
            flush=True,
        )
        return True

    def is_swapchain_recreate_result(self, result: int) -> bool:
        return int(result) in {
            int(getattr(self.vk, "VK_ERROR_OUT_OF_DATE_KHR", -1000001004)),
            int(getattr(self.vk, "VK_SUBOPTIMAL_KHR", 1000001003)),
            int(
                getattr(
                    self.vk,
                    "VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT",
                    -1000255000,
                )
            ),
        }

    def is_swapchain_out_of_date(self, result: int) -> bool:
        return int(result) == int(
            getattr(self.vk, "VK_ERROR_OUT_OF_DATE_KHR", -1000001004)
        )

    def _create_sync(self) -> None:
        vk = self.vk
        sem_info = vk.VkSemaphoreCreateInfo(sType=vk.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO)
        self.image_available = vk.vkCreateSemaphore(self.device, sem_info, None)
        self.render_finished = vk.vkCreateSemaphore(self.device, sem_info, None)
        self.fence = vk.vkCreateFence(self.device, vk.VkFenceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            flags=vk.VK_FENCE_CREATE_SIGNALED_BIT,
        ), None)

    def present(self, frame: Any) -> None:
        if self.window is None:
            raise RuntimeError("Vulkan local viewer has not been initialized")
        self.poll_events()
        cuda_rgba = frame_to_cuda_rgba(frame)
        if cuda_rgba is not None:
            height, width = (int(cuda_rgba.shape[0]), int(cuda_rgba.shape[1]))
        else:
            pixels, width, height = frame_to_rgba_bytes(frame)
        if (
            self._source is None
            or self._source.size != (width, height)
            or self._source.format != self.source_format
        ):
            if self._source is not None:
                self._source.close()
            self._source = _TransferSource(self, width, height)
        if not self._source.present(cuda_rgba if cuda_rgba is not None else pixels):
            return
        now = time.perf_counter()
        self._fps_frames += 1
        fps = present_fps_if_due(self._fps_frames, now - self._fps_started)
        if fps is not None:
            self._report_present_fps(fps, self._fps_frames)
            self._fps_frames = 0
            self._fps_started = now

    def close(self) -> None:
        try:
            if self.device is not None:
                self.vk.vkDeviceWaitIdle(self.device)
                if self._source is not None:
                    self._source.close()
                for semaphore in (self.image_available, self.render_finished):
                    if semaphore is not None:
                        self.vk.vkDestroySemaphore(self.device, semaphore, None)
                if self.fence is not None:
                    self.vk.vkDestroyFence(self.device, self.fence, None)
                if self.command_pool is not None:
                    self.vk.vkDestroyCommandPool(self.device, self.command_pool, None)
                self._destroy_swapchain()
                self.vk.vkDestroyDevice(self.device, None)
        finally:
            if self.surface is not None and self.instance is not None:
                self._instance_function(b"vkDestroySurfaceKHR", "void(*)(VkInstance, VkSurfaceKHR, const VkAllocationCallbacks *)")(self.instance, self.surface, self.vk.ffi.NULL)
            if self.instance is not None:
                self.vk.vkDestroyInstance(self.instance, None)
            if self.window is not None:
                self.glfw.destroy_window(self.window)
            if self.glfw is not None and self.config.manage_glfw_lifecycle:
                self.glfw.terminate()
            self.window = None


class _TransferSource:
    def __init__(self, owner: VulkanLocalViewer, width: int, height: int) -> None:
        self.owner, self.size = owner, (width, height)
        self.format = int(owner.source_format)
        self.capacity = width * height * 4
        self.buffer = self.memory = self.image = self.image_memory = None
        self._image_initialized = False
        self._interop_context: _LocalInteropContext | None = None
        self._external_image: VulkanExportableImage | None = None
        self._cuda_ready: VulkanExportableSemaphore | None = None
        self._cuda_importer: CudaVulkanImageImporter | None = None
        self._cuda_active = False
        self._slow_present_count = 0
        self._create()

    def _memory_type(self, bits: int, required: int) -> int:
        props = self.owner.vk.vkGetPhysicalDeviceMemoryProperties(self.owner.physical_device)
        for index, item in enumerate(props.memoryTypes):
            if bits & (1 << index) and (item.propertyFlags & required) == required:
                return index
        raise RuntimeError("no compatible Vulkan memory type for local viewer")

    def _create(self) -> None:
        vk, device = self.owner.vk, self.owner.device
        self.buffer = vk.vkCreateBuffer(device, vk.VkBufferCreateInfo(sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, size=self.capacity, usage=vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT, sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE), None)
        req = vk.vkGetBufferMemoryRequirements(device, self.buffer)
        self.memory = vk.vkAllocateMemory(device, vk.VkMemoryAllocateInfo(sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, allocationSize=req.size, memoryTypeIndex=self._memory_type(req.memoryTypeBits, vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)), None)
        vk.vkBindBufferMemory(device, self.buffer, self.memory, 0)
        width, height = self.size
        source_format = self.format
        self.image = vk.vkCreateImage(device, vk.VkImageCreateInfo(sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, imageType=vk.VK_IMAGE_TYPE_2D, format=source_format, extent=vk.VkExtent3D(width=width, height=height, depth=1), mipLevels=1, arrayLayers=1, samples=vk.VK_SAMPLE_COUNT_1_BIT, tiling=vk.VK_IMAGE_TILING_OPTIMAL, usage=vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT | vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT, sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE, initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED), None)
        req = vk.vkGetImageMemoryRequirements(device, self.image)
        self.image_memory = vk.vkAllocateMemory(device, vk.VkMemoryAllocateInfo(sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, allocationSize=req.size, memoryTypeIndex=self._memory_type(req.memoryTypeBits, vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)), None)
        vk.vkBindImageMemory(device, self.image, self.image_memory, 0)

        required = {
            *VulkanExportableImage.required_device_extensions(),
            *VulkanExportableSemaphore.required_device_extensions(),
        }
        if required.issubset(set(self.owner._interop_extensions)):
            try:
                self._interop_context = _LocalInteropContext(self.owner)
                self._external_image = VulkanExportableImage(
                    self._interop_context,
                    width,
                    height,
                    label="local-viewer-cuda-source",
                    format=source_format,
                )
                self._cuda_ready = VulkanExportableSemaphore(
                    self._interop_context,
                    label="local-viewer-cuda-ready",
                )
                is_rocm = False
                try:
                    import torch

                    is_rocm = bool(getattr(torch.version, "hip", None))
                    if is_rocm:
                        from viewer.rocm_vulkan_interop import (
                            RocmVulkanImageImporter,
                        )

                        self._cuda_importer = RocmVulkanImageImporter()
                    else:
                        self._cuda_importer = CudaVulkanImageImporter()
                except Exception:
                    self._cuda_importer = CudaVulkanImageImporter()
                # Imports and establishes GENERAL once, before any frame is sent.
                self._cuda_importer.register_slot(self._external_image)
                self._cuda_importer.register_semaphore(self._cuda_ready)
                self._cuda_active = True
                print(
                    "[VulkanLocalViewer] "
                    + ("ROCm" if is_rocm else "CUDA")
                    + " external-image zero-copy active",
                    flush=True,
                )
            except Exception as exc:
                self._disable_cuda_interop(exc)
        else:
            print(
                "[VulkanLocalViewer] CUDA external-image zero-copy unavailable: "
                "required Vulkan external-memory/semaphore extensions missing",
                flush=True,
            )

    def _disable_cuda_interop(
        self, reason: Exception | str, *, announce: bool = True
    ) -> None:
        if announce and (self._cuda_active or self._external_image is not None):
            detail = str(reason)
            print(
                "[VulkanLocalViewer] CUDA external-image zero-copy unavailable: "
                f"{type(reason).__name__}: {detail}" if isinstance(reason, Exception)
                else f"[VulkanLocalViewer] CUDA external-image zero-copy unavailable: {detail}",
                flush=True,
            )
        self._cuda_active = False
        if self._cuda_importer is not None:
            self._cuda_importer.close()
        if self._cuda_ready is not None:
            self._cuda_ready.close()
        if self._external_image is not None:
            self._external_image.close()
        self._cuda_ready = self._external_image = self._cuda_importer = None

    @staticmethod
    def transition_image(vk: Any, cmd: Any, image: Any, old: int, new: int) -> None:
        barrier = vk.VkImageMemoryBarrier(sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, oldLayout=old, newLayout=new, srcAccessMask=0, dstAccessMask=0, image=image, subresourceRange=vk.VkImageSubresourceRange(aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT, baseMipLevel=0, levelCount=1, baseArrayLayer=0, layerCount=1))
        vk.vkCmdPipelineBarrier(cmd, vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, None, 0, None, 1, [barrier])

    def _transition(self, cmd: Any, image: Any, old: int, new: int) -> None:
        self.transition_image(self.owner.vk, cmd, image, old, new)

    def present(self, pixels: Any) -> bool:
        vk, o = self.owner.vk, self.owner
        frame_started = time.perf_counter()
        stage_started = frame_started
        fence_value = vk.vkWaitForFences(
            o.device, 1, [o.fence], True, 1_000_000_000
        )
        fence_result = int(vk.VK_SUCCESS if fence_value is None else fence_value)
        fence_ms = (time.perf_counter() - stage_started) * 1000.0
        index_output = vk.ffi.new("uint32_t *")
        acquire = o._device_function(b"vkAcquireNextImageKHR", "VkResult(*)(VkDevice, VkSwapchainKHR, uint64_t, VkSemaphore, VkFence, uint32_t *)")
        stage_started = time.perf_counter()
        acquire_result = int(acquire(o.device, o.swapchain, 1_000_000_000, o.image_available, vk.ffi.NULL, index_output))
        acquire_ms = (time.perf_counter() - stage_started) * 1000.0
        if o.is_swapchain_out_of_date(acquire_result):
            o.recreate_swapchain()
            return False
        recreate_after_present = o.is_swapchain_recreate_result(acquire_result)
        if acquire_result != int(vk.VK_SUCCESS) and not recreate_after_present:
            raise RuntimeError(f"Vulkan local-viewer acquire failed ({acquire_result})")
        index = int(index_output[0])
        vk.vkResetFences(o.device, 1, [o.fence])

        cuda_source = self._cuda_active and bool(getattr(pixels, "is_cuda", False))
        stage_started = time.perf_counter()
        if cuda_source:
            try:
                self._cuda_importer.copy_tensor(pixels, self._external_image)
                self._cuda_importer.signal_semaphore(self._cuda_ready)
                # HIP (ROCm) async ops above must complete before the Vulkan
                # submit reads the imported external image. Synchronize on the
                # HIP stream so the next depth-frame stream sync (torch
                # cuda.synchronize) does not hang waiting on a surface the
                # driver left in-flight. CUDA importer exposes no synchronize
                # and keeps its established behavior.
                sync = getattr(self._cuda_importer, "synchronize", None)
                if callable(sync):
                    sync()
            except Exception as exc:
                self._disable_cuda_interop(exc)
                cuda_source = False
                pixels, _width, _height = frame_to_rgba_bytes(pixels)
        if not cuda_source:
            if not isinstance(pixels, (bytes, bytearray, memoryview)):
                # CUDA/ROCm interop unavailable: the caller may hand us a GPU
                # tensor; convert it to tightly packed RGBA8 host bytes first.
                pixels, _width, _height = frame_to_rgba_bytes(pixels)
            mapped = vk.vkMapMemory(o.device, self.memory, 0, self.capacity, 0)
            # PyVulkan's vkMapMemory already returns a writable cffi buffer;
            # wrapping it again in ffi.buffer() fails with a TypeError.
            mapped[:] = pixels
            vk.vkUnmapMemory(o.device, self.memory)
        upload_ms = (time.perf_counter() - stage_started) * 1000.0
        stage_started = time.perf_counter()
        cmd = o.command_buffer
        vk.vkResetCommandBuffer(cmd, 0)
        vk.vkBeginCommandBuffer(cmd, vk.VkCommandBufferBeginInfo(sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO))
        source_image = self.image
        if cuda_source:
            source_image = self._external_image.image
            self._transition(cmd, source_image, vk.VK_IMAGE_LAYOUT_GENERAL, vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
        else:
            self._transition(cmd, self.image, vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL if self._image_initialized else vk.VK_IMAGE_LAYOUT_UNDEFINED, vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
            vk.vkCmdCopyBufferToImage(cmd, self.buffer, self.image, vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, [vk.VkBufferImageCopy(bufferOffset=0, bufferRowLength=0, bufferImageHeight=0, imageSubresource=vk.VkImageSubresourceLayers(aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT, mipLevel=0, baseArrayLayer=0, layerCount=1), imageOffset=vk.VkOffset3D(x=0, y=0, z=0), imageExtent=vk.VkExtent3D(width=self.size[0], height=self.size[1], depth=1))])
            self._transition(cmd, self.image, vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
        target = o.swap_images[index]
        target_old_layout = (
            vk.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
            if o._swap_image_initialized[index]
            else vk.VK_IMAGE_LAYOUT_UNDEFINED
        )
        self._transition(cmd, target, target_old_layout, vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
        clear = vk.VkClearColorValue(float32=[0.0, 0.0, 0.0, 1.0])
        vk.vkCmdClearColorImage(cmd, target, vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, clear, 1, [vk.VkImageSubresourceRange(aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT, baseMipLevel=0, levelCount=1, baseArrayLayer=0, layerCount=1)])
        fit_mode = "contain"
        if o.config.display_fit_enabled and o.config.display_fit_mode_provider is not None:
            try:
                fit_mode = o.config.display_fit_mode_provider()
            except Exception:
                fit_mode = o.config.display_fit_mode
        elif o.config.display_fit_enabled:
            fit_mode = o.config.display_fit_mode
        fit_mode = normalize_display_fit_mode(fit_mode)
        # Dynamic tex_w,tex_h: query current window/monitor size per frame
        # so input aspect changes (e.g., window resized) are reflected.
        dyn_input_size = o.config.input_size
        try:
            cap_mode = str(getattr(o.config, "capture_mode", "") or "").strip()
            if cap_mode.casefold() == "window":
                title = str(getattr(o.config, "window_title", "") or "").strip()
                if title and sys.platform == "win32":
                    try:
                        import win32gui

                        hwnd = win32gui.FindWindow(None, title)
                        if hwnd:
                            _, _, w, h = win32gui.GetClientRect(hwnd)
                            if w > 0 and h > 0:
                                dyn_input_size = (int(w), int(h))
                    except Exception:
                        pass
            elif cap_mode:
                # Monitor mode: use current monitor size (may change with resolution)
                try:
                    from utils.display import get_monitor_size

                    # o.config.monitor_index is the stereo output monitor; input monitor
                    # is preview_monitor_index or monitor_index depending on config
                    inp_idx = int(getattr(o.config, "preview_monitor_index", 0) or 0)
                    if inp_idx <= 0:
                        inp_idx = int(getattr(o.config, "monitor_index", 0) or 0)
                    if inp_idx > 0:
                        dyn_input_size = get_monitor_size(inp_idx)
                except Exception:
                    pass
        except Exception:
            pass
        regions = presentation_blit_regions(
            self.size,
            o.extent,
            fit_mode,
            o.config.display_mode,
            input_size=dyn_input_size,
        )
        geometry = (fit_mode, o.config.display_mode, self.size, o.extent, regions)
        if geometry != o._last_presentation_geometry:
            prefix = "First-frame" if o._last_presentation_geometry is None else "Updated"
            o._last_presentation_geometry = geometry
            o._presentation_geometry_reported = True
            print(
                f"[VulkanLocalViewer] {prefix} presentation geometry: "
                f"fit_mode={fit_mode} display_mode={o.config.display_mode} "
                f"sbs_source={self.size[0]}x{self.size[1]} "
                f"swapchain_extent={o.extent[0]}x{o.extent[1]} "
                f"blit_regions={regions}",
                flush=True,
            )
        # Keep every driver call to one region.  NVIDIA Windows drivers can
        # block the imported CUDA-image transfer when a single vkCmdBlitImage
        # contains two packed-eye regions; separate commands preserve the
        # symmetric cover crop without entering that multi-region path.
        for source_rect, destination_rect in regions:
            sx0, sy0, sx1, sy1 = source_rect
            dx0, dy0, dx1, dy1 = destination_rect
            vk.vkCmdBlitImage(
                cmd,
                source_image,
                vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                target,
                vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                [vk.VkImageBlit(
                    srcSubresource=vk.VkImageSubresourceLayers(aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT, mipLevel=0, baseArrayLayer=0, layerCount=1),
                    srcOffsets=[vk.VkOffset3D(x=sx0, y=sy0, z=0), vk.VkOffset3D(x=sx1, y=sy1, z=1)],
                    dstSubresource=vk.VkImageSubresourceLayers(aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT, mipLevel=0, baseArrayLayer=0, layerCount=1),
                    dstOffsets=[vk.VkOffset3D(x=dx0, y=dy0, z=0), vk.VkOffset3D(x=dx1, y=dy1, z=1)],
                )],
                vk.VK_FILTER_LINEAR,
            )
        self._transition(cmd, target, vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, vk.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)
        if cuda_source:
            self._transition(cmd, source_image, vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, vk.VK_IMAGE_LAYOUT_GENERAL)
        vk.vkEndCommandBuffer(cmd)
        record_ms = (time.perf_counter() - stage_started) * 1000.0
        waits = [o.image_available]
        stages = [vk.VK_PIPELINE_STAGE_TRANSFER_BIT]
        if cuda_source:
            waits.append(self._cuda_ready.semaphore)
            stages.append(vk.VK_PIPELINE_STAGE_TRANSFER_BIT)
        submit = vk.VkSubmitInfo(sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO, waitSemaphoreCount=len(waits), pWaitSemaphores=waits, pWaitDstStageMask=stages, commandBufferCount=1, pCommandBuffers=[cmd], signalSemaphoreCount=1, pSignalSemaphores=[o.render_finished])
        stage_started = time.perf_counter()
        submit_value = vk.vkQueueSubmit(o.queue, 1, [submit], o.fence)
        submit_result = int(vk.VK_SUCCESS if submit_value is None else submit_value)
        submit_ms = (time.perf_counter() - stage_started) * 1000.0
        present_info = vk.VkPresentInfoKHR(sType=vk.VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, waitSemaphoreCount=1, pWaitSemaphores=[o.render_finished], swapchainCount=1, pSwapchains=[o.swapchain], pImageIndices=[index])
        present = o._device_function(b"vkQueuePresentKHR", "VkResult(*)(VkQueue, const VkPresentInfoKHR *)")
        stage_started = time.perf_counter()
        result = int(present(o.queue, vk.ffi.addressof(present_info)))
        present_ms = (time.perf_counter() - stage_started) * 1000.0
        total_ms = (time.perf_counter() - frame_started) * 1000.0
        if total_ms >= 100.0:
            self._slow_present_count += 1
            if self._slow_present_count <= 5 or self._slow_present_count % 30 == 0:
                print(
                    "[VulkanLocalViewer] Slow present stages: "
                    f"total={total_ms:.1f}ms fence={fence_ms:.1f}ms "
                    f"fence_result={fence_result} acquire={acquire_ms:.1f}ms "
                    f"acquire_result={acquire_result} "
                    f"upload={upload_ms:.1f}ms record={record_ms:.1f}ms "
                    f"submit={submit_ms:.1f}ms submit_result={submit_result} "
                    f"present={present_ms:.1f}ms present_result={result} "
                    f"cuda={cuda_source} fit={fit_mode}",
                    flush=True,
                )
        o._swap_image_initialized[index] = True
        if o.is_swapchain_recreate_result(result) or recreate_after_present:
            o.recreate_swapchain()
            self._image_initialized = True
            return False
        if result != int(vk.VK_SUCCESS):
            raise RuntimeError(f"Vulkan local-viewer present failed ({result})")
        self._image_initialized = True
        return True

    def close(self) -> None:
        vk, device = self.owner.vk, self.owner.device
        self._disable_cuda_interop("close", announce=False)
        if self.image is not None: vk.vkDestroyImage(device, self.image, None)
        if self.image_memory is not None: vk.vkFreeMemory(device, self.image_memory, None)
        if self.buffer is not None: vk.vkDestroyBuffer(device, self.buffer, None)
        if self.memory is not None: vk.vkFreeMemory(device, self.memory, None)


def run_vulkan_local_viewer(*, runtime_q: Any, shutdown_event: Any, config: VulkanLocalViewerConfig) -> None:
    """Run the Vulkan window and consume only the newest completed SBS frame."""
    viewer: VulkanLocalViewer | None = None
    preview_viewer: VulkanLocalViewer | None = None
    preview_disabled = not bool(config.window_preview)
    try:
        while not shutdown_event.is_set():
            try:
                result, _started = runtime_q.get(timeout=0.05)
            except queue.Empty:
                if viewer is not None:
                    viewer.poll_events()
                if preview_viewer is not None:
                    try:
                        preview_viewer.poll_events()
                    except StopIteration:
                        preview_viewer.close()
                        preview_viewer = None
                        preview_disabled = True
                continue
            if config.on_breakdown_inc is not None:
                config.on_breakdown_inc("viewer_get", 1)
            if not bool(getattr(runtime_q, "_d2s_ordered", False)):
                while True:
                    try:
                        result, _started = runtime_q.get_nowait()
                    except queue.Empty:
                        break
                    if config.on_breakdown_inc is not None:
                        config.on_breakdown_inc("viewer_get", 1)
                        config.on_breakdown_inc("viewer_drop", 1)
            frame = getattr(result, "sbs", None)
            if frame is None:
                continue
            if viewer is None:
                viewer = VulkanLocalViewer(config)
                viewer.initialize()
                print("[VulkanLocalViewer] Vulkan local window initialized", flush=True)
            if preview_viewer is None and not preview_disabled:
                preview_config = replace(
                    config,
                    title=f"{config.title} - Debug Preview",
                    monitor_index=(
                        int(config.preview_monitor_index)
                        if config.preview_monitor_index is not None
                        else int(config.monitor_index)
                    ),
                    fullscreen=False,
                    window_preview=False,
                    show_fps=False,
                    display_mode="Mono",
                    display_fit_mode="contain",
                    display_fit_mode_provider=None,
                    display_fit_enabled=True,
                    on_sbs_fps=None,
                    on_breakdown_inc=None,
                    on_breakdown_add_time=None,
                    manage_glfw_lifecycle=False,
                    exclude_from_capture=False,
                )
                try:
                    preview_viewer = VulkanLocalViewer(preview_config)
                    preview_viewer.initialize()
                    print(
                        "[VulkanLocalViewer] Additional debug preview initialized",
                        flush=True,
                    )
                except Exception as exc:
                    if preview_viewer is not None:
                        preview_viewer.close()
                    preview_viewer = None
                    preview_disabled = True
                    print(
                        "[VulkanLocalViewer] Additional debug preview unavailable: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
            present_started = time.perf_counter()
            viewer.present(frame)
            if config.on_breakdown_add_time is not None:
                config.on_breakdown_add_time(
                    "local_present", time.perf_counter() - present_started
                )
            if config.on_breakdown_inc is not None:
                config.on_breakdown_inc("local_presented_frame", 1)
            if preview_viewer is not None:
                preview_frame = depth_preview_frame(result)
                if preview_frame is None:
                    continue
                preview_started = time.perf_counter()
                try:
                    preview_viewer.present(preview_frame)
                    if config.on_breakdown_add_time is not None:
                        config.on_breakdown_add_time(
                            "local_preview_present",
                            time.perf_counter() - preview_started,
                        )
                    if config.on_breakdown_inc is not None:
                        config.on_breakdown_inc("local_preview_presented_frame", 1)
                except Exception as exc:
                    preview_viewer.close()
                    preview_viewer = None
                    preview_disabled = True
                    print(
                        "[VulkanLocalViewer] Additional debug preview stopped: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
    except StopIteration:
        shutdown_event.set()
    finally:
        if preview_viewer is not None:
            preview_viewer.close()
        if viewer is not None:
            viewer.close()
