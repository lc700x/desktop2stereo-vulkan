from __future__ import annotations

import ctypes
import ctypes.util
from concurrent.futures import Future, ThreadPoolExecutor
from collections import deque
import importlib
import json
import math
import os
import queue
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from viewer.vulkan_context import (
    MIN_VULKAN_API_VERSION,
    ImageState,
    VulkanContext,
    VulkanCapabilityError,
    _cffi_handle_address,
    _require_timeline_semaphore_features,
    find_graphics_queue_family,
    make_vulkan_version,
)
from viewer.vulkan_resources import (
    VulkanDepthAttachment,
    VulkanExportableImage,
    VulkanHostImage,
    VulkanHostReadbackBuffer,
    VulkanImageResource,
    VulkanTransientImage,
)
from viewer.vulkan_msdf_quad import VulkanMsdfQuadRenderer, VulkanMsdfQuadRequest
from viewer.vulkan_projection_screen import VulkanProjectionScreenPass
from viewer.vulkan_multiview_diagnostic import VulkanMultiviewEyeDiagnosticPass
from app_runtime.output_contract import VulkanStereoOutputFrame


_OUTPUT_FRAME_UNSET = object()

from .core_controller_actions import CoreControllerActionsMixin
from .core_input_helpers import CoreInputHelpersMixin
from .core_controller_input import CoreControllerInputMixin
from .core_controller_guide_input import CoreControllerGuideInputMixin
from .core_controller_shortcuts import CoreControllerShortcutsMixin
from .core_controller_pose import CoreControllerPoseMixin
from .core_controller_ray import CoreControllerRayMixin
from .controller_models import (
    controller_button_local_position,
    discover_controller_brands,
    select_controller_brand,
)
from viewer.controller_help import get_controller_help_rows
from .filters import OneEuroFilter3D
from .xr_math import (
    _fov_to_proj_mat4_d3d,
    _mat3_to_quat_xyzw,
    _pose_to_view_mat4,
    _xr_quat_to_mat4,
    euler_to_mat4,
    mat4_to_xr_posef,
)
from .overlay_textures import (
    build_controller_callout_rgba,
    build_cursor_rgba,
    build_fps_overlay_rgba,
    build_help_rgba,
    build_team_help_rgba,
    build_keyboard_rgba,
    build_screen_adjust_osd_rgba,
    build_screen_preset_osd_rgba,
    build_settings_menu_rgba,
    build_short_osd_rgba,
)
from .settings_menu import (
    OpenXrSettingsMenu,
    PICTURE_DEFAULTS,
    SETTINGS_MENU_WORLD_SIZE,
)
from .keyboard_layout import _KB_TEX_H, _KB_TEX_W
from .msdf_font_atlas import MsdfFontAtlas
from .windows_input import (
    _MOUSEEVENTF_LEFTDOWN,
    _MOUSEEVENTF_LEFTUP,
    _MOUSEEVENTF_RIGHTDOWN,
    _MOUSEEVENTF_RIGHTUP,
    _send_mouse_flags,
    _send_key,
    _set_cursor_pos,
    _start_physical_input_monitor,
    _physical_mouse_active,
    _physical_keyboard_active,
)
from .input import (
    _TOUCH_AVAILABLE,
    _TOUCH_CONTACT_ID_LEFT,
    _TOUCH_CONTACT_ID_RIGHT,
    _TOUCH_PINCH_SPREAD_GAIN,
    _touch_injector,
)
from utils import LANG
from gui.localization import normalize_locale
from gui.config import (
    discover_environment_keys,
    environment_display_label,
    load_environment_display_names,
)
from utils.xr_headset_presets import resolve_xr_headset_preset
from utils.screen_resolution_policy import (
    ScreenSamplingPlan,
    build_screen_sampling_plan,
)


_DEFAULT_XR_HEADSET_PRESET = resolve_xr_headset_preset(None)

_MSDF_OSD_SCALE = 0.58
_MSDF_OSD_RUN_GAP = 8.0
_MSDF_OSD_PADDING_X = 20.0
_MSDF_OSD_PADDING_Y = 14.0
_MSDF_OSD_REFERENCE_HEIGHT = 78.0
_TOOL_OVERLAY_UPDATE_INTERVAL = 1.0

# Virtual Desktop does not accept the stereo screen Quad swapchain used by the
# reprojection experiment. Keep the implementation isolated for diagnosis, but
# never allow it to take ownership of the primary screen presentation path.
_SCREEN_QUAD_REPROJECTION_SUPPORTED = False


def _env_flag(name: str, default: bool = False) -> bool:
    return str(os.environ.get(name, "1" if default else "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _env_number(name: str, default: float, *, minimum: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = float(default)
    if not math.isfinite(value):
        value = float(default)
    return max(float(minimum), value)


def _sbs_capture_options() -> dict[str, Any] | None:
    output_dir = str(os.environ.get("D2S_SBS_CAPTURE_DIR", "")).strip()
    if not output_dir:
        return None
    return {
        "output_dir": Path(output_dir),
        "delay_seconds": _env_number(
            "D2S_SBS_CAPTURE_DELAY_SECONDS", 15.0, minimum=0.0
        ),
        "sample_count": int(
            _env_number("D2S_SBS_CAPTURE_SAMPLE_COUNT", 300, minimum=1.0)
        ),
        "image_count": int(
            _env_number("D2S_SBS_CAPTURE_IMAGE_COUNT", 6, minimum=1.0)
        ),
        "eye_width": int(
            _env_number("D2S_SBS_CAPTURE_EYE_WIDTH", 640, minimum=64.0)
        ),
    }


def _vulkan_rgba_to_rgb(
    pixels: np.ndarray,
    *,
    format_value: int,
    vk: Any,
    image_origin: str,
) -> np.ndarray:
    rgba = pixels
    if int(format_value) in {
        int(vk.VK_FORMAT_B8G8R8A8_UNORM),
        int(vk.VK_FORMAT_B8G8R8A8_SRGB),
    }:
        rgba = rgba[..., [2, 1, 0, 3]]
    rgb = np.ascontiguousarray(rgba[..., :3])
    if str(image_origin).strip().lower() == "bottom_left":
        rgb = np.ascontiguousarray(rgb[::-1])
    return rgb


def _write_sbs_capture_png(
    output_path: Path,
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
) -> None:
    from PIL import Image

    sbs = np.concatenate((left_rgb, right_rgb), axis=1)
    Image.fromarray(sbs, mode="RGB").save(output_path, compress_level=1)


def _active_rgb_mean(rgb: np.ndarray, *, threshold: int = 16) -> tuple[int, tuple[float, float, float]]:
    """Return non-background pixel count and RGB mean for visual diagnostics."""
    active = np.max(rgb, axis=2) > int(threshold)
    count = int(np.count_nonzero(active))
    if count == 0:
        return 0, (0.0, 0.0, 0.0)
    mean = np.mean(rgb[active], axis=0)
    return count, tuple(float(value) for value in mean)


def _layout_msdf_osd_runs(
    atlas: MsdfFontAtlas,
    runs: tuple[tuple[str, tuple[int, int, int, int]], ...]
    | tuple[dict[str, Any], ...],
) -> tuple[int, int, tuple[dict[str, Any], ...]]:
    """Fit one-line MSDF runs to a padded Quad canvas."""
    normalized: list[tuple[str, tuple[int, int, int, int]]] = []
    for run in runs:
        if isinstance(run, dict):
            text = str(run.get("text", ""))
            color = tuple(run.get("color", (255, 255, 255, 255)))
        else:
            text, color = run
            text = str(text)
            color = tuple(color)
        if len(color) != 4:
            raise ValueError("MSDF OSD colors must contain four components")
        normalized.append((text, color))

    widths = [
        float(atlas.text_advance(text, scale=_MSDF_OSD_SCALE))
        for text, _color in normalized
    ]
    content_width = sum(widths) + _MSDF_OSD_RUN_GAP * max(0, len(widths) - 1)
    canvas_width = max(
        64,
        int(math.ceil(content_width + 2.0 * _MSDF_OSD_PADDING_X)),
    )
    canvas_height = max(
        48,
        int(
            math.ceil(
                float(atlas.line_height) * _MSDF_OSD_SCALE
                + 2.0 * _MSDF_OSD_PADDING_Y
            )
        ),
    )
    cursor = max(
        _MSDF_OSD_PADDING_X,
        (float(canvas_width) - content_width) * 0.5,
    )
    laid_out: list[dict[str, Any]] = []
    for (text, color), run_width in zip(normalized, widths):
        laid_out.append(
            {
                "text": text,
                "x": cursor,
                "y": _MSDF_OSD_PADDING_Y,
                "scale": _MSDF_OSD_SCALE,
                "color": color,
            }
        )
        cursor += run_width + _MSDF_OSD_RUN_GAP
    return canvas_width, canvas_height, tuple(laid_out)


def _build_msdf_panel_request(
    atlas: MsdfFontAtlas,
    runs: tuple[dict[str, Any], ...],
    *,
    width: int,
    height: int,
    background: tuple[int, int, int, int],
    radius: float = 14.0,
) -> VulkanMsdfQuadRequest:
    """Create a GPU MSDF panel request with explicit canvas geometry."""
    return VulkanMsdfQuadRequest(
        width=int(width),
        height=int(height),
        runs=tuple(runs),
        background=background,
        radius=float(radius),
    )


def _build_msdf_depth_osd_request(
    atlas: MsdfFontAtlas, depth_strength: float, message: str | None = None
) -> VulkanMsdfQuadRequest:
    """Build the legacy depth or stereo-mode prompt as a Quad MSDF panel."""
    if message:
        runs = ((str(message), (0, 210, 230, 255)),)
    else:
        runs = (
            ("Depth Strength", (150, 158, 185, 255)),
            (f"{max(0.0, float(depth_strength)):.2f}", (0, 210, 230, 255)),
        )
    width, height, laid_out = _layout_msdf_osd_runs(atlas, runs)
    return _build_msdf_panel_request(
        atlas,
        laid_out,
        width=width,
        height=height,
        background=(32, 32, 36, 210),
    )


def _build_msdf_fps_panel(
    atlas: MsdfFontAtlas,
    *,
    actual_fps: float,
    sbs_fps: float,
    capture_fps: float,
    latency_ms: float,
    screen_width: float,
    screen_height: float,
    screen_distance: float,
    depth_strength: float,
    vr_res: tuple[int, int],
    sbs_res: tuple[int, int],
    controller_brand: str,
    environment_visible: bool,
) -> VulkanMsdfQuadRequest:
    """Build the FPS panel as MSDF runs instead of a rasterized text bitmap."""
    scale = 24.0 / max(float(atlas.line_height), 1.0)
    label_color = (150, 158, 185, 255)
    value_colors = (
        (0, 230, 90, 255),
        (0, 210, 230, 255),
        (255, 190, 40, 255),
        (0, 210, 230, 255),
        (0, 210, 230, 255),
    )
    labels = (
        "[Performance]",
        "[3D Display]",
        "[Resolution]",
        "[Controller]",
        "[Environment]",
    )
    latency_text = f"{float(latency_ms):.0f}ms" if float(latency_ms or 0.0) > 0 else "N/A"
    values = (
        f"XR {float(actual_fps):.0f} FPS   SBS {float(sbs_fps):.0f} FPS   "
        f"Capture {float(capture_fps):.0f} FPS   Latency {latency_text}",
        (
            f"{float(screen_width):.2f} x {float(screen_height):.2f} m"
            f"  @  {float(screen_distance):.2f} m"
            f"   Depth Strength {float(depth_strength):.2f}"
        ),
        f"XR {int(vr_res[0])}x{int(vr_res[1])}/eye   Screen {int(sbs_res[0])}x{int(sbs_res[1])}",
        f"Model: {controller_brand}" if controller_brand else "",
        "ON" if environment_visible else "OFF",
    )
    pad_x = 14.0
    pad_y = 14.0
    row_gap = 34.0
    label_width = max(
        atlas.text_advance(label, scale=scale) for label in labels
    )
    value_x = pad_x + label_width + 10.0
    value_width = max(
        atlas.text_advance(value, scale=scale) for value in values
    )
    canvas_width = int(math.ceil(value_x + value_width + pad_x))
    canvas_height = int(math.ceil(pad_y * 2.0 + row_gap * len(labels)))
    runs: list[dict[str, Any]] = []
    for index, (label, value) in enumerate(zip(labels, values)):
        y = pad_y + index * row_gap
        runs.append(
            {
                "text": label,
                "x": pad_x,
                "y": y,
                "scale": scale,
                "color": label_color,
            }
        )
        if value:
            runs.append(
                {
                    "text": value,
                    "x": value_x,
                    "y": y,
                    "scale": scale,
                    "color": value_colors[index],
                }
            )
    return _build_msdf_panel_request(
        atlas,
        tuple(runs),
        width=canvas_width,
        height=canvas_height,
        background=(32, 32, 36, 210),
    )


def _build_msdf_help_panel(
    atlas: MsdfFontAtlas,
    rows: list[tuple[str, str, str, bool]],
    *,
    two_columns: bool,
    size_scale: float = 1.0,
    canvas_scale: float | None = None,
) -> VulkanMsdfQuadRequest:
    """Build the controller guide panel from its shared row definition."""
    size_scale = max(0.1, min(4.0, float(size_scale)))
    normal_px = (16.0 if two_columns else 21.0) * size_scale
    title_px = (18.0 if two_columns else 21.0) * size_scale
    normal_scale = normal_px / max(float(atlas.line_height), 1.0)
    title_scale = title_px / max(float(atlas.line_height), 1.0)
    column_widths = [0.0, 0.0, 0.0]
    for row in rows:
        is_title = bool(row[3])
        scale = title_scale if is_title else normal_scale
        for column in range(3):
            column_widths[column] = max(
                column_widths[column],
                atlas.text_advance(row[column], scale=scale),
            )

    gap = 20.0 * size_scale
    middle_gap = 50.0 * size_scale
    padding_x = 30.0 * size_scale
    padding_y = 20.0 * size_scale
    line_height = (16.0 + 6.0 if two_columns else 21.0 + 6.0) * size_scale
    inner_width = sum(column_widths) + gap * 2.0
    if two_columns:
        title_indices = [index for index, row in enumerate(rows) if bool(row[3])]
        middle_index = title_indices[4] if len(title_indices) > 4 else len(rows)
        left_rows = rows[:middle_index]
        right_rows = rows[middle_index:]
    else:
        # The screen-side vertical guide is one complete column. Do not apply
        # the controller-attached two-column split to this layout.
        left_rows = rows
        right_rows = []
    content_width = (
        inner_width * 2.0 + middle_gap + padding_x * 2.0
        if two_columns
        else inner_width + padding_x * 2.0
    )
    content_height = max(len(left_rows), len(right_rows)) * line_height + padding_y * 2.0
    canvas_width = content_width
    canvas_height = content_height
    content_offset_x = 0.0
    content_offset_y = 0.0
    if canvas_scale is not None:
        canvas_scale = max(1.0, float(canvas_scale))
        base_normal_px = (16.0 if two_columns else 21.0) * canvas_scale
        base_title_px = (18.0 if two_columns else 21.0) * canvas_scale
        base_column_widths = [0.0, 0.0, 0.0]
        for row in rows:
            is_title = bool(row[3])
            base_row_scale = (
                base_title_px if is_title else base_normal_px
            ) / max(float(atlas.line_height), 1.0)
            for column in range(3):
                base_column_widths[column] = max(
                    base_column_widths[column],
                    atlas.text_advance(row[column], scale=base_row_scale),
                )
        base_gap = 20.0 * canvas_scale
        base_middle_gap = 50.0 * canvas_scale
        base_padding_x = 30.0 * canvas_scale
        base_padding_y = 20.0 * canvas_scale
        base_inner_width = sum(base_column_widths) + base_gap * 2.0
        base_line_height = (
            16.0 + 6.0 if two_columns else 21.0 + 6.0
        ) * canvas_scale
        canvas_width = (
            base_inner_width * 2.0 + base_middle_gap + base_padding_x * 2.0
            if two_columns
            else base_inner_width + base_padding_x * 2.0
        )
        canvas_height = (
            max(len(left_rows), len(right_rows)) * base_line_height
            + base_padding_y * 2.0
        )
        content_offset_x = (canvas_width - content_width) * 0.5
        content_offset_y = (canvas_height - content_height) * 0.5
    runs: list[dict[str, Any]] = []

    def add_rows(group_rows, origin_x: float) -> None:
        for row_index, row in enumerate(group_rows):
            is_title = bool(row[3])
            scale = title_scale if is_title else normal_scale
            color = (90, 190, 255, 255) if is_title else (200, 210, 235, 255)
            y = content_offset_y + padding_y + row_index * line_height
            x = origin_x
            for column in range(3):
                text = str(row[column])
                if text:
                    runs.append(
                        {
                            "text": text,
                            "x": content_offset_x + x,
                            "y": y,
                            "scale": scale,
                            "color": color,
                        }
                    )
                x += column_widths[column] + gap

    add_rows(left_rows, padding_x)
    if two_columns:
        add_rows(right_rows, padding_x + inner_width + middle_gap)
    return _build_msdf_panel_request(
        atlas,
        tuple(runs),
        width=int(math.ceil(canvas_width)),
        height=int(math.ceil(canvas_height)),
        background=(18, 18, 28, 210),
    )


class OpenXrVulkanUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class OpenXrVulkanConfig:
    application_name: str = "Desktop2Stereo Vulkan"
    render_scale: float = 1.0
    clear_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    requested_vulkan_version: int = make_vulkan_version(1, 4, 0)
    # Keep the validated OpenXR projection target as sRGB. The Filament bridge
    # is configured for linear Rec709 output so the target performs one OETF.
    swapchain_color_mode: str = "srgb"
    controller_model: str = "PICO"
    headset_model: str = _DEFAULT_XR_HEADSET_PRESET.key
    # MSS 1-based monitor whose desktop is captured and shown in VR. Used to
    # map the laser screen UV to the correct virtual-desktop cursor position.
    monitor_index: int = 1
    controller_guide_max_distance: float = 0.4
    filament_bridge_path: str | None = None
    filament_glb_path: str | None = None
    filament_profile_path: str | None = None
    filament_panorama_path: str | None = None
    filament_scene_exposure_ev: float = 0.0
    filament_skybox_brightness: float = 1.0
    # Reuse the legacy headset preset for profiles without a screen. The
    # presenter only consumes these resolved values; it does not define a
    # second headset-geometry table.
    filament_screen_width: float = _DEFAULT_XR_HEADSET_PRESET.width_m
    filament_screen_distance: float = _DEFAULT_XR_HEADSET_PRESET.distance_m
    filament_ambient_light_color: tuple[float, float, float] = (0.14, 0.13, 0.15)
    filament_ambient_light_intensity_lux: float = 30000.0
    filament_controller_ambient_light_intensity_lux: float = 8000.0
    filament_controller_hdr_ambient_light_intensity_lux: float = 8000.0
    filament_controller_light_intensity_candela: float = 2000.0
    filament_fill_light_color: tuple[float, float, float] = (0.55, 0.55, 0.58)
    filament_fill_light_intensity: float = 1.0
    filament_fill_light_direction: tuple[float, float, float] = (-0.35, -1.0, -0.55)
    filament_controller_head_light_weight: float = 0.85
    filament_controller_top_light_weight: float = 0.60
    filament_controller_top_light_color: tuple[float, float, float] = (0.95, 0.97, 1.0)
    filament_controller_head_light_offset: tuple[float, float, float] = (0.0, 0.05, 0.0)
    filament_controller_top_light_offset: tuple[float, float, float] = (0.0, 0.45, -0.18)
    filament_controller_head_light_falloff: float = 2.0
    filament_controller_top_light_falloff: float = 2.0
    filament_controller_head_light_cast_shadows: bool = False
    filament_controller_top_light_cast_shadows: bool = False
    filament_controller_screen_light_enabled: bool = True
    filament_controller_screen_light_intensity_lux: float = 500.0
    filament_controller_screen_light_saturation: float = 0.65
    filament_controller_screen_light_max_luminance: float = 0.40
    filament_controller_screen_light_smoothing_seconds: float = 0.18
    filament_controller_screen_light_sample_hz: float = 12.0
    filament_controller_screen_light_cast_shadows: bool = False
    filament_environment_screen_light_enabled: bool = True
    filament_environment_screen_light_intensity_candela: float = 120.0
    filament_environment_screen_light_saturation: float = 0.70
    filament_environment_screen_light_max_luminance: float = 0.40
    filament_environment_screen_light_smoothing_seconds: float = 0.18
    filament_environment_screen_light_sample_hz: float = 12.0
    filament_environment_screen_light_falloff: float = 4.0
    filament_environment_screen_light_offset: float = 0.08
    filament_environment_screen_light_cast_shadows: bool = False
    filament_glow_sample_hz: float = 30.0
    filament_glow_smoothing_seconds: float = 0.10
    openxr_no_headset_retry_interval: float = 3.0
    openxr_standby_retry_interval: float = 3.0
    openxr_standby_retry_max_interval: float = 30.0
    headset_wait_inference_timeout: float = 60.0


@dataclass(slots=True)
class _EyeSwapchain:
    handle: Any
    images: list[Any]
    width: int
    height: int
    resources: list[VulkanImageResource] = field(default_factory=list)
    array_size: int = 1


class OpenXrCompositionBuilder:
    """Builds projection layers without owning OpenXR frame lifecycle."""

    def __init__(self, xr: Any, reference_space: Any) -> None:
        self.xr = xr
        self.reference_space = reference_space

    def projection_layer(
        self, views: list[Any], swapchains: list[_EyeSwapchain], *,
        layer_flags: Any = 0,
    ) -> Any:
        layered = len(swapchains) == 1 and swapchains[0].array_size >= 2
        eye_swapchains = (
            [(swapchains[0], 0), (swapchains[0], 1)]
            if layered
            else [(eye, 0) for eye in swapchains]
        )
        if len(views) < len(eye_swapchains):
            raise ValueError("projection layer requires one view per eye swapchain")
        projection_views = []
        for eye_index, (eye, array_index) in enumerate(eye_swapchains):
            projection_views.append(
                self.xr.CompositionLayerProjectionView(
                    pose=views[eye_index].pose,
                    fov=views[eye_index].fov,
                    sub_image=self.xr.SwapchainSubImage(
                        swapchain=eye.handle,
                        image_rect=self.xr.Rect2Di(
                            offset=self.xr.Offset2Di(x=0, y=0),
                            extent=self.xr.Extent2Di(width=eye.width, height=eye.height),
                        ),
                        image_array_index=array_index,
                    ),
                )
            )
        return self.xr.CompositionLayerProjection(
            layer_flags=layer_flags,
            space=self.reference_space,
            views=projection_views,
        )

    def quad_layer(
        self, swapchain: _EyeSwapchain, position: tuple[float, float, float],
        width: float, height: float, rotation: tuple[float, float, float],
        eye_index: int,
    ) -> Any:
        qx, qy, qz, qw = _euler_degrees_to_quaternion(rotation)
        return self.xr.CompositionLayerQuad(
            space=self.reference_space,
            eye_visibility=(self.xr.EyeVisibility.LEFT if eye_index == 0
                            else self.xr.EyeVisibility.RIGHT),
            sub_image=self.xr.SwapchainSubImage(
                swapchain=swapchain.handle,
                image_rect=self.xr.Rect2Di(
                    offset=self.xr.Offset2Di(x=0, y=0),
                    extent=self.xr.Extent2Di(width=swapchain.width, height=swapchain.height),
                ),
                image_array_index=eye_index if swapchain.array_size >= 2 else 0,
            ),
            pose=self.xr.Posef(
                orientation=self.xr.Quaternionf(x=qx, y=qy, z=qz, w=qw),
                position=self.xr.Vector3f(
                    x=float(position[0]), y=float(position[1]), z=float(position[2])
                ),
            ),
            size=self.xr.Extent2Df(width=float(width), height=float(height)),
        )


class OpenXrVulkanPresenter(
    CoreControllerActionsMixin,
    CoreControllerPoseMixin,
    CoreControllerRayMixin,
    CoreControllerInputMixin,
    CoreControllerGuideInputMixin,
    CoreControllerShortcutsMixin,
    CoreInputHelpersMixin,
):
    """OpenXR Vulkan projection-layer presenter with Filament controllers."""

    _VULKAN_EXTENSION = "XR_KHR_vulkan_enable2"
    _DEFAULT_SCREEN_CURVE_HALF_ANGLE = 0.72

    def __init__(
        self,
        config: OpenXrVulkanConfig | None = None,
        *,
        on_headset_state: Callable[[str], None] | None = None,
        on_controller_shortcut: Callable[..., bool | None] | None = None,
        on_breakdown_inc: Callable[[str, int | float], None] | None = None,
        on_breakdown_add_time: Callable[[str, float], None] | None = None,
        on_breakdown_set_latest: Callable[[str, Any], None] | None = None,
        on_runtime_fps: Callable[[], float] | None = None,
        on_capture_fps: Callable[[], float] | None = None,
        on_sbs_fps: Callable[[float], int] | None = None,
    ) -> None:
        self.config = config or OpenXrVulkanConfig()
        self._headset_preset = resolve_xr_headset_preset(self.config.headset_model)
        self._on_headset_state = on_headset_state
        self._on_controller_shortcut = on_controller_shortcut
        self._on_breakdown_inc = on_breakdown_inc
        self._on_breakdown_add_time = on_breakdown_add_time
        self._on_breakdown_set_latest = on_breakdown_set_latest
        self._on_runtime_fps = on_runtime_fps
        self._on_capture_fps = on_capture_fps
        self._on_sbs_fps = on_sbs_fps
        self._adaptive_capture_target_fps = 0
        # The projection presenter does not submit an asynchronous effect
        # source. Mark that validation branch inactive instead of reporting a
        # missing effect path for every otherwise-valid projection frame.
        if self._on_breakdown_set_latest is not None:
            self._on_breakdown_set_latest("openxr_async_effects_enabled", False)
        if self.config.render_scale <= 0:
            raise ValueError("render_scale must be greater than zero")
        if len(self.config.clear_color) != 4:
            raise ValueError("clear_color must contain four components")
        if self.config.controller_guide_max_distance <= 0:
            raise ValueError("controller_guide_max_distance must be greater than zero")

        self.xr: Any = None
        self.instance: Any = None
        self.system_id: Any = None
        self.session: Any = None
        self.reference_space: Any = None
        self._reference_space_type: Any = None
        self.vulkan: VulkanContext | None = None
        self.swapchain_format: int | None = None
        self.swapchains: list[_EyeSwapchain] = []
        self._controller_composition_swapchain: _EyeSwapchain | None = None
        self._controller_composition_layer_logged = False
        self._vulkan_controller_proxy_swapchains: list[_EyeSwapchain] = []
        self._vulkan_controller_proxy_layer_logged = False
        self._panorama_swapchain: _EyeSwapchain | None = None
        self._panorama_staging: VulkanHostImage | None = None
        self._panorama_size: tuple[int, int] | None = None
        self._panorama_layer: Any = None
        self._openxr_equirect_supported = False
        self._panorama_skip_logged = False
        self._panorama_failed = False
        self._vulkan_panorama_image = None
        self._vulkan_panorama_staging = None
        self._multiview_active = False
        self._filament_multiview_hdr_images: list[VulkanTransientImage] = []
        self._filament_multiview_ready_semaphores: list[Any] = []
        self._filament_multiview_slot_timelines: list[int] = []
        self._filament_multiview_slot = 0
        self._filament_multiview_current: VulkanTransientImage | None = None
        self._filament_multiview_current_slot: int | None = None
        self._filament_multiview_finished_consumed = False
        self._projection_array_eye_diagnostic = _env_flag(
            "D2S_OPENXR_PROJECTION_ARRAY_EYE_DIAGNOSTIC", default=False
        )
        self._vulkan_multiview_eye_diagnostic = _env_flag(
            "D2S_OPENXR_VULKAN_MULTIVIEW_EYE_DIAGNOSTIC", default=False
        )
        self._filament_multiview_projection_diagnostic = _env_flag(
            "D2S_FILAMENT_MULTIVIEW_PROJECTION_DIAGNOSTIC", default=False
        )
        self._filament_multiview_layer_readback_requested = _env_flag(
            "D2S_FILAMENT_MULTIVIEW_LAYER_READBACK", default=False
        )
        self._filament_multiview_layer_readback_done = False
        self._filament_multiview_layer_readback_frame = 0
        self._filament_multiview_layer_readback_delay_frames = 30
        self._vulkan_projection_composer_requested = _env_flag(
            "D2S_VULKAN_PROJECTION_COMPOSER", default=True
        )
        self._vulkan_projection_quality_chain_requested = _env_flag(
            "D2S_VULKAN_PROJECTION_QUALITY_CHAIN", default=True
        )
        self._filament_projection_only = _env_flag(
            "D2S_FILAMENT_PROJECTION_ONLY", default=False
        )
        self._filament_controller_overlay_after_composer = _env_flag(
            "D2S_FILAMENT_CONTROLLER_OVERLAY_AFTER_COMPOSER", default=True
        )
        self._vulkan_projection_composer_active = False
        self._vulkan_projection_composer_frame_id: int | None = None
        self._last_vulkan_projection_composer_status: tuple[Any, ...] | None = None
        self._last_vulkan_projection_composer_fallback: tuple[str, str] | None = None
        self._last_vulkan_projection_glow_error: tuple[str, str] | None = None
        self._last_vulkan_projection_laser_error: tuple[str, str] | None = None
        # The current Filament producer exposes color completion only.  Do not
        # draw a Vulkan laser without the producer depth attachment: that
        # would change the legacy root-occlusion behavior.
        self._vulkan_projection_laser_depth_available = False
        self._last_vulkan_projection_laser_depth_status: str | None = None
        self._last_vulkan_projection_laser_prepare_status: str | None = None
        screen_quad_reprojection_requested = _env_flag(
            "D2S_OPENXR_SCREEN_QUAD_REPROJECTION"
        )
        self._screen_quad_reprojection_requested = bool(
            screen_quad_reprojection_requested
            and _SCREEN_QUAD_REPROJECTION_SUPPORTED
        )
        if (
            screen_quad_reprojection_requested
            and not _SCREEN_QUAD_REPROJECTION_SUPPORTED
        ):
            print(
                "[OpenXRViewer] Screen Quad Reprojection disabled: "
                "Virtual Desktop stereo Quad swapchain is unsupported; "
                "using Projection swapchain",
                flush=True,
            )
        self._screen_quad_reprojection_active = False
        self._screen_quad_reprojection_frame_id: int | None = None
        self._last_screen_quad_reprojection_status: tuple[Any, ...] | None = None
        if self._on_breakdown_set_latest is not None:
            self._on_breakdown_set_latest(
                "openxr_vulkan_projection_composer_requested",
                self._vulkan_projection_composer_requested,
            )
            self._on_breakdown_set_latest(
                "openxr_vulkan_projection_composer_active", False
            )
            self._on_breakdown_set_latest(
                "openxr_vulkan_projection_composer_frame_id", -1
            )
            self._on_breakdown_set_latest(
                "openxr_vulkan_projection_laser_depth_available",
                self._vulkan_projection_laser_depth_available,
            )
            self._on_breakdown_set_latest(
                "openxr_screen_quad_reprojection_requested",
                self._screen_quad_reprojection_requested,
            )
            self._on_breakdown_set_latest(
                "openxr_screen_quad_reprojection_active", False
            )
        self._quad_swapchains: list[_EyeSwapchain] = []
        self._quad_swapchain_format: int | None = None
        self._tool_quad_swapchain_format: int | None = None
        self._quad_swapchain_extent: tuple[int, int] | None = None
        self.filament_bridge: Any | None = None
        self._filament_depth_attachments: list[VulkanDepthAttachment] = []
        self._filament_depth_attachments_bound = False
        self.session_state: Any = None
        self.session_running = False
        self.exit_requested = False
        self.frame_count = 0
        self._view_configuration_type: Any = None
        self._environment_blend_mode: Any = None
        self._vulkan_loader: Any = None
        self._vk_get_instance_proc_addr: Any = None
        self._graphics_binding: Any = None
        self._provisional_vk_instance: Any = None
        self._provisional_vk_device: Any = None
        self._profile_head_transform: np.ndarray | None = None
        self._profile_initial_head: np.ndarray | None = None
        self._profile_space_applied = False
        self._profile_space_calibration_pass = 0
        self._profile_space_pose_in_reference = np.eye(4, dtype=np.float32)
        self._profile_reference_head_anchor: np.ndarray | None = None
        self._profile_space_preserve_anchor = False
        self._profile_view_name: str | None = None
        self._filament_profile_data: dict[str, Any] = {}
        self._filament_view_poses: tuple[dict[str, Any], ...] = ()
        self._filament_view_pose_index = 0
        self._room_seat_height_offset = 0.0
        self._profile_auto_center_on_screen = False
        self._profile_reference_space_change_ignored_logged = False
        self._profile_alignment_logged = False
        self._head_position_w: np.ndarray | None = None
        self._head_forward_w: np.ndarray | None = None
        self._head_model_matrix: np.ndarray | None = None
        self._initial_head_y = 0.0
        self._profile_near_plane = 0.05
        self._profile_far_plane = 1000.0
        self._filament_scene_exposure = self.config.filament_scene_exposure_ev
        self._filament_skybox_brightness = self.config.filament_skybox_brightness
        self._filament_ambient_light_color = self.config.filament_ambient_light_color
        self._controller_ambient_light_color_override = None
        self._controller_hdr_ambient_light_color_override = None
        self._filament_ambient_light_intensity_lux = (
            self.config.filament_ambient_light_intensity_lux
        )
        self._controller_ambient_light_intensity_lux = (
            self.config.filament_controller_ambient_light_intensity_lux
        )
        self._controller_hdr_ambient_light_intensity_lux = (
            self.config.filament_controller_hdr_ambient_light_intensity_lux
        )
        self._controller_light_intensity_candela = (
            self.config.filament_controller_light_intensity_candela
        )
        self._last_screen_resolution_status = None
        self._last_screen_sampling_status = None
        self._active_screen_sampling_plan: ScreenSamplingPlan | None = None
        self._controller_hdr_lighting = False
        self._filament_fill_light_color = self.config.filament_fill_light_color
        self._filament_fill_light_intensity = self.config.filament_fill_light_intensity
        self._filament_fill_light_direction = self.config.filament_fill_light_direction
        self._controller_head_light_weight = self.config.filament_controller_head_light_weight
        self._controller_top_light_weight = self.config.filament_controller_top_light_weight
        self._controller_top_light_color = self.config.filament_controller_top_light_color
        self._controller_head_light_offset = self.config.filament_controller_head_light_offset
        self._controller_top_light_offset = self.config.filament_controller_top_light_offset
        self._controller_head_light_falloff = self.config.filament_controller_head_light_falloff
        self._controller_top_light_falloff = self.config.filament_controller_top_light_falloff
        self._controller_head_light_cast_shadows = (
            self.config.filament_controller_head_light_cast_shadows
        )
        self._controller_top_light_cast_shadows = (
            self.config.filament_controller_top_light_cast_shadows
        )
        self._controller_screen_light_enabled = (
            self.config.filament_controller_screen_light_enabled
        )
        self._controller_screen_light_intensity_lux = (
            self.config.filament_controller_screen_light_intensity_lux
        )
        self._controller_screen_light_saturation = (
            self.config.filament_controller_screen_light_saturation
        )
        self._controller_screen_light_max_luminance = (
            self.config.filament_controller_screen_light_max_luminance
        )
        self._controller_screen_light_smoothing_seconds = (
            self.config.filament_controller_screen_light_smoothing_seconds
        )
        self._controller_screen_light_sample_hz = (
            self.config.filament_controller_screen_light_sample_hz
        )
        self._controller_screen_light_cast_shadows = (
            self.config.filament_controller_screen_light_cast_shadows
        )
        self._filament_glow_sample_hz = self.config.filament_glow_sample_hz
        self._filament_glow_smoothing_seconds = (
            self.config.filament_glow_smoothing_seconds
        )
        self._controller_screen_light_smoothed_color = np.zeros(3, dtype=np.float64)
        self._controller_screen_light_smoothed_intensity = 0.0
        self._controller_screen_light_status = None
        self._controller_screen_light_applied = False
        self._environment_screen_light_enabled = (
            self.config.filament_environment_screen_light_enabled
        )
        self._environment_screen_light_intensity_candela = (
            self.config.filament_environment_screen_light_intensity_candela
        )
        self._environment_screen_area_light_intensity = 6.0
        self._environment_screen_light_saturation = (
            self.config.filament_environment_screen_light_saturation
        )
        self._environment_screen_light_max_luminance = (
            self.config.filament_environment_screen_light_max_luminance
        )
        self._environment_screen_light_smoothing_seconds = (
            self.config.filament_environment_screen_light_smoothing_seconds
        )
        self._environment_screen_light_sample_hz = (
            self.config.filament_environment_screen_light_sample_hz
        )
        self._environment_screen_light_falloff = (
            self.config.filament_environment_screen_light_falloff
        )
        self._environment_screen_light_offset = (
            self.config.filament_environment_screen_light_offset
        )
        self._environment_screen_light_cast_shadows = (
            self.config.filament_environment_screen_light_cast_shadows
        )
        self._environment_screen_light_applied = False
        self._environment_screen_light_status = None
        self._filament_lighting_presets: tuple[dict[str, Any], ...] = ()
        self._filament_lighting_preset_index = 0
        self._filament_glow_mode = "off"
        # Glow belongs to the blank Default environment only. GLB rooms have
        # authored lighting and geometry that must not receive this overlay.
        self._filament_glow_environment_enabled = not bool(
            self.config.filament_glb_path
        )
        # Keep the v2.5 effect constants intact. Glow samples its own small
        # Vulkan compute output, leaving the zero-copy screen image untouched.
        self._filament_glow_intensity = 0.175
        self._filament_glow_width = 0.75
        self._filament_glow_default_multiplier = 1.5
        self._filament_glow_intensity_multiplier = 0.0
        self._filament_glow_shell_default_multiplier = 1.85
        self._filament_glow_shell_intensity_multiplier = 0.0
        self._filament_glow_shell_radius = 20.0
        self._filament_glow_shell_height = 9.5
        self._veil_intensity = 1.5
        self._veil_alpha = 1.0
        self._last_filament_glow_source_serial = -1
        self._last_filament_glow_source_key: tuple[str, int] | None = None
        self._last_filament_glow_status: tuple[Any, ...] | None = None
        self._filament_screen: tuple[
            tuple[float, float, float], float, float, tuple[float, float, float]
        ] | None = None
        self._filament_screen_initial = None
        self._filament_screen_profile_authored = False
        self._filament_screen_head_initialized = False
        self._screen_curved = False
        self._screen_curve_half_angle = 0.0
        self._screen_initial_curve_half_angle = 0.0
        self._passthrough_backdrop = False
        self._controllers_root = Path(__file__).resolve().parent / "controllers"
        self._controller_brands = discover_controller_brands(self._controllers_root)
        requested_controller_model = (
            self.config.controller_model
            or os.environ.get("D2S_CONTROLLER_MODEL", "PICO")
        )
        self._vulkan_controller_proxy_enabled = (
            str(requested_controller_model).strip().lower() == "none"
        )
        self._controller_brand = select_controller_brand(
            self._controller_brands,
            requested_controller_model,
        )
        self._controller_calibration_mode = False
        self._controller_calibration_offset = np.asarray(
            self._controller_brand.offset
            if self._controller_brand
            else (0.0, 0.0, 0.0),
            dtype=np.float64,
        )
        self._controller_calibration_rotation_deg = float(
            self._controller_brand.rotation_deg
            if self._controller_brand
            else 0.0
        )
        self._controller_b_button_local: np.ndarray | None = None
        self._controller_b_button_resolved = False
        self._controller_inputs = ({}, {})
        self._last_controller_input_error: str | None = None
        self._aim_space_l = None
        self._aim_space_r = None
        self._grip_space_l = None
        self._grip_space_r = None
        self._aim_mat_l = None
        self._aim_mat_r = None
        self._grip_mat_l = None
        self._grip_mat_r = None
        self._frame_now = 0.0
        self._filament_animation_origin: float | None = None
        # Physical mouse/keyboard get priority over the controller beam and the
        # virtual keyboard: the low-level hooks (started once here) track only
        # non-injected input, so moving the real mouse or typing on the hardware
        # keyboard suppresses the beam's cursor emulation.
        _start_physical_input_monitor()
        # Keep the controller lifecycle aligned with the legacy renderer:
        # movement refreshes a per-hand activity timestamp and both the model
        # and laser are hidden after the idle timeout.
        controller_now = time.perf_counter()
        self._laser_last_move_l = controller_now
        self._laser_last_move_r = controller_now
        self._laser_prev_mat_l = None
        self._laser_prev_mat_r = None
        self._LASER_HIDE_AFTER = 5.0
        self._LASER_MOVE_THRESH = 0.015
        self._smooth_ray_origin_l = None
        self._smooth_ray_origin_r = None
        self._smooth_ray_quat_l = None
        self._smooth_ray_quat_r = None
        self._smooth_ray_fwd_l = None
        self._smooth_ray_fwd_r = None
        self._rot_smooth = 0.10
        self._ray_deadzone_rad = 0.0052
        # Match the legacy laser edge-release cone: once the ray is within
        # six degrees of the nearest screen edge, keep the cursor attached.
        self._ray_edge_deadzone_rad = math.radians(6.0)
        self._ray_filter_l = OneEuroFilter3D(8.0, 8.0, 8.0)
        self._ray_filter_r = OneEuroFilter3D(8.0, 8.0, 8.0)
        self._last_frame_dt = 1.0 / 90.0
        self._initialized = False
        self._presenter_thread_id: int | None = None
        self._presenter_commands: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=2)
        self._output_adapter: Any | None = None
        self._output_adapter_error: str | None = None
        self._next_output_frame_id = 0
        self._pending_output: VulkanStereoOutputFrame | None = None
        self._displayed_output: VulkanStereoOutputFrame | None = None
        self._rendering_output: VulkanStereoOutputFrame | None = None
        self._projection_busy = threading.Event()
        self._output_lock = threading.Lock()
        self._headset_wait_started = 0.0
        self._headset_hard_idle_notified = False
        self._headset_active_notified = False
        self._headset_wait_logged = False
        self._accept_output = False
        self._source_frame_wait_logged = False
        self._has_presented_frame = False
        self._last_quad_layers: list[Any] = []
        self._last_screen_quad_layers: list[Any] = []
        # One-shot first-frame visual diagnostics. The readback is deliberately
        # delayed until the normal render has completed, so it observes the
        # production Vulkan image and the final OpenXR projection target.
        self._visual_regression_capture_eyes: set[int] = set()
        self._visual_regression_capture_failed = False
        self._visual_regression_source_host_images: dict[int, VulkanHostImage] = {}
        self._visual_regression_projection_host_images: dict[int, VulkanHostImage] = {}
        self._sbs_capture_options = _sbs_capture_options()
        self._sbs_capture_origin: float | None = None
        self._sbs_capture_slots: list[dict[str, Any]] = []
        self._sbs_capture_write_futures: list[tuple[Future, dict[str, Any]]] = []
        self._sbs_capture_records: list[dict[str, Any]] = []
        self._sbs_capture_seen_frame_id: int | None = None
        self._sbs_capture_observed = 0
        self._sbs_capture_scheduled = 0
        self._sbs_capture_skipped = 0
        self._sbs_capture_finished = False
        self._sbs_capture_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="d2s-sbs-capture")
            if self._sbs_capture_options is not None
            else None
        )
        self._overlay_quad_entries: dict[str, dict[str, Any]] = {}
        self._settings_menu = OpenXrSettingsMenu()
        self._settings_menu_pose: tuple[tuple[float, ...], tuple[float, ...]] | None = None
        self._settings_menu_cursor_uv: tuple[float, float] | None = None
        self._settings_menu_values: dict[str, float | bool] = {}
        self._settings_menu_last_redraw = 0.0
        self._settings_menu_last_adjust = 0.0
        self._settings_menu_trigger_down = [False, False]
        self._settings_menu_allow_curve = True
        self._settings_menu_grab_hand: int | None = None
        self._settings_menu_grab_relative: np.ndarray | None = None
        self._settings_menu_grip_down = [False, False]
        self._openxr_render_scale = max(0.5, min(2.0, float(self.config.render_scale)))
        self._pending_openxr_render_scale: float | None = None
        self._view_configuration_views: tuple[Any, ...] = ()
        # Keep rasterized tool textures and their released swapchain image
        # alive. Static Quad layers must not perform a host upload every XR
        # frame; only the layer pose is rebuilt per frame.
        self._tool_quad_texture_cache: dict[str, np.ndarray] = {}
        self._tool_quad_texture_keys: dict[str, tuple[Any, ...]] = {}
        self._tool_overlay_xr_fps = 0.0
        self._tool_overlay_pending_xr_fps = 0.0
        self._tool_overlay_sbs_fps = 0.0
        self._tool_overlay_capture_fps = 0.0
        self._tool_overlay_latency_ms = 0.0
        self._tool_overlay_depth_strength = 0.0
        self._tool_overlay_depth_strength_pending: float | None = None
        self._tool_overlay_vr_res = (0, 0)
        self._tool_overlay_sbs_res = (0, 0)
        self._tool_overlay_pending_latency_ms = 0.0
        self._tool_overlay_xr_window_started = 0.0
        self._tool_overlay_xr_window_frames = 0
        self._tool_overlay_xr_frame_ts = deque(maxlen=60)
        self._tool_overlay_sbs_window_started = 0.0
        self._tool_overlay_sbs_window_frames = 0
        self._tool_overlay_last_output_id: int | None = None
        self._right_grip_screen_pointer_applied = False
        self._controller_callout_rgba: np.ndarray | None = None
        self._msdf_font_atlas: MsdfFontAtlas | None = None
        self._vulkan_msdf_quad_renderer: VulkanMsdfQuadRenderer | None = None
        self._vulkan_projection_screen_pass: VulkanProjectionScreenPass | None = None
        self._vulkan_multiview_diagnostic_pass: (
            VulkanMultiviewEyeDiagnosticPass | None
        ) = None
        # Legacy screen OSD state. These are rendered as Quad layers above
        # the virtual screen, never inside the projection scene.
        self._preset_name_overlay: str | None = None
        self._preset_osd_show_t = -999.0
        self._screen_osd_show_t = -999.0
        self._depth_osd_show_t = -999.0
        self._depth_osd_message: str | None = None
        # Legacy OpenXR shortcut state is kept in the presenter so both the
        # Vulkan projection path and future Quad Layer overlays read one state.
        self._keyboard_visible = False
        self._fps_overlay_visible = False
        self._operation_guide_visible = False
        self._screen_operation_guide_visible = False
        self._hand_fps_visible = False
        self._hand_operation_guide_visible = False
        self._aperture_visible = False
        self._init_controller_shortcuts()
        self._init_controller_guide_input()
        self._keyboard_width = 1.6
        self._keyboard_height = 0.33
        self._keyboard_keys = []
        self._kb_show_shifted = False
        self._mod_state = {
            "shift": [False, False, 0.0],
            "ctrl": [False, False, 0.0],
            "alt": [False, False, 0.0],
            "win": [False, False, 0.0],
        }
        self._caps_lock = False
        self._kb_trig_prev_l = 0.0
        self._kb_trig_prev_r = 0.0
        self._kb_hover_l = None
        self._kb_hover_r = None
        self._kb_held_key_l = None
        self._kb_held_key_r = None
        self._kb_held_mods_l = None
        self._kb_held_mods_r = None
        self._kb_rpt_t_l = 0.0   # last auto-repeat emit time (left)
        self._kb_rpt_t_r = 0.0
        self._kb_rpt_n_l = 0     # repeats emitted; 0 = still in initial delay
        self._kb_rpt_n_r = 0
        self._haptic_last_l = 0.0
        self._haptic_last_r = 0.0
        self._grip_l_now = False
        self._grip_r_now = False
        self._pointer_state = {"left": "idle", "right": "idle"}
        self._pointer_press_time = {"left": 0.0, "right": 0.0}
        # Windows multi-touch contacts (preferred over mouse clicks).
        self._touch_state = {"left": "idle", "right": "idle"}
        self._touch_px = {"left": (0, 0), "right": (0, 0)}
        self._touch_valid = {"left": False, "right": False}
        self._touch_trig_prev = {"left": 0.0, "right": 0.0}
        self._left_grab_anchor = None
        self._right_grab_anchor = None
        self._screen_hit_grab_anchor_l = None
        self._screen_hit_grab_anchor_r = None
        self._keyboard_position_offset = np.zeros(3, dtype=np.float64)
        self._keyboard_rotation_offset = np.zeros(2, dtype=np.float64)
        self._keyboard_grab_anchor = None
        self._kb_grab_local_l = None
        self._kb_grab_local_r = None
        self._screen_resize_anchor = None
        self._grip_target_l = None
        self._grip_target_r = None
        self._grip_rotation_anchor_l = None
        self._grip_rotation_anchor_r = None
        self._screen_rotation_anchor_l = None
        self._screen_rotation_anchor_r = None
        self._grip_screen_rotation_snapped_l = False
        self._both_grip_anchor = None
        self._scroll_accum_x = 0.0
        self._scroll_accum_y = 0.0
        for direction in ("left", "right", "up", "down"):
            setattr(self, f"_arrow_{direction}_held", False)
        self._status_panel_cycle = 0
        self._hand_panel_cycle = 0
        self._unsupported_shortcut_actions: set[str] = set()
        default_screen_width = max(0.25, float(self.config.filament_screen_width))
        default_screen_distance = max(0.25, float(self.config.filament_screen_distance))
        self._shortcut_screen_presets = (
            ('10" Tablet', 0.30, 0.4),
            ('27" Monitor', 0.60, 0.6),
            ('65" TV', 1.44, 2.0),
            ('100" Projector 1', 2.40, 2.0),
            ('100" Projector 2', 2.21, 2.5),
            ('Headset Recommended', default_screen_width, default_screen_distance),
            ('1000" IMAX', 22.0, 20.0),
        )
        self._shortcut_screen_preset_index = 5
        self._shortcut_saved_skybox_brightness = self._filament_skybox_brightness
        self._shortcut_light_levels = (0.0, 0.5, 1.0)
        # Right-grip screen controls accelerate while the stick is held. The
        # first frame remains precise, then reaches 10 m/s after five seconds.
        self._screen_control_min_speed = 0.10
        self._screen_control_max_speed = 10.0
        self._screen_control_acceleration = (
            self._screen_control_max_speed - self._screen_control_min_speed
        ) / 5.0
        self._screen_control_max_hold_seconds = 5.0
        self._screen_distance_hold_seconds = 0.0
        self._screen_distance_hold_direction = 0
        self._screen_size_hold_seconds = 0.0
        self._screen_size_hold_direction = 0

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def output_ready(self) -> bool:
        """Report readiness after the presenter-owned Filament Engine is ready."""
        return self._initialized

    def inference_backpressure_active(self) -> bool:
        """Whether an extra inference would compete with queued XR presentation."""
        if not self._initialized or not self.session_running:
            return False
        if self._projection_busy.is_set() or not self._presenter_commands.empty():
            return True
        with self._output_lock:
            return self._pending_output is not None or self._rendering_output is not None

    @property
    def source_ready_semaphore_available(self) -> bool:
        """Whether Projection Composer can consume exported source semaphores."""
        return bool(self._vulkan_projection_composer_requested and self.vulkan is not None)

    def _controller_ambient_light_color(self) -> tuple[float, float, float]:
        """Return the room ambient color without controller compensation."""
        if self._controller_ambient_light_color_override is not None:
            return tuple(self._controller_ambient_light_color_override)
        return tuple(float(component) for component in self._filament_ambient_light_color)

    def _controller_hdr_ambient_light_color(self) -> tuple[float, float, float]:
        """Return controller ambient light isolated from the room Scene."""
        multiplier = max(
            0.0,
            float(getattr(self._controller_brand, "ambient_light_multiplier", 1.0)),
        )
        base_color = (
            self._controller_hdr_ambient_light_color_override
            if self._controller_hdr_ambient_light_color_override is not None
            else self._filament_ambient_light_color
        )
        return tuple(
            float(component) * multiplier
            for component in base_color
        )

    def _apply_filament_bridge_lighting(self, bridge=None) -> None:
        bridge = bridge or self.filament_bridge
        if bridge is None:
            return
        controller_ambient_intensity = (
            self._controller_hdr_ambient_light_intensity_lux
            if self._controller_hdr_lighting
            else self._controller_ambient_light_intensity_lux
        )
        if hasattr(bridge, "set_lighting_config") and bridge.set_lighting_config(
            environment_ambient_color=self._controller_ambient_light_color(),
            environment_ambient_intensity_lux=self._filament_ambient_light_intensity_lux,
            controller_ambient_color=self._controller_hdr_ambient_light_color(),
            controller_ambient_intensity_lux=controller_ambient_intensity,
            controller_ambient_enabled=True,
            head_light_color=self._filament_fill_light_color,
            head_light_intensity_candela=(
                self._controller_light_intensity_candela
                * self._filament_fill_light_intensity
                * self._controller_head_light_weight
            ),
            head_light_offset=self._controller_head_light_offset,
            head_light_falloff=self._controller_head_light_falloff,
            head_light_cast_shadows=self._controller_head_light_cast_shadows,
            top_light_color=self._controller_top_light_color,
            top_light_intensity_candela=(
                self._controller_light_intensity_candela
                * self._filament_fill_light_intensity
                * self._controller_top_light_weight
            ),
            top_light_offset=self._controller_top_light_offset,
            top_light_falloff=self._controller_top_light_falloff,
            top_light_cast_shadows=self._controller_top_light_cast_shadows,
        ):
            return
        if hasattr(bridge, "set_ambient_light"):
            bridge.set_ambient_light(self._controller_ambient_light_color())
        if hasattr(bridge, "set_controller_ambient_light"):
            bridge.set_controller_ambient_light(
                self._controller_hdr_ambient_light_color(), True
            )
        if hasattr(bridge, "set_fill_light"):
            bridge.set_fill_light(
                self._filament_fill_light_color,
                self._filament_fill_light_intensity,
                self._filament_fill_light_direction,
            )

    def _apply_controller_material_profile(self, bridge=None, brand=None) -> None:
        bridge = bridge or self.filament_bridge
        brand = brand or self._controller_brand
        setter = getattr(bridge, "set_controller_material_override", None)
        if bridge is None or brand is None or not callable(setter):
            return
        roughness = getattr(brand, "material_roughness_factor", None)
        metallic = getattr(brand, "material_metallic_factor", None)
        specular = getattr(brand, "material_specular_color_factor", None)
        if roughness is None and metallic is None and specular is None:
            return
        if setter(
            roughness_factor=roughness,
            metallic_factor=metallic,
            specular_color_factor=specular,
        ):
            print(
                "[OpenXRViewer] Controller material override: "
                f"brand={brand.name} roughness={roughness} "
                f"metallic={metallic} specular={specular}",
                flush=True,
            )

    def _update_controller_screen_light(
        self, frame: VulkanStereoOutputFrame | None, bridge: Any
    ) -> None:
        setter = getattr(bridge, "set_controller_screen_light", None)
        metadata = dict(getattr(frame, "metadata", None) or {})
        sample = metadata.get("screen_light_linear_rgb")
        active = bool(
            callable(setter)
            and self._controller_screen_light_enabled
            and self._filament_screen is not None
            and isinstance(sample, (list, tuple))
            and len(sample) >= 3
        )
        if not callable(setter):
            return
        if not active:
            if self._controller_screen_light_applied:
                setter(
                    (0.0, 0.0, 0.0), 0.0, (0.0, 0.0, 1.0),
                    False, False,
                )
            self._controller_screen_light_applied = False
            self._controller_screen_light_smoothed_color.fill(0.0)
            self._controller_screen_light_smoothed_intensity = 0.0
            return

        rgb = np.maximum(0.0, np.asarray(sample[:3], dtype=np.float64))
        luminance = float(np.dot(rgb, np.asarray((0.2126, 0.7152, 0.0722))))
        max_luminance = max(0.0, float(self._controller_screen_light_max_luminance))
        target_luminance = min(luminance, max_luminance)
        maximum = float(np.max(rgb))
        if maximum > 1e-6 and target_luminance > 0.0:
            chroma = rgb / maximum
            saturation = max(0.0, min(1.0, float(
                self._controller_screen_light_saturation
            )))
            target_color = (1.0 - saturation) + saturation * chroma
        else:
            target_color = np.zeros(3, dtype=np.float64)
        screen_pose = self._filament_screen_pose_mat4().astype(np.float64)
        screen_position = screen_pose[:3, 3]
        controller_positions = [
            np.asarray(matrix[:3, 3], dtype=np.float64)
            for matrix in (self._grip_mat_l, self._grip_mat_r)
            if matrix is not None
        ]
        if controller_positions:
            light_target = np.mean(controller_positions, axis=0)
        elif self._head_position_w is not None:
            light_target = np.asarray(self._head_position_w, dtype=np.float64)
        else:
            light_target = screen_position + screen_pose[:3, 2]
        light_direction = light_target - screen_position
        direction_length = float(np.linalg.norm(light_direction))
        if direction_length > 1e-6:
            light_direction /= direction_length
        else:
            light_direction = screen_pose[:3, 2]
        target_intensity = (
            max(0.0, float(self._controller_screen_light_intensity_lux))
            * target_luminance
        )
        smoothing_seconds = max(
            0.0, float(self._controller_screen_light_smoothing_seconds)
        )
        dt = max(0.0, min(0.25, float(self._last_frame_dt)))
        alpha = 1.0 if smoothing_seconds <= 0.0 else (
            1.0 - math.exp(-dt / smoothing_seconds)
        )
        self._controller_screen_light_smoothed_color += alpha * (
            target_color - self._controller_screen_light_smoothed_color
        )
        self._controller_screen_light_smoothed_intensity += alpha * (
            target_intensity - self._controller_screen_light_smoothed_intensity
        )
        applied = setter(
            tuple(float(value) for value in self._controller_screen_light_smoothed_color),
            float(self._controller_screen_light_smoothed_intensity),
            tuple(float(value) for value in light_direction),
            bool(self._controller_screen_light_cast_shadows),
            True,
        )
        if not applied:
            return
        self._controller_screen_light_applied = True
        status = str(metadata.get("screen_light_sample_path", "unknown"))
        if status != self._controller_screen_light_status:
            self._controller_screen_light_status = status
            print(
                "Filament controller screen light active: "
                f"sample={status} foreground_only=True",
                flush=True,
            )

    def _update_environment_screen_lights(
        self, frame: VulkanStereoOutputFrame | None, bridge: Any
    ) -> None:
        setter = getattr(bridge, "set_environment_screen_area_light", None)
        metadata = dict(getattr(frame, "metadata", None) or {})
        samples = metadata.get("screen_edge_light_linear_rgb")
        active = bool(
            callable(setter)
            and self._environment_screen_light_enabled
            and self.config.filament_glb_path
            and self._filament_screen is not None
            and isinstance(samples, (list, tuple))
            and len(samples) == 24
        )
        if not callable(setter):
            return
        if not active:
            if self._environment_screen_light_applied:
                setter(
                    (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.0),
                    half_width=1.0, half_height=1.0, intensity=0.0,
                    enabled=False,
                )
            self._environment_screen_light_applied = False
            return

        screen_pose = self._filament_screen_pose_mat4().astype(np.float64)
        center = screen_pose[:3, 3]
        normal = screen_pose[:3, 2]
        width = float(self._filament_screen[1])
        height = float(self._filament_screen[2])
        saturation = max(0.0, min(
            1.0, float(self._environment_screen_light_saturation)
        ))
        max_luminance = max(
            0.0, float(self._environment_screen_light_max_luminance)
        )
        rgb = np.mean(np.maximum(
            0.0,
            np.asarray([sample[:3] for sample in samples], dtype=np.float64),
        ), axis=0)
        luminance = float(np.dot(rgb, (0.2126, 0.7152, 0.0722)))
        if max_luminance > 0.0 and luminance > max_luminance:
            rgb *= max_luminance / max(luminance, 1e-6)
            luminance = max_luminance
        gray = np.full(3, luminance, dtype=np.float64)
        color = np.maximum(0.0, gray + (rgb - gray) * saturation)
        # Room EV is applied later to the complete Filament resolve image.
        # Counter-scale only these virtual screen emitters so changing room
        # brightness dims the authored room but not the screen reflection.
        exposure_compensation = 2.0 ** (-max(
            -8.0, min(8.0, float(self._filament_scene_exposure))
        ))
        intensity = max(
            0.0, float(self._environment_screen_area_light_intensity)
        ) * exposure_compensation
        applied = setter(
            tuple(float(value) for value in center),
            tuple(float(value) for value in normal),
            tuple(float(value) for value in color),
            half_width=width * 0.5, half_height=height * 0.5,
            intensity=intensity,
            enabled=True,
        )
        if not applied:
            return
        self._environment_screen_light_applied = True
        status = str(metadata.get("screen_light_sample_path", "unknown"))
        if status != self._environment_screen_light_status:
            self._environment_screen_light_status = status
            print(
                "[OpenXRViewer] Filament room screen reflection active: "
                f"sample={status} mode=screen_area "
                f"intensity={intensity:.2f} luminance={luminance:.3f} "
                "environment_only=True",
                flush=True,
            )

    def initialize(self) -> None:
        if self._initialized:
            return
        self.exit_requested = False
        self.frame_count = 0
        self.session_state = None
        self.xr = _import_openxr()
        xr = self.xr
        available_extensions = {
            _decode_name(item.extension_name)
            for item in xr.enumerate_instance_extension_properties()
        }
        if self._VULKAN_EXTENSION not in available_extensions:
            raise OpenXrVulkanUnavailableError(
                f"OpenXR runtime does not expose {self._VULKAN_EXTENSION}"
            )

        optional_equirect = getattr(
            xr, "KHR_COMPOSITION_LAYER_EQUIRECT2_EXTENSION_NAME", None
        )
        enabled_extensions = [self._VULKAN_EXTENSION]
        if optional_equirect and optional_equirect in available_extensions:
            enabled_extensions.append(optional_equirect)
            self._openxr_equirect_supported = hasattr(
                xr, "CompositionLayerEquirect2KHR"
            )
        print(
            "[OpenXRViewer] Equirect background capability: "
            f"extension={bool(optional_equirect and optional_equirect in available_extensions)} "
            f"binding={hasattr(xr, 'CompositionLayerEquirect2KHR')} "
            f"active={self._openxr_equirect_supported}",
            flush=True,
        )
        try:
            self.instance = xr.create_instance(
                xr.InstanceCreateInfo(
                    application_info=xr.ApplicationInfo(
                        application_name=self.config.application_name,
                        application_version=1,
                        engine_name="D2S",
                        engine_version=1,
                        api_version=xr.Version(1, 0, 0),
                    ),
                    enabled_extension_names=enabled_extensions,
                )
            )
            self.system_id = xr.get_system(
                self.instance,
                xr.SystemGetInfo(form_factor=xr.FormFactor.HEAD_MOUNTED_DISPLAY),
            )
            requirements = _get_vulkan_graphics_requirements2(
                xr, self.instance, self.system_id
            )
            api_version = _select_vulkan_api_version(
                requirements, self.config.requested_vulkan_version
            )
            self._create_vulkan_objects(api_version)
            self._create_session_and_swapchains()
            self._report_projection_composer_boundary()
            self._xr_instance = self.instance
            self._xr_session = self.session
            self._xr_space = self.reference_space
            self._init_controller_actions()
            self._load_filament_profile()
            # Filament Engine and all resources remain owned by the Presenter
            # thread. Filament rejects rendering from a thread that was not
            # adopted by its JobSystem, so native GLB loading cannot migrate
            # the Engine to a background thread.
            self._initialize_filament_bridges()
            self._initialize_msdf_text_atlas()
            self._initialize_msdf_quad_renderer()
            self._initialized = True
        except Exception:
            self.close()
            raise

    def poll_events(self) -> None:
        self._ensure_initialized()
        xr = self.xr
        while True:
            try:
                event = xr.poll_event(self.instance)
            except xr.EventUnavailable:
                return

            if event.type == xr.StructureType.EVENT_DATA_SESSION_STATE_CHANGED:
                changed = ctypes.cast(
                    ctypes.byref(event),
                    ctypes.POINTER(xr.EventDataSessionStateChanged),
                ).contents
                self.session_state = changed.state
                if changed.state == xr.SessionState.READY and not self.session_running:
                    xr.begin_session(
                        self.session,
                        xr.SessionBeginInfo(
                            primary_view_configuration_type=self._view_configuration_type
                        ),
                    )
                    self.session_running = True
                elif changed.state == xr.SessionState.STOPPING and self.session_running:
                    xr.end_session(self.session)
                    self.session_running = False
                elif changed.state in (
                    xr.SessionState.EXITING,
                    xr.SessionState.LOSS_PENDING,
                ):
                    self.exit_requested = True
            elif event.type == xr.StructureType.EVENT_DATA_REFERENCE_SPACE_CHANGE_PENDING:
                self._recreate_reference_space_after_runtime_change()
            elif event.type == xr.StructureType.EVENT_DATA_INSTANCE_LOSS_PENDING:
                self.exit_requested = True

    def _recreate_reference_space_after_runtime_change(self) -> None:
        """Recreate the base XR space after a runtime relocation event."""
        if self.xr is None or self.session is None or self._reference_space_type is None:
            self._profile_space_applied = False
            return
        try:
            new_space = self.xr.create_reference_space(
                self.session,
                self.xr.ReferenceSpaceCreateInfo(
                    reference_space_type=self._reference_space_type
                ),
            )
        except Exception as exc:
            # Keep the current space alive if the runtime cannot create a
            # replacement; the next valid view will retry profile application.
            print(
                f"[OpenXRViewer] Reference space change pending; "
                f"recreate failed: {exc}",
                flush=True,
            )
            self._profile_space_applied = False
            return
        old_space = self.reference_space
        self.reference_space = new_space
        self._xr_space = new_space
        self._profile_space_applied = False
        self._profile_space_calibration_pass = 0
        self._profile_space_pose_in_reference = np.eye(4, dtype=np.float32)
        self._profile_reference_head_anchor = None
        self._profile_space_preserve_anchor = False
        self._profile_initial_head = None
        self._profile_alignment_logged = False
        self._head_position_w = None
        self._head_forward_w = None
        print(
            "[OpenXRViewer] Reference space change accepted: "
            "runtime recenter will be combined with the active room seat",
            flush=True,
        )
        if old_space is not None:
            try:
                self.xr.destroy_space(old_space)
            except Exception:
                pass

    def run_frame(self) -> bool:
        frame_started = time.perf_counter()
        self._ensure_initialized()
        events_started = time.perf_counter()
        self.poll_events()
        if self._on_breakdown_add_time is not None:
            self._on_breakdown_add_time(
                "openxr_events", time.perf_counter() - events_started
            )
        if self.exit_requested:
            return False
        if not self.session_running:
            self._notify_headset_waiting()
            time.sleep(0.01)
            return True

        xr = self.xr
        # Apply a released menu slider at the frame boundary, before the next
        # xrWaitFrame/xrBeginFrame pair. Swapchain dimensions are immutable.
        self._apply_pending_openxr_render_scale()
        wait_started = time.perf_counter()
        frame_state = xr.wait_frame(self.session)
        if self._on_breakdown_add_time is not None:
            self._on_breakdown_add_time(
                "openxr_wait_frame", time.perf_counter() - wait_started
            )
        # Keep xrWaitFrame at the frame boundary. Runtime-output conversion can
        # enqueue Vulkan work and must not delay the runtime's pacing decision.
        commands_started = time.perf_counter()
        self._drain_presenter_commands()
        if self._on_breakdown_add_time is not None:
            self._on_breakdown_add_time(
                "openxr_presenter_commands", time.perf_counter() - commands_started
            )
        if self._on_breakdown_inc is not None:
            self._on_breakdown_inc("openxr_loop", 1)
            self._on_breakdown_inc(
                "openxr_should_render" if frame_state.should_render else "openxr_no_render",
                1,
            )
        previous_frame_now = self._frame_now
        self._frame_now = time.perf_counter()
        if previous_frame_now > 0.0:
            self._last_frame_dt = max(
                0.001, min(0.1, self._frame_now - previous_frame_now)
            )
        if frame_state.should_render:
            self._notify_headset_active()
        else:
            self._notify_headset_waiting()
        controls_started = time.perf_counter()
        try:
            if self._on_breakdown_inc is not None:
                self._on_breakdown_inc("openxr_input_sample", 1)
            self._sync_controller_inputs(1.0 / 90.0)
            self._update_aim_poses(frame_state.predicted_display_time)
            self._update_grip_poses(frame_state.predicted_display_time)
            self._smooth_controller_poses()
            self._grip_l_now = bool(self._controller_input(0).get("grip", 0.0) > 0.5)
            self._grip_r_now = bool(self._controller_input(1).get("grip", 0.0) > 0.5)
            menu_consumed = self._handle_settings_menu_input()
            if not menu_consumed:
                self._handle_keyboard_input()
                self._handle_vulkan_pointer_input()
                self._handle_controller_shortcuts()
                self._handle_controller_guide_input(self._last_frame_dt)
            self._last_controller_input_error = None
        except Exception as exc:
            # Keep one bad optional input path from terminating XR, but make
            # the failure observable instead of silently disabling all
            # keyboard, drag, and shortcut handling for the session.
            error = f"{type(exc).__name__}: {exc}"
            if error != self._last_controller_input_error:
                self._last_controller_input_error = error
                if type(exc).__name__ == "SessionNotFocused":
                    message = (
                        "[OpenXRViewer] Controller input deferred: "
                        "OpenXR session is not focused"
                    )
                else:
                    message = f"[OpenXRViewer] Controller input update failed: {error}"
                print(
                    message,
                    flush=True,
                )
        finally:
            if self._on_breakdown_add_time is not None:
                self._on_breakdown_add_time(
                    "openxr_controls", time.perf_counter() - controls_started
                )
        begin_started = time.perf_counter()
        xr.begin_frame(self.session)
        if self._on_breakdown_add_time is not None:
            self._on_breakdown_add_time(
                "openxr_begin_frame", time.perf_counter() - begin_started
            )
        layer_structures: list[Any] = []
        layer_pointers: list[Any] = []
        try:
            if frame_state.should_render:
                locate_started = time.perf_counter()
                locate_call_started = time.perf_counter()
                view_state, views = xr.locate_views(
                    self.session,
                    xr.ViewLocateInfo(
                        view_configuration_type=self._view_configuration_type,
                        display_time=frame_state.predicted_display_time,
                        space=self.reference_space,
                    ),
                )
                if self._on_breakdown_add_time is not None:
                    self._on_breakdown_add_time(
                        "openxr_locate_call", time.perf_counter() - locate_call_started
                    )
                valid_flags = (
                    xr.ViewStateFlags.POSITION_VALID_BIT
                    | xr.ViewStateFlags.ORIENTATION_VALID_BIT
                )
                if view_state.view_state_flags & valid_flags == valid_flags:
                    # The first rebase can race VDXR's STAGE-floor update. Apply at
                    # most one calibration pass per XR tick so the final pass uses
                    # the next frame's measured pose rather than another locate in
                    # the same runtime tick.
                    if self._apply_profile_reference_space(views):
                        locate_call_started = time.perf_counter()
                        view_state, views = xr.locate_views(
                            self.session,
                            xr.ViewLocateInfo(
                                view_configuration_type=self._view_configuration_type,
                                display_time=frame_state.predicted_display_time,
                                space=self.reference_space,
                            ),
                        )
                        if self._on_breakdown_add_time is not None:
                            self._on_breakdown_add_time(
                                "openxr_locate_call",
                                time.perf_counter() - locate_call_started,
                            )
                    view_prepare_started = time.perf_counter()
                    self._cache_head_position(views)
                    self._report_profile_alignment()
                    self._initialize_filament_screen_from_head()
                    if self._on_breakdown_add_time is not None:
                        self._on_breakdown_add_time(
                            "openxr_view_prepare",
                            time.perf_counter() - view_prepare_started,
                        )
                    output_lock_started = time.perf_counter()
                    with self._output_lock:
                        output_frame = self._pending_output
                    if (
                        self._filament_multiview_projection_diagnostic
                        and output_frame is not None
                    ):
                        # This diagnostic renders only Filament. Do not retain
                        # an inference output as the displayed SBS frame when
                        # neither eye entered source-image sampling.
                        self._abort_output_frame(output_frame)
                        output_frame = None
                    if self._on_breakdown_add_time is not None:
                        self._on_breakdown_add_time(
                            "openxr_output_lock",
                            time.perf_counter() - output_lock_started,
                        )
                    metrics_started = time.perf_counter()
                    self._update_tool_overlay_metrics(output_frame)
                    if self._on_breakdown_add_time is not None:
                        self._on_breakdown_add_time(
                            "openxr_overlay_metrics",
                            time.perf_counter() - metrics_started,
                        )
                    # Match the legacy frame gate: runtime rendering readiness
                    # is separate from the availability of a fresh stereo frame.
                    if (
                        self._pending_output is None
                        and not self._has_presented_frame
                        and not self._projection_array_eye_diagnostic
                        and not self._vulkan_multiview_eye_diagnostic
                        and not self._filament_multiview_projection_diagnostic
                    ):
                        if not self._source_frame_wait_logged:
                            self._source_frame_wait_logged = True
                            print(
                                "[OpenXRViewer] OpenXR render ready; "
                                "waiting for first runtime eye frame",
                                flush=True,
                            )
                        layer = None
                    else:
                        self._source_frame_wait_logged = False
                        # Render the world at the current headset pose on
                        # every XR tick; only inference input may be reused.
                        projection_started = time.perf_counter()
                        layer = self._render_projection_layer(views, output_frame)
                        if self._on_breakdown_add_time is not None:
                            self._on_breakdown_add_time(
                                "openxr_projection_layer",
                                time.perf_counter() - projection_started,
                            )
                    if layer is not None:
                        layer_assembly_started = time.perf_counter()
                        panorama_layer = self._prepare_panorama_layer()
                        if panorama_layer is not None:
                            layer_structures.insert(0, panorama_layer)
                            layer_pointers.insert(0, ctypes.pointer(panorama_layer))
                        layer_structures.append(layer)
                        layer_pointers.append(ctypes.pointer(layer))
                        try:
                            quad_started = time.perf_counter()
                            self._last_quad_layers = self._render_quad_layers(output_frame)
                            if self._on_breakdown_add_time is not None:
                                self._on_breakdown_add_time(
                                    "openxr_quad_total",
                                    time.perf_counter() - quad_started,
                                )
                            if (
                                output_frame is not None
                                and not self._screen_quad_reprojection_active
                            ):
                                commit_started = time.perf_counter()
                                self._commit_output_frame(output_frame)
                                if self._on_breakdown_add_time is not None:
                                    self._on_breakdown_add_time(
                                        "openxr_output_commit",
                                        time.perf_counter() - commit_started,
                                    )
                        except Exception:
                            if output_frame is not None:
                                self._abort_output_frame(output_frame)
                            raise
                        self._has_presented_frame = True
                        layer_pointers_started = time.perf_counter()
                        layer_structures.extend(self._last_quad_layers)
                        layer_pointers.extend(
                            ctypes.pointer(item) for item in self._last_quad_layers
                        )
                        controller_layer = self._render_controller_composition_layer(
                            views
                        )
                        if controller_layer is not None:
                            layer_structures.append(controller_layer)
                            layer_pointers.append(ctypes.pointer(controller_layer))
                        controller_proxy_layer = self._render_vulkan_controller_proxy_layer(
                            views
                        )
                        if controller_proxy_layer is not None:
                            layer_structures.append(controller_proxy_layer)
                            layer_pointers.append(ctypes.pointer(controller_proxy_layer))
                        if self._on_breakdown_add_time is not None:
                            self._on_breakdown_add_time(
                                "openxr_layer_pointers",
                                time.perf_counter() - layer_pointers_started,
                            )
                        if self._on_breakdown_add_time is not None:
                            self._on_breakdown_add_time(
                                "openxr_layer_assembly",
                                time.perf_counter() - layer_assembly_started,
                            )
                if self._on_breakdown_add_time is not None:
                    self._on_breakdown_add_time(
                        "openxr_locate_total", time.perf_counter() - locate_started
                    )
        finally:
            if not bool(getattr(self.vulkan, "device_lost", False)):
                end_info = xr.FrameEndInfo(
                    display_time=frame_state.predicted_display_time,
                    environment_blend_mode=self._environment_blend_mode,
                    layer_count=len(layer_pointers),
                    layers=layer_pointers or None,
                )
                end_started = time.perf_counter()
                xr.end_frame(self.session, end_info)
                self._record_xr_presented_frame()
                if self._on_breakdown_add_time is not None:
                    self._on_breakdown_add_time(
                        "openxr_end_frame", time.perf_counter() - end_started
                    )
        self.frame_count += 1
        if self._on_breakdown_add_time is not None:
            self._on_breakdown_add_time(
                "openxr_frame_total", time.perf_counter() - frame_started
            )
        return not self.exit_requested

    def _set_shortcut_panel(self, name: str | None) -> None:
        # Legacy Menu/A cycle: hidden -> FPS -> FPS + vertical screen guide
        # -> hidden. The guide never replaces the FPS panel at state 2.
        # Menu and B panels are mutually exclusive so a stale guide cannot
        # remain visible when the user switches to the other control path.
        self._hand_panel_cycle = 0
        self._hand_fps_visible = False
        self._hand_operation_guide_visible = False
        self._fps_overlay_visible = name in {"fps", "guide"}
        self._screen_operation_guide_visible = name == "guide"
        self._aperture_visible = name == "aperture"
        self._operation_guide_visible = self._screen_operation_guide_visible

    def _set_hand_shortcut_panel(self, name: str | None) -> None:
        # Legacy B cycle: hidden -> hand FPS -> hand FPS + hand guide -> hidden.
        # Selecting the B panel clears the Menu-owned screen panel first.
        self._status_panel_cycle = 0
        self._fps_overlay_visible = False
        self._screen_operation_guide_visible = False
        self._aperture_visible = False
        self._hand_fps_visible = name in {"fps", "guide"}
        self._hand_operation_guide_visible = name == "guide"
        # Keep the legacy compatibility flag true for the controller-attached
        # B-panel; Menu uses _screen_operation_guide_visible above.
        self._operation_guide_visible = (
            self._screen_operation_guide_visible or self._hand_operation_guide_visible
        )

    def _set_shortcut_skybox_brightness(self, brightness: float) -> None:
        self._filament_skybox_brightness = max(0.0, float(brightness))
        if self.filament_bridge is not None:
            self.filament_bridge.set_skybox_brightness(
                self._filament_skybox_brightness
            )

    def _apply_filament_scene_exposure_to_bridge(self, bridge=None) -> None:
        """Apply exposure only when Filament owns the final color pipeline."""
        bridge = self.filament_bridge if bridge is None else bridge
        if bridge is None or self._multiview_active:
            # The layered producer must keep Filament post-processing disabled;
            # VulkanProjectionScreenPass applies this exposure once during the
            # final HDR resolve. Calling the native setter here would rebuild
            # ColorGrading and force the multiview room View back to PP=true.
            return
        bridge.set_scene_exposure(self._filament_scene_exposure)

    @staticmethod
    def _normalize_filament_glow_mode(value: Any) -> str:
        mode = str(value or "off").strip().lower()
        return {
            "none": "off",
            "false": "off",
            "0": "off",
            "screen": "glow",
        }.get(mode, mode) if mode in {
            "off", "none", "false", "0", "screen", "surround",
            "glow", "veil",
        } else "off"

    def _apply_filament_glow_profile_fields(self, values: dict[str, Any]) -> None:
        if "glow_mode" in values:
            self._filament_glow_mode = self._normalize_filament_glow_mode(
                values.get("glow_mode")
            )
        for key, attribute, minimum, maximum in (
            ("glow_intensity", "_filament_glow_intensity", 0.0, None),
            ("glow_width", "_filament_glow_width", 0.0, None),
            ("glow_intensity_multiplier", "_filament_glow_intensity_multiplier", 0.0, None),
            ("glow_shell_intensity_multiplier", "_filament_glow_shell_intensity_multiplier", 0.0, None),
            ("glow_shell_radius", "_filament_glow_shell_radius", 0.0, None),
            ("glow_shell_height", "_filament_glow_shell_height", 0.0, None),
            ("veil_intensity", "_veil_intensity", 0.0, None),
            ("veil_alpha", "_veil_alpha", 0.0, 1.0),
        ):
            if key not in values:
                continue
            try:
                number = max(float(minimum), float(values[key]))
                if maximum is not None:
                    number = min(float(maximum), number)
                setattr(self, attribute, number)
            except (TypeError, ValueError):
                continue

    def _cycle_filament_glow_mode(self) -> None:
        modes = ("surround", "glow", "veil", "off")
        current = self._normalize_filament_glow_mode(self._filament_glow_mode)
        if current not in modes:
            current = (
                "glow"
                if float(self._filament_glow_intensity_multiplier) > 0.0
                else "off"
            )
        next_mode = modes[(modes.index(current) + 1) % len(modes)]
        self._set_filament_glow_mode(next_mode)

    def _set_filament_glow_mode(self, mode: str) -> None:
        next_mode = self._normalize_filament_glow_mode(mode)
        self._filament_glow_mode = next_mode
        if next_mode == "off":
            self._filament_glow_intensity_multiplier = 0.0
            self._filament_glow_shell_intensity_multiplier = 0.0
        elif next_mode == "surround":
            self._filament_glow_intensity_multiplier = 0.0
            if self._filament_glow_shell_intensity_multiplier <= 0.0:
                self._filament_glow_shell_intensity_multiplier = (
                    self._filament_glow_shell_default_multiplier
                )
        else:
            self._filament_glow_shell_intensity_multiplier = 0.0
        if (
            next_mode not in {"off", "surround"}
            and self._filament_glow_intensity_multiplier <= 0.0
        ):
            self._filament_glow_intensity_multiplier = (
                self._filament_glow_default_multiplier
            )
        label = {
            "surround": "Surround Glow",
            "glow": "Glow",
            "veil": "Veil",
            "off": "Off",
        }[next_mode]
        self._preset_name_overlay = label
        self._preset_osd_show_t = time.perf_counter()
        self._last_filament_glow_status = None
        print(f"[OpenXRViewer] Glow mode: {next_mode}", flush=True)

    def _apply_filament_lighting_preset(
        self, preset: dict[str, Any], *, apply_bridge: bool = True
    ) -> None:
        """Apply the legacy environment lighting-preset fields to Filament."""
        if not isinstance(preset, dict):
            return
        for key, attribute in (
            ("preview_exposure", "_filament_scene_exposure"),
            ("preview_skybox_brightness", "_filament_skybox_brightness"),
            ("controller_head_light_intensity", "_filament_fill_light_intensity"),
            ("env_ambient_light_intensity_lux", "_filament_ambient_light_intensity_lux"),
            ("controller_ambient_light_intensity_lux", "_controller_ambient_light_intensity_lux"),
            ("controller_hdr_ambient_light_intensity_lux", "_controller_hdr_ambient_light_intensity_lux"),
            ("controller_light_intensity_candela", "_controller_light_intensity_candela"),
            ("controller_head_light_weight", "_controller_head_light_weight"),
            ("controller_top_light_weight", "_controller_top_light_weight"),
            ("controller_head_light_falloff", "_controller_head_light_falloff"),
            ("controller_top_light_falloff", "_controller_top_light_falloff"),
            ("controller_screen_light_intensity_lux", "_controller_screen_light_intensity_lux"),
            ("controller_screen_light_saturation", "_controller_screen_light_saturation"),
            ("controller_screen_light_max_luminance", "_controller_screen_light_max_luminance"),
            ("controller_screen_light_smoothing_seconds", "_controller_screen_light_smoothing_seconds"),
            ("controller_screen_light_sample_hz", "_controller_screen_light_sample_hz"),
            ("environment_screen_light_intensity_candela", "_environment_screen_light_intensity_candela"),
            ("environment_screen_light_saturation", "_environment_screen_light_saturation"),
            ("environment_screen_light_max_luminance", "_environment_screen_light_max_luminance"),
            ("environment_screen_light_smoothing_seconds", "_environment_screen_light_smoothing_seconds"),
            ("environment_screen_light_sample_hz", "_environment_screen_light_sample_hz"),
            ("environment_screen_light_falloff", "_environment_screen_light_falloff"),
            ("environment_screen_light_offset", "_environment_screen_light_offset"),
            ("glow_sample_hz", "_filament_glow_sample_hz"),
            ("glow_smoothing_seconds", "_filament_glow_smoothing_seconds"),
        ):
            if key in preset:
                try:
                    setattr(self, attribute, float(preset[key]))
                except (TypeError, ValueError):
                    pass
        for keys, attribute in (
            (("env_ambient_color", "ambient_color"), "_filament_ambient_light_color"),
            (("controller_head_light_color", "env_head_light_color", "head_light_color"), "_filament_fill_light_color"),
            (("controller_ambient_light_color",), "_controller_ambient_light_color_override"),
            (("controller_hdr_ambient_light_color",), "_controller_hdr_ambient_light_color_override"),
            (("controller_top_light_color",), "_controller_top_light_color"),
            (("controller_head_light_offset",), "_controller_head_light_offset"),
            (("controller_top_light_offset",), "_controller_top_light_offset"),
        ):
            for key in keys:
                value = preset.get(key)
                if isinstance(value, (list, tuple)) and len(value) >= 3:
                    try:
                        setattr(self, attribute, tuple(float(item) for item in value[:3]))
                    except (TypeError, ValueError):
                        pass
                    break
        for key, attribute in (
            ("controller_head_light_cast_shadows", "_controller_head_light_cast_shadows"),
            ("controller_top_light_cast_shadows", "_controller_top_light_cast_shadows"),
            ("controller_screen_light_enabled", "_controller_screen_light_enabled"),
            ("controller_screen_light_cast_shadows", "_controller_screen_light_cast_shadows"),
            ("environment_screen_light_enabled", "_environment_screen_light_enabled"),
            ("environment_screen_light_cast_shadows", "_environment_screen_light_cast_shadows"),
        ):
            if key in preset:
                setattr(self, attribute, bool(preset[key]))
        direction = preset.get("env_fill_light_direction", preset.get("fill_light_direction"))
        if isinstance(direction, (list, tuple)) and len(direction) >= 3:
            try:
                self._filament_fill_light_direction = tuple(
                    float(item) for item in direction[:3]
                )
            except (TypeError, ValueError):
                pass
        for attribute in (
            "_filament_ambient_light_intensity_lux",
            "_controller_ambient_light_intensity_lux",
            "_controller_hdr_ambient_light_intensity_lux",
            "_controller_light_intensity_candela",
            "_controller_head_light_weight",
            "_controller_top_light_weight",
        ):
            setattr(self, attribute, max(0.0, float(getattr(self, attribute))))
        self._controller_head_light_falloff = max(
            0.001, float(self._controller_head_light_falloff)
        )
        self._controller_top_light_falloff = max(
            0.001, float(self._controller_top_light_falloff)
        )
        self._controller_screen_light_intensity_lux = max(
            0.0, float(self._controller_screen_light_intensity_lux)
        )
        self._controller_screen_light_saturation = max(
            0.0, min(1.0, float(self._controller_screen_light_saturation))
        )
        self._controller_screen_light_max_luminance = max(
            0.0, float(self._controller_screen_light_max_luminance)
        )
        self._controller_screen_light_smoothing_seconds = max(
            0.0, float(self._controller_screen_light_smoothing_seconds)
        )
        self._controller_screen_light_sample_hz = max(
            1.0, float(self._controller_screen_light_sample_hz)
        )
        self._environment_screen_light_intensity_candela = max(
            0.0, float(self._environment_screen_light_intensity_candela)
        )
        self._environment_screen_light_saturation = max(
            0.0, min(1.0, float(self._environment_screen_light_saturation))
        )
        self._environment_screen_light_max_luminance = max(
            0.0, float(self._environment_screen_light_max_luminance)
        )
        self._environment_screen_light_smoothing_seconds = max(
            0.0, float(self._environment_screen_light_smoothing_seconds)
        )
        self._environment_screen_light_sample_hz = max(
            1.0, float(self._environment_screen_light_sample_hz)
        )
        self._environment_screen_light_falloff = max(
            0.01, float(self._environment_screen_light_falloff)
        )
        self._environment_screen_light_offset = max(
            0.0, float(self._environment_screen_light_offset)
        )
        self._filament_glow_sample_hz = max(
            1.0, float(self._filament_glow_sample_hz)
        )
        self._filament_glow_smoothing_seconds = max(
            0.0, float(self._filament_glow_smoothing_seconds)
        )
        self._apply_filament_glow_profile_fields(preset)
        if not apply_bridge or self.filament_bridge is None:
            return
        bridge = self.filament_bridge
        self._apply_filament_scene_exposure_to_bridge(bridge)
        bridge.set_skybox_brightness(self._filament_skybox_brightness)
        self._apply_filament_bridge_lighting(bridge)

    def _cycle_shortcut_screen_preset(self) -> None:
        if self._filament_screen is None:
            return
        self._shortcut_screen_preset_index = (
            self._shortcut_screen_preset_index + 1
        ) % len(self._shortcut_screen_presets)
        self._apply_shortcut_screen_preset(self._shortcut_screen_preset_index)

    def _apply_shortcut_screen_preset(self, index: int) -> None:
        """Apply the legacy screen preset size, distance, and head-facing pose."""
        if self._filament_screen is None or not self._shortcut_screen_presets:
            return
        index = int(index) % len(self._shortcut_screen_presets)
        _name, width, distance = self._shortcut_screen_presets[index]
        old_position, old_width, old_height, rotation = self._filament_screen
        if self._head_position_w is not None and self._head_forward_w is not None:
            hx, _hy, hz = self._head_position_w
            fx, _fy, fz = self._head_forward_w
            horizontal = math.sqrt(float(fx) * float(fx) + float(fz) * float(fz))
            if horizontal > 1e-4:
                fx /= horizontal
                fz /= horizontal
            else:
                fx, fz = 0.0, -1.0
            position = (
                float(hx) + float(fx) * float(distance),
                float(self._initial_head_y),
                float(hz) + float(fz) * float(distance),
            )
            rotation = (
                math.degrees(math.atan2(-float(fx), -float(fz))),
                0.0,
                0.0,
            )
        else:
            position = (0.0, 0.0, -float(distance))
            rotation = (0.0, 0.0, 0.0)
        height = float(width) * float(old_height) / max(float(old_width), 1e-6)
        self._filament_screen = (
            tuple(float(value) for value in position),
            float(width),
            height,
            rotation,
        )
        self._preset_name_overlay = (
            f"{_name}  {float(width):.2f} x {float(height):.2f} m"
            f"  @ {float(distance):.2f} m"
        )
        self._preset_osd_show_t = time.perf_counter()

    def _controller_callback_depth_strength(self) -> float | None:
        """Read the synchronously updated runtime value when available."""
        callback = self._on_controller_shortcut
        owner = getattr(callback, "__self__", None)
        context = getattr(owner, "context", None)
        state = getattr(context, "openxr_state", None)
        snapshot = getattr(state, "runtime_settings_snapshot", None)
        value = getattr(snapshot, "depth_strength", None)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        return max(0.0, value)

    def _dispatch_controller_shortcut(self, action: str, **values) -> None:
        """Apply shared shortcut actions to Vulkan-owned presentation state."""
        if action == "cycle_status_panel":
            self._status_panel_cycle = (self._status_panel_cycle + 1) % 3
            self._set_shortcut_panel(
                (None, "fps", "guide")[self._status_panel_cycle]
            )
        elif action == "cycle_hand_panel":
            # Match the legacy B long-press state machine exactly.
            self._hand_panel_cycle = (self._hand_panel_cycle + 1) % 3
            self._set_hand_shortcut_panel(
                (None, "fps", "guide")[self._hand_panel_cycle]
            )
        elif action == "toggle_keyboard":
            self._keyboard_visible = not self._keyboard_visible
            self._keyboard_position_offset[:] = 0.0
            self._keyboard_grab_anchor = None
            self._kb_grab_local_l = None
            self._kb_grab_local_r = None
            if self._keyboard_visible:
                screen_width = float(self._filament_screen[1]) if self._filament_screen else 2.4
                self._keyboard_width = max(0.3, screen_width * 0.8)
                self._keyboard_height = self._keyboard_width * _KB_TEX_H / float(_KB_TEX_W)
                self._keyboard_keys = []
                self._keyboard_texture_key = None
        elif action == "reset_screen":
            if self._filament_screen_profile_authored:
                if self._filament_screen_initial is not None:
                    self._filament_screen = self._filament_screen_initial
                    self._preset_name_overlay = "Screen Reset"
                    self._preset_osd_show_t = time.perf_counter()
            else:
                self._shortcut_screen_preset_index = 5
                self._apply_shortcut_screen_preset(5)
        elif action == "cycle_screen_preset":
            self._cycle_shortcut_screen_preset()
        elif action == "toggle_screen_shape":
            self._screen_curved = not self._screen_curved
            self._screen_curve_half_angle = 0.72 if self._screen_curved else 0.0
            self._preset_name_overlay = (
                "Curved Screen" if self._screen_curved else "Flat Screen"
            )
            self._preset_osd_show_t = time.perf_counter()
        elif action == "toggle_background":
            if self._filament_skybox_brightness > 0.0:
                self._shortcut_saved_skybox_brightness = (
                    self._filament_skybox_brightness
                )
                self._set_shortcut_skybox_brightness(0.0)
            else:
                self._set_shortcut_skybox_brightness(
                    self._shortcut_saved_skybox_brightness or 1.0
                )
        elif action == "cycle_environment_light":
            # The shared shortcut name predates the renderer split. In v2.5,
            # releasing X after 1-4 seconds cycles the screen-edge effects;
            # it does not cycle room-light presets.
            self._cycle_filament_glow_mode()
        elif action == "toggle_passthrough":
            bridge = self.filament_bridge
            if bridge is None or not getattr(
                bridge, "passthrough_backdrop_abi_available", False
            ):
                self._unsupported_shortcut_actions.add(action)
                return
            self._passthrough_backdrop = not self._passthrough_backdrop
            bridge.set_passthrough_backdrop(self._passthrough_backdrop)
        elif action == "switch_controller_brand":
            self._switch_shortcut_controller_brand()
        elif action == "toggle_controller_calibration":
            self._controller_calibration_mode = not self._controller_calibration_mode
            print(
                "[OpenXRViewer] Controller calibration: "
                f"{'on' if self._controller_calibration_mode else 'off'}",
                flush=True,
            )
        elif action == "adjust_controller_calibration":
            self._controller_calibration_offset[1] += float(values.get("offset_y", 0.0))
            self._controller_calibration_offset[2] += float(values.get("offset_z", 0.0))
            self._controller_calibration_rotation_deg += float(
                values.get("rotation_deg", 0.0)
            )
        elif action == "save_controller_calibration":
            self._save_shortcut_controller_calibration()
        elif action == "rotate_screen":
            if self._screen_ray_hit_for_hand(0) is not None:
                self._adjust_shortcut_screen_rotation(
                    float(values.get("yaw_delta", 0.0)),
                    float(values.get("pitch_delta", 0.0)),
                )
        elif action == "resize_screen":
            # The pointer path below already applies the legacy exponential
            # right-grip/right-stick curve. Do not add the guide mixin's old
            # fixed-speed delta a second time in the same XR frame.
            if (
                not self._right_grip_screen_pointer_applied
                and self._screen_ray_hit_for_hand(1) is not None
            ):
                self._adjust_shortcut_screen_size(
                    float(values.get("width_delta", 0.0)),
                    float(values.get("distance_delta", 0.0)),
                )
        elif action == "rotate_keyboard":
            self._keyboard_rotation_offset += np.asarray(
                (values.get("yaw_delta", 0.0), values.get("pitch_delta", 0.0)),
                dtype=np.float64,
            )
        elif action == "orbit_keyboard":
            self._keyboard_position_offset[0] += float(values.get("horizontal", 0.0)) * 0.4
            self._keyboard_position_offset[1] += float(values.get("vertical", 0.0)) * 0.4
        elif action == "resize_keyboard":
            self._adjust_shortcut_keyboard(
                float(values.get("width_delta", 0.0)),
                float(values.get("distance_delta", 0.0)),
            )
        elif action == "arrow_axes":
            self._send_arrow_impl(float(values.get("horizontal", 0.0)), "left", "right")
            self._send_arrow_impl(float(values.get("vertical", 0.0)), "up", "down")
        elif action == "scroll_axes":
            self._accum_scroll(
                float(values.get("horizontal", 0.0)),
                float(values.get("vertical", 0.0)),
                float(values.get("dt", self._last_frame_dt)),
            )
        elif action == "copy":
            _send_key(0x43, ctrl=True)
        elif action == "cut":
            _send_key(0x58, ctrl=True)
        elif action == "paste":
            _send_key(0x56, ctrl=True)
        elif action == "enter":
            _send_key(0x0D)
        else:
            handled = bool(
                self._on_controller_shortcut
                and self._on_controller_shortcut(action, **values)
            )
            if handled and action in {
                "adjust_depth_strength",
                "reset_depth",
                "toggle_stereo",
            }:
                # RuntimeCallbacks owns the depth value. The presenter owns
                # the legacy Quad prompt. Read the callback's synchronous
                # runtime snapshot so continuous stick input updates the
                # displayed value before the next rendered output arrives.
                callback_depth = self._controller_callback_depth_strength()
                previous_depth = self._tool_overlay_depth_strength
                if callback_depth is not None:
                    self._tool_overlay_depth_strength = callback_depth
                    self._tool_overlay_depth_strength_pending = callback_depth
                elif action == "adjust_depth_strength":
                    target = max(
                        0.0,
                        min(
                            10.0,
                            previous_depth + float(values.get("delta", 0.0)),
                        ),
                    )
                    self._tool_overlay_depth_strength = target
                    self._tool_overlay_depth_strength_pending = target
                if action == "toggle_stereo":
                    was_enabled = previous_depth > 0.0
                    self._depth_osd_message = (
                        "3D mode off"
                        if (callback_depth == 0.0 or (callback_depth is None and was_enabled))
                        else "3D mode on"
                    )
                else:
                    self._depth_osd_message = None
                self._depth_osd_show_t = time.perf_counter()
            if not handled:
                self._unsupported_shortcut_actions.add(action)

    def _input_deadzone(self) -> float:
        return 0.15

    def _adjust_shortcut_screen_rotation(
        self, yaw_delta: float, pitch_delta: float
    ) -> None:
        if self._filament_screen is None:
            return
        position, width, height, rotation = self._filament_screen
        next_rotation = (
            float(rotation[0]) + yaw_delta,
            max(-89.0, min(89.0, float(rotation[1]) + pitch_delta)),
            float(rotation[2]),
        )
        self._filament_screen = (position, width, height, next_rotation)
        self._screen_osd_show_t = time.perf_counter()

    def _adjust_shortcut_screen_size(
        self, width_delta: float, distance_delta: float
    ) -> None:
        if self._filament_screen is None:
            return
        position, width, height, rotation = self._filament_screen
        next_width = max(0.3, float(width) + width_delta)
        next_height = next_width * float(height) / max(float(width), 1e-6)
        head = np.asarray(
            self._head_position_w if self._head_position_w is not None else (0, 0, 0),
            dtype=np.float64,
        )
        radial = np.asarray(position, dtype=np.float64) - head
        distance = max(float(np.linalg.norm(radial)), 1e-6)
        next_distance = max(0.3, distance + distance_delta)
        next_position = head + radial / distance * next_distance
        self._filament_screen = (
            tuple(float(value) for value in next_position),
            next_width,
            next_height,
            rotation,
        )
        self._screen_osd_show_t = time.perf_counter()

    def _adjust_shortcut_keyboard(
        self, width_delta: float, distance_delta: float
    ) -> None:
        self._keyboard_width = max(0.3, min(4.0, self._keyboard_width + width_delta))
        self._keyboard_height = self._keyboard_width * _KB_TEX_H / float(_KB_TEX_W)
        self._keyboard_texture_key = None
        pose = self._keyboard_pose_mat4()
        head = np.asarray(
            self._head_position_w if self._head_position_w is not None else (0, 0, 0),
            dtype=np.float64,
        )
        radial = pose[:3, 3].astype(np.float64) - head
        distance = max(float(np.linalg.norm(radial)), 1e-6)
        self._keyboard_position_offset += radial / distance * distance_delta

    def _switch_shortcut_controller_brand(self) -> None:
        if (
            self._vulkan_controller_proxy_enabled
            or not self._controller_brands
        ):
            return
        names = sorted(self._controller_brands)
        current_name = getattr(self._controller_brand, "name", None)
        index = names.index(current_name) if current_name in names else -1
        next_brand = self._controller_brands[names[(index + 1) % len(names)]]
        previous = self._controller_brand
        bridge = self.filament_bridge
        try:
            if bridge is not None and hasattr(bridge, "load_controller"):
                bridge.load_controller(0, next_brand.left_glb.read_bytes())
                bridge.load_controller(1, next_brand.right_glb.read_bytes())
                self._apply_controller_material_profile(bridge, next_brand)
        except Exception:
            if bridge is not None and previous is not None:
                bridge.load_controller(0, previous.left_glb.read_bytes())
                bridge.load_controller(1, previous.right_glb.read_bytes())
                self._apply_controller_material_profile(bridge, previous)
            raise
        self._controller_brand = next_brand
        self._controller_calibration_offset = np.asarray(
            next_brand.offset, dtype=np.float64
        )
        self._controller_calibration_rotation_deg = float(next_brand.rotation_deg)
        ambient_multiplier = float(
            getattr(next_brand, "ambient_light_multiplier", 1.0)
        )
        if bridge is not None:
            self._apply_filament_bridge_lighting(bridge)
        anchor = self._resolve_controller_b_button_local(force=True)
        anchor_text = (
            "unresolved"
            if anchor is None
            else ", ".join(f"{value:.6f}" for value in anchor)
        )
        print(
            f"[OpenXRViewer] Switched controller: {next_brand.name}; "
            f"ambient_multiplier={ambient_multiplier:.2f}; "
            f"B-button anchor=({anchor_text})",
            flush=True,
        )

    def _controller_model_display_name(self) -> str:
        if self._vulkan_controller_proxy_enabled:
            return str(getattr(self._controller_brand, "profile_id", "None") or "None").title()
        return str(getattr(self._controller_brand, "name", "") or "Unknown")

    def _save_shortcut_controller_calibration(self) -> None:
        brand = self._controller_brand
        if brand is None:
            return
        profile_path = brand.root / "profile.json"
        try:
            profile = json.loads(profile_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            profile = {}
        overrides = profile.setdefault("overrides", {})
        overrides["model_offset"] = [
            round(float(value), 6) for value in self._controller_calibration_offset
        ]
        overrides["model_rotation_deg"] = round(
            float(self._controller_calibration_rotation_deg), 4
        )
        profile_path.write_text(
            json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._controller_calibration_mode = False
        print(f"[OpenXRViewer] Controller calibration saved: {profile_path}", flush=True)

    def _pulse_haptic(
        self,
        hand_path_str="/user/hand/right",
        *,
        amplitude=0.18,
        duration_s=0.018,
        min_interval_s=0.045,
    ) -> bool:
        """Send a short controller haptic pulse; failures are non-fatal."""
        action = getattr(self, "_act_haptic", None)
        xr = getattr(self, "xr", None)
        session = getattr(self, "session", None)
        if session is None:
            session = getattr(self, "_xr_session", None)
        if action is None or xr is None or session is None:
            return False

        now = time.perf_counter()
        last_attr = (
            "_haptic_last_l"
            if hand_path_str == "/user/hand/left"
            else "_haptic_last_r"
        )
        if now - float(getattr(self, last_attr, 0.0) or 0.0) < float(min_interval_s):
            return False

        try:
            path = getattr(
                self,
                "_path_left" if hand_path_str == "/user/hand/left" else "_path_right",
                None,
            )
            if path is None:
                instance = getattr(self, "instance", None)
                if instance is None:
                    instance = getattr(self, "_xr_instance", None)
                if instance is None:
                    return False
                path = xr.string_to_path(instance, hand_path_str)
            duration_ns = max(1, int(float(duration_s) * 1_000_000_000))
            vibration = xr.HapticVibration(
                duration=duration_ns,
                frequency=xr.FREQUENCY_UNSPECIFIED,
                amplitude=max(0.0, min(1.0, float(amplitude))),
            )
            xr.apply_haptic_feedback(
                session,
                xr.HapticActionInfo(action=action, subaction_path=path),
                vibration,
            )
            setattr(self, last_attr, now)
            return True
        except Exception:
            return False

    def _press_key(self, key, key_idx, held_key_attr, held_mods_attr):
        return self._press_key_impl(key, key_idx, held_key_attr, held_mods_attr)

    def _refresh_or_upload_keyboard_content(self) -> None:
        # Tool quads rebuild their RGBA payload from the current state each XR tick.
        return None

    def _keyboard_pose_mat4(self) -> np.ndarray:
        _position, _screen_width, screen_height, _rotation = self._filament_screen or (
            (0.0, 1.2, -2.0), 2.4, 1.35, (0.0, 0.0, 0.0)
        )
        screen_pose = self._filament_screen_pose_mat4()
        local_position = np.asarray(
            (0.0, -float(screen_height) / 2.0 - float(screen_height) * 0.15
             - float(self._keyboard_height) / 2.0, 0.0),
            dtype=np.float64,
        )
        keyboard_position = (
            screen_pose[:3, 3]
            + screen_pose[:3, :3] @ local_position
            + self._keyboard_position_offset
        )
        # The legacy keyboard is independently head-facing. Do not inherit a
        # room/profile screen rotation when that rotation is not head-facing.
        head = self._head_position_w
        if head is not None:
            direction_from_head = keyboard_position - np.asarray(head, dtype=np.float64)
            distance = float(np.linalg.norm(direction_from_head))
        else:
            direction_from_head = None
            distance = 0.0
        if direction_from_head is not None and distance > 1e-6:
            nx, ny, nz = direction_from_head / distance
            base_yaw = math.atan2(-float(nx), -float(nz))
            base_pitch = math.asin(max(-1.0, min(1.0, float(ny))))
            matrix = euler_to_mat4(
                base_yaw + math.radians(float(self._keyboard_rotation_offset[0])),
                base_pitch + math.radians(float(self._keyboard_rotation_offset[1])),
                0.0,
            ).astype(np.float64)
        else:
            matrix = screen_pose.copy().astype(np.float64)
            local_rotation = euler_to_mat4(
                0.0,
                math.radians(float(self._keyboard_rotation_offset[1])),
                math.radians(float(self._keyboard_rotation_offset[0])),
            ).astype(np.float64)
            matrix[:3, :3] = matrix[:3, :3] @ local_rotation[:3, :3]
        matrix[:3, 3] = keyboard_position
        return matrix.astype(np.float64)

    def _keyboard_plane_hit(self, origin, direction):
        if not self._keyboard_visible:
            return None, None
        if not self._keyboard_keys:
            _rgba, self._keyboard_keys = build_keyboard_rgba(
                self._kb_show_shifted, self._keyboard_width, self._keyboard_height
            )
        pose = self._keyboard_pose_mat4()
        normal = pose[:3, 2]
        denominator = float(np.dot(normal, direction))
        if abs(denominator) < 1e-6:
            return None, None
        distance = float(np.dot(normal, pose[:3, 3] - origin) / denominator)
        if distance <= 0.0:
            return None, None
        hit = np.asarray(origin, dtype=np.float64) + np.asarray(direction, dtype=np.float64) * distance
        local = np.linalg.inv(pose) @ np.append(hit, 1.0)
        x, y = float(local[0]), float(local[1])
        if abs(x) > self._keyboard_width / 2.0 or abs(y) > self._keyboard_height / 2.0:
            return None, None
        return x, y

    def _controller_interaction_ray(self, hand):
        """Return the legacy-calibrated ray used by the visible laser."""
        aim_matrix = self._aim_mat_l if hand == 0 else self._aim_mat_r
        if aim_matrix is None:
            return None, None
        grip_matrix = self._grip_mat_l if hand == 0 else self._grip_mat_r
        origin, direction = self._get_smoothed_ray(hand)
        has_smoothed_ray = origin is not None and direction is not None
        if not has_smoothed_ray:
            if grip_matrix is not None:
                raw_origin = (
                    grip_matrix[:3, 3] + grip_matrix[:3, 1] * 0.020
                ).astype(np.float64)
            else:
                raw_origin = aim_matrix[:3, 3].astype(np.float64)
            origin = raw_origin
            direction = (-aim_matrix[:3, 2]).astype(np.float64)
        else:
            origin = np.asarray(origin, dtype=np.float64)
            direction = np.asarray(direction, dtype=np.float64)
            if grip_matrix is not None:
                raw_origin = (
                    grip_matrix[:3, 3] + grip_matrix[:3, 1] * 0.020
                ).astype(np.float64)
            else:
                raw_origin = aim_matrix[:3, 3].astype(np.float64)
        direction /= max(float(np.linalg.norm(direction)), 1e-8)
        right_axis = aim_matrix[:3, 0].astype(np.float64)
        right_axis /= max(float(np.linalg.norm(right_axis)), 1e-8)
        angle = math.radians(12.0)
        direction = self._normalize_interaction_ray(
            direction * math.cos(angle)
            + np.cross(right_axis, direction) * math.sin(angle)
            + right_axis
            * float(np.dot(right_axis, direction))
            * (1.0 - math.cos(angle))
        )
        if has_smoothed_ray and not self._settings_menu.visible:
            # The smoothed ray may leave the finite screen by a small amount
            # while the unsmoothed hand pose is still close to an edge. Copy
            # the legacy edge constraint so the visible laser and interaction
            # hit remain latched to the nearest edge instead of disappearing.
            # A visible settings Quad has input priority over the screen. Its
            # cursor must follow the real controller ray instead of inheriting
            # the screen-only edge attraction behind the panel.
            if self._screen_ray_hit(aim_matrix, raw_origin, direction) is None:
                raw_direction = (-aim_matrix[:3, 2]).astype(np.float64)
                raw_direction /= max(float(np.linalg.norm(raw_direction)), 1e-8)
                raw_direction = self._normalize_interaction_ray(
                    raw_direction * math.cos(angle)
                    + np.cross(right_axis, raw_direction) * math.sin(angle)
                    + right_axis
                    * float(np.dot(right_axis, raw_direction))
                    * (1.0 - math.cos(angle))
                )
                if self._screen_ray_hit(aim_matrix, raw_origin, raw_direction) is None:
                    plane_uv = self._screen_plane_uv(raw_origin, direction)
                    if plane_uv is not None:
                        clamped_u = max(0.0, min(1.0, float(plane_uv[0])))
                        clamped_v = max(0.0, min(1.0, float(plane_uv[1])))
                        clamped_world = self._screen_uv_to_world(clamped_u, clamped_v)
                        edge_direction = clamped_world - raw_origin
                        edge_length = float(np.linalg.norm(edge_direction))
                        if edge_length > 1e-6:
                            edge_direction /= edge_length
                            edge_angle = math.acos(
                                max(-1.0, min(1.0, float(np.dot(raw_direction, edge_direction))))
                            )
                            if edge_angle < self._ray_edge_deadzone_rad:
                                direction = edge_direction
        direction /= max(float(np.linalg.norm(direction)), 1e-8)
        return origin, direction

    @staticmethod
    def _normalize_interaction_ray(direction: np.ndarray) -> np.ndarray:
        result = np.asarray(direction, dtype=np.float64)
        result /= max(float(np.linalg.norm(result)), 1e-8)
        return result

    def _screen_plane_uv(self, origin: np.ndarray, direction: np.ndarray):
        """Return unbounded UV on the screen-center plane for edge snapping."""
        if self._filament_screen is None:
            return None
        position, width, height, rotation = self._filament_screen
        pose = euler_to_mat4(
            *(math.radians(float(value)) for value in rotation)
        ).astype(np.float64)
        pose[:3, 3] = np.asarray(position, dtype=np.float64)
        normal = pose[:3, 2]
        denominator = float(np.dot(normal, direction))
        if abs(denominator) < 1e-6:
            return None
        distance = float(np.dot(normal, pose[:3, 3] - origin) / denominator)
        if distance <= 0.0:
            return None
        hit = np.asarray(origin, dtype=np.float64) + np.asarray(direction, dtype=np.float64) * distance
        local = np.linalg.inv(pose) @ np.append(hit, 1.0)
        return (
            float(local[0]) / max(float(width), 1e-6) + 0.5,
            float(local[1]) / max(float(height), 1e-6) + 0.5,
        )

    def _screen_ray_hit_for_hand(self, hand):
        aim_matrix = self._aim_mat_l if hand == 0 else self._aim_mat_r
        origin, direction = self._controller_interaction_ray(hand)
        if origin is None or direction is None:
            return None
        return self._screen_ray_hit(aim_matrix, origin, direction)

    def _screen_hit_world_for_hand(self, hand):
        """Return the current calibrated laser hit in screen world space."""
        hit = self._screen_ray_hit_for_hand(hand)
        if hit is None or self._filament_screen is None:
            return None
        u, v = (float(hit[0]), float(hit[1]))
        return self._screen_uv_to_world(u, v)

    def _effective_screen_curve_half_angle(self) -> float:
        """Return a safe active curve angle for curved-screen geometry."""
        if not self._screen_curved:
            return 0.0
        half_angle = float(self._screen_curve_half_angle)
        if not math.isfinite(half_angle) or half_angle <= 1e-6:
            half_angle = self._DEFAULT_SCREEN_CURVE_HALF_ANGLE
        return min(half_angle, math.pi / 2.0)

    def _screen_ray_hit(self, matrix, ray_origin=None, ray_direction=None):
        if matrix is None or self._filament_screen is None:
            return None
        position, width, height, rotation = self._filament_screen
        pose = euler_to_mat4(*(math.radians(float(value)) for value in rotation)).astype(np.float64)
        pose[:3, 3] = np.asarray(position, dtype=np.float64)
        origin = (
            matrix[:3, 3].astype(np.float64)
            if ray_origin is None
            else np.asarray(ray_origin, dtype=np.float64)
        )
        direction = (
            (-matrix[:3, 2]).astype(np.float64)
            if ray_direction is None
            else np.asarray(ray_direction, dtype=np.float64)
        )
        direction /= max(float(np.linalg.norm(direction)), 1e-10)
        if self._screen_curved:
            # Match the legacy _laser_screen_hit_uv cylinder geometry and
            # return the same bottom-to-top UV convention used by the GLB
            # screen mesh. Analytic roots avoid a frame-dependent scan.
            half_width = float(width) / 2.0
            half_height = float(height) / 2.0
            half_angle = self._effective_screen_curve_half_angle()
            radius = half_width / max(half_angle, 1e-8)
            local_origin = pose[:3, :3].T @ (origin - pose[:3, 3])
            local_direction = pose[:3, :3].T @ direction
            ox, oy, oz = (float(value) for value in local_origin)
            dx, _dy, dz = (float(value) for value in local_direction)
            qa = dx * dx + dz * dz
            qb = 2.0 * (ox * dx + (oz - radius) * dz)
            qc = ox * ox + (oz - radius) * (oz - radius) - radius * radius
            if abs(qa) < 1e-10:
                return None
            discriminant = qb * qb - 4.0 * qa * qc
            if discriminant < 0.0:
                return None
            root = math.sqrt(max(0.0, discriminant))
            roots = sorted(
                ((-qb - root) / (2.0 * qa), (-qb + root) / (2.0 * qa))
            )
            for distance in roots:
                if distance <= 0.01:
                    continue
                local_hit = local_origin + local_direction * distance
                if abs(float(local_hit[1])) > half_height + 1e-6:
                    continue
                angle = math.atan2(
                    float(local_hit[0]), radius - float(local_hit[2])
                )
                if angle < -half_angle - 1e-6 or angle > half_angle + 1e-6:
                    continue
                return (
                    float((angle + half_angle) / (2.0 * half_angle)),
                    float((float(local_hit[1]) + half_height) / (2.0 * half_height)),
                )
            return None
        normal = pose[:3, 2]
        denominator = float(np.dot(normal, direction))
        if abs(denominator) < 1e-6:
            return None
        distance = float(np.dot(normal, pose[:3, 3] - origin) / denominator)
        if distance <= 0.0:
            return None
        hit = origin + direction * distance
        local = np.linalg.inv(pose) @ np.append(hit, 1.0)
        if abs(float(local[0])) > width / 2.0 or abs(float(local[1])) > height / 2.0:
            return None
        return (
            max(0.0, min(1.0, float(local[0]) / width + 0.5)),
            max(0.0, min(1.0, 0.5 + float(local[1]) / height)),
        )

    def _screen_uv_to_world(self, u: float, v: float) -> np.ndarray | None:
        """Convert screen UV to the current flat or curved screen surface."""
        if self._filament_screen is None:
            return None
        position, width, height, rotation = self._filament_screen
        pose = euler_to_mat4(
            *(math.radians(float(value)) for value in rotation)
        ).astype(np.float64)
        pose[:3, 3] = np.asarray(position, dtype=np.float64)
        if self._screen_curved:
            half_angle = self._effective_screen_curve_half_angle()
            radius = float(width) / 2.0 / max(half_angle, 1e-8)
            angle = -half_angle + 2.0 * half_angle * float(u)
            local = np.asarray(
                (
                    radius * math.sin(angle),
                    (float(v) - 0.5) * float(height),
                    radius * (1.0 - math.cos(angle)),
                ),
                dtype=np.float64,
            )
        else:
            local = np.asarray(
                (
                    (float(u) - 0.5) * float(width),
                    (float(v) - 0.5) * float(height),
                    0.0,
                ),
                dtype=np.float64,
            )
        return pose[:3, 3] + pose[:3, :3] @ local

    def _set_filament_screen_pose(self, position, rotation=None) -> None:
        if self._filament_screen is None:
            return
        _old_position, width, height, old_rotation = self._filament_screen
        pose_rotation = tuple(rotation if rotation is not None else old_rotation)
        self._filament_screen = (tuple(float(value) for value in position), width, height, pose_rotation)

    def _set_keyboard_world_position(self, position) -> None:
        _screen_position, _width, screen_height, _rotation = self._filament_screen or (
            (0.0, 1.2, -2.0),
            2.4,
            1.35,
            (0.0, 0.0, 0.0),
        )
        # Keep the legacy absolute-position setter compatible with the new
        # screen-relative keyboard anchor.  The caller-provided world
        # position is converted to an offset from the current anchor, so the
        # next pose evaluation reproduces it exactly.
        local_position = np.asarray(
            (
                0.0,
                -float(screen_height) / 2.0
                - float(screen_height) * 0.15
                - float(self._keyboard_height) / 2.0,
                0.0,
            ),
            dtype=np.float64,
        )
        base_position = self._filament_screen_pose_mat4()[:3, 3] + (
            self._filament_screen_pose_mat4()[:3, :3] @ local_position
        )
        self._keyboard_position_offset = (
            np.asarray(position, dtype=np.float64) - base_position
        )

    @staticmethod
    def _rotation_delta_euler_degrees(rotation: np.ndarray) -> tuple[float, float, float]:
        """Convert a relative rotation matrix to the viewer yaw/pitch/roll order."""
        pitch = math.asin(max(-1.0, min(1.0, -float(rotation[1, 2]))))
        cos_pitch = math.cos(pitch)
        if abs(cos_pitch) > 1e-6:
            yaw = math.atan2(float(rotation[0, 2]), float(rotation[2, 2]))
            roll = math.atan2(float(rotation[1, 0]), float(rotation[1, 1]))
        else:
            yaw = math.atan2(-float(rotation[2, 0]), float(rotation[0, 0]))
            roll = 0.0
        return tuple(math.degrees(value) for value in (yaw, pitch, roll))

    def _apply_grip_screen_rotation(self, hand_index: int) -> None:
        # The legacy right-grip wrist-rotation feature was disabled. Screen
        # rotation remains available only through the legacy left-grip gesture
        # and the documented left-grip/right-stick shortcut.
        if int(hand_index) != 0:
            return
        if self._filament_screen is None:
            return
        suffix = "l" if hand_index == 0 else "r"
        grip_matrix = self._grip_mat_l if hand_index == 0 else self._grip_mat_r
        grip_anchor = getattr(self, f"_grip_rotation_anchor_{suffix}")
        screen_anchor = getattr(self, f"_screen_rotation_anchor_{suffix}")
        if grip_matrix is None or grip_anchor is None or screen_anchor is None:
            return
        relative = (
            np.asarray(grip_matrix[:3, :3], dtype=np.float64)
            @ np.asarray(grip_anchor, dtype=np.float64).T
        )
        _yaw, _pitch, roll = self._rotation_delta_euler_degrees(relative)
        if (
            abs(float(roll)) < 45.0
            or bool(getattr(self, "_grip_screen_rotation_snapped_l", False))
        ):
            return
        direction = 1.0 if float(roll) > 0.0 else -1.0
        rotation = (
            float(screen_anchor[0]),
            max(-89.0, min(89.0, float(screen_anchor[1]))),
            float(screen_anchor[2]) + direction * 90.0,
        )
        self._set_filament_screen_pose(self._filament_screen[0], rotation)
        self._grip_screen_rotation_snapped_l = True

    def _reset_screen_control_hold(self, control: str) -> None:
        setattr(self, f"_screen_{control}_hold_seconds", 0.0)
        setattr(self, f"_screen_{control}_hold_direction", 0)

    def _screen_hold_speed(self, axis_value: float, *, dt: float, control: str) -> float:
        """Return speed from hold duration, restarting after release/reversal."""
        value = float(axis_value)
        if abs(value) <= self._input_deadzone():
            self._reset_screen_control_hold(control)
            return 0.0
        direction = 1 if value > 0.0 else -1
        direction_attr = f"_screen_{control}_hold_direction"
        hold_attr = f"_screen_{control}_hold_seconds"
        if getattr(self, direction_attr) != direction:
            setattr(self, hold_attr, 0.0)
        hold_seconds = min(
            float(self._screen_control_max_hold_seconds),
            float(getattr(self, hold_attr)) + max(0.0, float(dt)),
        )
        setattr(self, direction_attr, direction)
        setattr(self, hold_attr, hold_seconds)
        return min(
            float(self._screen_control_max_speed),
            float(self._screen_control_min_speed)
            + float(self._screen_control_acceleration) * hold_seconds,
        )

    def _apply_right_grip_screen_distance(
        self, joystick_y: float, *, dt: float, laser_hit: Any
    ) -> None:
        """Move the screen radially with five-second hold-time acceleration."""
        if (
            self._filament_screen is None
            or self._head_position_w is None
            or laser_hit is None
            or abs(float(joystick_y)) <= self._input_deadzone()
        ):
            self._reset_screen_control_hold("distance")
            return
        # The Vulkan controller input contract exposes the thumbstick Y axis
        # with the sign flipped from the legacy raw OpenXR value. Restore the
        # legacy sign for this operation: pushing the stick forward must move
        # the screen away from the head.
        legacy_joystick_y = -float(joystick_y)
        speed = self._screen_hold_speed(
            legacy_joystick_y, dt=dt, control="distance"
        )
        if speed <= 0.0:
            return
        position, width, height, rotation = self._filament_screen
        head = np.asarray(self._head_position_w, dtype=np.float64)
        radial = np.asarray(position, dtype=np.float64) - head
        radius = float(np.linalg.norm(radial))
        if radius <= 1e-6:
            return
        radial /= radius
        # Match the legacy sign: positive raw OpenXR Y increases the
        # head-to-screen radius.
        next_radius = max(
            0.3,
            radius + speed * (1.0 if legacy_joystick_y > 0.0 else -1.0) * dt,
        )
        next_position = head + radial * next_radius
        dx, dy, dz = (next_position - head) / next_radius
        next_rotation = (
            math.degrees(math.atan2(-float(dx), -float(dz))),
            math.degrees(math.asin(max(-1.0, min(1.0, float(dy))))),
            float(rotation[2]),
        )
        self._set_filament_screen_pose(next_position, next_rotation)
        self._screen_osd_show_t = time.perf_counter()

    def _apply_right_grip_screen_resize(
        self, joystick_x: float, *, dt: float, laser_hit: Any
    ) -> None:
        """Resize the screen with five-second hold-time acceleration."""
        if (
            self._filament_screen is None
            or laser_hit is None
            or abs(float(joystick_x)) <= self._input_deadzone()
        ):
            self._reset_screen_control_hold("size")
            return
        speed = self._screen_hold_speed(
            float(joystick_x), dt=dt, control="size"
        )
        if speed <= 0.0:
            return
        position, width, height, rotation = self._filament_screen
        next_width = max(0.3, float(width) + math.copysign(speed * dt, float(joystick_x)))
        next_height = next_width * float(height) / max(float(width), 1e-6)
        self._filament_screen = (
            tuple(float(value) for value in position),
            next_width,
            next_height,
            rotation,
        )
        self._screen_osd_show_t = time.perf_counter()

    def _screen_projection_world_points(self) -> np.ndarray | None:
        if self._filament_screen is None:
            return None
        position, width, height, rotation = self._filament_screen
        if width <= 0.0 or height <= 0.0:
            return None
        screen_pose = euler_to_mat4(
            *(math.radians(float(value)) for value in rotation[:3])
        ).astype(np.float64)
        center = np.asarray(position, dtype=np.float64)
        right = screen_pose[:3, 0].astype(np.float64)
        up = screen_pose[:3, 1].astype(np.float64)
        forward = np.cross(right, up)
        half_width = float(width) * 0.5
        half_height = float(height) * 0.5
        if self._screen_curved:
            segments = 48
            half_angle = self._effective_screen_curve_half_angle()
            radius = half_width / half_angle
            points = []
            for segment in range(segments + 1):
                angle = -half_angle + 2.0 * half_angle * segment / segments
                local_x = radius * math.sin(angle)
                local_z = radius * (1.0 - math.cos(angle))
                column_center = center + right * local_x + forward * local_z
                points.extend((
                    column_center - up * half_height,
                    column_center + up * half_height,
                ))
            return np.asarray(points, dtype=np.float64)
        return np.asarray(
            (
                center - right * half_width - up * half_height,
                center + right * half_width - up * half_height,
                center + right * half_width + up * half_height,
                center - right * half_width + up * half_height,
            ),
            dtype=np.float64,
        )

    def _screen_projection_points(
        self,
        view: Any,
        swapchain_size: tuple[int, int],
    ) -> np.ndarray | None:
        try:
            sc_w, sc_h = int(swapchain_size[0]), int(swapchain_size[1])
            world_points = self._screen_projection_world_points()
            if sc_w <= 0 or sc_h <= 0 or world_points is None:
                return None
            eye_pose = _xr_view_pose_to_model_mat4(view.pose).astype(np.float64)
            camera_points = (
                np.linalg.inv(eye_pose)
                @ np.concatenate(
                    (world_points, np.ones((len(world_points), 1), dtype=np.float64)),
                    axis=1,
                ).T
            ).T[:, :3]
            depth = -camera_points[:, 2]
            valid = np.isfinite(depth) & (depth > 1e-6)
            if not np.all(valid):
                return None
            fov = view.fov
            tan_left = math.tan(float(fov.angle_left))
            tan_right = math.tan(float(fov.angle_right))
            tan_down = math.tan(float(fov.angle_down))
            tan_up = math.tan(float(fov.angle_up))
            if (
                not all(math.isfinite(value) for value in (
                    tan_left, tan_right, tan_down, tan_up,
                ))
                or tan_right <= tan_left
                or tan_up <= tan_down
            ):
                return None
            ndc_x = 2.0 * (
                camera_points[valid, 0] / depth[valid] - tan_left
            ) / (tan_right - tan_left) - 1.0
            ndc_y = 2.0 * (
                camera_points[valid, 1] / depth[valid] - tan_down
            ) / (tan_up - tan_down) - 1.0
            points = np.column_stack((
                (ndc_x * 0.5 + 0.5) * sc_w,
                (1.0 - (ndc_y * 0.5 + 0.5)) * sc_h,
            ))
            if len(points) != len(world_points) or not np.all(np.isfinite(points)):
                return None
            return points
        except (AttributeError, IndexError, TypeError, ValueError, np.linalg.LinAlgError):
            return None

    def _screen_projection_quad(
        self,
        view: Any,
        swapchain_size: tuple[int, int],
    ) -> np.ndarray | None:
        if self._screen_curved:
            return None
        points = self._screen_projection_points(view, swapchain_size)
        return points if points is not None and points.shape == (4, 2) else None

    def _screen_projection_bounds(
        self,
        view: Any,
        swapchain_size: tuple[int, int],
    ) -> tuple[float, float, float, float] | None:
        points = self._screen_projection_points(view, swapchain_size)
        if points is None:
            return None
        sc_w, sc_h = int(swapchain_size[0]), int(swapchain_size[1])
        return (
            max(float(np.min(points[:, 0])), 0.0),
            max(float(np.min(points[:, 1])), 0.0),
            min(float(np.max(points[:, 0])), float(sc_w)),
            min(float(np.max(points[:, 1])), float(sc_h)),
        )

    def _screen_footprint_pixels(
        self,
        view: Any,
        swapchain_size: tuple[int, int],
    ) -> tuple[float, float] | None:
        bounds = self._screen_projection_bounds(view, swapchain_size)
        if bounds is None:
            return None
        return max(0.0, bounds[2] - bounds[0]), max(0.0, bounds[3] - bounds[1])

    def _report_screen_resolution(
        self,
        views: list[Any],
        output_frame: VulkanStereoOutputFrame | None,
    ) -> None:
        """Log screen pixel dimensions once per actual resolution configuration."""

        if output_frame is None or self._filament_screen is None:
            return
        sources = (
            (
                int(getattr(output_frame.left_eye, "width", 0)),
                int(getattr(output_frame.left_eye, "height", 0)),
            ),
            (
                int(getattr(output_frame.right_eye, "width", 0)),
                int(getattr(output_frame.right_eye, "height", 0)),
            ),
        )
        targets = self._projection_eye_extents()
        if len(views) < 2 or len(targets) < 2:
            return
        footprints = tuple(
            self._screen_footprint_pixels(views[index], targets[index])
            for index in range(2)
        )
        metadata = dict(output_frame.metadata or {})
        render_size = metadata.get("render_size", metadata.get("source_render_size"))
        if isinstance(render_size, (list, tuple)) and len(render_size) >= 2:
            render_size_label = f"{int(render_size[0])}x{int(render_size[1])}"
        else:
            render_size_label = str(render_size or "unknown")
        screen = self._filament_screen

        # The projected footprint is useful in the message, but it is view-dependent
        # and must not decide whether a resolution diagnostic is emitted.
        resolution_status = (
            sources,
            targets,
            render_size_label,
        )
        if resolution_status == self._last_screen_resolution_status:
            return
        self._last_screen_resolution_status = resolution_status

        def format_size(size: tuple[int, int]) -> str:
            return f"{size[0]}x{size[1]}"

        def format_footprint(footprint: tuple[float, float] | None) -> str:
            if footprint is None:
                return "unknown"
            return f"{round(footprint[0])}x{round(footprint[1])}"

        def format_density(
            source: tuple[int, int], footprint: tuple[float, float] | None
        ) -> str:
            if footprint is None or footprint[0] <= 0.0 or footprint[1] <= 0.0:
                return "unknown"
            return f"{source[0] / footprint[0]:.2f}x{source[1] / footprint[1]:.2f}"

        print(
            "[OpenXRViewer] screen resolution "
            f"source_left={format_size(sources[0])} "
            f"source_right={format_size(sources[1])} "
            f"render_size={render_size_label} "
            f"screen_footprint_left={format_footprint(footprints[0])} "
            f"screen_footprint_right={format_footprint(footprints[1])} "
            f"projection_target_left={format_size(targets[0])} "
            f"projection_target_right={format_size(targets[1])} "
            f"source_per_screen_pixel_left={format_density(sources[0], footprints[0])} "
            f"source_per_screen_pixel_right={format_density(sources[1], footprints[1])} "
            f"screen_m={float(screen[1]):.3f}x{float(screen[2]):.3f} "
            f"distance_m={float(np.linalg.norm(np.asarray(screen[0], dtype=np.float64))):.3f} "
            f"curved={bool(self._screen_curved)}",
            flush=True,
        )

    def _target_monitor_rect(self) -> tuple[int, int, int, int]:
        """Return (left, top, width, height) of the captured monitor.

        Port of the legacy ``_get_target_monitor_rect``. Monitor indices
        follow the MSS 1-based convention; falls back to the primary monitor.
        Cached per monitor index so the Win32 enumeration runs once.
        """
        cached = getattr(self, "_target_mon_rect_cache", None)
        if cached is not None and cached[0] == self.config.monitor_index:
            return cached[1]
        if sys.platform != "win32":
            rect = (0, 0, 1920, 1080)
        else:
            from ctypes import wintypes

            class _RECT(ctypes.Structure):
                _fields_ = [
                    ("left", wintypes.LONG),
                    ("top", wintypes.LONG),
                    ("right", wintypes.LONG),
                    ("bottom", wintypes.LONG),
                ]

            class _MONITORINFO(ctypes.Structure):
                _fields_ = [
                    ("cbSize", wintypes.DWORD),
                    ("rcMonitor", _RECT),
                    ("rcWork", _RECT),
                    ("dwFlags", wintypes.DWORD),
                ]

            monitors: list[tuple[int, int, int, int]] = []

            def _enumerate_monitor(
                _handle, _hdc, _p_rect, _lparam
            ) -> bool:
                info = _MONITORINFO()
                info.cbSize = ctypes.sizeof(_MONITORINFO)
                ctypes.windll.user32.GetMonitorInfoW(
                    _handle, ctypes.byref(info)
                )
                r = info.rcMonitor
                monitors.append(
                    (r.left, r.top, r.right - r.left, r.bottom - r.top)
                )
                return True

            _CALLBACK = ctypes.WINFUNCTYPE(
                wintypes.BOOL,
                wintypes.HMONITOR,
                wintypes.HDC,
                ctypes.POINTER(_RECT),
                wintypes.LPARAM,
            )
            ctypes.windll.user32.EnumDisplayMonitors(
                None, None, _CALLBACK(_enumerate_monitor), 0
            )
            index = max(0, int(self.config.monitor_index) - 1)
            rect = (
                monitors[index] if index < len(monitors) else (0, 0, 1920, 1080)
            )
        self._target_mon_rect_cache = (int(self.config.monitor_index), rect)
        return rect

    def _cursor_pixel_for_screen_uv(self, u: float, v: float) -> tuple[int, int]:
        """Map the bottom-to-top screen UV to virtual-desktop cursor pixels.

        Matches the legacy ``_screen_uv_to_source_top_uv``: v=0 is the bottom
        of the screen quad (image bottom), so the Windows cursor y must use
        ``(1 - v)`` because Windows y grows downward.
        """
        left, top, width, height = self._target_monitor_rect()
        return (
            int(left + float(u) * width),
            int(top + (1.0 - float(v)) * height),
        )

    def _hand_keyboard_hit(self, hand_index: int) -> bool:
        """True when the hand's laser currently intersects the keyboard."""
        if not self._keyboard_visible:
            return False
        origin, direction = self._controller_interaction_ray(hand_index)
        if origin is None or direction is None:
            return False
        return self._keyboard_plane_hit(origin, direction) != (None, None)

    def _screen_edge_px_for_hand(self, hand_index: int):
        """Clamp the interaction ray to the screen edge in desktop pixels.

        An active touch drag that briefly grazes off-screen keeps updating at
        the clamped edge so Windows sees uninterrupted motion data (the legacy
        fast-drag "cursor stuck at edge" fix).
        """
        origin, direction = self._controller_interaction_ray(hand_index)
        if origin is None or direction is None:
            return None
        uv = self._screen_plane_uv(origin, direction)
        if uv is None:
            return None
        u = max(0.0, min(1.0, float(uv[0])))
        v = max(0.0, min(1.0, float(uv[1])))
        return self._cursor_pixel_for_screen_uv(u, v)

    def _clamp_touch_px(self, x: float, y: float) -> tuple[int, int]:
        """Clamp touch coordinates to the captured monitor's pixel bounds."""
        left, top, width, height = self._target_monitor_rect()
        if width > 0 and height > 0:
            x = max(float(left), min(float(left + width - 1), x))
            y = max(float(top), min(float(top + height - 1), y))
        return int(round(x)), int(round(y))

    def _update_touch_contacts(self, inputs, hits) -> bool:
        """Drive Windows multi-touch contacts for both hands (preferred).

        Returns True while touch injection is active, in which case the mouse
        click/drag fallback is disabled. Mirrors the legacy ``_handle_triggers``
        lifecycle with the documented bug fixes applied:
          * the raw laser-mapped pixel position is used directly (the second
            EMA stage caused drag lag / "fast drag = small move" / cursor hang),
          * an active contact survives off-screen by clamping to the screen
            edge, and
          * an UP fires at the current laser position (no post-release snap).
        ``SetCursorPos`` suppression during a two-contact gesture is handled
        by the caller via ``_touch_state``.
        """
        if not _TOUCH_AVAILABLE or _touch_injector is None:
            return False
        hands = []
        for name, hand_index, hit in (
            ("left", 0, hits[0]),
            ("right", 1, hits[1]),
        ):
            trigger = float(inputs[hand_index].get("trigger", 0.0) or 0.0)
            keyboard_hit = self._hand_keyboard_hit(hand_index)
            valid = hit is not None and not keyboard_hit
            if valid:
                self._touch_px[name] = self._cursor_pixel_for_screen_uv(
                    hit[0], hit[1]
                )
            else:
                edge = self._screen_edge_px_for_hand(hand_index)
                if edge is not None:
                    self._touch_px[name] = edge
            self._touch_valid[name] = valid
            suffix = "l" if hand_index == 0 else "r"
            grip = bool(
                float(inputs[hand_index].get("grip", 0.0) or 0.0) > 0.5
                and getattr(self, f"_grip_target_{suffix}") in ("screen", "keyboard")
            )
            trig_prev = self._touch_trig_prev[name]
            if self._touch_state[name] == "down":
                want_down = trigger > 0.3 and not grip
            else:
                want_down = (
                    trigger >= 0.7 and trig_prev < 0.7 and valid and not grip
                )
            hands.append(
                {
                    "name": name,
                    "contact_id": (
                        _TOUCH_CONTACT_ID_LEFT
                        if hand_index == 0
                        else _TOUCH_CONTACT_ID_RIGHT
                    ),
                    "want_down": want_down,
                    "px": self._touch_px[name],
                    "trig": trigger,
                }
            )
        # Two simultaneous DOWNs spread around their midpoint so pinch/zoom
        # responds with less physical travel (legacy gain).
        if len(hands) == 2 and all(hand["want_down"] for hand in hands):
            gain = float(_TOUCH_PINCH_SPREAD_GAIN)
            if gain > 1.0:
                p0 = hands[0]["px"]
                p1 = hands[1]["px"]
                cx = (float(p0[0]) + float(p1[0])) * 0.5
                cy = (float(p0[1]) + float(p1[1])) * 0.5
                hands[0]["px"] = self._clamp_touch_px(
                    cx + (float(p0[0]) - cx) * gain,
                    cy + (float(p0[1]) - cy) * gain,
                )
                hands[1]["px"] = self._clamp_touch_px(
                    cx + (float(p1[0]) - cx) * gain,
                    cy + (float(p1[1]) - cy) * gain,
                )
        for hand in hands:
            self._touch_trig_prev[hand["name"]] = hand["trig"]
            # Only publish a real transition (DOWN / held UPDATE / UP from a
            # held contact), matching the legacy _handle_triggers call pattern.
            if hand["want_down"] or self._touch_state[hand["name"]] == "down":
                px = hand["px"]
                _touch_injector.set(
                    hand["contact_id"], px[0], px[1], hand["want_down"]
                )
            self._touch_state[hand["name"]] = "down" if hand["want_down"] else "idle"
        _touch_injector.flush()
        return bool(_touch_injector.available)

    def _cancel_touch_contacts(self) -> None:
        """Lift every active touch contact (menu open / grip / shutdown)."""
        if not _TOUCH_AVAILABLE or _touch_injector is None:
            return
        for name, contact_id in (
            ("left", _TOUCH_CONTACT_ID_LEFT),
            ("right", _TOUCH_CONTACT_ID_RIGHT),
        ):
            if self._touch_state[name] == "down":
                px = self._touch_px[name]
                _touch_injector.set(contact_id, px[0], px[1], want_down=False)
                self._touch_state[name] = "idle"
        _touch_injector.flush()

    def _handle_vulkan_pointer_input(self) -> None:
        """Reuse legacy trigger hold/drag semantics for the Vulkan screen."""
        self._right_grip_screen_pointer_applied = False
        now = time.perf_counter()
        inputs = (self._controller_input(0), self._controller_input(1))
        hits = (self._screen_ray_hit_for_hand(0), self._screen_ray_hit_for_hand(1))
        left_grip = bool(inputs[0].get("grip", 0.0) > 0.5)
        right_grip = bool(inputs[1].get("grip", 0.0) > 0.5)
        if not (
            right_grip
            and not left_grip
            and getattr(self, "_grip_target_r", None) == "screen"
            and hits[1] is not None
        ):
            self._reset_screen_control_hold("distance")
            self._reset_screen_control_hold("size")
        stick_active = (
            abs(float(inputs[0].get("joystick_x", 0.0))) > self._input_deadzone()
            or abs(float(inputs[0].get("joystick_y", 0.0))) > self._input_deadzone(),
            abs(float(inputs[1].get("joystick_x", 0.0))) > self._input_deadzone()
            or abs(float(inputs[1].get("joystick_y", 0.0))) > self._input_deadzone(),
        )
        grip_matrices = (self._grip_mat_l, self._grip_mat_r)
        grip_values = (left_grip, right_grip)
        for index, suffix in enumerate(("l", "r")):
            target_attr = f"_grip_target_{suffix}"
            anchor_attr = "_left_grab_anchor" if index == 0 else "_right_grab_anchor"
            rotation_attr = f"_grip_rotation_anchor_{suffix}"
            screen_rotation_attr = f"_screen_rotation_anchor_{suffix}"
            if not grip_values[index]:
                setattr(self, target_attr, None)
                setattr(self, anchor_attr, None)
                setattr(self, rotation_attr, None)
                setattr(self, screen_rotation_attr, None)
                setattr(self, f"_kb_grab_local_{suffix}", None)
                if index == 0:
                    self._grip_screen_rotation_snapped_l = False
                setattr(self, f"_screen_hit_grab_anchor_{suffix}", None)
                continue
            if getattr(self, target_attr) is None:
                # Match the legacy rising-edge latch: a highlighted key is
                # sufficient to select the keyboard, even if the ray is in a
                # key gap on the next frame.
                keyboard_hit = self._keyboard_visible and getattr(
                    self, f"_kb_hover_{suffix}"
                ) is not None
                if keyboard_hit:
                    setattr(self, target_attr, "keyboard")
                elif hits[index] is not None:
                    setattr(self, target_attr, "screen")

            if (
                index == 0
                and getattr(self, target_attr) == "screen"
                and getattr(self, rotation_attr) is None
                and grip_matrices[index] is not None
                and self._filament_screen is not None
            ):
                # The old renderer records the left grip pose on the rising
                # edge. Without this anchor the later 45-degree snap test can
                # never observe wrist rotation.
                setattr(
                    self,
                    rotation_attr,
                    np.asarray(grip_matrices[index][:3, :3], dtype=np.float64).copy(),
                )
                screen_rotation = self._filament_screen[3]
                normalized_roll = ((float(screen_rotation[2]) + 180.0) % 360.0) - 180.0
                if abs(normalized_roll) < 45.0:
                    base_roll = 0.0
                elif 45.0 <= normalized_roll < 135.0:
                    base_roll = 90.0
                elif -135.0 < normalized_roll <= -45.0:
                    base_roll = -90.0
                else:
                    base_roll = 0.0
                setattr(
                    self,
                    screen_rotation_attr,
                    (
                        float(screen_rotation[0]),
                        float(screen_rotation[1]),
                        base_roll,
                    ),
                )
                self._grip_screen_rotation_snapped_l = False

        both_grips = left_grip and right_grip
        if both_grips and not any(stick_active) and all(
            matrix is not None for matrix in grip_matrices
        ):
            common_target = (
                self._grip_target_l
                if self._grip_target_l == self._grip_target_r
                else None
            )
            center = (
                grip_matrices[0][:3, 3].astype(np.float64)
                + grip_matrices[1][:3, 3].astype(np.float64)
            ) * 0.5
            if common_target == "screen" and self._filament_screen is not None:
                if self._both_grip_anchor is None:
                    self._both_grip_anchor = (
                        "screen",
                        np.asarray(self._filament_screen[0], dtype=np.float64) - center,
                    )
                self._set_filament_screen_pose(center + self._both_grip_anchor[1])
            elif common_target == "keyboard":
                if self._both_grip_anchor is None:
                    self._both_grip_anchor = (
                        "keyboard", self._keyboard_pose_mat4()[:3, 3] - center
                    )
                keyboard_position = center + self._both_grip_anchor[1]
                self._set_keyboard_world_position(keyboard_position)
        else:
            self._both_grip_anchor = None
            for index, suffix in enumerate(("l", "r")):
                if not grip_values[index] or grip_matrices[index] is None:
                    continue
                anchor_attr = "_left_grab_anchor" if index == 0 else "_right_grab_anchor"
                rotation_attr = f"_grip_rotation_anchor_{suffix}"
                screen_rotation_attr = f"_screen_rotation_anchor_{suffix}"
                if stick_active[index]:
                    setattr(self, anchor_attr, None)
                    setattr(self, rotation_attr, None)
                    setattr(self, f"_screen_hit_grab_anchor_{suffix}", None)
                    setattr(self, f"_kb_grab_local_{suffix}", None)
                    continue
                target = getattr(self, f"_grip_target_{suffix}")
                if target == "keyboard":
                    # Port the legacy keyboard grip-to-move behavior: keep
                    # the laser's original keyboard-local point attached
                    # while moving the panel on a sphere around the head.
                    ray_origin, ray_direction = self._controller_interaction_ray(index)
                    if ray_origin is None or ray_direction is None:
                        continue
                    keyboard_pose = self._keyboard_pose_mat4()
                    normal = keyboard_pose[:3, 2].astype(np.float64)
                    denominator = float(np.dot(normal, ray_direction))
                    if abs(denominator) < 1e-6:
                        continue
                    distance = float(
                        np.dot(normal, keyboard_pose[:3, 3] - ray_origin)
                        / denominator
                    )
                    if distance < 0.05:
                        continue
                    hit_world = (
                        np.asarray(ray_origin, dtype=np.float64)
                        + np.asarray(ray_direction, dtype=np.float64) * distance
                    )
                    local_hit = np.linalg.inv(keyboard_pose) @ np.append(
                        hit_world, 1.0
                    )
                    keyboard_local_attr = f"_kb_grab_local_{suffix}"
                    keyboard_local = getattr(self, keyboard_local_attr)
                    if keyboard_local is None:
                        setattr(
                            self,
                            keyboard_local_attr,
                            np.asarray(local_hit[:2], dtype=np.float64),
                        )
                        continue

                    desired_center = (
                        hit_world
                        - keyboard_pose[:3, 0] * float(keyboard_local[0])
                        - keyboard_pose[:3, 1] * float(keyboard_local[1])
                    )
                    if self._head_position_w is not None:
                        head = np.asarray(self._head_position_w, dtype=np.float64)
                        current_radius_vector = keyboard_pose[:3, 3] - head
                        current_radius = float(np.linalg.norm(current_radius_vector))
                        desired_radius_vector = desired_center - head
                        desired_radius = float(np.linalg.norm(desired_radius_vector))
                        if current_radius > 1e-6 and desired_radius > 1e-6:
                            desired_center = (
                                head
                                + desired_radius_vector / desired_radius * current_radius
                            )
                    self._set_keyboard_world_position(desired_center)
                elif target == "screen" and self._filament_screen is not None:
                    # Match the legacy renderer: the point selected by the
                    # visible laser stays attached to the same screen-local
                    # coordinate while the hand moves or rotates.
                    hit_world = self._screen_hit_world_for_hand(index)
                    if hit_world is None:
                        continue
                    screen_position, screen_width, screen_height, screen_rotation = (
                        self._filament_screen
                    )
                    screen_pose = self._filament_screen_pose_mat4()
                    screen_center = screen_pose[:3, 3].astype(np.float64)
                    screen_basis = screen_pose[:3, :3].astype(np.float64)
                    hit_anchor_attr = f"_screen_hit_grab_anchor_{suffix}"
                    hit_anchor = getattr(self, hit_anchor_attr)
                    if hit_anchor is None:
                        hit_anchor = screen_basis.T @ (hit_world - screen_center)
                        setattr(self, hit_anchor_attr, hit_anchor)
                    target_center = hit_world - screen_basis @ hit_anchor
                    target_rotation = screen_rotation
                    if index == 1 and self._head_position_w is not None:
                        # Right-hand legacy drag orbits around the head while
                        # preserving the current screen distance and keeps the
                        # screen normal aimed back at the head.
                        head = np.asarray(self._head_position_w, dtype=np.float64)
                        original_radius = float(np.linalg.norm(screen_center - head))
                        radial = target_center - head
                        radial_length = float(np.linalg.norm(radial))
                        if original_radius > 1e-6 and radial_length > 1e-6:
                            target_center = head + radial / radial_length * original_radius
                            dx, dy, dz = (target_center - head) / original_radius
                            target_rotation = (
                                math.degrees(math.atan2(-float(dx), -float(dz))),
                                math.degrees(math.asin(max(-1.0, min(1.0, float(dy))))),
                                float(screen_rotation[2]),
                            )
                    self._set_filament_screen_pose(target_center, target_rotation)
                    if index == 0:
                        self._apply_grip_screen_rotation(index)
        if (
            right_grip
            and not left_grip
            and getattr(self, "_grip_target_r", None) == "screen"
            and self._filament_screen is not None
        ):
            self._right_grip_screen_pointer_applied = True
            input_dt = max(0.001, min(0.1, float(self._last_frame_dt)))
            self._apply_right_grip_screen_resize(
                float(inputs[1].get("joystick_x", 0.0) or 0.0),
                dt=input_dt,
                laser_hit=hits[1],
            )
            self._apply_right_grip_screen_distance(
                float(inputs[1].get("joystick_y", 0.0) or 0.0),
                dt=input_dt,
                laser_hit=hits[1],
            )
        touch_active = self._update_touch_contacts(inputs, hits)
        # Physical mouse takes priority over the controller beam: while the real
        # mouse was moved/clicked recently, the beam neither moves the OS cursor
        # nor sends clicks (the injected input is released and ignored).
        physical_mouse_active = bool(_physical_mouse_active())
        for name, hand, hit, down_flag, up_flag in (
            ("left", inputs[0], hits[0], _MOUSEEVENTF_RIGHTDOWN, _MOUSEEVENTF_RIGHTUP),
            ("right", inputs[1], hits[1], _MOUSEEVENTF_LEFTDOWN, _MOUSEEVENTF_LEFTUP),
        ):
            trigger = float(hand.get("trigger", 0.0) or 0.0)
            state = self._pointer_state[name]
            hand_index = 0 if name == "left" else 1
            keyboard_hit = self._hand_keyboard_hit(hand_index)
            if touch_active:
                # Windows multi-touch is the preferred input. The OS pins the
                # cursor to an active contact, so only move SetCursorPos for
                # hover, and never fight a two-contact gesture (pinch/zoom/pan)
                # with it.
                if (
                    hit is not None
                    and not keyboard_hit
                    and not (
                        self._touch_state["left"] == "down"
                        and self._touch_state["right"] == "down"
                    )
                    and not physical_mouse_active
                ):
                    _set_cursor_pos(*self._cursor_pixel_for_screen_uv(hit[0], hit[1]))
                self._pointer_state[name] = "idle"
                continue
            if physical_mouse_active:
                # The hardware mouse owns the cursor now; release any beam-held
                # button and ignore the beam for this frame.
                if state != "idle":
                    _send_mouse_flags(up_flag)
                self._pointer_state[name] = "idle"
                continue
            if hit is None or keyboard_hit:
                if state != "idle":
                    _send_mouse_flags(up_flag)
                self._pointer_state[name] = "idle"
                continue
            # Move the OS cursor every frame the laser is on the screen,
            # matching the legacy _handle_cursor continuous tracking. The
            # trigger press below then clicks at this already-updated spot.
            _set_cursor_pos(*self._cursor_pixel_for_screen_uv(hit[0], hit[1]))
            if state == "idle" and trigger >= 0.7:
                _send_mouse_flags(down_flag)
                _send_mouse_flags(up_flag)
                self._pointer_press_time[name] = now
                self._pointer_state[name] = "pressed"
            elif state == "pressed":
                if trigger <= 0.3:
                    self._pointer_state[name] = "idle"
                elif now - self._pointer_press_time[name] >= 0.35:
                    _send_mouse_flags(down_flag)
                    self._pointer_state[name] = "dragging"
            elif state == "dragging":
                if trigger <= 0.3:
                    _send_mouse_flags(up_flag)
                    self._pointer_state[name] = "idle"

    def _open_settings_menu(self) -> None:
        self._cancel_touch_contacts()
        if self._head_position_w is None or self._head_model_matrix is None:
            return
        head_basis = np.asarray(self._head_model_matrix[:3, :3], dtype=np.float64)
        right = head_basis[:, 0]
        up = head_basis[:, 1]
        panel_normal = head_basis[:, 2]
        forward = -panel_normal
        basis = np.column_stack((right, up, panel_normal))
        position = np.asarray(self._head_position_w, dtype=np.float64) + forward * 1.1
        position -= up * 0.12
        self._settings_menu_pose = (
            tuple(float(value) for value in position),
            tuple(float(value) for value in _mat3_to_quat_xyzw(basis)),
        )
        self._refresh_settings_menu_values()
        self._settings_menu.open()

    def _settings_menu_matrix(self) -> np.ndarray | None:
        if self._settings_menu_pose is None:
            return None
        position, quaternion = self._settings_menu_pose
        qx, qy, qz, qw = quaternion
        quaternion_value = type(
            "MenuQuaternion", (), {"x": qx, "y": qy, "z": qz, "w": qw}
        )()
        matrix = _xr_quat_to_mat4(quaternion_value).astype(np.float64)
        matrix[:3, 3] = np.asarray(position, dtype=np.float64)
        return matrix

    def _set_settings_menu_matrix(self, matrix: np.ndarray) -> None:
        self._settings_menu_pose = (
            tuple(float(value) for value in matrix[:3, 3]),
            tuple(float(value) for value in _mat3_to_quat_xyzw(matrix[:3, :3])),
        )

    def _resolve_environment_selection(
        self, model: str
    ) -> tuple[Path | None, Path, Path | None]:
        environments_root = Path(__file__).resolve().parent / "environments"
        requested = str(model or "Default").strip() or "Default"
        canonical = next(
            (key for key in discover_environment_keys() if key.lower() == requested.lower()),
            requested,
        )
        room_dir = environments_root / canonical
        profile_path = room_dir / "profile.json"
        if not profile_path.is_file():
            raise FileNotFoundError(f"environment profile not found: {profile_path}")
        profile = json.loads(profile_path.read_text(encoding="utf-8-sig"))
        if not isinstance(profile, dict):
            raise ValueError(f"environment profile root must be an object: {profile_path}")
        glb_value = profile.get("glb", "environment.glb")
        glb_path = None
        if glb_value not in (None, "", False):
            glb_path = room_dir / str(glb_value)
            if not glb_path.is_file():
                raise FileNotFoundError(f"environment GLB not found: {glb_path}")
        background = profile.get("background")
        background = background if isinstance(background, dict) else {}
        image_value = (
            background.get("image") or background.get("path")
            or background.get("file") or profile.get("background_image")
        )
        panorama_path = None
        if image_value:
            panorama_path = room_dir / str(image_value)
            if not panorama_path.is_file():
                raise FileNotFoundError(
                    f"environment panorama not found: {panorama_path}"
                )
        if glb_path is None and panorama_path is None and canonical != "Default":
            raise ValueError(f"environment has neither GLB nor panorama: {canonical}")
        return glb_path, profile_path, panorama_path

    def _release_panorama_sources(self) -> None:
        if self._panorama_staging is not None:
            self._panorama_staging.close()
            self._panorama_staging = None
        if self._panorama_swapchain is not None:
            self._destroy_projection_swapchain(self._panorama_swapchain)
            self._panorama_swapchain = None
        for attribute in ("_vulkan_panorama_image", "_vulkan_panorama_staging"):
            image = getattr(self, attribute)
            if image is not None:
                image.close()
                setattr(self, attribute, None)
        self._panorama_size = None
        self._panorama_layer = None
        self._panorama_failed = False
        self._panorama_skip_logged = False
        if hasattr(self, "_panorama_initial_orientation"):
            del self._panorama_initial_orientation

    def _reset_environment_profile_state(self) -> None:
        self._filament_profile_data = {}
        self._filament_view_poses = ()
        self._filament_view_pose_index = 0
        self._room_seat_height_offset = 0.0
        self._filament_lighting_presets = ()
        self._filament_lighting_preset_index = 0
        self._profile_near_plane = 0.05
        self._profile_far_plane = 1000.0
        self._filament_scene_exposure = self.config.filament_scene_exposure_ev
        self._filament_skybox_brightness = self.config.filament_skybox_brightness
        self._filament_ambient_light_color = self.config.filament_ambient_light_color
        self._controller_ambient_light_color_override = None
        self._controller_hdr_ambient_light_color_override = None
        self._filament_ambient_light_intensity_lux = (
            self.config.filament_ambient_light_intensity_lux
        )
        self._controller_ambient_light_intensity_lux = (
            self.config.filament_controller_ambient_light_intensity_lux
        )
        self._controller_hdr_ambient_light_intensity_lux = (
            self.config.filament_controller_hdr_ambient_light_intensity_lux
        )
        self._controller_light_intensity_candela = (
            self.config.filament_controller_light_intensity_candela
        )
        self._filament_fill_light_color = self.config.filament_fill_light_color
        self._filament_fill_light_intensity = self.config.filament_fill_light_intensity
        self._filament_fill_light_direction = self.config.filament_fill_light_direction
        self._controller_head_light_weight = self.config.filament_controller_head_light_weight
        self._controller_top_light_weight = self.config.filament_controller_top_light_weight
        self._controller_top_light_color = self.config.filament_controller_top_light_color
        self._controller_head_light_offset = self.config.filament_controller_head_light_offset
        self._controller_top_light_offset = self.config.filament_controller_top_light_offset
        self._controller_head_light_falloff = self.config.filament_controller_head_light_falloff
        self._controller_top_light_falloff = self.config.filament_controller_top_light_falloff
        self._controller_head_light_cast_shadows = (
            self.config.filament_controller_head_light_cast_shadows
        )
        self._controller_top_light_cast_shadows = (
            self.config.filament_controller_top_light_cast_shadows
        )
        self._controller_screen_light_enabled = (
            self.config.filament_controller_screen_light_enabled
        )
        self._controller_screen_light_intensity_lux = (
            self.config.filament_controller_screen_light_intensity_lux
        )
        self._controller_screen_light_saturation = (
            self.config.filament_controller_screen_light_saturation
        )
        self._controller_screen_light_max_luminance = (
            self.config.filament_controller_screen_light_max_luminance
        )
        self._controller_screen_light_smoothing_seconds = (
            self.config.filament_controller_screen_light_smoothing_seconds
        )
        self._controller_screen_light_sample_hz = (
            self.config.filament_controller_screen_light_sample_hz
        )
        self._controller_screen_light_cast_shadows = (
            self.config.filament_controller_screen_light_cast_shadows
        )
        self._environment_screen_light_enabled = (
            self.config.filament_environment_screen_light_enabled
        )
        self._environment_screen_light_intensity_candela = (
            self.config.filament_environment_screen_light_intensity_candela
        )
        self._environment_screen_area_light_intensity = 6.0
        self._environment_screen_light_saturation = (
            self.config.filament_environment_screen_light_saturation
        )
        self._environment_screen_light_max_luminance = (
            self.config.filament_environment_screen_light_max_luminance
        )
        self._environment_screen_light_smoothing_seconds = (
            self.config.filament_environment_screen_light_smoothing_seconds
        )
        self._environment_screen_light_sample_hz = (
            self.config.filament_environment_screen_light_sample_hz
        )
        self._environment_screen_light_falloff = (
            self.config.filament_environment_screen_light_falloff
        )
        self._environment_screen_light_offset = (
            self.config.filament_environment_screen_light_offset
        )
        self._environment_screen_light_cast_shadows = (
            self.config.filament_environment_screen_light_cast_shadows
        )
        self._filament_glow_sample_hz = self.config.filament_glow_sample_hz
        self._filament_glow_smoothing_seconds = (
            self.config.filament_glow_smoothing_seconds
        )
        self._controller_hdr_lighting = False
        self._filament_screen = None
        self._filament_screen_initial = None
        self._filament_screen_profile_authored = False
        self._filament_screen_head_initialized = False
        self._settings_menu_allow_curve = True
        self._screen_curved = False
        self._screen_curve_half_angle = 0.0
        self._screen_initial_curve_half_angle = 0.0
        self._filament_glow_mode = "off"
        self._filament_glow_intensity = 0.175
        self._filament_glow_width = 0.75
        self._filament_glow_default_multiplier = 1.5
        self._filament_glow_intensity_multiplier = 0.0
        self._filament_glow_shell_default_multiplier = 1.85
        self._filament_glow_shell_intensity_multiplier = 0.0
        self._filament_glow_shell_radius = 20.0
        self._filament_glow_shell_height = 9.5
        self._veil_intensity = 1.5
        self._veil_alpha = 1.0

    def _hot_switch_environment(self, model: str) -> bool:
        try:
            glb_path, profile_path, panorama_path = (
                self._resolve_environment_selection(model)
            )
            glb_data = glb_path.read_bytes() if glb_path is not None else None
        except Exception as exc:
            self._preset_name_overlay = f"Room switch failed: {exc}"
            self._preset_osd_show_t = time.perf_counter()
            print(f"[OpenXRViewer] Environment hot switch rejected: {exc}", flush=True)
            return False

        old_config = self.config
        old_paths = (
            old_config.filament_glb_path,
            old_config.filament_profile_path,
            old_config.filament_panorama_path,
        )
        old_glb_data = None
        if old_paths[0]:
            try:
                old_glb_data = Path(old_paths[0]).read_bytes()
            except OSError:
                pass
        previous_head_transform = (
            None if self._profile_head_transform is None
            else np.asarray(self._profile_head_transform, dtype=np.float64).copy()
        )
        try:
            if self.vulkan is not None:
                self.vulkan.wait_idle()
            panorama_mode_changed = bool(old_paths[2]) != bool(panorama_path)
            if panorama_mode_changed and self._vulkan_projection_screen_pass is not None:
                self._vulkan_projection_screen_pass.close()
                self._vulkan_projection_screen_pass = None
            bridge = self.filament_bridge
            if bridge is not None:
                bridge.wait_for_idle()
                if glb_data is None:
                    bridge.unload_glb()
                else:
                    bridge.load_glb(glb_data)
            self._release_panorama_sources()
            self.config = replace(
                self.config,
                filament_glb_path=(str(glb_path) if glb_path is not None else None),
                filament_profile_path=str(profile_path),
                filament_panorama_path=(
                    str(panorama_path) if panorama_path is not None else None
                ),
            )
            self._reset_environment_profile_state()
            self._load_filament_profile()
            self._filament_glow_environment_enabled = bool(
                glb_path is None and panorama_path is None
            )
            if bridge is None and glb_path is not None:
                self._initialize_filament_bridges()
                bridge = self.filament_bridge
                if bridge is None:
                    raise RuntimeError(
                        "Filament Bridge is unavailable for the selected GLB room"
                    )
            if bridge is not None:
                self._apply_filament_scene_exposure_to_bridge(bridge)
                bridge.set_skybox_brightness(self._filament_skybox_brightness)
                self._apply_filament_bridge_lighting(bridge)
            self._move_settings_menu_with_profile_head(previous_head_transform)
            self._profile_space_applied = False
            self._profile_space_calibration_pass = 0
            self._profile_space_preserve_anchor = bool(
                self._profile_reference_head_anchor is not None
            )
            self._profile_reference_space_change_ignored_logged = False
            self._profile_alignment_logged = False
            selected = profile_path.parent.name
            self._settings_menu_values["room:model"] = selected
            self._refresh_settings_menu_values()
            self._dispatch_controller_shortcut(
                "select_environment_model", model=selected
            )
            self._preset_name_overlay = f"Room: {selected}"
            self._preset_osd_show_t = time.perf_counter()
            self._settings_menu.mark_dirty()
            print(
                f"[OpenXRViewer] Environment hot switch complete: model={selected}",
                flush=True,
            )
            return True
        except Exception as exc:
            print(
                f"[OpenXRViewer] Environment hot switch failed: "
                f"{type(exc).__name__}: {exc}; restoring previous environment",
                flush=True,
            )
            self.config = old_config
            try:
                if self.filament_bridge is not None:
                    self.filament_bridge.wait_for_idle()
                    if old_glb_data is None:
                        self.filament_bridge.unload_glb()
                    else:
                        self.filament_bridge.load_glb(old_glb_data)
                self._release_panorama_sources()
                self._reset_environment_profile_state()
                self._load_filament_profile()
                self._apply_filament_bridge_lighting()
            except Exception as rollback_exc:
                print(
                    "[OpenXRViewer] Environment rollback failed: "
                    f"{type(rollback_exc).__name__}: {rollback_exc}",
                    flush=True,
                )
            self._preset_name_overlay = f"Room switch failed: {type(exc).__name__}"
            self._preset_osd_show_t = time.perf_counter()
            self._settings_menu.mark_dirty()
            return False

    def _move_settings_menu_with_profile_head(
        self, previous_head_transform: np.ndarray | None
    ) -> None:
        if (
            previous_head_transform is None
            or self._profile_head_transform is None
        ):
            return
        menu_matrix = self._settings_menu_matrix()
        if menu_matrix is None:
            return
        seat_delta = (
            np.asarray(self._profile_head_transform, dtype=np.float64)
            @ np.linalg.inv(np.asarray(previous_head_transform, dtype=np.float64))
        )
        self._set_settings_menu_matrix(seat_delta @ menu_matrix)

    def _handle_settings_menu_grip_drag(
        self, inputs, hits: tuple[tuple[float, float] | None, ...]
    ) -> None:
        grip_matrices = (self._grip_mat_l, self._grip_mat_r)
        for hand in (0, 1):
            pressed = float(inputs[hand].get("grip", 0.0) or 0.0) > 0.5
            if pressed and not self._settings_menu_grip_down[hand]:
                self._settings_menu_grip_down[hand] = True
                if (
                    self._settings_menu_grab_hand is None
                    and hits[hand] is not None
                    and grip_matrices[hand] is not None
                ):
                    menu_matrix = self._settings_menu_matrix()
                    if menu_matrix is not None:
                        self._settings_menu_grab_hand = hand
                        self._settings_menu_grab_relative = (
                            np.linalg.inv(np.asarray(grip_matrices[hand], dtype=np.float64))
                            @ menu_matrix
                        )
            elif not pressed and self._settings_menu_grip_down[hand]:
                self._settings_menu_grip_down[hand] = False
                if self._settings_menu_grab_hand == hand:
                    self._settings_menu_grab_hand = None
                    self._settings_menu_grab_relative = None
        hand = self._settings_menu_grab_hand
        if (
            hand is not None
            and self._settings_menu_grab_relative is not None
            and grip_matrices[hand] is not None
        ):
            self._set_settings_menu_matrix(
                np.asarray(grip_matrices[hand], dtype=np.float64)
                @ self._settings_menu_grab_relative
            )

    def _refresh_settings_menu_values(self) -> None:
        snapshot = None
        callback = self._on_controller_shortcut
        owner = getattr(callback, "__self__", None)
        context = getattr(owner, "context", None)
        state = getattr(context, "openxr_state", None)
        snapshot = getattr(state, "runtime_settings_snapshot", None)
        defaults = {
            "color_brightness": 1.0, "color_contrast": 1.0,
            "color_saturation": 1.0, "color_gamma": 1.0,
            "color_temperature": 0.0, "color_tint": 0.0,
            "vulkan_projection_min_lod": 0.0,
            "vulkan_projection_max_lod": 0.35,
            "vulkan_projection_mip_lod_bias": -0.35,
            "vulkan_projection_rcas_sharpness": 0.5,
        }
        for key, default in defaults.items():
            value = getattr(snapshot, key, None) if snapshot is not None else None
            if value is not None:
                self._settings_menu_values[key] = float(value)
            else:
                self._settings_menu_values.setdefault(key, default)
        self._settings_menu_values.setdefault(
            "openxr_render_scale", self._openxr_render_scale
        )
        depth_value = getattr(snapshot, "depth_strength", None) if snapshot is not None else None
        if depth_value is not None:
            self._settings_menu_values["depth_strength"] = float(depth_value)
        else:
            self._settings_menu_values.setdefault("depth_strength", 0.25)
        cross_eyed = getattr(snapshot, "cross_eyed", None) if snapshot is not None else None
        if cross_eyed is not None:
            self._settings_menu_values["cross_eyed"] = bool(cross_eyed)
        else:
            self._settings_menu_values.setdefault("cross_eyed", False)
        locale = self._overlay_language()
        environment_keys = discover_environment_keys()
        display_names = load_environment_display_names(environment_keys)
        environments_root = Path(__file__).resolve().parent / "environments"
        room_keys = []
        for key in environment_keys:
            try:
                profile = json.loads(
                    (environments_root / key / "profile.json").read_text(
                        encoding="utf-8-sig"
                    )
                )
            except (OSError, ValueError):
                continue
            background = profile.get("background") if isinstance(profile, dict) else None
            has_panorama = (
                isinstance(background, dict)
                and background.get("image") not in (None, "", False)
                and str(profile.get("environment_type", "")).strip().lower()
                == "panorama"
            )
            if isinstance(profile, dict) and (
                key.lower() == "default"
                or
                profile.get("glb") not in (None, "", False) or has_panorama
            ):
                room_keys.append(key)
        self._settings_menu.room_models = tuple(
            (key, environment_display_label(key, locale, display_names))
            for key in room_keys
        )
        selected_room = (
            Path(self.config.filament_profile_path).parent.name
            if self.config.filament_profile_path else "Default"
        )
        self._settings_menu.room_tab_visible = selected_room.strip().lower() != "default"
        if not self._settings_menu.room_tab_visible and self._settings_menu.tab == "room":
            self._settings_menu.set_tab("picture")
        self._settings_menu_values["room:model"] = (
            selected_room
        )
        self._settings_menu_values["room:seat_index"] = int(
            self._filament_view_pose_index
        )
        self._settings_menu_values["room:seat_height"] = float(
            self._room_seat_height_offset
        )
        self._settings_menu_values["room:exposure"] = float(
            self._filament_scene_exposure
        )
        self._settings_menu_values["room:screen_reflection_enabled"] = bool(
            self._environment_screen_light_enabled
        )
        if self._filament_screen is not None and self._filament_screen_initial is not None:
            self._settings_menu_values["screen:width"] = (
                float(self._filament_screen[1])
                / max(float(self._filament_screen_initial[1]), 1e-6)
            )
            self._settings_menu_values["screen:height"] = (
                float(self._filament_screen[0][1])
                - float(self._filament_screen_initial[0][1])
            )
            head = np.asarray(
                self._head_position_w if self._head_position_w is not None
                else (0.0, 0.0, 0.0),
                dtype=np.float64,
            )
            initial_position = np.asarray(self._filament_screen_initial[0], dtype=np.float64)
            current_position = np.asarray(self._filament_screen[0], dtype=np.float64)
            self._settings_menu_values["screen:distance"] = float(
                np.linalg.norm(current_position - head)
                / max(float(np.linalg.norm(initial_position - head)), 1e-6)
            )
        self._settings_menu_values["screen:curve"] = bool(self._screen_curved)
        self._settings_menu_values["screen:curve_half_angle"] = float(
            self._screen_curve_half_angle
        )
        self._settings_menu_values["screen_allow_curve"] = bool(self._settings_menu_allow_curve)
        show_glow = bool(self._filament_glow_environment_enabled)
        self._settings_menu_values["show_glow_tab"] = show_glow
        self._settings_menu_values["glow:mode"] = self._normalize_filament_glow_mode(
            self._filament_glow_mode
        )
        if self._settings_menu.tab == "glow" and not show_glow:
            self._settings_menu.set_tab("picture")

    def _settings_menu_ray_hit(self, hand: int) -> tuple[float, float] | None:
        if not self._settings_menu.visible or self._settings_menu_pose is None:
            return None
        origin, direction = self._controller_interaction_ray(hand)
        if origin is None or direction is None:
            return None
        position, quaternion = self._settings_menu_pose
        qx, qy, qz, qw = quaternion
        quaternion_value = type(
            "MenuQuaternion", (), {"x": qx, "y": qy, "z": qz, "w": qw}
        )()
        basis4 = _xr_quat_to_mat4(quaternion_value).astype(np.float64)
        normal = basis4[:3, 2]
        denominator = float(np.dot(normal, direction))
        if abs(denominator) < 1e-6:
            return None
        distance = float(np.dot(normal, np.asarray(position) - origin) / denominator)
        if distance <= 0.0:
            return None
        world = np.asarray(origin) + np.asarray(direction) * distance
        local = basis4[:3, :3].T @ (world - np.asarray(position))
        u = float(local[0]) / SETTINGS_MENU_WORLD_SIZE[0] + 0.5
        v = 0.5 - float(local[1]) / SETTINGS_MENU_WORLD_SIZE[1]
        if 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0:
            return u, v
        return None

    def _apply_settings_menu_control(self, control, uv, *, persist: bool = False) -> None:
        key = control.key
        if key.startswith("tab:"):
            self._settings_menu.set_tab(key[4:])
            return
        if key.startswith("step:"):
            _prefix, operation, target_key = key.split(":", 2)
            target = next(
                (
                    item for item in self._settings_menu.controls(
                        allow_curve=self._settings_menu_allow_curve,
                        show_glow=self._filament_glow_environment_enabled,
                        lang=self._overlay_language(),
                    )
                    if item.key == target_key and item.kind == "slider"
                ),
                None,
            )
            if target is None:
                return
            current = float(
                self._settings_menu_values.get(
                    target_key, 0.0 if target.minimum <= 0.0 <= target.maximum
                    else target.minimum
                )
            )
            direction = -1.0 if operation == "minus" else 1.0
            value = max(
                target.minimum,
                min(target.maximum, current + direction * target.step),
            )
            value = round(value / target.step) * target.step
            fraction = (value - target.minimum) / max(
                target.maximum - target.minimum, 1e-9
            )
            slider_u = target.rect[0] + fraction * (
                target.rect[2] - target.rect[0]
            )
            self._apply_settings_menu_control(
                target, (slider_u, uv[1]), persist=True
            )
            if target_key == "openxr_render_scale":
                self._pending_openxr_render_scale = float(
                    self._settings_menu_values[target_key]
                )
                self._dispatch_controller_shortcut(
                    "persist_openxr_render_scale",
                    value=self._settings_menu_values[target_key],
                )
            return
        if key == "close":
            self._settings_menu.close()
            return
        if key == "section:reset_defaults" and self._settings_menu.tab == "picture":
            self._settings_menu_values.update(PICTURE_DEFAULTS)
            self._pending_openxr_render_scale = float(
                PICTURE_DEFAULTS["openxr_render_scale"]
            )
            self._dispatch_controller_shortcut(
                "set_runtime_settings",
                settings={
                    name: value for name, value in PICTURE_DEFAULTS.items()
                    if name != "openxr_render_scale"
                },
                persist=True,
            )
            self._dispatch_controller_shortcut(
                "persist_openxr_render_scale",
                value=self._pending_openxr_render_scale,
            )
            self._settings_menu.mark_dirty()
            return
        if key == "depth:toggle_stereo":
            self._dispatch_controller_shortcut("toggle_stereo")
            callback_depth = self._controller_callback_depth_strength()
            if callback_depth is not None:
                self._settings_menu_values["depth_strength"] = callback_depth
            self._settings_menu.mark_dirty()
            return
        if key == "depth:toggle_cross_eyed":
            value = not bool(self._settings_menu_values.get("cross_eyed", False))
            self._settings_menu_values["cross_eyed"] = value
            self._dispatch_controller_shortcut(
                "set_runtime_setting", name="cross_eyed", value=value, persist=True
            )
            self._settings_menu.mark_dirty()
            return
        if key == "section:reset_defaults" and self._settings_menu.tab == "depth":
            self._settings_menu_values.update({
                "depth_strength": 0.25,
                "cross_eyed": False,
            })
            self._dispatch_controller_shortcut(
                "set_runtime_settings",
                settings={"depth_strength": 0.25, "cross_eyed": False},
                persist=True,
            )
            self._settings_menu.mark_dirty()
            return
        if key.startswith("glow:"):
            if not self._filament_glow_environment_enabled:
                return
            self._set_filament_glow_mode(key.split(":", 1)[1])
            self._settings_menu_values["glow:mode"] = self._filament_glow_mode
            self._settings_menu.mark_dirty()
            return
        if key.startswith("room:model:"):
            model = key.split(":", 2)[2]
            self._hot_switch_environment(model)
            return
        if key.startswith("room:seat:"):
            indices = {"front": 0, "middle": 1, "back": 2}
            self._apply_settings_menu_seat(indices[key.rsplit(":", 1)[1]])
            return
        if key == "room:toggle_screen_reflection":
            enabled = not bool(self._environment_screen_light_enabled)
            self._environment_screen_light_enabled = enabled
            self._settings_menu_values[
                "room:screen_reflection_enabled"
            ] = enabled
            if not enabled and self.filament_bridge is not None:
                self._update_environment_screen_lights(
                    None, self.filament_bridge
                )
            self._settings_menu.mark_dirty()
            return
        if key == "section:reset_defaults" and self._settings_menu.tab == "screen":
            if self._filament_screen_initial is not None:
                self._filament_screen = self._filament_screen_initial
            self._screen_curve_half_angle = self._screen_initial_curve_half_angle
            self._screen_curved = self._screen_curve_half_angle > 1e-6
            self._refresh_settings_menu_values()
            self._settings_menu.mark_dirty()
            return
        if key.startswith("screen:rotate:"):
            if self._filament_screen is None:
                return
            delta = -90.0 if key.endswith("-90") else 90.0
            position, width, height, rotation = self._filament_screen
            self._filament_screen = (
                position, width, height,
                (float(rotation[0]), float(rotation[1]), float(rotation[2]) + delta),
            )
            self._settings_menu.mark_dirty()
            return
        if control.kind == "slider":
            value = control.value_from_u(float(uv[0]))
            if key.startswith("screen:"):
                if self._filament_screen is None or self._filament_screen_initial is None:
                    return
                position, width, height, rotation = self._filament_screen
                initial_position, initial_width, initial_height, _initial_rotation = self._filament_screen_initial
                if key == "screen:width":
                    width = float(initial_width) * value
                    height = float(initial_height) * value
                elif key == "screen:height":
                    position = (position[0], float(initial_position[1]) + value, position[2])
                elif key == "screen:distance":
                    head = np.asarray(
                        self._head_position_w if self._head_position_w is not None
                        else (0.0, 0.0, 0.0),
                        dtype=np.float64,
                    )
                    initial_vector = np.asarray(initial_position, dtype=np.float64) - head
                    initial_length = max(float(np.linalg.norm(initial_vector)), 1e-6)
                    position = tuple(
                        float(component) for component in (
                            head + initial_vector / initial_length * initial_length * value
                        )
                    )
                self._filament_screen = (position, width, height, rotation)
            elif key == "room:seat_height":
                self._apply_settings_menu_seat_height(value)
            elif key == "room:exposure":
                self._filament_scene_exposure = float(value)
                self._apply_filament_scene_exposure_to_bridge()
            elif not key.startswith("room:"):
                if key == "vulkan_projection_min_lod":
                    value = min(value, float(self._settings_menu_values.get("vulkan_projection_max_lod", 2.0)))
                elif key == "vulkan_projection_max_lod":
                    value = max(value, float(self._settings_menu_values.get("vulkan_projection_min_lod", 0.0)))
                if key != "openxr_render_scale":
                    self._dispatch_controller_shortcut(
                        "set_runtime_setting", name=key, value=value, persist=persist
                    )
            self._settings_menu_values[key] = value
            self._settings_menu.mark_dirty()
        elif key.startswith("screen:type:"):
            levels = {
                "screen:type:flat": 0.0,
                "screen:type:subtle": math.radians(20.0),
                "screen:type:medium": math.radians(30.0),
                "screen:type:deep": 0.72,
            }
            half_angle = levels[key]
            if half_angle > 0.0 and not self._settings_menu_allow_curve:
                return
            self._screen_curve_half_angle = half_angle
            self._screen_curved = half_angle > 0.0
            self._settings_menu_values["screen:curve_half_angle"] = half_angle
            self._settings_menu_values["screen:curve"] = self._screen_curved
            self._settings_menu.mark_dirty()

    def _apply_settings_menu_seat(self, index: int) -> None:
        if not self._filament_view_poses:
            return
        previous_head_transform = (
            None if self._profile_head_transform is None
            else np.asarray(self._profile_head_transform, dtype=np.float64).copy()
        )
        index = int(index) % min(3, len(self._filament_view_poses))
        self._filament_view_pose_index = index
        pose = self._filament_view_poses[index]
        self._set_profile_head_transform_from_view_pose(
            pose, self._filament_profile_data
        )
        self._move_settings_menu_with_profile_head(previous_head_transform)
        self._room_seat_height_offset = 0.0
        self._profile_space_applied = False
        self._profile_space_calibration_pass = 0
        self._profile_space_preserve_anchor = bool(
            self._profile_reference_head_anchor is not None
        )
        self._settings_menu_values["room:seat_index"] = index
        self._settings_menu_values["room:seat_height"] = 0.0
        self._settings_menu.mark_dirty()

    def _apply_settings_menu_seat_height(self, offset: float) -> None:
        if self._profile_head_transform is None:
            return
        previous_head_transform = np.asarray(
            self._profile_head_transform, dtype=np.float64
        ).copy()
        delta = float(offset) - float(self._room_seat_height_offset)
        self._profile_head_transform[1, 3] += delta
        self._move_settings_menu_with_profile_head(previous_head_transform)
        self._room_seat_height_offset = float(offset)
        self._profile_space_applied = False
        self._profile_space_calibration_pass = 0
        self._profile_space_preserve_anchor = bool(
            self._profile_reference_head_anchor is not None
        )

    def _set_profile_head_transform_from_view_pose(
        self, view_pose: dict[str, Any], profile: dict[str, Any]
    ) -> None:
        model_position = profile.get("model_position", [0.0, 0.0, 0.0])
        model_rotation_deg = profile.get("model_rotation_deg", [0.0, 0.0, 0.0])
        model_scale = profile.get("model_scale", [1.0, 1.0, 1.0])
        world = np.asarray(
            [float(view_pose[key]) for key in ("x", "y", "z")],
            dtype=np.float32,
        )
        rotation_deg = view_pose.get("rotation_deg")
        if not isinstance(rotation_deg, (list, tuple)) or len(rotation_deg) < 3:
            rotation_deg = [float(view_pose.get("angle", 0.0)), 0.0, 0.0]
        pose_space = str(view_pose.get(
            "view_pose_space", view_pose.get(
                "pose_space", profile.get("view_pose_space", "world")
            )
        )).strip().lower()
        if pose_space in {"scene", "glb", "local"}:
            position = world
        else:
            model = euler_to_mat4(*(
                math.radians(float(value)) for value in model_rotation_deg[:3]
            )).astype(np.float32)
            model[:3, 3] = np.asarray(model_position[:3], dtype=np.float32)
            model[:3, :3] = model[:3, :3] @ np.diag(
                np.asarray(model_scale[:3], dtype=np.float32)
            )
            position = (np.linalg.inv(model) @ np.append(world, 1.0))[:3]
        transform = euler_to_mat4(*(
            math.radians(float(value)) for value in rotation_deg[:3]
        )).astype(np.float32)
        transform[:3, 3] = position
        self._profile_head_transform = transform
        self._profile_view_name = str(view_pose.get("name", "profile"))

    def _handle_settings_menu_input(self) -> bool:
        inputs = (self._controller_input(0), self._controller_input(1))
        if not self._settings_menu.visible:
            for hand in (0, 1):
                trigger = float(inputs[hand].get("trigger", 0.0) or 0.0)
                if self._settings_menu_trigger_down[hand]:
                    if trigger <= 0.3:
                        self._settings_menu_trigger_down[hand] = False
                    continue
                origin, direction = self._controller_interaction_ray(hand)
                keyboard_hit = (
                    self._keyboard_visible and origin is not None
                    and self._keyboard_plane_hit(origin, direction) != (None, None)
                )
                outside = self._screen_ray_hit_for_hand(hand) is None and not keyboard_hit
                if self._settings_menu.sample_trigger(
                    hand, trigger,
                    outside_targets=outside,
                ):
                    self._open_settings_menu()
                    return True
            return False

        hits = (self._settings_menu_ray_hit(0), self._settings_menu_ray_hit(1))
        self._handle_settings_menu_grip_drag(inputs, hits)
        if self._settings_menu_grab_hand is not None:
            hits = (self._settings_menu_ray_hit(0), self._settings_menu_ray_hit(1))
        hover = None
        for hand in (0, 1):
            if hits[hand] is not None:
                hover = self._settings_menu.hit_test(
                    hits[hand], allow_curve=self._settings_menu_allow_curve,
                    show_glow=self._filament_glow_environment_enabled,
                    lang=self._overlay_language(),
                )
                break
        next_hover = None if hover is None else hover.key
        next_cursor = next((hit for hit in hits if hit is not None), None)
        if next_cursor != self._settings_menu_cursor_uv:
            self._settings_menu_cursor_uv = next_cursor
            self._settings_menu.mark_dirty()
        if next_hover != self._settings_menu.hover_key:
            self._settings_menu.hover_key = next_hover
            self._settings_menu.mark_dirty()
        for hand in (0, 1):
            trigger = float(inputs[hand].get("trigger", 0.0) or 0.0)
            if not self._settings_menu_trigger_down[hand] and trigger >= 0.7:
                if self._settings_menu.active_hand is None:
                    self._settings_menu.active_hand = hand
                    self._settings_menu.active_key = next_hover
                    self._settings_menu_trigger_down[hand] = True
                    if hits[hand] is None:
                        self._settings_menu.close()
                    elif hover is not None:
                        self._apply_settings_menu_control(hover, hits[hand])
            elif self._settings_menu_trigger_down[hand] and trigger <= 0.3:
                self._settings_menu_trigger_down[hand] = False
                if self._settings_menu.active_hand == hand:
                    active_key = self._settings_menu.active_key
                    if active_key and not active_key.startswith(("screen:", "room:")):
                        active_control = next(
                            (item for item in self._settings_menu.controls(
                                allow_curve=self._settings_menu_allow_curve,
                                show_glow=self._filament_glow_environment_enabled,
                                lang=self._overlay_language(),
                            ) if item.key == active_key),
                            None,
                        )
                        if active_control is not None and active_control.kind == "slider":
                            if active_key == "openxr_render_scale":
                                self._pending_openxr_render_scale = float(
                                    self._settings_menu_values[active_key]
                                )
                                self._dispatch_controller_shortcut(
                                    "persist_openxr_render_scale",
                                    value=self._settings_menu_values[active_key],
                                )
                            else:
                                self._dispatch_controller_shortcut(
                                    "set_runtime_setting",
                                    name=active_key,
                                    value=self._settings_menu_values[active_key],
                                    persist=True,
                                )
                    self._settings_menu.active_hand = None
                    self._settings_menu.active_key = None
            elif (
                self._settings_menu_trigger_down[hand]
                and self._settings_menu.active_hand == hand
                and hits[hand] is not None
            ):
                control = self._settings_menu.hit_test(
                    hits[hand], allow_curve=self._settings_menu_allow_curve,
                    show_glow=self._filament_glow_environment_enabled,
                    lang=self._overlay_language(),
                )
                if control is not None and control.kind == "slider" and control.key == self._settings_menu.active_key:
                    now = time.perf_counter()
                    if now - self._settings_menu_last_adjust >= 0.05:
                        self._apply_settings_menu_control(control, hits[hand])
                        self._settings_menu_last_adjust = now
        return True

    def run(self, frame_limit: int | None = None) -> int:
        self.initialize()
        while frame_limit is None or self.frame_count < frame_limit:
            if not self.run_frame():
                break
        return self.frame_count

    def run_until(self, shutdown_event: Any) -> int:
        """Run the XR frame loop until the application shutdown event is set."""
        self._presenter_thread_id = threading.get_ident()
        retry_count = 0
        try:
            while not shutdown_event.is_set() and not self.exit_requested:
                try:
                    if not self._initialized:
                        self.initialize()
                    retry_count = 0
                    while not shutdown_event.is_set() and not self.exit_requested:
                        if not self.run_frame():
                            break
                        if self._session_requires_reconnect():
                            self.close()
                            self.exit_requested = False
                            self._notify_headset_waiting()
                            break
                    if self._session_requires_reconnect():
                        self.close()
                        self.exit_requested = False
                        self._notify_headset_waiting()
                except Exception as exc:
                    if not self._is_no_headset_error(exc):
                        raise
                    print(
                        "[OpenXRViewer] OpenXR HMD form factor unavailable; "
                        "Vulkan/Filament initialization deferred until headset wake-up",
                        flush=True,
                    )
                    self.close()
                    self._notify_headset_waiting()

                if shutdown_event.is_set() or self.exit_requested:
                    break
                retry_count += 1
                delay = self._retry_delay(retry_count)
                print(
                    f"[OpenXRViewer] Waiting for VR headset connect... "
                    f"(retry in {delay:.1f}s)",
                    flush=True,
                )
                shutdown_event.wait(delay)
            return self.frame_count
        finally:
            self.close()

    @staticmethod
    def _is_no_headset_error(exc: BaseException) -> bool:
        return type(exc).__name__ == "FormFactorUnavailableError"

    def _session_requires_reconnect(self) -> bool:
        state = self.session_state
        state_name = str(getattr(state, "name", state)).upper()
        return state_name in {"STOPPING", "LOSS_PENDING"}

    def _retry_delay(self, retry_count: int) -> float:
        base = max(0.1, float(self.config.openxr_standby_retry_interval))
        maximum = max(base, float(self.config.openxr_standby_retry_max_interval))
        if self.session_state is None:
            base = max(0.1, float(self.config.openxr_no_headset_retry_interval))
        return min(maximum, base * (2 ** max(0, retry_count - 1)))

    def _notify_headset_state(self, state: str) -> None:
        callback = self._on_headset_state
        if callback is None:
            return
        try:
            callback(state)
        except Exception as exc:
            print(
                f"[OpenXRViewer] Headset state callback failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

    def _notify_headset_waiting(self) -> None:
        # Do not let a frame produced before standby cross the recovery boundary.
        self._accept_output = False
        self._drop_output_frames()
        now = time.perf_counter()
        if self._headset_wait_started <= 0.0:
            self._headset_wait_started = now
            self._headset_hard_idle_notified = False
            self._headset_active_notified = False
            self._headset_wait_logged = False
            self._notify_headset_state("waiting")
        if not self._headset_wait_logged:
            self._headset_wait_logged = True
            print(
                "[OpenXRViewer] Headset not detected or in standby; "
                "waiting for headset wake-up",
                flush=True,
            )
        timeout = max(0.0, float(self.config.headset_wait_inference_timeout))
        if (
            not self._headset_hard_idle_notified
            and now - self._headset_wait_started >= timeout
        ):
            self._headset_hard_idle_notified = True
            self._notify_headset_state("hard_idle")
            print(
                f"[OpenXRViewer] No headset detected for {timeout:.0f}s; "
                "stopping source inference",
                flush=True,
            )

    def _notify_headset_active(self) -> None:
        if self._headset_active_notified:
            return
        self._headset_wait_started = 0.0
        self._headset_hard_idle_notified = False
        self._headset_active_notified = True
        self._headset_wait_logged = False
        self._source_frame_wait_logged = False
        self._accept_output = True
        self._notify_headset_state("active")
        print("[OpenXRViewer] Headset detected; source inference resumed", flush=True)

    def close(self) -> None:
        if _TOUCH_AVAILABLE and _touch_injector is not None:
            try:
                _touch_injector.cancel_all()
            except Exception:
                pass
        xr = self.xr
        vulkan_device_lost = bool(
            self.vulkan is not None
            and getattr(self.vulkan, "device_lost", False)
        )
        if self.vulkan is not None and not vulkan_device_lost:
            try:
                self.vulkan.wait_idle()
            except Exception:
                pass

        self._close_sbs_sequence_capture()

        # Release output-frame leases while their adapters and synchronization
        # objects are still alive.
        self._drop_output_frames()

        if self.vulkan is not None and not vulkan_device_lost:
            try:
                self.vulkan.wait_idle()
            except Exception:
                pass

        # Destroy Filament's external-texture wrappers before the adapters
        # destroy the borrowed screen and Glow VkImages.
        if self.filament_bridge is not None:
            try:
                self.filament_bridge.close()
            except Exception:
                pass
            self.filament_bridge = None
        if self._controller_composition_swapchain is not None:
            try:
                self._destroy_projection_swapchain(
                    self._controller_composition_swapchain
                )
            except Exception:
                pass
            self._controller_composition_swapchain = None
        for eye in reversed(self._vulkan_controller_proxy_swapchains):
            try:
                self._destroy_projection_swapchain(eye)
            except Exception:
                pass
        self._vulkan_controller_proxy_swapchains.clear()
        for attachment in self._filament_depth_attachments:
            try:
                attachment.close()
            except Exception:
                pass
        self._filament_depth_attachments.clear()

        if self._output_adapter is not None:
            try:
                self._output_adapter.close()
            except Exception:
                pass
            self._output_adapter = None

        if self._vulkan_msdf_quad_renderer is not None:
            try:
                self._vulkan_msdf_quad_renderer.close()
            except Exception:
                pass
            self._vulkan_msdf_quad_renderer = None

        if self._vulkan_projection_screen_pass is not None:
            try:
                self._vulkan_projection_screen_pass.close()
            except Exception:
                pass
            self._vulkan_projection_screen_pass = None
        if self._vulkan_multiview_diagnostic_pass is not None:
            try:
                self._vulkan_multiview_diagnostic_pass.close()
            except Exception:
                pass
            self._vulkan_multiview_diagnostic_pass = None

        for image in self._filament_multiview_hdr_images:
            try:
                image.close()
            except Exception:
                pass
        self._filament_multiview_hdr_images.clear()
        if self.vulkan is not None and getattr(self.vulkan, "device", None) is not None:
            for semaphore in self._filament_multiview_ready_semaphores:
                try:
                    self.vulkan.vk.vkDestroySemaphore(
                        self.vulkan.device, semaphore, None
                    )
                except Exception:
                    pass
        self._filament_multiview_ready_semaphores.clear()
        self._filament_multiview_slot_timelines.clear()
        self._filament_multiview_current = None
        self._filament_multiview_current_slot = None
        self._filament_multiview_finished_consumed = False

        if xr is not None:
            if self._panorama_staging is not None:
                try:
                    self._panorama_staging.close()
                except Exception:
                    pass
                self._panorama_staging = None
            if self._panorama_swapchain is not None:
                try:
                    self._destroy_projection_swapchain(self._panorama_swapchain)
                except Exception:
                    pass
                self._panorama_swapchain = None
            for image in (self._vulkan_panorama_image, self._vulkan_panorama_staging):
                if image is not None:
                    try:
                        image.close()
                    except Exception:
                        pass
            self._vulkan_panorama_image = None
            self._vulkan_panorama_staging = None
            self._panorama_layer = None
            self._destroy_tool_quad_layers()
            self._destroy_quad_swapchains()
            for eye in reversed(self.swapchains):
                for resource in reversed(eye.resources):
                    try:
                        if self.vulkan is not None:
                            self.vulkan.unregister_external_image(resource)
                    except Exception:
                        pass
                try:
                    xr.destroy_swapchain(eye.handle)
                except Exception:
                    pass
            self.swapchains.clear()
            self._multiview_active = False

            if self.reference_space is not None:
                try:
                    xr.destroy_space(self.reference_space)
                except Exception:
                    pass
                self.reference_space = None

            if self.session is not None:
                if self.session_running:
                    try:
                        xr.end_session(self.session)
                    except Exception:
                        pass
                try:
                    xr.destroy_session(self.session)
                except Exception:
                    pass
                self.session = None
                self.session_running = False

        if self.vulkan is not None:
            try:
                self.vulkan.close()
            except Exception:
                pass
            self.vulkan = None
        elif self._provisional_vk_instance is not None:
            try:
                import vulkan as vk

                if self._provisional_vk_device is not None:
                    vk.vkDestroyDevice(self._provisional_vk_device, None)
                vk.vkDestroyInstance(self._provisional_vk_instance, None)
            except Exception:
                pass
        self._provisional_vk_device = None
        self._provisional_vk_instance = None

        if xr is not None and self.instance is not None:
            if not vulkan_device_lost:
                try:
                    xr.destroy_instance(self.instance)
                except Exception:
                    pass
            self.instance = None

        self.system_id = None
        self.swapchain_format = None
        self._tool_quad_swapchain_format = None
        self._graphics_binding = None
        self._initialized = False
        self._last_screen_resolution_status = None
        self._last_screen_resolution_log_t = 0.0
        self._clear_presenter_commands()
        self._drop_output_frames()
        self._has_presented_frame = False
        self._last_quad_layers = []
        self._last_screen_quad_layers = []
        for host_image in tuple(
            self._visual_regression_source_host_images.values()
        ) + tuple(self._visual_regression_projection_host_images.values()):
            try:
                host_image.close()
            except Exception:
                pass
        self._visual_regression_source_host_images.clear()
        self._visual_regression_projection_host_images.clear()
        self._visual_regression_capture_eyes.clear()
        self._visual_regression_capture_failed = False
        self._source_frame_wait_logged = False
        self._accept_output = False
        self._filament_animation_origin = None
        self._profile_initial_head = None
        self._profile_space_applied = False
        self._profile_space_calibration_pass = 0
        self._profile_space_pose_in_reference = np.eye(4, dtype=np.float32)
        self._profile_reference_head_anchor = None
        self._profile_space_preserve_anchor = False
        self._profile_reference_space_change_ignored_logged = False
        self._profile_alignment_logged = False
        self._reference_space_type = None
        self._presenter_thread_id = None
        self._next_output_frame_id = 0

    def __enter__(self) -> "OpenXrVulkanPresenter":
        self.initialize()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _create_vulkan_objects(self, api_version: int) -> None:
        xr = self.xr
        import vulkan as vk

        self._vulkan_loader, self._vk_get_instance_proc_addr = _load_vulkan_proc_addr(xr)
        platform = _openxr_platform_module(xr)

        app_info = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName=self.config.application_name,
            applicationVersion=1,
            pEngineName="D2S",
            engineVersion=1,
            apiVersion=int(api_version),
        )
        instance_create_info = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app_info,
        )
        xr_instance, vulkan_result = xr.create_vulkan_instance_khr(
            self.instance,
            xr.VulkanInstanceCreateInfoKHR(
                system_id=self.system_id,
                pfn_get_instance_proc_addr=self._vk_get_instance_proc_addr,
                vulkan_create_info=_cffi_struct_pointer(
                    vk, instance_create_info, platform.VkInstanceCreateInfo
                ),
            ),
        )
        _check_vulkan_result(vulkan_result, "xrCreateVulkanInstanceKHR")
        vk_instance = _ctypes_handle_to_cffi(vk, "VkInstance", xr_instance)
        self._provisional_vk_instance = vk_instance

        xr_physical_device = xr.get_vulkan_graphics_device2_khr(
            self.instance,
            xr.VulkanGraphicsDeviceGetInfoKHR(
                system_id=self.system_id,
                vulkan_instance=xr_instance,
            ),
        )
        vk_physical_device = _ctypes_handle_to_cffi(
            vk, "VkPhysicalDevice", xr_physical_device
        )
        queue_family_index = find_graphics_queue_family(vk, vk_physical_device)
        try:
            timeline_features, synchronization2_enabled = _require_timeline_semaphore_features(
                vk, vk_physical_device, require_multiview=True
            )
        except VulkanCapabilityError as exc:
            raise OpenXrVulkanUnavailableError(str(exc)) from exc
        queue_family_properties = vk.vkGetPhysicalDeviceQueueFamilyProperties(
            vk_physical_device
        )
        available_queue_count = int(
            queue_family_properties[queue_family_index].queueCount
        )
        requested_queue_count = 2 if available_queue_count >= 2 else 1
        queue_info = vk.VkDeviceQueueCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=queue_family_index,
            queueCount=requested_queue_count,
            pQueuePriorities=[1.0, 0.5][:requested_queue_count],
        )
        # XR_KHR_vulkan_enable2 does not expose xrGetVulkanDeviceExtensionsKHR.
        # Device extensions are selected from the application's Vulkan resource
        # requirements and validated against the runtime-selected physical device.
        external_extensions = VulkanExportableImage.required_device_extensions()
        available_extensions = {
            _decode_name(item.extensionName)
            for item in vk.vkEnumerateDeviceExtensionProperties(vk_physical_device, None)
        }
        missing_extensions = [
            name for name in external_extensions if name not in available_extensions
        ]
        if missing_extensions:
            raise OpenXrVulkanUnavailableError(
                "Vulkan external-memory extensions are unavailable: "
                + ", ".join(missing_extensions)
            )
        optional_external_semaphore = (
            VulkanExportableImage.optional_external_semaphore_extensions()
        )
        enabled_optional = (
            optional_external_semaphore
            if optional_external_semaphore
            and all(name in available_extensions for name in optional_external_semaphore)
            else ()
        )
        device_extensions = tuple(
            dict.fromkeys((*external_extensions, *enabled_optional))
        )
        device_create_info = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            pNext=timeline_features,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[queue_info],
            enabledExtensionCount=len(device_extensions),
            ppEnabledExtensionNames=list(device_extensions),
        )
        xr_device, vulkan_result = xr.create_vulkan_device_khr(
            self.instance,
            xr.VulkanDeviceCreateInfoKHR(
                system_id=self.system_id,
                pfn_get_instance_proc_addr=self._vk_get_instance_proc_addr,
                vulkan_physical_device=xr_physical_device,
                vulkan_create_info=_cffi_struct_pointer(
                    vk, device_create_info, platform.VkDeviceCreateInfo
                ),
            ),
        )
        _check_vulkan_result(vulkan_result, "xrCreateVulkanDeviceKHR")
        vk_device = _ctypes_handle_to_cffi(vk, "VkDevice", xr_device)
        self._provisional_vk_device = vk_device
        frame_context_count = int(
            _env_number("D2S_OPENXR_VULKAN_FRAME_CONTEXTS", 9, minimum=3.0)
        )
        self.vulkan = VulkanContext.adopt(
            instance=vk_instance,
            physical_device=vk_physical_device,
            device=vk_device,
            queue_family_index=queue_family_index,
            owns_instance=True,
            owns_device=True,
            timeline_semaphore_enabled=True,
            synchronization2_enabled=synchronization2_enabled,
            compute_queue_index=1 if requested_queue_count >= 2 else 0,
            # Projection may submit screen, Glow and controller-depth work in
            # one XR frame. Keep enough command slots for all three swapchain
            # images so a later pass does not immediately wait on an earlier
            # pass from the preceding frame.
            frame_context_count=frame_context_count,
        )
        print(
            "[OpenXRViewer] Vulkan queue topology: "
            f"graphics=family{queue_family_index}/queue0 "
            f"glow_compute=family{queue_family_index}/queue"
            f"{1 if requested_queue_count >= 2 else 0} "
            f"async={requested_queue_count >= 2} "
            f"frame_contexts={frame_context_count}",
            flush=True,
        )
        self._provisional_vk_device = None
        self._provisional_vk_instance = None
        self._graphics_binding = xr.GraphicsBindingVulkan2KHR(
            instance=xr_instance,
            physical_device=xr_physical_device,
            device=xr_device,
            queue_family_index=queue_family_index,
            queue_index=0,
        )
        # AMD ROCm: pre-create and warm the torch glow source before the session
        # presents, so the first live glow frame does not pay the ~85ms torch
        # kernel JIT + first-transition cost. Opt-in via D2S_ROCm_TORCH_GLOW;
        # the default glow path on ROCm is the stable cpu_fallback.
        self._prewarmed_glow_backend = None
        if os.environ.get("D2S_ROCm_TORCH_GLOW"):
            try:
                import torch

                if getattr(torch.version, "hip", None):
                    from stereo_runtime.rocm_torch_glow_source import (
                        RocmTorchGlowSource,
                    )

                    prewarm_backend = RocmTorchGlowSource(self.vulkan)
                    warm = torch.zeros(
                        (1, 3, 2160, 3840), dtype=torch.float32, device="cuda"
                    )
                    prewarm_backend.submit(
                        warm,
                        mode="glow",
                        screen_light_only=False,
                        temporal_smoothing_seconds=1.0,
                    )
                    prewarm_backend.release_frame(0)
                    self._prewarmed_glow_backend = prewarm_backend
                    print(
                        "[OpenXRViewer] ROCm torch glow pre-warmed",
                        flush=True,
                    )
            except Exception as exc:
                self._prewarmed_glow_backend = None
                print(
                    "[OpenXRViewer] ROCm torch glow pre-warm skipped: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

    def _create_session_and_swapchains(self) -> None:
        xr = self.xr
        vk = self.vulkan.vk
        self._view_configuration_type = xr.ViewConfigurationType.PRIMARY_STEREO
        self._environment_blend_mode = xr.EnvironmentBlendMode.OPAQUE
        self.session = xr.create_session(
            self.instance,
            xr.SessionCreateInfo(
                system_id=self.system_id,
                next=ctypes.cast(
                    ctypes.pointer(self._graphics_binding), ctypes.c_void_p
                ),
            ),
        )
        available_spaces = xr.enumerate_reference_spaces(self.session)
        self._reference_space_type = (
            xr.ReferenceSpaceType.STAGE
            if xr.ReferenceSpaceType.STAGE in available_spaces
            else xr.ReferenceSpaceType.LOCAL
        )
        self.reference_space = xr.create_reference_space(
            self.session,
            xr.ReferenceSpaceCreateInfo(
                reference_space_type=self._reference_space_type
            ),
        )
        print(
            f"[OpenXRViewer] Reference space selected: "
            f"{getattr(self._reference_space_type, 'name', self._reference_space_type)}",
            flush=True,
        )
        formats = list(xr.enumerate_swapchain_formats(self.session))
        self.swapchain_format = _select_swapchain_format(
            vk, formats, self.config.swapchain_color_mode
        )
        print(
            "OpenXR swapchain color mode: "
            f"requested={self.config.swapchain_color_mode} "
            f"selected={_vulkan_format_name(vk, self.swapchain_format)} "
            f"format={self.swapchain_format}",
            flush=True,
        )
        view_configs = xr.enumerate_view_configuration_views(
            self.instance, self.system_id, self._view_configuration_type
        )
        if len(view_configs) < 2:
            raise OpenXrVulkanUnavailableError(
                f"PRIMARY_STEREO returned {len(view_configs)} view(s)"
            )

        self._view_configuration_views = tuple(view_configs[:2])
        self._create_projection_swapchains_for_scale(self._openxr_render_scale)

    def _create_projection_swapchains_for_scale(self, render_scale: float) -> None:
        view_configs = self._view_configuration_views
        if len(view_configs) < 2:
            raise OpenXrVulkanUnavailableError(
                "PRIMARY_STEREO view configuration is unavailable"
            )
        eye_extents = []
        for view_config in view_configs[:2]:
            width = _scaled_dimension(
                view_config.recommended_image_rect_width,
                view_config.max_image_rect_width,
                render_scale,
            )
            height = _scaled_dimension(
                view_config.recommended_image_rect_height,
                view_config.max_image_rect_height,
                render_scale,
            )
            eye_extents.append((width, height))
        if (
            self._projection_array_eye_diagnostic
            or self._vulkan_multiview_eye_diagnostic
        ):
            if eye_extents[0] != eye_extents[1]:
                raise OpenXrVulkanUnavailableError(
                    "Projection array diagnostic requires equal eye extents: "
                    f"left={eye_extents[0]} right={eye_extents[1]}"
                )
            width, height = eye_extents[0]
            self.swapchains.append(
                self._create_projection_swapchain(width, height, array_size=2)
            )
            if self._vulkan_multiview_eye_diagnostic:
                self._multiview_active = True
            print(
                "[OpenXRViewer] Layered Projection diagnostic created: "
                f"array_size=2 extent={width}x{height}",
                flush=True,
            )
        else:
            for width, height in eye_extents:
                self.swapchains.append(
                    self._create_projection_swapchain(width, height)
                )
        if self._vulkan_controller_proxy_enabled and len(self.swapchains) == 2:
            try:
                self._vulkan_controller_proxy_swapchains = [
                    self._create_projection_swapchain(width, height)
                    for width, height in eye_extents
                ]
            except Exception:
                for eye in reversed(self._vulkan_controller_proxy_swapchains):
                    self._destroy_projection_swapchain(eye)
                self._vulkan_controller_proxy_swapchains.clear()
                raise

    def _release_projection_render_targets(self) -> None:
        if self.vulkan is None:
            return
        self.vulkan.wait_idle()
        if self.filament_bridge is not None:
            self.filament_bridge.close()
            self.filament_bridge = None
        if self._controller_composition_swapchain is not None:
            self._destroy_projection_swapchain(
                self._controller_composition_swapchain
            )
            self._controller_composition_swapchain = None
        for eye in reversed(self._vulkan_controller_proxy_swapchains):
            self._destroy_projection_swapchain(eye)
        self._vulkan_controller_proxy_swapchains.clear()
        for attachment in self._filament_depth_attachments:
            attachment.close()
        self._filament_depth_attachments.clear()
        self._filament_depth_attachments_bound = False
        for image in self._filament_multiview_hdr_images:
            image.close()
        self._filament_multiview_hdr_images.clear()
        for semaphore in self._filament_multiview_ready_semaphores:
            self.vulkan.vk.vkDestroySemaphore(self.vulkan.device, semaphore, None)
        self._filament_multiview_ready_semaphores.clear()
        self._filament_multiview_slot_timelines.clear()
        self._filament_multiview_current = None
        self._filament_multiview_current_slot = None
        self._filament_multiview_finished_consumed = False
        self._multiview_active = False
        if self._vulkan_projection_screen_pass is not None:
            self._vulkan_projection_screen_pass.close()
            self._vulkan_projection_screen_pass = None
        if self._vulkan_multiview_diagnostic_pass is not None:
            self._vulkan_multiview_diagnostic_pass.close()
            self._vulkan_multiview_diagnostic_pass = None
        for eye in reversed(self.swapchains):
            self._destroy_projection_swapchain(eye)
        self.swapchains.clear()

    def _apply_pending_openxr_render_scale(self) -> None:
        requested = self._pending_openxr_render_scale
        if requested is None:
            return
        requested = max(0.5, min(2.0, float(requested)))
        self._pending_openxr_render_scale = None
        if abs(requested - self._openxr_render_scale) < 1e-6:
            return
        previous = self._openxr_render_scale
        try:
            self._release_projection_render_targets()
            self._create_projection_swapchains_for_scale(requested)
            self._openxr_render_scale = requested
            self._settings_menu_values["openxr_render_scale"] = requested
            self._last_vulkan_projection_composer_status = None
            self._initialize_filament_bridges()
            extents = self._projection_eye_extents()
            print(
                "[OpenXRViewer] Projection render scale rebuilt: "
                f"scale={requested:.2f} extents={extents}",
                flush=True,
            )
        except Exception as exc:
            print(
                "[OpenXRViewer] Projection render scale rebuild failed: "
                f"requested={requested:.2f} {type(exc).__name__}: {exc}",
                flush=True,
            )
            self._release_projection_render_targets()
            self._create_projection_swapchains_for_scale(previous)
            self._openxr_render_scale = previous
            self._settings_menu_values["openxr_render_scale"] = previous
            self._initialize_filament_bridges()

    def _create_projection_swapchain(
        self, width: int, height: int, *, array_size: int = 1,
        sampled: bool = False,
    ) -> _EyeSwapchain:
        xr = self.xr
        usage_flags = (
            xr.SwapchainUsageFlags.COLOR_ATTACHMENT_BIT
            | xr.SwapchainUsageFlags.TRANSFER_DST_BIT
        )
        if sampled:
            usage_flags |= xr.SwapchainUsageFlags.SAMPLED_BIT
        if self._filament_multiview_layer_readback_requested and array_size >= 2:
            usage_flags |= xr.SwapchainUsageFlags.TRANSFER_SRC_BIT
        handle = xr.create_swapchain(
            self.session,
            xr.SwapchainCreateInfo(
                usage_flags=usage_flags,
                format=self.swapchain_format,
                sample_count=1,
                width=width,
                height=height,
                face_count=1,
                array_size=array_size,
                mip_count=1,
            ),
        )
        images = list(
            xr.enumerate_swapchain_images(handle, xr.SwapchainImageVulkan2KHR)
        )
        if not images:
            xr.destroy_swapchain(handle)
            raise OpenXrVulkanUnavailableError(
                "OpenXR runtime returned an empty Vulkan swapchain"
            )
        return _EyeSwapchain(
            handle=handle,
            images=images,
            width=width,
            height=height,
            resources=self._register_swapchain_images(images, width, height),
            array_size=array_size,
        )

    def _prepare_panorama_layer(self) -> Any | None:
        """Upload the selected equirectangular environment once and expose it as a background layer."""
        if (
            not self.config.filament_panorama_path
            or self._panorama_failed
            or not self._openxr_equirect_supported
        ):
            if (
                self.config.filament_panorama_path
                and not self._openxr_equirect_supported
                and not self._panorama_skip_logged
            ):
                print(
                    "[OpenXRViewer] HDR panorama deferred: runtime lacks "
                    "XR_KHR_composition_layer_equirect2; Vulkan projection "
                    "equirect pass required",
                    flush=True,
                )
                self._panorama_skip_logged = True
            return None
        if self._panorama_layer is not None:
            return self._panorama_layer
        path = Path(self.config.filament_panorama_path)
        try:
            import imageio.v2 as imageio
            # Keep startup bounded: the packaged 10k HDRs are reduced before
            # any Vulkan allocation, avoiding a multi-hundred-MB upload on the
            # XR presenter thread.
            raw_image = np.asarray(imageio.imread(path))
            raw_dtype = raw_image.dtype
            is_float_image = raw_image.dtype.kind == "f"
            image = np.asarray(raw_image, dtype=np.float32)
            if image.ndim == 2:
                image = np.repeat(image[..., None], 3, axis=2)
            if image.shape[-1] > 3:
                image = image[..., :3]
            if is_float_image:
                image = np.maximum(image, 0.0)
                image = image / (1.0 + image)
                image = np.power(np.clip(image, 0.0, 1.0), 1.0 / 2.2)
            elif raw_image.dtype.itemsize > 1:
                image = image / float(np.iinfo(raw_image.dtype).max)
            if raw_dtype.kind == "u" and raw_dtype.itemsize == 1:
                image = np.asarray(np.clip(image, 0.0, 255.0), dtype=np.uint8)
            else:
                image = np.asarray(np.clip(image * 255.0, 0.0, 255.0), dtype=np.uint8)
            image = np.concatenate(
                (image, np.full((*image.shape[:2], 1), 255, dtype=np.uint8)), axis=2
            )
            height, width = image.shape[:2]
            if max(width, height) > 1024:
                scale = 1024.0 / max(width, height)
                width = max(1, int(round(width * scale)))
                height = max(1, int(round(height * scale)))
                from PIL import Image
                image = np.asarray(
                    Image.fromarray(image, "RGBA").resize((width, height), Image.Resampling.BILINEAR)
                )
            if self._panorama_layer is not None and self._panorama_size == (width, height):
                return self._panorama_layer
            if self._panorama_swapchain is not None:
                self._destroy_projection_swapchain(self._panorama_swapchain)
            self._panorama_staging = VulkanHostImage(
                self.vulkan, width, height, format=int(self.swapchain_format),
                label="openxr-panorama-hdr",
            )
            self._panorama_staging.upload(image)
            self._panorama_swapchain = self._create_projection_swapchain(width, height)
            with _acquired_swapchain_image(self.xr, self._panorama_swapchain) as index:
                upload_timeline = self.vulkan.copy_image(
                    self._panorama_staging.resource,
                    self._panorama_swapchain.resources[index],
                )
                self.vulkan.wait_for_timeline(upload_timeline)
            self._panorama_size = (width, height)
            sub_image = self.xr.SwapchainSubImage(
                swapchain=self._panorama_swapchain.handle,
                image_rect=self.xr.Rect2Di(
                    offset=self.xr.Offset2Di(x=0, y=0),
                    extent=self.xr.Extent2Di(width=width, height=height),
                ),
                image_array_index=0,
            )
            if self._openxr_equirect_supported:
                self._panorama_layer = self.xr.CompositionLayerEquirect2KHR(
                    space=self.reference_space, sub_image=sub_image,
                    pose=self.xr.Posef(), radius=0.0,
                    central_horizontal_angle=float(math.tau),
                    upper_vertical_angle=float(math.pi * 0.5),
                    lower_vertical_angle=float(-math.pi * 0.5),
                )
            else:
                # A flat Quad is not a valid panorama fallback: it is not
                # head-locked, can occlude the SBS screen, and caused expensive
                # per-session uploads without producing a visible HDR room.
                # Leave the projection path untouched until the Vulkan
                # equirect pass is available.
                self._panorama_failed = True
                if not self._panorama_skip_logged:
                    print(
                        "[OpenXRViewer] HDR panorama deferred: runtime lacks "
                        "XR_KHR_composition_layer_equirect2; Vulkan projection "
                        "equirect pass required",
                        flush=True,
                    )
                    self._panorama_skip_logged = True
                return None
            print(
                f"[OpenXRViewer] HDR panorama background active: {path} ({width}x{height})",
                flush=True,
            )
            return self._panorama_layer
        except Exception as exc:
            self._panorama_failed = True
            print(f"[OpenXRViewer] HDR panorama background unavailable: {type(exc).__name__}: {exc}", flush=True)
            return None

    def _ensure_vulkan_panorama_source(self):
        """Upload the selected panorama once for the Vulkan Projection background pass."""
        if self._vulkan_panorama_image is not None:
            return self._vulkan_panorama_image.resource
        path_value = self.config.filament_panorama_path
        if not path_value or self.vulkan is None or self._vulkan_projection_screen_pass is None:
            return None
        try:
            import imageio.v2 as imageio
            raw = np.asarray(imageio.imread(path_value))
            if raw.ndim == 2:
                raw = np.repeat(raw[..., None], 3, axis=2)
            raw = raw[..., :3]
            if raw.dtype.kind == "f":
                raw = np.clip(raw / (1.0 + np.maximum(raw, 0.0)), 0.0, 1.0) * 255.0
            elif raw.dtype.itemsize > 1:
                raw = raw.astype(np.float32) / float(np.iinfo(raw.dtype).max) * 255.0
            rgba = np.concatenate((np.asarray(np.clip(raw, 0, 255), dtype=np.uint8),
                                   np.full((*raw.shape[:2], 1), 255, dtype=np.uint8)), axis=2)
            h, w = rgba.shape[:2]
            device_limit = int(
                self.vulkan.vk.vkGetPhysicalDeviceProperties(
                    self.vulkan.physical_device
                ).limits.maxImageDimension2D
            )
            if w > device_limit or h > device_limit:
                raise RuntimeError(
                    "panorama source exceeds Vulkan maxImageDimension2D: "
                    f"source={w}x{h} device_limit={device_limit}"
                )
            fmt = int(self.swapchain_format)
            self._vulkan_panorama_staging = VulkanHostImage(self.vulkan, w, h, format=fmt, label="panorama-staging")
            self._vulkan_panorama_staging.upload(rgba)
            self._vulkan_panorama_image = VulkanTransientImage(self.vulkan, w, h, format=fmt, label="panorama-source")
            timeline = self.vulkan.copy_image(self._vulkan_panorama_staging.resource, self._vulkan_panorama_image.resource)
            self.vulkan.wait_for_timeline(timeline)
            print(
                "[OpenXRViewer] Vulkan HDR panorama source uploaded: "
                f"source={w}x{h} uploaded={w}x{h} scale=original",
                flush=True,
            )
            return self._vulkan_panorama_image.resource
        except Exception as exc:
            self._panorama_failed = True
            print(f"[OpenXRViewer] Vulkan HDR panorama upload failed: {type(exc).__name__}: {exc}", flush=True)
            return None

    def _panorama_push_constants(self, view: Any) -> bytes:
        # Pass the OpenXR orientation and FOV directly. This avoids all CPU /
        # GLSL matrix-major and projection-sign conventions: the vertex shader
        # builds a view ray and rotates it into the stable reference space.
        orientation = np.asarray((
            float(view.pose.orientation.x),
            float(view.pose.orientation.y),
            float(view.pose.orientation.z),
            float(view.pose.orientation.w),
        ), dtype=np.float64)
        length = float(np.linalg.norm(orientation))
        if not math.isfinite(length) or length <= 1e-8:
            orientation = np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
        else:
            orientation /= length
        initial = getattr(self, "_panorama_initial_orientation", None)
        if initial is None:
            self._panorama_initial_orientation = orientation.copy()
            self._panorama_initial_center_uv = self._panorama_center_uv(
                view.fov, orientation
            )
            initial_uv = self._panorama_initial_center_uv
            print(
                "[OpenXRViewer] Vulkan panorama initial tracking: "
                f"q=({orientation[0]:+.4f},{orientation[1]:+.4f},"
                f"{orientation[2]:+.4f},{orientation[3]:+.4f}) "
                f"center_uv=({initial_uv[0]:.4f},{initial_uv[1]:.4f})",
                flush=True,
            )
        elif not getattr(self, "_panorama_live_rotation_logged", False):
            dot = min(1.0, max(-1.0, abs(float(np.dot(initial, orientation)))))
            delta_degrees = math.degrees(2.0 * math.acos(dot))
            if delta_degrees >= 3.0:
                self._panorama_live_rotation_logged = True
                center_uv = self._panorama_center_uv(view.fov, orientation)
                initial_uv = self._panorama_initial_center_uv
                delta_u = ((center_uv[0] - initial_uv[0] + 0.5) % 1.0) - 0.5
                print(
                    "[OpenXRViewer] Vulkan panorama live head rotation: "
                    f"delta={delta_degrees:.1f}deg "
                    f"q=({orientation[0]:+.4f},{orientation[1]:+.4f},"
                    f"{orientation[2]:+.4f},{orientation[3]:+.4f}) "
                    f"center_uv=({center_uv[0]:.4f},{center_uv[1]:.4f}) "
                    f"uv_delta=({delta_u:+.4f},"
                    f"{center_uv[1] - initial_uv[1]:+.4f})",
                    flush=True,
                )
        values = np.asarray((
            math.tan(float(view.fov.angle_left)),
            math.tan(float(view.fov.angle_right)),
            math.tan(float(view.fov.angle_down)),
            math.tan(float(view.fov.angle_up)),
            *orientation,
        ), dtype="<f4")
        if not getattr(self, "_panorama_rotation_only_logged", False):
            self._panorama_rotation_only_logged = True
            print(
                "[OpenXRViewer] Vulkan panorama mode: world-locked "
                "direct-fov + OpenXR-orientation",
                flush=True,
            )
        return values.tobytes()

    @staticmethod
    def _panorama_center_uv(fov: Any, orientation: np.ndarray) -> tuple[float, float]:
        view_direction = np.asarray((
            0.5 * (
                math.tan(float(fov.angle_left))
                + math.tan(float(fov.angle_right))
            ),
            0.5 * (
                math.tan(float(fov.angle_down))
                + math.tan(float(fov.angle_up))
            ),
            -1.0,
        ), dtype=np.float64)
        x, y, z, w = (float(value) for value in orientation)
        q = np.asarray((x, y, z), dtype=np.float64)
        world_direction = view_direction + 2.0 * np.cross(
            q, np.cross(q, view_direction) + w * view_direction
        )
        world_direction /= max(float(np.linalg.norm(world_direction)), 1e-8)
        u = math.atan2(float(world_direction[0]), -float(world_direction[2]))
        u = (u / math.tau + 0.5) % 1.0
        v = 0.5 - math.asin(min(1.0, max(-1.0, float(world_direction[1])))) / math.pi
        return float(u), float(v)

    def _destroy_projection_swapchain(self, eye: _EyeSwapchain) -> None:
        for resource in reversed(eye.resources):
            self.vulkan.unregister_external_image(resource)
        self.xr.destroy_swapchain(eye.handle)

    def _projection_eye_extents(self) -> tuple[tuple[int, int], ...]:
        if len(self.swapchains) == 1 and self.swapchains[0].array_size >= 2:
            extent = (self.swapchains[0].width, self.swapchains[0].height)
            return (extent, extent)
        return tuple((eye.width, eye.height) for eye in self.swapchains[:2])

    def _register_swapchain_images(
        self, images: list[Any], width: int, height: int,
        format_value: int | None = None,
    ) -> list[VulkanImageResource]:
        resources: list[VulkanImageResource] = []
        try:
            for index, item in enumerate(images):
                image = self.vulkan.image_handle_from_address(
                    _ctypes_handle_address(item.image)
                )
                resource = VulkanImageResource(
                    context=self.vulkan,
                    image=image,
                    view=None,
                    width=width,
                    height=height,
                    format=int(format_value if format_value is not None else self.swapchain_format),
                    layout=self.vulkan.vk.VK_IMAGE_LAYOUT_UNDEFINED,
                    access_mask=0,
                    stage_mask=0,
                    queue_family_index=self.vulkan.queue_family_index,
                    external=True,
                    label=f"openxr-swapchain-{index}",
                )
                self.vulkan.register_external_image(resource)
                resources.append(resource)
        except Exception:
            for resource in reversed(resources):
                try:
                    self.vulkan.unregister_external_image(resource)
                except Exception:
                    pass
            raise
        return resources

    def submit_output(self, frame: VulkanStereoOutputFrame) -> None:
        """Queue the newest Vulkan left/right frame for the next XR frame."""

        if not self._accept_output or not self.session_running:
            raise RuntimeError("OpenXR presenter is waiting for headset rendering")

        if not isinstance(frame.left_eye, VulkanImageResource) or not isinstance(
            frame.right_eye, VulkanImageResource
        ):
            raise TypeError("OpenXR Vulkan output requires VulkanImageResource eyes")
        if frame.left_eye.context is not self.vulkan or frame.right_eye.context is not self.vulkan:
            raise ValueError("OpenXR output images belong to a different Vulkan context")
        if (
            self._presenter_thread_id is not None
            and threading.get_ident() != self._presenter_thread_id
        ):
            self._enqueue_presenter_command("submit_output", frame)
            return
        self._submit_output_on_presenter(frame)

    def submit_runtime_result(self, runtime_result: Any, timestamp: float) -> None:
        """Marshal raw inference output to the Presenter-owned Vulkan path."""

        if not self._accept_output or not self.session_running:
            return
        payload = (runtime_result, float(timestamp))
        if (
            self._presenter_thread_id is not None
            and threading.get_ident() != self._presenter_thread_id
        ):
            self._enqueue_presenter_command("submit_runtime_result", payload)
            return
        self._submit_runtime_result_on_presenter(*payload)

    def _submit_runtime_result_on_presenter(
        self, runtime_result: Any, timestamp: float
    ) -> None:
        """Convert and publish inference output while owning the Vulkan context."""

        debug_info = dict(getattr(runtime_result, "debug_info", None) or {})
        requested_backend = (
            "vulkan_zero_copy"
            if getattr(runtime_result, "vulkan_compute_request", None) is not None
            else (
                "vulkan_host"
                if str(debug_info.get("stereo_compute_backend", "")).strip().lower()
                == "vulkan"
                else None
            )
        )
        if (
            self._output_adapter is not None
            and requested_backend is not None
            and getattr(self._output_adapter, "backend_name", None) != requested_backend
        ):
            close = getattr(self._output_adapter, "close", None)
            if callable(close):
                close()
            self._output_adapter = None
        if self._output_adapter is None:
            from app_runtime.gpu_producer import create_gpu_producer_adapter

            try:
                self._output_adapter = create_gpu_producer_adapter(
                    self,
                    backend=requested_backend,
                )
                self._output_adapter_error = None
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                if message != self._output_adapter_error:
                    print(
                        f"[OpenXRViewer] GPU producer adapter unavailable: {message}; "
                        "waiting for a compatible GPU interop adapter",
                        flush=True,
                    )
                    self._output_adapter_error = message
                return
        try:
            conversion_started = time.perf_counter()
            left_eye = getattr(runtime_result, "left_eye", None)
            right_eye = getattr(runtime_result, "right_eye", None)
            if not isinstance(left_eye, VulkanImageResource) or not isinstance(
                right_eye, VulkanImageResource
            ):
                frame = self._output_adapter.convert(
                    runtime_result,
                    frame_id=self._next_output_frame_id,
                    timestamp=timestamp,
                )
            else:
                debug_info = dict(getattr(runtime_result, "debug_info", None) or {})
                frame = VulkanStereoOutputFrame(
                    frame_id=self._next_output_frame_id,
                    timestamp=timestamp,
                    left_eye=left_eye,
                    right_eye=right_eye,
                    sbs=getattr(runtime_result, "sbs", None),
                    ready_timeline=getattr(runtime_result, "ready_timeline", None),
                    metadata=debug_info,
                    color_space=str(debug_info.get("output_color_space", "srgb")),
                    image_origin=str(debug_info.get("output_image_origin", "top_left")),
                )
            callback = self._on_breakdown_add_time
            if callback is not None:
                callback(
                    "openxr_vulkan_output_convert",
                    max(0.0, time.perf_counter() - conversion_started),
                )
                metadata = dict(getattr(frame, "metadata", None) or {})
                callback(
                    "openxr_vulkan_input_slot_wait",
                    max(
                        0.0,
                        float(metadata.get("vulkan_input_slot_wait_ms", 0.0))
                        / 1000.0,
                    ),
                )
                callback(
                    "openxr_vulkan_input_upload",
                    max(
                        0.0,
                        float(metadata.get("vulkan_input_upload_ms", 0.0))
                        / 1000.0,
                    ),
                )
            self._next_output_frame_id += 1
        except Exception as exc:
            print(
                f"[OpenXRViewer] Runtime output conversion failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return
        self._submit_output_on_presenter(frame)

    def _submit_output_on_presenter(self, frame: VulkanStereoOutputFrame) -> None:
        with self._output_lock:
            previous = self._pending_output
            self._pending_output = frame
        if previous is not None and previous is not frame:
            self._release_output_frame(previous)

    def _enqueue_presenter_command(self, kind: str, payload: Any) -> None:
        command = (str(kind), payload)
        while True:
            try:
                self._presenter_commands.put_nowait(command)
                return
            except queue.Full:
                try:
                    old_kind, old_payload = self._presenter_commands.get_nowait()
                except queue.Empty:
                    continue
                if old_kind == "submit_output":
                    self._release_output_frame(old_payload)

    def _drain_presenter_commands(self) -> None:
        latest_runtime_result = None
        while True:
            try:
                kind, payload = self._presenter_commands.get_nowait()
            except queue.Empty:
                break
            if kind == "submit_output":
                self._submit_output_on_presenter(payload)
            elif kind == "submit_runtime_result":
                # Convert at most one raw result per XR tick. Processing a
                # burst here can exhaust the output ring before the previous
                # frame reaches commit/release, causing the owner thread to
                # wait on its own slot lease indefinitely.
                latest_runtime_result = payload
        if latest_runtime_result is not None:
            self._submit_runtime_result_on_presenter(*latest_runtime_result)

    def _clear_presenter_commands(self) -> None:
        while True:
            try:
                kind, payload = self._presenter_commands.get_nowait()
            except queue.Empty:
                return
            if kind == "submit_output":
                self._release_output_frame(payload)
            elif kind == "submit_runtime_result":
                continue

    @staticmethod
    def _release_output_frame(frame: VulkanStereoOutputFrame | None) -> None:
        if frame is None:
            return
        metadata = frame.metadata or {}
        if metadata.get("_vulkan_release_attempted"):
            return
        # A frame can appear in displayed/rendering/pending bookkeeping while
        # unwinding an exception.  Its callbacks own idempotent CPU cleanup,
        # but Vulkan submissions must never be attempted twice—especially
        # after VK_ERROR_DEVICE_LOST made every handle terminal.
        metadata["_vulkan_release_attempted"] = True
        consumer_release = metadata.get("_vulkan_source_consumer_release")
        consumer_semaphores = metadata.get(
            "_vulkan_consumer_release_semaphores"
        )
        consumer_timeline = metadata.get("_vulkan_consumer_release_timeline")
        try:
            if callable(consumer_release) and consumer_timeline is not None:
                consumer_release(
                    frame.frame_id,
                    wait_for_timeline=int(consumer_timeline),
                )
            elif callable(consumer_release) and consumer_semaphores is not None:
                consumer_release(frame.frame_id, tuple(consumer_semaphores))
            else:
                callback = metadata.get("_vulkan_output_release")
                if callable(callback):
                    fallback_timeline = metadata.get("_vulkan_fallback_copy_timeline")
                    if fallback_timeline is not None:
                        try:
                            callback(
                                frame.frame_id,
                                wait_for_timeline=int(fallback_timeline),
                            )
                        except TypeError:
                            callback(frame.frame_id)
                    else:
                        callback(frame.frame_id)
        finally:
            glow_release = metadata.get("_vulkan_glow_release")
            if callable(glow_release):
                glow_release(frame.frame_id)

    def release_displayed_output_for_reuse(self, slot_index: int) -> bool:
        """Release a displayed source slot before a producer ring wrap blocks."""

        with self._output_lock:
            displayed = self._displayed_output
            if displayed is None:
                return False
            metadata = displayed.metadata or {}
            if metadata.get("vulkan_output_ring_slot") != int(slot_index):
                return False
            self._displayed_output = None
        self._release_output_frame(displayed)
        return True

    def _drop_output_frames(self) -> None:
        with self._output_lock:
            pending = self._pending_output
            displayed = self._displayed_output
            rendering = self._rendering_output
            self._pending_output = None
            self._displayed_output = None
            self._rendering_output = None
        self._release_output_frame(pending)
        if displayed is not pending and displayed is not rendering:
            self._release_output_frame(displayed)
        if rendering is not pending and rendering is not displayed:
            self._release_output_frame(rendering)

    def _commit_output_frame(self, frame: VulkanStereoOutputFrame) -> None:
        with self._output_lock:
            previous = self._displayed_output
            if self._pending_output is frame:
                self._pending_output = None
            if self._rendering_output is frame:
                self._rendering_output = None
            self._displayed_output = frame
        if previous is not None and previous is not frame:
            release_started = time.perf_counter()
            self._release_output_frame(previous)
            if self._on_breakdown_add_time is not None:
                self._on_breakdown_add_time(
                    "openxr_output_release",
                    time.perf_counter() - release_started,
                )

    def _abort_output_frame(self, frame: VulkanStereoOutputFrame) -> None:
        with self._output_lock:
            if self._rendering_output is frame:
                self._rendering_output = None
            if self._pending_output is frame:
                self._pending_output = None
        self._release_output_frame(frame)

    def _initialize_filament_bridges(self) -> None:
        if self._vulkan_controller_proxy_enabled and not self.config.filament_glb_path:
            print(
                "[OpenXRViewer] Vulkan controller proxy active: "
                "controller GLB and unused Filament engine are bypassed",
                flush=True,
            )
            return
        if self._vulkan_controller_proxy_enabled:
            print(
                "[OpenXRViewer] Vulkan controller proxy active: "
                "controller GLB bypassed; Filament environment remains enabled",
                flush=True,
            )
        if self._vulkan_multiview_eye_diagnostic:
            print(
                "[OpenXRViewer] Pure Vulkan multiview diagnostic: "
                "Filament initialization bypassed",
                flush=True,
            )
            return
        bridge_path = self.config.filament_bridge_path or os.environ.get(
            "D2S_FILAMENT_BRIDGE"
        )
        if not bridge_path:
            return

        from .filament_vulkan_bridge import FilamentVulkanBridge

        bridge = FilamentVulkanBridge(bridge_path)
        file_reader, asset_reads = self._start_filament_file_reads()
        try:
            bridge.create(
                instance=self.vulkan.instance,
                physical_device=self.vulkan.physical_device,
                device=self.vulkan.device,
                queue_family_index=self.vulkan.queue_family_index,
                queue_index=0,
            )
            # Use the validated layered Filament producer whenever the Vulkan
            # Projection Composer is enabled. The diagnostic launcher only
            # changes whether SBS/Glow are consumed; it must not select a
            # different controller rendering path.
            self._multiview_active = bool(
                self._vulkan_projection_composer_requested
                and self._try_enable_filament_multiview(bridge)
            )
            if (
                not self._multiview_active
            ):
                print(
                    "[OpenXRViewer] Filament multiview unavailable; "
                    "using per-eye swapchains",
                    flush=True,
                )
            elif self._multiview_active:
                mode = (
                    "diagnostic, SBS/Glow disabled"
                    if self._filament_multiview_projection_diagnostic
                    else "formal Projection Composer path"
                )
                print(f"[OpenXRViewer] Filament multiview active: {mode}", flush=True)
            if not self._multiview_active:
                depth_create_abi = bool(
                    getattr(bridge, "depth_swapchain_abi_available", False)
                )
                depth_output_abi = bool(
                    getattr(bridge, "depth_output_abi_available", False)
                )
                depth_query_abi = callable(
                    getattr(bridge, "get_depth_attachment", None)
                )
                print(
                    "[OpenXRViewer] Filament depth capability: "
                    f"create={int(depth_create_abi)} "
                    f"query={int(depth_query_abi)} "
                    f"output={int(depth_output_abi)} "
                    f"requested={int(_env_flag('D2S_FILAMENT_DEPTH_SWAPCHAIN', default=False))}",
                    flush=True,
                )
                use_depth_swapchain = bool(
                    _env_flag("D2S_FILAMENT_DEPTH_SWAPCHAIN", default=False)
                    and depth_create_abi
                    and depth_query_abi
                    and depth_output_abi
                )
                if (
                    getattr(bridge, "depth_swapchain_abi_available", False)
                    and not use_depth_swapchain
                ):
                    print(
                        "[OpenXRViewer] Filament depth swapchain injection disabled: "
                        "set D2S_FILAMENT_DEPTH_SWAPCHAIN=1 to test",
                        flush=True,
                    )
                if use_depth_swapchain:
                    try:
                        self._filament_depth_attachments = [
                            VulkanDepthAttachment(
                                self.vulkan,
                                eye.width,
                                eye.height,
                                label=f"filament-depth-eye{eye_index}",
                            )
                            for eye_index, eye in enumerate(self.swapchains)
                        ]
                    except Exception as exc:
                        for attachment in self._filament_depth_attachments:
                            attachment.close()
                        self._filament_depth_attachments.clear()
                        use_depth_swapchain = False
                        print(
                            "[OpenXRViewer] Filament depth attachments disabled: "
                            f"{type(exc).__name__}: {exc}",
                            flush=True,
                        )
                depth_query_mismatch = False
                for eye_index, eye in enumerate(self.swapchains):
                    if use_depth_swapchain:
                        depth = self._filament_depth_attachments[eye_index]
                        bridge.create_eye_swapchain_with_depth(
                            eye_index,
                            (image.image for image in eye.images),
                            format=self.swapchain_format,
                            width=eye.width,
                            height=eye.height,
                            depth_image=depth.image,
                            depth_format=depth.format,
                        )
                        query_depth = getattr(bridge, "get_depth_attachment", None)
                        if callable(query_depth):
                            native_depth = query_depth(eye_index)
                            expected_image_address = int(
                                depth.vk.ffi.cast("uintptr_t", depth.image)
                            )
                            if (
                                not native_depth
                                or int(native_depth[1]) != int(depth.format)
                            ):
                                depth_query_mismatch = True
                                print(
                                    "[OpenXRViewer] Filament depth attachment query mismatch: "
                                    f"eye={eye_index} native={native_depth} "
                                    f"expected_image=0x{expected_image_address:x} "
                                    f"expected_format={int(depth.format)}; continuing with "
                                    "the created depth swapchain",
                                    flush=True,
                                )
                    else:
                        bridge.create_eye_swapchain(
                            eye_index,
                            (image.image for image in eye.images),
                            format=self.swapchain_format,
                            width=eye.width,
                            height=eye.height,
                        )
                if use_depth_swapchain and depth_query_mismatch:
                    print(
                        "[OpenXRViewer] Filament depth swapchain rejected by native "
                        "image query; falling back to color-only swapchains",
                        flush=True,
                    )
                    for attachment in self._filament_depth_attachments:
                        attachment.close()
                    self._filament_depth_attachments.clear()
                    for eye_index, eye in enumerate(self.swapchains):
                        bridge.create_eye_swapchain(
                            eye_index,
                            (image.image for image in eye.images),
                            format=self.swapchain_format,
                            width=eye.width,
                            height=eye.height,
                        )
                    use_depth_swapchain = False
                self._filament_depth_attachments_bound = bool(use_depth_swapchain)
                if self._filament_depth_attachments_bound:
                    print(
                        "[OpenXRViewer] Filament depth attachments bound: "
                        f"eyes={len(self._filament_depth_attachments)}",
                        flush=True,
                    )
            glb_path = self.config.filament_glb_path
            if glb_path:
                bridge.load_glb(asset_reads["environment"].result())
            if (
                self._controller_brand is not None
                and self._controller_brand.left_glb is not None
                and self._controller_brand.right_glb is not None
                and getattr(bridge, "controller_abi_available", True)
                and hasattr(bridge, "load_controller")
            ):
                bridge.load_controller(0, asset_reads["controller_left"].result())
                bridge.load_controller(1, asset_reads["controller_right"].result())
                self._apply_controller_material_profile(
                    bridge, self._controller_brand
                )
                print(
                    "Filament controllers loaded: "
                    f"brand={self._controller_brand.name} "
                    f"abi={bridge.controller_abi_available} "
                    f"visibility_abi={getattr(bridge, 'controller_visibility_abi_available', False)} "
                    f"laser_abi={getattr(bridge, 'laser_abi_available', False)}",
                    flush=True,
                )
            if (
                not self._vulkan_controller_proxy_enabled
                and self._controller_brand is not None
                and getattr(self._controller_brand, "left_glb", True) is not None
                and getattr(self._controller_brand, "right_glb", True) is not None
                and getattr(bridge, "controller_guide_abi_available", False)
                and hasattr(bridge, "set_controller_guide_texture")
            ):
                if self._controller_callout_rgba is None:
                    self._controller_callout_rgba = build_controller_callout_rgba(lang="CN")
                bridge.set_controller_guide_texture(self._controller_callout_rgba)
                print(
                    "Filament controller guide loaded: projection_layer=True",
                    flush=True,
                )
            self._apply_filament_scene_exposure_to_bridge(bridge)
            bridge.set_skybox_brightness(self._filament_skybox_brightness)
            self._apply_filament_bridge_lighting(bridge)
            self.filament_bridge = bridge
        except Exception:
            bridge.close()
            self.filament_bridge = None
            raise
        finally:
            file_reader.shutdown(wait=True)

    def _report_projection_composer_boundary(self) -> None:
        print(
            "[OpenXRViewer] Vulkan projection composer boundary: "
            f"requested={self._vulkan_projection_composer_requested} "
            "active=False fallback=existing_projection_path",
            flush=True,
        )

    def _projection_screen_push_constants(
        self, view: Any, sampling_constants: bytes | None = None
    ) -> bytes:
        if self._filament_screen is None:
            raise RuntimeError("Vulkan Projection Composer screen is unavailable")
        position, width, height, rotation = self._filament_screen
        screen_rotation = euler_to_mat4(
            *(math.radians(float(value)) for value in rotation[:3])
        ).astype(np.float32)
        view_projection = (
            _fov_to_proj_mat4_d3d(
                view.fov,
                near=self._profile_near_plane,
                far=self._profile_far_plane,
            )
            @ _pose_to_view_mat4(view.pose)
        ).astype(np.float32)
        # Vulkan's positive-height viewport maps positive NDC Y downward.
        view_projection[1, :] *= -1.0
        if sampling_constants is None:
            sampling_values = np.zeros(4, dtype=np.float32)
        else:
            sampling_values = np.frombuffer(sampling_constants, dtype="<f4")
            if sampling_values.size != 4 or not np.all(np.isfinite(sampling_values)):
                raise ValueError("Vulkan Projection Composer sampling constants are invalid")
        values = np.concatenate((
            view_projection.reshape(-1, order="F"),
            np.asarray((*position, sampling_values[0]), dtype=np.float32),
            np.asarray((*screen_rotation[:3, 0], sampling_values[1]), dtype=np.float32),
            np.asarray((*screen_rotation[:3, 1], sampling_values[2]), dtype=np.float32),
            np.asarray((
                float(width) * 0.5,
                float(height) * 0.5,
                self._effective_screen_curve_half_angle(),
                sampling_values[3],
            ), dtype=np.float32),
        )).astype("<f4", copy=False)
        if values.size != 32 or not np.all(np.isfinite(values)):
            raise ValueError("Vulkan Projection Composer screen transform is invalid")
        return values.tobytes()

    def _projection_screen_sampling_constants(self, source: Any, target: Any) -> bytes:
        width = int(getattr(source, "width", 0))
        height = int(getattr(source, "height", 0))
        if width <= 0 or height <= 0:
            raise ValueError("Vulkan Projection Composer source size is invalid")
        target_width = int(getattr(target, "width", 0))
        target_height = int(getattr(target, "height", 0))
        if target_width <= 0 or target_height <= 0:
            raise ValueError("Vulkan Projection Composer target size is invalid")
        # The quality chain runs before world-space projection. The final
        # screen pass always samples the completed quality mip texture.
        values = np.asarray(
            (
                1.0 / float(width),
                1.0 / float(height),
                1.0,
                0.0,
            ),
            dtype="<f4",
        )
        return values.tobytes()

    def _projection_glow_state(self) -> tuple[int, bytes] | None:
        """Encode the legacy Filament Glow state for Vulkan shaders."""
        mode = self._normalize_filament_glow_mode(self._filament_glow_mode)
        if (
            mode == "off"
            or not self._filament_glow_environment_enabled
            or self._passthrough_backdrop
            or self._filament_screen is None
        ):
            return None
        mode_value = {
            "glow": 1,
            "veil": 2,
            "surround": 3,
        }.get(mode)
        if mode_value is None:
            return None
        glow_multiplier = max(0.0, float(self._filament_glow_intensity_multiplier))
        shell_multiplier = max(
            0.0, float(self._filament_glow_shell_intensity_multiplier)
        )
        if (mode_value == 3 and shell_multiplier <= 0.0) or (
            mode_value != 3 and glow_multiplier <= 0.0
        ):
            return None
        screen_center, screen_width, screen_height, _rotation = self._filament_screen
        head = np.asarray(
            self._head_position_w
            if self._head_position_w is not None
            else (0.0, 0.0, 0.0),
            dtype=np.float64,
        )
        screen_long = max(float(screen_width), float(screen_height), 2.4)
        distance = max(
            float(np.linalg.norm(head - np.asarray(screen_center, dtype=np.float64))),
            0.5,
        )
        glow_range = (
            max(float(self._filament_glow_width), 0.75)
            * (screen_long / 2.4)
            * (distance / 2.0)
            * 20.0
        )
        glow_size = (
            float(screen_width) + glow_range * 2.0,
            float(screen_height) + glow_range * 2.0,
        )
        values = np.asarray(
            (
                float(head[0]), float(head[1]), float(head[2]), float(mode_value),
                max(0.0, float(self._filament_glow_intensity)),
                max(0.0, float(self._filament_glow_width)),
                glow_multiplier,
                shell_multiplier,
                float(glow_size[0]),
                float(glow_size[1]),
                float(screen_width) * 0.5,
                float(screen_height) * 0.5,
                max(0.0, float(self._veil_intensity)),
                min(1.0, max(0.0, float(self._veil_alpha))),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                max(0.0, float(self._filament_glow_shell_radius)),
                max(0.0, float(self._filament_glow_shell_height)),
                screen_long,
                distance,
            ),
            dtype="<f4",
        )
        return int(mode_value), values.tobytes()

    def _projection_laser_params(self, hand: int) -> bytes | None:
        """Pack the legacy controller laser transform for Vulkan overlay draw."""
        laser_matrix = self._projection_laser_model(hand)
        if laser_matrix is None:
            return None
        values = np.zeros(20, dtype=np.float32)
        values[:16] = laser_matrix.reshape(-1, order="F")
        values[16] = float(math.fmod(self._frame_now, 1024.0))
        return values.astype("<f4", copy=False).tobytes()

    def _projection_laser_model(self, hand: int) -> np.ndarray | None:
        """Build the legacy beam transform without depending on Filament."""
        hand = int(hand)
        if hand not in (0, 1):
            return None
        grip_matrix = self._grip_mat_l if hand == 0 else self._grip_mat_r
        aim_matrix = self._aim_mat_l if hand == 0 else self._aim_mat_r
        last_move = self._laser_last_move_l if hand == 0 else self._laser_last_move_r
        if (
            grip_matrix is None
            or aim_matrix is None
            or self._frame_now - float(last_move) > self._LASER_HIDE_AFTER
        ):
            return None
        origin, direction = self._controller_interaction_ray(hand)
        if origin is None or direction is None:
            return None
        right_axis = aim_matrix[:3, 0].astype(np.float64)
        right_axis /= max(float(np.linalg.norm(right_axis)), 1e-8)
        beam_origin = origin.astype(np.float64) + direction * 0.11
        normal_axis = np.cross(right_axis, direction)
        normal_axis /= max(float(np.linalg.norm(normal_axis)), 1e-8)
        right_axis = np.cross(direction, normal_axis)
        right_axis /= max(float(np.linalg.norm(right_axis)), 1e-8)
        laser_matrix = np.eye(4, dtype=np.float32)
        laser_matrix[:3, 0] = (right_axis * 0.006).astype(np.float32)
        laser_matrix[:3, 1] = (direction * 0.4).astype(np.float32)
        laser_matrix[:3, 2] = (normal_axis * 0.006).astype(np.float32)
        laser_matrix[:3, 3] = beam_origin.astype(np.float32)
        return laser_matrix

    def _projection_controller_proxy_params(self) -> bytes | None:
        """Pack both profile-calibrated OpenXR grip poses for the Vulkan proxy."""
        if not self._vulkan_controller_proxy_enabled:
            return None
        offset = np.eye(4, dtype=np.float32)
        offset[:3, 3] = np.asarray(
            self._controller_calibration_offset, dtype=np.float32
        )
        rotation = euler_to_mat4(
            0.0, math.radians(self._controller_calibration_rotation_deg), 0.0
        ).astype(np.float32)
        values = np.zeros(72, dtype=np.float32)
        visible_count = 0
        for hand, grip_matrix in enumerate((self._grip_mat_l, self._grip_mat_r)):
            last_move = (
                self._laser_last_move_l if hand == 0 else self._laser_last_move_r
            )
            if (
                grip_matrix is None
                or self._frame_now - float(last_move) > self._LASER_HIDE_AFTER
            ):
                continue
            matrix = np.asarray(grip_matrix, dtype=np.float32)
            if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
                continue
            matrix = matrix @ rotation @ offset
            values[hand * 16 : (hand + 1) * 16] = matrix.reshape(-1, order="F")
            values[64 + hand] = 1.0
            visible_count += 1
            laser_matrix = self._projection_laser_model(hand)
            if laser_matrix is not None:
                start = 32 + hand * 16
                values[start : start + 16] = laser_matrix.reshape(-1, order="F")
                values[69 + hand] = 1.0
        if visible_count == 0:
            return None
        values[68] = float(math.fmod(self._frame_now, 1024.0))
        return values.astype("<f4", copy=False).tobytes()

    def _apply_vulkan_projection_sampling(
        self,
        frame: VulkanStereoOutputFrame,
        *,
        quality_chain_enabled: bool = True,
    ) -> None:
        screen_pass = self._vulkan_projection_screen_pass
        if screen_pass is None:
            return
        metadata = frame.metadata or {}
        try:
            if not quality_chain_enabled:
                screen_pass.set_sampling_config(
                    min_lod=0.0,
                    max_lod=0.0,
                    mip_lod_bias=0.0,
                    rcas_sharpness=0.0,
                )
                return
            screen_pass.set_sampling_config(
                min_lod=metadata.get("vulkan_projection_min_lod", 0.0),
                max_lod=metadata.get("vulkan_projection_max_lod", 0.35),
                mip_lod_bias=metadata.get("vulkan_projection_mip_lod_bias", -0.35),
                rcas_sharpness=metadata.get("vulkan_projection_rcas_sharpness", 0.5),
            )
        except (TypeError, ValueError):
            return

    def _resolve_filament_multiview_hdr(
        self,
        acquired_images: list[tuple[_EyeSwapchain, int]],
        wait_semaphores: list[Any] | tuple[Any, ...],
        *,
        projection_draws: list[dict[str, Any]] | None = None,
        load_target: bool = False,
    ) -> int:
        if (
            self.vulkan is None
            or self._filament_multiview_current is None
            or len(acquired_images) not in {1, 2}
        ):
            raise RuntimeError("Filament multiview HDR resolve has no valid targets")
        layered = len(acquired_images) == 1 and acquired_images[0][0].array_size >= 2
        target_format = int(acquired_images[0][0].resources[0].format)
        if self._vulkan_projection_screen_pass is None:
            self._vulkan_projection_screen_pass = VulkanProjectionScreenPass(
                self.vulkan,
                target_format,
                enable_panorama=bool(self.config.filament_panorama_path),
            )
        if projection_draws is None:
            projection_draws = []
            for eye_index in range(2):
                target_eye, image_index = (
                    acquired_images[0] if layered else acquired_images[eye_index]
                )
                projection_draws.append({
                    "target": target_eye.resources[image_index],
                    "array_layer": eye_index if layered else 0,
                    "frame_slot": int(self.frame_count) % 3,
                })
        timeline = self._vulkan_projection_screen_pass.submit_filament_hdr(
            projection_draws,
            tuple(self._filament_multiview_current.layer_resources),
            exposure_ev=float(self._filament_scene_exposure),
            wait_semaphores=wait_semaphores,
            load_target=bool(load_target),
        )
        self._filament_multiview_finished_consumed = True
        if self._filament_multiview_current_slot is not None:
            self._filament_multiview_slot_timelines[
                self._filament_multiview_current_slot
            ] = int(timeline)
        if not getattr(self, "_filament_multiview_hdr_resolve_logged", False):
            self._filament_multiview_hdr_resolve_logged = True
            print(
                "[OpenXRViewer] Filament multiview HDR resolve active: "
                f"exposure_ev={self._filament_scene_exposure:.2f} "
                "tone_mapper=LINEAR targets=per_eye_projection",
                flush=True,
            )
        return int(timeline)

    def _render_vulkan_projection_composer(
        self,
        frame: VulkanStereoOutputFrame,
        acquired_images: list[tuple[_EyeSwapchain, int]],
        views: list[Any],
        filament_wait_semaphores: list[Any] | tuple[Any, ...] = (),
        filament_hdr_sources: list[Any] | tuple[Any, ...] = (),
    ) -> int:
        if self.vulkan is None or len(acquired_images) not in {1, 2}:
            raise RuntimeError("Vulkan Projection Composer has no valid targets")
        diagnostic = _env_flag("D2S_VULKAN_PROJECTION_COMPOSER_EYE_DIAGNOSTIC")
        layered = len(acquired_images) == 1 and acquired_images[0][0].array_size >= 2
        if diagnostic:
            target_eye, image_index = acquired_images[0]
            colors = ((1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0))
            timelines = []
            for eye_index, color in enumerate(colors):
                eye_target, target_index = (
                    (target_eye, image_index)
                    if layered
                    else acquired_images[eye_index]
                )
                timelines.append(
                    self.vulkan.clear_color_image(
                        eye_target.resources[target_index].image,
                        color,
                        base_array_layer=eye_index if layered else 0,
                    )
                )
            print(
                "[OpenXRViewer] Vulkan projection composer eye diagnostic: "
                f"left=red layer={0 if layered else 'eye0'} "
                f"right=green layer={1 if layered else 'eye1'}",
                flush=True,
            )
            self._vulkan_projection_composer_frame_id = int(frame.frame_id)
            self._vulkan_projection_composer_active = True
            return max(timelines)
        if len(views) < 2:
            raise RuntimeError("Vulkan Projection Composer requires two views")
        prepare_source = (frame.metadata or {}).get(
            "_vulkan_source_prepare_for_sampling"
        )
        if not callable(prepare_source):
            raise RuntimeError(
                "Vulkan Projection Composer source preparation is unavailable"
            )
        source_inputs = (frame.left_eye, frame.right_eye)
        status = (
            layered,
            int(source_inputs[0].width),
            int(source_inputs[0].height),
            int(acquired_images[0][0].width),
            int(acquired_images[0][0].height),
            bool(self._screen_curved),
        )
        if status != self._last_vulkan_projection_composer_status:
            self._last_vulkan_projection_composer_status = status
            print(
                "[OpenXRViewer] Vulkan projection composer active: "
                f"mode=graphics_triangle_strip layered={layered} "
                f"source={status[1]}x{status[2]} "
                f"target={status[3]}x{status[4]} curved={status[5]}",
                flush=True,
            )
        target_format = int(acquired_images[0][0].resources[0].format)
        if self._vulkan_projection_screen_pass is None:
            self._vulkan_projection_screen_pass = VulkanProjectionScreenPass(
                self.vulkan,
                target_format,
                enable_panorama=bool(self.config.filament_panorama_path),
            )
        depth_sampling_timeline = 0
        depth_sampling_active = False
        self._vulkan_projection_laser_depth_available = False
        # Filament remains the owner of controller laser rendering.  The
        # Vulkan depth-laser experiment is intentionally disabled while the
        # producer output contract is being migrated.
        if self._vulkan_projection_laser_depth_available and (
            not self._multiview_active
            and self._filament_depth_attachments_bound
            and len(filament_wait_semaphores) == 2
        ):
            try:
                depth_sampling_active = bool(
                    self._vulkan_projection_screen_pass.set_laser_depth_attachments(
                        self._filament_depth_attachments
                    )
                )
                prepare_depth = getattr(
                    self.vulkan, "prepare_external_depth_for_sampling", None
                )
                if depth_sampling_active and callable(prepare_depth):
                    for eye_index, semaphore in enumerate(filament_wait_semaphores):
                        depth_sampling_timeline = max(
                            int(depth_sampling_timeline),
                            int(
                                prepare_depth(
                                    self._filament_depth_attachments[eye_index].resource,
                                    wait_semaphore=semaphore,
                                )
                            ),
                        )
                    self._vulkan_projection_laser_depth_available = True
                    filament_wait_semaphores = ()
                else:
                    depth_sampling_active = False
            except Exception as exc:
                depth_sampling_active = False
                self._vulkan_projection_laser_depth_available = False
                print(
                    "[OpenXRViewer] Vulkan projection depth sampling skipped: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
        self._apply_vulkan_projection_sampling(
            frame,
            quality_chain_enabled=(
                self._vulkan_projection_quality_chain_requested
                and not bool((frame.metadata or {}).get("output_quality_applied", 0))
            ),
        )
        plan = self._active_screen_sampling_plan
        use_quality_mip = bool(
            self._vulkan_projection_quality_chain_requested
            and plan is not None
            and not bool((frame.metadata or {}).get("output_quality_applied", 0))
        )
        projection_draws = []
        glow_source = (frame.metadata or {}).get("glow_vulkan_image")
        glow_state = self._projection_glow_state()
        controller_proxy_params = self._projection_controller_proxy_params()
        for eye_index, source in enumerate(source_inputs):
            source_prepare_started = time.perf_counter()
            wait_semaphore = prepare_source(frame.frame_id, eye_index)
            if self._on_breakdown_add_time is not None:
                self._on_breakdown_add_time(
                    "openxr_vulkan_composer_source_prepare",
                    time.perf_counter() - source_prepare_started,
                )
            draw_prepare_started = time.perf_counter()
            target_eye, image_index = (
                acquired_images[0] if layered else acquired_images[eye_index]
            )
            sampling_constants = self._projection_screen_sampling_constants(
                source, target_eye.resources[image_index]
            )
            screen_push_constants = self._projection_screen_push_constants(
                views[eye_index], sampling_constants
            )
            projection_draw = {
                "source": source,
                "target": target_eye.resources[image_index],
                "array_layer": eye_index if layered else 0,
                "eye_index": eye_index,
                "frame_slot": int(self.frame_count) % 3,
                "push_constants": screen_push_constants,
                "clear_color": self.config.clear_color,
                "wait_semaphore": wait_semaphore,
            }
            laser_params = (
                self._projection_laser_params(eye_index)
                if self._vulkan_projection_laser_depth_available
                else None
            )
            if laser_params is not None:
                projection_draw["laser_params"] = laser_params
                projection_draw["laser_push_constants"] = screen_push_constants[:64]
            if controller_proxy_params is not None:
                projection_draw["controller_proxy_params"] = controller_proxy_params
                projection_draw["controller_proxy_push_constants"] = (
                    screen_push_constants[:64]
                )
            if glow_source is not None and glow_state is not None:
                projection_draw["glow_source"] = glow_source
                projection_draw["glow_push_constants"] = screen_push_constants
                projection_draw["glow_mode"] = glow_state[0]
                projection_draw["glow_params"] = glow_state[1]
                projection_draw["glow_curved"] = bool(self._screen_curved)
            projection_draws.append(projection_draw)
            if self._on_breakdown_add_time is not None:
                self._on_breakdown_add_time(
                    "openxr_vulkan_composer_draw_prepare",
                    time.perf_counter() - draw_prepare_started,
                )
        panorama_source = self._ensure_vulkan_panorama_source()
        panorama_timeline = 0
        if panorama_source is not None:
            for eye_index, draw in enumerate(projection_draws):
                draw["panorama_push_constants"] = self._panorama_push_constants(views[eye_index])
                draw["panorama_source_ready"] = True
            panorama_timeline = self._vulkan_projection_screen_pass.submit_panorama(
                projection_draws, panorama_source, wait_for_timeline=0
            )
        filament_hdr_timeline = 0
        defer_filament_resolve = bool(
            filament_hdr_sources
            and all("glow_source" in draw for draw in projection_draws)
            and not self._filament_projection_only
        )
        if filament_hdr_sources and not defer_filament_resolve:
            # A live GLB -> panorama switch keeps the Filament engine for
            # controllers.  Preserve the panorama that was rendered into the
            # Projection target before resolving that transparent foreground;
            # the clear resolve is only valid when Filament is the first pass.
            filament_hdr_timeline = self._resolve_filament_multiview_hdr(
                acquired_images,
                filament_wait_semaphores,
                projection_draws=projection_draws,
                load_target=bool(panorama_timeline),
            )
            filament_wait_semaphores = ()
        if self._filament_projection_only and filament_hdr_timeline:
            if not getattr(self, "_filament_projection_only_logged", False):
                self._filament_projection_only_logged = True
                print(
                    "[OpenXRViewer] Filament projection-only diagnostic active: "
                    "SBS/Glow Vulkan overlays skipped",
                    flush=True,
                )
            self._vulkan_projection_composer_frame_id = int(frame.frame_id)
            self._vulkan_projection_composer_active = True
            return int(filament_hdr_timeline)
        laser_prepare_status = (
            "owner=filament "
            f"vulkan_depth={int(self._vulkan_projection_laser_depth_available)} "
            f"draws={sum(1 for draw in projection_draws if 'laser_params' in draw)} "
            f"aim_l={int(self._aim_mat_l is not None)} aim_r={int(self._aim_mat_r is not None)} "
            f"grip_l={int(self._grip_mat_l is not None)} grip_r={int(self._grip_mat_r is not None)}"
        )
        if laser_prepare_status != self._last_vulkan_projection_laser_prepare_status:
            self._last_vulkan_projection_laser_prepare_status = laser_prepare_status
            print(
                "[OpenXRViewer] Vulkan projection laser prepare: "
                f"{laser_prepare_status}",
                flush=True,
            )
        submit_started = time.perf_counter()
        timeline = None
        surround_timeline = 0
        surround_requested = bool(
            glow_state is not None
            and glow_state[0] == 3
            and all("glow_source" in draw for draw in projection_draws)
        )
        if os.environ.get("D2S_OPENXR_DISABLE_GLOW_DRAW"):
            # Diagnostic: isolate whether the composer's glow draw pass (sampling
            # the glow image with the glow fragment pipeline) breaks the Virtual
            # Desktop session; the screen-light reduction is kept.
            surround_requested = False
        surround_active = False
        if surround_requested:
            # Match the legacy Filament scene split: Surround is room/background
            # emission and must be occluded by the opaque SBS surface.  The
            # Glow and Veil are foreground effects drawn afterward.
            try:
                surround_timeline = (
                    self._vulkan_projection_screen_pass.submit_stereo_glow(
                        projection_draws,
                        wait_for_timeline=int(filament_hdr_timeline),
                        clear_target=not bool(filament_hdr_timeline),
                    )
                )
                surround_active = True
                self._last_vulkan_projection_glow_error = None
                if self._on_breakdown_inc is not None:
                    self._on_breakdown_inc("openxr_vulkan_composer_glow", 1)
                if os.environ.get("D2S_GLOW_DIAGNOSTIC"):
                    print(
                        "[OpenXRViewer] Glow draw: surround pass executed "
                        f"mode={glow_state[0]} draws={len(projection_draws)}",
                        flush=True,
                    )
            except Exception as exc:
                glow_error = (type(exc).__name__, str(exc))
                if glow_error != self._last_vulkan_projection_glow_error:
                    self._last_vulkan_projection_glow_error = glow_error
                    print(
                        "[OpenXRViewer] Vulkan projection Surround skipped: "
                        f"{glow_error[0]}: {glow_error[1]}",
                        flush=True,
                    )
        if (
            not surround_requested
            and not os.environ.get("D2S_OPENXR_DISABLE_GLOW_DRAW")
            and all("glow_source" in draw for draw in projection_draws)
        ):
            # Formal ordering is controller/environment -> Glow -> SBS screen.
            # Drawing Glow after the screen lets the translucent effect cover
            # opaque controller pixels in the Filament multiview source.
            try:
                surround_timeline = self._vulkan_projection_screen_pass.submit_stereo_glow(
                    projection_draws,
                    wait_for_timeline=int(filament_hdr_timeline),
                )
                surround_active = True
                self._last_vulkan_projection_glow_error = None
                if self._on_breakdown_inc is not None:
                    self._on_breakdown_inc("openxr_vulkan_composer_glow", 1)
                if os.environ.get("D2S_GLOW_DIAGNOSTIC"):
                    print(
                        "[OpenXRViewer] Glow draw: glow/veil pass executed "
                        f"mode={glow_state[0]} draws={len(projection_draws)}",
                        flush=True,
                    )
            except Exception as exc:
                glow_error = (type(exc).__name__, str(exc))
                if glow_error != self._last_vulkan_projection_glow_error:
                    self._last_vulkan_projection_glow_error = glow_error
                    print(
                        "[OpenXRViewer] Vulkan projection Glow skipped: "
                        f"{glow_error[0]}: {glow_error[1]}",
                        flush=True,
                    )
        if use_quality_mip:
            try:
                timeline = self._vulkan_projection_screen_pass.try_submit_stereo_quality_mip(
                    projection_draws,
                    mode=plan.mode,
                    filter_scale=plan.filter_scale,
                    upscale_scale=plan.upscale_scale,
                    load_target=bool(
                        surround_active
                        or filament_hdr_timeline
                        or filament_wait_semaphores
                        or depth_sampling_timeline
                        or panorama_timeline
                    ),
                        wait_for_timeline=max(
                            int(surround_timeline),
                        int(depth_sampling_timeline), int(panorama_timeline),
                    ),
                    extra_wait_semaphores=filament_wait_semaphores,
                )
            except Exception as exc:
                print(
                    "[OpenXRViewer] Vulkan projection quality chain skipped: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
        if self._on_breakdown_inc is not None and use_quality_mip:
            if timeline is not None:
                self._on_breakdown_inc("openxr_vulkan_composer_quality", 1)
            else:
                self._on_breakdown_inc("openxr_vulkan_composer_quality_skip", 1)
        if timeline is None:
            timeline = self._vulkan_projection_screen_pass.submit_stereo(
                projection_draws,
                load_target=bool(
                    surround_active
                    or filament_hdr_timeline
                    or filament_wait_semaphores
                    or depth_sampling_timeline
                    or panorama_timeline
                ),
                extra_wait_semaphores=filament_wait_semaphores,
                wait_for_timeline=max(
                    int(surround_timeline),
                    int(depth_sampling_timeline), int(panorama_timeline),
                ),
            )
        if defer_filament_resolve:
            # The formal foreground order is Glow -> SBS -> Filament
            # controller/environment -> laser. Use the LOAD overlay pass so
            # HDR resolve does not clear the already-composed screen.
            filament_hdr_timeline = self._resolve_filament_multiview_hdr(
                acquired_images,
                filament_wait_semaphores,
                projection_draws=projection_draws,
                load_target=True,
            )
            filament_wait_semaphores = ()
        if any("laser_params" in draw for draw in projection_draws):
            try:
                timeline = self._vulkan_projection_screen_pass.submit_stereo_laser(
                    projection_draws,
                    wait_for_timeline=int(timeline),
                )
                self._last_vulkan_projection_laser_error = None
                if self._on_breakdown_inc is not None:
                    self._on_breakdown_inc("openxr_vulkan_composer_laser", 1)
            except Exception as exc:
                laser_error = (type(exc).__name__, str(exc))
                if laser_error != self._last_vulkan_projection_laser_error:
                    self._last_vulkan_projection_laser_error = laser_error
                    print(
                        "[OpenXRViewer] Vulkan projection laser skipped: "
                        f"{laser_error[0]}: {laser_error[1]}",
                        flush=True,
                    )
        elif (
            not self._vulkan_controller_proxy_enabled
            and (
                self.filament_bridge is None
                or not getattr(self.filament_bridge, "laser_abi_available", False)
            )
        ):
            if self._last_vulkan_projection_laser_depth_status != "unavailable":
                self._last_vulkan_projection_laser_depth_status = "unavailable"
                print(
                    "[OpenXRViewer] Controller laser unavailable: "
                    "Filament laser ABI is not available",
                    flush=True,
                )
        if any("controller_proxy_params" in draw for draw in projection_draws):
            timeline = self._vulkan_projection_screen_pass.submit_stereo_controller_proxy(
                projection_draws,
                wait_for_timeline=int(timeline),
            )
        if depth_sampling_active:
            release_depth = getattr(
                self.vulkan, "release_external_depth_from_sampling", None
            )
            if callable(release_depth):
                for attachment in self._filament_depth_attachments:
                    timeline = max(
                        int(timeline),
                        int(
                            release_depth(
                                attachment.resource,
                                wait_for_timeline=int(timeline),
                            )
                        ),
                    )
        if self._on_breakdown_add_time is not None:
            self._on_breakdown_add_time(
                "openxr_vulkan_composer_submit",
                time.perf_counter() - submit_started,
            )
            submit_profile = self._vulkan_projection_screen_pass.last_submit_profile
            for stage, metric in (
                ("fence_wait", "openxr_vulkan_composer_fence_wait"),
                ("record", "openxr_vulkan_composer_record"),
                ("queue_submit", "openxr_vulkan_composer_queue_submit"),
            ):
                if stage in submit_profile:
                    self._on_breakdown_add_time(metric, submit_profile[stage])
            if self._on_breakdown_inc is not None:
                for stage, metric in (
                    ("mip_template_hit", "openxr_vulkan_mip_template_hit"),
                    ("mip_template_new", "openxr_vulkan_mip_template_new"),
                    (
                        "render_pass_template_hit",
                        "openxr_vulkan_render_pass_template_hit",
                    ),
                    (
                        "render_pass_template_new",
                        "openxr_vulkan_render_pass_template_new",
                    ),
                    (
                        "image_barrier_template_hit",
                        "openxr_vulkan_image_barrier_template_hit",
                    ),
                    (
                        "image_barrier_template_new",
                        "openxr_vulkan_image_barrier_template_new",
                    ),
                ):
                    if stage in submit_profile:
                        self._on_breakdown_inc(metric, submit_profile[stage])
        self._vulkan_projection_composer_frame_id = int(frame.frame_id)
        self._vulkan_projection_composer_active = True
        return int(timeline)

    def _try_enable_filament_multiview(self, bridge: Any) -> bool:
        if not (
            getattr(bridge, "multiview_abi_available", False)
            and getattr(bridge, "multiview_supported", False)
            and getattr(bridge, "multiview_depth_swapchain_abi_available", False)
            and getattr(bridge, "image_ready_semaphore_abi_available", False)
            and getattr(bridge, "finished_drawing_semaphore_abi_available", False)
            and getattr(
                bridge, "controller_composition_layer_abi_available", False
            )
            and self.vulkan is not None
            and len(self.swapchains) == 2
            and self._vulkan_projection_composer_requested
        ):
            return False
        left, right = self.swapchains
        if (left.width, left.height) != (right.width, right.height):
            print(
                "[OpenXRViewer] Filament multiview unavailable: eye extents differ",
                flush=True,
            )
            return False
        vk = self.vulkan.vk
        hdr_format = int(vk.VK_FORMAT_R16G16B16A16_SFLOAT)
        hdr_images: list[VulkanTransientImage] = []
        depth_attachment: VulkanDepthAttachment | None = None
        ready_semaphores: list[Any] = []
        controller_swapchain: _EyeSwapchain | None = None
        try:
            depth_attachment = VulkanDepthAttachment(
                self.vulkan,
                left.width,
                left.height,
                label="filament-multiview-depth",
                array_layers=2,
            )
            for slot in range(3):
                hdr_images.append(
                    VulkanTransientImage(
                        self.vulkan,
                        left.width,
                        left.height,
                        format=hdr_format,
                        array_layers=2,
                        label=f"filament-multiview-hdr-slot{slot}",
                    )
                )
                ready_semaphores.append(
                    vk.vkCreateSemaphore(
                        self.vulkan.device,
                        vk.VkSemaphoreCreateInfo(
                            sType=vk.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                        ),
                        None,
                    )
                )
            bridge.create_stereo_swapchain_with_depth(
                (image.image for image in hdr_images),
                format=hdr_format,
                width=left.width,
                height=left.height,
                depth_image=depth_attachment.image,
                depth_format=depth_attachment.format,
            )
            controller_swapchain = self._create_projection_swapchain(
                left.width, left.height, array_size=2
            )
            bridge.create_controller_overlay_stereo_swapchain(
                (image.image for image in controller_swapchain.images),
                format=int(self.swapchain_format),
                width=left.width,
                height=left.height,
                depth_image=depth_attachment.image,
                depth_format=depth_attachment.format,
            )
        except Exception as exc:
            if controller_swapchain is not None:
                self._destroy_projection_swapchain(controller_swapchain)
            if depth_attachment is not None:
                depth_attachment.close()
            for image in hdr_images:
                image.close()
            for semaphore in ready_semaphores:
                try:
                    vk.vkDestroySemaphore(self.vulkan.device, semaphore, None)
                except Exception:
                    pass
            print(
                "[OpenXRViewer] Filament multiview fallback: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return False
        self._controller_composition_swapchain = controller_swapchain
        self._filament_depth_attachments = [depth_attachment]
        self._filament_depth_attachments_bound = True
        self._filament_multiview_hdr_images = hdr_images
        self._filament_multiview_ready_semaphores = ready_semaphores
        self._filament_multiview_slot_timelines = [0] * len(hdr_images)
        self._filament_multiview_slot = 0
        self._filament_multiview_current = None
        self._filament_multiview_current_slot = None
        self._filament_multiview_finished_consumed = False
        print(
            "[OpenXRViewer] Filament projection path: "
            f"multiview_hdr slots={len(hdr_images)} array_size=2 "
            f"format=R16G16B16A16_SFLOAT extent={left.width}x{left.height} "
            f"depth_layers=2 depth_format={depth_attachment.format}",
            flush=True,
        )
        return True

    def _initialize_msdf_text_atlas(self) -> None:
        """Load the shared atlas for the MSDF-to-Quad OSD path."""
        try:
            atlas = MsdfFontAtlas()
        except Exception as exc:
            self._msdf_font_atlas = None
            print(
                "[OpenXRViewer] MSDF atlas unavailable; "
                f"using legacy Quad text ({type(exc).__name__}: {exc})",
                flush=True,
            )
            return
        self._msdf_font_atlas = atlas
        print(
            "[OpenXRViewer] MSDF atlas loaded for Quad OSD: "
            f"pages={len(atlas.pages)} glyphs={len(atlas.glyphs)} "
            f"distance_range={atlas.distance_range:g}",
            flush=True,
        )

    def _apply_screen_sampling_policy(
        self,
        output_frame: VulkanStereoOutputFrame | None,
    ) -> ScreenSamplingPlan | None:
        """Apply the GUI-headset/input-resolution matrix to the screen filter."""
        if output_frame is None or self._filament_screen is None:
            return None
        metadata = dict(output_frame.metadata or {})

        def metadata_size(value: Any) -> tuple[int, int] | None:
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                try:
                    width, height = int(value[0]), int(value[1])
                except (TypeError, ValueError):
                    return None
                return (width, height) if width > 0 and height > 0 else None
            text = str(value or "").strip().lower()
            if "x" not in text:
                return None
            left, right = text.split("x", 1)
            try:
                width, height = int(left), int(right)
            except ValueError:
                return None
            return (width, height) if width > 0 and height > 0 else None

        # The sampling policy must describe the texture that the projection
        # composer actually samples. capture_size can differ after the GUI 4K
        # render-scale policy and is retained only as source diagnostics.
        source = output_frame.left_eye
        eye_size = (
            int(getattr(source, "width", 0)),
            int(getattr(source, "height", 0)),
        )
        source_size = eye_size if eye_size[0] > 0 and eye_size[1] > 0 else None
        if source_size is None:
            source_size = next(
                (
                    metadata_size(metadata.get(key))
                    for key in ("render_size", "source_size", "input_size", "capture_size")
                    if metadata_size(metadata.get(key)) is not None
                ),
                None,
            )
        if source_size is None:
            return None
        try:
            plan = build_screen_sampling_plan(
                source_size[0],
                source_size[1],
                self._headset_preset.resolution_tier_k,
            )
        except (TypeError, ValueError):
            return None
        if int(metadata.get("output_quality_applied", 0) or 0):
            # Eye images already passed through the shared Lanczos2/EASU/RCAS
            # stage before output routing. Projection only owns MIP generation
            # and screen composition now; repeating the quality pass would
            # resize and sharpen the frame twice.
            plan = replace(
                plan,
                recommended_headset_tier_k=plan.input_tier_k,
                effective_tier_k=plan.input_tier_k,
                filter_scale=1.0,
                upscale_scale=1.0,
                mode="native_mip",
            )
        status = (
            plan.source_width,
            plan.source_height,
            plan.input_tier_k,
            plan.headset_tier_k,
            plan.recommended_headset_tier_k,
            plan.effective_tier_k,
            round(plan.filter_scale, 4),
            round(plan.upscale_scale, 4),
            plan.mode,
        )
        status_changed = status != self._last_screen_sampling_status
        if status_changed:
            self._last_screen_sampling_status = status
            print(
                "[OpenXRViewer] screen sampling policy "
                f"headset={self._headset_preset.key} "
                f"headset_tier={plan.headset_tier_k}K "
                f"input={plan.source_width}x{plan.source_height} "
                f"input_tier={plan.input_tier_k}K "
                f"recommended={plan.recommended_headset_tier_k}K "
                f"effective={plan.effective_tier_k}K "
                f"filter_scale={plan.filter_scale:.2f} mode={plan.mode} "
                f"upscale_scale={plan.upscale_scale:.2f} "
                "sampling_owner="
                + (
                    "vulkan_projection_composer"
                ),
                flush=True,
            )
        self._active_screen_sampling_plan = plan
        return plan

    def _initialize_msdf_quad_renderer(self) -> None:
        if self.vulkan is None or self._msdf_font_atlas is None:
            return
        try:
            self._vulkan_msdf_quad_renderer = VulkanMsdfQuadRenderer(
                self.vulkan, self._msdf_font_atlas
            )
            print(
                "[OpenXRViewer] Vulkan MSDF Quad renderer active: "
                "atlas_gpu=True output=storage_image",
                flush=True,
            )
        except Exception as exc:
            self._vulkan_msdf_quad_renderer = None
            print(
                "[OpenXRViewer] Vulkan MSDF Quad renderer unavailable; "
                f"using CPU MSDF compatibility path ({type(exc).__name__}: {exc})",
                flush=True,
            )

    def _submit_msdf_text_runs(self, runs: list[dict[str, Any]]) -> bool:
        """Submit merged MSDF runs; return false when the legacy path is needed."""
        bridge = self.filament_bridge
        atlas = self._msdf_font_atlas
        if bridge is None or atlas is None or not getattr(
            bridge, "text_overlay_abi_available", False
        ):
            return False
        grouped: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
        try:
            for run in runs:
                geometry = atlas.build_geometry(**run)
                for page, buffers in geometry.items():
                    grouped.setdefault(page, []).append(buffers)
            for page in range(len(atlas.pages)):
                buffers = grouped.get(page, ())
                if buffers:
                    vertices = np.ascontiguousarray(
                        np.concatenate([item[0] for item in buffers], axis=0),
                        dtype=np.float32,
                    )
                    index_parts = []
                    vertex_offset = 0
                    for item_vertices, item_indices in buffers:
                        index_parts.append(item_indices.astype(np.uint32) + vertex_offset)
                        vertex_offset += int(item_vertices.shape[0])
                    indices = np.ascontiguousarray(
                        np.concatenate(index_parts).astype(np.uint16), dtype=np.uint16
                    )
                    bridge.set_text_overlay_page(page, vertices, indices, visible=True)
                else:
                    bridge.set_text_overlay_page(
                        page,
                        np.zeros((0, 9), dtype=np.float32),
                        np.zeros(0, dtype=np.uint16),
                        visible=False,
                    )
        except Exception as exc:
            self._msdf_font_atlas = None
            print(
                "[OpenXRViewer] MSDF text disabled after submit failure; "
                "retaining legacy overlay: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return False
        return True

    def _start_filament_file_reads(
        self,
    ) -> tuple[ThreadPoolExecutor, dict[str, Future[bytes]]]:
        """Read assets off-thread without moving any Filament work off-owner."""
        executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="FilamentFileIO")
        reads: dict[str, Future[bytes]] = {}
        if self.config.filament_glb_path:
            reads["environment"] = executor.submit(
                Path(self.config.filament_glb_path).read_bytes
            )
        if (
            self._controller_brand is not None
            and self._controller_brand.left_glb is not None
            and self._controller_brand.right_glb is not None
        ):
            reads["controller_left"] = executor.submit(
                self._controller_brand.left_glb.read_bytes
            )
            reads["controller_right"] = executor.submit(
                self._controller_brand.right_glb.read_bytes
            )
        return executor, reads

    def _update_filament_controllers(self, bridge: Any) -> None:
        if (
            self._vulkan_controller_proxy_enabled
            or self._controller_brand is None
            or getattr(self._controller_brand, "left_glb", True) is None
            or getattr(self._controller_brand, "right_glb", True) is None
            or not getattr(bridge, "controller_abi_available", True)
            or not hasattr(bridge, "set_controller_pose")
            or not hasattr(bridge, "set_controller_inputs")
        ):
            return
        self._update_filament_controller_guide(bridge)
        offset = np.eye(4, dtype=np.float32)
        offset[:3, 3] = np.asarray(
            self._controller_calibration_offset, dtype=np.float32
        )
        # Controller profiles use the legacy model calibration convention:
        # model_rotation_deg is a rotation around the local X axis.
        rotation = euler_to_mat4(
            0.0, math.radians(self._controller_calibration_rotation_deg), 0.0
        ).astype(np.float32)
        for hand, (grip_matrix, aim_matrix) in enumerate(
            zip((self._grip_mat_l, self._grip_mat_r), (self._aim_mat_l, self._aim_mat_r))
        ):
            last_move = self._laser_last_move_l if hand == 0 else self._laser_last_move_r
            active = (
                grip_matrix is not None
                and self._frame_now - float(last_move) <= self._LASER_HIDE_AFTER
            )
            if getattr(bridge, "controller_visibility_abi_available", False):
                bridge.set_controller_visible(hand, active)
            if not active:
                self._reset_smoothed_ray(hand)
                if getattr(bridge, "laser_abi_available", False):
                    bridge.set_controller_laser(
                        hand, np.eye(4, dtype=np.float32), visible=False
                    )
                continue
            model_matrix = grip_matrix @ rotation @ offset
            bridge.set_controller_pose(hand, model_matrix)
            values = self._controller_input(hand)
            button_mask = 0
            for bit, name in enumerate(
                ("a_button", "b_button", "x_button", "y_button", "menu_button")
            ):
                if values.get(name, 0.0) > 0.5:
                    button_mask |= 1 << bit
            if values.get("stick_click", 0.0) > 0.5:
                button_mask |= 1 << 5
            if max(
                values.get("joystick_touched", 0.0),
                values.get("touchpad_touched", 0.0),
            ) > 0.5:
                # Keep the frozen C ABI: bit 6 carries the shared WebXR touch state.
                button_mask |= 1 << 6
            bridge.set_controller_inputs(
                hand,
                trigger=values.get("trigger", 0.0),
                grip=values.get("grip", 0.0),
                joystick_x=values.get("joystick_x", 0.0),
                joystick_y=values.get("joystick_y", 0.0),
                button_mask=button_mask,
            )
            if getattr(bridge, "laser_abi_available", False) and hasattr(bridge, "set_controller_laser"):
                if aim_matrix is None:
                    bridge.set_controller_laser(
                        hand, np.eye(4, dtype=np.float32), visible=False
                    )
                else:
                    smoothed_origin, direction = self._controller_interaction_ray(hand)
                    if smoothed_origin is None or direction is None:
                        bridge.set_controller_laser(
                            hand, np.eye(4, dtype=np.float32), visible=False
                        )
                        continue
                    right_axis = aim_matrix[:3, 0].astype(np.float64)
                    right_axis /= max(float(np.linalg.norm(right_axis)), 1e-8)
                    # Start the beam just beyond the grip shell.
                    beam_origin = (
                        smoothed_origin.astype(np.float64) + direction * 0.11
                    )
                    normal_axis = np.cross(right_axis, direction)
                    normal_axis /= max(float(np.linalg.norm(normal_axis)), 1e-8)
                    right_axis = np.cross(direction, normal_axis)
                    right_axis /= max(float(np.linalg.norm(right_axis)), 1e-8)
                    laser_matrix = np.eye(4, dtype=np.float32)
                    laser_matrix[:3, 0] = (right_axis * 0.006).astype(np.float32)
                    laser_matrix[:3, 1] = (direction * 0.4).astype(np.float32)
                    laser_matrix[:3, 2] = (normal_axis * 0.006).astype(np.float32)
                    laser_matrix[:3, 3] = beam_origin.astype(np.float32)
                    bridge.set_controller_laser(hand, laser_matrix, visible=True)
    def _update_filament_controller_guide(self, bridge: Any) -> None:
        if (
            getattr(bridge, "controller_guide_abi_available", False)
            and hasattr(bridge, "set_controller_guide")
        ):
            geometry = self._controller_guide_geometry()
            if geometry is None:
                bridge.set_controller_guide(np.eye(4, dtype=np.float32), visible=False)
            else:
                position, size, basis = geometry
                guide_matrix = np.eye(4, dtype=np.float32)
                guide_matrix[:3, 0] = (basis[:, 0] * size[0]).astype(np.float32)
                guide_matrix[:3, 1] = (basis[:, 1] * size[1]).astype(np.float32)
                guide_matrix[:3, 2] = basis[:, 2].astype(np.float32)
                guide_matrix[:3, 3] = np.asarray(position, dtype=np.float32)
                bridge.set_controller_guide(guide_matrix, visible=True)

    def _load_filament_profile(self) -> None:
        profile_path = self.config.filament_profile_path
        if not profile_path:
            return
        with open(profile_path, "r", encoding="utf-8-sig") as handle:
            profile = json.load(handle)
        if not isinstance(profile, dict):
            raise ValueError("Filament profile root must be an object")
        self._filament_profile_data = profile

        presets = profile.get("lighting_presets")
        self._filament_lighting_presets = tuple(
            item for item in presets if isinstance(item, dict)
        ) if isinstance(presets, list) else ()
        try:
            self._filament_lighting_preset_index = int(
                profile.get("lighting_preset_index", 0)
            )
        except (TypeError, ValueError):
            self._filament_lighting_preset_index = 0
        if self._filament_lighting_presets:
            self._filament_lighting_preset_index %= len(
                self._filament_lighting_presets
            )
        self._apply_filament_glow_profile_fields(profile)

        view_pose = profile.get("view_pose", profile.get("camera"))
        view_poses = profile.get("view_poses")
        if isinstance(view_poses, list) and view_poses:
            index = int(profile.get("view_pose_index", 0)) % len(view_poses)
            view_pose = view_poses[index]
            valid_poses = [item for item in view_poses if isinstance(item, dict)]
            named_poses = {}
            for item in valid_poses:
                name = str(item.get("name", "")).strip().lower()
                if "front" in name or "前" in name:
                    named_poses.setdefault("front", item)
                elif "back" in name or "后" in name:
                    named_poses.setdefault("back", item)
                elif "middle" in name or "center" in name or "中" in name:
                    named_poses.setdefault("middle", item)
            self._filament_view_poses = tuple(
                named_poses.get(name, valid_poses[position])
                if position < len(valid_poses) else valid_poses[0]
                for position, name in enumerate(("front", "middle", "back"))
            )
            self._filament_view_pose_index = next(
                (
                    position for position, item
                    in enumerate(self._filament_view_poses)
                    if item is view_pose or item == view_pose
                ),
                min(index, max(0, len(self._filament_view_poses) - 1)),
            )
        if not isinstance(view_pose, dict):
            # Default and panorama environments intentionally have no authored
            # room-space seat. Rebase identity to the initial leveled headset
            # pose, matching the legacy OpenXR default screen contract.
            view_pose = {
                "name": "Default",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "rotation_deg": [0.0, 0.0, 0.0],
            }

        try:
            model_position = profile.get(
                "model_position", profile.get("position", [0.0, 0.0, 0.0])
            )
            if not isinstance(model_position, (list, tuple)) or len(model_position) < 3:
                model_position = [0.0, 0.0, 0.0]
            model_rotation_deg = profile.get("model_rotation_deg", [0.0, 0.0, 0.0])
            if not isinstance(model_rotation_deg, (list, tuple)) or len(model_rotation_deg) < 3:
                model_rotation_deg = [0.0, 0.0, 0.0]
            model_scale = profile.get("model_scale", [1.0, 1.0, 1.0])
            if not isinstance(model_scale, (list, tuple)) or len(model_scale) < 3:
                model_scale = [1.0, 1.0, 1.0]

            world_position_vec = np.asarray(
                [float(view_pose[key]) for key in ("x", "y", "z")],
                dtype=np.float32,
            )
            rotation_deg = view_pose.get("rotation_deg")
            if not isinstance(rotation_deg, (list, tuple)) or len(rotation_deg) < 3:
                rotation_deg = [float(view_pose.get("angle", 0.0)), 0.0, 0.0]
            rotation_rad = [math.radians(float(value)) for value in rotation_deg[:3]]

            pose_space = str(
                view_pose.get(
                    "view_pose_space",
                    view_pose.get("pose_space", profile.get("view_pose_space", "world")),
                )
            ).strip().lower()
            if pose_space in {"scene", "glb", "local"}:
                glb_position = world_position_vec
            else:
                # view_poses are authored in environment world coordinates while
                # the imported GLB and calibrated OpenXR space use GLB-local
                # coordinates. Match the legacy viewer by applying the inverse
                # model transform before rebasing the reference space.
                model_matrix = euler_to_mat4(
                    *(math.radians(float(value)) for value in model_rotation_deg[:3])
                ).astype(np.float32)
                model_matrix[:3, 3] = np.asarray(model_position[:3], dtype=np.float32)
                scale = np.asarray(model_scale[:3], dtype=np.float32)
                model_matrix[:3, :3] = model_matrix[:3, :3] @ np.diag(scale)
                glb_position = (
                    np.linalg.inv(model_matrix)
                    @ np.append(world_position_vec, 1.0)
                )[:3]
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError("Filament profile view pose contains invalid values") from exc

        transform = euler_to_mat4(*rotation_rad).astype(np.float32)
        transform[:3, 3] = np.asarray(glb_position, dtype=np.float32)
        self._profile_head_transform = transform
        self._profile_view_name = str(view_pose.get("name", "profile"))
        self._profile_auto_center_on_screen = bool(
            view_pose.get("auto_center_on_screen", False)
        )
        self._profile_near_plane = max(0.001, float(profile.get("xr_projection_near", 0.05)))
        self._profile_far_plane = max(
            self._profile_near_plane + 1.0,
            float(profile.get("xr_projection_far", 1000.0)),
        )
        self._filament_scene_exposure = float(
            profile.get("preview_exposure", self._filament_scene_exposure)
        )
        self._filament_skybox_brightness = float(
            profile.get("preview_skybox_brightness", self._filament_skybox_brightness)
        )
        ambient_color = profile.get(
            "env_ambient_color", self._filament_ambient_light_color
        )
        if isinstance(ambient_color, (list, tuple)) and len(ambient_color) >= 3:
            self._filament_ambient_light_color = tuple(
                max(0.0, float(value)) for value in ambient_color[:3]
            )
        # Resolve host-authored controller lights; the native bridge owns no
        # color, intensity, or placement defaults.
        fill_color = profile.get(
            "controller_head_light_color",
            profile.get("env_head_light_color", self._filament_fill_light_color),
        )
        fill_direction = self._filament_fill_light_direction
        if isinstance(fill_color, (list, tuple)) and len(fill_color) >= 3:
            self._filament_fill_light_color = tuple(
                float(value) for value in fill_color[:3]
            )
        if isinstance(fill_direction, (list, tuple)) and len(fill_direction) >= 3:
            self._filament_fill_light_direction = tuple(
                float(value) for value in fill_direction[:3]
            )
        for key, attribute in (
            ("controller_ambient_light_color", "_controller_ambient_light_color_override"),
            ("controller_hdr_ambient_light_color", "_controller_hdr_ambient_light_color_override"),
        ):
            value = profile.get(key)
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                setattr(self, attribute, tuple(max(0.0, float(item)) for item in value[:3]))
        self._filament_fill_light_intensity = float(
            profile.get("controller_head_light_intensity", 1.0)
        )
        for key, attribute in (
            ("env_ambient_light_intensity_lux", "_filament_ambient_light_intensity_lux"),
            ("controller_ambient_light_intensity_lux", "_controller_ambient_light_intensity_lux"),
            ("controller_hdr_ambient_light_intensity_lux", "_controller_hdr_ambient_light_intensity_lux"),
            ("controller_light_intensity_candela", "_controller_light_intensity_candela"),
            ("controller_head_light_weight", "_controller_head_light_weight"),
            ("controller_top_light_weight", "_controller_top_light_weight"),
            ("controller_head_light_falloff", "_controller_head_light_falloff"),
            ("controller_top_light_falloff", "_controller_top_light_falloff"),
            ("controller_screen_light_intensity_lux", "_controller_screen_light_intensity_lux"),
            ("controller_screen_light_saturation", "_controller_screen_light_saturation"),
            ("controller_screen_light_max_luminance", "_controller_screen_light_max_luminance"),
            ("controller_screen_light_smoothing_seconds", "_controller_screen_light_smoothing_seconds"),
            ("controller_screen_light_sample_hz", "_controller_screen_light_sample_hz"),
            ("environment_screen_light_intensity_candela", "_environment_screen_light_intensity_candela"),
            ("screen_light_intensity", "_environment_screen_area_light_intensity"),
            ("environment_screen_light_saturation", "_environment_screen_light_saturation"),
            ("environment_screen_light_max_luminance", "_environment_screen_light_max_luminance"),
            ("environment_screen_light_smoothing_seconds", "_environment_screen_light_smoothing_seconds"),
            ("environment_screen_light_sample_hz", "_environment_screen_light_sample_hz"),
            ("environment_screen_light_falloff", "_environment_screen_light_falloff"),
            ("environment_screen_light_offset", "_environment_screen_light_offset"),
            ("glow_sample_hz", "_filament_glow_sample_hz"),
            ("glow_smoothing_seconds", "_filament_glow_smoothing_seconds"),
        ):
            if key in profile:
                minimum = 0.001 if key.endswith("_falloff") else 0.0
                setattr(self, attribute, max(minimum, float(profile[key])))
        for key, attribute in (
            ("controller_top_light_color", "_controller_top_light_color"),
            ("controller_head_light_offset", "_controller_head_light_offset"),
            ("controller_top_light_offset", "_controller_top_light_offset"),
        ):
            value = profile.get(key)
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                setattr(self, attribute, tuple(float(item) for item in value[:3]))
        self._controller_head_light_cast_shadows = bool(profile.get(
            "controller_head_light_cast_shadows",
            self._controller_head_light_cast_shadows,
        ))
        self._controller_top_light_cast_shadows = bool(profile.get(
            "controller_top_light_cast_shadows",
            self._controller_top_light_cast_shadows,
        ))
        self._controller_screen_light_enabled = bool(profile.get(
            "controller_screen_light_enabled",
            self._controller_screen_light_enabled,
        ))
        self._controller_screen_light_cast_shadows = bool(profile.get(
            "controller_screen_light_cast_shadows",
            self._controller_screen_light_cast_shadows,
        ))
        default_environment = bool(
            self.config.filament_profile_path
            and Path(self.config.filament_profile_path).parent.name.strip().lower()
            == "default"
        )
        self._environment_screen_light_enabled = (
            False
            if default_environment
            else bool(profile.get(
                "environment_screen_light_enabled",
                self._environment_screen_light_enabled,
            ))
        )
        self._environment_screen_light_cast_shadows = bool(profile.get(
            "environment_screen_light_cast_shadows",
            self._environment_screen_light_cast_shadows,
        ))
        self._controller_hdr_lighting = bool(
            profile.get("controller_hdr_lighting", False)
        )
        if self._filament_lighting_presets:
            self._apply_filament_lighting_preset(
                self._filament_lighting_presets[
                    self._filament_lighting_preset_index
                ],
                apply_bridge=False,
            )
        if profile.get("glb") or self.config.filament_glb_path:
            # Room exposure is an OpenXR session-only adjustment. Never inherit
            # a previous run or an authored preview EV as the menu starting
            # point; every GLB room starts at the neutral zero tick.
            self._filament_scene_exposure = 0.0
        screen = profile.get("screen")
        self._filament_screen_profile_authored = isinstance(screen, dict)
        if not isinstance(screen, dict):
            default_width = max(0.25, float(self.config.filament_screen_width))
            default_distance = max(0.25, float(self.config.filament_screen_distance))
            screen = {
                "position": [0.0, 0.0, -default_distance],
                "width": default_width,
                "height": default_width * 9.0 / 16.0,
                "rotation_deg": [0.0, 0.0, 0.0],
            }
        if isinstance(screen, dict):
            self._settings_menu_allow_curve = bool(screen.get("allow_curve", True))
            profile_curved = bool(screen.get("curved", False))
            try:
                profile_half_angle = float(
                    screen.get(
                        "curve_half_angle_rad",
                        0.72 if profile_curved else 0.0,
                    )
                )
            except (TypeError, ValueError):
                profile_half_angle = 0.72 if profile_curved else 0.0
            self._screen_curve_half_angle = max(
                0.0,
                min(profile_half_angle, math.pi / 2.0),
            )
            self._screen_curved = (
                self._settings_menu_allow_curve
                and self._screen_curve_half_angle > 1e-6
            )
            if not self._screen_curved:
                self._screen_curve_half_angle = 0.0
            self._screen_initial_curve_half_angle = self._screen_curve_half_angle
            screen_position = screen.get("position", [0.0, 1.2, -2.0])
            rotation = screen.get("rotation_deg", [0.0, 0.0, 0.0])
            if (
                isinstance(screen_position, (list, tuple))
                and len(screen_position) >= 3
                and isinstance(rotation, (list, tuple))
                and len(rotation) >= 3
            ):
                self._filament_screen = (
                    tuple(float(value) for value in screen_position[:3]),
                    float(screen.get("width", 2.4)),
                    float(screen.get(
                        "height",
                        float(screen.get("width", 2.4)) * 9.0 / 16.0,
                    )),
                    tuple(float(value) for value in rotation[:3]),
                )
                self._filament_screen_initial = self._filament_screen
        print(
            f"Loaded Filament profile view: {self._profile_view_name} "
            f"world_position={world_position_vec.tolist()} glb_position={glb_position.tolist()} "
            f"rotation_rad={rotation_rad}",
            flush=True,
        )
        environment_lighting = (
            "hdr_ibl_pending_profile_fallback"
            if self._controller_hdr_lighting
            else "room_profile"
        )
        print(
            "Filament controller lighting: "
            f"environment={environment_lighting} "
            "screen_light=disabled",
            flush=True,
        )

    def _apply_filament_profile(self, views: list[Any]) -> list[Any]:
        # The environment profile is applied once by rebasing the shared
        # OpenXR reference space. Runtime eye views must remain unmodified so
        # the compositor receives the matching headset poses.
        return views

    def _apply_profile_reference_space(self, views: list[Any]) -> bool:
        """Apply the saved seat pose, then close the loop on the measured XR pose."""
        if self._profile_space_applied or self._profile_head_transform is None:
            return False
        if len(views) < 2 or self.xr is None or self.session is None:
            return False
        eye_matrices = [_xr_view_pose_to_model_mat4(view.pose) for view in views[:2]]
        raw_head = eye_matrices[0].copy()
        raw_head[:3, 3] = (eye_matrices[0][:3, 3] + eye_matrices[1][:3, 3]) * 0.5
        # Views are expressed in the currently active child reference space.
        # Recover the head in the original STAGE/LOCAL reference before
        # calculating the replacement space. This state is essential for the
        # feedback pass; treating the second locate result as a base-space pose
        # would mix two coordinate systems and retain the physical head height.
        measured_reference_head = (
            self._profile_space_pose_in_reference.astype(np.float64) @ raw_head
        )
        measured_reference_head = self._level_head_model_mat4(
            measured_reference_head
        )
        preserve_anchor = bool(
            self._profile_space_preserve_anchor
            and self._profile_reference_head_anchor is not None
        )
        if preserve_anchor:
            # A live seat selection is normally made while the user is looking
            # at the settings menu. Reusing that instantaneous gaze as the new
            # calibration origin makes the room counter-rotate away from the
            # menu. Keep the stable anchor captured at startup/recenter and
            # only replace the authored target seat transform.
            reference_head = np.asarray(
                self._profile_reference_head_anchor, dtype=np.float64
            )
        else:
            reference_head = measured_reference_head
            self._profile_reference_head_anchor = reference_head.astype(
                np.float32
            )
        space_pose = reference_head @ np.linalg.inv(self._profile_head_transform)
        try:
            new_space = self.xr.create_reference_space(
                self.session,
                self.xr.ReferenceSpaceCreateInfo(
                    reference_space_type=(
                        self._reference_space_type
                        or self.xr.ReferenceSpaceType.LOCAL
                    ),
                    pose_in_reference_space=mat4_to_xr_posef(space_pose.astype(np.float32)),
                ),
            )
        except Exception as exc:
            print(f"[OpenXRViewer] Failed to apply profile reference space: {exc}", flush=True)
            return False
        old_space = self.reference_space
        self.reference_space = new_space
        # Controller action spaces must use the same calibrated world space.
        self._xr_space = new_space
        self._profile_space_pose_in_reference = space_pose.astype(np.float32)
        if preserve_anchor:
            self._profile_space_calibration_pass = 2
            self._profile_space_applied = True
            self._profile_space_preserve_anchor = False
        else:
            self._profile_space_calibration_pass += 1
            self._profile_space_applied = self._profile_space_calibration_pass >= 2
        self._profile_initial_head = raw_head
        if old_space is not None:
            try:
                self.xr.destroy_space(old_space)
            except Exception:
                pass
        phase = "final" if self._profile_space_applied else "provisional"
        print(
            f"[OpenXRViewer] Applied {phase} profile pose to stable OpenXR "
            f"reference space (pass={self._profile_space_calibration_pass})",
            flush=True,
        )
        return True

    @staticmethod
    def _level_head_model_mat4(head_mat: np.ndarray) -> np.ndarray:
        """Keep position and yaw while preserving a level environment."""
        pos = head_mat[:3, 3].copy()
        forward = -head_mat[:3, 2].astype(np.float32)
        forward[1] = 0.0
        norm = float(np.linalg.norm(forward))
        yaw = 0.0 if norm < 1e-6 else math.atan2(
            -float(forward[0] / norm), -float(forward[2] / norm)
        )
        leveled = euler_to_mat4(yaw, 0.0, 0.0).astype(np.float32)
        leveled[:3, 3] = pos
        return leveled

    def _render_filament_multiview(
        self,
        render_views: list[Any],
        presentation_frame: VulkanStereoOutputFrame | None,
        finished_semaphore_available: bool,
        record_time: Callable[[str, float], None],
    ) -> int | None:
        bridge = self.filament_bridge
        if (
            self.vulkan is None
            or not self._filament_multiview_hdr_images
            or not self._filament_multiview_ready_semaphores
        ):
            raise RuntimeError("Filament multiview HDR targets are unavailable")
        slot = int(self._filament_multiview_slot) % len(
            self._filament_multiview_hdr_images
        )
        hdr_image = self._filament_multiview_hdr_images[slot]
        ready_semaphore = self._filament_multiview_ready_semaphores[slot]
        previous_timeline = int(self._filament_multiview_slot_timelines[slot])
        vk = self.vulkan.vk
        source_state = self.vulkan.image_state(hdr_image.image)

        def prepare_hdr_target(command_buffer: Any) -> None:
            old_layout = int(source_state.layout)
            barrier = vk.VkImageMemoryBarrier(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                srcAccessMask=int(source_state.access_mask),
                dstAccessMask=vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                oldLayout=old_layout,
                newLayout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                image=hdr_image.image,
                subresourceRange=vk.VkImageSubresourceRange(
                    aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel=0,
                    levelCount=1,
                    baseArrayLayer=0,
                    layerCount=2,
                ),
            )
            vk.vkCmdPipelineBarrier(
                command_buffer,
                int(source_state.stage_mask)
                or vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                0,
                0,
                None,
                0,
                None,
                1,
                [barrier],
            )

        self.vulkan.submit_on(
            "graphics",
            prepare_hdr_target,
            wait_for_timeline=previous_timeline,
            signal_semaphore=ready_semaphore,
        )
        self.vulkan.register_image_state(
            hdr_image.image,
            ImageState(
                layout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                access_mask=vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                stage_mask=vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                queue_family_index=self.vulkan.queue_family_index,
            ),
        )
        state_started = time.perf_counter()
        bridge.set_active_eye(0)
        _update_filament_stereo_camera(
            bridge,
            render_views,
            near_plane=self._profile_near_plane,
            far_plane=self._profile_far_plane,
        )
        record_time("openxr_filament_multiview_state", state_started)
        bridge.set_active_eye(0)
        bridge.set_acquired_image(slot)
        bridge.set_image_ready_semaphore(
            _cffi_handle_address(vk, ready_semaphore)
        )
        queue_started = time.perf_counter()
        bridge.begin_frame()
        record_time("openxr_filament_multiview_queue", queue_started)
        finish_started = time.perf_counter()
        bridge.end_frame()
        record_time("openxr_filament_multiview_finish_wait", finish_started)
        self._filament_multiview_current = hdr_image
        self._filament_multiview_current_slot = slot
        self._filament_multiview_finished_consumed = False
        self._filament_multiview_slot = (slot + 1) % len(
            self._filament_multiview_hdr_images
        )
        return (
            bridge.get_finished_drawing_semaphore()
            if finished_semaphore_available
            else None
        )

    def _capture_filament_multiview_layers(
        self,
        acquired_images: list[tuple[_EyeSwapchain, int]],
        wait_for_timeline: int | None,
        wait_semaphore: Any | None = None,
    ) -> int | None:
        """Read back both Filament array layers once to locate stereo routing faults."""
        if (
            not self._filament_multiview_layer_readback_requested
            or self._filament_multiview_layer_readback_done
        ):
            return wait_for_timeline
        self._filament_multiview_layer_readback_done = True
        if len(acquired_images) != 1 or acquired_images[0][0].array_size < 2:
            print(
                "[OpenXRViewer] Filament multiview layer readback skipped: "
                "array_size=2 swapchain unavailable",
                flush=True,
            )
            return wait_for_timeline

        eye, image_index = acquired_images[0]
        source = eye.resources[image_index]
        output_width = min(320, int(source.width))
        output_height = max(
            1, round(int(source.height) * output_width / int(source.width))
        )
        host: VulkanHostReadbackBuffer | None = None
        try:
            vk = self.vulkan.vk
            # The custom external Filament platform deliberately disables the
            # PRESENT transition. Its finished semaphore therefore publishes
            # the layered target in COLOR_ATTACHMENT_OPTIMAL.
            self.vulkan.register_image_state(
                source.image,
                ImageState(
                    layout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    access_mask=vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    stage_mask=vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                    queue_family_index=self.vulkan.queue_family_index,
                ),
            )
            # Use an exact-size copy instead of a GPU blit. Some Windows
            # drivers create the linear host image successfully but do not
            # produce data for an optimal-to-linear scaled blit. Reuse one
            # full-size host allocation and shrink each completed layer on CPU.
            host = VulkanHostReadbackBuffer(
                self.vulkan,
                int(source.width),
                int(source.height),
                label="filament-multiview-readback",
            )
            from PIL import Image

            layers: list[np.ndarray] = []
            for layer in range(2):
                wait_for_timeline = self.vulkan.copy_image_to_host_buffer(
                    source,
                    host,
                    wait_for_timeline=wait_for_timeline,
                    wait_semaphore=wait_semaphore if layer == 0 else None,
                    source_array_layer=layer,
                )
                self.vulkan.wait_for_timeline(int(wait_for_timeline))
                full_rgb = _vulkan_rgba_to_rgb(
                    host.read_rgba(),
                    format_value=int(source.format),
                    vk=vk,
                    image_origin="bottom_left",
                )
                layers.append(
                    np.asarray(
                        Image.fromarray(full_rgb, mode="RGB").resize(
                            (output_width, output_height), Image.Resampling.BILINEAR
                        )
                    )
                )
            output_dir = Path(__file__).resolve().parents[2] / "artifacts"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / (
                "filament_multiview_layers_"
                f"{time.strftime('%Y%m%d_%H%M%S')}.png"
            )
            _write_sbs_capture_png(output_path, layers[0], layers[1])
            statistics = []
            for layer, rgb in enumerate(layers):
                active_count, mean = _active_rgb_mean(rgb)
                statistics.append(
                    f"layer{layer}: active={active_count} "
                    f"mean_rgb=({mean[0]:.1f},{mean[1]:.1f},{mean[2]:.1f})"
                )
            print(
                "[OpenXRViewer] Filament multiview layer readback saved: "
                f"{output_path} source={source.width}x{source.height} "
                f"format={source.format} copy=image-to-buffer | "
                f"{' | '.join(statistics)}",
                flush=True,
            )
        except Exception as exc:
            print(
                "[OpenXRViewer] Filament multiview layer readback failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        finally:
            if host is not None:
                host.close()
        return wait_for_timeline

    def _advance_filament_multiview_layer_readback(self) -> bool:
        """Delay diagnostic capture until Filament has compiled and rendered materials."""
        if (
            not self._filament_multiview_layer_readback_requested
            or self._filament_multiview_layer_readback_done
            or not self._multiview_active
        ):
            return False
        self._filament_multiview_layer_readback_frame += 1
        if self._filament_multiview_layer_readback_frame == 1:
            print(
                "[OpenXRViewer] Filament multiview layer readback deferred: "
                f"capture_frame={self._filament_multiview_layer_readback_delay_frames}",
                flush=True,
            )
        return (
            self._filament_multiview_layer_readback_frame
            >= self._filament_multiview_layer_readback_delay_frames
        )

    def _render_filament_for_projection_composer(
        self,
        render_views: list[Any],
        acquired_images: list[tuple[_EyeSwapchain, int]],
        animation_time: float,
        record_time: Callable[[str, float], None],
    ) -> list[Any]:
        """Render environment/controllers before Composer overlays the SBS image."""
        bridge = self.filament_bridge
        if bridge is None or len(acquired_images) != 2:
            return []
        finished_available = bool(
            getattr(bridge, "finished_drawing_semaphore_abi_available", False)
        )
        semaphores: list[Any] = []
        # Keep the established safe per-eye end-frame behavior until the
        # Composer/Filament shared queue contract has been validated.
        deferred = False
        for eye_index, (eye, image_index) in enumerate(acquired_images):
            state_started = time.perf_counter()
            bridge.set_active_eye(eye_index)
            _update_filament_camera(
                bridge,
                render_views[eye_index],
                near_plane=self._profile_near_plane,
                far_plane=self._profile_far_plane,
            )
            record_time(f"openxr_filament_eye{eye_index}_state", state_started)
            bridge.set_acquired_image(image_index)
            queue_started = time.perf_counter()
            if (
                self._filament_controller_overlay_after_composer
                and bool(getattr(bridge, "controller_overlay_abi_available", False))
                and bool(getattr(bridge, "background_frame_abi_available", False))
            ):
                bridge.begin_background_frame()
            else:
                bridge.begin_frame()
            record_time(f"openxr_filament_eye{eye_index}_queue", queue_started)
            finish_started = time.perf_counter()
            if deferred:
                bridge.end_frame_deferred()
            else:
                bridge.end_frame()
            record_time(
                f"openxr_filament_eye{eye_index}_finish_wait", finish_started
            )
            if finished_available:
                semaphore = bridge.get_finished_drawing_semaphore()
                if semaphore is not None:
                    semaphores.append(semaphore)
            if eye_index < len(self._filament_depth_attachments):
                adopt_depth = getattr(
                    self.vulkan, "adopt_external_depth_attachment", None
                )
                if callable(adopt_depth):
                    adopt_depth(self._filament_depth_attachments[eye_index].resource)
        return semaphores

    def _render_filament_controller_overlay(
        self,
        acquired_images: list[tuple[_EyeSwapchain, int]],
        record_time: Callable[[str, float], None],
    ) -> None:
        bridge = self.filament_bridge
        if (
            not self._filament_controller_overlay_after_composer
            or bridge is None
            or self._multiview_active
        ):
            return
        if not bool(getattr(bridge, "controller_overlay_abi_available", False)):
            if not getattr(self, "_filament_controller_overlay_unavailable_logged", False):
                self._filament_controller_overlay_unavailable_logged = True
                print(
                    "[OpenXRViewer] Filament controller overlay unavailable: "
                    "rebuild the native Bridge",
                    flush=True,
                )
            return
        for eye_index, (_eye, image_index) in enumerate(acquired_images):
            started = time.perf_counter()
            bridge.set_active_eye(eye_index)
            bridge.set_acquired_image(image_index)
            bridge.render_controller_overlay()
            record_time(f"openxr_filament_controller_overlay_eye{eye_index}", started)
        if not getattr(self, "_filament_controller_overlay_logged", False):
            self._filament_controller_overlay_logged = True
            print(
                "[OpenXRViewer] Filament controller overlay active: "
                "order=environment->screen/glow->controller/laser/guide",
                flush=True,
            )

    def _render_controller_composition_layer(
        self, views: list[Any]
    ) -> Any | None:
        bridge = self.filament_bridge
        swapchain = self._controller_composition_swapchain
        if (
            bridge is None
            or swapchain is None
            or not self._multiview_active
            or not bool(
                getattr(
                    bridge, "controller_composition_layer_abi_available", False
                )
            )
        ):
            return None
        started = time.perf_counter()
        with _acquired_swapchain_image(self.xr, swapchain) as image_index:
            bridge.render_controller_composition_layer(image_index)
        if self._on_breakdown_add_time is not None:
            self._on_breakdown_add_time(
                "openxr_controller_composition_layer",
                time.perf_counter() - started,
            )
        if not self._controller_composition_layer_logged:
            self._controller_composition_layer_logged = True
            print(
                "[OpenXRViewer] Controller composition layer active: "
                "order=projection->quad->controller/laser/guide",
                flush=True,
            )
        return OpenXrCompositionBuilder(
            self.xr, self.reference_space
        ).projection_layer(
            views,
            [swapchain],
            layer_flags=(
                self.xr.CompositionLayerFlags.BLEND_TEXTURE_SOURCE_ALPHA_BIT
            ),
        )

    def _render_vulkan_controller_proxy_layer(
        self, views: list[Any]
    ) -> Any | None:
        swapchains = self._vulkan_controller_proxy_swapchains
        screen_pass = self._vulkan_projection_screen_pass
        params = self._projection_controller_proxy_params()
        if (
            not self._vulkan_controller_proxy_enabled
            or screen_pass is None
            or params is None
            or len(swapchains) != 2
            or len(views) < 2
        ):
            return None
        acquired_images: list[tuple[_EyeSwapchain, int]] = []
        try:
            for eye in swapchains:
                image_index = self.xr.acquire_swapchain_image(eye.handle)
                acquired_images.append((eye, image_index))
            for eye, _image_index in acquired_images:
                self.xr.wait_swapchain_image(
                    eye.handle,
                    self.xr.SwapchainImageWaitInfo(timeout=self.xr.INFINITE_DURATION),
                )
            draws = []
            for eye_index, (eye, image_index) in enumerate(acquired_images):
                draws.append(
                    {
                        "target": eye.resources[image_index],
                        "array_layer": 0,
                        "eye_index": eye_index,
                        "frame_slot": int(self.frame_count) % 3,
                        "controller_proxy_params": params,
                        "controller_proxy_push_constants": (
                            self._projection_screen_push_constants(
                                views[eye_index]
                            )[:64]
                        ),
                        "clear_color": (0.0, 0.0, 0.0, 0.0),
                    }
                )
            screen_pass.submit_stereo_controller_proxy(
                draws,
                wait_for_timeline=0,
                clear_target=True,
            )
        except Exception as exc:
            print(
                "[OpenXRViewer] Vulkan controller proxy layer skipped: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return None
        finally:
            for eye, _image_index in acquired_images:
                self.xr.release_swapchain_image(eye.handle)
        if not self._vulkan_controller_proxy_layer_logged:
            self._vulkan_controller_proxy_layer_logged = True
            print(
                "[OpenXRViewer] Vulkan controller proxy layer active: "
                "order=projection->quad->controller/laser/guide",
                flush=True,
            )
        return OpenXrCompositionBuilder(self.xr, self.reference_space).projection_layer(
            views,
            swapchains,
            layer_flags=self.xr.CompositionLayerFlags.BLEND_TEXTURE_SOURCE_ALPHA_BIT,
        )

    def _render_projection_layer(
        self,
        views: list[Any],
        output_frame: VulkanStereoOutputFrame | None | object = _OUTPUT_FRAME_UNSET,
    ) -> Any | None:
        required_views = 2 if self._multiview_active else len(self.swapchains)
        if len(views) < required_views:
            return None
        # The profile adjusts the Filament camera relative to the model. The
        # composition layer must retain the runtime-provided eye poses so the
        # OpenXR compositor keeps the rendered image aligned with the headset.
        composition_views = views
        render_views = self._apply_filament_profile(views)
        xr = self.xr
        if output_frame is _OUTPUT_FRAME_UNSET:
            with self._output_lock:
                output_frame = self._pending_output
                self._pending_output = None
        else:
            with self._output_lock:
                if self._pending_output is output_frame:
                    self._pending_output = None
        if isinstance(output_frame, VulkanStereoOutputFrame):
            with self._output_lock:
                self._rendering_output = output_frame
        with self._output_lock:
            sampling_frame = self._displayed_output
        if self._filament_animation_origin is None:
            self._filament_animation_origin = self._frame_now
        animation_time = max(0.0, self._frame_now - self._filament_animation_origin)
        acquired_images: list[tuple[_EyeSwapchain, int]] = []
        consumer_release_semaphores: list[int | None] = [None, None]
        consumer_completion_timeline: int | None = None
        submitted_filament_eyes: list[int] = []
        completion_drain_attempted = False
        finished_semaphore_available = False
        filament_composer_wait_semaphores: list[Any] = []
        filament_composer_rendered = False
        render_succeeded = False
        self._vulkan_projection_composer_active = False
        self._screen_quad_reprojection_active = False
        composer_frame = (
            output_frame
            if isinstance(output_frame, VulkanStereoOutputFrame)
            else sampling_frame
        )
        presentation_frame = (
            composer_frame
            if isinstance(composer_frame, VulkanStereoOutputFrame)
            else None
        )
        use_vulkan_projection_composer = bool(
            self._vulkan_projection_composer_requested
            and isinstance(composer_frame, VulkanStereoOutputFrame)
        )
        use_screen_quad_reprojection = bool(
            self._screen_quad_reprojection_requested
            and (
                self._can_use_screen_quad_reprojection(composer_frame)
                or bool(self._last_screen_quad_layers)
            )
        )
        if use_screen_quad_reprojection:
            use_vulkan_projection_composer = False
        filament_queue_lock = getattr(self.vulkan, "_lock", None)
        filament_queue_locked = False
        projection_started = time.perf_counter()
        self._projection_busy.set()

        def record_time(name: str, started: float) -> None:
            callback = self._on_breakdown_add_time
            if callback is not None:
                callback(name, max(0.0, time.perf_counter() - started))

        def prepare_filament_rendering() -> None:
            nonlocal filament_queue_locked, finished_semaphore_available
            if (
                self.filament_bridge is not None
                and filament_queue_lock is not None
                and not filament_queue_locked
            ):
                filament_lock_started = time.perf_counter()
                filament_queue_lock.acquire()
                filament_queue_locked = True
                record_time("openxr_filament_lock_wait", filament_lock_started)
            if self.filament_bridge is None:
                return
            # Glow is composed by VulkanProjectionScreenPass after the SBS
            # draw. Filament only owns environment/controller renderables.
            # Controller transforms and GLB animation state are shared by
            # both eye Views. Updating them twice adds owner-thread work
            # without changing either eye's scene state.
            self._update_controller_screen_light(
                presentation_frame, self.filament_bridge
            )
            self._update_environment_screen_lights(
                presentation_frame, self.filament_bridge
            )
            self._update_filament_controllers(self.filament_bridge)
            if hasattr(self.filament_bridge, "apply_animations"):
                self.filament_bridge.apply_animations(animation_time)
            finished_semaphore_available = bool(
                getattr(
                    self.filament_bridge,
                    "finished_drawing_semaphore_abi_available",
                    False,
                )
            )

        try:
            # Acquire the complete stereo pair before entering either blocking
            # wait. This gives the runtime both requests up front and avoids
            # serializing the second acquire behind the first eye's wait.
            acquire_started = time.perf_counter()
            for eye in self.swapchains:
                image_index = xr.acquire_swapchain_image(eye.handle)
                acquired_images.append((eye, image_index))
            record_time("openxr_projection_acquire_pair", acquire_started)

            wait_pair_started = time.perf_counter()
            for eye_index, (eye, _image_index) in enumerate(acquired_images):
                wait_started = time.perf_counter()
                xr.wait_swapchain_image(
                    eye.handle,
                    xr.SwapchainImageWaitInfo(timeout=xr.INFINITE_DURATION),
                )
                record_time(
                    f"openxr_projection_wait_eye{eye_index}", wait_started
                )
            record_time("openxr_swapchain_wait", wait_pair_started)

            if self._projection_array_eye_diagnostic:
                self._render_projection_array_eye_diagnostic(acquired_images)
                render_succeeded = True
            elif self._vulkan_multiview_eye_diagnostic:
                self._render_vulkan_multiview_eye_diagnostic(acquired_images)
                render_succeeded = True

            shared_prepare_started = time.perf_counter()
            if use_vulkan_projection_composer and presentation_frame is not None:
                sampling_frame = presentation_frame
            self._report_screen_resolution(views, presentation_frame)
            self._apply_screen_sampling_policy(
                presentation_frame
            )
            if (
                self.filament_bridge is not None
                and not use_screen_quad_reprojection
            ):
                prepare_filament_rendering()
            record_time("openxr_projection_shared_prepare", shared_prepare_started)

            if use_screen_quad_reprojection:
                try:
                    if isinstance(composer_frame, VulkanStereoOutputFrame):
                        self._render_screen_quad_reprojection(composer_frame)
                    elif not self._last_screen_quad_layers:
                        raise RuntimeError("screen Quad cache is empty")
                    else:
                        self._screen_quad_reprojection_active = True
                    self._clear_projection_targets(acquired_images)
                    render_succeeded = True
                except Exception as exc:
                    use_screen_quad_reprojection = False
                    self._screen_quad_reprojection_active = False
                    if self._on_breakdown_inc is not None:
                        self._on_breakdown_inc("openxr_screen_quad_fallback", 1)
                    self._report_screen_quad_reprojection_status(
                        "fallback", type(exc).__name__
                    )
                    use_vulkan_projection_composer = bool(
                        self._vulkan_projection_composer_requested
                        and isinstance(composer_frame, VulkanStereoOutputFrame)
                    )
            if not render_succeeded and use_vulkan_projection_composer:
                try:
                    filament_hdr_sources: tuple[Any, ...] = ()
                    if self.filament_bridge is not None:
                        if self._multiview_active:
                            finished = self._render_filament_multiview(
                                render_views,
                                presentation_frame,
                                finished_semaphore_available,
                                record_time,
                            )
                            filament_composer_wait_semaphores = (
                                [finished] if finished is not None else []
                            )
                            if self._filament_multiview_current is None:
                                raise RuntimeError(
                                    "Filament multiview HDR frame was not published"
                                )
                            filament_hdr_sources = tuple(
                                self._filament_multiview_current.layer_resources
                            )
                        else:
                            filament_composer_wait_semaphores = (
                                self._render_filament_for_projection_composer(
                                    render_views,
                                    acquired_images,
                                    animation_time,
                                    record_time,
                                )
                            )
                        filament_composer_rendered = True
                    if filament_composer_wait_semaphores or filament_hdr_sources:
                        composer_timeline = self._render_vulkan_projection_composer(
                            composer_frame,
                            acquired_images,
                            composition_views,
                            filament_wait_semaphores=filament_composer_wait_semaphores,
                            filament_hdr_sources=filament_hdr_sources,
                        )
                    else:
                        composer_timeline = self._render_vulkan_projection_composer(
                            composer_frame, acquired_images, composition_views
                        )
                    self._render_filament_controller_overlay(
                        acquired_images, record_time
                    )
                    composer_frame.metadata["_vulkan_consumer_release_timeline"] = max(
                        int(
                            composer_frame.metadata.get(
                                "_vulkan_consumer_release_timeline", 0
                            )
                        ),
                        int(composer_timeline),
                    )
                    if self._filament_multiview_current_slot is not None:
                        self._filament_multiview_slot_timelines[
                            self._filament_multiview_current_slot
                        ] = int(composer_timeline)
                    render_succeeded = True
                except Exception as exc:
                    use_vulkan_projection_composer = False
                    # The fallback renders only the Filament environment and
                    # controllers. The SBS image stays Composer-only.
                    self._vulkan_projection_composer_active = False
                    fallback_status = (type(exc).__name__, str(exc))
                    if fallback_status != self._last_vulkan_projection_composer_fallback:
                        self._last_vulkan_projection_composer_fallback = fallback_status
                        print(
                            "[OpenXRViewer] Vulkan projection composer fallback: "
                            f"{fallback_status[0]}: {fallback_status[1]}\n"
                            f"{traceback.format_exc().rstrip()}",
                            flush=True,
                        )
                    if self._on_breakdown_inc is not None:
                        self._on_breakdown_inc(
                            "openxr_vulkan_projection_composer_fallback", 1
                        )
                    if filament_composer_rendered:
                        if (
                            filament_composer_wait_semaphores
                            and not self._filament_multiview_finished_consumed
                        ):
                            drain_started = time.perf_counter()
                            consumer_completion_timeline = self.vulkan.submit_on(
                                "graphics",
                                lambda _command_buffer: None,
                                wait_semaphore=filament_composer_wait_semaphores,
                            )
                            record_time("openxr_filament_completion_drain", drain_started)
                            if isinstance(sampling_frame, VulkanStereoOutputFrame):
                                sampling_frame.metadata[
                                    "_vulkan_consumer_release_timeline"
                                ] = max(
                                    int(
                                        sampling_frame.metadata.get(
                                            "_vulkan_consumer_release_timeline", 0
                                        )
                                    ),
                                    int(consumer_completion_timeline),
                                )
                        if self._multiview_active:
                            self._clear_projection_targets(acquired_images)
                        render_succeeded = True
                    else:
                        fallback_prepare_started = time.perf_counter()
                        prepare_filament_rendering()
                        record_time(
                            "openxr_vulkan_composer_fallback_prepare",
                            fallback_prepare_started,
                        )
            if (
                not render_succeeded
                and self._multiview_active
                and self.filament_bridge is not None
            ):
                finished = self._render_filament_multiview(
                    render_views,
                    presentation_frame,
                    finished_semaphore_available,
                    record_time,
                )
                consumer_completion_timeline = (
                    self._resolve_filament_multiview_hdr(
                        acquired_images,
                        [finished] if finished is not None else [],
                    )
                )
                render_succeeded = True
            if not render_succeeded:
                for eye_index, (eye, image_index) in enumerate(acquired_images):
                    if self.filament_bridge is not None:
                        bridge = self.filament_bridge
                        state_started = time.perf_counter()
                        bridge.set_active_eye(eye_index)
                        _update_filament_camera(
                            bridge,
                            render_views[eye_index],
                            near_plane=self._profile_near_plane,
                            far_plane=self._profile_far_plane,
                        )
                        record_time(
                            f"openxr_filament_eye{eye_index}_state", state_started
                        )
                        bridge.set_acquired_image(image_index)
                        queue_started = time.perf_counter()
                        bridge.begin_frame()
                        record_time(
                            f"openxr_filament_eye{eye_index}_queue", queue_started
                        )
                        finish_started = time.perf_counter()
                        bridge.end_frame()
                        record_time(
                            f"openxr_filament_eye{eye_index}_finish_wait",
                            finish_started,
                        )
                        submitted_filament_eyes.append(eye_index)
                        if finished_semaphore_available:
                            consumer_release_semaphores[eye_index] = (
                                bridge.get_finished_drawing_semaphore()
                            )
                    else:
                        image_address = _ctypes_handle_address(eye.images[image_index].image)
                        image = self.vulkan.image_handle_from_address(image_address)
                        self.vulkan.clear_color_image(image, self.config.clear_color)
            published_semaphores = tuple(
                semaphore
                for semaphore in consumer_release_semaphores
                if semaphore is not None
            )
            expected_semaphores = 1 if self._multiview_active else 2
            if (
                use_vulkan_projection_composer
                or filament_composer_rendered
                or self._filament_multiview_finished_consumed
            ):
                expected_semaphores = 0
            if finished_semaphore_available and len(published_semaphores) != expected_semaphores:
                raise RuntimeError(
                    "Filament did not publish the expected render-finished semaphores"
                )
            readback_ready = self._advance_filament_multiview_layer_readback()
            readback_consumed_filament_semaphore = False
            if (
                published_semaphores
                and readback_ready
            ):
                consumer_completion_timeline = self._capture_filament_multiview_layers(
                    acquired_images,
                    None,
                    published_semaphores,
                )
                readback_consumed_filament_semaphore = True
            if published_semaphores and not readback_consumed_filament_semaphore:
                drain_started = time.perf_counter()
                completion_drain_attempted = True
                consumer_completion_timeline = self.vulkan.submit_on(
                    "graphics",
                    lambda _command_buffer: None,
                    wait_semaphore=published_semaphores,
                )
                record_time("openxr_filament_completion_drain", drain_started)
                if isinstance(sampling_frame, VulkanStereoOutputFrame):
                    sampling_frame.metadata["_vulkan_consumer_release_timeline"] = max(
                        int(
                            sampling_frame.metadata.get(
                                "_vulkan_consumer_release_timeline", 0
                            )
                        ),
                        int(consumer_completion_timeline),
                    )
            if readback_ready and not readback_consumed_filament_semaphore:
                self._capture_filament_multiview_layers(
                    acquired_images,
                    consumer_completion_timeline,
                )
            render_succeeded = True
            if presentation_frame is not None and (
                use_vulkan_projection_composer
                or use_screen_quad_reprojection
                or self.filament_bridge is None
            ):
                if self._on_breakdown_inc is not None:
                    self._on_breakdown_inc(
                        "openxr_new_screen_frame"
                        if isinstance(output_frame, VulkanStereoOutputFrame)
                        else "openxr_reused_screen_frame",
                        1,
                    )
            if self._on_breakdown_set_latest is not None:
                self._on_breakdown_set_latest(
                    "openxr_vulkan_projection_composer_requested",
                    self._vulkan_projection_composer_requested,
                )
                self._on_breakdown_set_latest(
                    "openxr_vulkan_projection_quality_chain_requested",
                    self._vulkan_projection_quality_chain_requested,
                )
                self._on_breakdown_set_latest(
                    "openxr_vulkan_projection_composer_active",
                    self._vulkan_projection_composer_active,
                )
                self._on_breakdown_set_latest(
                    "openxr_vulkan_projection_composer_frame_id",
                    (
                        self._vulkan_projection_composer_frame_id
                        if self._vulkan_projection_composer_active
                        else -1
                    ),
                )
                self._on_breakdown_set_latest(
                    "openxr_screen_quad_reprojection_requested",
                    self._screen_quad_reprojection_requested,
                )
                self._on_breakdown_set_latest(
                    "openxr_screen_quad_reprojection_active",
                    self._screen_quad_reprojection_active,
                )
                self._on_breakdown_set_latest(
                    "openxr_projection_path",
                    (
                        "screen_quad_reprojection"
                        if self._screen_quad_reprojection_active
                        else (
                            "vulkan_composer"
                            if self._vulkan_projection_composer_active
                            else (
                                "filament_multiview"
                                if self._multiview_active
                                else "filament_per_eye"
                            )
                        )
                    ),
                )
            self._maybe_capture_sbs_sequence(presentation_frame)
        finally:
            if (
                not render_succeeded
                and self.filament_bridge is not None
                and submitted_filament_eyes
            ):
                try:
                    if not any(consumer_release_semaphores):
                        self.filament_bridge.wait_for_idle()
                    if finished_semaphore_available:
                        for eye_index in submitted_filament_eyes:
                            if consumer_release_semaphores[eye_index] is not None:
                                continue
                            self.filament_bridge.set_active_eye(eye_index)
                            consumer_release_semaphores[eye_index] = (
                                self.filament_bridge.get_finished_drawing_semaphore()
                            )
                    if (
                        not completion_drain_attempted
                        and any(consumer_release_semaphores)
                        and not bool(getattr(self.vulkan, "device_lost", False))
                    ):
                        completion_drain_attempted = True
                        consumer_completion_timeline = self.vulkan.submit_on(
                            "graphics",
                            lambda _command_buffer: None,
                            wait_semaphore=tuple(
                                semaphore
                                for semaphore in consumer_release_semaphores
                                if semaphore is not None
                            ),
                        )
                    if (
                        consumer_completion_timeline is not None
                        and isinstance(sampling_frame, VulkanStereoOutputFrame)
                    ):
                        sampling_frame.metadata[
                            "_vulkan_consumer_release_timeline"
                        ] = max(
                            int(
                                sampling_frame.metadata.get(
                                    "_vulkan_consumer_release_timeline", 0
                                )
                            ),
                            int(consumer_completion_timeline),
                        )
                except Exception:
                    pass
            if filament_queue_locked and filament_queue_lock is not None:
                filament_queue_lock.release()
                filament_queue_locked = False
            release_started = time.perf_counter()
            for eye, _image_index in acquired_images:
                xr.release_swapchain_image(eye.handle)
            record_time("openxr_projection_release_pair", release_started)
            if (
                isinstance(output_frame, VulkanStereoOutputFrame)
                and not render_succeeded
            ):
                self._abort_output_frame(output_frame)
            record_time("openxr_projection_total", projection_started)
            self._projection_busy.clear()
        return OpenXrCompositionBuilder(xr, self.reference_space).projection_layer(
            composition_views, self.swapchains
        )

    def _render_projection_array_eye_diagnostic(
        self, acquired_images: list[tuple[_EyeSwapchain, int]]
    ) -> None:
        if len(acquired_images) != 1 or acquired_images[0][0].array_size < 2:
            raise RuntimeError(
                "Projection array eye diagnostic requires one array_size=2 swapchain"
            )
        eye, image_index = acquired_images[0]
        target = eye.resources[image_index].image
        for array_layer, color in enumerate(
            ((1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0))
        ):
            self.vulkan.clear_color_image(
                target, color, base_array_layer=array_layer
            )
        if not getattr(self, "_projection_array_eye_diagnostic_logged", False):
            self._projection_array_eye_diagnostic_logged = True
            print(
                "[OpenXRViewer] Projection array eye diagnostic active: "
                "left=red layer=0 right=green layer=1",
                flush=True,
            )

    def _render_vulkan_multiview_eye_diagnostic(
        self, acquired_images: list[tuple[_EyeSwapchain, int]]
    ) -> None:
        if len(acquired_images) != 1 or acquired_images[0][0].array_size < 2:
            raise RuntimeError(
                "Vulkan multiview diagnostic requires one array_size=2 swapchain"
            )
        eye, image_index = acquired_images[0]
        if self._vulkan_multiview_diagnostic_pass is None:
            self._vulkan_multiview_diagnostic_pass = (
                VulkanMultiviewEyeDiagnosticPass(
                    self.vulkan, int(self.swapchain_format)
                )
            )
        timeline = self._vulkan_multiview_diagnostic_pass.submit(
            eye.resources[image_index]
        )
        self.vulkan.wait_for_timeline(timeline)
        if not getattr(self, "_vulkan_multiview_eye_diagnostic_logged", False):
            self._vulkan_multiview_eye_diagnostic_logged = True
            view_counts = self._vulkan_multiview_diagnostic_pass.read_view_counts()
            print(
                "[OpenXRViewer] Vulkan multiview eye diagnostic active: "
                "viewMask=0x3 vertex_gl_ViewIndex left=red right=green "
                f"fragment_counts={view_counts}",
                flush=True,
            )

    def _ensure_sbs_capture_slots(self, frame: VulkanStereoOutputFrame) -> bool:
        if self._sbs_capture_slots:
            return True
        if self.vulkan is None:
            return False
        options = self._sbs_capture_options
        if options is None:
            return False
        left = frame.left_eye
        right = frame.right_eye
        if not isinstance(left, VulkanImageResource) or not isinstance(
            right, VulkanImageResource
        ):
            return False
        source_width = int(left.width)
        source_height = int(left.height)
        if source_width <= 0 or source_height <= 0:
            return False
        target_width = min(source_width, int(options["eye_width"]))
        target_height = max(1, round(source_height * target_width / source_width))
        for slot_index in range(3):
            self._sbs_capture_slots.append(
                {
                    "left": VulkanHostImage(
                        self.vulkan,
                        target_width,
                        target_height,
                        format=int(left.format),
                        label=f"sbs-capture-left-{slot_index}",
                        readback=True,
                    ),
                    "right": VulkanHostImage(
                        self.vulkan,
                        target_width,
                        target_height,
                        format=int(right.format),
                        label=f"sbs-capture-right-{slot_index}",
                        readback=True,
                    ),
                    "timeline": 0,
                    "record": None,
                }
            )
        output_dir = Path(options["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        print(
            "[OpenXRViewer] SBS sequence capture armed: "
            f"delay={float(options['delay_seconds']):.1f}s "
            f"metadata_samples={int(options['sample_count'])} "
            f"images={min(int(options['image_count']), int(options['sample_count']))} "
            f"size={target_width * 2}x{target_height} dir={output_dir}",
            flush=True,
        )
        return True

    def _prune_sbs_capture_writes(self) -> None:
        pending: list[tuple[Future, dict[str, Any]]] = []
        for future, record in self._sbs_capture_write_futures:
            if not future.done():
                pending.append((future, record))
                continue
            try:
                future.result()
            except Exception as exc:
                record["status"] = "write_failed"
                record["error"] = f"{type(exc).__name__}: {exc}"
            else:
                record["status"] = "saved"
        self._sbs_capture_write_futures = pending

    def _drain_sbs_capture_readbacks(self, *, force: bool = False) -> None:
        if not self._sbs_capture_slots or self.vulkan is None:
            return
        self._prune_sbs_capture_writes()
        completed = self.vulkan.completed_timeline_value()
        if force:
            completed = self.vulkan.last_submitted_timeline_value
        if completed is None:
            return
        executor = self._sbs_capture_executor
        for slot in self._sbs_capture_slots:
            timeline = int(slot.get("timeline", 0))
            record = slot.get("record")
            if timeline <= 0 or timeline > int(completed):
                continue
            if record is None:
                slot["timeline"] = 0
                continue
            try:
                left_host = slot["left"]
                right_host = slot["right"]
                left_rgb = _vulkan_rgba_to_rgb(
                    left_host.read_rgba(),
                    format_value=int(left_host.format),
                    vk=left_host.vk,
                    image_origin=str(record["image_origin"]),
                )
                right_rgb = _vulkan_rgba_to_rgb(
                    right_host.read_rgba(),
                    format_value=int(right_host.format),
                    vk=right_host.vk,
                    image_origin=str(record["image_origin"]),
                )
                if executor is None:
                    raise RuntimeError("SBS capture writer is unavailable")
                future = executor.submit(
                    _write_sbs_capture_png,
                    Path(record["path"]),
                    left_rgb,
                    right_rgb,
                )
                record["status"] = "writing"
                record["gpu_timeline"] = timeline
                self._sbs_capture_write_futures.append((future, record))
            except Exception as exc:
                record["status"] = "readback_failed"
                record["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                slot["timeline"] = 0
                slot["record"] = None

    def _write_sbs_capture_manifest(self) -> None:
        options = self._sbs_capture_options
        if options is None:
            return
        output_dir = Path(options["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "stage": "vulkan_stereo_eye_images_before_projection_composition",
            "contains_projection": False,
            "delay_seconds": float(options["delay_seconds"]),
            "requested_samples": int(options["sample_count"]),
            "observed_samples": int(self._sbs_capture_observed),
            "requested_images": min(
                int(options["image_count"]), int(options["sample_count"])
            ),
            "scheduled_images": int(self._sbs_capture_scheduled),
            "skipped_images": int(self._sbs_capture_skipped),
            "records": self._sbs_capture_records,
        }
        (output_dir / "sbs_capture_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _maybe_capture_sbs_sequence(
        self, frame: VulkanStereoOutputFrame | None
    ) -> None:
        options = self._sbs_capture_options
        if options is None or self._sbs_capture_finished:
            return
        self._drain_sbs_capture_readbacks()
        if (
            self._sbs_capture_observed >= int(options["sample_count"])
            and not self._sbs_capture_write_futures
            and not any(int(slot.get("timeline", 0)) for slot in self._sbs_capture_slots)
        ):
            self._close_sbs_sequence_capture()
            return
        if frame is None:
            return
        if bool(frame.metadata.get("_vulkan_release_attempted")):
            return
        frame_id = int(frame.frame_id)
        if self._sbs_capture_seen_frame_id == frame_id:
            return
        self._sbs_capture_seen_frame_id = frame_id
        now = time.perf_counter()
        if self._sbs_capture_origin is None:
            self._sbs_capture_origin = now
            return
        if now - self._sbs_capture_origin < float(options["delay_seconds"]):
            return
        if self._sbs_capture_observed >= int(options["sample_count"]):
            return
        sample_index = int(self._sbs_capture_observed)
        self._sbs_capture_observed += 1
        record = {
            "sample_index": sample_index,
            "frame_id": frame_id,
            "source_timestamp": float(frame.timestamp),
            "capture_monotonic": now,
            "image_origin": str(frame.image_origin),
            "status": "metadata_only",
        }
        self._sbs_capture_records.append(record)
        sample_count = int(options["sample_count"])
        image_count = min(int(options["image_count"]), sample_count)
        if image_count <= 1:
            image_indices = {0}
        else:
            image_indices = {
                round(index * (sample_count - 1) / (image_count - 1))
                for index in range(image_count)
            }
        if sample_index not in image_indices:
            return
        if not self._ensure_sbs_capture_slots(frame):
            record["status"] = "capture_unavailable"
            self._sbs_capture_skipped += 1
            return
        self._prune_sbs_capture_writes()
        if len(self._sbs_capture_write_futures) >= 6:
            record["status"] = "capture_skipped_writer_busy"
            self._sbs_capture_skipped += 1
            return
        slot = next(
            (item for item in self._sbs_capture_slots if not item["timeline"]),
            None,
        )
        if slot is None:
            record["status"] = "capture_skipped_gpu_busy"
            self._sbs_capture_skipped += 1
            return
        output_path = Path(options["output_dir"]) / f"sbs_sample_{sample_index:04d}.png"
        record["path"] = str(output_path)
        record["file"] = output_path.name
        record["status"] = "gpu_pending"
        wait_timeline = max(
            int(frame.ready_timeline or 0),
            int(frame.metadata.get("_vulkan_consumer_release_timeline", 0)),
        )
        submitted_timeline = 0
        try:
            left_timeline = self.vulkan.copy_image(
                frame.left_eye,
                slot["left"].resource,
                wait_for_timeline=(wait_timeline or None),
                resize=True,
                filter_mode=self.vulkan.vk.VK_FILTER_LINEAR,
                destination_host_readable=True,
            )
            submitted_timeline = int(left_timeline)
            right_timeline = self.vulkan.copy_image(
                frame.right_eye,
                slot["right"].resource,
                wait_for_timeline=left_timeline,
                resize=True,
                filter_mode=self.vulkan.vk.VK_FILTER_LINEAR,
                destination_host_readable=True,
            )
            submitted_timeline = int(right_timeline)
        except Exception as exc:
            record["status"] = "submit_failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            self._sbs_capture_skipped += 1
            if submitted_timeline > 0:
                slot["timeline"] = submitted_timeline
                slot["record"] = None
                frame.metadata["_vulkan_consumer_release_timeline"] = max(
                    int(
                        frame.metadata.get(
                            "_vulkan_consumer_release_timeline", 0
                        )
                    ),
                    submitted_timeline,
                )
            return
        slot["timeline"] = int(right_timeline)
        slot["record"] = record
        self._sbs_capture_scheduled += 1
        frame.metadata["_vulkan_consumer_release_timeline"] = max(
            int(frame.metadata.get("_vulkan_consumer_release_timeline", 0)),
            int(right_timeline),
        )
        if self._sbs_capture_observed >= int(options["sample_count"]):
            print(
                "[OpenXRViewer] SBS pacing metadata collection complete; "
                "finishing sparse asynchronous image writes",
                flush=True,
            )

    def _close_sbs_sequence_capture(self) -> None:
        if self._sbs_capture_options is None or self._sbs_capture_finished:
            return
        self._drain_sbs_capture_readbacks(force=True)
        executor = self._sbs_capture_executor
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
        self._sbs_capture_executor = None
        self._prune_sbs_capture_writes()
        self._write_sbs_capture_manifest()
        for slot in self._sbs_capture_slots:
            for key in ("left", "right"):
                try:
                    slot[key].close()
                except Exception:
                    pass
        self._sbs_capture_slots.clear()
        self._sbs_capture_finished = True
        print(
            "[OpenXRViewer] SBS sequence capture saved: "
            f"{self._sbs_capture_options['output_dir']}",
            flush=True,
        )

    @staticmethod
    def _save_visual_regression_host_image(
        host_image: VulkanHostImage,
        output_path: Path,
    ) -> None:
        """Save raw Vulkan pixels as an RGB diagnostic without changing them."""
        from PIL import Image

        pixels = host_image.read_rgba()
        vk = host_image.vk
        if int(host_image.format) in {
            int(vk.VK_FORMAT_B8G8R8A8_UNORM),
            int(vk.VK_FORMAT_B8G8R8A8_SRGB),
        }:
            pixels = pixels[..., [2, 1, 0, 3]]
        Image.fromarray(pixels[..., :3].copy(), mode="RGB").save(output_path)

    def _maybe_capture_visual_regression_frame(
        self,
        output_frame: VulkanStereoOutputFrame,
        *,
        eye_index: int,
        source_resource: VulkanImageResource | None,
        projection_resource: VulkanImageResource | None,
        projection_array_layer: int = 0,
        source_layout: int,
        source_access_mask: int,
        source_stage_mask: int,
    ) -> None:
        """Capture input/output/projection stages once from the live XR frame."""
        if self._visual_regression_capture_failed:
            return
        metadata = output_frame.metadata or {}
        output_dir_text = str(metadata.get("visual_regression_dir", "")).strip()
        if not output_dir_text:
            # Runtime visual regression is opt-in. Without producer metadata
            # there is no explicit capture request, so do not read back or
            # write projection images during normal rendering.
            return
        if not output_dir_text or self.vulkan is None:
            return
        eye = int(eye_index)
        if eye in self._visual_regression_capture_eyes:
            return
        if source_resource is None or projection_resource is None:
            self._visual_regression_capture_failed = True
            print(
                "[OpenXRViewer] visual regression capture skipped: "
                "production source or projection resource is unavailable",
                flush=True,
            )
            return
        try:
            from stereo_runtime.stage_visual_regression import _write_contact_sheet

            output_dir = Path(output_dir_text)
            output_dir.mkdir(parents=True, exist_ok=True)
            source_host_image = self._visual_regression_source_host_images.get(eye)
            if source_host_image is None:
                source_host_image = VulkanHostImage(
                    self.vulkan,
                    int(source_resource.width),
                    int(source_resource.height),
                    format=int(source_resource.format),
                    label=f"visual-regression-live-source-eye-{eye}",
                    readback=True,
                )
                self._visual_regression_source_host_images[eye] = source_host_image
            projection_host_image = self._visual_regression_projection_host_images.get(eye)
            if projection_host_image is None:
                projection_host_image = VulkanHostImage(
                    self.vulkan,
                    int(projection_resource.width),
                    int(projection_resource.height),
                    format=int(projection_resource.format),
                    label=f"visual-regression-live-projection-eye-{eye}",
                    readback=True,
                )
                self._visual_regression_projection_host_images[eye] = projection_host_image

            # Native Filament owns the projection render pass and the
            # producer may have registered the output state through a
            # different path. Normalize only the Python-side tracker for this
            # diagnostic copy; this does not submit a Vulkan barrier.
            vk = self.vulkan.vk
            self.vulkan.register_image_state(
                source_resource.image,
                ImageState(
                    layout=int(source_layout),
                    access_mask=int(source_access_mask),
                    stage_mask=int(source_stage_mask),
                    queue_family_index=self.vulkan.queue_family_index,
                ),
            )
            self.vulkan.register_image_state(
                projection_resource.image,
                ImageState(
                    layout=int(vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL),
                    access_mask=int(vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT),
                    stage_mask=int(vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT),
                    queue_family_index=self.vulkan.queue_family_index,
                )
            )

            source_timeline = self.vulkan.copy_image(
                source_resource,
                source_host_image.resource,
                destination_host_readable=True,
            )
            self.vulkan.wait_for_timeline(source_timeline)
            self._save_visual_regression_host_image(
                source_host_image,
                output_dir / f"03_vulkan_output_{'left' if eye == 0 else 'right'}_eye.png",
            )

            try:
                projection_timeline = self.vulkan.copy_image(
                    projection_resource,
                    projection_host_image.resource,
                    source_array_layer=projection_array_layer,
                    destination_host_readable=True,
                )
            except VulkanCapabilityError as first_exc:
                # Some OpenXR runtimes leave the Python-side swapchain state
                # stale after Filament's native render pass. Reassert the
                # known completed color-attachment state and retry once.
                self.vulkan.register_image_state(
                    projection_resource.image,
                    ImageState(
                        layout=int(vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL),
                        access_mask=int(vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT),
                        stage_mask=int(vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT),
                        queue_family_index=self.vulkan.queue_family_index,
                    ),
                )
                try:
                    projection_timeline = self.vulkan.copy_image(
                        projection_resource,
                        projection_host_image.resource,
                        source_array_layer=projection_array_layer,
                        destination_host_readable=True,
                    )
                except Exception as retry_exc:
                    raise VulkanCapabilityError(
                        "projection image diagnostic copy failed after state retry: "
                        f"first={type(first_exc).__name__}: {first_exc}; "
                        f"retry={type(retry_exc).__name__}: {retry_exc}"
                    ) from retry_exc
            self.vulkan.wait_for_timeline(projection_timeline)
            self._save_visual_regression_host_image(
                projection_host_image,
                output_dir / f"06_openxr_projection_{'left' if eye == 0 else 'right'}_eye.png",
            )
            self._visual_regression_capture_eyes.add(eye)
            if len(self._visual_regression_capture_eyes) >= 2:
                manifest = {
                    "frame_id": int(output_frame.frame_id),
                    "source_stage": "vulkan_output_image",
                    "projection_stage": "openxr_projection_swapchain",
                    "readback": "temporary_host_image",
                    "color_space": str(output_frame.color_space),
                    "image_origin": str(output_frame.image_origin),
                    "vulkan_projection_quality_chain_requested": bool(
                        self._vulkan_projection_quality_chain_requested
                    ),
                    "source_size": [int(source_resource.width), int(source_resource.height)],
                    "projection_size": [int(projection_resource.width), int(projection_resource.height)],
                }
                (output_dir / "visual_regression_runtime_manifest.json").write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                _write_contact_sheet(output_dir)
                print(
                    f"[OpenXRViewer] automatic visual regression capture saved: {output_dir}",
                    flush=True,
                )
        except Exception as exc:
            self._visual_regression_capture_failed = True
            print(
                "[OpenXRViewer] automatic visual regression capture failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("OpenXrVulkanPresenter is not initialized")

    def _ensure_quad_swapchains(self, width: int, height: int) -> None:
        if self._quad_swapchain_extent == (width, height) and len(self._quad_swapchains) == 1:
            return
        if self.xr is None or self.session is None or self.vulkan is None:
            return
        self._destroy_quad_swapchains()
        vk = self.vulkan.vk
        formats = list(self.xr.enumerate_swapchain_formats(self.session))
        # The runtime output contract is display-referred sRGB. Match the
        # validated legacy Quad Layer path and prefer an sRGB target.
        quad_format = _select_swapchain_format(vk, formats, "srgb")
        handle = self.xr.create_swapchain(
            self.session,
            self.xr.SwapchainCreateInfo(
                usage_flags=(self.xr.SwapchainUsageFlags.COLOR_ATTACHMENT_BIT
                             | self.xr.SwapchainUsageFlags.TRANSFER_DST_BIT),
                format=quad_format, sample_count=1, width=width, height=height,
                face_count=1, array_size=2, mip_count=1,
            ),
        )
        images = list(self.xr.enumerate_swapchain_images(
            handle, self.xr.SwapchainImageVulkan2KHR
        ))
        self._quad_swapchains.append(_EyeSwapchain(
            handle, images, width, height,
            self._register_swapchain_images(images, width, height, quad_format),
            array_size=2,
        ))
        self._quad_swapchain_format = int(quad_format)
        self._quad_swapchain_extent = (width, height)
        print(
            f"[OpenXRViewer] Quad layer swapchains created: "
            f"format={_vulkan_format_name(vk, quad_format)} extent={width}x{height} array_size=2",
            flush=True,
        )

    def _destroy_quad_swapchains(self) -> None:
        if self.xr is None:
            self._quad_swapchains.clear()
            return
        for eye in reversed(self._quad_swapchains):
            for resource in reversed(eye.resources):
                try:
                    if self.vulkan is not None:
                        self.vulkan.unregister_external_image(resource)
                except Exception:
                    pass
            try:
                self.xr.destroy_swapchain(eye.handle)
            except Exception:
                pass
        self._quad_swapchains.clear()
        self._quad_swapchain_format = None
        self._quad_swapchain_extent = None

    def _destroy_tool_quad_layers(self) -> None:
        for entry in self._overlay_quad_entries.values():
            try:
                staging = entry.get("staging")
                if staging is not None:
                    staging.close()
            except Exception:
                pass
            for resource in reversed(entry.get("resources", ())):
                try:
                    if self.vulkan is not None:
                        self.vulkan.unregister_external_image(resource)
                except Exception:
                    pass
            try:
                if self.xr is not None:
                    self.xr.destroy_swapchain(entry["swapchain"])
            except Exception:
                pass
        self._overlay_quad_entries.clear()
        self._tool_quad_texture_cache.clear()
        self._tool_quad_texture_keys.clear()
        self._tool_overlay_xr_fps = 0.0
        self._tool_overlay_pending_xr_fps = 0.0
        self._tool_overlay_sbs_fps = 0.0
        self._tool_overlay_capture_fps = 0.0
        self._tool_overlay_latency_ms = 0.0
        self._tool_overlay_depth_strength = 0.0
        self._tool_overlay_depth_strength_pending = None
        self._depth_osd_message = None
        self._tool_overlay_vr_res = (0, 0)
        self._tool_overlay_sbs_res = (0, 0)
        self._tool_overlay_pending_latency_ms = 0.0
        self._tool_overlay_xr_window_started = 0.0
        self._tool_overlay_xr_window_frames = 0
        self._tool_overlay_xr_frame_ts.clear()
        self._tool_overlay_sbs_window_started = 0.0
        self._tool_overlay_sbs_window_frames = 0
        self._tool_overlay_last_output_id = None

    def _update_tool_overlay_metrics(
        self, output_frame: VulkanStereoOutputFrame | None
    ) -> None:
        """Update low-rate overlay metrics without touching GPU resources."""
        now = float(self._frame_now or time.perf_counter())
        if output_frame is not None:
            depth_value = (getattr(output_frame, "metadata", None) or {}).get(
                "depth_strength"
            )
            try:
                depth_value = float(depth_value)
            except (TypeError, ValueError):
                depth_value = None
            if depth_value is not None and math.isfinite(depth_value):
                depth_value = max(0.0, depth_value)
                pending = self._tool_overlay_depth_strength_pending
                if pending is not None:
                    if abs(depth_value - pending) <= 1e-3:
                        self._tool_overlay_depth_strength_pending = None
                    else:
                        # Do not let an older in-flight output frame overwrite
                        # the value just accepted by the controller callback.
                        depth_value = None
                if depth_value is not None:
                    self._tool_overlay_depth_strength = depth_value
        if self._tool_overlay_xr_window_started <= 0.0:
            self._tool_overlay_xr_window_started = now
        self._tool_overlay_xr_window_frames += 1
        xr_elapsed = now - self._tool_overlay_xr_window_started
        if xr_elapsed >= _TOOL_OVERLAY_UPDATE_INTERVAL:
            if self._tool_overlay_pending_xr_fps > 0.0:
                self._tool_overlay_xr_fps = self._tool_overlay_pending_xr_fps
            else:
                self._tool_overlay_xr_fps = (
                    self._tool_overlay_xr_window_frames / xr_elapsed
                )
            # Keep all displayed performance values on the same low-rate
            # snapshot. Rebuilding the PIL texture from per-frame latency
            # defeats the legacy overlay cache and stalls the presenter.
            self._tool_overlay_latency_ms = self._tool_overlay_pending_latency_ms
            if self._on_capture_fps is not None:
                try:
                    capture_fps = float(self._on_capture_fps())
                except (TypeError, ValueError):
                    capture_fps = 0.0
                self._tool_overlay_capture_fps = (
                    capture_fps if math.isfinite(capture_fps) and capture_fps > 0.0
                    else 0.0
                )
            self._tool_overlay_xr_window_started = now
            self._tool_overlay_xr_window_frames = 0

        if self._tool_overlay_sbs_window_started <= 0.0:
            self._tool_overlay_sbs_window_started = now
        if output_frame is not None:
            frame_id = int(output_frame.frame_id)
            if frame_id != self._tool_overlay_last_output_id:
                self._tool_overlay_last_output_id = frame_id
                self._tool_overlay_sbs_window_frames += 1
                timestamp = float(output_frame.timestamp)
                latency_ms = (now - timestamp) * 1000.0
                if 0.0 <= latency_ms <= 10000.0:
                    self._tool_overlay_pending_latency_ms = latency_ms
        sbs_elapsed = now - self._tool_overlay_sbs_window_started
        if sbs_elapsed >= _TOOL_OVERLAY_UPDATE_INTERVAL:
            self._tool_overlay_sbs_fps = (
                self._tool_overlay_sbs_window_frames / sbs_elapsed
            )
            if self._on_sbs_fps is not None:
                capture_target = int(
                    self._on_sbs_fps(self._tool_overlay_sbs_fps)
                )
                if capture_target != self._adaptive_capture_target_fps:
                    self._adaptive_capture_target_fps = capture_target
                    print(
                        "[OpenXRViewer] Adaptive capture target: "
                        f"{capture_target} FPS (SBS={self._tool_overlay_sbs_fps:.1f})",
                        flush=True,
                    )
                if self._on_breakdown_set_latest is not None:
                    self._on_breakdown_set_latest(
                        "adaptive_capture_target_fps", capture_target
                    )
            self._tool_overlay_sbs_window_started = now
            self._tool_overlay_sbs_window_frames = 0

    def _record_xr_presented_frame(self) -> None:
        timestamp = time.perf_counter()
        if self._on_breakdown_inc is not None:
            self._on_breakdown_inc("openxr_presented_frame", 1)
        self._tool_overlay_xr_frame_ts.append(timestamp)
        count = len(self._tool_overlay_xr_frame_ts)
        if count < 2:
            return
        span = timestamp - self._tool_overlay_xr_frame_ts[0]
        if span > 0.0:
            self._tool_overlay_pending_xr_fps = (count - 1) / span

    def _overlay_resolution_sizes(
        self, output_frame: VulkanStereoOutputFrame | None
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Return the live XR eye and per-eye output sizes for the FPS panel."""
        vr_res = tuple(self._tool_overlay_vr_res)
        if self.swapchains:
            eye = self.swapchains[0]
            candidate = (int(getattr(eye, "width", 0)), int(getattr(eye, "height", 0)))
            if candidate[0] > 0 and candidate[1] > 0:
                vr_res = candidate
                self._tool_overlay_vr_res = candidate

        sbs_res = tuple(self._tool_overlay_sbs_res)
        if output_frame is not None:
            metadata = dict(output_frame.metadata or {})
            candidate = metadata.get("render_size", metadata.get("source_render_size"))
            if isinstance(candidate, (list, tuple)) and len(candidate) >= 2:
                candidate_size = (int(candidate[0]), int(candidate[1]))
                if candidate_size[0] > 0 and candidate_size[1] > 0:
                    sbs_res = candidate_size
                    self._tool_overlay_sbs_res = candidate_size
            if sbs_res == (0, 0):
                eye = getattr(output_frame, "left_eye", None)
                candidate_size = (
                    int(getattr(eye, "width", 0)),
                    int(getattr(eye, "height", 0)),
                )
                if candidate_size[0] > 0 and candidate_size[1] > 0:
                    sbs_res = candidate_size
                    self._tool_overlay_sbs_res = candidate_size
        return vr_res, sbs_res

    def _render_quad_layers(self, output_frame: VulkanStereoOutputFrame | None) -> list[Any]:
        # The main SBS screen is Projection Composer-only. Quad layers carry
        # controller tools and 2D overlays; they never replace the screen.
        # A panel/guide render error must not take down the XR session.
        try:
            return self._render_tool_quad_layers(output_frame)
        except Exception:
            import traceback

            print(
                "[OpenXRViewer] quad layer render error; skipping overlays:\n"
                + traceback.format_exc().rstrip(),
                flush=True,
            )
            return []

    def _can_use_screen_quad_reprojection(
        self, frame: VulkanStereoOutputFrame | None
    ) -> bool:
        if (
            not isinstance(frame, VulkanStereoOutputFrame)
            or self._filament_screen is None
            or self.vulkan is None
            or self.xr is None
            or self.session is None
        ):
            return False
        metadata = frame.metadata or {}
        if not callable(metadata.get("_vulkan_source_prepare_for_sampling")):
            return False
        return all(
            int(getattr(source, "width", 0)) > 0
            and int(getattr(source, "height", 0)) > 0
            and getattr(source, "image", None) is not None
            for source in (frame.left_eye, frame.right_eye)
        )

    def _report_screen_quad_reprojection_status(self, *status: Any) -> None:
        current = tuple(status)
        if current == self._last_screen_quad_reprojection_status:
            return
        self._last_screen_quad_reprojection_status = current
        print(
            "[OpenXRViewer] Screen Quad Reprojection "
            + " ".join(str(value) for value in current),
            flush=True,
        )

    def _clear_projection_targets(
        self, acquired_images: list[tuple[_EyeSwapchain, int]]
    ) -> None:
        for eye_index, (eye, image_index) in enumerate(acquired_images):
            array_layers = range(eye.array_size) if len(acquired_images) == 1 else (0,)
            for array_layer in array_layers:
                self.vulkan.clear_color_image(
                    eye.resources[image_index].image,
                    self.config.clear_color,
                    base_array_layer=array_layer,
                )

    def _render_screen_quad_reprojection(
        self, frame: VulkanStereoOutputFrame
    ) -> None:
        if not self._can_use_screen_quad_reprojection(frame):
            raise RuntimeError("screen Quad prerequisites are unavailable")
        if self._screen_quad_reprojection_frame_id == int(frame.frame_id):
            self._screen_quad_reprojection_active = bool(self._last_screen_quad_layers)
            if self._screen_quad_reprojection_active and self._on_breakdown_inc is not None:
                self._on_breakdown_inc("openxr_screen_quad_reuse", 1)
            return
        width = int(frame.left_eye.width)
        height = int(frame.left_eye.height)
        if (int(frame.right_eye.width), int(frame.right_eye.height)) != (width, height):
            raise RuntimeError("stereo screen Quad source extents differ")
        upload_started = time.perf_counter()
        self._ensure_quad_swapchains(width, height)
        if len(self._quad_swapchains) != 1:
            raise RuntimeError("stereo screen Quad swapchains are unavailable")
        prepare_source = frame.metadata["_vulkan_source_prepare_for_sampling"]
        position, screen_width, screen_height, rotation = self._filament_screen
        diagnostic = _env_flag("D2S_OPENXR_SCREEN_QUAD_EYE_DIAGNOSTIC")
        screen_layers = []
        copy_timeline = 0
        quad_swapchain = self._quad_swapchains[0]
        with _acquired_swapchain_image(self.xr, quad_swapchain) as image_index:
            for eye_index, source in enumerate((frame.left_eye, frame.right_eye)):
                if diagnostic:
                    color = ((1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0))[eye_index]
                    copy_timeline = max(
                        copy_timeline,
                        int(self.vulkan.clear_color_image(
                            quad_swapchain.resources[image_index].image,
                            color,
                            base_array_layer=eye_index,
                        )),
                    )
                else:
                    visible_semaphore = prepare_source(frame.frame_id, eye_index)
                    copy_timeline = max(
                        copy_timeline,
                        int(
                            self.vulkan.copy_image(
                                source,
                                quad_swapchain.resources[image_index],
                                wait_semaphore=visible_semaphore,
                                destination_array_layer=eye_index,
                                flip_y=False,
                            )
                        ),
                    )
                screen_layers.append(
                    OpenXrCompositionBuilder(self.xr, self.reference_space).quad_layer(
                        quad_swapchain, position, screen_width, screen_height, rotation, eye_index
                    )
                )
        frame.metadata["_vulkan_consumer_release_timeline"] = max(
            int(frame.metadata.get("_vulkan_consumer_release_timeline", 0)),
            copy_timeline,
        )
        # Quad swapchain images now own the submitted content. Return the
        # producer-owned source immediately instead of holding a runtime slot
        # until a later head-pose-only Quad reuse.
        self._release_output_frame(frame)
        with self._output_lock:
            previous = self._displayed_output
            if self._rendering_output is frame:
                self._rendering_output = None
            self._displayed_output = None
            if self._pending_output is frame:
                self._pending_output = None
        if previous is not None and previous is not frame:
            self._release_output_frame(previous)
        self._last_screen_quad_layers = screen_layers
        self._screen_quad_reprojection_frame_id = int(frame.frame_id)
        self._screen_quad_reprojection_active = True
        self._report_screen_quad_reprojection_status(
            "eye_diagnostic=left_red_right_green" if diagnostic else "active",
            f"source={width}x{height}",
        )
        if self._on_breakdown_inc is not None:
            self._on_breakdown_inc("openxr_screen_quad_new", 1)
        if self._on_breakdown_add_time is not None:
            self._on_breakdown_add_time(
                "openxr_screen_quad_upload", time.perf_counter() - upload_started
            )

    def _overlay_language(self) -> str:
        return normalize_locale(LANG)

    def _filament_screen_pose_mat4(self) -> np.ndarray:
        position, _width, _height, rotation = self._filament_screen or (
            (0.0, 1.2, -2.0), 2.4, 1.35, (0.0, 0.0, 0.0)
        )
        pose = euler_to_mat4(
            *(math.radians(float(value)) for value in rotation)
        ).astype(np.float64)
        pose[:3, 3] = np.asarray(position, dtype=np.float64)
        return pose

    @staticmethod
    def _overlay_pose_from_matrix(matrix: np.ndarray) -> tuple[tuple[float, ...], tuple[float, ...]]:
        quaternion = _mat3_to_quat_xyzw(matrix[:3, :3].astype(np.float64))
        position = tuple(float(value) for value in matrix[:3, 3])
        return position, tuple(float(value) for value in quaternion)

    def _screen_overlay_pose(self, local_model: np.ndarray):
        return self._overlay_pose_from_matrix(
            self._filament_screen_pose_mat4() @ local_model
        )

    def _controller_overlay_pose(self, hand: int, panel_height: float, top_ref: float):
        grip = self._grip_mat_l if int(hand) == 0 else self._grip_mat_r
        aim = self._aim_mat_l if int(hand) == 0 else self._aim_mat_r
        panel_pos = panel_fwd = panel_up = None
        if grip is not None and aim is not None:
            grip_up = np.asarray(grip[:3, 1], dtype=np.float64)
            grip_up /= max(float(np.linalg.norm(grip_up)), 1e-10)
            laser_fwd = -np.asarray(aim[:3, 2], dtype=np.float64)
            right_axis = np.asarray(aim[:3, 0], dtype=np.float64)
            right_axis /= max(float(np.linalg.norm(right_axis)), 1e-10)
            angle = math.radians(12.0)
            laser_fwd = (
                laser_fwd * math.cos(angle)
                + np.cross(right_axis, laser_fwd) * math.sin(angle)
                + right_axis * float(np.dot(right_axis, laser_fwd))
                * (1.0 - math.cos(angle))
            )
            laser_fwd /= max(float(np.linalg.norm(laser_fwd)), 1e-10)
            grip_pos = np.asarray(grip[:3, 3], dtype=np.float64)
            laser_origin = grip_pos + grip_up * 0.020 + laser_fwd * 0.11
            panel_fwd = grip_up - laser_fwd
            panel_fwd /= max(float(np.linalg.norm(panel_fwd)), 1e-10)
            panel_up = grip_up
            panel_right = np.cross(panel_up, panel_fwd)
            panel_right /= max(float(np.linalg.norm(panel_right)), 1e-10)
            panel_up2 = np.cross(panel_fwd, panel_right)
            panel_up2 /= max(float(np.linalg.norm(panel_up2)), 1e-10)
            panel_pos = (
                laser_origin
                + panel_fwd * 0.05
                + panel_up2 * (top_ref - panel_height / 2.0)
            )
            basis = np.column_stack((panel_right, panel_up2, panel_fwd))
            matrix = np.eye(4, dtype=np.float64)
            matrix[:3, :3] = basis
            matrix[:3, 3] = panel_pos
            return self._overlay_pose_from_matrix(matrix)

        if self._head_position_w is None or self._head_forward_w is None:
            return None
        head = np.asarray(self._head_position_w, dtype=np.float64)
        forward = np.asarray(self._head_forward_w, dtype=np.float64)
        forward /= max(float(np.linalg.norm(forward)), 1e-10)
        panel_pos = head + forward * (1.0 if int(hand) == 0 else 1.2)
        panel_pos[1] += -0.15 if int(hand) == 0 else -0.3
        panel_fwd = -forward
        panel_up = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
        panel_right = np.cross(panel_up, panel_fwd)
        panel_right /= max(float(np.linalg.norm(panel_right)), 1e-10)
        panel_up2 = np.cross(panel_fwd, panel_right)
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = np.column_stack((panel_right, panel_up2, panel_fwd))
        matrix[:3, 3] = panel_pos
        return self._overlay_pose_from_matrix(matrix)

    def _operation_guide_environment_mode(self) -> bool:
        name = str(getattr(self, "_profile_view_name", "") or "").strip().lower()
        return bool(name and name not in {"default", "none"})

    def _cursor_overlay_specs(self, rgba, screen_pose, head):
        """Build the legacy laser hit rings as transparent tool quads."""
        specs = []
        for hand in (0, 1):
            origin, direction = self._controller_interaction_ray(hand)
            if origin is None or direction is None:
                continue
            keyboard_hit = None
            if self._keyboard_visible:
                keyboard_hit = self._keyboard_plane_hit(origin, direction)
            if keyboard_hit != (None, None) and keyboard_hit is not None:
                pose = self._keyboard_pose_mat4()
                x, y = (float(keyboard_hit[0]), float(keyboard_hit[1]))
                local = np.asarray((x, y, 0.0), dtype=np.float64)
                matrix = pose.copy()
                matrix[:3, 3] = (
                    pose[:3, 3] + pose[:3, :3] @ local + pose[:3, 2] * 0.003
                )
            else:
                hit = self._screen_ray_hit_for_hand(hand)
                if hit is None:
                    continue
                u, v = (float(hit[0]), float(hit[1]))
                _position, width, height, _rotation = self._filament_screen or (
                    (0.0, 1.2, -2.0), 2.4, 1.35, (0.0, 0.0, 0.0)
                )
                matrix = screen_pose.copy()
                if self._screen_curved:
                    half_angle = self._effective_screen_curve_half_angle()
                    angle = -half_angle + 2.0 * half_angle * u
                    tangent = np.asarray(
                        (math.cos(angle), 0.0, math.sin(angle)),
                        dtype=np.float64,
                    )
                    normal = np.asarray(
                        (-math.sin(angle), 0.0, math.cos(angle)),
                        dtype=np.float64,
                    )
                    curved_basis = np.column_stack(
                        (tangent, np.asarray((0.0, 1.0, 0.0)), normal)
                    )
                    matrix[:3, :3] = screen_pose[:3, :3] @ curved_basis
                    hit_world = self._screen_uv_to_world(u, v)
                    if hit_world is None:
                        continue
                    matrix[:3, 3] = (
                        hit_world + screen_pose[:3, :3] @ normal * 0.003
                    )
                else:
                    matrix[:3, 3] = (
                        screen_pose[:3, 3]
                        + screen_pose[:3, :3]
                        @ np.asarray(((u - 0.5) * float(width),
                                      (v - 0.5) * float(height), 0.0), dtype=np.float64)
                        + screen_pose[:3, 2] * 0.003
                    )
            distance = float(np.linalg.norm(matrix[:3, 3] - head))
            radius = 0.012 * float(np.clip(distance / 2.0, 0.35, 50.0))
            position, rotation = self._overlay_pose_from_matrix(matrix)
            specs.append(
                (
                    f"laser_cursor_{hand}",
                    rgba,
                    position,
                    (radius * 2.0, radius * 2.0),
                    rotation,
                )
            )
        return specs

    def _render_tool_quad_layers(
        self, output_frame: VulkanStereoOutputFrame | None = None
    ) -> list[Any]:
        """Submit the legacy keyboard and overlay quads with legacy poses."""
        if self.xr is None or self.session is None or self.vulkan is None:
            return []
        _position, width, height, _rotation = self._filament_screen or (
            (0.0, 1.2, -2.0), 2.4, 1.35, (0.0, 0.0, 0.0)
        )
        width = float(width)
        height = float(height)
        vr_res, sbs_res = self._overlay_resolution_sizes(output_frame)
        language = self._overlay_language()
        environment_mode = self._operation_guide_environment_mode()
        specs = []

        if self._keyboard_visible:
            keyboard_width = float(self._keyboard_width)
            keyboard_height = float(self._keyboard_height)
            hover_indices = tuple(
                sorted(
                    index
                    for index in (self._kb_hover_l, self._kb_hover_r)
                    if index is not None
                )
            )
            held_indices = tuple(
                sorted(
                    index
                    for index in (self._kb_held_key_l, self._kb_held_key_r)
                    if index is not None
                )
            )
            modifier_vks = {0x10: "shift", 0x11: "ctrl", 0x12: "alt", 0x5B: "win"}
            locked_indices = [
                index
                for index, key in enumerate(self._keyboard_keys)
                if key.vk in modifier_vks
                and bool(self._mod_state[modifier_vks[key.vk]][0])
            ]
            if self._caps_lock:
                locked_indices.extend(
                    index
                    for index, key in enumerate(self._keyboard_keys)
                    if key.vk == 0x14
                )
            locked_indices = tuple(sorted(set(locked_indices)))
            keyboard_cache_key = (
                bool(self._kb_show_shifted), keyboard_width, keyboard_height,
                hover_indices, held_indices, locked_indices,
            )
            rgba = self._tool_quad_texture_cache.get("keyboard")
            if rgba is None or self._tool_quad_texture_keys.get("keyboard") != keyboard_cache_key:
                rgba, self._keyboard_keys = build_keyboard_rgba(
                    self._kb_show_shifted,
                    keyboard_width,
                    keyboard_height,
                    hover_indices=hover_indices,
                    held_indices=held_indices,
                    locked_indices=locked_indices,
                )
                self._tool_quad_texture_cache["keyboard"] = rgba
                self._tool_quad_texture_keys["keyboard"] = keyboard_cache_key
            keyboard_pose = self._keyboard_pose_mat4()
            _keyboard_position, keyboard_quaternion = self._overlay_pose_from_matrix(
                keyboard_pose
            )
            specs.append(
                (
                    "keyboard", rgba, _keyboard_position,
                    (keyboard_width, keyboard_height), keyboard_quaternion,
                )
            )

        head = np.asarray(
            self._head_position_w if self._head_position_w is not None else (0, 0, 0),
            dtype=np.float64,
        )
        screen_pose = self._filament_screen_pose_mat4()
        screen_distance = float(np.linalg.norm(screen_pose[:3, 3] - head))
        now = float(self._frame_now or time.perf_counter())
        preset_osd_active = (
            self._preset_name_overlay
            and now - float(self._preset_osd_show_t) < 5.0
        )
        screen_osd_active = now - float(self._screen_osd_show_t) < 2.5
        if preset_osd_active or screen_osd_active:
            # Match the legacy screen OSD: dark rounded panel, grey labels,
            # cyan values, and a centered text group.
            if screen_osd_active:
                msdf_atlas = self._msdf_font_atlas
                osd_key = (
                    "screen_adjust_osd",
                    "gpu-msdf"
                    if self._vulkan_msdf_quad_renderer is not None
                    else ("cpu-msdf" if msdf_atlas is not None else "legacy"),
                    round(width, 2),
                    round(screen_distance, 2),
                )
                osd_rgba = self._tool_quad_texture_cache.get("screen_osd")
                if (
                    osd_rgba is None
                    or self._tool_quad_texture_keys.get("screen_osd") != osd_key
                ):
                    if msdf_atlas is not None:
                        runs = (
                            ("Size", (150, 158, 185, 255)),
                            (
                                f"{width:.2f} x {width * 9.0 / 16.0:.2f} m",
                                (0, 210, 230, 255),
                            ),
                            ("Dist", (150, 158, 185, 255)),
                            (f"{screen_distance:.2f} m", (0, 210, 230, 255)),
                        )
                        canvas_width, canvas_height, msdf_runs = _layout_msdf_osd_runs(
                            msdf_atlas, runs
                        )
                        if self._vulkan_msdf_quad_renderer is not None:
                            osd_rgba = VulkanMsdfQuadRequest(
                                width=canvas_width,
                                height=canvas_height,
                                runs=msdf_runs,
                            )
                        else:
                            from .overlay_textures import build_msdf_text_osd_rgba

                            osd_rgba = build_msdf_text_osd_rgba(
                                msdf_atlas,
                                size=(canvas_width, canvas_height),
                                runs=msdf_runs,
                            )
                    else:
                        canvas_width, canvas_height = 512, 78
                        osd_rgba = build_screen_adjust_osd_rgba(
                            width,
                            screen_distance,
                            size=(canvas_width, canvas_height),
                        )
                    self._tool_quad_texture_cache["screen_osd"] = osd_rgba
                    self._tool_quad_texture_keys["screen_osd"] = osd_key
                if isinstance(osd_rgba, VulkanMsdfQuadRequest):
                    canvas_width, canvas_height = osd_rgba.width, osd_rgba.height
                elif hasattr(osd_rgba, "shape"):
                    canvas_height, canvas_width = osd_rgba.shape[:2]
                else:
                    canvas_width, canvas_height = 512, 78
                osd_height = width * 0.03 * (
                    float(canvas_height) / _MSDF_OSD_REFERENCE_HEIGHT
                )
                osd_width = osd_height * (
                    float(canvas_width) / max(1.0, float(canvas_height))
                )
            else:
                msdf_atlas = self._msdf_font_atlas
                osd_key = (
                    "preset_osd",
                    "gpu-msdf"
                    if self._vulkan_msdf_quad_renderer is not None
                    else ("cpu-msdf" if msdf_atlas is not None else "legacy"),
                    str(self._preset_name_overlay),
                    round(width, 3),
                    round(height, 3),
                )
                osd_rgba = self._tool_quad_texture_cache.get("screen_osd")
                if (
                    osd_rgba is None
                    or self._tool_quad_texture_keys.get("screen_osd") != osd_key
                ):
                    if msdf_atlas is not None:
                        label = "Preset"
                        value = str(self._preset_name_overlay)
                        runs = (
                            {
                                "text": label,
                                "color": (150, 158, 185, 255),
                            },
                            {
                                "text": value,
                                "color": (0, 210, 230, 255),
                            },
                        )
                        canvas_width, canvas_height, msdf_runs = _layout_msdf_osd_runs(
                            msdf_atlas, runs
                        )
                        if self._vulkan_msdf_quad_renderer is not None:
                            osd_rgba = VulkanMsdfQuadRequest(
                                width=canvas_width,
                                height=canvas_height,
                                runs=msdf_runs,
                            )
                        else:
                            from .overlay_textures import build_msdf_text_osd_rgba

                            osd_rgba = build_msdf_text_osd_rgba(
                                msdf_atlas,
                                size=(canvas_width, canvas_height),
                                runs=msdf_runs,
                            )
                    else:
                        canvas_width, canvas_height = 768, 78
                        osd_rgba = build_screen_preset_osd_rgba(
                            str(self._preset_name_overlay),
                            size=(canvas_width, canvas_height),
                        )
                    self._tool_quad_texture_cache["screen_osd"] = osd_rgba
                    self._tool_quad_texture_keys["screen_osd"] = osd_key
                if isinstance(osd_rgba, VulkanMsdfQuadRequest):
                    canvas_width, canvas_height = osd_rgba.width, osd_rgba.height
                elif hasattr(osd_rgba, "shape"):
                    canvas_height, canvas_width = osd_rgba.shape[:2]
                else:
                    canvas_width, canvas_height = 768, 78
                osd_height = width * 0.03 * (
                    float(canvas_height) / _MSDF_OSD_REFERENCE_HEIGHT
                )
                osd_width = osd_height * (
                    float(canvas_width) / max(1.0, float(canvas_height))
                )
            osd_local = np.eye(4, dtype=np.float64)
            # Keep the legacy gap between the screen edge and the centered OSD.
            osd_local[:3, 3] = (
                0.0,
                height / 2.0 + width * 0.016 + osd_height / 2.0,
                0.0,
            )
            osd_position, osd_rotation = self._screen_overlay_pose(osd_local)
            specs.append(
                (
                    "screen_osd",
                    osd_rgba,
                    osd_position,
                    (osd_width, osd_height),
                    osd_rotation,
                )
            )
        depth_osd_active = now - float(self._depth_osd_show_t) < 2.5
        if depth_osd_active:
            msdf_atlas = self._msdf_font_atlas
            depth_key = (
                "depth_osd",
                "gpu-msdf"
                if self._vulkan_msdf_quad_renderer is not None
                else ("cpu-msdf" if msdf_atlas is not None else "legacy"),
                round(self._tool_overlay_depth_strength, 3),
                self._depth_osd_message,
            )
            depth_rgba = self._tool_quad_texture_cache.get("depth_osd")
            if (
                depth_rgba is None
                or self._tool_quad_texture_keys.get("depth_osd") != depth_key
            ):
                if msdf_atlas is not None:
                    depth_request = _build_msdf_depth_osd_request(
                        msdf_atlas,
                        self._tool_overlay_depth_strength,
                        self._depth_osd_message,
                    )
                    if self._vulkan_msdf_quad_renderer is not None:
                        depth_rgba = depth_request
                    else:
                        from .overlay_textures import build_msdf_text_osd_rgba

                        depth_rgba = build_msdf_text_osd_rgba(
                            msdf_atlas,
                            size=(depth_request.width, depth_request.height),
                            runs=depth_request.runs,
                            background=depth_request.background,
                            radius=int(depth_request.radius),
                        )
                else:
                    from .overlay_textures import build_short_osd_rgba

                    depth_rgba = build_short_osd_rgba(
                        [
                            self._depth_osd_message
                            or (
                                f"Depth Strength "
                                f"{self._tool_overlay_depth_strength:.2f}"
                            )
                        ]
                    )
                self._tool_quad_texture_cache["depth_osd"] = depth_rgba
                self._tool_quad_texture_keys["depth_osd"] = depth_key
            if isinstance(depth_rgba, VulkanMsdfQuadRequest):
                depth_canvas_width, depth_canvas_height = (
                    depth_rgba.width,
                    depth_rgba.height,
                )
            else:
                depth_canvas_height, depth_canvas_width = depth_rgba.shape[:2]
            depth_osd_height = width * 0.03 * (
                float(depth_canvas_height) / _MSDF_OSD_REFERENCE_HEIGHT
            )
            depth_osd_width = depth_osd_height * (
                float(depth_canvas_width) / max(1.0, float(depth_canvas_height))
            )
            depth_local = np.eye(4, dtype=np.float64)
            depth_local[:3, 3] = (
                0.0,
                height / 2.0 + width * 0.016 + depth_osd_height / 2.0,
                0.0,
            )
            depth_position, depth_rotation = self._screen_overlay_pose(depth_local)
            specs.append(
                (
                    "depth_osd",
                    depth_rgba,
                    depth_position,
                    (depth_osd_width, depth_osd_height),
                    depth_rotation,
                )
            )
        msdf_atlas = self._msdf_font_atlas
        controller_model_name = self._controller_model_display_name()
        fps_key = (
            "fps", language, environment_mode, round(width, 3), round(height, 3),
            round(self._tool_overlay_xr_fps, 1), round(self._tool_overlay_sbs_fps, 1),
            round(self._tool_overlay_capture_fps, 1),
            round(self._tool_overlay_latency_ms, 1),
            round(self._tool_overlay_depth_strength, 3),
            vr_res,
            sbs_res,
            controller_model_name,
        )
        if self._fps_overlay_visible or self._hand_fps_visible:
            rgba = self._tool_quad_texture_cache.get("fps")
            if rgba is None or self._tool_quad_texture_keys.get("fps") != fps_key:
                if msdf_atlas is not None:
                    msdf_request = _build_msdf_fps_panel(
                        msdf_atlas,
                        actual_fps=self._tool_overlay_xr_fps,
                        sbs_fps=self._tool_overlay_sbs_fps,
                        capture_fps=self._tool_overlay_capture_fps,
                        latency_ms=self._tool_overlay_latency_ms,
                        screen_width=width,
                        screen_height=height,
                        screen_distance=screen_distance,
                        depth_strength=self._tool_overlay_depth_strength,
                        vr_res=vr_res,
                        sbs_res=sbs_res,
                        controller_brand=controller_model_name,
                        environment_visible=environment_mode,
                    )
                    if self._vulkan_msdf_quad_renderer is not None:
                        rgba = msdf_request
                    else:
                        from .overlay_textures import build_msdf_text_osd_rgba

                        rgba = build_msdf_text_osd_rgba(
                            msdf_atlas,
                            size=(msdf_request.width, msdf_request.height),
                            runs=msdf_request.runs,
                            background=msdf_request.background,
                            radius=int(msdf_request.radius),
                        )
                else:
                    rgba = build_fps_overlay_rgba(
                        actual_fps=self._tool_overlay_xr_fps,
                        sbs_fps=self._tool_overlay_sbs_fps,
                        capture_fps=self._tool_overlay_capture_fps,
                        latency_ms=self._tool_overlay_latency_ms,
                        screen_width=width,
                        screen_height=height,
                        screen_distance=screen_distance,
                        depth_strength=self._tool_overlay_depth_strength,
                        vr_res=vr_res,
                        sbs_res=sbs_res,
                        controller_brand=controller_model_name,
                        environment_visible=environment_mode,
                    )
                self._tool_quad_texture_cache["fps"] = rgba
                self._tool_quad_texture_keys["fps"] = fps_key

        if self._fps_overlay_visible:
            overlay_h = height / 8.0
            overlay_w = overlay_h * float(rgba.shape[1]) / max(1.0, float(rgba.shape[0]))
            local = np.eye(4, dtype=np.float64)
            local[:3, 3] = (
                -width / 2.0 + overlay_w / 2.0,
                -height / 2.0 - height * 0.02 - overlay_h / 2.0,
                0.0,
            )
            fps_position, fps_rotation = self._screen_overlay_pose(local)
            specs.append(("screen_fps", rgba, fps_position, (overlay_w, overlay_h), fps_rotation))

        if self._screen_operation_guide_visible:
            # Keep the guide's glyph layout proportional to its canvas. The
            # whole Quad already follows the screen height, so shrinking the
            # glyphs inversely with a large screen would leave a tiny island of
            # text in the middle of an otherwise empty guide panel.
            screen_guide_scale = 1.0
            help_key = (
                "screen_help",
                language,
                round(screen_guide_scale, 4),
                "gpu-msdf"
                if self._vulkan_msdf_quad_renderer is not None
                else ("cpu-msdf" if msdf_atlas is not None else "legacy"),
            )
            rgba = self._tool_quad_texture_cache.get("screen_help")
            if rgba is None or self._tool_quad_texture_keys.get("screen_help") != help_key:
                # This is the legacy screen-side vertical guide. The
                # controller-attached two-column guide is a different panel.
                if msdf_atlas is not None:
                    rows, _env_rows = get_controller_help_rows(language)
                    msdf_request = _build_msdf_help_panel(
                        msdf_atlas,
                        rows,
                        two_columns=False,
                        size_scale=screen_guide_scale,
                        canvas_scale=1.0,
                    )
                    if self._vulkan_msdf_quad_renderer is not None:
                        rgba = msdf_request
                    else:
                        from .overlay_textures import build_msdf_text_osd_rgba

                        rgba = build_msdf_text_osd_rgba(
                            msdf_atlas,
                            size=(msdf_request.width, msdf_request.height),
                            runs=msdf_request.runs,
                            background=msdf_request.background,
                            radius=int(msdf_request.radius),
                        )
                else:
                    rgba = build_team_help_rgba(lang=language)
                self._tool_quad_texture_cache["screen_help"] = rgba
                self._tool_quad_texture_keys["screen_help"] = help_key
            # The panel always follows the screen height. Its MSDF texture
            # layout is scaled independently so text remains proportional.
            panel_h = height
            panel_w = panel_h * float(rgba.shape[1]) / max(1.0, float(rgba.shape[0]))
            gap = height * 0.02
            head_local = screen_pose[:3, :3].T @ (head - screen_pose[:3, 3])
            hinge = np.asarray((-width / 2.0 - gap, 0.0, 0.0), dtype=np.float64)
            to_user = head_local - hinge
            to_user /= max(float(np.linalg.norm(to_user)), 1e-10)
            theta = math.atan2(float(to_user[0]), float(to_user[2]))
            hinge_rotation = np.eye(4, dtype=np.float64)
            hinge_rotation[0, 0] = math.cos(theta)
            hinge_rotation[0, 2] = math.sin(theta)
            hinge_rotation[2, 0] = -math.sin(theta)
            hinge_rotation[2, 2] = math.cos(theta)
            hinge_translation = np.eye(4, dtype=np.float64)
            hinge_translation[0, 3] = -width / 2.0 - gap
            panel_offset = np.eye(4, dtype=np.float64)
            panel_offset[0, 3] = -panel_w / 2.0
            panel_position, panel_rotation = self._overlay_pose_from_matrix(
                screen_pose @ hinge_translation @ hinge_rotation @ panel_offset
            )
            specs.append(("screen_help", rgba, panel_position, (panel_w, panel_h), panel_rotation))

        if self._vulkan_controller_proxy_enabled:
            callout_key = ("controller_proxy_callout", language)
            controller_callout = self._tool_quad_texture_cache.get(
                "controller_proxy_callout"
            )
            if (
                controller_callout is None
                or self._tool_quad_texture_keys.get("controller_proxy_callout")
                != callout_key
            ):
                controller_callout = build_controller_callout_rgba(lang=language)
                self._tool_quad_texture_cache[
                    "controller_proxy_callout"
                ] = controller_callout
                self._tool_quad_texture_keys[
                    "controller_proxy_callout"
                ] = callout_key
            geometry = self._controller_guide_geometry()
            if geometry is not None:
                callout_position, callout_size, callout_basis = geometry
                callout_rotation = tuple(
                    float(value) for value in _mat3_to_quat_xyzw(callout_basis)
                )
                specs.append(
                    (
                        "controller_proxy_callout",
                        controller_callout,
                        callout_position,
                        callout_size,
                        callout_rotation,
                    )
                )

        if self._hand_fps_visible:
            pose = self._controller_overlay_pose(0, 0.075, 0.10)
            if pose is not None:
                hand_position, hand_rotation = pose
                overlay_w = 0.075 * float(rgba.shape[1]) / max(1.0, float(rgba.shape[0]))
                specs.append(("hand_fps", rgba, hand_position, (overlay_w, 0.075), hand_rotation))

        if self._hand_operation_guide_visible:
            help_key = (
                "hand_help",
                language,
                environment_mode,
                "gpu-msdf"
                if self._vulkan_msdf_quad_renderer is not None
                else ("cpu-msdf" if msdf_atlas is not None else "legacy"),
            )
            hand_help = self._tool_quad_texture_cache.get("hand_help")
            if hand_help is None or self._tool_quad_texture_keys.get("hand_help") != help_key:
                if msdf_atlas is not None:
                    rows, env_rows = get_controller_help_rows(language)
                    selected_rows = env_rows if environment_mode else rows
                    msdf_request = _build_msdf_help_panel(
                        msdf_atlas, selected_rows, two_columns=True
                    )
                    if self._vulkan_msdf_quad_renderer is not None:
                        hand_help = msdf_request
                    else:
                        from .overlay_textures import build_msdf_text_osd_rgba

                        hand_help = build_msdf_text_osd_rgba(
                            msdf_atlas,
                            size=(msdf_request.width, msdf_request.height),
                            runs=msdf_request.runs,
                            background=msdf_request.background,
                            radius=int(msdf_request.radius),
                        )
                else:
                    hand_help = build_help_rgba(environment_mode=environment_mode, lang=language)
                self._tool_quad_texture_cache["hand_help"] = hand_help
                self._tool_quad_texture_keys["hand_help"] = help_key
            panel_h = 0.2
            panel_w = panel_h * float(hand_help.shape[1]) / max(1.0, float(hand_help.shape[0]))
            pose = self._controller_overlay_pose(1, panel_h, panel_h + 0.025)
            if pose is not None:
                hand_position, hand_rotation = pose
                specs.append(("hand_help", hand_help, hand_position, (panel_w, panel_h), hand_rotation))

        if self._aperture_visible:
            aperture_key = ("Aperture", "B: close", 384, 64)
            rgba = self._tool_quad_texture_cache.get("aperture")
            if rgba is None or self._tool_quad_texture_keys.get("aperture") != aperture_key:
                rgba = build_short_osd_rgba(("Aperture", "B: close"), width=384, height=64)
                self._tool_quad_texture_cache["aperture"] = rgba
                self._tool_quad_texture_keys["aperture"] = aperture_key
            position, rotation = self._overlay_pose_from_matrix(screen_pose)
            specs.append(("aperture", rgba, position, (width * 0.24, height * 0.06), rotation))

        if self._settings_menu.visible and self._settings_menu_pose is not None:
            menu_now = time.perf_counter()
            menu_rgba = self._tool_quad_texture_cache.get("settings_menu")
            if (
                menu_rgba is None
                or (
                    self._settings_menu.dirty
                    and menu_now - self._settings_menu_last_redraw >= 0.05
                )
            ):
                menu_rgba = build_settings_menu_rgba(
                    self._settings_menu,
                    self._settings_menu_values,
                    hover_key=self._settings_menu.hover_key,
                    cursor_uv=self._settings_menu_cursor_uv,
                    lang=language,
                )
                self._tool_quad_texture_cache["settings_menu"] = menu_rgba
                self._tool_quad_texture_keys["settings_menu"] = (
                    "settings_menu", self._settings_menu.revision
                )
                self._settings_menu.dirty = False
                self._settings_menu_last_redraw = menu_now
            menu_position, menu_rotation = self._settings_menu_pose
            specs.append((
                "settings_menu", menu_rgba, menu_position,
                SETTINGS_MENU_WORLD_SIZE, menu_rotation
            ))

        if not self._settings_menu.visible:
            cursor_rgba = self._tool_quad_texture_cache.get("laser_cursor")
            if cursor_rgba is None:
                cursor_rgba = build_cursor_rgba(64)
                self._tool_quad_texture_cache["laser_cursor"] = cursor_rgba
                self._tool_quad_texture_keys["laser_cursor"] = ("legacy_cursor_ring", 64)
            specs.extend(self._cursor_overlay_specs(cursor_rgba, screen_pose, head))

        if self._vulkan_controller_proxy_enabled:
            controller_callouts = [
                spec for spec in specs if spec[0] == "controller_proxy_callout"
            ]
            specs = [
                spec for spec in specs if spec[0] != "controller_proxy_callout"
            ] + controller_callouts
        specs_ready = time.perf_counter()
        if not os.environ.get("D2S_OPENXR_ENABLE_TOOL_QUADS"):
            # Virtual Desktop's runtime fails MSDF tool-quad swapchain
            # enumeration (RuntimeFailureError) which precedes session drops.
            # The quads are cosmetic overlays (menu / cursor / callouts), never
            # the screen or the glow layers. Re-enable with
            # D2S_OPENXR_ENABLE_TOOL_QUADS=1.
            return []
        layers = [self._upload_tool_quad(*spec) for spec in specs]
        if self._on_breakdown_add_time is not None:
            self._on_breakdown_add_time(
                "openxr_quad_upload", time.perf_counter() - specs_ready
            )
        return layers

    def _cache_head_position(self, views: list[Any]) -> None:
        if len(views) < 2:
            self._head_position_w = None
            self._head_forward_w = None
            self._head_model_matrix = None
            return
        eye_positions = [
            np.asarray(
                (view.pose.position.x, view.pose.position.y, view.pose.position.z),
                dtype=np.float64,
            )
            for view in views[:2]
        ]
        self._head_position_w = (eye_positions[0] + eye_positions[1]) * 0.5
        head_matrix = _xr_view_pose_to_model_mat4(views[0].pose)
        self._head_model_matrix = np.asarray(head_matrix, dtype=np.float64)
        self._head_forward_w = -head_matrix[:3, 2].astype(np.float64)
        if self._head_position_w is not None:
            self._initial_head_y = float(self._head_position_w[1])

    def _report_profile_alignment(self) -> None:
        """Log the calibrated head pose against the authored GLB-local target once."""
        if self._profile_alignment_logged:
            return
        if not self._profile_space_applied:
            return
        if self._profile_head_transform is None or self._head_position_w is None:
            return
        target = np.asarray(self._profile_head_transform[:3, 3], dtype=np.float64)
        actual = np.asarray(self._head_position_w, dtype=np.float64)
        delta = actual - target
        self._profile_alignment_logged = True
        print(
            "[OpenXRViewer] profile head alignment: "
            f"target_glb=({target[0]:.3f},{target[1]:.3f},{target[2]:.3f}) "
            f"actual_xr=({actual[0]:.3f},{actual[1]:.3f},{actual[2]:.3f}) "
            f"delta=({delta[0]:+.3f},{delta[1]:+.3f},{delta[2]:+.3f}) "
            f"reference_space={getattr(self._reference_space_type, 'name', self._reference_space_type)}",
            flush=True,
        )

    def _initialize_filament_screen_from_head(self) -> None:
        """Initialize an unauthored screen with the legacy head-centered preset."""
        if (
            self._filament_screen_head_initialized
            or self._filament_screen_profile_authored
            or self._filament_screen is None
            or self._head_position_w is None
            or self._head_forward_w is None
        ):
            return
        self._shortcut_screen_preset_index = 5
        self._apply_shortcut_screen_preset(5)
        self._filament_screen_initial = self._filament_screen
        self._filament_screen_head_initialized = True

    def _controller_guide_geometry(self):
        """Return the world-space panel geometry for the Projection Layer guide."""
        if self._grip_mat_r is None or self._head_position_w is None:
            return None
        controller_position = np.asarray(self._grip_mat_r[:3, 3], dtype=np.float64)
        to_head = np.asarray(self._head_position_w, dtype=np.float64) - controller_position
        distance = float(np.linalg.norm(to_head))
        if distance <= 1e-6 or distance > self.config.controller_guide_max_distance:
            return None

        def normalized(vector):
            vector = np.asarray(vector, dtype=np.float64)
            return vector / max(float(np.linalg.norm(vector)), 1e-6)

        button_position = self._controller_b_button_world_position()
        if button_position is None:
            button_position = controller_position
        forward = normalized(np.asarray(self._head_position_w, dtype=np.float64) - button_position)
        world_up = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
        right = normalized(np.cross(world_up, forward))
        up = normalized(np.cross(forward, right))

        # Keep the Quad head-facing while solving its center from the B button
        # world position and the callout endpoint's local texture coordinate.
        endpoint_x = (540.0 / 1024.0 - 0.5) * 0.34
        endpoint_y = (0.5 - 300.0 / 768.0) * 0.255
        panel_position = (
            button_position
            - right * endpoint_x
            - up * endpoint_y
            + forward * 0.006
        )
        basis = np.column_stack((right, up, forward))
        return (
            tuple(float(value) for value in panel_position),
            (0.34, 0.255),
            basis,
        )

    def _controller_guide_pose(self):
        """Return the legacy pose representation used by geometry tests."""
        geometry = self._controller_guide_geometry()
        if geometry is None:
            return None
        position, size, basis = geometry
        rotation = _mat3_to_quat_xyzw(basis)
        return (
            position,
            size,
            tuple(float(value) for value in rotation),
        )

    def _resolve_controller_b_button_local(
        self, *, force: bool = False
    ) -> np.ndarray | None:
        if force:
            controller_button_local_position.cache_clear()
            self._controller_b_button_local = None
            self._controller_b_button_resolved = False
        if self._vulkan_controller_proxy_enabled:
            self._controller_b_button_local = (
                np.asarray((0.040, 0.040, -0.040), dtype=np.float64)
            )
            self._controller_b_button_resolved = True
            return self._controller_b_button_local
        if self._controller_brand is None:
            self._controller_b_button_local = None
            self._controller_b_button_resolved = True
            return None
        if not self._controller_b_button_resolved:
            if self._controller_brand.right_glb is None:
                return None
            resolved = controller_button_local_position(
                str(self._controller_brand.right_glb), "b_button"
            )
            self._controller_b_button_local = (
                None if resolved is None else np.asarray(resolved, dtype=np.float64)
            )
            self._controller_b_button_resolved = True
        return self._controller_b_button_local

    def _controller_b_button_world_position(self):
        if self._grip_mat_r is None:
            return None
        button_local = self._resolve_controller_b_button_local()
        if button_local is None:
            return None

        model_matrix = np.asarray(self._grip_mat_r, dtype=np.float64)
        offset = np.eye(4, dtype=np.float64)
        offset[:3, 3] = np.asarray(
            self._controller_calibration_offset, dtype=np.float64
        )
        rotation = euler_to_mat4(
            0.0, math.radians(self._controller_calibration_rotation_deg), 0.0
        ).astype(np.float64)
        model_matrix = model_matrix @ rotation @ offset
        local = np.ones(4, dtype=np.float64)
        local[:3] = button_local
        return (model_matrix @ local)[:3]

    def _upload_msdf_tool_quad(self, key, request, position, size, rotation):
        renderer = self._vulkan_msdf_quad_renderer
        if renderer is None:
            raise RuntimeError("Vulkan MSDF Quad renderer is unavailable")
        height, width = int(request.height), int(request.width)
        format_value = self._tool_quad_format()
        if not renderer.supports_destination_format(format_value):
            from .overlay_textures import build_msdf_text_osd_rgba

            return self._upload_tool_quad(
                key,
                build_msdf_text_osd_rgba(
                    self._msdf_font_atlas,
                    size=(width, height),
                    runs=request.runs,
                    background=request.background,
                    radius=int(request.radius),
                ),
                position,
                size,
                rotation,
            )
        entry = self._overlay_quad_entries.get(key)
        if (
            entry is None
            or entry["size"] != (width, height)
            or entry.get("format") != format_value
        ):
            if entry is not None:
                staging = entry.get("staging")
                if staging is not None:
                    staging.close()
                for resource in reversed(entry["resources"]):
                    self.vulkan.unregister_external_image(resource)
                self.xr.destroy_swapchain(entry["swapchain"])
            swapchain = self.xr.create_swapchain(
                self.session,
                self.xr.SwapchainCreateInfo(
                    usage_flags=(
                        self.xr.SwapchainUsageFlags.COLOR_ATTACHMENT_BIT
                        | self.xr.SwapchainUsageFlags.TRANSFER_DST_BIT
                    ),
                    format=format_value,
                    sample_count=1,
                    width=width,
                    height=height,
                    face_count=1,
                    array_size=1,
                    mip_count=1,
                ),
            )
            images = list(
                self.xr.enumerate_swapchain_images(
                    swapchain, self.xr.SwapchainImageVulkan2KHR
                )
            )
            entry = {
                "swapchain": swapchain,
                "size": (width, height),
                "format": format_value,
                "resources": self._register_swapchain_images(
                    images, width, height, format_value
                ),
                "staging": None,
                "image_index": None,
                "content": None,
            }
            self._overlay_quad_entries[key] = entry
        if entry.get("content") is not request or entry.get("image_index") is None:
            with _acquired_swapchain_image(
                self.xr,
                _EyeSwapchain(
                    entry["swapchain"], [], width, height, entry["resources"]
                ),
            ) as image_index:
                rendered = renderer.render(
                    request, destination_format=int(entry["format"])
                )
                timeline = self.vulkan.copy_image(
                    rendered, entry["resources"][image_index]
                )
                renderer.notify_copy_timeline(timeline)
                entry["image_index"] = image_index
            entry["content"] = request
        image_index = int(entry["image_index"])
        if len(rotation) == 4:
            qx, qy, qz, qw = (float(value) for value in rotation)
        else:
            qx, qy, qz, qw = _euler_degrees_to_quaternion(rotation)
        return self.xr.CompositionLayerQuad(
            layer_flags=(
                self.xr.CompositionLayerFlags.BLEND_TEXTURE_SOURCE_ALPHA_BIT
                | self.xr.CompositionLayerFlags.UNPREMULTIPLIED_ALPHA_BIT
            ),
            space=self.reference_space,
            eye_visibility=self.xr.EyeVisibility.BOTH,
            sub_image=self.xr.SwapchainSubImage(
                swapchain=entry["swapchain"],
                image_rect=self.xr.Rect2Di(
                    offset=self.xr.Offset2Di(x=0, y=0),
                    extent=self.xr.Extent2Di(width=width, height=height),
                ),
                image_array_index=0,
            ),
            pose=self.xr.Posef(
                orientation=self.xr.Quaternionf(x=qx, y=qy, z=qz, w=qw),
                position=self.xr.Vector3f(
                    x=float(position[0]), y=float(position[1]), z=float(position[2])
                ),
            ),
            size=self.xr.Extent2Df(width=float(size[0]), height=float(size[1])),
        )

    def _upload_tool_quad(self, key, rgba, position, size, rotation):
        if isinstance(rgba, VulkanMsdfQuadRequest):
            return self._upload_msdf_tool_quad(key, rgba, position, size, rotation)
        height, width = int(rgba.shape[0]), int(rgba.shape[1])
        entry = self._overlay_quad_entries.get(key)
        format_value = self._tool_quad_format()
        if (
            entry is None
            or entry["size"] != (width, height)
            or entry.get("format") != format_value
        ):
            if entry is not None:
                staging = entry.get("staging")
                if staging is not None:
                    staging.close()
                for resource in reversed(entry["resources"]):
                    self.vulkan.unregister_external_image(resource)
                self.xr.destroy_swapchain(entry["swapchain"])
            swapchain = self.xr.create_swapchain(
                self.session,
                self.xr.SwapchainCreateInfo(
                    usage_flags=(self.xr.SwapchainUsageFlags.COLOR_ATTACHMENT_BIT | self.xr.SwapchainUsageFlags.TRANSFER_DST_BIT),
                    format=format_value, sample_count=1, width=width, height=height,
                    face_count=1, array_size=1, mip_count=1,
                ),
            )
            images = list(self.xr.enumerate_swapchain_images(swapchain, self.xr.SwapchainImageVulkan2KHR))
            entry = {
                "swapchain": swapchain,
                "size": (width, height),
                "format": format_value,
                "resources": self._register_swapchain_images(images, width, height, format_value),
                "staging": VulkanHostImage(self.vulkan, width, height, format=format_value, label=f"overlay-{key}"),
                "image_index": None,
                "content": None,
            }
            self._overlay_quad_entries[key] = entry
        if entry.get("content") is not rgba or entry.get("image_index") is None:
            entry["staging"].upload(rgba)
            with _acquired_swapchain_image(self.xr, _EyeSwapchain(entry["swapchain"], [], width, height, entry["resources"])) as image_index:
                self.vulkan.copy_image(entry["staging"].resource, entry["resources"][image_index])
                entry["image_index"] = image_index
            entry["content"] = rgba
        image_index = int(entry["image_index"])
        if len(rotation) == 4:
            qx, qy, qz, qw = (float(value) for value in rotation)
        else:
            qx, qy, qz, qw = _euler_degrees_to_quaternion(rotation)
        return self.xr.CompositionLayerQuad(
            layer_flags=(
                self.xr.CompositionLayerFlags.BLEND_TEXTURE_SOURCE_ALPHA_BIT
                | self.xr.CompositionLayerFlags.UNPREMULTIPLIED_ALPHA_BIT
            ),
            space=self.reference_space,
            eye_visibility=self.xr.EyeVisibility.BOTH,
            sub_image=self.xr.SwapchainSubImage(
                swapchain=entry["swapchain"],
                image_rect=self.xr.Rect2Di(offset=self.xr.Offset2Di(x=0, y=0), extent=self.xr.Extent2Di(width=width, height=height)),
                image_array_index=0,
            ),
            pose=self.xr.Posef(
                orientation=self.xr.Quaternionf(x=qx, y=qy, z=qz, w=qw),
                position=self.xr.Vector3f(x=float(position[0]), y=float(position[1]), z=float(position[2])),
            ),
            size=self.xr.Extent2Df(width=float(size[0]), height=float(size[1])),
        )

    def _tool_quad_format(self) -> int:
        if self._tool_quad_swapchain_format is None:
            formats = self.xr.enumerate_swapchain_formats(self.session)
            self._tool_quad_swapchain_format = _select_swapchain_format(
                self.vulkan.vk, list(formats), "srgb"
            )
        return int(self._tool_quad_swapchain_format)


@contextmanager
def _acquired_swapchain_image(xr: Any, eye: _EyeSwapchain):
    """Guarantee release after every successful acquire, including wait errors."""

    image_index = xr.acquire_swapchain_image(eye.handle)
    try:
        xr.wait_swapchain_image(
            eye.handle,
            xr.SwapchainImageWaitInfo(timeout=xr.INFINITE_DURATION),
        )
        yield image_index
    finally:
        xr.release_swapchain_image(eye.handle)


def _xr_view_pose_to_model_mat4(pose: Any) -> np.ndarray:
    matrix = _xr_quat_to_mat4(pose.orientation).astype(np.float32)
    matrix[:3, 3] = (
        float(pose.position.x),
        float(pose.position.y),
        float(pose.position.z),
    )
    return matrix


def _euler_degrees_to_quaternion(rotation: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """Convert legacy profile yaw/pitch/roll degrees to OpenXR xyzw."""
    yaw, pitch, roll = (
        math.radians(float(value)) for value in rotation[:3]
    )
    matrix = euler_to_mat4(yaw, pitch, roll)
    return tuple(float(value) for value in _mat3_to_quat_xyzw(matrix[:3, :3]))


def _update_filament_camera(
    bridge: Any,
    view: Any,
    *,
    near_plane: float = 0.05,
    far_plane: float = 1000.0,
) -> None:
    pose = view.pose
    rotation = _xr_quat_to_mat4(pose.orientation)[:3, :3]
    position = (
        float(pose.position.x),
        float(pose.position.y),
        float(pose.position.z),
    )
    forward = rotation @ (0.0, 0.0, -1.0)
    up = rotation @ (0.0, 1.0, 0.0)
    center = tuple(position[index] + float(forward[index]) for index in range(3))
    bridge.set_camera_look_at(position, center, tuple(float(value) for value in up))

    fov = view.fov
    left = math.tan(float(fov.angle_left)) * near_plane
    right = math.tan(float(fov.angle_right)) * near_plane
    bottom = math.tan(float(fov.angle_down)) * near_plane
    top = math.tan(float(fov.angle_up)) * near_plane
    if hasattr(bridge, "set_camera_projection_frustum"):
        bridge.set_camera_projection_frustum(
            left, right, bottom, top,
            near_plane=near_plane,
            far_plane=far_plane,
        )
        return
    horizontal = max(0.01, abs(float(fov.angle_right) - float(fov.angle_left)))
    vertical = max(0.01, abs(float(fov.angle_up) - float(fov.angle_down)))
    aspect = math.tan(horizontal * 0.5) / max(math.tan(vertical * 0.5), 1e-6)
    bridge.set_camera_projection(
        math.degrees(vertical),
        aspect,
        near_plane=near_plane,
        far_plane=far_plane,
    )


def _update_filament_stereo_camera(
    bridge: Any,
    views: list[Any],
    *,
    near_plane: float = 0.05,
    far_plane: float = 1000.0,
) -> None:
    eye_models = [
        _xr_view_pose_to_model_mat4(view.pose) for view in views[:2]
    ]
    head_model = eye_models[0].copy()
    head_model[:3, 3] = 0.5 * (
        eye_models[0][:3, 3] + eye_models[1][:3, 3]
    )
    head_inverse = np.linalg.inv(head_model).astype(np.float32)
    position = tuple(float(value) for value in head_model[:3, 3])
    forward = head_model[:3, :3] @ (0.0, 0.0, -1.0)
    up = head_model[:3, :3] @ (0.0, 1.0, 0.0)
    center = tuple(position[index] + float(forward[index]) for index in range(3))
    bridge.set_camera_look_at(
        position, center, tuple(float(value) for value in up)
    )

    matrices: list[float] = []
    frustums: list[float] = []
    for view, eye_model in zip(views[:2], eye_models):
        matrices.extend(
            float(value)
            for value in (head_inverse @ eye_model).reshape(-1, order="F")
        )
        fov = view.fov
        frustums.extend(
            (
                math.tan(float(fov.angle_left)) * near_plane,
                math.tan(float(fov.angle_right)) * near_plane,
                math.tan(float(fov.angle_down)) * near_plane,
                math.tan(float(fov.angle_up)) * near_plane,
            )
        )
    bridge.set_stereo_camera(
        matrices,
        frustums,
        near_plane=near_plane,
        far_plane=far_plane,
    )


def _import_openxr() -> Any:
    try:
        import xr
    except (ImportError, OSError) as exc:
        raise OpenXrVulkanUnavailableError(
            "pyopenxr or the OpenXR loader is unavailable"
        ) from exc
    return xr


def _get_vulkan_graphics_requirements2(
    xr: Any, instance: Any, system_id: Any
) -> Any:
    function = ctypes.cast(
        xr.get_instance_proc_addr(
            instance.instance, "xrGetVulkanGraphicsRequirements2KHR"
        ),
        xr.platform.PFN_xrGetVulkanGraphicsRequirements2KHR,
    )
    requirements = xr.GraphicsRequirementsVulkan2KHR()
    result = xr.check_result(function(instance, system_id, ctypes.byref(requirements)))
    if result.is_exception():
        raise result
    return requirements


def _select_vulkan_api_version(requirements: Any, requested: int) -> int:
    minimum = make_vulkan_version(
        requirements.min_api_version_supported.major,
        requirements.min_api_version_supported.minor,
        requirements.min_api_version_supported.patch,
    )
    maximum = make_vulkan_version(
        requirements.max_api_version_supported.major,
        requirements.max_api_version_supported.minor,
        requirements.max_api_version_supported.patch,
    )
    if minimum > maximum:
        raise OpenXrVulkanUnavailableError(
            "OpenXR runtime returned an invalid Vulkan API version range"
        )
    if maximum < MIN_VULKAN_API_VERSION:
        raise OpenXrVulkanUnavailableError(
            "OpenXR runtime does not support the required Vulkan 1.2 minimum"
        )
    selected = max(minimum, min(int(requested), maximum))
    if selected < MIN_VULKAN_API_VERSION:
        raise OpenXrVulkanUnavailableError(
            "Negotiated Vulkan API version is below the required Vulkan 1.2 minimum"
        )
    return selected


def _select_swapchain_format(
    vk: Any, available_formats: list[int], color_mode: str = "srgb"
) -> int:
    mode = str(color_mode or "srgb").strip().lower()
    if mode not in {"srgb", "auto"}:
        raise ValueError(
            "OpenXR projection swapchain must use sRGB; "
            "linear UNORM output is not supported"
        )

    srgb = (
        vk.VK_FORMAT_R8G8B8A8_SRGB,
        vk.VK_FORMAT_B8G8R8A8_SRGB,
    )
    preferred = srgb
    for candidate in preferred:
        if int(candidate) in available_formats:
            return int(candidate)
    if available_formats:
        raise OpenXrVulkanUnavailableError(
            "OpenXR runtime exposes no sRGB projection swapchain format; "
            "refusing a color-space-changing UNORM fallback"
        )
    if not available_formats:
        raise OpenXrVulkanUnavailableError(
            "OpenXR runtime returned no swapchain formats"
        )
    return int(available_formats[0])


def _vulkan_format_name(vk: Any, value: int) -> str:
    names = {
        int(vk.VK_FORMAT_R8G8B8A8_SRGB): "R8G8B8A8_SRGB",
        int(vk.VK_FORMAT_B8G8R8A8_SRGB): "B8G8R8A8_SRGB",
        int(vk.VK_FORMAT_R8G8B8A8_UNORM): "R8G8B8A8_UNORM",
        int(vk.VK_FORMAT_B8G8R8A8_UNORM): "B8G8R8A8_UNORM",
    }
    return names.get(int(value), "runtime-preferred")


def _scaled_dimension(recommended: int, maximum: int, scale: float) -> int:
    return max(1, min(int(maximum), round(int(recommended) * float(scale))))


def _openxr_platform_module(xr: Any) -> Any:
    return importlib.import_module(xr.VulkanInstanceCreateInfoKHR.__module__)


def _load_vulkan_proc_addr(xr: Any) -> tuple[Any, Any]:
    if sys.platform == "win32":
        candidates = ["vulkan-1.dll"]
    elif sys.platform == "darwin":
        candidates = ["libvulkan.1.dylib", "libvulkan.dylib", "libMoltenVK.dylib"]
    else:
        candidates = ["libvulkan.so.1", "libvulkan.so"]
    discovered = ctypes.util.find_library("vulkan")
    if discovered:
        candidates.append(discovered)

    platform = _openxr_platform_module(xr)
    errors: list[str] = []
    for candidate in dict.fromkeys(candidates):
        try:
            loader = (
                ctypes.WinDLL(candidate)
                if sys.platform == "win32"
                else ctypes.CDLL(candidate)
            )
            function = ctypes.cast(
                loader.vkGetInstanceProcAddr, platform.PFN_vkGetInstanceProcAddr
            )
            return loader, function
        except (AttributeError, OSError) as exc:
            errors.append(f"{candidate}: {exc}")
    raise OpenXrVulkanUnavailableError(
        "Unable to load vkGetInstanceProcAddr: " + "; ".join(errors)
    )


def _cffi_struct_pointer(vk: Any, value: Any, ctypes_type: Any) -> Any:
    address = int(vk.ffi.cast("uintptr_t", vk.ffi.addressof(value)))
    return ctypes.cast(ctypes.c_void_p(address), ctypes.POINTER(ctypes_type))


def _ctypes_handle_to_cffi(vk: Any, type_name: str, handle: Any) -> Any:
    address = _ctypes_handle_address(handle)
    if not address:
        raise OpenXrVulkanUnavailableError(f"OpenXR returned a null {type_name}")
    return vk.ffi.cast(type_name, address)


def _ctypes_handle_address(handle: Any) -> int:
    return int(ctypes.cast(handle, ctypes.c_void_p).value or 0)


def _check_vulkan_result(result: Any, operation: str) -> None:
    value = int(result.value if hasattr(result, "value") else result)
    if value != 0:
        raise OpenXrVulkanUnavailableError(f"{operation} returned VkResult {value}")


def _decode_name(value: Any) -> str:
    if isinstance(value, bytes):
        return value.split(b"\0", 1)[0].decode("utf-8", errors="replace")
    return str(value)
