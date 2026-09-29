from __future__ import annotations

import ctypes
from dataclasses import replace
import inspect
import json
import math
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import vulkan as vk

from path_config import APP_ROOT
import xr

from app_runtime.output_contract import VulkanStereoOutputFrame
from xr_viewer import core_input_helpers
from xr_viewer import core_openxr_vulkan
from xr_viewer.core_input_helpers import CoreInputHelpersMixin
from viewer.vulkan_context import (
    ImageState,
    ImageStateTracker,
    QueueFamilySelection,
    VulkanCapabilityError,
    VulkanContext,
    VulkanTimelineTimeout,
    VulkanUnavailableError,
    _require_timeline_semaphore_features,
    format_vulkan_version,
    make_vulkan_version,
    unpack_vulkan_version,
    _find_queue_families,
)
from viewer.vulkan_resources import VulkanImageResource
from viewer.vulkan_projection_screen import VulkanProjectionScreenPass
from xr_viewer.core_openxr_vulkan import (
    OpenXrCompositionBuilder,
    OpenXrVulkanConfig,
    OpenXrVulkanPresenter,
    OpenXrVulkanUnavailableError,
    _EyeSwapchain,
    _xr_view_pose_to_model_mat4,
    _layout_msdf_osd_runs,
    _build_msdf_help_panel,
    _scaled_dimension,
    _select_swapchain_format,
    _select_vulkan_api_version,
    _sbs_capture_options,
    _update_filament_camera,
    _update_filament_stereo_camera,
    _vulkan_rgba_to_rgb,
)
from xr_viewer.controller_models import controller_button_local_position
from xr_viewer.overlay_textures import build_controller_callout_rgba, build_keyboard_rgba
from xr_viewer.msdf_font_atlas import MsdfFontAtlas
from viewer.controller_help import get_controller_help_rows
from viewer.vulkan_msdf_quad import VulkanMsdfQuadRequest
from xr_viewer.xr_math import _xr_quat_to_mat4, euler_to_mat4, mat4_to_xr_posef
from utils.screen_resolution_policy import build_screen_sampling_plan


def test_vulkan_version_round_trip() -> None:
    packed = make_vulkan_version(1, 3, 275)
    assert unpack_vulkan_version(packed) == (1, 3, 275)
    assert format_vulkan_version(packed) == "1.3.275"


def test_panorama_push_constants_are_translation_free_and_rotation_aware() -> None:
    presenter = OpenXrVulkanPresenter.__new__(OpenXrVulkanPresenter)
    presenter._profile_near_plane = 0.05
    presenter._profile_far_plane = 100.0
    presenter._panorama_rotation_only_logged = True
    fov = xr.Fovf(
        angle_left=-0.7, angle_right=0.7,
        angle_up=0.7, angle_down=-0.7,
    )

    def payload(position, orientation):
        view = SimpleNamespace(
            pose=SimpleNamespace(position=position, orientation=orientation),
            fov=fov,
        )
        return presenter._panorama_push_constants(view)

    identity = xr.Quaternionf(x=0.0, y=0.0, z=0.0, w=1.0)
    moved = payload(SimpleNamespace(x=4.0, y=-2.0, z=8.0), identity)
    origin = payload(SimpleNamespace(x=0.0, y=0.0, z=0.0), identity)
    yaw = math.radians(45.0) * 0.5
    rotated = payload(
        SimpleNamespace(x=0.0, y=0.0, z=0.0),
        xr.Quaternionf(x=0.0, y=math.sin(yaw), z=0.0, w=math.cos(yaw)),
    )

    assert len(origin) == 32
    assert moved == origin
    assert rotated[:16] == origin[:16]
    assert rotated[16:] != origin[16:]

    initial_uv = presenter._panorama_center_uv(
        fov, np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
    )
    rotated_uv = presenter._panorama_center_uv(
        fov,
        np.asarray((0.0, math.sin(yaw), 0.0, math.cos(yaw)), dtype=np.float64),
    )
    assert initial_uv == pytest.approx((0.5, 0.5), abs=1e-6)
    assert rotated_uv[0] == pytest.approx(0.375, abs=1e-6)
    assert rotated_uv[1] == pytest.approx(0.5, abs=1e-6)


def test_timeline_feature_chain_returns_feature_node_and_sync_flag():
    class FeatureNode:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeVulkan:
        VK_TRUE = 1
        VK_FALSE = 0
        VkPhysicalDeviceTimelineSemaphoreFeatures = FeatureNode
        VkPhysicalDeviceSynchronization2Features = FeatureNode
        VkPhysicalDeviceFeatures2 = FeatureNode

        @staticmethod
        def vkGetPhysicalDeviceFeatures2(_physical_device, features2):
            features2.pNext.timelineSemaphore = 1
            features2.pNext.pNext.synchronization2 = 1

    feature_chain, synchronization2_enabled = _require_timeline_semaphore_features(
        FakeVulkan(), object()
    )

    assert feature_chain.synchronization2 == 1
    assert feature_chain.pNext.timelineSemaphore == 1
    assert synchronization2_enabled is True


def test_openxr_feature_chain_enables_multiview():
    class FeatureNode:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeVulkan:
        VK_TRUE = 1
        VK_FALSE = 0
        VkPhysicalDeviceTimelineSemaphoreFeatures = FeatureNode
        VkPhysicalDeviceSynchronization2Features = FeatureNode
        VkPhysicalDeviceMultiviewFeatures = FeatureNode
        VkPhysicalDeviceFeatures2 = FeatureNode

        @staticmethod
        def vkGetPhysicalDeviceFeatures2(_physical_device, features2):
            features2.pNext.timelineSemaphore = 1
            features2.pNext.pNext.synchronization2 = 1
            features2.pNext.pNext.pNext.multiview = 1

    feature_chain, synchronization2_enabled = _require_timeline_semaphore_features(
        FakeVulkan(), object(), require_multiview=True
    )

    assert feature_chain.synchronization2 == 1
    assert feature_chain.pNext.multiview == 1
    assert feature_chain.pNext.pNext.timelineSemaphore == 1
    assert synchronization2_enabled is True


def test_queue_family_selection_prefers_dedicated_compute_and_transfer() -> None:
    vk = SimpleNamespace(
        VK_QUEUE_GRAPHICS_BIT=0x1,
        VK_QUEUE_COMPUTE_BIT=0x2,
        VK_QUEUE_TRANSFER_BIT=0x4,
        vkGetPhysicalDeviceQueueFamilyProperties=lambda _device: [
            SimpleNamespace(queueCount=1, queueFlags=0x1 | 0x2),
            SimpleNamespace(queueCount=1, queueFlags=0x2),
            SimpleNamespace(queueCount=1, queueFlags=0x4),
        ],
    )
    assert _find_queue_families(vk, object()) == QueueFamilySelection(0, 1, 2)


def test_queue_family_selection_falls_back_to_graphics() -> None:
    vk = SimpleNamespace(
        VK_QUEUE_GRAPHICS_BIT=0x1,
        VK_QUEUE_COMPUTE_BIT=0x2,
        VK_QUEUE_TRANSFER_BIT=0x4,
        vkGetPhysicalDeviceQueueFamilyProperties=lambda _device: [
            SimpleNamespace(queueCount=1, queueFlags=0x1 | 0x2 | 0x4),
        ],
    )
    assert _find_queue_families(vk, object()) == QueueFamilySelection(0, 0, 0)


def test_image_state_tracker_returns_explicit_undefined_state() -> None:
    tracker = ImageStateTracker(default_queue_family_index=3)
    state = tracker.get(17, undefined_layout=9)
    assert state == ImageState(9, 0, 0, 3)


def test_image_state_tracker_updates_and_clears_state() -> None:
    tracker = ImageStateTracker(default_queue_family_index=3)
    state = ImageState(4, 8, 16, 2)
    tracker.update(17, state)
    assert tracker.get(17, undefined_layout=9) == state
    tracker.clear()
    assert tracker.get(17, undefined_layout=9).layout == 9


def test_image_state_tracker_removes_released_image() -> None:
    tracker = ImageStateTracker(default_queue_family_index=3)
    tracker.update(17, ImageState(4, 8, 16, 3))
    tracker.remove(17)
    assert tracker.get(17, undefined_layout=9) == ImageState(9, 0, 0, 3)


def test_image_state_tracker_rejects_wrong_queue_owner() -> None:
    tracker = ImageStateTracker(default_queue_family_index=3)
    tracker.update(17, ImageState(4, 8, 16, 2))
    with pytest.raises(VulkanCapabilityError, match="owned by queue family 2"):
        tracker.require_owner(17, 3)


def test_image_state_tracker_owns_pending_queue_transfer_until_acquire() -> None:
    tracker = ImageStateTracker(default_queue_family_index=0)
    tracker.update(17, ImageState(4, 8, 16, 0))
    transfer = tracker.begin_ownership_transfer(
        17,
        source_queue_family_index=0,
        destination_queue_family_index=2,
        undefined_layout=9,
    )
    with pytest.raises(VulkanCapabilityError, match="pending queue ownership"):
        tracker.require_owner(17, 2)
    tracker.complete_ownership_transfer(transfer)
    assert tracker.require_owner(17, 2).queue_family_index == 2


def test_openxr_version_range_clamps_requested_vulkan_version() -> None:
    requirements = SimpleNamespace(
        min_api_version_supported=xr.Version(1, 2, 0),
        max_api_version_supported=xr.Version(1, 3, 0),
    )
    assert _select_vulkan_api_version(
        requirements, make_vulkan_version(1, 0, 0)
    ) == make_vulkan_version(1, 2, 0)
    assert _select_vulkan_api_version(
        requirements, make_vulkan_version(1, 2, 0)
    ) == make_vulkan_version(1, 2, 0)
    assert _select_vulkan_api_version(
        requirements, make_vulkan_version(1, 4, 0)
    ) == make_vulkan_version(1, 3, 0)


def test_invalid_openxr_version_range_is_rejected() -> None:
    requirements = SimpleNamespace(
        min_api_version_supported=xr.Version(1, 3, 0),
        max_api_version_supported=xr.Version(1, 1, 0),
    )
    with pytest.raises(OpenXrVulkanUnavailableError):
        _select_vulkan_api_version(requirements, make_vulkan_version(1, 2, 0))


def test_openxr_runtime_below_vulkan_12_is_rejected() -> None:
    requirements = SimpleNamespace(
        min_api_version_supported=xr.Version(1, 0, 0),
        max_api_version_supported=xr.Version(1, 1, 0),
    )
    with pytest.raises(OpenXrVulkanUnavailableError, match="Vulkan 1.2"):
        _select_vulkan_api_version(requirements, make_vulkan_version(1, 4, 0))


def test_swapchain_format_prefers_srgb() -> None:
    vk = SimpleNamespace(
        VK_FORMAT_R8G8B8A8_SRGB=43,
        VK_FORMAT_B8G8R8A8_SRGB=50,
        VK_FORMAT_R8G8B8A8_UNORM=37,
        VK_FORMAT_B8G8R8A8_UNORM=44,
    )
    assert _select_swapchain_format(vk, [44, 50, 43]) == 43
    with pytest.raises(OpenXrVulkanUnavailableError, match="no sRGB"):
        _select_swapchain_format(vk, [44])
    with pytest.raises(OpenXrVulkanUnavailableError):
        _select_swapchain_format(vk, [])


def test_swapchain_format_rejects_linear_unorm_mode() -> None:
    vk = SimpleNamespace(
        VK_FORMAT_R8G8B8A8_SRGB=43,
        VK_FORMAT_B8G8R8A8_SRGB=50,
        VK_FORMAT_R8G8B8A8_UNORM=37,
        VK_FORMAT_B8G8R8A8_UNORM=44,
    )
    with pytest.raises(ValueError, match="must use sRGB"):
        _select_swapchain_format(vk, [43, 44], "unorm")
    assert _select_swapchain_format(vk, [43, 44], "srgb") == 43
    assert _select_swapchain_format(vk, [43, 44], "auto") == 43


def test_swapchain_color_mode_rejects_unknown_value() -> None:
    vk = SimpleNamespace(
        VK_FORMAT_R8G8B8A8_SRGB=43,
        VK_FORMAT_B8G8R8A8_SRGB=50,
        VK_FORMAT_R8G8B8A8_UNORM=37,
        VK_FORMAT_B8G8R8A8_UNORM=44,
    )
    with pytest.raises(ValueError, match="must use sRGB"):
        _select_swapchain_format(vk, [43, 44], "linear")


def test_render_scale_is_bounded_by_runtime_limit() -> None:
    assert _scaled_dimension(1000, 1200, 0.5) == 500
    assert _scaled_dimension(1000, 1200, 2.0) == 1200
    assert _scaled_dimension(1000, 5000, 4.0) == 4000
    assert _scaled_dimension(1, 1, 0.1) == 1


def test_projection_swapchains_use_runtime_recommendation_times_render_scale() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._view_configuration_views = (
        SimpleNamespace(
            recommended_image_rect_width=1000,
            recommended_image_rect_height=800,
            max_image_rect_width=1800,
            max_image_rect_height=1400,
        ),
        SimpleNamespace(
            recommended_image_rect_width=900,
            recommended_image_rect_height=700,
            max_image_rect_width=1800,
            max_image_rect_height=1400,
        ),
    )
    created = []
    presenter._create_projection_swapchain = lambda width, height, **_kwargs: (
        created.append((width, height)) or SimpleNamespace()
    )

    presenter._create_projection_swapchains_for_scale(1.5)

    assert created == [(1500, 1200), (1350, 1050)]


def test_projection_pass_is_precreated_with_panorama_support(monkeypatch) -> None:
    presenter = OpenXrVulkanPresenter()
    presenter.config = replace(presenter.config, filament_panorama_path="test-panorama.hdr")
    presenter._view_configuration_views = (
        SimpleNamespace(
            recommended_image_rect_width=1000,
            recommended_image_rect_height=800,
            max_image_rect_width=1800,
            max_image_rect_height=1400,
        ),
        SimpleNamespace(
            recommended_image_rect_width=900,
            recommended_image_rect_height=700,
            max_image_rect_width=1800,
            max_image_rect_height=1400,
        ),
    )
    presenter.vulkan = object()
    presenter.swapchain_format = 43
    presenter._create_projection_swapchain = lambda *_args, **_kwargs: SimpleNamespace()
    created = []

    def create_projection_pass(
        context, target_format, *, enable_panorama, panorama_required
    ):
        created.append((context, target_format, enable_panorama, panorama_required))
        return SimpleNamespace(panorama_enabled=enable_panorama)

    monkeypatch.setattr(
        core_openxr_vulkan, "VulkanProjectionScreenPass", create_projection_pass
    )
    monkeypatch.setattr(presenter, "_is_rocm_backend", lambda: False)
    presenter._create_projection_swapchains_for_scale(1.0)

    assert created == [(presenter.vulkan, 43, True, False)]


@pytest.mark.parametrize("is_rocm", [False, True])
def test_default_environment_skips_optional_panorama_pipeline(monkeypatch, is_rocm) -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._view_configuration_views = (
        SimpleNamespace(
            recommended_image_rect_width=1000,
            recommended_image_rect_height=800,
            max_image_rect_width=1800,
            max_image_rect_height=1400,
        ),
        SimpleNamespace(
            recommended_image_rect_width=900,
            recommended_image_rect_height=700,
            max_image_rect_width=1800,
            max_image_rect_height=1400,
        ),
    )
    presenter.vulkan = object()
    presenter.swapchain_format = 43
    presenter._create_projection_swapchain = lambda *_args, **_kwargs: SimpleNamespace()
    created = []

    def create_projection_pass(
        context, target_format, *, enable_panorama, panorama_required
    ):
        created.append((context, target_format, enable_panorama, panorama_required))
        return SimpleNamespace(panorama_enabled=enable_panorama)

    monkeypatch.setattr(
        core_openxr_vulkan, "VulkanProjectionScreenPass", create_projection_pass
    )
    monkeypatch.setattr(presenter, "_is_rocm_backend", lambda: is_rocm)
    presenter._create_projection_swapchains_for_scale(1.0)

    assert created == [(presenter.vulkan, 43, False, False)]


def test_rocm_keeps_required_panorama_pipeline_with_native_equirect(monkeypatch) -> None:
    presenter = OpenXrVulkanPresenter()
    presenter.config = replace(presenter.config, filament_panorama_path="test-panorama.hdr")
    presenter._view_configuration_views = (
        SimpleNamespace(
            recommended_image_rect_width=1000,
            recommended_image_rect_height=800,
            max_image_rect_width=1800,
            max_image_rect_height=1400,
        ),
        SimpleNamespace(
            recommended_image_rect_width=900,
            recommended_image_rect_height=700,
            max_image_rect_width=1800,
            max_image_rect_height=1400,
        ),
    )
    presenter.vulkan = object()
    presenter.swapchain_format = 43
    presenter._openxr_equirect_supported = True
    presenter._create_projection_swapchain = lambda *_args, **_kwargs: SimpleNamespace()
    created = []

    def create_projection_pass(
        context, target_format, *, enable_panorama, panorama_required
    ):
        created.append((context, target_format, enable_panorama, panorama_required))
        return SimpleNamespace(panorama_enabled=enable_panorama)

    monkeypatch.setattr(presenter, "_is_rocm_backend", lambda: True)
    monkeypatch.setattr(
        core_openxr_vulkan, "VulkanProjectionScreenPass", create_projection_pass
    )
    presenter._create_projection_swapchains_for_scale(1.0)

    assert created == [(presenter.vulkan, 43, True, True)]


def test_presenter_validates_configuration() -> None:
    with pytest.raises(ValueError):
        OpenXrVulkanPresenter(OpenXrVulkanConfig(render_scale=0))


def test_openxr_defaults_to_validated_srgb_projection_target() -> None:
    config = OpenXrVulkanConfig()
    assert config.swapchain_color_mode == "srgb"
    assert config.clear_color == (0.0, 0.0, 0.0, 1.0)
    assert config.controller_model == "PICO"
    assert config.controller_guide_max_distance == pytest.approx(1.0)
    assert config.headset_model == "Pico 4 / 4 Ultra"


def test_presenter_config_keeps_default_screen_geometry() -> None:
    config = OpenXrVulkanConfig()
    assert config.filament_screen_width == pytest.approx(23.09)
    assert config.filament_screen_distance == pytest.approx(20.0)


def test_presenter_rejects_non_positive_controller_guide_distance() -> None:
    with pytest.raises(ValueError, match="controller_guide_max_distance"):
        OpenXrVulkanPresenter(OpenXrVulkanConfig(controller_guide_max_distance=0.0))


def test_controller_callout_texture_keeps_controller_center_transparent() -> None:
    rgba = build_controller_callout_rgba(lang="CN")

    assert rgba.shape == (1536, 2048, 4)
    assert rgba.dtype == np.uint8
    assert rgba[768, 1024, 3] == 0
    assert tuple(rgba[768, 1024, :3]) == (255, 255, 255)
    assert tuple(rgba[420, 1500]) == (255, 255, 255, 255)
    assert rgba[420, 1200, 3] == 0
    assert tuple(rgba[600, 1080]) == (255, 255, 255, 255)
    assert rgba[252, 300, 3] == 0
    assert int(rgba[..., 3].max()) == 255


def test_controller_guide_pose_hides_beyond_headset_distance() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._grip_mat_r = np.eye(4, dtype=np.float64)
    presenter._aim_mat_r = np.eye(4, dtype=np.float64)
    presenter._head_position_w = np.asarray((0.0, 0.0, 0.4), dtype=np.float64)

    pose = presenter._controller_guide_pose()
    assert pose is not None
    assert pose[1] == pytest.approx((0.34, 0.255))
    assert np.linalg.norm(np.asarray(pose[2], dtype=np.float64)) == pytest.approx(1.0)

    presenter._head_position_w[2] = 1.001
    assert presenter._controller_guide_pose() is None


@pytest.mark.parametrize(
    ("brand_name", "expected"),
    (
        ("HP", (-0.0235, 0.012129, -0.035076)),
        ("INDEX", (-0.021801, -0.001037, -0.051047)),
        ("PICO", (-0.00672205, 0.01771696, -0.02744452)),
        ("QUEST", (-0.0128, 0.001141, -0.028491)),
        ("VIVE", (-0.021922, 0.00029, -0.041995)),
        ("YVR", (-0.022195, 0.008466, -0.007238)),
    ),
)
def test_b_button_position_is_resolved_from_each_controller_glb(
    brand_name: str, expected: tuple[float, float, float]
) -> None:
    path = (
        APP_ROOT / f"xr_viewer/controllers/{brand_name}/right.glb"
    )

    position = controller_button_local_position(str(path), "b_button")

    assert position == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize(
    ("brand_name", "expected_multiplier"),
    (
        ("HP", 20.0),
        ("INDEX", 20.0),
        ("PICO", 1.0),
        ("QUEST", 1.0),
        ("VIVE", 20.0),
        ("YVR", 20.0),
    ),
)
def test_controller_profile_selects_ambient_light_multiplier(
    brand_name: str, expected_multiplier: float
) -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._controller_brand = presenter._controller_brands[brand_name]
    presenter._filament_ambient_light_color = (0.06, 0.05, 0.05)

    assert presenter._controller_brand.ambient_light_multiplier == pytest.approx(
        expected_multiplier
    )
    # The room ambient color is never multiplied by a controller brand.
    assert presenter._controller_ambient_light_color() == pytest.approx(
        (0.06, 0.05, 0.05)
    )
    assert presenter._controller_hdr_ambient_light_color() == pytest.approx(
        tuple(value * expected_multiplier for value in (0.06, 0.05, 0.05))
    )


def test_controller_brand_switch_recalculates_b_button_anchor() -> None:
    presenter = OpenXrVulkanPresenter()
    previous_brand = presenter._controller_brand
    presenter._controller_b_button_local = np.asarray(
        (99.0, 99.0, 99.0), dtype=np.float64
    )
    presenter._controller_b_button_resolved = True

    presenter._switch_shortcut_controller_brand()

    assert presenter._controller_brand is not previous_brand
    expected = controller_button_local_position(
        str(presenter._controller_brand.right_glb), "b_button"
    )
    assert presenter._controller_b_button_resolved is True
    assert presenter._controller_b_button_local == pytest.approx(expected)


def test_controller_brand_switch_refreshes_ambient_light() -> None:
    class Bridge:
        def __init__(self) -> None:
            self.ambient_colors: list[tuple[float, float, float]] = []
            self.controller_ambient: list[tuple[tuple[float, float, float], bool]] = []

        def load_controller(self, _hand: int, _data: bytes) -> None:
            pass

        def set_ambient_light(self, color) -> None:
            self.ambient_colors.append(tuple(color))

        def set_controller_ambient_light(self, color, enabled) -> None:
            self.controller_ambient.append((tuple(color), bool(enabled)))

    presenter = OpenXrVulkanPresenter()
    presenter._filament_ambient_light_color = (0.06, 0.05, 0.05)
    presenter._controller_brand = presenter._controller_brands["QUEST"]
    presenter.filament_bridge = Bridge()

    presenter._switch_shortcut_controller_brand()

    assert presenter._controller_brand.name == "VIVE"
    assert len(presenter.filament_bridge.ambient_colors) == 1
    assert presenter.filament_bridge.ambient_colors[0] == pytest.approx(
        (0.06, 0.05, 0.05)
    )
    assert presenter.filament_bridge.controller_ambient == [
        ((1.2, 1.0, 1.0), True)
    ]


def test_controller_lighting_config_is_resolved_outside_native_bridge() -> None:
    class Bridge:
        def __init__(self) -> None:
            self.config = None

        def set_lighting_config(self, **config) -> bool:
            self.config = config
            return True

    presenter = OpenXrVulkanPresenter()
    presenter._filament_ambient_light_color = (0.06, 0.05, 0.05)
    presenter._controller_hdr_lighting = True
    presenter._controller_brand = presenter._controller_brands["PICO"]
    presenter._controller_hdr_ambient_light_color_override = (0.2, 0.3, 0.4)
    bridge = Bridge()

    presenter._apply_filament_bridge_lighting(bridge)

    assert bridge.config["environment_ambient_intensity_lux"] == pytest.approx(30000.0)
    assert bridge.config["controller_ambient_intensity_lux"] == pytest.approx(8000.0)
    assert bridge.config["controller_ambient_color"] == pytest.approx((0.2, 0.3, 0.4))
    assert bridge.config["head_light_intensity_candela"] == pytest.approx(1700.0)
    assert bridge.config["top_light_intensity_candela"] == pytest.approx(1200.0)
    assert bridge.config["head_light_offset"] == pytest.approx((0.0, 0.05, 0.0))
    assert bridge.config["top_light_offset"] == pytest.approx((0.0, 0.45, -0.18))


@pytest.mark.parametrize("brand_name", ("HP", "INDEX", "PICO", "QUEST", "VIVE", "YVR"))
def test_controller_material_profile_matches_brand_defaults(
    brand_name: str,
) -> None:
    class Bridge:
        def __init__(self) -> None:
            self.config = None

        def set_controller_material_override(self, **config) -> bool:
            self.config = config
            return True

    presenter = OpenXrVulkanPresenter()
    bridge = Bridge()

    presenter._apply_controller_material_profile(
        bridge, presenter._controller_brands[brand_name]
    )

    if brand_name == "PICO":
        assert bridge.config == {
            "roughness_factor": pytest.approx(0.3),
            "metallic_factor": pytest.approx(0.0),
            "specular_color_factor": pytest.approx((1.0, 1.0, 1.0)),
        }
    else:
        assert bridge.config is None


def test_controller_screen_light_tracks_linear_screen_color_in_foreground() -> None:
    class Bridge:
        def __init__(self) -> None:
            self.calls = []

        def set_controller_screen_light(self, *args) -> bool:
            self.calls.append(args)
            return True

    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = (
        (1.0, 2.0, -3.0), 2.4, 1.35, (0.0, 0.0, 0.0)
    )
    presenter._controller_screen_light_smoothing_seconds = 0.0
    bridge = Bridge()
    frame = SimpleNamespace(metadata={
        "screen_light_linear_rgb": (0.0, 0.0, 1.0),
        "screen_light_sample_path": "vulkan_compute_reduction",
    })

    presenter._update_controller_screen_light(frame, bridge)

    color, intensity, direction, shadows, enabled = bridge.calls[-1]
    assert color == pytest.approx((0.35, 0.35, 1.0))
    assert intensity == pytest.approx(500.0 * 0.0722)
    assert direction == pytest.approx((0.0, 0.0, 1.0))
    assert shadows is False
    assert enabled is True

    presenter._filament_screen = (
        (0.0, 0.0, -20.0), 23.0, 13.0, (0.0, 0.0, 0.0)
    )
    presenter._grip_mat_l = np.eye(4, dtype=np.float64)
    presenter._update_controller_screen_light(frame, bridge)
    _color, distant_intensity, distant_direction, *_rest = bridge.calls[-1]
    assert distant_intensity == pytest.approx(500.0 * 0.0722)
    assert distant_direction == pytest.approx((0.0, 0.0, 1.0))


def test_environment_screen_reflection_uses_one_continuous_screen_area() -> None:
    class Bridge:
        def __init__(self) -> None:
            self.call = None

        def set_environment_screen_area_light(
            self, position, normal, color, **options
        ) -> bool:
            self.call = (position, normal, color, options)
            return True

    presenter = OpenXrVulkanPresenter()
    presenter.config = replace(presenter.config, filament_glb_path="room.glb")
    presenter._filament_screen = (
        (0.0, 0.0, -3.0), 4.0, 2.0, (0.0, 0.0, 0.0)
    )
    presenter._environment_screen_light_saturation = 1.0
    presenter._environment_screen_area_light_intensity = 3.5
    bridge = Bridge()
    samples = tuple((1.0, 0.0, 0.0) for _ in range(24))
    frame = SimpleNamespace(metadata={
        "screen_edge_light_linear_rgb": samples,
        "screen_light_sample_path": "vulkan_compute_reduction",
    })

    presenter._update_environment_screen_lights(frame, bridge)

    position, normal, color, options = bridge.call
    assert position == pytest.approx((0.0, 0.0, -3.0))
    assert normal == pytest.approx((0.0, 0.0, 1.0))
    assert color == pytest.approx((1.0, 0.0, 0.0))
    assert options == {
        "half_width": pytest.approx(2.0),
        "half_height": pytest.approx(1.0),
        "intensity": pytest.approx(3.5),
        "enabled": True,
    }

    presenter._filament_scene_exposure = -1.0
    presenter._update_environment_screen_lights(frame, bridge)
    assert bridge.call[3]["intensity"] == pytest.approx(7.0)


def test_controller_button_position_does_not_require_opengl_renderer(
    monkeypatch,
) -> None:
    import builtins

    path = (APP_ROOT /
            "xr_viewer/controllers/PICO/right.glb")
    original_import = builtins.__import__

    def reject_moderngl(name, *args, **kwargs):
        if name == "moderngl":
            raise ModuleNotFoundError("moderngl intentionally unavailable")
        return original_import(name, *args, **kwargs)

    controller_button_local_position.cache_clear()
    monkeypatch.setattr(builtins, "__import__", reject_moderngl)

    assert controller_button_local_position(str(path), "b_button") is not None


def test_controller_guide_stays_head_facing_while_endpoint_follows_b_button() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._head_position_w = np.asarray((0.0, 0.0, 0.2), dtype=np.float64)
    presenter._grip_mat_r = np.eye(4, dtype=np.float64)
    presenter._aim_mat_r = np.eye(4, dtype=np.float64)
    presenter._controller_b_button_local = np.asarray(
        (-0.00672205, 0.01771696, -0.02744452), dtype=np.float64
    )
    presenter._controller_b_button_resolved = True

    def endpoint_and_facing():
        position, size, quaternion = presenter._controller_guide_pose()
        orientation = xr.Quaternionf(
            x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3]
        )
        basis = _xr_quat_to_mat4(orientation)[:3, :3]
        endpoint_local = np.asarray((
            (540.0 / 1024.0 - 0.5) * size[0],
            (0.5 - 300.0 / 768.0) * size[1],
            0.0,
        ))
        endpoint = np.asarray(position) + basis @ endpoint_local
        button = presenter._controller_b_button_world_position()
        toward_head = presenter._head_position_w - np.asarray(position)
        toward_head /= np.linalg.norm(toward_head)
        return endpoint, button, float(np.dot(basis[:, 2], toward_head))

    initial_endpoint, initial_button, initial_facing = endpoint_and_facing()
    assert np.linalg.norm(initial_endpoint - initial_button) == pytest.approx(0.006, abs=1e-5)
    assert initial_facing > 0.99

    angle = math.radians(30.0)
    rotation = np.asarray((
        (math.cos(angle), -math.sin(angle), 0.0),
        (math.sin(angle), math.cos(angle), 0.0),
        (0.0, 0.0, 1.0),
    ), dtype=np.float64)
    presenter._grip_mat_r[:3, :3] = rotation
    presenter._aim_mat_r[:3, :3] = rotation
    rotated_endpoint, rotated_button, rotated_facing = endpoint_and_facing()

    assert not np.allclose(initial_button, rotated_button)
    assert np.linalg.norm(rotated_endpoint - rotated_button) == pytest.approx(0.006, abs=1e-5)
    assert rotated_facing > 0.99


def test_controller_callout_uses_projection_layer_not_quad_layer() -> None:
    source = (APP_ROOT /
              "xr_viewer/core_openxr_vulkan.py").read_text(encoding="utf-8")

    assert "return self._render_tool_quad_layers(output_frame)" in source
    assert "bridge.set_controller_guide_texture(self._controller_callout_rgba)" in source
    assert "bridge.set_controller_guide(guide_matrix, visible=True)" in source
    assert 'specs.append(("controller_callouts"' not in source
    assert "if self._screen_operation_guide_visible:" in source
    assert "build_help_rgba(environment_mode=environment_mode, lang=language)" in source
    assert 'self._tool_quad_texture_cache.get("screen_help")' in source
    assert '"keyboard", rgba, _keyboard_position' in source
    assert '"keyboard", rgba, keyboard_position' not in source
    assert 'entry.get("content") is not rgba' in source
    assert 'entry.get("image_index") is None' in source
    assert "_tool_overlay_xr_fps" in source
    assert "_tool_overlay_sbs_fps" in source
    assert "_tool_overlay_capture_fps" in source
    assert "self._update_tool_overlay_metrics(output_frame)" in source
    assert "actual_fps=self._tool_overlay_xr_fps" in source
    assert "capture_fps=self._tool_overlay_capture_fps" in source


def test_keyboard_quad_tracks_hover_and_held_indices() -> None:
    source = (APP_ROOT /
              "xr_viewer/core_openxr_vulkan.py").read_text(encoding="utf-8")

    assert "hover_indices = tuple(" in source
    assert "held_indices = tuple(" in source
    assert "hover_indices=hover_indices" in source
    assert "held_indices=held_indices" in source
    assert "hover_indices, held_indices," in source


def test_locked_keyboard_modifier_is_orange() -> None:
    _rgba, keys = build_keyboard_rgba(False, 1.0, 0.5)
    ctrl_index = next(index for index, key in enumerate(keys) if key.vk == 0x11)
    rgba, keys = build_keyboard_rgba(
        False,
        1.0,
        0.5,
        locked_indices=(ctrl_index,),
    )
    key = keys[ctrl_index]
    cx = int(round((key.rect_uv[0] + key.rect_uv[2]) * 0.5 * rgba.shape[1]))
    cy = int(round((key.rect_uv[1] + key.rect_uv[3]) * 0.5 * rgba.shape[0]))

    assert tuple(rgba[cy, cx]) == (240, 145, 35, 255)


def test_keyboard_haptic_pulse_uses_legacy_action_and_rate_limit() -> None:
    calls = []

    class FakeXR:
        FREQUENCY_UNSPECIFIED = -1.0

        class HapticVibration:
            def __init__(self, **values):
                self.values = values

        class HapticActionInfo:
            def __init__(self, **values):
                self.values = values

        @staticmethod
        def apply_haptic_feedback(session, action_info, vibration):
            calls.append((session, action_info.values, vibration.values))

    presenter = OpenXrVulkanPresenter()
    presenter.xr = FakeXR()
    presenter.session = object()
    presenter._act_haptic = object()
    presenter._path_right = 42

    assert presenter._pulse_haptic(
        "/user/hand/right", amplitude=0.18, duration_s=0.018, min_interval_s=10.0
    ) is True
    assert presenter._pulse_haptic(
        "/user/hand/right", amplitude=0.18, duration_s=0.018, min_interval_s=10.0
    ) is False
    assert len(calls) == 1
    assert calls[0][1]["subaction_path"] == 42
    assert calls[0][2] == {
        "duration": 18_000_000,
        "frequency": -1.0,
        "amplitude": 0.18,
    }


def test_keyboard_modifier_clicks_toggle_real_key_state_for_combinations(monkeypatch) -> None:
    events = []

    class FakeUser32:
        @staticmethod
        def keybd_event(virtual_key, _scan_code, flags, _extra):
            events.append((virtual_key, flags))

    monkeypatch.setattr(
        core_input_helpers.ctypes,
        "windll",
        SimpleNamespace(user32=FakeUser32()),
        raising=False,
    )

    class Host(CoreInputHelpersMixin):
        pass

    host = Host()
    host._mod_state = {
        "shift": [False, False, 0.0],
        "ctrl": [False, False, 0.0],
        "alt": [False, False, 0.0],
        "win": [False, False, 0.0],
    }
    host._caps_lock = False

    host._toggle_modifier_key("ctrl", 0x11)
    assert host._mod_state["ctrl"][0] is True
    assert events == [(0x11, 0)]

    key = SimpleNamespace(vk=0x41, shifted_vk=0x41)
    host._press_key_impl(key, 7, "held_key", "held_mods")
    assert events == [(0x11, 0), (0x41, 0)]
    assert host.held_mods == (False, False, False, False, 0x41)

    host._toggle_modifier_key("ctrl", 0x11)
    assert host._mod_state["ctrl"][0] is False
    assert events[-1] == (0x11, core_input_helpers._KEYEVENTF_KEYUP)


def test_tool_overlay_metrics_snapshot_latency_with_fps_window() -> None:
    presenter = OpenXrVulkanPresenter(on_capture_fps=lambda: 23.7)
    presenter._frame_now = 10.0

    frame = type("Frame", (), {"frame_id": 1, "timestamp": 10.0})()
    presenter._update_tool_overlay_metrics(frame)
    assert presenter._tool_overlay_latency_ms == 0.0

    presenter._frame_now = 10.25
    frame.frame_id = 2
    frame.timestamp = 10.20
    presenter._update_tool_overlay_metrics(frame)
    assert presenter._tool_overlay_latency_ms == 0.0
    assert presenter._tool_overlay_pending_latency_ms == pytest.approx(50.0)

    presenter._frame_now = 11.05
    frame.frame_id = 3
    frame.timestamp = 11.0
    frame.metadata = {"depth_strength": 1.75}
    presenter._update_tool_overlay_metrics(frame)
    assert presenter._tool_overlay_xr_fps == pytest.approx(3.0 / 1.05)
    assert presenter._tool_overlay_capture_fps == pytest.approx(23.7)
    assert presenter._tool_overlay_latency_ms == pytest.approx(50.0)
    assert presenter._tool_overlay_depth_strength == pytest.approx(1.75)


def test_tool_overlay_sbs_fps_counts_unique_producer_frames() -> None:
    presenter = OpenXrVulkanPresenter()
    frame = type(
        "Frame",
        (),
        {"frame_id": 1, "timestamp": 20.0, "metadata": {}},
    )()

    for now, output in ((20.0, frame), (20.25, frame), (20.5, None), (21.0, None)):
        presenter._frame_now = now
        presenter._update_tool_overlay_metrics(output)

    assert presenter._tool_overlay_xr_fps == pytest.approx(4.0)
    assert presenter._tool_overlay_sbs_fps == pytest.approx(1.0)

    presenter._frame_now = 22.0
    presenter._update_tool_overlay_metrics(None)
    assert presenter._tool_overlay_sbs_fps == 0.0


def test_tool_overlay_keeps_real_xr_present_rate() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._tool_overlay_xr_frame_ts.extend((10.0, 10.5))
    presenter._tool_overlay_xr_fps = 2.0
    presenter._tool_overlay_pending_xr_fps = 2.0
    presenter._tool_overlay_xr_window_started = 10.0
    presenter._tool_overlay_xr_window_frames = 36
    presenter._frame_now = 11.1

    presenter._update_tool_overlay_metrics(None)

    assert presenter._tool_overlay_xr_fps == pytest.approx(2.0)


def test_tool_overlay_snapshots_present_fps_before_invalidating_quad_texture(
    monkeypatch,
) -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._tool_overlay_xr_fps = 60.0
    presenter._tool_overlay_xr_frame_ts.extend((10.0, 10.1))
    monkeypatch.setattr("xr_viewer.core_openxr_vulkan.time.perf_counter", lambda: 10.2)

    presenter._record_xr_presented_frame()

    assert presenter._tool_overlay_xr_fps == 60.0
    assert presenter._tool_overlay_pending_xr_fps == pytest.approx(10.0)


def test_tool_overlay_does_not_report_runtime_producer_fps_as_presented_sbs() -> None:
    presenter = OpenXrVulkanPresenter(on_runtime_fps=lambda: 59.3)
    presenter._frame_now = 2.0

    presenter._update_tool_overlay_metrics(None)

    assert presenter._tool_overlay_xr_fps == 0.0
    assert presenter._tool_overlay_sbs_fps == 0.0


def test_presenter_inference_pressure_tracks_projection_and_queued_output() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._initialized = True
    presenter.session_running = True

    assert presenter.inference_backpressure_active() is False
    presenter._projection_busy.set()
    assert presenter.inference_backpressure_active() is True
    presenter._projection_busy.clear()

    presenter._pending_output = object()
    assert presenter.inference_backpressure_active() is True
    presenter._pending_output = None
    presenter._presenter_commands.put_nowait(("submit_runtime_result", object()))
    assert presenter.inference_backpressure_active() is True


def test_openxr_vulkan_uses_deeper_command_ring_for_multi_pass_projection() -> None:
    source = (APP_ROOT /
              "xr_viewer/core_openxr_vulkan.py").read_text(encoding="utf-8")

    assert '"D2S_OPENXR_VULKAN_FRAME_CONTEXTS", 9, minimum=3.0' in source
    assert "frame_context_count=frame_context_count" in source


def test_depth_shortcut_triggers_quad_osd_with_runtime_value() -> None:
    presenter = OpenXrVulkanPresenter(
        on_controller_shortcut=lambda _action, **_values: True
    )
    presenter._dispatch_controller_shortcut(
        "adjust_depth_strength", delta=0.25
    )
    assert presenter._depth_osd_show_t > 0.0

    presenter._filament_screen = (
        (0.0, 0.0, -2.0), 2.4, 1.35, (0.0, 0.0, 0.0)
    )
    presenter._head_position_w = np.asarray((0.0, 0.0, 0.0), dtype=np.float64)
    presenter._tool_overlay_depth_strength = 1.75
    presenter._depth_osd_show_t = 1.0
    presenter._frame_now = 1.1
    presenter.xr = object()
    presenter.session = object()
    presenter.vulkan = object()
    presenter._msdf_font_atlas = MsdfFontAtlas()
    presenter._vulkan_msdf_quad_renderer = object()
    presenter._cursor_overlay_specs = lambda *_args: []
    presenter._upload_tool_quad = lambda *args: args

    specs = presenter._render_tool_quad_layers()
    depth_osd = next(spec for spec in specs if spec[0] == "depth_osd")
    assert isinstance(depth_osd[1], VulkanMsdfQuadRequest)
    assert any(run["text"] == "Depth Strength" for run in depth_osd[1].runs)
    assert any(run["text"] == "1.75" for run in depth_osd[1].runs)


def test_toggle_stereo_shortcut_uses_legacy_mode_message() -> None:
    presenter = OpenXrVulkanPresenter(
        on_controller_shortcut=lambda _action, **_values: True
    )
    presenter._tool_overlay_depth_strength = 1.75
    presenter._dispatch_controller_shortcut("toggle_stereo")

    assert presenter._depth_osd_message == "3D mode off"
    assert presenter._depth_osd_show_t > 0.0


def test_depth_osd_uses_synchronous_runtime_value_before_output_frame() -> None:
    snapshot = SimpleNamespace(depth_strength=1.75)

    class RuntimeShortcut:
        context = SimpleNamespace(
            openxr_state=SimpleNamespace(runtime_settings_snapshot=snapshot)
        )

        def handle(self, action, **values):
            assert action == "adjust_depth_strength"
            snapshot.depth_strength += float(values["delta"])
            return True

    callback = RuntimeShortcut()
    presenter = OpenXrVulkanPresenter(on_controller_shortcut=callback.handle)
    presenter._tool_overlay_depth_strength = 1.75
    presenter._dispatch_controller_shortcut(
        "adjust_depth_strength", delta=0.25
    )

    assert presenter._tool_overlay_depth_strength == pytest.approx(2.0)
    assert presenter._tool_overlay_depth_strength_pending == pytest.approx(2.0)

    stale_frame = SimpleNamespace(
        frame_id=1, timestamp=0.0, metadata={"depth_strength": 1.75}
    )
    presenter._frame_now = 1.0
    presenter._update_tool_overlay_metrics(stale_frame)
    assert presenter._tool_overlay_depth_strength == pytest.approx(2.0)

    fresh_frame = SimpleNamespace(
        frame_id=2, timestamp=0.0, metadata={"depth_strength": 2.0}
    )
    presenter._update_tool_overlay_metrics(fresh_frame)
    assert presenter._tool_overlay_depth_strength == pytest.approx(2.0)
    assert presenter._tool_overlay_depth_strength_pending is None


def test_fps_overlay_resolution_uses_live_xr_and_output_sizes() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter.swapchains = [SimpleNamespace(width=3648, height=3648)]
    output_frame = SimpleNamespace(
        left_eye=SimpleNamespace(width=3840, height=2160),
        metadata={"render_size": (3840, 2160)},
    )

    assert presenter._overlay_resolution_sizes(output_frame) == (
        (3648, 3648),
        (3840, 2160),
    )
    assert presenter._overlay_resolution_sizes(None) == (
        (3648, 3648),
        (3840, 2160),
    )


@pytest.mark.parametrize(
    ("capture_size", "render_size", "eye_size", "expected_screen_size"),
    (
        ((1920, 1200), "3840x2400", (3840, 2400), (1920, 1200)),
        ("3840x2160", (1920, 1080), (1920, 1080), (3840, 2160)),
    ),
)
def test_fps_overlay_screen_resolution_prefers_source_capture_size(
    capture_size, render_size, eye_size, expected_screen_size
) -> None:
    presenter = OpenXrVulkanPresenter()
    presenter.swapchains = [SimpleNamespace(width=3648, height=3648)]
    output_frame = SimpleNamespace(
        left_eye=SimpleNamespace(width=eye_size[0], height=eye_size[1]),
        metadata={"capture_size": capture_size, "render_size": render_size},
    )

    assert presenter._overlay_resolution_sizes(output_frame) == (
        (3648, 3648),
        expected_screen_size,
    )


def test_screen_geometry_uses_capture_aspect_over_stale_render_metadata() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = (
        (0.0, 1.6, -2.0),
        2.4,
        1.35,
        (-90.0, 0.0, 0.0),
    )
    output_frame = SimpleNamespace(
        left_eye=SimpleNamespace(width=3840, height=2160),
        metadata={
            "capture_size": (1920, 1200),
            "render_size": (3840, 2160),
        },
    )

    presenter._sync_screen_aspect_to_frame(output_frame)

    assert presenter._filament_screen[2] == pytest.approx(1.5)


def test_filament_controller_guide_tracks_geometry_and_visibility() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._head_position_w = np.asarray((0.0, 0.0, 0.2), dtype=np.float64)
    presenter._grip_mat_r = np.eye(4, dtype=np.float64)
    presenter._controller_b_button_local = np.asarray(
        (-0.00672205, 0.01771696, -0.02744452), dtype=np.float64
    )
    presenter._controller_b_button_resolved = True

    class Bridge:
        controller_guide_abi_available = True

        def __init__(self):
            self.calls = []

        def set_controller_guide(self, matrix, *, visible):
            self.calls.append((np.asarray(matrix).copy(), visible))

    bridge = Bridge()
    presenter._update_filament_controller_guide(bridge)

    matrix, visible = bridge.calls[-1]
    assert visible is True
    assert matrix.shape == (4, 4)
    assert np.linalg.norm(matrix[:3, 0]) == pytest.approx(0.34)
    assert np.linalg.norm(matrix[:3, 1]) == pytest.approx(0.255)
    assert np.dot(matrix[:3, 2], presenter._head_position_w - matrix[:3, 3]) > 0.0

    presenter._head_position_w[2] = 1.001
    presenter._update_filament_controller_guide(bridge)
    _, visible = bridge.calls[-1]
    assert visible is False


def test_presenter_defaults_to_composer_only_screen_path(monkeypatch) -> None:
    monkeypatch.delenv("D2S_VULKAN_PROJECTION_COMPOSER", raising=False)
    presenter = OpenXrVulkanPresenter()
    assert presenter._vulkan_projection_composer_requested is True
    assert not hasattr(presenter, "_filament_screen_image_enabled")


def test_presenter_defaults_to_controller_overlay_after_composer(monkeypatch) -> None:
    monkeypatch.delenv(
        "D2S_FILAMENT_CONTROLLER_OVERLAY_AFTER_COMPOSER", raising=False
    )
    presenter = OpenXrVulkanPresenter()
    assert presenter._filament_controller_overlay_after_composer is True


def test_controller_overlay_after_composer_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("D2S_FILAMENT_CONTROLLER_OVERLAY_AFTER_COMPOSER", "0")
    presenter = OpenXrVulkanPresenter()
    assert presenter._filament_controller_overlay_after_composer is False


def test_projection_composer_defaults_on_and_can_be_explicitly_disabled(monkeypatch) -> None:
    monkeypatch.delenv("D2S_VULKAN_PROJECTION_COMPOSER", raising=False)
    presenter = OpenXrVulkanPresenter()
    assert presenter._vulkan_projection_composer_requested is True
    assert presenter._vulkan_projection_composer_active is False

    monkeypatch.setenv("D2S_VULKAN_PROJECTION_COMPOSER", "0")
    presenter = OpenXrVulkanPresenter()
    assert presenter._vulkan_projection_composer_requested is False
    assert presenter._vulkan_projection_composer_active is False


def test_filament_multiview_is_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("D2S_FILAMENT_MULTIVIEW", raising=False)
    presenter = OpenXrVulkanPresenter()
    assert presenter._filament_multiview_requested is False

    monkeypatch.setenv("D2S_FILAMENT_MULTIVIEW", "1")
    presenter = OpenXrVulkanPresenter()
    assert presenter._filament_multiview_requested is True


def test_projection_quality_chain_defaults_to_direct_gpu_and_can_be_enabled(monkeypatch) -> None:
    monkeypatch.delenv("D2S_VULKAN_PROJECTION_QUALITY_CHAIN", raising=False)
    presenter = OpenXrVulkanPresenter()
    assert presenter._vulkan_projection_quality_chain_requested is False

    monkeypatch.setenv("D2S_VULKAN_PROJECTION_QUALITY_CHAIN", "1")
    presenter = OpenXrVulkanPresenter()
    assert presenter._vulkan_projection_quality_chain_requested is True


def test_rocm_defaults_to_stable_vulkan_controller_and_no_tool_quads(monkeypatch) -> None:
    monkeypatch.setattr(
        OpenXrVulkanPresenter,
        "_is_rocm_backend",
        staticmethod(lambda: True),
    )
    monkeypatch.delenv("D2S_ROCM_DISABLE_OPENXR_OVERLAYS", raising=False)
    presenter = OpenXrVulkanPresenter(OpenXrVulkanConfig(controller_model="QUEST"))

    assert presenter._rocm_openxr_stable_path is False
    assert presenter._vulkan_controller_proxy_enabled is False
    assert presenter._controller_brand.profile_id == "quest"
    assert presenter._tool_quads_disabled() is False
    assert presenter._rocm_projection_quality_chain_enabled is False


def test_rocm_overlay_isolation_is_explicit_and_reversible(monkeypatch) -> None:
    monkeypatch.setattr(
        OpenXrVulkanPresenter,
        "_is_rocm_backend",
        staticmethod(lambda: True),
    )
    monkeypatch.setenv("D2S_ROCM_DISABLE_OPENXR_OVERLAYS", "1")
    presenter = OpenXrVulkanPresenter(OpenXrVulkanConfig(controller_model="QUEST"))

    assert presenter._rocm_openxr_stable_path is True
    assert presenter._vulkan_controller_proxy_enabled is False
    presenter._activate_rocm_openxr_stable_path()
    assert presenter._vulkan_controller_proxy_enabled is True
    assert presenter._controller_brand.profile_id == "none"
    assert presenter._tool_quads_disabled() is True
    assert presenter._rocm_projection_quality_chain_enabled is False


def test_rocm_overlay_enable_override_keeps_gpu_tool_quads(monkeypatch) -> None:
    monkeypatch.setattr(
        OpenXrVulkanPresenter,
        "_is_rocm_backend",
        staticmethod(lambda: True),
    )
    monkeypatch.setenv("D2S_ROCM_DISABLE_OPENXR_OVERLAYS", "1")
    monkeypatch.setenv("D2S_ROCM_ENABLE_OPENXR_OVERLAYS", "1")
    presenter = OpenXrVulkanPresenter(OpenXrVulkanConfig(controller_model="QUEST"))

    assert presenter._rocm_openxr_stable_path is False
    assert presenter._vulkan_controller_proxy_enabled is False
    assert presenter._tool_quads_disabled() is False


def test_rocm_tool_quad_startup_can_fit_controller_and_help_panels() -> None:
    pool = (1024, 1024)

    assert OpenXrVulkanPresenter._tool_quad_precreate_size(
        "controller_proxy_callout", pool
    ) == (2048, 1536)
    assert OpenXrVulkanPresenter._tool_quad_precreate_size(
        "screen_help", pool
    ) == (1280, 1536)
    assert OpenXrVulkanPresenter._tool_quad_precreate_size(
        "hand_help", pool
    ) == (2048, 1024)


def test_msdf_startup_prewarm_is_rocm_only(monkeypatch) -> None:
    prewarm_calls = []

    class FakeRenderer:
        outputs = [object()]

        def __init__(self, _vulkan, _atlas) -> None:
            pass

        def prewarm_outputs(self, sizes) -> None:
            prewarm_calls.append(tuple(sizes))

    monkeypatch.setattr(core_openxr_vulkan, "VulkanMsdfQuadRenderer", FakeRenderer)
    nvidia = OpenXrVulkanPresenter()
    nvidia.vulkan = object()
    nvidia._msdf_font_atlas = object()
    nvidia._rocm_backend = False
    nvidia._initialize_msdf_quad_renderer()

    assert prewarm_calls == []

    rocm = OpenXrVulkanPresenter()
    rocm.vulkan = object()
    rocm._msdf_font_atlas = object()
    rocm._rocm_backend = True
    rocm._initialize_msdf_quad_renderer()

    assert prewarm_calls == [
        ((256, 64), (512, 64), (1024, 64), (512, 256), (1024, 512), (1024, 1024))
    ]


def test_projection_composer_sampling_does_not_mutate_filament_bridge() -> None:
    calls = []

    class Bridge:
        screen_sampling_abi_available = True

        def set_screen_sampling(self, value):
            calls.append(("sampling", value))

        def set_screen_upscale(self, value):
            calls.append(("upscale", value))

    presenter = OpenXrVulkanPresenter()
    presenter.filament_bridge = Bridge()
    presenter._filament_screen = ((0.0, 0.0, -2.0), 2.0, 1.0, (0.0, 0.0, 0.0))
    frame = VulkanStereoOutputFrame(
        frame_id=1,
        timestamp=0.0,
        left_eye=SimpleNamespace(width=1920, height=1080),
        right_eye=SimpleNamespace(width=1920, height=1080),
        metadata={"capture_size": (3840, 2160), "render_size": (1920, 1080)},
    )

    plan = presenter._apply_screen_sampling_policy(frame)

    assert plan is not None
    assert (plan.source_width, plan.source_height) == (1920, 1080)
    assert plan.mode == "upscale_easu"
    assert presenter._active_screen_sampling_plan is plan
    assert calls == []


def test_projection_composer_sampling_uses_scaled_2k_eye_texture_not_4k_capture() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = ((0.0, 0.0, -2.0), 2.0, 1.0, (0.0, 0.0, 0.0))
    frame = VulkanStereoOutputFrame(
        frame_id=1,
        timestamp=0.0,
        left_eye=SimpleNamespace(width=2880, height=1620),
        right_eye=SimpleNamespace(width=2880, height=1620),
        metadata={"capture_size": (3840, 2160), "render_size": (2880, 1620)},
    )

    plan = presenter._apply_screen_sampling_policy(frame)

    assert plan is not None
    assert (plan.source_width, plan.source_height) == (2880, 1620)
    assert plan.input_tier_k == 2
    assert plan.mode == "upscale_easu"


def test_projection_sampling_uses_openxr_screen_footprint_for_quality_size() -> None:
    presenter = OpenXrVulkanPresenter(
        OpenXrVulkanConfig(headset_model="Meta Quest 2")
    )
    presenter._filament_screen = (
        (0.0, 0.0, -2.0), 2.0, 1.0, (0.0, 0.0, 0.0)
    )
    presenter._projection_eye_extents = lambda: ((3000, 1688), (3000, 1688))
    presenter._screen_footprint_pixels = lambda _view, _target: (3000.0, 1687.0)
    frame = VulkanStereoOutputFrame(
        frame_id=1,
        timestamp=0.0,
        left_eye=SimpleNamespace(width=3840, height=2160),
        right_eye=SimpleNamespace(width=3840, height=2160),
        metadata={},
    )

    plan = presenter._apply_screen_sampling_policy(frame, [object(), object()])

    assert plan is not None
    assert plan.quality_size == (3000, 1688)
    assert plan.quality_width > 2048


def test_projection_does_not_repeat_the_shared_output_quality_pass() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = ((0.0, 0.0, -2.0), 2.0, 1.0, (0.0, 0.0, 0.0))
    frame = VulkanStereoOutputFrame(
        frame_id=1,
        timestamp=0.0,
        left_eye=SimpleNamespace(width=3840, height=2160),
        right_eye=SimpleNamespace(width=3840, height=2160),
        metadata={
            "output_quality_applied": 1,
            "output_quality_mode": "upscale_easu",
        },
    )

    plan = presenter._apply_screen_sampling_policy(frame)

    assert plan is not None
    assert plan.mode == "native_mip"
    assert plan.filter_scale == 1.0
    assert plan.upscale_scale == 1.0


def test_screen_sampling_log_coalesces_pose_driven_policy_changes(monkeypatch, capsys) -> None:
    now = [100.0]
    monkeypatch.setattr(core_openxr_vulkan.time, "perf_counter", lambda: now[0])
    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = ((0.0, 0.0, -2.0), 2.0, 1.0, (0.0, 0.0, 0.0))
    frame = VulkanStereoOutputFrame(
        frame_id=1,
        timestamp=0.0,
        left_eye=SimpleNamespace(width=1920, height=1080),
        right_eye=SimpleNamespace(width=1920, height=1080),
        metadata={"render_size": (1920, 1080)},
    )

    presenter._apply_screen_sampling_policy(frame)
    assert "screen sampling policy" in capsys.readouterr().out

    frame.metadata["output_quality_applied"] = 1
    now[0] = 101.0
    presenter._apply_screen_sampling_policy(frame)
    assert capsys.readouterr().out == ""

    now[0] = 102.1
    presenter._apply_screen_sampling_policy(frame)
    output = capsys.readouterr().out
    assert "screen sampling policy" in output
    assert "mode=native_mip" in output


def test_projection_composer_final_pass_samples_completed_quality_mips() -> None:
    presenter = OpenXrVulkanPresenter()
    source = SimpleNamespace(width=3840, height=2160)
    target = SimpleNamespace(width=3648, height=3648)
    presenter._active_screen_sampling_plan = build_screen_sampling_plan(3840, 2160, 2)

    values = np.frombuffer(
        presenter._projection_screen_sampling_constants(source, target), dtype="<f4"
    )

    assert values.tolist() == pytest.approx((1.0 / 3840.0, 1.0 / 2160.0, 1.0, 0.0))

    presenter._active_screen_sampling_plan = build_screen_sampling_plan(1920, 1080, 2)
    values = np.frombuffer(
        presenter._projection_screen_sampling_constants(source, target), dtype="<f4"
    )
    assert values.tolist() == pytest.approx((1.0 / 3840.0, 1.0 / 2160.0, 1.0, 0.0))


def test_projection_composer_routes_sampling_policy_to_quality_chain() -> None:
    source = inspect.getsource(OpenXrVulkanPresenter._render_vulkan_projection_composer)

    assert "try_submit_stereo_quality_mip(" in source
    assert "mode=plan.mode" in source
    assert "filter_scale=plan.filter_scale" in source
    assert "upscale_scale=plan.upscale_scale" in source
    assert "extra_wait_semaphores=filament_wait_semaphores" in source
    assert "or filament_wait_semaphores" in source
    assert "self._vulkan_projection_quality_chain_requested" in source
    assert "cached_sources" not in source
    assert "if timeline is None:" in source


def test_projection_quality_chain_is_not_disabled_by_filament_waits() -> None:
    source = inspect.getsource(OpenXrVulkanPresenter._render_vulkan_projection_composer)

    # Filament waits synchronize the producer; they must not suppress the
    # projection quality pass or make GUI LOD/MIP/RCAS settings ineffective.
    assert "and not filament_wait_semaphores" not in source


def test_projection_panorama_survives_live_filament_foreground() -> None:
    source = inspect.getsource(OpenXrVulkanPresenter._render_vulkan_projection_composer)

    # GLB -> HDR hot switching intentionally keeps Filament alive for the
    # controllers. Its transparent HDR resolve must LOAD the panorama pass
    # instead of clearing the Projection target.
    panorama = source.index("panorama_timeline = self._vulkan_projection_screen_pass.submit_panorama")
    foreground = source.index("if filament_hdr_sources and not defer_filament_resolve:")
    screen = source.index("if use_quality_mip:")
    assert panorama < foreground < screen
    foreground_block = source[foreground:screen]
    assert "load_target=bool(panorama_timeline)" in foreground_block


def test_vulkan_panorama_upload_is_bounded_and_releases_staging() -> None:
    source = inspect.getsource(OpenXrVulkanPresenter._ensure_vulkan_panorama_source)

    assert "_VULKAN_PANORAMA_MAX_SIDE" in source
    assert "Image.Resampling.BILINEAR" in source
    assert "self._vulkan_panorama_staging.close()" in source
    assert "source={source_w}x{source_h} uploaded={w}x{h}" in source


def test_filament_controller_overlay_runs_after_vulkan_composer() -> None:
    calls = []

    class Bridge:
        controller_overlay_abi_available = True

        def set_active_eye(self, eye_index):
            calls.append(("eye", eye_index))

        def set_acquired_image(self, image_index):
            calls.append(("image", image_index))

        def render_controller_overlay(self):
            calls.append(("overlay", None))

    presenter = OpenXrVulkanPresenter()
    presenter.filament_bridge = Bridge()
    presenter._filament_controller_overlay_after_composer = True
    presenter._render_filament_controller_overlay(
        [(object(), 3), (object(), 5)], lambda *_args: None
    )

    assert calls == [
        ("eye", 0), ("image", 3), ("overlay", None),
        ("eye", 1), ("image", 5), ("overlay", None),
    ]
    source = inspect.getsource(OpenXrVulkanPresenter._render_projection_layer)
    composer = source.index("composer_timeline = self._render_vulkan_projection_composer")
    overlay = source.index("self._render_filament_controller_overlay", composer)
    output_commit = source.index("composer_frame.metadata", overlay)
    assert composer < overlay < output_commit


def test_projection_composer_base_pass_defers_controller_layers() -> None:
    source = inspect.getsource(
        OpenXrVulkanPresenter._render_filament_for_projection_composer
    )

    assert 'getattr(bridge, "background_frame_abi_available", False)' in source
    assert "bridge.begin_background_frame()" in source
    assert "bridge.begin_frame()" in source


def test_projection_quality_chain_disabled_forces_lod0_sampling() -> None:
    source = inspect.getsource(OpenXrVulkanPresenter._apply_vulkan_projection_sampling)

    assert "quality_chain_enabled" in source
    assert "min_lod=0.0" in source
    assert "max_lod=0.0" in source
    assert "mip_lod_bias=0.0" in source
    assert "rcas_sharpness=0.0" in source


def test_projection_quality_chain_skips_mip_after_shared_output_processing() -> None:
    source = inspect.getsource(OpenXrVulkanPresenter._render_vulkan_projection_composer)

    assert source.count('get("output_quality_applied", 0)') >= 2


def test_projection_quality_chain_bypasses_rcas_when_sharpness_is_zero() -> None:
    source = inspect.getsource(VulkanProjectionScreenPass.try_submit_stereo_quality_mip)
    mip_source = inspect.getsource(VulkanProjectionScreenPass.try_submit_stereo_mip)

    assert 'mode == "native_mip"' in source
    assert "try_submit_stereo_mip(" in source
    assert "use_rcas = self.rcas_sharpness > 0.0" in source
    assert "use_rcas = self.rcas_sharpness > 0.0" in mip_source
    assert "_record_rcas_draw" in mip_source
    assert "if use_rcas:" in mip_source
    assert "descriptor_sets=self.rcas_descriptor_sets" in source
    assert "self._complete_quality_mip_draw(quality, copy_draw, screen, timeline)" in source


def test_projection_mip_recording_templates_are_reused() -> None:
    class FakeFfi:
        @staticmethod
        def cast(_type_name, value):
            return value

    class FakeVk:
        ffi = FakeFfi()

        def __getattr__(self, name):
            if name.startswith("VK_"):
                return abs(hash(name)) + 1
            if name.startswith("Vk"):
                return lambda **kwargs: (name, kwargs)
            raise AttributeError(name)

    class Image:
        image = 123
        width = 3840
        height = 2160
        mip_levels = 4

    screen_pass = object.__new__(VulkanProjectionScreenPass)
    screen_pass.vk = FakeVk()
    screen_pass._mip_recording_templates = {}
    screen_pass._mip_template_hits = 0
    screen_pass._mip_template_misses = 0

    first = screen_pass._mip_recording_template(
        Image(), screen_pass.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    )
    second = screen_pass._mip_recording_template(
        Image(), screen_pass.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    )

    assert second is first
    assert len(first["levels"]) == 3
    assert screen_pass._mip_template_hits == 1
    assert screen_pass._mip_template_misses == 1


def test_projection_render_pass_and_barrier_templates_are_reused() -> None:
    class FakeFfi:
        @staticmethod
        def cast(_type_name, value):
            return value

    class FakeVk:
        ffi = FakeFfi()

        def __getattr__(self, name):
            if name.startswith("VK_"):
                return abs(hash(name)) + 1
            if name.startswith("Vk"):
                return lambda **kwargs: (name, kwargs)
            raise AttributeError(name)

    screen_pass = object.__new__(VulkanProjectionScreenPass)
    screen_pass.vk = FakeVk()
    screen_pass._render_pass_recording_templates = {}
    screen_pass._render_pass_template_hits = 0
    screen_pass._render_pass_template_misses = 0
    screen_pass._image_barrier_templates = {}
    screen_pass._image_barrier_template_hits = 0
    screen_pass._image_barrier_template_misses = 0

    render_first = screen_pass._render_pass_recording_template(
        100, 200, 3648, 3648, (0.0, 0.0, 0.0, 1.0)
    )
    render_second = screen_pass._render_pass_recording_template(
        100, 200, 3648, 3648, (0.0, 0.0, 0.0, 1.0)
    )
    barrier_first = screen_pass._image_barrier_template(
        300,
        src_access=1,
        dst_access=2,
        old_layout=3,
        new_layout=4,
        base_array_layer=1,
    )
    barrier_second = screen_pass._image_barrier_template(
        300,
        src_access=1,
        dst_access=2,
        old_layout=3,
        new_layout=4,
        base_array_layer=1,
    )

    assert render_second is render_first
    assert barrier_second is barrier_first
    assert screen_pass._render_pass_template_hits == 1
    assert screen_pass._render_pass_template_misses == 1
    assert screen_pass._image_barrier_template_hits == 1
    assert screen_pass._image_barrier_template_misses == 1


def test_projection_mip_lod_bias_defaults_and_clamps(monkeypatch) -> None:
    monkeypatch.delenv("D2S_VULKAN_PROJECTION_MIP_LOD_BIAS", raising=False)
    assert VulkanProjectionScreenPass._mip_lod_bias_from_env() == -0.35
    monkeypatch.setenv("D2S_VULKAN_PROJECTION_MIP_LOD_BIAS", "-2.0")
    assert VulkanProjectionScreenPass._mip_lod_bias_from_env() == -1.5

    monkeypatch.setenv("D2S_VULKAN_PROJECTION_MIP_LOD_BIAS", "invalid")
    assert VulkanProjectionScreenPass._mip_lod_bias_from_env() == -0.35


def test_projection_sampling_config_clamps_and_orders_lods() -> None:
    assert VulkanProjectionScreenPass._normalize_sampling_config(1.0, 0.25, -2.0, 2.0) == (
        1.0,
        1.0,
        -1.5,
        1.0,
    )

def test_projection_rcas_sharpness_defaults_and_clamps(monkeypatch) -> None:
    monkeypatch.delenv("D2S_VULKAN_PROJECTION_RCAS_SHARPNESS", raising=False)
    assert VulkanProjectionScreenPass._rcas_sharpness_from_env() == 0.5

    monkeypatch.setenv("D2S_VULKAN_PROJECTION_RCAS_SHARPNESS", "2.0")
    assert VulkanProjectionScreenPass._rcas_sharpness_from_env() == 1.0

    monkeypatch.setenv("D2S_VULKAN_PROJECTION_RCAS_SHARPNESS", "invalid")
    assert VulkanProjectionScreenPass._rcas_sharpness_from_env() == 0.5


def test_projection_max_mip_lod_defaults_and_clamps(monkeypatch) -> None:
    monkeypatch.delenv("D2S_VULKAN_PROJECTION_MAX_LOD", raising=False)
    assert VulkanProjectionScreenPass._max_mip_lod_from_env() == 0.35

    monkeypatch.setenv("D2S_VULKAN_PROJECTION_MAX_LOD", "20.0")
    assert VulkanProjectionScreenPass._max_mip_lod_from_env() == 16.0

    monkeypatch.setenv("D2S_VULKAN_PROJECTION_MAX_LOD", "invalid")
    assert VulkanProjectionScreenPass._max_mip_lod_from_env() == 0.35


def test_screen_quad_reprojection_is_disabled_for_virtual_desktop(monkeypatch) -> None:
    monkeypatch.delenv("D2S_OPENXR_SCREEN_QUAD_REPROJECTION", raising=False)
    presenter = OpenXrVulkanPresenter()
    assert presenter._screen_quad_reprojection_requested is False
    assert presenter._screen_quad_reprojection_active is False

    monkeypatch.setenv("D2S_OPENXR_SCREEN_QUAD_REPROJECTION", "1")
    presenter = OpenXrVulkanPresenter()
    assert presenter._screen_quad_reprojection_requested is False
    assert presenter._screen_quad_reprojection_active is False


def test_screen_quad_reprojection_uploads_stereo_layers_and_releases_source(monkeypatch) -> None:
    calls = []

    class FakeXr:
        LEFT = "left"
        RIGHT = "right"
        INFINITE_DURATION = 1

        @staticmethod
        def acquire_swapchain_image(_handle):
            return 0

        @staticmethod
        def wait_swapchain_image(_handle, _wait_info):
            return None

        @staticmethod
        def release_swapchain_image(_handle):
            return None

        @staticmethod
        def SwapchainImageWaitInfo(*, timeout):
            return timeout

    presenter = OpenXrVulkanPresenter()
    presenter.xr = FakeXr
    presenter.session = object()
    presenter.reference_space = object()
    presenter._filament_screen = ((0.0, 1.0, -2.0), 2.0, 1.0, (0.0, 0.0, 0.0))
    presenter._ensure_quad_swapchains = lambda width, height: presenter._quad_swapchains.extend(
        [] if presenter._quad_swapchains else [
            _EyeSwapchain("stereo-quad", [], width, height, ["quad-target"], array_size=2),
        ]
    )
    presenter.vulkan = SimpleNamespace(
        copy_image=lambda source, target, **kwargs: calls.append(
            (source.image, target, kwargs["wait_semaphore"])
        ) or 19,
    )
    monkeypatch.setattr(
        OpenXrCompositionBuilder,
        "quad_layer",
        lambda _self, _swapchain, _position, _width, _height, _rotation, eye: (
            SimpleNamespace(eye_visibility=("left", "right")[eye])
        ),
    )
    frame = VulkanStereoOutputFrame(
        frame_id=33,
        timestamp=0.0,
        left_eye=SimpleNamespace(image="left-source", width=4, height=2),
        right_eye=SimpleNamespace(image="right-source", width=4, height=2),
        metadata={
            "_vulkan_source_prepare_for_sampling": lambda _frame, eye: (
                "left-ready", "right-ready"
            )[eye],
            "_vulkan_source_consumer_release": lambda frame_id, **kwargs: calls.append(
                ("release", frame_id, kwargs["wait_for_timeline"])
            ),
        },
    )

    presenter._render_screen_quad_reprojection(frame)

    assert calls == [
        ("left-source", "quad-target", "left-ready"),
        ("right-source", "quad-target", "right-ready"),
        ("release", 33, 19),
    ]
    assert presenter._screen_quad_reprojection_active is True
    assert [layer.eye_visibility for layer in presenter._last_screen_quad_layers] == [
        "left", "right"
    ]


def test_screen_quad_reprojection_eye_diagnostic_clears_each_eye(monkeypatch) -> None:
    calls = []

    class FakeXr:
        LEFT = "left"
        RIGHT = "right"
        INFINITE_DURATION = 1
        acquire_swapchain_image = staticmethod(lambda _handle: 0)
        wait_swapchain_image = staticmethod(lambda _handle, _wait_info: None)
        release_swapchain_image = staticmethod(lambda _handle: None)
        SwapchainImageWaitInfo = staticmethod(lambda *, timeout: timeout)

    monkeypatch.setenv("D2S_OPENXR_SCREEN_QUAD_EYE_DIAGNOSTIC", "1")
    presenter = OpenXrVulkanPresenter()
    presenter.xr = FakeXr
    presenter.session = object()
    presenter.reference_space = object()
    presenter._filament_screen = ((0.0, 1.0, -2.0), 2.0, 1.0, (0.0, 0.0, 0.0))
    presenter._ensure_quad_swapchains = lambda width, height: presenter._quad_swapchains.extend(
        [] if presenter._quad_swapchains else [
            _EyeSwapchain(
                "stereo-quad", [], width, height, [SimpleNamespace(image="quad-target")], array_size=2
            ),
        ]
    )
    presenter.vulkan = SimpleNamespace(
        clear_color_image=lambda image, color, **kwargs: calls.append((image, color, kwargs)) or 19,
        copy_image=lambda *_args, **_kwargs: pytest.fail("diagnostic must not copy source frames"),
    )
    monkeypatch.setattr(
        OpenXrCompositionBuilder,
        "quad_layer",
        lambda _self, _swapchain, _position, _width, _height, _rotation, eye: SimpleNamespace(
            eye_visibility=("left", "right")[eye]
        ),
    )
    frame = VulkanStereoOutputFrame(
        frame_id=33,
        timestamp=0.0,
        left_eye=SimpleNamespace(image="left-source", width=4, height=2),
        right_eye=SimpleNamespace(image="right-source", width=4, height=2),
        metadata={
            "_vulkan_source_prepare_for_sampling": lambda *_args: pytest.fail("diagnostic must not prepare sources"),
            "_vulkan_source_consumer_release": lambda *_args, **_kwargs: None,
        },
    )

    presenter._render_screen_quad_reprojection(frame)

    assert calls == [
        ("quad-target", (1.0, 0.0, 0.0, 1.0), {"base_array_layer": 0}),
        ("quad-target", (0.0, 1.0, 0.0, 1.0), {"base_array_layer": 1}),
    ]


def test_projection_composer_uses_direct_vulkan_rasterization_in_opaque_runtime() -> None:
    source = (APP_ROOT /
              "xr_viewer/core_openxr_vulkan.py").read_text(encoding="utf-8")
    render_source = inspect.getsource(
        OpenXrVulkanPresenter._render_vulkan_projection_composer
    )

    assert "use_vulkan_projection_composer = bool(" in source
    assert "self._environment_blend_mode != xr.EnvironmentBlendMode.OPAQUE" not in source
    assert "opaque_runtime_requires_full_projection_background" not in source
    assert ".copy_image(" not in render_source
    assert "_screen_projection_bounds" not in render_source
    assert "Vulkan projection composer active:" in source
    assert "Vulkan projection composer eye diagnostic:" in source
    assert "self._vulkan_projection_composer_active" in source


def test_projection_pass_reports_creation_stage_and_fallback_traceback() -> None:
    pass_source = inspect.getsource(VulkanProjectionScreenPass)
    create_source = inspect.getsource(VulkanProjectionScreenPass._create)
    composer_source = inspect.getsource(
        OpenXrVulkanPresenter._render_vulkan_projection_composer
    )
    presenter_source = inspect.getsource(OpenXrVulkanPresenter._render_projection_layer)

    assert 'self.creation_stage = "create_shader_modules"' in pass_source
    assert 'self.creation_stage = "create_screen_pipeline"' in pass_source
    assert 'self.creation_stage = "create_panorama_pipeline"' in pass_source
    assert "if self.panorama_enabled:" in create_source
    assert "Vulkan projection pass is unavailable" in composer_source
    projection_source = inspect.getsource(
        OpenXrVulkanPresenter._create_projection_swapchains_for_scale
    )
    assert "bool(self.config.filament_panorama_path)" in projection_source
    assert "self._is_rocm_backend()" in projection_source
    assert "Vulkan projection pass creation failed:" in pass_source
    assert "Optional panorama pipeline unavailable" in pass_source
    assert "traceback.format_exc().rstrip()" in presenter_source


def test_filament_screen_image_abi_is_absent_from_presenter() -> None:
    source = (APP_ROOT /
              "xr_viewer/core_openxr_vulkan.py").read_text(encoding="utf-8")

    for symbol in (
        "set_screen_image(",
        "set_screen_ready_semaphore",
        "_can_use_filament_screen_image",
        "D2S_ENABLE_FILAMENT_SCREEN_IMAGE",
    ):
        assert symbol not in source


def test_release_output_frame_waits_for_filament_finished_semaphores() -> None:
    calls = []
    frame = SimpleNamespace(
        frame_id=17,
        metadata={
            "_vulkan_source_consumer_release": lambda frame_id, semaphores: calls.append(
                (frame_id, semaphores)
            ),
            "_vulkan_consumer_release_semaphores": ("left-finished", "right-finished"),
            "_vulkan_output_release": lambda frame_id: calls.append(("fallback", frame_id)),
        },
    )

    OpenXrVulkanPresenter._release_output_frame(frame)

    assert calls == [(17, ("left-finished", "right-finished"))]


def test_release_output_frame_prefers_consumed_filament_timeline() -> None:
    calls = []
    frame = SimpleNamespace(
        frame_id=18,
        metadata={
            "_vulkan_source_consumer_release": lambda frame_id, **kwargs: calls.append(
                (frame_id, kwargs)
            ),
            "_vulkan_consumer_release_timeline": 41,
            "_vulkan_consumer_release_semaphores": ("stale-left", "stale-right"),
        },
    )

    OpenXrVulkanPresenter._release_output_frame(frame)

    assert calls == [(18, {"wait_for_timeline": 41})]


def test_release_output_frame_calls_consumer_without_sync_argument() -> None:
    calls = []
    frame = SimpleNamespace(
        frame_id=181,
        metadata={
            "_vulkan_source_consumer_release": (
                lambda frame_id: calls.append(frame_id)
            ),
            "_vulkan_output_release": (
                lambda frame_id: calls.append(("fallback", frame_id))
            ),
        },
    )

    OpenXrVulkanPresenter._release_output_frame(frame)

    assert calls == [181]


def test_release_output_frame_releases_glow_after_screen_consumer() -> None:
    calls = []
    frame = SimpleNamespace(
        frame_id=19,
        metadata={
            "_vulkan_source_consumer_release": lambda frame_id, semaphores: calls.append(
                ("screen", frame_id, semaphores)
            ),
            "_vulkan_consumer_release_semaphores": ("left", "right"),
            "_vulkan_glow_release": lambda frame_id: calls.append(
                ("glow", frame_id)
            ),
        },
    )

    OpenXrVulkanPresenter._release_output_frame(frame)

    assert calls == [
        ("screen", 19, ("left", "right")),
        ("glow", 19),
    ]


def test_release_output_frame_is_idempotent_during_exception_unwind() -> None:
    calls = []
    frame = SimpleNamespace(
        frame_id=20,
        metadata={
            "_vulkan_output_release": lambda frame_id: calls.append(frame_id),
            "_vulkan_glow_release": lambda frame_id: calls.append(("glow", frame_id)),
        },
    )

    OpenXrVulkanPresenter._release_output_frame(frame)
    OpenXrVulkanPresenter._release_output_frame(frame)

    assert calls == [20, ("glow", 20)]


def test_release_displayed_output_for_reuse_releases_matching_ring_slot() -> None:
    calls = []
    presenter = OpenXrVulkanPresenter()
    presenter._displayed_output = SimpleNamespace(
        frame_id=23,
        metadata={
            "vulkan_output_ring_slot": 2,
            "_vulkan_output_release": lambda frame_id: calls.append(frame_id),
        },
    )

    assert presenter.release_displayed_output_for_reuse(2) is True
    assert presenter._displayed_output is None
    assert calls == [23]
    assert presenter.release_displayed_output_for_reuse(2) is False


def test_host_image_upload_writes_padded_rows_without_pointer_cast() -> None:
    from viewer.vulkan_resources import VulkanHostImage
    import numpy as np

    image = VulkanHostImage.__new__(VulkanHostImage)
    image.width = 2
    image.height = 2
    image._layout = SimpleNamespace(offset=2, rowPitch=12, size=26)
    mapped = bytearray(26)
    image.vk = SimpleNamespace(
        vkMapMemory=lambda *args: mapped,
        vkUnmapMemory=lambda *args: None,
    )
    image.context = SimpleNamespace(device=object())
    image.memory = object()
    image.upload(np.arange(16, dtype=np.uint8).reshape(2, 2, 4))
    assert mapped[2:10] == bytes(range(8))
    assert mapped[14:22] == bytes(range(8, 16))


def test_sbs_capture_options_are_explicit_and_delayed(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("D2S_SBS_CAPTURE_DIR", raising=False)
    assert _sbs_capture_options() is None

    monkeypatch.setenv("D2S_SBS_CAPTURE_DIR", str(tmp_path))
    options = _sbs_capture_options()

    assert options is not None
    assert options["output_dir"] == tmp_path
    assert options["delay_seconds"] == 15.0
    assert options["sample_count"] == 300
    assert options["image_count"] == 6
    assert options["eye_width"] == 640


def test_sbs_capture_converts_bgra_and_bottom_left_origin() -> None:
    pixels = np.array(
        [
            [[[1, 2, 3, 255]]],
            [[[4, 5, 6, 255]]],
        ],
        dtype=np.uint8,
    ).reshape(2, 1, 4)

    rgb = _vulkan_rgba_to_rgb(
        pixels,
        format_value=vk.VK_FORMAT_B8G8R8A8_UNORM,
        vk=vk,
        image_origin="bottom_left",
    )

    assert rgb.tolist() == [[[6, 5, 4]], [[3, 2, 1]]]


def test_host_image_readback_reads_padded_rows_in_one_mapping() -> None:
    from viewer.vulkan_resources import VulkanHostImage

    image = VulkanHostImage.__new__(VulkanHostImage)
    image.width = 2
    image.height = 2
    image._layout = SimpleNamespace(offset=2, rowPitch=12, size=26)
    mapped = bytearray(26)
    mapped[2:10] = bytes(range(8))
    mapped[14:22] = bytes(range(8, 16))
    image.vk = SimpleNamespace(
        vkMapMemory=lambda *args: mapped,
        vkUnmapMemory=lambda *args: None,
    )
    image.context = SimpleNamespace(device=object())
    image.memory = object()

    assert image.read_rgba().reshape(-1).tolist() == list(range(16))


def test_presenter_uses_controller_action_mixin_initializer() -> None:
    presenter = OpenXrVulkanPresenter()
    assert hasattr(presenter, "_init_controller_actions")
    assert not hasattr(presenter, "_initialize_controller_actions")
    assert presenter._LASER_HIDE_AFTER == 5.0
    assert presenter._laser_prev_mat_l is None
    assert presenter._laser_prev_mat_r is None


def test_filament_controller_lifecycle_hides_each_idle_hand_independently() -> None:
    class Bridge:
        controller_abi_available = True
        controller_visibility_abi_available = True
        laser_abi_available = True

        def __init__(self) -> None:
            self.visible = []
            self.poses = []
            self.inputs = []
            self.lasers = []

        def set_controller_visible(self, hand, visible) -> None:
            self.visible.append((hand, visible))

        def set_controller_pose(self, hand, matrix) -> None:
            self.poses.append((hand, matrix.copy()))

        def set_controller_inputs(self, hand, **values) -> None:
            self.inputs.append((hand, values))

        def set_controller_laser(self, hand, matrix, *, visible) -> None:
            self.lasers.append((hand, matrix.copy(), visible))

    presenter = OpenXrVulkanPresenter()
    presenter._controller_brand = SimpleNamespace(
        offset=(0.0, 0.0, 0.0), rotation_deg=0.0
    )
    presenter._frame_now = 20.0
    presenter._laser_last_move_l = 14.9
    presenter._laser_last_move_r = 19.0
    presenter._grip_mat_l = np.eye(4, dtype=np.float32)
    presenter._grip_mat_r = np.eye(4, dtype=np.float32)
    presenter._aim_mat_l = None
    presenter._aim_mat_r = None
    presenter._controller_inputs = ({}, {"joystick_touched": 1.0})
    bridge = Bridge()

    presenter._update_filament_controllers(bridge)

    assert bridge.visible == [(0, False), (1, True)]
    assert [hand for hand, _matrix in bridge.poses] == [1]
    assert [hand for hand, _values in bridge.inputs] == [1]
    assert bridge.inputs[0][1]["button_mask"] == 1 << 6
    assert [(hand, visible) for hand, _matrix, visible in bridge.lasers] == [
        (0, False), (1, False)
    ]


def test_controller_touch_actions_cover_thumbstick_trackpad_and_thumbrest() -> None:
    root = Path(__file__).resolve().parents[1]
    actions = (APP_ROOT / "xr_viewer/core_controller_actions.py").read_text(
        encoding="utf-8"
    )
    inputs = (APP_ROOT / "xr_viewer/core_controller_input.py").read_text(
        encoding="utf-8"
    )

    assert "/input/thumbstick/touch" in actions
    assert "/input/trackpad/touch" in actions
    assert "/input/thumbrest/touch" in actions
    assert 'left["stick_click"]' in inputs
    assert '"joystick_touched": 1.0 if left_touched else 0.0' in inputs
    assert '"touchpad_touched": 1.0 if right_touched else 0.0' in inputs


def test_active_filament_controller_uses_legacy_laser_calibration() -> None:
    class Bridge:
        controller_abi_available = True
        controller_visibility_abi_available = True
        laser_abi_available = True

        def __init__(self) -> None:
            self.laser_matrix = None

        def set_controller_visible(self, hand, visible) -> None:
            pass

        def set_controller_pose(self, hand, matrix) -> None:
            pass

        def set_controller_inputs(self, hand, **values) -> None:
            pass

        def set_controller_laser(self, hand, matrix, *, visible) -> None:
            if hand == 0 and visible:
                self.laser_matrix = matrix.copy()

    presenter = OpenXrVulkanPresenter()
    presenter._controller_brand = SimpleNamespace(
        offset=(0.0, 0.0, 0.0), rotation_deg=0.0
    )
    presenter._frame_now = 20.0
    presenter._laser_last_move_l = 19.0
    presenter._laser_last_move_r = 0.0
    presenter._grip_mat_l = np.eye(4, dtype=np.float32)
    presenter._grip_mat_r = None
    presenter._aim_mat_l = np.eye(4, dtype=np.float32)
    presenter._aim_mat_r = None
    presenter._controller_inputs = ({}, {})
    bridge = Bridge()

    presenter._update_filament_controllers(bridge)

    assert bridge.laser_matrix is not None
    assert np.linalg.norm(bridge.laser_matrix[:3, 0]) == pytest.approx(0.006)
    assert np.linalg.norm(bridge.laser_matrix[:3, 1]) == pytest.approx(0.4)
    assert np.linalg.norm(bridge.laser_matrix[:3, 2]) == pytest.approx(0.006)
    assert bridge.laser_matrix[1, 3] > 0.0
    assert bridge.laser_matrix[2, 3] < 0.0


def test_vulkan_presenter_exposes_legacy_overlay_shortcut_state() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._controller_inputs = (
        {"x_button": 0.0, "menu_button": 0.0},
        {"a_button": 0.0, "b_button": 0.0, "menu_button": 0.0},
    )
    presenter._handle_controller_shortcuts()
    presenter._controller_inputs = (
        {"x_button": 1.0, "menu_button": 0.0},
        {"a_button": 0.0, "b_button": 0.0, "menu_button": 0.0},
    )
    presenter._handle_controller_shortcuts()
    presenter._controller_inputs = ({"x_button": 0.0}, {})
    presenter._handle_controller_shortcuts()
    assert presenter._keyboard_visible is True
    presenter._controller_inputs = ({"x_button": 1.0}, {})
    presenter._handle_controller_shortcuts()
    presenter._controller_inputs = ({"x_button": 0.0}, {})
    presenter._handle_controller_shortcuts()
    assert presenter._keyboard_visible is False


def test_vulkan_b_long_press_cycles_hand_fps_and_operation_guide() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._frame_now = 1.0
    presenter._controller_inputs = ({}, {"b_button": 1.0})
    presenter._handle_controller_shortcuts()

    presenter._frame_now = 2.01
    presenter._handle_controller_shortcuts()

    # First B hold: hand FPS only.
    assert presenter._hand_fps_visible is True
    assert presenter._hand_operation_guide_visible is False
    assert presenter._operation_guide_visible is False
    assert presenter._fps_overlay_visible is False
    assert presenter._aperture_visible is False

    presenter._dispatch_controller_shortcut("cycle_hand_panel")
    assert presenter._hand_fps_visible is True
    assert presenter._hand_operation_guide_visible is True
    assert presenter._operation_guide_visible is True
    assert presenter._fps_overlay_visible is False

    presenter._dispatch_controller_shortcut("cycle_hand_panel")
    assert presenter._hand_fps_visible is False
    assert presenter._hand_operation_guide_visible is False
    assert presenter._operation_guide_visible is False


def test_menu_panel_cycle_keeps_fps_when_vertical_screen_guide_is_shown() -> None:
    presenter = OpenXrVulkanPresenter()

    presenter._dispatch_controller_shortcut("cycle_status_panel")
    assert presenter._fps_overlay_visible is True
    assert presenter._screen_operation_guide_visible is False

    presenter._dispatch_controller_shortcut("cycle_status_panel")
    assert presenter._fps_overlay_visible is True
    assert presenter._screen_operation_guide_visible is True

    presenter._dispatch_controller_shortcut("cycle_status_panel")
    assert presenter._fps_overlay_visible is False
    assert presenter._screen_operation_guide_visible is False


def test_menu_and_b_panel_cycles_do_not_leave_the_other_guide_visible() -> None:
    presenter = OpenXrVulkanPresenter()

    presenter._dispatch_controller_shortcut("cycle_status_panel")
    presenter._dispatch_controller_shortcut("cycle_status_panel")
    assert presenter._screen_operation_guide_visible is True

    presenter._dispatch_controller_shortcut("cycle_hand_panel")
    assert presenter._hand_fps_visible is True
    assert presenter._hand_operation_guide_visible is False
    assert presenter._screen_operation_guide_visible is False
    assert presenter._fps_overlay_visible is False


def test_screen_operation_guide_keeps_screen_height_and_scales_text() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = (
        (0.0, 0.0, -20.0), 23.09, 12.988, (0.0, 0.0, 0.0)
    )
    presenter._head_position_w = np.asarray((0.0, 0.0, 0.0), dtype=np.float64)
    presenter._screen_operation_guide_visible = True
    presenter.xr = object()
    presenter.session = object()
    presenter.vulkan = object()
    presenter._msdf_font_atlas = MsdfFontAtlas()
    presenter._vulkan_msdf_quad_renderer = object()
    presenter._cursor_overlay_specs = lambda *_args: []
    presenter._upload_tool_quad = lambda *args: args

    specs = presenter._render_tool_quad_layers()
    guide = next(spec for spec in specs if spec[0] == "screen_help")

    assert guide[3][1] == pytest.approx(12.988)
    assert isinstance(guide[1], VulkanMsdfQuadRequest)
    assert max(run["scale"] for run in guide[1].runs) == pytest.approx(
        21.0 / presenter._msdf_font_atlas.line_height
    )


def test_vertical_msdf_operation_guide_contains_all_legacy_rows() -> None:
    rows, _environment_rows = get_controller_help_rows("CN")
    request = _build_msdf_help_panel(
        MsdfFontAtlas(), rows, two_columns=False
    )
    rendered_text = {str(run["text"]) for run in request.runs}

    for row in rows:
        for text in row[:3]:
            if text:
                assert text in rendered_text


def test_keyboard_pose_faces_head_independently_of_profile_screen_rotation() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = (
        (0.0, 1.0, -2.0), 2.4, 1.35, (35.0, 20.0, 10.0)
    )
    presenter._head_position_w = np.asarray((0.8, 1.2, 0.0), dtype=np.float64)

    pose = presenter._keyboard_pose_mat4()
    toward_head = presenter._head_position_w - pose[:3, 3]
    toward_head /= np.linalg.norm(toward_head)

    assert float(np.dot(pose[:3, 2], toward_head)) > 0.99


def test_laser_cursor_ring_is_emitted_at_screen_hit() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = (
        (0.0, 0.0, -2.0), 2.4, 1.35, (0.0, 0.0, 0.0)
    )
    presenter._head_position_w = np.asarray((0.0, 0.0, 0.0), dtype=np.float64)
    presenter._aim_mat_r = np.eye(4, dtype=np.float32)
    presenter._grip_mat_r = np.eye(4, dtype=np.float32)

    specs = presenter._cursor_overlay_specs(
        np.ones((64, 64, 4), dtype=np.uint8),
        presenter._filament_screen_pose_mat4(),
        presenter._head_position_w,
    )

    assert len(specs) == 1
    assert specs[0][0] == "laser_cursor_1"
    assert specs[0][2][2] > -2.0


def test_vulkan_shortcuts_cycle_screen_preset_and_background() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._head_position_w = (0.0, 0.0, 0.0)
    presenter._head_forward_w = (0.0, 0.0, -1.0)
    presenter._filament_screen = (
        (0.0, 0.0, -16.0),
        16.0,
        9.0,
        (0.0, 0.0, 0.0),
    )

    presenter._dispatch_controller_shortcut("cycle_screen_preset")

    position, width, height, _rotation = presenter._filament_screen
    assert position == pytest.approx((0.0, 0.0, -20.0))
    assert width == pytest.approx(22.0)
    assert height == pytest.approx(12.375)
    assert presenter._preset_name_overlay.startswith('1000" IMAX')
    assert presenter._preset_osd_show_t > 0.0

    presenter._dispatch_controller_shortcut("toggle_background")
    assert presenter._filament_skybox_brightness == pytest.approx(0.0)
    presenter._dispatch_controller_shortcut("toggle_background")
    assert presenter._filament_skybox_brightness == pytest.approx(1.0)


def test_x_long_press_action_cycles_v25_glow_modes_not_room_lighting() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_glow_mode = "off"
    presenter._filament_glow_intensity_multiplier = 0.0

    observed = []
    for _ in range(4):
        presenter._dispatch_controller_shortcut("cycle_environment_light")
        observed.append(presenter._filament_glow_mode)

    assert observed == ["surround", "glow", "veil", "off"]
    assert presenter._filament_glow_intensity_multiplier == pytest.approx(0.0)
    assert presenter._filament_glow_shell_intensity_multiplier == pytest.approx(0.0)
    assert presenter._preset_name_overlay == "Off"


def test_screen_adjustment_osd_is_submitted_as_quad_layer() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = (
        (0.0, 0.0, -2.0), 2.4, 1.35, (0.0, 0.0, 0.0)
    )
    presenter._head_position_w = np.asarray((0.0, 0.0, 0.0), dtype=np.float64)
    presenter._screen_osd_show_t = -999.0
    presenter._adjust_shortcut_screen_size(0.1, 0.0)
    presenter._frame_now = presenter._screen_osd_show_t
    presenter.xr = object()
    presenter.session = object()
    presenter.vulkan = object()
    presenter._cursor_overlay_specs = lambda *_args: []
    presenter._upload_tool_quad = lambda *args: args

    specs = presenter._render_tool_quad_layers()

    osd_specs = [spec for spec in specs if spec[0] == "screen_osd"]
    assert len(osd_specs) == 1
    assert osd_specs[0][3] == pytest.approx((2.5 * 0.03 * (512.0 / 78.0), 2.5 * 0.03))
    assert osd_specs[0][2][1] > presenter._filament_screen[0][1]

    presenter._filament_screen = (
        (0.0, 0.0, -20.0), 4.8, 2.7, (0.0, 0.0, 0.0)
    )
    presenter._screen_osd_show_t = presenter._frame_now
    large_specs = presenter._render_tool_quad_layers()
    large_osd = [spec for spec in large_specs if spec[0] == "screen_osd"]

    assert large_osd[0][3] == pytest.approx((4.8 * 0.03 * (512.0 / 78.0), 4.8 * 0.03))


def test_screen_osd_keeps_text_in_quad_texture_when_msdf_abi_exists() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = (
        (0.0, 0.0, -2.0), 2.4, 1.35, (0.0, 0.0, 0.0)
    )
    presenter._head_position_w = np.asarray((0.0, 0.0, 0.0), dtype=np.float64)
    presenter._screen_osd_show_t = 1.0
    presenter._frame_now = 1.1
    presenter.xr = object()
    presenter.session = object()
    presenter.vulkan = object()
    presenter.filament_bridge = SimpleNamespace(text_overlay_abi_available=True)
    presenter._msdf_font_atlas = MsdfFontAtlas()
    presenter._cursor_overlay_specs = lambda *_args: []
    presenter._upload_tool_quad = lambda *args: args
    presenter._submit_msdf_text_runs = lambda *_args: pytest.fail(
        "screen OSD must not submit Projection-layer MSDF geometry"
    )

    specs = presenter._render_tool_quad_layers()
    osd = next(spec for spec in specs if spec[0] == "screen_osd")

    # The panel itself has alpha 210; text glyphs have alpha 255. Seeing 255
    # proves the complete text+background bitmap was kept in the Quad layer.
    assert int(np.max(osd[1][..., 3])) == 255


def test_screen_osd_selects_gpu_msdf_request_for_quad_upload() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = (
        (0.0, 0.0, -2.0), 2.4, 1.35, (0.0, 0.0, 0.0)
    )
    presenter._head_position_w = np.asarray((0.0, 0.0, 0.0), dtype=np.float64)
    presenter._screen_osd_show_t = 1.0
    presenter._frame_now = 1.1
    presenter.xr = object()
    presenter.session = object()
    presenter.vulkan = object()
    presenter._msdf_font_atlas = MsdfFontAtlas()
    presenter._vulkan_msdf_quad_renderer = object()
    presenter._cursor_overlay_specs = lambda *_args: []
    presenter._upload_tool_quad = lambda *args: args

    specs = presenter._render_tool_quad_layers()
    osd = next(spec for spec in specs if spec[0] == "screen_osd")

    assert isinstance(osd[1], VulkanMsdfQuadRequest)
    assert osd[1].height < 78
    assert osd[1].width < 512
    assert osd[1].width > osd[1].height
    assert osd[3][0] / osd[3][1] == pytest.approx(
        osd[1].width / osd[1].height
    )


def test_fps_and_operation_guides_select_gpu_msdf_quad_requests() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = (
        (0.0, 0.0, -2.0), 2.4, 1.35, (0.0, 0.0, 0.0)
    )
    presenter._head_position_w = np.asarray((0.0, 0.0, 0.0), dtype=np.float64)
    presenter._frame_now = 1.1
    presenter._fps_overlay_visible = True
    presenter._screen_operation_guide_visible = True
    presenter._hand_operation_guide_visible = True
    presenter.xr = object()
    presenter.session = object()
    presenter.vulkan = object()
    presenter._msdf_font_atlas = MsdfFontAtlas()
    presenter._vulkan_msdf_quad_renderer = object()
    presenter._controller_overlay_pose = lambda *_args: (
        (0.0, 0.0, -0.5),
        (0.0, 0.0, 0.0, 1.0),
    )
    presenter._cursor_overlay_specs = lambda *_args: []
    presenter._upload_tool_quad = lambda *args: args

    specs = presenter._render_tool_quad_layers()
    by_name = {
        spec[0]: spec[1]
        for spec in specs
        if spec[0] in {"screen_fps", "screen_help", "hand_help"}
    }

    assert set(by_name) == {"screen_fps", "screen_help", "hand_help"}
    assert all(isinstance(value, VulkanMsdfQuadRequest) for value in by_name.values())
    assert all(value.width > 0 and value.height > 0 for value in by_name.values())


def test_msdf_osd_canvas_width_follows_text_advance() -> None:
    atlas = MsdfFontAtlas()
    short_width, short_height, _ = _layout_msdf_osd_runs(
        atlas,
        (
            {"text": "Preset", "color": (255, 255, 255, 255)},
            {"text": "A", "color": (0, 255, 255, 255)},
        ),
    )
    long_width, long_height, _ = _layout_msdf_osd_runs(
        atlas,
        (
            {"text": "Preset", "color": (255, 255, 255, 255)},
            {"text": "Headset Recommended", "color": (0, 255, 255, 255)},
        ),
    )

    assert long_width > short_width
    assert long_height == short_height


def test_vulkan_reset_screen_restores_only_initial_size_and_position() -> None:
    presenter = OpenXrVulkanPresenter()
    initial = ((0.0, 0.0, -2.5), 2.4, 1.35, (0.0, 0.0, 0.0))
    presenter._filament_screen_initial = initial
    presenter._filament_screen_profile_authored = True
    presenter._filament_screen = (
        (1.0, 0.5, -20.0), 22.0, 12.375, (5.0, 10.0, 0.0)
    )
    presenter._screen_curve_half_angle = math.radians(16.0)
    presenter._screen_crop_width_percent = 12.0
    presenter._screen_crop_height_percent = 9.0
    presenter._screen_dynamic_crop = True

    presenter._dispatch_controller_shortcut("reset_screen")

    assert presenter._filament_screen == (
        initial[0], initial[1], initial[2], (5.0, 10.0, 0.0)
    )
    assert presenter._screen_curve_half_angle == pytest.approx(math.radians(16.0))
    assert presenter._screen_crop_width_percent == 12.0
    assert presenter._screen_crop_height_percent == 9.0
    assert presenter._screen_dynamic_crop is True


def test_right_grip_moves_screen_without_resizing_it() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = (
        (0.0, 0.0, -2.0), 2.4, 1.35, (0.0, 0.0, 0.0)
    )
    presenter._aim_mat_r = np.eye(4, dtype=np.float32)
    presenter._grip_mat_r = np.eye(4, dtype=np.float32)
    presenter._controller_inputs = ({}, {"grip": 1.0})

    presenter._handle_vulkan_pointer_input()
    presenter._grip_mat_r[0, 3] = 0.2
    presenter._handle_vulkan_pointer_input()

    position, width, height, _rotation = presenter._filament_screen
    assert position == pytest.approx((0.2, 0.0, -2.0))
    assert width == pytest.approx(2.4)
    assert height == pytest.approx(1.35)


def _prepare_pointer_priority_presenter(monkeypatch):
    generation = {"value": 0}
    monkeypatch.setattr(
        core_openxr_vulkan,
        "_physical_input_generation",
        lambda: generation["value"],
    )
    presenter = OpenXrVulkanPresenter()
    presenter._controller_inputs = ({}, {"trigger": 0.0})
    presenter._screen_ray_hit_for_hand = lambda hand: (
        (0.5, 0.5) if hand == 1 else None
    )
    presenter._hand_keyboard_hit = lambda _hand: False
    presenter._last_physical_input_generation = 0
    return presenter, generation


def test_physical_mouse_event_blocks_remote_cursor_and_resumes_next_frame(monkeypatch) -> None:
    presenter, generation = _prepare_pointer_priority_presenter(monkeypatch)
    cursor_positions = []
    monkeypatch.setattr(
        core_openxr_vulkan,
        "_set_cursor_pos",
        lambda *position: cursor_positions.append(position),
    )

    generation["value"] = 1
    presenter._handle_vulkan_pointer_input()
    assert cursor_positions == []

    presenter._handle_vulkan_pointer_input()
    assert len(cursor_positions) == 1


def test_physical_keyboard_event_blocks_remote_cursor(monkeypatch) -> None:
    presenter, generation = _prepare_pointer_priority_presenter(monkeypatch)
    cursor_positions = []
    monkeypatch.setattr(
        core_openxr_vulkan,
        "_set_cursor_pos",
        lambda *position: cursor_positions.append(position),
    )

    generation["value"] = 1
    presenter._handle_vulkan_pointer_input()

    assert cursor_positions == []


def test_repeated_physical_events_keep_remote_cursor_blocked(monkeypatch) -> None:
    presenter, generation = _prepare_pointer_priority_presenter(monkeypatch)
    cursor_positions = []
    monkeypatch.setattr(
        core_openxr_vulkan,
        "_set_cursor_pos",
        lambda *position: cursor_positions.append(position),
    )

    for value in (1, 2, 3):
        generation["value"] = value
        presenter._handle_vulkan_pointer_input()

    assert cursor_positions == []


def test_physical_event_cancels_remote_mouse_and_touch_drag(monkeypatch) -> None:
    presenter, generation = _prepare_pointer_priority_presenter(monkeypatch)
    mouse_events = []
    monkeypatch.setattr(
        core_openxr_vulkan,
        "_send_mouse_flags",
        lambda flag: mouse_events.append(flag),
    )
    presenter._pointer_state["right"] = "dragging"

    class FakeTouchInjector:
        available = True

        def __init__(self):
            self.events = []

        def set(self, contact_id, x, y, want_down):
            self.events.append((contact_id, x, y, want_down))

        def flush(self):
            self.events.append("flush")

    injector = FakeTouchInjector()
    presenter._touch_state["right"] = "down"
    presenter._touch_px["right"] = (321, 654)
    monkeypatch.setattr(core_openxr_vulkan, "_TOUCH_AVAILABLE", True)
    monkeypatch.setattr(core_openxr_vulkan, "_touch_injector", injector)

    generation["value"] = 1
    presenter._handle_vulkan_pointer_input()

    assert presenter._pointer_state["right"] == "idle"
    assert presenter._touch_state["right"] == "idle"
    release_events = [
        event for event in injector.events if isinstance(event, tuple) and not event[3]
    ]
    assert release_events == [(1, 321, 654, False)]
    assert mouse_events == [core_openxr_vulkan._MOUSEEVENTF_LEFTUP]


def test_non_windows_physical_input_generation_is_noop() -> None:
    if sys.platform == "win32":
        pytest.skip("non-Windows fallback behavior")
    from xr_viewer import windows_input

    assert windows_input._physical_input_generation() == 0


def test_right_grip_moves_keyboard_using_laser_local_anchor() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = (
        (0.0, 0.0, -2.0), 2.4, 1.35, (0.0, 0.0, 0.0)
    )
    presenter._head_position_w = np.asarray((0.0, 0.0, 0.0), dtype=np.float64)
    presenter._keyboard_visible = True
    presenter._kb_hover_r = 0
    presenter._aim_mat_r = np.eye(4, dtype=np.float32)
    presenter._grip_mat_r = np.eye(4, dtype=np.float32)
    presenter._controller_inputs = ({}, {"grip": 1.0})

    initial_pose = presenter._keyboard_pose_mat4()
    ray = {
        "origin": initial_pose[:3, 3] - initial_pose[:3, 2],
        "direction": initial_pose[:3, 2].copy(),
    }
    presenter._controller_interaction_ray = lambda _hand: (
        ray["origin"], ray["direction"]
    )

    presenter._handle_vulkan_pointer_input()
    assert presenter._grip_target_r == "keyboard"
    assert presenter._kb_grab_local_r is not None

    ray["origin"] = (
        initial_pose[:3, 3]
        + initial_pose[:3, 0] * 0.3
        - initial_pose[:3, 2]
    )
    presenter._handle_vulkan_pointer_input()

    moved_position = presenter._keyboard_pose_mat4()[:3, 3]
    assert moved_position[0] > 0.1


def test_right_grip_drag_orbits_screen_around_head_and_faces_head() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = (
        (0.0, 0.0, -2.0), 2.4, 1.35, (0.0, 0.0, 0.0)
    )
    presenter._head_position_w = np.asarray((0.0, 0.0, 0.0), dtype=np.float64)
    presenter._aim_mat_r = np.eye(4, dtype=np.float32)
    presenter._grip_mat_r = np.eye(4, dtype=np.float32)
    presenter._controller_inputs = ({}, {"grip": 1.0})

    presenter._handle_vulkan_pointer_input()
    presenter._grip_mat_r[0, 3] = 0.4
    presenter._handle_vulkan_pointer_input()

    position, _width, _height, rotation = presenter._filament_screen
    assert np.linalg.norm(np.asarray(position)) == pytest.approx(2.0)
    pose = presenter._filament_screen_pose_mat4()
    toward_head = -np.asarray(position, dtype=np.float64)
    toward_head /= np.linalg.norm(toward_head)
    assert float(np.dot(pose[:3, 2], toward_head)) > 0.99
    assert rotation[0] != pytest.approx(0.0)


def test_screen_drag_uses_the_calibrated_visible_controller_ray() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = (
        (0.0, 0.0, -2.0), 2.4, 1.35, (0.0, 0.0, 0.0)
    )
    angle = math.radians(-19.0)
    presenter._aim_mat_r = np.asarray(
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, math.cos(angle), -math.sin(angle), 0.0),
            (0.0, math.sin(angle), math.cos(angle), 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )
    presenter._grip_mat_r = np.eye(4, dtype=np.float32)
    presenter._controller_inputs = ({}, {"grip": 1.0})

    # The uncalibrated Aim -Z ray misses the screen, while the visible legacy
    # laser ray (12 degrees around local X) lands inside it.
    assert presenter._screen_ray_hit(presenter._aim_mat_r) is None
    assert presenter._screen_ray_hit_for_hand(1) is not None

    presenter._handle_vulkan_pointer_input()

    assert presenter._grip_target_r == "screen"


def test_screen_edge_ray_snaps_to_legacy_edge() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = (
        (0.0, 0.0, -2.0), 2.4, 1.35, (0.0, 0.0, 0.0)
    )
    yaw = math.radians(-35.0)
    presenter._aim_mat_r = np.asarray(
        (
            (math.cos(yaw), 0.0, math.sin(yaw), 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (-math.sin(yaw), 0.0, math.cos(yaw), 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )
    presenter._grip_mat_r = np.eye(4, dtype=np.float32)
    presenter._smooth_ray_origin_r = np.asarray(
        (0.0, 0.0, 0.0), dtype=np.float64
    )
    presenter._smooth_ray_fwd_r = (
        -presenter._aim_mat_r[:3, 2].astype(np.float64)
    )

    # The calibrated ray is just outside the finite screen. The legacy
    # six-degree edge cone must clamp it to the nearest screen edge.
    origin = presenter._smooth_ray_origin_r
    raw_origin = presenter._grip_mat_r[:3, 3] + presenter._grip_mat_r[:3, 1] * 0.020
    raw_direction = -presenter._aim_mat_r[:3, 2].astype(np.float64)
    right_axis = presenter._aim_mat_r[:3, 0].astype(np.float64)
    angle = math.radians(12.0)
    raw_direction = (
        raw_direction * math.cos(angle)
        + np.cross(right_axis, raw_direction) * math.sin(angle)
        + right_axis * np.dot(right_axis, raw_direction) * (1.0 - math.cos(angle))
    )
    assert presenter._screen_plane_uv(raw_origin, raw_direction)[0] > 1.0
    assert presenter._screen_ray_hit(
        presenter._aim_mat_r, raw_origin, raw_direction
    ) is None
    hit = presenter._screen_ray_hit_for_hand(1)
    assert hit is not None
    assert hit[0] == pytest.approx(1.0)


def test_left_grip_rotation_snaps_screen_to_quarter_turn() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = (
        (0.0, 0.0, -2.0), 2.4, 1.35, (0.0, 0.0, 0.0)
    )
    presenter._grip_mat_l = np.eye(4, dtype=np.float32)
    presenter._grip_rotation_anchor_l = np.eye(3, dtype=np.float64)
    presenter._screen_rotation_anchor_l = (0.0, 0.0, 0.0)
    angle = math.radians(100.0)
    presenter._grip_mat_l[:3, :3] = np.asarray(
        (
            (math.cos(angle), -math.sin(angle), 0.0),
            (math.sin(angle), math.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )

    presenter._apply_grip_screen_rotation(0)

    assert presenter._filament_screen[3] == pytest.approx((0.0, 0.0, 90.0))


def test_left_grip_input_records_anchor_and_snaps_screen_by_ninety_degrees() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = (
        (0.0, 0.0, -2.0), 2.4, 1.35, (0.0, 0.0, 0.0)
    )
    presenter._aim_mat_l = np.eye(4, dtype=np.float32)
    presenter._grip_mat_l = np.eye(4, dtype=np.float32)
    presenter._controller_inputs = ({"grip": 1.0}, {})

    presenter._handle_vulkan_pointer_input()
    assert presenter._grip_rotation_anchor_l is not None
    assert presenter._screen_rotation_anchor_l == pytest.approx((0.0, 0.0, 0.0))

    angle = math.radians(50.0)
    presenter._grip_mat_l[:3, :3] = np.asarray(
        (
            (math.cos(angle), -math.sin(angle), 0.0),
            (math.sin(angle), math.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )
    presenter._handle_vulkan_pointer_input()

    assert presenter._filament_screen[3][2] == pytest.approx(90.0)


def test_right_grip_stick_y_uses_legacy_accelerated_radial_distance() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._head_position_w = np.asarray((0.0, 0.0, 0.0), dtype=np.float64)
    presenter._filament_screen = (
        (0.0, 0.0, -2.0), 2.4, 1.35, (0.0, 0.0, 0.0)
    )

    presenter._apply_right_grip_screen_distance(
        -1.0,
        dt=1.0 / 90.0,
        laser_hit=(0.5, 0.5, 2.0),
    )

    first_frame_speed = (
        presenter._screen_control_min_speed
        + presenter._screen_control_acceleration / 90.0
    )
    assert presenter._filament_screen[0] == pytest.approx(
        (0.0, 0.0, -2.0 - first_frame_speed / 90.0)
    )


def test_right_grip_stick_x_resizes_screen_without_22m_cap() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = (
        (0.0, 0.0, -2.0), 22.0, 12.375, (0.0, 0.0, 0.0)
    )

    presenter._apply_right_grip_screen_resize(
        1.0,
        dt=5.0,
        laser_hit=(0.5, 0.5, 2.0),
    )

    expected_width = 22.0 + presenter._screen_control_max_speed * 5.0
    assert presenter._filament_screen[1] == pytest.approx(expected_width)
    assert presenter._filament_screen[2] == pytest.approx(
        12.375 * expected_width / 22.0
    )


def test_screen_control_speed_ramps_from_precision_to_ten_meters_per_second() -> None:
    presenter = OpenXrVulkanPresenter()

    first = presenter._screen_hold_speed(
        1.0, dt=1.0 / 90.0, control="size"
    )
    one_second = presenter._screen_hold_speed(
        1.0, dt=1.0 - 1.0 / 90.0, control="size"
    )
    held = presenter._screen_hold_speed(
        1.0, dt=4.0, control="size"
    )

    assert first == pytest.approx(
        presenter._screen_control_min_speed
        + presenter._screen_control_acceleration / 90.0
    )
    assert one_second == pytest.approx(2.08)
    assert held == pytest.approx(presenter._screen_control_max_speed)


def test_screen_control_deadzone_remaps_partial_stick_for_precision() -> None:
    presenter = OpenXrVulkanPresenter()

    assert presenter._screen_control_axis_value(0.19) == pytest.approx(0.0)
    assert presenter._screen_control_axis_value(-0.60) == pytest.approx(-0.5)


def test_screen_control_stick_selects_only_one_axis_with_hysteresis() -> None:
    presenter = OpenXrVulkanPresenter()

    axis, value = presenter._screen_control_stick_axis(0.8, 0.7)
    assert axis == "size"
    assert value > 0.0
    axis, value = presenter._screen_control_stick_axis(0.7, 0.8)
    assert axis == "size"
    assert value > 0.0
    axis, value = presenter._screen_control_stick_axis(0.2, 0.9)
    assert axis == "distance"
    assert value > 0.0
    axis, value = presenter._screen_control_stick_axis(0.05, 0.05)
    assert axis is None
    assert value == pytest.approx(0.0)


def test_pointer_exponential_resize_is_not_followed_by_fixed_guide_delta() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = (
        (0.0, 0.0, -2.0), 2.4, 1.35, (0.0, 0.0, 0.0)
    )
    presenter._head_position_w = np.asarray((0.0, 0.0, 0.0), dtype=np.float64)

    presenter._apply_right_grip_screen_distance(
        -0.5,
        dt=1.0 / 90.0,
        laser_hit=(0.5, 0.5, 2.0),
    )
    expected_position = presenter._filament_screen[0]

    presenter._right_grip_screen_pointer_applied = True
    presenter._dispatch_controller_shortcut(
        "resize_screen", width_delta=0.6, distance_delta=0.5
    )

    assert presenter._filament_screen[0] == pytest.approx(expected_position)
    assert presenter._filament_screen[1] == pytest.approx(2.4)


def test_right_grip_wrist_rotation_does_not_rotate_screen() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = (
        (0.0, 0.0, -2.0), 2.4, 1.35, (10.0, 5.0, 3.0)
    )
    presenter._grip_mat_r = np.eye(4, dtype=np.float32)
    presenter._grip_rotation_anchor_r = np.eye(3, dtype=np.float64)
    presenter._screen_rotation_anchor_r = (10.0, 5.0, 3.0)
    presenter._grip_mat_r[:3, :3] = np.asarray(
        (
            (0.0, -1.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )

    presenter._apply_grip_screen_rotation(1)

    assert presenter._filament_screen[3] == pytest.approx((10.0, 5.0, 3.0))


def test_keyboard_world_position_is_converted_to_screen_relative_offset() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = (
        (1.0, 2.0, -3.0), 2.4, 1.0, (0.0, 0.0, 0.0)
    )

    presenter._set_keyboard_world_position((1.5, 1.0, -2.5))

    assert presenter._keyboard_pose_mat4()[:3, 3] == pytest.approx(
        (1.5, 1.0, -2.5)
    )


def test_continuous_screen_shortcuts_apply_only_while_laser_hits_screen() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = (
        (0.0, 0.0, -2.0), 2.4, 1.35, (0.0, 0.0, 0.0)
    )
    presenter._head_position_w = (0.0, 0.0, 0.0)
    presenter._aim_mat_l = np.eye(4, dtype=np.float32)
    presenter._aim_mat_r = np.eye(4, dtype=np.float32)

    presenter._dispatch_controller_shortcut(
        "rotate_screen", yaw_delta=10.0, pitch_delta=5.0
    )
    presenter._dispatch_controller_shortcut(
        "resize_screen", width_delta=0.6, distance_delta=0.5
    )

    position, width, _height, rotation = presenter._filament_screen
    assert rotation == pytest.approx((10.0, 5.0, 0.0))
    assert width == pytest.approx(3.0)
    assert np.linalg.norm(np.asarray(position)) == pytest.approx(2.5)


def test_controller_brand_switch_and_calibration_save_use_live_profile(
    tmp_path,
) -> None:
    class Bridge:
        def __init__(self) -> None:
            self.loaded: list[tuple[int, bytes]] = []

        def load_controller(self, hand: int, data: bytes) -> None:
            self.loaded.append((hand, data))

    def brand(name: str, marker: bytes):
        root = tmp_path / name
        root.mkdir()
        left = root / "left.glb"
        right = root / "right.glb"
        left.write_bytes(marker + b"L")
        right.write_bytes(marker + b"R")
        (root / "profile.json").write_text("{}\n", encoding="utf-8")
        return SimpleNamespace(
            name=name,
            root=root,
            left_glb=left,
            right_glb=right,
            offset=(0.0, 0.0, 0.0),
            rotation_deg=0.0,
        )

    first = brand("A", b"A")
    second = brand("B", b"B")
    presenter = OpenXrVulkanPresenter()
    presenter._controller_brands = {"A": first, "B": second}
    presenter._controller_brand = first
    presenter.filament_bridge = Bridge()

    presenter._dispatch_controller_shortcut("switch_controller_brand")
    presenter._controller_calibration_offset[:] = (0.1, 0.2, 0.3)
    presenter._controller_calibration_rotation_deg = 12.5
    presenter._controller_calibration_mode = True
    presenter._dispatch_controller_shortcut("save_controller_calibration")

    profile = json.loads((second.root / "profile.json").read_text(encoding="utf-8"))
    assert presenter._controller_brand is second
    assert presenter.filament_bridge.loaded == [(0, b"BL"), (1, b"BR")]
    assert profile["overrides"] == {
        "model_offset": [0.1, 0.2, 0.3],
        "model_rotation_deg": 12.5,
    }
    assert presenter._controller_calibration_mode is False


def test_none_controller_model_without_room_bypasses_filament_and_enables_vulkan_proxy(
    capsys,
) -> None:
    presenter = OpenXrVulkanPresenter(
        OpenXrVulkanConfig(
            controller_model="None",
            filament_bridge_path="must-not-be-loaded.dll",
        )
    )

    presenter._initialize_filament_bridges()

    assert presenter._vulkan_controller_proxy_enabled is True
    assert presenter._controller_brand.profile_id == "none"
    assert presenter._controller_brand.left_glb is None
    assert presenter._controller_brand.right_glb is None
    assert presenter._controller_calibration_offset == pytest.approx(
        presenter._controller_brand.offset
    )
    assert presenter._controller_calibration_rotation_deg == pytest.approx(
        presenter._controller_brand.rotation_deg
    )
    assert presenter.filament_bridge is None
    assert "controller GLB and unused Filament engine are bypassed" in (
        capsys.readouterr().out
    )


def test_none_controller_model_keeps_filament_room_enabled(monkeypatch, tmp_path) -> None:
    room_path = tmp_path / "room.glb"
    room_path.write_bytes(b"textured-room")
    calls: list[tuple[str, object]] = []

    class FakeBridge:
        multiview_abi_available = False

        def __init__(self, path):
            calls.append(("bridge", path))

        def create(self, **_kwargs):
            calls.append(("create", None))

        def create_eye_swapchain(self, eye_index, images, **_kwargs):
            calls.append(("swapchain", (eye_index, list(images))))

        def load_glb(self, data):
            calls.append(("room", data))

        def load_controller(self, *_args):
            raise AssertionError("NONE must not load controller GLBs")

        def set_scene_exposure(self, _value):
            pass

        def set_skybox_brightness(self, _value):
            pass

        def set_fill_light(self, _color, _intensity, _direction):
            pass

        def close(self):
            pass

    import xr_viewer.filament_vulkan_bridge as bridge_module

    monkeypatch.setattr(bridge_module, "FilamentVulkanBridge", FakeBridge)
    presenter = OpenXrVulkanPresenter(
        OpenXrVulkanConfig(
            controller_model="NONE",
            filament_bridge_path="bridge.dll",
            filament_glb_path=str(room_path),
        )
    )
    presenter.vulkan = SimpleNamespace(
        instance=1,
        physical_device=2,
        device=3,
        queue_family_index=4,
    )
    presenter.swapchain_format = 43
    presenter.swapchains = [
        _EyeSwapchain("left", [SimpleNamespace(image="left")], 10, 20),
        _EyeSwapchain("right", [SimpleNamespace(image="right")], 10, 20),
    ]

    presenter._initialize_filament_bridges()

    assert presenter.filament_bridge is not None
    assert ("room", b"textured-room") in calls


def test_none_controller_model_never_updates_unloaded_filament_controllers() -> None:
    presenter = OpenXrVulkanPresenter(OpenXrVulkanConfig(controller_model="NONE"))

    class EnvironmentOnlyBridge:
        controller_abi_available = True
        controller_visibility_abi_available = True
        controller_guide_abi_available = True

        def __getattr__(self, name):
            if name.startswith("set_controller"):
                raise AssertionError(f"unexpected Filament controller call: {name}")
            raise AttributeError(name)

    presenter._update_filament_controllers(EnvironmentOnlyBridge())


def test_none_controller_model_rejects_controller_brand_switch() -> None:
    presenter = OpenXrVulkanPresenter(OpenXrVulkanConfig(controller_model="None"))

    presenter._dispatch_controller_shortcut("switch_controller_brand")

    assert presenter._controller_brand.profile_id == "none"


def test_none_controller_model_has_explicit_fps_panel_name() -> None:
    presenter = OpenXrVulkanPresenter(OpenXrVulkanConfig(controller_model="None"))

    assert presenter._controller_model_display_name() == "None"


def test_vulkan_controller_proxy_packs_both_openxr_grip_poses() -> None:
    presenter = OpenXrVulkanPresenter(OpenXrVulkanConfig(controller_model="None"))
    presenter._frame_now = 12.0
    presenter._laser_last_move_l = 11.0
    presenter._laser_last_move_r = 11.0
    presenter._grip_mat_l = np.eye(4, dtype=np.float32)
    presenter._grip_mat_r = np.eye(4, dtype=np.float32)
    presenter._aim_mat_l = np.eye(4, dtype=np.float32)
    presenter._aim_mat_r = np.eye(4, dtype=np.float32)
    presenter._grip_mat_l[:3, 3] = (1.0, 2.0, 3.0)
    presenter._grip_mat_r[:3, 3] = (-1.0, -2.0, -3.0)
    presenter._controller_interaction_ray = lambda hand: (
        np.asarray((float(hand), 0.0, 0.0), dtype=np.float64),
        np.asarray((0.0, 0.0, -1.0), dtype=np.float64),
    )

    payload = presenter._projection_controller_proxy_params()

    assert payload is not None
    assert len(payload) == VulkanProjectionScreenPass._CONTROLLER_OVERLAY_PARAM_SIZE
    values = np.frombuffer(payload, dtype="<f4")
    left = values[:16].reshape((4, 4), order="F")
    right = values[16:32].reshape((4, 4), order="F")
    offset = np.eye(4, dtype=np.float32)
    offset[:3, 3] = presenter._controller_calibration_offset
    rotation = euler_to_mat4(
        0.0, math.radians(presenter._controller_calibration_rotation_deg), 0.0
    )
    assert left == pytest.approx(presenter._grip_mat_l @ rotation @ offset)
    assert right == pytest.approx(presenter._grip_mat_r @ rotation @ offset)
    assert values[64:66] == pytest.approx((1.0, 1.0))
    assert values[68:71] == pytest.approx((12.0, 1.0, 1.0))

    vertex_shader = (
        APP_ROOT
        / "shaders"
        / "d2s_projection_controller_proxy_vert.vert"
    ).read_text(encoding="utf-8")
    fragment_shader = (
        APP_ROOT
        / "shaders"
        / "d2s_projection_controller_proxy_frag.frag"
    ).read_text(encoding="utf-8")
    assert "CUBE_VERTEX_COUNT = 36" in vertex_shader
    assert "face_position(face, corner) * 0.040" in vertex_shader
    assert "if (face == 2) return vec3(a, 1.0, -b);" in vertex_shader
    assert "if (face == 3) return vec3(a, -1.0, b);" in vertex_shader
    assert "high_y ? (1.0 / 6.0) : 0.5" in vertex_shader
    assert "laser_uv.y - laser_time * 0.4" in fragment_shader
    assert "if (!gl_FrontFacing)" in fragment_shader
    assert "noperspective in vec2 face_uv" in fragment_shader
    assert "fwidth(nearest_edge) * 1.5" in fragment_shader
    assert "mix(vec3(0.0), base_color, interior)" in fragment_shader

    presenter._frame_now = 16.01
    assert presenter._projection_controller_proxy_params() is None


def test_vulkan_controller_proxy_uses_front_right_top_corner_for_b_anchor() -> None:
    presenter = OpenXrVulkanPresenter(OpenXrVulkanConfig(controller_model="None"))
    presenter._grip_mat_r = np.eye(4, dtype=np.float32)
    presenter._grip_mat_r[:3, 3] = (1.0, 2.0, 3.0)

    anchor = presenter._resolve_controller_b_button_local()

    assert anchor == pytest.approx((0.040, 0.040, -0.040))
    offset = np.eye(4, dtype=np.float64)
    offset[:3, 3] = presenter._controller_calibration_offset
    rotation = euler_to_mat4(
        0.0, math.radians(presenter._controller_calibration_rotation_deg), 0.0
    ).astype(np.float64)
    expected = presenter._grip_mat_r @ rotation @ offset @ np.asarray(
        (0.040, 0.040, -0.040, 1.0), dtype=np.float64
    )
    assert presenter._controller_b_button_world_position() == pytest.approx(expected[:3])
    source = inspect.getsource(OpenXrVulkanPresenter._render_tool_quad_layers)
    assert '"controller_proxy_callout"' in source
    assert "build_controller_callout_rgba(lang=language)" in source


def test_vulkan_controller_proxy_uses_final_transparent_projection_layer() -> None:
    source = inspect.getsource(OpenXrVulkanPresenter._render_vulkan_controller_proxy_layer)

    assert '"clear_color": (0.0, 0.0, 0.0, 0.0)' in source
    assert "clear_target=True" in source
    assert "BLEND_TEXTURE_SOURCE_ALPHA_BIT" in source
    assert "order=projection->quad->controller/laser/guide" in source


def test_vulkan_controller_proxy_callout_is_ordered_after_menu() -> None:
    source = inspect.getsource(OpenXrVulkanPresenter._render_tool_quad_layers)

    assert "controller_callouts = [" in source
    assert "specs = [" in source
    assert "controller_proxy_callout" in source


def test_vulkan_shortcut_delegates_runtime_owned_actions() -> None:
    actions: list[str] = []
    presenter = OpenXrVulkanPresenter(
        on_controller_shortcut=lambda action: actions.append(action) or True
    )

    presenter._dispatch_controller_shortcut("toggle_stereo")
    presenter._dispatch_controller_shortcut("reset_depth")

    assert actions == ["toggle_stereo", "reset_depth"]
    assert presenter._unsupported_shortcut_actions == set()


def test_vulkan_shortcut_toggles_composer_curved_screen() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = (
        (0.0, 0.0, -2.0), 2.4, 1.35, (0.0, 0.0, 0.0)
    )

    presenter._dispatch_controller_shortcut("toggle_screen_shape")
    presenter._dispatch_controller_shortcut("toggle_screen_shape")

    assert presenter._screen_curved is False


def test_vulkan_shortcut_toggles_legacy_green_passthrough_backdrop() -> None:
    class Bridge:
        passthrough_backdrop_abi_available = True

        def __init__(self) -> None:
            self.values: list[bool] = []

        def set_passthrough_backdrop(self, enabled: bool) -> None:
            self.values.append(enabled)

    presenter = OpenXrVulkanPresenter()
    presenter.filament_bridge = Bridge()

    presenter._dispatch_controller_shortcut("toggle_passthrough")
    presenter._dispatch_controller_shortcut("toggle_passthrough")

    assert presenter.filament_bridge.values == [True, False]


def test_openxr_frame_gate_waits_for_runtime_output_before_filament() -> None:
    source = (APP_ROOT /
              "xr_viewer/core_openxr_vulkan.py").read_text(encoding="utf-8")

    assert "self._pending_output is None" in source
    assert "and not self._has_presented_frame" in source
    assert "and not self._projection_array_eye_diagnostic" in source
    assert "and not self._vulkan_multiview_eye_diagnostic" in source
    assert "and not self._filament_multiview_projection_diagnostic" in source
    assert "waiting for first runtime eye frame" in source
    assert "layer = self._render_projection_layer(views, output_frame)" in source
    assert "Vulkan Projection Composer" in source
    assert "bridge.set_screen_image(" not in source


def test_quad_layer_uses_runtime_output_size_and_openxr_visibility() -> None:
    source = (APP_ROOT /
              "xr_viewer/core_openxr_vulkan.py").read_text(encoding="utf-8")

    assert "_ensure_quad_swapchains(width, height)" in source
    assert '_select_swapchain_format(vk, formats, "srgb")' in source
    assert "flip_x=True" not in source
    assert "flip_y=False" in source


def test_tool_quad_layer_enables_unpremultiplied_source_alpha() -> None:
    source = (APP_ROOT /
              "xr_viewer/core_openxr_vulkan.py").read_text(encoding="utf-8")

    assert "CompositionLayerFlags.BLEND_TEXTURE_SOURCE_ALPHA_BIT" in source
    assert "CompositionLayerFlags.UNPREMULTIPLIED_ALPHA_BIT" in source
    assert "format_value if format_value is not None" in source
    assert "CompositionLayerQuad" in source
    assert "EyeVisibility.LEFT" in source


def test_profile_reference_space_is_shared_with_controller_pose_queries() -> None:
    source = (APP_ROOT /
              "xr_viewer/core_openxr_vulkan.py").read_text(encoding="utf-8")

    assert "self.reference_space = new_space" in source
    assert "self._xr_space = new_space" in source


def test_profile_screen_height_defaults_to_16_9_width() -> None:
    source = (APP_ROOT /
              "xr_viewer/core_openxr_vulkan.py").read_text(encoding="utf-8")

    assert 'float(screen.get("width", 2.4)) * 9.0 / 16.0' in source
    assert "EyeVisibility.RIGHT" in source
    assert "_has_presented_frame" in source
    assert "self._last_quad_layers" in source
    assert "Render the world at the current headset pose" in source


def test_vulkan_copy_allows_srgb_unorm_quad_conversion() -> None:
    source = (APP_ROOT /
              "viewer/vulkan_context.py").read_text(encoding="utf-8")

    assert "formats_are_srgb_compatible" in source
    assert "or not formats_match" in source


def test_profile_pose_uses_two_frame_closed_loop_reference_space_calibration() -> None:
    source = (APP_ROOT /
              "xr_viewer/core_openxr_vulkan.py").read_text(encoding="utf-8")

    assert "_apply_profile_reference_space(views)" in source
    assert "self._profile_space_calibration_pass >= 2" in source
    assert "self._profile_space_pose_in_reference.astype(np.float64) @ raw_head" in source
    assert "space_pose = reference_head @ np.linalg.inv(self._profile_head_transform)" in source
    assert "xr.ReferenceSpaceType.STAGE" in source
    assert "enumerate_reference_spaces(self.session)" in source


def test_profile_reference_space_feedback_removes_measured_stage_height() -> None:
    created_poses = []

    class FakeXr:
        ReferenceSpaceType = xr.ReferenceSpaceType
        ReferenceSpaceCreateInfo = xr.ReferenceSpaceCreateInfo

        @staticmethod
        def create_reference_space(_session, create_info):
            created_poses.append(create_info.pose_in_reference_space)
            return f"space-{len(created_poses)}"

        @staticmethod
        def destroy_space(_space):
            pass

    def views_at(y):
        return [
            SimpleNamespace(
                pose=SimpleNamespace(
                    position=SimpleNamespace(x=x, y=y, z=0.0),
                    orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
                )
            )
            for x in (-0.03, 0.03)
        ]

    presenter = OpenXrVulkanPresenter()
    presenter.xr = FakeXr
    presenter.session = object()
    presenter.reference_space = "base"
    presenter._xr_space = "base"
    presenter._reference_space_type = xr.ReferenceSpaceType.STAGE
    presenter._profile_head_transform = np.eye(4, dtype=np.float32)
    presenter._profile_head_transform[1, 3] = 10.0

    # The first startup locate can still report the provisional STAGE origin.
    assert presenter._apply_profile_reference_space(views_at(0.0))
    assert not presenter._profile_space_applied
    assert created_poses[0].position.y == pytest.approx(-10.0)

    # VDXR then settles one metre higher and reports target + physical height.
    # The feedback pass recovers the base-space head through the provisional
    # transform and compensates the measured metre without a fixed constant.
    assert presenter._apply_profile_reference_space(views_at(11.0))
    assert presenter._profile_space_applied
    assert created_poses[1].position.y == pytest.approx(-9.0)
    assert presenter._profile_space_calibration_pass == 2


def test_authored_profile_accepts_runtime_reference_space_change() -> None:
    destroyed = []

    class FakeXr:
        ReferenceSpaceCreateInfo = xr.ReferenceSpaceCreateInfo

        @staticmethod
        def create_reference_space(_session, _create_info):
            return "replacement-base-space"

        @staticmethod
        def destroy_space(space):
            destroyed.append(space)

    presenter = OpenXrVulkanPresenter()
    presenter.xr = FakeXr
    presenter.session = object()
    presenter._reference_space_type = xr.ReferenceSpaceType.STAGE
    presenter.reference_space = "calibrated-space"
    presenter._xr_space = "calibrated-space"
    presenter._profile_space_applied = True
    presenter._profile_space_calibration_pass = 2
    presenter._profile_auto_center_on_screen = False
    presenter._profile_reference_head_anchor = np.eye(4, dtype=np.float32)

    presenter._recreate_reference_space_after_runtime_change()

    assert presenter.reference_space == "replacement-base-space"
    assert presenter._xr_space == "replacement-base-space"
    assert not presenter._profile_space_applied
    assert presenter._profile_space_calibration_pass == 0
    assert presenter._profile_reference_head_anchor is None
    assert destroyed == ["calibrated-space"]


def test_auto_center_profile_recalibrates_after_reference_space_change() -> None:
    destroyed = []

    class FakeXr:
        ReferenceSpaceCreateInfo = xr.ReferenceSpaceCreateInfo

        @staticmethod
        def create_reference_space(_session, _create_info):
            return "replacement-base-space"

        @staticmethod
        def destroy_space(space):
            destroyed.append(space)

    presenter = OpenXrVulkanPresenter()
    presenter.xr = FakeXr
    presenter.session = object()
    presenter._reference_space_type = xr.ReferenceSpaceType.STAGE
    presenter.reference_space = "calibrated-space"
    presenter._xr_space = "calibrated-space"
    presenter._profile_space_applied = True
    presenter._profile_space_calibration_pass = 2
    presenter._profile_auto_center_on_screen = True

    presenter._recreate_reference_space_after_runtime_change()

    assert presenter.reference_space == "replacement-base-space"
    assert presenter._xr_space == "replacement-base-space"
    assert not presenter._profile_space_applied
    assert presenter._profile_space_calibration_pass == 0
    assert destroyed == ["calibrated-space"]


def test_filament_profile_keeps_glb_and_screen_positions_separate(tmp_path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps({
        "model_position": [0.0, 0.0, 0.0],
        "view_poses": [{"x": 1.0, "y": 2.0, "z": 3.0}],
        "screen": {"position": [10.0, 20.0, 30.0]},
    }), encoding="utf-8")
    presenter = OpenXrVulkanPresenter(OpenXrVulkanConfig(
        filament_profile_path=str(profile_path),
    ))
    presenter._load_filament_profile()
    assert presenter._profile_head_transform[:3, 3].tolist() == [1.0, 2.0, 3.0]
    assert presenter._filament_screen[0] == (10.0, 20.0, 30.0)


def test_filament_profile_restores_authored_screen_curve(tmp_path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps({
        "screen": {
            "position": [0.0, 1.2, -2.0],
            "curved": True,
            "curve_half_angle_rad": math.radians(30.0),
        },
    }), encoding="utf-8")
    presenter = OpenXrVulkanPresenter(OpenXrVulkanConfig(
        filament_profile_path=str(profile_path),
    ))

    presenter._load_filament_profile()

    assert presenter._screen_curved is True
    assert presenter._screen_curve_half_angle == pytest.approx(math.radians(30.0))


def test_glb_room_exposure_always_starts_at_zero_for_new_session(tmp_path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps({
        "glb": "environment.glb",
        "preview_exposure": 4.0,
        "lighting_presets": [{"preview_exposure": -3.0}],
    }), encoding="utf-8")
    presenter = OpenXrVulkanPresenter(OpenXrVulkanConfig(
        filament_profile_path=str(profile_path),
        filament_glb_path=str(tmp_path / "environment.glb"),
        filament_scene_exposure_ev=6.0,
    ))

    presenter._load_filament_profile()

    assert presenter._filament_scene_exposure == pytest.approx(0.0)


def test_filament_profile_view_pose_is_converted_to_glb_local_space(tmp_path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps({
        "model_position": [0.0, 843.0, 0.0],
        "view_poses": [{"x": -24.0, "y": 900.0, "z": -961.0}],
    }), encoding="utf-8")
    presenter = OpenXrVulkanPresenter(OpenXrVulkanConfig(
        filament_profile_path=str(profile_path),
    ))
    presenter._load_filament_profile()
    assert presenter._profile_head_transform[:3, 3].tolist() == [-24.0, 57.0, -961.0]


def test_default_filament_profile_creates_identity_view_and_screen(tmp_path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps({"glb": None, "screen_light_intensity": 3.5}),
        encoding="utf-8",
    )
    presenter = OpenXrVulkanPresenter(
        OpenXrVulkanConfig(filament_profile_path=str(profile_path))
    )

    presenter._load_filament_profile()

    assert presenter._profile_view_name == "Default"
    assert presenter._profile_head_transform == pytest.approx(np.eye(4))
    assert presenter._filament_screen == (
        (0.0, 0.0, -20.0),
        23.09,
        12.988125,
        (0.0, 0.0, 0.0),
    )


def test_default_screen_is_initialized_from_head_pose() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = (
        (0.0, 0.0, -20.0),
        23.09,
        12.988125,
        (0.0, 0.0, 0.0),
    )
    presenter._head_position_w = np.asarray((1.0, 1.6, 2.0), dtype=np.float64)
    presenter._head_forward_w = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    presenter._initial_head_y = 1.6

    presenter._initialize_filament_screen_from_head()

    position, width, height, rotation = presenter._filament_screen
    assert position == pytest.approx((21.0, 1.6, 2.0))
    assert width == pytest.approx(23.09)
    assert height == pytest.approx(12.988125)
    assert rotation == pytest.approx((-90.0, 0.0, 0.0))


def test_persisted_screen_state_is_applied_and_reset_restores_head_pose(tmp_path) -> None:
    profile_path = tmp_path / "Default" / "profile.json"
    profile_path.parent.mkdir()
    profile_path.write_text(json.dumps({"glb": None}), encoding="utf-8")
    persisted = {
        "Default": {
            "position": [1.0, 1.6, -2.5],
            "width": 3.2,
            "height": 1.8,
            "rotation_deg": [-90.0, 5.0, 0.0],
            "curved": True,
            "curve_half_angle_rad": 0.72,
        }
    }
    calls = []
    presenter = OpenXrVulkanPresenter(
        OpenXrVulkanConfig(
            filament_profile_path=str(profile_path),
            filament_screen_states=persisted,
        ),
        on_controller_shortcut=lambda action, **values: (
            calls.append((action, values)) or True
        ),
    )

    presenter._load_filament_profile()
    presenter._head_position_w = np.asarray((1.0, 1.6, 2.0), dtype=np.float64)
    presenter._head_forward_w = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    presenter._initialize_filament_screen_from_head()

    assert presenter._filament_screen == (
        (1.0, 1.6, -2.5),
        3.2,
        1.8,
        (-90.0, 5.0, 0.0),
    )
    assert presenter._screen_curved is True

    presenter._dispatch_controller_shortcut("reset_screen")

    position, width, height, rotation = presenter._filament_screen
    assert position == pytest.approx((21.0, 1.6, 2.0))
    assert width == pytest.approx(23.09)
    assert height == pytest.approx(12.988125)
    assert rotation == pytest.approx((-90.0, 5.0, 0.0))
    assert presenter._screen_curved is True
    assert presenter._screen_curve_half_angle == pytest.approx(0.72)
    assert calls[-1][0] == "persist_openxr_screen_state"
    assert presenter.config.filament_screen_states["Default"]["position"] == pytest.approx(
        (21.0, 1.6, 2.0)
    )


def test_default_screen_uses_current_head_height_not_stale_cached_height() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = (
        (0.0, 0.0, -20.0),
        23.09,
        12.988125,
        (0.0, 0.0, 0.0),
    )
    presenter._head_position_w = np.asarray((1.0, 1.6, 2.0), dtype=np.float64)
    presenter._head_forward_w = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    presenter._initial_head_y = 9.0

    presenter._initialize_filament_screen_from_head()

    assert presenter._filament_screen[0][1] == pytest.approx(1.6)


def test_packaged_default_profile_uses_neutral_filament_exposure() -> None:
    profile_path = (
        APP_ROOT
        / "xr_viewer/environments/Default/profile.json"
    )
    profile = json.loads(profile_path.read_text(encoding="utf-8"))

    assert profile["glb"] is None
    assert profile["preview_exposure"] == 0.0
    assert profile["lighting_presets"][0]["glow_mode"] == "surround"
    assert profile["lighting_presets"][0]["glow_shell_intensity_multiplier"] == 1.85
    assert profile["lighting_presets"][0]["glow_shell_radius"] == 20.0
    assert profile["lighting_presets"][0]["glow_shell_height"] == 9.5


def test_packaged_3d_room_profiles_use_neutral_filament_exposure() -> None:
    environments_dir = (
        APP_ROOT / "xr_viewer/environments"
    )
    room_profiles = []
    for profile_path in environments_dir.glob("*/profile.json"):
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        if profile.get("glb"):
            room_profiles.append((profile_path.parent.name, profile))

    assert room_profiles
    for room_name, profile in room_profiles:
        assert profile.get("preview_exposure") == 0.0, room_name
        assert profile.get("preview_skybox_brightness") == 1.0, room_name


def test_legacy_environment_exposure_does_not_override_filament_ev() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_scene_exposure = 1.25

    presenter._apply_filament_lighting_preset(
        {"env_exposure": 0.5}, apply_bridge=False
    )
    assert presenter._filament_scene_exposure == pytest.approx(1.25)

    presenter._apply_filament_lighting_preset(
        {"preview_exposure": -0.75}, apply_bridge=False
    )
    assert presenter._filament_scene_exposure == pytest.approx(-0.75)


def test_3d_cinema_profile_screen_faces_the_default_audience() -> None:
    profile_path = (
        APP_ROOT
        / "xr_viewer/environments/3d_cinema/profile.json"
    )
    profile = json.loads(profile_path.read_text(encoding="utf-8"))

    assert profile["display_name"]["CN"] == "3D_巨幕影院"
    assert profile["screen"]["rotation_deg"] == [-180.0, 0.0, 0.0]


def test_controller_profile_rotation_uses_local_x_axis() -> None:
    source = (APP_ROOT /
              "xr_viewer/core_openxr_vulkan.py").read_text(encoding="utf-8")
    assert (
        "0.0, math.radians(self._controller_calibration_rotation_deg), 0.0"
        in source
    )


def test_quad_profile_rotation_uses_legacy_yaw_pitch_roll_order() -> None:
    from xr_viewer.core_openxr_vulkan import _euler_degrees_to_quaternion

    x, y, z, w = _euler_degrees_to_quaternion((90.0, 0.0, 0.0))
    assert abs(x) < 1e-6
    assert abs(y - 2 ** -0.5) < 1e-6
    assert abs(z) < 1e-6
    assert abs(w - 2 ** -0.5) < 1e-6


def test_projection_layer_routes_runtime_eyes_to_vulkan_composer() -> None:
    source = (APP_ROOT /
              "xr_viewer/core_openxr_vulkan.py").read_text(encoding="utf-8")
    assert "_render_vulkan_projection_composer(" in source
    assert "bridge.set_screen_image(" not in source
    assert "The main SBS screen is Projection Composer-only" in source


def test_filament_screen_footprint_matches_projected_swapchain_pixels() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = ((0.0, 0.0, -2.0), 2.0, 1.0, (0.0, 0.0, 0.0))
    view = SimpleNamespace(
        pose=SimpleNamespace(
            position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
        fov=SimpleNamespace(
            angle_left=-math.pi / 4.0,
            angle_right=math.pi / 4.0,
            angle_down=-math.pi / 4.0,
            angle_up=math.pi / 4.0,
        ),
    )

    footprint = presenter._screen_footprint_pixels(view, (1000, 1000))

    assert footprint == pytest.approx((500.0, 250.0))


def test_projection_points_use_vulkan_top_left_y_coordinates() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = ((0.0, 0.0, -2.0), 2.0, 1.0, (0.0, 0.0, 0.0))
    view = SimpleNamespace(
        pose=SimpleNamespace(
            position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
        fov=SimpleNamespace(
            angle_left=-math.pi / 4.0,
            angle_right=math.pi / 4.0,
            angle_down=-math.pi / 4.0,
            angle_up=math.pi / 4.0,
        ),
    )

    points = presenter._screen_projection_points(view, (1000, 1000))

    assert points is not None
    assert points[0, 1] > points[2, 1]


def test_projection_points_reject_screen_crossing_eye_plane() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = ((0.0, 0.0, 0.0), 2.0, 1.0, (0.0, 0.0, 0.0))
    view = SimpleNamespace(
        pose=SimpleNamespace(
            position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
        fov=SimpleNamespace(
            angle_left=-math.pi / 4.0,
            angle_right=math.pi / 4.0,
            angle_down=-math.pi / 4.0,
            angle_up=math.pi / 4.0,
        ),
    )

    assert presenter._screen_projection_points(view, (1000, 1000)) is None


def test_projection_screen_keeps_eye_plane_crossing_for_gpu_clipping() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = ((0.0, 0.0, 0.0), 2.0, 1.0, (0.0, 0.0, 0.0))
    view = SimpleNamespace(
        pose=SimpleNamespace(
            position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
        fov=SimpleNamespace(
            angle_left=-math.pi / 4.0,
            angle_right=math.pi / 4.0,
            angle_down=-math.pi / 4.0,
            angle_up=math.pi / 4.0,
        ),
    )

    payload = presenter._projection_screen_push_constants(view)

    assert len(payload) == 128
    assert np.frombuffer(payload, dtype="<f4")[-4:] == pytest.approx(
        (1.0, 0.5, 0.0, 0.0)
    )


def test_projection_glow_state_preserves_legacy_screen_glow_parameters() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = ((0.0, 0.0, -2.0), 2.0, 1.0, (0.0, 0.0, 0.0))
    presenter._filament_glow_mode = "glow"
    presenter._filament_glow_environment_enabled = True
    presenter._filament_glow_width = 0.25
    presenter._filament_glow_intensity = 0.5
    presenter._filament_glow_intensity_multiplier = 1.5
    presenter._head_position_w = np.asarray((0.0, 0.0, 0.0), dtype=np.float64)
    state = presenter._projection_glow_state()

    assert state is not None
    mode, payload = state
    values = np.frombuffer(payload, dtype="<f4")
    assert mode == 1
    assert len(payload) == 96
    assert values[:4] == pytest.approx((0.0, 0.0, 0.0, 1.0))
    assert values[4:8] == pytest.approx((0.5, 0.25, 1.5, 0.0))
    assert values[8:10] == pytest.approx((32.0, 31.0))
    assert values[10:12] == pytest.approx((1.0, 0.5))
    assert values[22:24] == pytest.approx((2.4, 2.0))


def test_projection_glow_is_absent_when_mode_is_off() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_glow_mode = "off"

    assert presenter._projection_glow_state() is None


def test_rocm_glow_prewarm_does_not_auto_enable_visual_glow() -> None:
    source = (APP_ROOT / "xr_viewer" / "core_openxr_vulkan.py").read_text(
        encoding="utf-8"
    )
    marker = "# AMD ROCm: pre-create"
    start = source.index(marker)
    end = source.index("    def _create_session_and_swapchains", start)
    prewarm = source[start:end]

    assert "VulkanGlowSourceComputeBackend" in prewarm
    assert "prewarm_backend.submit" in prewarm
    assert "Filament glow state preserved" in prewarm
    assert '_set_filament_glow_mode("glow")' not in prewarm


@pytest.mark.parametrize(
    ("mode", "mode_value"),
    (("glow", 1), ("veil", 2), ("surround", 3)),
)
def test_projection_glow_maps_supported_modes(mode: str, mode_value: int) -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_glow_environment_enabled = True
    presenter._filament_screen = ((0.0, 0.0, -2.0), 2.0, 1.0, (0.0, 0.0, 0.0))
    presenter._filament_glow_mode = mode
    presenter._filament_glow_intensity_multiplier = 1.0
    presenter._filament_glow_shell_intensity_multiplier = 1.0

    state = presenter._projection_glow_state()

    assert state is not None
    assert state[0] == mode_value
    assert np.frombuffer(state[1], dtype="<f4")[3] == pytest.approx(mode_value)


def test_projection_glow_uses_reduced_surround_density_and_stable_order() -> None:
    assert VulkanProjectionScreenPass._GLOW_SEGMENTS == 64
    assert VulkanProjectionScreenPass._GLOW_SHELL_SEGMENTS == 48
    assert VulkanProjectionScreenPass._GLOW_SHELL_RADIAL_SEGMENTS == 24
    assert VulkanProjectionScreenPass._SURROUND_VERTEX_COUNT == 4 * 24 * 48 * 6

    pipeline_source = inspect.getsource(VulkanProjectionScreenPass._create)
    surround_setup = pipeline_source.split("self.surround_pipeline", 1)[1]
    assert "maximum=True" in surround_setup
    assert "vk.VK_BLEND_OP_MAX if maximum else vk.VK_BLEND_OP_ADD" in pipeline_source

    shader = (
        APP_ROOT
        / "shaders"
        / "d2s_projection_glow_vert.vert"
    ).read_text(encoding="utf-8")
    assert "const int GLOW_SHELL_SEGMENTS = 48;" in shader
    assert "const int SHELL_RADIAL_SEGMENTS = 24;" in shader

    source = inspect.getsource(
        OpenXrVulkanPresenter._render_vulkan_projection_composer
    )
    assert "glow_state[0] == 3" in source
    assert "clear_target=not bool(filament_hdr_timeline)" in source
    assert "or filament_hdr_timeline" in source
    assert "or filament_wait_semaphores" in source


def test_projection_glow_does_not_repeat_the_producer_y_flip() -> None:
    shader = (
        APP_ROOT
        / "shaders"
        / "d2s_projection_glow_frag.frag"
    ).read_text(encoding="utf-8")

    assert "vec2 content_uv = raw;" in shader
    assert "vec2 sample_uv = uv;" in shader
    assert "q.y = 1.0 - q.y;" not in shader


def test_projection_glow_shader_applies_shared_opacity_to_all_modes() -> None:
    shader = (
        APP_ROOT / "shaders" / "d2s_projection_glow_frag.frag"
    ).read_text(encoding="utf-8")

    assert "state.glow.x * state.glow.z * state.veil.y" in shader
    assert "state.glow.x * state.glow.w * state.veil.y" in shader
    assert "state.veil.y * state.veil.x" in shader


def test_projection_laser_packs_legacy_beam_transform_and_animation() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._frame_now = 12.0
    presenter._laser_last_move_l = 11.0
    presenter._grip_mat_l = np.eye(4, dtype=np.float32)
    presenter._aim_mat_l = np.eye(4, dtype=np.float32)
    presenter._controller_interaction_ray = lambda _hand: (
        np.asarray((0.0, 0.0, 0.0), dtype=np.float64),
        np.asarray((0.0, 0.0, -1.0), dtype=np.float64),
    )

    payload = presenter._projection_laser_params(0)

    assert payload is not None
    values = np.frombuffer(payload, dtype="<f4")
    assert len(payload) == 80
    assert values[0:4] == pytest.approx((0.006, 0.0, 0.0, 0.0))
    assert values[4:8] == pytest.approx((0.0, 0.0, -0.4, 0.0))
    assert values[8:12] == pytest.approx((0.0, 0.006, 0.0, 0.0))
    assert values[12:16] == pytest.approx((0.0, 0.0, -0.11, 1.0))
    assert values[16] == pytest.approx(12.0)


def test_screen_resolution_log_reports_source_and_projected_pixels(capsys) -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = ((0.0, 0.0, -2.0), 2.0, 1.0, (0.0, 0.0, 0.0))
    presenter.swapchains = [
        SimpleNamespace(width=1000, height=1000),
        SimpleNamespace(width=1000, height=1000),
    ]
    view = SimpleNamespace(
        pose=SimpleNamespace(
            position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
        fov=SimpleNamespace(
            angle_left=-math.pi / 4.0,
            angle_right=math.pi / 4.0,
            angle_down=-math.pi / 4.0,
            angle_up=math.pi / 4.0,
        ),
    )
    frame = SimpleNamespace(
        left_eye=SimpleNamespace(width=3840, height=2160),
        right_eye=SimpleNamespace(width=3840, height=2160),
        metadata={"render_size": (3840, 2160)},
    )

    presenter._report_screen_resolution([view, view], frame)

    output = capsys.readouterr().out
    assert "source_left=3840x2160" in output
    assert "render_size=3840x2160" in output
    assert "screen_footprint_left=500x250" in output
    assert "projection_target_left=1000x1000" in output
    assert "source_per_screen_pixel_left=7.68x8.64" in output


def test_screen_resolution_log_ignores_pose_jitter(capsys) -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = ((0.0, 0.0, -2.0), 2.0, 1.0, (0.0, 0.0, 0.0))
    presenter.swapchains = [
        SimpleNamespace(width=1000, height=1000),
        SimpleNamespace(width=1000, height=1000),
    ]
    view = SimpleNamespace(
        pose=SimpleNamespace(
            position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
        fov=SimpleNamespace(
            angle_left=-math.pi / 4.0,
            angle_right=math.pi / 4.0,
            angle_down=-math.pi / 4.0,
            angle_up=math.pi / 4.0,
        ),
    )
    frame = SimpleNamespace(
        left_eye=SimpleNamespace(width=3840, height=2160),
        right_eye=SimpleNamespace(width=3840, height=2160),
        metadata={"render_size": (3840, 2160)},
    )

    presenter._report_screen_resolution([view, view], frame)
    capsys.readouterr()

    view.pose.position.x = 1.0
    presenter._report_screen_resolution([view, view], frame)

    assert capsys.readouterr().out == ""


def test_screen_resolution_log_coalesces_rapid_configuration_changes(
    monkeypatch, capsys
) -> None:
    now = [100.0]
    monkeypatch.setattr(core_openxr_vulkan.time, "perf_counter", lambda: now[0])
    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = ((0.0, 0.0, -2.0), 2.0, 1.0, (0.0, 0.0, 0.0))
    presenter.swapchains = [
        SimpleNamespace(width=1000, height=1000),
        SimpleNamespace(width=1000, height=1000),
    ]
    view = SimpleNamespace(
        pose=SimpleNamespace(
            position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
        fov=SimpleNamespace(
            angle_left=-math.pi / 4.0,
            angle_right=math.pi / 4.0,
            angle_down=-math.pi / 4.0,
            angle_up=math.pi / 4.0,
        ),
    )
    frame = SimpleNamespace(
        left_eye=SimpleNamespace(width=3840, height=2160),
        right_eye=SimpleNamespace(width=3840, height=2160),
        metadata={"render_size": (3840, 2160)},
    )

    presenter._report_screen_resolution([view, view], frame)
    assert "render_size=3840x2160" in capsys.readouterr().out

    frame.metadata["render_size"] = (1920, 1080)
    now[0] = 101.0
    presenter._report_screen_resolution([view, view], frame)
    assert capsys.readouterr().out == ""

    now[0] = 102.1
    presenter._report_screen_resolution([view, view], frame)
    assert "render_size=1920x1080" in capsys.readouterr().out


def test_presenter_run_until_owns_shutdown_close() -> None:
    presenter = OpenXrVulkanPresenter()
    shutdown = threading.Event()
    calls = []

    presenter.initialize = lambda: calls.append("initialize")
    presenter.run_frame = lambda: (calls.append("frame"), shutdown.set(), True)[2]
    presenter.close = lambda: calls.append("close")

    assert presenter.run_until(shutdown) == 0
    assert calls == ["initialize", "frame", "close"]


def test_presenter_device_loss_stops_runtime_event_for_fresh_process_restart() -> None:
    presenter = OpenXrVulkanPresenter()
    shutdown = threading.Event()
    presenter.vulkan = SimpleNamespace(device_lost=True)
    presenter.initialize = lambda: setattr(presenter, "_initialized", True)

    def fail_frame():
        presenter._request_fatal_device_loss()
        return False

    presenter.run_frame = fail_frame
    presenter.close = lambda: None

    presenter.run_until(shutdown)

    assert shutdown.is_set()
    assert presenter.fatal_device_loss is True


def test_runtime_failure_waits_for_openxr_runtime_release_before_reconnect() -> None:
    RuntimeFailureError = type("RuntimeFailureError", (Exception,), {})
    presenter = OpenXrVulkanPresenter()
    shutdown = threading.Event()
    waits = []
    presenter.initialize = lambda: setattr(presenter, "_initialized", True)
    presenter.run_frame = lambda: (_ for _ in ()).throw(
        RuntimeFailureError("runtime rejected frame")
    )
    presenter.close = lambda: None

    def wait(delay):
        waits.append(delay)
        shutdown.set()
        return True

    shutdown.wait = wait

    presenter.run_until(shutdown)

    assert waits == [6.0]


def test_presenter_close_destroys_bound_vulkan_before_openxr() -> None:
    calls = []

    class FakeVulkan:
        device_lost = False

        def wait_idle(self):
            pass

        def close(self):
            calls.append("vulkan")

    presenter = OpenXrVulkanPresenter()
    presenter.xr = SimpleNamespace(
        destroy_instance=lambda _instance: calls.append("openxr")
    )
    presenter.instance = object()
    presenter.vulkan = FakeVulkan()

    presenter.close()

    assert calls == ["vulkan", "openxr"]


def test_rocm_prewarm_closes_before_presenter_vulkan_device() -> None:
    events = []
    presenter = OpenXrVulkanPresenter()
    presenter._prewarmed_glow_backend = SimpleNamespace(close=lambda: events.append("glow"))
    presenter.vulkan = SimpleNamespace(
        device_lost=False, wait_idle=lambda: None, close=lambda: events.append("device"),
    )
    presenter.close()
    assert events == ["glow", "device"]
    assert presenter._prewarmed_glow_backend is None


def test_presenter_close_skips_openxr_instance_destroy_after_device_loss() -> None:
    calls = []

    class FakeVulkan:
        device_lost = True

        def wait_idle(self):
            raise AssertionError("device-lost Vulkan must not wait for idle")

        def close(self):
            calls.append("vulkan")

    presenter = OpenXrVulkanPresenter()
    presenter.xr = SimpleNamespace(
        destroy_instance=lambda _instance: calls.append("openxr")
    )
    presenter.instance = object()
    presenter.vulkan = FakeVulkan()

    presenter.close()

    # VDXR can access freed loader state when xrDestroyInstance follows a
    # device loss. The fatal path exits the process, so avoiding that call is
    # safer than attempting an in-process reconnect with a dead device.
    assert calls == ["vulkan"]
    assert presenter.instance is None


def test_presenter_skips_end_frame_after_vulkan_device_loss() -> None:
    calls = []
    device = SimpleNamespace(device_lost=False)

    def locate_views(*_args, **_kwargs):
        device.device_lost = True
        raise RuntimeError("original Vulkan failure")

    presenter = OpenXrVulkanPresenter()
    presenter._initialized = True
    presenter.session_running = True
    presenter.session = object()
    presenter.reference_space = object()
    presenter.vulkan = device
    presenter.xr = SimpleNamespace(
        wait_frame=lambda _session: SimpleNamespace(
            should_render=True,
            predicted_display_time=1,
        ),
        begin_frame=lambda _session: calls.append("begin"),
        end_frame=lambda *_args: calls.append("end"),
        locate_views=locate_views,
        ViewLocateInfo=lambda **kwargs: kwargs,
    )
    presenter._drain_presenter_commands = lambda: None
    presenter.poll_events = lambda: None
    presenter._sync_controller_inputs = lambda *_args: None
    presenter._update_aim_poses = lambda *_args: None
    presenter._update_grip_poses = lambda *_args: None
    presenter._smooth_controller_poses = lambda: None
    presenter._controller_input = lambda *_args: {}
    presenter._handle_keyboard_input = lambda: None
    presenter._handle_vulkan_pointer_input = lambda: None
    presenter._handle_controller_shortcuts = lambda: None
    presenter._handle_controller_guide_input = lambda *_args: None

    with pytest.raises(RuntimeError, match="original Vulkan failure"):
        presenter.run_frame()

    assert calls == ["begin"]


def test_presenter_recovers_from_invalid_openxr_composition_rect(capsys) -> None:
    calls = []

    class SwapchainRectInvalidError(RuntimeError):
        pass

    class FakeXr:
        FrameEndInfo = staticmethod(lambda **kwargs: kwargs)

        def end_frame(self, _session, info):
            calls.append(info["layer_count"])
            if len(calls) == 1:
                raise SwapchainRectInvalidError("stale rect")

    presenter = OpenXrVulkanPresenter()
    presenter.xr = FakeXr()
    presenter.session = object()
    presenter._end_openxr_frame(42, [object()])

    assert calls == [1, 0]
    assert "retrying the frame without optional layers" in capsys.readouterr().out


def test_presenter_invalid_optional_layer_recovery_keeps_primary_layers() -> None:
    calls = []

    class SwapchainRectInvalidError(RuntimeError):
        pass

    class FakeXr:
        FrameEndInfo = staticmethod(lambda **kwargs: kwargs)

        def end_frame(self, _session, info):
            calls.append(info["layers"])
            if len(calls) == 1:
                raise SwapchainRectInvalidError("stale optional quad")

    projection = object()
    optional_quad = object()
    presenter = OpenXrVulkanPresenter()
    presenter.xr = FakeXr()
    presenter.session = object()
    presenter._end_openxr_frame(
        42,
        [projection, optional_quad],
        fallback_layer_pointers=[projection],
    )

    assert calls == [[projection, optional_quad], [projection]]




def test_presenter_waits_for_headset_and_retries_initialization(capsys) -> None:
    presenter = OpenXrVulkanPresenter(OpenXrVulkanConfig(
        openxr_no_headset_retry_interval=0.001,
        openxr_standby_retry_interval=0.001,
        openxr_standby_retry_max_interval=0.001,
    ))
    shutdown = threading.Event()
    calls = []

    def initialize():
        calls.append("initialize")
        if calls.count("initialize") == 1:
            raise type("FormFactorUnavailableError", (RuntimeError,), {})()

    def run_frame():
        calls.append("frame")
        shutdown.set()
        return True

    presenter.initialize = initialize
    presenter.run_frame = run_frame
    presenter.close = lambda: calls.append("close")

    assert presenter.run_until(shutdown) == 0
    assert calls == ["initialize", "close", "initialize", "frame", "close"]
    assert "Vulkan/Filament initialization deferred" in capsys.readouterr().out


def test_presenter_wait_enters_hard_idle_after_configured_timeout(capsys) -> None:
    states = []
    presenter = OpenXrVulkanPresenter(
        OpenXrVulkanConfig(headset_wait_inference_timeout=0.0),
        on_headset_state=states.append,
    )

    presenter._notify_headset_waiting()
    assert states == ["waiting", "hard_idle"]
    assert "Headset not detected or in standby" in capsys.readouterr().out

    presenter._notify_headset_active()
    assert states[-1] == "active"


def test_presenter_output_ready_tracks_openxr_acceptance_gate() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._initialized = True
    presenter.session_running = True

    assert not presenter.output_ready

    presenter._notify_headset_active()
    assert presenter.output_ready

    presenter._notify_headset_waiting()
    assert not presenter.output_ready


def test_presenter_rejects_output_while_headset_is_waiting() -> None:
    from types import SimpleNamespace

    presenter = OpenXrVulkanPresenter()

    with pytest.raises(RuntimeError, match="waiting for headset rendering"):
        presenter.submit_output(object())

    presenter._notify_headset_active()
    presenter.session_running = True
    with pytest.raises(TypeError, match="VulkanImageResource"):
        presenter.submit_output(SimpleNamespace(left_eye=None, right_eye=None))


def test_presenter_queues_output_from_non_owner_thread() -> None:
    presenter = OpenXrVulkanPresenter()
    context = object()
    resource = VulkanImageResource(
        context=context,
        image="image-left",
        view=None,
        width=8,
        height=8,
        format=43,
        layout=0,
        access_mask=0,
        stage_mask=0,
        queue_family_index=0,
    )
    other_resource = VulkanImageResource(
        context=context,
        image="image-right",
        view=None,
        width=8,
        height=8,
        format=43,
        layout=0,
        access_mask=0,
        stage_mask=0,
        queue_family_index=0,
    )
    presenter.vulkan = context
    presenter.session_running = True
    presenter._accept_output = True
    presenter._presenter_thread_id = threading.get_ident() + 1
    frame = SimpleNamespace(
        left_eye=resource,
        right_eye=other_resource,
        metadata={},
    )

    presenter.submit_output(frame)

    assert presenter._pending_output is None
    presenter._drain_presenter_commands()
    assert presenter._pending_output is frame


def test_presenter_submits_raw_runtime_result_on_owner_thread() -> None:
    presenter = OpenXrVulkanPresenter()
    context = object()
    left = VulkanImageResource(
        context=context, image="left", view=None, width=8, height=8, format=43,
        layout=0, access_mask=0, stage_mask=0, queue_family_index=0,
    )
    right = VulkanImageResource(
        context=context, image="right", view=None, width=8, height=8, format=43,
        layout=0, access_mask=0, stage_mask=0, queue_family_index=0,
    )
    presenter.vulkan = context
    presenter.session_running = True
    presenter._accept_output = True
    presenter._presenter_thread_id = threading.get_ident()

    presenter.submit_runtime_result(
        SimpleNamespace(left_eye=left, right_eye=right), 1.0
    )

    assert presenter._pending_output is not None
    assert presenter._pending_output.left_eye is left


def test_presenter_drains_only_latest_raw_runtime_result(monkeypatch) -> None:
    presenter = OpenXrVulkanPresenter()
    calls = []
    monkeypatch.setattr(
        presenter,
        "_submit_runtime_result_on_presenter",
        lambda result, timestamp: calls.append((result, timestamp)),
    )
    presenter._presenter_commands.put(("submit_runtime_result", ("old", 1.0)))
    presenter._presenter_commands.put(("submit_runtime_result", ("new", 2.0)))

    presenter._drain_presenter_commands()

    assert calls == [("new", 2.0)]


def test_runtime_output_conversion_error_is_reported_once_until_recovery(
    capsys,
) -> None:
    class FailingAdapter:
        def convert(self, *_args, **_kwargs):
            raise ValueError("conversion failed")

    presenter = OpenXrVulkanPresenter()
    presenter._output_adapter = FailingAdapter()
    result = SimpleNamespace(debug_info={})

    presenter._submit_runtime_result_on_presenter(result, 1.0)
    presenter._submit_runtime_result_on_presenter(result, 2.0)

    output = capsys.readouterr().out
    assert output.count("Runtime output conversion failed") == 1


def test_presenter_latches_fatal_device_loss_from_output_conversion() -> None:
    class LostContext:
        device_lost = True

    class FailingAdapter:
        def convert(self, *_args, **_kwargs):
            raise RuntimeError("VkErrorDeviceLost")

    presenter = OpenXrVulkanPresenter()
    presenter.vulkan = LostContext()
    presenter._output_adapter = FailingAdapter()

    presenter._submit_runtime_result_on_presenter(
        SimpleNamespace(debug_info={}), 1.0
    )

    assert presenter.fatal_device_loss is True
    assert presenter.exit_requested is True
    assert presenter._accept_output is False


def test_presenter_close_stops_worker_output_before_teardown() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._accept_output = True
    presenter.session_running = True
    presenter.close()

    assert presenter._accept_output is False


def test_filament_bridge_binds_each_openxr_eye(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    class FakeBridge:
        def __init__(self, path):
            calls.append(("load", path))

        def create(self, **kwargs):
            calls.append(("create", kwargs["device"]))

        def create_eye_swapchain(self, eye_index, images, **kwargs):
            calls.append(("swapchain", (eye_index, list(images), kwargs["format"])))

        def set_scene_exposure(self, _value):
            pass

        def set_skybox_brightness(self, _value):
            pass

        def set_fill_light(self, _color, _intensity, _direction):
            pass

        def close(self):
            calls.append(("close", None))

    import xr_viewer.filament_vulkan_bridge as bridge_module

    monkeypatch.setattr(bridge_module, "FilamentVulkanBridge", FakeBridge)
    presenter = OpenXrVulkanPresenter(
        OpenXrVulkanConfig(filament_bridge_path="bridge.dll")
    )
    presenter.vulkan = SimpleNamespace(
        instance=1,
        physical_device=2,
        device=3,
        queue_family_index=4,
    )
    presenter.swapchain_format = 43
    presenter.swapchains = [
        _EyeSwapchain("left", [SimpleNamespace(image="left-image")], 10, 20),
        _EyeSwapchain("right", [SimpleNamespace(image="right-image")], 30, 40),
    ]

    presenter._initialize_filament_bridges()

    assert presenter.filament_bridge is not None
    assert calls == [
        ("load", "bridge.dll"),
        ("create", 3),
        ("swapchain", (0, ["left-image"], 43)),
        ("swapchain", (1, ["right-image"], 43)),
    ]


def test_filament_multiview_uses_private_hdr_targets_and_keeps_eye_swapchains(
    monkeypatch,
) -> None:
    presenter = OpenXrVulkanPresenter()
    left = _EyeSwapchain("left", [], 10, 20)
    right = _EyeSwapchain("right", [], 10, 20)
    presenter.swapchains = [left, right]
    presenter.swapchain_format = 43
    presenter._vulkan_projection_composer_requested = True
    presenter._filament_multiview_requested = True
    closed = []

    class FakeDepthImage:
        def __init__(self, _context, width, height, *, label, array_layers):
            self.image = "depth"
            self.format = 126
            assert (width, height, array_layers) == (10, 20, 2)
            assert label == "filament-multiview-depth"

        def close(self):
            closed.append(self.image)

    class FakeHdrImage:
        count = 0

        def __init__(self, _context, width, height, *, format, array_layers, label):
            slot = type(self).count
            type(self).count += 1
            self.image = f"hdr-{slot}"
            self.layer_resources = [f"layer-{slot}-0", f"layer-{slot}-1"]
            assert (width, height, format, array_layers) == (10, 20, 97, 2)
            assert label.startswith("filament-multiview-hdr-slot")

        def close(self):
            closed.append(self.image)

    monkeypatch.setattr(
        "xr_viewer.core_openxr_vulkan.VulkanTransientImage", FakeHdrImage
    )
    monkeypatch.setattr(
        "xr_viewer.core_openxr_vulkan.VulkanDepthAttachment", FakeDepthImage
    )
    presenter.vulkan = SimpleNamespace(
        device="device",
        vk=SimpleNamespace(
            VK_FORMAT_R16G16B16A16_SFLOAT=97,
            VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO=1,
            VkSemaphoreCreateInfo=lambda **kwargs: kwargs,
            vkCreateSemaphore=lambda *_args: object(),
            vkDestroySemaphore=lambda *_args: None,
        ),
    )
    class FakeBridge:
        multiview_abi_available = True
        multiview_supported = True
        multiview_depth_swapchain_abi_available = True
        image_ready_semaphore_abi_available = True
        finished_drawing_semaphore_abi_available = True
        def create_stereo_swapchain_with_depth(self, images, **kwargs):
            assert list(images) == ["hdr-0", "hdr-1", "hdr-2"]
            assert kwargs == {
                "format": 97,
                "width": 10,
                "height": 20,
                "depth_image": "depth",
                "depth_format": 126,
            }

    assert presenter._try_enable_filament_multiview(FakeBridge())
    assert presenter.swapchains == [left, right]
    assert len(presenter._filament_multiview_hdr_images) == 3
    assert len(presenter._filament_depth_attachments) == 1
    assert presenter._controller_composition_swapchain is None
    assert not closed


def test_filament_multiview_failure_preserves_two_swapchain_fallback(monkeypatch) -> None:
    presenter = OpenXrVulkanPresenter()
    left = _EyeSwapchain("left", [], 10, 20)
    right = _EyeSwapchain("right", [], 10, 20)
    presenter.swapchains = [left, right]
    presenter.swapchain_format = 43
    presenter._vulkan_projection_composer_requested = True
    presenter._filament_multiview_requested = True
    closed = []

    class FakeDepthImage:
        def __init__(self, *_args, **_kwargs):
            self.image = "depth"
            self.format = 126

        def close(self):
            closed.append(self.image)

    class FakeHdrImage:
        count = 0

        def __init__(self, *_args, **_kwargs):
            slot = type(self).count
            type(self).count += 1
            self.image = f"hdr-{slot}"

        def close(self):
            closed.append(self.image)

    monkeypatch.setattr(
        "xr_viewer.core_openxr_vulkan.VulkanTransientImage", FakeHdrImage
    )
    monkeypatch.setattr(
        "xr_viewer.core_openxr_vulkan.VulkanDepthAttachment", FakeDepthImage
    )
    presenter.vulkan = SimpleNamespace(
        device="device",
        vk=SimpleNamespace(
            VK_FORMAT_R16G16B16A16_SFLOAT=97,
            VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO=1,
            VkSemaphoreCreateInfo=lambda **kwargs: kwargs,
            vkCreateSemaphore=lambda *_args: object(),
            vkDestroySemaphore=lambda *_args: None,
        ),
    )

    class FakeBridge:
        multiview_abi_available = True
        multiview_supported = True
        multiview_depth_swapchain_abi_available = True
        image_ready_semaphore_abi_available = True
        finished_drawing_semaphore_abi_available = True
        controller_composition_layer_abi_available = True

        @staticmethod
        def create_stereo_swapchain_with_depth(_images, **_kwargs):
            raise RuntimeError("layered target rejected")

    assert not presenter._try_enable_filament_multiview(FakeBridge())
    assert presenter.swapchains == [left, right]
    assert closed == ["depth", "hdr-0", "hdr-1", "hdr-2"]


def test_filament_multiview_keeps_fallback_for_mismatched_eye_extents() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter.swapchains = [
        _EyeSwapchain("left", [], 10, 20),
        _EyeSwapchain("right", [], 11, 20),
    ]
    bridge = SimpleNamespace(
        multiview_abi_available=True,
        multiview_supported=True,
    )

    assert not presenter._try_enable_filament_multiview(bridge)
    assert [eye.handle for eye in presenter.swapchains] == ["left", "right"]


def test_filament_multiview_projection_diagnostic_activates_layered_path(
    monkeypatch,
) -> None:
    monkeypatch.setenv("D2S_VULKAN_PROJECTION_COMPOSER", "1")
    monkeypatch.setenv("D2S_FILAMENT_MULTIVIEW", "1")
    monkeypatch.setenv("D2S_FILAMENT_MULTIVIEW_PROJECTION_DIAGNOSTIC", "1")
    calls = []

    class FakeBridge:
        def __init__(self, _path):
            pass

        def create(self, **_kwargs):
            pass

        def set_scene_exposure(self, _value):
            pass

        def set_skybox_brightness(self, _value):
            pass

        def set_fill_light(self, _color, _intensity, _direction):
            pass

        def close(self):
            pass

    import xr_viewer.filament_vulkan_bridge as bridge_module

    monkeypatch.setattr(bridge_module, "FilamentVulkanBridge", FakeBridge)
    presenter = OpenXrVulkanPresenter(
        OpenXrVulkanConfig(filament_bridge_path="bridge.dll")
    )
    presenter.vulkan = SimpleNamespace(
        instance=1,
        physical_device=2,
        device=3,
        queue_family_index=4,
    )
    presenter.swapchains = [
        _EyeSwapchain("left", [], 10, 20),
        _EyeSwapchain("right", [], 10, 20),
    ]
    presenter._try_enable_filament_multiview = lambda bridge: (
        calls.append(bridge),
        True,
    )[-1]

    presenter._initialize_filament_bridges()

    assert calls == [presenter.filament_bridge]
    assert presenter._multiview_active


def test_active_rgb_mean_ignores_black_background() -> None:
    from xr_viewer.core_openxr_vulkan import _active_rgb_mean

    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[0, 0] = (255, 0, 0)
    rgb[1, 1] = (0, 255, 0)

    count, mean = _active_rgb_mean(rgb)

    assert count == 2
    assert mean == pytest.approx((127.5, 127.5, 0.0))


def test_filament_multiview_readback_requests_transfer_source(monkeypatch) -> None:
    monkeypatch.setenv("D2S_FILAMENT_MULTIVIEW_LAYER_READBACK", "1")
    presenter = OpenXrVulkanPresenter()

    assert presenter._filament_multiview_layer_readback_requested
    source = inspect.getsource(OpenXrVulkanPresenter._create_projection_swapchain)
    assert "SwapchainUsageFlags.TRANSFER_SRC_BIT" in source


def test_filament_multiview_readback_waits_for_stable_rendering(monkeypatch) -> None:
    monkeypatch.setenv("D2S_FILAMENT_MULTIVIEW_LAYER_READBACK", "1")
    presenter = OpenXrVulkanPresenter()
    presenter._multiview_active = True

    for _ in range(29):
        assert not presenter._advance_filament_multiview_layer_readback()

    assert presenter._filament_multiview_layer_readback_frame == 29
    assert presenter._advance_filament_multiview_layer_readback()

    presenter._filament_multiview_layer_readback_done = True
    assert not presenter._advance_filament_multiview_layer_readback()
    assert presenter._filament_multiview_layer_readback_frame == 30


def test_pure_vulkan_multiview_diagnostic_bypasses_filament(monkeypatch) -> None:
    monkeypatch.setenv("D2S_OPENXR_VULKAN_MULTIVIEW_EYE_DIAGNOSTIC", "1")
    presenter = OpenXrVulkanPresenter(
        OpenXrVulkanConfig(filament_bridge_path="bridge.dll")
    )

    presenter._initialize_filament_bridges()

    assert presenter.filament_bridge is None
    root = Path(__file__).resolve().parents[1]
    vertex_shader = (APP_ROOT / "shaders/d2s_multiview_eye_diag.vert").read_text(
        encoding="utf-8"
    )
    fragment_shader = (APP_ROOT / "shaders/d2s_multiview_eye_diag.frag").read_text(
        encoding="utf-8"
    )
    assert "viewIndexFromVertex = gl_ViewIndex" in vertex_shader
    assert "flat out uint viewIndexFromVertex" in vertex_shader
    assert "flat in uint viewIndexFromVertex" in fragment_shader
    assert "gl_ViewIndex" not in fragment_shader
    source = inspect.getsource(
        OpenXrVulkanPresenter._render_vulkan_multiview_eye_diagnostic
    )
    assert "VulkanMultiviewEyeDiagnosticPass" in source
    assert "wait_for_timeline" in source


def test_filament_camera_receives_openxr_pose_and_fov() -> None:
    calls: list[tuple[str, tuple[float, ...]]] = []

    class FakeBridge:
        def set_camera_look_at(self, eye, center, up):
            calls.append(("look_at", (*eye, *center, *up)))

        def set_camera_projection(self, fov_degrees, aspect, **kwargs):
            calls.append(("projection", (fov_degrees, aspect, kwargs)))

    view = SimpleNamespace(
        pose=SimpleNamespace(
            position=SimpleNamespace(x=1.0, y=2.0, z=3.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
        fov=SimpleNamespace(
            angle_left=-0.7,
            angle_right=0.7,
            angle_up=0.6,
            angle_down=-0.6,
        ),
    )

    _update_filament_camera(FakeBridge(), view)

    assert calls[0][0] == "look_at"
    assert calls[0][1][:3] == (1.0, 2.0, 3.0)
    assert calls[0][1][3:6] == (1.0, 2.0, 2.0)
    assert calls[0][1][6:] == (0.0, 1.0, 0.0)
    assert calls[1][0] == "projection"
    assert calls[1][1][0] == pytest.approx(68.7549, rel=1e-4)
    assert calls[1][1][2]["far_plane"] == 1000.0


def test_filament_stereo_camera_receives_head_and_relative_eyes() -> None:
    calls = []
    views = [
        SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=x, y=2.0, z=3.0),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
            fov=SimpleNamespace(
                angle_left=-0.5,
                angle_right=0.5,
                angle_down=-0.4,
                angle_up=0.4,
            ),
        )
        for x in (-0.03, 0.03)
    ]
    bridge = SimpleNamespace(
        set_camera_look_at=lambda eye, center, up: calls.append(
            ("head", eye, center, up)
        ),
        set_stereo_camera=lambda matrices, frustums, **kwargs: calls.append(
            ("eyes", matrices, frustums, kwargs)
        )
    )

    _update_filament_stereo_camera(bridge, views, near_plane=0.1, far_plane=50.0)

    assert calls[0][0] == "head"
    assert calls[0][1] == pytest.approx((0.0, 2.0, 3.0))
    assert calls[0][2] == pytest.approx((0.0, 2.0, 2.0))
    _, matrices, frustums, options = calls[1]
    assert len(matrices) == 32
    assert matrices[12] == pytest.approx(-0.03)
    assert matrices[28] == pytest.approx(0.03)
    assert matrices[:16] != matrices[16:]
    assert len(frustums) == 8
    assert options == {"near_plane": 0.1, "far_plane": 50.0}


def test_swapchain_image_is_released_when_wait_fails() -> None:
    calls: list[str] = []

    class FakeXr:
        INFINITE_DURATION = 1

        @staticmethod
        def acquire_swapchain_image(_handle):
            calls.append("acquire")
            return 0

        @staticmethod
        def wait_swapchain_image(_handle, _wait_info):
            calls.append("wait")
            raise RuntimeError("wait failed")

        @staticmethod
        def release_swapchain_image(_handle):
            calls.append("release")

        @staticmethod
        def SwapchainImageWaitInfo(*, timeout):
            return timeout

    presenter = OpenXrVulkanPresenter()
    presenter.xr = FakeXr
    presenter.swapchains = [
        _EyeSwapchain(
            handle=object(),
            images=[SimpleNamespace(image=None)],
            width=1,
            height=1,
        )
    ]
    with pytest.raises(RuntimeError, match="wait failed"):
        presenter._render_projection_layer([object()])
    assert calls == ["acquire", "wait", "release"]


def test_swapchain_image_is_released_after_wait_when_render_fails() -> None:
    calls: list[str] = []

    class FakeXr:
        INFINITE_DURATION = 1

        @staticmethod
        def acquire_swapchain_image(_handle):
            calls.append("acquire")
            return 0

        @staticmethod
        def wait_swapchain_image(_handle, _wait_info):
            calls.append("wait")

        @staticmethod
        def release_swapchain_image(_handle):
            calls.append("release")

        @staticmethod
        def SwapchainImageWaitInfo(*, timeout):
            return timeout

    class FakeVulkan:
        @staticmethod
        def image_handle_from_address(_address):
            return object()

        @staticmethod
        def clear_color_image(_image, _color):
            raise RuntimeError("clear failed")

    presenter = OpenXrVulkanPresenter()
    presenter.xr = FakeXr
    presenter.vulkan = FakeVulkan()
    presenter.swapchains = [
        _EyeSwapchain(
            handle=object(),
            images=[SimpleNamespace(image=ctypes.c_void_p(1))],
            width=1,
            height=1,
        )
    ]
    with pytest.raises(RuntimeError, match="clear failed"):
        presenter._render_projection_layer([object()])
    assert calls == ["acquire", "wait", "release"]


def test_projection_acquires_stereo_pair_before_waiting(monkeypatch) -> None:
    calls: list[str] = []
    timing_names: list[str] = []

    class FakeXr:
        INFINITE_DURATION = 1

        @staticmethod
        def acquire_swapchain_image(handle):
            calls.append(f"acquire:{handle}")
            return 0

        @staticmethod
        def wait_swapchain_image(handle, _wait_info):
            calls.append(f"wait:{handle}")

        @staticmethod
        def release_swapchain_image(handle):
            calls.append(f"release:{handle}")

        @staticmethod
        def SwapchainImageWaitInfo(*, timeout):
            return timeout

    class FakeVulkan:
        @staticmethod
        def image_handle_from_address(address):
            return address

        @staticmethod
        def clear_color_image(_image, _color):
            calls.append("clear")

    presenter = OpenXrVulkanPresenter(
        on_breakdown_add_time=lambda name, _seconds: timing_names.append(name)
    )
    presenter.xr = FakeXr
    presenter.vulkan = FakeVulkan()
    presenter.swapchains = [
        _EyeSwapchain("left", [SimpleNamespace(image=ctypes.c_void_p(1))], 1, 1),
        _EyeSwapchain("right", [SimpleNamespace(image=ctypes.c_void_p(2))], 1, 1),
    ]
    presenter._apply_filament_profile = lambda views: views
    presenter._report_screen_resolution = lambda *_args: None
    presenter._apply_screen_sampling_policy = lambda *_args: None
    monkeypatch.setattr(
        OpenXrCompositionBuilder,
        "projection_layer",
        lambda *_args: "projection",
    )

    assert presenter._render_projection_layer([object(), object()], None) == "projection"
    assert calls[:4] == [
        "acquire:left",
        "acquire:right",
        "wait:left",
        "wait:right",
    ]
    assert calls[-2:] == ["release:left", "release:right"]
    assert "openxr_projection_acquire_pair" in timing_names
    assert "openxr_projection_wait_eye0" in timing_names
    assert "openxr_projection_wait_eye1" in timing_names
    assert "openxr_projection_total" in timing_names


def test_projection_updates_shared_filament_state_once_per_stereo_pair(
    monkeypatch,
) -> None:
    calls: list[str] = []

    class FakeXr:
        INFINITE_DURATION = 1

        @staticmethod
        def acquire_swapchain_image(_handle):
            return 0

        @staticmethod
        def wait_swapchain_image(_handle, _wait_info):
            return None

        @staticmethod
        def release_swapchain_image(_handle):
            return None

        @staticmethod
        def SwapchainImageWaitInfo(*, timeout):
            return timeout

    class FakeBridge:
        def apply_animations(self, _seconds):
            calls.append("animations")

        def set_active_eye(self, eye_index):
            calls.append(f"active:{eye_index}")

        def set_acquired_image(self, image_index):
            calls.append(f"image:{image_index}")

        def begin_frame(self):
            calls.append("begin")

        def end_frame(self):
            calls.append("end")

    presenter = OpenXrVulkanPresenter()
    presenter.xr = FakeXr
    presenter.swapchains = [
        _EyeSwapchain("left", [SimpleNamespace(image=None)], 1, 1),
        _EyeSwapchain("right", [SimpleNamespace(image=None)], 1, 1),
    ]
    presenter.filament_bridge = FakeBridge()
    presenter._apply_filament_profile = lambda views: views
    presenter._report_screen_resolution = lambda *_args: None
    presenter._apply_screen_sampling_policy = lambda *_args: None
    presenter._update_filament_controllers = lambda *_args: calls.append("controllers")
    monkeypatch.setattr(
        "xr_viewer.core_openxr_vulkan._update_filament_camera",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        OpenXrCompositionBuilder,
        "projection_layer",
        lambda *_args: "projection",
    )

    assert presenter._render_projection_layer([object(), object()], None) == "projection"
    assert calls.count("controllers") == 1
    assert calls.count("animations") == 1
    assert calls.count("begin") == 2
    assert calls.count("end") == 2


def test_projection_composer_failure_falls_back_to_filament_in_same_frame(
    monkeypatch,
) -> None:
    calls: list[str] = []

    class FakeXr:
        INFINITE_DURATION = 1

        @staticmethod
        def acquire_swapchain_image(_handle):
            return 0

        @staticmethod
        def wait_swapchain_image(_handle, _wait_info):
            return None

        @staticmethod
        def release_swapchain_image(_handle):
            return None

        @staticmethod
        def SwapchainImageWaitInfo(*, timeout):
            return timeout

    class FakeBridge:
        def apply_animations(self, _seconds):
            calls.append("animations")

        def set_active_eye(self, eye_index):
            calls.append(f"active:{eye_index}")

        def set_acquired_image(self, image_index):
            calls.append(f"image:{image_index}")

        def begin_frame(self):
            calls.append("begin")

        def end_frame(self):
            calls.append("end")

    fallback_count: list[int] = []
    presenter = OpenXrVulkanPresenter(
        on_breakdown_inc=lambda name, value: (
            fallback_count.append(value)
            if name == "openxr_vulkan_projection_composer_fallback"
            else None
        )
    )
    presenter._vulkan_projection_composer_requested = True
    presenter.xr = FakeXr
    presenter.vulkan = SimpleNamespace(
        _lock=threading.RLock(),
        vk=SimpleNamespace(
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL=1,
            VK_ACCESS_SHADER_READ_BIT=2,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT=4,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT=8,
        ),
    )
    presenter.swapchains = [
        _EyeSwapchain(
            "left", [SimpleNamespace(image=None)], 1, 1, resources=[object()]
        ),
        _EyeSwapchain(
            "right", [SimpleNamespace(image=None)], 1, 1, resources=[object()]
        ),
    ]
    presenter.filament_bridge = FakeBridge()
    presenter._filament_screen = ((0.0, 0.0, -2.0), 2.0, 1.0, (0.0, 0.0, 0.0))
    presenter._apply_filament_profile = lambda views: views
    presenter._report_screen_resolution = lambda *_args: None
    presenter._apply_screen_sampling_policy = lambda *_args: None
    presenter._update_filament_controllers = lambda *_args: calls.append("controllers")
    presenter._maybe_capture_visual_regression_frame = lambda *_args, **_kwargs: None
    presenter._render_vulkan_projection_composer = lambda *_args: (
        _ for _ in ()
    ).throw(RuntimeError("submit failed"))
    monkeypatch.setattr(
        "xr_viewer.core_openxr_vulkan._update_filament_camera",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        OpenXrCompositionBuilder,
        "projection_layer",
        lambda *_args: "projection",
    )
    frame = VulkanStereoOutputFrame(
        frame_id=7,
        timestamp=0.0,
        left_eye=SimpleNamespace(image="left", width=1, height=1, format=43),
        right_eye=SimpleNamespace(image="right", width=1, height=1, format=43),
        metadata={"_vulkan_source_prepare_for_sampling": lambda *_args: None},
    )

    assert presenter._render_projection_layer([object(), object()], frame) == "projection"
    assert "screen" not in calls
    assert fallback_count == [1]
    assert not presenter._vulkan_projection_composer_active
    assert calls.count("controllers") == 1
    assert calls.count("animations") == 1
    assert calls.count("begin") == 2
    assert calls.count("end") == 2


def test_projection_keeps_unsafe_stereo_batch_disabled(
    monkeypatch,
) -> None:
    calls: list[str] = []
    timing_names: list[str] = []

    class TrackingLock:
        held = False

        def acquire(self):
            self.held = True

        def release(self):
            self.held = False

    queue_lock = TrackingLock()

    class FakeXr:
        INFINITE_DURATION = 1

        @staticmethod
        def acquire_swapchain_image(_handle):
            return 0

        @staticmethod
        def wait_swapchain_image(_handle, _wait_info):
            return None

        @staticmethod
        def release_swapchain_image(_handle):
            return None

        @staticmethod
        def SwapchainImageWaitInfo(*, timeout):
            return timeout

    class FakeBridge:
        stereo_batch_submit_abi_available = True
        finished_drawing_semaphore_abi_available = True

        def __init__(self):
            self.active_eye = 0

        def apply_animations(self, _seconds):
            pass

        def set_active_eye(self, eye_index):
            self.active_eye = eye_index
            calls.append(f"active:{eye_index}")

        def set_acquired_image(self, image_index):
            calls.append(f"image:{image_index}")

        def begin_frame(self):
            assert queue_lock.held
            calls.append("begin")

        def end_frame(self):
            calls.append("end")

        def end_frame_deferred(self):
            raise AssertionError("unsafe stereo batch must remain disabled")

        def finish_frame_batch(self):
            raise AssertionError("unsafe stereo batch must remain disabled")

        def wait_for_idle(self):
            assert queue_lock.held
            calls.append("wait-idle")

        def get_finished_drawing_semaphore(self):
            return f"finished:{self.active_eye}"

    presenter = OpenXrVulkanPresenter(
        on_breakdown_add_time=lambda name, _seconds: timing_names.append(name)
    )
    presenter.xr = FakeXr
    presenter.swapchains = [
        _EyeSwapchain("left", [SimpleNamespace(image=None)], 1, 1),
        _EyeSwapchain("right", [SimpleNamespace(image=None)], 1, 1),
    ]
    presenter.filament_bridge = FakeBridge()
    presenter.vulkan = SimpleNamespace(
        _timeline_semaphore=object(),
        _lock=queue_lock,
        submit_on=lambda role, record, **kwargs: (
            record("command-buffer"),
            calls.append(("drain", role, kwargs)),
            23,
        )[-1],
    )
    presenter._displayed_output = VulkanStereoOutputFrame(
        frame_id=7,
        timestamp=0.0,
        left_eye=object(),
        right_eye=object(),
        metadata={},
    )
    presenter._apply_filament_profile = lambda views: views
    presenter._report_screen_resolution = lambda *_args: None
    presenter._apply_screen_sampling_policy = lambda *_args: None
    presenter._update_filament_controllers = lambda *_args: None
    monkeypatch.setattr(
        "xr_viewer.core_openxr_vulkan._update_filament_camera",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        OpenXrCompositionBuilder,
        "projection_layer",
        lambda *_args: "projection",
    )

    assert presenter._render_projection_layer([object(), object()], None) == "projection"
    assert calls.count("begin") == 2
    assert calls.count("end") == 2
    assert calls.count("wait-idle") == 0
    assert len([call for call in calls if isinstance(call, tuple)]) == 1
    assert "openxr_filament_stereo_finish_wait" not in timing_names
    assert "openxr_filament_completion_drain" in timing_names
    assert presenter._displayed_output.metadata[
        "_vulkan_consumer_release_timeline"
    ] == 23
    assert not queue_lock.held


def test_multiview_projection_renders_one_layered_filament_frame(monkeypatch) -> None:
    render_source = inspect.getsource(
        OpenXrVulkanPresenter._render_filament_multiview
    )
    composer_source = inspect.getsource(
        OpenXrVulkanPresenter._render_vulkan_projection_composer
    )
    projection_source = inspect.getsource(
        OpenXrVulkanPresenter._render_projection_layer
    )
    assert "set_image_ready_semaphore" in render_source
    assert "layerCount=2" in render_source
    assert "_resolve_filament_multiview_hdr" in composer_source
    assert "_resolve_filament_multiview_hdr" in projection_source
    assert "or self._filament_multiview_finished_consumed" in projection_source
    return

    monkeypatch.setenv("D2S_VULKAN_PROJECTION_COMPOSER", "0")
    calls = []
    capture_calls = []

    class FakeXr:
        INFINITE_DURATION = 1

        @staticmethod
        def acquire_swapchain_image(handle):
            calls.append(("acquire", handle))
            return 0

        @staticmethod
        def wait_swapchain_image(handle, _wait_info):
            calls.append(("wait", handle))

        @staticmethod
        def release_swapchain_image(handle):
            calls.append(("release", handle))

        @staticmethod
        def SwapchainImageWaitInfo(*, timeout):
            return timeout

    class FakeBridge:
        finished_drawing_semaphore_abi_available = True

        def set_active_eye(self, eye_index):
            calls.append(("active", eye_index))

        def set_acquired_image(self, image_index):
            calls.append(("image", image_index))

        def begin_frame(self):
            calls.append("begin")

        def end_frame(self):
            calls.append("end")

        def end_frame_deferred(self):
            raise AssertionError("unsafe deferred path must remain disabled")

        def finish_frame_batch(self):
            raise AssertionError("unsafe deferred path must remain disabled")

        def get_finished_drawing_semaphore(self):
            return "filament-finished"

        def apply_animations(self, _seconds):
            pass

    lock = threading.RLock()

    def submit_on(role, record, **kwargs):
        record("command-buffer")
        calls.append(("submit", role, kwargs.get("wait_semaphore")))
        return 41 if kwargs.get("wait_semaphore") == ("left-ready", "right-ready") else 42

    presenter = OpenXrVulkanPresenter()
    presenter.xr = FakeXr
    presenter.vulkan = SimpleNamespace(
        _lock=lock,
        submit_on=submit_on,
        vk=SimpleNamespace(
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL=1,
            VK_ACCESS_SHADER_READ_BIT=2,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT=4,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT=8,
        ),
    )
    presenter.swapchains = [
        _EyeSwapchain(
            "layered",
            [SimpleNamespace(image=ctypes.c_void_p(1))],
            10,
            20,
            resources=["layered-resource"],
            array_size=2,
        )
    ]
    presenter._multiview_active = True
    presenter.filament_bridge = FakeBridge()
    presenter._filament_screen = ((0.0, 0.0, -2.0), 2.0, 1.0, (0.0, 0.0, 0.0))
    presenter._apply_filament_profile = lambda views: views
    presenter._report_screen_resolution = lambda *_args: None
    presenter._apply_screen_sampling_policy = lambda *_args: None
    presenter._update_filament_controllers = lambda *_args: None
    monkeypatch.setattr(
        "xr_viewer.core_openxr_vulkan._update_filament_stereo_camera",
        lambda *_args, **_kwargs: calls.append("stereo-camera"),
    )
    monkeypatch.setattr(
        OpenXrCompositionBuilder,
        "projection_layer",
        lambda *_args: "projection",
    )
    frame = VulkanStereoOutputFrame(
        frame_id=7,
        timestamp=0.0,
        left_eye=SimpleNamespace(
            image="left", width=10, height=20, format=43, resource="left-resource"
        ),
        right_eye=SimpleNamespace(
            image="right", width=10, height=20, format=43, resource="right-resource"
        ),
        metadata={},
    )

    assert presenter._render_projection_layer([object(), object()], frame) == "projection"
    assert calls.count(("acquire", "layered")) == 1
    assert calls.count(("wait", "layered")) == 1
    assert calls.count(("release", "layered")) == 1
    assert calls.count("begin") == 1
    assert calls.count("end") == 1
    assert not any(
        call == ("submit", "graphics", ("left-ready", "right-ready"))
        for call in calls
    )
    assert ("submit", "graphics", ("filament-finished",)) in calls
    assert "_vulkan_consumer_release_timeline" not in frame.metadata
    assert not capture_calls


def test_multiview_render_keeps_active_eye_zero(monkeypatch) -> None:
    active_eyes = []

    class FakeBridge:
        def set_active_eye(self, eye_index):
            active_eyes.append(eye_index)

        def set_acquired_image(self, _image_index):
            pass

        def set_image_ready_semaphore(self, _semaphore):
            pass

        def begin_frame(self):
            pass

        def end_frame(self):
            pass

    presenter = OpenXrVulkanPresenter()
    presenter.filament_bridge = FakeBridge()
    presenter.vulkan = SimpleNamespace(
        queue_family_index=0,
        vk=SimpleNamespace(
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT=1,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL=2,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT=4,
        ),
        image_state=lambda _image: ImageState(0, 0, 0, 0),
        submit_on=lambda *_args, **_kwargs: 1,
        register_image_state=lambda *_args, **_kwargs: None,
    )
    presenter._filament_multiview_hdr_images = [
        SimpleNamespace(image="hdr", layer_resources=["left", "right"])
    ]
    presenter._filament_multiview_ready_semaphores = ["ready"]
    presenter._filament_multiview_slot_timelines = [0]
    presenter._filament_screen = ((0.0, 0.0, -2.0), 2.0, 1.0, (0.0, 0.0, 0.0))
    frame = VulkanStereoOutputFrame(
        frame_id=7,
        timestamp=0.0,
        left_eye=SimpleNamespace(image="left", width=10, height=20, format=43),
        right_eye=SimpleNamespace(image="right", width=10, height=20, format=43),
        metadata={"_vulkan_source_prepare_for_sampling": lambda *_args: None},
    )
    monkeypatch.setattr(
        "xr_viewer.core_openxr_vulkan._update_filament_stereo_camera",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "xr_viewer.core_openxr_vulkan._cffi_handle_address",
        lambda *_args: 123,
    )

    presenter._render_filament_multiview(
        [object(), object()], frame, False, lambda *_args: None
    )

    assert active_eyes[-1] == 0


def test_projection_reuses_displayed_output_without_releasing_it(monkeypatch) -> None:
    monkeypatch.setenv("D2S_VULKAN_PROJECTION_COMPOSER", "0")
    calls: list[tuple[str, object]] = []
    from utils.breakdown import FPSBreakdown

    breakdown = FPSBreakdown(enabled=True, target_fps=60)

    class FakeXr:
        INFINITE_DURATION = 1

        @staticmethod
        def acquire_swapchain_image(_handle):
            return 0

        @staticmethod
        def wait_swapchain_image(_handle, _wait_info):
            return None

        @staticmethod
        def release_swapchain_image(_handle):
            return None

        @staticmethod
        def SwapchainImageWaitInfo(*, timeout):
            return timeout

    class FakeBridge:
        def set_active_eye(self, _eye_index):
            pass

        def set_acquired_image(self, _image_index):
            pass

        def begin_frame(self):
            pass

        def end_frame(self):
            pass

        def apply_animations(self, _seconds):
            pass

    displayed = VulkanStereoOutputFrame(
        frame_id=17,
        timestamp=0.0,
        left_eye=SimpleNamespace(image="left", width=10, height=20, format=43),
        right_eye=SimpleNamespace(image="right", width=10, height=20, format=43),
        metadata={
            "_vulkan_source_prepare_for_sampling": (
                lambda _frame_id, eye_index: ("left-ready", "right-ready")[eye_index]
            )
        },
    )
    presenter = OpenXrVulkanPresenter(
        on_breakdown_inc=breakdown.inc,
        on_breakdown_set_latest=breakdown.set_latest,
    )
    presenter.xr = FakeXr
    presenter.vulkan = SimpleNamespace()
    presenter.swapchains = [
        _EyeSwapchain("left", [SimpleNamespace(image=None)], 1, 1),
        _EyeSwapchain("right", [SimpleNamespace(image=None)], 1, 1),
    ]
    presenter.filament_bridge = FakeBridge()
    presenter._filament_screen = ((0.0, 0.0, -2.0), 2.0, 1.0, (0.0, 0.0, 0.0))
    presenter._displayed_output = displayed
    presenter._apply_filament_profile = lambda views: views
    presenter._report_screen_resolution = lambda *_args: None
    presenter._apply_screen_sampling_policy = lambda *_args: None
    presenter._update_filament_controllers = lambda *_args: None
    monkeypatch.setattr(
        "xr_viewer.core_openxr_vulkan._update_filament_camera",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        OpenXrCompositionBuilder,
        "projection_layer",
        lambda *_args: "projection",
    )

    assert presenter._render_projection_layer([object(), object()], None) == "projection"
    assert presenter._displayed_output is displayed
    assert not any(name == "screen" for name, _value in calls)
    assert "openxr_reused_screen_frame" not in breakdown.stats
    assert "screen_present" in breakdown.validate_openxr_async().missing


def test_tool_quad_layers_never_reuse_the_main_screen() -> None:
    from utils.breakdown import FPSBreakdown

    breakdown = FPSBreakdown(enabled=True, target_fps=60)
    presenter = OpenXrVulkanPresenter(
        on_breakdown_inc=breakdown.inc,
        on_breakdown_set_latest=breakdown.set_latest,
    )
    presenter._last_screen_quad_layers = ["cached-screen"]
    presenter._render_tool_quad_layers = lambda _frame: ["tools"]

    assert presenter._render_quad_layers(None) == ["tools"]
    assert "openxr_reused_screen_frame" not in breakdown.stats
    assert "screen_present" in breakdown.validate_openxr_async().missing


def test_settings_menu_uses_one_both_eye_cached_tool_quad() -> None:
    presenter = OpenXrVulkanPresenter()
    assert presenter._settings_menu.visible is False
    presenter._head_position_w = np.asarray((0.0, 1.6, 0.0), dtype=np.float64)
    presenter._head_forward_w = np.asarray((0.0, 0.0, -1.0), dtype=np.float64)
    presenter._head_model_matrix = np.eye(4, dtype=np.float64)
    presenter._open_settings_menu()
    assert presenter._settings_menu.visible is True
    assert presenter._settings_menu_pose is not None
    assert presenter._settings_menu_pose[0] == pytest.approx((0.0, 1.48, -1.1))


def test_settings_menu_grip_drag_preserves_controller_relative_pose() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._settings_menu_pose = ((0.0, 1.0, -1.0), (0.0, 0.0, 0.0, 1.0))
    presenter._settings_menu.visible = True
    presenter._grip_mat_l = np.eye(4, dtype=np.float64)
    inputs = ({"grip": 1.0}, {"grip": 0.0})
    presenter._handle_settings_menu_grip_drag(inputs, ((0.5, 0.5), None))
    presenter._grip_mat_l[:3, 3] = (0.25, 0.1, -0.2)
    presenter._handle_settings_menu_grip_drag(inputs, ((0.5, 0.5), None))
    assert presenter._settings_menu_pose[0] == pytest.approx((0.25, 1.1, -1.2))


def test_settings_menu_render_scale_defers_rebuild_until_slider_release() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._settings_menu.set_tab("picture")
    control = next(
        item for item in presenter._settings_menu.controls()
        if item.key == "openxr_render_scale"
    )

    presenter._apply_settings_menu_control(control, (control.rect[2], 0.5))

    assert presenter._settings_menu_values["openxr_render_scale"] == 4.0
    assert presenter._pending_openxr_render_scale is None


def test_settings_menu_plus_button_applies_exact_slider_step() -> None:
    calls = []
    presenter = OpenXrVulkanPresenter(
        on_controller_shortcut=lambda action, **values: calls.append(
            (action, values)
        ) or True
    )
    presenter._settings_menu_values["color_brightness"] = 1.0
    presenter._settings_menu.set_tab("picture")
    plus = next(
        item for item in presenter._settings_menu.controls()
        if item.key == "step:plus:color_brightness"
    )

    presenter._apply_settings_menu_control(plus, (0.0, 0.0))

    assert presenter._settings_menu_values["color_brightness"] == pytest.approx(1.1)
    assert calls[-1] == (
        "set_runtime_setting",
        {"name": "color_brightness", "value": pytest.approx(1.1), "persist": True},
    )


def test_settings_menu_depth_toggle_and_cross_eyed_are_dispatched() -> None:
    calls = []
    presenter = OpenXrVulkanPresenter(
        on_controller_shortcut=lambda action, **values: calls.append(
            (action, values)
        ) or True
    )
    presenter._settings_menu.set_tab("depth")
    presenter._settings_menu_values.update({
        "depth_strength": 0.25, "cross_eyed": False,
    })
    controls = {item.key: item for item in presenter._settings_menu.controls()}

    presenter._apply_settings_menu_control(
        controls["depth:toggle_stereo"], (0.0, 0.0)
    )
    presenter._apply_settings_menu_control(
        controls["depth:toggle_cross_eyed"], (0.0, 0.0)
    )

    assert calls[0][0] == "toggle_stereo"
    assert calls[1] == (
        "set_runtime_setting",
        {"name": "cross_eyed", "value": True, "persist": True},
    )


def test_settings_menu_glow_mode_uses_existing_runtime_state_machine() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_glow_environment_enabled = True
    presenter._settings_menu.set_tab("glow")
    controls = {
        item.key: item for item in presenter._settings_menu.controls(show_glow=True)
    }

    presenter._apply_settings_menu_control(
        controls["glow:surround"], (0.0, 0.0)
    )
    assert presenter._filament_glow_mode == "surround"
    assert presenter._filament_glow_intensity_multiplier == 0.0
    assert presenter._filament_glow_shell_intensity_multiplier > 0.0

    presenter._apply_settings_menu_control(controls["glow:off"], (0.0, 0.0))
    assert presenter._filament_glow_mode == "off"
    assert presenter._filament_glow_intensity_multiplier == 0.0
    assert presenter._filament_glow_shell_intensity_multiplier == 0.0


def test_default_glow_mode_change_is_persisted_and_reloaded() -> None:
    persisted = {"Default": "off"}
    calls = []
    presenter = OpenXrVulkanPresenter(
        OpenXrVulkanConfig(filament_glow_modes=persisted),
        on_controller_shortcut=lambda action, **values: (
            calls.append((action, values)) or True
        ),
    )
    presenter._filament_glow_environment_enabled = True
    presenter._filament_screen_state_environment = "Default"

    presenter._set_filament_glow_mode("glow", persist=True)

    assert calls[-1] == (
        "persist_openxr_glow_mode",
        {"environment": "Default", "mode": "glow"},
    )
    assert persisted["Default"] == "glow"


def test_persisted_default_glow_mode_overrides_profile_defaults() -> None:
    presenter = OpenXrVulkanPresenter(
        # This is how PyYAML loads the unquoted value written for "off".
        OpenXrVulkanConfig(filament_glow_modes={"Default": False})
    )
    presenter._filament_screen_state_environment = "Default"

    presenter._apply_filament_glow_profile_fields(
        {
            "glow_mode": "veil",
            "glow_intensity_multiplier": 1.85,
            "glow_shell_intensity_multiplier": 0.0,
        }
    )

    assert presenter._filament_glow_mode == "off"
    assert presenter._filament_glow_intensity_multiplier == 0.0
    assert presenter._filament_glow_shell_intensity_multiplier == 0.0


def test_persisted_default_glow_mode_wins_after_lighting_preset() -> None:
    presenter = OpenXrVulkanPresenter(
        OpenXrVulkanConfig(filament_glow_modes={"Default": "glow"})
    )
    presenter._filament_screen_state_environment = "Default"
    presenter._apply_filament_glow_profile_fields({
        "glow_mode": "veil",
        "glow_intensity_multiplier": 1.85,
        "glow_shell_intensity_multiplier": 0.0,
    })

    presenter._apply_filament_lighting_preset(
        {
            "glow_mode": "surround",
            "glow_intensity_multiplier": 0.0,
            "glow_shell_intensity_multiplier": 1.85,
        },
        apply_bridge=False,
    )
    # This is the final load-stage reapplication used after the profile's
    # selected preset has been resolved.
    presenter._apply_filament_glow_profile_fields({})

    assert presenter._filament_glow_mode == "glow"
    assert presenter._filament_glow_intensity_multiplier > 0.0
    assert presenter._filament_glow_shell_intensity_multiplier == 0.0


def test_glow_transparency_slider_updates_gpu_parameters_and_persists() -> None:
    calls = []
    presenter = OpenXrVulkanPresenter(
        on_controller_shortcut=lambda action, **values: (
            calls.append((action, values)) or True
        ),
    )
    presenter._filament_glow_environment_enabled = True
    presenter._settings_menu.set_tab("glow")
    controls = {
        item.key: item for item in presenter._settings_menu.controls(show_glow=True)
    }
    transparency = controls["glow:transparency"]
    fraction = (0.35 - transparency.minimum) / (
        transparency.maximum - transparency.minimum
    )
    u = transparency.rect[0] + fraction * (
        transparency.rect[2] - transparency.rect[0]
    )

    presenter._apply_settings_menu_control(transparency, (u, 0.0))

    assert presenter._veil_alpha == pytest.approx(0.65)
    assert calls[-1] == (
        "persist_openxr_glow_transparency",
        {"environment": "Default", "transparency": pytest.approx(0.35)},
    )
    presenter._filament_glow_mode = "glow"
    presenter._filament_glow_intensity_multiplier = 1.0
    presenter._filament_screen = ((0.0, 0.0, -2.0), 2.0, 1.0, (0.0, 0.0, 0.0))
    glow_state = presenter._projection_glow_state()
    assert glow_state is not None
    assert np.frombuffer(glow_state[1], dtype="<f4")[13] == pytest.approx(0.65)


def test_settings_menu_glow_control_is_ignored_outside_default_environment() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_glow_environment_enabled = False
    presenter._filament_glow_mode = "off"
    presenter._apply_settings_menu_control(
        type("Control", (), {"key": "glow:glow"})(), (0.0, 0.0)
    )
    assert presenter._filament_glow_mode == "off"


def test_settings_menu_room_exposure_updates_filament_immediately() -> None:
    presenter = OpenXrVulkanPresenter()
    calls = []
    persisted = []
    presenter.filament_bridge = type(
        "Bridge", (), {"set_scene_exposure": lambda _self, value: calls.append(value)}
    )()
    presenter._dispatch_controller_shortcut = (
        lambda action, **values: persisted.append((action, values))
    )
    presenter._settings_menu.set_tab("room")
    control = next(
        item for item in presenter._settings_menu.controls()
        if item.key == "room:exposure"
    )
    presenter._apply_settings_menu_control(
        control, (control.rect[2], 0.5), persist=True
    )
    assert presenter._filament_scene_exposure == pytest.approx(8.0)
    assert calls == [pytest.approx(8.0)]
    assert persisted == []


def test_settings_menu_room_exposure_uses_only_vulkan_resolve_in_multiview() -> None:
    calls = []
    presenter = OpenXrVulkanPresenter()
    presenter._multiview_active = True
    presenter.filament_bridge = type(
        "Bridge", (), {"set_scene_exposure": lambda _self, value: calls.append(value)}
    )()
    presenter._settings_menu.set_tab("room")
    control = next(
        item for item in presenter._settings_menu.controls()
        if item.key == "room:exposure"
    )

    presenter._apply_settings_menu_control(control, (control.rect[2], 0.5))
    presenter._apply_settings_menu_control(control, (
        (control.rect[0] + control.rect[2]) * 0.5, 0.5
    ))

    assert presenter._filament_scene_exposure == pytest.approx(0.0)
    assert calls == []


def test_multiview_hdr_resolve_consumes_current_room_exposure() -> None:
    source = inspect.getsource(OpenXrVulkanPresenter._resolve_filament_multiview_hdr)

    assert "exposure_ev=float(self._filament_scene_exposure)" in source


def test_settings_menu_room_screen_reflection_toggle_removes_lights_immediately() -> None:
    calls = []

    class Bridge:
        def set_environment_screen_area_light(self, *args, **kwargs):
            calls.append((args, kwargs))
            return True

    presenter = OpenXrVulkanPresenter()
    presenter.filament_bridge = Bridge()
    presenter._environment_screen_light_enabled = True
    presenter._environment_screen_light_applied = True
    presenter._settings_menu.set_tab("room")
    control = next(
        item for item in presenter._settings_menu.controls()
        if item.key == "room:toggle_screen_reflection"
    )

    presenter._apply_settings_menu_control(control, (0.0, 0.0))

    assert presenter._environment_screen_light_enabled is False
    assert presenter._environment_screen_light_applied is False
    assert presenter._settings_menu_values[
        "room:screen_reflection_enabled"
    ] is False
    assert calls[-1][1]["enabled"] is False

    presenter._apply_settings_menu_control(control, (0.0, 0.0))
    assert presenter._environment_screen_light_enabled is True


def test_settings_menu_room_model_requests_hot_switch(monkeypatch) -> None:
    calls = []
    presenter = OpenXrVulkanPresenter()
    monkeypatch.setattr(
        presenter, "_hot_switch_environment",
        lambda model: calls.append(model) or True,
    )
    control = type(
        "Control", (), {"key": "room:model:3d_theater", "label": "Theater"}
    )()
    presenter._apply_settings_menu_control(control, (0.0, 0.0))
    assert calls == ["3d_theater"]


def test_settings_menu_room_list_includes_glb_and_panorama_profiles() -> None:
    presenter = OpenXrVulkanPresenter()

    presenter._refresh_settings_menu_values()

    assert presenter._settings_menu.room_tab_visible is True
    assert "tab:room" in {
        control.key for control in presenter._settings_menu.controls()
    }
    room_keys = {key for key, _label in presenter._settings_menu.room_models}
    assert "3d_theater" in room_keys
    assert "hdr_lakesky" in room_keys
    assert "hdr_universe" in room_keys
    assert "Default" in room_keys


def test_environment_hot_switch_unloads_glb_for_panorama_and_then_persists(
    tmp_path, monkeypatch, capsys
) -> None:
    old_profile = tmp_path / "old" / "profile.json"
    old_glb = tmp_path / "old" / "environment.glb"
    new_profile = tmp_path / "hdr" / "profile.json"
    panorama = tmp_path / "hdr" / "sky.hdr"
    old_profile.parent.mkdir()
    new_profile.parent.mkdir()
    old_profile.write_text(json.dumps({"glb": "environment.glb"}), encoding="utf-8")
    old_glb.write_bytes(b"old-glb")
    new_profile.write_text(json.dumps({
        "environment_type": "panorama",
        "background": {"image": "sky.hdr"},
        "glb": None,
    }), encoding="utf-8")
    panorama.write_bytes(b"hdr")

    class Bridge:
        def __init__(self):
            self.calls = []
        def wait_for_idle(self): self.calls.append("idle")
        def unload_glb(self): self.calls.append("unload")
        def load_glb(self, data): self.calls.append(("load", bytes(data)))
        def set_scene_exposure(self, value): self.calls.append(("exposure", value))
        def set_skybox_brightness(self, value): self.calls.append(("skybox", value))

    persisted = []
    presenter = OpenXrVulkanPresenter(OpenXrVulkanConfig(
        filament_glb_path=str(old_glb),
        filament_profile_path=str(old_profile),
    ))
    bridge = Bridge()
    presenter.filament_bridge = bridge

    class ProjectionPass:
        panorama_enabled = True
        def __init__(self): self.closed = False
        def close(self): self.closed = True

    projection_pass = ProjectionPass()
    presenter._vulkan_projection_screen_pass = projection_pass
    monkeypatch.setattr(
        presenter, "_resolve_environment_selection",
        lambda _model: (None, new_profile, panorama),
    )
    monkeypatch.setattr(presenter, "_apply_filament_bridge_lighting", lambda *_args: None)
    monkeypatch.setattr(presenter, "_refresh_settings_menu_values", lambda: None)
    monkeypatch.setattr(
        presenter, "_dispatch_controller_shortcut",
        lambda action, **values: persisted.append((action, values)),
    )

    assert presenter._hot_switch_environment("hdr") is True
    assert bridge.calls[:2] == ["idle", "unload"]
    assert presenter.config.filament_glb_path is None
    assert presenter.config.filament_panorama_path == str(panorama)
    assert projection_pass.closed is False
    assert presenter._vulkan_projection_screen_pass is projection_pass
    assert persisted == [("select_environment_model", {"model": "hdr"})]
    output = capsys.readouterr().out
    assert "[OpenXRViewer] Environment hot switch complete: model=hdr\n" in output
    assert not any(
        " glb=" in line or " panorama=" in line
        for line in output.splitlines()
        if "Environment hot switch complete:" in line
    )


def test_environment_hot_switch_rolls_back_glb_and_does_not_persist_on_failure(
    tmp_path, monkeypatch
) -> None:
    old_profile = tmp_path / "old" / "profile.json"
    old_glb = tmp_path / "old" / "environment.glb"
    new_profile = tmp_path / "new" / "profile.json"
    new_glb = tmp_path / "new" / "environment.glb"
    old_profile.parent.mkdir()
    new_profile.parent.mkdir()
    old_profile.write_text(json.dumps({"glb": "environment.glb"}), encoding="utf-8")
    old_glb.write_bytes(b"old-glb")
    new_profile.write_text(json.dumps({"glb": "environment.glb"}), encoding="utf-8")
    new_glb.write_bytes(b"new-glb")

    class Bridge:
        def __init__(self): self.loads = []
        def wait_for_idle(self): pass
        def load_glb(self, data):
            payload = bytes(data)
            self.loads.append(payload)
            if payload == b"new-glb":
                raise RuntimeError("target rejected")
        def unload_glb(self): pass

    persisted = []
    presenter = OpenXrVulkanPresenter(OpenXrVulkanConfig(
        filament_glb_path=str(old_glb),
        filament_profile_path=str(old_profile),
    ))
    bridge = Bridge()
    presenter.filament_bridge = bridge
    monkeypatch.setattr(
        presenter, "_resolve_environment_selection",
        lambda _model: (new_glb, new_profile, None),
    )
    monkeypatch.setattr(presenter, "_apply_filament_bridge_lighting", lambda *_args: None)
    monkeypatch.setattr(
        presenter, "_dispatch_controller_shortcut",
        lambda action, **values: persisted.append((action, values)),
    )

    assert presenter._hot_switch_environment("new") is False
    assert bridge.loads == [b"new-glb", b"old-glb"]
    assert presenter.config.filament_glb_path == str(old_glb)
    assert persisted == []


def test_settings_menu_three_seat_switch_restarts_profile_calibration() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_profile_data = {
        "model_position": [0.0, 0.0, 0.0],
        "model_rotation_deg": [0.0, 0.0, 0.0],
        "model_scale": [1.0, 1.0, 1.0],
    }
    presenter._filament_view_poses = (
        {"name": "Front", "x": 0.0, "y": 1.0, "z": -1.0},
        {"name": "Middle", "x": 0.0, "y": 1.2, "z": 0.0},
        {"name": "Back", "x": 0.0, "y": 1.4, "z": 1.0},
    )
    presenter._profile_space_applied = True
    presenter._profile_space_calibration_pass = 2
    presenter._profile_head_transform = np.eye(4, dtype=np.float32)
    presenter._profile_head_transform[:3, 3] = (0.0, 1.2, 0.0)
    presenter._profile_reference_head_anchor = np.eye(4, dtype=np.float32)
    presenter._settings_menu_pose = (
        (0.25, 1.0, -1.0), (0.0, 0.0, 0.0, 1.0)
    )

    presenter._apply_settings_menu_seat(2)

    assert presenter._filament_view_pose_index == 2
    assert presenter._profile_view_name == "Back"
    assert presenter._profile_head_transform[:3, 3] == pytest.approx(
        (0.0, 1.4, 1.0)
    )
    assert presenter._profile_space_applied is False
    assert presenter._profile_space_calibration_pass == 0
    assert presenter._profile_space_preserve_anchor is True
    assert presenter._settings_menu_pose[0] == pytest.approx((0.25, 1.2, 0.0))


def test_live_seat_switch_does_not_bake_current_menu_gaze_into_space() -> None:
    created_poses = []

    class FakeXr:
        ReferenceSpaceType = xr.ReferenceSpaceType
        ReferenceSpaceCreateInfo = xr.ReferenceSpaceCreateInfo

        @staticmethod
        def create_reference_space(_session, create_info):
            created_poses.append(create_info.pose_in_reference_space)
            return "seat-space"

        @staticmethod
        def destroy_space(_space):
            pass

    presenter = OpenXrVulkanPresenter()
    presenter.xr = FakeXr
    presenter.session = object()
    presenter.reference_space = "old-seat-space"
    presenter._xr_space = "old-seat-space"
    presenter._reference_space_type = xr.ReferenceSpaceType.STAGE
    presenter._profile_reference_head_anchor = np.eye(4, dtype=np.float32)
    presenter._profile_space_preserve_anchor = True
    presenter._profile_head_transform = euler_to_mat4(
        math.radians(90.0), 0.0, 0.0
    ).astype(np.float32)

    gaze = euler_to_mat4(math.radians(30.0), 0.0, 0.0)
    gaze_quat = mat4_to_xr_posef(gaze).orientation
    views = [
        SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=x, y=0.0, z=0.0),
                orientation=gaze_quat,
            )
        )
        for x in (-0.03, 0.03)
    ]

    assert presenter._apply_profile_reference_space(views)
    created = _xr_view_pose_to_model_mat4(created_poses[0])
    expected = np.linalg.inv(presenter._profile_head_transform)
    assert created == pytest.approx(expected, abs=1e-5)
    assert presenter._profile_space_applied
    assert not presenter._profile_space_preserve_anchor


def test_settings_menu_follows_live_seat_height_change() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._profile_head_transform = np.eye(4, dtype=np.float32)
    presenter._profile_head_transform[:3, 3] = (2.0, 1.5, -3.0)
    presenter._settings_menu_pose = (
        (2.25, 1.2, -4.0), (0.0, 0.0, 0.0, 1.0)
    )

    presenter._apply_settings_menu_seat_height(0.75)

    assert presenter._profile_head_transform[:3, 3] == pytest.approx(
        (2.0, 2.25, -3.0)
    )
    assert presenter._settings_menu_pose[0] == pytest.approx((2.25, 1.95, -4.0))


def test_settings_menu_screen_rotation_and_reset_restore_profile_pose() -> None:
    presenter = OpenXrVulkanPresenter()
    initial = ((1.0, 2.0, -3.0), 4.0, 2.25, (10.0, 20.0, 30.0))
    presenter._filament_screen_initial = initial
    presenter._filament_screen = initial
    presenter._screen_initial_curve_half_angle = math.radians(20.0)
    presenter._screen_curve_half_angle = math.radians(20.0)
    presenter._screen_curved = True
    presenter._settings_menu.set_tab("screen")
    controls = {item.key: item for item in presenter._settings_menu.controls()}

    presenter._apply_settings_menu_control(
        controls["screen:rotate:+90"], (0.0, 0.0)
    )
    assert presenter._filament_screen[3] == pytest.approx((10.0, 20.0, 120.0))

    presenter._apply_settings_menu_control(
        controls["section:reset_defaults"], (0.0, 0.0)
    )
    assert presenter._filament_screen == (
        initial[0], initial[1], initial[2], (10.0, 20.0, 120.0)
    )
    assert presenter._screen_curve_half_angle == pytest.approx(math.radians(20.0))


def test_settings_menu_render_scale_step_schedules_one_rebuild() -> None:
    calls = []
    presenter = OpenXrVulkanPresenter(
        on_controller_shortcut=lambda action, **values: calls.append(
            (action, values)
        ) or True
    )
    presenter._settings_menu_values["openxr_render_scale"] = 1.0
    presenter._settings_menu.set_tab("picture")
    minus = next(
        item for item in presenter._settings_menu.controls()
        if item.key == "step:minus:openxr_render_scale"
    )

    presenter._apply_settings_menu_control(minus, (0.0, 0.0))

    assert presenter._settings_menu_values["openxr_render_scale"] == pytest.approx(0.95)
    assert presenter._pending_openxr_render_scale == pytest.approx(0.95)
    assert calls[-1][0] == "persist_openxr_render_scale"


def test_rocm_automatic_render_scale_is_capped_without_changing_manual_or_cuda(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        OpenXrVulkanPresenter,
        "_is_rocm_backend",
        staticmethod(lambda: True),
    )
    rocm_startup_auto = OpenXrVulkanPresenter(
        OpenXrVulkanConfig(
            render_scale=1.24,
            render_scale_auto=True,
            headset_model="Meta Quest 2",
        )
    )
    assert rocm_startup_auto._openxr_render_scale == pytest.approx(1.0)

    rocm_auto = OpenXrVulkanPresenter(
        OpenXrVulkanConfig(
            render_scale=0.8,
            render_scale_auto=True,
            headset_model="Meta Quest 2",
        )
    )
    assert rocm_auto._openxr_render_scale == pytest.approx(0.8)
    rebuild_scales = []
    rocm_auto._release_projection_render_targets = lambda: None
    rocm_auto._create_projection_swapchains_for_scale = (
        lambda scale: rebuild_scales.append(scale)
    )
    rocm_auto._initialize_filament_bridges = lambda: None
    rocm_auto._projection_eye_extents = lambda: ()
    rocm_auto._pending_openxr_render_scale = 1.24

    rocm_auto._apply_pending_openxr_render_scale()

    assert rebuild_scales == [pytest.approx(1.0)]
    assert rocm_auto._openxr_render_scale == pytest.approx(1.0)

    rocm_manual = OpenXrVulkanPresenter(
        OpenXrVulkanConfig(
            render_scale=1.24,
            render_scale_auto=False,
            headset_model="Meta Quest 2",
        )
    )
    assert rocm_manual._openxr_render_scale == pytest.approx(1.24)

    monkeypatch.setattr(
        OpenXrVulkanPresenter,
        "_is_rocm_backend",
        staticmethod(lambda: False),
    )
    cuda_auto = OpenXrVulkanPresenter(
        OpenXrVulkanConfig(
            render_scale=1.24,
            render_scale_auto=True,
            headset_model="Meta Quest 2",
        )
    )
    assert cuda_auto._openxr_render_scale == pytest.approx(1.24)


def test_visible_settings_menu_disables_screen_edge_ray_attraction() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._settings_menu.visible = True
    presenter._aim_mat_l = np.eye(4, dtype=np.float64)
    presenter._grip_mat_l = np.eye(4, dtype=np.float64)
    presenter._get_smoothed_ray = lambda _hand: (
        np.asarray((0.0, 0.0, 0.0), dtype=np.float64),
        np.asarray((0.0, 0.0, -1.0), dtype=np.float64),
    )
    presenter._screen_ray_hit = lambda *_args: None
    presenter._screen_plane_uv = lambda *_args: (1.02, 0.5)
    presenter._screen_uv_to_world = lambda *_args: np.asarray(
        (0.1, 0.0, -1.0), dtype=np.float64
    )

    _origin, direction = presenter._controller_interaction_ray(0)

    expected = presenter._normalize_interaction_ray(
        np.asarray((0.0, 0.0, -1.0), dtype=np.float64)
        * math.cos(math.radians(12.0))
        + np.cross(
            np.asarray((1.0, 0.0, 0.0), dtype=np.float64),
            np.asarray((0.0, 0.0, -1.0), dtype=np.float64),
        )
        * math.sin(math.radians(12.0))
    )
    assert direction == pytest.approx(expected)


@pytest.mark.parametrize(
    ("key", "half_angle"),
    (
        ("screen:type:flat", 0.0),
        ("screen:type:subtle", math.radians(20.0)),
        ("screen:type:medium", math.radians(30.0)),
        ("screen:type:deep", 0.72),
    ),
)
def test_settings_menu_screen_type_sets_real_curve_half_angle(key, half_angle):
    presenter = OpenXrVulkanPresenter()
    presenter._settings_menu.set_tab("screen")
    control = next(item for item in presenter._settings_menu.controls() if item.key == key)
    presenter._apply_settings_menu_control(control, (0.5, 0.5))
    assert presenter._screen_curve_half_angle == pytest.approx(half_angle)
    assert presenter._screen_curved is (half_angle > 0.0)


def test_settings_menu_quad_is_submitted_before_laser_cursor() -> None:
    source = inspect.getsource(OpenXrVulkanPresenter._render_tool_quad_layers)
    menu = source.index('"settings_menu", menu_rgba')
    cursor = source.index('self._cursor_overlay_specs')
    assert menu < cursor
    assert "if not self._settings_menu.visible" in source


def test_tool_quad_format_is_enumerated_once_per_session() -> None:
    calls = 0

    class FakeXr:
        @staticmethod
        def enumerate_swapchain_formats(_session):
            nonlocal calls
            calls += 1
            return [43]

    presenter = OpenXrVulkanPresenter()
    presenter.xr = FakeXr
    presenter.session = object()
    presenter.vulkan = SimpleNamespace(
        vk=SimpleNamespace(
            VK_FORMAT_R8G8B8A8_SRGB=43,
            VK_FORMAT_B8G8R8A8_SRGB=50,
            VK_FORMAT_R8G8B8A8_UNORM=37,
            VK_FORMAT_B8G8R8A8_UNORM=44,
        )
    )

    assert presenter._tool_quad_format() == 43
    assert presenter._tool_quad_format() == 43
    assert calls == 1


def test_projection_layer_builder_owns_only_layer_assembly() -> None:
    class FakeXr:
        CompositionLayerProjectionView = staticmethod(lambda **kwargs: kwargs)
        SwapchainSubImage = staticmethod(lambda **kwargs: kwargs)
        Rect2Di = staticmethod(lambda **kwargs: kwargs)
        Offset2Di = staticmethod(lambda **kwargs: kwargs)
        Extent2Di = staticmethod(lambda **kwargs: kwargs)
        CompositionLayerProjection = staticmethod(lambda **kwargs: kwargs)

    views = [
        SimpleNamespace(pose="left-pose", fov="left-fov"),
        SimpleNamespace(pose="right-pose", fov="right-fov"),
    ]
    swapchains = [
        _EyeSwapchain("left-chain", [], 10, 20),
        _EyeSwapchain("right-chain", [], 30, 40),
    ]
    layer = OpenXrCompositionBuilder(FakeXr, "local-space").projection_layer(
        views, swapchains
    )
    assert layer["space"] == "local-space"
    assert [view["pose"] for view in layer["views"]] == ["left-pose", "right-pose"]
    assert layer["views"][1]["sub_image"]["image_rect"]["extent"] == {
        "width": 30,
        "height": 40,
    }


def test_projection_layer_builder_maps_multiview_to_array_layers() -> None:
    class FakeXr:
        CompositionLayerProjectionView = staticmethod(lambda **kwargs: kwargs)
        SwapchainSubImage = staticmethod(lambda **kwargs: kwargs)
        Rect2Di = staticmethod(lambda **kwargs: kwargs)
        Offset2Di = staticmethod(lambda **kwargs: kwargs)
        Extent2Di = staticmethod(lambda **kwargs: kwargs)
        CompositionLayerProjection = staticmethod(lambda **kwargs: kwargs)

    views = [
        SimpleNamespace(pose="left-pose", fov="left-fov"),
        SimpleNamespace(pose="right-pose", fov="right-fov"),
    ]
    swapchain = _EyeSwapchain("stereo-chain", [], 10, 20, array_size=2)

    layer = OpenXrCompositionBuilder(FakeXr, "local-space").projection_layer(
        views, [swapchain], layer_flags=7
    )

    assert [
        view["sub_image"]["image_array_index"] for view in layer["views"]
    ] == [0, 1]
    assert [view["sub_image"]["swapchain"] for view in layer["views"]] == [
        "stereo-chain",
        "stereo-chain",
    ]
    assert layer["layer_flags"] == 7


def test_projection_array_eye_diagnostic_clears_two_layers(monkeypatch) -> None:
    calls = []
    presenter = OpenXrVulkanPresenter()
    presenter.vulkan = SimpleNamespace(
        clear_color_image=lambda image, color, **kwargs: calls.append(
            (image, color, kwargs)
        )
    )
    swapchain = _EyeSwapchain(
        "stereo-chain",
        [],
        10,
        20,
        resources=[SimpleNamespace(image="stereo-image")],
        array_size=2,
    )

    presenter._render_projection_array_eye_diagnostic([(swapchain, 0)])

    assert calls == [
        ("stereo-image", (1.0, 0.0, 0.0, 1.0), {"base_array_layer": 0}),
        ("stereo-image", (0.0, 1.0, 0.0, 1.0), {"base_array_layer": 1}),
    ]


def test_projection_array_eye_diagnostic_requires_layered_swapchain() -> None:
    presenter = OpenXrVulkanPresenter()

    with pytest.raises(RuntimeError, match="one array_size=2 swapchain"):
        presenter._render_projection_array_eye_diagnostic(
            [(_EyeSwapchain("left", [], 10, 20), 0)]
        )


def test_quad_layer_builder_maps_each_eye_to_its_array_layer() -> None:
    class FakeXr:
        EyeVisibility = SimpleNamespace(LEFT="left", RIGHT="right")
        CompositionLayerQuad = staticmethod(lambda **kwargs: kwargs)
        SwapchainSubImage = staticmethod(lambda **kwargs: kwargs)
        Rect2Di = staticmethod(lambda **kwargs: kwargs)
        Offset2Di = staticmethod(lambda **kwargs: kwargs)
        Extent2Di = staticmethod(lambda **kwargs: kwargs)
        Posef = staticmethod(lambda **kwargs: kwargs)
        Quaternionf = staticmethod(lambda **kwargs: kwargs)
        Vector3f = staticmethod(lambda **kwargs: kwargs)
        Extent2Df = staticmethod(lambda **kwargs: kwargs)

    swapchain = _EyeSwapchain("stereo-quad", [], 10, 20, array_size=2)
    builder = OpenXrCompositionBuilder(FakeXr, "local-space")

    layers = [
        builder.quad_layer(swapchain, (0.0, 0.0, -1.0), 1.0, 1.0, (0.0, 0.0, 0.0), eye)
        for eye in (0, 1)
    ]

    assert [layer["eye_visibility"] for layer in layers] == ["left", "right"]
    assert [layer["sub_image"]["image_array_index"] for layer in layers] == [0, 1]


def test_copy_image_reads_selected_source_layer_into_destination_layer_zero() -> None:
    barriers = []
    copies = []

    class RecordingVk:
        def __getattr__(self, name):
            return getattr(vk, name)

        def vkCmdPipelineBarrier(self, *args):
            barriers.append(args[-1])

        def vkCmdCopyImage(self, *args):
            copies.append(args[-1][0])

    context = object.__new__(VulkanContext)
    context.vk = RecordingVk()
    context.queue_family_index = 0
    context._closed = False
    context._device_lost = False
    context._image_states = ImageStateTracker(default_queue_family_index=0)
    context.submit_on = lambda _role, record, **_kwargs: (record("commands"), 7)[1]
    source = SimpleNamespace(
        context=context,
        image=vk.ffi.cast("VkImage", 1),
        width=8,
        height=4,
        format=vk.VK_FORMAT_R8G8B8A8_UNORM,
    )
    destination = SimpleNamespace(
        context=context,
        image=vk.ffi.cast("VkImage", 2),
        width=8,
        height=4,
        format=vk.VK_FORMAT_R8G8B8A8_UNORM,
    )
    context._image_states.update(1, ImageState(10, 11, 12, 0))
    context._image_states.update(2, ImageState(20, 21, 22, 0))

    assert context.copy_image(source, destination, source_array_layer=1) == 7
    assert barriers[0][0].subresourceRange.baseArrayLayer == 1
    assert barriers[0][1].subresourceRange.baseArrayLayer == 0
    assert copies[0].srcSubresource.baseArrayLayer == 1
    assert copies[0].dstSubresource.baseArrayLayer == 0
    assert barriers[1][0].subresourceRange.baseArrayLayer == 1
    assert barriers[1][1].subresourceRange.baseArrayLayer == 0


def test_copy_image_can_leave_destination_host_readable() -> None:
    barriers = []

    class RecordingVk:
        def __getattr__(self, name):
            return getattr(vk, name)

        def vkCmdPipelineBarrier(self, *args):
            barriers.append(args[-1])

        def vkCmdCopyImage(self, *_args):
            return None

    context = object.__new__(VulkanContext)
    context.vk = RecordingVk()
    context.queue_family_index = 0
    context._closed = False
    context._device_lost = False
    context._image_states = ImageStateTracker(default_queue_family_index=0)
    context.submit_on = lambda _role, record, **_kwargs: (record("commands"), 9)[1]
    source = SimpleNamespace(
        context=context,
        image=vk.ffi.cast("VkImage", 3),
        width=8,
        height=4,
        format=vk.VK_FORMAT_R8G8B8A8_UNORM,
    )
    destination = SimpleNamespace(
        context=context,
        image=vk.ffi.cast("VkImage", 4),
        width=8,
        height=4,
        format=vk.VK_FORMAT_R8G8B8A8_UNORM,
    )
    context._image_states.update(3, ImageState(10, 11, 12, 0))
    context._image_states.update(4, ImageState(20, 21, 22, 0))

    assert context.copy_image(
        source, destination, destination_host_readable=True
    ) == 9
    final_barrier = barriers[1][1]
    assert final_barrier.newLayout == vk.VK_IMAGE_LAYOUT_GENERAL
    assert final_barrier.dstAccessMask == vk.VK_ACCESS_HOST_READ_BIT
    destination_state = context._image_states.get(
        4, undefined_layout=vk.VK_IMAGE_LAYOUT_UNDEFINED
    )
    assert destination_state.layout == vk.VK_IMAGE_LAYOUT_GENERAL
    assert destination_state.access_mask == vk.VK_ACCESS_HOST_READ_BIT


def test_standalone_vulkan_context_smoke() -> None:
    try:
        context = VulkanContext.create()
    except (
        VulkanUnavailableError,
        VulkanCapabilityError,
        vk.VkErrorIncompatibleDriver,
    ) as exc:
        pytest.skip(str(exc))
    try:
        assert context.device_info.name
        assert context.device_info.queue_family_index >= 0
    finally:
        context.close()
    assert context.closed


def test_vulkan_context_device_loss_close_skips_driver_destruction() -> None:
    calls = []

    class FakeVk:
        def __getattr__(self, name):
            def record(*_args):
                calls.append(name)

            return record

    context = object.__new__(VulkanContext)
    context.vk = FakeVk()
    context.device = "dead-device"
    context.instance = "dead-instance"
    context._timeline_semaphore = "timeline"
    context._frame_contexts = [
        SimpleNamespace(
            queue_resources={
                "graphics": SimpleNamespace(
                    fence="fence", command_pool="command-pool"
                )
            }
        )
    ]
    context._owns_device = True
    context._owns_instance = True
    context._external_image_registry = None
    context._image_states = ImageStateTracker(default_queue_family_index=0)
    context._command_buffer = "command-buffer"
    context._command_pool = "command-pool"
    context._fence = "fence"
    context.queue = "queue"
    context.physical_device = "physical-device"
    context._closed = False
    context._device_lost = True
    context._lock = threading.RLock()

    context.close()

    assert calls == []
    assert context.closed is True


def test_screen_crop_uv_effective_geometry_curve_and_physical_mouse_mapping() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = (
        (0.0, 0.0, -2.0), 4.0, 2.0, (0.0, 0.0, 0.0)
    )
    presenter._screen_crop_width_percent = 10.0
    presenter._screen_crop_height_percent = 20.0
    presenter._screen_curved = True
    presenter._screen_curve_half_angle = 0.50
    presenter._target_monitor_rect = lambda: (100, 200, 1000, 500)

    assert presenter._screen_crop_uv() == pytest.approx((0.1, 0.2, 0.8, 0.6))
    effective = presenter._effective_filament_screen()
    assert effective is not None
    assert effective[1:3] == pytest.approx((3.2, 1.2))
    assert presenter._effective_screen_curve_half_angle() == pytest.approx(0.4)
    assert presenter._cursor_pixel_for_screen_uv(0.0, 0.0) == (200, 600)
    assert presenter._cursor_pixel_for_screen_uv(1.0, 1.0) == (1000, 300)


def test_crop_controls_reset_and_manual_change_disable_dynamic_crop() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = (
        (0.0, 0.0, -2.0), 2.4, 1.35, (0.0, 0.0, 0.0)
    )
    presenter._settings_menu.set_screen_section("crop")
    controls = {item.key: item for item in presenter._settings_menu.controls()}
    presenter._screen_dynamic_crop = True

    width = controls["screen:crop_width"]
    presenter._apply_settings_menu_control(
        width, (width.rect[0] + (width.rect[2] - width.rect[0]) * (12.0 / 45.0), 0.0)
    )

    assert presenter._screen_crop_width_percent == pytest.approx(12.0)
    assert presenter._screen_dynamic_crop is False

    presenter._screen_crop_height_percent = 8.0
    presenter._screen_dynamic_crop = True
    presenter._apply_settings_menu_control(controls["screen:reset_crop"], (0.0, 0.0))

    assert presenter._screen_crop_width_percent == 0.0
    assert presenter._screen_crop_height_percent == 0.0
    assert presenter._screen_dynamic_crop is False


def test_crop_detector_one_shot_and_dynamic_three_result_hysteresis() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = (
        (0.0, 0.0, -2.0), 2.4, 1.35, (0.0, 0.0, 0.0)
    )
    presenter._on_controller_shortcut = lambda *_args, **_kwargs: True
    presenter._screen_auto_crop_pending = True
    presenter._screen_crop_detector_inflight = True

    presenter._apply_screen_crop_detection((0.0, 0.1, 1.0, 0.8), 1)

    assert presenter._screen_auto_crop_pending is False
    assert presenter._screen_dynamic_crop is False
    assert presenter._screen_crop_width_percent == 0.0
    assert presenter._screen_crop_height_percent == pytest.approx(10.0)

    presenter._screen_dynamic_crop = True
    presenter._screen_crop_width_percent = 14.0
    presenter._screen_crop_height_percent = 10.0
    for serial in (2, 3):
        presenter._apply_screen_crop_detection((0.0, 0.0, 1.0, 1.0), serial)
    assert presenter._screen_crop_width_percent == 14.0
    presenter._apply_screen_crop_detection((0.0, 0.0, 1.0, 1.0), 4)
    assert presenter._screen_crop_width_percent == 0.0
    assert presenter._screen_crop_height_percent == 0.0


def test_unavailable_crop_detector_keeps_static_crop_and_disables_dynamic_mode() -> None:
    presenter = OpenXrVulkanPresenter(
        on_controller_shortcut=lambda *_args, **_kwargs: True
    )
    presenter._filament_screen = (
        (0.0, 0.0, -2.0), 2.4, 1.35, (0.0, 0.0, 0.0)
    )
    presenter._screen_crop_width_percent = 14.0
    presenter._screen_crop_height_percent = 10.0
    presenter._screen_dynamic_crop = True
    presenter._screen_auto_crop_pending = True

    presenter._screen_crop_detection_unavailable()

    assert presenter._screen_crop_width_percent == 14.0
    assert presenter._screen_crop_height_percent == 10.0
    assert presenter._screen_dynamic_crop is False
    assert presenter._screen_auto_crop_pending is False


def test_projection_quality_cache_can_release_jittered_extents() -> None:
    class CachedImage:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    first = CachedImage()
    second = CachedImage()
    projection = object.__new__(VulkanProjectionScreenPass)
    projection.quality_images = {(100, 50, 37): [first]}
    projection.mip_images = {(100, 50, 37): [second]}

    projection._close_quality_image_cache()

    assert first.closed is True
    assert second.closed is True
    assert projection.quality_images == {}
    assert projection.mip_images == {}


def test_crop_state_is_persisted_per_openxr_environment() -> None:
    calls = []
    presenter = OpenXrVulkanPresenter(
        on_controller_shortcut=lambda *args, **kwargs: calls.append((args, kwargs)) or True
    )
    presenter._filament_screen_state_environment = "Cinema"
    presenter._filament_screen = (
        (0.0, 0.0, -2.0), 2.4, 1.35, (0.0, 0.0, 0.0)
    )
    presenter._screen_crop_width_percent = 11.0
    presenter._screen_crop_height_percent = 9.0
    presenter._screen_dynamic_crop = True

    presenter._persist_screen_state(force=True)

    state = presenter.config.filament_screen_states["Cinema"]
    assert state["crop_width_percent"] == 11.0
    assert state["crop_height_percent"] == 9.0
    assert state["dynamic_crop"] is True
    assert calls[-1][0][0] == "persist_openxr_screen_state"


def test_vulkan_timeline_timeout_releases_context_lock_without_device_loss() -> None:
    class VkTimeout(Exception):
        pass

    class WaitInfo:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    lock_available = threading.Event()

    class FakeVk:
        VK_SUCCESS = 0
        VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO = 1
        VkSemaphoreWaitInfo = WaitInfo

        @staticmethod
        def vkWaitSemaphores(_device, _wait_info, _timeout_ns):
            with context._lock:
                lock_available.set()
            raise VkTimeout("timeline wait expired")

    context = object.__new__(VulkanContext)
    context.vk = FakeVk()
    context.device = "device"
    context._timeline_semaphore = "timeline"
    context._lock = threading.RLock()
    context._wait_condition = threading.Condition(context._lock)
    context._active_blocking_waits = 0
    context._closed = False
    context._device_lost = False
    context._device_lost_error = None
    errors = []

    def wait() -> None:
        try:
            context.wait_for_timeline(7)
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(
        target=wait,
    )
    worker.start()
    assert lock_available.wait(1.0)
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], VulkanTimelineTimeout)
    assert context.device_lost is False
    assert context._active_blocking_waits == 0


def test_vulkan_timeline_timeout_is_recoverable_error() -> None:
    class WaitInfo:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeVk:
        VK_SUCCESS = 0
        VK_TIMEOUT = 2
        VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO = 1
        VkSemaphoreWaitInfo = WaitInfo

        @staticmethod
        def vkWaitSemaphores(_device, _wait_info, _timeout_ns):
            return 2

    context = object.__new__(VulkanContext)
    context.vk = FakeVk()
    context.device = "device"
    context._timeline_semaphore = "timeline"
    context._lock = threading.RLock()
    context._wait_condition = threading.Condition(context._lock)
    context._active_blocking_waits = 0
    context._closed = False
    context._device_lost = False
    context._device_lost_error = None

    with pytest.raises(VulkanTimelineTimeout):
        context.wait_for_timeline(7)
    assert context.device_lost is False
