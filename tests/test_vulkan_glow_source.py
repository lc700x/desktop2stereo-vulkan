from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from path_config import APP_ROOT

from path_config import APP_ROOT

from stereo_runtime.vulkan_glow_source import VulkanGlowSourceComputeBackend
from stereo_runtime.vulkan_glow_source_pass import VulkanGlowSourcePass


def test_glow_prefilter_uses_legacy_mip_footprint_in_source_pixels() -> None:
    assert VulkanGlowSourceComputeBackend.prefilter_scale("veil") == 1.0
    assert VulkanGlowSourceComputeBackend.prefilter_scale("glow") == 256.0
    assert VulkanGlowSourceComputeBackend.prefilter_scale("surround") == 256.0


def test_glow_pass_contract_is_fixed_rgba_target() -> None:
    effect_pass = object.__new__(VulkanGlowSourcePass)
    effect_pass.target_width = 320
    effect_pass.target_height = 180

    assert effect_pass.group_counts == (40, 23, 1)
    assert effect_pass.input_buffer_size(3840, 2160) == 3840 * 2160 * 12
    assert effect_pass.PUSH_CONSTANTS_SIZE == 64


def test_glow_input_capacity_covers_monitor_aspect_changes() -> None:
    assert VulkanGlowSourceComputeBackend.INPUT_BUFFER_CAPACITY == 512 * 512 * 3 * 4


def test_glow_source_crop_normalization_keeps_a_valid_inner_rectangle() -> None:
    assert VulkanGlowSourceComputeBackend._normalize_source_crop_uv(
        (0.1, 0.2, 0.8, 0.6)
    ) == (0.1, 0.2, 0.8, 0.6)
    assert VulkanGlowSourceComputeBackend._normalize_source_crop_uv(
        (float("nan"), 0.0, 1.0, 1.0)
    ) == (0.0, 0.0, 1.0, 1.0)


def test_glow_fence_poll_handles_pyvulkan_not_ready_exception() -> None:
    class VkNotReady(Exception):
        pass

    class Vk:
        def __init__(self) -> None:
            self.ready = False

        def vkGetFenceStatus(self, _device, _fence):
            if not self.ready:
                raise VkNotReady()
            return None

    backend = object.__new__(VulkanGlowSourceComputeBackend)
    backend.vk = Vk()
    backend.vk.VkNotReady = VkNotReady
    backend.context = SimpleNamespace(device=object())

    assert backend._fence_complete(object()) is False
    backend.vk.ready = True
    assert backend._fence_complete(object()) is True


def test_glow_poll_short_circuits_after_device_loss() -> None:
    backend = object.__new__(VulkanGlowSourceComputeBackend)
    backend._closed = False
    backend.context = SimpleNamespace(device_lost=True)
    backend.slots = SimpleNamespace(
        __iter__=lambda _self: (_ for _ in ()).throw(
            AssertionError("dead Vulkan device must not poll fences")
        )
    )

    backend.poll()


def test_glow_queue_submit_uses_dedicated_lock() -> None:
    class TrackingLock:
        held = False

        def __enter__(self):
            self.held = True

        def __exit__(self, *_args):
            self.held = False

    lock = TrackingLock()
    submissions = []

    class Vk:
        @staticmethod
        def vkQueueSubmit(*args):
            assert lock.held
            submissions.append(args)

    backend = object.__new__(VulkanGlowSourceComputeBackend)
    backend.context = SimpleNamespace(_lock=lock)
    backend._submit_lock = lock
    backend.vk = Vk()

    backend._submit_queue("queue", 1, ["submit"], "fence")

    assert submissions == [("queue", 1, ["submit"], "fence")]
    assert not lock.held


def test_glow_input_wait_is_backend_specific() -> None:
    backend = object.__new__(VulkanGlowSourceComputeBackend)
    backend.vk = SimpleNamespace(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT=32)
    semaphore = object()

    backend._rocm_interop = False
    assert backend._input_ready_wait(semaphore) == ((semaphore,), (32,))

    backend._rocm_interop = True
    assert backend._input_ready_wait(semaphore) == ((), ())


def test_glow_frame_lease_is_counted_once_and_released() -> None:
    slot = SimpleNamespace(
        image=SimpleNamespace(
            resource=SimpleNamespace(width=320, height=180, format=37)
        ),
        lease_count=0,
        screen_light_buffer=SimpleNamespace(),
    )
    backend = object.__new__(VulkanGlowSourceComputeBackend)
    backend._closed = False
    backend._current_slot = slot
    backend._frame_slots = {}
    backend._serial = 3
    backend._last_submit_ms = 0.2
    backend._reuse_count = 0
    backend._budget_skip_count = 0
    backend._screen_light_rgb = (0.1, 0.2, 0.3)
    backend._edge_light_rgb = tuple((0.1, 0.2, 0.3) for _ in range(24))
    backend._host_input = False
    backend.poll = lambda: None

    first = backend.acquire(8)
    second = backend.acquire(8)
    assert first["glow_vulkan_image"] is slot.image.resource
    assert second["glow_vulkan_serial"] == 3
    assert first["screen_light_linear_rgb"] == (0.1, 0.2, 0.3)
    assert first["screen_light_sample_path"] == "vulkan_compute_reduction"
    assert len(first["screen_edge_light_linear_rgb"]) == 24
    assert slot.lease_count == 1

    backend.release_frame(8)
    assert slot.lease_count == 0


def test_glow_shader_and_spirv_are_checked_in_together() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (APP_ROOT / "shaders/d2s_glow_source.comp").read_text(encoding="utf-8")
    spirv = (APP_ROOT / "shaders/d2s_glow_source.spv").read_bytes()

    assert "max(base_footprint, vec2(max(params.prefilter_scale, 1.0)))" in source
    assert "srgb_to_linear" in source
    assert "vec2(output_uv.x, 1.0 - output_uv.y)" in source
    assert "uint surround_region_average;" in source
    assert "vec4 source_crop;" in source
    assert "detect_letterbox_crop" in source
    assert "vec2 grid = vec2(8.0, 6.0);" in source
    assert "float edge_band_pixels = clamp(" in source
    assert "/ 270.0)" in source
    assert "4.0, 16.0" in source
    assert "if (region.x < 0.5)" in source
    assert "region.x > grid.x - 1.5" in source
    assert "if (region.y < 0.5)" in source
    assert "region.y > grid.y - 1.5" in source
    assert "footprint.x = edge_band_pixels;" in source
    assert "footprint.y = edge_band_pixels;" in source
    assert "source_encoded_at" in source
    assert "preserve perceptual black" in source
    assert "average = srgb_to_linear(average);" in source
    assert "ScreenLightBuffer" in source
    assert "reduce_screen_light_linear" in source
    assert "GlowHistoryBuffer" in source
    assert "EdgeLightBuffer" in source
    assert "edge_light_linear[24]" in source
    assert "clockwise: top 8, right 4, bottom 8, left 4" in source
    assert "float temporal_alpha;" in source
    assert "uint write_glow;" in source
    assert "params.write_glow == 0u" in source
    assert "glow_history[history_index]" in source
    assert spirv[:4] == b"\x03\x02#\x07"
