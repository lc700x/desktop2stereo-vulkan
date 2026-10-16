from __future__ import annotations

import struct
from pathlib import Path

from path_config import APP_ROOT
from types import SimpleNamespace

import pytest

import stereo_runtime.vulkan_stereo_pass as module
import stereo_runtime.vulkan_stereo_image_pass as image_module
from stereo_runtime.vulkan_stereo_pass import (
    VULKAN_HOLE_FILL_BALANCED,
    VULKAN_HOLE_FILL_NONE,
    VULKAN_HOLE_FILL_QUALITY,
    VulkanLayeredStereoParams,
    VulkanLayeredStereoPass,
    VulkanStereoFusedParams,
    VulkanStereoFusedPass,
    resolve_vulkan_hole_fill_mode,
    resolve_vulkan_hole_fill_parameters,
    vulkan_hole_fill_backend_name,
)


def test_vulkan_stereo_shaders_skip_fill_neighborhood_when_disabled():
    root = Path(__file__).resolve().parents[1]
    shader_paths = (
        APP_ROOT / "shaders" / "d2s_stereo_fused.comp",
        APP_ROOT / "shaders" / "d2s_stereo_layered.comp",
        APP_ROOT / "shaders" / "d2s_stereo_layered_tiled.comp",
        APP_ROOT / "shaders" / "d2s_stereo_layered_output.comp",
    )

    for shader_path in shader_paths:
        source = shader_path.read_text(encoding="utf-8")
        fill_function = source[source.index("vec3 fill_eye"):]
        early_return = fill_function.index("params.fill_strength <= 1.0e-5")
        neighborhood_sample = fill_function.index("box_average(")
        assert early_return < neighborhood_sample, shader_path.name

    fused_source = shader_paths[0].read_text(encoding="utf-8")
    assert "params.hole_fill_mode != HOLE_FILL_NONE" in fused_source
    assert "fill_enabled ? feathered_mask_at" in fused_source
    for shader_path in shader_paths[1:]:
        source = shader_path.read_text(encoding="utf-8")
        assert "params.hole_fill_mode != HOLE_FILL_NONE" in source
        assert "? 0.0 : feathered_mask_at" in source


def test_vulkan_layered_shaders_implement_quality_content_aware_fill():
    root = Path(__file__).resolve().parents[1]
    for name in (
        "d2s_stereo_fused.comp",
        "d2s_stereo_layered.comp",
        "d2s_stereo_layered_tiled.comp",
        "d2s_stereo_layered_output.comp",
    ):
        source = (APP_ROOT / "shaders" / name).read_text(encoding="utf-8")
        assert "const uint HOLE_FILL_NONE = 2u;" in source
        assert "abs(right_shift - left_shift) > 0.05" in source
        assert "for (int step = 1; step <= 3; ++step)" in source
        assert "directional * 0.75 + blurred * 0.25" in source
        assert "depth_protection" in source


@pytest.mark.parametrize(
    ("hole_fill", "hole_fill_mode", "expected_mode", "expected_backend"),
    (
        ("edge_aware", "balanced", VULKAN_HOLE_FILL_BALANCED, "vulkan_balanced"),
        (
            "edge_aware",
            "quality",
            VULKAN_HOLE_FILL_QUALITY,
            "vulkan_directional_content_aware_radius3",
        ),
        ("none", "quality", VULKAN_HOLE_FILL_NONE, "none"),
        ("edge_aware", "none", VULKAN_HOLE_FILL_NONE, "none"),
    ),
)
def test_vulkan_hole_fill_mode_mapping(
    hole_fill, hole_fill_mode, expected_mode, expected_backend
):
    mode = resolve_vulkan_hole_fill_mode(hole_fill, hole_fill_mode)

    assert mode == expected_mode
    assert vulkan_hole_fill_backend_name(mode) == expected_backend


@pytest.mark.parametrize(
    ("mode", "expected"),
    (
        (VULKAN_HOLE_FILL_BALANCED, (2, 0.6)),
        (VULKAN_HOLE_FILL_QUALITY, (3, 1.0)),
        (VULKAN_HOLE_FILL_NONE, (0, 0.0)),
    ),
)
def test_vulkan_hole_fill_parameters_are_mode_complete(mode, expected):
    assert resolve_vulkan_hole_fill_parameters(
        mode,
        fill_radius=2,
        fill_strength=0.6,
    ) == expected


def test_vulkan_stereo_params_match_shader_push_constant_layout():
    payload = VulkanStereoFusedParams(
        depth_strength=2.0,
        max_disparity_px=96.0,
        convergence=0.1,
        edge_threshold=0.03,
        fill_strength=0.6,
        fill_radius=1,
        mask_feather_radius=3,
        symmetric=True,
        hole_fill_mode=VULKAN_HOLE_FILL_QUALITY,
    ).pack(3840, 2160)

    assert len(payload) == VulkanStereoFusedPass.PUSH_CONSTANTS_SIZE
    unpacked = struct.unpack("<IIfffffIIII", payload)
    assert unpacked[:2] == (3840, 2160)
    assert unpacked[-4:] == (1, 3, 1, VULKAN_HOLE_FILL_QUALITY)


def test_vulkan_stereo_params_reject_invalid_dimensions_and_radius():
    with pytest.raises(ValueError, match="dimensions"):
        VulkanStereoFusedParams().pack(0, 1)
    with pytest.raises(ValueError, match="radii"):
        VulkanStereoFusedParams(fill_radius=-1).pack(1, 1)


def test_vulkan_layered_params_match_shader_push_constant_layout():
    payload = VulkanLayeredStereoParams(layers=3, hole_fill_mode=VULKAN_HOLE_FILL_NONE).pack(3840, 2160)

    assert len(payload) == VulkanLayeredStereoPass.PUSH_CONSTANTS_SIZE
    unpacked = struct.unpack("<IIfffffIIIIffffIIIII", payload)
    assert unpacked[:2] == (3840, 2160)
    assert unpacked[10] == 3
    assert unpacked[-3:] == (VULKAN_HOLE_FILL_NONE, 1, 0)
    image_payload = VulkanLayeredStereoParams().pack_image(3840, 2160, packed_output=True)
    assert len(image_payload) == 84
    assert struct.unpack("<II", image_payload[-8:]) == (1, 0)


def test_vulkan_stereo_pass_uses_bounded_descriptor_sets(monkeypatch):
    class FakeVk:
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7

    class FakePipeline:
        def __init__(self, context, shader_path, **kwargs):
            self.descriptor_set_layout = "layout"
            self.calls = []

        def record_dispatch(self, command_buffer, **kwargs):
            self.calls.append((command_buffer, kwargs))

        def close(self):
            pass

    class FakeArena:
        def __init__(self, context, budget):
            self.sets = []

        def allocate(self, layout):
            value = f"set-{len(self.sets)}"
            self.sets.append(value)
            return value

        def update_storage_buffer(self, descriptor_set, binding, buffer):
            pass

        def close(self):
            pass

    monkeypatch.setattr(module, "VulkanComputePipeline", FakePipeline)
    monkeypatch.setattr(module, "VulkanDescriptorArena", FakeArena)
    context = SimpleNamespace(vk=FakeVk(), frame_context_count=3)
    stereo_pass = VulkanStereoFusedPass(context, width=32, height=24)

    sizes = stereo_pass.buffer_sizes
    buffers = tuple(
        SimpleNamespace(context=context, size=sizes[name])
        for name in sizes
    )
    recorded = []

    def submit_on(role, record, **kwargs):
        assert role == "compute"
        record("command-buffer")
        recorded.append(kwargs)
        return 5

    context.submit_on = submit_on
    assert stereo_pass.submit(*buffers, frame_id=1, config_version=1) == 5
    assert len(stereo_pass.descriptor_sets) == 3
    assert recorded == [{}]
    assert stereo_pass.pipeline.calls[0][1]["group_count_x"] == 2
    assert stereo_pass.pipeline.calls[0][1]["group_count_y"] == 2
    assert len(stereo_pass.pipeline.calls[0][1]["push_constants"]) == 44
    stereo_pass.close()


def test_vulkan_stereo_image_pass_writes_two_storage_images(monkeypatch):
    class FakeVk:
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE = 8
        VK_IMAGE_LAYOUT_GENERAL = 9

    class FakePipeline:
        def __init__(self, context, shader_path, **kwargs):
            self.descriptor_set_layout = "layout"
            self.calls = []
            assert kwargs["push_constants_size"] == 84
            assert [item.descriptor_type for item in kwargs["descriptor_bindings"]] == [7, 7, 8, 8]

        def record_dispatch(self, command_buffer, **kwargs):
            self.calls.append((command_buffer, kwargs))

        def close(self):
            pass

    class FakeArena:
        def __init__(self, context, budget):
            self.sets = []
            assert budget.storage_buffers_per_set == 2
            assert budget.storage_images_per_set == 2

        def allocate(self, layout):
            value = f"set-{len(self.sets)}"
            self.sets.append(value)
            return value

        def update_storage_buffer(self, descriptor_set, binding, buffer):
            pass

        def update_storage_image(self, descriptor_set, binding, image):
            pass

        def close(self):
            pass

    monkeypatch.setattr(image_module, "VulkanComputePipeline", FakePipeline)
    monkeypatch.setattr(image_module, "VulkanDescriptorArena", FakeArena)
    submit_kwargs = []

    def submit_on(role, record, **kwargs):
        assert role == "compute"
        record("command-buffer")
        submit_kwargs.append(kwargs)
        return 11

    context = SimpleNamespace(
        vk=FakeVk(),
        frame_context_count=2,
        image_state=lambda image: SimpleNamespace(layout=9),
        submit_on=submit_on,
    )
    stereo_pass = image_module.VulkanStereoImagePass(context, width=32, height=24)
    sizes = stereo_pass.input_buffer_sizes
    buffers = tuple(SimpleNamespace(context=context, size=sizes[name]) for name in ("rgb", "depth"))
    images = tuple(
        SimpleNamespace(context=context, width=32, height=24, image=object())
        for _ in range(2)
    )

    assert stereo_pass.submit(
        *buffers,
        *images,
        params=VulkanLayeredStereoParams(),
        frame_id=1,
        config_version=1,
        wait_semaphore="input-ready",
        signal_semaphore="input-released",
    ) == 11
    assert submit_kwargs == [{
        "wait_semaphore": "input-ready",
        "signal_semaphore": "input-released",
    }]
    assert stereo_pass.pipeline.calls[0][1]["group_count_x"] == 2
    assert stereo_pass.pipeline.calls[0][1]["group_count_y"] == 2
    stereo_pass.close()
