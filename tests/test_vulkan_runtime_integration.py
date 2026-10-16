from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest

from path_config import APP_ROOT
import torch

import stereo_runtime.runtime as runtime_module
from stereo_runtime import StereoRuntime, StereoRuntimeConfig
from stereo_runtime.depth_provider import DepthProfileResult
from stereo_runtime.openxr_render import OpenXRRenderConfig
from stereo_runtime.synthesis import StereoResult


class _Provider:
    def load(self):
        return None

    def predict_profile(self, rgb):
        b, _, height, width = rgb.shape
        depth = torch.linspace(0.0, 1.0, width, dtype=torch.float32).view(1, 1, 1, width)
        depth = depth.expand(b, 1, height, width).contiguous()
        return DepthProfileResult(depth=depth, preprocess_ms=0.0, model_ms=0.0, postprocess_ms=0.0)

    def close(self):
        return None


class _FakeVulkanBackend:
    def submit_frame(self, rgb, depth, *, params):
        self.last_fused_params = params
        mask = torch.zeros_like(depth)
        return rgb, rgb.clone(), mask, {
            "vulkan_device": "fake-vulkan",
            "vulkan_fused_backend": "vulkan_stereo_fused",
            "vulkan_readback": "host_visible_storage_buffer",
        }

    def submit_layered_frame(self, rgb, depth, *, params):
        del params
        mask = torch.zeros_like(depth)
        return rgb, rgb.clone(), mask, {
            "vulkan_device": "fake-vulkan",
            "vulkan_fused_backend": "vulkan_stereo_layered",
            "vulkan_readback": "host_visible_storage_buffer",
        }

    def close(self):
        return None


def test_openxr_prewarp_is_enabled_by_default_for_presenter_owned_output(monkeypatch):
    monkeypatch.delenv("D2S_OPENXR_PREWARP_EYES", raising=False)

    assert runtime_module._openxr_prewarp_eyes_enabled() is True


def test_openxr_triton_prewarp_reuses_full_synthesis_with_configured_hole_fill(monkeypatch):
    monkeypatch.setenv("D2S_OPENXR_RUNTIME_OUTPUT_UINT8", "0")
    config = StereoRuntimeConfig(
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        stereo_quality="quality_4k",
        stereo_compute_backend="triton",
        temporal=False,
        hole_fill_mode="quality",
        hole_fill_radius=3,
        hole_fill_strength=1.0,
    )
    runtime = StereoRuntime(config, depth_provider=_Provider(), collect_memory_stats=False)
    monkeypatch.setattr(runtime, "_resolve_stereo_compute_backend", lambda _rgb: "cuda_triton")
    runtime._resolved_stereo_compute_backend = "cuda_triton"

    calls = []

    def fake_synthesize(rgb, depth, synthesis_config, temporal_state=None):
        calls.append((depth, synthesis_config, temporal_state))
        return StereoResult(
            left_eye=torch.full_like(rgb, 0.25),
            right_eye=torch.full_like(rgb, 0.75),
            sbs=torch.empty(0),
            debug_info={
                "backend": "quality_4k",
                "occlusion_mask_backend": "triton_occlusion",
                "hole_fill_backend": "triton_directional_content_aware_radius3",
                "hole_fill_mode": synthesis_config.hole_fill_mode,
                "hole_fill_radius": synthesis_config.hole_fill_radius,
                "hole_fill_strength": synthesis_config.hole_fill_strength,
            },
        )

    monkeypatch.setattr(runtime_module, "synthesize_stereo", fake_synthesize)
    monkeypatch.setattr(
        runtime_module,
        "render_openxr_stereo",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("grid-sample fallback used")),
    )

    result = runtime.process_openxr_frame(
        torch.rand(1, 3, 8, 12),
        OpenXRRenderConfig(output_mode="full_synthesis_eyes"),
    )

    assert len(calls) == 1
    _, synthesis_config, temporal_state = calls[0]
    assert synthesis_config.output_format == "mono"
    assert synthesis_config.hole_fill_mode == "quality"
    assert synthesis_config.hole_fill_radius == 3
    assert synthesis_config.hole_fill_strength == 1.0
    assert temporal_state is runtime.temporal_state
    assert torch.all(result.left_eye == 0.25)
    assert torch.all(result.right_eye == 0.75)
    assert result.debug_info["stereo_compute_backend"] == "cuda_triton"
    assert result.debug_info["hole_fill_backend"] == "triton_directional_content_aware_radius3"
    assert result.debug_info["openxr_prewarp_backend"] == "triton_full_synthesis_eyes"
    assert result.debug_info["openxr_grid_sample_fallback"] == 0
    assert result.timing["synthesis_ms"] > 0.0
    runtime.close()


def test_openxr_triton_no_fill_uses_fused_rgba_u8_output(monkeypatch):
    config = StereoRuntimeConfig(
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        stereo_quality="quality_4k",
        stereo_compute_backend="triton",
        temporal=False,
        hole_fill_mode="none",
    )
    runtime = StereoRuntime(config, depth_provider=_Provider(), collect_memory_stats=False)
    monkeypatch.setattr(runtime, "_resolve_stereo_compute_backend", lambda _rgb: "cuda_triton")
    runtime._resolved_stereo_compute_backend = "cuda_triton"

    synthesis_left = torch.full((8, 12, 4), 25, dtype=torch.uint8)
    synthesis_right = torch.full((8, 12, 4), 75, dtype=torch.uint8)
    calls = []

    def fake_no_fill(rgb, depth, synthesis_config, cuda_events):
        calls.append((rgb, depth, synthesis_config, cuda_events))
        return (
            synthesis_left,
            synthesis_right,
            {
                "backend": "quality_4k",
                "warp_composite_backend": "triton_warp_composite2_rgba_u8",
                "occlusion_mask_backend": "skipped_no_consumer",
                "hole_fill_backend": "none",
            },
        ), "used"

    monkeypatch.setattr(runtime_module, "_try_openxr_no_fill_fused_rgba_u8", fake_no_fill)
    monkeypatch.setattr(
        runtime_module,
        "synthesize_stereo",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("canonical synthesis used")),
    )
    monkeypatch.setattr(
        runtime_module,
        "render_openxr_stereo",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("grid-sample fallback used")),
    )

    result = runtime.process_openxr_frame(
        torch.rand(1, 3, 8, 12),
        OpenXRRenderConfig(output_mode="full_synthesis_eyes"),
    )

    assert len(calls) == 1
    assert result.left_eye is synthesis_left
    assert result.right_eye is synthesis_right
    assert result.left_eye.dtype == torch.uint8
    assert result.debug_info["sbs_backend"] == "openxr_triton_no_fill_fused_rgba_u8"
    assert result.debug_info["openxr_prewarp_backend"] == "triton_no_fill_fused_rgba_u8"
    assert result.debug_info["openxr_no_fill_fused_reason"] == "used"
    assert result.debug_info["openxr_grid_sample_fallback"] == 0
    assert result.debug_info["occlusion_mask_backend"] == "skipped_no_consumer"
    assert result.debug_info["hole_fill_backend"] == "none"
    assert result.debug_info["runtime_output_pack_backend"] == "triton_warp_composite2_rgba_u8"
    runtime.close()


def test_presenter_owned_vulkan_compute_declares_cuda_external_input_path():
    source = (
        APP_ROOT
        / "stereo_runtime/vulkan_backend.py"
    ).read_text(encoding="utf-8")
    assert "VulkanExportableBuffer" in source
    assert "copy_tensor_to_buffer" in source
    assert 'input_mode = "cuda_external_buffer"' in source
    assert 'input_mode = "host_visible_buffer"' in source
    assert "wait_semaphore=wait_semaphore" in source
    assert '"vulkan_output_image_direct": True' in source
    assert '"vulkan_zero_cpu_readback": True' in source


def test_local_gpu_viewer_uses_common_external_buffer_path_for_cuda_and_rocm():
    source = (
        APP_ROOT
        / "viewer"
        / "vulkan_local_viewer.py"
    ).read_text(encoding="utf-8")

    assert "VulkanExportableBuffer" in source
    assert "copy_tensor_to_buffer" in source
    assert "vkCmdCopyBufferToImage" in source
    assert '"ROCm" if is_rocm else "CUDA"' in source
    assert "VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT" in source
    assert "memory_handle_type=rocm_handle_type" in source
    assert "single_packed_blit" in source


def test_vulkan_output_shader_decodes_srgb_before_unorm_store():
    shader = (
        APP_ROOT
        / "shaders"
        / "d2s_stereo_layered_output.comp"
    ).read_text(encoding="utf-8")

    assert "display-referred sRGB" in shader
    assert "pow((encoded + 0.055) / 1.055" in shader
    assert "output image is an UNORM storage image" in shader
    assert "float mask = screen_edge_suppressed" not in shader
    assert "if (found >= 1.0) return 1.0;" in shader


def test_vulkan_stereo_shaders_clamp_displaced_edges_instead_of_reflecting():
    shader_root = APP_ROOT / "shaders"
    for name in (
        "d2s_stereo_fused.comp",
        "d2s_stereo_layered.comp",
        "d2s_stereo_layered_output.comp",
        "d2s_stereo_layered_tiled.comp",
    ):
        shader = (shader_root / name).read_text(encoding="utf-8")
        assert "float clamp_coordinate" in shader
        assert "reflect_coordinate" not in shader


def test_vulkan_msdf_quad_shader_is_a_gpu_atlas_to_storage_image_pass():
    shader = (
        APP_ROOT
        / "shaders"
        / "d2s_msdf_quad.comp"
    ).read_text(encoding="utf-8")
    assert "readonly uniform image2D msdf_atlas" in shader
    assert "writeonly uniform image2D quad_output" in shader
    assert "median3(sampleMsdf" in shader
    assert "imageStore(" in shader
    assert "srgbToLinear" in shader


def test_vulkan_msdf_quad_request_keeps_one_quad_texture_contract():
    from viewer.vulkan_msdf_quad import VulkanMsdfQuadRequest

    request = VulkanMsdfQuadRequest(
        width=512,
        height=78,
        runs=({"text": "Size", "x": 10.0, "y": 18.0, "scale": 0.58},),
    )
    assert request.shape == (78, 512, 4)


def test_vulkan_openxr_output_shader_uses_legacy_openxr_eye_order():
    shader = (
        APP_ROOT
        / "shaders"
        / "d2s_stereo_layered_output.comp"
    ).read_text(encoding="utf-8")

    assert "fill_eye(ix, iy, -1.0, fill_mask)" in shader
    assert "params.symmetric != 0u ? 1.0 : 0.9" in shader


def test_vulkan_generic_sbs_shader_keeps_synthesis_eye_order():
    shader = (
        APP_ROOT
        / "shaders"
        / "d2s_stereo_layered.comp"
    ).read_text(encoding="utf-8")

    assert "fill_eye(ix, iy, 1.0" in shader
    assert "params.symmetric != 0u ? -1.0 : -0.9" in shader


def test_vulkan_tiled_reference_shader_keeps_layered_pass_abi():
    shader = (
        APP_ROOT
        / "shaders"
        / "d2s_stereo_layered_tiled.comp"
    ).read_text(encoding="utf-8")
    backend_source = (
        APP_ROOT
        / "stereo_runtime"
        / "vulkan_backend.py"
    ).read_text(encoding="utf-8")

    assert "layout(local_size_x = 16, local_size_y = 16" in shader
    assert "shared float shared_depth" in shader
    assert "layout(set = 0, binding = 4, std430) writeonly buffer MaskBuffer" in shader
    assert "layered_shader_path" in backend_source
    assert "d2s_stereo_layered.spv" in backend_source


def test_vulkan_fallback_openxr_output_normalizes_generic_eye_order():
    source = (
        APP_ROOT
        / "stereo_runtime"
        / "runtime.py"
    ).read_text(encoding="utf-8")

    assert "left_eye = vulkan_stereo.left_eye" in source
    assert "right_eye = vulkan_stereo.right_eye" in source


def test_openxr_depth_temporal_is_disabled_with_stereo_temporal(monkeypatch):
    monkeypatch.setenv("D2S_OPENXR_RGB_DEPTH_TEMPORAL_ALPHA", "0.9")
    config = StereoRuntimeConfig(
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        stereo_compute_backend="vulkan",
        temporal=False,
    )
    runtime = StereoRuntime(config, depth_provider=_Provider(), collect_memory_stats=False)
    first = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    second = torch.ones((1, 1, 2, 2), dtype=torch.float32)

    assert torch.equal(
        runtime._stabilize_openxr_rgb_depth(first, enabled=False), first
    )
    assert torch.equal(
        runtime._stabilize_openxr_rgb_depth(second, enabled=False), second
    )
    runtime.close()


def test_openxr_depth_temporal_blends_only_color_stable_pixels(monkeypatch):
    monkeypatch.setenv("D2S_OPENXR_RGB_DEPTH_TEMPORAL_ALPHA", "0.9")
    config = StereoRuntimeConfig(
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        stereo_compute_backend="vulkan",
        temporal=True,
    )
    runtime = StereoRuntime(config, depth_provider=_Provider(), collect_memory_stats=False)
    first_depth = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    first_rgb = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    moved_rgb = first_rgb.clone()
    moved_rgb[..., 0, 0] = 1.0
    current_depth = torch.ones_like(first_depth)

    runtime._stabilize_openxr_rgb_depth(first_depth, rgb=first_rgb)
    result = runtime._stabilize_openxr_rgb_depth(current_depth, rgb=moved_rgb)

    assert result[0, 0, 0, 0] == 1.0
    assert torch.allclose(result[0, 0, 1, 1], torch.tensor(0.1))
    assert torch.isfinite(result).all()
    runtime.close()


def test_fused_depth_temporal_respects_disabled_setting(monkeypatch):
    monkeypatch.setenv("D2S_DEPTH_TEMPORAL", "1")
    config = StereoRuntimeConfig(
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        stereo_compute_backend="vulkan",
        temporal=False,
    )
    runtime = StereoRuntime(config, depth_provider=_Provider(), collect_memory_stats=False)
    runtime._fused_depth_prev = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    runtime._fused_rgb_prev = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    depth = torch.ones((1, 1, 2, 2), dtype=torch.float32)

    result = runtime._temporal_stabilize_depth(
        depth,
        torch.zeros((1, 3, 2, 2), dtype=torch.float32),
        enabled=False,
    )

    assert torch.equal(result, depth)
    assert runtime._fused_depth_prev is None
    assert runtime._fused_rgb_prev is None
    runtime.close()


@pytest.mark.parametrize("depth_strength", (0.0, 1.75))
def test_openxr_deferred_vulkan_request_uses_runtime_depth_strength(monkeypatch, depth_strength):
    monkeypatch.setattr(
        runtime_module,
        "probe_triton_runtime",
        lambda device=None: SimpleNamespace(available=True, vendor="nvidia"),
    )
    config = StereoRuntimeConfig(
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        stereo_quality="quality_4k",
        stereo_compute_backend="vulkan",
        temporal=False,
        foreground_shift_scale=1.15,
        midground_shift_scale=1.05,
        background_shift_scale=1.05,
    )
    runtime = StereoRuntime(config, depth_provider=_Provider(), collect_memory_stats=False)

    result = runtime.process_openxr_frame(
        torch.rand(1, 3, 8, 12),
        OpenXRRenderConfig(depth_strength=depth_strength),
    )

    assert result.vulkan_compute_request is not None
    assert result.debug_info["vulkan_zero_copy_deferred"] == 1
    assert result.debug_info["stereo_compute_backend"] == "vulkan"
    assert result.debug_info["vulkan_zero_copy_reason"] == "ready"
    assert result.vulkan_compute_request.params.foreground_scale == 1.15
    assert result.vulkan_compute_request.params.midground_scale == 1.05
    assert result.vulkan_compute_request.params.background_scale == 1.05
    assert result.vulkan_compute_request.params.depth_strength == depth_strength
    runtime.close()


@pytest.mark.parametrize(
    ("hole_fill_mode", "fill_radius", "fill_strength", "expected_mode", "expected_backend"),
    (
        (
            "quality",
            3,
            1.0,
            1,
            "vulkan_directional_content_aware_radius3",
        ),
        ("none", 0, 0.0, 2, "none"),
    ),
)
def test_openxr_deferred_vulkan_request_preserves_hole_fill_mode(
    monkeypatch,
    hole_fill_mode,
    fill_radius,
    fill_strength,
    expected_mode,
    expected_backend,
):
    monkeypatch.setattr(
        runtime_module,
        "probe_triton_runtime",
        lambda device=None: SimpleNamespace(available=False, vendor="unsupported"),
    )
    config = StereoRuntimeConfig(
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        stereo_quality="quality_4k",
        stereo_compute_backend="vulkan",
        temporal=False,
        hole_fill_mode=hole_fill_mode,
        hole_fill_radius=fill_radius,
        hole_fill_strength=fill_strength,
    )
    runtime = StereoRuntime(config, depth_provider=_Provider(), collect_memory_stats=False)

    result = runtime.process_openxr_frame(
        torch.rand(1, 3, 8, 12),
        OpenXRRenderConfig(output_mode="full_synthesis_eyes"),
    )

    assert result.vulkan_compute_request is not None
    assert result.vulkan_compute_request.params.hole_fill_mode == expected_mode
    assert result.vulkan_compute_request.params.fill_radius == fill_radius
    assert result.vulkan_compute_request.params.fill_strength == fill_strength
    assert result.debug_info["vulkan_hole_fill_mode"] == expected_mode
    assert result.debug_info["hole_fill_backend"] == expected_backend
    runtime.close()


def test_runtime_routes_fast_plus_to_vulkan_fused_backend(monkeypatch):
    monkeypatch.setattr(
        runtime_module,
        "probe_triton_runtime",
        lambda device=None: SimpleNamespace(available=False, vendor="unknown"),
    )
    config = StereoRuntimeConfig(
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        stereo_quality="fast_plus",
        stereo_compute_backend="vulkan",
        output_format="half_sbs",
        temporal=False,
    )
    runtime = StereoRuntime(config, depth_provider=_Provider(), collect_memory_stats=False)
    runtime._vulkan_stereo_backend = _FakeVulkanBackend()

    result = runtime.process_rgb_frame(torch.rand(1, 3, 8, 12))

    assert result.debug_info["stereo_compute_backend"] == "vulkan"
    assert result.debug_info["vulkan_fused_backend"] == "vulkan_stereo_fused"
    assert result.debug_info["vulkan_device"] == "fake-vulkan"
    assert result.left_eye.shape == (1, 3, 8, 12)
    assert result.right_eye.shape == (1, 3, 8, 12)
    runtime.close()


def test_intel_network_mode_defers_supported_vulkan_stereo_request(monkeypatch):
    monkeypatch.setenv("D2S_INTEL_VULKAN_SBS", "1")
    config = StereoRuntimeConfig(
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        stereo_quality="fast_plus",
        stereo_compute_backend="vulkan",
        output_format="half_sbs",
        temporal=False,
    )
    runtime = StereoRuntime(config, depth_provider=_Provider(), collect_memory_stats=False)

    result = runtime.process_rgb_frame(torch.rand(1, 3, 8, 12))

    assert result.vulkan_compute_request is not None
    assert result.vulkan_compute_request.rgb.shape == (1, 3, 8, 12)
    assert result.vulkan_compute_request.depth.shape == (1, 1, 8, 12)
    assert result.debug_info["vulkan_zero_copy_request"] == 1
    assert result.debug_info["sbs_backend"] == "vulkan_deferred_intel_network"
    runtime.close()


@pytest.mark.parametrize(
    ("hole_fill_mode", "expected_mode", "expected_radius", "expected_strength", "expected_backend"),
    (
        ("balanced", 0, 1, 0.6, "vulkan_balanced"),
        ("quality", 1, 3, 1.0, "vulkan_directional_content_aware_radius3"),
        ("none", 2, 0, 0.0, "none"),
    ),
)
def test_fast_plus_vulkan_fused_preserves_hole_fill_mode(
    monkeypatch,
    hole_fill_mode,
    expected_mode,
    expected_radius,
    expected_strength,
    expected_backend,
):
    monkeypatch.setattr(
        runtime_module,
        "probe_triton_runtime",
        lambda device=None: SimpleNamespace(available=False, vendor="unsupported"),
    )
    config = StereoRuntimeConfig(
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        stereo_quality="fast_plus",
        stereo_compute_backend="vulkan",
        output_format="half_sbs",
        temporal=False,
        hole_fill_mode=hole_fill_mode,
        hole_fill_radius=1,
        hole_fill_strength=0.6,
    )
    runtime = StereoRuntime(config, depth_provider=_Provider(), collect_memory_stats=False)
    runtime._vulkan_stereo_backend = _FakeVulkanBackend()

    result = runtime.process_rgb_frame(torch.rand(1, 3, 8, 12))
    params = runtime._vulkan_stereo_backend.last_fused_params

    assert params.hole_fill_mode == expected_mode
    assert params.fill_radius == expected_radius
    assert params.fill_strength == expected_strength
    assert result.debug_info["vulkan_hole_fill_mode"] == expected_mode
    assert result.debug_info["hole_fill_backend"] == expected_backend
    runtime.close()


def test_runtime_auto_routes_to_vulkan_when_triton_probe_fails(monkeypatch):
    monkeypatch.setattr(
        runtime_module,
        "probe_triton_runtime",
        lambda device=None: SimpleNamespace(available=False, vendor="unsupported"),
    )
    config = StereoRuntimeConfig(
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        stereo_quality="fast_plus",
        stereo_compute_backend="auto",
        temporal=False,
    )
    runtime = StereoRuntime(config, depth_provider=_Provider(), collect_memory_stats=False)
    runtime._vulkan_stereo_backend = _FakeVulkanBackend()

    result = runtime.process_rgb_frame(torch.rand(1, 3, 8, 12))

    assert runtime._resolved_stereo_compute_backend == "vulkan"
    assert result.debug_info["stereo_compute_backend"] == "vulkan"
    runtime.close()


def test_openxr_prewarp_path_prefers_deferred_request_after_vulkan_backend_init(monkeypatch):
    monkeypatch.setattr(
        runtime_module,
        "probe_triton_runtime",
        lambda device=None: SimpleNamespace(available=False, vendor="unsupported"),
    )
    config = StereoRuntimeConfig(
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        stereo_quality="fast_plus",
        stereo_compute_backend="vulkan",
        temporal=False,
    )
    runtime = StereoRuntime(config, depth_provider=_Provider(), collect_memory_stats=False)
    runtime._vulkan_stereo_backend = _FakeVulkanBackend()

    result = runtime.process_openxr_frame(
        torch.rand(1, 3, 8, 12),
        OpenXRRenderConfig(output_mode="full_synthesis_eyes"),
    )

    assert result.vulkan_compute_request is not None
    assert result.debug_info["vulkan_zero_copy_request"] == 1
    assert result.debug_info["stereo_compute_backend"] == "vulkan"
    assert result.output_format == "openxr_eye_views"
    runtime.close()


def test_openxr_prewarp_vulkan_defers_to_presenter_zero_copy_request(monkeypatch):
    monkeypatch.setattr(
        runtime_module,
        "probe_triton_runtime",
        lambda device=None: SimpleNamespace(available=False, vendor="unsupported"),
    )
    config = StereoRuntimeConfig(
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        stereo_quality="quality_4k",
        stereo_compute_backend="vulkan",
        temporal=False,
    )
    runtime = StereoRuntime(config, depth_provider=_Provider(), collect_memory_stats=False)

    result = runtime.process_openxr_frame(
        torch.rand(1, 3, 8, 12),
        OpenXRRenderConfig(output_mode="full_synthesis_eyes"),
    )

    assert result.vulkan_compute_request is not None
    assert result.debug_info["vulkan_zero_copy_request"] == 1
    assert result.vulkan_compute_request.rgb.shape == (1, 3, 8, 12)
    assert result.vulkan_compute_request.depth.shape == (1, 1, 8, 12)
    runtime.close()


def test_runtime_routes_quality_4k_to_vulkan_layered_backend(monkeypatch):
    monkeypatch.setattr(
        runtime_module,
        "probe_triton_runtime",
        lambda device=None: SimpleNamespace(available=False, vendor="unsupported"),
    )
    config = StereoRuntimeConfig(
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        stereo_quality="quality_4k",
        stereo_compute_backend="vulkan",
        output_format="half_sbs",
        temporal=False,
    )
    runtime = StereoRuntime(config, depth_provider=_Provider(), collect_memory_stats=False)
    runtime._vulkan_stereo_backend = _FakeVulkanBackend()

    result = runtime.process_rgb_frame(torch.rand(1, 3, 8, 12))

    assert result.debug_info["stereo_compute_backend"] == "vulkan"
    assert result.debug_info["vulkan_fused_backend"] == "vulkan_stereo_layered"
    assert result.debug_info["sbs_backend"] == "vulkan_layered_stereo"
    runtime.close()


def test_macos_advanced_stream_uses_canonical_stereo_synthesis(monkeypatch):
    monkeypatch.setattr(runtime_module.sys, "platform", "darwin")
    monkeypatch.setenv("D2S_MAC_STREAM_CANONICAL_SYNTHESIS", "1")
    monkeypatch.delenv("D2S_INTEL_VULKAN_SBS", raising=False)
    monkeypatch.delenv("D2S_RUNTIME_OUTPUT_UINT8", raising=False)

    config = StereoRuntimeConfig(
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        stereo_quality="quality_4k",
        stereo_compute_backend="auto",
        output_format="half_sbs",
        temporal=False,
    )
    runtime = StereoRuntime(config, depth_provider=_Provider(), collect_memory_stats=False)
    vulkan_calls = []
    synthesis_calls = []

    def fake_vulkan(*args, **kwargs):
        del kwargs
        vulkan_calls.append(args)
        rgb = args[0]
        return (
            StereoResult(
                left_eye=rgb,
                right_eye=rgb,
                sbs=rgb,
                debug_info={"sbs_backend": "vulkan_layered_stereo"},
            ),
            "used",
        )

    def fake_synthesize(rgb, depth, synthesis_config, temporal_state=None, sbs_only=False):
        del depth, temporal_state, sbs_only
        synthesis_calls.append(synthesis_config)
        return StereoResult(
            left_eye=rgb,
            right_eye=rgb,
            sbs=rgb,
            debug_info={"sbs_backend": "canonical_synthesis"},
        )

    monkeypatch.setattr(runtime, "_try_vulkan_fused_stereo", fake_vulkan)
    monkeypatch.setattr(runtime_module, "synthesize_stereo", fake_synthesize)

    result = runtime.process_rgb_frame(torch.rand(1, 3, 8, 12))

    assert not vulkan_calls
    assert len(synthesis_calls) == 1
    assert result.debug_info["sbs_backend"] == "canonical_synthesis"
    assert result.debug_info["macos_stream_canonical_synthesis"] == 1
    runtime.close()
