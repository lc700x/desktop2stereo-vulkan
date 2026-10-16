import sys
from pathlib import Path

from path_config import APP_ROOT

import pytest
import torch
import torch.nn.functional as F

import stereo_runtime.synthesis as synthesis_module

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from stereo_runtime.hole_fill import (
    box_blur,
    directional_edge_aware_fill,
    directional_edge_aware_fill_backend,
    edge_aware_fill,
)
from stereo_runtime.baseline_shift import ShiftParams, compute_shift_px, make_base_grid, warp_horizontal
from stereo_runtime.depth_upsample import upsample_depth
from stereo_runtime.layers import composite_layers, depth_edges, make_depth_layers
from stereo_runtime.occlusion import make_occlusion_mask, suppress_screen_edge_mask
from stereo_runtime.output import OUTPUT_FORMAT_CHOICES, make_sbs, match_depth, sbs_backend
from stereo_runtime.openxr_render import OpenXRRenderConfig
from stereo_runtime.synthesis import StereoConfig, _try_fused_warp_composite2, synthesize_stereo
from stereo_runtime.temporal import TemporalState, _scene_sample


def test_triton_occlusion_hot_path_has_no_cpu_scalar_sync():
    source = (APP_ROOT / "stereo_runtime" / "occlusion_triton.py").read_text(encoding="utf-8")
    assert ".item()" not in source


def test_scene_sample_uses_bounded_spatial_sample():
    rgb = torch.rand(1, 3, 216, 384)
    sample = _scene_sample(rgb, sample_size=64)
    assert sample.shape == (1, 1, 64, 64)
    assert sample.is_contiguous()


def test_scene_reset_hot_path_has_no_cuda_cpu_sync():
    source = (APP_ROOT / "stereo_runtime" / "temporal.py").read_text(encoding="utf-8")
    assert ".item()" not in source
    assert "event.query()" not in source
    assert "pending_scene_event" not in source


def make_inputs(width=64, height=32):
    y = torch.linspace(0, 1, height)
    x = torch.linspace(0, 1, width)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    rgb = torch.stack([xx, yy, torch.ones_like(xx) * 0.5], dim=0).unsqueeze(0)
    depth = xx.unsqueeze(0).unsqueeze(0)
    return rgb, depth


def test_shift_params_use_explicit_max_disparity_budget():
    depth = torch.ones(1, 1, 1, 1)
    shift = compute_shift_px(
        depth,
        1000,
        ShiftParams(
            depth_strength=1.0,
            convergence=0.0,
            max_disparity_px=20.0,
            parallax_preset="standard",
        ),
    )
    assert torch.allclose(shift, torch.full_like(depth, -10.0))


def test_convergence_zeroes_shift_at_screen_plane():
    depth = torch.full((1, 1, 2, 2), 0.45)
    shift = compute_shift_px(depth, 3840, ShiftParams(convergence=0.45, max_disparity_px=20.0))
    assert torch.count_nonzero(shift) == 0

def test_normal_sbs_eye_order_uses_left_then_right_views(monkeypatch):
    monkeypatch.setenv("D2S_SBS_AA", "0")
    rgb, depth = make_inputs(width=64, height=32)
    config = StereoConfig(
        backend="fast",
        output_format="full_sbs",
        temporal=False,
        fused=False,
        depth_strength=2.0,
        convergence=0.0,
        max_disparity_px=6.0,
        depth_antialias_strength=0.0,
    )
    result = synthesize_stereo(rgb, depth, config)
    expected_shift = compute_shift_px(
        depth,
        rgb.shape[-1],
        ShiftParams(
            depth_strength=config.depth_strength,
            convergence=config.convergence,
            max_disparity_px=config.max_disparity_px,
            parallax_preset=config.parallax_preset,
        ),
    )
    expected_left = warp_horizontal(rgb, expected_shift, eye_sign=1.0)
    expected_right = warp_horizontal(rgb, expected_shift, eye_sign=-1.0)

    assert torch.equal(result.left_eye, expected_left)
    assert torch.equal(result.right_eye, expected_right)
    assert torch.equal(result.sbs[..., :, :64], expected_left)
    assert torch.equal(result.sbs[..., :, 64:], expected_right)


def test_layered_quality_uses_resolved_parallax_budget_fields(monkeypatch):
    monkeypatch.setenv("D2S_SBS_AA", "0")
    rgb, depth = make_inputs(width=64, height=32)
    config = StereoConfig(
        backend="quality_4k",
        layers=2,
        output_format="half_sbs",
        debug_output=True,
        temporal=False,
        fused=False,
        depth_strength=1.0,
        convergence=0.0,
        max_disparity_px=6.0,
        parallax_preset="strong",
        depth_antialias_strength=0.0,
    )

    result = synthesize_stereo(rgb, depth, config)

    expected_shift = compute_shift_px(
        depth,
        rgb.shape[-1],
        ShiftParams(
            depth_strength=config.depth_strength,
            convergence=config.convergence,
            max_disparity_px=config.max_disparity_px,
            parallax_preset=config.parallax_preset,
        ),
    )
    assert torch.allclose(result.debug_info["shift_px"], expected_shift)
    assert result.debug_info["parallax_budget_preset"] == "strong"
    assert result.debug_info["resolved_max_disparity_px"] == 6.0
    assert result.debug_info["parallax_resolver_version"] == 1


def test_synthesis_debug_records_resolved_parallax_budget_override():
    rgb, depth = make_inputs(width=64, height=32)
    result = synthesize_stereo(
        rgb,
        depth,
        StereoConfig(
            backend="fast",
            output_format="half_sbs",
            debug_output=True,
            temporal=False,
            fused=False,
            max_disparity_px=20.0,
        ),
    )

    assert result.debug_info["resolved_max_disparity_px"] == 20.0
    assert result.debug_info["parallax_budget_preset"] == "standard"


def test_synthesis_debug_keeps_runtime_contract_scalars_without_debug_output():
    rgb, depth = make_inputs(width=64, height=32)
    config = StereoConfig(
        backend="fast",
        output_format="half_sbs",
        debug_output=False,
        temporal=True,
        temporal_strength=0.35,
        convergence=0.42,
        hole_fill_mode="quality",
        hole_fill_radius=5,
        hole_fill_strength=0.8,
        edge_threshold=0.07,
        edge_dilation=3,
        mask_feather_radius=2,
        fused=False,
    )

    result = synthesize_stereo(rgb, depth, config, temporal_state=TemporalState())

    assert result.debug_info["convergence"] == 0.42
    assert result.debug_info["temporal_enabled"] == 1
    assert result.debug_info["temporal_strength"] == 0.35
    assert result.debug_info["hole_fill_mode"] == "quality"
    assert result.debug_info["hole_fill_radius"] == 5
    assert result.debug_info["hole_fill_strength"] == 0.8
    assert result.debug_info["edge_threshold"] == 0.07
    assert result.debug_info["edge_dilation"] == 3
    assert result.debug_info["mask_feather_radius"] == 2
    for key in (
        "scene_detect_ms",
        "fast_depth_postprocess_ms",
        "fast_baseline_ms",
        "fast_debug_ms",
        "temporal_ms",
        "output_depth_ms",
        "sbs_backend_ms",
        "make_sbs_ms",
        "synthesis_total_ms",
        "synthesis_accounted_ms",
        "synthesis_unaccounted_ms",
    ):
        assert isinstance(result.debug_info[key], float)
        assert result.debug_info[key] >= 0.0
    assert "shift_px" not in result.debug_info


@pytest.mark.parametrize("backend", ["fast", "fast_plus", "quality_4k", "hq_4k"])
def test_zero_max_disparity_bypasses_all_binocular_difference(backend):
    rgb, depth = make_inputs(width=64, height=32)
    config = StereoConfig(
        backend=backend,
        output_format="full_sbs",
        temporal=False,
        fused=False,
        depth_strength=10.0,
        convergence=0.0,
        max_disparity_px=0.0,
        parallax_preset="standard",
        debug_output=True,
    )

    result = synthesize_stereo(rgb, depth, config)

    assert torch.count_nonzero(result.debug_info["shift_px"]) == 0
    assert torch.allclose(result.left_eye, rgb)
    assert torch.allclose(result.right_eye, rgb)
    assert torch.allclose(result.left_eye, result.right_eye)


def test_half_sbs_shape():
    rgb, depth = make_inputs()
    result = synthesize_stereo(rgb, depth, StereoConfig(backend="fast", output_format="half_sbs"))
    assert result.left_eye.shape == rgb.shape
    assert result.right_eye.shape == rgb.shape
    assert result.sbs.shape == rgb.shape



def test_fast_plus_adds_light_occlusion_fill_debug_info():
    rgb, depth = make_inputs()
    rgb = rgb.clone()
    depth = depth.clone()
    rgb[:, :, :, 28:36] = 1.0
    depth[:, :, :, 32:] = 1.0
    depth[:, :, :, :32] = 0.0
    fast = synthesize_stereo(rgb, depth, StereoConfig(backend="fast", output_format="half_sbs", temporal=False))
    result = synthesize_stereo(
        rgb,
        depth,
        StereoConfig(backend="fast_plus", output_format="half_sbs", debug_output=True, temporal=False, fused=False),
    )

    assert result.sbs.shape == rgb.shape
    assert result.debug_info["backend"] == "fast_plus"
    assert result.debug_info["fast_plus_edge_threshold"] == 0.03
    assert result.debug_info["fast_plus_edge_dilation"] == 1
    assert result.debug_info["fast_plus_hole_fill_radius"] == 1
    assert result.debug_info["fast_plus_hole_fill_strength"] == 0.60
    assert "occlusion_mask" in result.debug_info
    assert result.debug_info["hole_fill_backend"] == "torch_directional_content_aware"
    assert not torch.equal(result.sbs, fast.sbs)


def test_fast_plus_hole_fill_off_skips_fill_algorithms(monkeypatch):
    rgb, depth = make_inputs()

    def unexpected_fill(*args, **kwargs):
        raise AssertionError("hole fill must not run when disabled")

    def unexpected_occlusion(*args, **kwargs):
        raise AssertionError("occlusion mask must not run without a consumer")

    monkeypatch.setattr(synthesis_module, "directional_edge_aware_fill_backend", unexpected_fill)
    monkeypatch.setattr(synthesis_module, "directional_edge_aware_fill", unexpected_fill)
    monkeypatch.setattr(synthesis_module, "occlusion_backend", unexpected_occlusion)
    monkeypatch.setattr(synthesis_module, "make_occlusion_mask", unexpected_occlusion)

    result = synthesize_stereo(
        rgb,
        depth,
        StereoConfig(
            backend="fast_plus",
            output_format="half_sbs",
            temporal=False,
            fused=False,
            hole_fill="none",
            hole_fill_mode="none",
        ),
    )

    assert result.debug_info["hole_fill_backend"] == "none"
    assert result.debug_info["occlusion_mask_backend"] == "skipped_no_consumer"
    assert result.debug_info["hole_fill_mode"] == "none"
    assert result.debug_info["hole_fill_radius"] == 0
    assert result.debug_info["hole_fill_strength"] == 0.0
    assert result.debug_info["fast_plus_hole_fill_radius"] == 0
    assert result.debug_info["fast_plus_hole_fill_strength"] == 0.0


def test_layered_hole_fill_mode_controls_radius_and_strength():
    rgb, depth = make_inputs()
    result = synthesize_stereo(
        rgb,
        depth,
        StereoConfig(
            backend="quality_4k",
            output_format="half_sbs",
            debug_output=True,
            temporal=False,
            fused=False,
            hole_fill_mode="balanced",
            hole_fill_radius=1,
            hole_fill_strength=0.6,
        ),
    )

    assert result.debug_info["hole_fill_mode"] == "balanced"
    assert result.debug_info["hole_fill_radius"] == 1
    assert result.debug_info["hole_fill_strength"] == 0.6


def test_layered_balanced_hole_fill_uses_fast_edge_aware_path():
    rgb, depth = make_inputs()
    result = synthesize_stereo(
        rgb,
        depth,
        StereoConfig(
            backend="quality_4k",
            output_format="half_sbs",
            debug_output=True,
            temporal=False,
            fused=False,
            hole_fill_mode="balanced",
        ),
    )

    assert result.debug_info["hole_fill_mode"] == "balanced"
    assert result.debug_info["hole_fill_backend"] == "torch_avg_pool"


def test_layered_quality_hole_fill_uses_directional_content_aware_path():
    rgb, depth = make_inputs()
    result = synthesize_stereo(
        rgb,
        depth,
        StereoConfig(
            backend="quality_4k",
            output_format="half_sbs",
            debug_output=True,
            temporal=False,
            fused=False,
            hole_fill_mode="quality",
        ),
    )

    assert result.debug_info["hole_fill_mode"] == "quality"
    assert result.debug_info["hole_fill_backend"] == "torch_directional_content_aware"


def test_layered_hole_fill_off_skips_all_fill_algorithms(monkeypatch):
    rgb, depth = make_inputs()

    def unexpected_fill(*args, **kwargs):
        raise AssertionError("hole fill must not run when disabled")

    def unexpected_occlusion(*args, **kwargs):
        raise AssertionError("occlusion mask must not run without a consumer")

    monkeypatch.setattr(synthesis_module, "edge_aware_fill_backend", unexpected_fill)
    monkeypatch.setattr(synthesis_module, "edge_aware_fill", unexpected_fill)
    monkeypatch.setattr(synthesis_module, "directional_edge_aware_fill_backend", unexpected_fill)
    monkeypatch.setattr(synthesis_module, "directional_edge_aware_fill", unexpected_fill)
    monkeypatch.setattr(synthesis_module, "occlusion_backend", unexpected_occlusion)
    monkeypatch.setattr(synthesis_module, "make_occlusion_mask", unexpected_occlusion)

    result = synthesize_stereo(
        rgb,
        depth,
        StereoConfig(
            backend="quality_4k",
            output_format="half_sbs",
            temporal=False,
            fused=False,
            hole_fill="none",
            hole_fill_mode="none",
        ),
    )

    assert result.debug_info["hole_fill_backend"] == "none"
    assert result.debug_info["occlusion_mask_backend"] == "skipped_no_consumer"
    assert result.debug_info["hole_fill_mode"] == "none"
    assert result.debug_info["hole_fill_radius"] == 0
    assert result.debug_info["hole_fill_strength"] == 0.0


def test_triton_directional_content_aware_matches_torch_reference_when_available():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(23)
    image = torch.rand(2, 3, 31, 47, device="cuda")
    mask = torch.rand(1, 1, 31, 47, device="cuda")
    depth = torch.rand(1, 1, 31, 47, device="cuda")
    shift_px = (depth - 0.5) * 24.0

    expected = directional_edge_aware_fill(
        image,
        mask,
        depth,
        shift_px,
        radius=3,
        strength=1.0,
        mask_feather_radius=3,
        fused=False,
    )
    actual = directional_edge_aware_fill(
        image,
        mask,
        depth,
        shift_px,
        radius=3,
        strength=1.0,
        mask_feather_radius=3,
        fused=True,
    )

    assert directional_edge_aware_fill_backend(
        image,
        mask,
        depth,
        shift_px,
        radius=3,
        mask_feather_radius=3,
        fused=True,
    ) == "triton_directional_content_aware_radius3"
    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_mask_feather_radius_softens_hole_fill_blend():
    image = torch.zeros(1, 3, 9, 9)
    image[:, :, :, 5:] = 1.0
    mask = torch.zeros(1, 1, 9, 9)
    mask[:, :, 4, 4] = 1.0

    hard = edge_aware_fill(image, mask, radius=1, strength=1.0, fused=False, mask_feather_radius=0)
    soft = edge_aware_fill(image, mask, radius=1, strength=1.0, fused=False, mask_feather_radius=2)

    assert not torch.equal(hard, soft)
    hard_changed = ((hard - image).abs() > 1e-6).sum()
    soft_changed = ((soft - image).abs() > 1e-6).sum()
    assert soft_changed > hard_changed


def test_directional_hole_fill_prefers_background_side_at_depth_edge():
    image = torch.zeros(1, 3, 5, 7)
    image[..., :, :3] = 0.2
    image[..., :, 3:] = 0.9
    depth = torch.zeros(1, 1, 5, 7)
    depth[..., :, :3] = 0.0
    depth[..., :, 3:] = 1.0
    mask = torch.zeros(1, 1, 5, 7)
    mask[..., :, 3:4] = 1.0

    out = directional_edge_aware_fill(
        image,
        mask,
        depth=depth,
        shift_px=None,
        radius=1,
        strength=1.0,
        mask_feather_radius=0,
        depth_edge_threshold=0.01,
    )

    assert out[..., :, 3].mean() < image[..., :, 3].mean()
    assert out[..., :, 3].mean() < 0.75


def test_directional_hole_fill_protects_high_frequency_ui_edges():
    image = torch.zeros(1, 3, 5, 7)
    image[..., :, 3] = 1.0
    depth = torch.zeros(1, 1, 5, 7)
    mask = torch.zeros(1, 1, 5, 7)
    mask[..., :, 3:4] = 1.0

    protected = directional_edge_aware_fill(
        image,
        mask,
        depth=depth,
        shift_px=None,
        radius=1,
        strength=1.0,
        mask_feather_radius=0,
        depth_edge_threshold=0.01,
    )
    legacy = edge_aware_fill(image, mask, radius=1, strength=1.0, fused=False, mask_feather_radius=0)

    assert protected[..., :, 3].mean() > legacy[..., :, 3].mean()

def test_full_sbs_shape():
    rgb, depth = make_inputs()
    result = synthesize_stereo(rgb, depth, StereoConfig(backend="fast", output_format="full_sbs"))
    assert result.sbs.shape == (1, 3, 32, 128)


def test_tab_mono_and_depth_map_shapes(monkeypatch):
    monkeypatch.setenv("D2S_SBS_AA", "0")
    rgb, depth = make_inputs()
    half_tab = synthesize_stereo(rgb, depth, StereoConfig(backend="fast", output_format="half_tab"))
    full_tab = synthesize_stereo(rgb, depth, StereoConfig(backend="fast", output_format="full_tab"))
    mono = synthesize_stereo(rgb, depth, StereoConfig(backend="fast", output_format="mono"))
    depth_map = synthesize_stereo(
        rgb, depth, StereoConfig(backend="fast", output_format="depth_map", depth_antialias_strength=0.0)
    )
    assert half_tab.sbs.shape == rgb.shape
    assert full_tab.sbs.shape == (1, 3, 64, 64)
    assert mono.sbs.shape == rgb.shape
    assert depth_map.sbs.shape == rgb.shape
    assert torch.equal(mono.sbs, mono.left_eye)
    assert torch.equal(depth_map.sbs[:, 0:1], depth)


def test_non_debug_mono_skips_unused_output_depth_postprocess(monkeypatch: pytest.MonkeyPatch):
    import stereo_runtime.synthesis as synthesis_module

    calls = 0
    original = synthesis_module.postprocess_depth

    def wrapped_postprocess_depth(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(synthesis_module, "postprocess_depth", wrapped_postprocess_depth)
    rgb, depth = make_inputs()

    synthesize_stereo(
        rgb,
        depth,
        StereoConfig(backend="quality_4k", output_format="mono", debug_output=False, temporal=False),
    )

    assert calls == 1


def test_composite_display_output_semantics():
    left = torch.zeros(1, 3, 4, 6)
    right = torch.ones(1, 3, 4, 6)

    anaglyph = make_sbs(left, right, "anaglyph", fused=False)
    assert torch.equal(anaglyph[:, 0:1], left[:, 0:1])
    assert torch.equal(anaglyph[:, 1:], right[:, 1:])

    interleaved = make_sbs(left, right, "interleaved", fused=False)
    assert torch.equal(interleaved[..., 0::2, :], left[..., 0::2, :])
    assert torch.equal(interleaved[..., 1::2, :], right[..., 1::2, :])

    leia = make_sbs(left, right, "leia", fused=False)
    assert torch.equal(leia[..., :, 0::2], left[..., :, 0::2])
    assert torch.equal(leia[..., :, 1::2], right[..., :, 1::2])


def test_half_sbs_fallback_uses_nonnegative_area_downsample():
    left = torch.arange(1 * 1 * 2 * 4, dtype=torch.float32).view(1, 1, 2, 4)
    right = left + 100.0
    actual = make_sbs(left, right, "half_sbs", fused=False)
    expected = torch.cat(((left[..., 0::2] + left[..., 1::2]) * 0.5,
                          (right[..., 0::2] + right[..., 1::2]) * 0.5), dim=-1)
    assert torch.equal(actual, expected)


def test_guided_depth_upsample_preserves_shape_range_and_uses_rgb_edges():
    depth = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]], dtype=torch.float32)
    rgb = torch.zeros(1, 3, 4, 4)
    rgb[..., :, 2:] = 1.0

    bilinear = upsample_depth(depth, 4, 4, rgb=rgb, mode="bilinear")
    guided = upsample_depth(depth, 4, 4, rgb=rgb, mode="guided", edge_strength=1.0)

    assert guided.shape == (1, 1, 4, 4)
    assert guided.amin() >= 0
    assert guided.amax() <= 1
    assert not torch.equal(guided, bilinear)


def test_anaglyph_methods_have_stable_defaults():
    left = torch.zeros(1, 3, 4, 6)
    right = torch.ones(1, 3, 4, 6)

    red_cyan = make_sbs(left, right, "anaglyph", fused=False)
    green_magenta = make_sbs(left, right, "anaglyph", fused=False, anaglyph_method="green_magenta")
    amber_blue = make_sbs(left, right, "anaglyph", fused=False, anaglyph_method="amber_blue")
    gray = make_sbs(left, right, "anaglyph", fused=False, anaglyph_method="gray")

    assert torch.equal(red_cyan[:, 0:1], left[:, 0:1])
    assert torch.equal(red_cyan[:, 1:], right[:, 1:])
    assert torch.equal(green_magenta[:, 0:1], right[:, 0:1])
    assert torch.equal(green_magenta[:, 1:2], left[:, 1:2])
    assert torch.equal(green_magenta[:, 2:3], right[:, 2:3])
    assert torch.equal(amber_blue[:, 0:2], left[:, 0:2])
    assert torch.equal(amber_blue[:, 2:3], right[:, 2:3])
    assert torch.equal(gray[:, 0:1], left.mean(dim=1, keepdim=True))
    assert torch.equal(gray[:, 1:], right.mean(dim=1, keepdim=True).expand(-1, 2, -1, -1))


def test_cross_eyed_swaps_output_eyes():
    rgb, depth = make_inputs(width=64, height=32)
    normal = synthesize_stereo(rgb, depth, StereoConfig(backend="fast", output_format="full_sbs", temporal=False))
    crossed = synthesize_stereo(
        rgb,
        depth,
        StereoConfig(backend="fast", output_format="full_sbs", temporal=False, cross_eyed=True, debug_output=True),
    )

    assert torch.equal(crossed.left_eye, normal.right_eye)
    assert torch.equal(crossed.right_eye, normal.left_eye)
    assert torch.equal(crossed.sbs[..., :, :64], normal.right_eye)
    assert torch.equal(crossed.sbs[..., :, 64:], normal.left_eye)
    assert crossed.debug_info["cross_eyed"] == 1


@pytest.mark.parametrize(
    ("width", "height"),
    [
        (1280, 720),
        (1920, 1080),
        (720, 1280),
        (641, 359),
    ],
)
def test_output_formats_support_non_4k_resolutions(width: int, height: int):
    left = torch.zeros(1, 3, height, width)
    right = torch.ones(1, 3, height, width)
    depth = torch.linspace(0, 1, width).view(1, 1, 1, width).expand(1, 1, height, width)

    expected_shapes = {
        "half_sbs": (1, 3, height, width),
        "full_sbs": (1, 3, height, width * 2),
        "half_tab": (1, 3, height, width),
        "full_tab": (1, 3, height * 2, width),
        "mono": (1, 3, height, width),
        "depth_map": (1, 3, height, width),
        "anaglyph": (1, 3, height, width),
        "interleaved": (1, 3, height, width),
        "leia": (1, 3, height, width),
    }

    for output_format in OUTPUT_FORMAT_CHOICES:
        kwargs = {"depth": depth} if output_format == "depth_map" else {}
        actual = make_sbs(left, right, output_format, fused=False, **kwargs)
        assert actual.shape == expected_shapes[output_format]


def test_fast_synthesis_supports_odd_and_portrait_resolutions():
    for width, height in ((641, 359), (360, 641)):
        rgb, depth = make_inputs(width=width, height=height)
        for output_format in OUTPUT_FORMAT_CHOICES:
            result = synthesize_stereo(rgb, depth, StereoConfig(backend="fast", output_format=output_format))
            if output_format == "full_sbs":
                assert result.sbs.shape == (1, 3, height, width * 2)
            elif output_format == "full_tab":
                assert result.sbs.shape == (1, 3, height * 2, width)
            else:
                assert result.sbs.shape == rgb.shape


def test_quality_depth_map_uses_matched_output_depth(monkeypatch):
    monkeypatch.setenv("D2S_SBS_AA", "0")
    rgb, depth = make_inputs(width=64, height=32)
    low_res_depth = F.interpolate(depth, size=(16, 32), mode="bilinear", align_corners=False)
    result = synthesize_stereo(
        rgb,
        low_res_depth,
        StereoConfig(
            backend="quality_4k", layers=2, output_format="depth_map", debug_output=True,
            temporal=False, depth_antialias_strength=0.0,
        ),
    )
    expected = match_depth(low_res_depth, rgb.shape[-2], rgb.shape[-1], rgb=rgb, edge_aware=True)
    assert result.sbs.shape == rgb.shape
    assert torch.equal(result.debug_info["output_depth"], expected)
    assert torch.equal(result.sbs[:, 0:1], expected)


def test_disabled_depth_postprocess_preserves_output_depth():
    rgb, depth = make_inputs(width=64, height=32)
    result = synthesize_stereo(
        rgb,
        depth,
        StereoConfig(
            backend="quality_4k", layers=2, output_format="depth_map", debug_output=True,
            temporal=False, depth_antialias_strength=0.0,
        ),
    )
    assert torch.equal(result.debug_info["output_depth"], depth)


def test_depth_pop_and_depth_antialias_affect_output_depth():
    rgb, depth = make_inputs(width=64, height=32)
    scaled = synthesize_stereo(
        rgb,
        depth,
        StereoConfig(
            backend="quality_4k",
            layers=2,
            output_format="depth_map",
            debug_output=True,
            temporal=False,
            depth_pop=0.5,
        ),
    )
    assert not torch.equal(scaled.debug_info["output_depth"], depth)

    hard_edge = torch.zeros_like(depth)
    hard_edge[..., :, depth.shape[-1] // 2 :] = 1.0
    smoothed = synthesize_stereo(
        rgb,
        hard_edge,
        StereoConfig(
            backend="quality_4k",
            layers=2,
            output_format="depth_map",
            debug_output=True,
            temporal=False,
            depth_antialias_strength=1.0,
        ),
    )
    assert smoothed.debug_info["output_depth"].amin() >= 0
    assert smoothed.debug_info["output_depth"].amax() <= 1
    assert not torch.equal(smoothed.debug_info["output_depth"], hard_edge)


def test_edge_dilation_parameter_affects_occlusion_mask():
    rgb, depth = make_inputs(width=64, height=32)
    depth = (depth > 0.5).float()
    no_dilation = synthesize_stereo(
        rgb,
        depth,
        StereoConfig(backend="quality_4k", layers=2, debug_output=True, temporal=False, edge_dilation=0, fused=False),
    )
    dilation = synthesize_stereo(
        rgb,
        depth,
        StereoConfig(backend="quality_4k", layers=2, debug_output=True, temporal=False, edge_dilation=3, fused=False),
    )
    assert dilation.debug_info["occlusion_mask"].sum() >= no_dilation.debug_info["occlusion_mask"].sum()


def test_screen_edge_mask_suppression_clears_only_border():
    mask = torch.ones(1, 1, 8, 10)
    actual = suppress_screen_edge_mask(mask, border_px=2)

    assert actual[..., :2, :].sum() == 0
    assert actual[..., -2:, :].sum() == 0
    assert actual[..., :, :2].sum() == 0
    assert actual[..., :, -2:].sum() == 0
    assert torch.equal(actual[..., 2:-2, 2:-2], torch.ones(1, 1, 4, 6))


def test_screen_edge_mask_suppression_preserves_internal_occlusion():
    rgb, depth = make_inputs(width=64, height=32)
    depth = torch.zeros_like(depth)
    depth[..., :, :1] = 1.0
    depth[..., :, 31:33] = 1.0
    result = synthesize_stereo(
        rgb,
        depth,
        StereoConfig(
            backend="quality_4k",
            layers=2,
            debug_output=True,
            temporal=False,
            edge_dilation=1,
            screen_edge_mask_suppression=2,
            fused=False,
        ),
    )
    mask = result.debug_info["occlusion_mask"]

    assert mask[..., :, :2].sum() == 0
    assert mask[..., :, -2:].sum() == 0
    assert mask[..., :, 30:34].sum() > 0


def test_quality_debug_outputs():
    rgb, depth = make_inputs()
    config = StereoConfig(backend="quality_4k", layers=2, output_format="half_sbs", debug_output=True)
    result = synthesize_stereo(rgb, depth, config)
    assert result.sbs.shape == rgb.shape
    assert "occlusion_mask" in result.debug_info
    assert result.debug_info["occlusion_mask"].shape == (1, 1, 32, 64)


def test_hq_uses_at_least_three_layers():
    rgb, depth = make_inputs()
    result = synthesize_stereo(rgb, depth, StereoConfig(backend="hq_4k", layers=2, debug_output=True))
    assert result.debug_info["layers"] == 3


def test_temporal_state_runs_twice():
    rgb, depth = make_inputs()
    state = TemporalState()
    config = StereoConfig(backend="quality_4k", temporal=True)
    first = synthesize_stereo(rgb, depth, config, temporal_state=state)
    second = synthesize_stereo(rgb, depth, config, temporal_state=state)
    assert first.sbs.shape == second.sbs.shape


def test_triton_temporal_masked_matches_torch_when_available():
    if not torch.cuda.is_available():
        return
    from stereo_runtime.temporal_triton import apply_temporal_masked

    torch.manual_seed(41)
    left = torch.rand(1, 3, 24, 40, device="cuda")
    right = torch.rand(1, 3, 24, 40, device="cuda")
    prev_left = torch.rand_like(left)
    prev_right = torch.rand_like(right)
    mask = torch.rand(1, 1, 24, 40, device="cuda") * 0.25

    actual_left, actual_right = apply_temporal_masked(left, right, prev_left, prev_right, mask, alpha=1.0)
    expected_left = left * (1.0 - mask) + prev_left * mask
    expected_right = right * (1.0 - mask) + prev_right * mask

    assert torch.allclose(actual_left, expected_left, atol=1e-6, rtol=1e-6)
    assert torch.allclose(actual_right, expected_right, atol=1e-6, rtol=1e-6)


def test_auto_temporal_reset_is_disabled_on_cpu_but_temporal_blending_stays_enabled():
    depth = torch.zeros(1, 1, 8, 8)
    state = TemporalState()
    config = StereoConfig(
        backend="fast",
        temporal=True,
        temporal_strength=0.75,
        auto_reset_temporal=True,
        scene_reset_threshold=0.2,
        debug_output=True,
    )
    dark = torch.zeros(1, 3, 8, 8)
    bright = torch.ones(1, 3, 8, 8)

    first = synthesize_stereo(dark, depth, config, temporal_state=state)
    second = synthesize_stereo(bright, depth, config, temporal_state=state)

    assert first.debug_info["temporal_reset"] == 0
    assert second.debug_info["temporal_reset"] == 0
    assert "temporal_reset_reason" not in second.debug_info
    assert second.debug_info["temporal_enabled"] == 1
    assert state.reset_count == 0


def test_box_blur_matches_2d_kernel():
    torch.manual_seed(7)
    image = torch.rand(2, 3, 16, 18)
    radius = 3
    k = radius * 2 + 1
    weight = torch.ones(3, 1, k, k, dtype=image.dtype) / float(k * k)
    expected = F.conv2d(image, weight, padding=radius, groups=3)
    actual = box_blur(image, radius=radius)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_edge_aware_fill_cuda_triton_matches_fallback_when_available():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(17)
    image = torch.rand(2, 3, 32, 40, device="cuda")
    mask = (torch.rand(2, 1, 32, 40, device="cuda") > 0.7).float()
    expected = torch.lerp(image, box_blur(image, radius=3), mask)
    actual = edge_aware_fill(image, mask, radius=3, strength=1.0)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_depth_edges_matches_padded_gradient_formula():
    torch.manual_seed(11)
    depth = torch.rand(2, 1, 12, 14)
    threshold = 0.04
    dx = F.pad((depth[..., :, 1:] - depth[..., :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((depth[..., 1:, :] - depth[..., :-1, :]).abs(), (0, 0, 0, 1))
    expected = ((dx + dy) > threshold).float()
    actual = depth_edges(depth, threshold=threshold)
    assert torch.equal(actual, expected)


def test_composite_layers_two_layer_fast_path_matches_formula():
    torch.manual_seed(13)
    warped = [torch.rand(2, 3, 8, 10), torch.rand(2, 3, 8, 10)]
    weights = torch.rand(2, 2, 8, 10)
    expected = warped[0] * weights[:, 0:1] + warped[1] * weights[:, 1:2]
    actual = composite_layers(warped, weights)
    assert torch.equal(actual, expected)


def test_legacy_fused_warp_composite_cuda_matches_layered_torch_path_when_available():
    if not torch.cuda.is_available():
        return
    rgb, depth = make_inputs(width=40, height=24)
    rgb = rgb.cuda()
    depth = depth.cuda()
    # An externally supplied shift tensor retains the legacy layered API.
    # Budget-backed source visibility is covered by the projection tests.
    base_shift = compute_shift_px(depth, rgb.shape[-1], ShiftParams()).clone()
    weights = make_depth_layers(depth, layers=2)
    left_layers = []
    right_layers = []
    for idx in range(2):
        layer_shift = base_shift * (0.75 + 0.25 * (idx + 1) / 2)
        left_layers.append(warp_horizontal(rgb, layer_shift, eye_sign=1.0))
        right_layers.append(warp_horizontal(rgb, layer_shift, eye_sign=-1.0))
    expected_left = composite_layers(left_layers, weights)
    expected_right = composite_layers(right_layers, weights)
    actual = _try_fused_warp_composite2(rgb, depth, base_shift, layers=2, symmetric=True)
    assert actual is not None
    actual_left, actual_right = actual
    assert torch.allclose(actual_left, expected_left, atol=5e-4, rtol=1e-4)
    assert torch.allclose(actual_right, expected_right, atol=5e-4, rtol=1e-4)


def test_fused_occlusion_cuda_matches_torch_path_when_available():
    if not torch.cuda.is_available():
        return
    rgb, depth = make_inputs(width=40, height=24)
    depth = depth.cuda()
    base_shift = compute_shift_px(depth, rgb.shape[-1], ShiftParams())
    expected = make_occlusion_mask(depth, base_shift, fused=False)
    actual = make_occlusion_mask(depth, base_shift, fused=True)
    assert torch.equal(actual, expected)


def test_triton_occlusion_radius2_accepts_float_rounding_when_available():
    if not torch.cuda.is_available():
        return
    from stereo_runtime.occlusion import occlusion_backend

    depth = torch.rand(1, 1, 24, 40, device="cuda", dtype=torch.float32)
    shift_px = torch.rand_like(depth)

    backend = occlusion_backend(
        depth,
        shift_px,
        edge_threshold=0.04000000000000001,
        dilation=2,
        fused=True,
    )

    assert backend == "triton_occlusion_radius2"


def test_triton_occlusion_radius1_accepts_realtime_threshold_when_available():
    if not torch.cuda.is_available():
        return
    from stereo_runtime.occlusion import occlusion_backend

    depth = torch.rand(1, 1, 24, 40, device="cuda", dtype=torch.float32)
    shift_px = torch.rand_like(depth)

    backend = occlusion_backend(
        depth,
        shift_px,
        edge_threshold=0.04,
        dilation=1,
        fused=True,
    )

    assert backend == "triton_occlusion_radius1"


def test_lightweight_balanced_fill_uses_triton_radius1_when_available():
    if not torch.cuda.is_available():
        return
    rgb, depth = make_inputs(width=40, height=24)
    result = synthesize_stereo(
        rgb.cuda(),
        depth.cuda(),
        StereoConfig(
            backend="quality_4k",
            output_format="mono",
            debug_output=True,
            temporal=False,
            edge_dilation=1,
            edge_threshold=0.04,
            mask_feather_radius=0,
            hole_fill_mode="balanced",
            hole_fill_radius=1,
            hole_fill_strength=0.6,
        ),
    )

    assert result.debug_info["occlusion_mask_backend"] == "triton_occlusion_radius1"
    assert result.debug_info["hole_fill_backend"] == "triton_radius1"


def test_fused_half_sbs_cuda_matches_torch_path_when_available():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(19)
    left = torch.rand(1, 3, 24, 40, device="cuda")
    right = torch.rand(1, 3, 24, 40, device="cuda")
    expected = make_sbs(left, right, "half_sbs", fused=False)
    actual = make_sbs(left, right, "half_sbs", fused=True)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_fused_full_sbs_cuda_matches_torch_path_when_available():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(23)
    left = torch.rand(1, 3, 24, 40, device="cuda")
    right = torch.rand(1, 3, 24, 40, device="cuda")
    expected = make_sbs(left, right, "full_sbs", fused=False)
    actual = make_sbs(left, right, "full_sbs", fused=True)
    assert torch.equal(actual, expected)
    assert sbs_backend(left, right, "full_sbs", fused=True) == "triton_full_sbs"


def test_fused_tab_outputs_cuda_match_torch_path_when_available():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(29)
    left = torch.rand(1, 3, 24, 40, device="cuda")
    right = torch.rand(1, 3, 24, 40, device="cuda")
    expected_half = make_sbs(left, right, "half_tab", fused=False)
    actual_half = make_sbs(left, right, "half_tab", fused=True)
    expected_full = make_sbs(left, right, "full_tab", fused=False)
    actual_full = make_sbs(left, right, "full_tab", fused=True)
    assert torch.allclose(actual_half, expected_half, atol=1e-6, rtol=1e-6)
    assert torch.equal(actual_full, expected_full)
    assert sbs_backend(left, right, "half_tab", fused=True) == "triton_half_tab"
    assert sbs_backend(left, right, "full_tab", fused=True) == "triton_full_tab"


def test_fused_depth_map_cuda_matches_torch_path_when_available():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(31)
    left = torch.rand(1, 3, 24, 40, device="cuda")
    right = torch.rand(1, 3, 24, 40, device="cuda")
    depth = torch.rand(1, 1, 24, 40, device="cuda")
    expected = make_sbs(left, right, "depth_map", fused=False, depth=depth)
    actual = make_sbs(left, right, "depth_map", fused=True, depth=depth)
    assert torch.equal(actual, expected)
    assert sbs_backend(left, right, "depth_map", fused=True, depth=depth) == "triton_depth_map"


def test_fused_composite_display_outputs_cuda_match_torch_path_when_available():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(37)
    left = torch.rand(1, 3, 24, 40, device="cuda")
    right = torch.rand(1, 3, 24, 40, device="cuda")
    for output_format, backend in (
        ("anaglyph", "triton_anaglyph"),
        ("interleaved", "triton_interleaved"),
        ("leia", "triton_leia"),
    ):
        expected = make_sbs(left, right, output_format, fused=False)
        actual = make_sbs(left, right, output_format, fused=True)
        assert torch.equal(actual, expected)
        assert sbs_backend(left, right, output_format, fused=True) == backend


def test_odd_resolution_cuda_output_backends_fall_back_when_required():
    if not torch.cuda.is_available():
        return
    left = torch.rand(1, 3, 23, 39, device="cuda")
    right = torch.rand(1, 3, 23, 39, device="cuda")

    assert sbs_backend(left, right, "half_sbs", fused=True) == "torch_interpolate"
    assert sbs_backend(left, right, "half_tab", fused=True) == "torch_interpolate_vertical"
    assert sbs_backend(left, right, "full_sbs", fused=True) == "triton_full_sbs"
    assert sbs_backend(left, right, "full_tab", fused=True) == "triton_full_tab"
    assert sbs_backend(left, right, "anaglyph", fused=True) == "triton_anaglyph"
    assert sbs_backend(left, right, "interleaved", fused=True) == "triton_interleaved"
    assert sbs_backend(left, right, "leia", fused=True) == "triton_leia"


def test_depth_map_requires_depth():
    left = torch.rand(1, 3, 8, 10)
    right = torch.rand(1, 3, 8, 10)
    with pytest.raises(ValueError, match="requires depth"):
        make_sbs(left, right, "depth_map")


def test_fused_config_false_uses_torch_backends():
    rgb, depth = make_inputs(width=40, height=24)
    result = synthesize_stereo(
        rgb,
        depth,
        StereoConfig(backend="quality_4k", layers=2, debug_output=True, temporal=False, fused=False),
    )
    assert result.debug_info["warp_composite_backend"] == "torch_grid_sample"
    assert result.debug_info["occlusion_mask_backend"] == "torch_max_pool"
    assert result.debug_info["hole_fill_backend"] == "torch_avg_pool"
    assert result.debug_info["sbs_backend"] == "torch_interpolate"

    full_result = synthesize_stereo(
        rgb,
        depth,
        StereoConfig(backend="quality_4k", layers=2, output_format="full_sbs", debug_output=True, temporal=False, fused=False),
    )
    assert full_result.debug_info["sbs_backend"] == "torch_cat"

    half_tab_result = synthesize_stereo(
        rgb,
        depth,
        StereoConfig(backend="quality_4k", layers=2, output_format="half_tab", debug_output=True, temporal=False, fused=False),
    )
    assert half_tab_result.debug_info["sbs_backend"] == "torch_interpolate"

    depth_map_result = synthesize_stereo(
        rgb,
        depth,
        StereoConfig(backend="quality_4k", layers=2, output_format="depth_map", debug_output=True, temporal=False, fused=False),
    )
    assert depth_map_result.debug_info["sbs_backend"] == "torch_depth_map"

    for output_format in ("anaglyph", "interleaved", "leia"):
        format_result = synthesize_stereo(
            rgb,
            depth,
            StereoConfig(backend="quality_4k", layers=2, output_format=output_format, debug_output=True, temporal=False, fused=False),
        )
        assert format_result.debug_info["sbs_backend"] == f"torch_{output_format}"


def test_disable_triton_env_uses_torch_backends(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("STEREO_RUNTIME_DISABLE_TRITON", "1")
    rgb, depth = make_inputs(width=40, height=24)
    result = synthesize_stereo(
        rgb,
        depth,
        StereoConfig(backend="quality_4k", layers=2, debug_output=True, temporal=False),
    )
    assert result.debug_info["warp_composite_backend"] == "torch_grid_sample"
    assert result.debug_info["occlusion_mask_backend"] == "torch_max_pool"
    assert result.debug_info["hole_fill_backend"] == "torch_avg_pool"
    assert result.debug_info["sbs_backend"] == "torch_interpolate"

    full_result = synthesize_stereo(
        rgb,
        depth,
        StereoConfig(backend="quality_4k", layers=2, output_format="full_sbs", debug_output=True, temporal=False),
    )
    assert full_result.debug_info["sbs_backend"] == "torch_cat"

    half_tab_result = synthesize_stereo(
        rgb,
        depth,
        StereoConfig(backend="quality_4k", layers=2, output_format="half_tab", debug_output=True, temporal=False),
    )
    assert half_tab_result.debug_info["sbs_backend"] == "torch_interpolate"

    depth_map_result = synthesize_stereo(
        rgb,
        depth,
        StereoConfig(backend="quality_4k", layers=2, output_format="depth_map", debug_output=True, temporal=False),
    )
    assert depth_map_result.debug_info["sbs_backend"] == "torch_depth_map"

    for output_format in ("anaglyph", "interleaved", "leia"):
        format_result = synthesize_stereo(
            rgb,
            depth,
            StereoConfig(backend="quality_4k", layers=2, output_format=output_format, debug_output=True, temporal=False),
        )
        assert format_result.debug_info["sbs_backend"] == f"torch_{output_format}"


def test_warp_horizontal_matches_cached_grid_formula():
    rgb, depth = make_inputs(width=32, height=16)
    eye_sign = -1.0
    b, _, h, w = rgb.shape
    shift_px = depth * 0.75
    grid = make_base_grid(b, h, w, rgb.device, rgb.dtype).clone()
    shift_norm = (2.0 * shift_px.squeeze(1) / max(w - 1, 1)) * eye_sign
    grid[..., 0] = grid[..., 0] + shift_norm
    expected = F.grid_sample(rgb, grid, mode="bilinear", padding_mode="border", align_corners=True)
    actual = warp_horizontal(rgb, shift_px, eye_sign=eye_sign)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_warp_horizontal_clamps_out_of_range_edges_instead_of_mirroring():
    rgb = torch.arange(8, dtype=torch.float32).reshape(1, 1, 1, 8)
    shift = torch.full((1, 1, 1, 8), 3.0)

    actual = warp_horizontal(rgb, shift, eye_sign=1.0)

    # The rightmost destination samples beyond the source.  Reflection would
    # bring them back to interior values (4, 3, ...), producing a double edge;
    # border mode keeps the actual edge pixel (7) at the boundary.
    assert torch.equal(actual[..., -2:], torch.tensor([[[[7.0, 7.0]]]]))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
def test_mps_warp_horizontal_supports_border_clamping():
    rgb = torch.arange(8, dtype=torch.float32, device="mps").reshape(1, 1, 1, 8)
    shift = torch.full((1, 1, 1, 8), 3.0, device="mps")

    actual = warp_horizontal(rgb, shift, eye_sign=1.0)

    assert actual.device.type == "mps"
    assert torch.allclose(
        actual[..., -2:].cpu(),
        torch.tensor([[[[7.0, 7.0]]]]),
    )


def test_mps_source_projection_is_preferred_and_explicit_disable_is_respected(monkeypatch):
    from types import SimpleNamespace
    from stereo_runtime import _fused_warp_mps

    monkeypatch.delenv("D2S_MAC_STREAM_MPS_FUSED", raising=False)
    monkeypatch.setenv("STEREO_RUNTIME_DISABLE_TRITON", "1")
    expected = (torch.zeros(1, 3, 2, 2), torch.ones(1, 3, 2, 2))
    monkeypatch.setattr(_fused_warp_mps, "mps_warp_composite2", lambda *args: expected)
    rgb = SimpleNamespace(device=SimpleNamespace(type="mps"))
    depth = torch.zeros(1, 1, 2, 2)

    assert _try_fused_warp_composite2(rgb, depth, depth, layers=2, symmetric=True) is expected
    monkeypatch.setenv("D2S_MAC_STREAM_MPS_FUSED", "0")
    assert _try_fused_warp_composite2(rgb, depth, depth, layers=2, symmetric=True) is None


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
def test_macos_stream_mps_fused_warp_uses_common_layer_math(monkeypatch):
    monkeypatch.setenv("D2S_MAC_STREAM_MPS_FUSED", "1")
    rgb = torch.rand(1, 3, 16, 32, device="mps")
    depth = torch.linspace(0.0, 1.0, 16 * 32, device="mps").reshape(1, 1, 16, 32)
    config = StereoConfig(
        backend="quality_4k",
        output_format="full_sbs",
        layers=2,
        symmetric=True,
        hole_fill="none",
        temporal=False,
        fused=True,
        depth_strength=0.25,
    )

    result = synthesize_stereo(rgb, depth, config)

    assert result.debug_info["warp_composite_backend"] == "metal_mps_warp_composite2"
    assert torch.isfinite(result.left_eye).all().item()
    assert torch.isfinite(result.right_eye).all().item()


def test_openxr_stereo_warp_defaults_to_edge_clamping():
    assert OpenXRRenderConfig().padding_mode == "border"


def test_zero_depth_pop_skips_duplicate_depth_pop_pass(monkeypatch: pytest.MonkeyPatch):
    import stereo_runtime.depth_postprocess as depth_postprocess_module

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("zero depth_pop should skip apply_depth_pop")

    monkeypatch.setattr(depth_postprocess_module, "apply_depth_pop", fail_if_called)
    depth = torch.tensor([[[[-0.5, 0.25, 1.5]]]], dtype=torch.float32)

    actual = depth_postprocess_module.postprocess_depth(depth, depth_pop=0.0, antialias_strength=0.0)

    assert torch.equal(actual, torch.tensor([[[[0.0, 0.25, 1.0]]]], dtype=torch.float32))


def test_negative_depth_pop_uses_realtime_compression_without_pow():
    from stereo_runtime.depth_postprocess import apply_depth_pop

    depth = torch.tensor([[[[0.0, 0.25, 0.5, 0.75, 1.0]]]], dtype=torch.float32)
    actual = apply_depth_pop(depth, -0.5)
    expected = torch.tensor([[[[0.25, 0.375, 0.5, 0.625, 0.75]]]], dtype=torch.float32)
    assert torch.allclose(actual, expected)

    source = APP_ROOT / "stereo_runtime" / "depth_postprocess.py"
    code = source.read_text(encoding="utf-8")
    negative_branch = code.index("if depth_pop < 0.0:")
    pow_branch = code.index(".pow(exponent)")
    assert negative_branch < pow_branch
    assert "compressed = centered * (1.0 - strength)" in code[negative_branch:pow_branch]
