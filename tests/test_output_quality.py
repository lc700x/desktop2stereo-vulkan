from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from stereo_runtime.output_quality import (
    _downsample_pixel_coverage_srgb,
    _torch_easu,
    _torch_rcas,
    apply_output_quality,
    output_quality_requires_eye_images,
    output_mip_lod_for_config,
)
from utils.screen_resolution_policy import build_output_sampling_plan
from stereo_runtime.synthesis import StereoConfig, synthesize_stereo


def _headset_config(**overrides):
    values = {
        "output_quality_enabled": True,
        "output_headset_tier_k": 4,
        "output_rcas_sharpness": 0.5,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for EASU")
def test_triton_easu_and_rcas_preserve_a_constant_eye_image(monkeypatch) -> None:
    # Fixed-canvas presenters cap the headset upscale by default. Exercise the
    # legacy EASU/RCAS chain explicitly here so this kernel stays covered.
    monkeypatch.setenv("D2S_CAP_OUTPUT_UPSCALE", "0")
    image = torch.full((1, 3, 8, 12), 0.25, device="cuda")

    left, right, debug = apply_output_quality(image, image, _headset_config())
    torch.cuda.synchronize()

    assert tuple(left.shape) == (1, 3, 16, 24)
    assert torch.equal(left, right)
    assert torch.allclose(left, torch.full_like(left, 0.25))
    assert debug["output_quality_backend"] == "triton_easu+triton_rcas"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for EASU")
def test_triton_quality_matches_the_device_independent_reference() -> None:
    from stereo_runtime.output_quality_triton import apply_rcas, easu_resize

    torch.manual_seed(7)
    image = torch.rand((1, 3, 7, 11), device="cuda")
    expected_easu = _torch_easu(image, 14, 22)
    actual_easu = easu_resize(image, 14, 22)
    expected_rcas = _torch_rcas(expected_easu, 0.5)
    actual_rcas = apply_rcas(actual_easu, 0.5)
    torch.cuda.synchronize()

    assert torch.allclose(actual_easu, expected_easu, atol=1e-4, rtol=1e-4)
    assert torch.allclose(actual_rcas, expected_rcas, atol=1e-4, rtol=1e-4)


def test_common_quality_runs_before_half_sbs_packing(monkeypatch) -> None:
    monkeypatch.setenv("D2S_CAP_OUTPUT_UPSCALE", "0")
    rgb = torch.rand(1, 3, 8, 12)
    depth = torch.linspace(0.0, 1.0, 12).view(1, 1, 1, 12).expand(1, 1, 8, 12)
    config = StereoConfig(
        backend="fast",
        output_format="half_sbs",
        temporal=False,
        output_quality_enabled=True,
        output_headset_tier_k=4,
        output_rcas_sharpness=0.0,
    )

    result = synthesize_stereo(rgb, depth, config)

    assert tuple(result.left_eye.shape[-2:]) == (16, 24)
    assert tuple(result.sbs.shape[-2:]) == (16, 24)
    assert result.debug_info["output_quality_mode"] == "upscale_easu"
    assert result.debug_info["output_quality_backend"] == "torch_easu_reference"


def test_forced_mip_lod_filters_native_headset_output_before_packing(monkeypatch) -> None:
    monkeypatch.setenv("D2S_CAP_OUTPUT_UPSCALE", "0")
    checker = (torch.arange(16).view(1, 1, 1, 16) % 2).float().expand(1, 3, 8, 16)
    baseline, _, _ = apply_output_quality(
        checker,
        checker,
        _headset_config(output_rcas_sharpness=0.0),
    )
    config = _headset_config(
        output_min_lod=2.0,
        output_max_lod=2.0,
        output_mip_lod_bias=-0.7,
        output_rcas_sharpness=0.0,
    )

    filtered, _, debug = apply_output_quality(checker, checker, config)

    assert tuple(filtered.shape) == tuple(baseline.shape)
    assert filtered.var() < baseline.var()
    assert debug["output_quality_mip_lod"] == 2.0
    assert debug["output_quality_applied"] == 1
    assert debug["output_quality_backend"].startswith("torch_trilinear_mip+")
    assert output_quality_requires_eye_images(config, 16, 8) is True


@pytest.mark.parametrize("scale", ["100%", "85%", "75%", "50%"])
def test_cap_keeps_quality_scale_sources_eligible_for_fused_direct_sbs(monkeypatch, scale) -> None:
    """Quality Scale must reduce GPU work, not disable the fused warp path."""
    from stereo_runtime.output_quality import (
        output_quality_requires_eye_images,
        output_sampling_plan_for_config,
        output_upscale_capped,
    )
    from stereo_runtime.render_size import RenderSizeConfig, resolve_render_size

    monkeypatch.delenv("D2S_CAP_OUTPUT_UPSCALE", raising=False)
    assert output_upscale_capped() is True
    size = resolve_render_size((3840, 2160), RenderSizeConfig(scale_factor=scale))
    config = _headset_config(
        output_min_lod=0.0,
        output_max_lod=0.35,
        output_mip_lod_bias=-0.35,
    )
    plan = output_sampling_plan_for_config(config, *size)
    assert plan is not None
    assert plan.mode == "native_mip"
    assert (plan.target_width, plan.target_height) == size
    assert output_quality_requires_eye_images(config, *size) is False


def test_native_4k_plan_is_identical_with_and_without_the_cap(monkeypatch) -> None:
    """At 100% the cap is provably a no-op.

    An earlier benchmark showed 100% as ~0.5 ms slower with the cap on, but
    that was first-case warm-up in a fresh process: the resolved plan and
    fused-path eligibility are identical for both cap states.
    """
    from stereo_runtime.output_quality import (
        output_quality_requires_eye_images,
        output_sampling_plan_for_config,
    )
    from stereo_runtime.render_size import RenderSizeConfig, resolve_render_size

    size = resolve_render_size((3840, 2160), RenderSizeConfig(scale_factor="100%"))
    config = _headset_config(
        output_min_lod=0.0,
        output_max_lod=0.35,
        output_mip_lod_bias=-0.35,
    )
    snapshots = []
    for cap in ("1", "0"):
        monkeypatch.setenv("D2S_CAP_OUTPUT_UPSCALE", cap)
        plan = output_sampling_plan_for_config(config, *size)
        snapshots.append(
            (
                plan.mode,
                plan.target_width,
                plan.target_height,
                plan.scale,
                plan.target_kind,
                output_quality_requires_eye_images(config, *size),
            )
        )
    assert snapshots[0] == snapshots[1] == ("native_mip", *size, 1.0, "headset", False)


def test_legacy_headset_upscale_would_disable_the_fused_path(monkeypatch) -> None:
    from stereo_runtime.output_quality import (
        output_quality_requires_eye_images,
        output_sampling_plan_for_config,
    )
    from stereo_runtime.render_size import RenderSizeConfig, resolve_render_size

    monkeypatch.setenv("D2S_CAP_OUTPUT_UPSCALE", "0")
    size = resolve_render_size((3840, 2160), RenderSizeConfig(scale_factor="85%"))
    config = _headset_config()
    plan = output_sampling_plan_for_config(config, *size)
    assert plan is not None
    assert plan.mode == "upscale_easu"
    assert (plan.target_width, plan.target_height) == (6528, 3672)
    assert output_quality_requires_eye_images(config, *size) is True


def test_pixel_coverage_downsample_avoids_ringing_and_sharpens_at_final_size(monkeypatch) -> None:
    import stereo_runtime.output_quality as output_quality

    plan = SimpleNamespace(
        mode="downsample_lanczos_rcas",
        target_width=4,
        target_height=2,
        scale=0.5,
        target_kind="test",
    )
    monkeypatch.setattr(
        output_quality, "output_sampling_plan_for_config", lambda *_args: plan
    )
    sharpness = []

    def record_rcas(image, value):
        sharpness.append((tuple(image.shape), value))
        return image

    monkeypatch.setattr(output_quality, "_torch_rcas", record_rcas)
    image = torch.zeros((1, 3, 4, 8))
    image[..., 1::2] = 1.0
    expected = 0.735356983

    reduced, right, debug = apply_output_quality(
        image,
        image,
        _headset_config(
            output_min_lod=0.0,
            output_max_lod=0.0,
            output_mip_lod_bias=0.0,
            output_rcas_sharpness=0.5,
        ),
    )

    assert tuple(reduced.shape) == (1, 3, 2, 4)
    assert torch.equal(reduced, right)
    assert torch.allclose(reduced, torch.full_like(reduced, expected), atol=1e-5)
    assert sharpness == [((1, 3, 2, 4), 0.5), ((1, 3, 2, 4), 0.5)]
    assert debug["output_quality_backend"] == (
        "torch_pixel_coverage_srgb+torch_rcas"
    )
    assert tuple(_downsample_pixel_coverage_srgb(image, 4, 2).shape) == (1, 3, 2, 4)


def test_max_lod_changes_shared_output_without_raising_min_lod() -> None:
    plan = build_output_sampling_plan(3840, 2160, headset_tier_k=4)

    default_lod = output_mip_lod_for_config(
        _headset_config(output_min_lod=0.0, output_max_lod=0.35, output_mip_lod_bias=-0.35),
        plan,
        3840,
        2160,
    )
    raised_lod = output_mip_lod_for_config(
        _headset_config(output_min_lod=0.0, output_max_lod=0.40, output_mip_lod_bias=-0.35),
        plan,
        3840,
        2160,
    )
    strong_lod = output_mip_lod_for_config(
        _headset_config(output_min_lod=0.0, output_max_lod=2.0, output_mip_lod_bias=-0.7),
        plan,
        3840,
        2160,
    )

    assert default_lod == 0.0
    assert raised_lod == pytest.approx(0.05)
    assert strong_lod == pytest.approx(1.3)


@pytest.mark.parametrize("ratio", [2, 3, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("transpose", [False, True])
def test_fused_srgb_area_preserves_original_filter_on_gpu(ratio, dtype, transpose):
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm GPU unavailable")
    from stereo_runtime.output import _horizontal_area_reduce
    from stereo_runtime.output_quality_triton import horizontal_area_srgb

    image = torch.rand((1, 3, 24, 36), device="cuda", dtype=dtype) * 1.2 - 0.1
    if transpose:
        image = image.transpose(-2, -1)
    width = image.shape[-1] // ratio
    rgb = image.float().clamp(0.0, 1.0)
    linear = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055).clamp_min(0).pow(2.4))
    reduced = _horizontal_area_reduce(linear, width).clamp(0.0, 1.0)
    expected = torch.where(reduced <= 0.0031308, reduced * 12.92,
                           1.055 * reduced.pow(1.0 / 2.4) - 0.055).clamp(0.0, 1.0).to(dtype)
    actual = horizontal_area_srgb(image, width)
    assert actual is not None
    torch.testing.assert_close(actual, expected, atol=1e-3 if dtype == torch.float16 else 3e-6, rtol=1e-5)


def test_fused_srgb_area_preserves_cpu_and_noninteger_fallback():
    from stereo_runtime.output import downsample_horizontal_area_srgb

    image = torch.rand((1, 4, 5, 11))
    assert downsample_horizontal_area_srgb(image, 7).shape == (1, 4, 5, 7)
    with pytest.raises(ValueError):
        downsample_horizontal_area_srgb(image, 0)


def test_fused_srgb_area_compiler_failure_retains_torch_reference(monkeypatch):
    pytest.importorskip("triton", reason="Triton is optional on macOS")
    from triton.errors import TritonError
    import stereo_runtime.output_quality_triton as kernels
    from stereo_runtime.output import downsample_horizontal_area_srgb

    class FailedKernel:
        def __getitem__(self, _grid):
            def compile_failure(*_args):
                raise TritonError("compiler unavailable")
            return compile_failure
    monkeypatch.setattr(kernels, "can_use_output_quality_triton", lambda _image: True)
    monkeypatch.setattr(kernels, "_horizontal_area_srgb_kernel", FailedKernel())
    image = torch.full((1, 3, 4, 8), 0.5)
    assert kernels.horizontal_area_srgb(image, 4) is None
    torch.testing.assert_close(downsample_horizontal_area_srgb(image, 4), image[..., :4])
