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
def test_triton_easu_and_rcas_preserve_a_constant_eye_image() -> None:
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


def test_common_quality_runs_before_half_sbs_packing() -> None:
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


def test_forced_mip_lod_filters_native_headset_output_before_packing() -> None:
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
