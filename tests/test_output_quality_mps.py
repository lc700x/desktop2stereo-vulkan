"""MPS output-quality fallback tests (ported macOS behavior)."""

import os

import pytest
import torch
import torch.nn.functional as F
from types import SimpleNamespace

from stereo_runtime.output_quality import (
    _mps_rcas,
    _mps_upscale,
    apply_output_quality,
    output_sampling_plan_for_config,
)
from stereo_runtime.output import downsample_horizontal_area_srgb, make_sbs, sbs_backend
from stereo_runtime.output_mps import make_half_sbs_srgb

MPS = torch.device("mps") if torch.backends.mps.is_available() else None

pytestmark = pytest.mark.skipif(MPS is None, reason="requires MPS")


def test_mps_upscale_matches_target_size() -> None:
    image = torch.rand(1, 3, 540, 960, device=MPS)
    upscaled = _mps_upscale(image, 1080, 1920)
    assert upscaled.shape == (1, 3, 1080, 1920)
    assert upscaled.dtype == torch.float32
    assert bool((upscaled >= 0.0).all() and (upscaled <= 1.0).all())


def test_mps_rcas_preserves_shape_and_clamps() -> None:
    image = torch.rand(1, 3, 64, 64, device=MPS)
    sharpened = _mps_rcas(image, 0.5)
    assert sharpened.shape == image.shape
    assert bool((sharpened >= 0.0).all() and (sharpened <= 1.0).all())


def test_upscale_easu_falls_back_to_mps_bicubic_on_mps() -> None:
    left = torch.rand(1, 3, 540, 960, device=MPS)
    right = torch.rand(1, 3, 540, 960, device=MPS)
    cfg = SimpleNamespace(
        output_quality_enabled=True, output_rcas_sharpness=0.0, output_headset_tier_k=4
    )
    _, _, info = apply_output_quality(left, right, cfg)
    assert "mps_bicubic" in info["output_quality_backend"]


def test_d2s_cap_output_upscale_keeps_native_resolution(monkeypatch) -> None:
    monkeypatch.setenv("D2S_CAP_OUTPUT_UPSCALE", "1")
    cfg = SimpleNamespace(output_quality_enabled=True, output_headset_tier_k=4)
    plan = output_sampling_plan_for_config(cfg, 1920, 1080)
    assert plan is not None
    assert plan.mode == "native_mip"
    assert (plan.target_width, plan.target_height) == (1920, 1080)


def test_mps_half_sbs_srgb_matches_reference_and_is_independent_of_depth_aa(monkeypatch) -> None:
    torch.manual_seed(17)
    left = torch.rand(1, 3, 24, 40, device=MPS)
    right = torch.rand_like(left)
    reference = torch.cat(
        (downsample_horizontal_area_srgb(left), downsample_horizontal_area_srgb(right)),
        dim=-1,
    )

    filtered_path = make_half_sbs_srgb(left, right)
    monkeypatch.setenv("D2S_SBS_AA", "0")
    without_depth_aa = make_sbs(left, right, "half_sbs")
    monkeypatch.setenv("D2S_SBS_AA", "1")
    with_depth_aa = make_sbs(left, right, "half_sbs")
    torch.mps.synchronize()

    assert torch.allclose(filtered_path, reference, atol=1e-6, rtol=1e-6)
    assert torch.equal(without_depth_aa, with_depth_aa)
    assert sbs_backend(left, right, "half_sbs") == "mps_half_sbs_srgb"
