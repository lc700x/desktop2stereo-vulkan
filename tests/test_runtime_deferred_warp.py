"""Deferred Metal warp runtime handoff tests (ported macOS behavior)."""

import os

import pytest
import torch

from stereo_runtime.runtime import (
    StereoRuntimeResult,
    _half_res_synth_enabled,
    _pack_sbs_host_frame,
    _viewer_host_frame_enabled,
    _warp_depth_postprocess,
)

MPS = torch.device("mps") if torch.backends.mps.is_available() else None


def test_result_carries_viewer_handoff_fields() -> None:
    result = StereoRuntimeResult(
        depth=torch.zeros(1, 1, 8, 8),
        left_eye=torch.zeros(1, 3, 8, 8),
        right_eye=torch.zeros(1, 3, 8, 8),
        sbs=torch.zeros(1, 3, 8, 8),
        viewer_rgb=None,
        viewer_depth=None,
        viewer_frame_np=None,
        viewer_bgra=None,
    )
    assert result.viewer_rgb is None
    assert result.viewer_frame_np is None
    assert result.viewer_bgra is None


def test_viewer_env_flags_default_off() -> None:
    os.environ.pop("D2S_HALF_RES_SYNTH", None)
    os.environ.pop("D2S_VIEWER_HOST_FRAME", None)
    assert _half_res_synth_enabled() is False
    assert _viewer_host_frame_enabled() is False


def test_viewer_env_flags_opt_in(monkeypatch) -> None:
    monkeypatch.setenv("D2S_HALF_RES_SYNTH", "1")
    monkeypatch.setenv("D2S_VIEWER_HOST_FRAME", "1")
    assert _half_res_synth_enabled() is True
    assert _viewer_host_frame_enabled() is True


@pytest.mark.skipif(MPS is None, reason="requires MPS")
def test_pack_sbs_host_frame_returns_hwc_rgba8() -> None:
    sbs = torch.rand(1, 3, 64, 128, device=MPS)
    packed = _pack_sbs_host_frame(sbs)
    assert packed is not None
    host, width, height = packed
    assert (width, height) == (128, 64)
    assert host.shape == (64, 128, 4)
    assert host.dtype == torch.uint8 or str(host.dtype) == "uint8"


@pytest.mark.skipif(MPS is None, reason="requires MPS")
def test_warp_depth_postprocess_noop_when_knobs_off() -> None:
    from types import SimpleNamespace

    depth = torch.rand(1, 1, 64, 64, device=MPS)
    config = SimpleNamespace(depth_pop=0.0, depth_antialias_strength=0.0)
    out = _warp_depth_postprocess(depth, config, depth)
    torch.mps.synchronize()
    assert out is depth  # zero extra passes
