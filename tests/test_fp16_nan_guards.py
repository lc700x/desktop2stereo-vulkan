"""fp16 NaN/Inf hardening tests (ported macOS guards)."""

import pytest
import torch

from stereo_runtime.depth_provider import _normalize_depth
from stereo_runtime.runtime import _pack_sbs_host_frame, _quantize_depth_for_warp

MPS = torch.device("mps") if torch.backends.mps.is_available() else None
DEV = MPS if MPS is not None else torch.device("cpu")


def _poisoned_depth():
    depth = torch.rand(1, 1, 32, 32, device=DEV)
    depth[0, 0, 3, 3] = float("nan")
    depth[0, 0, 5, 5] = float("inf")
    depth[0, 0, 7, 7] = float("-inf")
    return depth


def test_normalize_depth_sanitizes_non_finite() -> None:
    out = _normalize_depth(_poisoned_depth())
    assert torch.isfinite(out).all()
    assert bool((out >= 0.0).all() and (out <= 1.0).all())


def test_quantize_depth_for_warp_sanitizes_non_finite(monkeypatch) -> None:
    monkeypatch.setattr("stereo_runtime.runtime.sys.platform", "darwin")
    q = _quantize_depth_for_warp(_poisoned_depth())
    assert q.dtype == torch.uint8
    assert torch.isfinite(q.float()).all()


def test_pack_sbs_host_frame_sanitizes_non_finite() -> None:
    sbs = torch.rand(1, 3, 32, 32, device=DEV)
    sbs[0, 0, 2, 2] = float("nan")
    packed = _pack_sbs_host_frame(sbs)
    assert packed is not None
    host, _, _ = packed
    values = host.reshape(-1)
    assert not torch.isnan(torch.as_tensor(values)).any()


@pytest.mark.skipif(MPS is None, reason="requires MPS")
def test_fp16_pipeline_depth_stays_finite() -> None:
    # End-to-end: fp16 CoreML engine output must never leak non-finite depth.
    import os

    from stereo_runtime.providers.apple.pytorch_mps import GenericAutoDepthMpsProvider

    cache = os.environ.get("D2S_BENCH_MODELS", "")
    if not cache or not os.path.isdir(cache):
        pytest.skip("D2S_BENCH_MODELS cache not configured")
    provider = GenericAutoDepthMpsProvider(
        model_id="xingyang1/Distill-Any-Depth-Small-hf",
        model_name="xingyang1/Distill-Any-Depth-Small-hf",
        device="mps", cache_dir=cache, local_files_only=True, force_download=False,
        depth_upsample="bilinear", depth_upsample_edge_strength=0.35,
        depth_resolution=336, patch_size=14, dtype=torch.float16,
        use_coreml=True, recompile_coreml=False,
    )
    frame = torch.rand(1, 3, 336, 336, device="mps")
    result = provider.predict_profile(frame)
    torch.mps.synchronize()
    assert torch.isfinite(result.depth).all(), "fp16 CoreML depth must be finite"
