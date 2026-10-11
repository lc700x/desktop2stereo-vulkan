from __future__ import annotations

import torch

from stereo_runtime.depth_postprocess import anti_alias_depth_global_gaussian
from stereo_runtime.depth_postprocess import _depth_gaussian_spacing
from stereo_runtime.depth_postprocess import _global_gaussian_kernel
from stereo_runtime.display_antialias import antialias_depth_map


def _depth_step() -> torch.Tensor:
    depth = torch.zeros((1, 1, 32, 48), dtype=torch.float32)
    depth[..., 24:] = 1.0
    return depth


def test_global_gaussian_softens_depth_edges_and_preserves_flat_regions() -> None:
    depth = _depth_step()

    filtered = anti_alias_depth_global_gaussian(depth)

    assert filtered.shape == depth.shape
    assert filtered.dtype == depth.dtype
    assert torch.equal(filtered[..., :18], depth[..., :18])
    assert torch.equal(filtered[..., 30:], depth[..., 30:])
    assert torch.all((filtered[..., 22:26] > 0.0) & (filtered[..., 22:26] < 1.0))
    assert (filtered[..., 23] - filtered[..., 24]).abs().mean() < 1.0


def test_depth_map_uses_shared_display_aa_switch(monkeypatch) -> None:
    depth = _depth_step()

    monkeypatch.setenv("D2S_SBS_AA", "0")
    unfiltered = antialias_depth_map(depth)
    assert torch.equal(unfiltered, depth)

    monkeypatch.setenv("D2S_SBS_AA", "1")
    filtered = antialias_depth_map(depth)
    assert torch.equal(filtered, anti_alias_depth_global_gaussian(depth))
    assert not torch.equal(filtered, depth)


def test_global_gaussian_scales_sampling_spacing_for_4k() -> None:
    assert _depth_gaussian_spacing(1080, 1920) == 1.0
    assert _depth_gaussian_spacing(1920, 3840) == 1.5


def test_global_gaussian_suppresses_4k_checkerboard_without_nonfinite_values() -> None:
    depth = (torch.arange(3840, dtype=torch.float32) % 2).view(1, 1, 1, 3840)

    filtered = anti_alias_depth_global_gaussian(depth)

    assert filtered.shape == depth.shape
    assert torch.isfinite(filtered).all()
    assert filtered.min() >= 0.0
    assert filtered.max() <= 1.0
    assert (filtered[..., 8:-8].max() - filtered[..., 8:-8].min()) < 0.1


def test_global_gaussian_kernel_is_normalized_for_4k_spacing() -> None:
    kernel = _global_gaussian_kernel("cpu", None, torch.float32, 1.5)

    assert kernel.shape == (1, 1, 7, 7)
    assert torch.isfinite(kernel).all()
    assert torch.all(kernel >= 0.0)
    assert torch.allclose(kernel.sum(), torch.tensor(1.0), atol=1e-6)
