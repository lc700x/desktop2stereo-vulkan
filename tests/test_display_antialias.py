from __future__ import annotations

import torch
import pytest

from stereo_runtime.depth_postprocess import anti_alias_depth
from stereo_runtime.depth_postprocess import anti_alias_depth_global_gaussian
from stereo_runtime.depth_postprocess import _depth_gaussian_spacing
from stereo_runtime.depth_postprocess import _global_gaussian_kernel
from stereo_runtime.depth_postprocess import model_depth_aa_applied, postprocess_depth
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


def test_model_aa_skips_duplicate_output_resolution_gaussian() -> None:
    depth = _depth_step()

    with model_depth_aa_applied(True):
        filtered = postprocess_depth(depth, global_gaussian=True)

    assert torch.equal(filtered, depth)


def test_edge_aware_global_gaussian_does_not_create_intermediate_depths() -> None:
    depth = _depth_step()

    filtered = anti_alias_depth_global_gaussian(depth, edge_threshold=0.04)

    assert torch.equal(filtered, depth)
    assert torch.isfinite(filtered).all()


@pytest.mark.parametrize("shape", ((12, 25), (6, 3840)))
@pytest.mark.parametrize(
    "device",
    (
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/ROCm device unavailable"),
        ),
    ),
)
def test_final_gaussian_softens_silhouette_with_model_edge_protection_enabled(shape, device) -> None:
    depth = torch.zeros((1, 1, *shape), dtype=torch.float32)
    boundary = shape[-1] // 2
    depth[..., boundary:] = 1.0
    reference = anti_alias_depth_global_gaussian(depth)

    actual = postprocess_depth(
        depth.to(device),
        global_gaussian=True,
        edge_threshold=0.04,
    ).cpu()

    assert torch.isfinite(actual).all()
    assert torch.allclose(actual, reference, atol=2e-5, rtol=2e-5)
    transition = actual[..., boundary - 1 : boundary + 1]
    assert torch.all((transition > 0.0) & (transition < 1.0))
    assert torch.equal(actual[..., :boundary - 4], depth[..., :boundary - 4])


def test_edge_aware_global_gaussian_still_smooths_within_a_depth_surface() -> None:
    noise = torch.tensor((0.0, 0.01, -0.01, 0.02, -0.02), dtype=torch.float32)
    depth = (0.5 + noise.repeat(16, 10)).view(1, 1, 16, 50)

    filtered = anti_alias_depth_global_gaussian(depth, edge_threshold=0.04)

    assert torch.isfinite(filtered).all()
    assert filtered.var() < depth.var()


def test_edge_aware_gaussian_keeps_fractional_spacing_samples_on_their_depth_side() -> None:
    depth = torch.zeros((1, 1, 8, 3840), dtype=torch.float32)
    depth[..., 1920:] = 1.0

    filtered = anti_alias_depth_global_gaussian(depth, edge_threshold=0.04)

    assert torch.equal(filtered, depth)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/ROCm device unavailable")
@pytest.mark.parametrize("shape", ((12, 25), (6, 3840)))
def test_edge_aware_global_gaussian_gpu_matches_torch_reference(shape) -> None:
    depth = torch.rand((1, 1, *shape), dtype=torch.float32)
    depth[..., shape[-1] // 2 :] = (depth[..., shape[-1] // 2 :] > 0.5).float()
    reference = anti_alias_depth_global_gaussian(depth, edge_threshold=0.04)

    actual = anti_alias_depth_global_gaussian(
        depth.cuda(),
        edge_threshold=0.04,
    ).cpu()

    assert torch.isfinite(actual).all()
    assert torch.allclose(actual, reference, atol=2e-5, rtol=2e-5)


def test_model_scale_edge_aware_gaussian_preserves_step_and_smooths_flat_region() -> None:
    depth = _depth_step()
    noise = torch.tensor((0.0, 0.01, -0.01, 0.02, -0.02), dtype=torch.float32)
    smooth_input = (0.5 + noise.repeat(32, 5)[:, :24]).unsqueeze(0).unsqueeze(0)

    filtered_step = anti_alias_depth(
        depth,
        1.5,
        edge_threshold=0.04,
    )
    filtered_flat = anti_alias_depth(
        smooth_input,
        1.5,
        edge_threshold=0.04,
    )

    assert torch.equal(filtered_step, depth)
    assert torch.isfinite(filtered_flat).all()
    assert filtered_flat.var() < smooth_input.var()


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
