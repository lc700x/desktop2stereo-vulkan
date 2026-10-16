from __future__ import annotations

import torch
import pytest

from stereo_runtime.depth_upsample import (
    filter_model_depth,
    model_depth_antialiasing,
    upsample_depth,
)


@pytest.mark.parametrize("fail_kernel", [False, True])
def test_cuda_joint_upsample_allocates_bilinear_only_for_fallback(monkeypatch, fail_kernel):
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm is unavailable")
    import stereo_runtime.triton_runtime as triton_runtime
    from stereo_runtime import _depth_upsample_triton as kernels

    depth = torch.rand(1, 1, 2, 3, device="cuda")
    rgb = torch.rand(1, 3, 8, 10, device="cuda")
    interpolate = torch.nn.functional.interpolate
    expected = interpolate(depth, size=(8, 10), mode="bilinear", align_corners=False)
    calls = []
    def track_interpolate(input, *args, **kwargs):
        calls.append((input.shape[1], kwargs.get("size")))
        return interpolate(input, *args, **kwargs)
    def kernel(*args, **kwargs):
        if fail_kernel:
            raise RuntimeError("kernel unavailable")
        return expected
    monkeypatch.setattr(triton_runtime, "triton_runtime_available", lambda device: True)
    monkeypatch.setattr(kernels, "joint_bilateral_upsample", kernel)
    monkeypatch.setattr(torch.nn.functional, "interpolate", track_interpolate)

    actual = upsample_depth(depth, 8, 10, rgb=rgb, mode="joint_bilateral")

    assert torch.equal(actual, expected)
    assert ((1, (8, 10)) in calls) is fail_kernel


@pytest.mark.parametrize(
    "device",
    (
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/ROCm unavailable"),
        ),
    ),
)
def test_model_aa_uses_separable_legacy_kernel_then_bilinear(device: str) -> None:
    depth = torch.rand((1, 1, 18, 27), device=device)
    rgb = torch.rand((1, 3, 72, 108), device=device)
    filtered = filter_model_depth(depth, rgb, 2.0)
    from stereo_runtime.depth_postprocess import anti_alias_depth

    expected_filter = anti_alias_depth(depth, 4.0)
    expected = torch.nn.functional.interpolate(
        expected_filter, size=(72, 108), mode="bilinear", align_corners=False
    )
    with model_depth_antialiasing(2.0) as state:
        actual = upsample_depth(depth, 72, 108, rgb=rgb, mode="joint_bilateral")

    assert state.applied
    assert torch.isfinite(filtered).all()
    assert torch.isfinite(actual).all()
    assert torch.allclose(filtered, expected_filter, atol=2e-6, rtol=2e-6)
    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_joint_bilateral_upsample_keeps_depth_step_on_rgb_edge() -> None:
    height, width = 64, 64
    depth = torch.full((1, 1, 8, 8), 0.1)
    depth[..., 4:] = 0.9
    rgb = torch.zeros((1, 3, height, width))
    rgb[..., :, width // 2 :] = 1.0

    sampled = upsample_depth(
        depth, height, width, rgb=rgb, mode="joint_bilateral"
    )
    profile = sampled[0, 0, height // 2]

    assert profile[width // 2 - 1] < 0.11
    assert profile[width // 2] > 0.89


def test_joint_bilateral_upsample_preserves_smooth_depth_without_rgb_edge() -> None:
    depth = torch.linspace(0.0, 1.0, 8).view(1, 1, 1, 8).expand(1, 1, 8, 8)
    rgb = torch.full((1, 3, 64, 64), 0.5)

    sampled = upsample_depth(
        depth, 64, 64, rgb=rgb, mode="joint_bilateral"
    )
    reference = torch.nn.functional.interpolate(
        depth, size=(64, 64), mode="bilinear", align_corners=False
    )

    assert torch.allclose(sampled, reference, atol=1e-5, rtol=0.0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_joint_bilateral_upsample_does_not_treat_a_depth_plane_as_an_edge(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm is unavailable")
    depth_height, depth_width = 12, 12
    low_y, low_x = torch.meshgrid(
        torch.arange(depth_height, device=device, dtype=torch.float32),
        torch.arange(depth_width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    depth = (0.2 + 0.03 * low_x + 0.025 * low_y)[None, None]
    height, width = 96, 96
    rgb = torch.zeros((1, 3, height, width), device=device)
    rgb[..., width // 2:] = 1.0

    sampled = upsample_depth(depth, height, width, rgb=rgb, mode="joint_bilateral")
    reference = torch.nn.functional.interpolate(
        depth, size=(height, width), mode="bilinear", align_corners=False
    )

    assert torch.allclose(sampled, reference, atol=1e-5, rtol=0.0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_joint_bilateral_upsample_is_continuous_near_depth_threshold(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm is unavailable")
    rgb = torch.zeros((1, 3, 32, 32), device=device)
    rgb[..., 16:] = 1.0
    outputs = []
    for edge_span in (0.0399, 0.0401):
        depth = torch.full((1, 1, 4, 4), 0.5, device=device)
        depth[..., 2:] += edge_span
        outputs.append(
            upsample_depth(depth, 32, 32, rgb=rgb, mode="joint_bilateral")
        )
    if device == "cuda":
        torch.cuda.synchronize()

    assert all(torch.isfinite(output).all() for output in outputs)
    assert float((outputs[1] - outputs[0]).abs().max()) < 5.0e-4


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
@pytest.mark.parametrize("slope", [0.01, 0.03, 0.06])
def test_joint_upsample_does_not_amplify_curvature_threshold_crossing(device: str, slope: float) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm is unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal is unavailable")
    rgb = torch.zeros((1, 3, 64, 64), device=device)
    rgb[..., 32:] = 1.0
    outputs = []
    # Curvature used to switch reconstruction abruptly at threshold * 0.5.
    # A slope makes that switch jump much farther than the input perturbation.
    for span in (0.01999, 0.02001):
        depth = (0.2 + slope * torch.arange(8, device=device))[None, None, None]
        depth = depth.expand(1, 1, 8, 8).clone()
        depth[..., 4:] += span
        outputs.append(upsample_depth(depth, 64, 64, rgb=rgb, mode="joint_bilateral"))

    assert all(torch.isfinite(output).all() for output in outputs)
    assert float((outputs[1] - outputs[0]).abs().max()) < 5.0e-4


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
@pytest.mark.parametrize("depth_size", [(1, 1), (1, 5), (5, 1), (2, 2)])
def test_joint_upsample_curvature_preserves_small_constant_maps(device: str, depth_size: tuple[int, int]) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm is unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal is unavailable")
    depth = torch.full((2, 1, *depth_size), 0.4, device=device)
    rgb = torch.rand((2, 3, 19, 27), device=device)

    actual = upsample_depth(depth, 19, 27, rgb=rgb, mode="joint_bilateral")

    assert torch.isfinite(actual).all()
    assert torch.allclose(actual, torch.full_like(actual, 0.4), atol=1.0e-6, rtol=0.0)


def test_joint_bilateral_upsample_does_not_make_oblique_depth_ramp() -> None:
    """A low-resolution person contour must stay in one depth class.

    Bilinear depth values across an oblique foreground/background boundary
    become intermediate disparities.  Those values turn into a second,
    displaced contour in DIBR.  The RGB edge is deliberately subpixel shifted
    so this catches the production failure rather than only an axis-aligned
    fixture.
    """
    height, width = 64, 64
    depth_height, depth_width = 8, 8
    low_y, low_x = torch.meshgrid(
        torch.arange(depth_height, dtype=torch.float32),
        torch.arange(depth_width, dtype=torch.float32),
        indexing="ij",
    )
    slope = 0.375
    low_depth = torch.where(
        low_x + 0.5 >= 2.0 + slope * (low_y + 0.5),
        torch.tensor(0.8),
        torch.tensor(0.2),
    )[None, None]
    high_y, high_x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    boundary = (2.0 + slope * (high_y / 8.0 + 0.5)) * 8.0 + 0.25
    rgb = torch.where(high_x + 0.5 >= boundary, 0.82, 0.16)
    rgb = rgb.expand(3, -1, -1).unsqueeze(0)

    sampled = upsample_depth(
        low_depth, height, width, rgb=rgb, mode="joint_bilateral"
    )
    edge_band = (high_x - boundary).abs() <= 3.0
    intermediate = ((sampled[0, 0] > 0.25) & (sampled[0, 0] < 0.75))[edge_band]

    assert float(intermediate.float().mean()) < 0.08


def test_mps_joint_bilateral_matches_reference_edge_when_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("Metal is unavailable")
    depth_cpu = torch.full((1, 1, 8, 8), 0.1)
    depth_cpu[..., 4:] = 0.9
    rgb_cpu = torch.zeros((1, 3, 64, 64))
    rgb_cpu[..., :, 32:] = 1.0
    expected = upsample_depth(
        depth_cpu, 64, 64, rgb=rgb_cpu, mode="joint_bilateral"
    )
    actual = upsample_depth(
        depth_cpu.to("mps"),
        64,
        64,
        rgb=rgb_cpu.to("mps"),
        mode="joint_bilateral",
    ).cpu()

    assert torch.allclose(actual, expected, atol=1e-4, rtol=0.0)
    assert actual[0, 0, 32, 31] < 0.11
    assert actual[0, 0, 32, 32] > 0.89


@pytest.mark.parametrize("resolution", [196, 238, 294, 336, 392, 448, 518])
def test_joint_upsample_gpu_matches_reference_at_multiple_model_scales(resolution: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm is unavailable")
    pytest.importorskip("triton")
    from stereo_runtime._depth_upsample_triton import joint_bilateral_upsample
    from stereo_runtime.depth_upsample import _joint_bilateral_upsample

    height, width = 144, 256
    depth_width = max(3, round(width * resolution / 3840))
    depth_height = max(3, round(height * resolution / 3840))
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    radius = ((x - 127.25).square() + (y - 71.75).square()).float().sqrt()
    coverage = ((45.0 - radius) / 1.5 + 0.5).clamp(0.0, 1.0)
    rgb = (0.12 + 0.7 * coverage)[None, None].expand(1, 3, -1, -1).contiguous()
    depth = torch.nn.functional.interpolate(
        (0.15 + 0.7 * coverage)[None, None],
        size=(depth_height, depth_width), mode="bilinear", align_corners=False,
    )
    expected = _joint_bilateral_upsample(depth, rgb, height, width)
    guide = torch.nn.functional.interpolate(rgb, size=depth.shape[-2:], mode="bilinear", align_corners=False)
    # A compiler failure must fail here rather than use the bilinear fallback.
    actual = joint_bilateral_upsample(depth.cuda(), rgb.cuda(), guide.cuda(), height, width).cpu()

    assert torch.isfinite(actual).all()
    assert torch.allclose(actual, expected, atol=3e-4, rtol=0.0)
    edge_band = (radius - 45.0).abs() < 2.0
    assert float((actual[0, 0] - (0.15 + 0.7 * coverage)).abs()[edge_band].mean()) < 0.18
