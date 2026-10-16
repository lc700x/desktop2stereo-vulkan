from __future__ import annotations

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_warp_composite_rgba_u8_matches_float_warp_then_pack():
    pytest.importorskip("triton")
    from stereo_runtime.output_triton import make_chw_rgb_to_hwc_rgba_u8
    from stereo_runtime.warp_composite_triton import warp_composite2, warp_composite2_rgba_u8

    torch.manual_seed(1234)
    rgb = torch.rand((1, 3, 31, 47), device="cuda", dtype=torch.float32)
    depth = torch.rand((1, 1, 31, 47), device="cuda", dtype=torch.float32)
    base_shift = (torch.rand((1, 1, 31, 47), device="cuda", dtype=torch.float32) - 0.5) * 14.0

    float_left, float_right = warp_composite2(rgb, depth, base_shift)
    expected_left = make_chw_rgb_to_hwc_rgba_u8(float_left)
    expected_right = make_chw_rgb_to_hwc_rgba_u8(float_right)
    actual_left, actual_right = warp_composite2_rgba_u8(rgb, depth, base_shift)
    torch.cuda.synchronize()

    assert torch.equal(actual_left, expected_left)
    assert torch.equal(actual_right, expected_right)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_direct_half_sbs_matches_openxr_lanczos2_pack():
    pytest.importorskip("triton")
    from stereo_runtime.output import make_sbs
    from stereo_runtime.warp_composite_triton import (
        warp_composite2,
        warp_composite2_half_sbs,
    )

    torch.manual_seed(4321)
    rgb = torch.rand((1, 3, 24, 40), device="cuda", dtype=torch.float32)
    depth = torch.rand((1, 1, 24, 40), device="cuda", dtype=torch.float32)
    base_shift = (
        torch.rand((1, 1, 24, 40), device="cuda", dtype=torch.float32) - 0.5
    ) * 12.0

    left, right = warp_composite2(rgb, depth, base_shift)
    expected = make_sbs(left, right, "half_sbs", fused=True)
    actual = warp_composite2_half_sbs(rgb, depth, base_shift)
    torch.cuda.synchronize()

    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/ROCm is required")
@pytest.mark.parametrize("height,width", [(24, 40), (31, 66)])
def test_direct_half_sbs_uint8_preserves_float_composite(height, width):
    pytest.importorskip("triton")
    from stereo_runtime.output import make_sbs
    from stereo_runtime.warp_composite_triton import warp_composite2, warp_composite2_half_sbs

    torch.manual_seed(4321)
    rgb = torch.rand((1, 3, height, width), device="cuda")
    depth = torch.rand((1, 1, height, width), device="cuda")
    shift = (torch.rand_like(depth) - 0.5) * 12.0
    left, right = warp_composite2(rgb, depth, shift)
    expected = make_sbs(left, right, "half_sbs", fused=True).clamp(0, 1).mul(255).round().to(torch.uint8)

    actual = warp_composite2_half_sbs(rgb, depth, shift, output_uint8=True)

    assert actual.shape == rgb.shape
    assert actual.dtype == torch.uint8
    assert (actual.short() - expected.short()).abs().max() <= 1
    assert (actual != expected).float().mean() < 0.001


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/ROCm is required")
def test_edge_classification_is_continuous_near_depth_thresholds():
    pytest.importorskip("triton")
    from stereo_runtime.warp_composite_triton import warp_composite2

    height = width = 24
    ramp = torch.linspace(0.0, 1.0, width, device="cuda").view(1, 1, 1, width)
    rgb = torch.cat(
        (
            ramp.expand(1, 1, height, width),
            (1.0 - ramp).expand(1, 1, height, width),
            torch.zeros((1, 1, height, width), device="cuda"),
        ),
        dim=1,
    )
    shift = torch.full((1, 1, height, width), 8.0, device="cuda")

    for threshold in (0.025, 0.04, 0.05):
        stereo_outputs = []
        for delta in (threshold - 1.0e-4, threshold + 1.0e-4):
            depth = torch.full((1, 1, height, width), 0.5, device="cuda")
            depth[..., width // 2 :] += delta
            left, right = warp_composite2(rgb, depth, shift)
            stereo_outputs.append((left, right))
        torch.cuda.synchronize()

        for before, after in zip(stereo_outputs[0], stereo_outputs[1]):
            assert float((after - before).abs().max()) < 1.0e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/ROCm is required")
def test_source_projection_does_not_copy_foreground_into_disoccluded_background():
    pytest.importorskip("triton")
    from stereo_runtime.baseline_shift import ShiftParams, compute_shift_px
    from stereo_runtime.warp_composite_triton import warp_composite2

    height, width = 32, 160
    rgb = torch.full((1, 3, height, width), 0.12, device="cuda")
    rgb[..., 40:100] = 0.85
    depth = torch.full((1, 1, height, width), 0.2, device="cuda")
    depth[..., 40:100] = 0.8
    shift = compute_shift_px(depth, width, ShiftParams(convergence=0.5, depth_strength=1.0, max_disparity_px=40))

    left, right = warp_composite2(rgb, depth, shift)

    assert torch.isfinite(left).all() and torch.isfinite(right).all()
    # The near surface moves six pixels; the right eye reveals background
    # rather than drawing the source outline again in the uncovered strip.
    assert float((right[..., 96:108] - 0.12).abs().max()) < 1e-5
    assert float((left[..., 108:120] - 0.12).abs().max()) < 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/ROCm is required")
def test_projected_rgba_and_sbs_preserve_float_visibility():
    pytest.importorskip("triton")
    from stereo_runtime.baseline_shift import ShiftParams, compute_shift_px
    from stereo_runtime.output import make_sbs
    from stereo_runtime.output_triton import make_chw_rgb_to_hwc_rgba_u8
    from stereo_runtime.warp_composite_triton import warp_composite2, warp_composite2_half_sbs, warp_composite2_rgba_u8

    rgb = torch.rand((1, 3, 32, 64), device="cuda")
    depth = torch.full((1, 1, 32, 64), 0.15, device="cuda")
    depth[..., 18:40] = 0.85
    shift = compute_shift_px(depth, 64, ShiftParams(convergence=0.5, max_disparity_px=12))
    left, right = warp_composite2(rgb, depth, shift)
    rgba_left, rgba_right = warp_composite2_rgba_u8(rgb, depth, shift)
    sbs = warp_composite2_half_sbs(rgb, depth, shift)

    assert torch.equal(rgba_left, make_chw_rgb_to_hwc_rgba_u8(left))
    assert torch.equal(rgba_right, make_chw_rgb_to_hwc_rgba_u8(right))
    assert torch.allclose(sbs, make_sbs(left, right, "half_sbs", fused=True), atol=2e-6, rtol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/ROCm is required")
def test_tiled_projected_rgba_matches_linear_kernel_exactly():
    pytest.importorskip("triton")
    import triton
    from stereo_runtime.baseline_shift import ShiftParams, compute_shift_px
    from stereo_runtime.warp_composite_triton import (
        _warp_composite2_rgba_u8_kernel,
        warp_composite2_rgba_u8,
    )

    torch.manual_seed(20261017)
    height, width = 37, 70
    rgb = torch.rand((1, 3, height, width), device="cuda")
    depth = torch.full((1, 1, height, width), 0.15, device="cuda")
    depth[..., 18:43] = 0.85
    shift = compute_shift_px(
        depth,
        width,
        ShiftParams(convergence=0.5, depth_strength=1.0, max_disparity_px=12.0),
    )
    pmin, pmax = shift._d2s_projection_bounds
    pixels = height * width
    expected_left = torch.empty((height, width, 4), device="cuda", dtype=torch.uint8)
    expected_right = torch.empty_like(expected_left)
    _warp_composite2_rgba_u8_kernel[(triton.cdiv(pixels, 256),)](
        rgb,
        depth,
        shift,
        expected_left,
        expected_right,
        pixels,
        width,
        0.08,
        pmin,
        pmax,
        False,
        256,
        num_warps=4,
    )

    actual_left, actual_right = warp_composite2_rgba_u8(rgb, depth, shift)

    assert torch.equal(actual_left, expected_left)
    assert torch.equal(actual_right, expected_right)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/ROCm is required")
@pytest.mark.parametrize("output_uint8", [False, True])
def test_tiled_half_sbs_matches_linear_kernel_exactly(output_uint8: bool):
    pytest.importorskip("triton")
    import triton
    from stereo_runtime.baseline_shift import ShiftParams, compute_shift_px
    from stereo_runtime.warp_composite_triton import (
        _warp_composite2_half_sbs_kernel,
        warp_composite2_half_sbs,
    )

    torch.manual_seed(20261018)
    height, width = 37, 70
    rgb = torch.rand((1, 3, height, width), device="cuda")
    depth = torch.full((1, 1, height, width), 0.15, device="cuda")
    depth[..., 18:43] = 0.85
    shift = compute_shift_px(
        depth,
        width,
        ShiftParams(convergence=0.5, depth_strength=1.0, max_disparity_px=12.0),
    )
    pmin, pmax = shift._d2s_projection_bounds
    pixels = height * width
    expected = torch.empty_like(rgb, dtype=torch.uint8 if output_uint8 else rgb.dtype)
    _warp_composite2_half_sbs_kernel[(triton.cdiv(pixels, 128),)](
        rgb,
        depth,
        shift,
        expected,
        pixels,
        width,
        width // 2,
        pixels,
        0.08,
        pmin,
        pmax,
        output_uint8,
        128,
        num_warps=4,
    )

    actual = warp_composite2_half_sbs(rgb, depth, shift, output_uint8=output_uint8)

    assert torch.equal(actual, expected)
