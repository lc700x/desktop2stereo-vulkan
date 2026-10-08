from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

from stereo_runtime.display_antialias import antialias_eye, antialias_sbs, antialias_sbs_half
from stereo_runtime.output import _make_sbs_unfiltered, make_sbs


def _diagonal(width=96, height=64, slope=0.43, phase=0.3, scale_x=1):
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    line = slope * (yy + 0.5) + 21.0 + phase
    binary = (xx + 0.5 >= line).float()[None, None].expand(1, 3, height, width).clone()
    coverage = torch.zeros(height, width // scale_x)
    for sy in range(8):
        for sx in range(8):
            y = torch.arange(height)[:, None] + (sy + 0.5) / 8
            x = torch.arange(width // scale_x)[None, :] * scale_x + (sx + 0.5) / 8 * scale_x
            coverage += (x >= slope * y + 21.0 + phase).float() / 64
    reference = torch.where(coverage <= 0.0031308, coverage * 12.92,
                            1.055 * coverage.clamp_min(0).pow(1 / 2.4) - 0.055)
    mask = ((coverage > 0) & (coverage < 1)).float()[None, None]
    mask = F.max_pool2d(mask, 3, stride=1, padding=1).bool().expand(1, 3, -1, -1)
    return binary, reference[None, None].expand(1, 3, -1, -1), mask


@pytest.mark.parametrize("output_format,scale_x", [("full_sbs", 1), ("half_sbs", 2)])
def test_fxaa_before_pack_improves_analytic_8x8_diagonal_coverage(monkeypatch, output_format, scale_x):
    monkeypatch.setenv("D2S_SBS_AA", "1")
    source, reference, mask = _diagonal(scale_x=scale_x)
    before = _make_sbs_unfiltered(source, source, output_format, fused=False)
    after = make_sbs(source, source, output_format, fused=False)
    eye_width = after.shape[-1] // 2
    before_error = (before[..., :eye_width] - reference)[mask].square().mean().sqrt()
    after_error = (after[..., :eye_width] - reference)[mask].square().mean().sqrt()

    assert after_error < before_error * 0.8
    assert torch.equal(after[..., eye_width:], after[..., :eye_width])
    # The Half-SBS path now uses a linear-light area average, so compare flat
    # regions against the same unfiltered resampler rather than point sampling.
    assert torch.equal(after[..., :eye_width][~mask], before[..., :eye_width][~mask])


def test_fxaa_preserves_two_pixel_lines_in_full_and_half_sbs(monkeypatch):
    monkeypatch.setenv("D2S_SBS_AA", "1")
    image = torch.zeros(1, 3, 32, 48)
    image[..., 22:24] = 1.0
    for output_format in ("full_sbs", "half_sbs"):
        output = make_sbs(image, image, output_format, fused=False)
        peak_per_row = output[..., :output.shape[-1] // 2].amax(dim=-1)
        assert torch.all(peak_per_row > 0.5)


def test_eye_seam_flat_colors_and_alpha_survive_odd_sizes():
    image = torch.zeros(2, 4, 12, 19)
    image[:, 0, :, :9] = 1.0
    image[:, 2, :, 9:] = 1.0
    image[:, 3] = torch.arange(19)[None, :].expand(12, -1) / 19

    assert torch.equal(antialias_sbs(image, "full_sbs"), image)
    assert torch.equal(antialias_sbs(image, "depth_map"), image)


def test_half_sbs_odd_eye_width_preserves_extent_and_eye_isolation(monkeypatch):
    monkeypatch.setenv("D2S_SBS_AA", "1")
    eyes = torch.zeros(1, 3, 8, 10)
    eyes[:, 1, :, :5] = 1.0
    eyes[:, 2, :, 5:] = 1.0

    packed = antialias_sbs_half(eyes)

    assert packed.shape == (1, 3, 8, 5)
    assert torch.allclose(packed[:, 1, :, :2], torch.ones_like(packed[:, 1, :, :2]), atol=1e-6)
    assert torch.allclose(packed[:, 2, :, 2:], torch.ones_like(packed[:, 2, :, 2:]), atol=1e-6)
    assert torch.count_nonzero(packed[:, 2, :, :2]) == 0
    assert torch.count_nonzero(packed[:, 1, :, 2:]) == 0


def test_display_only_filter_handles_rgba_u8_openxr_layout_without_touching_alpha():
    source, _, _ = _diagonal()
    rgb = (source.squeeze(0).permute(1, 2, 0) * 255).to(torch.uint8)
    alpha = torch.arange(rgb.shape[1], dtype=torch.uint8)[None, :, None].expand(rgb.shape[0], -1, -1)
    rgba = torch.cat((rgb, alpha), dim=-1)
    output = antialias_eye(rgba)

    assert output.shape == rgba.shape and output.dtype == rgba.dtype
    assert torch.equal(output[..., 3], rgba[..., 3])
    assert not torch.equal(output[..., :3], rgba[..., :3])


@pytest.mark.skipif(os.environ.get("D2S_TEST_MPS_AA") != "1", reason="explicit MPS pixel verification")
def test_mps_fxaa_matches_tensor_reference_and_preserves_eye_seam():
    source, _, _ = _diagonal(width=94)
    rgba = torch.cat((source, torch.full_like(source[:, :1], 0.7)), dim=1).repeat(2, 1, 1, 1)
    actual = antialias_sbs(rgba.to("mps"), "full_sbs").cpu()
    expected = antialias_sbs(rgba, "full_sbs")

    assert actual.shape == rgba.shape and actual.dtype == rgba.dtype
    assert torch.equal(actual[:, 3], rgba[:, 3])
    assert not torch.equal(actual[:, :3], rgba[:, :3])
    assert torch.allclose(actual, expected, atol=1e-4, rtol=0)

    changed = rgba.clone()
    changed[..., changed.shape[-1] // 2 :] = 0.25
    changed[:, 3, :, changed.shape[-1] // 2 :] = 0.7
    changed_output = antialias_sbs(changed.to("mps"), "full_sbs").cpu()
    eye_width = changed.shape[-1] // 2
    assert torch.allclose(actual[..., :eye_width], changed_output[..., :eye_width], atol=1e-6, rtol=0)


@pytest.mark.skipif(os.environ.get("D2S_TEST_MPS_AA") != "1", reason="explicit MPS pixel verification")
def test_mps_fxaa_near_ties_match_tensor_reference_in_full_and_half_sbs():
    torch.manual_seed(19)
    source = torch.randint(0, 256, (1, 3, 64, 96), dtype=torch.uint8)
    y = torch.arange(64, dtype=torch.int32)[:, None]
    x = torch.arange(48, dtype=torch.int32)[None, :]
    source[..., :48] = ((x * 3 + y * 5)[None, None] % 256).to(torch.uint8)
    source[..., 8:11, 8:40] = 255
    source[..., 35:38, 20:44] = 0

    expected_full = antialias_sbs(source, "full_sbs")
    actual_full = antialias_sbs(source.to("mps"), "full_sbs").cpu()
    expected_half = antialias_sbs_half(source)
    actual_half = antialias_sbs_half(source.to("mps")).cpu()

    assert int((actual_full.to(torch.int16) - expected_full.to(torch.int16)).abs().max()) <= 1
    assert int((actual_half.to(torch.int16) - expected_half.to(torch.int16)).abs().max()) <= 1


@pytest.mark.skipif(os.environ.get("D2S_TEST_MPS_AA") != "1", reason="explicit MPS pixel verification")
def test_mps_half_pack_matches_full_eye_aa_then_area_for_odd_widths():
    from stereo_runtime.output import downsample_horizontal_area_srgb

    torch.manual_seed(4)
    eyes = torch.rand(1, 4, 12, 50, dtype=torch.float32, device="mps")
    expected_full = antialias_sbs(eyes, "full_sbs")
    expected = torch.cat(
        (
            downsample_horizontal_area_srgb(expected_full[..., :25].cpu(), 12),
            downsample_horizontal_area_srgb(expected_full[..., 25:].cpu(), 13),
        ),
        dim=-1,
    )
    actual = antialias_sbs_half(eyes).cpu()

    assert actual.shape == (1, 4, 12, 25)
    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(os.environ.get("D2S_TEST_MPS_AA") != "1", reason="explicit MPS pixel verification")
def test_mps_uint8_half_pack_matches_full_eye_aa_then_area():
    from stereo_runtime.output import downsample_horizontal_area_srgb

    torch.manual_seed(9)
    eyes = torch.randint(0, 256, (1, 4, 12, 48), dtype=torch.uint8, device="mps")
    expected_full = antialias_sbs(eyes, "full_sbs").cpu()
    expected = torch.cat(
        (
            downsample_horizontal_area_srgb(expected_full[..., :24], 12),
            downsample_horizontal_area_srgb(expected_full[..., 24:], 12),
        ),
        dim=-1,
    )
    actual = antialias_sbs_half(eyes).cpu()

    assert actual.dtype == torch.uint8
    assert actual.shape == (1, 4, 12, 24)
    assert int((actual.to(torch.int16) - expected.to(torch.int16)).abs().max()) <= 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/ROCm display kernel requires a GPU")
@pytest.mark.parametrize("dtype", (torch.float32, torch.uint8))
def test_triton_fxaa_matches_tensor_reference_on_cuda_or_rocm(dtype):
    from stereo_runtime._display_antialias_triton import fxaa

    torch.manual_seed(12)
    source = torch.rand((1, 3, 48, 96), dtype=torch.float32)
    if dtype == torch.uint8:
        source = (source * 255).round().to(torch.uint8)
        source[..., 20:23] = 255
    else:
        source[..., 20:23] = 1.0
    source[..., 48:] = source[..., 48:].roll(1, dims=-1)
    expected = antialias_sbs(source, "full_sbs")
    actual = fxaa(source.to("cuda"), eye_width=48).cpu()

    if dtype == torch.uint8:
        assert int((actual.to(torch.int16) - expected.to(torch.int16)).abs().max()) <= 1
    else:
        assert torch.allclose(actual, expected, atol=1 / 255, rtol=0)
