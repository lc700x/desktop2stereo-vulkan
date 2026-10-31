import math
import sys
from pathlib import Path

from path_config import APP_ROOT

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from capture.preprocess_triton import bgr_to_rgb_resize_norm, can_use_triton_preprocess


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _torch_bilinear_reference(frame_raw, out_height, out_width):
    frame_rgb = frame_raw[..., [2, 1, 0]].permute(2, 0, 1).contiguous().float().mul_(1.0 / 255.0)
    if frame_rgb.shape[-2:] == (out_height, out_width):
        return frame_rgb
    return F.interpolate(
        frame_rgb.unsqueeze(0),
        size=(out_height, out_width),
        mode="bilinear",
        align_corners=False,
        antialias=False,
    ).squeeze(0)


def _area_reference(frame_raw, out_height, out_width):
    frame_rgb = frame_raw[..., [2, 1, 0]].detach().cpu().float().mul_(1.0 / 255.0)
    in_height, in_width = frame_rgb.shape[:2]
    scale_y = in_height / out_height
    scale_x = in_width / out_width
    out = torch.zeros((3, out_height, out_width), dtype=torch.float32)
    for oy in range(out_height):
        y_start = oy * scale_y
        y_end = y_start + scale_y
        for ox in range(out_width):
            x_start = ox * scale_x
            x_end = x_start + scale_x
            value = torch.zeros(3, dtype=torch.float32)
            for sy in range(math.floor(y_start), math.ceil(y_end)):
                wy = max(0.0, min(y_end, sy + 1.0) - max(y_start, float(sy)))
                for sx in range(math.floor(x_start), math.ceil(x_end)):
                    wx = max(0.0, min(x_end, sx + 1.0) - max(x_start, float(sx)))
                    value += frame_rgb[sy, sx] * (wx * wy)
            out[:, oy, ox] = value / (scale_x * scale_y)
    return out.cuda()


@pytest.mark.parametrize("channels", [3, 4])
def test_triton_preprocess_preserves_pixels_without_scaling(channels):
    frame = torch.arange(5 * 6 * channels, device="cuda", dtype=torch.uint8).reshape(5, 6, channels)

    actual = bgr_to_rgb_resize_norm(frame, 5, 6)
    expected = _torch_bilinear_reference(frame, 5, 6)

    assert can_use_triton_preprocess(frame)
    assert actual.shape == (3, 5, 6)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(torch.version.cuda is None, reason="NVIDIA CUDA required")
def test_triton_preprocess_same_size_bgra_matches_reference_exactly():
    frame = torch.randint(0, 256, (37, 53, 4), device="cuda", dtype=torch.uint8)

    actual = bgr_to_rgb_resize_norm(frame, 37, 53)
    expected = _torch_bilinear_reference(frame, 37, 53)

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("out_size", [(4, 5), (6, 7)])
def test_triton_preprocess_matches_area_filter_when_downscaling(channels, out_size):
    frame = torch.arange(8 * 10 * channels, device="cuda", dtype=torch.uint8).reshape(8, 10, channels)
    out_height, out_width = out_size

    actual = bgr_to_rgb_resize_norm(frame, out_height, out_width)
    expected = _area_reference(frame, out_height, out_width)

    assert actual.shape == (3, out_height, out_width)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
