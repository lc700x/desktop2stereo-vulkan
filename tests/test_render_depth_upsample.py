from __future__ import annotations

import torch
import pytest

from stereo_runtime.depth_upsample import upsample_depth


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
