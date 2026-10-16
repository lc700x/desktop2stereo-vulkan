import pytest
import torch

from desktop2stereo.stereo_runtime.temporal import TemporalState, stabilize_captured_depth


def test_motion_depth_stabilization_rejects_mismatched_frame_pair() -> None:
    state = TemporalState(
        motion_depth=torch.zeros((1, 1, 2, 2)),
        motion_luma=torch.zeros((2, 2)),
        motion_capture_frame_id=10,
        motion_depth_frame_id=10,
    )
    depth = torch.ones((1, 1, 2, 2))
    rgb = torch.zeros((1, 3, 2, 2))

    result, debug = stabilize_captured_depth(
        depth,
        rgb,
        state,
        capture_frame_id=11,
        depth_frame_id=10,
    )

    assert result is depth
    assert debug["motion_depth_reason"] == "depth_frame_mismatch"
    assert state.motion_depth is None
    assert state.motion_luma is None
    assert state.motion_capture_frame_id is None


def test_motion_depth_stabilization_requires_capture_id() -> None:
    state = TemporalState(
        motion_depth=torch.zeros((1, 1, 2, 2)),
        motion_luma=torch.zeros((2, 2)),
        motion_capture_frame_id=10,
        motion_depth_frame_id=10,
    )
    depth = torch.ones((1, 1, 2, 2))
    rgb = torch.zeros((1, 3, 2, 2))

    result, debug = stabilize_captured_depth(
        depth,
        rgb,
        state,
        capture_frame_id=None,
    )

    assert result is depth
    assert debug["motion_depth_reason"] == "capture_id_missing"
    assert state.motion_depth is None


def test_motion_depth_stabilization_honors_kill_switch(monkeypatch) -> None:
    monkeypatch.setenv("D2S_DEPTH_TEMPORAL", "0")
    state = TemporalState(
        motion_depth=torch.zeros((1, 1, 2, 2)),
        motion_luma=torch.zeros((2, 2)),
        motion_capture_frame_id=10,
        motion_depth_frame_id=10,
    )
    depth = torch.ones((1, 1, 2, 2))
    rgb = torch.zeros((1, 3, 2, 2))

    result, debug = stabilize_captured_depth(
        depth,
        rgb,
        state,
        capture_frame_id=11,
    )

    assert result is depth
    assert debug["motion_depth_reason"] == "disabled"
    assert state.motion_depth is None
    assert state.motion_luma is None


def test_temporal_reset_clears_motion_depth_history() -> None:
    state = TemporalState(
        motion_depth=torch.zeros((1, 1, 2, 2)),
        motion_luma=torch.zeros((2, 2)),
        motion_capture_frame_id=10,
        motion_depth_frame_id=10,
    )

    state.reset_stereo()

    assert state.motion_depth is None
    assert state.motion_luma is None
    assert state.motion_capture_frame_id is None
    assert state.motion_depth_frame_id is None


def test_motion_stabilization_does_not_blend_across_moving_depth_edge() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA or ROCm device is not available")
    try:
        from desktop2stereo.stereo_runtime.temporal_triton import (
            _downsample_motion_luma,
            can_use_motion_depth_stabilization,
            stabilize_depth_motion_compensated,
        )
    except ImportError:
        pytest.skip("Triton is not available")

    device = torch.device("cuda")
    height, width = 135, 240
    generator = torch.Generator(device=device).manual_seed(7)
    previous_rgb = torch.rand(
        (1, 3, height, width),
        device=device,
        generator=generator,
    )
    current_rgb = torch.roll(previous_rgb, shifts=1, dims=-1)
    previous_depth = torch.full((1, 1, height, width), 0.2, device=device)
    current_depth = torch.full_like(previous_depth, 0.2)
    previous_depth[:, :, :, 120:] = 0.8
    current_depth[:, :, :, 121:] = 0.8
    if not can_use_motion_depth_stabilization(current_depth, current_rgb):
        pytest.skip("Triton motion stabilization is unavailable on this device")
    previous_luma = _downsample_motion_luma(previous_rgb)
    output, _ = stabilize_depth_motion_compensated(
        current_depth,
        previous_depth,
        current_rgb,
        previous_luma,
        edge_threshold=0.04,
    )
    torch.cuda.synchronize(device)

    expected = torch.tensor([0.2] * 4 + [0.8] * 4, device=device)
    torch.testing.assert_close(output[0, 0, height // 2, 117:125], expected, atol=1e-5, rtol=0.0)
    assert bool(torch.isfinite(output).all().item())
