from __future__ import annotations

from dataclasses import dataclass
import logging
import os

import torch

LOGGER = logging.getLogger(__name__)


@dataclass
class TemporalState:
    left: torch.Tensor | None = None
    right: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    scene_reference: torch.Tensor | None = None
    cooldown_remaining: int = 0
    reset_count: int = 0
    last_scene_delta: float = 0.0
    last_scene_reset: bool = False
    scene_check_counter: int = 0
    motion_depth: torch.Tensor | None = None
    motion_luma: torch.Tensor | None = None
    motion_capture_frame_id: int | None = None
    motion_depth_frame_id: int | None = None
    motion_disabled: bool = False

    def reset(self) -> None:
        self.reset_stereo()
        self.scene_reference = None
        self.cooldown_remaining = 0
        self.last_scene_delta = 0.0
        self.last_scene_reset = False
        self.scene_check_counter = 0

    def reset_stereo(self) -> None:
        self.left = None
        self.right = None
        self.mask = None
        self.reset_motion_depth()

    def reset_motion_depth(self) -> None:
        self.motion_depth = None
        self.motion_luma = None
        self.motion_capture_frame_id = None
        self.motion_depth_frame_id = None
        self.motion_disabled = False


def stabilize_captured_depth(
    depth: torch.Tensor,
    rgb: torch.Tensor,
    state: TemporalState | None,
    *,
    capture_frame_id: int | None,
    depth_frame_id: int | None = None,
    strength: float = 0.65,
    edge_threshold: float = 0.04,
    enabled: bool = True,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Use current-frame optical flow to stabilize depth edges without pixel-space trails."""
    if state is None:
        return depth, {"motion_depth_stabilization": 0, "motion_depth_reason": "no_state"}
    if not enabled or os.environ.get("D2S_DEPTH_TEMPORAL", "1").strip().lower() in {
        "0",
        "false",
        "off",
        "no",
    }:
        state.reset_motion_depth()
        return depth, {"motion_depth_stabilization": 0, "motion_depth_reason": "disabled"}
    if state.motion_disabled:
        return depth, {"motion_depth_stabilization": 0, "motion_depth_reason": "kernel_disabled"}
    if capture_frame_id is None:
        state.reset_motion_depth()
        return depth, {"motion_depth_stabilization": 0, "motion_depth_reason": "capture_id_missing"}
    try:
        current_capture_id = int(capture_frame_id)
        current_depth_id = current_capture_id if depth_frame_id is None else int(depth_frame_id)
    except (TypeError, ValueError):
        state.reset_motion_depth()
        return depth, {"motion_depth_stabilization": 0, "motion_depth_reason": "invalid_frame_id"}
    if current_depth_id != current_capture_id:
        state.reset_motion_depth()
        return depth, {"motion_depth_stabilization": 0, "motion_depth_reason": "depth_frame_mismatch"}

    try:
        from .temporal_triton import (
            _downsample_motion_luma,
            can_use_motion_depth_stabilization,
            stabilize_depth_motion_compensated,
        )

        if not can_use_motion_depth_stabilization(depth, rgb):
            state.reset_motion_depth()
            return depth, {"motion_depth_stabilization": 0, "motion_depth_reason": "triton_unavailable"}
        if state.motion_capture_frame_id == current_capture_id:
            if (
                state.motion_depth is not None
                and state.motion_depth_frame_id == current_depth_id
                and state.motion_depth.shape == depth.shape
                and state.motion_depth.device == depth.device
            ):
                return state.motion_depth, {
                    "motion_depth_stabilization": 0,
                    "motion_depth_reason": "same_capture_reused",
                }
            state.reset_motion_depth()

        previous_capture_id = state.motion_capture_frame_id
        if previous_capture_id is not None and (
            current_capture_id <= previous_capture_id
            or current_capture_id - previous_capture_id > 3
        ):
            state.reset_motion_depth()

        previous = state.motion_depth
        previous_luma = state.motion_luma
        if (
            previous is None
            or previous_luma is None
            or previous.shape != depth.shape
            or previous.device != depth.device
            or previous_luma.device != rgb.device
        ):
            current_luma = _downsample_motion_luma(rgb)
            state.motion_depth = depth.detach().contiguous().clone()
            state.motion_luma = current_luma.detach()
            state.motion_capture_frame_id = current_capture_id
            state.motion_depth_frame_id = current_depth_id
            return depth, {"motion_depth_stabilization": 0, "motion_depth_reason": "history_seeded"}

        stabilized, current_luma = stabilize_depth_motion_compensated(
            depth,
            previous,
            rgb,
            previous_luma,
            strength=strength,
            edge_threshold=edge_threshold,
        )
    except Exception as exc:
        state.reset_motion_depth()
        state.motion_disabled = True
        LOGGER.warning(
            "Motion-compensated depth stabilization disabled after %s: %s",
            type(exc).__name__,
            exc,
        )
        return depth, {"motion_depth_stabilization": 0, "motion_depth_reason": "kernel_failed"}

    state.motion_depth = stabilized.detach()
    state.motion_luma = current_luma.detach()
    state.motion_capture_frame_id = current_capture_id
    state.motion_depth_frame_id = current_depth_id
    return stabilized, {
        "motion_depth_stabilization": 1,
        "motion_depth_backend": "triton_forward_backward_flow",
    }


def detect_scene_change(
    rgb: torch.Tensor,
    state: TemporalState | None,
    *,
    threshold: float = 0.22,
    sample_size: int = 64,
    scene_check_interval: int = 6,
) -> bool:
    return False


def detect_scene_gate(
    rgb: torch.Tensor,
    state: TemporalState | None,
    *,
    threshold: float = 0.22,
    sample_size: int = 64,
    scene_check_interval: int = 6,
) -> torch.Tensor | None:
    if state is None or not rgb.is_cuda or not torch.cuda.is_available():
        return None
    state.last_scene_reset = False
    interval = max(1, int(scene_check_interval))
    if state.scene_reference is not None:
        state.scene_check_counter = (state.scene_check_counter + 1) % interval
        if state.scene_check_counter != 0:
            return None
    sample = _scene_sample(rgb, sample_size=sample_size)
    if state.scene_reference is None or state.scene_reference.shape != sample.shape:
        state.scene_reference = sample.detach()
        state.last_scene_delta = 0.0
        return None

    delta_tensor = (sample - state.scene_reference.to(sample.device)).abs().mean()
    state.scene_reference = sample.detach()
    return (delta_tensor < float(threshold)).to(dtype=sample.dtype)

def apply_temporal(
    left: torch.Tensor,
    right: torch.Tensor,
    mask: torch.Tensor | None,
    state: TemporalState | None,
    strength: float = 0.85,
    scene_gate: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if state is None:
        return left, right
    if state.left is None or state.right is None or state.left.shape != left.shape or state.right.shape != right.shape:
        state.left = left.detach()
        state.right = right.detach()
        state.mask = mask.detach() if mask is not None else None
        return left, right

    alpha = float(max(0.0, min(strength, 0.98)))
    if mask is None:
        blend = alpha
    else:
        blend = (mask * alpha).to(device=left.device, dtype=left.dtype)
    if scene_gate is not None:
        blend = blend * scene_gate.to(device=left.device, dtype=left.dtype)
    elif mask is not None:
        try:
            from .temporal_triton import apply_temporal_masked, can_use_triton_temporal_masked

            if can_use_triton_temporal_masked(left, right, state.left, state.right, blend):
                left_out, right_out = apply_temporal_masked(
                    left,
                    right,
                    state.left,
                    state.right,
                    blend,
                    alpha=1.0,
                )
                state.left = left_out.detach()
                state.right = right_out.detach()
                state.mask = mask.detach()
                return left_out, right_out
        except Exception:
            pass
    left_out = left * (1.0 - blend) + state.left.to(left.device) * blend
    right_out = right * (1.0 - blend) + state.right.to(right.device) * blend
    state.left = left_out.detach()
    state.right = right_out.detach()
    state.mask = mask.detach() if mask is not None else None
    return left_out, right_out


def _scene_sample(rgb: torch.Tensor, *, sample_size: int) -> torch.Tensor:
    if rgb.ndim == 3:
        rgb = rgb.unsqueeze(0)
    if rgb.ndim != 4:
        raise ValueError(f"rgb must be CHW or BCHW, got shape {tuple(rgb.shape)}")
    sample = rgb.detach()
    height = int(sample.shape[-2])
    width = int(sample.shape[-1])
    target = max(1, int(sample_size))
    step_y = max(1, height // target)
    step_x = max(1, width // target)
    offset_y = max(0, step_y // 2)
    offset_x = max(0, step_x // 2)
    sample = sample[..., offset_y::step_y, offset_x::step_x]
    sample = sample[..., :target, :target].float()
    if sample.shape[1] == 3:
        sample = sample[:, 0:1] * 0.299 + sample[:, 1:2] * 0.587 + sample[:, 2:3] * 0.114
    elif sample.shape[1] != 1:
        sample = sample.mean(dim=1, keepdim=True)
    return sample.contiguous()
