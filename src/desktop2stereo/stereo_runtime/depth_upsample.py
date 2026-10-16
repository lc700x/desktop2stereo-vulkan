from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from .output import ensure_b1hw, ensure_bchw

DepthUpsampleMode = Literal["bilinear", "guided", "joint_bilateral"]


@dataclass
class _ModelDepthAntialiasState:
    strength: float
    edge_threshold: float = 0.04
    applied: bool = False


_MODEL_DEPTH_ANTIALIAS_STATE: ContextVar[_ModelDepthAntialiasState | None] = ContextVar(
    "d2s_model_depth_antialias_state", default=None
)


@contextmanager
def model_depth_antialiasing(strength: float, edge_threshold: float = 0.04):
    """Apply the configured spatial depth filter before provider upsampling.

    Providers share ``upsample_depth`` but own their model execution. A
    context-local setting lets the runtime enable identical postprocessing for
    PyTorch, TensorRT, ONNX, ROCm, and MPS providers without changing their
    inference code or constructor contracts.
    """
    state = _ModelDepthAntialiasState(
        max(0.0, float(strength)),
        max(0.0, float(edge_threshold)),
    )
    token = _MODEL_DEPTH_ANTIALIAS_STATE.set(state)
    try:
        yield state
    finally:
        _MODEL_DEPTH_ANTIALIAS_STATE.reset(token)


def filter_model_depth(
    depth: torch.Tensor,
    rgb: torch.Tensor | None,
    strength: float,
    *,
    edge_threshold: float = 0.04,
) -> torch.Tensor:
    if strength <= 0.0:
        return depth
    # v2.5 maps the 0..2 GUI choice to AA_STRENGTH=0..4. Keep that mapping
    # while filtering on the model grid instead of the much larger RGB grid.
    model_strength = min(4.0, 2.0 * float(strength))
    from .depth_postprocess import anti_alias_depth

    return anti_alias_depth(depth, model_strength)


def _gather_depth_samples(depth: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = depth.shape
    flat = depth.reshape(batch, channels, height * width)
    return flat.gather(2, indices.reshape(1, 1, -1).expand(batch, channels, -1))


def _joint_bilateral_upsample(
    depth: torch.Tensor,
    rgb: torch.Tensor,
    height: int,
    width: int,
    *,
    color_sigma: float = 0.12,
    depth_edge_threshold: float = 0.04,
) -> torch.Tensor:
    """Reconstruct same-frame depth with continuous model-grid RGB support."""
    batch, _, depth_height, depth_width = depth.shape
    rgb = ensure_bchw(rgb, name="rgb").to(device=depth.device, dtype=torch.float32)
    if rgb.shape[0] == 1 and batch != 1:
        rgb = rgb.expand(batch, -1, -1, -1)
    if rgb.shape[0] != batch:
        raise ValueError("rgb and depth batch dimensions must match")
    if rgb.shape[-2:] != (height, width):
        rgb = F.interpolate(rgb, size=(height, width), mode="bilinear", align_corners=False)
    rgb = rgb[:, :3].clamp(0.0, 1.0)
    guide = F.interpolate(rgb, size=(depth_height, depth_width), mode="bilinear", align_corners=False)
    y = ((torch.arange(height, device=depth.device, dtype=torch.float32) + 0.5)
         * depth_height / height - 0.5).clamp(0, depth_height - 1)
    x = ((torch.arange(width, device=depth.device, dtype=torch.float32) + 0.5)
         * depth_width / width - 0.5).clamp(0, depth_width - 1)
    cy, cx = (y + 0.5).floor().long(), (x + 0.5).floor().long()
    flat_guide = guide.flatten(2)
    spatial_total = torch.zeros_like(rgb[:, :1])
    depth_sum = torch.zeros_like(spatial_total)
    depth_square_sum = torch.zeros_like(spatial_total)
    color_sum = torch.zeros_like(rgb)
    color_square_sum = torch.zeros_like(spatial_total)
    total = torch.zeros_like(spatial_total)
    guided_sum = torch.zeros_like(spatial_total)
    matching_distance_sum = torch.zeros_like(spatial_total)
    max_log_weight = torch.full_like(spatial_total, -1.0e30)
    sigma = max(float(color_sigma) / 3.0, 1.0e-4)
    threshold = max(float(depth_edge_threshold), 1.0e-6)
    bilinear = F.interpolate(depth, size=(height, width), mode="bilinear", align_corners=False)
    if depth_width >= 3:
        padded_x = F.pad(depth, (1, 1, 0, 0), mode="replicate")
        curvature_x = (padded_x[..., :-2] - 2.0 * depth + padded_x[..., 2:]).abs()
        curvature_x = torch.cat(
            (
                (depth[..., :1] - 2.0 * depth[..., 1:2] + depth[..., 2:3]).abs(),
                curvature_x[..., 1:-1],
                (depth[..., -3:-2] - 2.0 * depth[..., -2:-1] + depth[..., -1:]).abs(),
            ),
            dim=-1,
        )
    else:
        curvature_x = torch.zeros_like(depth)
    if depth_height >= 3:
        padded_y = F.pad(depth, (0, 0, 1, 1), mode="replicate")
        curvature_y = (padded_y[..., :-2, :] - 2.0 * depth + padded_y[..., 2:, :]).abs()
        curvature_y = torch.cat(
            (
                (depth[..., :1, :] - 2.0 * depth[..., 1:2, :] + depth[..., 2:3, :]).abs(),
                curvature_y[..., 1:-1, :],
                (depth[..., -3:-2, :] - 2.0 * depth[..., -2:-1, :] + depth[..., -1:, :]).abs(),
            ),
            dim=-2,
        )
    else:
        curvature_y = torch.zeros_like(depth)
    edge_signal = F.interpolate(
        torch.maximum(curvature_x, curvature_y), size=(height, width),
        mode="bilinear", align_corners=False,
    )
    low_count = torch.zeros_like(total)
    high_count = torch.zeros_like(total)
    low_delta_sum = torch.zeros_like(total)
    high_delta_sum = torch.zeros_like(total)
    low_total = torch.zeros_like(total)
    high_total = torch.zeros_like(total)
    low_sum = torch.zeros_like(total)
    high_sum = torch.zeros_like(total)

    def smoothstep(value: torch.Tensor, edge0: float, edge1: float) -> torch.Tensor:
        t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0)
        return t.square() * (3.0 - 2.0 * t)

    edge_weight = smoothstep(edge_signal, threshold * 0.25, threshold * 0.5)
    # Zero support at +/-1.5 texels makes the moving 3x3 window continuous.
    # Only per-tap tensors are materialized; no full-frame unfold is needed.
    for oy in (-1, 0, 1):
        wy = 1.0 - smoothstep((y - cy - oy).abs(), 0.5, 1.5)
        sy = (cy + oy).clamp(0, depth_height - 1)
        for ox in (-1, 0, 1):
            wx = 1.0 - smoothstep((x - cx - ox).abs(), 0.5, 1.5)
            sx = (cx + ox).clamp(0, depth_width - 1)
            spatial = (wy[:, None] * wx[None, :])[None, None]
            indices = sy[:, None] * depth_width + sx[None, :]
            sample = _gather_depth_samples(depth, indices).reshape(batch, 1, height, width)
            color = flat_guide.gather(
                2, indices.reshape(1, 1, -1).expand(batch, 3, -1)
            ).reshape(batch, 3, height, width)
            difference = rgb - color
            distance = difference.square().mean(dim=1, keepdim=True)
            delta = difference.abs().mean(dim=1, keepdim=True)
            log_weight = torch.where(
                spatial > 0.0,
                spatial.clamp_min(1.0e-30).log() - distance / (2.0 * sigma * sigma),
                -1.0e30,
            )
            new_max = torch.maximum(max_log_weight, log_weight)
            rescale = (max_log_weight - new_max).exp()
            weight = torch.where(spatial > 0.0, (log_weight - new_max).exp(), 0.0)
            total = total * rescale + weight
            guided_sum = guided_sum * rescale + weight * sample
            matching_distance_sum = matching_distance_sum * rescale + weight * distance
            max_log_weight = new_max
            low = 1.0 - smoothstep(sample - bilinear, -0.5 * threshold, 0.5 * threshold)
            high = 1.0 - low
            low_count += spatial * low
            high_count += spatial * high
            low_delta_sum += spatial * low * delta
            high_delta_sum += spatial * high * delta
            low_total = low_total * rescale + weight * low
            high_total = high_total * rescale + weight * high
            low_sum = low_sum * rescale + weight * low * sample
            high_sum = high_sum * rescale + weight * high * sample
            spatial_total += spatial
            depth_sum += spatial * sample
            depth_square_sum += spatial * sample.square()
            color_sum += spatial * color
            color_square_sum += spatial * color.square().mean(dim=1, keepdim=True)
    denominator = spatial_total.clamp_min(1.0e-8)
    depth_variance = (depth_square_sum / denominator - (depth_sum / denominator).square()).clamp_min(0.0)
    color_variance = (
        color_square_sum / denominator - (color_sum / denominator).square().mean(dim=1, keepdim=True)
    ).clamp_min(0.0)
    confidence = smoothstep(depth_variance, (threshold * 0.25) ** 2, (threshold * 0.5) ** 2)
    confidence *= smoothstep(color_variance, 0.005 ** 2, 0.02 ** 2)
    guided = torch.where(total > 1.0e-8, guided_sum / total.clamp_min(1.0e-8), bilinear)
    low_delta = low_delta_sum / low_count.clamp_min(1.0e-8)
    high_delta = high_delta_sum / high_count.clamp_min(1.0e-8)
    prefer_high = smoothstep(low_delta - high_delta, -0.02, 0.02)
    selected = torch.lerp(low_sum / low_total.clamp_min(1.0e-8),
                          high_sum / high_total.clamp_min(1.0e-8), prefer_high)
    support = smoothstep(torch.minimum(low_count, high_count), 0.001, 0.1)
    class_confidence = smoothstep((low_delta - high_delta).abs(), 0.0025, 0.03) * support
    class_confidence *= 1.0 - smoothstep(matching_distance_sum / total.clamp_min(1.0e-8),
                                        0.000025, 0.0004)
    guided = torch.lerp(guided, selected, class_confidence)
    return (bilinear + edge_weight * confidence * (guided - bilinear)).clamp(0.0, 1.0)


def upsample_depth(
    depth: torch.Tensor,
    height: int,
    width: int,
    *,
    rgb: torch.Tensor | None = None,
    mode: DepthUpsampleMode = "bilinear",
    edge_strength: float = 0.35,
    depth_edge_threshold: float = 0.04,
) -> torch.Tensor:
    """Upsample normalized depth to the RGB frame size.

    `bilinear` preserves current behavior. `guided` keeps the same output
    resolution but softly pulls high-gradient RGB edges from a nearest path to
    reduce bleeding around silhouettes without changing inference resolution.
    """

    depth = ensure_b1hw(depth).float()
    antialias_state = _MODEL_DEPTH_ANTIALIAS_STATE.get()
    model_antialias_applied = False
    if antialias_state is not None and antialias_state.strength > 0.0:
        depth = filter_model_depth(
            depth,
            rgb,
            antialias_state.strength,
            edge_threshold=antialias_state.edge_threshold,
        )
        antialias_state.applied = True
        model_antialias_applied = True
    if depth.shape[-2:] == (height, width):
        return depth

    if model_antialias_applied:
        return F.interpolate(depth, size=(height, width), mode="bilinear", align_corners=False)
    if mode == "bilinear":
        return F.interpolate(depth, size=(height, width), mode="bilinear", align_corners=False)
    if mode == "joint_bilateral":
        if rgb is None:
            return F.interpolate(depth, size=(height, width), mode="bilinear", align_corners=False)
        rgb = ensure_bchw(rgb, name="rgb").to(device=depth.device, dtype=torch.float32)
        if rgb.shape[-2:] != (height, width):
            rgb = F.interpolate(rgb, size=(height, width), mode="bilinear", align_corners=False)
        if depth.device.type == "mps":
            try:
                from ._fused_warp_mps import mps_joint_bilateral_upsample

                return mps_joint_bilateral_upsample(
                    depth,
                    rgb,
                    height,
                    width,
                    edge_threshold=depth_edge_threshold,
                )
            except Exception:
                # Preserve the stable MPS fallback rather than dispatching the
                # much slower multi-kernel tensor implementation.
                pass
        if depth.is_cuda:
            try:
                from .triton_runtime import triton_runtime_available

                if triton_runtime_available(depth.device):
                    from ._depth_upsample_triton import joint_bilateral_upsample

                    guide = F.interpolate(
                        rgb[:, :3], size=depth.shape[-2:], mode="bilinear",
                        align_corners=False,
                    )
                    return joint_bilateral_upsample(
                        depth,
                        rgb,
                        guide,
                        height,
                        width,
                        depth_edge_threshold=depth_edge_threshold,
                    )
            except Exception:
                pass
        if depth.device.type in {"cuda", "mps"}:
            return F.interpolate(depth, size=(height, width), mode="bilinear", align_corners=False)
        return _joint_bilateral_upsample(
            depth,
            rgb,
            height,
            width,
            depth_edge_threshold=depth_edge_threshold,
        )
    if mode != "guided":
        raise ValueError(f"unknown depth upsample mode: {mode!r}")
    bilinear = F.interpolate(depth, size=(height, width), mode="bilinear", align_corners=False)
    if rgb is None:
        return bilinear

    rgb = ensure_bchw(rgb, name="rgb").to(device=bilinear.device).float()
    if rgb.shape[-2:] != (height, width):
        rgb = F.interpolate(rgb, size=(height, width), mode="bilinear", align_corners=False)
    rgb = rgb.clamp(0, 1)

    nearest = F.interpolate(depth, size=(height, width), mode="nearest")
    luma = rgb.mean(dim=1, keepdim=True)
    dx = F.pad((luma[..., :, 1:] - luma[..., :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((luma[..., 1:, :] - luma[..., :-1, :]).abs(), (0, 0, 0, 1))
    edge = (dx + dy).clamp(0, 1)
    edge = F.max_pool2d(edge, kernel_size=3, stride=1, padding=1)
    weight = (edge * float(edge_strength)).clamp(0, 1)
    return torch.lerp(bilinear, nearest, weight).clamp(0, 1)
