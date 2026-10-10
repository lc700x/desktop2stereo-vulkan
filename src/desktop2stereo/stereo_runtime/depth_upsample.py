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
    applied: bool = False


_MODEL_DEPTH_ANTIALIAS_STATE: ContextVar[_ModelDepthAntialiasState | None] = ContextVar(
    "d2s_model_depth_antialias_state", default=None
)


@contextmanager
def model_depth_antialiasing(strength: float):
    """Apply the configured spatial depth filter before provider upsampling.

    Providers share ``upsample_depth`` but own their model execution. A
    context-local setting lets the runtime enable identical postprocessing for
    PyTorch, TensorRT, ONNX, ROCm, and MPS providers without changing their
    inference code or constructor contracts.
    """
    state = _ModelDepthAntialiasState(max(0.0, float(strength)))
    token = _MODEL_DEPTH_ANTIALIAS_STATE.set(state)
    try:
        yield state
    finally:
        _MODEL_DEPTH_ANTIALIAS_STATE.reset(token)


def filter_model_depth(
    depth: torch.Tensor,
    rgb: torch.Tensor | None,
    strength: float,
) -> torch.Tensor:
    if strength <= 0.0:
        return depth
    # v2.5 maps the 0..2 GUI choice to AA_STRENGTH=0..4. Keep that mapping
    # while filtering on the model grid instead of the much larger RGB grid.
    model_strength = min(4.0, 2.0 * float(strength))
    from .depth_postprocess import anti_alias_depth, anti_alias_depth_guided

    if rgb is None:
        return anti_alias_depth(depth, model_strength)
    guide = ensure_bchw(rgb, name="rgb").to(device=depth.device, dtype=torch.float32)
    guide = F.interpolate(
        guide[:, :3],
        size=depth.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)
    # The smooth Gaussian removes depth-grid stair steps; the RGB edge mask
    # retains 35% of the original depth at strong image edges to limit contour
    # widening. Both constants are mirrored by the native Metal kernel.
    return anti_alias_depth_guided(
        depth,
        guide,
        model_strength,
        sigma_color=0.1,
        max_keep=0.35,
    )


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
    """Bilinear depth interpolation with same-frame RGB edge validation."""
    batch, _, depth_height, depth_width = depth.shape
    rgb = ensure_bchw(rgb, name="rgb").to(device=depth.device, dtype=torch.float32)
    if rgb.shape[0] == 1 and batch != 1:
        rgb = rgb.expand(batch, -1, -1, -1)
    if rgb.shape[0] != batch:
        raise ValueError("rgb and depth batch dimensions must match")
    if rgb.shape[-2:] != (height, width):
        rgb = F.interpolate(rgb, size=(height, width), mode="bilinear", align_corners=False)
    rgb = rgb[:, :3].float().clamp(0.0, 1.0)

    y = ((torch.arange(height, device=depth.device, dtype=torch.float32) + 0.5)
         * (float(depth_height) / float(height)) - 0.5).clamp(0, depth_height - 1)
    x = ((torch.arange(width, device=depth.device, dtype=torch.float32) + 0.5)
         * (float(depth_width) / float(width)) - 0.5).clamp(0, depth_width - 1)
    y0, x0 = y.floor().long(), x.floor().long()
    y1 = (y0 + 1).clamp_max(depth_height - 1)
    x1 = (x0 + 1).clamp_max(depth_width - 1)
    fy, fx = y - y0, x - x0

    i00 = y0[:, None] * depth_width + x0[None, :]
    i10 = y0[:, None] * depth_width + x1[None, :]
    i01 = y1[:, None] * depth_width + x0[None, :]
    i11 = y1[:, None] * depth_width + x1[None, :]
    d00, d10 = _gather_depth_samples(depth, i00), _gather_depth_samples(depth, i10)
    d01, d11 = _gather_depth_samples(depth, i01), _gather_depth_samples(depth, i11)
    fx, fy = fx.view(1, 1, 1, width), fy.view(1, 1, height, 1)
    spatial = (
        (1.0 - fy) * (1.0 - fx),
        (1.0 - fy) * fx,
        fy * (1.0 - fx),
        fy * fx,
    )

    # Compare against color represented at each model-depth texel centre.
    # Bilinear downsampling is supported by MPS for arbitrary display sizes.
    low_rgb = F.interpolate(
        rgb, size=(depth_height, depth_width), mode="bilinear", align_corners=False
    )
    flat_rgb = rgb.reshape(batch, 3, height * width)
    flat_low_rgb = low_rgb.reshape(batch, 3, depth_height * depth_width)
    indices = (i00, i10, i01, i11)
    samples = (d00, d10, d01, d11)
    weights = []
    color_deltas = []
    for index, spatial_weight in zip(indices, spatial):
        guide = flat_low_rgb.gather(
            2, index.reshape(1, 1, -1).expand(batch, 3, -1)
        )
        color_delta = (flat_rgb - guide).abs().mean(dim=1, keepdim=True)
        color_deltas.append(color_delta)
        weights.append(
            spatial_weight.reshape(1, 1, -1)
            * torch.exp(-color_delta / max(float(color_sigma), 1e-4))
        )

    total = sum(weights)
    minimum = torch.minimum(torch.minimum(d00, d10), torch.minimum(d01, d11))
    maximum = torch.maximum(torch.maximum(d00, d10), torch.maximum(d01, d11))
    guided = sum(value * weight for value, weight in zip(samples, weights)) / total.clamp_min(1e-8)

    # At a depth discontinuity, averaging the two sides creates a third
    # surface.  That surface receives its own disparity during DIBR and shows
    # up as a second contour.  Use the high-resolution RGB as a classifier and
    # normalize only the taps belonging to the winning depth class.  Smooth
    # regions retain the original joint-bilateral interpolation.
    midpoint = (minimum + maximum) * 0.5
    high_masks = [value >= midpoint for value in samples]
    color_stack = torch.stack(color_deltas, dim=0)
    high_stack = torch.stack([mask.float() for mask in high_masks], dim=0)
    nearest_color = color_stack.argmin(dim=0, keepdim=True)
    choose_high = torch.gather(high_stack, 0, nearest_color).squeeze(0).bool()
    selected_weights = [
        weight * torch.where(choose_high, mask, ~mask).float()
        for weight, mask in zip(weights, high_masks)
    ]
    selected_total = sum(selected_weights)
    selected = sum(
        value * weight for value, weight in zip(samples, selected_weights)
    ) / selected_total.clamp_min(1e-8)
    color_contrast = color_stack.amax(dim=0) - color_stack.amin(dim=0)
    class_edge = (maximum - minimum) >= float(depth_edge_threshold)
    use_class = (
        class_edge
        & (selected_total > 1e-8)
        & (color_contrast >= 0.005)
    )
    guided = torch.where(use_class, selected, guided)
    has_depth_edge = (maximum - minimum) >= float(depth_edge_threshold)
    bilinear = F.interpolate(depth, size=(height, width), mode="bilinear", align_corners=False)
    use_guided = (has_depth_edge & (total > 1e-8)).reshape(batch, 1, height, width)
    return torch.where(
        use_guided, guided.reshape(batch, 1, height, width), bilinear
    ).clamp(0, 1)


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
    if antialias_state is not None and antialias_state.strength > 0.0:
        depth = filter_model_depth(depth, rgb, antialias_state.strength)
        antialias_state.applied = True
    if depth.shape[-2:] == (height, width):
        return depth

    bilinear = F.interpolate(depth, size=(height, width), mode="bilinear", align_corners=False)
    if mode == "bilinear":
        return bilinear
    if mode == "joint_bilateral":
        if rgb is None:
            return bilinear
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
                return bilinear
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
                return bilinear
            return bilinear
        return _joint_bilateral_upsample(
            depth,
            rgb,
            height,
            width,
            depth_edge_threshold=depth_edge_threshold,
        )
    if mode != "guided":
        raise ValueError(f"unknown depth upsample mode: {mode!r}")
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
