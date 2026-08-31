from __future__ import annotations

import torch
import torch.nn.functional as F

from .output import ensure_b1hw, ensure_bchw


def apply_depth_pop(depth: torch.Tensor, depth_pop: float, mid: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    depth = ensure_b1hw(depth).float().clamp(0.0, 1.0)
    if abs(depth_pop) < eps:
        return depth
    if depth_pop <= -1.0:
        raise ValueError("depth_pop must be greater than -1.0")
    if depth_pop < 0.0:
        strength = min(1.0, max(0.0, -float(depth_pop)))
        centered = depth - float(mid)
        # Negative Depth Pop is a realtime compression control. Avoid torch.pow here:
        # high exponents near -1.0 are extremely slow on 4K CUDA tensors.
        compressed = centered * (1.0 - strength)
        return (float(mid) + compressed).clamp(0.0, 1.0)
    exponent = 1.0 / (1.0 + float(depth_pop))
    centered = depth - float(mid)
    out = float(mid) + torch.sign(centered) * torch.abs(centered).pow(exponent)
    return out.clamp(0.0, 1.0)


def anti_alias_depth(depth: torch.Tensor, strength: float) -> torch.Tensor:
    depth = ensure_b1hw(depth).float()
    if strength <= 0.0:
        return depth
    kernel_size = int(3 * float(strength)) | 1
    if kernel_size < 3:
        return depth
    sigma = max(0.5 * float(strength), 1e-4)
    coords = torch.arange(kernel_size, device=depth.device, dtype=depth.dtype) - kernel_size // 2
    kernel = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum().clamp_min(1e-6)
    out = F.conv2d(depth, kernel.view(1, 1, 1, -1), padding=(0, kernel_size // 2))
    out = F.conv2d(out, kernel.view(1, 1, -1, 1), padding=(kernel_size // 2, 0))
    return out.clamp(0.0, 1.0)


def anti_alias_depth_guided(
    depth: torch.Tensor,
    guide: torch.Tensor,
    strength: float,
    sigma_color: float = 0.1,
    max_keep: float = 0.85,
) -> torch.Tensor:
    """Edge-aware antialiasing guided by the RGB frame (dispatch-cheap form).

    arXiv 1911.07036 stage-1 intent -- denoise depth WITHOUT widening
    object edges -- realized as an edge-masked blend instead of a
    per-neighbour joint bilateral: the bilateral issued ~60 tiny kernels
    per frame and drained the MPS dispatch budget (docs/33 "MPS drain
    wall", ~1ms per submit cycle), collapsing Local Viewer FPS. Here the
    plain gaussian runs once (2 convs), a 3x3 high-pass of the guide
    yields an edge-likeness map, and the result blends back toward the
    raw depth by ``max_keep * edge``. Flat regions denoise fully; object
    edges keep most of their sharpness but never all of it, bounding the
    disparity step so the warp does not tear into double contours.
    """
    depth = ensure_b1hw(depth).float()
    guide = ensure_bchw(guide, name="guide").float()
    if strength <= 0.0:
        return depth
    smooth = anti_alias_depth(depth, strength)
    blur = F.avg_pool2d(guide, kernel_size=3, stride=1, padding=1)
    grad = (guide - blur).abs().sum(dim=1, keepdim=True)
    edge = (grad / (sigma_color * float(guide.shape[1]))).clamp(0.0, 1.0)
    w = max_keep * edge
    return (w * depth + (1.0 - w) * smooth).clamp(0.0, 1.0)


def postprocess_depth(
    depth: torch.Tensor,
    *,
    depth_pop: float = 0.0,
    antialias_strength: float = 0.0,
    guide: torch.Tensor | None = None,
) -> torch.Tensor:
    out = ensure_b1hw(depth).float().clamp(0.0, 1.0)
    if abs(float(depth_pop)) >= 1e-6:
        out = apply_depth_pop(out, depth_pop)
    if guide is None:
        out = anti_alias_depth(out, antialias_strength)
    else:
        out = anti_alias_depth_guided(out, guide, antialias_strength)
    return out
