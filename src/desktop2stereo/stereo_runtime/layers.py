from __future__ import annotations

import torch
import torch.nn.functional as F

from .output import ensure_b1hw

_LAYER_CENTERS_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}


def _layer_centers(layers: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (layers, str(device), dtype)
    centers = _LAYER_CENTERS_CACHE.get(key)
    if centers is None:
        centers = torch.linspace(0.0, 1.0, layers, device=device, dtype=dtype).view(1, layers, 1, 1)
        _LAYER_CENTERS_CACHE[key] = centers
    return centers


def make_depth_layers(
    depth: torch.Tensor,
    layers: int = 2,
    softness: float = 0.08,
    *,
    rgb: torch.Tensor | None = None,
    edge_threshold: float = 0.04,
) -> torch.Tensor:
    if layers < 1:
        raise ValueError("layers must be >= 1")
    depth = ensure_b1hw(depth).clamp(0, 1)
    if layers == 1:
        return torch.ones_like(depth)

    centers = _layer_centers(layers, depth.device, depth.dtype)
    depth_blhw = depth.expand(-1, layers, -1, -1)
    weights = torch.exp(-((depth_blhw - centers) ** 2) / max(softness, 1e-4))
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    if rgb is None or layers != 2:
        return weights

    # Keep a foreground/background depth step from becoming a third layer.
    # This is the tensor fallback equivalent of the native RGB-guided depth
    # classifier; flat regions keep the Gaussian layer weights above.
    rgb = rgb[:, :3].float().to(device=depth.device)
    if rgb.shape[-2:] != depth.shape[-2:]:
        rgb = F.interpolate(rgb, size=depth.shape[-2:], mode="bilinear", align_corners=False)
    luminance = rgb[:, 0:1] * 0.299 + rgb[:, 1:2] * 0.587 + rgb[:, 2:3] * 0.114
    padded_depth = F.pad(depth, (1, 1, 1, 1), mode="replicate")
    left = padded_depth[..., 1:-1, :-2]
    right = padded_depth[..., 1:-1, 2:]
    up = padded_depth[..., :-2, 1:-1]
    down = padded_depth[..., 2:, 1:-1]
    minimum = torch.minimum(torch.minimum(left, right), torch.minimum(up, down))
    maximum = torch.maximum(torch.maximum(left, right), torch.maximum(up, down))
    midpoint = (minimum + maximum) * 0.5
    low_left, low_right = left < midpoint, right < midpoint
    low_up, low_down = up < midpoint, down < midpoint
    padded_luma = F.pad(luminance, (1, 1, 1, 1), mode="replicate")
    ll = padded_luma[..., 1:-1, :-2]
    lr = padded_luma[..., 1:-1, 2:]
    lu = padded_luma[..., :-2, 1:-1]
    ld = padded_luma[..., 2:, 1:-1]
    low_count = sum(mask.float() for mask in (low_left, low_right, low_up, low_down))
    high_count = 4.0 - low_count
    low_luma = (
        ll * low_left.float() + lr * low_right.float()
        + lu * low_up.float() + ld * low_down.float()
    ) / low_count.clamp_min(1.0)
    high_luma = (
        ll * (~low_left).float() + lr * (~low_right).float()
        + lu * (~low_up).float() + ld * (~low_down).float()
    ) / high_count.clamp_min(1.0)
    use_class = (
        (maximum - minimum >= float(edge_threshold))
        & (low_count > 0.0)
        & (high_count > 0.0)
        & ((high_luma - low_luma).abs() >= 0.02)
    )
    choose_high = (luminance - high_luma).abs() <= (luminance - low_luma).abs()
    hard = torch.cat((~choose_high, choose_high), dim=1).to(weights.dtype)
    return torch.where(use_class.expand_as(weights), hard, weights)


def composite_layers(warped: list[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
    if not warped:
        raise ValueError("warped layer list is empty")
    weights = weights[:, : len(warped)]
    if len(warped) == 1:
        return warped[0] * weights[:, 0:1]
    if len(warped) == 2:
        return warped[0] * weights[:, 0:1] + warped[1] * weights[:, 1:2]
    out = torch.zeros_like(warped[0])
    for idx, layer in enumerate(warped):
        out.addcmul_(layer, weights[:, idx : idx + 1])
    return out


def depth_edges(depth: torch.Tensor, threshold: float = 0.04) -> torch.Tensor:
    depth = ensure_b1hw(depth).float()
    edges = torch.zeros_like(depth)
    edges[..., :, :-1].add_((depth[..., :, 1:] - depth[..., :, :-1]).abs())
    edges[..., :-1, :].add_((depth[..., 1:, :] - depth[..., :-1, :]).abs())
    return (edges > threshold).float()
