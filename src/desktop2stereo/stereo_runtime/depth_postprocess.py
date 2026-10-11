from __future__ import annotations

import functools
import logging
import math
import threading
import torch
import torch.nn.functional as F

from .output import ensure_b1hw, ensure_bchw


_LOGGER = logging.getLogger(__name__)
_TRITON_GAUSSIAN_FAILURE_LOCK = threading.Lock()
_TRITON_GAUSSIAN_FAILURE_RECORDED = False


_GLOBAL_GAUSSIAN_MSL = r"""
#include <metal_stdlib>
using namespace metal;
struct DepthGaussianParams { uint width; uint height; uint batches; };
kernel void d2s_depth_global_gaussian(
    device const float* source [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant DepthGaussianParams& p [[buffer(2)]],
    constant float& spacing [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.width || gid.y >= p.height || gid.z >= p.batches) return;
    constexpr float weights[5] = {
        0.15246914f, 0.22184130f, 0.25137912f,
        0.22184130f, 0.15246914f
    };
    float sum = 0.0f;
    for (int dy = -2; dy <= 2; ++dy) {
        float sample_y = clamp(float(gid.y) + float(dy) * spacing, 0.0f, float(p.height - 1));
        uint y0 = uint(floor(sample_y));
        uint y1 = min(y0 + 1u, p.height - 1u);
        float fy = sample_y - float(y0);
        for (int dx = -2; dx <= 2; ++dx) {
            float sample_x = clamp(float(gid.x) + float(dx) * spacing, 0.0f, float(p.width - 1));
            uint x0 = uint(floor(sample_x));
            uint x1 = min(x0 + 1u, p.width - 1u);
            float fx = sample_x - float(x0);
            uint base = gid.z * p.width * p.height;
            float top = mix(source[base + y0 * p.width + x0], source[base + y0 * p.width + x1], fx);
            float bottom = mix(source[base + y1 * p.width + x0], source[base + y1 * p.width + x1], fx);
            sum += mix(top, bottom, fy) * weights[dy + 2] * weights[dx + 2];
        }
    }
    uint index = gid.z * p.width * p.height + gid.y * p.width + gid.x;
    output[index] = clamp(sum, 0.0f, 1.0f);
}
"""


@functools.lru_cache(maxsize=1)
def _mps_depth_gaussian_library():
    return torch.mps.compile_shader(_GLOBAL_GAUSSIAN_MSL)


@functools.lru_cache(maxsize=16)
def _global_gaussian_axis_weights(spacing: float) -> tuple[float, ...]:
    radius = math.ceil(2.0 * spacing)
    axis = [0.0] * (2 * radius + 1)
    sample_weights = [math.exp(-(offset * offset) / 8.0) for offset in range(-2, 3)]
    weight_sum = sum(sample_weights)
    for offset, weight in zip(range(-2, 3), sample_weights):
        position = offset * spacing
        lower = math.floor(position)
        fraction = position - lower
        axis[lower + radius] += weight / weight_sum * (1.0 - fraction)
        if fraction > 0.0:
            axis[lower + radius + 1] += weight / weight_sum * fraction
    return tuple(axis)


@functools.lru_cache(maxsize=16)
def _global_gaussian_kernel(device_type: str, device_index: int | None, dtype: torch.dtype, spacing: float):
    device = torch.device(device_type, device_index) if device_index is not None else torch.device(device_type)
    axis_tensor = torch.tensor(
        _global_gaussian_axis_weights(spacing), dtype=dtype, device=device
    )
    return (axis_tensor[:, None] * axis_tensor[None, :]).view(
        1, 1, len(axis_tensor), len(axis_tensor)
    )


def _depth_gaussian_spacing(height: int, width: int) -> float:
    long_edge = max(int(height), int(width))
    return 1.0 + 0.5 * min(1.0, max(0.0, (long_edge - 1920.0) / 1920.0))


def anti_alias_depth_global_gaussian(depth: torch.Tensor) -> torch.Tensor:
    """Apply one resolution-scaled 5x5, sigma-2 Gaussian to normalized depth."""
    depth = ensure_b1hw(depth).float().contiguous()
    batch, _, height, width = map(int, depth.shape)
    if min(batch, height, width) < 1:
        return depth
    spacing = _depth_gaussian_spacing(height, width)

    if depth.device.type == "mps":
        try:
            output = torch.empty_like(depth)
            params = torch.tensor((width, height, batch), dtype=torch.int32, device="mps")
            sample_spacing = torch.tensor((spacing,), dtype=torch.float32, device="mps")
            _mps_depth_gaussian_library().d2s_depth_global_gaussian(
                depth, output, params, sample_spacing, threads=(width, height, batch)
            )
            return output
        except Exception:
            pass

    if depth.device.type == "cuda":
        try:
            from ._depth_gaussian_triton import gaussian_1

            return gaussian_1(depth, spacing=spacing)
        except Exception as exc:
            global _TRITON_GAUSSIAN_FAILURE_RECORDED
            with _TRITON_GAUSSIAN_FAILURE_LOCK:
                if not _TRITON_GAUSSIAN_FAILURE_RECORDED:
                    _LOGGER.warning(
                        "Triton 5x5 depth Gaussian failed; falling back to Torch: %s",
                        exc,
                    )
                    _TRITON_GAUSSIAN_FAILURE_RECORDED = True

    kernel = _global_gaussian_kernel(
        depth.device.type, depth.device.index, depth.dtype, spacing
    )
    radius = int(kernel.shape[-1]) // 2
    padded = F.pad(depth, (radius, radius, radius, radius), mode="replicate")
    return F.conv2d(padded, kernel).clamp(0.0, 1.0)


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
    global_gaussian: bool = False,
) -> torch.Tensor:
    out = ensure_b1hw(depth).float().clamp(0.0, 1.0)
    if abs(float(depth_pop)) >= 1e-6:
        out = apply_depth_pop(out, depth_pop)
    if guide is None:
        out = anti_alias_depth(out, antialias_strength)
    else:
        out = anti_alias_depth_guided(out, guide, antialias_strength)
    if global_gaussian:
        out = anti_alias_depth_global_gaussian(out)
    return out
