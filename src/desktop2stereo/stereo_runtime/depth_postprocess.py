from __future__ import annotations

import functools
import logging
import math
import threading
from contextlib import contextmanager
from contextvars import ContextVar
import torch
import torch.nn.functional as F

from .output import ensure_b1hw, ensure_bchw


_LOGGER = logging.getLogger(__name__)
_TRITON_GAUSSIAN_FAILURE_LOCK = threading.Lock()
_TRITON_GAUSSIAN_FAILURE_RECORDED = False
_MODEL_DEPTH_AA_APPLIED: ContextVar[bool] = ContextVar(
    "d2s_model_depth_aa_applied", default=False
)


@contextmanager
def model_depth_aa_applied(applied: bool):
    token = _MODEL_DEPTH_AA_APPLIED.set(bool(applied))
    try:
        yield
    finally:
        _MODEL_DEPTH_AA_APPLIED.reset(token)


def output_depth_gaussian_enabled() -> bool:
    return not _MODEL_DEPTH_AA_APPLIED.get()


_GLOBAL_GAUSSIAN_MSL = r"""
#include <metal_stdlib>
using namespace metal;
struct DepthGaussianParams { uint width; uint height; uint batches; };
kernel void d2s_depth_global_gaussian(
    device const float* source [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant DepthGaussianParams& p [[buffer(2)]],
    constant float& spacing [[buffer(3)]],
    constant float& edge_threshold [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.width || gid.y >= p.height || gid.z >= p.batches) return;
    constexpr float sample_weights[5] = {
        0.15246914f, 0.22184130f, 0.25137912f,
        0.22184130f, 0.15246914f
    };
    float axis_weights[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int offset = -2; offset <= 2; ++offset) {
        float position = float(offset) * spacing;
        int lower = int(floor(position));
        float fraction = position - float(lower);
        float weight = sample_weights[offset + 2];
        axis_weights[lower + 3] += weight * (1.0f - fraction);
        if (fraction > 0.0f) axis_weights[lower + 4] += weight * fraction;
    }
    uint base = gid.z * p.width * p.height;
    uint center_index = base + gid.y * p.width + gid.x;
    float center = source[center_index];
    float sum = 0.0f;
    float weight_sum = 0.0f;
    for (int dy = -3; dy <= 3; ++dy) {
        float wy = axis_weights[dy + 3];
        if (wy <= 0.0f) continue;
        uint yy = uint(clamp(int(gid.y) + dy, 0, int(p.height) - 1));
        for (int dx = -3; dx <= 3; ++dx) {
            float wx = axis_weights[dx + 3];
            if (wx <= 0.0f) continue;
            uint xx = uint(clamp(int(gid.x) + dx, 0, int(p.width) - 1));
            float sample = source[base + yy * p.width + xx];
            float edge_weight = 1.0f;
            if (edge_threshold >= 0.0f) {
                float distance = abs(sample - center);
                edge_weight = edge_threshold > 1.0e-6f
                    ? 1.0f - smoothstep(0.5f * edge_threshold,
                                        edge_threshold, distance)
                    : float(distance <= 1.0e-6f);
            }
            float weight = wx * wy * edge_weight;
            sum += sample * weight;
            weight_sum += weight;
        }
    }
    uint index = gid.z * p.width * p.height + gid.y * p.width + gid.x;
    output[index] = clamp(sum / max(weight_sum, 1.0e-8f), 0.0f, 1.0f);
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


def anti_alias_depth_global_gaussian(
    depth: torch.Tensor,
    *,
    edge_threshold: float | None = None,
) -> torch.Tensor:
    """Apply one resolution-scaled 5x5, sigma-2 Gaussian to normalized depth."""
    return _anti_alias_depth_global_gaussian(depth, edge_threshold=edge_threshold)


def _edge_aware_gaussian_torch(
    depth: torch.Tensor,
    kernel: torch.Tensor,
    edge_threshold: float,
) -> torch.Tensor:
    batch, _, height, width = map(int, depth.shape)
    kernel_size = int(kernel.shape[-1])
    radius = kernel_size // 2
    padded = F.pad(depth, (radius, radius, radius, radius), mode="replicate")
    weights = kernel.view(1, 1, 1, 1, kernel_size, kernel_size)
    max_patch_elements = 16 * 1024 * 1024
    rows_per_chunk = max(
        1,
        min(height, max_patch_elements // max(1, batch * width * kernel_size * kernel_size)),
    )
    output_rows = []
    for row_start in range(0, height, rows_per_chunk):
        row_end = min(height, row_start + rows_per_chunk)
        row_count = row_end - row_start
        tile = padded[..., row_start : row_end + 2 * radius, :]
        patches = tile.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
        center = depth[..., row_start:row_end, :].unsqueeze(-1).unsqueeze(-1)
        distance = (patches - center).abs()
        threshold = max(0.0, float(edge_threshold))
        if threshold > 1e-6:
            transition = ((distance - 0.5 * threshold) / (0.5 * threshold)).clamp(0.0, 1.0)
            edge_weight = 1.0 - transition.square() * (3.0 - 2.0 * transition)
        else:
            edge_weight = (distance <= 1e-6).to(depth.dtype)
        included_weights = edge_weight * weights
        filtered = (patches * included_weights).sum(dim=(-1, -2)) / included_weights.sum(
            dim=(-1, -2)
        ).clamp_min(1.0e-8)
        output_rows.append(filtered)
    return torch.cat(output_rows, dim=-2).clamp(0.0, 1.0)


def _anti_alias_depth_global_gaussian(
    depth: torch.Tensor,
    *,
    edge_threshold: float | None = None,
) -> torch.Tensor:
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
            sample_threshold = torch.tensor(
                (-1.0 if edge_threshold is None else float(edge_threshold),),
                dtype=torch.float32,
                device="mps",
            )
            _mps_depth_gaussian_library().d2s_depth_global_gaussian(
                depth,
                output,
                params,
                sample_spacing,
                sample_threshold,
                threads=(width, height, batch),
            )
            return output
        except Exception:
            pass

    if depth.device.type == "cuda":
        try:
            from ._depth_gaussian_triton import gaussian_1

            return gaussian_1(depth, spacing=spacing, edge_threshold=edge_threshold)
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
    if edge_threshold is not None:
        return _edge_aware_gaussian_torch(depth, kernel, float(edge_threshold))
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


@functools.lru_cache(maxsize=32)
def _model_gaussian_kernel(
    device_type: str,
    device_index: int | None,
    dtype: torch.dtype,
    strength: float,
):
    device = torch.device(device_type, device_index) if device_index is not None else torch.device(device_type)
    kernel_size = int(3 * strength) | 1
    sigma = max(0.5 * strength, 1e-4)
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    kernel = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    return kernel / kernel.sum().clamp_min(1e-6)


def anti_alias_depth(
    depth: torch.Tensor,
    strength: float,
    *,
    edge_threshold: float | None = None,
) -> torch.Tensor:
    depth = ensure_b1hw(depth).float()
    if strength <= 0.0:
        return depth
    strength = float(strength)
    kernel_size = int(3 * strength) | 1
    if kernel_size < 3:
        return depth
    kernel = _model_gaussian_kernel(
        depth.device.type, depth.device.index, depth.dtype, strength
    )
    if edge_threshold is not None:
        kernel_2d = (kernel[:, None] * kernel[None, :]).view(
            1, 1, kernel_size, kernel_size
        )
        return _edge_aware_gaussian_torch(depth, kernel_2d, float(edge_threshold))
    radius = kernel_size // 2
    horizontal = F.conv2d(
        F.pad(depth, (radius, radius, 0, 0), mode="replicate"),
        kernel.view(1, 1, 1, -1),
    )
    out = F.conv2d(
        F.pad(horizontal, (0, 0, radius, radius), mode="replicate"),
        kernel.view(1, 1, -1, 1),
    )
    return out.clamp(0.0, 1.0)


def anti_alias_depth_guided(
    depth: torch.Tensor,
    guide: torch.Tensor,
    strength: float,
    sigma_color: float = 0.1,
    max_keep: float = 0.85,
    depth_edge_threshold: float | None = None,
) -> torch.Tensor:
    """Edge-aware antialiasing guided by the RGB frame (dispatch-cheap form).

    arXiv 1911.07036 stage-1 intent -- denoise depth WITHOUT widening
    object edges -- realized as an edge-masked blend instead of a
    per-neighbour joint bilateral: the bilateral issued ~60 tiny kernels
    per frame and drained the MPS dispatch budget (docs/33 "MPS drain
    wall", ~1ms per submit cycle), collapsing Local Viewer FPS. When a
    depth threshold is supplied, the Gaussian omits cross-edge neighbors
    and renormalizes the remaining weights. A 3x3 high-pass of the guide
    yields an edge-likeness map, and the result blends back toward the raw
    depth by ``max_keep * edge``. Flat regions denoise fully without adding
    an intermediate depth surface at object boundaries.
    """
    depth = ensure_b1hw(depth).float()
    guide = ensure_bchw(guide, name="guide").float()
    if strength <= 0.0:
        return depth
    smooth = anti_alias_depth(
        depth,
        strength,
        edge_threshold=depth_edge_threshold,
    )
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
    edge_threshold: float | None = None,
) -> torch.Tensor:
    out = ensure_b1hw(depth).float().clamp(0.0, 1.0)
    if abs(float(depth_pop)) >= 1e-6:
        out = apply_depth_pop(out, depth_pop)
    if guide is None:
        out = anti_alias_depth(
            out,
            antialias_strength,
            edge_threshold=edge_threshold,
        )
    else:
        out = anti_alias_depth_guided(
            out,
            guide,
            antialias_strength,
            depth_edge_threshold=edge_threshold,
        )
    if global_gaussian and output_depth_gaussian_enabled():
        # Soften the silhouette before reprojection. The configured model-grid
        # filter keeps its depth-edge protection independently of this final pass.
        out = _anti_alias_depth_global_gaussian(out)
    return out
