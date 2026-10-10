"""Display-only FXAA on completed eye pixels; no depth or inference changes.

The native Metal, Vulkan, and tensor kernels use the same bounded endpoint
search. Edge detection uses encoded RGB; fractional coverage blends linear RGB.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

_LOGGER = logging.getLogger(__name__)
_FAILED_BACKENDS: set[tuple[str, str]] = set()
_SEARCH_STEPS = (1.5, 2.0, 2.0, 2.0, 4.0, 8.0)


def _decode_srgb(value: torch.Tensor) -> torch.Tensor:
    return torch.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055).clamp_min(0).pow(2.4))


def _encode_srgb(value: torch.Tensor) -> torch.Tensor:
    return torch.where(value <= 0.0031308, value * 12.92, 1.055 * value.clamp_min(0).pow(1.0 / 2.4) - 0.055)


def fxaa_reference(image: torch.Tensor) -> torch.Tensor:
    """Device-independent reference; input is BCHW encoded RGB or RGBA."""
    if image.ndim != 4 or image.shape[1] not in (3, 4):
        raise ValueError("FXAA expects BCHW RGB or RGBA")
    batch, _, height, width = image.shape
    source = image.float() / 255.0 if image.dtype == torch.uint8 else image.float()
    rgb = source[:, :3].clamp(0.0, 1.0)
    luma = rgb[:, 0:1] * 0.299 + rgb[:, 1:2] * 0.587 + rgb[:, 2:3] * 0.114
    padded = F.pad(luma, (1, 1, 1, 1), mode="replicate")
    center = luma
    north, south = padded[..., :-2, 1:-1], padded[..., 2:, 1:-1]
    west, east = padded[..., 1:-1, :-2], padded[..., 1:-1, 2:]
    nw, ne = padded[..., :-2, :-2], padded[..., :-2, 2:]
    sw, se = padded[..., 2:, :-2], padded[..., 2:, 2:]
    minimum = torch.minimum(center, torch.minimum(torch.minimum(north, south), torch.minimum(west, east)))
    maximum = torch.maximum(center, torch.maximum(torch.maximum(north, south), torch.maximum(west, east)))
    contrast = maximum - minimum
    edge = contrast >= torch.maximum(maximum * 0.125, maximum.new_tensor(0.0312)) + 1e-5
    horizontal_metric = (
        2.0 * (north + south - 2.0 * center).abs()
        + (nw + sw - 2.0 * west).abs() + (ne + se - 2.0 * east).abs()
    )
    vertical_metric = (
        2.0 * (west + east - 2.0 * center).abs()
        + (nw + ne - 2.0 * north).abs() + (sw + se - 2.0 * south).abs()
    )
    # Break near-equal edge-direction metrics consistently across float32
    # backends instead of letting a one-ULP difference rotate the resolve.
    horizontal = horizontal_metric >= vertical_metric - 1e-6
    first = torch.where(horizontal, north, west)
    second = torch.where(horizontal, south, east)
    gradient_first, gradient_second = (first - center).abs(), (second - center).abs()
    first_side = gradient_first >= gradient_second - 1e-6
    normal_sign = torch.where(first_side, -1.0, 1.0)
    # Keep endpoint decisions stable across normalized-grid (Torch) and direct
    # pixel-coordinate (Metal/Vulkan) samplers at exact quarter-gradient ties.
    # Keep endpoint classification stable across GPU implementations. The
    # extra 1e-5 margin is much larger than float32 interpolation error while
    # remaining visually negligible at this luma scale.
    threshold = torch.maximum(gradient_first, gradient_second) * 0.25 - 1.1e-4
    edge_luma = (center + torch.where(first_side, first, second)) * 0.5

    yy, xx = torch.meshgrid(
        torch.arange(height, device=image.device, dtype=torch.float32),
        torch.arange(width, device=image.device, dtype=torch.float32), indexing="ij",
    )
    xx, yy = xx[None, None], yy[None, None]
    normal_x, normal_y = (~horizontal).float() * normal_sign, horizontal.float() * normal_sign
    tangent_x, tangent_y = horizontal.float(), (~horizontal).float()
    edge_x, edge_y = xx + normal_x * 0.5, yy + normal_y * 0.5

    def sample(texture: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Pixel-space gathers match the Metal/GLSL/Triton kernels and avoid
        # normalized-grid rounding changing FXAA endpoint decisions.
        x, y = x.expand(batch, 1, height, width), y.expand(batch, 1, height, width)
        x0f, y0f = torch.floor(x), torch.floor(y)
        fx, fy = x - x0f, y - y0f
        x0 = x0f.to(torch.long).clamp(0, width - 1)
        x1 = (x0f.to(torch.long) + 1).clamp(0, width - 1)
        y0 = y0f.to(torch.long).clamp(0, height - 1)
        y1 = (y0f.to(torch.long) + 1).clamp(0, height - 1)
        flat = texture.reshape(batch, texture.shape[1], height * width)

        def gather(px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
            index = (py * width + px).reshape(batch, 1, height * width)
            index = index.expand(-1, texture.shape[1], -1)
            return torch.gather(flat, 2, index).reshape(batch, texture.shape[1], height, width)

        a, b = gather(x0, y0), gather(x1, y0)
        c, d = gather(x0, y1), gather(x1, y1)
        top = a * (1.0 - fx) + b * fx
        bottom = c * (1.0 - fx) + d * fx
        return top * (1.0 - fy) + bottom * fy

    distance_neg, distance_pos = torch.ones_like(center), torch.ones_like(center)
    delta_neg = sample(luma, edge_x - tangent_x, edge_y - tangent_y) - edge_luma
    delta_pos = sample(luma, edge_x + tangent_x, edge_y + tangent_y) - edge_luma
    done_neg, done_pos = delta_neg.abs() >= threshold, delta_pos.abs() >= threshold
    for step in _SEARCH_STEPS:
        distance_neg = distance_neg + (~done_neg).float() * step
        distance_pos = distance_pos + (~done_pos).float() * step
        next_neg = sample(luma, edge_x - tangent_x * distance_neg, edge_y - tangent_y * distance_neg) - edge_luma
        next_pos = sample(luma, edge_x + tangent_x * distance_pos, edge_y + tangent_y * distance_pos) - edge_luma
        delta_neg, delta_pos = torch.where(done_neg, delta_neg, next_neg), torch.where(done_pos, delta_pos, next_pos)
        done_neg, done_pos = done_neg | (delta_neg.abs() >= threshold), done_pos | (delta_pos.abs() >= threshold)
    nearer_neg = distance_neg <= distance_pos
    endpoint_delta = torch.where(nearer_neg, delta_neg, delta_pos)
    valid_span = (endpoint_delta < 0.0) != ((center - edge_luma) < 0.0)
    coverage = 0.5 - torch.minimum(distance_neg, distance_pos) / (distance_neg + distance_pos).clamp_min(1e-6)
    coverage = torch.where(valid_span, coverage, 0.0)
    neighborhood = (2.0 * (north + south + west + east) + nw + ne + sw + se) / 12.0
    subpixel = ((neighborhood - center).abs() / contrast.clamp_min(1e-6)).clamp(0.0, 1.0)
    # Slightly over-unity coverage softens stair steps from the 336-grid depth
    # silhouette; the same factor is mirrored by the GPU implementations.
    subpixel = (subpixel * subpixel * (3.0 - 2.0 * subpixel)).square() * 1.25
    offset = torch.maximum(coverage, subpixel)
    covered = sample(_decode_srgb(rgb), xx + normal_x * offset, yy + normal_y * offset)
    result = torch.where(edge.expand_as(rgb), _encode_srgb(covered), source[:, :3])
    if image.shape[1] == 4:
        result = torch.cat((result, source[:, 3:4]), dim=1)
    if image.dtype == torch.uint8:
        # Match Metal/Triton's positive half-up UNORM conversion.
        return torch.floor(result * 255.0 + 0.5).clamp(0, 255).to(torch.uint8)
    return result.to(image.dtype)


@functools.lru_cache(maxsize=1)
def _mps_library():
    header = Path(__file__).parent / "providers/apple/native/sbs_fxaa_msl.h"
    text = header.read_text()
    shader = text.split('R"MSL(', 1)[1].rsplit(')MSL"', 1)[0]
    return torch.mps.compile_shader(shader)


@functools.lru_cache(maxsize=16)
def _mps_params(width: int, height: int, channels: int, layout: int, eye_width: int, batch: int):
    return torch.tensor((width, height, channels, layout, eye_width, batch), dtype=torch.int32, device="mps")


def _mps_filter(source: torch.Tensor, eye_width: int, *, half_sbs: bool = False) -> torch.Tensor:
    """Use the same bounded FXAA resolve as Vulkan, Triton, and Torch."""
    batch, channels, height, width = map(int, source.shape)
    output_width = width // 2 if half_sbs else width
    output = torch.empty((batch, channels, height, output_width), dtype=source.dtype, device="mps")
    layout = 2 if source.dtype == torch.uint8 else 1
    params = _mps_params(width, height, channels, layout, eye_width, batch)
    library = _mps_library()
    kernel = library.d2s_sbs_fxaa_half if half_sbs else library.d2s_sbs_fxaa
    kernel(source, source, output, output, params, threads=(output_width, height, batch))
    return output


def _filter(image: torch.Tensor, eye_width: int) -> torch.Tensor:
    source = image.contiguous()
    backend = source.device.type
    key = (backend, str(source.device))
    if key not in _FAILED_BACKENDS and source.dtype in (torch.float32, torch.uint8):
        try:
            if backend == "mps":
                return _mps_filter(source, eye_width)
            if backend == "cuda":
                from ._display_antialias_triton import fxaa
                return fxaa(source, eye_width)
        except Exception as exc:
            # A display-kernel/compiler failure must not take down inference.
            _FAILED_BACKENDS.add(key)
            _LOGGER.warning("Display FXAA %s kernel unavailable; using tensor fallback: %s", backend, exc)
    if eye_width and source.shape[-1] > eye_width:
        return torch.cat((fxaa_reference(source[..., :eye_width]), fxaa_reference(source[..., eye_width:])), dim=-1)
    return fxaa_reference(source)


def antialias_eye(image: torch.Tensor) -> torch.Tensor:
    """Filter one completed eye, retaining its shape, dtype, and alpha."""
    channels_last = image.ndim == 3 and image.shape[0] not in (3, 4) and image.shape[-1] in (3, 4)
    tensor = image.permute(2, 0, 1) if channels_last else image
    single = tensor.ndim == 3
    source = tensor.unsqueeze(0) if single else tensor
    if source.ndim != 4 or source.shape[1] not in (3, 4):
        return image
    result = _filter(source, 0)
    result = result.squeeze(0) if single else result
    return result.permute(1, 2, 0).contiguous() if channels_last else result


def antialias_sbs(image: torch.Tensor, output_format: str) -> torch.Tensor:
    """Filter complete packed eyes without ever sampling across their seam."""
    if output_format not in {"half_sbs", "full_sbs"}:
        return image
    single = image.ndim == 3
    source = image.unsqueeze(0) if single else image
    if source.ndim != 4 or source.shape[1] not in (3, 4):
        return image
    result = _filter(source, int(source.shape[-1]) // 2)
    return result.squeeze(0) if single else result


def antialias_sbs_half(image: torch.Tensor) -> torch.Tensor:
    """Antialias each full eye, then area-reduce in linear light to Half-SBS."""
    channels_last = image.ndim == 3 and image.shape[0] not in (3, 4) and image.shape[-1] in (3, 4)
    tensor = image.permute(2, 0, 1) if channels_last else image
    single = tensor.ndim == 3
    source = tensor.unsqueeze(0) if single else tensor
    if source.ndim != 4 or source.shape[1] not in (3, 4) or source.shape[-1] % 2:
        return image
    source = source.contiguous()
    batch, channels, height, width = map(int, source.shape)
    half = None
    key = (source.device.type, str(source.device))
    if (source.device.type == "mps" and key not in _FAILED_BACKENDS
            and source.dtype in (torch.float32, torch.uint8)):
        try:
            half = _mps_filter(source, width // 2, half_sbs=True)
        except Exception as exc:
            _FAILED_BACKENDS.add(key)
            _LOGGER.warning("Display FXAA Half-SBS Metal kernel unavailable; using shared fallback: %s", exc)
    if half is None:
        eye_width = width // 2
        full = _filter(source, eye_width)
        from .output import downsample_horizontal_area_srgb

        left_target = eye_width // 2
        right_target = eye_width - left_target
        left = downsample_horizontal_area_srgb(full[..., :eye_width], left_target)
        right = downsample_horizontal_area_srgb(full[..., eye_width:], right_target)
        half = torch.cat((left, right), dim=-1)
    result = half.squeeze(0) if single else half
    return result.permute(1, 2, 0).contiguous() if channels_last else result
