from __future__ import annotations

import math
import os
from typing import Literal

import torch
import torch.nn.functional as F

OutputFormat = Literal[
    "half_sbs",
    "full_sbs",
    "half_tab",
    "full_tab",
    "mono",
    "depth_map",
    "anaglyph",
    "interleaved",
    "leia",
]
AnaglyphMethod = Literal["red_cyan", "green_magenta", "amber_blue", "gray"]
OUTPUT_FORMAT_CHOICES = (
    "half_sbs",
    "full_sbs",
    "half_tab",
    "full_tab",
    "mono",
    "depth_map",
    "anaglyph",
    "interleaved",
    "leia",
)


def output_edge_aa_enabled() -> bool:
    """Shared display-only edge-AA switch for renderer backends."""
    return os.environ.get("D2S_SBS_AA", "1").strip().lower() not in {
        "0", "false", "off", "no"
    }


def ensure_bchw(x: torch.Tensor, *, name: str) -> torch.Tensor:
    if x.ndim == 3:
        return x.unsqueeze(0)
    if x.ndim == 4:
        return x
    raise ValueError(f"{name} must be CHW or BCHW, got shape {tuple(x.shape)}")


def ensure_b1hw(depth: torch.Tensor) -> torch.Tensor:
    if depth.ndim == 2:
        return depth.unsqueeze(0).unsqueeze(0)
    if depth.ndim == 3:
        return depth.unsqueeze(1)
    if depth.ndim == 4 and depth.shape[1] == 1:
        return depth
    raise ValueError(f"depth must be HW, BHW, or B1HW, got shape {tuple(depth.shape)}")


def match_depth(
    depth: torch.Tensor,
    height: int,
    width: int,
    *,
    rgb: torch.Tensor | None = None,
    edge_aware: bool = False,
) -> torch.Tensor:
    depth = ensure_b1hw(depth).float()
    if depth.shape[-2:] == (height, width):
        return depth
    if edge_aware and rgb is not None:
        from .depth_upsample import upsample_depth

        return upsample_depth(depth, height, width, rgb=rgb, mode="joint_bilateral")
    return F.interpolate(depth, size=(height, width), mode="bilinear", align_corners=False)


def downsample_horizontal_lanczos2(
    image: torch.Tensor,
    target_width: int | None = None,
) -> torch.Tensor:
    """Match the OpenXR Lanczos2 route for an approximately 2x horizontal reduction."""
    image = ensure_bchw(image, name="image")
    width = int(image.shape[-1])
    target_width = width // 2 if target_width is None else int(target_width)
    if target_width <= 0 or target_width > width:
        raise ValueError("Lanczos2 target width must be between 1 and the source width")
    if width != target_width * 2:
        # Odd-sized inputs need different left/right half widths to preserve the
        # packed frame width. Evaluate the same Lanczos2 kernel at each target
        # pixel centre instead of assuming the exact 2x phase below.
        positions = (
            (torch.arange(target_width, device=image.device, dtype=torch.float32) + 0.5)
            * (float(width) / float(target_width))
            - 0.5
        )
        bases = torch.floor(positions).to(torch.int64)
        result = torch.zeros(
            (*image.shape[:-1], target_width),
            device=image.device,
            dtype=image.dtype,
        )
        total_weight = torch.zeros(target_width, device=image.device, dtype=torch.float32)
        for offset in (-1, 0, 1, 2):
            indices = bases + offset
            distance = positions - indices.to(torch.float32)
            pi_distance = torch.pi * distance
            sinc = torch.where(
                distance.abs() < 1e-6,
                torch.ones_like(distance),
                torch.sin(pi_distance) / pi_distance,
            )
            half_pi_distance = pi_distance * 0.5
            window = torch.where(
                distance.abs() < 1e-6,
                torch.ones_like(distance),
                torch.sin(half_pi_distance) / half_pi_distance,
            )
            weight = torch.where(distance.abs() < 2.0, sinc * window, 0.0)
            result = result + image.index_select(-1, indices.clamp(0, width - 1)) * weight.to(image.dtype)
            total_weight = total_weight + weight
        return result / total_weight.to(image.dtype).view(1, 1, 1, -1)
    padded = F.pad(image, (1, 2, 0, 0), mode="replicate")
    return (
        -padded[..., 0:width:2]
        + 9.0 * padded[..., 1 : width + 1 : 2]
        + 9.0 * padded[..., 2 : width + 2 : 2]
        - padded[..., 3 : width + 3 : 2]
    ) * (1.0 / 16.0)


def downsample_horizontal_area(
    image: torch.Tensor,
    target_width: int | None = None,
) -> torch.Tensor:
    """Reduce SBS eye width with non-negative pixel-area weights."""
    image = ensure_bchw(image, name="image")
    width = int(image.shape[-1])
    target_width = width // 2 if target_width is None else int(target_width)
    if target_width <= 0 or target_width > width:
        raise ValueError("area target width must be between 1 and the source width")
    source = image.float() if image.dtype == torch.uint8 else image
    reduced = _horizontal_area_reduce(source, target_width)
    if image.dtype == torch.uint8:
        return reduced.round().clamp(0, 255).to(torch.uint8)
    return reduced.to(image.dtype)


def _horizontal_area_reduce(image: torch.Tensor, target_width: int) -> torch.Tensor:
    """Exact box-area reduction; integer ratios use the native area kernel."""
    source_width = int(image.shape[-1])
    if target_width == source_width:
        return image
    if source_width % target_width == 0:
        return F.interpolate(
            image,
            size=(int(image.shape[-2]), target_width),
            mode="area",
        )

    scale = source_width / target_width
    output_x = torch.arange(target_width, device=image.device, dtype=torch.float32)
    source_begin = output_x * scale
    source_end = (output_x + 1.0) * scale
    first = torch.floor(source_begin).to(torch.long)
    taps = math.ceil(scale) + 1
    reduced = torch.zeros(
        (*image.shape[:-1], target_width),
        dtype=image.dtype,
        device=image.device,
    )
    for tap in range(taps):
        source_x = first + tap
        weight = (
            torch.minimum(source_end, source_x.to(torch.float32) + 1.0)
            - torch.maximum(source_begin, source_x.to(torch.float32))
        ).clamp_min(0.0)
        sample = image.index_select(-1, source_x.clamp_max(source_width - 1))
        reduced = reduced + sample * weight.view(*([1] * (image.ndim - 1)), target_width)
    return reduced / scale


def downsample_horizontal_area_srgb(
    image: torch.Tensor,
    target_width: int | None = None,
) -> torch.Tensor:
    """Area-reduce encoded RGB in linear light and preserve alpha coverage."""
    image = ensure_bchw(image, name="image")
    if image.shape[1] not in (3, 4):
        raise ValueError("sRGB area sampling expects RGB or RGBA")
    scale = 255.0 if image.dtype == torch.uint8 else 1.0
    source = image.float() / scale
    rgb = source[:, :3].clamp(0.0, 1.0)
    linear = torch.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055).clamp_min(0.0).pow(2.4),
    )
    width = int(source.shape[-1])
    target_width = width // 2 if target_width is None else int(target_width)
    if target_width <= 0 or target_width > width:
        raise ValueError("area target width must be between 1 and the source width")
    reduced_rgb = _horizontal_area_reduce(linear, target_width).clamp(0.0, 1.0)
    encoded = torch.where(
        reduced_rgb <= 0.0031308,
        reduced_rgb * 12.92,
        1.055 * reduced_rgb.pow(1.0 / 2.4) - 0.055,
    )
    if image.shape[1] == 4:
        alpha = _horizontal_area_reduce(source[:, 3:4], target_width)
        encoded = torch.cat((encoded, alpha), dim=1)
    encoded = encoded * scale
    if image.dtype == torch.uint8:
        return encoded.round().clamp(0.0, 255.0).to(torch.uint8)
    return encoded.clamp(0.0, 1.0).to(image.dtype)


def downsample_vertical_lanczos2(
    image: torch.Tensor,
    target_height: int | None = None,
) -> torch.Tensor:
    """Apply the same Lanczos2 reduction vertically for Half-TAB output."""
    image = ensure_bchw(image, name="image")
    return downsample_horizontal_lanczos2(
        image.transpose(-2, -1),
        target_height,
    ).transpose(-2, -1)


def make_sbs(
    left: torch.Tensor,
    right: torch.Tensor,
    output_format: OutputFormat,
    fused: bool = True,
    depth: torch.Tensor | None = None,
    anaglyph_method: AnaglyphMethod = "red_cyan",
) -> torch.Tensor:
    if output_edge_aa_enabled() and output_format in {"half_sbs", "full_sbs"}:
        left_bchw = ensure_bchw(left, name="left")
        right_bchw = ensure_bchw(right, name="right")
        if left_bchw.shape != right_bchw.shape:
            raise ValueError(
                f"left and right shapes must match, got {left_bchw.shape} and {right_bchw.shape}"
            )
        if left_bchw.shape[1] in (3, 4):
            from .display_antialias import antialias_sbs, antialias_sbs_half

            packed = torch.cat((left_bchw, right_bchw), dim=-1)
            if output_format == "half_sbs":
                # One device dispatch performs both eye-local FXAA and the
                # positive-weight linear-light Half-SBS reduction.
                return antialias_sbs_half(packed)
            return antialias_sbs(packed, output_format)
    return _make_sbs_unfiltered(left, right, output_format, fused, depth, anaglyph_method)


def _make_sbs_unfiltered(
    left: torch.Tensor,
    right: torch.Tensor,
    output_format: OutputFormat,
    fused: bool = True,
    depth: torch.Tensor | None = None,
    anaglyph_method: AnaglyphMethod = "red_cyan",
) -> torch.Tensor:
    left = ensure_bchw(left, name="left")
    right = ensure_bchw(right, name="right")
    if left.shape != right.shape:
        raise ValueError(f"left and right shapes must match, got {left.shape} and {right.shape}")

    if output_format == "mono":
        return left

    if output_format == "anaglyph":
        if sbs_backend(left, right, output_format, fused=fused, anaglyph_method=anaglyph_method) == "triton_anaglyph":
            from .output_triton import make_anaglyph

            return make_anaglyph(left, right)
        return make_anaglyph_torch(left, right, method=anaglyph_method)

    if output_format == "interleaved":
        if sbs_backend(left, right, output_format, fused=fused) == "triton_interleaved":
            from .output_triton import make_interleaved

            return make_interleaved(left, right)
        out = torch.empty_like(left)
        out[..., 0::2, :] = left[..., 0::2, :]
        out[..., 1::2, :] = right[..., 1::2, :]
        return out

    if output_format == "leia":
        if sbs_backend(left, right, output_format, fused=fused) == "triton_leia":
            from .output_triton import make_leia

            return make_leia(left, right)
        out = torch.empty_like(left)
        out[..., :, 0::2] = left[..., :, 0::2]
        out[..., :, 1::2] = right[..., :, 1::2]
        return out

    if output_format == "depth_map":
        if depth is None:
            raise ValueError("depth_map output requires depth")
        depth = match_depth(depth, left.shape[-2], left.shape[-1])
        if sbs_backend(left, right, output_format, fused=fused, depth=depth) == "triton_depth_map":
            from .output_triton import make_depth_map

            return make_depth_map(depth, left.shape[1])
        return depth.repeat(1, left.shape[1], 1, 1)

    if output_format == "full_sbs":
        if sbs_backend(left, right, output_format, fused=fused) == "triton_full_sbs":
            from .output_triton import make_full_sbs

            return make_full_sbs(left, right)
        return torch.cat([left, right], dim=-1)

    if output_format == "half_sbs":
        if sbs_backend(left, right, output_format, fused=fused) == "triton_half_sbs":
            from .output_triton import make_half_sbs

            return make_half_sbs(left, right, linear_srgb=output_edge_aa_enabled())
        width = int(left.shape[-1])
        downsample = (
            downsample_horizontal_area_srgb
            if output_edge_aa_enabled() and left.shape[1] in (3, 4)
            else downsample_horizontal_area
        )
        left_half = downsample(left, width // 2)
        right_half = downsample(right, width - width // 2)
        return torch.cat([left_half, right_half], dim=-1)

    if output_format == "full_tab":
        if sbs_backend(left, right, output_format, fused=fused) == "triton_full_tab":
            from .output_triton import make_full_tab

            return make_full_tab(left, right)
        return torch.cat([left, right], dim=-2)

    if output_format == "half_tab":
        if sbs_backend(left, right, output_format, fused=fused) == "triton_half_tab":
            from .output_triton import make_half_tab

            return make_half_tab(left, right)
        h = int(left.shape[-2])
        left_h = max(1, h // 2)
        right_h = max(1, h - left_h)
        left_half = downsample_vertical_lanczos2(left, left_h)
        right_half = downsample_vertical_lanczos2(right, right_h)
        return torch.cat([left_half, right_half], dim=-2)

    raise ValueError(f"unknown output_format: {output_format}")


def sbs_backend(
    left: torch.Tensor,
    right: torch.Tensor,
    output_format: OutputFormat,
    fused: bool = True,
    depth: torch.Tensor | None = None,
    anaglyph_method: AnaglyphMethod = "red_cyan",
) -> str:
    if output_format in {"full_sbs", "full_tab"} and (not fused or _triton_disabled_by_env()):
        return "torch_cat"
    if output_format in {"half_sbs", "half_tab"} and (not fused or _triton_disabled_by_env()):
        return "torch_interpolate"
    if output_format == "depth_map" and (not fused or _triton_disabled_by_env()):
        return "torch_depth_map"
    if output_format == "mono":
        return "torch_mono_left"
    if output_format in {"anaglyph", "interleaved", "leia"} and (not fused or _triton_disabled_by_env()):
        return f"torch_{output_format}"
    if output_format not in {"half_sbs", "full_sbs", "half_tab", "full_tab", "depth_map", "anaglyph", "interleaved", "leia"}:
        return "torch_output"
    try:
        from .output_triton import (
            can_use_triton_anaglyph,
            can_use_triton_depth_map,
            can_use_triton_full_sbs,
            can_use_triton_full_tab,
            can_use_triton_half_sbs,
            can_use_triton_half_tab,
            can_use_triton_interleaved,
            can_use_triton_leia,
        )
    except Exception:
        if output_format in {"anaglyph", "interleaved", "leia"}:
            return f"torch_{output_format}"
        if output_format in {"full_sbs", "full_tab"}:
            return "torch_cat"
        if output_format == "depth_map":
            return "torch_depth_map"
        return "torch_interpolate"
    if output_format == "full_sbs":
        return "triton_full_sbs" if can_use_triton_full_sbs(left, right) else "torch_cat"
    if output_format == "half_sbs":
        return "triton_half_sbs" if can_use_triton_half_sbs(left, right) else "torch_interpolate"
    if output_format == "full_tab":
        return "triton_full_tab" if can_use_triton_full_tab(left, right) else "torch_cat_vertical"
    if output_format == "half_tab":
        return "triton_half_tab" if can_use_triton_half_tab(left, right) else "torch_interpolate_vertical"
    if output_format == "depth_map" and depth is not None:
        return "triton_depth_map" if can_use_triton_depth_map(depth, left.shape[1]) else "torch_depth_map"
    if output_format == "anaglyph":
        return "triton_anaglyph" if anaglyph_method == "red_cyan" and can_use_triton_anaglyph(left, right) else "torch_anaglyph"
    if output_format == "interleaved":
        return "triton_interleaved" if can_use_triton_interleaved(left, right) else "torch_interleaved"
    if output_format == "leia":
        return "triton_leia" if can_use_triton_leia(left, right) else "torch_leia"
    return "torch_depth_map"


def _triton_disabled_by_env() -> bool:
    return (
        os.environ.get("STEREO_RUNTIME_DISABLE_TRITON", "").lower() in {"1", "true", "yes", "on"}
        or os.environ.get("STEREO_LAB_DISABLE_TRITON", "").lower() in {"1", "true", "yes", "on"}
    )


def make_anaglyph_torch(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    method: AnaglyphMethod = "red_cyan",
) -> torch.Tensor:
    out = torch.empty_like(left)
    if method == "red_cyan":
        out[:, 0:1] = left[:, 0:1]
        out[:, 1:] = right[:, 1:]
        return out
    if method == "green_magenta":
        out[:, 0:1] = right[:, 0:1]
        out[:, 1:2] = left[:, 1:2]
        out[:, 2:3] = right[:, 2:3]
        return out
    if method == "amber_blue":
        out[:, 0:2] = left[:, 0:2]
        out[:, 2:3] = right[:, 2:3]
        return out
    if method == "gray":
        left_gray = left.mean(dim=1, keepdim=True)
        right_gray = right.mean(dim=1, keepdim=True)
        out[:, 0:1] = left_gray
        out[:, 1:] = right_gray
        return out
    raise ValueError(f"unknown anaglyph_method: {method}")


def to_uint8_image(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().clamp(0, 1)
    return (x * 255.0).round().to(torch.uint8)
