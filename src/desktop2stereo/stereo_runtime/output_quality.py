from __future__ import annotations

import math
import os
from typing import Any

import torch
import torch.nn.functional as F

from utils.screen_resolution_policy import OutputSamplingPlan, build_output_sampling_plan

from .output import downsample_horizontal_lanczos2, downsample_vertical_lanczos2


def output_sampling_plan_for_config(
    config: Any,
    source_width: int,
    source_height: int,
) -> OutputSamplingPlan | None:
    if not bool(getattr(config, "output_quality_enabled", False)):
        return None
    plan = build_output_sampling_plan(
        source_width,
        source_height,
        headset_tier_k=int(getattr(config, "output_headset_tier_k", 4)),
    )
    # Local Viewer presents to a physical display; inflating every frame to
    # the headset tier before presenting wastes the whole GPU budget there.
    # Other modes (XR/streaming consumers of the tier canvas) are unaffected.
    if (
        plan.mode == "upscale_easu"
        and os.environ.get("D2S_CAP_OUTPUT_UPSCALE", "") == "1"
    ):
        return OutputSamplingPlan(
            source_width=plan.source_width,
            source_height=plan.source_height,
            target_width=plan.source_width,
            target_height=plan.source_height,
            mode="native_mip",
            target_kind=plan.target_kind,
        )
    return plan


def output_quality_requires_eye_images(config: Any, width: int, height: int) -> bool:
    plan = output_sampling_plan_for_config(config, width, height)
    return plan is not None and (
        plan.mode != "native_mip" or output_mip_lod_for_config(config, plan, width, height) > 0.0
    )


def output_mip_lod_for_config(
    config: Any,
    plan: OutputSamplingPlan,
    width: int,
    height: int,
) -> float:
    minimum = max(0.0, min(16.0, float(getattr(config, "output_min_lod", 0.0))))
    maximum = max(minimum, min(16.0, float(getattr(config, "output_max_lod", 0.35))))
    bias = max(-1.5, min(0.0, float(getattr(config, "output_mip_lod_bias", -0.35))))
    base_lod = max(0.0, math.log2(1.0 / plan.scale))
    available_lod = max(0.0, math.log2(max(1, min(int(width), int(height)))))
    requested_lod = max(base_lod, maximum) + bias
    return min(available_lod, max(minimum, min(maximum, requested_lod)))


def _apply_mip_lod(image: torch.Tensor, lod: float) -> torch.Tensor:
    if lod <= 1.0e-6:
        return image
    source = image.float()
    source_height, source_width = source.shape[-2:]
    lower_level = int(math.floor(lod))
    upper_level = int(math.ceil(lod))

    def mip_level(level: int) -> torch.Tensor:
        if level <= 0:
            return source
        target_height = max(1, source_height >> level)
        target_width = max(1, source_width >> level)
        reduced = F.interpolate(source, size=(target_height, target_width), mode="area")
        return F.interpolate(reduced, size=(source_height, source_width), mode="bilinear", align_corners=False)

    lower = mip_level(lower_level)
    if upper_level == lower_level:
        return lower
    return torch.lerp(lower, mip_level(upper_level), lod - lower_level)


def _torch_rcas(image: torch.Tensor, sharpness: float) -> torch.Tensor:
    padded = F.pad(image.float(), (1, 1, 1, 1), mode="replicate")
    center = padded[..., 1:-1, 1:-1]
    up = padded[..., :-2, 1:-1]
    left = padded[..., 1:-1, :-2]
    right = padded[..., 1:-1, 2:]
    down = padded[..., 2:, 1:-1]
    luma_weights = image.new_tensor((0.299, 0.587, 0.114), dtype=torch.float32).view(1, 3, 1, 1)
    lumas = [(item * luma_weights).sum(dim=1, keepdim=True) for item in (up, left, center, right, down)]
    noise = (0.25 * (lumas[0] + lumas[1] + lumas[3] + lumas[4]) - lumas[2]).abs()
    luma_max = torch.stack(lumas, dim=0).amax(dim=0)
    luma_min = torch.stack(lumas, dim=0).amin(dim=0)
    noise = 1.0 - 0.5 * (noise / (luma_max - luma_min).abs().clamp_min(1.0e-6)).clamp(0.0, 1.0)
    min4 = torch.minimum(torch.minimum(up, left), torch.minimum(right, down))
    max4 = torch.maximum(torch.maximum(up, left), torch.maximum(right, down))
    hit_min = torch.minimum(min4, center) / (4.0 * max4).clamp_min(1.0e-6)
    hit_max = (1.0 - torch.maximum(max4, center)) / (4.0 * min4 - 4.0).clamp_max(-1.0e-6)
    lobe = torch.maximum(-hit_min, hit_max).amax(dim=1, keepdim=True)
    contrast = 2.0 ** (-2.0 * (1.0 - max(0.0, min(1.0, float(sharpness)))))
    lobe = lobe.clamp(-0.1875, 0.0) * contrast * noise
    return ((lobe * (up + left + right + down) + center) / (4.0 * lobe + 1.0).abs().clamp_min(1.0e-6)).clamp(0.0, 1.0)


def _torch_easu(
    image: torch.Tensor,
    target_height: int,
    target_width: int,
) -> torch.Tensor:
    """Device-independent EASU reference, evaluated in bounded row tiles."""
    source = image.float()
    if image.dtype == torch.uint8:
        source = source * (1.0 / 255.0)
    _, _, height, width = source.shape
    output = torch.empty(
        (1, 3, target_height, target_width),
        device=source.device,
        dtype=torch.float32,
    )
    x = torch.arange(target_width, device=source.device, dtype=torch.float32)
    source_x = (x + 0.5) * (width / float(target_width)) - 0.5
    base_x = source_x.floor().to(torch.int64)
    pp_x_line = source_x - base_x.float()

    def sample(x_index: torch.Tensor, y_index: torch.Tensor) -> torch.Tensor:
        return source[0, :, y_index.clamp(0, height - 1), x_index.clamp(0, width - 1)].unsqueeze(0)

    def luma(color: torch.Tensor) -> torch.Tensor:
        return color[:, 0] * 0.299 + color[:, 1] * 0.587 + color[:, 2] * 0.114

    for row_start in range(0, target_height, 64):
        row_end = min(target_height, row_start + 64)
        y = torch.arange(row_start, row_end, device=source.device, dtype=torch.float32)
        source_y = (y + 0.5) * (height / float(target_height)) - 0.5
        base_y = source_y.floor().to(torch.int64)
        pp_y = (source_y - base_y.float())[:, None]
        bx = base_x[None, :].expand(row_end - row_start, -1)
        by = base_y[:, None].expand(-1, target_width)
        pp_x = pp_x_line[None, :]

        b = sample(bx, by - 1); c = sample(bx + 1, by - 1)
        e = sample(bx - 1, by); f = sample(bx, by)
        g = sample(bx + 1, by); h = sample(bx + 2, by)
        i = sample(bx - 1, by + 1); j = sample(bx, by + 1)
        k = sample(bx + 1, by + 1); item_l = sample(bx + 2, by + 1)
        n = sample(bx, by + 2); item_o = sample(bx + 1, by + 2)
        bl, cl, el, fl, gl, hl = (luma(item) for item in (b, c, e, f, g, h))
        il, jl, kl, ll, nl, ol = (luma(item) for item in (i, j, k, item_l, n, item_o))
        direction_x = torch.zeros_like(pp_x + pp_y)
        direction_y = torch.zeros_like(direction_x)
        length_value = torch.zeros_like(direction_x)

        def update(weight, a, left_luma, center_luma, right_luma, down_luma):
            nonlocal direction_x, direction_y, length_value
            gradient_x = right_luma - left_luma
            inverse_x = 1.0 / torch.maximum(
                (right_luma - center_luma).abs(),
                (center_luma - left_luma).abs(),
            ).clamp_min(1.0e-6)
            direction_x = direction_x + gradient_x * weight
            normalized_x = (gradient_x.abs() * inverse_x).clamp(0.0, 1.0)
            length_value = length_value + normalized_x.square() * weight
            gradient_y = down_luma - a
            inverse_y = 1.0 / torch.maximum(
                (down_luma - center_luma).abs(),
                (center_luma - a).abs(),
            ).clamp_min(1.0e-6)
            direction_y = direction_y + gradient_y * weight
            normalized_y = (gradient_y.abs() * inverse_y).clamp(0.0, 1.0)
            length_value = length_value + normalized_y.square() * weight

        update((1.0 - pp_x) * (1.0 - pp_y), bl, el, fl, gl, jl)
        update(pp_x * (1.0 - pp_y), cl, fl, gl, hl, kl)
        update((1.0 - pp_x) * pp_y, fl, il, jl, kl, nl)
        update(pp_x * pp_y, gl, jl, kl, ll, ol)
        direction_length = direction_x.square() + direction_y.square()
        inverse_direction = direction_length.clamp_min(1.0e-12).rsqrt()
        flat = direction_length < 0.000030517578125
        direction_x = torch.where(flat, 1.0, direction_x * inverse_direction)
        direction_y = torch.where(flat, 0.0, direction_y * inverse_direction)
        length_value = 0.25 * length_value.square()
        stretch = 1.0 / torch.maximum(direction_x.abs(), direction_y.abs()).clamp_min(1.0e-6)
        length_x = 1.0 + (stretch - 1.0) * length_value
        length_y = 1.0 - 0.5 * length_value
        lobe = 0.5 + (0.21 - 0.5) * length_value
        clip_value = 1.0 / lobe.clamp_min(1.0e-6)
        color = torch.zeros_like(f)
        weight_sum = torch.zeros_like(direction_x)

        def tap(offset_x, offset_y, sample_color):
            nonlocal color, weight_sum
            rotated_x = (offset_x * direction_x + offset_y * direction_y) * length_x
            rotated_y = (offset_x * -direction_y + offset_y * direction_x) * length_y
            distance_squared = torch.minimum(rotated_x.square() + rotated_y.square(), clip_value)
            weight_b = 0.4 * distance_squared - 1.0
            weight_a = lobe * distance_squared - 1.0
            weight_b = 1.5625 * weight_b.square() - 0.5625
            weight = weight_b * weight_a.square()
            color = color + sample_color * weight[:, None]
            weight_sum = weight_sum + weight

        tap(-pp_x, -1.0 - pp_y, b); tap(1.0 - pp_x, -1.0 - pp_y, c)
        tap(-1.0 - pp_x, 1.0 - pp_y, i); tap(-pp_x, 1.0 - pp_y, j)
        tap(-pp_x, -pp_y, f); tap(-1.0 - pp_x, -pp_y, e)
        tap(1.0 - pp_x, 1.0 - pp_y, k); tap(2.0 - pp_x, 1.0 - pp_y, item_l)
        tap(2.0 - pp_x, -pp_y, h); tap(1.0 - pp_x, -pp_y, g)
        tap(1.0 - pp_x, 2.0 - pp_y, item_o)
        safe_weight = torch.where(weight_sum.abs() <= 1.0e-6, 1.0, weight_sum)
        color = torch.where(
            (weight_sum.abs() <= 1.0e-6)[:, None],
            f,
            color / safe_weight[:, None],
        )
        min4 = torch.minimum(torch.minimum(f, g), torch.minimum(j, k))
        max4 = torch.maximum(torch.maximum(f, g), torch.maximum(j, k))
        output[..., row_start:row_end, :] = torch.minimum(max4, torch.maximum(min4, color))
    return output


def _mps_upscale(image: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
    """Bicubic stand-in for the Triton EASU pass on MPS.

    Neither the Triton kernels nor the tile-looped ``_torch_easu`` reference
    are viable on Apple GPUs (the reference costs seconds per frame), so the
    upscale degrades to hardware bicubic there.
    """
    upscaled = F.interpolate(
        image.float(),
        size=(target_height, target_width),
        mode="bicubic",
        align_corners=False,
    )
    return upscaled.clamp_(0.0, 1.0)


def _mps_rcas(image: torch.Tensor, sharpness: float) -> torch.Tensor:
    """Unsharp-mask stand-in for RCAS on MPS (see _mps_upscale rationale)."""
    source = image.float()
    blurred = F.avg_pool2d(source, kernel_size=3, stride=1, padding=1, count_include_pad=False)
    amount = 1.5 * max(0.0, min(1.0, float(sharpness)))
    return (source + amount * (source - blurred)).clamp_(0.0, 1.0)


def apply_output_quality(
    left: torch.Tensor,
    right: torch.Tensor,
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int | float | str]]:
    width = int(left.shape[-1])
    height = int(left.shape[-2])
    plan = output_sampling_plan_for_config(config, width, height)
    if plan is None:
        return left, right, {"output_quality_mode": "disabled"}
    backend = "native"
    mip_lod = output_mip_lod_for_config(config, plan, width, height)
    if mip_lod > 0.0:
        left = _apply_mip_lod(left, mip_lod)
        right = _apply_mip_lod(right, mip_lod)
        backend = "torch_trilinear_mip"
    if plan.mode == "downsample_lanczos_rcas":
        left = downsample_vertical_lanczos2(
            downsample_horizontal_lanczos2(left, plan.target_width),
            plan.target_height,
        )
        right = downsample_vertical_lanczos2(
            downsample_horizontal_lanczos2(right, plan.target_width),
            plan.target_height,
        )
        backend = "torch_lanczos2" if backend == "native" else backend + "+torch_lanczos2"
    elif plan.mode == "upscale_easu":
        try:
            from .output_quality_triton import can_use_output_quality_triton, easu_resize

            if not can_use_output_quality_triton(left) or not can_use_output_quality_triton(right):
                raise RuntimeError("Triton EASU is unavailable for this device or tensor")
            left = easu_resize(left, plan.target_height, plan.target_width)
            right = easu_resize(right, plan.target_height, plan.target_width)
            backend = "triton_easu" if backend == "native" else backend + "+triton_easu"
        except Exception:
            if left.device.type == "mps":
                left = _mps_upscale(left, plan.target_height, plan.target_width)
                right = _mps_upscale(right, plan.target_height, plan.target_width)
                backend = "mps_bicubic" if backend == "native" else backend + "+mps_bicubic"
            else:
                left = _torch_easu(left, plan.target_height, plan.target_width)
                right = _torch_easu(right, plan.target_height, plan.target_width)
                backend = "torch_easu_reference" if backend == "native" else backend + "+torch_easu_reference"
    sharpness = max(0.0, min(1.0, float(getattr(config, "output_rcas_sharpness", 0.5))))
    if plan.mode != "native_mip" and sharpness > 0.0:
        try:
            from .output_quality_triton import apply_rcas, can_use_output_quality_triton

            if not can_use_output_quality_triton(left) or not can_use_output_quality_triton(right):
                raise RuntimeError("Triton RCAS is unavailable for this device or tensor")
            left = apply_rcas(left, sharpness)
            right = apply_rcas(right, sharpness)
            backend += "+triton_rcas"
        except Exception:
            if left.device.type == "mps":
                left = _mps_rcas(left, sharpness)
                right = _mps_rcas(right, sharpness)
                backend += "+mps_unsharp_rcas"
            else:
                left = _torch_rcas(left, sharpness)
                right = _torch_rcas(right, sharpness)
                backend += "+torch_rcas"
    return left, right, {
        "output_quality_applied": int(plan.mode != "native_mip" or mip_lod > 0.0),
        "output_quality_mode": plan.mode,
        "output_quality_backend": backend,
        "output_quality_source_width": width,
        "output_quality_source_height": height,
        "output_quality_target_width": plan.target_width,
        "output_quality_target_height": plan.target_height,
        "output_quality_mip_lod": mip_lod,
        "output_quality_rcas_sharpness": sharpness,
        "output_quality_target_kind": plan.target_kind,
    }
