from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Literal

import torch

from .baseline_shift import ShiftParams, compute_shift_px, shift_debug_info, synthesize_baseline, warp_horizontal
from .depth_postprocess import postprocess_depth
from .hole_fill import (
    directional_edge_aware_fill,
    directional_edge_aware_fill_backend,
    edge_aware_fill,
    edge_aware_fill_backend,
)
from .layers import composite_layers, make_depth_layers
from .occlusion import make_occlusion_mask, occlusion_backend
from .output import (
    AnaglyphMethod,
    OutputFormat,
    ensure_bchw,
    make_sbs,
    match_depth,
    output_edge_aa_enabled,
    sbs_backend,
)
from .output_quality import (
    apply_output_quality,
    output_quality_requires_eye_images,
    output_sampling_plan_for_config,
)
from .refine import refine_local
from .temporal import TemporalState, apply_temporal, detect_scene_gate

Backend = Literal["fast", "fast_plus", "quality_4k", "hq_4k"]
HoleFill = Literal["none", "fast", "edge_aware"]
HoleFillMode = Literal["none", "balanced", "quality", "content_aware", "directional"]


@dataclass
class StereoConfig:
    backend: Backend = "quality_4k"
    layers: int = 2
    occlusion: bool = True
    symmetric: bool = True
    hole_fill: HoleFill = "edge_aware"
    temporal: bool = True
    output_format: OutputFormat = "half_sbs"
    debug_output: bool = False
    depth_strength: float = 2.0
    convergence: float | torch.Tensor = 0.0
    max_disparity_px: float | None = None
    parallax_preset: str = "standard"
    foreground_shift_scale: float = 1.0
    midground_shift_scale: float = 1.0
    background_shift_scale: float = 1.0
    dynamic_convergence_enabled: bool = False
    dynamic_convergence_strength: float = 0.0
    dynamic_convergence_target: float = 0.5
    dynamic_convergence_alpha: float = 0.85
    temporal_strength: float = 0.85
    auto_reset_temporal: bool = False
    scene_reset_threshold: float = 0.22
    depth_pop: float = 0.0
    depth_antialias_strength: float = 0.0
    edge_dilation: int = 2
    edge_threshold: float = 0.04
    mask_feather_radius: int = 3
    hole_fill_mode: HoleFillMode = "balanced"
    hole_fill_radius: int = 1
    hole_fill_strength: float = 0.6
    screen_edge_mask_suppression: int = 0
    cross_eyed: bool = False
    anaglyph_method: AnaglyphMethod = "red_cyan"
    refine: bool = False
    fused: bool = True
    output_quality_enabled: bool = False
    output_headset_tier_k: int = 4
    output_min_lod: float = 0.0
    output_max_lod: float = 0.35
    output_mip_lod_bias: float = -0.35
    output_rcas_sharpness: float = 0.5


@dataclass
class StereoResult:
    left_eye: torch.Tensor
    right_eye: torch.Tensor
    sbs: torch.Tensor
    debug_info: dict[str, torch.Tensor | float | int | str] = field(default_factory=dict)
    cuda_timing_events: dict[str, object] = field(default_factory=dict)


def _record_cuda_event(events: dict[str, object], name: str, tensor: torch.Tensor | None) -> None:
    if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
        return
    try:
        event = torch.cuda.Event(blocking=False, enable_timing=True)
        event.record(torch.cuda.current_stream(tensor.device))
        events[name] = event
    except Exception:
        return


def _layered_synthesis(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    config: StereoConfig,
    cuda_events: dict[str, object] | None = None,
    *,
    sbs_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict]:
    stage_times: dict[str, float] = {}
    stage_start = time.perf_counter()
    cuda_events = cuda_events if cuda_events is not None else {}
    params = ShiftParams(
        depth_strength=config.depth_strength,
        convergence=config.convergence,
        max_disparity_px=config.max_disparity_px,
        parallax_preset=config.parallax_preset,
        foreground_shift_scale=config.foreground_shift_scale,
        midground_shift_scale=config.midground_shift_scale,
        background_shift_scale=config.background_shift_scale,
    )
    rgb = ensure_bchw(rgb, name="rgb").float()
    depth = postprocess_depth(
        match_depth(
            depth,
            rgb.shape[-2],
            rgb.shape[-1],
            rgb=rgb,
            edge_aware=True,
        ),
        depth_pop=config.depth_pop,
        antialias_strength=config.depth_antialias_strength,
    )
    _record_cuda_event(cuda_events, "synth_depth_postprocess", rgb)
    base_shift = compute_shift_px(depth, rgb.shape[-1], params)
    parallax_debug = shift_debug_info(depth, rgb.shape[-1], params)
    _record_cuda_event(cuda_events, "synth_shift_response", rgb)
    _record_cuda_event(cuda_events, "synth_depth_shift", rgb)
    stage_times["depth_postprocess_shift_ms"] = (time.perf_counter() - stage_start) * 1000.0
    stage_start = time.perf_counter()

    layer_count = max(1, int(config.layers))
    direct_sbs = None
    direct_sbs_backend = None
    direct_sbs_eligible = (
        sbs_only
        and config.output_format in {"half_sbs", "full_sbs"}
        and config.backend == "quality_4k"
        and layer_count == 2
        and bool(config.symmetric)
        and bool(config.fused)
        and str(config.hole_fill).strip().lower() == "none"
        and not bool(config.temporal)
        and not bool(config.refine)
        and not bool(config.debug_output)
        and not bool(config.cross_eyed)
        and int(rgb.shape[-1]) % 2 == 0
        and not output_quality_requires_eye_images(
            config, int(rgb.shape[-1]), int(rgb.shape[-2])
        )
    )
    direct_sbs_ms = 0.0
    if direct_sbs_eligible:
        try:
            from .warp_composite_triton import (
                can_use_triton_warp_composite2,
                warp_composite2_full_sbs,
                warp_composite2_half_sbs,
            )
            if can_use_triton_warp_composite2(
                rgb, depth, base_shift, layers=layer_count, symmetric=config.symmetric
            ):
                direct_start = time.perf_counter()
                direct_sbs = (
                    warp_composite2_half_sbs(rgb, depth, base_shift)
                    if config.output_format == "half_sbs" and not output_edge_aa_enabled()
                    else warp_composite2_full_sbs(rgb, depth, base_shift)
                )
                direct_sbs_backend = (
                    "triton_warp_composite2_half_sbs"
                    if config.output_format == "half_sbs"
                    else "triton_warp_composite2_full_sbs"
                )
                direct_sbs_ms = (time.perf_counter() - direct_start) * 1000.0
        except Exception:
            direct_sbs = None
            direct_sbs_backend = None
    mps_direct_eligible = (
        sbs_only
        and rgb.device.type == "mps"
        and config.output_format in {"half_sbs", "full_sbs", "half_tab", "full_tab"}
        and config.backend == "quality_4k"
        and layer_count == 2
        and bool(config.symmetric)
        and bool(config.fused)
        and str(config.hole_fill).strip().lower() == "none"
        and not bool(config.temporal)
        and not bool(config.refine)
        and not bool(config.debug_output)
        and not bool(config.cross_eyed)
        and not output_quality_requires_eye_images(
            config, int(rgb.shape[-1]), int(rgb.shape[-2])
        )
        and os.environ.get("D2S_MAC_STREAM_MPS_FUSED", "0").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    if direct_sbs is None and mps_direct_eligible:
        try:
            from ._fused_warp_mps import mps_warp_composite2_u8

            direct_sbs = mps_warp_composite2_u8(
                rgb, depth, base_shift,
                "full_sbs" if config.output_format == "half_sbs" and output_edge_aa_enabled() else config.output_format,
            )
            if direct_sbs is not None:
                direct_sbs_backend = "metal_mps_warp_composite2_u8"
        except Exception:
            direct_sbs = None
            direct_sbs_backend = None
    if direct_sbs is not None:
        left, right = rgb, rgb
        warp_composite_backend = direct_sbs_backend
        edge_aa_backend = "disabled"
    else:
        fused = _try_fused_warp_composite2(
            rgb,
            depth,
            base_shift,
            layers=layer_count,
            symmetric=config.symmetric,
            enabled=config.fused,
        )
        if fused is None:
            warp_composite_backend = "torch_grid_sample"
        elif (
            rgb.device.type == "mps"
            and os.environ.get("D2S_MAC_STREAM_MPS_FUSED", "0").strip().lower()
            in {"1", "true", "yes", "on"}
        ):
            warp_composite_backend = "metal_mps_warp_composite2"
        else:
            warp_composite_backend = "triton_warp_composite2"
        if fused is not None:
            left, right = fused
        else:
            weights = make_depth_layers(
                depth,
                layers=layer_count,
                rgb=rgb,
                edge_threshold=config.edge_threshold,
            )
            left_layers: list[torch.Tensor] = []
            right_layers: list[torch.Tensor] = []
            for idx in range(layer_count):
                layer_shift = base_shift * (0.75 + 0.25 * (idx + 1) / layer_count)
                left_layers.append(warp_horizontal(rgb, layer_shift, eye_sign=1.0))
                sign = -1.0 if config.symmetric else -0.9
                right_layers.append(warp_horizontal(rgb, layer_shift, eye_sign=sign))

            left = composite_layers(left_layers, weights)
            right = composite_layers(right_layers, weights)
        edge_aa_backend = "disabled"
    _record_cuda_event(cuda_events, "synth_warp", rgb)
    stage_times["warp_composite_ms"] = direct_sbs_ms if direct_sbs is not None else (time.perf_counter() - stage_start) * 1000.0
    stage_start = time.perf_counter()
    occlusion_mask_needed = bool(config.occlusion) and (
        config.hole_fill != "none"
        or bool(config.refine)
        or bool(config.temporal)
        or bool(config.debug_output)
    )
    if occlusion_mask_needed:
        occlusion_mask_backend = occlusion_backend(
            depth,
            base_shift,
            edge_threshold=config.edge_threshold,
            dilation=config.edge_dilation,
            fused=config.fused,
        )
        mask = make_occlusion_mask(
            depth,
            base_shift,
            edge_threshold=config.edge_threshold,
            dilation=config.edge_dilation,
            fused=config.fused,
            screen_edge_suppression=config.screen_edge_mask_suppression,
        )
    else:
        occlusion_mask_backend = "skipped_no_consumer" if config.occlusion else "none"
        mask = None

    _record_cuda_event(cuda_events, "synth_occlusion", rgb)
    stage_times["occlusion_ms"] = (time.perf_counter() - stage_start) * 1000.0
    stage_start = time.perf_counter()
    hole_fill_backend = "none"
    if config.hole_fill != "none":
        radius = int(config.hole_fill_radius)
        strength = float(config.hole_fill_strength)
        if config.hole_fill == "fast" and config.hole_fill_mode == "balanced":
            radius = 2
            strength = 0.65
        eyes = torch.cat([left, right], dim=0)
        fill_mask = mask.expand(eyes.shape[0], -1, -1, -1)
        use_directional_fill = str(config.hole_fill_mode).strip().lower() in {
            "quality",
            "content_aware",
            "directional",
        }
        if use_directional_fill:
            hole_fill_backend = directional_edge_aware_fill_backend(
                eyes,
                fill_mask,
                depth,
                base_shift,
                radius=radius,
                mask_feather_radius=config.mask_feather_radius,
                fused=config.fused,
            )
            eyes = directional_edge_aware_fill(
                eyes,
                fill_mask,
                depth=depth,
                shift_px=base_shift,
                radius=radius,
                strength=strength,
                mask_feather_radius=config.mask_feather_radius,
                depth_edge_threshold=config.edge_threshold,
                fused=config.fused,
            )
        else:
            hole_fill_backend = edge_aware_fill_backend(
                eyes,
                fill_mask,
                radius=radius,
                strength=strength,
                fused=config.fused,
                mask_feather_radius=config.mask_feather_radius,
            )
            eyes = edge_aware_fill(
                eyes,
                fill_mask,
                radius=radius,
                strength=strength,
                fused=config.fused,
                mask_feather_radius=config.mask_feather_radius,
            )
        left, right = eyes.chunk(2, dim=0)

    _record_cuda_event(cuda_events, "synth_hole_fill", rgb)
    stage_times["hole_fill_ms"] = (time.perf_counter() - stage_start) * 1000.0
    stage_start = time.perf_counter()
    left = refine_local(left, mask, enabled=config.refine)
    right = refine_local(right, mask, enabled=config.refine)
    _record_cuda_event(cuda_events, "synth_refine", rgb)
    stage_times["refine_ms"] = (time.perf_counter() - stage_start) * 1000.0
    return left, right, mask, {
        "layers": layer_count,
        "shift_px": base_shift,
        "occlusion_mask": mask,
        "warp_composite_backend": warp_composite_backend,
        "edge_aa_backend": edge_aa_backend,
        "direct_sbs_backend": direct_sbs_backend or "none",
        "direct_sbs_ms": float(direct_sbs_ms),
        "direct_sbs": direct_sbs,
        "occlusion_mask_backend": occlusion_mask_backend,
        "hole_fill_backend": hole_fill_backend,
        "mask_feather_radius": int(config.mask_feather_radius),
        "hole_fill_mode": str(config.hole_fill_mode),
        "hole_fill_radius": int(radius) if config.hole_fill != "none" else 0,
        "hole_fill_strength": float(strength) if config.hole_fill != "none" else 0.0,
        **parallax_debug,
        **stage_times,
    }


def _try_fused_warp_composite2(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    base_shift: torch.Tensor,
    *,
    layers: int,
    symmetric: bool,
    enabled: bool = True,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not enabled:
        return None
    if (
        rgb.device.type == "mps"
        and layers == 2
        and symmetric
        and os.environ.get("D2S_MAC_STREAM_MPS_FUSED", "0").strip().lower()
        in {"1", "true", "yes", "on"}
    ):
        try:
            from ._fused_warp_mps import mps_warp_composite2

            fused = mps_warp_composite2(rgb, depth, base_shift)
            if fused is not None:
                return fused
        except Exception:
            pass
    if _triton_disabled_by_env():
        return None
    try:
        from .warp_composite_triton import can_use_triton_warp_composite2, warp_composite2
    except Exception:
        return None
    try:
        if not can_use_triton_warp_composite2(
            rgb, depth, base_shift, layers=layers, symmetric=symmetric
        ):
            return None
        return warp_composite2(rgb, depth, base_shift)
    except Exception:
        # Optional output acceleration may fail to compile on a device/runtime;
        # keep stereo synthesis on its PyTorch fallback instead of aborting inference.
        return None


def _triton_disabled_by_env() -> bool:
    return (
        os.environ.get("STEREO_RUNTIME_DISABLE_TRITON", "").lower() in {"1", "true", "yes", "on"}
        or os.environ.get("STEREO_LAB_DISABLE_TRITON", "").lower() in {"1", "true", "yes", "on"}
    )


def synthesize_stereo(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    config: StereoConfig | None = None,
    temporal_state: TemporalState | None = None,
    *,
    sbs_only: bool = False,
) -> StereoResult:
    synthesis_start = time.perf_counter()
    stage_times: dict[str, float] = {}
    cuda_events: dict[str, object] = {}
    _record_cuda_event(cuda_events, "synth_start", rgb)
    config = config or StereoConfig()
    temporal_reset = False
    scene_gate = None
    direct_sbs = None

    stage_start = time.perf_counter()
    if config.temporal and config.auto_reset_temporal and temporal_state is not None and rgb.is_cuda:
        scene_gate = detect_scene_gate(
            rgb,
            temporal_state,
            threshold=config.scene_reset_threshold,
        )
        temporal_reset = bool(temporal_state.last_scene_reset)
    _record_cuda_event(cuda_events, "synth_scene", rgb)
    stage_times["scene_detect_ms"] = (time.perf_counter() - stage_start) * 1000.0

    if config.backend in {"fast", "fast_plus"}:
        params = ShiftParams(
            depth_strength=config.depth_strength,
            convergence=config.convergence,
            max_disparity_px=config.max_disparity_px,
            parallax_preset=config.parallax_preset,
        )
        stage_start = time.perf_counter()
        depth = postprocess_depth(
            depth,
            depth_pop=config.depth_pop,
            antialias_strength=config.depth_antialias_strength,
        )
        _record_cuda_event(cuda_events, "synth_depth_shift", rgb)
        stage_times["fast_depth_postprocess_ms"] = (time.perf_counter() - stage_start) * 1000.0

        stage_start = time.perf_counter()
        left, right, shift_px = synthesize_baseline(rgb, depth, params)
        edge_aa_backend = "disabled"
        _record_cuda_event(cuda_events, "synth_warp", rgb)
        stage_times["fast_baseline_ms"] = (time.perf_counter() - stage_start) * 1000.0
        if config.backend == "fast_plus":
            stage_start = time.perf_counter()
            fast_plus_mask_needed = (
                config.hole_fill != "none"
                or bool(config.temporal)
                or bool(config.debug_output)
            )
            if fast_plus_mask_needed:
                depth_for_mask = match_depth(
                    depth,
                    left.shape[-2],
                    left.shape[-1],
                    rgb=rgb,
                    edge_aware=True,
                )
                mask = make_occlusion_mask(
                    depth_for_mask,
                    shift_px,
                    edge_threshold=0.03,
                    dilation=1,
                    fused=config.fused,
                    screen_edge_suppression=config.screen_edge_mask_suppression,
                )
                occlusion_mask_backend = occlusion_backend(
                    depth_for_mask,
                    shift_px,
                    edge_threshold=0.03,
                    dilation=1,
                    fused=config.fused,
                )
            else:
                depth_for_mask = None
                mask = None
                occlusion_mask_backend = "skipped_no_consumer"
            _record_cuda_event(cuda_events, "synth_occlusion", rgb)
            stage_times["fast_plus_mask_ms"] = (time.perf_counter() - stage_start) * 1000.0

            stage_start = time.perf_counter()
            hole_fill_backend = "none"
            fast_plus_hole_fill_radius = 0
            fast_plus_hole_fill_strength = 0.0
            if config.hole_fill != "none":
                eyes = torch.cat([left, right], dim=0)
                fill_mask = mask.expand(eyes.shape[0], -1, -1, -1)
                hole_fill_backend = directional_edge_aware_fill_backend(
                    eyes,
                    fill_mask,
                    depth_for_mask,
                    shift_px,
                    radius=1,
                    mask_feather_radius=config.mask_feather_radius,
                    fused=config.fused,
                )
                eyes = directional_edge_aware_fill(
                    eyes,
                    fill_mask,
                    depth=depth_for_mask,
                    shift_px=shift_px,
                    radius=1,
                    strength=0.60,
                    mask_feather_radius=config.mask_feather_radius,
                    depth_edge_threshold=0.03,
                    fused=config.fused,
                )
                left, right = eyes.chunk(2, dim=0)
                fast_plus_hole_fill_radius = 1
                fast_plus_hole_fill_strength = 0.60
            _record_cuda_event(cuda_events, "synth_hole_fill", rgb)
            stage_times["fast_plus_fill_ms"] = (time.perf_counter() - stage_start) * 1000.0

            stage_start = time.perf_counter()
            debug = {
                "backend": config.backend,
                "edge_aa_backend": edge_aa_backend,
                "shift_px": shift_px,
                "occlusion_mask": mask,
                "occlusion_mask_backend": occlusion_mask_backend,
                "hole_fill_backend": hole_fill_backend,
                "fast_plus_edge_threshold": 0.03,
                "fast_plus_edge_dilation": 1,
                "fast_plus_hole_fill_radius": fast_plus_hole_fill_radius,
                "fast_plus_hole_fill_strength": fast_plus_hole_fill_strength,
                "mask_feather_radius": int(config.mask_feather_radius),
                **shift_debug_info(depth, left.shape[-1], params),
            }
            stage_times["fast_debug_ms"] = (time.perf_counter() - stage_start) * 1000.0
        else:
            mask = None
            stage_start = time.perf_counter()
            debug = {"backend": config.backend, "edge_aa_backend": edge_aa_backend, "shift_px": shift_px, **shift_debug_info(depth, left.shape[-1], params)}
            stage_times["fast_debug_ms"] = (time.perf_counter() - stage_start) * 1000.0
    else:
        if config.backend == "hq_4k" and config.layers < 3:
            config = StereoConfig(**{**config.__dict__, "layers": 3})
        stage_start = time.perf_counter()
        left, right, mask, debug = _layered_synthesis(
            rgb,
            depth,
            config,
            cuda_events,
            sbs_only=sbs_only,
        )
        direct_sbs = debug.pop("direct_sbs", None)
        stage_times["layered_total_ms"] = (time.perf_counter() - stage_start) * 1000.0
        debug["backend"] = config.backend

    stage_start = time.perf_counter()
    if config.temporal:
        left, right = apply_temporal(left, right, mask, temporal_state, strength=config.temporal_strength, scene_gate=scene_gate)
    _record_cuda_event(cuda_events, "synth_temporal", rgb)
    stage_times["temporal_ms"] = (time.perf_counter() - stage_start) * 1000.0

    stage_start = time.perf_counter()
    needs_output_depth = config.output_format == "depth_map" or bool(config.debug_output)
    output_depth = None
    if needs_output_depth:
        output_depth = postprocess_depth(
            match_depth(
                depth,
                left.shape[-2],
                left.shape[-1],
                rgb=rgb,
                edge_aware=True,
            ),
            depth_pop=config.depth_pop,
            antialias_strength=config.depth_antialias_strength,
        )
    _record_cuda_event(cuda_events, "synth_output_depth", rgb)
    stage_times["output_depth_ms"] = (time.perf_counter() - stage_start) * 1000.0

    stage_start = time.perf_counter()
    if config.cross_eyed:
        left, right = right, left
    stage_times["cross_eye_ms"] = (time.perf_counter() - stage_start) * 1000.0

    stage_start = time.perf_counter()
    if direct_sbs is None:
        if os.environ.get("D2S_OPENXR_NO_EYE_QUALITY"):
            left, right, quality_debug = left, right, {"output_quality_mode": "skipped_env"}
        else:
            left, right, quality_debug = apply_output_quality(left, right, config)
    else:
        plan = output_sampling_plan_for_config(
            config, int(left.shape[-1]), int(left.shape[-2])
        )
        quality_debug = {
            "output_quality_applied": 0,
            "output_quality_mode": "native_mip" if plan is not None else "disabled",
            "output_quality_backend": "direct_sbs_native",
        }
    debug.update(quality_debug)
    _record_cuda_event(cuda_events, "synth_output_quality", left)
    stage_times["output_quality_ms"] = (time.perf_counter() - stage_start) * 1000.0

    stage_start = time.perf_counter()
    if config.debug_output:
        debug["output_depth"] = output_depth
        debug["temporal_reset"] = int(temporal_reset)
        if temporal_state is not None:
            debug["scene_delta"] = float(temporal_state.last_scene_delta)
            debug["temporal_reset_count"] = int(temporal_state.reset_count)
    if temporal_reset:
        debug["temporal_reset_reason"] = "scene_reset"
    debug["cross_eyed"] = int(config.cross_eyed)
    debug["anaglyph_method"] = config.anaglyph_method
    debug["convergence"] = float(config.convergence)
    debug["temporal_enabled"] = int(bool(config.temporal))
    debug["temporal_strength"] = float(config.temporal_strength)
    hole_fill_enabled = config.hole_fill != "none"
    debug["hole_fill_mode"] = str(config.hole_fill_mode) if hole_fill_enabled else "none"
    debug["hole_fill_radius"] = int(config.hole_fill_radius) if hole_fill_enabled else 0
    debug["hole_fill_strength"] = float(config.hole_fill_strength) if hole_fill_enabled else 0.0
    debug["edge_threshold"] = float(config.edge_threshold)
    debug["edge_dilation"] = int(config.edge_dilation)
    debug["mask_feather_radius"] = int(config.mask_feather_radius)
    stage_times["debug_finalize_ms"] = (time.perf_counter() - stage_start) * 1000.0

    stage_start = time.perf_counter()
    if direct_sbs is not None:
        debug["sbs_backend"] = debug.get("direct_sbs_backend", "triton_direct_sbs")
        stage_times["sbs_backend_ms"] = 0.0
        sbs = direct_sbs
        if output_edge_aa_enabled() and config.output_format in {"half_sbs", "full_sbs"}:
            if config.output_format == "half_sbs":
                from .display_antialias import antialias_sbs_half

                sbs = antialias_sbs_half(sbs)
            else:
                from .display_antialias import antialias_sbs

                sbs = antialias_sbs(sbs, "full_sbs")
        stage_times["make_sbs_ms"] = (time.perf_counter() - stage_start) * 1000.0
    else:
        debug["sbs_backend"] = sbs_backend(
            left,
            right,
            config.output_format,
            fused=config.fused,
            depth=output_depth if config.output_format == "depth_map" else None,
            anaglyph_method=config.anaglyph_method,
        )
        stage_times["sbs_backend_ms"] = (time.perf_counter() - stage_start) * 1000.0

        stage_start = time.perf_counter()
        sbs = make_sbs(
            left,
            right,
            config.output_format,
            fused=config.fused,
            depth=output_depth if config.output_format == "depth_map" else None,
            anaglyph_method=config.anaglyph_method,
        )
        stage_times["make_sbs_ms"] = (time.perf_counter() - stage_start) * 1000.0

    debug["edge_aa_backend"] = (
        "output_fxaa" if output_edge_aa_enabled() and config.output_format in {"half_sbs", "full_sbs"}
        else "disabled"
    )
    _record_cuda_event(cuda_events, "synth_sbs", rgb)
    synthesis_total_ms = (time.perf_counter() - synthesis_start) * 1000.0
    stage_accounted_ms = sum(stage_times.values())
    debug.update(stage_times)
    debug["synthesis_total_ms"] = float(synthesis_total_ms)
    debug["synthesis_accounted_ms"] = float(stage_accounted_ms)
    debug["synthesis_unaccounted_ms"] = max(0.0, float(synthesis_total_ms - stage_accounted_ms))
    if not config.debug_output:
        debug = {k: v for k, v in debug.items() if isinstance(v, (float, int, str))}
    return StereoResult(left_eye=left, right_eye=right, sbs=sbs, debug_info=debug, cuda_timing_events=cuda_events)
