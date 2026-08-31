from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from dataclasses import replace
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch

from .adapter import StereoRuntimeConfig, depth_provider_config_from_runtime, stereo_config_from_runtime
from .baseline_shift import ShiftParams, compute_shift_px, shift_debug_info
from .depth_postprocess import (
    anti_alias_depth_guided,
    apply_depth_pop,
    postprocess_depth,
)
from .depth_provider import DepthProfileResult, create_depth_provider
from .openxr_render import OpenXRRenderConfig, render_openxr_stereo
from .parallax import parallax_debug_info, resolve_parallax_budget
from .render_size import runtime_output_size_text
from .settings_snapshot import (
    RuntimeSettingsPipelineRebuildRequired,
    RuntimeSettingsRestartRequired,
    RuntimeSettingsSnapshot,
    SnapshotChangeClass,
)
from .compute_backend import probe_opengl_stereo_backend, resolve_stereo_compute_backend
from .output import ensure_bchw, make_sbs, match_depth
from .output_quality import apply_output_quality, output_quality_requires_eye_images
from .synthesis import StereoConfig, StereoResult, synthesize_stereo
from .temporal import TemporalState, apply_temporal
from .triton_runtime import probe_triton_runtime


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DepthRuntimeResult:
    depth: torch.Tensor
    timing: dict[str, float] = field(default_factory=dict)
    provider_info: dict[str, Any] = field(default_factory=dict)


class DepthRuntime:
    """Persistent host-facing runtime for RGB frame -> depth only."""

    def __init__(
        self,
        config: StereoRuntimeConfig,
        *,
        depth_provider: Any | None = None,
        stats_window: int = 300,
        collect_memory_stats: bool = True,
    ) -> None:
        self.config = config
        self.depth_config = depth_provider_config_from_runtime(config)
        self.depth_provider = depth_provider if depth_provider is not None else create_depth_provider(self.depth_config)
        self._loaded = False
        self._active = True
        self.last_timing: dict[str, float] = {}
        self.last_memory: dict[str, float] = {}
        self.stats = RollingRuntimeStats(maxlen=stats_window)
        self.collect_memory_stats = bool(collect_memory_stats)

    def load(self) -> None:
        if self._loaded:
            return
        load = getattr(self.depth_provider, "load", None)
        if callable(load):
            load()
        self._loaded = True

    def set_inference_active(self, active: bool) -> None:
        self._active = bool(active)

    def reset_stats(self) -> None:
        self.stats.reset()
        self.last_timing = {}
        self.last_memory = {}

    def close(self) -> None:
        close = getattr(self.depth_provider, "close", None)
        if callable(close):
            close()
        self._loaded = False

    def provider_report(self) -> dict[str, Any]:
        return _provider_report(self.depth_provider)

    def to_report(self) -> dict[str, Any]:
        report = self.config.to_report()
        report["depth_provider"] = self.provider_report()
        report["depth_backend_resolved"] = report["depth_provider"].get(
            "depth_backend", self.depth_config.backend
        )
        report["last_timing"] = dict(self.last_timing)
        report["last_memory"] = dict(self.last_memory)
        report["rolling_stats"] = self.stats.to_report()
        report["inference_active"] = self._active
        return report

    def predict_depth_frame(self, rgb_frame: torch.Tensor) -> DepthRuntimeResult:
        if not self._active:
            raise RuntimeError("DepthRuntime inference is paused")
        self.load()
        self._reset_cuda_peak_if_needed()
        rgb_frame = _validate_runtime_rgb_frame(rgb_frame)

        total_start = time.perf_counter()
        profile = self._predict_depth_profile(rgb_frame)
        total_ms = (time.perf_counter() - total_start) * 1000.0
        timing = {
            "depth_preprocess_ms": float(profile.preprocess_ms),
            "depth_model_ms": float(profile.model_ms),
            "depth_slot_wait_ms": float(profile.slot_wait_ms),
            "depth_postprocess_ms": float(profile.postprocess_ms),
            "depth_total_ms": float(total_ms),
            "total_ms": float(total_ms),
        }
        memory = self._collect_memory_stats(rgb_frame)
        self.last_timing = timing
        self.last_memory = memory
        self.stats.update(timing, memory)
        return DepthRuntimeResult(depth=profile.depth, timing=timing, provider_info=self.provider_report())

    def _predict_depth_profile(self, rgb_frame: torch.Tensor) -> DepthProfileResult:
        predict_profile = getattr(self.depth_provider, "predict_profile", None)
        if callable(predict_profile):
            result = predict_profile(rgb_frame)
            if isinstance(result, DepthProfileResult):
                return result
            depth = getattr(result, "depth", None)
            if depth is not None:
                return DepthProfileResult(
                    depth=depth,
                    preprocess_ms=float(getattr(result, "preprocess_ms", 0.0)),
                    model_ms=float(getattr(result, "model_ms", 0.0)),
                    postprocess_ms=float(getattr(result, "postprocess_ms", 0.0)),
                    cuda_timing_events=dict(getattr(result, "cuda_timing_events", None) or {}),
                    slot_wait_ms=float(getattr(result, "slot_wait_ms", 0.0)),
                )

        start = time.perf_counter()
        depth = self.depth_provider.predict(rgb_frame)
        elapsed = (time.perf_counter() - start) * 1000.0
        return DepthProfileResult(depth=depth, preprocess_ms=0.0, model_ms=float(elapsed), postprocess_ms=0.0)

    def _reset_cuda_peak_if_needed(self) -> None:
        if not self.collect_memory_stats or not torch.cuda.is_available():
            return
        device = self._runtime_cuda_device()
        if device is None:
            return
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass

    def _collect_memory_stats(self, rgb_frame: torch.Tensor) -> dict[str, float]:
        if not self.collect_memory_stats or not torch.cuda.is_available():
            return {}
        device = self._runtime_cuda_device(rgb_frame)
        if device is None:
            return {}
        try:
            return {
                "cuda_memory_allocated_mb": torch.cuda.memory_allocated(device) / (1024.0 * 1024.0),
                "cuda_memory_reserved_mb": torch.cuda.memory_reserved(device) / (1024.0 * 1024.0),
                "cuda_peak_memory_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0),
                "cuda_peak_memory_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0),
            }
        except Exception:
            return {}

    def _runtime_cuda_device(self, rgb_frame: torch.Tensor | None = None) -> torch.device | None:
        if isinstance(rgb_frame, torch.Tensor) and rgb_frame.is_cuda:
            return rgb_frame.device
        try:
            device = torch.device(self.config.device)
        except Exception:
            return None
        return device if device.type == "cuda" else None


@dataclass(frozen=True)
class StereoRuntimeResult:
    depth: torch.Tensor
    left_eye: torch.Tensor
    right_eye: torch.Tensor
    sbs: torch.Tensor
    output_eye_size: tuple[int, int] | None = None
    output_display_size: tuple[int, int] | None = None
    output_format: str | None = None
    output_dtype: str | None = None
    output_pack_backend: str | None = None
    # Deferred shader-warp handoff (macOS Local Viewer): the viewer samples
    # these directly in its fragment shader, so no SBS is synthesized here.
    viewer_rgb: torch.Tensor | None = None
    viewer_depth: torch.Tensor | None = None
    # Host-packed presentation frame (HWC RGBA8 numpy): produced on the
    # runtime thread right after synthesis while the MPS queue is drained,
    # so the viewer never pays an MPS sync mid-present.
    viewer_frame_np: Any | None = None
    # Owned zero-copy capture frame (SCK CVPixelBuffer+CVMetalTexture):
    # the warp viewer samples it directly instead of uploading packed RGB.
    viewer_bgra: Any | None = None
    active_settings_version: int | None = None
    hot_reload_class: str | None = None
    hot_reload_changed_fields: tuple[str, ...] = ()
    debug_info: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)
    provider_info: dict[str, Any] = field(default_factory=dict)
    cuda_ready_event: Any | None = None
    cuda_timing_events: dict[str, Any] = field(default_factory=dict)
    # Optional final composed BGRA8 D3D11 texture supplied by a native
    # compositor. When present, Intel native output can import it directly.
    native_final_sbs_surface: Any | None = None
    # Optional deferred Vulkan request used by a network sink that owns its
    # own Vulkan image ring rather than an OpenXR presenter.
    vulkan_compute_request: "VulkanComputeRequest | None" = None


@dataclass(frozen=True)
class VulkanComputeRequest:
    """Presenter-thread request for direct Vulkan stereo image synthesis."""

    rgb: torch.Tensor
    depth: torch.Tensor
    params: Any


@dataclass(frozen=True)
class OpenXRRuntimeResult:
    depth: torch.Tensor
    left_eye: torch.Tensor
    right_eye: torch.Tensor
    source_rgb: torch.Tensor | None = None
    output_eye_size: tuple[int, int] | None = None
    output_display_size: tuple[int, int] | None = None
    output_format: str | None = None
    output_dtype: str | None = None
    output_pack_backend: str | None = None
    active_settings_version: int | None = None
    hot_reload_class: str | None = None
    hot_reload_changed_fields: tuple[str, ...] = ()
    shader_uniforms: dict[str, Any] | None = None
    debug_info: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)
    provider_info: dict[str, Any] = field(default_factory=dict)
    cuda_ready_event: Any | None = None
    cuda_timing_events: dict[str, Any] = field(default_factory=dict)
    vulkan_compute_request: VulkanComputeRequest | None = None
    # Optional final composed BGRA8 D3D11 texture supplied by a native
    # compositor; kept separate from the Vulkan presenter request.
    native_final_sbs_surface: Any | None = None


def openxr_result_from_stereo_result(
    stereo_result: StereoRuntimeResult,
    source_rgb: torch.Tensor | None = None,
) -> OpenXRRuntimeResult:
    debug = dict(stereo_result.debug_info or {})
    left_eye = stereo_result.left_eye
    right_eye = stereo_result.right_eye
    cuda_events = dict(getattr(stereo_result, "cuda_timing_events", None) or {})
    display_size = _runtime_frame_size(left_eye)
    stereo_output_format = getattr(stereo_result, "output_format", None) or debug.get("runtime_output_format")
    if stereo_output_format == "half_sbs" and _should_split_half_sbs_for_openxr(debug):
        split_eyes = _split_half_sbs_frame(stereo_result.sbs)
        if split_eyes is not None:
            left_eye, right_eye = split_eyes
            display_size = _runtime_frame_size(stereo_result.sbs)
            debug.setdefault("runtime_output_pack_backend", "split_half_sbs")

    pack_start = time.perf_counter()
    _record_cuda_event(cuda_events, "openxr_pack_start", left_eye if isinstance(left_eye, torch.Tensor) else None)
    if _openxr_runtime_output_uint8_enabled():
        packed_left, left_pack_backend = _pack_openxr_eye_rgba_u8_with_backend(left_eye)
        packed_right, right_pack_backend = _pack_openxr_eye_rgba_u8_with_backend(right_eye)
        if packed_left is not left_eye or packed_right is not right_eye:
            left_eye = packed_left
            right_eye = packed_right
            pack_backend = _merge_openxr_rgba_pack_backend(left_pack_backend, right_pack_backend)
            previous = debug.get("runtime_output_pack_backend")
            debug["runtime_output_pack_backend"] = (
                pack_backend if previous in (None, "none") else f"{previous}+{pack_backend}"
            )
    _record_cuda_event(cuda_events, "openxr_pack", left_eye if isinstance(left_eye, torch.Tensor) else None)
    _record_cuda_event(cuda_events, "end", left_eye if isinstance(left_eye, torch.Tensor) else None)
    pack_ms = (time.perf_counter() - pack_start) * 1000.0

    debug["application_runtime_target"] = "openxr"
    debug["stereo_synthesis_mode"] = "full_synthesis_eyes"
    debug["runtime_output_format"] = "openxr_full_synthesis_eyes"
    debug["runtime_output_dtype"] = _runtime_eye_dtype(left_eye, right_eye)
    debug["runtime_output_eye_size"] = _runtime_eye_size(left_eye)
    debug["runtime_output_display_size"] = _runtime_size_text(display_size)
    debug.setdefault("runtime_output_pack_backend", "none")
    output_eye_size = _runtime_frame_size(left_eye)
    output_display_size = display_size
    timing = {**dict(stereo_result.timing or {}), "pack_ms": float(pack_ms)}

    return OpenXRRuntimeResult(
        depth=stereo_result.depth,
        left_eye=left_eye,
        right_eye=right_eye,
        source_rgb=source_rgb,
        output_eye_size=output_eye_size,
        output_display_size=output_display_size,
        output_format="openxr_full_synthesis_eyes",
        output_dtype=debug["runtime_output_dtype"],
        output_pack_backend=debug.get("runtime_output_pack_backend"),
        active_settings_version=getattr(stereo_result, "active_settings_version", None),
        hot_reload_class=getattr(stereo_result, "hot_reload_class", None),
        hot_reload_changed_fields=tuple(getattr(stereo_result, "hot_reload_changed_fields", ()) or ()),
        debug_info=debug,
        timing=timing,
        provider_info=dict(stereo_result.provider_info or {}),
        cuda_ready_event=getattr(stereo_result, "cuda_ready_event", None),
        cuda_timing_events=cuda_events,
    )



def _pack_openxr_eye_rgba_u8(eye: torch.Tensor) -> torch.Tensor:
    return _pack_openxr_eye_rgba_u8_with_backend(eye)[0]


def _pack_openxr_eye_rgba_u8_with_backend(eye: torch.Tensor) -> tuple[torch.Tensor, str]:
    if not isinstance(eye, torch.Tensor):
        return eye, "none"
    tensor = eye.detach()
    triton_packed = _try_pack_openxr_eye_rgba_u8_triton(tensor)
    if triton_packed is not None:
        return triton_packed, "triton_openxr_rgba_u8"
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            return eye, "none"
        tensor = tensor[0]
    if tensor.ndim == 3 and tensor.shape[0] in (3, 4):
        tensor = tensor[:3].permute(1, 2, 0)
    elif tensor.ndim == 3 and tensor.shape[-1] == 4 and tensor.dtype == torch.uint8:
        return tensor.contiguous(), "torch_openxr_rgba_u8"
    elif tensor.ndim == 3 and tensor.shape[-1] >= 3:
        tensor = tensor[..., :3]
    else:
        return eye, "none"
    if tensor.is_floating_point():
        tensor = tensor.clamp(0.0, 1.0).mul(255.0).round()
    rgb = tensor.contiguous().clamp(0, 255).to(torch.uint8)
    h, w = rgb.shape[:2]
    rgba = torch.empty((h, w, 4), dtype=torch.uint8, device=rgb.device)
    rgba[..., :3].copy_(rgb[..., :3])
    rgba[..., 3].fill_(255)
    return rgba, "torch_openxr_rgba_u8"


def _try_pack_openxr_eye_rgba_u8_triton(tensor: torch.Tensor) -> torch.Tensor | None:
    if not (
        tensor.is_cuda
        and tensor.dtype == torch.float32
        and tensor.ndim == 4
        and tensor.shape[0] == 1
        and tensor.shape[1] == 3
    ):
        return None
    try:
        from .output_triton import make_chw_rgb_to_hwc_rgba_u8

        return make_chw_rgb_to_hwc_rgba_u8(tensor)
    except Exception:
        return None


def _merge_openxr_rgba_pack_backend(left_backend: str, right_backend: str) -> str:
    if left_backend == right_backend:
        return left_backend
    backends = [backend for backend in (left_backend, right_backend) if backend and backend != "none"]
    return "+".join(backends) if backends else "none"


def _record_cuda_event(events: dict[str, Any], name: str, frame: torch.Tensor | None) -> None:
    if not isinstance(frame, torch.Tensor) or not frame.is_cuda:
        return
    try:
        event = torch.cuda.Event(blocking=False, enable_timing=True)
        event.record(torch.cuda.current_stream(frame.device))
        events[name] = event
    except Exception:
        return


def _try_openxr_no_fill_fused_rgba_u8(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    config: StereoConfig,
    cuda_events: dict[str, Any],
) -> tuple[tuple[torch.Tensor, torch.Tensor, dict[str, Any]] | None, str]:
    if str(getattr(config, "backend", "")) != "quality_4k":
        return None, f"backend={getattr(config, 'backend', 'unknown')}"
    if str(getattr(config, "hole_fill", "edge_aware")).strip().lower() != "none":
        return None, "hole_fill_enabled"
    if bool(getattr(config, "temporal", False)):
        return None, "temporal_enabled"
    if bool(getattr(config, "refine", False)):
        return None, "refine_enabled"
    if bool(getattr(config, "debug_output", False)):
        return None, "debug_output"
    if int(getattr(config, "layers", 2)) != 2:
        return None, f"layers={getattr(config, 'layers', 'unknown')}"
    if not bool(getattr(config, "symmetric", True)):
        return None, "asymmetric"
    if bool(getattr(config, "cross_eyed", False)):
        return None, "cross_eyed"
    if not bool(getattr(config, "fused", True)):
        return None, "fused_disabled"
    if not _openxr_runtime_output_uint8_enabled():
        return None, "rgba_u8_output_disabled"
    if not isinstance(rgb, torch.Tensor) or not rgb.is_cuda or rgb.dtype != torch.float32:
        return None, "unsupported_rgb"
    if str(os.environ.get("STEREO_RUNTIME_DISABLE_TRITON", "")).strip().lower() in {"1", "true", "yes", "on"}:
        return None, "triton_disabled"
    if str(os.environ.get("STEREO_LAB_DISABLE_TRITON", "")).strip().lower() in {"1", "true", "yes", "on"}:
        return None, "triton_disabled"

    try:
        from .warp_composite_triton import can_use_triton_warp_composite2, warp_composite2_rgba_u8
    except Exception as exc:
        return None, f"import_failed:{type(exc).__name__}"

    _record_cuda_event(cuda_events, "synth_start", rgb)
    depth_shift_start = time.perf_counter()
    processed_depth = postprocess_depth(
        match_depth(depth, rgb.shape[-2], rgb.shape[-1]),
        depth_pop=float(getattr(config, "depth_pop", 0.0)),
        antialias_strength=float(getattr(config, "depth_antialias_strength", 0.0)),
    )
    _record_cuda_event(cuda_events, "synth_depth_postprocess", rgb)
    params = ShiftParams(
        depth_strength=float(getattr(config, "depth_strength", 1.0)),
        convergence=getattr(config, "convergence", 0.0),
        max_disparity_px=getattr(config, "max_disparity_px", None),
        parallax_preset=str(getattr(config, "parallax_preset", "standard")),
        foreground_shift_scale=float(getattr(config, "foreground_shift_scale", 1.0)),
        midground_shift_scale=float(getattr(config, "midground_shift_scale", 1.0)),
        background_shift_scale=float(getattr(config, "background_shift_scale", 1.0)),
    )
    base_shift = compute_shift_px(processed_depth, int(rgb.shape[-1]), params)
    _record_cuda_event(cuda_events, "synth_shift_response", rgb)
    _record_cuda_event(cuda_events, "synth_depth_shift", rgb)
    depth_shift_ms = (time.perf_counter() - depth_shift_start) * 1000.0

    if not can_use_triton_warp_composite2(
        rgb,
        processed_depth,
        base_shift,
        layers=2,
        symmetric=True,
    ):
        return None, "unsupported_tensor"

    warp_start = time.perf_counter()
    _record_cuda_event(cuda_events, "openxr_pack_start", rgb)
    left, right = warp_composite2_rgba_u8(rgb, processed_depth, base_shift)
    _record_cuda_event(cuda_events, "synth_warp", left)
    _record_cuda_event(cuda_events, "synth_occlusion", left)
    _record_cuda_event(cuda_events, "synth_hole_fill", left)
    _record_cuda_event(cuda_events, "synth_refine", left)
    _record_cuda_event(cuda_events, "synth_temporal", left)
    _record_cuda_event(cuda_events, "synth_output_depth", left)
    _record_cuda_event(cuda_events, "synth_sbs", left)
    _record_cuda_event(cuda_events, "openxr_pack", left)
    warp_ms = (time.perf_counter() - warp_start) * 1000.0

    debug = {
        "backend": "quality_4k",
        "layers": 2,
        "shift_px": base_shift,
        "warp_composite_backend": "triton_warp_composite2_rgba_u8",
        "occlusion_mask_backend": "skipped_no_consumer",
        "hole_fill_backend": "none",
        "hole_fill_mode": "none",
        "hole_fill_radius": 0,
        "hole_fill_strength": 0.0,
        "depth_postprocess_shift_ms": depth_shift_ms,
        "warp_composite_ms": warp_ms,
        "occlusion_ms": 0.0,
        "hole_fill_ms": 0.0,
        "refine_ms": 0.0,
        "temporal_ms": 0.0,
        "output_depth_ms": 0.0,
        "make_sbs_ms": 0.0,
        **shift_debug_info(processed_depth, int(rgb.shape[-1]), params),
    }
    return (left, right, debug), "used"


def _apply_color_adjustment(rgb_frame: torch.Tensor, config: StereoRuntimeConfig) -> torch.Tensor:
    """Apply display color controls after depth inference and before output fan-out."""
    brightness = float(getattr(config, "color_brightness", 1.0))
    contrast = float(getattr(config, "color_contrast", 1.0))
    saturation = float(getattr(config, "color_saturation", 1.0))
    gamma = float(getattr(config, "color_gamma", 1.0))
    temperature = float(getattr(config, "color_temperature", 0.0))
    tint = float(getattr(config, "color_tint", 0.0))
    if (brightness == 1.0 and contrast == 1.0 and saturation == 1.0 and gamma == 1.0
            and temperature == 0.0 and tint == 0.0):
        return rgb_frame

    rgb = rgb_frame if rgb_frame.ndim == 4 else rgb_frame.unsqueeze(0)
    rgb = rgb.float().clone()
    rgb = rgb * max(brightness, 0.0)
    temperature_factor = max(-1.0, min(1.0, temperature / 100.0))
    tint_factor = max(-1.0, min(1.0, tint / 100.0))
    rgb[:, 0:1] *= 1.0 + 0.12 * temperature_factor
    rgb[:, 2:3] *= 1.0 - 0.12 * temperature_factor
    rgb[:, 1:2] *= 1.0 - 0.08 * tint_factor
    rgb[:, 0:1] *= 1.0 + 0.08 * tint_factor
    luminance = rgb[:, 0:1] * 0.2126 + rgb[:, 1:2] * 0.7152 + rgb[:, 2:3] * 0.0722
    rgb = luminance + (rgb - luminance) * saturation
    rgb = (rgb - 0.5) * contrast + 0.5
    if gamma != 1.0:
        rgb = rgb.clamp(0.0, 1.0).pow(1.0 / max(gamma, 0.01))
    rgb = rgb.clamp(0.0, 1.0)
    return rgb if rgb_frame.ndim == 4 else rgb[0]


def _should_split_half_sbs_for_openxr(debug: dict[str, Any]) -> bool:
    fused_backend = str(debug.get("fast_plus_fused_backend", "") or "").strip().lower()
    sbs_backend = str(debug.get("sbs_backend", "") or "").strip().lower()
    if fused_backend and fused_backend not in {"not_used", "none", "n/a"}:
        return True
    return "fused_half_sbs" in sbs_backend


def _split_half_sbs_frame(frame: Any) -> tuple[Any, Any] | None:
    shape = tuple(getattr(frame, "shape", ()))
    if len(shape) < 2:
        return None
    if len(shape) == 4 and shape[1] in (1, 3, 4):
        width_dim = 3
    elif len(shape) == 4 and shape[-1] in (1, 3, 4):
        width_dim = 2
    elif len(shape) == 3 and shape[0] in (1, 3, 4):
        width_dim = 2
    elif len(shape) == 3 and shape[-1] in (1, 3, 4):
        width_dim = 1
    else:
        width_dim = len(shape) - 1
    width = int(shape[width_dim])
    half_width = width // 2
    if half_width <= 0:
        return None

    left_slice = [slice(None)] * len(shape)
    right_slice = [slice(None)] * len(shape)
    left_slice[width_dim] = slice(0, half_width)
    right_slice[width_dim] = slice(half_width, half_width * 2)
    left = frame[tuple(left_slice)]
    right = frame[tuple(right_slice)]
    if hasattr(left, "contiguous"):
        left = left.contiguous()
    if hasattr(right, "contiguous"):
        right = right.contiguous()
    return left, right


class RollingRuntimeStats:
    def __init__(self, *, maxlen: int = 300) -> None:
        self.maxlen = int(max(1, maxlen))
        self._samples: deque[dict[str, float]] = deque(maxlen=self.maxlen)
        self._memory_samples: deque[dict[str, float]] = deque(maxlen=self.maxlen)

    @property
    def count(self) -> int:
        return len(self._samples)

    def reset(self) -> None:
        self._samples.clear()
        self._memory_samples.clear()

    def update(self, timing: dict[str, float], memory: dict[str, float] | None = None) -> None:
        self._samples.append({key: float(value) for key, value in timing.items()})
        if memory:
            self._memory_samples.append({key: float(value) for key, value in memory.items()})

    def to_report(self) -> dict[str, Any]:
        report: dict[str, Any] = {
            "window": self.maxlen,
            "count": self.count,
            "stages": {},
            "fps": {},
            "memory": {},
        }
        if not self._samples:
            return report

        keys = sorted({key for sample in self._samples for key in sample})
        for key in keys:
            values = [sample[key] for sample in self._samples if key in sample]
            report["stages"][key] = _series_stats(values)

        total_values = [sample["total_ms"] for sample in self._samples if sample.get("total_ms", 0.0) > 0]
        if total_values:
            total_stats = _series_stats(total_values)
            report["fps"] = {
                "latest": 1000.0 / total_values[-1],
                "mean_from_mean_ms": 1000.0 / total_stats["mean"] if total_stats["mean"] > 0 else 0.0,
                "p90_from_p90_ms": 1000.0 / total_stats["p90"] if total_stats["p90"] > 0 else 0.0,
                "p99_from_p99_ms": 1000.0 / total_stats["p99"] if total_stats["p99"] > 0 else 0.0,
            }

        if self._memory_samples:
            memory_keys = sorted({key for sample in self._memory_samples for key in sample})
            for key in memory_keys:
                values = [sample[key] for sample in self._memory_samples if key in sample]
                report["memory"][key] = _series_stats(values)
        return report


def _series_stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    count = len(ordered)
    if count == 0:
        return {"count": 0.0, "latest": 0.0, "min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p99": 0.0}
    return {
        "count": float(count),
        "latest": float(values[-1]),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "mean": float(sum(ordered) / count),
        "median": _percentile_sorted(ordered, 0.50),
        "p90": _percentile_sorted(ordered, 0.90),
        "p99": _percentile_sorted(ordered, 0.99),
    }


def _percentile_sorted(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return float(values[lo] * (1.0 - frac) + values[hi] * frac)


def _snapshot_changed_fields(snapshot: RuntimeSettingsSnapshot) -> tuple[str, ...]:
    return tuple(
        sorted(
            name
            for name in snapshot.__dataclass_fields__
            if getattr(snapshot, name) is not None and name not in {"version", "timestamp"}
        )
    )


def _merge_runtime_settings_snapshot(
    base: RuntimeSettingsSnapshot,
    updates: RuntimeSettingsSnapshot,
) -> RuntimeSettingsSnapshot:
    values = {"version": int(updates.version), "timestamp": float(updates.timestamp)}
    for name in updates.__dataclass_fields__:
        if name in {"version", "timestamp"}:
            continue
        value = getattr(updates, name)
        if value is not None:
            values[name] = value
        else:
            values[name] = getattr(base, name)
    return RuntimeSettingsSnapshot(**values)


_DEPTH_PROVIDER_REBUILD_FIELDS = frozenset(
    {"depth_backend", "model_id", "export_height", "export_width", "profile_sync", "use_cuda_graph"}
)
_RUNTIME_HANDLED_PIPELINE_REBUILD_FIELDS = _DEPTH_PROVIDER_REBUILD_FIELDS
_TEMPORAL_RESET_HOT_RELOAD_FIELDS = frozenset(
    {
        "temporal",
        "temporal_enabled",
        "depth_strength",
        "convergence",
        "max_disparity_px",
        "parallax_preset",
        "parallax_budget_preset",
        "foreground_shift_scale",
        "midground_shift_scale",
        "background_shift_scale",
        "dynamic_convergence_enabled",
        "dynamic_convergence_strength",
        "dynamic_convergence_target",
        "dynamic_convergence_alpha",
    }
)


def _append_temporal_reset_reason(debug: dict[str, Any], reason: str) -> None:
    current = debug.get("temporal_reset_reason")
    if not current:
        debug["temporal_reset_reason"] = reason
        return
    reasons = [part.strip() for part in str(current).split(",") if part.strip()]
    if reason not in reasons:
        reasons.append(reason)
    debug["temporal_reset_reason"] = ",".join(reasons)


def _consume_pending_temporal_reset_reasons(runtime: "StereoRuntime", debug: dict[str, Any]) -> None:
    for reason in runtime._pending_temporal_reset_reasons:
        _append_temporal_reset_reason(debug, reason)
    runtime._pending_temporal_reset_reasons = ()


def _add_active_settings_debug_info(debug: dict[str, Any], snapshot: RuntimeSettingsSnapshot) -> None:
    for field_name in (
        "source",
        "application_runtime_target",
        "runtime_quality_mode",
        "stereo_synthesis_mode",
        "render_size_policy",
        "stereo_render_scale",
        "output_transport",
        "presentation_flags",
        "debug_flags",
        "output_format",
        "max_disparity_px",
        "parallax_preset",
        "parallax_budget_preset",
        "convergence",
        "foreground_shift_scale",
        "midground_shift_scale",
        "background_shift_scale",
        "dynamic_convergence_enabled",
        "dynamic_convergence_strength",
        "dynamic_convergence_target",
        "dynamic_convergence_alpha",
        "hole_fill_mode",
    ):
        value = getattr(snapshot, field_name)
        if value is not None:
            debug[field_name] = value


def _add_runtime_config_debug_info(debug: dict[str, Any], config: StereoConfig) -> None:
    debug.setdefault("runtime_quality_mode", str(config.backend))
    debug.setdefault("output_format", str(config.output_format))
    debug.setdefault("stereo_synthesis_mode", "packed_synthesis")
    debug.setdefault("depth_strength", float(config.depth_strength))
    debug.setdefault("max_disparity_px", None if config.max_disparity_px is None else float(config.max_disparity_px))
    debug.setdefault("runtime_temporal_enabled", int(bool(config.temporal)))
    debug.setdefault("runtime_refine_enabled", int(bool(config.refine)))
    debug.setdefault("runtime_occlusion_enabled", int(bool(config.occlusion)))
    debug.setdefault("parallax_preset", str(config.parallax_preset))
    debug.setdefault("convergence", _debug_scalar_no_sync(config.convergence))
    debug.setdefault("foreground_shift_scale", float(getattr(config, "foreground_shift_scale", 1.0)))
    debug.setdefault("midground_shift_scale", float(getattr(config, "midground_shift_scale", 1.0)))
    debug.setdefault("background_shift_scale", float(getattr(config, "background_shift_scale", 1.0)))
    debug.setdefault("dynamic_convergence_enabled", bool(getattr(config, "dynamic_convergence_enabled", False)))
    hole_fill_enabled = str(getattr(config, "hole_fill", "edge_aware")).strip().lower() != "none"
    debug.setdefault("hole_fill_mode", str(getattr(config, "hole_fill_mode", "balanced")) if hole_fill_enabled else "none")
    debug.setdefault("hole_fill_radius", int(getattr(config, "hole_fill_radius", 1)) if hole_fill_enabled else 0)
    debug.setdefault("hole_fill_strength", float(getattr(config, "hole_fill_strength", 0.6)) if hole_fill_enabled else 0.0)


def _debug_scalar_no_sync(value: Any) -> float | str | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.is_cuda:
            return "cuda_tensor"
        return float(value.detach())
    return float(value)


def _add_depth_contract_debug_info(
    debug: dict[str, Any],
    depth: torch.Tensor,
    provider_info: dict[str, Any],
) -> None:
    debug["depth_render_size"] = runtime_output_size_text(_runtime_frame_size(depth))
    resolved_backend = provider_info.get("depth_backend")
    if resolved_backend:
        debug["depth_backend_resolved"] = str(resolved_backend)
    provider_runtime = provider_info.get("runtime")
    if provider_runtime:
        debug["depth_runtime"] = str(provider_runtime)
    fallback_reason = provider_info.get("fallback_reason")
    if fallback_reason:
        debug["depth_fallback_reason"] = str(fallback_reason)
    provider_size = _provider_size_label(provider_info)
    if provider_size is not None:
        debug["depth_provider_size"] = provider_size


def _layered_parallax_enabled(config: Any) -> bool:
    return any(
        abs(float(getattr(config, field_name, 1.0)) - 1.0) > 1e-6
        for field_name in ("foreground_shift_scale", "midground_shift_scale", "background_shift_scale")
    )


def _dynamic_convergence_config_for_depth(
    runtime: Any,
    depth: torch.Tensor,
    stereo_config: Any,
    *,
    prefer_gpu_tensor: bool = True,
) -> tuple[Any, dict[str, Any]]:
    enabled = bool(getattr(stereo_config, "dynamic_convergence_enabled", False))
    strength = max(0.0, min(1.0, float(getattr(stereo_config, "dynamic_convergence_strength", 0.0))))
    manual = float(getattr(stereo_config, "convergence", 0.0))
    if not enabled or strength <= 0.0:
        runtime._dynamic_convergence_value = None
        runtime._dynamic_convergence_last_measured = None
        runtime._dynamic_convergence_pending_measurement = None
        runtime._dynamic_convergence_pending_event = None
        return stereo_config, {"dynamic_convergence_effective": manual}
    target = max(0.0, min(1.0, float(getattr(stereo_config, "dynamic_convergence_target", 0.5))))
    alpha = max(0.0, min(0.98, float(getattr(stereo_config, "dynamic_convergence_alpha", 0.85))))
    measured = _dynamic_convergence_measurement(runtime, depth, target, prefer_gpu_tensor=prefer_gpu_tensor)
    previous = getattr(runtime, "_dynamic_convergence_value", None)
    if isinstance(measured, torch.Tensor):
        manual_tensor = measured.new_tensor(manual)
        desired = manual_tensor + (measured - manual_tensor) * strength
        if isinstance(previous, torch.Tensor) and previous.device == desired.device:
            effective = previous.to(dtype=desired.dtype) * alpha + desired * (1.0 - alpha)
        else:
            effective = desired
        effective = effective.detach()
        runtime._dynamic_convergence_value = effective
        runtime._dynamic_convergence_last_measured = measured.detach()
        return replace(stereo_config, convergence=effective), {
            "dynamic_convergence_effective": _debug_scalar_no_sync(effective),
            "dynamic_convergence_measured": _debug_scalar_no_sync(measured),
            "dynamic_convergence_manual": float(manual),
            "dynamic_convergence_strength": float(strength),
            "dynamic_convergence_target": float(target),
            "dynamic_convergence_alpha": float(alpha),
        }
    previous_float = _dynamic_convergence_previous_float(previous)
    if measured is None:
        effective = manual if previous_float is None else previous_float
    else:
        desired = manual + (measured - manual) * strength
        effective = desired if previous_float is None else previous_float * alpha + desired * (1.0 - alpha)
    runtime._dynamic_convergence_value = float(effective)
    runtime._dynamic_convergence_last_measured = measured
    return replace(stereo_config, convergence=float(effective)), {
        "dynamic_convergence_effective": float(effective),
        "dynamic_convergence_measured": None if measured is None else float(measured),
        "dynamic_convergence_manual": float(manual),
        "dynamic_convergence_strength": float(strength),
        "dynamic_convergence_target": float(target),
        "dynamic_convergence_alpha": float(alpha),
    }


def _dynamic_convergence_previous_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.is_cuda:
            return None
        return float(value.detach())
    return float(value)


def _dynamic_convergence_measurement(
    runtime: Any,
    depth: torch.Tensor,
    quantile: float,
    *,
    prefer_gpu_tensor: bool = True,
) -> float | torch.Tensor | None:
    tensor = _depth_quantile_tensor(depth, quantile).detach().float()
    if not tensor.is_cuda:
        runtime._dynamic_convergence_pending_measurement = None
        runtime._dynamic_convergence_pending_event = None
        return float(tensor)
    if prefer_gpu_tensor:
        runtime._dynamic_convergence_pending_measurement = None
        runtime._dynamic_convergence_pending_event = None
        return tensor

    pending = getattr(runtime, "_dynamic_convergence_pending_measurement", None)
    event = getattr(runtime, "_dynamic_convergence_pending_event", None)
    if pending is not None and event is not None and event.query():
        runtime._dynamic_convergence_last_measured = float(pending.detach())
        runtime._dynamic_convergence_pending_measurement = None
        runtime._dynamic_convergence_pending_event = None
    if getattr(runtime, "_dynamic_convergence_pending_measurement", None) is None:
        pending = _cpu_scalar_buffer(tensor)
        pending.copy_(tensor, non_blocking=True)
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(tensor.device))
        runtime._dynamic_convergence_pending_measurement = pending
        runtime._dynamic_convergence_pending_event = event
    return getattr(runtime, "_dynamic_convergence_last_measured", None)


def _cpu_scalar_buffer(tensor: torch.Tensor) -> torch.Tensor:
    try:
        return torch.empty((), dtype=torch.float32, device="cpu", pin_memory=bool(getattr(tensor, "is_cuda", False)))
    except Exception:
        return torch.empty((), dtype=torch.float32, device="cpu")


def _depth_quantile_tensor(depth: torch.Tensor, quantile: float, *, max_samples: int = 8192) -> torch.Tensor:
    tensor = depth.detach().float().clamp(0.0, 1.0).flatten()
    if tensor.numel() == 0:
        return depth.new_tensor(0.0, dtype=torch.float32)
    if tensor.numel() > max_samples:
        stride = max(1, int(tensor.numel() // max_samples))
        tensor = tensor[::stride]
    count = int(tensor.numel())
    index = min(count - 1, max(0, int(round(float(quantile) * float(count - 1)))))
    return torch.sort(tensor).values[index]


def _provider_size_label(provider_info: dict[str, Any]) -> str | None:
    provider_size = provider_info.get("depth_provider_size")
    if provider_size is not None:
        return _size_label(provider_size, height_width=False)
    for key in ("input_size", "fixed_input_size"):
        value = provider_info.get(key)
        if value is not None:
            return _size_label(value, height_width=True)
    depth_resolution = provider_info.get("depth_resolution")
    if depth_resolution is None:
        return None
    try:
        size = int(depth_resolution)
    except (TypeError, ValueError):
        return str(depth_resolution)
    return f"{size}x{size}"


def _size_label(value, *, height_width: bool) -> str:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            first = int(value[0])
            second = int(value[1])
            return runtime_output_size_text((second, first) if height_width else (first, second))
        except (TypeError, ValueError):
            pass
    return str(value)


class StereoRuntime:
    """Persistent host-facing runtime for RGB frame -> depth -> stereo output."""

    def __init__(
        self,
        config: StereoRuntimeConfig,
        *,
        depth_provider: Any | None = None,
        temporal_state: TemporalState | None = None,
        stats_window: int = 300,
        collect_memory_stats: bool = True,
    ) -> None:
        self.config = config
        self.depth_config = depth_provider_config_from_runtime(config)
        self.stereo_config = stereo_config_from_runtime(config)
        self.depth_provider = depth_provider if depth_provider is not None else create_depth_provider(self.depth_config)
        self.temporal_state = temporal_state if temporal_state is not None else TemporalState()
        self._openxr_depth_temporal: torch.Tensor | None = None
        self._openxr_rgb_depth_dumped = False
        self._loaded = False
        self._active = True
        self.last_timing: dict[str, float] = {}
        self.last_memory: dict[str, float] = {}
        self.active_settings_snapshot = RuntimeSettingsSnapshot(version=0, timestamp=0.0)
        self.active_settings_version = 0
        self.last_settings_change_class = SnapshotChangeClass.NO_CHANGE.value
        self.last_settings_changed_fields: tuple[str, ...] = ()
        self._pending_temporal_reset_reasons: tuple[str, ...] = ()
        self._last_runtime_perf_log_ts = 0.0
        self._runtime_frame_refresh_log_count = 0
        self._dynamic_convergence_value: float | None = None
        self._dynamic_convergence_last_measured: float | None = None
        self._dynamic_convergence_pending_measurement: torch.Tensor | None = None
        self._dynamic_convergence_pending_event: Any | None = None
        self._vulkan_stereo_backend: Any | None = None
        self._vulkan_stereo_backend_error: str | None = None
        self._resolved_stereo_compute_backend: str | None = None
        self._stereo_compute_backend_reason = "not_resolved"
        self._last_stereo_backend_log_key: tuple[str, str] | None = None
        self.stats = RollingRuntimeStats(maxlen=stats_window)
        self.collect_memory_stats = bool(collect_memory_stats)

    def load(self) -> None:
        if self._loaded:
            return
        load = getattr(self.depth_provider, "load", None)
        if callable(load):
            load()
        self._loaded = True

    def set_inference_active(self, active: bool) -> None:
        """Pause or resume depth and stereo processing for headset idle state."""
        self._active = bool(active)

    def reset_temporal(self) -> None:
        self.temporal_state.reset()

    def configure_stereo(self, stereo_config: Any, *, reset_temporal: bool = False) -> None:
        self.stereo_config = stereo_config
        if reset_temporal:
            self.temporal_state.reset_stereo()

    def apply_settings_snapshot(
        self,
        snapshot: RuntimeSettingsSnapshot,
        *,
        active_preset: str | None = None,
    ) -> SnapshotChangeClass:
        change_class = snapshot.classify()
        changed_fields = _snapshot_changed_fields(snapshot)
        merged_snapshot = _merge_runtime_settings_snapshot(self.active_settings_snapshot, snapshot)
        if change_class is SnapshotChangeClass.NO_CHANGE:
            self.active_settings_snapshot = merged_snapshot
            self.active_settings_version = int(snapshot.version)
            self.last_settings_change_class = change_class.value
            self.last_settings_changed_fields = changed_fields
            return change_class
        if change_class is SnapshotChangeClass.SESSION_RESTART:
            raise RuntimeSettingsRestartRequired(snapshot)
        if (
            change_class is SnapshotChangeClass.PIPELINE_REBUILD
            and not set(changed_fields).issubset(_RUNTIME_HANDLED_PIPELINE_REBUILD_FIELDS)
        ):
            raise RuntimeSettingsPipelineRebuildRequired(snapshot, changed_fields)

        updates = snapshot.to_config_updates()
        if active_preset is not None:
            updates["stereo_preset"] = active_preset
        self.config = replace(self.config, **updates)
        self.stereo_config = stereo_config_from_runtime(self.config)
        self.active_settings_snapshot = merged_snapshot
        self.active_settings_version = int(snapshot.version)
        self.last_settings_change_class = change_class.value
        self.last_settings_changed_fields = changed_fields

        if _TEMPORAL_RESET_HOT_RELOAD_FIELDS.intersection(changed_fields):
            self.temporal_state.reset_stereo()
            self._openxr_depth_temporal = None
            self._pending_temporal_reset_reasons = (*self._pending_temporal_reset_reasons, "settings_changed")

        if change_class is SnapshotChangeClass.PIPELINE_REBUILD and _DEPTH_PROVIDER_REBUILD_FIELDS.intersection(changed_fields):
            self._rebuild_depth_provider()
        return change_class

    def _temporal_stabilize_depth(self, depth):
        """EMA-smooth the shipped warp depth across frames (fused leg only).

        Kills frame-to-frame dither (ANE fp16 quantization, ViT global
        attention sensitivity) that the native-resolution warp converts
        into visible edge shimmer. Kill switch: D2S_DEPTH_TEMPORAL=0.
        """
        if os.environ.get("D2S_DEPTH_TEMPORAL", "1") in {"0", "false", "off"}:
            return depth
        prev = getattr(self, "_fused_depth_prev", None)
        if (
            prev is None
            or prev.shape != depth.shape
            or prev.device != depth.device
        ):
            self._fused_depth_prev = depth.detach().clone()
            return depth
        # Out-of-place: the previous state must never be mutated while the
        # packer thread may still be reading the tensor we shipped last frame.
        st = self._fused_depth_prev.mul(0.35).add_(depth.detach(), alpha=0.65)
        # One poisoned fp16 frame must not poison the EMA forever; sanitize
        # the blended state and reseed on non-finite input.
        if st.is_floating_point() and not torch.isfinite(st).all():
            st = torch.nan_to_num(st, nan=1.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
            self._fused_depth_prev = st.detach().clone()
        else:
            self._fused_depth_prev = st
        return st

    def _rebuild_depth_provider(self) -> None:
        close = getattr(self.depth_provider, "close", None)
        if callable(close):
            close()
        self.depth_config = depth_provider_config_from_runtime(self.config)
        self.depth_provider = create_depth_provider(self.depth_config)
        self._loaded = False

    def warmup_stereo_kernels_for_frame(self, rgb_frame: torch.Tensor) -> None:
        """Compile stereo synthesis kernels for the actual runtime frame shape."""
        rgb_frame = _validate_runtime_rgb_frame(rgb_frame)
        device = rgb_frame.device
        if device.type != "cuda" or not torch.cuda.is_available():
            return
        if str(os.environ.get("D2S_DISABLE_STEREO_WARMUP", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}:
            return
        self.load()
        preprocessor = getattr(self.depth_provider, "_preprocessor", None)
        engine = getattr(self.depth_provider, "_engine", None)
        if bool(getattr(self.config, "use_cuda_graph", False)) and preprocessor is not None and engine is not None:
            input_size = preprocessor.input_size(int(rgb_frame.shape[-2]), int(rgb_frame.shape[-1]))
            try:
                engine.capture_graph((1, 3, int(input_size[0]), int(input_size[1])))
            except RuntimeError as exc:
                setattr(self.depth_provider, "_cuda_graph_disabled_reason", f"{type(exc).__name__}: {exc}")
                clear_graph = getattr(engine, "clear_graph", None)
                if callable(clear_graph):
                    clear_graph()
                try:
                    torch.cuda.synchronize(device)
                except Exception:
                    pass
                print(
                    "\033[31m[TensorRT] CUDA graph warmup capture failed; disabled for this process. "
                    f"Using native TensorRT enqueue until restart: {type(exc).__name__}: {exc}\033[0m",
                    flush=True,
                )
                raise
        if rgb_frame.ndim == 3:
            _, height, width = rgb_frame.shape
        else:
            _, _, height, width = rgb_frame.shape
        rgb = torch.zeros((1, 3, int(height), int(width)), device=device, dtype=torch.float32)
        depth = torch.linspace(0.0, 1.0, int(width), device=device, dtype=torch.float32).view(1, 1, 1, int(width)).expand(1, 1, int(height), int(width)).contiguous()
        base = self.stereo_config
        depth_pop_values = {round(float(base.depth_pop), 3), 0.0, -0.7, 0.5}
        antialias_values = {round(float(base.depth_antialias_strength), 3), 0.0, 2.0}
        configs = []
        for depth_pop in sorted(depth_pop_values):
            for antialias in sorted(antialias_values):
                configs.append(
                    replace(
                        base,
                        temporal=False,
                        depth_pop=float(depth_pop),
                        depth_antialias_strength=float(antialias),
                    )
                )
        start = time.perf_counter()
        seen = set()
        for config in configs:
            key = (config.backend, config.output_format, config.layers, config.hole_fill, config.edge_dilation, round(float(config.depth_pop), 3), round(float(config.depth_antialias_strength), 3))
            if key in seen:
                continue
            seen.add(key)
            synthesize_stereo(rgb, depth, config, temporal_state=None)
        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        print(
            f"[StereoRuntime] stereo kernel warmup complete: {len(seen)} configs {int(width)}x{int(height)} in {elapsed_ms:.1f}ms",
            flush=True,
        )
    def reset_stats(self) -> None:
        self.stats.reset()
        self.last_timing = {}
        self.last_memory = {}

    def close(self) -> None:
        close = getattr(self.depth_provider, "close", None)
        if callable(close):
            close()
        close_vulkan = getattr(self._vulkan_stereo_backend, "close", None)
        if callable(close_vulkan):
            close_vulkan()
        self._vulkan_stereo_backend = None
        self._loaded = False

    def provider_report(self) -> dict[str, Any]:
        return _provider_report(self.depth_provider)

    def to_report(self) -> dict[str, Any]:
        report = self.config.to_report()
        report["depth_provider"] = self.provider_report()
        report["depth_backend_resolved"] = report["depth_provider"].get(
            "depth_backend", self.depth_config.backend
        )
        report["stereo_backend"] = self.stereo_config.backend
        report["stereo_compute_backend"] = (
            self._resolved_stereo_compute_backend
            or getattr(self.config, "stereo_compute_backend", "auto")
        )
        report["stereo_compute_backend_reason"] = self._stereo_compute_backend_reason
        report["last_timing"] = dict(self.last_timing)
        report["last_memory"] = dict(self.last_memory)
        report["rolling_stats"] = self.stats.to_report()
        report["inference_active"] = self._active
        return report

    def process_rgb_frame(
        self,
        rgb_frame: torch.Tensor,
        *,
        skip_sbs_output: bool = False,
        depth_profile: DepthProfileResult | None = None,
        pixel_buffer: Any | None = None,
    ) -> StereoRuntimeResult:
        if not self._active:
            raise RuntimeError("StereoRuntime inference is paused")
        self.load()

        # Owned zero-copy capture frame (SCK): either ship it to the warp
        # viewer or release it here — never leak it.
        viewer_bgra = None
        if pixel_buffer is not None:
            if (
                not skip_sbs_output
                and os.environ.get("D2S_METAL_SHADER_WARP", "0") == "1"
                and _viewer_color_adjustments_neutral(self.config)
                and hasattr(pixel_buffer, "mtl_texture")
            ):
                viewer_bgra = pixel_buffer
            else:
                try:
                    pixel_buffer.release()
                except Exception:
                    pass

        self._reset_cuda_peak_if_needed()
        rgb_frame = _validate_runtime_rgb_frame(rgb_frame)

        cuda_events: dict[str, Any] = {}
        _record_cuda_event(cuda_events, "start", rgb_frame)
        total_start = time.perf_counter()
        depth_start = time.perf_counter()
        profile = depth_profile or self._predict_depth_profile(rgb_frame)
        depth_total_ms = (
            float(profile.total_ms)
            if depth_profile is not None
            else (time.perf_counter() - depth_start) * 1000.0
        )
        depth = profile.depth
        output_rgb = _apply_color_adjustment(rgb_frame, self.config)
        cuda_events.update(getattr(profile, "cuda_timing_events", None) or {})
        _record_cuda_event(cuda_events, "depth", rgb_frame)
        stereo_config, convergence_debug = _dynamic_convergence_config_for_depth(self, depth, self.stereo_config)

        synth_start = time.perf_counter()
        deferred_vulkan_request = None
        deferred_vulkan_reason = "disabled"
        # Deferred Metal shader warp (macOS Local Viewer): skip torch-side
        # synthesis entirely; the viewer's fragment shader samples rgb+depth
        # at draw time (v2.5 approach). runtime_entry sets the env default to
        # "1" only for Darwin Local Viewer, so stream/OpenXR paths never see
        # it unless a user opts in explicitly.
        deferred_warp = (
            not skip_sbs_output
            and os.environ.get("D2S_METAL_SHADER_WARP", "0") == "1"
        )
        if deferred_warp:
            stereo = StereoResult(
                left_eye=output_rgb,
                right_eye=output_rgb,
                sbs=output_rgb,
                debug_info={
                    "backend": stereo_config.backend,
                    "sbs_backend": "metal_shader_warp",
                    "fast_plus_fused_backend": "not_used",
                    "fast_plus_fused_skip": "shader_warp",
                },
            )
        else:
            if _intel_vulkan_network_path_enabled() and not skip_sbs_output:
                deferred_vulkan_request, deferred_vulkan_reason = self._build_vulkan_compute_request(
                    output_rgb,
                    depth,
                    stereo_config,
                )
            if deferred_vulkan_request is not None:
                vulkan_stereo = None
                vulkan_skip = "intel_network_deferred"
                fused_sbs = None
                fused_skip = "intel_network_deferred"
                stereo = StereoResult(
                    left_eye=output_rgb,
                    right_eye=output_rgb,
                    sbs=output_rgb,
                    debug_info={
                        "backend": stereo_config.backend,
                        "sbs_backend": "vulkan_deferred_intel_network",
                        "stereo_compute_backend": "vulkan",
                        "vulkan_zero_copy_deferred": 1,
                        "vulkan_zero_copy_reason": deferred_vulkan_reason,
                    },
                )
            else:
                vulkan_stereo, vulkan_skip = self._try_vulkan_fused_stereo(
                    output_rgb,
                    depth,
                    stereo_config,
                    skip_sbs_output=skip_sbs_output,
                )
                fused_sbs, fused_skip = (None, "vulkan_selected") if vulkan_stereo is not None else (
                    (None, "skip_sbs_output")
                    if skip_sbs_output
                    else self._try_fast_plus_fused_sbs(output_rgb, depth, stereo_config)
                )
            if vulkan_stereo is not None:
                stereo = vulkan_stereo
                stereo.debug_info.setdefault("fast_plus_fused_backend", "not_used")
                stereo.debug_info.setdefault("fast_plus_fused_skip", vulkan_skip)
            elif fused_sbs is not None:
                stereo = StereoResult(
                    left_eye=output_rgb,
                    right_eye=output_rgb,
                    sbs=fused_sbs,
                    debug_info={
                        "backend": stereo_config.backend,
                        "sbs_backend": "triton_fast_plus_fused_half_sbs_uint8",
                        "fast_plus_fused_backend": "triton_half_sbs_uint8",
                        "fast_plus_fused_temporal_bypass": int(bool(stereo_config.temporal)),
                        "occlusion_mask_backend": "triton_fused_radius1",
                        "hole_fill_backend": "triton_fused_directional_4tap",
                    },
                )
            elif deferred_vulkan_request is not None:
                # The deferred branch already created a lightweight placeholder;
                # the Intel network sink performs the actual image dispatch.
                pass
            else:
                stereo_config_for_frame = stereo_config
                if skip_sbs_output:
                    stereo_config_for_frame = replace(stereo_config, output_format="mono")
                stereo = synthesize_stereo(
                    output_rgb,
                    depth,
                    stereo_config_for_frame,
                    temporal_state=self.temporal_state,
                    sbs_only=not skip_sbs_output,
                )
                cuda_events.update(getattr(stereo, "cuda_timing_events", None) or {})
                stereo.debug_info.setdefault("fast_plus_fused_backend", "not_used")
                stereo.debug_info.setdefault("fast_plus_fused_skip", fused_skip)
                if stereo_config_for_frame is not stereo_config:
                    stereo.debug_info["sbs_backend"] = "openxr_eyes_only"
                    stereo.debug_info["make_sbs_ms"] = 0.0
        synthesis_ms = (time.perf_counter() - synth_start) * 1000.0
        _record_cuda_event(cuda_events, "synthesis", rgb_frame)
        total_ms = (time.perf_counter() - total_start) * 1000.0

        timing = {
            "depth_preprocess_ms": float(profile.preprocess_ms),
            "depth_model_ms": float(profile.model_ms),
            "depth_slot_wait_ms": float(profile.slot_wait_ms),
            "depth_postprocess_ms": float(profile.postprocess_ms),
            "depth_total_ms": float(depth_total_ms),
            "synthesis_ms": float(synthesis_ms),
            "total_ms": float(total_ms),
        }
        memory = self._collect_memory_stats(rgb_frame)
        self.last_timing = timing
        self.last_memory = memory
        self.stats.update(timing, memory)

        debug = dict(stereo.debug_info)
        debug["runtime_depth_backend"] = self.depth_config.backend
        debug["runtime_output_format"] = self.stereo_config.output_format
        debug["packing_format"] = self.stereo_config.output_format
        debug["runtime_depth_upsample"] = self.config.depth_upsample
        debug["vulkan_zero_copy_request"] = int(deferred_vulkan_request is not None)
        debug["vulkan_zero_copy_reason"] = str(deferred_vulkan_reason)
        debug["active_settings_version"] = int(self.active_settings_version)
        debug["hot_reload_class"] = self.last_settings_change_class
        debug["hot_reload_changed_fields"] = list(self.last_settings_changed_fields)
        _consume_pending_temporal_reset_reasons(self, debug)
        _add_active_settings_debug_info(debug, self.active_settings_snapshot)
        _add_runtime_config_debug_info(debug, stereo_config)
        debug.update(convergence_debug)
        provider_info = self.provider_report()
        _add_depth_contract_debug_info(debug, depth, provider_info)
        _add_preprocess_debug_info(debug, rgb_frame)
        if memory:
            debug.update(memory)

        sbs = stereo.sbs
        pack_start = time.perf_counter()
        if deferred_warp:
            # The viewer consumes rgb+depth directly; nothing to pack.
            debug["runtime_output_pack_backend"] = "metal_shader_warp"
            debug["runtime_output_dtype"] = "deferred_to_viewer"
            sbs = output_rgb
        elif skip_sbs_output:
            debug["runtime_output_pack_backend"] = "openxr_eyes_only"
            debug["runtime_output_dtype"] = str(sbs.dtype).replace("torch.", "")
        elif _runtime_output_uint8_enabled() and sbs.is_floating_point():
            # The packed SBS is authoritative. Direct quality_4k synthesis can
            # intentionally retain source RGB placeholders in left_eye/right_eye;
            # rebuilding from those placeholders duplicates the same eye.
            sbs = sbs.detach().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
            debug["runtime_output_pack_backend"] = "torch_packed_sbs_to_uint8"
            debug["runtime_output_dtype"] = "uint8"
        else:
            if sbs.dtype == torch.uint8:
                debug.setdefault("runtime_output_pack_backend", debug.get("fast_plus_fused_backend", "prepacked_uint8"))
                debug["runtime_output_dtype"] = "uint8"
            else:
                debug["runtime_output_dtype"] = str(sbs.dtype).replace("torch.", "")
        output_eye_size, output_display_size = _add_runtime_output_size_debug_info(debug, stereo.left_eye, sbs)
        pack_ms = (time.perf_counter() - pack_start) * 1000.0
        _record_cuda_event(cuda_events, "pack", rgb_frame)
        _record_cuda_event(cuda_events, "end", rgb_frame)
        timing["pack_ms"] = float(pack_ms)
        # Host presentation frame: pack HWC RGBA8 on the device and pull one
        # copy on the runtime thread (its MPS queue is drained here), so the
        # viewer thread does a plain memcpy instead of an MPS sync mid-frame.
        viewer_frame_np = None
        # When the dedicated packer thread owns packing
        # (D2S_PACKER_THREAD installed entry-side, signaled here as
        # D2S_RUNTIME_INLINE_HOST_PACK=0), ship the device SBS untouched so
        # the runtime loop never waits on the MPS->host copy.
        inline_host_pack = os.environ.get(
            "D2S_RUNTIME_INLINE_HOST_PACK", "1"
        ) not in {"0", "false", "off"}
        if _viewer_host_frame_enabled() and inline_host_pack and sbs is not None:
            host_start = time.perf_counter()
            host_parts: dict[str, float] = {}
            if os.environ.get("D2S_SBS_HOST_SYNC_PRE", "0") == "1":
                sync_start = time.perf_counter()
                torch.mps.synchronize()
                timing["sbs_host_presync_ms"] = (
                    time.perf_counter() - sync_start
                ) * 1000.0
            viewer_frame_np = _pack_sbs_host_frame(sbs, timings=host_parts)
            timing["sbs_host_ms"] = (time.perf_counter() - host_start) * 1000.0
            for key, value in host_parts.items():
                timing[f"sbs_host_{key}_ms"] = float(value)
            if viewer_frame_np is not None:
                debug["runtime_output_pack_backend"] = (
                    str(debug.get("runtime_output_pack_backend", "")) + "+host_np"
                ).lstrip("+")
        slow_log_ms = float(os.environ.get("D2S_SLOW_RUNTIME_LOG_MS", "200") or "200")
        refresh_log_s = float(os.environ.get("D2S_RUNTIME_FRAME_LOG_REFRESH_S", "5") or "0")
        now_log = time.perf_counter()
        is_slow_frame = total_ms >= slow_log_ms
        is_refresh_frame = refresh_log_s > 0.0 and (now_log - self._last_runtime_perf_log_ts) >= refresh_log_s
        if is_slow_frame or is_refresh_frame:
            self._last_runtime_perf_log_ts = now_log
            depth_accounted_ms = float(profile.preprocess_ms) + float(profile.model_ms) + float(profile.postprocess_ms)
            depth_gap_ms = max(0.0, float(depth_total_ms) - depth_accounted_ms)
            log_kind = "slow frame" if is_slow_frame else "frame refresh"
            should_log_perf = True
            if log_kind == "frame refresh":
                should_log_perf = self._runtime_frame_refresh_log_count < 5
                if should_log_perf:
                    self._runtime_frame_refresh_log_count += 1
            if should_log_perf:
                LOGGER.debug(
                    f"[StereoRuntime] {log_kind}:"
                    f" total_ms={total_ms:.1f}"
                    f" depth_total_ms={depth_total_ms:.1f}"
                    f" depth_pre_ms={float(profile.preprocess_ms):.1f}"
                    f" depth_model_ms={float(profile.model_ms):.1f}"
                    f" depth_post_ms={float(profile.postprocess_ms):.1f}"
                    f" depth_gap_ms={depth_gap_ms:.1f}"
                    f" synthesis_ms={synthesis_ms:.1f}"
                    f" pack_ms={pack_ms:.1f}"
                    f" backend={debug.get('backend', stereo_config.backend)}"
                    f" depth_pop={stereo_config.depth_pop:.3f}"
                    f" antialias={stereo_config.depth_antialias_strength:.3f}"
                    f" output_dtype={debug.get('runtime_output_dtype', sbs.dtype)}"
                    f" pack_backend={debug.get('runtime_output_pack_backend', 'n/a')}"
                    f" sbs_backend={debug.get('sbs_backend', 'n/a')}"
                    f" fast_plus_fused={debug.get('fast_plus_fused_backend', 'n/a')}"
                    f" fast_plus_skip={debug.get('fast_plus_fused_skip', 'n/a')}"
                    f" stage_scene={float(debug.get('scene_detect_ms', 0.0)):.1f}"
                    f" stage_layered={float(debug.get('layered_total_ms', 0.0)):.1f}"
                    f" stage_depth_shift={float(debug.get('depth_postprocess_shift_ms', 0.0)):.1f}"
                    f" stage_warp={float(debug.get('warp_composite_ms', 0.0)):.1f}"
                    f" stage_occ={float(debug.get('occlusion_ms', 0.0)):.1f}"
                    f" stage_fill={float(debug.get('hole_fill_ms', 0.0)):.1f}"
                    f" stage_refine={float(debug.get('refine_ms', 0.0)):.1f}"
                    f" stage_temporal={float(debug.get('temporal_ms', 0.0)):.1f}"
                    f" stage_output_depth={float(debug.get('output_depth_ms', 0.0)):.1f}"
                    f" stage_sbs_backend={float(debug.get('sbs_backend_ms', 0.0)):.1f}"
                    f" stage_sbs={float(debug.get('make_sbs_ms', 0.0)):.1f}"
                    f" stage_synth_gap={float(debug.get('synthesis_unaccounted_ms', 0.0)):.1f}"
                )

        if deferred_warp:
            _vd_post = _warp_depth_postprocess(depth, stereo_config, output_rgb)
            if fused_warp_active():
                # Fused warp-pack reads f32 directly: antialias only.
                # Temporal stabilize damps ANE-fp16 / model dither that
                # native-res warp otherwise amplifies into edge shimmer.
                viewer_depth = _rz_ship(
                    self._temporal_stabilize_depth(_vd_post), raw=depth
                )
            else:
                viewer_depth = _rz_ship(
                    _quantize_depth_for_warp(_vd_post), raw=depth
                )
        else:
            viewer_depth = None

        return StereoRuntimeResult(
            depth=depth,
            left_eye=stereo.left_eye,
            right_eye=stereo.right_eye,
            sbs=sbs,
            output_eye_size=output_eye_size,
            output_display_size=output_display_size,
            output_format=str(debug.get("runtime_output_format")),
            output_dtype=str(debug.get("runtime_output_dtype")),
            output_pack_backend=_optional_debug_str(debug.get("runtime_output_pack_backend")),
            viewer_rgb=output_rgb if deferred_warp else None,
            viewer_depth=viewer_depth,
            viewer_frame_np=viewer_frame_np,
            viewer_bgra=viewer_bgra,
            active_settings_version=int(self.active_settings_version),
            hot_reload_class=self.last_settings_change_class,
            hot_reload_changed_fields=tuple(self.last_settings_changed_fields),
            debug_info=debug,
            timing=timing,
            provider_info=provider_info,
            cuda_ready_event=None,
            cuda_timing_events=cuda_events,
            vulkan_compute_request=deferred_vulkan_request,
        )

    def process_openxr_frame(
        self,
        rgb_frame: torch.Tensor,
        openxr_config: OpenXRRenderConfig | None = None,
        *,
        depth_profile: DepthProfileResult | None = None,
    ) -> OpenXRRuntimeResult:
        if not self._active:
            raise RuntimeError("StereoRuntime inference is paused")
        self.load()
        self._reset_cuda_peak_if_needed()
        rgb_frame = _validate_runtime_rgb_frame(rgb_frame)

        cuda_events: dict[str, Any] = {}
        _record_cuda_event(cuda_events, "start", rgb_frame)
        total_start = time.perf_counter()
        depth_start = time.perf_counter()
        profile = depth_profile or self._predict_depth_profile(rgb_frame)
        depth_total_ms = (
            float(profile.total_ms)
            if depth_profile is not None
            else (time.perf_counter() - depth_start) * 1000.0
        )
        cuda_events.update(getattr(profile, "cuda_timing_events", None) or {})
        depth = profile.depth
        output_rgb = _apply_color_adjustment(rgb_frame, self.config)
        output_mode = str(getattr(openxr_config, "output_mode", "auto") or "auto")
        if output_mode == "rgb_depth":
            prewarp_eyes = False
        elif output_mode == "full_synthesis_eyes":
            prewarp_eyes = True
        else:
            prewarp_eyes = _openxr_prewarp_eyes_enabled()
        stereo_config, convergence_debug = _dynamic_convergence_config_for_depth(
            self,
            depth,
            self.stereo_config,
            prefer_gpu_tensor=prewarp_eyes,
        )
        convergence = stereo_config.convergence
        if openxr_config is not None:
            openxr_config_for_frame = replace(
                openxr_config,
                convergence=convergence,
                foreground_shift_scale=float(getattr(stereo_config, "foreground_shift_scale", 1.0)),
                midground_shift_scale=float(getattr(stereo_config, "midground_shift_scale", 1.0)),
                background_shift_scale=float(getattr(stereo_config, "background_shift_scale", 1.0)),
            )
        else:
            openxr_config_for_frame = OpenXRRenderConfig(
                depth_strength=float(stereo_config.depth_strength),
                convergence=convergence,
                max_disparity_px=stereo_config.max_disparity_px,
                parallax_preset=str(stereo_config.parallax_preset),
                foreground_shift_scale=float(getattr(stereo_config, "foreground_shift_scale", 1.0)),
                midground_shift_scale=float(getattr(stereo_config, "midground_shift_scale", 1.0)),
                background_shift_scale=float(getattr(stereo_config, "background_shift_scale", 1.0)),
            )
        openxr_stereo_config = replace(
            stereo_config,
            depth_strength=max(0.0, float(openxr_config_for_frame.depth_strength)),
        )
        if bool(getattr(openxr_stereo_config, "output_quality_enabled", False)):
            # The shared quality stage runs on completed eye images before any
            # output format branches, so OpenXR cannot remain on rgb+depth DIBR.
            prewarp_eyes = True
        _record_cuda_event(cuda_events, "depth", rgb_frame)

        openxr_render_ms = 0.0
        synthesis_ms = 0.0
        pack_ms = 0.0
        pack_backend = "none"
        source_rgb = rgb_frame
        raw_depth = depth
        visual_regression_dir: str | None = None
        vulkan_compute_request: VulkanComputeRequest | None = None
        deferred_reason = "prewarp_disabled"
        if prewarp_eyes:
            deferred_request, deferred_reason = self._build_vulkan_compute_request(
                output_rgb,
                depth,
                openxr_stereo_config,
            )
            # OpenXR must defer the final stereo dispatch to the Presenter so
            # it can write directly into Presenter-owned images. This remains
            # true even if the runtime used its regular Vulkan backend for an
            # earlier non-OpenXR frame.
            if deferred_request is not None:
                vulkan_compute_request = deferred_request
                visual_regression_dir = self._maybe_dump_openxr_rgb_depth(
                    source_rgb=output_rgb,
                    raw_depth=raw_depth,
                    prepared_depth=deferred_request.depth,
                )
                left_eye = output_rgb
                right_eye = output_rgb
                output_format = "openxr_eye_views"
                from .vulkan_stereo_pass import vulkan_hole_fill_backend_name

                render_backend = {
                    "backend": str(openxr_stereo_config.backend),
                    "sbs_backend": "vulkan_deferred_stereo",
                    "warp_composite_backend": "vulkan_layered_stereo_output_image",
                    "occlusion_mask_backend": "vulkan_layered_stereo_output_image",
                    "hole_fill_backend": vulkan_hole_fill_backend_name(
                        deferred_request.params.hole_fill_mode
                    ),
                    "vulkan_hole_fill_mode": int(deferred_request.params.hole_fill_mode),
                    "stereo_compute_backend": "vulkan",
                    "vulkan_openxr_prewarp": 1,
                    "vulkan_zero_copy_request": 1,
                    "vulkan_zero_copy_deferred": 1,
                    "vulkan_zero_copy_reason": deferred_reason,
                }
                render_backend["vulkan_openxr_prewarp"] = 1
                _record_cuda_event(cuda_events, "openxr_render", rgb_frame)
            else:
                vulkan_stereo, vulkan_skip = self._try_vulkan_fused_stereo(
                    output_rgb,
                    depth,
                    openxr_stereo_config,
                    skip_sbs_output=True,
                )
                if vulkan_stereo is not None:
                    left_eye = vulkan_stereo.left_eye
                    right_eye = vulkan_stereo.right_eye
                    if bool(getattr(openxr_stereo_config, "cross_eyed", False)):
                        left_eye, right_eye = right_eye, left_eye
                    output_format = "openxr_eye_views"
                    render_backend = dict(vulkan_stereo.debug_info)
                    render_backend["vulkan_openxr_prewarp"] = 1
                    render_backend["vulkan_fused_skip"] = vulkan_skip
                    _record_cuda_event(cuda_events, "openxr_render", rgb_frame)
                elif _is_triton_stereo_compute_backend(self._resolve_stereo_compute_backend(output_rgb)):
                    render_start = time.perf_counter()
                    if output_quality_requires_eye_images(
                        openxr_stereo_config,
                        int(output_rgb.shape[-1]),
                        int(output_rgb.shape[-2]),
                    ):
                        no_fill_fused = None
                        no_fill_fused_reason = "common_output_quality_requires_eyes"
                    else:
                        no_fill_fused, no_fill_fused_reason = _try_openxr_no_fill_fused_rgba_u8(
                            output_rgb,
                            depth,
                            openxr_stereo_config,
                            cuda_events,
                        )
                    if no_fill_fused is not None:
                        synthesis_left, synthesis_right, fused_debug = no_fill_fused
                        left_eye = synthesis_left
                        right_eye = synthesis_right
                        if bool(getattr(openxr_stereo_config, "cross_eyed", False)):
                            left_eye, right_eye = right_eye, left_eye
                        output_format = "openxr_eye_views"
                        render_backend = dict(fused_debug)
                        render_backend["sbs_backend"] = "openxr_triton_no_fill_fused_rgba_u8"
                        render_backend["stereo_compute_backend"] = str(
                            self._resolved_stereo_compute_backend or "cuda_triton"
                        )
                        render_backend["openxr_prewarp_backend"] = "triton_no_fill_fused_rgba_u8"
                        render_backend["openxr_no_fill_fused_reason"] = no_fill_fused_reason
                        render_backend["openxr_grid_sample_fallback"] = 0
                        pack_backend = "triton_warp_composite2_rgba_u8"
                    else:
                        # Reuse the canonical synthesis path when the dedicated
                        # no-fill kernel cannot preserve the selected behavior.
                        triton_config = replace(openxr_stereo_config, output_format="mono")
                        triton_stereo = synthesize_stereo(
                            output_rgb,
                            depth,
                            triton_config,
                            temporal_state=self.temporal_state,
                        )
                        cuda_events.update(getattr(triton_stereo, "cuda_timing_events", None) or {})

                        left_eye = triton_stereo.left_eye
                        right_eye = triton_stereo.right_eye
                        output_format = "openxr_eye_views"
                        render_backend = dict(triton_stereo.debug_info)
                        render_backend["sbs_backend"] = "openxr_triton_eyes_only"
                        render_backend["stereo_compute_backend"] = str(
                            self._resolved_stereo_compute_backend or "cuda_triton"
                        )
                        render_backend["openxr_prewarp_backend"] = "triton_full_synthesis_eyes"
                        render_backend["openxr_no_fill_fused_reason"] = no_fill_fused_reason
                        render_backend["openxr_grid_sample_fallback"] = 0

                        pack_start = time.perf_counter()
                        _record_cuda_event(cuda_events, "openxr_pack_start", left_eye)
                        if _openxr_runtime_output_uint8_enabled():
                            packed_left, left_pack_backend = _pack_openxr_eye_rgba_u8_with_backend(left_eye)
                            packed_right, right_pack_backend = _pack_openxr_eye_rgba_u8_with_backend(right_eye)
                            if packed_left is not left_eye or packed_right is not right_eye:
                                left_eye = packed_left
                                right_eye = packed_right
                                pack_backend = _merge_openxr_rgba_pack_backend(left_pack_backend, right_pack_backend)
                        pack_ms = (time.perf_counter() - pack_start) * 1000.0
                        _record_cuda_event(cuda_events, "openxr_pack", left_eye)

                    openxr_render_ms = (time.perf_counter() - render_start) * 1000.0
                    synthesis_ms = openxr_render_ms
                    _record_cuda_event(cuda_events, "openxr_render", left_eye)
                else:
                    render_start = time.perf_counter()
                    openxr = render_openxr_stereo(output_rgb, depth, openxr_config_for_frame)
                    openxr_render_ms = (time.perf_counter() - render_start) * 1000.0
                    _record_cuda_event(cuda_events, "openxr_render", rgb_frame)

                    pack_start = time.perf_counter()
                    left_eye = openxr.left_eye
                    right_eye = openxr.right_eye
                    if bool(getattr(openxr_stereo_config, "cross_eyed", False)):
                        left_eye, right_eye = right_eye, left_eye
                    left_eye, right_eye, quality_debug = apply_output_quality(
                        left_eye,
                        right_eye,
                        openxr_stereo_config,
                    )
                    _record_cuda_event(cuda_events, "openxr_pack_start", left_eye)
                    if _openxr_runtime_output_uint8_enabled():
                        packed_left, left_pack_backend = _pack_openxr_eye_rgba_u8_with_backend(left_eye)
                        packed_right, right_pack_backend = _pack_openxr_eye_rgba_u8_with_backend(right_eye)
                        if packed_left is not left_eye or packed_right is not right_eye:
                            left_eye = packed_left
                            right_eye = packed_right
                            pack_backend = _merge_openxr_rgba_pack_backend(left_pack_backend, right_pack_backend)
                    pack_ms = (time.perf_counter() - pack_start) * 1000.0
                    _record_cuda_event(cuda_events, "openxr_pack", left_eye)
                    output_format = "openxr_eye_views"
                    render_backend = dict(openxr.debug_info)
                    render_backend.update(quality_debug)
                    render_backend["openxr_prewarp_backend"] = "grid_sample_fallback"
                    render_backend["openxr_grid_sample_fallback"] = 1
                    render_backend["openxr_grid_sample_fallback_reason"] = str(vulkan_skip)
        else:
            depth = self._prepare_openxr_rgb_depth(depth)
            _record_cuda_event(cuda_events, "openxr_depth_prepare", rgb_frame)
            visual_regression_dir = self._maybe_dump_openxr_rgb_depth(
                source_rgb=source_rgb,
                raw_depth=raw_depth,
                prepared_depth=depth,
            )
            left_eye = output_rgb
            right_eye = output_rgb
            output_format = "openxr_rgb_depth"
            render_backend = {"backend": "openxr_viewer_shader_dibr"}
        if visual_regression_dir is None:
            # Keep the automatic input/depth evidence available even when a
            # frame cannot use the deferred Vulkan request and falls back.
            visual_regression_dir = self._maybe_dump_openxr_rgb_depth(
                source_rgb=output_rgb,
                raw_depth=raw_depth,
                prepared_depth=depth,
            )
        total_ms = (time.perf_counter() - total_start) * 1000.0
        _record_cuda_event(cuda_events, "end", left_eye if isinstance(left_eye, torch.Tensor) else rgb_frame)

        timing = {
            "depth_preprocess_ms": float(profile.preprocess_ms),
            "depth_model_ms": float(profile.model_ms),
            "depth_slot_wait_ms": float(profile.slot_wait_ms),
            "depth_postprocess_ms": float(profile.postprocess_ms),
            "depth_total_ms": float(depth_total_ms),
            "synthesis_ms": float(synthesis_ms),
            "openxr_render_ms": float(openxr_render_ms),
            "pack_ms": float(pack_ms),
            "total_ms": float(total_ms),
        }
        memory = self._collect_memory_stats(rgb_frame)
        self.last_timing = timing
        self.last_memory = memory
        self.stats.update(timing, memory)

        debug = dict(render_backend)
        debug["application_runtime_target"] = "openxr"
        debug["stereo_synthesis_mode"] = "full_synthesis_eyes" if prewarp_eyes else "rgb_depth_direct"
        debug["vulkan_zero_copy_request"] = int(vulkan_compute_request is not None)
        debug["vulkan_zero_copy_reason"] = str(deferred_reason)
        debug["runtime_depth_backend"] = self.depth_config.backend
        debug["runtime_depth_execution_slot"] = getattr(profile, "execution_slot", None)
        debug["runtime_depth_execution_slot_count"] = max(
            1,
            int(getattr(profile, "execution_slot_count", 1)),
        )
        debug["runtime_output_format"] = output_format
        debug["packing_format"] = "none"
        debug["active_settings_version"] = int(self.active_settings_version)
        debug["runtime_output_dtype"] = _runtime_eye_dtype(left_eye, right_eye)
        if visual_regression_dir is not None:
            debug["visual_regression_dir"] = visual_regression_dir
        debug["cross_eyed"] = int(bool(getattr(stereo_config, "cross_eyed", False)))
        debug["hot_reload_class"] = self.last_settings_change_class
        debug["hot_reload_changed_fields"] = list(self.last_settings_changed_fields)
        _consume_pending_temporal_reset_reasons(self, debug)
        _add_active_settings_debug_info(debug, self.active_settings_snapshot)
        _add_runtime_config_debug_info(debug, openxr_stereo_config)
        debug.update(convergence_debug)
        provider_info = self.provider_report()
        _add_depth_contract_debug_info(debug, depth, provider_info)
        output_eye_size, output_display_size = _add_runtime_output_size_debug_info(debug, left_eye, left_eye)
        debug["runtime_output_pack_backend"] = pack_backend
        shader_uniforms = None
        if openxr_config is not None:
            shader_uniforms = _add_openxr_config_debug_info(debug, openxr_config_for_frame, left_eye)
        debug["runtime_depth_upsample"] = self.config.depth_upsample
        _add_preprocess_debug_info(debug, rgb_frame)
        if memory:
            debug.update(memory)
        if total_ms >= float(os.environ.get("D2S_SLOW_RUNTIME_LOG_MS", "120") or "120") and os.environ.get('D2S_DEBUG', '0') in ('1', 'true', 'yes', 'on'):
            print(
                "[StereoRuntime] slow openxr frame:"
                f" total_ms={total_ms:.1f}"
                f" depth_total_ms={depth_total_ms:.1f}"
                f" depth_model_ms={float(profile.model_ms):.1f}"
                f" depth_postprocess_ms={float(profile.postprocess_ms):.1f}"
                f" openxr_render_ms={openxr_render_ms:.1f}"
                f" pack_ms={pack_ms:.1f}"
                f" output_dtype={debug.get('runtime_output_dtype', 'n/a')}"
                f" eye_size={debug.get('runtime_output_eye_size', 'n/a')}"
                f" render_backend={debug.get('backend', debug.get('openxr_backend', 'n/a'))}"
                f" depth_backend={debug.get('runtime_depth_backend', 'n/a')}",
                flush=True,
            )

        return OpenXRRuntimeResult(
            depth=depth,
            left_eye=left_eye,
            right_eye=right_eye,
            source_rgb=source_rgb,
            output_eye_size=output_eye_size,
            output_display_size=output_display_size,
            output_format=str(debug.get("runtime_output_format")),
            output_dtype=str(debug.get("runtime_output_dtype")),
            output_pack_backend=_optional_debug_str(debug.get("runtime_output_pack_backend")),
            active_settings_version=int(self.active_settings_version),
            hot_reload_class=self.last_settings_change_class,
            hot_reload_changed_fields=tuple(self.last_settings_changed_fields),
            shader_uniforms=shader_uniforms,
            debug_info=debug,
            timing=timing,
            provider_info=provider_info,
            cuda_timing_events=cuda_events,
            vulkan_compute_request=vulkan_compute_request,
        )

    def _build_vulkan_compute_request(
        self,
        rgb_frame: torch.Tensor,
        depth: torch.Tensor,
        stereo_config: Any,
    ) -> tuple[VulkanComputeRequest | None, str]:
        """Build a Presenter-owned Vulkan request without touching Vulkan here."""
        if output_quality_requires_eye_images(
            stereo_config, int(rgb_frame.shape[-1]), int(rgb_frame.shape[-2])
        ):
            return None, "common_output_quality_requires_eyes"
        backend = str(getattr(stereo_config, "backend", ""))
        if backend not in {"fast_plus", "quality_4k", "hq_4k"}:
            return None, f"backend={backend or 'unknown'}"
        if str(getattr(stereo_config, "output_format", "")) == "depth_map":
            return None, "depth_map_not_supported"
        if isinstance(getattr(stereo_config, "convergence", None), torch.Tensor):
            return None, "dynamic_convergence_tensor"
        if bool(getattr(stereo_config, "refine", False)):
            return None, "local_refine_not_supported"
        # Temporal blending and eye swapping require a prior output image or a
        # separate image pass. Keep those semantics on the existing path until
        # their Vulkan stateful kernels are migrated.
        if bool(getattr(stereo_config, "temporal", False)):
            return None, "temporal_stateful_kernel_not_supported"
        if bool(getattr(stereo_config, "cross_eyed", False)):
            return None, "cross_eyed_output_not_supported"
        if self._resolve_stereo_compute_backend(rgb_frame) != "vulkan":
            return None, f"selected={self._resolved_stereo_compute_backend}"
        try:
            processed_depth = postprocess_depth(
                match_depth(depth, rgb_frame.shape[-2], rgb_frame.shape[-1]),
                depth_pop=float(getattr(stereo_config, "depth_pop", 0.0)),
                antialias_strength=float(getattr(stereo_config, "depth_antialias_strength", 0.0)),
            )
            budget = resolve_parallax_budget(
                render_width=int(rgb_frame.shape[-1]),
                render_height=int(rgb_frame.shape[-2]),
                preset=getattr(stereo_config, "parallax_preset", "standard"),
                convergence=float(getattr(stereo_config, "convergence", 0.0)),
                max_disparity_px=getattr(stereo_config, "max_disparity_px", None),
            )
            from .vulkan_stereo_pass import (
                VulkanLayeredStereoParams,
                resolve_vulkan_hole_fill_mode,
                resolve_vulkan_hole_fill_parameters,
            )

            hole_fill = str(getattr(stereo_config, "hole_fill", "edge_aware")).strip().lower()
            hole_fill_mode = str(getattr(stereo_config, "hole_fill_mode", "balanced")).strip().lower()
            vulkan_hole_fill_mode = resolve_vulkan_hole_fill_mode(hole_fill, hole_fill_mode)
            fill_radius, fill_strength = resolve_vulkan_hole_fill_parameters(
                vulkan_hole_fill_mode,
                fill_radius=getattr(stereo_config, "hole_fill_radius", 3),
                fill_strength=getattr(stereo_config, "hole_fill_strength", 1.0),
            )
            params = VulkanLayeredStereoParams(
                depth_strength=max(0.0, float(getattr(stereo_config, "depth_strength", 1.0))),
                max_disparity_px=float(budget.max_disparity_px),
                convergence=float(getattr(stereo_config, "convergence", 0.0)),
                edge_threshold=float(getattr(stereo_config, "edge_threshold", 0.04)),
                fill_strength=fill_strength,
                fill_radius=fill_radius,
                mask_feather_radius=max(0, min(3, int(getattr(stereo_config, "mask_feather_radius", 3)))),
                symmetric=bool(getattr(stereo_config, "symmetric", True)),
                layers=(1 if backend == "fast_plus" else max(1, min(4, int(getattr(stereo_config, "layers", 2))))),
                softness=0.08,
                foreground_scale=max(0.0, float(getattr(stereo_config, "foreground_shift_scale", 1.0))),
                midground_scale=max(0.0, float(getattr(stereo_config, "midground_shift_scale", 1.0))),
                background_scale=max(0.0, float(getattr(stereo_config, "background_shift_scale", 1.0))),
                edge_dilation=max(0, min(3, int(getattr(stereo_config, "edge_dilation", 2)))),
                screen_edge_suppression=max(0, int(getattr(stereo_config, "screen_edge_mask_suppression", 0))),
                hole_fill_mode=vulkan_hole_fill_mode,
                occlusion_enabled=bool(getattr(stereo_config, "occlusion", True)),
            )
            return VulkanComputeRequest(rgb=rgb_frame, depth=processed_depth, params=params), "ready"
        except Exception as exc:
            return None, f"request_failed:{type(exc).__name__}"

    def _prepare_openxr_rgb_depth(self, depth: torch.Tensor) -> torch.Tensor:
        depth = depth.detach().contiguous().float().clamp(0.0, 1.0)
        depth = _openxr_rgb_depth_percentile_normalize(depth, percentile=_openxr_rgb_depth_percentile())
        gamma = _openxr_rgb_depth_gamma()
        if abs(gamma - 1.0) > 1e-4:
            depth = depth.pow(gamma)
        depth = postprocess_depth(
            depth,
            depth_pop=float(getattr(self.stereo_config, "depth_pop", 0.0)),
            antialias_strength=float(getattr(self.stereo_config, "depth_antialias_strength", 0.0)),
        )
        return self._stabilize_openxr_rgb_depth(
            depth,
            enabled=bool(getattr(self.stereo_config, "temporal", False)),
        )

    def _maybe_dump_openxr_rgb_depth(
        self,
        *,
        source_rgb: torch.Tensor,
        raw_depth: torch.Tensor,
        prepared_depth: torch.Tensor,
    ) -> str | None:
        # Runtime visual regression is opt-in. The normal application path
        # must not perform host readback or write diagnostic images.
        dump_dir = str(
            self.config.openxr_visual_regression_dir
            or os.environ.get("D2S_OPENXR_RGB_DEPTH_DUMP_DIR", "")
        ).strip()
        if not dump_dir:
            return None
        out_dir = Path(dump_dir)
        if self._openxr_rgb_depth_dumped:
            return str(out_dir)
        try:
            from .io import save_depth, save_rgb

            out_dir.mkdir(parents=True, exist_ok=True)
            save_rgb(source_rgb.detach().float().clamp(0.0, 1.0), out_dir / "00_capture_rgb.png")
            save_depth(raw_depth.detach().float().clamp(0.0, 1.0), out_dir / "01_raw_depth.png")
            save_depth(prepared_depth.detach().float().clamp(0.0, 1.0), out_dir / "02_prepared_depth.png")
            print(f"[StereoRuntime] OpenXR visual regression capture saved: {out_dir}", flush=True)
            self._openxr_rgb_depth_dumped = True
            return str(out_dir)
        except Exception as exc:
            print(f"[StereoRuntime] OpenXR visual regression capture failed: {type(exc).__name__}: {exc}", flush=True)
            return None

    def _stabilize_openxr_rgb_depth(
        self,
        depth: torch.Tensor,
        *,
        enabled: bool = True,
    ) -> torch.Tensor:
        alpha = _openxr_rgb_depth_temporal_alpha() if enabled else 0.0
        depth = depth.detach().contiguous().float()
        if alpha <= 0.0:
            self._openxr_depth_temporal = None
            return depth
        prev = self._openxr_depth_temporal
        if prev is None or prev.shape != depth.shape or prev.device != depth.device:
            self._openxr_depth_temporal = depth
            return depth
        out = prev.mul(alpha).add(depth, alpha=(1.0 - alpha))
        self._openxr_depth_temporal = out.detach()
        return out

    def _predict_depth_profile(self, rgb_frame: torch.Tensor) -> DepthProfileResult:
        predict_profile = getattr(self.depth_provider, "predict_profile", None)
        if callable(predict_profile):
            result = predict_profile(rgb_frame)
            if isinstance(result, DepthProfileResult):
                return result
            depth = getattr(result, "depth", None)
            if depth is not None:
                return DepthProfileResult(
                    depth=depth,
                    preprocess_ms=float(getattr(result, "preprocess_ms", 0.0)),
                    model_ms=float(getattr(result, "model_ms", 0.0)),
                    postprocess_ms=float(getattr(result, "postprocess_ms", 0.0)),
                    slot_wait_ms=float(getattr(result, "slot_wait_ms", 0.0)),
                )

        start = time.perf_counter()
        depth = self.depth_provider.predict(rgb_frame)
        elapsed = (time.perf_counter() - start) * 1000.0
        return DepthProfileResult(depth=depth, preprocess_ms=0.0, model_ms=float(elapsed), postprocess_ms=0.0)

    def predict_openxr_depth(self, rgb_frame: torch.Tensor) -> DepthProfileResult:
        """Run only depth inference for the bounded parallel scheduler."""
        if not self._active:
            raise RuntimeError("StereoRuntime inference is paused")
        self.load()
        return self._predict_depth_profile(_validate_runtime_rgb_frame(rgb_frame))


    def _try_fast_plus_fused_sbs(self, rgb_frame: torch.Tensor, depth: torch.Tensor, stereo_config: Any) -> tuple[torch.Tensor | None, str]:
        if not _fast_plus_fused_enabled():
            return None, "disabled"
        if stereo_config.backend != "fast_plus":
            return None, f"backend={stereo_config.backend}"
        if str(getattr(stereo_config, "hole_fill", "edge_aware")).strip().lower() == "none":
            return None, "hole_fill_disabled"
        if stereo_config.output_format != "half_sbs":
            return None, f"format={stereo_config.output_format}"
        if not _runtime_output_uint8_enabled():
            return None, "runtime_uint8_off"
        if bool(getattr(stereo_config, "cross_eyed", False)):
            return None, "cross_eyed"
        if bool(getattr(stereo_config, "debug_output", False)):
            return None, "debug_output"
        if output_quality_requires_eye_images(
            stereo_config, int(rgb_frame.shape[-1]), int(rgb_frame.shape[-2])
        ):
            return None, "common_output_quality_requires_eyes"
        if _layered_parallax_enabled(stereo_config):
            return None, "layered_parallax"
        if isinstance(getattr(stereo_config, "convergence", None), torch.Tensor):
            return None, "dynamic_convergence_tensor"
        try:
            from .fast_plus_fused_triton import can_use_fast_plus_fused_half_sbs_uint8, make_fast_plus_fused_half_sbs_uint8
            from .output import match_depth
        except Exception as exc:
            return None, f"import_failed:{type(exc).__name__}"
        depth = match_depth(depth, rgb_frame.shape[-2], rgb_frame.shape[-1])
        if not can_use_fast_plus_fused_half_sbs_uint8(rgb_frame, depth):
            return None, f"unsupported_tensor:rgb={tuple(rgb_frame.shape)}/{rgb_frame.dtype}/{rgb_frame.device};depth={tuple(depth.shape)}/{depth.dtype}/{depth.device}"
        budget = resolve_parallax_budget(
            render_width=int(rgb_frame.shape[-1]),
            render_height=int(rgb_frame.shape[-2]),
            preset=getattr(stereo_config, "parallax_preset", "standard"),
            convergence=float(getattr(stereo_config, "convergence", 0.0)),
            max_disparity_px=getattr(stereo_config, "max_disparity_px", None),
        )
        try:
            return make_fast_plus_fused_half_sbs_uint8(
                rgb_frame,
                depth,
                convergence=float(getattr(stereo_config, "convergence", 0.0)),
                max_disparity_px=float(budget.max_disparity_px),
                depth_strength=max(0.0, float(getattr(stereo_config, "depth_strength", 1.0))),
                edge_threshold=0.03,
            ), "used"
        except Exception as exc:
            return None, f"kernel_failed:{type(exc).__name__}"

    def _resolve_stereo_compute_backend(self, rgb_frame: torch.Tensor) -> str:
        requested = str(getattr(self.config, "stereo_compute_backend", "auto") or "auto")
        probe = probe_triton_runtime(getattr(rgb_frame, "device", None))
        opengl_available, opengl_reason = probe_opengl_stereo_backend()
        if requested.strip().lower() in {"triton", "cuda", "cuda_triton"}:
            requested = "triton"
        try:
            resolved = resolve_stereo_compute_backend(
                requested,
                vendor_id=0,
                cuda_available=bool(getattr(rgb_frame, "is_cuda", False)),
                vulkan_available=self._vulkan_stereo_backend_error is None,
                triton_available=probe.available,
                triton_vendor=probe.vendor,
                opengl_available=opengl_available,
            )
            self._stereo_compute_backend_reason = (
                "selected_by_priority"
                if str(requested).strip().lower() in {"auto", "vendor", "vendor_default"}
                else "explicit_request"
            )
        except Exception as exc:
            self._stereo_compute_backend_reason = (
                f"{type(exc).__name__}: {exc}; opengl_probe={opengl_reason}"
            )
            resolved = resolve_stereo_compute_backend(
                "auto",
                vendor_id=0,
                cuda_available=bool(getattr(rgb_frame, "is_cuda", False)),
                vulkan_available=False,
                opengl_available=opengl_available,
            )
            LOGGER.warning(
                "Stereo synthesis backend fallback: requested=%s resolved=%s reason=%s",
                requested,
                resolved.value,
                self._stereo_compute_backend_reason,
            )
        self._resolved_stereo_compute_backend = str(resolved.value)
        log_key = (
            self._resolved_stereo_compute_backend,
            str(self._stereo_compute_backend_reason),
        )
        if log_key != self._last_stereo_backend_log_key:
            if self._resolved_stereo_compute_backend == "vulkan":
                LOGGER.info(
                    "Stereo synthesis backend selected: Vulkan (reason=%s)",
                    self._stereo_compute_backend_reason,
                )
            elif self._resolved_stereo_compute_backend == "opengl":
                LOGGER.info(
                    "Stereo synthesis backend selected: OpenGL (reason=%s)",
                    self._stereo_compute_backend_reason,
                )
            elif self._resolved_stereo_compute_backend == "cuda_triton":
                LOGGER.info(
                    "Stereo synthesis backend selected: vendor Triton (reason=%s)",
                    self._stereo_compute_backend_reason,
                )
            else:
                LOGGER.warning(
                    "Stereo synthesis fallback: selected=%s reason=%s opengl_probe=%s",
                    self._resolved_stereo_compute_backend,
                    self._stereo_compute_backend_reason,
                    opengl_reason,
                )
            self._last_stereo_backend_log_key = log_key
        return str(resolved.value)

    def _try_vulkan_fused_stereo(
        self,
        rgb_frame: torch.Tensor,
        depth: torch.Tensor,
        stereo_config: Any,
        *,
        skip_sbs_output: bool,
    ) -> tuple[StereoResult | None, str]:
        if output_quality_requires_eye_images(
            stereo_config, int(rgb_frame.shape[-1]), int(rgb_frame.shape[-2])
        ):
            return None, "common_output_quality_requires_eyes"
        backend = str(getattr(stereo_config, "backend", ""))
        if backend in {"quality_4k", "hq_4k"}:
            return self._try_vulkan_layered_stereo(
                rgb_frame,
                depth,
                stereo_config,
                skip_sbs_output=skip_sbs_output,
            )
        if backend != "fast_plus":
            return None, f"backend={getattr(stereo_config, 'backend', 'unknown')}"
        if str(getattr(stereo_config, "output_format", "")) == "depth_map":
            return None, "depth_map_not_supported"
        if _layered_parallax_enabled(stereo_config):
            return None, "layered_parallax"
        if isinstance(getattr(stereo_config, "convergence", None), torch.Tensor):
            return None, "dynamic_convergence_tensor"

        backend_name = self._resolve_stereo_compute_backend(rgb_frame)
        if backend_name != "vulkan":
            return None, f"selected={backend_name}"
        if self._vulkan_stereo_backend_error is not None:
            return None, f"backend_unavailable:{self._vulkan_stereo_backend_error}"

        try:
            if self._vulkan_stereo_backend is None:
                from .vulkan_backend import VulkanStereoComputeBackend

                self._vulkan_stereo_backend = VulkanStereoComputeBackend()
            processed_depth = postprocess_depth(
                match_depth(depth, rgb_frame.shape[-2], rgb_frame.shape[-1]),
                depth_pop=float(getattr(stereo_config, "depth_pop", 0.0)),
                antialias_strength=float(getattr(stereo_config, "depth_antialias_strength", 0.0)),
            )
            budget = resolve_parallax_budget(
                render_width=int(rgb_frame.shape[-1]),
                render_height=int(rgb_frame.shape[-2]),
                preset=getattr(stereo_config, "parallax_preset", "standard"),
                convergence=float(getattr(stereo_config, "convergence", 0.0)),
                max_disparity_px=getattr(stereo_config, "max_disparity_px", None),
            )
            from .vulkan_stereo_pass import (
                VulkanStereoFusedParams,
                resolve_vulkan_hole_fill_mode,
                resolve_vulkan_hole_fill_parameters,
                vulkan_hole_fill_backend_name,
            )

            hole_fill = str(
                getattr(stereo_config, "hole_fill", "edge_aware")
            ).strip().lower()
            hole_fill_mode = str(
                getattr(stereo_config, "hole_fill_mode", "balanced")
            ).strip().lower()
            vulkan_hole_fill_mode = resolve_vulkan_hole_fill_mode(
                hole_fill,
                hole_fill_mode,
            )
            fill_radius, fill_strength = resolve_vulkan_hole_fill_parameters(
                vulkan_hole_fill_mode,
                fill_radius=getattr(stereo_config, "hole_fill_radius", 1),
                fill_strength=getattr(stereo_config, "hole_fill_strength", 0.6),
            )

            left, right, mask, backend_debug = self._vulkan_stereo_backend.submit_frame(
                rgb_frame,
                processed_depth,
                params=VulkanStereoFusedParams(
                    depth_strength=max(0.0, float(getattr(stereo_config, "depth_strength", 1.0))),
                    max_disparity_px=float(budget.max_disparity_px),
                    convergence=float(getattr(stereo_config, "convergence", 0.0)),
                    edge_threshold=float(getattr(stereo_config, "edge_threshold", 0.03)),
                    fill_strength=fill_strength,
                    fill_radius=fill_radius,
                    mask_feather_radius=max(0, min(3, int(getattr(stereo_config, "mask_feather_radius", 3)))),
                    symmetric=bool(getattr(stereo_config, "symmetric", True)),
                    hole_fill_mode=vulkan_hole_fill_mode,
                ),
            )
            if bool(getattr(stereo_config, "temporal", False)):
                left, right = apply_temporal(
                    left,
                    right,
                    mask,
                    self.temporal_state,
                    strength=float(getattr(stereo_config, "temporal_strength", 0.75)),
                )
            if bool(getattr(stereo_config, "cross_eyed", False)):
                left, right = right, left
            output_format = "mono" if skip_sbs_output else str(stereo_config.output_format)
            sbs = make_sbs(left, right, output_format, fused=False)
            debug = {
                "backend": str(stereo_config.backend),
                "sbs_backend": "vulkan_fused_stereo",
                "warp_composite_backend": "vulkan_fused_stereo",
                "occlusion_mask_backend": "vulkan_fused_stereo",
                "hole_fill_backend": vulkan_hole_fill_backend_name(vulkan_hole_fill_mode),
                "vulkan_hole_fill_mode": int(vulkan_hole_fill_mode),
                "stereo_compute_backend": "vulkan",
                "fast_plus_fused_temporal_bypass": 0,
                "occlusion_mask": mask,
                **backend_debug,
            }
            return StereoResult(left_eye=left, right_eye=right, sbs=sbs, debug_info=debug), "used"
        except Exception as exc:
            self._vulkan_stereo_backend_error = f"{type(exc).__name__}: {exc}"
            close = getattr(self._vulkan_stereo_backend, "close", None)
            if callable(close):
                close()
            self._vulkan_stereo_backend = None
            LOGGER.warning("Vulkan stereo fused pass unavailable; falling back: %s", self._vulkan_stereo_backend_error)
            return None, f"kernel_failed:{self._vulkan_stereo_backend_error}"

    def _try_vulkan_layered_stereo(
        self,
        rgb_frame: torch.Tensor,
        depth: torch.Tensor,
        stereo_config: Any,
        *,
        skip_sbs_output: bool,
    ) -> tuple[StereoResult | None, str]:
        if str(getattr(stereo_config, "output_format", "")) == "depth_map":
            return None, "depth_map_not_supported"
        if isinstance(getattr(stereo_config, "convergence", None), torch.Tensor):
            return None, "dynamic_convergence_tensor"
        if bool(getattr(stereo_config, "refine", False)):
            return None, "local_refine_not_supported"

        backend_name = self._resolve_stereo_compute_backend(rgb_frame)
        if backend_name != "vulkan":
            return None, f"selected={backend_name}"
        if self._vulkan_stereo_backend_error is not None:
            return None, f"backend_unavailable:{self._vulkan_stereo_backend_error}"

        try:
            if self._vulkan_stereo_backend is None:
                from .vulkan_backend import VulkanStereoComputeBackend

                self._vulkan_stereo_backend = VulkanStereoComputeBackend()
            processed_depth = postprocess_depth(
                match_depth(depth, rgb_frame.shape[-2], rgb_frame.shape[-1]),
                depth_pop=float(getattr(stereo_config, "depth_pop", 0.0)),
                antialias_strength=float(getattr(stereo_config, "depth_antialias_strength", 0.0)),
            )
            budget = resolve_parallax_budget(
                render_width=int(rgb_frame.shape[-1]),
                render_height=int(rgb_frame.shape[-2]),
                preset=getattr(stereo_config, "parallax_preset", "standard"),
                convergence=float(getattr(stereo_config, "convergence", 0.0)),
                max_disparity_px=getattr(stereo_config, "max_disparity_px", None),
            )
            from .vulkan_stereo_pass import (
                VulkanLayeredStereoParams,
                resolve_vulkan_hole_fill_mode,
                resolve_vulkan_hole_fill_parameters,
                vulkan_hole_fill_backend_name,
            )

            hole_fill = str(getattr(stereo_config, "hole_fill", "edge_aware")).strip().lower()
            hole_fill_mode = str(getattr(stereo_config, "hole_fill_mode", "balanced")).strip().lower()
            vulkan_hole_fill_mode = resolve_vulkan_hole_fill_mode(hole_fill, hole_fill_mode)
            fill_radius, fill_strength = resolve_vulkan_hole_fill_parameters(
                vulkan_hole_fill_mode,
                fill_radius=getattr(stereo_config, "hole_fill_radius", 3),
                fill_strength=getattr(stereo_config, "hole_fill_strength", 1.0),
            )
            left, right, mask, backend_debug = self._vulkan_stereo_backend.submit_layered_frame(
                rgb_frame,
                processed_depth,
                params=VulkanLayeredStereoParams(
                    depth_strength=max(0.0, float(getattr(stereo_config, "depth_strength", 1.0))),
                    max_disparity_px=float(budget.max_disparity_px),
                    convergence=float(getattr(stereo_config, "convergence", 0.0)),
                    edge_threshold=float(getattr(stereo_config, "edge_threshold", 0.04)),
                    fill_strength=fill_strength,
                    fill_radius=fill_radius,
                    mask_feather_radius=max(0, min(3, int(getattr(stereo_config, "mask_feather_radius", 3)))),
                    symmetric=bool(getattr(stereo_config, "symmetric", True)),
                    layers=max(1, min(4, int(getattr(stereo_config, "layers", 2)))),
                    softness=0.08,
                    foreground_scale=max(0.0, float(getattr(stereo_config, "foreground_shift_scale", 1.0))),
                    midground_scale=max(0.0, float(getattr(stereo_config, "midground_shift_scale", 1.0))),
                    background_scale=max(0.0, float(getattr(stereo_config, "background_shift_scale", 1.0))),
                    edge_dilation=max(0, min(3, int(getattr(stereo_config, "edge_dilation", 2)))),
                    screen_edge_suppression=max(0, int(getattr(stereo_config, "screen_edge_mask_suppression", 0))),
                    hole_fill_mode=vulkan_hole_fill_mode,
                    occlusion_enabled=bool(getattr(stereo_config, "occlusion", True)),
                ),
            )
            if bool(getattr(stereo_config, "temporal", False)):
                left, right = apply_temporal(
                    left,
                    right,
                    mask,
                    self.temporal_state,
                    strength=float(getattr(stereo_config, "temporal_strength", 0.75)),
                )
            if bool(getattr(stereo_config, "cross_eyed", False)):
                left, right = right, left
            output_format = "mono" if skip_sbs_output else str(stereo_config.output_format)
            sbs = make_sbs(left, right, output_format, fused=False)
            debug = {
                "backend": str(stereo_config.backend),
                "sbs_backend": "vulkan_layered_stereo",
                "warp_composite_backend": "vulkan_layered_stereo",
                "occlusion_mask_backend": "vulkan_layered_stereo",
                "hole_fill_backend": vulkan_hole_fill_backend_name(vulkan_hole_fill_mode),
                "vulkan_hole_fill_mode": int(vulkan_hole_fill_mode),
                "stereo_compute_backend": "vulkan",
                "layers": int(getattr(stereo_config, "layers", 2)),
                "occlusion_mask": mask,
                **backend_debug,
            }
            return StereoResult(left_eye=left, right_eye=right, sbs=sbs, debug_info=debug), "used"
        except Exception as exc:
            self._vulkan_stereo_backend_error = f"{type(exc).__name__}: {exc}"
            close = getattr(self._vulkan_stereo_backend, "close", None)
            if callable(close):
                close()
            self._vulkan_stereo_backend = None
            LOGGER.warning("Vulkan layered stereo pass unavailable; falling back: %s", self._vulkan_stereo_backend_error)
            return None, f"kernel_failed:{self._vulkan_stereo_backend_error}"

    def _reset_cuda_peak_if_needed(self) -> None:
        if not self.collect_memory_stats or not torch.cuda.is_available():
            return
        device = self._runtime_cuda_device()
        if device is None:
            return
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass

    def _collect_memory_stats(self, rgb_frame: torch.Tensor) -> dict[str, float]:
        if not self.collect_memory_stats or not torch.cuda.is_available():
            return {}
        device = self._runtime_cuda_device(rgb_frame)
        if device is None:
            return {}
        try:
            return {
                "cuda_memory_allocated_mb": torch.cuda.memory_allocated(device) / (1024.0 * 1024.0),
                "cuda_memory_reserved_mb": torch.cuda.memory_reserved(device) / (1024.0 * 1024.0),
                "cuda_peak_memory_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0),
                "cuda_peak_memory_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0),
            }
        except Exception:
            return {}

    def _runtime_cuda_device(self, rgb_frame: torch.Tensor | None = None) -> torch.device | None:
        if isinstance(rgb_frame, torch.Tensor) and rgb_frame.is_cuda:
            return rgb_frame.device
        try:
            device = torch.device(self.config.device)
        except Exception:
            return None
        return device if device.type == "cuda" else None


StereoLabRuntime = StereoRuntime
StereoLabRuntimeResult = StereoRuntimeResult
StereoLabOpenXRRuntimeResult = OpenXRRuntimeResult
StereoLabDepthRuntime = DepthRuntime
StereoLabDepthRuntimeResult = DepthRuntimeResult


def _provider_report(depth_provider: Any) -> dict[str, Any]:
    info = getattr(depth_provider, "info", None)
    if info is None:
        report: dict[str, Any] = {}
    else:
        to_report = getattr(info, "to_report", None)
        if callable(to_report):
            report = to_report()
        elif isinstance(info, dict):
            report = dict(info)
        else:
            report = {"info": str(info)}
    attempts = getattr(depth_provider, "attempts", None)
    if attempts:
        report["attempts"] = [dict(item) for item in attempts]
    return report


def _validate_runtime_rgb_frame(rgb_frame: Any) -> torch.Tensor:
    """Validate the capture/runtime boundary without doing capture adaptation."""
    if not isinstance(rgb_frame, torch.Tensor):
        raise TypeError("rgb_frame must be a torch.Tensor prepared by the capture layer")
    if rgb_frame.ndim not in (3, 4):
        raise ValueError(f"rgb_frame must be CHW or BCHW, got shape {tuple(rgb_frame.shape)}")
    channel_dim = 0 if rgb_frame.ndim == 3 else 1
    if rgb_frame.shape[channel_dim] != 3:
        raise ValueError(f"rgb_frame must be RGB with 3 channels in CHW/BCHW layout, got shape {tuple(rgb_frame.shape)}")
    if not rgb_frame.is_floating_point():
        raise TypeError(f"rgb_frame must be float 0..1; got dtype {rgb_frame.dtype}")
    return rgb_frame


def _add_runtime_output_size_debug_info(
    debug: dict[str, Any],
    eye_frame: torch.Tensor,
    display_frame: torch.Tensor,
) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    output_eye_size = _runtime_frame_size(eye_frame)
    output_display_size = _runtime_frame_size(display_frame)
    debug["runtime_output_eye_size"] = runtime_output_size_text(output_eye_size)
    debug["runtime_output_display_size"] = runtime_output_size_text(output_display_size)
    return output_eye_size, output_display_size


def _optional_debug_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _openxr_shader_uniforms(
    config: OpenXRRenderConfig,
    *,
    render_size: tuple[int, int] | None,
    max_disparity_px: float | None,
    depth_response: str | None,
) -> dict[str, Any]:
    return {
        "max_disparity_px": 0.0 if max_disparity_px is None else float(max_disparity_px),
        "parallax_preset": str(config.parallax_preset),
        "depth_response": str(depth_response or "unknown"),
        "depth_strength": float(config.depth_strength),
        "convergence": float(config.convergence),
        "foreground_shift_scale": float(getattr(config, "foreground_shift_scale", 1.0)),
        "midground_shift_scale": float(getattr(config, "midground_shift_scale", 1.0)),
        "background_shift_scale": float(getattr(config, "background_shift_scale", 1.0)),
        "render_size": None if render_size is None else (int(render_size[0]), int(render_size[1])),
        "screen_roll": float(config.screen_roll),
    }


def _add_openxr_config_debug_info(debug: dict[str, Any], config: OpenXRRenderConfig, eye_frame: torch.Tensor) -> dict[str, Any]:
    render_size = _runtime_frame_size(eye_frame)
    max_disparity_px = None
    depth_response = None
    if render_size is not None:
        budget = resolve_parallax_budget(
            render_width=render_size[0],
            render_height=render_size[1],
            preset=config.parallax_preset,
            convergence=config.convergence,
            max_disparity_px=config.max_disparity_px,
        )
        debug.update(parallax_debug_info(budget))
        max_disparity_px = float(budget.max_disparity_px)
        depth_response = str(budget.depth_response_name)
    elif config.max_disparity_px is not None:
        max_disparity_px = float(config.max_disparity_px)
        debug["resolved_max_disparity_px"] = max_disparity_px
        debug["parallax_budget_preset"] = str(config.parallax_preset)
    debug["openxr_convergence"] = float(config.convergence)
    if max_disparity_px is not None:
        debug["openxr_max_disparity_px"] = float(max_disparity_px)
    debug["openxr_parallax_preset"] = str(config.parallax_preset)
    debug["openxr_foreground_shift_scale"] = float(getattr(config, "foreground_shift_scale", 1.0))
    debug["openxr_midground_shift_scale"] = float(getattr(config, "midground_shift_scale", 1.0))
    debug["openxr_background_shift_scale"] = float(getattr(config, "background_shift_scale", 1.0))
    debug["vulkan_projection_min_lod"] = float(getattr(config, "vulkan_projection_min_lod", 0.0))
    debug["vulkan_projection_max_lod"] = float(getattr(config, "vulkan_projection_max_lod", 0.35))
    debug["vulkan_projection_mip_lod_bias"] = float(getattr(config, "vulkan_projection_mip_lod_bias", -0.35))
    debug["vulkan_projection_rcas_sharpness"] = float(getattr(config, "vulkan_projection_rcas_sharpness", 0.5))
    uniforms = _openxr_shader_uniforms(
        config,
        render_size=render_size,
        max_disparity_px=max_disparity_px,
        depth_response=depth_response or debug.get("depth_response"),
    )
    debug["openxr_shader_uniforms"] = uniforms
    return uniforms


def _add_preprocess_debug_info(debug: dict[str, Any], rgb_frame: torch.Tensor) -> None:
    mapping = {
        "_d2s_preprocess_backend": "preprocess_backend",
        "_d2s_preprocess_input_kind": "preprocess_input_kind",
        "_d2s_preprocess_device_origin": "preprocess_device_origin",
        "_d2s_preprocess_device_output": "preprocess_device_output",
        "_d2s_preprocess_device_transfer": "preprocess_device_transfer",
    }
    for attr, key in mapping.items():
        value = getattr(rgb_frame, attr, None)
        if value is not None:
            debug[key] = str(value)



def _runtime_frame_size(frame) -> tuple[int, int] | None:
    shape = tuple(getattr(frame, "shape", ()))
    if len(shape) == 4:
        shape = shape[1:]
    if len(shape) == 3 and shape[0] in (1, 3, 4):
        return int(shape[2]), int(shape[1])
    if len(shape) == 3 and shape[-1] in (1, 3, 4):
        return int(shape[1]), int(shape[0])
    if len(shape) >= 2:
        return int(shape[-1]), int(shape[-2])
    return None


def _runtime_size_text(size: tuple[int, int] | None) -> str:
    if size is None:
        return "unknown"
    return f"{int(size[0])}x{int(size[1])}"


def _runtime_eye_dtype(left_eye, right_eye) -> str:
    left_dtype = str(getattr(left_eye, "dtype", "unknown")).replace("torch.", "")
    right_dtype = str(getattr(right_eye, "dtype", "unknown")).replace("torch.", "")
    if left_dtype == right_dtype:
        return left_dtype
    return f"left={left_dtype},right={right_dtype}"


def _runtime_eye_size(eye) -> str:
    shape = tuple(getattr(eye, "shape", ()))
    if len(shape) == 4:
        shape = shape[1:]
    if len(shape) == 3 and shape[0] in (3, 4):
        return f"{int(shape[2])}x{int(shape[1])}"
    if len(shape) == 3 and shape[-1] >= 3:
        return f"{int(shape[1])}x{int(shape[0])}"
    return "unknown"
def _runtime_output_uint8_enabled() -> bool:
    return _env_flag("D2S_RUNTIME_OUTPUT_UINT8", "0")

def _half_res_synth_enabled() -> bool:
    """Viewer fast path: warp at half eye resolution, upscale the SBS.

    Default-on for the Darwin Local Viewer (runtime_entry setdefaults "1");
    synthesis is the dominant stage and scales with pixel count.
    """
    return _env_flag("D2S_HALF_RES_SYNTH", "0")


def _needs_half_res_downscale(output_rgb, *, skip_sbs_output: bool) -> bool:
    """True when the runtime must shrink RGB before synthesis.

    False when preprocess already delivered synth-scale frames
    (D2S_PREPROCESS_AT_SYNTH_SCALE marks them _d2s_synth_scale).
    """
    return (
        _half_res_synth_enabled()
        and not skip_sbs_output
        and output_rgb.ndim == 3
        and not getattr(output_rgb, "_d2s_synth_scale", False)
    )


def _viewer_host_frame_enabled() -> bool:
    """Ship a ready-to-upload HWC RGBA8 numpy frame beside the SBS tensor."""
    return _env_flag("D2S_VIEWER_HOST_FRAME", "0")


def _viewer_color_adjustments_neutral(config: Any) -> bool:
    """True when no color adjustment would be skipped by sampling raw BGRA.

    The zero-copy warp path samples the captured frame directly; any
    non-neutral brightness/contrast/saturation/gamma/temperature/tint must
    fall back to the preprocessed-tensor path.
    """
    if config is None:
        return True
    for attr in (
        "color_brightness",
        "color_contrast",
        "color_saturation",
        "color_gamma",
    ):
        try:
            if abs(float(getattr(config, attr, 1.0)) - 1.0) > 1e-6:
                return False
        except (TypeError, ValueError):
            continue
    for attr in ("color_temperature", "color_tint"):
        try:
            if abs(float(getattr(config, attr, 0.0))) > 1e-6:
                return False
        except (TypeError, ValueError):
            continue
    return True


def _pack_sbs_host_frame(
    sbs: torch.Tensor,
    timings: dict[str, float] | None = None,
    host_out=None,
) -> Any | None:
    """Pack an accelerator SBS tensor to HWC RGBA8 host memory.

    One flat CHW pull from the device, then numpy HWC/alpha transform:
    avoids a second device materialization (permute+alpha live on device)
    whose lazy MPS completion lands as an extra full-queue stall.
    """

    def _mark(name: str, _t: list) -> None:
        if timings is not None:
            timings[name] = (time.perf_counter() - _t[0]) * 1000.0
            _t[0] = time.perf_counter()

    try:
        import numpy as np

        _t = [time.perf_counter()]
        if not (hasattr(sbs, "device") and getattr(sbs.device, "type", "cpu") != "cpu"):
            return None
        image = sbs.detach()
        if image.ndim == 4:
            image = image[0]
        elif image.ndim != 3:
            return None
        # Normalize to CHW (channels-first).
        if image.shape[-1] in (1, 3, 4) and image.shape[0] > 8:
            image = image.permute(2, 0, 1)  # HWC source -> CHW
        channels = int(image.shape[0])
        if channels not in (1, 3, 4):
            return None
        if image.is_floating_point():
            # NaN/Inf must not reach the u8 host frame (silent 0 = black).
            image = torch.nan_to_num(image, nan=1.0, posinf=1.0, neginf=0.0)
            image = (image.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
        elif image.dtype != torch.uint8:
            return None
        height, width = int(image.shape[-2]), int(image.shape[-1])
        if (
            sys.platform == "darwin"
            and os.environ.get("D2S_DEVICE_PACK", "1") in ("1", "true", "on")
        ):
            # Do CHW->HWC (+alpha) on the GPU: the .cpu() pull then lands as
            # final HWC RGBA8, removing the second full-frame host copy
            # (moveaxis+ascontiguousarray). Safe since this runs in the
            # packer thread where the MPS-stream wait is already paid.
            import torch as _torch

            chw = image
            if channels == 1:
                chw = chw.expand(3, -1, -1)
            if channels in (1, 3):
                alpha = _torch.empty(
                    (1, height, width), dtype=_torch.uint8, device=chw.device
                ).fill_(255)
                chw = _torch.cat((chw, alpha), dim=0)
            dev_out = chw.permute(1, 2, 0).contiguous()
            if host_out is None:
                # Prefer the IOSurface stage ring when enabled: the DMA
                # lands directly in shared surface memory.
                try:
                    from utils.iosurface_stage import acquire_stage

                    got = acquire_stage(width, height)
                    if got is not None:
                        host_out = got[1].writable_view()
                except Exception:
                    host_out = None
            if host_out is not None:
                # Direct D2H into caller-provided memory (IOSurface stage):
                # no intermediate CPU tensor allocation at all.
                try:
                    import torch as _t2

                    dst = _t2.frombuffer(host_out, dtype=_torch.uint8)
                    dst.copy_(dev_out.reshape(-1))
                    _mark("copy", _t)
                    return (
                        np.frombuffer(host_out, dtype=np.uint8).reshape(
                            height, width, 4
                        ),
                        int(width),
                        int(height),
                    )
                except Exception:
                    pass  # fall through to legacy .cpu() path
            host = dev_out.cpu().numpy()
            _mark("copy", _t)
            return host, int(width), int(height)
        if not image.is_contiguous():
            image = image.contiguous()
        host = image.cpu().numpy()  # CHW uint8, single device pull
        _mark("copy", _t)
        if channels == 1:
            out = np.empty((height, width, 4), dtype=np.uint8)
            out[..., :3] = host[0][..., None]
            out[..., 3] = 255
        elif channels == 3:
            out = np.empty((height, width, 4), dtype=np.uint8)
            out[..., :3] = np.moveaxis(host, 0, -1)
            out[..., 3] = 255
        else:
            out = np.ascontiguousarray(np.moveaxis(host, 0, -1))
        _mark("numpy", _t)
        return out, int(width), int(height)
    except Exception:
        return None


def _rz_ship(t, raw=None):
    try:
        from utils.residency import mark as _rz_mark

        if raw is not None:
            _rz_mark("depth_out", raw)
        _rz_mark("warp_depth_ship", t)
    except Exception:
        pass
    return t


def fused_warp_active() -> bool:
    """Fused Metal warp-pack for the Vulkan viewer path (darwin only)."""
    if sys.platform != "darwin":
        return False
    try:
        from stereo_runtime._fused_warp_mps import fused_enabled

        return fused_enabled()
    except Exception:
        return False


def _quantize_depth_for_warp(depth: torch.Tensor) -> torch.Tensor:
    """u8 depth for the Metal warp viewer: 4x smaller host pull.

    Stays the DEFAULT transport (round-18 A/B: 37.5 -> 43.1 fps on the
    Metal leg); "0"/"false"/"off" ships the original fp32 tensor instead.
    Depth-edge flicker is addressed upstream by the v2.5-parity antialias
    pass (`_warp_depth_postprocess`), not by dropping u8. R8Unorm sampling
    reproduces the same [0,1] values.
    """
    if sys.platform != "darwin" or os.environ.get(
        "D2S_WARP_DEPTH_U8", "1"
    ) in {"0", "false", "off"}:
        return depth
    try:
        # Clone: detach() shares storage and in-place ops would corrupt
        # the caller's depth tensor.
        q = depth.detach().clone()
        if q.is_floating_point():
            # NaN/Inf (fp16 dither) must not survive the u8 transport:
            # torch casts NaN to 0 silently, punching a near-plane hole.
            q = torch.nan_to_num(q, nan=1.0, posinf=1.0, neginf=0.0)
            q = q.clamp_(0.0, 1.0).mul_(255.0).round_()
        return q.to(torch.uint8)
    except Exception:
        return depth


def _needs_depth_postprocess(stereo_config: Any) -> bool:
    """True only when Depth Pop or Anti-aliasing is actually enabled.

    Mode-isolation guard: cinema/game/image presets run with both knobs at
    their preset values, and every non-warp consumer must stay on the
    exact pre-existing code path (zero extra passes) unless the user
    explicitly raised a knob.
    """
    pop = float(getattr(stereo_config, "depth_pop", 0.0) or 0.0)
    aa = float(getattr(stereo_config, "depth_antialias_strength", 0.0) or 0.0)
    return abs(pop) >= 1e-6 or aa > 0.0


def _warp_depth_postprocess(
    depth: torch.Tensor,
    stereo_config: Any,
    guide_rgb: torch.Tensor | None = None,
) -> torch.Tensor:
    """v2.5-parity depth prep for the deferred warp (traditional) leg.

    The torch and Vulkan synthesis paths run `postprocess_depth` (Depth
    Pop + Anti-aliasing) before warping; the deferred Metal-shader-warp
    path -- the v2.5 traditional method -- shipped the raw model output
    and skipped both, so depth edges kept their per-frame aliasing and
    flicker. Apply the same pass here. Zero extra passes when both knobs
    are off (the common realtime default).

    With a guide frame, the antialiasing becomes RGB-guided edge-aware
    smoothing (arXiv 1911.07036 "3D Photography" stage 1): depth averages
    only across color-similar neighbors, denoising flat regions WITHOUT
    widening object-edge transition bands. D2S_GUIDED_DEPTH_SMOOTH=0
    restores the plain gaussian.
    """
    pop = float(getattr(stereo_config, "depth_pop", 0.0) or 0.0)
    aa = float(getattr(stereo_config, "depth_antialias_strength", 0.0) or 0.0)
    if not _needs_depth_postprocess(stereo_config):
        return depth
    guide = None
    if aa > 0.0 and guide_rgb is not None and os.environ.get(
        "D2S_GUIDED_DEPTH_SMOOTH", "1"
    ) not in {"0", "false", "off"}:
        guide = _match_guide_to_depth(guide_rgb, depth)
    return _postprocess_depth_fast(depth, pop, aa, guide)


def _postprocess_depth_fast(
    depth: torch.Tensor,
    depth_pop: float,
    antialias_strength: float,
    guide: torch.Tensor | None,
):
    """Realtime form of ``postprocess_depth`` for native-res frames.

    The guided AA is dispatch-cheap at synth-scale but measurably heavy at
    1080p+ (rt_loop 20ms -> ~60ms measured). Compute the gaussian + guided
    blend at half resolution and bilinear back up: ~4x cheaper, visually
    near-identical (the edge map is low-frequency). Small planes keep the
    exact full-res math. Kill switch: D2S_GUIDED_AA_HALFRES=0.
    """
    if guide is None:
        return postprocess_depth(
            depth, depth_pop=depth_pop, antialias_strength=antialias_strength
        )
    if os.environ.get("D2S_GUIDED_AA_HALFRES", "1") in {"0", "false", "off"}:
        return postprocess_depth(
            depth, depth_pop=depth_pop, antialias_strength=antialias_strength
        )
    h, w = int(depth.shape[-2]), int(depth.shape[-1])
    if min(h, w) < 256 or antialias_strength <= 0.0:
        return postprocess_depth(
            depth,
            depth_pop=depth_pop,
            antialias_strength=antialias_strength,
            guide=guide,
        )
    out = depth.clamp(0.0, 1.0)
    if abs(float(depth_pop)) >= 1e-6:
        out = apply_depth_pop(out, depth_pop)
    half = torch.nn.functional.interpolate(
        out, scale_factor=0.5, mode="bilinear", align_corners=False
    ).clamp(0.0, 1.0)
    g_lo = torch.nn.functional.interpolate(
        guide, size=half.shape[-2:], mode="bilinear", align_corners=False
    )
    lo = anti_alias_depth_guided(half, g_lo, antialias_strength)
    return torch.nn.functional.interpolate(
        lo, size=(h, w), mode="bilinear", align_corners=False
    )


def _match_guide_to_depth(guide_rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    """Downscale the render-res RGB guide to the depth plane resolution."""
    g = ensure_bchw(guide_rgb, name="guide_rgb").float()
    th, tw = int(depth.shape[-2]), int(depth.shape[-1])
    if int(g.shape[-2]) == th and int(g.shape[-1]) == tw:
        return g
    return torch.nn.functional.interpolate(
        g, size=(th, tw), mode="bilinear", align_corners=False
    )


def _upscale_sbs_to_output(sbs: torch.Tensor, output_rgb: torch.Tensor) -> torch.Tensor:
    """Bilinear-upscale a half-res SBS back to the display frame size."""
    target_h, target_w = int(output_rgb.shape[-2]), int(output_rgb.shape[-1])
    if tuple(sbs.shape[-2:]) == (target_h, target_w):
        return sbs
    x = sbs
    squeezed = None
    if x.ndim == 3:  # CHW
        squeezed = 0
        x = x.unsqueeze(0)
    if x.ndim != 4:
        return sbs
    upscaled = torch.nn.functional.interpolate(
        x,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )
    if squeezed is not None:
        upscaled = upscaled.squeeze(squeezed)
    if isinstance(upscaled, torch.Tensor) and upscaled.is_contiguous():
        return upscaled
    return upscaled.contiguous()





def _openxr_runtime_output_uint8_enabled() -> bool:
    return _env_flag("D2S_OPENXR_RUNTIME_OUTPUT_UINT8", os.environ.get("D2S_RUNTIME_OUTPUT_UINT8", "1"))


def _openxr_prewarp_eyes_enabled() -> bool:
    # OpenXR uses the Presenter-owned Vulkan image path by default. The
    # environment variable remains an explicit escape hatch for diagnostics.
    return _env_flag("D2S_OPENXR_PREWARP_EYES", "1")


def _intel_vulkan_network_path_enabled() -> bool:
    """Return whether Intel native network output may defer stereo to Vulkan."""
    return _env_flag("D2S_INTEL_VULKAN_SBS", "0")


def _is_triton_stereo_compute_backend(value: object) -> bool:
    return str(value or "").strip().lower() in {"triton", "cuda_triton", "amd_triton", "rocm_triton"}


def _openxr_rgb_depth_temporal_alpha() -> float:
    raw = os.environ.get("D2S_OPENXR_RGB_DEPTH_TEMPORAL_ALPHA", "0.9")
    try:
        return max(0.0, min(0.98, float(raw)))
    except Exception:
        return 0.9


def _openxr_rgb_depth_gamma() -> float:
    raw = os.environ.get("D2S_OPENXR_RGB_DEPTH_GAMMA", "1.2")
    try:
        return max(0.1, min(4.0, float(raw)))
    except Exception:
        return 1.2


def _openxr_rgb_depth_percentile() -> float:
    raw = os.environ.get("D2S_OPENXR_RGB_DEPTH_PERCENTILE", "0")
    try:
        return max(0.0, min(20.0, float(raw)))
    except Exception:
        return 0.0


def _openxr_rgb_depth_percentile_normalize(depth: torch.Tensor, *, percentile: float) -> torch.Tensor:
    depth = depth.detach().contiguous().float().clamp(0.0, 1.0)
    if percentile <= 0.0:
        return depth
    flat = depth.flatten(start_dim=2)
    lo_q = max(0.0, min(1.0, float(percentile) / 100.0))
    hi_q = 1.0 - lo_q
    count = flat.shape[-1]
    if count <= 1:
        return depth
    lo_idx = min(count - 1, max(0, int(round(lo_q * (count - 1)))))
    hi_idx = min(count - 1, max(0, int(round(hi_q * (count - 1)))))
    sorted_vals = torch.sort(flat, dim=-1).values
    lo = sorted_vals[..., lo_idx].view(depth.shape[0], 1, 1, 1)
    hi = sorted_vals[..., hi_idx].view(depth.shape[0], 1, 1, 1)
    return ((depth - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)


def _env_flag(name: str, default: object = "0") -> bool:
    return str(os.environ.get(name, default) or default).strip().lower() in {"1", "true", "yes", "on"}


def _fast_plus_fused_enabled() -> bool:
    return str(os.environ.get("D2S_FAST_PLUS_FUSED", "1") or "1").strip().lower() in {"1", "true", "yes", "on"}
