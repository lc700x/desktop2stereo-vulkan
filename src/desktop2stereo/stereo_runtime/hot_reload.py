from __future__ import annotations

import os
import time
from dataclasses import replace
from typing import Callable

from stereo_runtime import stereo_config_for_preset
from stereo_runtime.adapter import (
    _depth_export_size_from_settings,
    _normalize_depth_backend,
    _normalize_hole_fill_mode,
    _normalize_output_format,
    _normalize_runtime_mode,
    _normalize_stereo_quality,
    preset_for_runtime_mode,
)
from stereo_runtime.presets import normalize_preset
from stereo_runtime.settings_snapshot import RuntimeSettingsSnapshot
from utils.xr_headset_presets import resolve_xr_headset_preset

def read_yaml(path: str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def runtime_stereo_overrides(runtime) -> dict:
    config = runtime.config
    return {
        "backend": config.stereo_quality,
        "depth_strength": config.depth_strength,
        "convergence": config.convergence,
        "max_disparity_px": getattr(config, "max_disparity_px", None),
        "parallax_preset": getattr(config, "parallax_preset", "standard"),
        "foreground_shift_scale": getattr(config, "foreground_shift_scale", 1.0),
        "midground_shift_scale": getattr(config, "midground_shift_scale", 1.0),
        "background_shift_scale": getattr(config, "background_shift_scale", 1.0),
        "dynamic_convergence_enabled": getattr(config, "dynamic_convergence_enabled", False),
        "dynamic_convergence_strength": getattr(config, "dynamic_convergence_strength", 0.0),
        "dynamic_convergence_target": getattr(config, "dynamic_convergence_target", 0.5),
        "dynamic_convergence_alpha": getattr(config, "dynamic_convergence_alpha", 0.85),
        "temporal": config.temporal,
        "temporal_strength": config.temporal_strength,
        "auto_reset_temporal": config.auto_reset_temporal,
        "scene_reset_threshold": config.scene_reset_threshold,
        "depth_pop": config.depth_pop,
        "depth_antialias_strength": config.depth_antialias_strength,
        "edge_threshold": config.edge_threshold,
        "edge_dilation": config.edge_dilation,
        "mask_feather_radius": config.mask_feather_radius,
        "hole_fill": "none" if config.hole_fill_mode == "none" else getattr(config, "hole_fill", "edge_aware"),
        "hole_fill_mode": config.hole_fill_mode,
        "hole_fill_radius": config.hole_fill_radius,
        "hole_fill_strength": config.hole_fill_strength,
        "screen_edge_mask_suppression": config.screen_edge_mask_suppression,
        "cross_eyed": config.cross_eyed,
        "anaglyph_method": config.anaglyph_method,
        "debug_output": getattr(config, "debug_output", False),
        "fused": config.fused,
        "output_quality_enabled": getattr(config, "output_quality_enabled", False),
        "output_headset_tier_k": getattr(config, "output_headset_tier_k", 4),
        "output_min_lod": getattr(config, "output_min_lod", 0.0),
        "output_max_lod": getattr(config, "output_max_lod", 0.35),
        "output_mip_lod_bias": getattr(config, "output_mip_lod_bias", -0.35),
        "output_rcas_sharpness": getattr(config, "output_rcas_sharpness", 0.5),
    }


def to_bool_hot_reload(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def clamp_depth_pop_hot_reload(value) -> float:
    return max(-0.9, min(5.0, float(value)))


def optional_float_hot_reload(value):
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return float(value)


def _is_fast_quality(settings_dict: dict, config) -> bool:
    raw = settings_dict.get("Stereo Quality", settings_dict.get("Synthetic View", getattr(config, "stereo_quality", "")))
    key = str(raw).strip().lower().replace("-", "_").replace("+", "_plus").replace(" ", "_")
    return key == "fast"


def _depth_backend_hot_reload(settings_dict: dict, config):
    backend_keys = {"Depth Backend", "MIGraphX", "TensorRT", "ONNX"}
    if not backend_keys.intersection(settings_dict):
        return None
    import torch

    # Mirror adapter.runtime_config_from_d2s_settings: TensorRT is NVIDIA-only,
    # so on AMD ROCm the stale "TensorRT" flag must not select tensorrt_native
    # (that would diverge from the runtime's migraphx_rocm and request a
    # pipeline rebuild on every hot reload).
    is_rocm = bool(getattr(torch.version, "hip", None))
    if to_bool_hot_reload(settings_dict.get("MIGraphX", False)):
        return "migraphx_rocm"
    if to_bool_hot_reload(settings_dict.get("TensorRT", False)) and not is_rocm:
        return "tensorrt_native"
    if to_bool_hot_reload(settings_dict.get("ONNX", False)):
        return "onnx_cuda"
    if settings_dict.get("Depth Backend"):
        return _normalize_depth_backend(settings_dict["Depth Backend"])
    return "migraphx_rocm" if is_rocm else "pytorch_cuda"


def _add_runtime_quality_mode_if_changed(values: dict, settings_dict: dict, config) -> None:
    mode_keys = ("Runtime Quality Mode", "Stereo Runtime Mode")
    raw = next((settings_dict[key] for key in mode_keys if key in settings_dict), None)
    if raw is None:
        return
    mode = _normalize_runtime_mode(raw)
    if mode != getattr(config, "mode", None):
        values["runtime_quality_mode"] = mode
        values["mode"] = mode


def _add_rebuild_fields_if_changed(values: dict, settings_dict: dict, config) -> None:
    if "Depth Model" in settings_dict or "model_id" in settings_dict:
        model_id = str(settings_dict.get("Depth Model") or settings_dict.get("model_id"))
        if model_id != getattr(config, "model_id", None):
            values["model_id"] = model_id
    depth_backend = _depth_backend_hot_reload(settings_dict, config)
    if depth_backend is not None and depth_backend != getattr(config, "depth_backend", None):
        values["depth_backend"] = depth_backend
    if "Depth Profile Sync" in settings_dict or "Profile Sync" in settings_dict:
        profile_sync = to_bool_hot_reload(settings_dict.get("Depth Profile Sync", settings_dict.get("Profile Sync")))
        if profile_sync != getattr(config, "profile_sync", None):
            values["profile_sync"] = profile_sync
    if "Depth CUDA Graph" in settings_dict or "CUDA Graph" in settings_dict:
        use_cuda_graph = to_bool_hot_reload(settings_dict.get("Depth CUDA Graph", settings_dict.get("CUDA Graph")))
        if use_cuda_graph != getattr(config, "use_cuda_graph", None):
            values["use_cuda_graph"] = use_cuda_graph
    size_keys = {"Depth Resolution", "depth_resolution", "Export Height", "export_height", "Export Width", "export_width"}
    if size_keys.intersection(settings_dict):
        export_height, export_width = _depth_export_size_from_settings(
            settings_dict,
            default_height=getattr(config, "export_height", 294),
            default_width=getattr(config, "export_width", 518),
        )
        if export_height != getattr(config, "export_height", None):
            values["export_height"] = export_height
        if export_width != getattr(config, "export_width", None):
            values["export_width"] = export_width


def hot_reload_value_snapshot(settings_dict: dict, config) -> dict:
    has_hole_fill_mode = "Hole Fill Mode" in settings_dict
    hole_fill_mode, hole_fill_radius, hole_fill_strength = _normalize_hole_fill_mode(
        settings_dict.get("Hole Fill Mode", getattr(config, "hole_fill_mode", "balanced"))
    )
    if not has_hole_fill_mode:
        hole_fill_radius = int(settings_dict.get("Hole Fill Radius", hole_fill_radius))
        hole_fill_strength = float(settings_dict.get("Hole Fill Strength", hole_fill_strength))
    debug_output_enabled = to_bool_hot_reload(
        settings_dict.get("Debug Stereo Output", getattr(config, "debug_output", False))
    )
    stereo_preset = normalize_preset(
        settings_dict.get(
            "Stereo Preset",
            settings_dict.get(
                "Stereo Mode Preset",
                config.stereo_preset or preset_for_runtime_mode(config.mode),
            ),
        )
    )
    temporal_strength = float(settings_dict.get("Temporal Strength", config.temporal_strength))
    scene_reset_threshold = float(settings_dict.get("Scene Reset Threshold", config.scene_reset_threshold))
    temporal_enabled = (
        to_bool_hot_reload(settings_dict["Temporal"])
        if "Temporal" in settings_dict
        else temporal_strength > 0.0
    )
    auto_reset_temporal = (
        to_bool_hot_reload(settings_dict.get("Auto Scene Reset", settings_dict.get("Auto Reset Temporal")))
        if "Auto Scene Reset" in settings_dict or "Auto Reset Temporal" in settings_dict
        else scene_reset_threshold > 0.0
    )
    values = {
        "depth_strength": float(settings_dict.get("Depth Strength", config.depth_strength)),
        "convergence": float(settings_dict.get("Convergence", config.convergence)),
        "stereo_quality": _normalize_stereo_quality(
            settings_dict.get("Stereo Quality", settings_dict.get("Synthetic View", config.stereo_quality))
        ),
        "stereo_preset": stereo_preset,
        "output_format": _normalize_output_format(settings_dict.get("Display Mode", config.output_format)),
        "max_disparity_px": optional_float_hot_reload(
            settings_dict.get(
                "Max Disparity Px",
                settings_dict.get("Max Disparity PX", getattr(config, "max_disparity_px", None)),
            )
        ),
        "parallax_preset": str(
            settings_dict.get(
                "Parallax Budget Preset",
                settings_dict.get("Parallax Preset", getattr(config, "parallax_preset", "standard")),
            )
        ),
        "foreground_shift_scale": float(settings_dict.get("Foreground Pop", getattr(config, "foreground_shift_scale", 1.0))),
        "midground_shift_scale": float(settings_dict.get("Midground Pop", getattr(config, "midground_shift_scale", 1.0))),
        "background_shift_scale": float(settings_dict.get("Background Pop", getattr(config, "background_shift_scale", 1.0))),
        "dynamic_convergence_enabled": to_bool_hot_reload(settings_dict.get("Dynamic Convergence", settings_dict.get("Dynamic Convergence Enabled", getattr(config, "dynamic_convergence_enabled", False)))),
        "dynamic_convergence_strength": float(settings_dict.get("Dynamic Convergence Strength", getattr(config, "dynamic_convergence_strength", 0.0))),
        "dynamic_convergence_target": float(settings_dict.get("Dynamic Convergence Target", getattr(config, "dynamic_convergence_target", 0.5))),
        "dynamic_convergence_alpha": float(settings_dict.get("Dynamic Convergence Alpha", getattr(config, "dynamic_convergence_alpha", 0.85))),
        "temporal": temporal_enabled,
        "temporal_strength": temporal_strength,
        "auto_reset_temporal": auto_reset_temporal,
        "scene_reset_threshold": scene_reset_threshold,
        "depth_pop": clamp_depth_pop_hot_reload(settings_dict.get("Depth Pop", config.depth_pop)),
        "depth_antialias_strength": float(
            settings_dict.get(
                "Depth Antialias Strength",
                settings_dict.get("Anti-aliasing", config.depth_antialias_strength),
            )
        ),
        "edge_dilation": int(settings_dict.get("Edge Dilation", config.edge_dilation)),
        "mask_feather_radius": int(settings_dict.get("Mask Feather Radius", config.mask_feather_radius)),
        "hole_fill_mode": hole_fill_mode,
        "hole_fill_radius": hole_fill_radius,
        "hole_fill_strength": hole_fill_strength,
        "screen_edge_mask_suppression": int(
            settings_dict.get("Screen Edge Mask Suppression", config.screen_edge_mask_suppression)
        ),
        "edge_threshold": float(settings_dict.get("Edge Threshold", config.edge_threshold)),
        "anaglyph_method": str(settings_dict.get("Anaglyph Method", config.anaglyph_method)),
        "color_brightness": float(settings_dict.get("Color Brightness", getattr(config, "color_brightness", 1.0))),
        "color_contrast": float(settings_dict.get("Color Contrast", getattr(config, "color_contrast", 1.0))),
        "color_saturation": float(settings_dict.get("Color Saturation", getattr(config, "color_saturation", 1.0))),
        "color_gamma": float(settings_dict.get("Color Gamma", getattr(config, "color_gamma", 1.0))),
        "color_temperature": float(settings_dict.get("Color Temperature", getattr(config, "color_temperature", 0.0))),
        "color_tint": float(settings_dict.get("Color Tint", getattr(config, "color_tint", 0.0))),
        "vulkan_projection_min_lod": max(0.0, min(16.0, float(settings_dict.get("Vulkan Projection Min LOD", 0.0)))),
        "vulkan_projection_max_lod": max(0.0, min(16.0, float(settings_dict.get("Vulkan Projection Max LOD", 0.35)))),
        "vulkan_projection_mip_lod_bias": max(-1.5, min(0.0, float(settings_dict.get("Vulkan Projection MIP LOD Bias", -0.35)))),
        "vulkan_projection_rcas_sharpness": max(0.0, min(1.0, float(settings_dict.get("Vulkan Projection RCAS Sharpness", 0.5)))),
        "output_headset_tier_k": int(
            resolve_xr_headset_preset(settings_dict.get("XR Headset Model")).resolution_tier_k
        ),
        "cross_eyed": to_bool_hot_reload(settings_dict.get("Cross Eyed", config.cross_eyed)),
        "presentation_flags": {
            "show_fps": to_bool_hot_reload(settings_dict.get("Show FPS", False)),
            "display_fit_mode": str(
                settings_dict.get("Display Fit Mode", "contain") or "contain"
            ).strip().casefold(),
        },
        "debug_output": debug_output_enabled,
        "debug_flags": {"debug_output": debug_output_enabled},
        "audio_delay": max(
            -10.0,
            min(10.0, float(settings_dict.get("Audio Delay", -0.1))),
        ),
    }
    _add_runtime_quality_mode_if_changed(values, settings_dict, config)
    _add_rebuild_fields_if_changed(values, settings_dict, config)
    if _is_fast_quality(settings_dict, config):
        values.update(
            {
                "temporal": False,
                "temporal_strength": 0.0,
                "auto_reset_temporal": False,
                "scene_reset_threshold": 0.0,
                "depth_pop": 0.0,
                "depth_antialias_strength": 0.0,
            }
        )
    return values


def hot_reload_runtime_settings_snapshot(
    settings_dict: dict,
    config,
    *,
    version: int,
    timestamp: float,
) -> RuntimeSettingsSnapshot:
    values = hot_reload_value_snapshot(settings_dict, config)
    snapshot_values = {
        key: value
        for key, value in values.items()
        if key in RuntimeSettingsSnapshot.__dataclass_fields__
    }
    return RuntimeSettingsSnapshot(
        version=version,
        timestamp=timestamp,
        source="settings_yaml_hot_reload",
        **snapshot_values,
    )


class StereoHotReloader:
    def __init__(
        self,
        *,
        settings_path: str,
        interval_s: float = 0.25,
        read_settings: Callable[[str], dict] = read_yaml,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.settings_path = settings_path
        self.interval_s = interval_s
        self.read_settings = read_settings
        self.clock = clock
        self.last_check = 0.0
        self.last_mtime = os.path.getmtime(settings_path) if os.path.exists(settings_path) else 0.0
        self.last_values = None

    def poll_settings_snapshot_if_needed(self, *, runtime, active_preset):
        now = self.clock()
        if now - self.last_check < self.interval_s:
            return None
        self.last_check = now
        try:
            mtime = os.path.getmtime(self.settings_path)
        except OSError:
            return None
        if mtime <= self.last_mtime and self.last_values is not None:
            return None
        try:
            settings_dict = self.read_settings(self.settings_path)
            values = hot_reload_value_snapshot(settings_dict, runtime.config)
            snapshot = hot_reload_runtime_settings_snapshot(
                settings_dict,
                runtime.config,
                version=int(mtime * 1_000_000_000),
                timestamp=mtime,
            )
        except Exception as exc:
            print(f"[Main] Stereo hot reload skipped: {type(exc).__name__}: {exc}", flush=True)
            self.last_mtime = mtime
            return None
        if values == self.last_values:
            self.last_mtime = mtime
            return None

        snapshot_preset = values.get("stereo_preset")
        applied_preset = active_preset if snapshot_preset == "auto" else (snapshot_preset or active_preset)
        self.last_values = values
        self.last_mtime = mtime
        return snapshot, applied_preset, values

    def log_settings_snapshot(self, values: dict, *, on_mode_log: Callable[[str], None]) -> None:
        if os.environ.get('D2S_DEBUG', '0') in ('1', 'true', 'yes', 'on'):
            print(
                "[Main] Stereo hot reload:"
                f" depth_strength={values['depth_strength']:.3f}"
                f" convergence={values['convergence']:.3f}"
                f" parallax_preset={values['parallax_preset']}"
                f" stereo_preset={values['stereo_preset']}"
                f" stereo_quality={values['stereo_quality']}"
                f" temporal_strength={values['temporal_strength']:.3f}"
                f" scene_reset={values['scene_reset_threshold']:.3f}"
                f" depth_pop={values['depth_pop']:.3f}"
                f" fg_shift={values['foreground_shift_scale']:.3f}"
                f" mg_shift={values['midground_shift_scale']:.3f}"
                f" bg_shift={values['background_shift_scale']:.3f}"
                f" dynamic_convergence={int(values['dynamic_convergence_enabled'])}/{values['dynamic_convergence_strength']:.3f}"
                f" antialias={values['depth_antialias_strength']:.3f}"
                f" edge_dilation={values['edge_dilation']}"
                f" mask_feather={values['mask_feather_radius']}"
                f" hole_fill={values['hole_fill_mode']}({values['hole_fill_radius']}/{values['hole_fill_strength']:.2f})"
                f" screen_edge_mask={values['screen_edge_mask_suppression']}"
                f" edge_threshold={values['edge_threshold']:.3f}"
                f" anaglyph={values['anaglyph_method']}"
                f" cross_eyed={int(values['cross_eyed'])}"
                f" debug_output={int(values['debug_output'])}",
                flush=True,
            )
        on_mode_log("hot-reload")

    def apply_if_needed(
        self,
        *,
        runtime,
        active_preset,
        on_openxr_config_update: Callable[..., None],
        on_mode_log: Callable[[str], None],
    ) -> bool:
        polled = self.poll_settings_snapshot_if_needed(runtime=runtime, active_preset=active_preset)
        if polled is None:
            return False
        snapshot, applied_preset, values = polled
        if hasattr(runtime, "apply_settings_snapshot"):
            runtime.apply_settings_snapshot(snapshot, active_preset=applied_preset)
        else:
            config_values = {key: value for key, value in values.items() if hasattr(runtime.config, key)}
            if applied_preset is not None:
                config_values["stereo_preset"] = applied_preset
            runtime.config = replace(runtime.config, **config_values)
            current = runtime.stereo_config
            runtime.configure_stereo(
                stereo_config_for_preset(
                    applied_preset or runtime.config.stereo_preset or preset_for_runtime_mode(runtime.config.mode),
                    output_format=current.output_format,
                    overrides=runtime_stereo_overrides(runtime),
                ),
                reset_temporal=False,
            )
        on_openxr_config_update(snapshot=snapshot)
        self.log_settings_snapshot(values, on_mode_log=on_mode_log)
        return True
