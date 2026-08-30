from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from .model_artifacts import artifact_paths_for_model
from .model_registry import ModelRegistry

if TYPE_CHECKING:
    from .depth_provider import DepthProviderConfig
    from .synthesis import StereoConfig

RuntimeMode = Literal["auto", "movie", "game", "image", "debug"]
StereoQuality = Literal["fast", "fast_plus", "quality_4k", "hq_4k"]
DepthBackend = Literal["auto", "tensorrt_native", "onnx_cuda", "pytorch_cuda", "pytorch_rocm", "migraphx_rocm", "directml"]
OnnxDtypeMode = Literal["auto", "fp16", "fp32"]
DepthUpsampleMode = Literal["bilinear", "guided"]
OutputFormat = Literal["half_sbs", "full_sbs", "half_tab", "full_tab", "mono", "anaglyph", "interleaved", "leia", "depth_map"]
HoleFill = Literal["none", "fast", "edge_aware"]
StereoComputeBackendMode = Literal["auto", "triton", "vulkan", "opengl"]


@dataclass(frozen=True)
class StereoRuntimeConfig:
    """Host-facing runtime config.

    The host owns user selection and settings persistence. stereo_runtime owns
    model registry resolution, model_dir derivation, download, artifact paths,
    ONNX/TensorRT preparation, and depth/stereo inference.
    """

    model_id: str
    model_dir: str | Path | None = None
    cache_dir: str | Path = "./models"
    mode: RuntimeMode = "movie"
    stereo_preset: str | None = None
    stereo_quality: StereoQuality = "quality_4k"
    output_format: OutputFormat = "half_sbs"
    depth_backend: DepthBackend = "auto"
    device: str = "cuda"
    onnx_dtype: OnnxDtypeMode = "auto"
    export_height: int = 294
    export_width: int = 518
    build_trt_engine: bool = False
    force_rebuild_trt: bool = False
    build_migraphx_graph: bool = False
    force_rebuild_migraphx: bool = False
    trt_workspace_gb: int = 4
    use_cuda_graph: bool = False
    profile_sync: bool = False
    parallel_inference: bool = False
    parallel_inference_workers: int = 1
    # Optional program-controlled OpenXR visual regression output directory.
    openxr_visual_regression_dir: str | Path | None = None
    depth_upsample: DepthUpsampleMode = "bilinear"
    depth_upsample_edge_strength: float = 0.35
    depth_strength: float = 2.0
    convergence: float = 0.0
    max_disparity_px: float | None = None
    parallax_preset: str = "standard"
    foreground_shift_scale: float = 1.0
    midground_shift_scale: float = 1.0
    background_shift_scale: float = 1.0
    dynamic_convergence_enabled: bool = False
    dynamic_convergence_strength: float = 0.0
    dynamic_convergence_target: float = 0.5
    dynamic_convergence_alpha: float = 0.85
    layers: int = 2
    occlusion: bool = True
    symmetric: bool = True
    hole_fill: HoleFill = "edge_aware"
    temporal: bool = True
    temporal_strength: float = 0.75
    auto_reset_temporal: bool = False
    scene_reset_threshold: float = 0.22
    depth_pop: float = 0.0
    depth_antialias_strength: float = 0.0
    edge_threshold: float = 0.04
    edge_dilation: int = 2
    mask_feather_radius: int = 3
    hole_fill_mode: str = "balanced"
    hole_fill_radius: int = 3
    hole_fill_strength: float = 1.0
    screen_edge_mask_suppression: int = 0
    cross_eyed: bool = False
    anaglyph_method: str = "red_cyan"
    color_brightness: float = 1.0
    color_contrast: float = 1.0
    color_saturation: float = 1.0
    color_gamma: float = 1.0
    color_temperature: float = 0.0
    color_tint: float = 0.0
    debug_output: bool = False
    fused: bool = True
    stereo_compute_backend: StereoComputeBackendMode = "auto"
    output_quality_enabled: bool = False
    output_headset_tier_k: int = 4
    output_min_lod: float = 0.0
    output_max_lod: float = 0.35
    output_mip_lod_bias: float = -0.35
    output_rcas_sharpness: float = 0.5

    @property
    def resolved_model_id(self) -> str:
        return ModelRegistry.default().resolve_model_id(self.model_id)

    @property
    def model_path(self) -> Path:
        if self.model_dir:
            return Path(self.model_dir)
        return self._artifact_paths().model_dir

    @property
    def onnx_path(self) -> Path:
        paths = self._artifact_paths()
        if self.onnx_dtype == "fp32":
            return paths.onnx_fp32_path
        return paths.onnx_fp16_path

    @property
    def fp32_onnx_path(self) -> Path:
        return self._artifact_paths().onnx_fp32_path

    @property
    def trt_engine_path(self) -> Path:
        paths = self._artifact_paths()
        if self.onnx_dtype == "fp32":
            return paths.trt_path_for_dtype("fp32")
        return paths.trt_fp16_path

    @property
    def migraphx_graph_path(self) -> Path:
        return self._artifact_paths().migraphx_fp16_path

    def _artifact_paths(self):
        return artifact_paths_for_model(
            self.resolved_model_id,
            cache_dir=self.cache_dir,
            model_dir=self.model_dir,
            export_height=self.export_height,
            export_width=self.export_width,
        )

    def artifact_paths(self) -> dict[str, str]:
        return {
            "model_dir": str(self.model_path),
            "onnx_path": str(self.onnx_path),
            "fp32_onnx_path": str(self.fp32_onnx_path),
            "trt_engine_path": str(self.trt_engine_path),
            "migraphx_graph_path": str(self.migraphx_graph_path),
        }

    def frame_contract(self) -> dict[str, str]:
        return runtime_frame_contract(self)

    def to_report(self) -> dict[str, Any]:
        report = asdict(self)
        report["resolved_model_id"] = self.resolved_model_id
        report["model_dir"] = str(self.model_path)
        report.update(self.artifact_paths())
        report["frame_contract"] = self.frame_contract()
        return report


def runtime_frame_contract(config: StereoRuntimeConfig) -> dict[str, str]:
    """Return the host-facing RGB frame contract.

    The host/capture pipeline owns capture-side color preprocessing and passes
    an already-RGB image frame. stereo_runtime starts at depth-provider input
    preparation and does not own BGR/BGRA-to-RGB conversion.
    """

    return {
        "input": "rgb_frame",
        "host_responsibility": "capture current image frame, perform capture-side color preprocessing, and pass an RGB frame at source resolution",
        "stereo_runtime_responsibility": "prepare RGB frame for depth inference, run depth provider, and synthesize stereo output",
        "not_stereo_runtime_responsibility": "desktop capture, BGR/BGRA-to-RGB conversion, window/monitor source handling",
        "backend_detail": "TensorRT/ONNX/PyTorch/Triton packing remains internal to stereo_runtime",
        "quality_rule": "host must not downscale or alter depth inference resolution semantics",
    }


def runtime_config_from_d2s_settings(
    settings: dict[str, Any],
    *,
    cache_dir: str | Path = "./models",
    device: str = "cuda",
    depth_only: bool = True,
) -> StereoRuntimeConfig:
    """Build a runtime config from Desktop2Stereo-style settings.

    GUI/settings keep user-facing choices. Runtime owns model resolution,
    artifact paths, dtype probing, and backend/provider details.
    """

    model_name = settings.get("Depth Model") or settings.get("model_id")
    if not model_name:
        raise ValueError("D2S settings must include 'Depth Model' or 'model_id'")

    depth_backend: DepthBackend
    import torch

    # TensorRT is NVIDIA-only.  On AMD ROCm the stale "TensorRT" setting
    # (defaulted on by the GUI for CUDA machines) must not select the
    # tensorrt_native backend, otherwise the frame loop fails importing
    # tensorrt.  ROCm defaults to the AMD-native MIGraphX path instead.
    is_rocm = bool(getattr(torch.version, "hip", None))
    if settings.get("MIGraphX", False):
        depth_backend = "migraphx_rocm"
    elif settings.get("TensorRT", False) and not is_rocm:
        depth_backend = "tensorrt_native"
    elif settings.get("ONNX", False):
        depth_backend = "onnx_cuda"
    elif settings.get("Depth Backend"):
        depth_backend = _normalize_depth_backend(settings["Depth Backend"])
    elif is_rocm:
        depth_backend = "migraphx_rocm"
    else:
        depth_backend = "auto"

    onnx_dtype: OnnxDtypeMode = "fp16" if _to_bool(settings.get("FP16", True)) else "fp32"
    preset_value = settings.get("Stereo Preset", settings.get("Stereo Mode Preset"))
    mode_source = settings.get("Stereo Runtime Mode", settings.get("Run Mode"))
    if mode_source is None:
        mode_source = preset_value if preset_value is not None else "movie"
    mode = _normalize_runtime_mode(mode_source)
    output_format = _normalize_output_format(settings.get("Display Mode", "half_sbs"))
    stereo_quality = _normalize_stereo_quality(settings.get("Stereo Quality", settings.get("Synthetic View", "fast" if depth_only else "quality_4k")))

    has_hole_fill_mode = "Hole Fill Mode" in settings
    hole_fill_mode, hole_fill_radius, hole_fill_strength = _normalize_hole_fill_mode(
        settings.get("Hole Fill Mode", settings.get("Hole Fill", "balanced"))
    )
    if not has_hole_fill_mode or hole_fill_mode == "balanced":
        hole_fill_radius = int(settings.get("Hole Fill Radius", hole_fill_radius))
        hole_fill_strength = float(settings.get("Hole Fill Strength", hole_fill_strength))
    export_height, export_width = _depth_export_size_from_settings(settings)
    parallel_workers = _parallel_inference_workers(settings)
    temporal_strength = max(0.0, float(settings.get("Temporal Strength", 0.75)))
    temporal_enabled = (
        hole_fill_mode != "none"
        and temporal_strength > 0.0
        and _to_bool(settings.get("Temporal", True))
    )
    if not temporal_enabled:
        temporal_strength = 0.0

    headset_tier = _output_headset_tier_from_settings(settings)
    return StereoRuntimeConfig(
        model_id=str(model_name),
        cache_dir=cache_dir,
        mode=mode,
        stereo_preset=str(preset_value) if preset_value is not None else None,
        stereo_quality=stereo_quality,
        output_format=output_format,
        depth_backend=depth_backend,
        device=device,
        onnx_dtype=onnx_dtype,
        build_trt_engine=bool(settings.get("TensorRT", False)),
        force_rebuild_trt=bool(settings.get("Recompile TensorRT", False)),
        build_migraphx_graph=bool(settings.get("MIGraphX", False)),
        force_rebuild_migraphx=bool(settings.get("Recompile MIGraphX", False)),
        openxr_visual_regression_dir=(
            str(settings.get("OpenXR Visual Regression Directory")).strip()
            if str(settings.get("OpenXR Visual Regression Directory", "")).strip()
            else None
        ),
        export_height=export_height,
        export_width=export_width,
        depth_strength=float(settings.get("Depth Strength", 2.0)),
        convergence=float(settings.get("Convergence", 0.0)),
        max_disparity_px=_optional_float_setting(settings, "Max Disparity Px", "Max Disparity PX"),
        parallax_preset=str(settings.get("Parallax Budget Preset", settings.get("Parallax Preset", "standard"))),
        foreground_shift_scale=float(settings.get("Foreground Pop", 1.0)),
        midground_shift_scale=float(settings.get("Midground Pop", 1.0)),
        background_shift_scale=float(settings.get("Background Pop", 1.0)),
        dynamic_convergence_enabled=_to_bool(settings.get("Dynamic Convergence", settings.get("Dynamic Convergence Enabled", False))),
        dynamic_convergence_strength=float(settings.get("Dynamic Convergence Strength", 0.0)),
        dynamic_convergence_target=float(settings.get("Dynamic Convergence Target", 0.5)),
        dynamic_convergence_alpha=float(settings.get("Dynamic Convergence Alpha", 0.85)),
        temporal=temporal_enabled,
        temporal_strength=temporal_strength,
        auto_reset_temporal=_to_bool(settings.get("Auto Scene Reset", settings.get("Auto Reset Temporal", True))),
        scene_reset_threshold=float(settings.get("Scene Reset Threshold", 0.22)),
        depth_pop=float(settings.get("Depth Pop", 0.0)),
        depth_antialias_strength=float(settings.get("Depth Antialias Strength", settings.get("Anti-aliasing", 0.0))),
        edge_threshold=float(settings.get("Edge Threshold", 0.04)),
        edge_dilation=int(settings.get("Edge Dilation", 2)),
        mask_feather_radius=int(settings.get("Mask Feather Radius", 3)),
        hole_fill="none" if hole_fill_mode == "none" else "edge_aware",
        hole_fill_mode=hole_fill_mode,
        hole_fill_radius=hole_fill_radius,
        hole_fill_strength=hole_fill_strength,
        screen_edge_mask_suppression=int(settings.get("Screen Edge Mask Suppression", 0)),
        cross_eyed=_to_bool(settings.get("Cross Eyed", False)),
        anaglyph_method=str(settings.get("Anaglyph Method", "red_cyan")),
        color_brightness=float(settings.get("Color Brightness", 1.0)),
        color_contrast=float(settings.get("Color Contrast", 1.0)),
        color_saturation=float(settings.get("Color Saturation", 1.0)),
        color_gamma=float(settings.get("Color Gamma", 1.0)),
        color_temperature=float(settings.get("Color Temperature", 0.0)),
        color_tint=float(settings.get("Color Tint", 0.0)),
        debug_output=_to_bool(settings.get("Debug Stereo Output", False)),
        profile_sync=_to_bool(settings.get("Depth Profile Sync", settings.get("Profile Sync", False))),
        parallel_inference=parallel_workers > 1,
        parallel_inference_workers=parallel_workers,
        stereo_compute_backend=_normalize_stereo_compute_backend(
            settings.get("Stereo Compute Backend", "auto")
        ),
        # The headset output-quality upscale (EASU to the headset 4K tier) is
        # meaningless for network streaming and costs hundreds of ms per frame
        # through the torch fallback. A 1920x1200 input must stay 1920x1200 for
        # the streamers; the 16:9 transport canvas is applied downstream.
        output_quality_enabled=not _streamer_mode_output_quality_disabled(settings),
        output_headset_tier_k=headset_tier,
        output_min_lod=max(0.0, min(16.0, float(settings.get("Vulkan Projection Min LOD", 0.0)))),
        output_max_lod=max(0.0, min(16.0, float(settings.get("Vulkan Projection Max LOD", 0.35)))),
        output_mip_lod_bias=max(-1.5, min(0.0, float(settings.get("Vulkan Projection MIP LOD Bias", -0.35)))),
        output_rcas_sharpness=max(
            0.0,
            min(1.0, float(settings.get("Vulkan Projection RCAS Sharpness", 0.5))),
        ),
    )



def _optional_float_setting(settings: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in settings and settings[key] is not None:
            return float(settings[key])
    return None


def _output_headset_tier_from_settings(settings: dict[str, Any]) -> int:
    from utils.xr_headset_presets import resolve_xr_headset_preset

    preset = resolve_xr_headset_preset(settings.get("XR Headset Model"))
    return int(preset.resolution_tier_k)


def _streamer_mode_output_quality_disabled(settings: dict[str, Any]) -> bool:
    """Return whether the headset output-quality upscale should be skipped.

    Streamers publish to browsers/network players, not a headset, so upscaling
    the packed SBS to the headset 4K tier only adds hundreds of ms of EASU
    work per frame (the torch fallback is ~300 ms at 1920x1200). Local Viewer,
    3D Monitor and OpenXR keep the upscale.
    """
    try:
        from utils.run_mode import normalize_run_mode
    except Exception:
        return False
    run_mode = normalize_run_mode(str(settings.get("Run Mode", "") or ""))
    return run_mode in {
        "MJPEG Streamer",
        "RTMP Streamer",
        "Streamer",
        "MJPEG",
        "RTMP",
    }


def _optional_int_setting(settings: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        if key in settings and settings[key] is not None:
            return int(settings[key])
    return None


def _depth_export_size_from_settings(
    settings: dict[str, Any],
    *,
    default_height: int = 294,
    default_width: int = 518,
) -> tuple[int, int]:
    explicit_height = _optional_int_setting(settings, "Export Height", "export_height")
    explicit_width = _optional_int_setting(settings, "Export Width", "export_width")
    depth_resolution = _optional_int_setting(settings, "Depth Resolution", "depth_resolution")
    width = explicit_width if explicit_width is not None else depth_resolution
    if width is None:
        width = int(default_width)
    if explicit_height is not None:
        height = explicit_height
    elif int(width) != int(default_width):
        height = round(float(width) * float(default_height) / float(default_width))
    else:
        height = int(default_height)
    return max(1, int(height)), max(1, int(width))


def _normalize_depth_backend(value: Any) -> DepthBackend:
    key = str(value).strip().lower().replace("-", "_")
    mapping: dict[str, DepthBackend] = {
        "auto": "auto",
        "tensorrt": "tensorrt_native",
        "tensorrt_native": "tensorrt_native",
        "tensorrt_native_graph": "tensorrt_native",
        "trt": "tensorrt_native",
        "onnx": "onnx_cuda",
        "onnx_cuda": "onnx_cuda",
        "onnx_cuda_iobinding": "onnx_cuda",
        "pytorch": "pytorch_cuda",
        "pytorch_cuda": "pytorch_cuda",
        "pytorch_rocm": "pytorch_rocm",
        "rocm": "pytorch_rocm",
        "amd_rocm": "pytorch_rocm",
        "migraphx": "migraphx_rocm",
        "migraphx_rocm": "migraphx_rocm",
        "directml": "directml",
        "dml": "directml",
        "rocm_migraphx": "migraphx_rocm",
    }
    try:
        return mapping[key]
    except KeyError as exc:
        raise ValueError(f"unknown depth backend: {value!r}") from exc


def _normalize_runtime_mode(value: Any) -> RuntimeMode:
    key = "_".join(part for part in str(value).strip().lower().replace("-", "_").replace("/", "_").replace(" ", "_").split("_") if part)
    if key in {"auto"}:
        return "auto"
    if key in {"game", "game_low_latency"}:
        return "game"
    if key in {"image", "still", "still_image", "still_image_hq"}:
        return "image"
    if key in {"debug", "debug_export"}:
        return "debug"
    return "movie"


def _normalize_stereo_quality(value: Any) -> StereoQuality:
    key = str(value).strip().lower().replace("-", "_").replace("+", "_plus").replace(" ", "_")
    mapping: dict[str, StereoQuality] = {
        "fast": "fast",
        "fast_plus": "fast_plus",
        "fastplus": "fast_plus",
        "quality": "quality_4k",
        "quality_4k": "quality_4k",
        "hq": "hq_4k",
        "hq_4k": "hq_4k",
    }
    return mapping.get(key, "quality_4k")


def _normalize_stereo_compute_backend(value: Any) -> StereoComputeBackendMode:
    key = str(value or "auto").strip().lower().replace("-", "_")
    if key in {"auto", "vendor", "vendor_default"}:
        return "auto"
    if key in {"triton", "cuda", "cuda_triton", "amd_triton", "rocm_triton"}:
        return "triton"
    if key in {"vulkan", "vulkan_compute"}:
        return "vulkan"
    if key in {"opengl", "opengl_compute", "gl"}:
        return "opengl"
    raise ValueError(f"unknown stereo compute backend: {value!r}")


def _normalize_hole_fill_mode(value: Any) -> tuple[str, int, float]:
    key = _normalize_hole_fill_mode_key(value)
    mapping = {
        "none": ("none", 0, 0.0),
        "off": ("none", 0, 0.0),
        "off_no_fill": ("none", 0, 0.0),
        "disabled": ("none", 0, 0.0),
        "no_fill": ("none", 0, 0.0),
        "no_hole_fill": ("none", 0, 0.0),
        "balanced": ("balanced", 1, 0.6),
        "quality": ("quality", 3, 1.0),
        "highest_quality": ("quality", 3, 1.0),
        "content_aware": ("quality", 3, 1.0),
        "content_aware_highest_quality": ("quality", 3, 1.0),
        "directional": ("quality", 3, 1.0),
        "directional_content_aware": ("quality", 3, 1.0),
    }
    return mapping.get(key, ("balanced", 1, 0.6))


def _normalize_hole_fill_mode_key(value: Any) -> str:
    text = str(value or "balanced").strip().lower()
    for old, new in (
        ("不补洞", "no fill"),
        ("关闭", "off"),
        ("禁用", "disabled"),
        ("均衡", "balanced"),
        ("最高质量", "highest quality"),
        ("内容感知", "content aware"),
        ("方向", "directional"),
    ):
        text = text.replace(old, new)
    for ch in ("-", "/", "+", "\\", "|", "(", ")", "[", "]"):
        text = text.replace(ch, " ")
    return "_".join(part for part in text.split() if part)


def _normalize_output_format(value: Any) -> OutputFormat:
    key = "_".join(
        part
        for part in str(value).strip().lower()
        .replace("-", "_")
        .replace("/", "_")
        .replace("+", "_")
        .split("_")
        if part
    )
    key = "_".join(part for part in key.replace(" ", "_").split("_") if part)
    mapping: dict[str, OutputFormat] = {
        "half_sbs": "half_sbs",
        "half_side_by_side": "half_sbs",
        "sbs": "half_sbs",
        "side_by_side": "half_sbs",
        "full_sbs": "full_sbs",
        "full_side_by_side": "full_sbs",
        "half_tab": "half_tab",
        "half_top_bottom": "half_tab",
        "tab": "half_tab",
        "top_bottom": "half_tab",
        "full_tab": "full_tab",
        "full_top_bottom": "full_tab",
        "mono": "mono",
        "anaglyph": "anaglyph",
        "interleaved": "interleaved",
        "leia": "leia",
        "depth_map": "depth_map",
    }
    return mapping.get(key, "half_sbs")


def preset_for_runtime_mode(mode: str) -> str:
    key = str(mode).strip().lower().replace("-", "_")
    mapping = {
        "auto": "auto",
        "traditional": "traditional_fastest",
        "traditional_fastest": "traditional_fastest",
        "movie": "cinema",
        "cinema": "cinema",
        "video": "cinema",
        "game": "game_low_latency",
        "game_low_latency": "game_low_latency",
        "image": "still_image_hq",
        "still": "still_image_hq",
        "still_image": "still_image_hq",
        "still_image_hq": "still_image_hq",
        "debug": "debug_export",
        "debug_export": "debug_export",
    }
    try:
        return mapping[key]
    except KeyError as exc:
        raise ValueError(f"unknown runtime mode: {mode!r}") from exc


def depth_provider_config_from_runtime(config: StereoRuntimeConfig) -> "DepthProviderConfig":
    from .depth_provider import DepthProviderConfig

    backend = config.depth_backend
    if backend == "auto":
        backend = "auto"
    if backend == "onnx_cuda":
        backend = "onnx_cuda_iobinding"
    if backend == "pytorch_cuda":
        backend = "pytorch_cuda"
    onnx_path = config.onnx_path if backend == "migraphx_rocm" else None
    engine_path = config.migraphx_graph_path if backend == "migraphx_rocm" else None
    build_engine = config.build_migraphx_graph if backend == "migraphx_rocm" else config.build_trt_engine
    force_rebuild = config.force_rebuild_migraphx if backend == "migraphx_rocm" else config.force_rebuild_trt

    return DepthProviderConfig(
        backend=backend,
        model_id=config.resolved_model_id,
        model_name=ModelRegistry.default().get(config.resolved_model_id).name,
        device=config.device,
        cache_dir=config.model_path.parent,
        onnx_path=onnx_path,
        engine_path=engine_path,
        depth_resolution=int(config.export_width),
        onnx_dtype=config.onnx_dtype,
        local_files_only=False,
        prefer_native_tensorrt=backend == "tensorrt_native",
        prefer_tensorrt=backend == "tensorrt_native",
        prefer_onnx=backend == "onnx_cuda_iobinding",
        use_iobinding=True,
        use_dlpack=backend == "onnx_cuda_iobinding",
        build_engine=build_engine,
        force_rebuild=force_rebuild,
        use_cuda_graph=config.use_cuda_graph,
        profile_sync=config.profile_sync,
        execution_slot_count=config.parallel_inference_workers,
        depth_upsample=config.depth_upsample,
        depth_upsample_edge_strength=config.depth_upsample_edge_strength,
    )


def openxr_render_config_from_snapshot(
    snapshot,
    *,
    render_size: tuple[int, int] | None = None,
    preset: str = "standard",
    screen_roll: float = 0.0,
    padding_mode: str = "reflection",
):
    """Convert normalized runtime settings into OpenXR render-core uniforms."""
    from .openxr_render import OpenXRRenderConfig
    from .parallax import resolve_parallax_budget

    depth_strength = 2.0 if snapshot.depth_strength is None else float(snapshot.depth_strength)
    convergence = 0.0 if snapshot.convergence is None else float(snapshot.convergence)
    parallax_preset = str(snapshot.parallax_preset or preset or "standard")
    max_disparity_px = snapshot.max_disparity_px
    if max_disparity_px is None and render_size is not None:
        budget = resolve_parallax_budget(
            render_width=int(render_size[0]),
            render_height=int(render_size[1]),
            preset=parallax_preset,
            convergence=convergence,
        )
        max_disparity_px = float(budget.max_disparity_px)

    return OpenXRRenderConfig(
        depth_strength=depth_strength,
        convergence=convergence,
        max_disparity_px=max_disparity_px,
        parallax_preset=parallax_preset,
        foreground_shift_scale=1.0 if snapshot.foreground_shift_scale is None else float(snapshot.foreground_shift_scale),
        midground_shift_scale=1.0 if snapshot.midground_shift_scale is None else float(snapshot.midground_shift_scale),
        background_shift_scale=1.0 if snapshot.background_shift_scale is None else float(snapshot.background_shift_scale),
        screen_roll=float(screen_roll),
        padding_mode=padding_mode,
        vulkan_projection_min_lod=0.0 if snapshot.vulkan_projection_min_lod is None else float(snapshot.vulkan_projection_min_lod),
        vulkan_projection_max_lod=0.35 if snapshot.vulkan_projection_max_lod is None else float(snapshot.vulkan_projection_max_lod),
        vulkan_projection_mip_lod_bias=-0.35 if snapshot.vulkan_projection_mip_lod_bias is None else float(snapshot.vulkan_projection_mip_lod_bias),
        vulkan_projection_rcas_sharpness=0.5 if snapshot.vulkan_projection_rcas_sharpness is None else float(snapshot.vulkan_projection_rcas_sharpness),
    )


def stereo_config_from_runtime(config: StereoRuntimeConfig) -> "StereoConfig":
    from .presets import normalize_preset, stereo_config_for_preset

    preset = normalize_preset(config.stereo_preset or preset_for_runtime_mode(config.mode))
    layers = config.layers
    if config.stereo_quality == "hq_4k" and layers < 3:
        layers = 3

    temporal = config.temporal
    temporal_strength = config.temporal_strength
    auto_reset_temporal = config.auto_reset_temporal
    scene_reset_threshold = config.scene_reset_threshold
    depth_pop = config.depth_pop
    depth_antialias_strength = config.depth_antialias_strength
    if config.stereo_quality == "fast":
        temporal = False
        temporal_strength = 0.0
        auto_reset_temporal = False
        scene_reset_threshold = 0.0
        depth_pop = 0.0
        depth_antialias_strength = 0.0

    return stereo_config_for_preset(
        preset,
        output_format=config.output_format,
        overrides={
            "backend": config.stereo_quality,
            "layers": layers,
            "occlusion": config.occlusion,
            "symmetric": config.symmetric,
            "hole_fill": "none" if config.hole_fill_mode == "none" else config.hole_fill,
            "temporal": temporal,
            "depth_strength": config.depth_strength,
            "convergence": config.convergence,
            "max_disparity_px": config.max_disparity_px,
            "parallax_preset": config.parallax_preset,
            "foreground_shift_scale": config.foreground_shift_scale,
            "midground_shift_scale": config.midground_shift_scale,
            "background_shift_scale": config.background_shift_scale,
            "dynamic_convergence_enabled": config.dynamic_convergence_enabled,
            "dynamic_convergence_strength": config.dynamic_convergence_strength,
            "dynamic_convergence_target": config.dynamic_convergence_target,
            "dynamic_convergence_alpha": config.dynamic_convergence_alpha,
            "temporal_strength": temporal_strength,
            "auto_reset_temporal": auto_reset_temporal,
            "scene_reset_threshold": scene_reset_threshold,
            "depth_pop": depth_pop,
            "depth_antialias_strength": depth_antialias_strength,
            "edge_threshold": config.edge_threshold,
            "edge_dilation": config.edge_dilation,
            "mask_feather_radius": config.mask_feather_radius,
            "hole_fill_mode": config.hole_fill_mode,
            "hole_fill_radius": config.hole_fill_radius,
            "hole_fill_strength": config.hole_fill_strength,
            "screen_edge_mask_suppression": config.screen_edge_mask_suppression,
            "cross_eyed": config.cross_eyed,
            "anaglyph_method": config.anaglyph_method,
            "debug_output": config.debug_output,
            "fused": config.fused,
            "output_quality_enabled": config.output_quality_enabled,
            "output_headset_tier_k": config.output_headset_tier_k,
            "output_min_lod": config.output_min_lod,
            "output_max_lod": config.output_max_lod,
            "output_mip_lod_bias": config.output_mip_lod_bias,
            "output_rcas_sharpness": config.output_rcas_sharpness,
        },
    )


StereoLabRuntimeConfig = StereoRuntimeConfig
DepthRuntimeConfig = StereoRuntimeConfig
StereoLabDepthRuntimeConfig = StereoRuntimeConfig

def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parallel_inference_workers(settings: dict[str, Any]) -> int:
    raw = settings.get("Parallel Inference Workers")
    if raw is None:
        return 2 if _to_bool(settings.get("Parallel Inference", False)) else 1
    try:
        return min(3, max(1, int(raw)))
    except (TypeError, ValueError):
        return 2 if _to_bool(settings.get("Parallel Inference", False)) else 1


def _normalize_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    key = str(value).strip().lower()
    if key in {"auto", "none", ""}:
        return None
    return key in {"1", "true", "yes", "on"}
