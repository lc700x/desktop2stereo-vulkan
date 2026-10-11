from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from .model_registry import DepthModelSpec, ModelRegistry, resolve_model_dir
from .progress import stage_progress

OnnxDtypeMode = Literal["auto", "fp16", "fp32"]
ArtifactBackend = Literal["onnx", "tensorrt", "migraphx"]
MODEL_WEIGHT_FILENAMES = (
    "model.safetensors",
    "model.pt",
    "model.ckpt",
    "pytorch_model.bin",
)
VIDEO_WEIGHT_FILENAMES = (
    "video_depth_anything_vits.pth",
    "video_depth_anything_vitb.pth",
    "video_depth_anything_vitl.pth",
    "metric_video_depth_anything_vits.pth",
    "metric_video_depth_anything_vitb.pth",
    "metric_video_depth_anything_vitl.pth",
)
ZIPDEPTH_CHECKPOINT_FILENAMES = {
    "zipdepth": "zipdepth_base.pth",
    "zipdepth-npu": "zipdepth_base_npu.pth",
}
MODEL_CONFIG_FILENAMES = ("config.json",)
# These model families build their network from the bundled implementation.
# Their Hugging Face repositories provide weights, not the runtime structure.
LOCAL_CONFIG_MODEL_FAMILIES = frozenset({
    "infinidepth",
    "zipdepth",
    "zipdepth-npu",
    "da3",
    "video-depth-anything",
    "metric-video-depth-anything",
})


@dataclass(frozen=True)
class ModelArtifactPaths:
    model_id: str
    model_name: str
    model_dir: Path
    onnx_fp16_path: Path
    onnx_fp32_path: Path
    trt_fp16_path: Path
    migraphx_fp16_path: Path
    export_height: int
    export_width: int

    def onnx_path_for_dtype(self, dtype_name: str) -> Path:
        if dtype_name == "fp16":
            return self.onnx_fp16_path
        if dtype_name == "fp32":
            return self.onnx_fp32_path
        raise ValueError(f"unknown ONNX dtype: {dtype_name!r}")

    def trt_path_for_dtype(self, dtype_name: str) -> Path:
        if dtype_name == "fp16":
            return self.trt_fp16_path
        if dtype_name == "fp32":
            return self.trt_fp16_path.with_name(self.trt_fp16_path.name.replace("model_fp16_", "model_fp32_"))
        raise ValueError(f"unknown TensorRT dtype: {dtype_name!r}")

    def to_report(self) -> dict[str, str | int]:
        report = asdict(self)
        report["model_dir"] = str(self.model_dir)
        report["onnx_fp16_path"] = str(self.onnx_fp16_path)
        report["onnx_fp32_path"] = str(self.onnx_fp32_path)
        report["trt_fp16_path"] = str(self.trt_fp16_path)
        report["migraphx_fp16_path"] = str(self.migraphx_fp16_path)
        return report


@dataclass(frozen=True)
class PreparedModelArtifacts:
    paths: ModelArtifactPaths
    downloaded: bool
    onnx_ready: bool
    trt_ready: bool
    selected_onnx_path: Path | None = None
    migraphx_ready: bool = False
    selected_migraphx_path: Path | None = None
    notes: tuple[str, ...] = ()

    def to_report(self) -> dict[str, object]:
        return {
            "paths": self.paths.to_report(),
            "downloaded": self.downloaded,
            "onnx_ready": self.onnx_ready,
            "trt_ready": self.trt_ready,
            "selected_onnx_path": str(self.selected_onnx_path) if self.selected_onnx_path else None,
            "migraphx_ready": self.migraphx_ready,
            "selected_migraphx_path": str(self.selected_migraphx_path) if self.selected_migraphx_path else None,
            "notes": list(self.notes),
        }


def _nearest_multiple(value: int, patch: int) -> int:
    down = (value // patch) * patch
    up = down + patch
    return max(patch, up if abs(up - value) <= abs(value - down) else down)


def export_size_for_model(model_id: str, export_height: int = 294, export_width: int = 518) -> tuple[int, int]:
    model_lower = str(model_id).lower()
    patch = 32 if "zipdepth" in model_lower else 16 if "infinidepth" in model_lower else 14
    return _nearest_multiple(int(export_height), patch), _nearest_multiple(int(export_width), patch)


def artifact_paths_for_model(
    model_name_or_id: str,
    *,
    cache_dir: str | Path = "./models",
    model_dir: str | Path | None = None,
    export_height: int = 294,
    export_width: int = 518,
    registry: ModelRegistry | None = None,
) -> ModelArtifactPaths:
    registry = registry or ModelRegistry.default()
    spec = registry.get(model_name_or_id)
    if spec.family in ZIPDEPTH_CHECKPOINT_FILENAMES and (export_height, export_width) == (294, 518):
        export_height = 384
        export_width = int(round(export_height * 518 / 294))
    export_height, export_width = export_size_for_model(spec.model_id, export_height, export_width)
    root = Path(model_dir) if model_dir is not None else resolve_model_dir(spec.model_id, cache_dir)
    artifact_prefix = (
        f"{spec.name.lower().replace('-', '_')}_"
        if spec.family in ZIPDEPTH_CHECKPOINT_FILENAMES
        else ""
    )
    return ModelArtifactPaths(
        model_id=spec.model_id,
        model_name=spec.name,
        model_dir=root,
        onnx_fp16_path=root / f"{artifact_prefix}model_fp16_{export_height}x{export_width}.onnx",
        onnx_fp32_path=root / f"{artifact_prefix}model_fp32_{export_height}x{export_width}.onnx",
        trt_fp16_path=root / f"{artifact_prefix}model_fp16_{export_height}x{export_width}.trt",
        migraphx_fp16_path=root / f"{artifact_prefix}model_fp16_{export_height}x{export_width}.mgx",
        export_height=int(export_height),
        export_width=int(export_width),
    )


def ensure_model_downloaded(
    model_spec: DepthModelSpec,
    *,
    cache_dir: str | Path = "./models",
    local_files_only: bool = False,
    force_download: bool = False,
) -> Path:
    """Download or validate a Hugging Face model without importing D2S settings."""
    model_dir = model_spec.model_dir(cache_dir)
    if not force_download and _model_cache_is_complete(model_spec, model_dir):
        return model_dir
    if model_spec.family == "infinidepth":
        from .depth_provider import _resolve_hf_model_file

        _resolve_hf_model_file(
            model_spec.model_id,
            cache_dir,
            local_files_only=local_files_only,
            force_download=force_download,
        )
        if not model_dir.exists():
            raise FileNotFoundError(f"model weights resolved but model directory was not found: {model_dir}")
        return model_dir

    if model_spec.family in ZIPDEPTH_CHECKPOINT_FILENAMES:
        filename = ZIPDEPTH_CHECKPOINT_FILENAMES[model_spec.family]
        if local_files_only:
            raise FileNotFoundError(f"local ZipDepth checkpoint is missing: {model_dir / filename}")
        from .depth_provider import _download_hf_file_direct, _reachable_hf_endpoints

        last_error: Exception | None = None
        for endpoint in _reachable_hf_endpoints(model_spec.model_id):
            try:
                _download_hf_file_direct(
                    model_spec.model_id,
                    filename,
                    cache_dir,
                    endpoint,
                )
                if _model_cache_is_complete(model_spec, model_dir):
                    return model_dir
            except Exception as exc:
                last_error = exc
        raise FileNotFoundError(
            f"ZipDepth checkpoint download failed: {model_dir / filename}"
        ) from last_error

    from .depth_provider import (
        _download_hf_file_direct,
        _hf_download_progress_patch,
        _hf_endpoint,
        _reachable_hf_endpoints,
    )
    from huggingface_hub import snapshot_download

    endpoints = _reachable_hf_endpoints(model_spec.model_id)
    last_error: Exception | None = None
    for endpoint in endpoints:
        try:
            # snapshot_download reads HF_ENDPOINT when the Hub client is built.
            # Keep the probe and the actual download on the same endpoint.
            with _hf_endpoint(endpoint), _hf_download_progress_patch():
                snapshot_download(
                    repo_id=model_spec.model_id,
                    cache_dir=str(cache_dir),
                    local_files_only=local_files_only,
                    force_download=force_download,
                )
            if _model_cache_is_complete(model_spec, model_dir):
                return model_dir
        except Exception as exc:
            last_error = exc
            if local_files_only:
                raise
    if local_files_only:
        raise FileNotFoundError(f"local model files are incomplete: {model_dir}") from last_error
    weight_filenames = MODEL_WEIGHT_FILENAMES
    if model_spec.family in {"video-depth-anything", "metric-video-depth-anything"}:
        weight_filenames = (*MODEL_WEIGHT_FILENAMES, *VIDEO_WEIGHT_FILENAMES)
    for endpoint in endpoints:
        for filename in weight_filenames:
            try:
                existing = model_dir / filename
                if existing.is_file() and existing.stat().st_size > 0:
                    break
                _download_hf_file_direct(model_spec.model_id, filename, cache_dir, endpoint)
                break
            except Exception as exc:
                last_error = exc
        for filename in MODEL_CONFIG_FILENAMES:
            try:
                existing = model_dir / filename
                if existing.is_file() and existing.stat().st_size > 0:
                    continue
                _download_hf_file_direct(model_spec.model_id, filename, cache_dir, endpoint)
            except Exception as exc:
                last_error = exc
        if _model_cache_is_complete(model_spec, model_dir):
            return model_dir
    if not model_dir.exists():
        raise FileNotFoundError(f"download completed but model directory was not found: {model_dir}")
    raise FileNotFoundError(f"download completed but model files are incomplete: {model_dir}") from last_error


def _model_cache_is_complete(model_spec: DepthModelSpec, model_dir: str | Path) -> bool:
    """Require model metadata as well as weights before allowing ONNX export."""
    root = Path(model_dir)
    if model_spec.family in ZIPDEPTH_CHECKPOINT_FILENAMES:
        weight = root / ZIPDEPTH_CHECKPOINT_FILENAMES[model_spec.family]
        return weight.is_file() and weight.stat().st_size > 0
    if find_local_model_weight(root) is None:
        return False
    if model_spec.family in LOCAL_CONFIG_MODEL_FAMILIES:
        return True
    return any(
        path.is_file() and path.stat().st_size > 0
        for filename in MODEL_CONFIG_FILENAMES
        for path in (root / filename, *root.rglob(filename))
    )


def find_local_model_weight(model_dir: str | Path) -> Path | None:
    root = Path(model_dir)
    if not root.exists():
        return None
    for filename in ZIPDEPTH_CHECKPOINT_FILENAMES.values():
        checkpoint = root / filename
        if checkpoint.is_file() and checkpoint.stat().st_size > 0:
            return checkpoint
    if root.name.lower() == "models--pablovela5620--zipdepth":
        return None
    for filename in MODEL_WEIGHT_FILENAMES:
        direct = root / filename
        if direct.is_file() and direct.stat().st_size > 0:
            return direct
        for path in root.rglob(filename):
            if path.is_file() and path.stat().st_size > 0:
                return path
    # Video-Depth-Anything repositories publish encoder checkpoints as named
    # .pth files rather than the standard Transformers filenames.
    for path in root.rglob("*.pth"):
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def select_existing_onnx(paths: ModelArtifactPaths, dtype: OnnxDtypeMode = "auto") -> Path | None:
    if dtype == "fp16":
        if paths.onnx_fp16_path.exists():
            return paths.onnx_fp16_path
        return paths.onnx_fp32_path if paths.onnx_fp32_path.exists() else None
    if dtype == "fp32":
        return paths.onnx_fp32_path if paths.onnx_fp32_path.exists() else None
    if paths.onnx_fp16_path.exists():
        return paths.onnx_fp16_path
    if paths.onnx_fp32_path.exists():
        return paths.onnx_fp32_path
    return None


def select_existing_trt(paths: ModelArtifactPaths, dtype: OnnxDtypeMode = "auto") -> Path | None:
    fp32_path = paths.trt_path_for_dtype("fp32")
    if dtype == "fp16":
        if paths.trt_fp16_path.exists():
            return paths.trt_fp16_path
        return fp32_path if fp32_path.exists() else None
    if dtype == "fp32":
        return fp32_path if fp32_path.exists() else None
    if paths.trt_fp16_path.exists():
        return paths.trt_fp16_path
    if fp32_path.exists():
        return fp32_path
    return None


def select_existing_migraphx(paths: ModelArtifactPaths, dtype: OnnxDtypeMode = "auto") -> Path | None:
    if dtype == "fp32":
        return None
    return paths.migraphx_fp16_path if paths.migraphx_fp16_path.exists() else None


def dtype_from_onnx_path(path: str | Path) -> str:
    name = Path(path).name
    if "model_fp32_" in name:
        return "fp32"
    return "fp16"


def ensure_onnx_exported(
    model_spec: DepthModelSpec,
    *,
    cache_dir: str | Path = "./models",
    model_dir: str | Path | None = None,
    height: int = 294,
    width: int = 518,
    dtype: OnnxDtypeMode = "auto",
    local_files_only: bool = False,
    export_if_missing: bool = False,
) -> Path:
    paths = artifact_paths_for_model(
        model_spec.name,
        cache_dir=cache_dir,
        model_dir=model_dir,
        export_height=height,
        export_width=width,
    )
    existing = select_existing_onnx(paths, dtype)
    if existing is not None:
        return existing
    if not export_if_missing:
        raise FileNotFoundError(f"ONNX artifact not found for {model_spec.model_id}: {paths.onnx_fp16_path} or {paths.onnx_fp32_path}")

    from .onnx_export import export_depth_model_onnx

    output_path = paths.onnx_fp16_path if dtype in ("auto", "fp16") else paths.onnx_fp32_path
    result = export_depth_model_onnx(
        model_id=model_spec.model_id,
        model_name=model_spec.name,
        output_path=output_path,
        cache_dir=cache_dir,
        # Use the same patch-aligned dimensions used in the artifact filename.
        # Custom models such as VDA reject the unaligned request before export.
        height=paths.export_height,
        width=paths.export_width,
        dtype=dtype,
        local_files_only=local_files_only,
        force_download=False,
    )
    return result.output_path


def ensure_tensorrt_engine(
    model_spec: DepthModelSpec,
    *,
    onnx_path: str | Path,
    engine_path: str | Path,
    build_if_missing: bool = False,
    force_rebuild: bool = False,
    workspace_gb: int = 4,
) -> Path:
    engine_path = Path(engine_path)
    if engine_path.exists() and not force_rebuild:
        return engine_path
    if not build_if_missing and not force_rebuild:
        raise FileNotFoundError(f"TensorRT engine not found for {model_spec.model_id}: {engine_path}")

    from .providers.nvidia.tensorrt_native import build_native_tensorrt_engine

    return build_native_tensorrt_engine(
        onnx_path,
        engine_path,
        fp16=dtype_from_onnx_path(onnx_path) == "fp16",
        workspace_gb=workspace_gb,
        force=force_rebuild,
    )


def ensure_migraphx_graph(
    model_spec: DepthModelSpec,
    *,
    onnx_path: str | Path,
    graph_path: str | Path,
    build_if_missing: bool = False,
    force_rebuild: bool = False,
) -> Path:
    graph_path = Path(graph_path)
    if graph_path.exists() and not force_rebuild:
        return graph_path
    if not build_if_missing and not force_rebuild:
        raise FileNotFoundError(f"MIGraphX graph not found for {model_spec.model_id}: {graph_path}")

    from .providers.amd.migraphx import build_migraphx_graph

    return build_migraphx_graph(
        onnx_path,
        graph_path,
        fp16=dtype_from_onnx_path(onnx_path) == "fp16",
        force=force_rebuild,
    )


def prepare_model_artifacts(
    model_name_or_id: str,
    *,
    cache_dir: str | Path = "./models",
    model_dir: str | Path | None = None,
    export_height: int = 294,
    export_width: int = 518,
    onnx_dtype: OnnxDtypeMode = "auto",
    local_files_only: bool = False,
    force_download: bool = False,
    download_if_missing: bool = False,
    export_onnx_if_missing: bool = False,
    artifact_backend: ArtifactBackend = "onnx",
    build_trt_if_missing: bool = False,
    force_rebuild_trt: bool = False,
    trt_workspace_gb: int = 4,
    build_migraphx_if_missing: bool = False,
    force_rebuild_migraphx: bool = False,
) -> PreparedModelArtifacts:
    registry = ModelRegistry.default()
    spec = registry.get(model_name_or_id)
    paths = artifact_paths_for_model(
        spec.name,
        cache_dir=cache_dir,
        model_dir=model_dir,
        export_height=export_height,
        export_width=export_width,
        registry=registry,
    )

    notes: list[str] = []
    with stage_progress("Runtime preparation", 4) as progress:
        progress.set_postfix_str("checking local artifacts", refresh=True)
        wants_trt = artifact_backend == "tensorrt"
        wants_migraphx = artifact_backend == "migraphx"
        selected_onnx = select_existing_onnx(paths, onnx_dtype)
        existing_trt = select_existing_trt(paths, onnx_dtype) if wants_trt and not force_rebuild_trt else None
        existing_migraphx = select_existing_migraphx(paths, onnx_dtype) if wants_migraphx and not force_rebuild_migraphx else None
        trt_path = existing_trt or (paths.trt_path_for_dtype(dtype_from_onnx_path(selected_onnx)) if selected_onnx is not None else paths.trt_path_for_dtype(onnx_dtype if onnx_dtype in ("fp16", "fp32") else "fp16"))
        migraphx_path = existing_migraphx or paths.migraphx_fp16_path
        trt_ready = existing_trt is not None
        migraphx_ready = existing_migraphx is not None
        progress.update(1)

        progress.set_postfix_str("checking ONNX artifact", refresh=True)
        executable_ready = trt_ready or migraphx_ready
        needs_onnx_export = export_onnx_if_missing or (wants_trt and build_trt_if_missing) or (wants_migraphx and build_migraphx_if_missing)
        if selected_onnx is None and not executable_ready and needs_onnx_export:
            model_cache_complete = _model_cache_is_complete(spec, paths.model_dir)
            if force_download or not _model_cache_is_complete(spec, paths.model_dir):
                ensure_model_downloaded(
                    spec,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                    force_download=force_download,
                )
            progress.set_postfix_str("exporting ONNX", refresh=True)
            selected_onnx = ensure_onnx_exported(
                spec,
                cache_dir=cache_dir,
                model_dir=paths.model_dir,
                height=export_height,
                width=export_width,
                dtype=onnx_dtype,
                # A downloaded weight does not imply that the Hub metadata is
                # cached. Let from_pretrained use the selected endpoint unless
                # the user explicitly requested offline mode.
                local_files_only=local_files_only or model_cache_complete,
                export_if_missing=True,
            )
            trt_path = paths.trt_path_for_dtype(dtype_from_onnx_path(selected_onnx))
            trt_ready = wants_trt and trt_path.exists()
        elif selected_onnx is None and not executable_ready:
            notes.append("onnx artifact missing")
        progress.update(1)

        progress.set_postfix_str("checking runtime artifact", refresh=True)
        if wants_trt and not trt_ready and build_trt_if_missing and selected_onnx is not None:
            progress.set_postfix_str("building TensorRT engine", refresh=True)
            ensure_tensorrt_engine(
                spec,
                onnx_path=selected_onnx,
                engine_path=trt_path,
                build_if_missing=True,
                force_rebuild=force_rebuild_trt,
                workspace_gb=trt_workspace_gb,
            )
            trt_ready = trt_path.exists()
        elif wants_migraphx and not migraphx_ready and build_migraphx_if_missing and selected_onnx is not None:
            progress.set_postfix_str("building MIGraphX graph", refresh=True)
            ensure_migraphx_graph(
                spec,
                onnx_path=selected_onnx,
                graph_path=migraphx_path,
                build_if_missing=True,
                force_rebuild=force_rebuild_migraphx,
            )
            migraphx_ready = migraphx_path.exists()
        elif wants_trt and not trt_ready:
            notes.append("TensorRT engine missing")
        elif wants_migraphx and not migraphx_ready:
            notes.append("MIGraphX graph missing")
        progress.set_postfix_str("ready", refresh=True)
        progress.update(1)

        if not paths.model_dir.exists() and local_files_only and not (selected_onnx or trt_ready or migraphx_ready):
            raise FileNotFoundError(f"model directory not found: {paths.model_dir}")
        downloaded = paths.model_dir.exists()

    return PreparedModelArtifacts(
        paths=paths,
        downloaded=downloaded,
        onnx_ready=selected_onnx is not None,
        trt_ready=trt_ready,
        selected_onnx_path=selected_onnx,
        migraphx_ready=migraphx_ready,
        selected_migraphx_path=migraphx_path if migraphx_ready else None,
        notes=tuple(notes),
    )
