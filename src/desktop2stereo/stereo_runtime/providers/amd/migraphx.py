from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from stereo_runtime.depth_onnx_provider import ModelOnnxPreprocessor, _dtype_from_onnx_name
from stereo_runtime.depth_provider import (
    DISTILL_ANY_DEPTH_BASE_MODEL_ID,
    DepthProfileResult,
    DepthProviderInfo,
    _is_zipdepth_model,
    ZipDepthProvider,
    _postprocess_generic_depth,
)
from stereo_runtime.depth_upsample import DepthUpsampleMode, upsample_depth
from stereo_runtime.output import ensure_bchw
from stereo_runtime.progress import activity_progress

from .pytorch_rocm import create_pytorch_rocm_provider, is_rocm_torch_available

ZIPDEPTH_MIGRAPHX_MAX_ABS_DIFF = 0.05
ZIPDEPTH_MIGRAPHX_RAW_MAX_ABS_DIFF = 0.005


def _migraphx_output_index(name: str) -> int:
    suffix = name.rsplit("#output_", 1)[-1].lstrip(":0")
    return int(suffix or 0)


def is_migraphx_available() -> bool:
    try:
        import migraphx  # noqa: F401
    except Exception:
        return False
    return True


def build_migraphx_graph(
    onnx_path: str | Path,
    graph_path: str | Path,
    *,
    fp16: bool = True,
    fp8: bool = True,
    force_fp32: bool = False,
    force: bool = False,
) -> Path:
    onnx_path = Path(onnx_path)
    graph_path = Path(graph_path)
    if graph_path.exists() and not force:
        return graph_path
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    import migraphx as mx

    graph_path.parent.mkdir(parents=True, exist_ok=True)
    with activity_progress(f"Parsing ONNX for MIGraphX: {onnx_path.name}"):
        prog = mx.parse_onnx(str(onnx_path))
    if fp16 and not force_fp32:
        with activity_progress(f"Optimizing MIGraphX graph: {graph_path.name}"):
            if fp8 and hasattr(mx, "autocast_fp8"):
                try:
                    mx.autocast_fp8(prog)
                except Exception:
                    mx.quantize_fp16(prog)
            else:
                mx.quantize_fp16(prog)
    with activity_progress(f"Building MIGraphX graph: {graph_path.name}"):
        prog.compile(mx.get_target("gpu"), offload_copy=False)
    with activity_progress(f"Saving MIGraphX graph: {graph_path.name}"):
        mx.save(prog, str(graph_path))
    return graph_path


class MIGraphXEngine:
    def __init__(self, graph_path: str | Path, *, device: str | torch.device = "cuda") -> None:
        import migraphx as mx

        self.mx = mx
        self.graph_path = Path(graph_path)
        self.device = torch.device(device)
        with activity_progress(f"Loading MIGraphX graph: {self.graph_path.name}"):
            self.prog = mx.load(str(self.graph_path))
        self._logged_zero_copy = False
        param_shapes = self.prog.get_parameter_shapes()
        self.input_name = None
        self._in_shape = None
        self._out_params = {}
        for name, shape in param_shapes.items():
            if "#output_" in name:
                self._out_params[name] = shape
            else:
                self.input_name = name
                self._in_shape = shape
        if self.input_name is None or self._in_shape is None:
            raise RuntimeError("MIGraphX graph has no input parameter")
        if not self._out_params:
            raise RuntimeError(
                "MIGraphX graph has no output parameter buffers. "
                "Recompile it with offload_copy=False or delete the cached .mgx file."
            )
        self._out_names = tuple(sorted(self._out_params, key=_migraphx_output_index))
        self._out_name = self._out_names[0]
        self._out_lens = list(self._out_params[self._out_name].lens())
        mx_to_torch = {1: torch.float16, 2: torch.float32, 3: torch.float64}
        self._mgx_in_dtype = mx_to_torch.get(int(self._in_shape.type()), torch.float32)
        self._mgx_out_dtypes = {
            name: mx_to_torch.get(int(shape.type()), torch.float32)
            for name, shape in self._out_params.items()
        }

    @property
    def input_shape(self) -> tuple[int, ...]:
        return tuple(int(dim) for dim in self._in_shape.lens())

    @property
    def input_image_size(self) -> tuple[int, int] | None:
        shape = self.input_shape
        if len(shape) not in (4, 5) or any(dim < 1 for dim in shape[-2:]):
            return None
        return int(shape[-2]), int(shape[-1])

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        tensor = tensor.to(device=self.device)
        if len(self.input_shape) == 5 and tensor.ndim == 4:
            # Video models are exported with a singleton temporal dimension;
            # the runtime still processes one frame at a time.
            tensor = tensor.unsqueeze(1)
        if tensor.ndim != len(self.input_shape):
            raise RuntimeError(
                f"MIGraphX input rank mismatch: graph={self.input_shape} tensor={tuple(tensor.shape)}"
            )
        if tensor.dtype != self._mgx_in_dtype or not tensor.is_contiguous():
            tensor = tensor.contiguous().to(dtype=self._mgx_in_dtype)
        in_arg = self.mx.argument_from_pointer(self._in_shape, tensor.data_ptr())
        outputs = {
            name: torch.empty(
                tuple(self._out_params[name].lens()),
                dtype=self._mgx_out_dtypes[name],
                device=self.device,
            )
            for name in self._out_names
        }
        output_args = {
            name: self.mx.argument_from_pointer(self._out_params[name], output.data_ptr())
            for name, output in outputs.items()
        }
        stream = torch.cuda.current_stream(self.device)
        self.prog.run_async(
            {self.input_name: in_arg, **output_args},
            stream.cuda_stream,
            "ihipStream_t",
        )
        if not self._logged_zero_copy:
            output_summary = (
                f"{len(outputs)} GPU outputs" if len(outputs) > 1 else str(tuple(outputs[self._out_name].shape))
            )
            print(
                f"[MIGraphX] Zero-copy GPU path active | input={tuple(tensor.shape)} {tensor.dtype} "
                f"-> output={output_summary}"
            )
            self._logged_zero_copy = True
        if len(self._out_names) == 1:
            return outputs[self._out_name]
        return tuple(outputs[name] for name in self._out_names)


class MIGraphXDepthProvider:
    def __init__(
        self,
        *,
        device: str | torch.device = "cuda",
        cache_dir: str | Path | None = None,
        onnx_path: str | Path | None = None,
        graph_path: str | Path | None = None,
        build_graph: bool = False,
        force_rebuild: bool = False,
        depth_upsample: DepthUpsampleMode = "bilinear",
        depth_upsample_edge_strength: float = 0.35,
        model_id: str = DISTILL_ANY_DEPTH_BASE_MODEL_ID,
        model_name: str | None = None,
        depth_resolution: int = 518,
        patch_size: int | None = 14,
        local_files_only: bool = True,
        force_download: bool = False,
        allow_pytorch_fallback: bool = True,
        use_torch_compile: bool = False,
    ) -> None:
        self.device = torch.device(device)
        # cuDNN autotuning is disabled for the ROCm depth path (per project policy).
        torch.backends.cudnn.benchmark = False
        self.profile_sync = False
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.onnx_path = Path(onnx_path) if onnx_path is not None else None
        self.graph_path = Path(graph_path) if graph_path is not None else None
        self.build_graph = bool(build_graph)
        self.force_rebuild = bool(force_rebuild)
        self.depth_upsample = depth_upsample
        self.depth_upsample_edge_strength = float(depth_upsample_edge_strength)
        self.model_id = str(model_id)
        self.model_name = str(model_name or model_id)
        self.depth_resolution = max(1, int(depth_resolution))
        self.local_files_only = bool(local_files_only)
        self.force_download = bool(force_download)
        self.allow_pytorch_fallback = bool(allow_pytorch_fallback)
        self.use_torch_compile = bool(use_torch_compile)
        self._zipdepth_parity_checked = False
        self._zipdepth_fallback: ZipDepthProvider | None = None
        self._zipdepth_model_provider: ZipDepthProvider | None = None
        self._zipdepth_runtime_logged = False
        self.patch_size = (
            32
            if _is_zipdepth_model(self.model_id)
            else int(patch_size or (16 if "infinidepth" in self.model_id.lower() else 14))
        )
        # The input dtype follows the ONNX artifact; MIGraphX quantizes model
        # internals to FP8 when available and otherwise to FP16.
        self._preprocessor_dtype = _dtype_from_onnx_name(self.onnx_path, torch.float16)
        self.preprocessor = ModelOnnxPreprocessor(
            model_id=self.model_id,
            device=self.device,
            dtype=self._preprocessor_dtype,
            target_resolution=self.depth_resolution,
            patch_size=self.patch_size,
        )
        self.engine: MIGraphXEngine | None = None
        self.info = DepthProviderInfo(
            provider="MIGraphX",
            model_name=self.model_name,
            model_id=self.model_id,
            depth_resolution=self.depth_resolution,
            cache_dir=str(self.cache_dir or ""),
            load_mode="local_files_only",
            depth_backend="migraphx_rocm",
            runtime="migraphx",
            onnx_path=str(self.onnx_path) if self.onnx_path else None,
            execution_provider="MIGraphX ROCm",
            output_device=str(self.device),
        )

    def _zipdepth_encoder_input_size(self) -> tuple[int, int]:
        if self.onnx_path is not None and self.onnx_path.is_file():
            import onnx

            model = onnx.load(str(self.onnx_path), load_external_data=False)
            for value in model.graph.input:
                dims = value.type.tensor_type.shape.dim
                if len(dims) >= 4:
                    height = int(dims[-2].dim_value)
                    width = int(dims[-1].dim_value)
                    if height > 0 and width > 0:
                        return height, width

        if self.graph_path is not None and self.graph_path.is_file():
            import migraphx as mx

            graph = mx.load(str(self.graph_path))
            for name, shape in graph.get_parameter_shapes().items():
                if "#output_" not in name:
                    dims = shape.lens()
                    if len(dims) >= 4:
                        return int(dims[-2]), int(dims[-1])

        return self.preprocessor.input_size(294, 518)

    def _load_zipdepth_encoder(self) -> MIGraphXEngine:
        if self.graph_path is None:
            raise FileNotFoundError("MIGraphX graph path is required")

        if self._zipdepth_model_provider is None:
            self._zipdepth_model_provider = ZipDepthProvider(
                model_id=self.model_id,
                model_name=self.model_name,
                device=self.device,
                cache_dir=self.cache_dir,
                dtype=self._preprocessor_dtype,
                depth_resolution=self.depth_resolution,
                patch_size=self.patch_size,
                local_files_only=self.local_files_only,
                force_download=self.force_download,
                use_torch_compile=False,
            )
        torch_model = self._zipdepth_model_provider.load()

        encoder_graph_path = self.graph_path.with_name(f"{self.graph_path.stem}_encoder_v3.mgx")
        encoder_onnx_path = self.graph_path.with_name(f"{self.graph_path.stem}_encoder_v3.onnx")
        if self.force_rebuild or not encoder_graph_path.is_file():
            if self.force_rebuild or not encoder_onnx_path.is_file():
                from ...onnx_export import export_zipdepth_encoder_onnx

                height, width = self._zipdepth_encoder_input_size()
                export_zipdepth_encoder_onnx(
                    torch_model,
                    encoder_onnx_path,
                    height=height,
                    width=width,
                )
            build_migraphx_graph(
                encoder_onnx_path,
                encoder_graph_path,
                fp16=self._preprocessor_dtype == torch.float16,
                fp8=False,
                force=self.force_rebuild,
            )

        self.engine = MIGraphXEngine(encoder_graph_path, device=self.device)
        self.info = replace(
            self.info,
            execution_provider="MIGraphX ROCm encoder + bundled PyTorch decoder",
        )
        if not self._zipdepth_runtime_logged:
            print(
                "[MIGraphX] ZipDepth uses zero-copy MIGraphX ROCm encoder "
                "and bundled PyTorch ROCm decoder",
                flush=True,
            )
            self._zipdepth_runtime_logged = True
        return self.engine

    def load(self) -> MIGraphXEngine:
        if self.engine is not None:
            return self.engine
        if self.graph_path is None:
            raise FileNotFoundError("MIGraphX graph path is required")
        if _is_zipdepth_model(self.model_id):
            return self._load_zipdepth_encoder()
        if self.onnx_path is None and (self.build_graph or self.force_rebuild or not self.graph_path.exists()):
            raise FileNotFoundError("ONNX path is required to build a MIGraphX graph")
        if self.onnx_path is not None and (self.build_graph or self.force_rebuild or not self.graph_path.exists()):
            build_migraphx_graph(
                self.onnx_path,
                self.graph_path,
                fp16=True,
                force=self.force_rebuild,
            )
        if not self.graph_path.exists():
            raise FileNotFoundError(f"MIGraphX graph not found: {self.graph_path}")
        self.engine = MIGraphXEngine(self.graph_path, device=self.device)
        return self.engine

    def predict(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.predict_profile(rgb).depth

    def predict_profile(self, rgb: torch.Tensor) -> DepthProfileResult:
        import time

        if self._zipdepth_fallback is not None:
            return self._zipdepth_fallback.predict_profile(rgb)

        def sync() -> None:
            if self.profile_sync and self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(self.device)

        cuda_events: dict[str, Any] = {}

        def record(name: str) -> None:
            if self.device.type != "cuda":
                return
            try:
                event = torch.cuda.Event(blocking=False, enable_timing=True)
                event.record(torch.cuda.current_stream(self.device))
                cuda_events[name] = event
            except Exception:
                pass

        engine = self.load()
        sync()
        record("depth_pre_start")
        start = time.perf_counter()
        rgb = ensure_bchw(rgb, name="rgb").to(self.device).float().clamp(0, 1)
        _, _, height, width = rgb.shape
        input_size = engine.input_image_size or self.preprocessor.input_size(height, width)
        tensor = self.preprocessor.prepare(rgb, height=input_size[0], width=input_size[1]).contiguous()
        sync()
        record("depth_pre_end")
        preprocess_ms = (time.perf_counter() - start) * 1000.0

        sync()
        record("depth_model_start")
        start = time.perf_counter()
        with torch.inference_mode():
            engine_output = engine(tensor)
            if _is_zipdepth_model(self.model_id):
                if not isinstance(engine_output, tuple) or len(engine_output) != 6:
                    raise RuntimeError(
                        "MIGraphX ZipDepth encoder must return stem features and four stage features"
                    )
                if self._zipdepth_model_provider is None:
                    raise RuntimeError("MIGraphX ZipDepth decoder model was not loaded")
                model = self._zipdepth_model_provider.load()
                predicted = model.decoder(
                    engine_output[0],
                    list(engine_output[2:]),
                    input_size,
                )
            else:
                predicted = engine_output
        sync()
        record("depth_model_end")
        model_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        record("depth_post_start")
        record("depth_norm_start")
        depth = _postprocess_generic_depth(predicted, self.model_id)
        record("depth_norm_end")
        record("depth_upsample_start")
        depth = upsample_depth(
            depth,
            height,
            width,
            rgb=rgb,
            mode=self.depth_upsample,
            edge_strength=self.depth_upsample_edge_strength,
        )
        sync()
        record("depth_upsample_end")
        record("depth_post_end")
        postprocess_ms = (time.perf_counter() - start) * 1000.0
        # GPU timings are resolved by the pipeline after its existing ready
        # event completes. Profiling must not drain capture/presentation work
        # on every stage; normal inference remains ordered on the HIP stream.
        result = DepthProfileResult(
            depth,
            preprocess_ms,
            model_ms,
            postprocess_ms,
            cuda_timing_events=cuda_events,
        )
        if self.model_name.strip().casefold() != "zipdepth-small" or self._zipdepth_parity_checked:
            return result

        fallback = self._zipdepth_model_provider
        if fallback is None:
            raise RuntimeError("MIGraphX ZipDepth reference model was not loaded")
        with torch.inference_mode():
            reference_prediction = fallback._run_depth_model(tensor)
        finite = (
            bool(torch.isfinite(result.depth).all())
            and bool(torch.isfinite(predicted).all())
            and bool(torch.isfinite(reference_prediction).all())
        )
        if predicted.shape == reference_prediction.shape and finite:
            max_abs_diff = float(
                (predicted.float() - reference_prediction.float()).abs().amax().item()
            )
        else:
            max_abs_diff = float("inf")
        if max_abs_diff <= ZIPDEPTH_MIGRAPHX_RAW_MAX_ABS_DIFF:
            self._zipdepth_parity_checked = True
            return result

        reason = (
            "non-finite depth output"
            if not finite
            else f"MIGraphX/PyTorch raw max_abs_diff={max_abs_diff:.6f}"
        )
        if not self.allow_pytorch_fallback:
            raise RuntimeError(f"MIGraphX ZipDepth-Small validation failed: {reason}")

        self._zipdepth_parity_checked = True
        self._zipdepth_fallback = fallback
        depth_backend = (
            "pytorch_rocm"
            if torch.version.hip
            else "pytorch_cuda"
            if self.device.type == "cuda"
            else "pytorch_cpu"
        )
        self.info = replace(
            self.info,
            provider=fallback.info.provider,
            depth_backend=depth_backend,
            runtime=fallback.info.runtime,
            execution_provider="PyTorch fallback",
            fallback_reason=f"MIGraphX ZipDepth-Small validation failed: {reason}",
        )
        print(
            f"[MIGraphX] ZipDepth-Small validation failed ({reason}); "
            "using bundled PyTorch for this session",
            flush=True,
        )
        return fallback.predict_profile(rgb)


def create_migraphx_rocm_provider(
    *,
    model_id: str = DISTILL_ANY_DEPTH_BASE_MODEL_ID,
    model_name: str | None = None,
    device: str | torch.device = "cuda",
    cache_dir: str | Path | None = None,
    onnx_path: str | Path | None = None,
    graph_path: str | Path | None = None,
    build_graph: bool = False,
    force_rebuild: bool = False,
    local_files_only: bool = True,
    force_download: bool = False,
    allow_pytorch_fallback: bool = True,
    depth_resolution: int = 518,
    patch_size: int | None = 14,
    depth_upsample: DepthUpsampleMode = "bilinear",
    depth_upsample_edge_strength: float = 0.35,
    use_torch_compile: bool = False,
):
    reason = None
    if not is_rocm_torch_available():
        reason = "torch.version.hip is not available"
    elif not is_migraphx_available():
        reason = "migraphx is not installed"

    if reason is not None:
        if not allow_pytorch_fallback:
            raise RuntimeError(reason)
        if _is_zipdepth_model(model_id):
            return ZipDepthProvider(
                model_id=model_id,
                model_name=model_name,
                device=device,
                cache_dir=cache_dir,
                depth_resolution=depth_resolution,
                patch_size=32,
                local_files_only=local_files_only,
                force_download=force_download,
                depth_upsample=depth_upsample,
                depth_upsample_edge_strength=depth_upsample_edge_strength,
                use_torch_compile=use_torch_compile,
            )
        provider = create_pytorch_rocm_provider(
            model_id=model_id,
            model_name=model_name,
            device=device,
            cache_dir=cache_dir,
            depth_resolution=depth_resolution,
            patch_size=patch_size,
            local_files_only=local_files_only,
            force_download=force_download,
            depth_upsample=depth_upsample,
            depth_upsample_edge_strength=depth_upsample_edge_strength,
        )
        provider.info = replace(provider.info, fallback_reason=reason)
        return provider

    # Runtime configuration can contain stale/nonexistent explicit paths (for
    # example after selecting a new model or input size). Resolve artifacts in
    # that case before asking the engine to rebuild; a non-None path alone does
    # not mean that the ONNX source exists.
    onnx_missing_for_build = (
        onnx_path is not None
        and not Path(onnx_path).is_file()
        and (build_graph or force_rebuild)
    )
    if onnx_path is None or graph_path is None or onnx_missing_for_build:
        from stereo_runtime.model_artifacts import prepare_model_artifacts

        # ZipDepth interprets the configured resolution as its short edge;
        # other families retain the existing long-edge convention.
        if _is_zipdepth_model(model_id):
            export_height = max(1, int(depth_resolution))
            export_width = max(1, int(round(export_height * 518 / 294)))
        else:
            export_width = max(1, int(depth_resolution))
            export_height = max(1, int(round(export_width * 294 / 518)))
        artifacts = prepare_model_artifacts(
            (
                model_name
                if _is_zipdepth_model(model_id)
                and model_name
                and model_name.casefold() in {"zipdepth-base", "zipdepth-small"}
                else model_id
            ),
            cache_dir=cache_dir or "./models",
            model_dir=Path(onnx_path).parent if onnx_path is not None else None,
            local_files_only=local_files_only,
            force_download=force_download,
            download_if_missing=not local_files_only,
            onnx_dtype="fp16",
            export_onnx_if_missing=True,
            artifact_backend="migraphx",
            export_height=export_height,
            export_width=export_width,
            build_migraphx_if_missing=(build_graph or force_rebuild) and not _is_zipdepth_model(model_id),
            force_rebuild_migraphx=force_rebuild and not _is_zipdepth_model(model_id),
        )
        if onnx_path is None or not Path(onnx_path).is_file():
            onnx_path = artifacts.selected_onnx_path
        graph_path = graph_path or artifacts.selected_migraphx_path or artifacts.paths.migraphx_fp16_path

    return MIGraphXDepthProvider(
        device=device,
        cache_dir=cache_dir,
        onnx_path=onnx_path,
        graph_path=graph_path,
        build_graph=build_graph,
        force_rebuild=force_rebuild,
        depth_upsample=depth_upsample,
        depth_upsample_edge_strength=depth_upsample_edge_strength,
        model_id=model_id,
        model_name=model_name,
        depth_resolution=depth_resolution,
        patch_size=patch_size,
        local_files_only=local_files_only,
        force_download=force_download,
        allow_pytorch_fallback=allow_pytorch_fallback,
        use_torch_compile=use_torch_compile,
    )


__all__ = [
    "MIGraphXDepthProvider",
    "MIGraphXEngine",
    "build_migraphx_graph",
    "create_migraphx_rocm_provider",
    "is_migraphx_available",
]
