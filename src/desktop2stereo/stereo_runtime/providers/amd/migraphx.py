from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from stereo_runtime.depth_onnx_provider import DistillPreprocessor, _dtype_from_onnx_name
from stereo_runtime.depth_provider import (
    DISTILL_ANY_DEPTH_BASE_MODEL_ID,
    DepthProfileResult,
    DepthProviderInfo,
    _normalize_depth,
)
from stereo_runtime.depth_upsample import DepthUpsampleMode, upsample_depth
from stereo_runtime.output import ensure_b1hw, ensure_bchw

from .pytorch_rocm import create_pytorch_rocm_provider, is_rocm_torch_available


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
    prog = mx.parse_onnx(str(onnx_path))
    if fp16 and not force_fp32:
        if fp8 and hasattr(mx, "autocast_fp8"):
            try:
                mx.autocast_fp8(prog)
            except Exception:
                mx.quantize_fp16(prog)
        else:
            mx.quantize_fp16(prog)
    prog.compile(mx.get_target("gpu"), offload_copy=False)
    mx.save(prog, str(graph_path))
    return graph_path


class MIGraphXEngine:
    def __init__(self, graph_path: str | Path, *, device: str | torch.device = "cuda") -> None:
        import migraphx as mx

        self.mx = mx
        self.graph_path = Path(graph_path)
        self.device = torch.device(device)
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
        self._out_name = next(iter(self._out_params))
        self._out_lens = list(self._out_params[self._out_name].lens())
        mx_to_torch = {1: torch.float16, 2: torch.float32, 3: torch.float64}
        self._mgx_in_dtype = mx_to_torch.get(int(self._in_shape.type()), torch.float32)
        self._mgx_out_dtype = mx_to_torch.get(int(self._out_params[self._out_name].type()), torch.float32)

    @property
    def input_shape(self) -> tuple[int, ...]:
        return tuple(int(dim) for dim in self._in_shape.lens())

    @property
    def input_image_size(self) -> tuple[int, int] | None:
        shape = self.input_shape
        if len(shape) != 4 or any(dim < 1 for dim in shape):
            return None
        return int(shape[-2]), int(shape[-1])

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.to(device=self.device)
        if tensor.dtype != self._mgx_in_dtype or not tensor.is_contiguous():
            tensor = tensor.contiguous().to(dtype=self._mgx_in_dtype)
        in_arg = self.mx.argument_from_pointer(self._in_shape, tensor.data_ptr())
        out = torch.empty(tuple(self._out_lens), dtype=self._mgx_out_dtype, device=self.device)
        out_arg = self.mx.argument_from_pointer(self._out_params[self._out_name], out.data_ptr())
        stream = torch.cuda.current_stream(self.device)
        self.prog.run_async({self.input_name: in_arg, self._out_name: out_arg}, stream.cuda_stream, "ihipStream_t")
        if not self._logged_zero_copy:
            print(
                f"[MIGraphX] Zero-copy GPU path active | input={tuple(tensor.shape)} {tensor.dtype} "
                f"-> output={tuple(out.shape)} {out.dtype}"
            )
            self._logged_zero_copy = True
        return out


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
    ) -> None:
        self.device = torch.device(device)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.onnx_path = Path(onnx_path) if onnx_path is not None else None
        self.graph_path = Path(graph_path) if graph_path is not None else None
        self.build_graph = bool(build_graph)
        self.force_rebuild = bool(force_rebuild)
        self.depth_upsample = depth_upsample
        self.depth_upsample_edge_strength = float(depth_upsample_edge_strength)
        # MIGraphX graphs are built fp16 by default (build_migraphx_graph quantizes
        # to fp16 unless force_fp32), so the preprocessor must emit fp16 tensors to
        # feed the engine without a dtype cast on the hot path.
        self._preprocessor_dtype = _dtype_from_onnx_name(self.onnx_path, torch.float16)
        self.preprocessor = DistillPreprocessor(device=self.device, dtype=self._preprocessor_dtype)
        self.engine: MIGraphXEngine | None = None
        self.info = DepthProviderInfo(
            provider="MIGraphX",
            model_name="Distill-Any-Depth-Base",
            model_id=DISTILL_ANY_DEPTH_BASE_MODEL_ID,
            depth_resolution=518,
            cache_dir=str(self.cache_dir or ""),
            load_mode="local_files_only",
            depth_backend="migraphx_rocm",
            runtime="migraphx",
            onnx_path=str(self.onnx_path) if self.onnx_path else None,
            execution_provider="MIGraphX ROCm",
            output_device=str(self.device),
        )

    def load(self) -> MIGraphXEngine:
        if self.engine is not None:
            return self.engine
        if self.graph_path is None:
            raise FileNotFoundError("MIGraphX graph path is required")
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

        def sync() -> None:
            if self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()

        sync()
        start = time.perf_counter()
        rgb = ensure_bchw(rgb, name="rgb").to(self.device).float().clamp(0, 1)
        _, _, height, width = rgb.shape
        engine = self.load()
        input_size = engine.input_image_size or (294, 518)
        tensor = self.preprocessor.prepare(rgb, height=input_size[0], width=input_size[1]).contiguous()
        sync()
        preprocess_ms = (time.perf_counter() - start) * 1000.0

        sync()
        start = time.perf_counter()
        with torch.inference_mode():
            predicted = engine(tensor)
        sync()
        model_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        depth = ensure_b1hw(predicted)
        depth = _normalize_depth(depth)
        depth = upsample_depth(
            depth,
            height,
            width,
            rgb=rgb,
            mode=self.depth_upsample,
            edge_strength=self.depth_upsample_edge_strength,
        )
        sync()
        postprocess_ms = (time.perf_counter() - start) * 1000.0
        return DepthProfileResult(depth, preprocess_ms, model_ms, postprocess_ms)


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
):
    reason = None
    if not is_rocm_torch_available():
        reason = "torch.version.hip is not available"
    elif not is_migraphx_available():
        reason = "migraphx is not installed"
    elif model_id != DISTILL_ANY_DEPTH_BASE_MODEL_ID:
        reason = "MIGraphX provider currently supports Distill-Any-Depth-Base only"

    if reason is not None:
        if not allow_pytorch_fallback:
            raise RuntimeError(reason)
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

    if onnx_path is None or graph_path is None:
        from stereo_runtime.model_artifacts import prepare_model_artifacts

        artifacts = prepare_model_artifacts(
            model_id,
            cache_dir=cache_dir or "./models",
            model_dir=Path(onnx_path).parent if onnx_path is not None else None,
            local_files_only=local_files_only,
            force_download=force_download,
            download_if_missing=not local_files_only,
            onnx_dtype="fp16",
            export_onnx_if_missing=True,
            artifact_backend="migraphx",
            build_migraphx_if_missing=build_graph or force_rebuild,
            force_rebuild_migraphx=force_rebuild,
        )
        onnx_path = onnx_path or artifacts.selected_onnx_path
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
    )


__all__ = [
    "MIGraphXDepthProvider",
    "MIGraphXEngine",
    "build_migraphx_graph",
    "create_migraphx_rocm_provider",
    "is_migraphx_available",
]
