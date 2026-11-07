from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import sys

import torch

from ...depth_onnx_provider import (
    ModelOnnxPreprocessor,
    _dtype_from_onnx_name,
    _input_size_from_artifact_name,
    default_onnx_path,
)
from ...depth_provider import (
    DISTILL_ANY_DEPTH_BASE_MODEL_ID,
    DISTILL_ANY_DEPTH_BASE_NAME,
    DISTILL_ANY_DEPTH_BASE_RESOLUTION,
    DepthProfileResult,
    DepthProviderConfig,
    DepthProviderInfo,
    TorchDepthProvider,
    _prepare_accelerated_artifacts,
    _normalize_depth,
    default_lab_cache_dir,
)
from ...depth_upsample import DepthUpsampleMode, upsample_depth
from ...output import ensure_b1hw, ensure_bchw, match_depth
from .onnx_cuda import OnnxCudaDepthProvider
from runtime_paths import python_site_packages
from utils.cpu_warnings import describe_tensor, warn_cpu_fallback, warn_cpu_transfer


def default_distill_base_trt_cache_dir(cache_dir: str | Path | None = None) -> Path:
    cache = Path(cache_dir) if cache_dir is not None else default_lab_cache_dir()
    return cache / "models--lc700x--Distill-Any-Depth-Base-hf" / "trt_cache"


def default_tensorrt_cache_dir(cache_dir: str | Path | None = None) -> Path:
    return default_distill_base_trt_cache_dir(cache_dir)


def candidate_tensorrt_lib_dirs() -> list[Path]:
    candidates = [
        Path(sys.prefix) / "Lib" / "site-packages" / "tensorrt_libs",
        Path(sys.executable).resolve().parent / "Lib" / "site-packages" / "tensorrt_libs",
        python_site_packages() / "tensorrt_libs",
    ]
    env_value = os.environ.get("STEREO_LAB_TENSORRT_LIBS")
    if env_value:
        candidates.insert(0, Path(env_value))

    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _dtype_label(dtype: torch.dtype) -> str:
    return "fp16" if dtype == torch.float16 else "fp32" if dtype == torch.float32 else str(dtype).replace("torch.", "")


def ensure_tensorrt_dll_path() -> list[str]:
    added: list[str] = []
    for path in candidate_tensorrt_lib_dirs():
        if not (path / "nvinfer_10.dll").exists():
            continue
        path_str = str(path)
        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        if path_str not in path_entries:
            os.environ["PATH"] = path_str + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(path_str)
            except OSError:
                pass
        added.append(path_str)
    return added


class DistillAnyDepthBaseTensorRtOrt:
    def __init__(
        self,
        *,
        device: str | torch.device = "cuda",
        cache_dir: str | Path | None = None,
        onnx_path: str | Path | None = None,
        onnx_dtype: str = "auto",
        trt_cache_dir: str | Path | None = None,
        model_id: str = DISTILL_ANY_DEPTH_BASE_MODEL_ID,
        model_name: str = DISTILL_ANY_DEPTH_BASE_NAME,
        local_files_only: bool = False,
        force_download: bool = False,
        depth_upsample: DepthUpsampleMode = "bilinear",
        depth_upsample_edge_strength: float = 0.35,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise RuntimeError("TensorRT depth provider requires CUDA")
        self.cache_dir = Path(cache_dir) if cache_dir is not None else default_lab_cache_dir()
        self._explicit_onnx_path = Path(onnx_path) if onnx_path is not None else None
        self.onnx_dtype = str(onnx_dtype)
        self.onnx_path = self._explicit_onnx_path or default_onnx_path(self.cache_dir)
        self.trt_cache_dir = Path(trt_cache_dir) if trt_cache_dir is not None else default_tensorrt_cache_dir(self.cache_dir)
        self.dtype = _dtype_from_onnx_name(self.onnx_path, torch.float16)
        self.model_id = model_id
        self.model_name = model_name
        self.local_files_only = bool(local_files_only)
        self.force_download = bool(force_download)
        self.depth_upsample = depth_upsample
        self.depth_upsample_edge_strength = float(depth_upsample_edge_strength)
        self.info = DepthProviderInfo(
            provider="onnxruntime.InferenceSession",
            model_name=self.model_name,
            model_id=self.model_id,
            depth_resolution=DISTILL_ANY_DEPTH_BASE_RESOLUTION,
            cache_dir=str(self.cache_dir),
            load_mode="local_onnx_tensorrt_ep",
            depth_backend="tensorrt",
            runtime="onnxruntime-tensorrt",
            onnx_path=str(self.onnx_path),
            io_binding=True,
            output_device="cuda",
        )
        self._session = None
        self._artifact_input_size: tuple[int, int] | None = None
        self._preprocessor = ModelOnnxPreprocessor(
            model_id=self.model_id,
            device=self.device,
            dtype=self.dtype,
            fixed_input_size=_input_size_from_artifact_name(self.onnx_path) if self._explicit_onnx_path else None,
        )

    def _set_onnx_path(self, path: Path, input_size: tuple[int, int]) -> None:
        dtype = _dtype_from_onnx_name(path, self.dtype)
        if path != self.onnx_path or dtype != self.dtype:
            self._session = None
        self.onnx_path = path
        self.dtype = dtype
        self._artifact_input_size = input_size
        self._preprocessor = ModelOnnxPreprocessor(
            model_id=self.model_id,
            device=self.device,
            dtype=self.dtype,
            fixed_input_size=input_size,
        )
        self.info = replace(self.info, onnx_path=str(self.onnx_path))

    def _ensure_artifacts_for_input(self, height: int, width: int) -> None:
        input_size = self._preprocessor.input_size(height, width)
        if self._explicit_onnx_path is not None:
            fixed_size = _input_size_from_artifact_name(self._explicit_onnx_path) or input_size
            if self._artifact_input_size != fixed_size:
                self._set_onnx_path(self._explicit_onnx_path, fixed_size)
            return
        if self._artifact_input_size == input_size and self.onnx_path.exists():
            return
        cfg = DepthProviderConfig(
            model_id=self.model_id,
            model_name=self.model_name,
            device=self.device,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            force_download=self.force_download,
            onnx_dtype=self.onnx_dtype,
        )
        artifacts = _prepare_accelerated_artifacts(cfg, input_size=input_size)
        if artifacts.selected_onnx_path is None:
            raise FileNotFoundError(f"ONNX artifact not found for {self.model_id} at input size {input_size}")
        self._set_onnx_path(Path(artifacts.selected_onnx_path), input_size)

    def load(self):
        if self._session is not None:
            return self._session
        if self._explicit_onnx_path is None and self._artifact_input_size is None:
            return None
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX file not found: {self.onnx_path}")

        trt_lib_dirs = ensure_tensorrt_dll_path()
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls(directory="")
        available = ort.get_available_providers()
        if "TensorrtExecutionProvider" not in available:
            raise RuntimeError(f"TensorRTExecutionProvider unavailable; available providers: {available}")

        self.trt_cache_dir.mkdir(parents=True, exist_ok=True)
        trt_options = {
            "trt_fp16_enable": self.dtype == torch.float16,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(self.trt_cache_dir),
        }
        session = ort.InferenceSession(
            str(self.onnx_path),
            providers=[
                ("TensorrtExecutionProvider", trt_options),
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )
        active = session.get_providers()
        if "TensorrtExecutionProvider" not in active:
            raise RuntimeError(f"TensorRT provider did not activate; active providers: {active}")
        print(
            "[TensorRT] ONNXRuntime TensorRT EP loaded:"
            f" onnx={self.onnx_path}"
            f" dtype={_dtype_label(self.dtype)}"
            f" trt_cache={self.trt_cache_dir}"
            f" active_providers={active}"
            f" available_providers={available}"
            f" dll_dirs={trt_lib_dirs or 'none'}",
            flush=True,
        )
        self.info = replace(self.info, execution_provider=active[0] if active else None, trt_lib_dirs=trt_lib_dirs)
        self._session = session
        return session

    def predict(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.predict_profile(rgb).depth

    def predict_profile(self, rgb: torch.Tensor) -> DepthProfileResult:
        import time

        def sync() -> None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        sync()
        start = time.perf_counter()
        rgb = ensure_bchw(rgb, name="rgb")
        _, _, height, width = rgb.shape
        self._ensure_artifacts_for_input(height, width)
        tensor = self._preprocessor(rgb)
        warn_cpu_fallback(
            "TensorRT ORT depth input binding",
            "numpy_input_required",
            detail="using tensor.detach().cpu().numpy() before OrtValue.ortvalue_from_numpy",
            key="tensorrt_ort_input_numpy_required",
        )
        warn_cpu_transfer(
            "TensorRT ORT depth input tensor",
            ".cpu().numpy()",
            detail=describe_tensor(tensor),
            key="tensorrt_ort_input_cpu_transfer",
        )
        ort_input = tensor.detach().cpu().numpy()
        sync()
        preprocess_ms = (time.perf_counter() - start) * 1000.0

        session = self.load()

        import numpy as np
        import onnxruntime as ort

        sync()
        start = time.perf_counter()
        input_ort = ort.OrtValue.ortvalue_from_numpy(ort_input, "cuda", 0)
        io_binding = session.io_binding()
        io_binding.bind_ortvalue_input("pixel_values", input_ort)
        io_binding.bind_output("predicted_depth", "cuda")
        session.run_with_iobinding(io_binding)
        output_ort = io_binding.get_outputs()[0]
        warn_cpu_transfer(
            "TensorRT ORT depth output",
            "OrtValue.numpy()",
            detail="output becomes CPU numpy before torch.from_numpy",
            key="tensorrt_ort_output_numpy_transfer",
        )
        output_np = output_ort.numpy()
        if not isinstance(output_np, np.ndarray):
            warn_cpu_transfer(
                "TensorRT ORT depth output",
                "np.asarray()",
                detail=f"type={type(output_np).__name__}",
                key="tensorrt_ort_output_asarray_transfer",
            )
            output_np = np.asarray(output_np)
        sync()
        model_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        depth = torch.from_numpy(output_np).float()
        depth = ensure_b1hw(depth)
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


TensorRtOrtDepthProvider = DistillAnyDepthBaseTensorRtOrt


def estimate_depth_nvidia_chain(
    rgb: torch.Tensor,
    *,
    device: str | torch.device = "cuda",
    cache_dir: str | Path | None = None,
    onnx_path: str | Path | None = None,
    trt_cache_dir: str | Path | None = None,
    prefer_tensorrt: bool = True,
    prefer_onnx: bool = True,
    allow_pytorch_fallback: bool = True,
    require_tensorrt: bool = False,
    local_files_only: bool = False,
    force_download: bool = False,
) -> tuple[torch.Tensor, DepthProviderInfo]:
    fallback_reasons: list[str] = []
    if prefer_tensorrt:
        try:
            provider = TensorRtOrtDepthProvider(
                device=device,
                cache_dir=cache_dir,
                onnx_path=onnx_path,
                trt_cache_dir=trt_cache_dir,
                local_files_only=local_files_only,
                force_download=force_download,
            )
            return provider.predict(rgb), provider.info
        except Exception as exc:
            reason = f"tensorrt:{type(exc).__name__}: {exc}"
            fallback_reasons.append(reason)
            if require_tensorrt:
                raise RuntimeError(reason) from exc

    if prefer_onnx:
        try:
            provider = OnnxCudaDepthProvider(
                device=device,
                cache_dir=cache_dir,
                onnx_path=onnx_path,
                use_iobinding=True,
                local_files_only=local_files_only,
                force_download=force_download,
            )
            depth = provider.predict(rgb)
            info = replace(provider.info, fallback_reason="; ".join(fallback_reasons) or None)
            return depth, info
        except Exception as exc:
            fallback_reasons.append(f"onnx_cuda_iobinding:{type(exc).__name__}: {exc}")
            if not allow_pytorch_fallback:
                raise

    provider = TorchDepthProvider(
        device=device,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        force_download=force_download,
    )
    depth = provider.predict(rgb)
    info = replace(
        provider.info,
        fallback_reason="; ".join(fallback_reasons) or None,
        onnx_path=str(onnx_path or default_onnx_path(cache_dir)),
    )
    return depth, info


estimate_distill_any_depth_base_518_nvidia = estimate_depth_nvidia_chain
