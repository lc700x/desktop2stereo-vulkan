from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re

import torch
import torch.nn.functional as F

from .depth_provider import (
    DISTILL_ANY_DEPTH_BASE_MODEL_ID,
    DISTILL_ANY_DEPTH_BASE_NAME,
    DISTILL_ANY_DEPTH_BASE_RESOLUTION,
    DISTILL_ANY_DEPTH_PATCH_SIZE,
    INFINIDEPTH_PATCH_SIZE,
    DepthProfileResult,
    DepthProviderInfo,
    DepthProviderConfig,
    TorchDepthProvider,
    _prepare_accelerated_artifacts,
    _is_infinidepth_model,
    _model_input_size,
    _normalize_depth,
    default_lab_cache_dir,
)
from .depth_upsample import DepthUpsampleMode, upsample_depth
from .output import ensure_b1hw, ensure_bchw, match_depth
from utils.cpu_warnings import describe_tensor, warn_cpu_fallback, warn_cpu_transfer


def default_distill_base_onnx_path(cache_dir: str | Path | None = None) -> Path:
    cache = Path(cache_dir) if cache_dir is not None else default_lab_cache_dir()
    return cache / "models--lc700x--Distill-Any-Depth-Base-hf" / "model_fp16_294x518.onnx"


def default_onnx_path(cache_dir: str | Path | None = None) -> Path:
    return default_distill_base_onnx_path(cache_dir)


def _input_size_from_artifact_name(path: Path) -> tuple[int, int] | None:
    match = re.search(r"_(\d+)x(\d+)(?:\.[^.]+)?$", path.name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _dtype_from_onnx_name(path: Path, fallback: torch.dtype) -> torch.dtype:
    name = path.name.lower()
    if "fp32" in name:
        return torch.float32
    if "fp16" in name:
        return torch.float16
    return fallback


def _preprocess_distill_rgb(rgb: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    rgb = ensure_bchw(rgb, name="rgb").to(device).float().clamp(0, 1)
    _, _, height, width = rgb.shape
    input_h, input_w = _model_input_size(
        height,
        width,
        DISTILL_ANY_DEPTH_BASE_RESOLUTION,
        DISTILL_ANY_DEPTH_PATCH_SIZE,
    )
    tensor = F.interpolate(
        rgb,
        size=(input_h, input_w),
        mode="bicubic" if device.type == "cuda" else "bilinear",
        align_corners=False,
        antialias=True if device.type == "cuda" else False,
    ).to(dtype)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, 3, 1, 1)
    return (tensor - mean) / std


class ModelOnnxPreprocessor:
    def __init__(
        self,
        *,
        model_id: str,
        device: torch.device,
        dtype: torch.dtype,
        fixed_input_size: tuple[int, int] | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.fixed_input_size = fixed_input_size
        self._shape_cache: dict[tuple[int, int], tuple[int, int]] = {}
        self._is_infinidepth = _is_infinidepth_model(model_id)
        self._patch_size = INFINIDEPTH_PATCH_SIZE if self._is_infinidepth else DISTILL_ANY_DEPTH_PATCH_SIZE
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, 3, 1, 1)

    def input_size(self, height: int, width: int) -> tuple[int, int]:
        if self.fixed_input_size is not None:
            return self.fixed_input_size
        key = (height, width)
        cached = self._shape_cache.get(key)
        if cached is not None:
            return cached
        size = _model_input_size(
            height,
            width,
            DISTILL_ANY_DEPTH_BASE_RESOLUTION,
            self._patch_size,
        )
        self._shape_cache[key] = size
        return size

    def __call__(self, rgb: torch.Tensor) -> torch.Tensor:
        rgb = ensure_bchw(rgb, name="rgb").to(self.device).float().clamp(0, 1)
        _, _, height, width = rgb.shape
        input_h, input_w = self.input_size(height, width)
        tensor = F.interpolate(
            rgb,
            size=(input_h, input_w),
            mode="bicubic" if self.device.type == "cuda" else "bilinear",
            align_corners=False,
            antialias=True if self.device.type == "cuda" else False,
        ).to(self.dtype)
        if self._is_infinidepth:
            return tensor
        return (tensor - self._mean) / self._std

    def prepare(self, rgb: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
        """Resize ``rgb`` to an explicit ``(height, width)`` and normalize.

        Same math as ``__call__``, but with the target size fixed by the caller.
        Used when an inference engine fixes the input resolution (e.g. a compiled
        MIGraphX graph), so the resize does not have to follow the model-default
        input size computation.
        """
        rgb = ensure_bchw(rgb, name="rgb").to(self.device).float().clamp(0, 1)
        tensor = F.interpolate(
            rgb,
            size=(height, width),
            mode="bicubic" if self.device.type == "cuda" else "bilinear",
            align_corners=False,
            antialias=True if self.device.type == "cuda" else False,
        ).to(self.dtype)
        if self._is_infinidepth:
            return tensor
        return (tensor - self._mean) / self._std


class DistillPreprocessor(ModelOnnxPreprocessor):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        fixed_input_size: tuple[int, int] | None = None,
    ) -> None:
        super().__init__(
            model_id=DISTILL_ANY_DEPTH_BASE_MODEL_ID,
            device=device,
            dtype=dtype,
            fixed_input_size=fixed_input_size,
        )


class DistillAnyDepthBaseOnnxCuda:
    def __init__(
        self,
        *,
        device: str | torch.device = "cuda",
        cache_dir: str | Path | None = None,
        onnx_path: str | Path | None = None,
        onnx_dtype: str = "auto",
        model_id: str = DISTILL_ANY_DEPTH_BASE_MODEL_ID,
        model_name: str = DISTILL_ANY_DEPTH_BASE_NAME,
        use_iobinding: bool = True,
        use_dlpack: bool = False,
        local_files_only: bool = False,
        force_download: bool = False,
        depth_upsample: DepthUpsampleMode = "bilinear",
        depth_upsample_edge_strength: float = 0.35,
    ) -> None:
        self.device = torch.device(device)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else default_lab_cache_dir()
        self._explicit_onnx_path = Path(onnx_path) if onnx_path is not None else None
        self.onnx_dtype = str(onnx_dtype)
        self.onnx_path = self._explicit_onnx_path or default_onnx_path(self.cache_dir)
        self.dtype = _dtype_from_onnx_name(self.onnx_path, torch.float16 if self.device.type == "cuda" else torch.float32)
        self.model_id = model_id
        self.model_name = model_name
        self.use_iobinding = bool(use_iobinding and self.device.type == "cuda")
        self.use_dlpack = bool(use_dlpack and self.use_iobinding)
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
            load_mode="local_onnx",
            depth_backend="onnx_cuda" if self.device.type == "cuda" else "onnx_cpu",
            runtime="onnxruntime",
            onnx_path=str(self.onnx_path),
            io_binding=self.use_iobinding,
            dlpack=self.use_dlpack,
            output_device="cuda" if self.use_iobinding else "cpu",
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

        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls(directory="")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device.type == "cuda" else ["CPUExecutionProvider"]
        session = ort.InferenceSession(str(self.onnx_path), providers=providers)
        active = session.get_providers()
        if self.device.type == "cuda" and "CUDAExecutionProvider" not in active:
            raise RuntimeError(f"ONNX Runtime CUDA provider unavailable; active providers: {active}")
        self.info = replace(self.info, execution_provider=active[0] if active else None)
        self._session = session
        return session

    def predict(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.predict_profile(rgb).depth

    def predict_profile(self, rgb: torch.Tensor) -> DepthProfileResult:
        import time

        def sync() -> None:
            if self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()

        sync()
        start = time.perf_counter()
        rgb = ensure_bchw(rgb, name="rgb")
        _, _, height, width = rgb.shape
        self._ensure_artifacts_for_input(height, width)
        tensor = self._preprocessor(rgb)
        if self.use_dlpack:
            ort_input = tensor
        else:
            warn_cpu_fallback(
                "ONNX depth input binding",
                "dlpack_disabled",
                detail="using tensor.detach().cpu().numpy() for ORT input",
                key="depth_onnx_input_dlpack_disabled",
            )
            warn_cpu_transfer(
                "ONNX depth input tensor",
                ".cpu().numpy()",
                detail=describe_tensor(tensor),
                key="depth_onnx_input_cpu_transfer",
            )
            ort_input = tensor.detach().cpu().numpy()
        sync()
        preprocess_ms = (time.perf_counter() - start) * 1000.0

        session = self.load()
        sync()
        start = time.perf_counter()
        if self.use_iobinding:
            predicted = self._run_iobinding(session, ort_input)
        else:
            predicted = session.run(["predicted_depth"], {"pixel_values": ort_input})[0]
            predicted = torch.from_numpy(predicted)
        sync()
        model_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        depth = predicted.float()
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

    def _run_iobinding(self, session, ort_input) -> torch.Tensor:
        import numpy as np
        import onnxruntime as ort
        import torch

        if isinstance(ort_input, torch.Tensor):
            input_ort = ort.OrtValue.from_dlpack(torch.utils.dlpack.to_dlpack(ort_input.contiguous()))
        else:
            input_ort = ort.OrtValue.ortvalue_from_numpy(ort_input, "cuda", 0)
        io_binding = session.io_binding()
        io_binding.bind_ortvalue_input("pixel_values", input_ort)
        io_binding.bind_output("predicted_depth", "cuda")
        session.run_with_iobinding(io_binding)
        output_ort = io_binding.get_outputs()[0]
        if isinstance(ort_input, torch.Tensor):
            return torch.utils.dlpack.from_dlpack(output_ort)
        output_np = output_ort.numpy()
        if not isinstance(output_np, np.ndarray):
            output_np = np.asarray(output_np)
        return torch.from_numpy(output_np)


def estimate_distill_any_depth_base_518_nvidia(
    rgb: torch.Tensor,
    *,
    device: str | torch.device = "cuda",
    cache_dir: str | Path | None = None,
    onnx_path: str | Path | None = None,
    prefer_onnx: bool = True,
    use_iobinding: bool = True,
    allow_pytorch_fallback: bool = True,
    local_files_only: bool = False,
    force_download: bool = False,
) -> tuple[torch.Tensor, DepthProviderInfo]:
    from .providers.nvidia.tensorrt_ort import estimate_depth_nvidia_chain

    return estimate_depth_nvidia_chain(
        rgb,
        device=device,
        cache_dir=cache_dir,
        onnx_path=onnx_path,
        prefer_tensorrt=False,
        prefer_onnx=prefer_onnx,
        allow_pytorch_fallback=allow_pytorch_fallback,
        local_files_only=local_files_only,
        force_download=force_download,
    )


def estimate_depth_onnx_cuda(
    rgb: torch.Tensor,
    *,
    device: str | torch.device = "cuda",
    cache_dir: str | Path | None = None,
    onnx_path: str | Path | None = None,
    use_iobinding: bool = True,
    allow_pytorch_fallback: bool = True,
    local_files_only: bool = False,
    force_download: bool = False,
) -> tuple[torch.Tensor, DepthProviderInfo]:
    fallback_reason = None
    try:
        provider = OnnxCudaDepthProvider(
            device=device,
            cache_dir=cache_dir,
            onnx_path=onnx_path,
            use_iobinding=use_iobinding,
            use_dlpack=False,
            local_files_only=local_files_only,
            force_download=force_download,
        )
        return provider.predict(rgb), provider.info
    except Exception as exc:
        fallback_reason = f"{type(exc).__name__}: {exc}"
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
        fallback_reason=fallback_reason,
        onnx_path=str(onnx_path or default_onnx_path(cache_dir)),
    )
    return depth, info


OnnxCudaDepthProvider = DistillAnyDepthBaseOnnxCuda
estimate_distill_any_depth_base_518_onnx_cuda = estimate_depth_onnx_cuda
