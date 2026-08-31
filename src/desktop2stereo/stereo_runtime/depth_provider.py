from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field, replace
import importlib
import importlib.util
import os
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from .depth_upsample import DepthUpsampleMode, upsample_depth
from .output import ensure_bchw, ensure_b1hw, match_depth
from .progress import DownloadProgress, progress_write, status_write
from utils.network import huggingface_endpoint_candidates

DISTILL_ANY_DEPTH_BASE_NAME = "Distill-Any-Depth-Base"
DISTILL_ANY_DEPTH_BASE_MODEL_ID = "lc700x/Distill-Any-Depth-Base-hf"
DISTILL_ANY_DEPTH_BASE_RESOLUTION = 518
DISTILL_ANY_DEPTH_PATCH_SIZE = 14
INFINIDEPTH_PATCH_SIZE = 16
HF_ENDPOINT_DEFAULTS = ("https://hf-mirror.com", "https://huggingface.co")
HF_DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Accept": "*/*",
    "Connection": "close",
}
INFINIDEPTH_ENCODERS = {
    "lc700x/infinidepth-small": "vits16",
    "lc700x/infinidepth-smallplus": "vits16plus",
    "lc700x/infinidepth-base": "vitb16",
    "lc700x/infinidepth-large": "vitl16",
}
_RESOLVED_HF_MODEL_FILES: dict[tuple[str, str], str] = {}
_MODEL_WEIGHT_FILENAMES = ("model.safetensors", "model.pt", "model.ckpt")

def _onnxruntime_available() -> bool:
    return importlib.util.find_spec("onnxruntime") is not None


@dataclass(frozen=True)
class DepthProviderInfo:
    provider: str
    model_name: str
    model_id: str
    depth_resolution: int
    cache_dir: str
    load_mode: str = "online"
    depth_backend: str = "pytorch_cuda"
    runtime: str = "transformers"
    onnx_path: str | None = None
    execution_provider: str | None = None
    fallback_reason: str | None = None
    io_binding: bool = False
    dlpack: bool = False
    output_device: str | None = None
    trt_lib_dirs: list[str] | None = None

    def to_report(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DepthProfileResult:
    depth: torch.Tensor
    preprocess_ms: float
    model_ms: float
    postprocess_ms: float
    cuda_timing_events: dict[str, Any] = field(default_factory=dict)
    execution_slot: int | None = None
    execution_slot_count: int = 1
    slot_wait_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.model_ms + self.postprocess_ms

    def to_report(self) -> dict[str, float]:
        return {
            "preprocess_ms": float(self.preprocess_ms),
            "model_ms": float(self.model_ms),
            "postprocess_ms": float(self.postprocess_ms),
            "slot_wait_ms": float(self.slot_wait_ms),
            "total_ms": float(self.total_ms),
        }


@dataclass(frozen=True)
class DepthProviderConfig:
    backend: str = "distill_base_nvidia"
    model_id: str = DISTILL_ANY_DEPTH_BASE_MODEL_ID
    model_name: str = DISTILL_ANY_DEPTH_BASE_NAME
    depth_resolution: int = DISTILL_ANY_DEPTH_BASE_RESOLUTION
    patch_size: int = DISTILL_ANY_DEPTH_PATCH_SIZE
    device: str | torch.device = "cuda"
    cache_dir: str | Path | None = None
    onnx_path: str | Path | None = None
    onnx_dtype: str = "auto"
    trt_cache_dir: str | Path | None = None
    engine_path: str | Path | None = None
    local_files_only: bool = True
    force_download: bool = False
    prefer_tensorrt: bool = True
    prefer_native_tensorrt: bool = False
    prefer_onnx: bool = True
    allow_pytorch_fallback: bool = True
    require_tensorrt: bool = False
    use_iobinding: bool = True
    use_dlpack: bool = False
    build_engine: bool = False
    force_rebuild: bool = False
    use_cuda_graph: bool = False
    profile_sync: bool = False
    execution_slot_count: int = 1
    depth_upsample: DepthUpsampleMode = "bilinear"
    depth_upsample_edge_strength: float = 0.35
    use_coreml: bool = False
    recompile_coreml: bool = False


def default_lab_cache_dir() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "models"


def distill_base_518_info(cache_dir: str | Path | None = None) -> DepthProviderInfo:
    cache = Path(cache_dir) if cache_dir is not None else default_lab_cache_dir()
    return DepthProviderInfo(
        provider="transformers.AutoModelForDepthEstimation",
        model_name=DISTILL_ANY_DEPTH_BASE_NAME,
        model_id=DISTILL_ANY_DEPTH_BASE_MODEL_ID,
        depth_resolution=DISTILL_ANY_DEPTH_BASE_RESOLUTION,
        cache_dir=str(cache),
    )


def _nearest_multiple(value: int, patch: int) -> int:
    down = (value // patch) * patch
    up = down + patch
    return max(1, up if abs(up - value) <= abs(value - down) else down)


def _model_input_size(height: int, width: int, target: int, patch: int) -> tuple[int, int]:
    longest = max(height, width)
    scale = target / float(longest) if longest != target else 1.0
    resized_h = max(1, int(round(height * scale)))
    resized_w = max(1, int(round(width * scale)))
    return _nearest_multiple(resized_h, patch), _nearest_multiple(resized_w, patch)


def _normalize_depth(depth: torch.Tensor, subsample_cap: int = 6_144) -> torch.Tensor:
    depth = ensure_b1hw(depth).float()
    flat = depth.flatten(start_dim=2)
    count = flat.shape[-1]
    if count <= 1:
        amin = flat.amin(dim=-1).view(depth.shape[0], 1, 1, 1)
        amax = flat.amax(dim=-1).view(depth.shape[0], 1, 1, 1)
    else:
        sampled = flat
        if count > subsample_cap:
            step = (count + subsample_cap - 1) // subsample_cap
            sampled = flat[..., ::step]
        sample_count = sampled.shape[-1]
        lo_idx = min(sample_count - 1, max(0, int(round(0.02 * (sample_count - 1)))))
        hi_idx = min(sample_count - 1, max(0, int(round(0.98 * (sample_count - 1)))))
        sorted_vals = torch.sort(sampled, dim=-1).values
        amin = sorted_vals[..., lo_idx].view(depth.shape[0], 1, 1, 1)
        amax = sorted_vals[..., hi_idx].view(depth.shape[0], 1, 1, 1)
    normalized = (depth - amin) / (amax - amin).clamp_min(1e-6)
    if normalized.is_floating_point():
        # fp16 depth sources can poison normalization with NaN/Inf; sanitize
        # at this choke point so every downstream consumer (synthesis, warp,
        # host-frame pack) only ever sees finite [0,1] depth.
        normalized = torch.nan_to_num(normalized, nan=1.0, posinf=1.0, neginf=0.0)
    return normalized.clamp(0, 1)


def _is_infinidepth_model(model_id: str) -> bool:
    return "infinidepth" in str(model_id).lower()


def _infinidepth_encoder_for_model(model_id: str) -> str:
    return INFINIDEPTH_ENCODERS.get(str(model_id).strip().lower(), "vitl16")


def _find_local_model_weight(model_dir: str | Path) -> str | None:
    root = Path(model_dir)
    if not root.exists():
        return None
    for filename in _MODEL_WEIGHT_FILENAMES:
        direct = root / filename
        if _is_valid_model_weight_file(direct):
            return str(direct)
        for path in root.rglob(filename):
            if _is_valid_model_weight_file(path):
                return str(path)
    return None


def _is_valid_model_weight_file(path: str | Path) -> bool:
    try:
        p = Path(path)
        return p.is_file() and p.stat().st_size > 0
    except OSError:
        return False


def _remove_invalid_model_weight(path: str | Path) -> None:
    try:
        p = Path(path)
        if p.exists() and p.stat().st_size <= 0:
            p.unlink()
    except OSError:
        pass


def _download_hf_file_direct(model_id: str, filename: str, cache_dir: str | Path, endpoint: str) -> str:
    import requests
    from .model_registry import resolve_model_dir

    target_dir = resolve_model_dir(model_id, cache_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / filename
    tmp = target.with_suffix(target.suffix + ".tmp")
    url = _hf_resolve_url(endpoint, model_id, filename)
    _progress_print(f"[Main] Direct model download fallback: {url} -> {target}")
    response = requests.get(url, headers=HF_DOWNLOAD_HEADERS, stream=True, timeout=30)
    response.raise_for_status()
    with open(tmp, "wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    if not _is_valid_model_weight_file(tmp):
        _remove_invalid_model_weight(tmp)
        raise FileNotFoundError(f"direct download produced empty model weight: {target}")
    os.replace(tmp, target)
    return str(target)


def _progress_print(message):
    progress_write(str(message))


@contextmanager
def _hf_download_progress_patch():
    originals = []
    try:
        for module_name, attr in (
            ("huggingface_hub.utils.tqdm", "tqdm"),
            ("huggingface_hub.file_download", "tqdm"),
            ("huggingface_hub._snapshot_download", "hf_tqdm"),
        ):
            module = importlib.import_module(module_name)
            originals.append((module, attr, getattr(module, attr, None)))
            setattr(module, attr, DownloadProgress)
        yield
    finally:
        for module, attr, original in reversed(originals):
            if original is not None:
                setattr(module, attr, original)


@contextmanager
def _hf_endpoint(endpoint: str):
    previous = os.environ.get("HF_ENDPOINT")
    os.environ["HF_ENDPOINT"] = endpoint
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("HF_ENDPOINT", None)
        else:
            os.environ["HF_ENDPOINT"] = previous


def _hf_endpoint_candidates() -> tuple[str, ...]:
    endpoint = os.environ.get("HF_ENDPOINT")
    if endpoint:
        if endpoint in HF_ENDPOINT_DEFAULTS:
            return (endpoint,) + tuple(candidate for candidate in HF_ENDPOINT_DEFAULTS if candidate != endpoint)
        return (endpoint,)
    return huggingface_endpoint_candidates()


def _hf_resolve_url(endpoint: str, model_id: str, filename: str = "") -> str:
    base = endpoint.rstrip("/")
    if filename:
        return f"{base}/{model_id}/resolve/main/{filename}"
    return f"{base}/{model_id}"


def _probe_download_url(url: str, opener: Callable[..., Any] | None = None) -> bool:
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    open_url = opener or urlopen
    for method in ("HEAD", "GET"):
        headers = dict(HF_DOWNLOAD_HEADERS)
        if method == "GET":
            headers["Range"] = "bytes=0-0"
        try:
            request = Request(url, headers=headers, method=method)
            with open_url(request, timeout=10) as response:
                final_url = getattr(response, "url", None) or response.geturl()
                status = getattr(response, "status", None) or response.getcode()
                length = response.headers.get("Content-Length", "unknown")
                ctype = response.headers.get("Content-Type", "unknown")
                _progress_print(
                    f"[Main] Download probe {method}: HTTP {status}, "
                    f"size={length}, type={ctype}, final={final_url}"
                )
                return True
        except HTTPError as exc:
            _progress_print(f"[Main] Download probe {method} failed: HTTP {exc.code} {exc.reason}, url={exc.geturl()}")
        except Exception as exc:
            _progress_print(f"[Main] Download probe {method} failed: {type(exc).__name__}: {exc}")
    return False


def _reachable_hf_endpoints(model_id: str) -> tuple[str, ...]:
    reachable = []
    for endpoint in _hf_endpoint_candidates():
        model_url = _hf_resolve_url(endpoint, model_id)
        _progress_print(f"[Main] Checking model download endpoint: {model_url}")
        if _probe_download_url(model_url):
            reachable.append(endpoint)
    if not reachable:
        candidates = ", ".join(_hf_endpoint_candidates())
        raise RuntimeError(
            f"unable to access model download endpoints ({candidates}). "
            "Please check your network or enable VPN, then retry."
        )
    return tuple(reachable)


def _load_hf_with_endpoint_fallback(load_fn: Callable[[str], Any], model_id: str):
    last_error = None
    for endpoint in _reachable_hf_endpoints(model_id):
        model_url = _hf_resolve_url(endpoint, model_id)
        _progress_print(f"[Main] Loading depth model from {endpoint}: {model_id}")
        _progress_print(f"[Main] Model download URL: {model_url}")
        _probe_download_url(model_url)
        try:
            with _hf_endpoint(endpoint), _hf_download_progress_patch():
                return load_fn(model_id)
        except Exception as exc:
            last_error = exc
            _progress_print(f"[Main] Depth model load failed from {endpoint}: {type(exc).__name__}: {exc}")
    raise last_error


def _print_download_preparing_progress(filename):
    _progress_print(f"[Main] Preparing download {filename}: waiting for server response...")


def _raise_model_resolution_error(model_id: str, last_error: Exception | None, *, local_only: bool) -> None:
    scope = "local InfiniDepth weights" if local_only else "InfiniDepth weights"
    if last_error is None:
        raise RuntimeError(f"unable to resolve {scope} for {model_id!r}")
    raise RuntimeError(
        f"unable to resolve {scope} for {model_id!r}: "
        f"{type(last_error).__name__}: {last_error}"
    ) from last_error


def _resolve_hf_model_file(
    model_id: str,
    cache_dir: str | Path,
    *,
    local_files_only: bool = False,
    force_download: bool = False,
) -> str:
    from huggingface_hub import hf_hub_download

    filenames = _MODEL_WEIGHT_FILENAMES
    cache_dir_text = str(cache_dir)
    cache_key = (str(model_id), str(Path(cache_dir).resolve()))
    if not force_download:
        cached = _RESOLVED_HF_MODEL_FILES.get(cache_key)
        if cached and _is_valid_model_weight_file(cached):
            return cached
        if cached:
            _remove_invalid_model_weight(cached)
            _RESOLVED_HF_MODEL_FILES.pop(cache_key, None)
        from .model_registry import resolve_model_dir

        local_path = _find_local_model_weight(resolve_model_dir(model_id, cache_dir))
        if local_path:
            _progress_print(f"[Main] Depth model local weight hit: {local_path}")
            _RESOLVED_HF_MODEL_FILES[cache_key] = local_path
            return local_path
    last_error: Exception | None = None
    _progress_print(f"[Main] Runtime preparation: checking depth model {model_id}")
    if not force_download:
        _progress_print(f"[Main] Checking local depth model cache: {model_id} in {cache_dir_text}")
        for filename in filenames:
            try:
                path = hf_hub_download(
                    repo_id=model_id,
                    filename=filename,
                    cache_dir=cache_dir_text,
                    local_files_only=True,
                )
                if not _is_valid_model_weight_file(path):
                    _remove_invalid_model_weight(path)
                    raise FileNotFoundError(f"cached model weight is empty: {path}")
                _progress_print(f"[Main] Depth model cache hit: {path}")
                _RESOLVED_HF_MODEL_FILES[cache_key] = path
                return path
            except Exception as exc:
                last_error = exc
    if local_files_only:
        _raise_model_resolution_error(model_id, last_error, local_only=True)

    for endpoint in _reachable_hf_endpoints(model_id):
        _progress_print(f"[Main] Depth model not found in local cache; preparing download from {endpoint}: {model_id}")
        with _hf_endpoint(endpoint):
            for filename in filenames:
                try:
                    _progress_print(
                        f"[Main] Preparing depth model download: {model_id}/{filename} "
                        f"to {cache_dir_text}. First download may take several minutes."
                    )
                    status_write(
                        f"正在下载深度模型权重：{model_id}/{filename}。首次下载可能需要几分钟，进度请看上方进度条。"
                    )
                    download_url = _hf_resolve_url(endpoint, model_id, filename)
                    _progress_print(f"[Main] Model download URL: {download_url}")
                    _probe_download_url(download_url)
                    _print_download_preparing_progress(filename)
                    path = hf_hub_download(
                        repo_id=model_id,
                        filename=filename,
                        cache_dir=cache_dir_text,
                        force_download=force_download,
                        tqdm_class=DownloadProgress,
                    )
                    if not _is_valid_model_weight_file(path):
                        _remove_invalid_model_weight(path)
                        path = hf_hub_download(
                            repo_id=model_id,
                            filename=filename,
                            cache_dir=cache_dir_text,
                            force_download=True,
                            tqdm_class=DownloadProgress,
                        )
                    if not _is_valid_model_weight_file(path):
                        _remove_invalid_model_weight(path)
                        path = _download_hf_file_direct(model_id, filename, cache_dir, endpoint)
                    status_write("深度模型权重下载完成，正在准备下一步。")
                    _RESOLVED_HF_MODEL_FILES[cache_key] = path
                    return path
                except Exception as exc:
                    last_error = exc
                    try:
                        path = _download_hf_file_direct(model_id, filename, cache_dir, endpoint)
                        status_write("深度模型权重下载完成，正在准备下一步。")
                        _RESOLVED_HF_MODEL_FILES[cache_key] = path
                        return path
                    except Exception as direct_exc:
                        last_error = direct_exc
                    _progress_print(f"[Main] Depth model download failed from {endpoint}: {type(exc).__name__}: {exc}")
    _raise_model_resolution_error(model_id, last_error, local_only=False)


class DistillAnyDepthBase518:
    def __init__(
        self,
        *,
        device: str | torch.device = "cuda",
        cache_dir: str | Path | None = None,
        dtype: torch.dtype | None = None,
        local_files_only: bool = False,
        force_download: bool = False,
        depth_upsample: DepthUpsampleMode = "bilinear",
        depth_upsample_edge_strength: float = 0.35,
    ) -> None:
        self.device = torch.device(device)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else default_lab_cache_dir()
        self.dtype = dtype or (torch.float16 if self.device.type == "cuda" else torch.float32)
        self.local_files_only = bool(local_files_only)
        self.force_download = bool(force_download)
        self.depth_upsample = depth_upsample
        self.depth_upsample_edge_strength = float(depth_upsample_edge_strength)
        self.info = DepthProviderInfo(
            provider="transformers.AutoModelForDepthEstimation",
            model_name=DISTILL_ANY_DEPTH_BASE_NAME,
            model_id=DISTILL_ANY_DEPTH_BASE_MODEL_ID,
            depth_resolution=DISTILL_ANY_DEPTH_BASE_RESOLUTION,
            cache_dir=str(self.cache_dir),
            load_mode="local_files_only" if self.local_files_only else "online_force_download" if self.force_download else "online",
            depth_backend="pytorch_cuda" if self.device.type == "cuda" else "pytorch_cpu",
            runtime="transformers",
        )
        self._model = None

    def load(self):
        if self._model is not None:
            return self._model

        from transformers import AutoModelForDepthEstimation

        kwargs = {
            "cache_dir": str(self.cache_dir),
            "dtype": self.dtype,
            "weights_only": True,
            "local_files_only": self.local_files_only,
            "force_download": self.force_download,
        }
        model = _load_hf_with_endpoint_fallback(
            lambda model_id: AutoModelForDepthEstimation.from_pretrained(model_id, **kwargs),
            DISTILL_ANY_DEPTH_BASE_MODEL_ID,
        )

        self._model = model.to(self.device).eval()
        return self._model

    def predict(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.predict_profile(rgb).depth

    def predict_profile(self, rgb: torch.Tensor) -> DepthProfileResult:
        def sync() -> None:
            if self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()

        sync()
        start = time.perf_counter()
        rgb = ensure_bchw(rgb, name="rgb").to(self.device).float().clamp(0, 1)
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
            mode="bicubic" if self.device.type == "cuda" else "bilinear",
            align_corners=False,
            antialias=True if self.device.type == "cuda" else False,
        ).to(self.dtype)

        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        sync()
        preprocess_ms = (time.perf_counter() - start) * 1000.0

        sync()
        start = time.perf_counter()
        predicted = self._run_depth_model(tensor).predicted_depth
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

    def _run_depth_model(self, tensor: torch.Tensor):
        """Model-call hook; override to route through an alternate engine.

        Must be safe to call from multiple threads (parallel depth slots);
        the default runs the torch module with the CUDA autocast policy.
        """
        model = self.load()
        use_autocast = self.device.type == "cuda" and self.dtype == torch.float16
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type, enabled=use_autocast
        ):
            return model(pixel_values=tensor)


class GenericAutoDepthProvider:
    def __init__(
        self,
        *,
        model_id: str,
        model_name: str | None = None,
        device: str | torch.device = "cuda",
        cache_dir: str | Path | None = None,
        dtype: torch.dtype | None = None,
        depth_resolution: int = DISTILL_ANY_DEPTH_BASE_RESOLUTION,
        patch_size: int | None = DISTILL_ANY_DEPTH_PATCH_SIZE,
        local_files_only: bool = False,
        force_download: bool = False,
        depth_upsample: DepthUpsampleMode = "bilinear",
        depth_upsample_edge_strength: float = 0.35,
    ) -> None:
        self.model_id = model_id
        self.model_name = model_name or model_id.rsplit("/", 1)[-1].replace("-hf", "")
        self.device = torch.device(device)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else default_lab_cache_dir()
        self.dtype = dtype or (torch.float16 if self.device.type == "cuda" else torch.float32)
        self.depth_resolution = int(depth_resolution)
        self.patch_size = patch_size
        self.local_files_only = bool(local_files_only)
        self.force_download = bool(force_download)
        self.depth_upsample = depth_upsample
        self.depth_upsample_edge_strength = float(depth_upsample_edge_strength)
        self.info = DepthProviderInfo(
            provider="transformers.AutoModelForDepthEstimation",
            model_name=self.model_name,
            model_id=self.model_id,
            depth_resolution=self.depth_resolution,
            cache_dir=str(self.cache_dir),
            load_mode="local_files_only" if self.local_files_only else "online_force_download" if self.force_download else "online",
            depth_backend="pytorch_cuda" if self.device.type == "cuda" else "pytorch_cpu",
            runtime="transformers-generic",
        )
        self._model = None

    def load(self):
        if self._model is not None:
            return self._model

        from transformers import AutoModelForDepthEstimation

        kwargs = {
            "cache_dir": str(self.cache_dir),
            "dtype": self.dtype,
            "weights_only": True,
            "local_files_only": self.local_files_only,
            "force_download": self.force_download,
        }
        model = _load_hf_with_endpoint_fallback(
            lambda model_id: AutoModelForDepthEstimation.from_pretrained(model_id, **kwargs),
            self.model_id,
        )
        self._model = model.to(self.device).eval()
        return self._model

    def _run_depth_model(self, tensor):
        use_autocast = self.device.type == "cuda" and self.dtype == torch.float16
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, enabled=use_autocast):
            return self.load()(pixel_values=tensor)

    def predict(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.predict_profile(rgb).depth

    def predict_profile(self, rgb: torch.Tensor) -> DepthProfileResult:
        def sync() -> None:
            if self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()

        sync()
        start = time.perf_counter()
        rgb = ensure_bchw(rgb, name="rgb").to(self.device).float().clamp(0, 1)
        _, _, height, width = rgb.shape
        input_h, input_w = _model_input_size(
            height,
            width,
            self.depth_resolution,
            self.patch_size or 1,
        )

        tensor = F.interpolate(
            rgb,
            size=(input_h, input_w),
            mode="bicubic" if self.device.type == "cuda" else "bilinear",
            align_corners=False,
            antialias=True if self.device.type == "cuda" else False,
        ).to(self.dtype)

        mean, std = _normalization_tensors_for_model(self.model_id, self.device, self.dtype)
        tensor = (tensor - mean) / std
        sync()
        preprocess_ms = (time.perf_counter() - start) * 1000.0

        sync()
        start = time.perf_counter()
        output = self._run_depth_model(tensor)
        predicted = _extract_depth_output(output)
        sync()
        model_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        depth = ensure_b1hw(predicted)
        depth = _postprocess_generic_depth(depth, self.model_id)
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


class InfiniDepthProvider:
    def __init__(
        self,
        *,
        model_id: str,
        model_name: str | None = None,
        device: str | torch.device = "cuda",
        cache_dir: str | Path | None = None,
        dtype: torch.dtype | None = None,
        depth_resolution: int = DISTILL_ANY_DEPTH_BASE_RESOLUTION,
        local_files_only: bool = False,
        force_download: bool = False,
        depth_upsample: DepthUpsampleMode = "bilinear",
        depth_upsample_edge_strength: float = 0.35,
    ) -> None:
        self.model_id = model_id
        self.model_name = model_name or model_id.rsplit("/", 1)[-1]
        self.device = torch.device(device)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else default_lab_cache_dir()
        self.dtype = dtype or (torch.float16 if self.device.type == "cuda" else torch.float32)
        self.depth_resolution = int(depth_resolution)
        self.encoder = _infinidepth_encoder_for_model(model_id)
        self.local_files_only = bool(local_files_only)
        self.force_download = bool(force_download)
        self.depth_upsample = depth_upsample
        self.depth_upsample_edge_strength = float(depth_upsample_edge_strength)
        self.info = DepthProviderInfo(
            provider="stereo_runtime.model_impl.InfiniDepth.api.InfiniDepthModel",
            model_name=self.model_name,
            model_id=self.model_id,
            depth_resolution=self.depth_resolution,
            cache_dir=str(self.cache_dir),
            load_mode="local_files_only" if self.local_files_only else "online_force_download" if self.force_download else "online",
            depth_backend="pytorch_cuda" if self.device.type == "cuda" else "pytorch_cpu",
            runtime="infinidepth",
        )
        self._model = None

    def load(self):
        if self._model is not None:
            return self._model

        from stereo_runtime.model_impl.InfiniDepth.api import InfiniDepthModel

        model_path = _resolve_hf_model_file(
            self.model_id,
            self.cache_dir,
            local_files_only=self.local_files_only,
            force_download=self.force_download,
        )
        model = InfiniDepthModel(model_path=model_path, encoder=self.encoder)
        self._model = model.to(self.device, dtype=self.dtype).eval()
        return self._model

    def predict(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.predict_profile(rgb).depth

    def predict_profile(self, rgb: torch.Tensor) -> DepthProfileResult:
        def sync() -> None:
            if self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()

        sync()
        start = time.perf_counter()
        rgb = ensure_bchw(rgb, name="rgb").to(self.device).float().clamp(0, 1)
        _, _, height, width = rgb.shape
        input_h, input_w = _model_input_size(
            height,
            width,
            self.depth_resolution,
            INFINIDEPTH_PATCH_SIZE,
        )

        tensor = F.interpolate(
            rgb,
            size=(input_h, input_w),
            mode="bicubic" if self.device.type == "cuda" else "bilinear",
            align_corners=False,
            antialias=True if self.device.type == "cuda" else False,
        ).to(self.dtype)
        sync()
        preprocess_ms = (time.perf_counter() - start) * 1000.0

        model = self.load()
        use_autocast = self.device.type == "cuda" and self.dtype == torch.float16
        sync()
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, enabled=use_autocast):
            predicted = model.predict_depth(tensor, fp32=self.dtype != torch.float16)
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


TorchDepthProvider = DistillAnyDepthBase518
GenericTorchDepthProvider = GenericAutoDepthProvider
InfiniDepthTorchProvider = InfiniDepthProvider


def _normalization_tensors_for_model(model_id: str, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    model_lower = model_id.lower()
    if any(key in model_lower for key in ("depthpro", "zoedepth", "dpt")):
        mean_values = [0.5, 0.5, 0.5]
        std_values = [0.5, 0.5, 0.5]
    else:
        mean_values = [0.485, 0.456, 0.406]
        std_values = [0.229, 0.224, 0.225]
    mean = torch.tensor(mean_values, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(std_values, device=device, dtype=dtype).view(1, 3, 1, 1)
    return mean, std


def _extract_depth_output(output):
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "predicted_depth"):
        return output.predicted_depth
    if isinstance(output, dict) and "predicted_depth" in output:
        return output["predicted_depth"]
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    raise RuntimeError(f"unsupported model output type: {type(output).__name__}")


def _is_metric_model(model_id: str) -> bool:
    model_lower = model_id.lower()
    return any(key in model_lower for key in ("metric", "kitti", "nyu", "depth-ai", "da3"))


def _postprocess_generic_depth(depth: torch.Tensor, model_id: str) -> torch.Tensor:
    depth = ensure_b1hw(depth).float()
    if _is_metric_model(model_id):
        depth = depth.clamp_min(5e-3).reciprocal()
    return _normalize_depth(depth)


def _prepare_accelerated_artifacts(
    cfg: DepthProviderConfig,
    *,
    build_trt: bool = False,
    input_size: tuple[int, int] | None = None,
):
    from .model_artifacts import prepare_model_artifacts

    cache_dir = Path(cfg.cache_dir) if cfg.cache_dir is not None else default_lab_cache_dir()
    model_dir = Path(cfg.onnx_path).parent if cfg.onnx_path is not None else None
    kwargs: dict[str, int] = {}
    if input_size is not None:
        kwargs["export_height"], kwargs["export_width"] = int(input_size[0]), int(input_size[1])
    result = prepare_model_artifacts(
        cfg.model_id,
        cache_dir=cache_dir,
        model_dir=model_dir,
        local_files_only=cfg.local_files_only,
        force_download=cfg.force_download,
        download_if_missing=not cfg.local_files_only,
        onnx_dtype=cfg.onnx_dtype,
        export_onnx_if_missing=True,
        artifact_backend="tensorrt" if build_trt else "onnx",
        build_trt_if_missing=build_trt,
        force_rebuild_trt=cfg.force_rebuild,
        **kwargs,
    )
    return result


class AutoDepthProvider:
    """Lazy automatic depth provider with explicit fallback telemetry."""

    def __init__(self, config: DepthProviderConfig):
        self.config = config
        self._provider = None
        self._attempts: list[dict[str, str]] = []
        self._factories = None
        self._next_candidate = 0
        self.info = DepthProviderInfo(
            provider="auto",
            model_name=config.model_name,
            model_id=config.model_id,
            depth_resolution=config.depth_resolution,
            cache_dir=str(config.cache_dir or ""),
            load_mode="auto",
            depth_backend="auto",
            runtime="auto",
            execution_provider="automatic",
            output_device=str(config.device),
        )

    @property
    def attempts(self) -> tuple[dict[str, str], ...]:
        return tuple(dict(item) for item in self._attempts)

    def _device_type(self) -> str:
        try:
            return str(torch.device(self.config.device).type)
        except (TypeError, RuntimeError, ValueError):
            return "cpu"

    def _torch_provider(self, device: str | torch.device):
        cfg = self.config
        common = {
            "cache_dir": cfg.cache_dir,
            "local_files_only": cfg.local_files_only,
            "force_download": cfg.force_download,
            "depth_upsample": cfg.depth_upsample,
            "depth_upsample_edge_strength": cfg.depth_upsample_edge_strength,
        }
        if cfg.model_id != DISTILL_ANY_DEPTH_BASE_MODEL_ID:
            if _is_infinidepth_model(cfg.model_id):
                return InfiniDepthProvider(
                    model_id=cfg.model_id,
                    model_name=cfg.model_name,
                    device=device,
                    depth_resolution=cfg.depth_resolution,
                    **common,
                )
            return GenericAutoDepthProvider(
                model_id=cfg.model_id,
                model_name=cfg.model_name,
                device=device,
                depth_resolution=cfg.depth_resolution,
                patch_size=cfg.patch_size,
                **common,
            )
        return DistillAnyDepthBase518(device=device, **common)

    def _candidate_factories(self):
        if self._factories is not None:
            return self._factories
        cfg = self.config
        device_type = self._device_type()
        factories: list[tuple[str, Callable[[], Any]]] = []
        if device_type == "cuda" and not getattr(torch.version, "hip", None):
            from .providers.nvidia.tensorrt_native import NativeTensorRtDepthProvider

            factories.append((
                "原生 TensorRT",
                lambda: NativeTensorRtDepthProvider(
                    device=cfg.device,
                    cache_dir=cfg.cache_dir,
                    onnx_path=cfg.onnx_path,
                    onnx_dtype=cfg.onnx_dtype,
                    engine_path=cfg.engine_path,
                    local_files_only=cfg.local_files_only,
                    force_download=cfg.force_download,
                    model_id=cfg.model_id,
                    model_name=cfg.model_name,
                    depth_resolution=cfg.depth_resolution,
                    build_engine=cfg.build_engine,
                    force_rebuild=cfg.force_rebuild,
                    use_cuda_graph=cfg.use_cuda_graph,
                    profile_sync=cfg.profile_sync,
                    execution_slot_count=cfg.execution_slot_count,
                    depth_upsample=cfg.depth_upsample,
                    depth_upsample_edge_strength=cfg.depth_upsample_edge_strength,
                ),
            ))
            factories.append(("PyTorch CUDA", lambda: self._torch_provider(cfg.device)))
        elif device_type == "cuda" and getattr(torch.version, "hip", None):
            from .providers.amd import create_pytorch_rocm_provider

            factories.append((
                "PyTorch ROCm",
                lambda: create_pytorch_rocm_provider(
                    model_id=cfg.model_id,
                    model_name=cfg.model_name,
                    device=cfg.device,
                    cache_dir=cfg.cache_dir,
                    depth_resolution=cfg.depth_resolution,
                    patch_size=cfg.patch_size,
                    local_files_only=cfg.local_files_only,
                    force_download=cfg.force_download,
                    depth_upsample=cfg.depth_upsample,
                    depth_upsample_edge_strength=cfg.depth_upsample_edge_strength,
                ),
            ))
        elif device_type == "privateuseone" and os.name == "nt":
            from .providers.directml import create_directml_provider

            factories.append((
                "DirectML",
                lambda: create_directml_provider(
                    model_id=cfg.model_id,
                    model_name=cfg.model_name,
                    device=cfg.device,
                    cache_dir=cfg.cache_dir,
                    depth_resolution=cfg.depth_resolution,
                    patch_size=cfg.patch_size,
                    local_files_only=cfg.local_files_only,
                    force_download=cfg.force_download,
                    depth_upsample=cfg.depth_upsample,
                    depth_upsample_edge_strength=cfg.depth_upsample_edge_strength,
                ),
            ))
        elif device_type == "xpu":
            from .providers.intel import create_pytorch_xpu_provider

            factories.append((
                "PyTorch XPU",
                lambda: create_pytorch_xpu_provider(
                    model_id=cfg.model_id,
                    model_name=cfg.model_name,
                    device=cfg.device,
                    cache_dir=cfg.cache_dir,
                    depth_resolution=cfg.depth_resolution,
                    patch_size=cfg.patch_size,
                    local_files_only=cfg.local_files_only,
                    force_download=cfg.force_download,
                    depth_upsample=cfg.depth_upsample,
                    depth_upsample_edge_strength=cfg.depth_upsample_edge_strength,
                ),
            ))
        elif device_type == "mps":
            from .providers.apple import create_pytorch_mps_provider

            factories.append((
                "PyTorch MPS",
                lambda: create_pytorch_mps_provider(
                    model_id=cfg.model_id,
                    model_name=cfg.model_name,
                    device=cfg.device,
                    cache_dir=cfg.cache_dir,
                    depth_resolution=cfg.depth_resolution,
                    patch_size=cfg.patch_size,
                    local_files_only=cfg.local_files_only,
                    force_download=cfg.force_download,
                    depth_upsample=cfg.depth_upsample,
                    depth_upsample_edge_strength=cfg.depth_upsample_edge_strength,
                    use_coreml=cfg.use_coreml,
                    recompile_coreml=cfg.recompile_coreml,
                ),
            ))
        if os.name == "nt":
            from .providers.directml import create_directml_provider, is_directml_available

            if is_directml_available() and not any(name == "DirectML" for name, _ in factories):
                factories.append((
                    "DirectML",
                    lambda: create_directml_provider(
                        model_id=cfg.model_id,
                        model_name=cfg.model_name,
                        cache_dir=cfg.cache_dir,
                        depth_resolution=cfg.depth_resolution,
                        patch_size=cfg.patch_size,
                        local_files_only=cfg.local_files_only,
                        force_download=cfg.force_download,
                        depth_upsample=cfg.depth_upsample,
                        depth_upsample_edge_strength=cfg.depth_upsample_edge_strength,
                    ),
                ))
        factories.append(("CPU", lambda: self._torch_provider("cpu")))
        self._factories = factories
        return factories

    def _activate_next(self) -> Any:
        if self._provider is not None:
            return self._provider
        factories = self._candidate_factories()
        last_error = None
        while self._next_candidate < len(factories):
            name, factory = factories[self._next_candidate]
            self._next_candidate += 1
            try:
                provider = factory()
                load = getattr(provider, "load", None)
                if callable(load):
                    load()
                self._provider = provider
                fallback_reason = "; ".join(
                    f"{item['backend']}: {item['reason']}" for item in self._attempts
                ) or None
                self.info = replace(provider.info, fallback_reason=fallback_reason)
                print(
                    f"[DepthBackend] selected={name} attempts={len(self._attempts)}",
                    flush=True,
                )
                return provider
            except Exception as exc:
                last_error = exc
                reason = f"{type(exc).__name__}: {exc}"
                self._attempts.append({
                    "backend": name,
                    "status": "failed",
                    "reason": reason,
                })
                print(f"[DepthBackend] fallback={name} reason={reason}", flush=True)
        raise RuntimeError(
            "No depth inference backend is available: "
            + "; ".join(
                f"{item['backend']}={item['reason']}" for item in self._attempts
            )
        ) from last_error

    def load(self):
        provider = self._activate_next()
        load = getattr(provider, "load", None)
        return load() if callable(load) else None

    def predict_profile(self, rgb: torch.Tensor) -> DepthProfileResult:
        provider = self._activate_next()
        try:
            result = provider.predict_profile(rgb)
        except Exception as exc:
            close = getattr(provider, "close", None)
            if callable(close):
                close()
            self._provider = None
            self._attempts.append({
                "backend": str(getattr(provider.info, "depth_backend", type(provider).__name__)),
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            })
            print(
                f"[DepthBackend] active provider failed; retrying chain: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            provider = self._activate_next()
            result = provider.predict_profile(rgb)
        self.info = replace(
            provider.info,
            fallback_reason=(
                "; ".join(
                    f"{item['backend']}: {item['reason']}" for item in self._attempts
                ) or None
            ),
        )
        return result

    def predict(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.predict_profile(rgb).depth

    def close(self) -> None:
        if self._provider is not None:
            close = getattr(self._provider, "close", None)
            if callable(close):
                close()
        self._provider = None


def create_depth_provider(config: DepthProviderConfig | dict[str, Any] | None = None):
    cfg = config if isinstance(config, DepthProviderConfig) else DepthProviderConfig(**(config or {}))
    backend = cfg.backend
    device = torch.device(cfg.device)

    if backend in {"auto", "automatic", "nvidia_auto"}:
        return AutoDepthProvider(cfg)

    if backend in {"migraphx_rocm", "rocm_migraphx", "migraphx"}:
        from .providers.amd import create_migraphx_rocm_provider

        return create_migraphx_rocm_provider(
            model_id=cfg.model_id,
            model_name=cfg.model_name,
            device=device,
            cache_dir=cfg.cache_dir,
            onnx_path=cfg.onnx_path,
            graph_path=cfg.engine_path,
            build_graph=cfg.build_engine,
            force_rebuild=cfg.force_rebuild,
            local_files_only=cfg.local_files_only,
            force_download=cfg.force_download,
            allow_pytorch_fallback=cfg.allow_pytorch_fallback,
            depth_resolution=cfg.depth_resolution,
            patch_size=cfg.patch_size,
            depth_upsample=cfg.depth_upsample,
            depth_upsample_edge_strength=cfg.depth_upsample_edge_strength,
        )

    if backend in {"pytorch_rocm", "rocm", "amd_rocm"}:
        from .providers.amd import create_pytorch_rocm_provider

        return create_pytorch_rocm_provider(
            model_id=cfg.model_id,
            model_name=cfg.model_name,
            device=device,
            cache_dir=cfg.cache_dir,
            depth_resolution=cfg.depth_resolution,
            patch_size=cfg.patch_size,
            local_files_only=cfg.local_files_only,
            force_download=cfg.force_download,
            depth_upsample=cfg.depth_upsample,
            depth_upsample_edge_strength=cfg.depth_upsample_edge_strength,
        )

    if backend in {"pytorch_xpu", "xpu", "intel_xpu"}:
        from .providers.intel import create_pytorch_xpu_provider

        return create_pytorch_xpu_provider(
            model_id=cfg.model_id,
            model_name=cfg.model_name,
            device=device,
            cache_dir=cfg.cache_dir,
            depth_resolution=cfg.depth_resolution,
            patch_size=cfg.patch_size,
            local_files_only=cfg.local_files_only,
            force_download=cfg.force_download,
            depth_upsample=cfg.depth_upsample,
            depth_upsample_edge_strength=cfg.depth_upsample_edge_strength,
        )

    if backend in {"directml", "pytorch_directml", "dml"}:
        from .providers.directml import create_directml_provider

        return create_directml_provider(
            model_id=cfg.model_id,
            model_name=cfg.model_name,
            device=cfg.device,
            cache_dir=cfg.cache_dir,
            depth_resolution=cfg.depth_resolution,
            patch_size=cfg.patch_size,
            local_files_only=cfg.local_files_only,
            force_download=cfg.force_download,
            depth_upsample=cfg.depth_upsample,
            depth_upsample_edge_strength=cfg.depth_upsample_edge_strength,
        )

    if backend in {"pytorch_mps", "mps", "apple_mps"}:
        from .providers.apple import create_pytorch_mps_provider

        return create_pytorch_mps_provider(
            model_id=cfg.model_id,
            model_name=cfg.model_name,
            device=device,
            cache_dir=cfg.cache_dir,
            depth_resolution=cfg.depth_resolution,
            patch_size=cfg.patch_size,
            local_files_only=cfg.local_files_only,
            force_download=cfg.force_download,
            depth_upsample=cfg.depth_upsample,
            depth_upsample_edge_strength=cfg.depth_upsample_edge_strength,
            use_coreml=cfg.use_coreml,
            recompile_coreml=cfg.recompile_coreml,
        )

    if backend in {"tensorrt_native", "native_tensorrt", "tensorrt_native_graph"} or (
        backend in {"distill_base_nvidia", "nvidia_chain"} and cfg.prefer_native_tensorrt
    ):
        from .providers.nvidia.tensorrt_native import NativeTensorRtDepthProvider

        return NativeTensorRtDepthProvider(
            device=device,
            cache_dir=cfg.cache_dir,
            onnx_path=cfg.onnx_path,
            onnx_dtype=cfg.onnx_dtype,
            engine_path=cfg.engine_path,
            local_files_only=cfg.local_files_only,
            force_download=cfg.force_download,
            model_id=cfg.model_id,
            model_name=cfg.model_name,
            depth_resolution=cfg.depth_resolution,
            build_engine=cfg.build_engine,
            force_rebuild=cfg.force_rebuild,
            use_cuda_graph=cfg.use_cuda_graph or backend == "tensorrt_native_graph",
            profile_sync=cfg.profile_sync,
            execution_slot_count=cfg.execution_slot_count,
            depth_upsample=cfg.depth_upsample,
            depth_upsample_edge_strength=cfg.depth_upsample_edge_strength,
        )

    ort_available = _onnxruntime_available()

    if backend in {"distill_base_nvidia", "nvidia_chain", "tensorrt", "tensorrt_ort"} and cfg.prefer_tensorrt:
        if not ort_available:
            if cfg.require_tensorrt or not cfg.allow_pytorch_fallback:
                raise RuntimeError("ONNX Runtime is not installed; TensorRT ORT depth provider is unavailable")
        else:
            from .providers.nvidia.tensorrt_ort import TensorRtOrtDepthProvider

            return TensorRtOrtDepthProvider(
                device=device,
                cache_dir=cfg.cache_dir,
                onnx_path=cfg.onnx_path,
                onnx_dtype=cfg.onnx_dtype,
                trt_cache_dir=cfg.trt_cache_dir,
                local_files_only=cfg.local_files_only,
                force_download=cfg.force_download,
                model_id=cfg.model_id,
                model_name=cfg.model_name,
                depth_upsample=cfg.depth_upsample,
                depth_upsample_edge_strength=cfg.depth_upsample_edge_strength,
            )

    if backend in {"distill_base_nvidia", "nvidia_chain", "onnx_cuda", "onnx_cuda_iobinding"} and cfg.prefer_onnx:
        if not ort_available:
            if not cfg.allow_pytorch_fallback:
                raise RuntimeError("ONNX Runtime is not installed; ONNX CUDA depth provider is unavailable")
        else:
            from .providers.nvidia.onnx_cuda import OnnxCudaDepthProvider

            return OnnxCudaDepthProvider(
                device=device,
                cache_dir=cfg.cache_dir,
                onnx_path=cfg.onnx_path,
                onnx_dtype=cfg.onnx_dtype,
                model_id=cfg.model_id,
                model_name=cfg.model_name,
                use_iobinding=cfg.use_iobinding,
                use_dlpack=cfg.use_dlpack,
                local_files_only=cfg.local_files_only,
                force_download=cfg.force_download,
                depth_upsample=cfg.depth_upsample,
                depth_upsample_edge_strength=cfg.depth_upsample_edge_strength,
            )

    if backend in {"distill_base_518", "distill_base_nvidia", "nvidia_chain", "tensorrt", "tensorrt_ort", "onnx_cuda", "onnx_cuda_iobinding", "pytorch_cuda", "pytorch"}:
        if device.type == "mps":
            from .providers.apple import create_pytorch_mps_provider

            return create_pytorch_mps_provider(
                model_id=cfg.model_id,
                model_name=cfg.model_name,
                device=device,
                cache_dir=cfg.cache_dir,
                depth_resolution=cfg.depth_resolution,
                patch_size=cfg.patch_size,
                local_files_only=cfg.local_files_only,
                force_download=cfg.force_download,
                depth_upsample=cfg.depth_upsample,
                depth_upsample_edge_strength=cfg.depth_upsample_edge_strength,
                use_coreml=cfg.use_coreml,
                recompile_coreml=cfg.recompile_coreml,
            )
        if cfg.model_id != DISTILL_ANY_DEPTH_BASE_MODEL_ID:
            if _is_infinidepth_model(cfg.model_id):
                return InfiniDepthProvider(
                    model_id=cfg.model_id,
                    model_name=cfg.model_name,
                    device=device,
                    cache_dir=cfg.cache_dir,
                    depth_resolution=cfg.depth_resolution,
                    local_files_only=cfg.local_files_only,
                    force_download=cfg.force_download,
                    depth_upsample=cfg.depth_upsample,
                    depth_upsample_edge_strength=cfg.depth_upsample_edge_strength,
                )
            return GenericAutoDepthProvider(
                model_id=cfg.model_id,
                model_name=cfg.model_name,
                device=device,
                cache_dir=cfg.cache_dir,
                depth_resolution=cfg.depth_resolution,
                patch_size=cfg.patch_size,
                local_files_only=cfg.local_files_only,
                force_download=cfg.force_download,
                depth_upsample=cfg.depth_upsample,
                depth_upsample_edge_strength=cfg.depth_upsample_edge_strength,
            )
        return DistillAnyDepthBase518(
            device=device,
            cache_dir=cfg.cache_dir,
            local_files_only=cfg.local_files_only,
            force_download=cfg.force_download,
            depth_upsample=cfg.depth_upsample,
            depth_upsample_edge_strength=cfg.depth_upsample_edge_strength,
        )

    raise ValueError(f"unknown depth backend: {backend}")


def estimate_depth(
    rgb: torch.Tensor,
    config: DepthProviderConfig | dict[str, Any] | None = None,
) -> tuple[torch.Tensor, DepthProviderInfo]:
    provider = create_depth_provider(config)
    depth = provider.predict(rgb)
    return depth, provider.info


def estimate_distill_any_depth_base_518(
    rgb: torch.Tensor,
    *,
    device: str | torch.device = "cuda",
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    force_download: bool = False,
) -> tuple[torch.Tensor, DepthProviderInfo]:
    provider = DistillAnyDepthBase518(
        device=device,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        force_download=force_download,
    )
    return provider.predict(rgb), provider.info
