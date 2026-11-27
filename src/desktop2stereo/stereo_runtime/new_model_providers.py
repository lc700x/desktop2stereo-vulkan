"""PyTorch providers for DepthPro2, MoGe-3, and FlashDepth."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from .depth_provider import (
    DepthProfileResult,
    DepthProviderInfo,
    _model_input_size_for_model,
    _postprocess_model_depth,
)
from .depth_upsample import DepthUpsampleMode, _upsample_depth_canonical
from .model_impl.custom_models import (
    load_depth_pro_2,
    load_flashdepth,
    load_moge_3,
    normalize_imagenet,
    predict_flashdepth,
    predict_moge_3_depth,
    resize_rgb,
    resolve_spec,
)
from .model_registry import DepthModelSpec
from .output import ensure_bchw

FAMILY_PATCH_SIZE = {
    "depth-pro-2": 32,
    "moge-3": 14,
    "flashdepth": 14,
}


def _backend_for_device(device: torch.device) -> str:
    if device.type == "cuda":
        return "pytorch_rocm" if torch.version.hip else "pytorch_cuda"
    if device.type == "mps":
        return "pytorch_mps"
    if device.type == "cpu":
        return "pytorch_cpu"
    return f"pytorch_{device.type}"


def _sync(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _vram_resolution_cap(spec: DepthModelSpec, device: torch.device, requested: int) -> int:
    """Keep the largest variants usable on commonly available 8 GB GPUs."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return requested
    try:
        index = device.index or torch.cuda.current_device()
        total_gib = float(torch.cuda.get_device_properties(index).total_memory) / (1024.0 ** 3)
    except (AttributeError, RuntimeError, ValueError):
        return requested
    if total_gib >= 10.0:
        return requested
    caps = {
        "DepthPro2-H": 294,
        "DepthPro2-L": 336,
        "DepthPro2-B": 448,
        "MoGe3-G": 336,
        "FlashDepth-L": 448,
        "FlashDepth-Full": 448,
    }
    return min(requested, caps.get(spec.name, requested))


class _FamilyProvider:
    family = ""

    def __init__(
        self,
        *,
        model_id: str,
        model_name: str | None = None,
        device: str | torch.device = "cuda",
        cache_dir: str | Path | None = None,
        dtype: torch.dtype | None = None,
        depth_resolution: int = 518,
        local_files_only: bool = False,
        force_download: bool = False,
        compat_profile: bool = True,
        depth_upsample: DepthUpsampleMode = "joint_bilateral",
        depth_upsample_edge_strength: float = 0.35,
    ) -> None:
        self.model_id = model_id
        self.model_name = model_name or model_id
        self.device = torch.device(device)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else Path("./models")
        self.dtype = dtype
        self.depth_resolution = int(depth_resolution)
        self.local_files_only = bool(local_files_only)
        self.force_download = bool(force_download)
        self.compat_profile = bool(compat_profile)
        self.depth_upsample = depth_upsample
        self.depth_upsample_edge_strength = float(depth_upsample_edge_strength)
        self.spec = resolve_spec(self.model_name)
        if self.spec.family != self.family:
            raise ValueError(
                f"{self.model_name!r} belongs to {self.spec.family!r}, "
                f"not {self.family!r}"
            )
        requested_resolution = self.depth_resolution
        self.depth_resolution = _vram_resolution_cap(
            self.spec,
            self.device,
            requested_resolution,
        )
        self.auto_resolution_cap = (
            self.depth_resolution < requested_resolution
        )
        profile = (
            self.spec.compat_profile
            if self.compat_profile and self.spec.compat_profile
            else "full-fidelity"
        )
        self.model_profile = (
            f"{profile}@{self.depth_resolution}"
            if self.auto_resolution_cap
            else profile
        )
        self._model: Any | None = None
        self.info = DepthProviderInfo(
            provider=f"stereo_runtime.model_impl.{self.family}",
            model_name=self.spec.name,
            model_id=self.spec.model_id,
            depth_resolution=self.depth_resolution,
            cache_dir=str(self.cache_dir),
            load_mode=(
                "local_files_only"
                if self.local_files_only
                else "online_force_download"
                if self.force_download
                else "online"
            ),
            depth_backend=_backend_for_device(self.device),
            runtime=self.family,
            output_device=str(self.device),
            model_profile=self.model_profile,
            zero_copy_output=True,
            host_readback_bytes=0,
            raw_output_semantics=self.spec.output_semantics,
        )

    @property
    def pipeline_slot_count(self) -> int:
        return 1

    def _load(self) -> Any:
        raise NotImplementedError

    def load(self) -> Any:
        if self._model is None:
            self._model = self._load()
        return self._model

    def _prepare_input(self, rgb: torch.Tensor) -> torch.Tensor:
        rgb = ensure_bchw(rgb, name="rgb").to(self.device).float()
        rgb = torch.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        _, _, height, width = rgb.shape
        patch = FAMILY_PATCH_SIZE[self.family]
        input_h, input_w = _model_input_size_for_model(
            height,
            width,
            self.depth_resolution,
            patch,
            self.spec.model_id,
        )
        rgb = resize_rgb(rgb, input_h, input_w)
        if self.family in {"depth-pro-2", "flashdepth"}:
            rgb = normalize_imagenet(rgb, dtype=torch.float32)
        return rgb

    def _predict_model(self, model: Any, rgb: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def predict_profile(self, rgb: torch.Tensor) -> DepthProfileResult:
        model = self.load()
        _sync(self.device)
        start = time.perf_counter()
        tensor = self._prepare_input(rgb)
        _sync(self.device)
        preprocess_ms = (time.perf_counter() - start) * 1000.0

        _sync(self.device)
        start = time.perf_counter()
        predicted = self._predict_model(model, tensor)
        _sync(self.device)
        model_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        depth = _postprocess_model_depth(predicted, self.spec.name)
        _, _, height, width = ensure_bchw(rgb, name="rgb").shape
        depth = _upsample_depth_canonical(
            depth,
            height,
            width,
            rgb=rgb,
            mode=self.depth_upsample,
            edge_strength=self.depth_upsample_edge_strength,
        )
        _sync(self.device)
        postprocess_ms = (time.perf_counter() - start) * 1000.0
        return DepthProfileResult(
            depth,
            preprocess_ms,
            model_ms,
            postprocess_ms,
            finite_depth=True,
            nonfinite_count=0,
            raw_output_semantics=self.spec.output_semantics,
            model_profile=self.model_profile,
            zero_copy_output=True,
            host_readback_bytes=0,
        )

    def predict(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.predict_profile(rgb).depth

    def close(self) -> None:
        self._model = None


class DepthPro2Provider(_FamilyProvider):
    family = "depth-pro-2"

    def _load(self) -> Any:
        model, spec = load_depth_pro_2(
            model_name_or_id=self.spec.name,
            cache_dir=self.cache_dir,
            device=self.device,
            dtype=self.dtype,
            local_files_only=self.local_files_only,
            force_download=self.force_download,
        )
        self.spec = spec
        return model

    def _predict_model(self, model: Any, rgb: torch.Tensor) -> torch.Tensor:
        # DepthPro2's public inference API returns focal-normalized inverse
        # depth.  Keep that response on the GPU and let the shared D2S
        # normalizer apply percentile clipping and finite-value repair.
        prediction = model.infer(rgb[0])
        inverse_depth = prediction.get("inverse_depth") if isinstance(prediction, dict) else None
        if inverse_depth is None:
            raise RuntimeError("DepthPro2 infer() returned no inverse_depth")
        return inverse_depth.unsqueeze(0).unsqueeze(0)


class MoGe3Provider(_FamilyProvider):
    family = "moge-3"

    def _load(self) -> Any:
        model, spec = load_moge_3(
            model_name_or_id=self.spec.name,
            cache_dir=self.cache_dir,
            device=self.device,
            compat_profile=self.compat_profile,
            local_files_only=self.local_files_only,
            force_download=self.force_download,
        )
        self.spec = spec
        return model

    def _predict_model(self, model: Any, rgb: torch.Tensor) -> torch.Tensor:
        return predict_moge_3_depth(
            model,
            rgb,
            target_resolution=self.depth_resolution,
        ).unsqueeze(1)


class FlashDepthProvider(_FamilyProvider):
    family = "flashdepth"

    def _load(self) -> Any:
        model, spec = load_flashdepth(
            model_name_or_id=self.spec.name,
            cache_dir=self.cache_dir,
            device=self.device,
            local_files_only=self.local_files_only,
            force_download=self.force_download,
        )
        self.spec = spec
        return model

    def _predict_model(self, model: Any, rgb: torch.Tensor) -> torch.Tensor:
        return predict_flashdepth(model, rgb).unsqueeze(1)


def create_family_provider(
    *,
    family: str,
    model_id: str,
    model_name: str | None = None,
    **kwargs: Any,
) -> _FamilyProvider:
    classes = {
        "depth-pro-2": DepthPro2Provider,
        "moge-3": MoGe3Provider,
        "flashdepth": FlashDepthProvider,
    }
    try:
        provider_cls = classes[family]
    except KeyError as exc:
        raise ValueError(f"unsupported native depth family: {family}") from exc
    return provider_cls(model_id=model_id, model_name=model_name, **kwargs)


def family_for_spec(spec: DepthModelSpec) -> str | None:
    return spec.family if spec.family in FAMILY_PATCH_SIZE else None


__all__ = [
    "DepthPro2Provider",
    "FlashDepthProvider",
    "MoGe3Provider",
    "create_family_provider",
    "family_for_spec",
]
