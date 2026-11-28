"""Strict Core ML providers for DepthPro2, MoGe-3, and FlashDepth."""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

from ...depth_provider import (
    DepthProfileResult,
    _model_input_size_for_model,
    _normalization_tensors_for_model,
    _normalize_depth_canonical,
)
from ...depth_upsample import _upsample_depth_canonical
from ...model_impl.custom_models import (
    _moge_num_tokens,
    _moge_point_map_to_depth,
    load_depth_pro_2,
    load_flashdepth,
    load_moge_3,
    predict_flashdepth,
)
from ...model_registry import FAMILY_INPUT_MULTIPLE
from ...new_model_providers import _FamilyProvider
from ...output import ensure_bchw
from .pytorch_mps import (
    CoreMLEngine,
    _safe_model_tag,
    coreml_safe_dinov2_positions,
    coreml_safe_interpolate,
    install_coremltools_workarounds,
)


class CoreMLFamilyAccelerationError(RuntimeError):
    """Raised when a family model cannot satisfy the accelerator-only gate."""


FAMILY_EXPORT_SCHEMA_VERSION = 1


def depth_to_d2s_response(depth: torch.Tensor, semantics: str) -> torch.Tensor:
    """Convert official model output to Distill-compatible near-high response."""
    if semantics in {"relative_z_depth", "metric_z_depth"}:
        return depth.clamp_min(5e-3).reciprocal()
    if semantics in {"relative_inverse_depth", "raw_disparity"}:
        return depth
    raise ValueError(f"unsupported depth output semantics: {semantics!r}")


class _DepthPro2CoreMLWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        inverse_depth = self.model.predict_inverse_depth(pixel_values)
        response = depth_to_d2s_response(inverse_depth, "relative_inverse_depth")
        return response.unsqueeze(1) if response.ndim == 3 else response


class _MoGe3CoreMLWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, target_resolution: int) -> None:
        super().__init__()
        self.model = model
        self.num_tokens = _moge_num_tokens(model, target_resolution)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        output = self.model.forward(
            pixel_values.float(),
            num_tokens=self.num_tokens,
            refine_steps=0,
            return_per_step=False,
        )
        points = output.get("points")
        if points is None:
            raise RuntimeError("MoGe-3 returned no point map")
        mask = output.get("mask")
        depth = _moge_point_map_to_depth(
            points.float(),
            mask=mask.float() if isinstance(mask, torch.Tensor) else None,
        )
        metric_scale = output.get("metric_scale")
        if isinstance(metric_scale, torch.Tensor):
            depth = depth * metric_scale.float()[:, None, None]
        # MoGe's official depth output is metric Z. Convert it to the same
        # near-high response consumed by Distill before graph normalization.
        return depth_to_d2s_response(depth, "metric_z_depth").unsqueeze(1)


class _FlashDepthCoreMLWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, input_height: int, input_width: int) -> None:
        super().__init__()
        self.model = model
        self.input_height = int(input_height)
        self.input_width = int(input_width)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        inverse_depth = predict_flashdepth(self.model, pixel_values)
        return depth_to_d2s_response(inverse_depth, "relative_inverse_depth").unsqueeze(1)


def _device_type(device: Any) -> str:
    return type(device).__name__


def summarize_compute_plan_usages(usages: list[Any]) -> dict[str, Any]:
    preferred: Counter[str] = Counter()
    supported: Counter[str] = Counter()
    cpu_only: list[str] = []
    for item in usages:
        if item is None:
            continue
        preferred[_device_type(item.preferred_compute_device)] += 1
        devices = {_device_type(device) for device in item.supported_compute_devices}
        for name in devices:
            supported[name] += 1
        if devices == {"MLCPUComputeDevice"}:
            cpu_only.append(str(getattr(item, "operator_name", "unknown")))
    return {
        "operation_count": sum(preferred.values()),
        "preferred": dict(sorted(preferred.items())),
        "supported": dict(sorted(supported.items())),
        "cpu_only_operator_count": len(cpu_only),
        "cpu_only_operators": sorted(set(cpu_only)),
    }


def compute_plan_passes(report: dict[str, Any], unit: str) -> bool:
    operation_count = int(report.get("operation_count", 0))
    if operation_count <= 0 or int(report.get("cpu_only_operator_count", 0)) > 0:
        return False
    expected = {
        "gpu": "MLGPUComputeDevice",
        "ane": "MLNeuralEngineComputeDevice",
    }.get(unit)
    if expected is None:
        return False
    preferred = report.get("preferred") or {}
    return int(preferred.get(expected, 0)) == operation_count


def _coreml_compute_unit(unit: str):
    import coremltools as ct

    units = {
        "ane": ct.ComputeUnit.CPU_AND_NE,
        "gpu": ct.ComputeUnit.CPU_AND_GPU,
    }
    try:
        return units[unit]
    except KeyError as exc:
        raise CoreMLFamilyAccelerationError(
            f"Core ML accelerator selection must be gpu or ane, got {unit!r}"
        ) from exc


def _inspect_compute_plans(model_path: Path, requested: str) -> dict[str, Any]:
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan

    load_unit = "gpu" if requested == "auto" else requested
    loaded = ct.models.MLModel(
        str(model_path),
        compute_units=_coreml_compute_unit(load_unit),
    )
    compiled_path = loaded.get_compiled_model_path()
    candidates: dict[str, Any] = {}
    units_to_inspect = ("ane", "gpu") if requested == "auto" else (requested,)
    for unit in units_to_inspect:
        plan = MLComputePlan.load_from_path(
            compiled_path,
            compute_units=_coreml_compute_unit(unit),
        )
        operations = plan.model_structure.program.functions["main"].block.operations
        usages = []
        for operation in operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(operation=operation)
            if usage is not None:
                usages.append(
                    SimpleNamespace(
                        preferred_compute_device=usage.preferred_compute_device,
                        supported_compute_devices=usage.supported_compute_devices,
                        operator_name=operation.operator_name,
                    )
                )
        report = summarize_compute_plan_usages(usages)
        report["passes"] = compute_plan_passes(report, unit)
        candidates[unit] = report

    if requested == "auto":
        selected = "ane" if candidates["ane"]["passes"] else "gpu"
    elif requested in {"ane", "gpu"}:
        selected = requested
    else:
        raise CoreMLFamilyAccelerationError(
            f"unsupported Core ML compute unit {requested!r}; use auto, gpu, or ane"
        )
    if not candidates[selected]["passes"]:
        raise CoreMLFamilyAccelerationError(
            f"{selected.upper()} compute plan failed accelerator-only gate: "
            f"{json.dumps(candidates[selected], ensure_ascii=False)}"
        )
    return {
        "requested": requested,
        "selected": selected,
        "candidates": candidates,
    }


def _weight_fingerprint(provider: _FamilyProvider) -> list[dict[str, Any]]:
    rows = []
    for path in provider.spec.weight_paths(provider.cache_dir):
        stat = path.stat() if path.is_file() else None
        rows.append(
            {
                "path": str(path),
                "size": int(stat.st_size) if stat else 0,
                "mtime_ns": int(stat.st_mtime_ns) if stat else 0,
            }
        )
    return rows


class _CoreMLFamilyProvider(_FamilyProvider):
    """Family provider whose frame path is Core ML only, with fail-closed gating."""

    strict_native_io = True

    def __init__(
        self,
        *,
        use_coreml: bool = True,
        recompile_coreml: bool = False,
        coreml_compute_units: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.use_coreml = bool(use_coreml)
        self.recompile_coreml = bool(recompile_coreml)
        requested_units = str(
            coreml_compute_units
            or os.environ.get("D2S_COREML_COMPUTE_UNITS", "auto")
        ).strip().lower()
        if requested_units in {"auto", "all", "cpu_and_gpu", "gpu"}:
            self.coreml_compute_units = "gpu"
        elif requested_units in {"ane", "cpu_and_ne", "cpu"}:
            raise CoreMLFamilyAccelerationError(
                f"{self.spec.name} is GPU-only; refusing compute units {requested_units!r}"
            )
        else:
            raise CoreMLFamilyAccelerationError(
                f"unsupported Core ML compute units {requested_units!r}; use GPU"
            )
        self._coreml_engines: dict[tuple[int, int], CoreMLEngine] = {}
        self._coreml_plan: dict[str, Any] | None = None

    @property
    def pipeline_slot_count(self) -> int:
        return 1

    def native_temporal_history_weight(self) -> float:
        if self.spec.name != "DepthPro2-S":
            return 0.0
        raw = os.environ.get("D2S_COREML_DP2_S_TEMPORAL_WEIGHT", "0.85")
        try:
            return min(0.95, max(0.0, float(raw)))
        except ValueError:
            return 0.85

    def load(self) -> None:
        if not self.use_coreml:
            raise CoreMLFamilyAccelerationError(
                f"{self.spec.name} requires Core ML on the macOS accelerated path"
            )

    def _load(self) -> Any:
        raise NotImplementedError

    def _wrap_model(self, model: torch.nn.Module, input_height: int, input_width: int) -> torch.nn.Module:
        raise NotImplementedError

    def _input_shape(self, rgb: torch.Tensor) -> tuple[int, int]:
        _, _, height, width = ensure_bchw(rgb, name="rgb").shape
        return _model_input_size_for_model(
            height,
            width,
            self.depth_resolution,
            FAMILY_INPUT_MULTIPLE[self.family],
            self.spec.model_id,
        )

    def _prepare_input(self, rgb: torch.Tensor) -> torch.Tensor:
        rgb = ensure_bchw(rgb, name="rgb").to(self.device).float()
        rgb = torch.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        input_height, input_width = self._input_shape(rgb)
        if tuple(rgb.shape[-2:]) != (input_height, input_width):
            rgb = F.interpolate(
                rgb,
                size=(input_height, input_width),
                mode="bilinear",
                align_corners=False,
                antialias=False,
            )
        mean, std = _normalization_tensors_for_model(
            self.spec.model_id,
            self.device,
            torch.float32,
        )
        return (rgb - mean) / std

    def _artifact_paths(
        self, input_height: int, input_width: int, precision: str
    ) -> tuple[Path, Path]:
        root = self.cache_dir / "coreml" / _safe_model_tag(self.spec.name)
        stem = f"model_{precision}_{int(input_height)}x{int(input_width)}"
        return root / f"{stem}.mlpackage", root / f"{stem}.manifest.json"

    def _manifest_payload(
        self, input_height: int, input_width: int, precision: str
    ) -> dict[str, Any]:
        import coremltools as ct

        return {
            "model_name": self.spec.name,
            "model_id": self.spec.model_id,
            "family": self.family,
            "input_height": int(input_height),
            "input_width": int(input_width),
            "compute_precision": precision.upper(),
            "minimum_deployment_target": "macOS15",
            "coremltools": str(ct.__version__),
            "torch": str(torch.__version__),
            "weights": _weight_fingerprint(self),
            "output_semantics": "relative_inverse_depth",
            "source_output_semantics": self.spec.output_semantics,
            "requested_compute_units": self.coreml_compute_units,
            "export_schema_version": FAMILY_EXPORT_SCHEMA_VERSION,
        }

    def _compile_engine(self, input_height: int, input_width: int) -> CoreMLEngine:
        import coremltools as ct
        import numpy as np

        # Core MLTools 9 crashes or stalls compiling FP16 graphs for the
        # DINOv3/DINOv2 transformer families. FP32 GPU is the reliable
        # accelerator-only path for these artifacts.
        precision = "fp32"
        model_path, manifest_path = self._artifact_paths(
            input_height, input_width, precision
        )
        manifest = self._manifest_payload(input_height, input_width, precision)
        cached_manifest = None
        cached = model_path.is_dir() and not self.recompile_coreml
        if cached and manifest_path.is_file():
            try:
                cached_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                cached_manifest = None
            cached_plan = (cached_manifest or {}).get("compute_plan") or {}
            cached_units = (
                (cached_manifest or {}).get("requested_compute_units")
                or cached_plan.get("selected")
            )
            cached = (
                cached_manifest is not None
                and cached_manifest.get("export_schema_version", 1)
                == FAMILY_EXPORT_SCHEMA_VERSION
                and cached_units == self.coreml_compute_units
                and all(
                    cached_manifest.get(key) == value
                    for key, value in manifest.items()
                    if key not in {"requested_compute_units", "export_schema_version"}
                )
            )

        selected_units = self.coreml_compute_units
        plan = None
        if not cached:
            install_coremltools_workarounds()
            previous_dtype = self.dtype
            self.dtype = torch.float32
            torch_model = self._load()
            self.dtype = previous_dtype
            wrapped = self._wrap_model(torch_model, input_height, input_width).float().eval()
            dummy = torch.randn(
                1,
                3,
                int(input_height),
                int(input_width),
                device=self.device,
                dtype=torch.float32,
            )
            print(
                f"[CoreMLFamily] converting {self.spec.name} "
                f"{input_height}x{input_width}",
                flush=True,
            )
            started = time.perf_counter()
            with torch.no_grad(), coreml_safe_interpolate(), coreml_safe_dinov2_positions(
                torch_model
            ):
                traced = torch.jit.trace(wrapped, dummy, strict=False)
                traced = torch.jit.freeze(traced)
            mlmodel = ct.convert(
                traced,
                inputs=[
                    ct.TensorType(
                        name="pixel_values",
                        shape=tuple(dummy.shape),
                        dtype=np.float32,
                    )
                ],
                compute_precision=(
                    ct.precision.FLOAT32
                    if precision == "fp32"
                    else ct.precision.FLOAT16
                ),
                compute_units=ct.ComputeUnit.ALL,
                minimum_deployment_target=ct.target.macOS15,
            )
            model_path.parent.mkdir(parents=True, exist_ok=True)
            mlmodel.save(str(model_path))
            plan = _inspect_compute_plans(model_path, self.coreml_compute_units)
            selected_units = str(plan["selected"])
            manifest["compute_plan"] = plan
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            self._model = None
            if self.device.type == "mps":
                torch.mps.empty_cache()
            print(
                f"[CoreMLFamily] converted {self.spec.name} in "
                f"{time.perf_counter() - started:.1f}s units={selected_units}",
                flush=True,
            )
        else:
            if cached_manifest is None:
                cached_manifest = manifest
            plan = cached_manifest.get("compute_plan")
            if not plan:
                plan = _inspect_compute_plans(model_path, self.coreml_compute_units)
            cached_manifest = {**manifest, "compute_plan": plan}
            manifest_path.write_text(
                json.dumps(cached_manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            selected_units = str(plan["selected"])
            print(
                f"[CoreMLFamily] using cached {self.spec.name} "
                f"{input_height}x{input_width} units={selected_units}",
                flush=True,
            )

        self._coreml_plan = plan
        engine = CoreMLEngine(
            model_path,
            self.device,
            compute_units_choice=selected_units,
        )
        mean, std = _normalization_tensors_for_model(
            self.spec.model_id,
            torch.device("cpu"),
            torch.float32,
        )
        engine.enable_native_io(
            model_path,
            int(input_width),
            int(input_height),
            normalization=(
                tuple(float(value) for value in mean.flatten().tolist()),
                tuple(float(value) for value in std.flatten().tolist()),
            ),
            compute_units=selected_units,
            temporal_history_weight=self.native_temporal_history_weight(),
            temporal_change_threshold=0.1,
        )
        if engine.native_io is None:
            _ = engine.model
        self.info = self.info.__class__(
            **{
                **self.info.to_report(),
                "provider": f"stereo_runtime.providers.apple.coreml_families.{self.family}",
                "depth_backend": "coreml",
                "runtime": "coreml",
                "execution_provider": "Apple Core ML",
                "fallback_reason": None,
                "native_io_binding": engine.native_io is not None,
                "native_io_reason": engine.native_io_reason,
                "raw_output_semantics": "relative_inverse_depth",
                "coreml_compute_units": selected_units,
                "coreml_compute_plan": plan,
                "coreml_artifact_path": str(model_path),
                "temporal_history_weight": self.native_temporal_history_weight(),
            }
        )
        return engine

    def _engine_for_frame(self, rgb: torch.Tensor) -> CoreMLEngine:
        input_height, input_width = self._input_shape(rgb)
        key = (int(input_height), int(input_width))
        engine = self._coreml_engines.get(key)
        if engine is None:
            engine = self._compile_engine(*key)
            self._coreml_engines[key] = engine
        return engine

    def _postprocess(self, predicted: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        depth = _normalize_depth_canonical(ensure_bchw(predicted, name="predicted_depth").float())
        _, _, height, width = ensure_bchw(rgb, name="rgb").shape
        return _upsample_depth_canonical(
            depth,
            height,
            width,
            rgb=rgb,
            mode=self.depth_upsample,
            edge_strength=self.depth_upsample_edge_strength,
        )

    def predict_profile(self, rgb: torch.Tensor) -> DepthProfileResult:
        rgb = ensure_bchw(rgb, name="rgb")
        engine = self._engine_for_frame(rgb)
        start = time.perf_counter()
        tensor = self._prepare_input(rgb)
        if self.device.type == "mps":
            torch.mps.synchronize()
        preprocess_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        output = engine(pixel_values=tensor)
        if self.device.type == "mps":
            torch.mps.synchronize()
        model_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        depth = self._postprocess(output.predicted_depth, rgb)
        if self.device.type == "mps":
            torch.mps.synchronize()
        postprocess_ms = (time.perf_counter() - start) * 1000.0
        element_size = depth.element_size()
        return DepthProfileResult(
            depth,
            preprocess_ms,
            model_ms,
            postprocess_ms,
            finite_depth=True,
            nonfinite_count=int(getattr(output, "nonfinite_count", 0)),
            raw_output_semantics="relative_inverse_depth",
            model_profile=self.model_profile,
            zero_copy_output=False,
            host_readback_bytes=int(output.predicted_depth.numel() * output.predicted_depth.element_size())
            if isinstance(output.predicted_depth, torch.Tensor)
            else int(depth.numel() * element_size),
        )

    def predict_profile_native(
        self,
        rgb: torch.Tensor,
        pixel_buffer: Any,
        frame_id: int,
    ) -> DepthProfileResult | None:
        if pixel_buffer is None:
            raise CoreMLFamilyAccelerationError(
                f"{self.spec.name} requires a native CVPixelBuffer for GPU-only inference"
            )
        engine = self._engine_for_frame(rgb)
        native_frame = engine.predict_native(pixel_buffer, int(frame_id))
        if native_frame is None:
            raise CoreMLFamilyAccelerationError(
                f"{self.spec.name} native Core ML bridge is unavailable; "
                "refusing host or PyTorch fallback"
            )
        self.info = self.info.__class__(
            **{
                **self.info.to_report(),
                "native_io_binding": bool(native_frame.input_shared),
                "native_output_backing": bool(native_frame.output_backing_used),
                "native_io_reason": "active",
            }
        )
        return DepthProfileResult(
            depth=native_frame,
            preprocess_ms=float(native_frame.preprocess_ms),
            model_ms=float(native_frame.model_ms),
            postprocess_ms=float(native_frame.postprocess_ms),
            finite_depth=bool(native_frame.finite_depth),
            nonfinite_count=int(native_frame.nonfinite_count),
            raw_output_semantics="relative_inverse_depth",
            model_profile=self.model_profile,
            native_depth=native_frame,
            native_resource_handle=native_frame,
            native_zero_copy=bool(native_frame.output_zero_copy),
            zero_copy_output=bool(native_frame.output_zero_copy),
            host_readback_bytes=0,
        )

    def native_io_ready_for_frame(self, width: int, height: int) -> bool:
        probe = torch.empty((1, 3, int(height), int(width)), device="meta", dtype=torch.float32)
        try:
            engine = self._engine_for_frame(probe)
            return engine.native_io is not None
        except Exception as exc:  # noqa: BLE001 - preflight reports every backend failure
            self.info = self.info.__class__(
                **{
                    **self.info.to_report(),
                    "fallback_reason": f"{type(exc).__name__}: {exc}",
                }
            )
            return False

    def close(self) -> None:
        for engine in tuple(self._coreml_engines.values()):
            engine.close()
        self._coreml_engines.clear()
        self._model = None


class DepthPro2CoreMLProvider(_CoreMLFamilyProvider):
    family = "depth-pro-2"

    def _load(self) -> torch.nn.Module:
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

    def _wrap_model(self, model: torch.nn.Module, input_height: int, input_width: int) -> torch.nn.Module:
        return _DepthPro2CoreMLWrapper(model)


class MoGe3CoreMLProvider(_CoreMLFamilyProvider):
    family = "moge-3"

    def _load(self) -> torch.nn.Module:
        model, spec = load_moge_3(
            model_name_or_id=self.spec.name,
            cache_dir=self.cache_dir,
            device=self.device,
            compat_profile=True,
            local_files_only=self.local_files_only,
            force_download=self.force_download,
        )
        self.spec = spec
        return model

    def _wrap_model(self, model: torch.nn.Module, input_height: int, input_width: int) -> torch.nn.Module:
        return _MoGe3CoreMLWrapper(model, max(int(input_height), int(input_width)))


class FlashDepthCoreMLProvider(_CoreMLFamilyProvider):
    family = "flashdepth"

    def _load(self) -> torch.nn.Module:
        model, spec = load_flashdepth(
            model_name_or_id=self.spec.name,
            cache_dir=self.cache_dir,
            device=self.device,
            local_files_only=self.local_files_only,
            force_download=self.force_download,
        )
        self.spec = spec
        return model

    def _wrap_model(self, model: torch.nn.Module, input_height: int, input_width: int) -> torch.nn.Module:
        return _FlashDepthCoreMLWrapper(model, input_height, input_width)


def create_coreml_family_provider(
    *,
    family: str,
    model_id: str,
    model_name: str | None = None,
    **kwargs: Any,
) -> _CoreMLFamilyProvider:
    classes = {
        "depth-pro-2": DepthPro2CoreMLProvider,
        "moge-3": MoGe3CoreMLProvider,
        "flashdepth": FlashDepthCoreMLProvider,
    }
    try:
        provider_cls = classes[family]
    except KeyError as exc:
        raise ValueError(f"unsupported Core ML depth family: {family}") from exc
    return provider_cls(model_id=model_id, model_name=model_name, **kwargs)


def coreml_family_available() -> bool:
    return sys.platform == "darwin"


__all__ = [
    "FAMILY_EXPORT_SCHEMA_VERSION",
    "CoreMLFamilyAccelerationError",
    "DepthPro2CoreMLProvider",
    "FlashDepthCoreMLProvider",
    "MoGe3CoreMLProvider",
    "compute_plan_passes",
    "coreml_family_available",
    "create_coreml_family_provider",
    "depth_to_d2s_response",
    "summarize_compute_plan_usages",
]
