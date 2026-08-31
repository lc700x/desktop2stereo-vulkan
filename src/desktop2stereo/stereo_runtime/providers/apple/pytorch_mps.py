from __future__ import annotations

import contextlib
import sys
import types
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch

from stereo_runtime.depth_provider import (
    DISTILL_ANY_DEPTH_BASE_MODEL_ID,
    DISTILL_ANY_DEPTH_BASE_RESOLUTION,
    DISTILL_ANY_DEPTH_PATCH_SIZE,
    DepthProviderInfo,
    DistillAnyDepthBase518,
    GenericAutoDepthProvider,
)
from stereo_runtime.depth_upsample import DepthUpsampleMode


def is_mps_torch_available() -> bool:
    try:
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


class _MpsInfoMixin:
    def _mark_mps_info(self, info: DepthProviderInfo) -> DepthProviderInfo:
        return replace(
            info,
            depth_backend="pytorch_mps",
            runtime="transformers-mps",
            execution_provider="Apple MPS PyTorch",
            fallback_reason=None if is_mps_torch_available() else "torch.backends.mps is not available",
            output_device=str(self.device),
        )


def is_coreml_available() -> bool:
    """CoreML acceleration is macOS-only and needs coremltools."""
    if sys.platform != "darwin":
        return False
    try:
        import coremltools  # noqa: F401

        return True
    except Exception:
        return False


@contextlib.contextmanager
def coreml_safe_interpolate():
    """Swap bicubic resampling to bilinear during TorchScript tracing.

    CoreML cannot represent bicubic interpolation; the patch only applies
    inside this block.
    """
    orig_interpolate = torch.nn.functional.interpolate

    def patched_interpolate(
        input,
        size=None,
        scale_factor=None,
        mode="nearest",
        align_corners=None,
        recompute_scale_factor=None,
        antialias=False,
    ):
        if mode == "bicubic":
            mode = "bilinear"
        return orig_interpolate(
            input,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor,
            antialias=antialias,
        )

    torch.nn.functional.interpolate = patched_interpolate
    try:
        yield
    finally:
        torch.nn.functional.interpolate = orig_interpolate


class ModelForCoreML(torch.nn.Module):
    """Return a single depth tensor so TorchScript tracing stays simple."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        if hasattr(self.model, "predict_depth"):
            out = self.model.predict_depth(x)
            if isinstance(out, torch.Tensor):
                return out
            raise RuntimeError("Unsupported predict_depth return type for CoreML export")
        try:
            out = self.model(pixel_values=x)
        except TypeError:
            out = self.model(x)
        if hasattr(out, "predicted_depth"):
            return out.predicted_depth
        if isinstance(out, dict):
            if "predicted_depth" in out:
                return out["predicted_depth"]
            for value in out.values():
                if isinstance(value, torch.Tensor):
                    return value
        if isinstance(out, (tuple, list)):
            if len(out) > 0 and isinstance(out[0], torch.Tensor):
                return out[0]
        raise RuntimeError("Unsupported model output type for CoreML export")


class CoreMLEngine:
    """Callable stand-in for the torch depth model.

    Returns an object exposing ``predicted_depth`` (on the provider device) so
    both provider base classes run their normal postprocessing unchanged.
    """

    def __init__(self, model_path: str | Path, device: str | torch.device) -> None:
        import coremltools as ct

        # ComputeUnit.ALL lets CoreML schedule depth on the GPU, whose Metal
        # work then serializes with the MPS synthesis stream ACROSS frames
        # (frame N's host pull drains frame N-1's CoreML-GPU tail). Pinning
        # to ANE(+CPU) frees the GPU for torch MPS entirely.
        # D2S_COREML_COMPUTE_UNITS: all (default) | ane | cpu_and_gpu
        import os

        choice = os.environ.get("D2S_COREML_COMPUTE_UNITS", "all").strip().lower()
        units = {
            "ane": getattr(ct.ComputeUnit, "CPU_AND_NE", ct.ComputeUnit.ALL),
            "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
            "cpu": ct.ComputeUnit.CPU_ONLY,
        }.get(choice, ct.ComputeUnit.ALL)
        self.model = ct.models.MLModel(
            str(model_path),
            compute_units=units,
        )
        self.device = torch.device(device)

    def __call__(self, pixel_values: torch.Tensor):
        import numpy as np

        # Skip the detach/cpu round trip when the caller already hands us a
        # contiguous CPU float32 tensor (from_numpy/numpy share memory).
        if (
            isinstance(pixel_values, torch.Tensor)
            and pixel_values.device.type == "cpu"
            and pixel_values.dtype == torch.float32
            and pixel_values.is_contiguous()
        ):
            np_input = pixel_values.numpy()
        else:
            np_input = pixel_values.detach().cpu().numpy()
        out = self.model.predict({"pixel_values": np_input})
        value = next(iter(out.values())) if isinstance(out, dict) else out
        # from_numpy wraps the prediction buffer without a copy; only pay the
        # device transfer when postprocessing actually runs off-CPU.
        tensor = torch.from_numpy(np.ascontiguousarray(value))
        if self.device.type != "cpu":
            tensor = tensor.to(self.device)
        if tensor.is_floating_point():
            # fp16/fp32 engines can emit NaN/Inf on pathological inputs
            # (denormal overflow, 0/0 in attention). Sanitize here so depth
            # normalization downstream never sees non-finite values; far
            # plane (1.0) is the conservative fallback for a poisoned pixel.
            tensor = torch.nan_to_num(tensor, nan=1.0, posinf=1.0, neginf=0.0)
        return SimpleNamespace(predicted_depth=tensor)


def _safe_model_tag(model_id: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in str(model_id)).strip("-")


def install_coremltools_workarounds() -> bool:
    """Patch coremltools' aten::int lowering for torch >= 2.8 traces.

    Newer tracers feed length-1 constant arrays into ``aten::int``; stock
    coremltools calls ``int(x.val)`` on the array and crashes with
    "only 0-dimensional arrays can be converted to Python scalars".
    Idempotent and limited to this process.
    """
    try:
        import numpy as np
        from coremltools.converters.mil.frontend.torch import torch_op_registry as treg
        from coremltools.converters.mil.frontend.torch import ops as tops
        from coremltools.converters.mil import Builder as mb
    except Exception:
        return False

    if getattr(tops, "_d2s_int_cast_patched", False):
        return True

    def _safe_int_cast(context, node):
        inputs = tops._get_inputs(context, node, expected=1)
        x = inputs[0]
        if not (len(x.shape) == 0 or all(d == 1 for d in x.shape)):
            raise ValueError("input to cast must be either a scalar or a length 1 tensor")
        if x.can_be_folded_to_const():
            values = np.asarray(x.val).ravel()
            value = int(values[0]) if values.size else 0
            res = mb.const(val=value, name=node.name)
        elif len(x.shape) > 0:
            squeezed = mb.squeeze(x=x, name=node.name + "_item")
            res = mb.cast(x=squeezed, dtype="int32", name=node.name)
        else:
            res = mb.cast(x=x, dtype="int32", name=node.name)
        context.add(res, node.name)

    registry = getattr(treg, "_TORCH_OPS_REGISTRY", None)
    mapping = getattr(registry, "name_to_func_mapping", None)
    if mapping is not None:
        mapping["int"] = _safe_int_cast
    else:
        treg["int"] = _safe_int_cast
    tops._d2s_int_cast_patched = True
    return True


@contextlib.contextmanager
def coreml_safe_dinov2_positions(model):
    """Use pure-Python ints for DINOv2 positional-embedding reshapes.

    transformers builds the sqrt(num_positions) dims with ``torch_int()``,
    which emits tensor-typed reshape sizes that trip coremltools on newer
    torch versions.
    """
    patched = []
    try:
        backbone = getattr(model, "backbone", None)
        emb = getattr(backbone, "embeddings", None)
        if emb is not None and hasattr(emb, "position_embeddings"):

            def static_interpolate_pos_encoding(self, embeddings, height, width):
                num_positions = self.position_embeddings.shape[1] - 1
                if (
                    not torch.jit.is_tracing()
                    and embeddings.shape[1] - 1 == num_positions
                    and height == width
                ):
                    return self.position_embeddings
                class_pos_embed = self.position_embeddings[:, :1]
                patch_pos_embed = self.position_embeddings[:, 1:]
                dim = embeddings.shape[-1]
                new_height = height // self.patch_size
                new_width = width // self.patch_size
                sqrt_num_positions = int(num_positions**0.5)
                patch_pos_embed = patch_pos_embed.reshape(
                    1, sqrt_num_positions, sqrt_num_positions, dim
                ).permute(0, 3, 1, 2)
                target_dtype = patch_pos_embed.dtype
                patch_pos_embed = torch.nn.functional.interpolate(
                    patch_pos_embed.to(torch.float32),
                    size=(new_height, new_width),
                    mode="bilinear",
                    align_corners=False,
                ).to(dtype=target_dtype)
                patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
                return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

            original = getattr(emb, "_d2s_original_interpolate", None)
            if original is None:
                original = emb.interpolate_pos_encoding
                emb._d2s_original_interpolate = original
            if isinstance(emb.interpolate_pos_encoding, types.MethodType) and getattr(
                emb.interpolate_pos_encoding.__func__, "_d2s_static", False
            ):
                pass
            else:
                static_interpolate_pos_encoding._d2s_static = True
                emb.interpolate_pos_encoding = types.MethodType(
                    static_interpolate_pos_encoding, emb
                )
            patched.append(emb)
        yield
    finally:
        for emb in patched:
            original = getattr(emb, "_d2s_original_interpolate", None)
            if original is not None:
                emb.interpolate_pos_encoding = original


class _CoreMLMixin:
    """Lazily convert/load fixed-shape CoreML engines on Apple Silicon."""

    use_coreml: bool = False
    recompile_coreml: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import threading

        self._coreml_engine_lock = threading.Lock()

    @property
    def pipeline_slot_count(self) -> int:
        # MLModel.predict is documented thread-safe; depth fast path has no
        # temporal cross-frame state, so two slots can overlap safely.
        return 2 if self._coreml_enabled() else 1

    def _coreml_enabled(self) -> bool:
        return bool(getattr(self, "use_coreml", False)) and sys.platform == "darwin"

    def _coreml_engine_for_frame(self, rgb: torch.Tensor):
        from stereo_runtime.depth_provider import (
            DISTILL_ANY_DEPTH_BASE_RESOLUTION,
            DISTILL_ANY_DEPTH_PATCH_SIZE,
            _model_input_size,
            ensure_bchw,
        )

        rgb = ensure_bchw(rgb, name="rgb")
        _, _, height, width = rgb.shape
        resolution = int(
            getattr(self, "depth_resolution", None) or DISTILL_ANY_DEPTH_BASE_RESOLUTION
        )
        patch_size = int(getattr(self, "patch_size", None) or 14)
        input_h, input_w = _model_input_size(height, width, resolution, patch_size)

        engines = getattr(self, "_coreml_engines", None)
        if engines is None:
            engines = {}
            self._coreml_engines = engines
        key = (int(input_h), int(input_w))
        if key in engines:
            return engines[key]

        # Parallel depth slots may race the one-time conversion; hold a lock
        # for the create path and re-check the cache inside it.
        with getattr(self, "_coreml_engine_lock", contextlib.nullcontext()):
            if key in engines:
                return engines[key]
            return self._coreml_engine_create_locked(key, engines)

    def _coreml_engine_create_locked(self, key, engines):
        # ``key`` is the (input_h, input_w) engine-size tuple computed by
        # _coreml_engine_for_frame; those names were previously referenced
        # here without being defined, raising NameError on EVERY frame and
        # silently knocking depth back to plain PyTorch MPS.
        input_h, input_w = int(key[0]), int(key[1])
        dtype_tag = "fp16" if self.dtype == torch.float16 else "fp32"
        model_dir = Path(str(self.cache_dir)) / "coreml" / _safe_model_tag(self.model_id)
        model_path = model_dir / f"model_{dtype_tag}_{input_h}x{input_w}.mlpackage"

        engine = None
        try:
            import coremltools as ct
        except Exception:
            print("[CoreML] coremltools is not installed; using PyTorch MPS", flush=True)
            engines[key] = None
            return None

        if not model_path.exists() or self.recompile_coreml:
            print(
                f"[CoreML] Compiling {model_path.name} "
                "(this may take a while on first run)...",
                flush=True,
            )
            install_coremltools_workarounds()
            torch_model = self.load()
            wrapped = ModelForCoreML(torch_model).float().eval()
            # Trace on the same device as the model weights (MPS); the
            # converted mlpackage is device-independent.
            trace_device = self.device if self.device.type != "cpu" else "cpu"
            dummy = torch.randn(1, 3, input_h, input_w, device=trace_device, dtype=torch.float32)
            try:
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
                            shape=dummy.shape,
                            dtype=__import__("numpy").float32,
                        )
                    ],
                    compute_precision=ct.precision.FLOAT16,
                    compute_units=ct.ComputeUnit.ALL,
                    minimum_deployment_target=ct.target.macOS13,
                )
                model_dir.mkdir(parents=True, exist_ok=True)
                mlmodel.save(str(model_path))
                print(f"[CoreML] Model saved to {model_path}", flush=True)
            except Exception as exc:
                print(f"[CoreML] conversion failed ({exc}); using PyTorch MPS", flush=True)
                engines[key] = None
                return None
        else:
            print(f"[CoreML] Using cached model {model_path.name}", flush=True)

        engine = CoreMLEngine(model_path, self.device)
        print("[CoreML] Ready (compute units: ALL)", flush=True)
        engines[key] = engine
        return engine

    def _run_depth_model(self, tensor: torch.Tensor):
        # Route only the model call through CoreML; base pre/post stays.
        # No shared-state mutation (the old self._model swap raced under
        # parallel depth slots).
        if not self._coreml_enabled():
            return super()._run_depth_model(tensor)
        engine = self._coreml_engine_for_frame(tensor)
        if engine is None:
            return super()._run_depth_model(tensor)
        with torch.inference_mode():
            return engine(pixel_values=tensor)


class DistillAnyDepthBaseMps(_CoreMLMixin, _MpsInfoMixin, DistillAnyDepthBase518):
    def __init__(
        self,
        *,
        device: str | torch.device = "mps",
        cache_dir: str | Path | None = None,
        dtype: torch.dtype | None = None,
        local_files_only: bool = False,
        force_download: bool = False,
        depth_resolution: int = DISTILL_ANY_DEPTH_BASE_RESOLUTION,
        patch_size: int | None = DISTILL_ANY_DEPTH_PATCH_SIZE,
        depth_upsample: DepthUpsampleMode = "bilinear",
        depth_upsample_edge_strength: float = 0.35,
        use_coreml: bool = False,
        recompile_coreml: bool = False,
    ) -> None:
        super().__init__(
            device=device,
            cache_dir=cache_dir,
            dtype=dtype,
            local_files_only=local_files_only,
            force_download=force_download,
            depth_upsample=depth_upsample,
            depth_upsample_edge_strength=depth_upsample_edge_strength,
        )
        self.model_id = DISTILL_ANY_DEPTH_BASE_MODEL_ID
        self.depth_resolution = int(depth_resolution)
        self.patch_size = patch_size
        self.use_coreml = bool(use_coreml)
        self.recompile_coreml = bool(recompile_coreml)
        self.info = self._mark_mps_info(self.info)


class GenericAutoDepthMpsProvider(_CoreMLMixin, _MpsInfoMixin, GenericAutoDepthProvider):
    def __init__(
        self,
        *,
        model_id: str,
        model_name: str | None = None,
        device: str | torch.device = "mps",
        cache_dir: str | Path | None = None,
        dtype: torch.dtype | None = None,
        depth_resolution: int = 518,
        patch_size: int | None = 14,
        local_files_only: bool = False,
        force_download: bool = False,
        depth_upsample: DepthUpsampleMode = "bilinear",
        depth_upsample_edge_strength: float = 0.35,
        use_coreml: bool = False,
        recompile_coreml: bool = False,
    ) -> None:
        super().__init__(
            model_id=model_id,
            model_name=model_name,
            device=device,
            cache_dir=cache_dir,
            dtype=dtype,
            depth_resolution=depth_resolution,
            patch_size=patch_size,
            local_files_only=local_files_only,
            force_download=force_download,
            depth_upsample=depth_upsample,
            depth_upsample_edge_strength=depth_upsample_edge_strength,
        )
        self.use_coreml = bool(use_coreml)
        self.recompile_coreml = bool(recompile_coreml)
        self.info = self._mark_mps_info(self.info)


TorchMpsDepthProvider = DistillAnyDepthBaseMps
GenericTorchMpsDepthProvider = GenericAutoDepthMpsProvider


def create_pytorch_mps_provider(
    *,
    model_id: str = DISTILL_ANY_DEPTH_BASE_MODEL_ID,
    model_name: str | None = None,
    device: str | torch.device = "mps",
    cache_dir: str | Path | None = None,
    depth_resolution: int = 518,
    patch_size: int | None = 14,
    local_files_only: bool = True,
    force_download: bool = False,
    depth_upsample: DepthUpsampleMode = "bilinear",
    depth_upsample_edge_strength: float = 0.35,
    use_coreml: bool = False,
    recompile_coreml: bool = False,
):
    if model_id != DISTILL_ANY_DEPTH_BASE_MODEL_ID:
        return GenericAutoDepthMpsProvider(
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
            use_coreml=use_coreml,
            recompile_coreml=recompile_coreml,
        )
    return DistillAnyDepthBaseMps(
        device=device,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        force_download=force_download,
        depth_resolution=depth_resolution,
        patch_size=patch_size,
        depth_upsample=depth_upsample,
        depth_upsample_edge_strength=depth_upsample_edge_strength,
        use_coreml=use_coreml,
        recompile_coreml=recompile_coreml,
    )


__all__ = [
    "CoreMLEngine",
    "GenericAutoDepthMpsProvider",
    "GenericTorchMpsDepthProvider",
    "DistillAnyDepthBaseMps",
    "TorchMpsDepthProvider",
    "coreml_safe_interpolate",
    "create_pytorch_mps_provider",
    "is_coreml_available",
    "is_mps_torch_available",
]
