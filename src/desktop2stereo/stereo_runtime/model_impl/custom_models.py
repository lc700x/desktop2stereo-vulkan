"""Adapters for the native model families added in the depth-model expansion.

The upstream projects expose different inference APIs and several of them
accept optional training-time dependencies.  Keep those differences here so
the runtime providers can use one small, explicit contract:

* ``load_*`` returns an ``eval()`` model on the requested device.
* ``predict_*_depth`` returns a ``[B, H, W]`` GPU tensor without a host
  round-trip.
* ONNX wrappers expose only the canonical RGB input and depth output.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ..model_registry import DepthModelSpec, ModelRegistry


class _HybridConfig(dict):
    """Both attribute and mapping access are required by FlashDepth."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resolve_spec(model_name_or_id: str) -> DepthModelSpec:
    return ModelRegistry.default().get(model_name_or_id)


def _weight_paths(
    spec: DepthModelSpec,
    cache_dir: str | Path,
    *,
    local_files_only: bool = False,
    force_download: bool = False,
) -> tuple[Path, ...]:
    from ..model_artifacts import ensure_model_downloaded

    root = ensure_model_downloaded(
        spec,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        force_download=force_download,
    )
    paths = spec.weight_paths(root)
    missing = [str(path) for path in paths if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        raise FileNotFoundError(
            "model weights are incomplete for "
            f"{spec.name}: {', '.join(missing)}"
        )
    return paths


def _module_dtype(device: torch.device) -> torch.dtype:
    if device.type == "mps":
        return torch.bfloat16
    if device.type == "cuda":
        index = device.index or torch.cuda.current_device()
        major, _ = torch.cuda.get_device_capability(index)
        # DepthPro2 is trained for bf16 autocast on Ampere and newer.  Turing
        # (including the 8 GB RTX 2070 target) needs the fp16 compatibility
        # profile instead.
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def load_depth_pro_2(
    *,
    model_name_or_id: str,
    cache_dir: str | Path,
    device: str | torch.device,
    dtype: torch.dtype | None = None,
    local_files_only: bool = False,
    force_download: bool = False,
) -> tuple[Any, DepthModelSpec]:
    from .depth_pro_2.model import PRESETS, create_model_and_transforms

    torch_device = torch.device(device)
    spec = resolve_spec(model_name_or_id)
    weights = _weight_paths(
        spec,
        cache_dir,
        local_files_only=local_files_only,
        force_download=force_download,
    )
    preset_name = spec.name.lower()
    if preset_name not in PRESETS:
        raise ValueError(f"unknown DepthPro2 preset: {spec.name}")
    config = replace(
        PRESETS[preset_name],
        repo_id=None,
        subfolder=None,
        checkpoint_uri=str(weights[0]),
        fov_checkpoint_uri=str(weights[1]),
    )
    precision = dtype or _module_dtype(torch_device)
    model, _ = create_model_and_transforms(
        config,
        device=torch_device,
        precision=precision,
    )
    return model.eval(), spec


def load_moge_3(
    *,
    model_name_or_id: str,
    cache_dir: str | Path,
    device: str | torch.device,
    compat_profile: bool = True,
    local_files_only: bool = False,
    force_download: bool = False,
) -> tuple[Any, DepthModelSpec]:
    from .moge.model.v3 import MoGeModel

    torch_device = torch.device(device)
    spec = resolve_spec(model_name_or_id)
    weights = _weight_paths(
        spec,
        cache_dir,
        local_files_only=local_files_only,
        force_download=force_download,
    )
    model_kwargs = {"refiner": None} if compat_profile else None
    model = MoGeModel.from_pretrained(
        str(weights[0]),
        model_kwargs=model_kwargs,
    )
    return model.to(torch_device).eval(), spec


def _moge_num_tokens(model: Any, target_resolution: int) -> int:
    minimum, maximum = getattr(model, "num_tokens_range", (1200, 3600))
    scale = max(0.0, min(1.0, float(target_resolution) / 518.0))
    return max(int(minimum), round(minimum + scale * scale * (maximum - minimum)))


def _moge_point_map_to_depth(
    points: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Recover metric Z depth without the NumPy focal solver.

    MoGe's official utility downsamples a point map and calls NumPy to solve
    focal/shift.  That would introduce a host readback in the middle of the
    inference path.  The same least-squares system is solved here with
    batched PyTorch operations, keeping the output on the model device.
    """
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError(f"MoGe point map must be [B,H,W,3], got {tuple(points.shape)}")

    from .moge.utils.geometry_torch import normalized_view_plane_uv

    batch, height, width, _ = points.shape
    uv = normalized_view_plane_uv(
        width,
        height,
        aspect_ratio=width / height,
        dtype=points.dtype,
        device=points.device,
    ).unsqueeze(0).expand(batch, -1, -1, -1)
    valid = (
        torch.ones((batch, height, width), dtype=torch.bool, device=points.device)
        if mask is None
        else mask.to(device=points.device, dtype=torch.bool)
    )
    weight = valid.to(dtype=points.dtype).unsqueeze(-1)
    # The official solver uses rows [x*z, y*z] = [x, y, -uv] * [focal,
    # shift].  The weighted normal equations avoid a Python/NumPy transfer.
    a = torch.stack((points[..., :2], -uv), dim=-1) * weight.unsqueeze(-1)
    rhs = (uv * points[..., 2:]).unsqueeze(-1) * weight.unsqueeze(-1)
    normal = (a.transpose(-1, -2) @ a).sum(dim=(1, 2))
    normal = normal + torch.eye(2, dtype=points.dtype, device=points.device).view(1, 2, 2) * 1e-6
    rhs_normal = (a.transpose(-1, -2) @ rhs).sum(dim=(1, 2)).squeeze(-1)
    # Explicit 2x2 solve keeps the graph exportable to ONNX/MIGraphX and
    # avoids a solver kernel for a problem that is always 2x2.
    determinant = normal[:, 0, 0] * normal[:, 1, 1] - normal[:, 0, 1] * normal[:, 1, 0]
    determinant = determinant.clamp_min(1e-8)
    shift = (
        normal[:, 0, 0] * rhs_normal[:, 1]
        - normal[:, 1, 0] * rhs_normal[:, 0]
    ) / determinant
    depth = points[..., 2] + shift[..., None, None]
    # MoGe's official infer() marks invalid pixels as infinite Z. Preserve that
    # far-plane meaning so the shared Z-to-disparity conversion yields far=0
    # instead of incorrectly turning masked pixels into near foreground.
    return torch.where(valid, depth, torch.full_like(depth, torch.inf))


def predict_moge_3_depth(
    model: Any,
    rgb: torch.Tensor,
    *,
    target_resolution: int = 518,
    apply_mask: bool = True,
) -> torch.Tensor:
    """Run the stateless MoGe-3 path and return metric Z depth ``[B,H,W]``."""
    rgb = rgb.to(dtype=model.dtype, device=model.device).clamp_(0.0, 1.0)
    num_tokens = _moge_num_tokens(model, target_resolution)
    with torch.inference_mode(), torch.autocast(
        device_type=model.device.type,
        dtype=torch.float16,
        enabled=model.device.type in {"cuda", "mps"} and model.dtype != torch.float16,
    ):
        output = model.forward(
            rgb,
            num_tokens=num_tokens,
            refine_steps=0,
            return_per_step=False,
        )
    points = output.get("points")
    if points is None:
        raise RuntimeError("MoGe-3 returned no point map")
    mask = output.get("mask") if apply_mask else None
    depth = _moge_point_map_to_depth(
        points.float(),
        mask=mask.float() if isinstance(mask, torch.Tensor) else None,
    )
    metric_scale = output.get("metric_scale")
    if isinstance(metric_scale, torch.Tensor):
        depth = depth * metric_scale.float()[:, None, None]
    return depth


def _flashdepth_config(variant: str) -> dict[str, Any]:
    lowered = variant.casefold()
    if lowered.endswith("-full"):
        vit_size = "vits"
        use_hybrid = True
        mamba_layer = [1]
    elif lowered.endswith("-l"):
        vit_size = "vitl"
        use_hybrid = False
        mamba_layer = [3]
    else:
        vit_size = "vits"
        use_hybrid = False
        mamba_layer = [1]
    return {
        "vit_size": vit_size,
        "patch_size": 14,
        "use_mamba": False,
        "downsample_mamba": [0.1],
        "mamba_in_dpt_layer": mamba_layer,
        "mamba_type": "add",
        "num_mamba_layers": 4,
        "mamba_d_conv": 4,
        "mamba_d_state": 256,
        "use_xlstm": False,
        "use_transformer_rnn": False,
        "training": False,
        "hybrid_configs": _HybridConfig(
            use_hybrid=use_hybrid,
            teacher_resolution=490,
            layers_to_skip=[1, 2, 3],
            num_blocks=4,
            mlp_expand=2,
            num_heads=2,
        ),
    }


def load_flashdepth(
    *,
    model_name_or_id: str,
    cache_dir: str | Path,
    device: str | torch.device,
    local_files_only: bool = False,
    force_download: bool = False,
) -> tuple[Any, DepthModelSpec]:
    from .flashdepth_upstream.flashdepth.model import FlashDepth

    torch_device = torch.device(device)
    spec = resolve_spec(model_name_or_id)
    weights = _weight_paths(
        spec,
        cache_dir,
        local_files_only=local_files_only,
        force_download=force_download,
    )
    model = FlashDepth(**_flashdepth_config(spec.name))
    checkpoint = torch.load(str(weights[0]), map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
        checkpoint = checkpoint["model"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"unsupported FlashDepth checkpoint format: {weights[0]}")
    state_dict = {
        str(key).removeprefix("module."): value
        for key, value in checkpoint.items()
    }
    current_keys = set(model.state_dict())
    overlapping = current_keys.intersection(state_dict)
    if not overlapping:
        raise RuntimeError(
            f"FlashDepth checkpoint does not match {spec.name}: {weights[0]}"
        )
    model.load_state_dict(state_dict, strict=False)
    return model.to(torch_device).eval(), spec


def predict_flashdepth(
    model: Any,
    rgb: torch.Tensor,
) -> torch.Tensor:
    """Run the stateless FlashDepth compatibility path.

    The temporal Mamba path is deliberately disabled for the portable profile;
    the still-image DPT decoder is the same depth response used by the model's
    single-frame evaluation path.
    """
    rgb = rgb.to(dtype=next(model.parameters()).dtype, device=next(model.parameters()).device)
    height, width = int(rgb.shape[-2]), int(rgb.shape[-1])
    patch_h, patch_w = height // 14, width // 14
    if patch_h < 1 or patch_w < 1:
        raise ValueError(f"FlashDepth input is too small: {height}x{width}")
    with torch.inference_mode(), torch.autocast(
        device_type=rgb.device.type,
        dtype=torch.float16,
        enabled=rgb.device.type in {"cuda", "mps"} and rgb.dtype != torch.float16,
    ):
        features = model.get_dpt_features(
            rgb,
            input_shape=(int(rgb.shape[0]), 3, height, width),
        )
        return model.final_head(features, patch_h, patch_w)


def normalize_imagenet(rgb: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=rgb.device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=rgb.device, dtype=dtype).view(1, 3, 1, 1)
    return (rgb.to(dtype=dtype) - mean) / std


def resize_rgb(rgb: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if tuple(rgb.shape[-2:]) == (height, width):
        return rgb
    return F.interpolate(
        rgb,
        size=(height, width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )


__all__ = [
    "load_depth_pro_2",
    "load_flashdepth",
    "load_moge_3",
    "normalize_imagenet",
    "predict_flashdepth",
    "predict_moge_3_depth",
    "resize_rgb",
    "resolve_spec",
]
