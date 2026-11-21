"""MLX Metal depth provider for the Hugging Face Depth Anything models.

The provider deliberately accepts only the native ``DepthAnythingForDepthEstimation``
checkpoint family used by Desktop2Stereo. Model weights stay in MLX arrays and
frame tensors cross the PyTorch/MLX boundary through Metal DLPack, so the normal
inference path never materializes RGB or depth on the CPU.
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

import torch

from stereo_runtime.depth_provider import (
    DISTILL_ANY_DEPTH_BASE_RESOLUTION,
    DISTILL_ANY_DEPTH_PATCH_SIZE,
    DepthProfileResult,
    DepthProviderInfo,
    _model_input_size_for_model,
    _model_output_semantics,
    _normalization_tensors_for_model,
    _postprocess_generic_depth,
)
from stereo_runtime.depth_upsample import DepthUpsampleMode, _upsample_depth_canonical
from stereo_runtime.model_artifacts import (
    ensure_model_downloaded,
    find_local_model_snapshot,
    find_local_model_weight,
)
from stereo_runtime.model_registry import ModelRegistry
from stereo_runtime.output import ensure_bchw


try:
    import mlx.core as mx
    import mlx.nn as nn
except Exception:  # MLX is an optional macOS-only dependency.
    mx = None
    nn = None


_MlxModuleBase = nn.Module if nn is not None else object


def is_mlx_available() -> bool:
    """Return whether MLX can execute on the local Metal GPU."""
    if sys.platform != "darwin":
        return False
    try:
        import mlx.core as mlx_core

        return bool(
            mlx_core.is_available(mlx_core.gpu)
            and mlx_core.metal.is_available()
        )
    except Exception:
        return False


def _require_mlx() -> tuple[Any, Any]:
    if not is_mlx_available():
        raise RuntimeError(
            "MLX Metal is unavailable; install the macOS MLX package and use an "
            "Apple Silicon GPU"
        )
    if mx is None or nn is None:
        # This can happen when the module was imported before an optional
        # environment was activated. Re-importing the module is the supported
        # recovery path, but keep the error actionable for embedded callers.
        raise RuntimeError("MLX was not importable in this Python process")
    return mx, nn


@contextmanager
def _mlx_gpu_scope(mlx_core: Any) -> Iterator[None]:
    """Run MLX graph construction and execution on the Metal device."""
    with mlx_core.stream(mlx_core.gpu):
        yield


def _mlx_dtype() -> Any:
    choice = os.environ.get("D2S_MLX_DTYPE", "float16").strip().lower()
    if choice in {"float32", "fp32", "f32"}:
        return mx.float32
    if choice in {"bfloat16", "bf16"} and hasattr(mx, "bfloat16"):
        return mx.bfloat16
    return mx.float16


def _stage_weights_on_gpu(mlx_core: Any, weights: list[tuple[str, Any]], dtype: Any) -> list[tuple[str, Any]]:
    """Materialize safetensors as MLX arrays resident on the Metal stream."""
    staged: list[tuple[str, Any]] = []
    for name, value in weights:
        if value.dtype != dtype:
            value = value.astype(dtype)
        # MLX 0.32.x cannot evaluate safetensors loaded with stream=gpu. A
        # no-op GPU operation performs one device-local transfer at startup.
        staged.append((name, mlx_core.add(value, 0, stream=mlx_core.gpu)))
    return staged


def _import_torch_mps_dlpack(mlx_core: Any, tensor: torch.Tensor) -> tuple[Any, bool]:
    """Prefer shared Metal storage, with a device-local copy fallback."""
    try:
        return mlx_core.from_dlpack(tensor, copy=False), True
    except (RuntimeError, TypeError, ValueError):
        try:
            # PyTorch currently exports private MPS buffers. MLX rejects those
            # with copy=False, but copy=None still keeps the transfer on GPU.
            return mlx_core.from_dlpack(tensor, copy=None), False
        except Exception as exc:
            raise RuntimeError(
                "MLX could not import the PyTorch MPS tensor through Metal DLPack"
            ) from exc


def _export_torch_mps_dlpack(tensor: Any) -> tuple[torch.Tensor, bool]:
    """Prefer shared MLX/Metal storage when importing depth into PyTorch."""
    try:
        return torch.utils.dlpack.from_dlpack(tensor, copy=False), True
    except (RuntimeError, TypeError, ValueError):
        try:
            # Older MPS stacks may expose a private Metal buffer.  copy=None lets
            # PyTorch select a device-local copy without falling back to the CPU.
            return torch.utils.dlpack.from_dlpack(tensor, copy=None), False
        except Exception as exc:
            raise RuntimeError(
                "PyTorch could not import the MLX Metal depth tensor through DLPack"
            ) from exc


def _resize_nhwc(
    x: Any,
    height: int,
    width: int,
    *,
    mode: str = "linear",
    align_corners: bool = False,
) -> Any:
    if tuple(x.shape[1:3]) == (int(height), int(width)):
        return x
    return nn.Upsample(
        scale_factor=(float(height) / float(x.shape[1]), float(width) / float(x.shape[2])),
        mode=mode,
        align_corners=align_corners,
    )(x)


class _MlxConv2d(_MlxModuleBase):
    """Conv2d with PyTorch OIHW parameter layout and MLX NHWC execution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.weight = mx.zeros((out_channels, in_channels, *kernel_size), dtype=mx.float32)
        if bias:
            self.bias = mx.zeros((out_channels,), dtype=mx.float32)
        self.stride = stride
        self.padding = padding

    def __call__(self, x: Any) -> Any:
        weight = self.weight.transpose(0, 2, 3, 1)
        y = mx.conv2d(x, weight, self.stride, self.padding)
        if "bias" in self:
            y = y + self.bias.reshape(1, 1, 1, -1)
        return y


class _MlxConvTranspose2d(_MlxModuleBase):
    """ConvTranspose2d with PyTorch IOKK parameter layout."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, *, stride: int) -> None:
        super().__init__()
        self.weight = mx.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=mx.float32)
        self.bias = mx.zeros((out_channels,), dtype=mx.float32)
        self.stride = stride

    def __call__(self, x: Any) -> Any:
        weight = self.weight.transpose(1, 2, 3, 0)
        y = mx.conv_transpose2d(x, weight, self.stride)
        return y + self.bias.reshape(1, 1, 1, -1)


class _DinoLayerScale(_MlxModuleBase):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.lambda1 = mx.ones((hidden_size,), dtype=mx.float32)

    def __call__(self, x: Any) -> Any:
        return x * self.lambda1


class _DinoAttentionOutput(_MlxModuleBase):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)


class _DinoAttentionProjection(_MlxModuleBase):
    def __init__(self, hidden_size: int, heads: int) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.heads = int(heads)

    def __call__(self, x: Any) -> Any:
        batch, tokens, hidden = x.shape
        head_dim = hidden // self.heads
        query = self.query(x).reshape(batch, tokens, self.heads, head_dim).transpose(0, 2, 1, 3)
        key = self.key(x).reshape(batch, tokens, self.heads, head_dim).transpose(0, 2, 1, 3)
        value = self.value(x).reshape(batch, tokens, self.heads, head_dim).transpose(0, 2, 1, 3)
        attended = mx.fast.scaled_dot_product_attention(
            query,
            key,
            value,
            scale=float(head_dim) ** -0.5,
        )
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, tokens, hidden)
        return attended


class _DinoAttention(_MlxModuleBase):
    def __init__(self, hidden_size: int, heads: int) -> None:
        super().__init__()
        self.attention = _DinoAttentionProjection(hidden_size, heads)
        self.output = _DinoAttentionOutput(hidden_size)

    def __call__(self, x: Any) -> Any:
        return self.output.dense(self.attention(x))


class _DinoMlp(_MlxModuleBase):
    def __init__(self, hidden_size: int, mlp_ratio: float) -> None:
        super().__init__()
        intermediate = int(hidden_size * mlp_ratio)
        self.fc1 = nn.Linear(hidden_size, intermediate)
        self.fc2 = nn.Linear(intermediate, hidden_size)

    def __call__(self, x: Any) -> Any:
        return self.fc2(nn.gelu(self.fc1(x)))


class _DinoLayer(_MlxModuleBase):
    def __init__(self, hidden_size: int, heads: int, mlp_ratio: float, eps: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=eps)
        self.attention = _DinoAttention(hidden_size, heads)
        self.layer_scale1 = _DinoLayerScale(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size, eps=eps)
        self.mlp = _DinoMlp(hidden_size, mlp_ratio)
        self.layer_scale2 = _DinoLayerScale(hidden_size)

    def __call__(self, x: Any) -> Any:
        residual = x
        x = residual + self.layer_scale1(self.attention(self.norm1(x)))
        residual = x
        return residual + self.layer_scale2(self.mlp(self.norm2(x)))


class _DinoPatchEmbeddings(_MlxModuleBase):
    def __init__(self, hidden_size: int, patch_size: int) -> None:
        super().__init__()
        self.projection = _MlxConv2d(3, hidden_size, patch_size, stride=patch_size)

    def __call__(self, x: Any) -> Any:
        projected = self.projection(x)
        return projected.reshape(projected.shape[0], -1, projected.shape[-1])


class _DinoEmbeddings(_MlxModuleBase):
    def __init__(self, hidden_size: int, patch_size: int, image_size: int, use_mask_token: bool) -> None:
        super().__init__()
        patch_count = (image_size // patch_size) ** 2
        self.cls_token = mx.zeros((1, 1, hidden_size), dtype=mx.float32)
        self.mask_token = mx.zeros((1, hidden_size), dtype=mx.float32) if use_mask_token else None
        self.patch_embeddings = _DinoPatchEmbeddings(hidden_size, patch_size)
        self.position_embeddings = mx.zeros((1, patch_count + 1, hidden_size), dtype=mx.float32)
        self.patch_size = int(patch_size)

    def __call__(self, x: Any) -> Any:
        _, height, width, _ = x.shape
        patches = self.patch_embeddings(x)
        cls = mx.broadcast_to(self.cls_token, (patches.shape[0], 1, patches.shape[-1]))
        tokens = mx.concatenate([cls, patches], axis=1)
        positions = self.position_embeddings
        patch_count = positions.shape[1] - 1
        if tokens.shape[1] - 1 != patch_count or height != width:
            base = int(round(patch_count ** 0.5))
            patch_positions = positions[:, 1:].reshape(1, base, base, -1)
            patch_positions = _resize_nhwc(
                patch_positions,
                height // self.patch_size,
                width // self.patch_size,
                mode="cubic",
            )
            positions = mx.concatenate(
                [positions[:, :1], patch_positions.reshape(1, -1, positions.shape[-1])],
                axis=1,
            )
        return tokens + positions


class _DinoEncoder(_MlxModuleBase):
    def __init__(self, layers: int, hidden_size: int, heads: int, mlp_ratio: float, eps: float) -> None:
        super().__init__()
        self.layer = [
            _DinoLayer(hidden_size, heads, mlp_ratio, eps)
            for _ in range(layers)
        ]

    def __call__(self, x: Any, output_indices: set[int]) -> list[Any]:
        outputs: list[Any] = []
        for index, layer in enumerate(self.layer, start=1):
            x = layer(x)
            if index in output_indices:
                outputs.append(x)
        return outputs


class _DinoBackbone(_MlxModuleBase):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        image_size = int(config.get("image_size", 518))
        patch_size = int(config.get("patch_size", 14))
        hidden_size = int(config["hidden_size"])
        layers = int(config.get("num_hidden_layers") or 12)
        heads = int(config["num_attention_heads"])
        self.embeddings = _DinoEmbeddings(
            hidden_size,
            patch_size,
            image_size,
            bool(config.get("use_mask_token") if config.get("use_mask_token") is not None else True),
        )
        self.encoder = _DinoEncoder(
            layers,
            hidden_size,
            heads,
            float(config.get("mlp_ratio") or 4.0),
            float(config.get("layer_norm_eps") or 1e-6),
        )
        self.layernorm = nn.LayerNorm(hidden_size, eps=float(config.get("layer_norm_eps") or 1e-6))
        self.out_indices = {int(index) for index in config.get("out_indices", [3, 6, 9, 12])}
        self.apply_layernorm = bool(
            config.get("apply_layernorm") if config.get("apply_layernorm") is not None else True
        )

    def __call__(self, x: Any) -> list[Any]:
        tokens = self.embeddings(x)
        outputs = self.encoder(tokens, self.out_indices)
        if self.apply_layernorm:
            outputs = [self.layernorm(value) for value in outputs]
        return outputs


class _ReassembleLayer(_MlxModuleBase):
    def __init__(self, hidden_size: int, channels: int, factor: float) -> None:
        super().__init__()
        self.projection = _MlxConv2d(hidden_size, channels, 1)
        if factor > 1:
            scale = int(round(factor))
            self.resize = _MlxConvTranspose2d(channels, channels, scale, stride=scale)
        elif factor == 1:
            self.resize = None
        else:
            self.resize = _MlxConv2d(channels, channels, 3, stride=int(round(1 / factor)), padding=1)

    def __call__(self, x: Any) -> Any:
        x = self.projection(x)
        return self.resize(x) if self.resize is not None else x


class _ReassembleStage(_MlxModuleBase):
    def __init__(self, hidden_size: int, channels: list[int], factors: list[float]) -> None:
        super().__init__()
        self.layers = [
            _ReassembleLayer(hidden_size, int(channel), float(factor))
            for channel, factor in zip(channels, factors)
        ]

    def __call__(self, hidden_states: list[Any], patch_height: int, patch_width: int) -> list[Any]:
        features = []
        for hidden_state, layer in zip(hidden_states, self.layers):
            patches = hidden_state[:, 1:].reshape(
                hidden_state.shape[0], patch_height, patch_width, hidden_state.shape[-1]
            )
            features.append(layer(patches))
        return features


class _ResidualLayer(_MlxModuleBase):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.convolution1 = _MlxConv2d(channels, channels, 3, padding=1)
        self.convolution2 = _MlxConv2d(channels, channels, 3, padding=1)

    def __call__(self, x: Any) -> Any:
        residual = x
        x = mx.maximum(self.convolution1(mx.maximum(x, 0.0)), 0.0)
        return self.convolution2(x) + residual


class _FusionLayer(_MlxModuleBase):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.projection = _MlxConv2d(channels, channels, 1)
        self.residual_layer1 = _ResidualLayer(channels)
        self.residual_layer2 = _ResidualLayer(channels)

    def __call__(self, hidden_state: Any, residual: Any = None, size: tuple[int, int] | None = None) -> Any:
        if residual is not None:
            if tuple(hidden_state.shape[1:3]) != tuple(residual.shape[1:3]):
                residual = _resize_nhwc(
                    residual,
                    hidden_state.shape[1],
                    hidden_state.shape[2],
                    align_corners=False,
                )
            hidden_state = hidden_state + self.residual_layer1(residual)
        hidden_state = self.residual_layer2(hidden_state)
        if size is None:
            size = (hidden_state.shape[1] * 2, hidden_state.shape[2] * 2)
        hidden_state = _resize_nhwc(
            hidden_state,
            int(size[0]),
            int(size[1]),
            align_corners=True,
        )
        return self.projection(hidden_state)


class _FusionStage(_MlxModuleBase):
    def __init__(self, channels: int, count: int) -> None:
        super().__init__()
        self.layers = [_FusionLayer(channels) for _ in range(count)]

    def __call__(self, hidden_states: list[Any]) -> list[Any]:
        hidden_states = list(reversed(hidden_states))
        fused = None
        outputs = []
        for index, layer in enumerate(self.layers):
            residual = None if fused is None else hidden_states[index]
            size = (
                (hidden_states[index + 1].shape[1], hidden_states[index + 1].shape[2])
                if index < len(hidden_states) - 1
                else None
            )
            fused = layer(hidden_states[index] if fused is None else fused, residual, size)
            outputs.append(fused)
        return outputs


class _DepthNeck(_MlxModuleBase):
    def __init__(self, config: dict[str, Any], hidden_size: int):
        super().__init__()
        channels = [int(value) for value in config["neck_hidden_sizes"]]
        fusion_hidden = int(config["fusion_hidden_size"])
        factors = [float(value) for value in config["reassemble_factors"]]
        self.reassemble_stage = _ReassembleStage(hidden_size, channels, factors)
        self.convs = [
            _MlxConv2d(channel, fusion_hidden, 3, padding=1, bias=False)
            for channel in channels
        ]
        self.fusion_stage = _FusionStage(fusion_hidden, len(channels))

    def __call__(self, hidden_states: list[Any], patch_height: int, patch_width: int) -> list[Any]:
        features = self.reassemble_stage(hidden_states, patch_height, patch_width)
        features = [conv(feature) for conv, feature in zip(self.convs, features)]
        return self.fusion_stage(features)


class _DepthHead(_MlxModuleBase):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        fusion_hidden = int(config["fusion_hidden_size"])
        head_hidden = int(config["head_hidden_size"])
        self.patch_size = int(config.get("patch_size", 14))
        self.conv1 = _MlxConv2d(fusion_hidden, fusion_hidden // 2, 3, padding=1)
        self.conv2 = _MlxConv2d(fusion_hidden // 2, head_hidden, 3, padding=1)
        self.conv3 = _MlxConv2d(head_hidden, 1, 1)
        self.relative = str(config.get("depth_estimation_type") or "relative") == "relative"
        self.max_depth = float(config.get("max_depth") or 1.0)

    def __call__(self, hidden_states: list[Any], patch_height: int, patch_width: int) -> Any:
        x = hidden_states[-1]
        x = self.conv1(x)
        x = _resize_nhwc(
            x,
            patch_height * self.patch_size,
            patch_width * self.patch_size,
            align_corners=True,
        )
        x = mx.maximum(self.conv2(x), 0.0)
        x = self.conv3(x)
        x = mx.maximum(x, 0.0) if self.relative else mx.sigmoid(x)
        return x[..., 0] * self.max_depth


class _MlxDepthAnything(_MlxModuleBase):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        backbone_config = dict(config.get("backbone_config") or {})
        backbone_config.setdefault("patch_size", config.get("patch_size", 14))
        backbone_config.setdefault("image_size", 518)
        hidden_size = int(backbone_config["hidden_size"])
        self.backbone = _DinoBackbone(backbone_config)
        self.neck = _DepthNeck(config, hidden_size)
        self.head = _DepthHead(config)
        self.patch_size = int(config.get("patch_size", 14))

    def __call__(self, pixel_values: Any) -> Any:
        hidden_states = self.backbone(pixel_values)
        patch_height = pixel_values.shape[1] // self.patch_size
        patch_width = pixel_values.shape[2] // self.patch_size
        fused = self.neck(hidden_states, patch_height, patch_width)
        return self.head(fused, patch_height, patch_width)


def _load_model_config(snapshot: Path) -> dict[str, Any]:
    try:
        config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"MLX requires a readable Hugging Face config.json: {exc}") from exc
    backbone = config.get("backbone_config") or {}
    if config.get("model_type") != "depth_anything" or backbone.get("model_type") != "dinov2":
        raise RuntimeError(
            "MLX currently supports Distill-Any-Depth/Depth Anything DINOv2 checkpoints only"
        )
    if backbone.get("use_swiglu_ffn"):
        raise RuntimeError("MLX does not yet support DINOv2 SwiGLU checkpoints")
    return config


class MlxDepthProvider:
    """GPU-only MLX provider for DINOv2-backed Depth Anything checkpoints."""

    def __init__(
        self,
        *,
        model_id: str,
        model_name: str | None = None,
        device: str | torch.device = "mps",
        cache_dir: str | Path | None = None,
        depth_resolution: int = DISTILL_ANY_DEPTH_BASE_RESOLUTION,
        patch_size: int = DISTILL_ANY_DEPTH_PATCH_SIZE,
        local_files_only: bool = False,
        force_download: bool = False,
        depth_upsample: DepthUpsampleMode = "bilinear",
        depth_upsample_edge_strength: float = 0.35,
    ) -> None:
        self.model_id = str(model_id)
        self.model_name = model_name or self.model_id.rsplit("/", 1)[-1].replace("-hf", "")
        self.device = torch.device(device)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else Path("./models")
        self.depth_resolution = int(depth_resolution)
        self.patch_size = int(patch_size)
        self.local_files_only = bool(local_files_only)
        self.force_download = bool(force_download)
        self.depth_upsample = depth_upsample
        self.depth_upsample_edge_strength = float(depth_upsample_edge_strength)
        self.info = DepthProviderInfo(
            provider="mlx.nn.DepthAnythingForDepthEstimation",
            model_name=self.model_name,
            model_id=self.model_id,
            depth_resolution=self.depth_resolution,
            cache_dir=str(self.cache_dir),
            load_mode="local_files_only" if self.local_files_only else "online",
            depth_backend="mlx",
            runtime="mlx-metal",
            execution_provider="Apple MLX Metal",
            dlpack=True,
            output_device="mps",
        )
        self._model = None
        self._snapshot: Path | None = None
        self._config: dict[str, Any] | None = None
        self._input_zero_copy: bool | None = None
        self._output_zero_copy: bool | None = None
        self._mlx_mean = None
        self._mlx_std = None

    def _resolve_snapshot(self) -> Path:
        if self._snapshot is not None:
            return self._snapshot
        snapshot = find_local_model_snapshot(self.model_id, self.cache_dir)
        if snapshot is None and not self.local_files_only:
            spec = ModelRegistry.default().get(self.model_id)
            ensure_model_downloaded(
                spec,
                cache_dir=self.cache_dir,
                local_files_only=False,
                force_download=self.force_download,
            )
            snapshot = find_local_model_snapshot(self.model_id, self.cache_dir)
        if snapshot is None:
            raise FileNotFoundError(
                f"MLX model files are missing for {self.model_id}; expected a local "
                "config.json and model.safetensors checkpoint"
            )
        self._snapshot = snapshot
        return snapshot

    def load(self) -> Any:
        if self._model is not None:
            return self._model
        mlx_core, _ = _require_mlx()
        if self.device.type != "mps" or sys.platform != "darwin":
            raise RuntimeError("The MLX backend requires a macOS MPS device")
        if not torch.backends.mps.is_available():
            raise RuntimeError("The MLX backend requires a PyTorch MPS device")
        snapshot = self._resolve_snapshot()
        config = _load_model_config(snapshot)
        weight_path = find_local_model_weight(snapshot)
        if weight_path is None or Path(weight_path).suffix.lower() != ".safetensors":
            raise RuntimeError(
                "MLX requires the original model.safetensors checkpoint; it does not "
                "convert PyTorch .bin/.pt files on the inference path"
            )
        with _mlx_gpu_scope(mlx_core):
            model = _MlxDepthAnything(config)
            dtype = _mlx_dtype()
            weights = _stage_weights_on_gpu(
                mlx_core,
                list(mlx_core.load(str(weight_path)).items()),
                dtype,
            )
            model.load_weights(weights, strict=True)
            mlx_core.eval(model.parameters())
            mean, std = _normalization_tensors_for_model(
                self.model_id,
                torch.device("cpu"),
                torch.float32,
            )
            self._mlx_mean = mlx_core.add(
                mlx_core.array(mean.flatten().tolist()).reshape(1, 1, 1, 3),
                0,
                stream=mlx_core.gpu,
            )
            self._mlx_std = mlx_core.add(
                mlx_core.array(std.flatten().tolist()).reshape(1, 1, 1, 3),
                0,
                stream=mlx_core.gpu,
            )
        self._config = config
        self.info = replace(
            self.info,
            depth_resolution=int(config.get("backbone_config", {}).get("image_size", self.depth_resolution)),
            native_io_reason=f"MLX Metal model weights; DLPack input interop; dtype={str(dtype)}",
        )
        self._model = model
        return model

    def predict(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.predict_profile(rgb).depth

    def predict_profile(self, rgb: torch.Tensor) -> DepthProfileResult:
        mlx_core, _ = _require_mlx()
        if self.device.type != "mps":
            raise RuntimeError("MLX inference is GPU-only and requires a torch MPS input")
        if not torch.backends.mps.is_available():
            raise RuntimeError("The MLX backend requires a PyTorch MPS device")
        rgb = ensure_bchw(rgb, name="rgb")
        if rgb.device.type != "mps":
            raise RuntimeError("MLX inference refuses CPU RGB tensors; use the MPS capture path")
        _, _, height, width = rgb.shape
        input_h, input_w = _model_input_size_for_model(
            height,
            width,
            self.depth_resolution,
            self.patch_size,
            self.model_id,
        )
        with _mlx_gpu_scope(mlx_core):
            start = time.perf_counter()
            rgb = torch.nan_to_num(rgb.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            torch.mps.synchronize()
            mlx_input, input_zero_copy = _import_torch_mps_dlpack(mlx_core, rgb.detach())
            if self._input_zero_copy is None or self._input_zero_copy != input_zero_copy:
                self._input_zero_copy = input_zero_copy
                self.info = replace(
                    self.info,
                    native_io_reason=(
                        "torch MPS -> MLX Metal DLPack zero-copy"
                        if input_zero_copy
                        else "torch MPS -> MLX Metal DLPack device-local copy"
                    ),
                )
            mlx_input = mlx_input.transpose(0, 2, 3, 1)
            mlx_input = _resize_nhwc(mlx_input, input_h, input_w, mode="cubic")
            model = self.load()
            mlx_input = ((mlx_input - self._mlx_mean) / self._mlx_std).astype(_mlx_dtype())
            mlx_core.eval(mlx_input)
            preprocess_ms = (time.perf_counter() - start) * 1000.0

            start = time.perf_counter()
            mlx_depth = model(mlx_input)
            mlx_core.eval(mlx_depth)
            mlx_core.synchronize()
            model_ms = (time.perf_counter() - start) * 1000.0

            start = time.perf_counter()
            depth, output_zero_copy = _export_torch_mps_dlpack(mlx_depth)
            self._output_zero_copy = output_zero_copy
            self.info = replace(
                self.info,
                native_io_reason=(
                    "torch MPS <-> MLX Metal DLPack zero-copy"
                    if self._input_zero_copy and output_zero_copy
                    else "torch MPS <-> MLX Metal DLPack device-local copy"
                ),
            )
            depth = depth.unsqueeze(1)
            if depth.device.type != "mps":
                raise RuntimeError(
                    f"MLX DLPack returned a non-MPS depth tensor: {depth.device}"
                )
            depth = _postprocess_generic_depth(depth, self.model_id)
            depth = _upsample_depth_canonical(
                depth,
                height,
                width,
                rgb=rgb,
                mode=self.depth_upsample,
                edge_strength=self.depth_upsample_edge_strength,
            )
            postprocess_ms = (time.perf_counter() - start) * 1000.0
        return DepthProfileResult(
            depth,
            preprocess_ms,
            model_ms,
            postprocess_ms,
            finite_depth=True,
            native_zero_copy=bool(self._input_zero_copy and self._output_zero_copy),
            raw_output_semantics=_model_output_semantics(self.model_id),
        )


def create_mlx_depth_provider(
    *,
    model_id: str,
    model_name: str | None = None,
    device: str | torch.device = "mps",
    cache_dir: str | Path | None = None,
    depth_resolution: int = DISTILL_ANY_DEPTH_BASE_RESOLUTION,
    patch_size: int = DISTILL_ANY_DEPTH_PATCH_SIZE,
    local_files_only: bool = False,
    force_download: bool = False,
    depth_upsample: DepthUpsampleMode = "bilinear",
    depth_upsample_edge_strength: float = 0.35,
):
    if sys.platform != "darwin":
        raise RuntimeError("The MLX backend is available only on macOS")
    return MlxDepthProvider(
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


__all__ = ["MlxDepthProvider", "create_mlx_depth_provider", "is_mlx_available"]
