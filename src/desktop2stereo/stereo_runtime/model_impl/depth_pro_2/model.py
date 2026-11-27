#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import Callable, Iterable, Iterator, Literal, Mapping, Sequence

import numpy as np
import torch
from huggingface_hub import PyTorchModelHubMixin, snapshot_download
from torch import nn
from torch.nn import functional as F

from .network.decoder import DPTDecoder, create_inverse_depth_head
from .network.encoder import KVCache, MultiViewEncoder, autocast_for
from .network.fov import FOVEstimator
from .streaming import DepthStream, Prediction
from .utils import default_device

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

INPUT_MULTIPLE = 32

VideoMode = Literal["buffered", "unbuffered"]
DEFAULT_WINDOW = 16
DEFAULT_OVERLAP = 4

CHECKPOINT_DIR = Path("./checkpoints")
CHECKPOINT_ROOT = "https://ml-site.cdn-apple.com/models/depth-pro-2"
HUB_REPOSITORY = "apple/DepthPro2"

LOGGER = logging.getLogger(__name__)


def resolve_checkpoint(uri: str | Path) -> Path:
    """Return a local path to a checkpoint given as a local path or a URL.

    A URL is served from ``./checkpoints/<name>`` if that exists, else downloaded once into the
    torch hub cache (``$TORCH_HOME/hub/checkpoints``).
    """
    text = str(uri)
    if text.startswith(("https://", "http://")):
        name = text.rsplit("/", 1)[-1]
        local = CHECKPOINT_DIR / name
        if local.exists():
            return local
        cached = Path(torch.hub.get_dir()) / "checkpoints" / name
        if not cached.exists():
            LOGGER.info("Downloading %s to %s.", text, cached)
            cached.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(text, str(cached), progress=True)
        return cached

    path = Path(text).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}.")
    return path


def load_checkpoint(uri: str | Path) -> Mapping[str, torch.Tensor]:
    """Memory-map a checkpoint's weights, or read them in full if its format cannot be mapped."""
    path = resolve_checkpoint(uri)
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except RuntimeError:
        return torch.load(path, map_location="cpu", weights_only=True)


@dataclass
class DepthPro2Config:
    backbone_preset: str
    reassembled_blocks: tuple[int, int, int, int]
    cross_frame_blocks: tuple[int, ...]
    dims_reassemble: tuple[int, int, int, int]
    dim_decoder: int

    repo_id: str | None = None
    subfolder: str | None = None
    checkpoint_uri: str | None = None
    fov_checkpoint_uri: str | None = f"{CHECKPOINT_ROOT}/depthpro2-fov-ew55zg4kq7.pt"


PRESETS: dict[str, DepthPro2Config] = {
    "depthpro2-s": DepthPro2Config(
        backbone_preset="vit_small_patch16_dinov3",
        reassembled_blocks=(2, 5, 8, 11),
        cross_frame_blocks=(5, 7, 9, 11),
        dims_reassemble=(48, 96, 192, 384),
        dim_decoder=64,
        repo_id=HUB_REPOSITORY,
        subfolder="depthpro2-s",
        checkpoint_uri=f"{CHECKPOINT_ROOT}/depthpro2-s-bgcc6f8bg3.pt",
    ),
    "depthpro2-b": DepthPro2Config(
        backbone_preset="vit_base_patch16_dinov3",
        reassembled_blocks=(2, 5, 8, 11),
        cross_frame_blocks=(5, 7, 9, 11),
        dims_reassemble=(96, 192, 384, 768),
        dim_decoder=128,
        repo_id=HUB_REPOSITORY,
        subfolder="depthpro2-b",
        checkpoint_uri=f"{CHECKPOINT_ROOT}/depthpro2-b-xb7m9egktw.pt",
    ),
    "depthpro2-l": DepthPro2Config(
        backbone_preset="vit_large_patch16_dinov3.lvd1689m",
        reassembled_blocks=(5, 11, 17, 23),
        cross_frame_blocks=(9, 11, 13, 15, 17, 19, 21, 23),
        dims_reassemble=(256, 512, 1024, 1024),
        dim_decoder=256,
        repo_id=HUB_REPOSITORY,
        subfolder="depthpro2-l",
        checkpoint_uri=f"{CHECKPOINT_ROOT}/depthpro2-l-tegstu7fim.pt",
    ),
    "depthpro2-h": DepthPro2Config(
        backbone_preset="vit_huge_plus_patch16_dinov3",
        reassembled_blocks=(7, 15, 23, 31),
        cross_frame_blocks=(11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31),
        dims_reassemble=(256, 512, 1024, 1024),
        dim_decoder=256,
        repo_id=HUB_REPOSITORY,
        subfolder="depthpro2-h",
        checkpoint_uri=f"{CHECKPOINT_ROOT}/depthpro2-h-eivtm5w7m5.pt",
    ),
}

DEFAULT_MODEL = "depthpro2-h"


def create_model_and_transforms(
    config: DepthPro2Config | str = DEFAULT_MODEL,
    device: torch.device | str | None = None,
    precision: torch.dtype | None = None,
) -> tuple[DepthPro2, Callable[[np.ndarray], torch.Tensor]]:
    """Create a Depth Pro 2 model and load weights from the configured checkpoints.

    Args:
        config: A model configuration, or the name of a preset in ``PRESETS``.
        device: The device to load the model onto; by default CUDA, then MPS, then the CPU.
        precision: Autocast dtype of the backbones and the decoder; the heads run in float32.
            Defaults to bfloat16 on CUDA and MPS, as trained and evaluated, and to float32 on the
            CPU, where bfloat16 autocast is much slower. ``torch.float16`` is closer to float32.

    Returns:
        The model in eval mode, and a transform from an RGB uint8 ``(H, W, 3)`` array to a
        normalized ``(3, H, W)`` tensor.
    """
    if isinstance(config, str):
        if config not in PRESETS:
            raise ValueError(f"Unknown preset {config!r}; choose one of {', '.join(PRESETS)}.")
        config = PRESETS[config]
    device = default_device() if device is None else torch.device(device)
    if precision is None:
        precision = torch.float32 if device.type == "cpu" else torch.bfloat16

    # Initializing H+'s 0.9 B parameters takes seconds on a CPU and a fraction of one on a GPU.
    with torch.device(device):
        if config.checkpoint_uri is None and config.repo_id is not None:
            model = DepthPro2.from_pretrained(
                config.repo_id, subfolder=config.subfolder, precision=precision, strict=True
            )
        else:
            model = DepthPro2(
                backbone_preset=config.backbone_preset,
                reassembled_blocks=config.reassembled_blocks,
                cross_frame_blocks=config.cross_frame_blocks,
                dims_reassemble=config.dims_reassemble,
                dim_decoder=config.dim_decoder,
                fov=config.fov_checkpoint_uri is not None,
                precision=precision,
            )

    if config.checkpoint_uri is not None:
        state_dict = load_checkpoint(config.checkpoint_uri)
        if model.fov is not None:
            fov_state_dict = load_checkpoint(config.fov_checkpoint_uri)
            state_dict.update({f"fov.{k}": v for k, v in fov_state_dict.items()})
        model.load_state_dict(state_dict, strict=True)

    model = model.to(device).eval()

    return model, partial(to_normalized_tensor, device=device)


@lru_cache(maxsize=None)
def _normalization(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(3, 1, 1)
    return mean, std


def to_normalized_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    """Normalize an RGB ``(H, W, 3)`` uint8 array to a ``(3, H, W)`` float tensor on ``device``."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise TypeError(
            "transform takes an RGB uint8 (H, W, 3) array, e.g. from load_rgb, "
            f"got {image.dtype} {tuple(image.shape)}."
        )
    x = torch.from_numpy(np.ascontiguousarray(image)).to(device, non_blocking=True)
    x = x.permute(2, 0, 1).contiguous().to(torch.float32).div_(255.0)
    mean, std = _normalization(device)
    return x.sub_(mean).div_(std)


class DepthPro2(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        backbone_preset: str,
        reassembled_blocks: Sequence[int],
        cross_frame_blocks: Sequence[int],
        dims_reassemble: Sequence[int],
        dim_decoder: int,
        fov: bool = True,
        precision: torch.dtype = torch.float32,
    ):
        """Initialize Depth Pro 2; ``precision`` is the autocast dtype, float32 disabling it."""
        super().__init__()
        self.encoder = MultiViewEncoder(
            backbone_preset=backbone_preset,
            reassembled_blocks=reassembled_blocks,
            cross_frame_blocks=cross_frame_blocks,
            dims_reassemble=dims_reassemble,
        )
        self.decoder = DPTDecoder(dims_reassemble=dims_reassemble, dim_decoder=dim_decoder)
        self.head = create_inverse_depth_head(self.decoder.dim_out)
        self.fov = FOVEstimator(precision) if fov else None
        self.precision = precision

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *, subfolder=None, **kwargs):
        """Load a model ``save_pretrained`` wrote, ``subfolder`` naming its folder if it has one."""
        if subfolder is not None:
            path = Path(pretrained_model_name_or_path)
            if not path.is_dir():
                options = {key: kwargs[key] for key in ("revision", "token") if key in kwargs}
                files = snapshot_download(str(path), allow_patterns=f"{subfolder}/*", **options)
                path = Path(files)
            pretrained_model_name_or_path = path / subfolder
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    def forward(self, x: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        """Predict focal-normalized inverse depth ``(T, 1, H, W)`` for frames ``(T, 3, H, W)``.

        With a ``cache``, ``x`` is one frame ``(1, 3, H, W)``, attending the cached frames too.
        """
        with autocast_for(x.device, self.precision)():
            features, grid_size = self.encoder.encode(x, cache)

        inverse_depth = []
        for frame in range(x.shape[0]):
            tokens = [f[frame : frame + 1] for f in features]
            inverse_depth.append(self.decode_frame(tokens, grid_size))
        return torch.cat(inverse_depth)

    def decode_frame(
        self, features: Sequence[torch.Tensor], grid_size: tuple[int, int]
    ) -> torch.Tensor:
        """Decode one frame's tapped tokens into inverse depth ``(1, 1, H, W)``."""
        device = features[0].device
        with autocast_for(device, self.precision)():
            decoded = self.decoder(self.encoder.reassemble_tokens(features, grid_size))
        return self.head(decoded.float())

    def predict_inverse_depth(self, x: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        """Inverse depth ``(T, H, W)`` for frames ``(T, 3, H, W)`` at their native resolution."""
        height, width = x.shape[-2:]
        pad_h = -height % INPUT_MULTIPLE
        pad_w = -width % INPUT_MULTIPLE
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return self.forward(x, cache)[:, 0, :height, :width]

    @torch.no_grad()
    def infer(
        self,
        x: torch.Tensor,
        f_px: float | torch.Tensor | None = None,
    ) -> Prediction:
        """Infer inverse depth, metric depth and focal length at the native input resolution.

        Args:
            x: Normalized image ``(3, H, W)`` or clip ``(T, 3, H, W)``. A clip's frames attend to
                each other and share one focal length, so pass unrelated photos one at a time.
            f_px: Focal length in pixels of ``x``, e.g. from EXIF; one estimate per clip by default.

        Returns:
            ``inverse_depth`` normalized by focal length and ``depth`` in meters along the optical
            axis, both ``(H, W)`` or ``(T, H, W)``; depth is infinite where infinitely far, such as
            sky. ``depth`` and ``focallength_px``, a 0-d tensor on the model's device, are ``None``
            if no focal length is given or estimated.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"infer takes a tensor from transform(image), got {type(x).__name__}.")
        if x.ndim not in (3, 4) or x.shape[-3] != 3 or not x.is_floating_point():
            raise ValueError(
                "infer takes a float (3, H, W) or (T, 3, H, W) tensor from transform(image), "
                f"got {x.dtype} {tuple(x.shape)}."
            )
        is_image = x.ndim == 3
        if is_image:
            x = x.unsqueeze(0)
        width = x.shape[-1]
        inverse_depth = self.predict_inverse_depth(x)

        if f_px is not None:
            f_px = torch.as_tensor(f_px, dtype=torch.float32, device=x.device).squeeze()
        elif self.fov is not None:
            f_px = self.fov.focal_length_px(x)

        depth = None if f_px is None else f_px / (width * inverse_depth)

        if is_image:
            inverse_depth = inverse_depth[0]
            depth = None if depth is None else depth[0]
        return {"inverse_depth": inverse_depth, "depth": depth, "focallength_px": f_px}

    def stream(
        self,
        window: int = DEFAULT_WINDOW,
        overlap: int = DEFAULT_OVERLAP,
        f_px: float | torch.Tensor | None = None,
        *,
        mode: VideoMode = "buffered",
    ) -> DepthStream:
        """Open a session over one video: ``step`` takes each frame and ``flush`` ends the video.

        Args:
            window: Frames attended together: the clip length (buffered) or the cache size,
                the current frame included (unbuffered).
            overlap: Frames shared by consecutive clips, fewer than ``window``; ``buffered`` only.
            f_px: Focal length in pixels of the frames. Defaults to the median estimate over the
                frames so far, kept once ``window`` frames are in.
            mode: ``buffered`` predicts overlapping clips, so a frame also sees later frames and is
                returned once its clip is done; ``unbuffered`` predicts each frame as it arrives,
                from itself and the frames before it.

        Returns:
            A :class:`~depth_pro_2.streaming.DepthStream`, whose ``step`` and ``flush`` return
            lists of the predictions finished, as :meth:`infer_video` yields them.
        """
        return DepthStream(self, window, overlap, f_px, mode=mode)

    @torch.no_grad()
    def infer_video(
        self,
        frames: Iterable[torch.Tensor],
        window: int = DEFAULT_WINDOW,
        overlap: int = DEFAULT_OVERLAP,
        f_px: float | torch.Tensor | None = None,
        *,
        mode: VideoMode = "buffered",
    ) -> Iterator[Prediction]:
        """Infer depth for a video, one prediction per frame, in order.

        Args:
            frames: Normalized frames ``(3, H, W)``, read lazily.
            window: Frames attended together: the clip length (buffered) or the cache size,
                the current frame included (unbuffered).
            overlap: Frames shared by consecutive clips, fewer than ``window``; ``buffered`` only.
            f_px: Focal length in pixels of the frames. Defaults to the median estimate over the
                first window.
            mode: ``buffered`` processes overlapping clips, so a frame also sees later frames;
                ``unbuffered`` predicts each frame from itself and the frames before it.

        Yields:
            ``inverse_depth`` and ``depth`` in meters, both ``(H, W)``, and ``focallength_px``.
        """
        if f_px is None and self.fov is not None:
            # Unlike a live feed, a file can look ahead: every frame gets the first window's focal.
            frames = iter(frames)
            head = list(itertools.islice(frames, window))
            if not head:
                return
            f_px = self.fov.focal_length_px(torch.stack(head))
            frames = itertools.chain(head, frames)
        stream = self.stream(window, overlap, f_px, mode=mode)
        for frame in frames:
            yield from stream.step(frame)
        yield from stream.flush()
