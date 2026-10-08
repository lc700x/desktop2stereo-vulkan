"""SMAA 1x High display filtering after DIBR visibility has been resolved."""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import torch

_NATIVE = Path(__file__).parent / "providers/apple/native"
_SHADERS = Path(__file__).resolve().parents[1] / "shaders"


@functools.lru_cache(maxsize=1)
def _mps_library():
    text = (_NATIVE / "sbs_smaa_msl.h").read_text(encoding="utf-8")
    source = text.split('R"MSL(', 1)[1].rsplit(')MSL"', 1)[0]
    return torch.mps.compile_shader(source)


@functools.lru_cache(maxsize=1)
def _mps_luts() -> tuple[torch.Tensor, torch.Tensor]:
    area = np.fromfile(_SHADERS / "smaa_area.rg8", dtype=np.uint8).copy()
    search = np.fromfile(_SHADERS / "smaa_search.r8", dtype=np.uint8).copy()
    return torch.from_numpy(area).to("mps"), torch.from_numpy(search).to("mps")


@functools.lru_cache(maxsize=16)
def _mps_params(width: int, height: int, channels: int, layout: int,
                eye_width: int, batch: int, half_sbs: int) -> torch.Tensor:
    return torch.tensor((width, height, channels, layout, eye_width, batch, half_sbs, 0),
                        dtype=torch.int32, device="mps")


def _mps_filter(
    image: torch.Tensor,
    eye_width: int = 0,
    *,
    half_sbs: bool = False,
) -> torch.Tensor:
    """Run the three SMAA stages in Metal; tables and working layout persist."""
    if image.device.type != "mps" or image.ndim != 4 or image.shape[1] not in (3, 4):
        raise ValueError("SMAA Metal expects MPS BCHW RGB/RGBA")
    source = image.contiguous()
    batch, channels, height, width = map(int, source.shape)
    edges = torch.empty((batch, height, width, 2), dtype=torch.uint8, device="mps")
    # Four normalized SMAA weights need one byte each; 8-bit storage keeps the
    # full-resolution scratch small and matches the official RG8 lookup path.
    weights = torch.empty((batch, height, width, 4), dtype=torch.uint8, device="mps")
    output_width = width // 2 if half_sbs else width
    output = torch.empty((batch, channels, height, output_width),
                         dtype=source.dtype, device="mps")
    params = _mps_params(width, height, channels,
                         2 if source.dtype == torch.uint8 else 1,
                         int(eye_width), batch, int(half_sbs))
    area, search = _mps_luts()
    library = _mps_library()
    grid = (width, height, batch)
    library.d2s_smaa_edges(source, source, edges, params, threads=grid)
    library.d2s_smaa_weights(edges, area, search, weights, params, threads=grid)
    library.d2s_smaa_resolve(
        source, source, weights, output, output, params,
        threads=(output_width, height, batch),
    )
    return output
