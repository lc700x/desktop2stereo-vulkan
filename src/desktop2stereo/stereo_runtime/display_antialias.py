"""Depth-map edge smoothing for display and stereo synthesis."""

from __future__ import annotations

import torch
from .depth_postprocess import anti_alias_depth_global_gaussian


def antialias_depth_map(depth: torch.Tensor) -> torch.Tensor:
    """Smooth a displayed depth map without changing its grayscale encoding."""
    from .output import ensure_b1hw, output_edge_aa_enabled

    value = ensure_b1hw(depth).float()
    if not output_edge_aa_enabled():
        return value
    return anti_alias_depth_global_gaussian(value)
