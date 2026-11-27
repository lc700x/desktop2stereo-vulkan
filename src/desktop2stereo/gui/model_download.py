"""GUI-facing model download helpers.

Downloads run inside the GUI process so the existing structured progress
events can update the log panel.  Keep environment mutation scoped to this
helper; tokens are never written to settings or logs.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from stereo_runtime.model_artifacts import ensure_model_downloaded
from stereo_runtime.model_registry import DepthModelSpec, ModelRegistry


def gui_model_cache_dir() -> Path:
    """Return the model cache used by the child runtime."""
    return Path(__file__).resolve().parents[1] / "models"


def resolve_model_spec(model_name: str) -> DepthModelSpec:
    return ModelRegistry.default().get(str(model_name))


@contextmanager
def _temporary_hf_token(token: str | None) -> Iterator[None]:
    token = str(token or "").strip()
    if not token:
        yield
        return

    previous = os.environ.get("HF_TOKEN")
    os.environ["HF_TOKEN"] = token
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("HF_TOKEN", None)
        else:
            os.environ["HF_TOKEN"] = previous


def download_model_to_gui_cache(
    model_name: str,
    *,
    cache_dir: str | Path | None = None,
    token: str | None = None,
    force_download: bool = False,
) -> Path:
    """Download or validate one model without persisting credentials."""
    spec = resolve_model_spec(model_name)
    target_cache = Path(cache_dir) if cache_dir is not None else gui_model_cache_dir()
    with _temporary_hf_token(token):
        return ensure_model_downloaded(
            spec,
            cache_dir=target_cache,
            force_download=bool(force_download),
        )


__all__ = [
    "download_model_to_gui_cache",
    "gui_model_cache_dir",
    "resolve_model_spec",
]
