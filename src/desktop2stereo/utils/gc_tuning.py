"""Optional GC pause accounting and reversible real-time tuning."""
from __future__ import annotations

import gc
import os
import time

_enabled = os.environ.get("D2S_GC_TUNING", "0").strip().casefold() in {
    "1", "true", "yes", "on"
}
_installed = False
_tuned = False
_started_at = 0.0
_saved_thresholds: tuple[int, int, int] | None = None
_stats = {"count": 0, "total_ms": 0.0, "max_ms": 0.0, "last_ms": 0.0, "last_generation": -1}


def _on_gc(phase: str, info: dict) -> None:
    global _started_at
    if phase == "start":
        _started_at = time.perf_counter()
    elif phase == "stop":
        pause_ms = max(0.0, (time.perf_counter() - _started_at) * 1000.0)
        _stats["count"] += 1
        _stats["total_ms"] += pause_ms
        _stats["max_ms"] = max(_stats["max_ms"], pause_ms)
        _stats["last_ms"] = pause_ms
        _stats["last_generation"] = int(info.get("generation", -1))


def install_pause_watch() -> None:
    """Install the allocation-light GC observer once per runtime."""
    global _installed
    if not _installed:
        _stats.update(
            count=0,
            total_ms=0.0,
            max_ms=0.0,
            last_ms=0.0,
            last_generation=-1,
        )
        gc.callbacks.append(_on_gc)
        _installed = True


def tune_gc_for_realtime(label: str) -> bool:
    """Apply experimental tuning only when explicitly enabled by the user."""
    global _saved_thresholds, _tuned
    if not _enabled or _tuned:
        return False
    _saved_thresholds = gc.get_threshold()
    try:
        gc.freeze()
    except (AttributeError, TypeError, ValueError):
        _saved_thresholds = None
        return False
    try:
        gen0 = max(
            _saved_thresholds[0],
            int(os.environ.get("D2S_GC_GEN0_THRESHOLD", "20000")),
        )
        gc.set_threshold(
            gen0,
            max(_saved_thresholds[1], 50),
            max(_saved_thresholds[2], 50),
        )
    except (TypeError, ValueError):
        unfreeze = getattr(gc, "unfreeze", None)
        if callable(unfreeze):
            unfreeze()
        gc.set_threshold(*_saved_thresholds)
        _saved_thresholds = None
        return False
    _tuned = True
    print(f"[GC] realtime tuning enabled for {label}: thresholds={gc.get_threshold()}", flush=True)
    return True


def take_pause_snapshot() -> dict[str, int | float]:
    """Return and reset pause totals for one diagnostics window."""
    snapshot = dict(_stats)
    _stats["count"] = 0
    _stats["total_ms"] = 0.0
    _stats["max_ms"] = 0.0
    return snapshot


def restore_gc_settings() -> None:
    """Restore interpreter settings when the processing runtime exits."""
    global _installed, _tuned, _saved_thresholds
    if _tuned:
        if _saved_thresholds is not None:
            gc.set_threshold(*_saved_thresholds)
        unfreeze = getattr(gc, "unfreeze", None)
        if callable(unfreeze):
            unfreeze()
    if _installed:
        try:
            gc.callbacks.remove(_on_gc)
        except ValueError:
            pass
    _installed = False
    _tuned = False
    _saved_thresholds = None
