"""GUI-specific alias for the shared process instance lock."""
from __future__ import annotations

import os
import time
from pathlib import Path

from utils.instance_lock import ProcessInstanceLock

GUIInstanceLock = ProcessInstanceLock


def request_gui_activation(path: str) -> str:
    """Publish an atomic request for the primary GUI to raise its window."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}-{time.time_ns()}"
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    temporary.write_text(f"{token}\n", encoding="ascii")
    os.replace(temporary, target)
    return token


def clear_gui_activation_request(path: str) -> None:
    try:
        Path(path).unlink()
    except OSError:
        pass


__all__ = [
    "GUIInstanceLock",
    "clear_gui_activation_request",
    "request_gui_activation",
]
