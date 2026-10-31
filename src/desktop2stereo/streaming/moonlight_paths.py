"""Platform-specific paths for the bundled Moonlight host."""

from __future__ import annotations

import os
from pathlib import Path
import sys


def moonlight_state_directory(
    *, platform_name: str | None = None, environ: dict[str, str] | None = None
) -> Path:
    platform_name = platform_name or sys.platform
    environ = os.environ if environ is None else environ
    if platform_name == "win32":
        root = Path(environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif platform_name == "darwin":
        root = Path.home() / "Library" / "Application Support"
    else:
        root = Path(environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return root / "Desktop2Stereo" / "Moonlight"


def sunshine_executable(
    server_dir: str | Path, *, environ=None, platform_name: str | None = None
) -> Path:
    environ = os.environ if environ is None else environ
    platform_name = platform_name or sys.platform
    override = str(environ.get("D2S_SUNSHINE_EXE", "") or "").strip()
    if override:
        return Path(override).expanduser()
    server_dir = Path(server_dir)
    if platform_name == "darwin":
        return server_dir / "macos" / "Sunshine.app" / "Contents" / "MacOS" / "Sunshine"
    return server_dir / "sunshine.exe"


def sunshine_working_directory(executable: str | Path) -> Path:
    executable = Path(executable)
    if executable.parent.name == "MacOS" and executable.parent.parent.name == "Contents":
        return executable.parent.parent / "Resources"
    return executable.parent
