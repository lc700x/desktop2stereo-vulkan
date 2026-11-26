#!/usr/bin/env python3
"""Build the macOS arm64 Desktop2Stereo release from the current source tree."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

PACKAGE_NAME = "Desktop2Stereo-macOS-arm64"
ALLOWED_DYLIB_PREFIXES = (
    "desktop2stereo/gui/flet_clients/flet-macos/",
    "desktop2stereo/streaming/moonlight_server/macos/",
    "desktop2stereo/xr_viewer/native/macos/",
)
IGNORED_NAMES = (
    "__pycache__",
    "logs",
    "models",
    "windows",
    "linux",
    "amd",
    "intel",
    "nvidia",
    "directml",
    "desktop_duplication",
    "native_rtsp_output",
    "amd_encoder",
    "ffmpeg",
    "mediamtx",
    "flet-windows*",
    "flet-linux*",
    "*.pyc",
    "*.pyo",
    "*.log",
    "*.dll",
    "*.exe",
    "*.so",
    "*.bat",
    "*.ps1",
    "*.reg",
    "*.pyd",
    "*.lib",
    "*windows*",
    "*linux*",
    "*darwin-amd64*",
    "*darwin_amd64*",
    "settings.yaml",
    "auto.crt",
    "auto.key",
    ".DS_Store",
)
REQUIRED_FILES = (
    "LICENSE",
    "src/run_mac",
    "src/env_install/README_MACOS.txt",
    "src/env_install/install-mps",
    "src/env_install/requirements-mps.txt",
    "src/env_install/requirements-pip-options.txt",
    "src/env_install/requirements.txt",
    "src/desktop2stereo/main.py",
    "src/desktop2stereo/streaming/rtmp/runtime-manifest.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_staged_tree(root: Path, *, machine_checks: bool) -> None:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if path.suffix.lower() in {".dll", ".exe", ".so", ".bat", ".ps1", ".pyd", ".lib"}:
            raise RuntimeError(f"forbidden platform payload in package: {relative}")
        if path.name == "settings.yaml":
            raise RuntimeError(f"machine-specific settings are not packageable: {relative}")
        if path.suffix.lower() == ".dylib":
            if not relative.startswith(ALLOWED_DYLIB_PREFIXES):
                raise RuntimeError(f"unexpected dylib in package: {relative}")
            if machine_checks and shutil.which("file"):
                probe = subprocess.run(
                    ["file", "-b", str(path)], capture_output=True, text=True, check=False
                )
                if "arm64" not in probe.stdout:
                    raise RuntimeError(f"dylib is not arm64: {relative}")


def stage_package(source_root: Path, stage_root: Path) -> None:
    missing = [name for name in REQUIRED_FILES if not (source_root / name).is_file()]
    if missing:
        raise FileNotFoundError("missing required release inputs: " + ", ".join(missing))

    package = stage_root / PACKAGE_NAME
    package.mkdir()
    shutil.copy2(source_root / "LICENSE", package / "LICENSE")
    shutil.copy2(source_root / "src/run_mac", package / "run_mac")
    shutil.copy2(
        source_root / "src/env_install/README_MACOS.txt", package / "README_MACOS.txt"
    )
    shutil.copy2(
        source_root / "src/env_install/install_mac.command", package / "install_mac.command"
    )

    env_install = package / "env_install"
    env_install.mkdir()
    for name in (
        "install-mps",
        "requirements-mps.txt",
        "requirements-pip-options.txt",
        "requirements.txt",
    ):
        shutil.copy2(source_root / "src/env_install" / name, env_install / name)

    shutil.copytree(
        source_root / "src/desktop2stereo",
        package / "desktop2stereo",
        ignore=shutil.ignore_patterns(*IGNORED_NAMES),
        symlinks=True,
        ignore_dangling_symlinks=True,
    )
    rtmp = package / "desktop2stereo/streaming/rtmp"
    macos_config = rtmp / "mediamtx.macos.yml"
    if macos_config.is_file():
        shutil.copy2(macos_config, rtmp / "mediamtx.yml")

    for name in ("run_mac", "install_mac.command", "env_install/install-mps"):
        path = package / name
        path.chmod(path.stat().st_mode | 0o111)


def build(source_root: Path, output: Path, seven_zip: str, machine_checks: bool) -> None:
    with tempfile.TemporaryDirectory(prefix="d2s-macos-package-") as temp_name:
        temp_root = Path(temp_name)
        stage_root = temp_root / "stage"
        stage_root.mkdir()
        stage_package(source_root, stage_root)
        validate_staged_tree(stage_root / PACKAGE_NAME, machine_checks=machine_checks)

        archive = temp_root / output.name
        subprocess.run(
            [seven_zip, "a", "-t7z", "-mx=9", str(archive), PACKAGE_NAME],
            cwd=stage_root,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        subprocess.run(
            [seven_zip, "t", str(archive)], check=True, stdout=subprocess.DEVNULL
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        if output.is_file():
            backup_dir = source_root / ".tmp" / "release-backup"
            backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output, backup_dir / f"{output.name}.previous")
        staged_output = output.with_suffix(output.suffix + ".tmp")
        shutil.copy2(archive, staged_output)
        os.replace(staged_output, output)
        output.with_suffix(output.suffix + ".sha256").write_text(
            f"{sha256(output)}  {output.name}\n", encoding="ascii"
        )


def main() -> None:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=default_root)
    parser.add_argument(
        "--output",
        type=Path,
        default=default_root / "dist" / "Desktop2Stereo_v3.0beta03-macOS-arm64.7z",
    )
    parser.add_argument("--seven-zip", default=shutil.which("7zz") or shutil.which("7z") or "7zz")
    parser.add_argument("--skip-machine-checks", action="store_true")
    args = parser.parse_args()
    build(args.source_root.resolve(), args.output.resolve(), args.seven_zip, not args.skip_machine_checks)


if __name__ == "__main__":
    main()
