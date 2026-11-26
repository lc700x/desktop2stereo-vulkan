from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "tools" / "build_macos_portable.py"
SEVEN_ZIP = shutil.which("7zz") or shutil.which("7z")


@pytest.mark.skipif(SEVEN_ZIP is None, reason="7-Zip is required to build the release")
def test_macos_builder_creates_a_clean_arm64_package(tmp_path: Path) -> None:
    source = tmp_path / "source"
    app = source / "src" / "desktop2stereo"
    rtmp = app / "streaming" / "rtmp"
    for path in (
        app / "gui",
        app / "capture" / "native" / "windows",
        app / "xr_viewer" / "native" / "macos",
        rtmp / "mediamtx",
        source / "src" / "env_install",
    ):
        path.mkdir(parents=True, exist_ok=True)

    (source / "LICENSE").write_text("license\n", encoding="utf-8")
    (source / "src" / "run_mac").write_text("#!/bin/bash\n", encoding="utf-8")
    (source / "src" / "env_install" / "README_MACOS.txt").write_text("readme\n", encoding="utf-8")
    (source / "src" / "env_install" / "install_mac.command").write_text("#!/bin/bash\n", encoding="utf-8")
    (source / "src" / "env_install" / "install-mps").write_text("#!/bin/bash\n", encoding="utf-8")
    (source / "src" / "env_install" / "requirements-mps.txt").write_text("", encoding="utf-8")
    (source / "src" / "env_install" / "requirements-pip-options.txt").write_text("", encoding="utf-8")
    (source / "src" / "env_install" / "requirements.txt").write_text("", encoding="utf-8")
    (app / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (app / "settings.yaml").write_text("Media Path: /Users/nick/Movies/test.mp4\n", encoding="utf-8")
    (app / "capture" / "native" / "windows" / "bridge.dll").write_bytes(b"windows")
    (app / "xr_viewer" / "native" / "macos" / "libfilament_bridge.dylib").write_bytes(b"mac")
    (rtmp / "runtime-manifest.json").write_text("{}", encoding="utf-8")
    (rtmp / "d2s-ffmpeg-darwin-arm64.zip").write_bytes(b"arm64")
    (rtmp / "d2s-ffmpeg-windows-amd64.zip").write_bytes(b"windows")
    (rtmp / "mediamtx_darwin_arm64.tar.gz").write_bytes(b"arm64")
    (rtmp / "mediamtx_v1.20.1_darwin_amd64.tar.gz").write_bytes(b"amd64")
    (rtmp / "mediamtx_linux_amd64.tar.gz").write_bytes(b"linux")
    (rtmp / "mediamtx.macos.yml").write_text("paths: {}\n", encoding="utf-8")

    output = tmp_path / "Desktop2Stereo_v3.0beta03-macOS-arm64.7z"
    result = subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--source-root",
            str(source),
            "--output",
            str(output),
            "--seven-zip",
            SEVEN_ZIP,
            "--skip-machine-checks",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    listing = subprocess.run(
        [SEVEN_ZIP, "l", "-ba", str(output)], capture_output=True, text=True, check=True
    ).stdout
    assert "README_MACOS.txt" in listing
    assert "desktop2stereo/main.py" in listing
    assert "d2s-ffmpeg-darwin-arm64.zip" in listing
    assert "mediamtx_darwin_arm64.tar.gz" in listing
    assert ".dll" not in listing
    assert "windows-amd64.zip" not in listing
    assert "darwin_amd64.tar.gz" not in listing
    assert "linux_amd64.tar.gz" not in listing
    assert "settings.yaml" not in listing
    assert (tmp_path / (output.name + ".sha256")).is_file()
