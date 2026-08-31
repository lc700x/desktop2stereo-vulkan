from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from streaming import runtime_manager
from path_config import APP_ROOT


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)


def _write_zip_with_symlink(path: Path, members: dict[str, bytes], symlink: tuple[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
        name, target = symlink
        info = zipfile.ZipInfo(name)
        info.create_system = 3  # Unix
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, target)


def test_ensure_runtime_installs_complete_ffmpeg_package(
    tmp_path: Path, monkeypatch
) -> None:
    manifest = {
        "schema_version": 1,
        "runtimes": {
            "Windows-amd64": {
                "ffmpeg_archive": "ffmpeg.zip",
                "mediamtx_archive": "mediamtx.zip",
                "ffmpeg_executable": "ffmpeg/bin/ffmpeg.exe",
                "mediamtx_executable": "mediamtx/mediamtx.exe",
                "mediamtx_config": "mediamtx.yml",
                "mediamtx_template": "mediamtx/mediamtx.yml",
            }
        },
    }
    (tmp_path / "runtime-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    _write_zip(
        tmp_path / "ffmpeg.zip",
        {
            "ffmpeg/bin/ffmpeg.exe": b"ffmpeg",
            "ffmpeg/bin/avcodec.dll": b"codec",
            "ffmpeg/lib/avcodec.lib": b"import-library",
        },
    )
    _write_zip(
        tmp_path / "mediamtx.zip",
        {
            "mediamtx.exe": b"mediamtx",
            "mediamtx.yml": b"paths: {}",
        },
    )
    monkeypatch.setattr(runtime_manager, "_runtime_key", lambda: "Windows-amd64")

    ffmpeg, mediamtx, config = runtime_manager.ensure_runtime(tmp_path)

    assert ffmpeg.read_bytes() == b"ffmpeg"
    assert (tmp_path / "ffmpeg/bin/avcodec.dll").read_bytes() == b"codec"
    assert (tmp_path / "ffmpeg/lib/avcodec.lib").read_bytes() == b"import-library"
    assert mediamtx.read_bytes() == b"mediamtx"
    assert config.read_bytes() == b"paths: {}"


def test_zip_symlink_members_extract_as_symlinks(tmp_path: Path) -> None:
    # The darwin FFmpeg zip stores versioned aliases (libavdevice.63.dylib)
    # as symlinks; zipfile.extractall() would materialize them as plain
    # files and dyld would reject them ("not a mach-o file").
    _write_zip_with_symlink(
        tmp_path / "bundle.zip",
        {"pkg/lib/libavdevice.63.1.101.dylib": b"MACHO"},
        ("pkg/lib/libavdevice.63.dylib", "libavdevice.63.1.101.dylib"),
    )
    destination = tmp_path / "out"
    destination.mkdir()

    runtime_manager._safe_extract(tmp_path / "bundle.zip", destination)

    alias = destination / "pkg/lib/libavdevice.63.dylib"
    assert alias.is_symlink()
    assert os.readlink(alias) == "libavdevice.63.1.101.dylib"
    assert (destination / "pkg/lib/libavdevice.63.1.101.dylib").read_bytes() == b"MACHO"


def test_safe_extract_rejects_symlink_escaping_archive(tmp_path: Path) -> None:
    _write_zip_with_symlink(
        tmp_path / "evil.zip",
        {},
        ("pkg/escape.dylib", "../../../../etc/passwd"),
    )
    destination = tmp_path / "out"
    destination.mkdir()

    with pytest.raises(ValueError, match="unsafe archive symlink"):
        runtime_manager._safe_extract(tmp_path / "evil.zip", destination)


def test_repair_darwin_dylib_paths_is_noop_without_runner_refs(tmp_path: Path) -> None:
    if sys.platform != "darwin":
        pytest.skip("darwin-only repair")
    binary = tmp_path / "ffmpeg"
    binary.write_bytes(b"not-a-macho")
    before = binary.read_bytes()

    runtime_manager._repair_darwin_dylib_paths(binary)

    assert binary.read_bytes() == before  # nothing to rewrite, no changes


@pytest.mark.skipif(sys.platform != "darwin", reason="requires macOS runtime archives")
def test_darwin_runtime_extracts_runnable_ffmpeg(tmp_path: Path) -> None:
    if shutil.which("install_name_tool") is None:
        pytest.skip("requires Xcode Command Line Tools")
    rtmp = APP_ROOT / "streaming" / "rtmp"
    manifest = json.loads((rtmp / "runtime-manifest.json").read_text(encoding="utf-8"))
    entry = manifest["runtimes"].get("Darwin-arm64") or manifest["runtimes"].get("Darwin-amd64")
    if entry is None:
        pytest.skip("no darwin runtime entry")
    for name in ("runtime-manifest.json", entry["ffmpeg_archive"], entry["mediamtx_archive"]):
        shutil.copy2(rtmp / name, tmp_path / name)

    ffmpeg, _mediamtx, _config = runtime_manager.ensure_runtime(tmp_path)

    result = subprocess.run(
        [str(ffmpeg), "-version"], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert "ffmpeg version" in (result.stdout + result.stderr)


@pytest.mark.skipif(sys.platform != "darwin", reason="requires a repaired macOS runtime")
def test_repair_darwin_dylib_paths_is_noop_when_already_repaired(tmp_path: Path) -> None:
    if shutil.which("install_name_tool") is None or shutil.which("otool") is None:
        pytest.skip("requires Xcode Command Line Tools")
    rtmp = APP_ROOT / "streaming" / "rtmp"
    ffmpeg_bin = rtmp / "ffmpeg" / "bin" / "ffmpeg"
    if not ffmpeg_bin.is_file():
        pytest.skip("darwin ffmpeg runtime not extracted")
    lib_dir = ffmpeg_bin.parent.parent / "lib"
    dylibs = sorted(
        path for path in lib_dir.glob("lib*.dylib") if not path.is_symlink()
    )
    if not dylibs:
        pytest.skip("no dylibs in the extracted runtime")

    def snapshot() -> dict:
        state = {}
        for path in dylibs:
            id_line = subprocess.run(
                ["otool", "-D", str(path)], capture_output=True, text=True
            ).stdout.strip()
            state[str(path)] = (id_line, path.stat().st_mtime_ns)
        return state

    before = snapshot()
    # A second repair over an already-@rpath runtime must not rewrite any
    # install name (install_name_tool invalidates ad-hoc signatures even when
    # the replacement equals the current id, so the rewrite must be skipped).
    runtime_manager._repair_darwin_dylib_paths(ffmpeg_bin)
    after = snapshot()

    assert before == after
