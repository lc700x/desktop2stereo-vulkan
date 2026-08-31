from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path


def _runtime_key() -> str:
    system = platform.system()
    machine = platform.machine().lower()
    arch = "arm64" if machine in {"arm64", "aarch64"} else "amd64"
    return f"{system}-{arch}"


def _safe_extract(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.infolist()
            for member in members:
                target = (destination / member.filename).resolve()
                if destination_resolved not in target.parents and target != destination_resolved:
                    raise ValueError(f"unsafe archive member: {member.filename}")
                mode = (member.external_attr >> 16) & 0o170000
                if mode == 0o120000:  # POSIX symlink
                    # zipfile.extractall() would write the link target as a
                    # plain file; recreate the symlink so dyld can resolve
                    # versioned aliases like libavdevice.63.dylib.
                    link_target = bundle.read(member).decode("utf-8")
                    resolved = (target.parent / link_target).resolve()
                    if (
                        destination_resolved not in resolved.parents
                        and resolved != destination_resolved
                    ):
                        raise ValueError(f"unsafe archive symlink: {member.filename}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.is_symlink() or target.exists():
                        target.unlink()
                    os.symlink(link_target, target)
                elif member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(member) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
        return
    with tarfile.open(archive, "r:*") as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if destination_resolved not in target.parents and target != destination_resolved:
                raise ValueError(f"unsafe archive member: {member.name}")
        bundle.extractall(destination)


def _repair_darwin_dylib_paths(ffmpeg_bin: Path) -> None:
    """Rewrite the bundled FFmpeg's CI-absolute dylib load paths to @rpath.

    The darwin FFmpeg builds link against ``/Users/runner/work/...`` paths
    from the build machine, so dyld cannot load ``libavdevice.63.dylib`` on
    any user Mac (``Library not loaded`` + SIGKILL). The binaries already
    carry the right LC_RPATH entries (``@loader_path/../lib`` on the binary,
    ``@loader_path`` on the dylibs); only the LC_LOAD_DYLIB / LC_ID_DYLIB
    paths are absolute, so rewriting them to ``@rpath/<name>`` makes the
    shipped runtime self-contained. ``install_name_tool`` invalidates the
    code signature, so an ad-hoc re-sign follows -- required on Apple
    Silicon. Idempotent: no-op once no runner paths remain.
    """
    if sys.platform != "darwin":
        return
    tool = shutil.which("install_name_tool")
    if tool is None:
        return
    lib_dir = ffmpeg_bin.parent.parent / "lib"
    targets = [ffmpeg_bin]
    # The darwin FFmpeg package ships ffprobe next to ffmpeg; it links the
    # same CI-absolute dylibs and must be repaired (and re-signed) too.
    for sibling in ("ffprobe",):
        probe_bin = ffmpeg_bin.parent / sibling
        if probe_bin.is_file():
            targets.append(probe_bin)
    if lib_dir.is_dir():
        targets.extend(
            path for path in sorted(lib_dir.glob("lib*.dylib")) if path.is_symlink() is False
        )
    runner_dep = re.compile(r"/Users/runner/[^\s]+/lib/[^\s]+\.dylib")
    changed = False
    for target in targets:
        info = subprocess.run(
            ["otool", "-L", str(target)], capture_output=True, text=True
        ).stdout
        for dep in runner_dep.findall(info):
            subprocess.run(
                [tool, "-change", dep, f"@rpath/{Path(dep).name}", str(target)],
                check=True,
            )
            changed = True
        if target.suffix == ".dylib":
            # Keep the id consistent so sibling dylibs resolve by the same
            # @rpath name; only rewrite when it differs so every app start is
            # a true no-op (install_name_tool invalidates the ad-hoc code
            # signature even when the new id equals the old one, and the
            # re-sign below must follow any rewrite).
            current_id = subprocess.run(
                ["otool", "-D", str(target)], capture_output=True, text=True
            ).stdout.strip()
            expected_id = f"@rpath/{target.name}"
            if not current_id or current_id.splitlines()[-1] != expected_id:
                subprocess.run(
                    [tool, "-id", expected_id, str(target)], check=False
                )
                changed = True
    if changed:
        codesign = shutil.which("codesign")
        if codesign is not None:
            subprocess.run([codesign, "--force", "--sign", "-", *targets], check=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_manifest(manifest: dict, entry: dict, runtime_key: str) -> None:
    """Reject an unprovenanced runtime before it can be published."""
    if int(manifest.get("schema_version", 1)) < 2:
        return
    provenance = manifest.get("provenance")
    required_provenance = {
        "ffmpeg_source_repository",
        "ffmpeg_source_ref",
        "ffmpeg_build_id",
        "ffmpeg_configuration",
        "compiler",
    }
    if not isinstance(provenance, dict) or not required_provenance.issubset(provenance):
        missing = sorted(required_provenance - set(provenance or {}))
        raise RuntimeError(
            f"streaming runtime manifest provenance is incomplete for {runtime_key}: {missing}"
        )
    archive_hash = str(entry.get("ffmpeg_archive_sha256", "")).strip().casefold()
    if len(archive_hash) != 64 or any(char not in "0123456789abcdef" for char in archive_hash):
        raise RuntimeError(
            f"streaming runtime manifest has no valid FFmpeg archive SHA-256 for {runtime_key}"
        )


def _verify_archive(path: Path, expected_hash: str) -> None:
    actual_hash = _sha256(path)
    if actual_hash.casefold() != expected_hash.casefold():
        raise RuntimeError(
            f"streaming runtime archive SHA-256 mismatch: {path.name}; "
            f"expected={expected_hash} actual={actual_hash}"
        )


def ensure_runtime(runtime_root: Path) -> tuple[Path, Path, Path]:
    manifest_path = runtime_root / "runtime-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runtime_key = _runtime_key()
    entry = manifest["runtimes"].get(runtime_key)
    if entry is None:
        raise RuntimeError(f"no streaming runtime registered for {runtime_key}")
    _validate_manifest(manifest, entry, runtime_key)
    ffmpeg = runtime_root / entry["ffmpeg_executable"]
    mediamtx = runtime_root / entry["mediamtx_executable"]
    config = runtime_root / entry.get("mediamtx_config", "mediamtx.yml")
    template = runtime_root / entry.get(
        "mediamtx_template", "mediamtx/mediamtx.yml"
    )
    if ffmpeg.is_file() and mediamtx.is_file() and template.is_file():
        if not config.is_file():
            config.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(template, config)
        if os.name != "nt":
            ffmpeg.chmod(ffmpeg.stat().st_mode | 0o111)
            for sibling in (ffmpeg.parent / "ffprobe",):
                if sibling.is_file():
                    sibling.chmod(sibling.stat().st_mode | 0o111)
            mediamtx.chmod(mediamtx.stat().st_mode | 0o111)
        _repair_darwin_dylib_paths(ffmpeg)
        return ffmpeg, mediamtx, config
    archives = [runtime_root / entry["ffmpeg_archive"], runtime_root / entry["mediamtx_archive"]]
    missing = [str(path) for path in archives if not path.is_file()]
    if not missing and int(manifest.get("schema_version", 1)) >= 2:
        _verify_archive(archives[0], entry["ffmpeg_archive_sha256"])
    if missing:
        raise FileNotFoundError("missing streaming runtime archive(s): " + ", ".join(missing))
    with tempfile.TemporaryDirectory(prefix="d2s-runtime-", dir=runtime_root) as temp_name:
        temp_root = Path(temp_name)
        ffmpeg_extract = temp_root / "ffmpeg"
        mediamtx_extract = temp_root / "mediamtx"
        ffmpeg_extract.mkdir()
        mediamtx_extract.mkdir()
        _safe_extract(archives[0], ffmpeg_extract)
        _safe_extract(archives[1], mediamtx_extract)
        # rglob(name) also matches the archive's top-level DIRECTORY (e.g. the
        # darwin ffmpeg zip extracts to <root>/ffmpeg/bin/ffmpeg); only a file
        # is a usable runtime, so filter is_file() or the wrong node is picked
        # and the package copy lands double-nested (ffmpeg/ffmpeg/bin/ffmpeg).
        def _first_file(root: Path, name: str) -> Path:
            for candidate in root.rglob(name):
                if candidate.is_file():
                    return candidate
            raise FileNotFoundError(f"no {name!r} file extracted under {root}")

        ffmpeg_source = _first_file(ffmpeg_extract, Path(entry["ffmpeg_executable"]).name)
        mediamtx_source = _first_file(mediamtx_extract, Path(entry["mediamtx_executable"]).name)
        template_source = next(mediamtx_extract.rglob("mediamtx.yml"), None)
        ffmpeg_package_source = ffmpeg_source.parent.parent
        ffmpeg_package_destination = runtime_root / Path(entry["ffmpeg_executable"]).parts[0]
        if ffmpeg_package_destination.exists():
            shutil.rmtree(ffmpeg_package_destination)
        # symlinks=True keeps the darwin build's versioned alias symlinks
        # (libavdevice.63.dylib -> libavdevice.63.1.101.dylib); dereferencing
        # them turns dyld resolution into "not a mach-o file".
        shutil.copytree(ffmpeg_package_source, ffmpeg_package_destination, symlinks=True)
        mediamtx.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mediamtx_source, mediamtx)
        if not template.is_file() and template_source is not None:
            template.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(template_source, template)
    if not config.is_file() and template.is_file():
        config.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(template, config)
    if not config.is_file():
        raise FileNotFoundError(f"MediaMTX config template not found: {template}")
    if os.name != "nt":
        ffmpeg.chmod(ffmpeg.stat().st_mode | 0o111)
        for sibling in (ffmpeg.parent / "ffprobe",):
            if sibling.is_file():
                sibling.chmod(sibling.stat().st_mode | 0o111)
        mediamtx.chmod(mediamtx.stat().st_mode | 0o111)
    _repair_darwin_dylib_paths(ffmpeg)
    return ffmpeg, mediamtx, config
