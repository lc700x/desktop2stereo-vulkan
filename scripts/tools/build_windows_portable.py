from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
APP_SOURCE = ROOT / "src" / "desktop2stereo"
SOURCE_ROOT = ROOT / "src"
PYTHON_ARCHIVE_DEFAULT = Path.home() / "Downloads" / "python3.7z"
PYTHON_VERSION = (3, 12, 10)
ROCM_VERSION = "10.1.0a20260822"
BASE_MODEL_CACHE = "models--lc700x--Distill-Any-Depth-Base-hf"
OFFICIAL_PACKAGE_INDEXES = {
    "pypi": "https://pypi.org/simple",
    "rocm": "https://rocm.nightlies.amd.com/whl-multi-arch/",
    "pytorch_cuda": "https://download.pytorch.org/whl/cu128",
    "nvidia": "https://pypi.nvidia.com",
}
PATCH_SCRIPTS = (
    "apply-wc-rocm-patches.py",
    "apply-wc-cuda-patches.py",
    "apply-triton-portable-patches.py",
)


VARIANTS: dict[str, dict[str, Any]] = {
    "rocm-gfx1200-gfx1201": {
        "title": "ROCm gfx1200 + gfx1201",
        "profile": "rocm",
        "requirements": "requirements-rocm7-9000.txt",
        "constraints": {
            "rocm": f"=={ROCM_VERSION}",
            "torch": f"==2.11.0+rocm{ROCM_VERSION}",
            "torchvision": f"==0.26.0+rocm{ROCM_VERSION}",
            "triton-windows": "==3.8.0.post29",
            "wc-rocm": "==0.1.3.post1",
        },
        "expected_arches": ("gfx1200", "gfx1201"),
    },
    "rocm-rdna1-3": {
        "title": "ROCm RDNA1–3",
        "profile": "rocm",
        "requirements": "requirements-rocm7.txt",
        "constraints": {
            "rocm": f"=={ROCM_VERSION}",
            "torch": f"==2.11.0+rocm{ROCM_VERSION}",
            "torchvision": f"==0.26.0+rocm{ROCM_VERSION}",
            "triton-windows": "==3.8.0.post29",
            "wc-rocm": "==0.1.3.post1",
        },
        "expected_arches": (
            "gfx1152",
            "gfx1151",
            "gfx1150",
            "gfx1103",
            "gfx1102",
            "gfx1101",
            "gfx1100",
            "gfx1030",
            "gfx1031",
            "gfx1032",
            "gfx1033",
            "gfx1034",
            "gfx1035",
            "gfx1036",
            "gfx1010",
            "gfx1011",
            "gfx1012",
        ),
    },
    "cuda": {
        "title": "CUDA 12.8",
        "profile": "cuda",
        "requirements": "requirements-cuda.txt",
        "constraints": {
            "torch": "==2.11.0+cu128",
            "torchvision": "==0.26.0+cu128",
            "triton-windows": "==3.6.0.post26",
            "tensorrt-cu12": "==10.14.1.48.post1",
            "tensorrt-cu12-bindings": "==10.14.1.48.post1",
            "tensorrt-cu12-libs": "==10.14.1.48.post1",
            "packaging": "==26.0",
            "wheel": "==0.45.1",
            "pygltflib": "==1.16.5",
            "pyyaml": "==6.0.3",
            "pywin32": "==312",
        },
        "expected_arches": (),
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    log_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    printable = subprocess.list2cmdline(command)
    print(f"\n> {printable}", flush=True)
    if log_path is None:
        return subprocess.run(command, cwd=cwd, env=env, text=True, check=True)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)
    return subprocess.CompletedProcess(command, code, "", "")


def current_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def find_7z(explicit: str | None) -> Path:
    candidates = [
        Path(explicit).expanduser() if explicit else None,
        Path(shutil.which("7z") or "") if shutil.which("7z") else None,
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "7-Zip" / "7z.exe",
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "7-Zip" / "7z.exe",
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("7-Zip was not found. Pass --7z with the path to 7z.exe.")


def clean_environment(runtime: Path, *, profile: str, slug: str) -> dict[str, str]:
    env = os.environ.copy()
    for name in tuple(env):
        if name.startswith("PIP_"):
            env.pop(name, None)
    for name in (
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONUSERBASE",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "PIP_TARGET",
        "PIP_PREFIX",
        "PIP_USER",
        "PIP_CONFIG_FILE",
        "CC",
        "CXX",
        "HIP_PATH",
        "HIP_CLANG_PATH",
        "ROCM_PATH",
        "ROCM_HOME",
        "CUDA_PATH",
        "CUDA_HOME",
        "INCLUDE",
        "LIB",
        "LIBPATH",
        "VCINSTALLDIR",
        "VCToolsInstallDir",
        "WindowsSdkDir",
        "WindowsSDKVersion",
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_NO_WINSDK",
        "CMAKE_PREFIX_PATH",
        "CMAKE_ARGS",
        "HF_HOME",
        "TRANSFORMERS_CACHE",
        "TORCH_HOME",
    ):
        env.pop(name, None)

    system_root = env.get("SystemRoot", r"C:\Windows")
    env["PYTHONHOME"] = str(runtime)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PIP_CONFIG_FILE"] = "NUL"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["D2S_PORTABLE_ROOT"] = str(runtime.parent.parent)
    env["D2S_PORTABLE_FLAVOR"] = slug
    path_entries = [str(runtime), str(runtime / "Scripts")]

    if profile == "rocm":
        site = runtime / "Lib" / "site-packages"
        sdk = site / "_rocm_sdk_core"
        libraries = site / "_rocm_sdk_libraries"
        env["ROCM_PATH"] = str(sdk)
        env["HIP_PATH"] = str(sdk)
        env["HIP_CLANG_PATH"] = str(sdk / "lib" / "llvm" / "bin")
        env["CC"] = str(sdk / "lib" / "llvm" / "bin" / "clang-cl.exe")
        env["CXX"] = env["CC"]
        env["TRITON_NO_WINSDK"] = "1"
        path_entries.extend(
            [
                str(sdk / "bin"),
                str(sdk / "lib" / "llvm" / "bin"),
                str(libraries / "bin"),
                str(libraries / "lib"),
                str(site / "_rocm_sdk_devel" / "bin"),
            ]
        )

    path_entries.extend(
        [
            str(runtime.parent / "desktop2stereo" / "msvcstub" / "bin"),
            str(Path(system_root) / "System32"),
            system_root,
            str(Path(system_root) / "System32" / "Wbem"),
            str(Path(system_root) / "System32" / "WindowsPowerShell" / "v1.0"),
        ]
    )
    env["PATH"] = os.pathsep.join(dict.fromkeys(path_entries))
    return env


def windows_short_path(path: Path) -> str:
    path.mkdir(parents=True, exist_ok=True)
    buffer = ctypes.create_unicode_buffer(32768)
    length = ctypes.windll.kernel32.GetShortPathNameW(str(path), buffer, len(buffer))
    short_path = buffer.value if length else str(path)
    try:
        short_path.encode("ascii")
    except UnicodeEncodeError as exc:
        raise RuntimeError(f"A Windows 8.3 path is required for GPU compiler caches: {path}") from exc
    return short_path


def validate_torch_compile(
    runtime: Path,
    profile: str,
    slug: str,
    package_root: Path,
    expected_arches: tuple[str, ...] = (),
) -> dict[str, Any]:
    cache_root = package_root / "runtime-cache" / slug
    triton_cache = windows_short_path(cache_root / "triton")
    inductor_cache = windows_short_path(cache_root / "inductor")
    env = clean_environment(runtime, profile=profile, slug=slug)
    env["TRITON_CACHE_DIR"] = triton_cache
    env["TORCHINDUCTOR_CACHE_DIR"] = inductor_cache
    env["D2S_PORTABLE_EXPECTED_ARCHES"] = json.dumps(expected_arches)
    if profile == "rocm":
        msvc_stub = package_root / "src" / "desktop2stereo" / "msvcstub"
        env["INCLUDE"] = str(msvc_stub)
        env["LIB"] = str(msvc_stub / "lib")
        env["LIBPATH"] = env["LIB"]
    code = r"""
import json
import os
import torch

if not torch.cuda.is_available():
    print(json.dumps({"status": "skipped", "reason": "No supported GPU or driver is available"}))
else:
    device = torch.cuda.get_device_name(0)
    expected_arches = json.loads(os.environ.get("D2S_PORTABLE_EXPECTED_ARCHES", "[]"))
    properties = torch.cuda.get_device_properties(0)
    architecture = str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]
    if expected_arches and architecture and architecture not in expected_arches:
        print(json.dumps({
            "status": "skipped",
            "reason": "Build host GPU architecture is not included in this package",
            "device": device,
            "architecture": architecture,
            "package_architectures": expected_arches,
        }))
    else:
        sample = torch.randn((128, 128), device="cuda")
        compiled = torch.compile(lambda value: torch.sin(value) * 2.0 + value, fullgraph=True)
        result = compiled(sample)
        torch.cuda.synchronize()
        if not torch.isfinite(result).all().item():
            raise RuntimeError("torch.compile produced non-finite values")
        print(json.dumps({
            "status": "passed",
            "torch": torch.__version__,
            "device": device,
            "architecture": architecture or None,
            "finite": True,
        }))
"""
    result = subprocess.run(
        [str(runtime / "python.exe"), "-c", code],
        cwd=package_root / "src" / "desktop2stereo",
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=300,
    )
    if result.returncode:
        output = (result.stdout + result.stderr).strip()
        raise RuntimeError(f"Bundled {profile} torch.compile smoke test failed for {slug}:\n{output[-6000:]}")
    try:
        report = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Bundled {profile} torch.compile smoke test returned no result for {slug}") from exc
    return report


def app_ignore(source: str, names: list[str]) -> set[str]:
    source_path = Path(source).resolve()
    models_path = (APP_SOURCE / "models").resolve()
    base_cache_path = (models_path / BASE_MODEL_CACHE).resolve()
    ignored: set[str] = set()
    for name in names:
        path = source_path / name
        if (
            name in {
                "__pycache__",
                ".pytest_cache",
                ".ruff_cache",
                ".mypy_cache",
                "logs",
                "runtime-crash.log",
                "settings.yaml",
                "settings.yaml.tmp",
            }
            or name.endswith((".pyc", ".pyo", ".log"))
        ):
            ignored.add(name)
        if source_path == (APP_SOURCE / "gui").resolve() and name == "flet_clients":
            ignored.add(name)
        if source_path == models_path and path.resolve() != base_cache_path:
            ignored.add(name)
        if source_path == base_cache_path and name == "blobs":
            ignored.add(name)
        if source_path == base_cache_path and name.lower().endswith((".mgx", ".onnx", ".trt")):
            ignored.add(name)
    return ignored


def copy_application(stage: Path) -> None:
    source_app = stage / "src" / "desktop2stereo"
    if not source_app.exists():
        shutil.copytree(APP_SOURCE, source_app, ignore=app_ignore)
    patch_dir = stage / "src" / "env_install"
    patch_dir.mkdir(parents=True, exist_ok=True)
    for filename in PATCH_SCRIPTS:
        source = ROOT / "src" / "env_install" / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, patch_dir / filename)
    for filename in ("LICENSE", "NOTICE"):
        source = ROOT / filename
        if source.is_file():
            shutil.copy2(source, stage / filename)


def copy_vc_runtime(runtime: Path) -> list[str]:
    program_files_x86 = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
    redist_root = program_files_x86 / "Microsoft Visual Studio" / "2022" / "BuildTools" / "VC" / "Redist" / "MSVC"
    candidates = sorted(
        redist_root.glob("*/x64/Microsoft.VC143.CRT"),
        key=lambda path: path.parts[-4],
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"Visual C++ x64 redistributable files not found under {redist_root}")

    source = candidates[0]
    copied: list[str] = []
    for item in source.iterdir():
        if item.is_file() and item.suffix.lower() in {".dll", ".manifest"}:
            shutil.copy2(item, runtime / item.name)
            copied.append(item.name)
    if not any(name.lower() == "vcruntime140.dll" for name in copied):
        raise RuntimeError(f"The Visual C++ redistributable is incomplete: {source}")

    license_candidates = (
        source.parent.parent / "redist.txt",
        source.parent.parent.parent / "redist.txt",
        redist_root / "redist.txt",
    )
    for candidate in license_candidates:
        if candidate.is_file():
            shutil.copy2(candidate, runtime / "VC_REDIST_LICENSE.txt")
            break
    return sorted(copied, key=str.lower)


def base_model_manifest(app_root: Path) -> dict[str, Any]:
    cache = app_root / "models" / BASE_MODEL_CACHE
    ref = cache / "refs" / "main"
    if not ref.is_file():
        raise FileNotFoundError(f"Default Distill Base model cache is missing: {ref}")
    revision = ref.read_text(encoding="utf-8").strip()
    snapshot = cache / "snapshots" / revision
    required = ("config.json", "model.safetensors", "preprocessor_config.json")
    missing = [name for name in required if not (snapshot / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete Distill Base snapshot {revision}: {', '.join(missing)}")
    weights = snapshot / "model.safetensors"
    return {
        "model_id": "lc700x/Distill-Any-Depth-Base-hf",
        "revision": revision,
        "weights_sha256": sha256_file(weights),
        "weights_size_bytes": weights.stat().st_size,
        "snapshot_files": {
            name: sha256_file(snapshot / name)
            for name in required
        },
    }


def make_constraints(variant: dict[str, Any], target: Path) -> None:
    rows = [
        "# Generated by scripts/tools/build_windows_portable.py.",
        "# Exact accelerator and reproducibility constraints for this Windows package.",
    ]
    for name, value in sorted(variant["constraints"].items()):
        rows.append(f"{name}{value}")
    if variant["profile"] == "rocm":
        rows.extend(
            [
                "pygltflib==1.16.5",
                "pyyaml==6.0.3",
                "pywin32==312",
            ]
        )
    target.write_text("\n".join(rows) + "\n", encoding="utf-8")


def make_official_requirements(
    source: Path,
    target: Path,
    *,
    rocm_arches: tuple[str, ...] = (),
    package_pins: dict[str, str] | None = None,
) -> None:
    """Copy a requirements file while removing inherited indexes and mirror includes."""
    rows: list[str] = []
    architecture_extras = ",".join(f"device-{architecture}" for architecture in rocm_arches)
    normalized_pins = {
        name.casefold(): version for name, version in (package_pins or {}).items()
    }
    for original in source.read_text(encoding="utf-8").splitlines():
        line = original.strip()
        if not line or line.startswith("#"):
            rows.append(original)
            continue
        if re.match(r"^(?:-r\s|--requirement\b|--index-url\b|--extra-index-url\b|--find-links\b)", line, re.I):
            continue
        if architecture_extras:
            line = re.sub(r"\bdevice-all\b", architecture_extras, line)
        line = line.replace("https://ghfast.top/github.com/", "https://github.com/")
        if re.search(r"(?:ghfast\.top|huaweicloud\.com|mirrors\.aliyun\.com)", line, re.I):
            raise ValueError(f"Non-official package source in requirements: {line}")
        requirement = re.match(
            r"^(?P<name>[A-Za-z0-9_.-]+)(?P<extras>\[[^\]]+\])?",
            line,
        )
        if requirement:
            pinned_version = normalized_pins.get(requirement.group("name").casefold())
            if pinned_version:
                extras = requirement.group("extras") or ""
                marker_offset = line.find(";")
                marker = line[marker_offset:] if marker_offset >= 0 else ""
                line = f"{requirement.group('name')}{extras}{pinned_version}{marker}"
        rows.append(line)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_portable_default_settings(runtime: Path, app_root: Path, profile: str, slug: str) -> None:
    settings_path = app_root / "settings.yaml"
    if settings_path.exists():
        if not settings_path.read_text(encoding="utf-8").startswith("# Generated portable defaults"):
            raise RuntimeError(f"Refusing to overwrite an existing settings file: {settings_path}")
    else:
        settings_path.write_text(
            "# Generated portable defaults seed; replaced with GUI defaults during build.\n"
            "Depth Model: Distill-Any-Depth-Base\n"
            "Depth Resolution: 518\n"
            "Anti-aliasing: 2\n",
            encoding="utf-8",
        )
    env = clean_environment(runtime, profile=profile, slug=slug)
    env["PYTHONPATH"] = str(runtime.parent)
    code = "\n".join(
        [
            "from pathlib import Path",
            "import yaml",
            "from gui.config import DEFAULTS",
            "config = dict(DEFAULTS)",
            'config.update({"Depth Model": "Distill-Any-Depth-Base", "Depth Resolution": 518, "Anti-aliasing": 2, "Depth Antialias Strength": 2.0, "OpenVINO": False})',
            'config["Monitor Identity"] = None',
            'config["Stereo Output Identity"] = None',
            f"target = Path({json.dumps(str(settings_path))})",
            'target.write_text("# Generated portable defaults from gui.config.DEFAULTS.\\n" + yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")',
        ]
    )
    result = subprocess.run(
        [str(runtime / "python.exe"), "-c", code],
        env=env,
        cwd=app_root,
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(f"Could not generate portable default settings: {result.stderr or result.stdout}")


def official_index_args(profile: str) -> list[str]:
    args = ["--index-url", OFFICIAL_PACKAGE_INDEXES["pypi"]]
    if profile == "rocm":
        args.extend(["--extra-index-url", OFFICIAL_PACKAGE_INDEXES["rocm"]])
    else:
        args.extend(
            [
                "--extra-index-url",
                OFFICIAL_PACKAGE_INDEXES["pytorch_cuda"],
                "--extra-index-url",
                OFFICIAL_PACKAGE_INDEXES["nvidia"],
            ]
        )
    return args


def merge_install_reports(paths: list[Path], destination: Path) -> list[dict[str, Any]]:
    combined: list[dict[str, Any]] = []
    for path in paths:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            combined.extend(data.get("install", []))
    destination.write_text(
        json.dumps({"version": 1, "install": combined}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return combined


def write_dependency_files(stage: Path, runtime: Path, reports: list[dict[str, Any]]) -> None:
    dependencies = stage / "dependencies"
    dependencies.mkdir(parents=True, exist_ok=True)
    freeze = subprocess.run(
        [str(runtime / "python.exe"), "-m", "pip", "freeze", "--all"],
        env=clean_environment(runtime, profile="cuda", slug="inventory"),
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    (dependencies / "requirements-resolved.txt").write_text(freeze, encoding="utf-8")

    report_path = dependencies / "pip-install-report.json"
    report_path.write_text(
        json.dumps({"version": 1, "install": reports}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rows = ["package\tversion\tsha256\twheel_url"]
    for record in sorted(reports, key=lambda item: (item.get("metadata", {}).get("name", "").lower(), item.get("metadata", {}).get("version", ""))):
        metadata = record.get("metadata", {})
        download = record.get("download_info", {})
        archive = download.get("archive_info", {})
        hashes = archive.get("hashes", {})
        digest = hashes.get("sha256") or archive.get("hash", "")
        digest = digest.removeprefix("sha256=")
        fields = (
            str(metadata.get("name", "")),
            str(metadata.get("version", "")),
            str(digest),
            str(download.get("url", "")),
        )
        rows.append("\t".join(value.replace("\t", " ").replace("\n", " ") for value in fields))
    (dependencies / "wheel-sources-sha256.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def app_file_hashes(app_root: Path, destination: Path) -> int:
    rows: list[str] = []
    for path in sorted(app_root.rglob("*")):
        if path.is_file():
            rows.append(f"{sha256_file(path)}  {path.relative_to(app_root).as_posix()}")
    destination.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return len(rows)


def installed_versions(runtime: Path, profile: str) -> dict[str, str]:
    code = (
        "import importlib.metadata as m, json; "
        "print(json.dumps({d.metadata['Name']: d.version for d in m.distributions() if d.metadata.get('Name')}))"
    )
    env = clean_environment(runtime, profile=profile, slug="inventory")
    result = subprocess.run(
        [str(runtime / "python.exe"), "-c", code],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


def device_inventory(versions: dict[str, str]) -> dict[str, list[str]]:
    rocm: set[str] = set()
    torch: set[str] = set()
    torchvision: set[str] = set()
    names: list[str] = []
    pattern = re.compile(r"(gfx[0-9a-z-]+)", re.IGNORECASE)
    for name in versions:
        lower = name.lower()
        if "device" not in lower or "gfx" not in lower:
            continue
        names.append(name)
        arch = pattern.search(lower)
        if not arch:
            continue
        tag = arch.group(1).replace("-", "")
        if "torchvision" in lower:
            torchvision.add(tag)
        elif "torch" in lower:
            torch.add(tag)
        elif "rocm" in lower:
            rocm.add(tag)
    supported = sorted(rocm & torch & torchvision)
    return {
        "rocm_sdk": sorted(rocm),
        "torch": sorted(torch),
        "torchvision": sorted(torchvision),
        "intersection": supported,
        "device_packages": sorted(names, key=str.lower),
    }


def launcher_text(slug: str, profile: str) -> str:
    lines = [
        "@echo off",
        "setlocal DisableDelayedExpansion",
        'set "PORTABLE_ROOT=%~sdp0"',
        'set "SRC_ROOT=%PORTABLE_ROOT%src"',
        'set "APP_DIR=%SRC_ROOT%\\desktop2stereo"',
        'set "PYTHON_ROOT=%SRC_ROOT%\\python3"',
        'set "PYTHON_EXE=%PYTHON_ROOT%\\python.exe"',
        'if not exist "%PYTHON_EXE%" (',
        '    echo Bundled Python was not found: "%PYTHON_EXE%"',
        "    pause",
        "    exit /b 1",
        ")",
        "for %%V in (PYTHONPATH PYTHONHOME PYTHONUSERBASE VIRTUAL_ENV CONDA_PREFIX PIP_TARGET PIP_PREFIX PIP_USER PIP_CONFIG_FILE CC CXX HIP_PATH HIP_CLANG_PATH ROCM_PATH ROCM_HOME CUDA_PATH CUDA_HOME INCLUDE LIB LIBPATH VCINSTALLDIR VCToolsInstallDir WindowsSdkDir WindowsSDKVersion TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR) do set \"%%V=\"",
        'set "PYTHONHOME=%PYTHON_ROOT%"',
        'set "PYTHONPATH=%SRC_ROOT%"',
        'set "PYTHONNOUSERSITE=1"',
        'set "PYTHONUTF8=1"',
        'set "PIP_CONFIG_FILE=NUL"',
        'set "D2S_PORTABLE_ROOT=%PORTABLE_ROOT%"',
        f'set "D2S_PORTABLE_FLAVOR={slug}"',
        f'set "D2S_PORTABLE_CACHE=%PORTABLE_ROOT%runtime-cache\\{slug}"',
        'set "HF_HOME=%APP_DIR%\\models\\huggingface"',
        'set "TRANSFORMERS_CACHE=%HF_HOME%\\transformers"',
        'set "TORCH_HOME=%APP_DIR%\\models\\torch"',
        'set "TRITON_CACHE_DIR=%D2S_PORTABLE_CACHE%\\triton"',
        'set "TORCHINDUCTOR_CACHE_DIR=%D2S_PORTABLE_CACHE%\\inductor"',
        'if not exist "%D2S_PORTABLE_CACHE%" mkdir "%D2S_PORTABLE_CACHE%"',
    ]
    path_entries = [
        "%PYTHON_ROOT%",
        "%PYTHON_ROOT%\\Scripts",
    ]
    if profile == "rocm":
        lines.extend(
            [
                'set "ROCM_PATH=%PYTHON_ROOT%\\Lib\\site-packages\\_rocm_sdk_core"',
                'set "HIP_PATH=%ROCM_PATH%"',
                'set "HIP_CLANG_PATH=%ROCM_PATH%\\lib\\llvm\\bin"',
                'set "CC=%HIP_CLANG_PATH%\\clang-cl.exe"',
                'set "CXX=%CC%"',
                'set "TRITON_NO_WINSDK=1"',
            ]
        )
        path_entries.extend(
            [
                "%ROCM_PATH%\\bin",
                "%HIP_CLANG_PATH%",
                "%PYTHON_ROOT%\\Lib\\site-packages\\_rocm_sdk_libraries\\bin",
                "%PYTHON_ROOT%\\Lib\\site-packages\\_rocm_sdk_libraries\\lib",
                "%PYTHON_ROOT%\\Lib\\site-packages\\_rocm_sdk_devel\\bin",
            ]
        )
    path_entries.extend(
        [
            "%SystemRoot%\\System32",
            "%SystemRoot%",
            "%SystemRoot%\\System32\\Wbem",
            "%SystemRoot%\\System32\\WindowsPowerShell\\v1.0",
        ]
    )
    lines.extend(
        [
            'set "PATH=' + ";".join(path_entries) + '"',
            'cd /d "%APP_DIR%"',
            'if errorlevel 1 exit /b 1',
            'echo Starting Desktop2Stereo portable (' + slug + ')',
            '"%PYTHON_EXE%" "%APP_DIR%\\main.py" %*',
            "exit /b %errorlevel%",
            "",
        ]
    )
    return "\r\n".join(lines)


def readme_text(slug: str, variant: dict[str, Any], model: dict[str, Any], vc_files: list[str]) -> tuple[str, str]:
    backend = "ROCm 7 / Triton Windows 3.8.0.post29" if variant["profile"] == "rocm" else "CUDA 12.8 / Triton Windows 3.6.0.post26"
    arch = ""
    if variant["profile"] == "rocm":
        arch = "\nDevice package set: " + ", ".join(variant["expected_arches"]) + "\n"
    en = f"""Desktop2Stereo Windows Portable

Build flavor: {slug}
Accelerator runtime: {backend}
Python: 3.12.10 x64, bundled under src/python3
Default depth model: Distill-Any-Depth-Base, offline snapshot {model["revision"]}
Default depth resolution: 518
Default anti-aliasing: 2
{arch}
Run Launch Desktop2Stereo.bat. The package does not install Python or GPU software on Windows.
The graphics driver is supplied by the target computer. OpenXR also requires a compatible
headset runtime such as Virtual Desktop VDXR, SteamVR, or Meta Quest Link.

The first run can take longer while the target GPU builds its local Triton, Inductor,
TensorRT, or MIGraphX cache. Those device-specific engine files are intentionally not shipped.
The bundled Distill Base weights work without downloading a model. Other model weights are
downloaded when selected.

Python packages and compiler SDK files are in src/python3. The launcher rejects ambient
Python, CUDA, ROCm, compiler, and user-site package paths and uses package-local caches.

MSVC runtime DLLs included in src/python3: {", ".join(vc_files)}
Resolved Python dependencies: dependencies/requirements-resolved.txt
Wheel URLs and SHA256 values: dependencies/wheel-sources-sha256.tsv
"""
    zh = f"""Desktop2Stereo Windows Portable

版本：{slug}
加速运行时：{backend}
Python：内置 Python 3.12.10 x64，位于 src/python3
默认深度模型：Distill-Any-Depth-Base，离线权重快照 {model["revision"]}
默认深度分辨率：518
默认抗锯齿：2
{arch}
双击 Launch Desktop2Stereo.bat 启动。本包不会向 Windows 安装 Python 或 GPU 软件。
显卡驱动由目标电脑提供。OpenXR 还需要兼容的头显运行时，例如 Virtual Desktop VDXR、
SteamVR 或 Meta Quest Link。

首次启动时，程序可能需要为目标 GPU 生成 Triton、Inductor、TensorRT 或 MIGraphX 缓存；
这些与设备相关的引擎文件不随发行包提供。内置的 Distill Base 权重可离线使用；
选择其他模型时会按需下载权重。

Python 包和编译 SDK 都在 src/python3 中。启动器会隔离外部 Python、CUDA、ROCm、
编译器和用户级 Python 包路径，并将编译缓存放在包内。

内置 MSVC 运行库：{", ".join(vc_files)}
已解析的 Python 依赖：dependencies/requirements-resolved.txt
Wheel 来源和 SHA256：dependencies/wheel-sources-sha256.tsv
"""
    return en, zh


def validate_imports(runtime: Path, profile: str, slug: str, app_root: Path) -> dict[str, Any]:
    env = clean_environment(runtime, profile=profile, slug=slug)
    env["PYTHONPATH"] = str(runtime.parent)
    env["D2S_RUNTIME_PATHS_FILE"] = str(app_root / "runtime_paths.env")
    env["TRITON_CACHE_DIR"] = str(runtime.parent.parent / "runtime-cache" / "validation" / slug / "triton")
    env["TORCHINDUCTOR_CACHE_DIR"] = str(runtime.parent.parent / "runtime-cache" / "validation" / slug / "inductor")
    code = r"""
import importlib.metadata as metadata
import importlib.util
import json
import sys
from pathlib import Path

import numpy
import cv2
import torch
import torchvision
import triton
import transformers
import flet
import onnx
import onnxscript
import sounddevice
import soundcard
import windows_capture
import wincam
import pynput
import pyperclip
import pygltflib
import xr
import vulkan
from OpenGL import GL
import yaml

settings_path = Path(sys.prefix).parent / "desktop2stereo" / "settings.yaml"
settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
assert settings["Depth Model"] == "Distill-Any-Depth-Base", settings["Depth Model"]
assert int(settings["Depth Resolution"]) == 518, settings["Depth Resolution"]
assert int(settings["Anti-aliasing"]) == 2, settings["Anti-aliasing"]
assert settings.get("Monitor Identity") is None, "A machine-specific monitor binding was packaged"
assert settings.get("Stereo Output Identity") is None, "A machine-specific output binding was packaged"

required = {
    "torch": "2.11.0",
    "torchvision": "0.26.0",
}
versions = {name: metadata.version(name) for name in required}
for name, prefix in required.items():
    assert versions[name].startswith(prefix), (name, versions[name])
assert sys.version_info[:3] == (3, 12, 10), sys.version
assert Path(torch.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert Path(triton.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())

result = {
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "triton": triton.__version__,
    "triton_distribution": metadata.version("triton-windows"),
    "numpy": numpy.__version__,
    "opencv": cv2.__version__,
    "torch_cuda": torch.version.cuda,
    "torch_hip": torch.version.hip,
    "cuda_available": bool(torch.cuda.is_available()),
    "torch_path_bundled": True,
    "flet": metadata.version("flet"),
    "onnx": onnx.__version__,
    "openxr_module": xr.__file__,
}
expected_triton = (
    "3.8.0.post29"
    if __import__("os").environ["D2S_PORTABLE_FLAVOR"].startswith("rocm")
    else "3.6.0.post26"
)
assert result["triton_distribution"] == expected_triton, result["triton_distribution"]
if __import__("os").environ["D2S_PORTABLE_FLAVOR"].startswith("rocm"):
    import migraphx
    import wc_rocm
    assert torch.version.hip is not None, torch.version
    result["migraphx"] = metadata.version("migraphx")
    result["wc_rocm"] = metadata.version("wc-rocm")
    assert Path(migraphx.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
else:
    import tensorrt
    import wc_cuda
    assert torch.version.cuda == "12.8", torch.version.cuda
    result["tensorrt"] = tensorrt.__version__
    result["wc_cuda"] = metadata.version("wc-cuda")
    assert Path(tensorrt.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())

print(json.dumps(result, sort_keys=True))
"""
    try:
        result = subprocess.run(
            [str(runtime / "python.exe"), "-c", code],
            env=env,
            text=True,
            capture_output=True,
            check=True,
            cwd=app_root,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Bundled dependency import probe failed: {exc.stderr or exc.stdout}") from exc
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Runtime import probe did not return JSON: {result.stdout}") from exc


def run_app_probe(runtime: Path, app_root: Path, profile: str, slug: str) -> dict[str, Any]:
    env = clean_environment(runtime, profile=profile, slug=slug)
    env["PYTHONPATH"] = str(runtime.parent)
    env["D2S_RUNTIME_PATHS_FILE"] = str(app_root / "runtime_paths.env")
    result = subprocess.run(
        [str(runtime / "python.exe"), str(app_root / "main.py"), "--probe"],
        env=env,
        cwd=app_root,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=180,
    )
    return {
        "exit_code": result.returncode,
        "completed": result.returncode == 0,
        "output_lines": len((result.stdout + result.stderr).splitlines()),
    }


def verify_package(
    package_root: Path,
    variant: dict[str, Any],
    slug: str,
    validation_dir: Path,
) -> dict[str, Any]:
    runtime = package_root / "src" / "python3"
    app_root = package_root / "src" / "desktop2stereo"
    if not (runtime / "python.exe").is_file():
        raise FileNotFoundError(runtime / "python.exe")
    for name in ("Include", "libs", "Lib", "DLLs"):
        if not (runtime / name).is_dir():
            raise FileNotFoundError(f"Bundled Python is missing {name}")
    settings_path = app_root / "settings.yaml"
    if not settings_path.is_file():
        raise FileNotFoundError(f"Portable default settings are missing: {settings_path}")
    if not settings_path.read_text(encoding="utf-8").startswith("# Generated portable defaults"):
        raise RuntimeError("A local user settings file was copied into the package")
    if (app_root / "runtime-crash.log").exists():
        raise RuntimeError("A runtime log was copied into the package")
    required_assets = (
        app_root / "gui" / "flet_packages" / "flet-windows.zip",
        app_root / "xr_viewer" / "native" / "windows" / "filament_bridge.dll",
        app_root / "capture" / "native" / "desktop_duplication" / "d2s_desktop_duplication.dll",
        app_root / "streaming" / "amd_encoder" / "d2s_amd_encoder.dll",
        app_root / "streaming" / "nvenc_cudaarray_bridge" / "d2s_nvenc_cudaarray_bridge.dll",
        app_root / "streaming" / "native_rtsp_output" / "opus.dll",
        app_root / "streaming" / "vulkan_ffmpeg_bridge" / "windows" / "d2s_vulkan_ffmpeg_bridge.dll",
        app_root / "streaming" / "rtmp" / "d2s-ffmpeg-9.0.1-d2s.2-windows-amd64.zip",
        app_root / "streaming" / "rtmp" / "mediamtx_v1.20.1_windows_amd64.zip",
        runtime / "Lib" / "site-packages" / "xr" / "library" / "openxr_loader.dll",
    )
    missing_assets = [str(path.relative_to(package_root)) for path in required_assets if not path.is_file()]
    if missing_assets:
        raise FileNotFoundError("Portable package is missing runtime assets: " + ", ".join(missing_assets))
    model = base_model_manifest(app_root)
    pip_env = clean_environment(runtime, profile=variant["profile"], slug=slug)
    pip_check = subprocess.run(
        [str(runtime / "python.exe"), "-m", "pip", "check"],
        env=pip_env,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    imports = validate_imports(runtime, variant["profile"], slug, app_root)
    probe = run_app_probe(runtime, app_root, variant["profile"], slug)
    if not probe["completed"]:
        raise RuntimeError(f"Application capability probe failed for {slug}")
    compile_smoke = validate_torch_compile(
        runtime,
        variant["profile"],
        slug,
        package_root,
        tuple(variant["expected_arches"]),
    )
    return {
        "dependency_check": pip_check or "No broken requirements found.",
        "imports": imports,
        "app_probe": probe,
        "torch_compile_smoke": compile_smoke,
        "model": model,
    }


def build_variant(
    *,
    slug: str,
    variant: dict[str, Any],
    archive: Path,
    stage: Path,
    python_archive: Path,
    seven_zip: Path,
    build_id: str,
    resume: bool,
    log_root: Path,
    verification_root: Path,
) -> dict[str, Any]:
    source_python = stage / "src" / "python3"
    app_root = stage / "src" / "desktop2stereo"
    source_root = stage / "src"
    if stage.exists() and not resume:
        raise FileExistsError(f"Staging directory already exists; use --resume to continue: {stage}")
    if not stage.exists():
        stage.mkdir(parents=True)
        source_root.mkdir()
        run([str(seven_zip), "x", str(python_archive), f"-o{source_python}", "-y"], log_path=log_root / "python-extract.log")
        copy_application(stage)
        vc_files = copy_vc_runtime(source_python)
    else:
        if not (source_python / "python.exe").is_file() or not app_root.is_dir():
            raise RuntimeError(f"The staging directory is incomplete and cannot be resumed: {stage}")
        vc_files = sorted(path.name for path in source_python.glob("*vcruntime*.dll"))

    if not vc_files:
        raise RuntimeError("Bundled Visual C++ runtime DLLs are missing")
    if sys.version_info < (3, 10):
        raise RuntimeError("The host Python is too old for this builder")

    python_check = subprocess.run(
        [
            str(source_python / "python.exe"),
            "-c",
            "import ensurepip, ssl, sqlite3, sys; "
            "assert sys.version_info[:3] == (3, 12, 10), sys.version; "
            "assert sys.maxsize.bit_length() == 63",
        ],
        env=clean_environment(source_python, profile=variant["profile"], slug=slug),
        text=True,
        capture_output=True,
    )
    if python_check.returncode:
        raise RuntimeError(f"The extracted Python runtime failed its validation: {python_check.stderr}")

    constraints = stage / "dependencies" / "portable-constraints.txt"
    constraints.parent.mkdir(parents=True, exist_ok=True)
    make_constraints(variant, constraints)
    env = clean_environment(source_python, profile=variant["profile"], slug=slug)
    requirements_dir = stage / "dependencies" / "official-requirements"
    backend_requirements = requirements_dir / variant["requirements"]
    common_requirements = requirements_dir / "requirements.txt"
    make_official_requirements(
        ROOT / "src" / "env_install" / variant["requirements"],
        backend_requirements,
        rocm_arches=variant["expected_arches"] if variant["profile"] == "rocm" else (),
        package_pins={
            name: value
            for name, value in variant["constraints"].items()
            if name == "triton-windows"
        },
    )
    make_official_requirements(ROOT / "src" / "env_install" / "requirements.txt", common_requirements)
    index_args = official_index_args(variant["profile"])
    pip = [
        str(source_python / "python.exe"),
        "-m",
        "pip",
        "--isolated",
        "install",
        "--no-cache-dir",
        "--prefer-binary",
        "--no-warn-script-location",
        "--disable-pip-version-check",
        *index_args,
        "-c",
        str(constraints),
    ]
    report_dir = stage / "dependencies" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    if variant["profile"] == "cuda":
        run(
            pip + ["setuptools==78.1.0", "wheel==0.45.1", "packaging==26.0", "--report", str(report_dir / "build-tools.json")],
            env=env,
            log_path=log_root / "cuda-build-tools.log",
        )

    backend_report = report_dir / "backend.json"
    run(
        pip + ["-r", str(backend_requirements), "--report", str(backend_report)],
        env=env,
        log_path=log_root / "backend-dependencies.log",
    )

    extras = ["pywin32==312", "PyYAML==6.0.3"]
    cuda_no_isolation = ["--no-build-isolation"] if variant["profile"] == "cuda" else []
    run(
        pip + cuda_no_isolation + extras + ["-r", str(common_requirements), "--report", str(report_dir / "common.json")],
        env=env,
        log_path=log_root / "common-dependencies.log",
    )

    if variant["profile"] == "rocm":
        run(
            [str(source_python / "python.exe"), "-m", "rocm_sdk", "init"],
            env=env,
            log_path=log_root / "rocm-sdk-init.log",
        )
        run(
            [str(source_python / "python.exe"), str(source_root / "env_install" / "apply-wc-rocm-patches.py")],
            env=env,
            log_path=log_root / "wc-rocm-patch.log",
        )
        run(
            [
                str(source_python / "python.exe"),
                str(source_root / "env_install" / "apply-triton-portable-patches.py"),
                str(source_python / "Lib" / "site-packages"),
            ],
            env=env,
            log_path=log_root / "triton-portable-patch.log",
        )
    else:
        run(
            [str(source_python / "python.exe"), str(source_root / "env_install" / "apply-wc-cuda-patches.py")],
            env=env,
            log_path=log_root / "wc-cuda-patch.log",
        )
        run(
            [
                str(source_python / "python.exe"),
                str(source_root / "env_install" / "apply-triton-portable-patches.py"),
                str(source_python / "Lib" / "site-packages"),
                "--unicode-env-only",
            ],
            env=env,
            log_path=log_root / "triton-unicode-patch.log",
        )

    report_paths = [report_dir / "build-tools.json", backend_report, report_dir / "common.json"]
    reports = merge_install_reports(report_paths, report_dir / "combined.json")
    write_dependency_files(stage, source_python, reports)
    write_portable_default_settings(source_python, app_root, variant["profile"], slug)
    versions = installed_versions(source_python, variant["profile"])
    inventory = device_inventory(versions) if variant["profile"] == "rocm" else {}
    for expected in variant["expected_arches"]:
        if expected not in " ".join(inventory.get("device_packages", [])).lower():
            raise RuntimeError(f"ROCm package {slug} does not include the {expected} device package")

    model = base_model_manifest(app_root)
    wheel_hash_file = stage / "dependencies" / "wheel-sources-sha256.tsv"
    source_count = app_file_hashes(app_root, stage / "SOURCE_SHA256SUMS.txt")
    vc_files = sorted(path.name for path in source_python.glob("*.dll") if path.name.lower().startswith(("vcruntime", "msvcp", "concrt", "vccorlib")))
    manifest = {
        "product": "Desktop2Stereo Windows Portable",
        "build_id": build_id,
        "source_commit": current_commit(),
        "variant": slug,
        "accelerator": variant["profile"],
        "python": "3.12.10 x64",
        "default_settings": {
            "depth_model": "Distill-Any-Depth-Base",
            "depth_resolution": 518,
            "anti_aliasing": 2,
        },
        "model": model,
        "python_package_count": len(versions),
        "device_packages": inventory,
        "vc_runtime_dlls": vc_files,
        "app_source_file_count": source_count,
        "python_archive": {
            "name": python_archive.name,
            "sha256": sha256_file(python_archive),
        },
        "wheel_source_sha256_file": wheel_hash_file.relative_to(stage).as_posix(),
        "package_download_policy": {
            "indexes": [OFFICIAL_PACKAGE_INDEXES["pypi"]]
            + ([OFFICIAL_PACKAGE_INDEXES["rocm"]] if variant["profile"] == "rocm" else [
                OFFICIAL_PACKAGE_INDEXES["pytorch_cuda"],
                OFFICIAL_PACKAGE_INDEXES["nvidia"],
            ]),
            "mirror_indexes_disabled": True,
            "direct_migraphx_wheel_source": "https://github.com/lc700x/MIGraphX-Windows/releases/",
        },
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    (stage / "PORTABLE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    english, chinese = readme_text(slug, variant, model, vc_files)
    (stage / "README.txt").write_text(english, encoding="utf-8")
    (stage / "使用说明.txt").write_text(chinese, encoding="utf-8")
    (stage / "THIRD_PARTY_NOTICES.txt").write_text(
        "Desktop2Stereo application license: see LICENSE.\n"
        "Python and third-party package licenses remain in their bundled distribution metadata.\n"
        "Microsoft Visual C++ runtime files are redistributed app-locally from the Visual Studio x64 CRT redist.\n",
        encoding="utf-8",
    )
    (stage / "Launch Desktop2Stereo.bat").write_text(launcher_text(slug, variant["profile"]), encoding="utf-8")

    archive.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            str(seven_zip),
            "a",
            "-t7z",
            str(archive),
            ".",
            "-m0=LZMA2",
            "-mx=1",
            "-mmt=on",
            "-ms=off",
        ],
        cwd=stage,
        log_path=log_root / "archive.log",
    )

    verify_root = verification_root / f"{build_id}-{slug}"
    if verify_root.exists():
        raise FileExistsError(f"Package verification folder already exists: {verify_root}")
    verify_root.mkdir(parents=True)
    run(
        [str(seven_zip), "x", str(archive), f"-o{verify_root}", "-y"],
        log_path=log_root / "package-extract.log",
    )
    validation = verify_package(verify_root, variant, slug, verify_root)
    (verify_root / "validation-report.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "slug": slug,
        "archive": archive,
        "archive_sha256": sha256_file(archive),
        "archive_size_bytes": archive.stat().st_size,
        "staging": stage,
        "verification_dir": verify_root,
        "validation": validation,
        "manifest": manifest,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build relocatable Windows Portable packages for Desktop2Stereo.")
    parser.add_argument("--python-archive", type=Path, default=PYTHON_ARCHIVE_DEFAULT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--staging-root", type=Path)
    parser.add_argument("--verification-root", type=Path)
    parser.add_argument("--7z", dest="seven_zip")
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--resume", action="store_true", help="Continue an existing staging folder created by this builder.")
    return parser.parse_args()


def main() -> int:
    if os.name != "nt":
        raise SystemExit("This builder is only supported on Windows.")
    args = parse_args()
    python_archive = args.python_archive.expanduser().resolve()
    if not python_archive.is_file():
        raise FileNotFoundError(f"Python runtime archive not found: {python_archive}")
    seven_zip = find_7z(args.seven_zip)
    commit = current_commit()
    build_id = f"{dt.date.today().isoformat()}-{commit[:7]}"
    output_root = (args.output_root or ROOT / "dist" / "windows-portable" / build_id).resolve()
    staging_root = (args.staging_root or Path(os.environ.get("TEMP", r"C:\Temp")) / f"d2sps-{build_id}").resolve()
    system_drive = Path(os.environ.get("SystemDrive", r"C:") + "\\")
    verification_root = (
        args.verification_root
        or system_drive / "Temp" / "Desktop2Stereo Portable 验证"
    ).resolve()
    if staging_root.drive.casefold() == verification_root.drive.casefold():
        raise ValueError("Portable relocation validation must extract to a drive different from staging.")
    if " " not in str(verification_root) or not any(
        "\u3400" <= character <= "\u9fff" for character in str(verification_root)
    ):
        raise ValueError("Portable relocation validation path must include spaces and Chinese characters.")
    output_root.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True, exist_ok=True)
    archive_digest = sha256_file(python_archive)
    print(f"Source commit: {commit}", flush=True)
    print(f"Python archive: {python_archive.name} sha256={archive_digest}", flush=True)
    print(f"Staging: {staging_root}", flush=True)
    print(f"Output: {output_root}", flush=True)

    results = []
    for slug in args.variants:
        variant = VARIANTS[slug]
        stage = staging_root / slug
        archive = output_root / f"Desktop2Stereo-{build_id}-{slug}.7z"
        if archive.exists() and not args.resume:
            raise FileExistsError(f"Archive already exists; select another build directory: {archive}")
        result = build_variant(
            slug=slug,
            variant=variant,
            archive=archive,
            stage=stage,
            python_archive=python_archive,
            seven_zip=seven_zip,
            build_id=build_id,
            resume=args.resume,
            log_root=staging_root / "build-logs" / slug,
            verification_root=verification_root,
        )
        results.append(result)

    sums = []
    for result in results:
        sums.append(f'{result["archive_sha256"]}  {result["archive"].name}')
    (output_root / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    source_manifest = {
        "build_id": build_id,
        "source_commit": commit,
        "python_archive_sha256": archive_digest,
        "packages": [result["slug"] for result in results],
        "note": "Per-package file hashes and resolved dependency hashes are inside each portable archive.",
    }
    (output_root / "SOURCE_MANIFEST.json").write_text(
        json.dumps(source_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    readme_lines_en = [
        "Desktop2Stereo Windows Portable packages",
        "",
        f"Build: {build_id}",
        f"Source commit: {commit}",
        f"Bundled Python archive SHA256: {archive_digest}",
        "",
        "Packages:",
    ]
    readme_lines_zh = [
        "Desktop2Stereo Windows Portable 发行包",
        "",
        f"构建编号：{build_id}",
        f"源码提交：{commit}",
        f"内置 Python 压缩包 SHA256：{archive_digest}",
        "",
        "发行包：",
    ]
    for result in results:
        readme_lines_en.append(
            f'- {result["archive"].name} ({result["archive_size_bytes"]:,} bytes; '
            f'SHA256 {result["archive_sha256"]})'
        )
        readme_lines_zh.append(
            f'- {result["archive"].name}（{result["archive_size_bytes"]:,} 字节；'
            f'SHA256 {result["archive_sha256"]}）'
        )
        package_deps = output_root / "dependencies" / result["slug"]
        package_deps.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result["staging"] / "dependencies" / "requirements-resolved.txt", package_deps)
        shutil.copy2(result["staging"] / "dependencies" / "wheel-sources-sha256.tsv", package_deps)
        shutil.copy2(result["staging"] / "PORTABLE_MANIFEST.json", output_root / f"PORTABLE_MANIFEST-{result['slug']}.json")
        shutil.copy2(result["staging"] / "SOURCE_SHA256SUMS.txt", output_root / f"SOURCE_SHA256SUMS-{result['slug']}.txt")
    readme_lines_en.extend(
        [
            "",
            "Each archive contains its bundled Python environment, GPU runtime dependencies,",
            "application assets, offline Distill Base model, package manifest, and instructions.",
            "The target system supplies its GPU driver; OpenXR requires a headset runtime.",
            "See validation-report.json and SHA256SUMS.txt.",
        ]
    )
    readme_lines_zh.extend(
        [
            "",
            "每个压缩包均包含内置 Python、GPU 运行依赖、应用资源、离线 Distill Base 权重、",
            "版本清单和使用说明。目标系统需要提供显卡驱动；OpenXR 需要头显运行时。",
            "详见 validation-report.json 和 SHA256SUMS.txt。",
        ]
    )
    (output_root / "README.txt").write_text("\n".join(readme_lines_en) + "\n", encoding="utf-8")
    (output_root / "使用说明.txt").write_text("\n".join(readme_lines_zh) + "\n", encoding="utf-8")
    summary = {
        "build_id": build_id,
        "source_commit": commit,
        "python_archive_sha256": archive_digest,
        "packages": [
            {
                "variant": result["slug"],
                "file": result["archive"].name,
                "size_bytes": result["archive_size_bytes"],
                "sha256": result["archive_sha256"],
                "validation": result["validation"],
            }
            for result in results
        ],
    }
    (output_root / "validation-report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\nPortable packages created and relocated-package checks passed:", flush=True)
    for result in results:
        print(
            f'  {result["archive"]} ({result["archive_size_bytes"]:,} bytes, sha256 {result["archive_sha256"]})',
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
