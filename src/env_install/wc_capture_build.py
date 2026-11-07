"""Build the bundled GPU capture extension with WGC session filtering."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def build_native_capture(
    runtime: Path,
    *,
    crate_name: str,
    package_name: str,
    module_name: str,
) -> bool:
    cargo = shutil.which("cargo")
    if cargo is None:
        print(
            f"[{package_name}] Rust is unavailable; using capture exclusion fallback.",
            flush=True,
        )
        return False

    project_root = Path(__file__).resolve().parents[2]
    crate_root = project_root / "native" / crate_name
    target_path = runtime / "Lib" / "site-packages" / package_name / f"{module_name}.pyd"
    target_path.relative_to(runtime)
    with tempfile.TemporaryDirectory(prefix=f"d2s-{crate_name}-") as build_dir:
        environment = os.environ.copy()
        environment["PYO3_PYTHON"] = str(runtime / "python.exe")
        subprocess.run(
            [
                cargo,
                "build",
                "--locked",
                "--release",
                "--manifest-path",
                str(crate_root / "Cargo.toml"),
                "--target-dir",
                str(Path(build_dir) / "target"),
            ],
            cwd=crate_root,
            env=environment,
            check=True,
        )
        built_path = Path(build_dir) / "target" / "release" / f"{module_name}.dll"
        if not built_path.is_file():
            raise FileNotFoundError(f"Cargo did not produce {built_path}")
        staged_path = target_path.with_name(f"{module_name}.pyd.new")
        shutil.copyfile(built_path, staged_path)
        os.replace(staged_path, target_path)

    return True
