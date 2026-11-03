# Scripts Layout

This directory keeps runnable helper scripts grouped by purpose.

| Directory | Purpose |
|---|---|
| `benchmark/` | Performance benchmarks and profiling entry points. |
| `examples/` | Small demos and preview generators. |
| `smoke/` | Fast smoke checks for the public API and host integration examples. |
| `tools/` | Model export, depth generation, comparison, and consistency utilities. |
| `windows/` | Visible Windows launcher scripts for manual testing. |
| `dev/` | Low-level development probes for TensorRT/Triton kernels. |

Scripts under subdirectories should resolve the project root with:

```python
ROOT = Path(__file__).resolve().parents[2]
```

On Windows, `scripts/tools/build_windows_portable.py` builds bundled Python
packages for ROCm gfx1200/gfx1201, the explicit RDNA1–3 device set, and CUDA
from `%USERPROFILE%\Downloads\python3.7z`. Run:

    src\python3\python.exe scripts\tools\build_windows_portable.py

Build only the new RDNA1–3 package without rebuilding CUDA or gfx1200/gfx1201:

    src\python3\python.exe scripts\tools\build_windows_portable.py --variants rocm-rdna1-3

The RDNA1–3 package contains device packages for gfx1152, gfx1151, gfx1150,
gfx1103, gfx1102, gfx1101, gfx1100, gfx1030, gfx1031, gfx1032, gfx1033,
gfx1034, gfx1035, gfx1036, gfx1010, gfx1011, and gfx1012. The builder removes
the device-all variant and writes these architecture extras only to its staged
requirements copy; do not edit the repository requirements files for packaging.

The builder requires 7-Zip and the Visual Studio x64 C runtime redistributable
files available on the build machine. It stages each runtime separately and
writes archives, dependency/source manifests, and relocation validation
results under `dist\windows-portable\<date>-<source-sha>\`. It does not modify
the development Python environments.

Package installation uses the official PyPI, AMD ROCm, PyTorch CUDA, and NVIDIA
indexes as applicable. It ignores machine pip configuration and does not read
`requirements-pip-options.txt`; the repository requirements files are left
untouched. The MIGraphX wheel is fetched from its direct GitHub Releases URL.
After all archives and relocation checks pass, `scripts/tools/cleanup_windows_portable.py`
can remove only the matching build staging and verification caches. It verifies
the release manifests and archive SHA256 values before cleanup and defaults to
a dry run. Add `--archives-moved` when archives have been moved out of the release
directory; it then relies on the recorded SHA256 and validation reports. Add
`--apply` to remove the two cache roots:

```powershell
src\python3\python.exe scripts\tools\cleanup_windows_portable.py `
    --release-root "dist\windows-portable\<release-folder>" `
    --staging-root ".portable-build\<build-id>-moonlight" `
    --verification-root "C:\Temp\Desktop2Stereo Portable 验证\<build-id>-moonlight" --archives-moved --apply
```

The package workflow, current build scope, and verified release record are in
the repository root `CONTEXT.md`.

Keep root-level `scripts/` empty except for this README and category folders.
