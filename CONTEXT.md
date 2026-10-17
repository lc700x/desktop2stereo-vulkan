# Project Context

## Windows Portable Builds

Builds use the Python 3.12.10 x64 archive at
`C:\Users\zjuli\Downloads\python3.7z`. The Windows builder is
[`scripts/tools/build_windows_portable.py`](scripts/tools/build_windows_portable.py);
it requires 7-Zip and the Visual C++ runtime redistributable files on the build
machine. Run it with the bundled Python:

```powershell
src\python3\python.exe scripts\tools\build_windows_portable.py --variants rocm-rdna1-3
```

`--variants` accepts one or more package slugs. Use `rocm-rdna1-3` by itself to
build only the new RDNA1–3 package, leaving the existing CUDA and gfx1200/gfx1201
archives untouched. The builder's full variant set is `rocm-gfx1200-gfx1201`,
`rocm-rdna1-3`, and `cuda`; the former `rocm-device-all` variant is removed.
Build outputs go under `dist/windows-portable/<date>-<source-sha>/`; staging and
relocation checks use separate temporary directories.

The RDNA1–3 ROCm package installs exactly these device extras:

```text
gfx1152 gfx1151 gfx1150 gfx1103 gfx1102 gfx1101 gfx1100
gfx1030 gfx1031 gfx1032 gfx1033 gfx1034 gfx1035 gfx1036
gfx1010 gfx1011 gfx1012
```

The builder starts from `src/env_install/requirements-rocm7.txt`, then writes a
staged copy with each `device-all` extra replaced by the explicit list above.
Never edit the repository `requirements*.txt` files to select portable package
architectures. This keeps the development dependencies unchanged.

The builder uses isolated pip with explicit official sources: PyPI, AMD's ROCm
nightly index, PyTorch's CUDA wheel index, NVIDIA's package index, and the
direct MIGraphX GitHub release URL when applicable. It strips inherited index
options from the staged copies and ignores machine-level pip configuration.
ROCm uses SDK/Torch `10.1.0a20260822` and `triton-windows==3.8.0.post29`; CUDA
uses Torch `2.11.0+cu128`, TensorRT `10.14.1.48.post1`, and
`triton-windows==3.6.0.post26`.

All packages include Python, application dependencies, runtime libraries, and
the offline Distill-Any-Depth-Base snapshot. GPU drivers remain supplied by the
target machine. The build emits dependency/source manifests, SHA256 sums,
English and Chinese usage notes, and relocation validation results alongside
the `.7z` archives.

Latest RDNA1–3 package build: `dist/windows-portable/2026-10-17-5f3ad64/`
contains `Desktop2Stereo-2026-10-17-5f3ad64-rocm-rdna1-3.7z` (SHA256
`a308b423d756b3fb158279f02dfba7a6ae5b7aa56dc7e8b4bff86c66fe593790`). The
build host is an RX 9060 XT (`gfx1200`), outside this package's device set, so
the host-specific `torch.compile` smoke check is recorded as skipped. Package
dependency checks, imports, application probe, and relocated extraction passed;
actual RDNA1–3 GPU compilation still needs matching hardware.
