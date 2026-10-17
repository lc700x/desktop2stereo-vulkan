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
packages for ROCm gfx1200/gfx1201, ROCm device-all, and CUDA from
`%USERPROFILE%\Downloads\python3.7z`. Run:

    src\python3\python.exe scripts\tools\build_windows_portable.py

The builder requires 7-Zip and the Visual Studio x64 C runtime redistributable
files available on the build machine. It stages each runtime separately and
writes archives, dependency/source manifests, and relocation validation
results under `dist\windows-portable\<date>-<source-sha>\`. It does not modify
the development Python environments.

Package installation uses the official PyPI, AMD ROCm, PyTorch CUDA, and NVIDIA
indexes as applicable. It ignores machine pip configuration and does not read
`requirements-pip-options.txt`; the repository requirements files are left
untouched. The MIGraphX wheel is fetched from its direct GitHub Releases URL.

Keep root-level `scripts/` empty except for this README and category folders.
