# Project Context

## Functional Changes Since 2026-10-17

Compared with commit `a5eaa747`, the product changes through `322a05208` include:

- Added Windows/macOS Moonlight streaming through bundled Sunshine, with PIN pairing/settings and native GPU/video encoders.
- Stabilized AMD AMF streaming, improved macOS streaming throughput/reconnect pacing, and added a Windows cross-adapter CUDA→D3D12 output bridge for OpenXR.
- OpenXR now prefers the active ALVR runtime and smooths real-time depth adjustment; Vulkan Glow uses an asynchronous compute queue.
- Windows Graphics Capture callbacks run outside the GIL to reduce capture-thread contention.
- Replaced the broad `rocm-device-all` package with a dedicated RDNA1–3 variant.

The user-facing log is in `changelog.md`.

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

### Existing RTX 2070 CUDA test environment

Use `src/python3_cuda/python.exe` for RTX 2070 tests. Verified Python 3.12.10,
Torch `2.11.0+cu128`, CUDA 12.8, and `NVIDIA GeForce RTX 2070`. The default
`src/python3/python.exe` is ROCm 10.1 and exposes the AMD RX 9060 XT; do not use
it for RTX inference. Reuse `src/python3_cuda` instead of creating a duplicate
CUDA environment.

A Quest 2 / VDXR OpenXR run on 2026-11-03 using this environment selected
TensorRT CUDA on the RTX 2070, matched the Vulkan adapter LUID, and enabled
CUDA/Vulkan semaphore interop. Warmed-up output reached about 71.7 XR presents/s
with `xr_projection_p95` around 2.3 ms; the XR frame-time p95 was about 15.3 ms
against a 72 Hz target. The earlier ROCm RX 9060 XT → Vulkan RTX 2070 run used
host-eye staging and measured about 1.5–1.6 XR presents/s with roughly 665 ms
projection p95. The comparison changes both adapter and inference backend, so
it supports the RTX configuration but does not isolate the adapter effect or
test `D2S_PRESENT_ASAP`. That patch only affects the Local Viewer loop; OpenXR
uses `xrWaitFrame` pacing and a separate latest-output handoff. The Perfetto
trace `quest2-vdxr-openxr-baseline` targets `VirtualDesktop.Android` and has no
D2S app-frame interval markers, so it is not an input-to-photon measurement.

A windowed RTX 2070 Local Viewer A/B on 2026-11-03 (VSync on, 1280x720,
CUDA/TensorRT) compared the existing loop with `D2S_PRESENT_ASAP=1`. Across
sample-window medians, latency p50/p95 moved from about 29/60 ms to 21/42 ms,
`viewer_queue_ms_p95` from 16 to 5 ms, and `present_call_ms_p95` from 24 to 7 ms.
Present FPS was about 52 vs 49; content FPS rose from about 44 to 46, and reuse
fell from 15% to 7%. The full `test_vulkan_local_viewer.py` module passed (64
checks). This measures the desktop viewer path, not headset photon timing. The
patch belongs only in `run_vulkan_local_viewer` (Windows Local Viewer/3D
Monitor and macOS's default Vulkan viewer); OpenXR and direct stream outputs
have separate pacing paths.

All packages include Python, application dependencies, runtime libraries, and
the offline Distill-Any-Depth-Base snapshot. GPU drivers remain supplied by the
target machine. The build emits dependency/source manifests, SHA256 sums,
English and Chinese usage notes, and relocation validation results alongside
the `.7z` archives.

Latest verified Windows portable release: commit `322a05208696fc8fd36113717e565fa8b40e8ff0`, with ROCm gfx1200/gfx1201, ROCm RDNA1–3, and CUDA 12.8. Compared with 2026-10-17's `rocm-device-all` build, RDNA1–3 is now an explicit 17-architecture set while gfx1200/gfx1201 and CUDA remain separate variants. The user moved the validated archives out of `dist/windows-portable/2026-11-01-322a052-moonlight-complete/`; the SHA256 sums, per-package manifests, and validation report remain there.

All three packages passed relocated extraction, dependency checks, application probes, and Moonlight checks. Each contains the same 130-file Sunshine payload (Moonlight asset SHA256 `0a595c0918ad6ae2c8941ca76642df75b615554d2fef2502901a895bebccd54c`), PyNvVideoCodec 2.2.0, Sunshine `2026.914.233613`, zlib, and the D2S AMD encoder (reported ABI 4; minimum supported ABI is 2). Sunshine's source `config` directory and macOS host binaries are excluded. Torch compilation passed for gfx1200 and CUDA; the RDNA1–3 check skipped on the gfx1200 build host because that architecture is outside the package set.

The user moved the validated `.7z` archives out of the workspace; their new location is not recorded here. Two interrupted CUDA repack attempts also created partial files, but no `.7z` files remain in either workspace output folder. Only accept a relocated archive if its SHA256 matches `SHA256SUMS.txt` and its extraction validation passed.

### Portable build cache cleanup

Use `scripts/tools/cleanup_windows_portable.py` only after a complete release exists. It verifies the three archive hashes and validation records before removing the selected staging and relocation-verification directories. It defaults to a dry run; add `--apply` to remove them. If the user has moved the archives, add `--archives-moved`; the tool then checks recorded hashes and validation reports and leaves release files untouched. It never removes earlier staging directories. Example for the verified release:

```powershell
src\python3\python.exe scripts\tools\cleanup_windows_portable.py `
  --release-root "dist\windows-portable\2026-11-01-322a052-moonlight-complete" `
  --staging-root ".portable-build\2026-11-01-322a052-moonlight" `
  --verification-root "C:\Temp\Desktop2Stereo Portable 验证\2026-11-01-322a052-moonlight" `
  --archives-moved --apply
```

## GUI verification

After every GUI modification, launch Desktop2Stereo and verify the affected
screen and interaction in the running app. Pair this with focused automated
checks; unit tests alone do not count as GUI verification.
