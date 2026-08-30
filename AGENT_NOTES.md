# Agent Experience Notes — AMD ROCm GPU zero-copy

Session: get local viewer + OpenXR running on AMD (ROCm) with GPU zero-copy (esp. depth), using `src\python3` as the ROCm env, without breaking the CUDA path.

## Environment facts
- `src\python3` = AMD ROCm env: Python 3.12, torch 2.10.0+rocm7.14.0, migraphx, wc_rocm (WindowsCaptureROCm capture), triton 3.7.1.
- `src\python3_cuda` = CUDA env (torch 2.11+cu128) — used to verify CUDA path unchanged.
- Machine: AMD Radeon RX 9060 XT (primary). NVIDIA RTX 5070 Ti was present but is now disabled. Virtual display adapters exist.
- Monitor on the AMD GPU: D270H (identity `DISPLAY\TUP0270\5&3b49acb3&8&UID4357`, manuf TUP, 1920x1200). Dell U2410(DP) `DELF017` is elsewhere.
- OpenXR runtime registered: Virtual Desktop Streamer (`VirtualDesktopXR (Bundled)`). Headset now connected.
- **MSVC / Windows SDK NOT installed.** This breaks triton: `triton.runtime.build.get_cc()` prefers the ROCm clang-cl, which needs MSVC headers for `stdlib.h` -> every `hip_utils` compile fails. Fix: point `CC` at triton's bundled tcc (`triton/runtime/tcc/tcc.exe`, has its own libc). Verified simple + atomic kernels run.

## Fixes (each committed)
1. **MIGraphX dtype bug**: `DistillPreprocessor.__init__` requires keyword-only `dtype`. MIGraphX provider now passes `dtype=_dtype_from_onnx_name(onnx_path, torch.float16)`; added `ModelOnnxPreprocessor.prepare(rgb, *, height, width)` for the graph-fixed input size.
2. **ROCm backend resolution** (`adapter.py` + `hot_reload.py`): on ROCm the stale `TensorRT: true` setting must NOT select `tensorrt_native` (that import fails and would request a divergent pipeline rebuild). Default is now `migraphx_rocm` on ROCm; explicit `Depth Backend` still respected. CUDA unchanged (verified side-by-side).
3. **Triton tcc** (`triton_runtime.py`): probe sets `CC` to bundled tcc when `find_msvc` finds nothing, so the honest triton probe succeeds and stereo kernels compile without MSVC.
4. **Viewer present path** (`vulkan_local_viewer.py`): `vkMapMemory` already returns a cffi buffer (remove double `ffi.buffer` wrap); convert GPU tensor pixels to bytes; wire `RocmVulkanImageImporter` on ROCm.
5. **ROCm HIP interop** (`rocm_vulkan_interop.py`): discover `amdhip64.dll` via `rocm_sdk` + `_rocm_sdk_*` bin dirs; drop `COLOR_ATTACHMENT` flag on `hipExternalMemoryGetMappedMipmappedArray` (local viewer image has no color-attachment usage -> HIP error 1).

## Current behavior (ROCm)
- Capture: wc_rocm -> HIP tensor `[H,W,4]` uint8 on `cuda:0` — GPU, both Monitor and Window input.
- Depth: MIGraphX zero-copy, `[MIGraphX] Zero-copy GPU path active | input=(1,3,294,518) fp16 -> output=(1,294,518) fp16`, ~12.5 ms/frame (engine `argument_from_pointer` + `run_async` on torch stream, `offload_copy=False`).
- Stereo: triton kernels (via tcc), warmup ~150 ms.
- Present (local viewer, ROCm): the importer's async `hipMemcpy2DToArrayAsync` + `hipSignalExternalSemaphoresAsync` on the **shared torch stream** left the stream in-flight so the next depth frame's `torch.cuda.synchronize()` hung (silent stall after frame 1). Fix: call importer `synchronize()` (hipStreamSynchronize) after copy+signal -> **~58 FPS full GPU** (monitor) / ~50-56 FPS (window). CPU-present fallback ~30 FPS.
- Streaming (MJPEG): `SBS FPS 61-63`, `convert_ms=0 submit_ms=0` (GPU), serving on the stream port.
- OpenXR: headset session active; MIGraphX zero-copy depth; Filament multiview projection composer renders the screen. Three blocking bugs fixed (HIP timeline external-semaphore unsupported):
  1. `prepare_source_for_sampling` indexed empty semaphore lists -> IndexError -> blank screen. Guard: when semaphores absent, transition the imported image to shader-read and return None.
  2. `release_consumer_frame` indexed empty release semaphores -> IndexError -> presenter thread death. Guard: when absent, return the image to GENERAL.
  3. Release now transitions SHADER_READ -> GENERAL so the next frame's sampling sees GENERAL (fixes "first frame then black" — the image was left shader-read after frame 1).
- **Open (validate with headset ON)**: confirm the OpenXR screen renders continuously (not "first frame then black") after the GENERAL-transition fix. OpenXR SBS ~12.5 FPS at 4K-per-eye (4608x4896) — perf tuning possible later.
- Glow (OpenXR): **GPU glow is DISABLED on ROCm (cpu_fallback)** — it renders a **black screen and stops the OpenXR compositor** on AMD LLPC (present ~0, headset idle), even after removing the device-wide-lock contention (per-glow submit lock) and the VkErrorUnknown (dropped semaphore wait + synchronous hipMemcpy). Verified by enabling (dark) then reverting (visible, ~73+ FPS present). v2.5's glow worked because it used a different **HIP-GL** approach (CUDART_GL registering GL PBOs), not the heavier `VulkanGlowSourceComputeBackend`. To get a usable glow on ROCm, port v2.5's HIP-GL glow.
- **GUI capture tool on AMD**: `get_capture_tool_options` keyed the vendor branch on the label containing `"CUDA"`; the build path passes the resolved capture-tool name (`"WindowsCaptureROCm"` — no `"CUDA"` substring), so on AMD it fell back to `WindowsCapture` and the saved `WindowsCaptureROCm` never loaded → CPU capture (low SBS / freeze). Fix: branch on `devices_module.IS_ROCM` first. GUI must run on the **ROCm env** (`src\python3`) for `IS_ROCM` to be True. To get the GPU path: select `Capture Tool → WindowsCaptureROCm` in the GUI and save.

## Display / monitor refresh (Windows)
- App resolves a saved display by `stable_id` (e.g. `DISPLAY\TUP0270\9&1ce1d36&1&UID1796`) + `Monitor Index`; a re-plugged monitor changes the EDID instance, so the saved stable_id goes stale -> `configured input display is unavailable; refresh and select it again` at startup. Fix: update settings.yaml `Monitor Identity` / `Stereo Output Identity` stable_id to the current value from `utils.display_info.enumerate_displays()`.
- `utils.display_info.enumerate_displays()` lists index + name + manufacturer + model + serial + stable_id + size (authoritative for identity matching).

## HIP runtime reference (from old d2s v2.5.0)
- Old xr_viewer used HIP-GL interop: `hipGraphicsGLRegisterBuffer(pbo)` -> `hipGraphicsMapResources(1,&res,stream)` -> **synchronous `hipMemcpy`(D2D)** so the GPU write is complete before the GL consumer samples. Use the same ordering guarantee (sync/ordered copy) for HIP-Vulkan interop; HIP implements **binary** external semaphores but NOT **timeline** external semaphores.

## Notes
- cudnn.benchmark is False (already in DA3 path; also set in migraphx provider init).
- Keep paths relative/portable (no hard-coded absolute paths in app code); v2.5.0 `depth.py` derives ROCm paths from torch's location.
- v2.5.0 `depth.py` `MIGraphXEngine` is the reference zero-copy design; current `providers/amd/migraphx.py` matches it.
- Test artifacts live in `F:\desktop2stereo-vulkan\.tmp` (untracked).
