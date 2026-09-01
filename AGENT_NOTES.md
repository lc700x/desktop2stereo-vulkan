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
- Glow (OpenXR): the `VulkanGlowSourceComputeBackend` source upload was the blocker. The HIP external-memory **buffer** import (`hipExternalMemoryGetMappedBuffer` + `hipMemcpy`) reads/writes **black** on AMD ROCm (isolated root cause: glow source buffer stays black -> black glow image darkens the scene). The HIP external-memory **image** import (`hipExternalMemoryGetMappedMipmappedArray` + `hipMemcpy2DToArrayAsync`) **works** — the local viewer uses it at ~58 FPS. Fix: on ROCm the glow source is uploaded as an RGBA8 Vulkan image (binding 0 = storage image in `d2s_glow_source_image.comp`), then `importer.synchronize()` completes the copy before the compute dispatch reads it. CUDA/NVIDIA keeps the planar sRGB storage-buffer source (binding 0 = storage buffer in `d2s_glow_source.comp`), byte-identical. HIP docs reference: rocm-examples `HIP-Basic/vulkan_interop` and HIP external-interop how-to (`hipExternalMemoryGetMappedMipmappedArray`/`hipMemcpy2DToArrayAsync`).
- Glow validation on AMD RX 9060 XT (headless VulkanContext, ROCm env): `VulkanGlowSourcePass(input_is_image=True)` and `(input_is_image=False)` both build compute pipelines + descriptor layouts OK (`vkCreateComputePipelines`). `RocmVulkanImageImporter.register_slot` on an RGBA8 `VulkanExportableImage` -> image lands in `VK_IMAGE_LAYOUT_GENERAL`; `copy_tensor` (hipMemcpy2DToArrayAsync) of an RGBA8 HIP tensor + `synchronize()` completes with no HIP error; `capabilities.external_memory=True, external_semaphore=True`. **Full backend submit verified**: built a device matching the OpenXR presenter (one shared family, `queueCreateInfoCount=1`, 2 queues, `compute_queue_index=1`) and ran `VulkanGlowSourceComputeBackend.submit` on a synthetic mid-gray sRGB `[1,3,64,64]=0.5` source -> `screen_light_rgb=(0.2159,0.2159,0.2159)` (== linear of sRGB 0.5), `sample_path=vulkan_compute_reduction`. This proves the glow compute reads real pixels via the HIP image import (the root cause of black glow is fixed and GPU-verified). Final visual confirm in the actual OpenXR headset is the only remaining step.
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

## RTMP Streamer on Windows ROCm (direct_sbs.py)
- Root cause of "Nothing was written into output file": `VulkanDirectSbsOutput._select_video_encoder`
  probed `h264_vulkan` on AMD LLPC, failed, and previously raised the raw FFmpeg probe stderr as
  `RuntimeError(report.detail)`. Now falls back to the shared vendor chain
  (`super()._select_video_encoder`) -> on this machine `h264_amf` is selected. NVIDIA/macOS are
  untouched: their Vulkan probe succeeds and never enters the fallback.
- Second bug (surfaced once video worked): `FFmpeg exited with code 4294967291: [in#1] Error opening
  input: I/O error` at startup. Root cause: settings.yaml has no "Stereo Mix" key, so runtime_entry
  resolves `selected_audio = "soundcard:"` (empty device name after the prefix); `_audio_input_args`
  then built dshow `-i audio=` with an empty name -> FFmpeg aborts at input open.
  Fix (Windows-only, in `_audio_input_args`): empty `soundcard:`/`wasapi:` name -> auto-pick a dshow
  loopback device via `_auto_select_windows_audio` (Stereo Mix / virtual-audio-capturer / What U Hear
  only; never a random microphone), else return [] -> video-only stream. Both `_start_ffmpeg` and
  `submit_frame` additionally retry once with audio disabled when FFmpeg dies on the audio input.
  No loopback device exists on this machine -> RTMP now starts video-only (verified: command contains
  only `-i pipe:0`, no empty `audio=`).
- RTMP smoke test on this machine (main.py --runtime --runtime-seconds 30, WEBRTC):
  h264_amf publishes "live" (1 track H264) to MediaMTX for the full window with
  no audio errors. Two more pre-existing encoder-option bugs were fixed after
  the audio fix unmasked them:
  1. h264_amf has no FFmpeg "preset" option -> the shared hardware branch's
     "-preset fast" made AMF die at option-apply time; strip "-preset" in the
     AMF-only cleanup branch (NVENC/QSV/VAAPI/VideoToolbox/libx264 untouched).
  2. Windows force_key_frames used "expr:eq(n%fps\,0)": the % modulo operator
     is rejected by this bundled FFmpeg's force_key_frames evaluator for EVERY
     encoder (libx264 included: "Missing ')' or too many args"), and the "\," 
     escaping is wrong for argv-list spawning. Use "expr:eq(mod(n,fps),0)"
     unescaped (verified exits 0 for h264_amf and libx264). The block is
     Windows-only, so macOS is untouched; NVIDIA native paths (PyNv NVENC and
     the Vulkan native mux) build their own commands without force_key_frames
     and never select AMF, so they stay byte-identical.
  - AMD LLPC still cannot create the native Vulkan FFmpeg encoder ("native
    Vulkan FFmpeg encoder creation failed") and hip_gl_interop=False, so the
    auto chain lands on the stable h264_amf host-upload path
    (gpu_to_cpu=True zero_copy=False) - same design as the glow CPU fallback.
- Browser showed black frame for the AMD WebRTC stream even though MediaMTX
  relayed all RTP (0 lost/0 discarded). Verified two causes in the RTSP SPS
  (4D0433 = Main profile, level 5.1):
  1. Profile mismatch: MediaMTX answers the WHEP offer with Constrained
     Baseline (profile-level-id 42e01f) but AMF emitted Main (4D) NALs; the
     browser negotiated CB then refused Main packets.
  2. Level under-declared: level 5.1 caps 4K at ~30.3 fps (MaxMBPS 983040 /
     32400 MB per frame) and the stream ran 4K@40 -> invalid SPS -> strict
     browser decoders rejected it.
  Fix in the AMF branch of _ffmpeg_command (WEBRTC + H.264 + Windows only):
  force -profile:v constrained_baseline and the level required by
  resolution/fps via _required_h264_level/_format_h264_level (5.2 for
  4K@40, 4.0 for 1080p@30, 4.2 for 1080p@60). Verified SPS is now 424034
  (CB, level 5.2) and FFmpeg decodes the RTSP feed at ~50 fps. HEVC AMF
  keeps defaults; NVENC/QSV/VideoToolbox/libx264 and the NVIDIA/macOS
  selection never enter the "_amf" branch.
- FINAL browser black-frame root cause (AMF): the Constrained Baseline +
  level fix was necessary but not sufficient. Wire-level NAL capture
  (interleaved RTSP reader) showed AMF -usage ultralowlatency emits only ONE
  true IDR (NAL type 5) at stream start; -g and -force_key_frames turn into
  non-IDR I-slices (open-GOP). Browser WebRTC H.264 depacketizer needs a real
  IDR to start -> framesReceived stays 0 with 50k+ RTP packets flowing.
  Verification (180 frames, NAL type-5 count): ultralowlatency=1, lowlatency=1,
  webcam=6, transcoding=6. Fix: WEBRTC+H.264 AMF uses -usage webcam (low
  latency, true IDRs); SRT/RTSP keep ultralowlatency; HEVC/NVIDIA/macOS
  untouched. Verified: real 4K stream decodes in browser (framesDecoded 200,
  readyState 4, videoWidth 3840).
- Diagnostic tools built under .tmp: cdp_probe*.mjs (headless Chrome + CDP:
  video state, getStats codec/frames, real WHEP answer capture), rtsp_analyze*.py
  (interleaved RTP NAL/PT capture). Keep for future streaming debugging.
