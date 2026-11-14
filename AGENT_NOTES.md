# Agent Experience Notes — AMD ROCm GPU zero-copy

Session: get local viewer + OpenXR running on AMD (ROCm) with GPU zero-copy (esp. depth), using `src\python3` as the ROCm env, without breaking the CUDA path.

## Environment facts
- `src\python3` = AMD ROCm env: Python 3.12, torch 2.11.0+rocm7.14.0, migraphx, wc_rocm (WindowsCaptureROCm capture), triton 3.7.1.
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
- RTMP/WebRTC audio on Windows: the bundled FFmpeg has NO native wasapi input
  (ffmpeg -devices shows only dshow/gdigrab/lavfi/vfwcap), so `-f wasapi` is
  impossible and the GUI's wasapi: device labels cannot be captured by FFmpeg
  directly. dshow can only see capture-side loopback devices (Stereo Mix /
  What U Hear / virtual cables); on machines with none of those (e.g. only a
  default speaker + VB-Cable), _auto_select_windows_audio returned "" and the
  stream ran video-only.
- Fix (direct_sbs.py, Windows-only): (1) _auto_select_windows_audio also
  matches VB-Audio virtual-cable devices ("CABLE Output/Input (VB-Audio
  Virtual Cable)", vb-audio, vb-cable, virtual cable, what u hear, wave out
  mix) while excluding microphone-named devices; (2) when still no dshow
  loopback exists, grab the default render device directly via the Python
  soundcard WASAPI loopback and feed its 48k stereo PCM to FFmpeg over
  localhost UDP (_start_wasapi_loopback_audio -> -f s16le -i udp://...).
  This is the answer to "can the sound card grab audio directly?": yes via
  WASAPI loopback (verified mean_volume -13.5 dB with a 440Hz tone), just not
  through FFmpeg itself on this build.
- Verified: MediaMTX "[path live] 2 tracks (H264, Opus)", WebRTC sessions read
  2 tracks, browser getStats audio inbound-rtp 1000 pkts/0 lost/jitter 0.024/
  totalSamplesReceived 943200 (48k decoded), video 760 frames decoded.

- REGRESSION FOLLOW-UP ("FPS becomes very low and no sound", fixed):
  Root cause was the dshow VB-Audio virtual-cable input, NOT the AMF usage
  mode. A/B in-app (same -usage webcam, only the audio input differs):
    * dshow "CABLE Output (VB-Audio Virtual Cable)" -> ~10 FPS, submit_ms
      ~90-130ms, and the cable captures digital silence when nothing is
      routed to it (no sound).
    * WASAPI default-speaker loopback (soundcard -> s16le udp://127.0.0.1) ->
      60 FPS steady, submit_ms 2.4ms, 2 tracks (H264, Opus).
  Standalone FFmpeg reproduces part of it: encode rate 167 FPS video-only vs
  72 FPS with the dshow cable input; with real-time pacing + RTSP muxing the
  muxer waits on the stalled/interleaved audio and the app's pipe write
  blocks (submit_ms ~90ms -> ~10 FPS).
  Also busted two earlier false leads: (a) "webcam caps at 27 FPS at 4K" was
  a testsrc2 producer artifact - with rawvideo 4K input every usage encodes
  ~75 FPS; (b) gops_per_idr / forced-idr / bf0 / sc_threshold cannot make
  ultralowlatency emit periodic IDR (still 1 per stream).
  Fix: _auto_select_windows_audio no longer falls back to virtual-cable
  dshow devices; machines without a real Stereo Mix loopback now go straight
  to the WASAPI default-speaker loopback (real system audio + full FPS).
  Real loopback dshow devices (Stereo Mix / virtual-audio-capturer) are still
  preferred when present. NVIDIA/macOS paths untouched. Diagnostics kept:
  D2S_AMF_USAGE (usage override, default webcam for WEBRTC+H.264),
  D2S_AMF_EXTRA (extra AMF args), D2S_FFMPEG_ECHO=1 (echo ffmpeg argv),
  D2S_FFMPEG_STATS=1 (echo ffmpeg frame/fps stats).

- "still no sound" follow-up (2026-09-01): full chain VERIFIED AUDIBLE end-to-end.
  Method: play 440Hz tone to the PC default speaker (soundcard player), run the
  app, connect NON-HEADLESS Chrome via WHEP, capture the video element's audio
  with WebAudio (AnalyserNode RMS). Result vs app: webaudio_rms 0.185-0.34 (loud
  tone), audio inbound-rtp 950 pkts / 897k samples / 0 concealment. So the
  capture->UDP->FFmpeg->Opus->RTSP->MediaMTX->WebRTC chain carries the PC's
  default-speaker audio correctly.
  CRITICAL MEASUREMENT TRAP: headless Chrome reports totalAudioEnergy=0 and
  audioLevel=0 (and often no audio inbound-rtp / muted track) for REAL audio,
  because headless mode has no audio output device. Always verify audio with
  NON-headless Chrome + WebAudio RMS (cdp_probe12.mjs), not headless getStats.
  Also: MediaMTX RTSP readers get 404 in the app context (only WebRTC/WHEP is
  meant to be consumed); the RTSP pull works against a manually-started
  MediaMTX with the same yml - so 404 there is not an audio failure.
  User-facing implication: the stream carries whatever plays on the Windows
  DEFAULT playback device (the one in "WASAPI loopback active: speaker=...").
  If the stream is silent: (a) nothing is playing to that device (check the
  new "WASAPI capture status: packets=N silent=M peak=..." log line every 5s),
  or (b) the viewer is muted/volume 0. GUI Stereo Mix dropdown lets the user
  pick a different loopback speaker to capture.

- "sound works but FPS 10 again" (2026-09-01): root cause = ANY dshow audio
  input throttles the AMF video pipeline, not just the VB-Cable. When the
  Realtek driver flakily exposes a dshow "Stereo Mix" capture device,
  _auto_select_windows_audio picked it -> real sound (Stereo Mix loops the
  Realtek output) but ~10 FPS (submit_ms ~90-145ms), exactly like the
  CABLE-Output case. Fixed by routing ALL Windows GUI audio sources
  ("soundcard:" and "wasapi:" labels, named or bare) through the Python
  soundcard WASAPI loopback sender; dshow is no longer used for prefixed
  labels. _auto_select_windows_audio removed (dead). Verified: WASAPI active,
  capture peaks 0.28-0.41, 60+ FPS steady (submit_ms 2.4ms), 2 tracks
  (H264, Opus), command shows "-f s16le -i udp://127.0.0.1:PORT" not dshow.
  Named speakers all open in loopback (Realtek 2nd output, D270H monitor,
  Realtek Digital, CABLE Input).

- "no sound for real test" resolution (2026-09-01): two independent fixes.
  (1) WASAPI-only audio (dshow throttles AMF video to 10 FPS) - committed
  a5f0cf6. (2) WebRTC ICE candidate poisoning: MediaMTX advertises ALL
  interface IPs as candidates, including the WSL/Hyper-V vEthernet
  (192.168.64.1) which remote LAN clients cannot reach -> sessions drop
  every 3-10s (reconnect loop) and audio never arrives. Verified in MediaMTX
  logs: user client (192.168.1.99) sessions rotated candidates 192.168.64.1/
  192.168.1.70 and died every 3-10s; after restricting candidates to real
  NICs (MTX_WEBRTCIPSFROMINTERFACES=no + MTX_WEBRTCADDITIONALHOSTS=<lan ips>
  from Get-NetAdapter Status=Up Virtual=False) the same client stayed
  connected for the whole run and the user heard sound. The app sets these
  env vars in _server_environment (Windows+WEBRTC only).
  IMPORTANT for testing: the GUI must be fully restarted to load code
  changes - a GUI started before a fix keeps running the old code (user's
  22:18 GUI ran the pre-ICE-fix code).

- "Video OK but no sound" root cause (2026-09-01, found 23:1x): the Windows
  WASAPI soundcard audio input used `-use_wallclock_as_timestamps 1` on the
  s16le/UDP demuxer. On the bundled FFmpeg build this SILENCES THE WHOLE
  AUDIO CHAIN: the Opus track is declared in the SDP but ZERO audio packets
  are ever produced. Symptoms: MediaMTX shows "2 tracks (H264, Opus)", the
  client subscribes to both, but the browser gets video only - inbound-rtp
  audio absent, audio receiver track muted, RTSP audio-only pull hangs with
  no astats report, local mp4 mux of the same chain = 0 bytes.
  Isolated with a 6-variant bisect (file mux): demuxer wallclock kills
  audio regardless of UDP-vs-file or aresample; no wallclock works.
  Fix (committed): drop the demuxer wallclock option on the soundcard input
  and re-anchor audio PTS to the wall clock in the FILTER graph instead:
  `-af asetpts=RTCTIME<+delay_us>,aresample=async=1` (order matters! asetpts
  must come BEFORE aresample; asetpts-after-aresample and asetpts-alone both
  produce empty audio on this build). The v2.5.0 -itsoffset audio delay is
  folded into the RTCTIME offset. Video input keeps use_wallclock_as_timestamps.
  Verified end-to-end: app run + tone -> WHEP probe shows audio inbound-rtp
  growing, receiver muted=false, WebAudio RMS 0.23-0.44; RTSP pull astats
  RMS -12 dB. Control (plain 0-based audio + wallclock video) confirmed the
  muxer then drops the audio (PTS gap), so the asetpts re-anchor is required.
  MediaMTX built-in player page (/live/) defaults video.muted=true - clients
  must unmute or open with ?muted=0; the page/reader.js otherwise handles
  audio correctly (recvonly transceivers for video+audio).

- Sound "jittering" root cause (2026-09-01/02): the WASAPI sender delivered
  each ~85 ms capture block as a ~0 ms burst of 17 datagrams followed by an
  ~85 ms gap (recorder.record returns one blocksize chunk per wakeup; the
  old code sent all datagrams back-to-back). FFmpeg's asetpts=RTCTIME stamps
  at processing time, so the burst compressed ~85 ms of audio into ~0 ms of
  timestamps -> client plays audio in fast bursts with gaps (jitter).
  Fix in wasapi_audio.py: dedicated paced sender thread draining the capture
  buffer at exactly one 240-frame datagram per 5 ms (udp_frames/samplerate),
  decoupled from the blocking capture loop. Also raise the Windows process
  timer resolution (winmm timeBeginPeriod(1)) and use time.perf_counter
  (QPC) for the send deadline: time.monotonic/Event.wait tick at ~15.6 ms on
  Windows and would quantize the 5 ms cadence into 3-packet bursts (measured
  artifact-free cadence now: min 2.6ms, median 5.1ms, p95 6.1ms, max 9ms,
  zero bursts, zero >50ms gaps; 199.8 pkt/s).
  IMPORTANT measurement trap: time.monotonic() on Windows ticks at ~15.6 ms;
  measuring UDP arrival deltas with it produces fake "0ms bursts + 16ms
  gaps". Always use time.perf_counter() for inter-packet timing.
  NVIDIA PyNvSrtVideoOutput (nvidia_encoder.py) audio filter migrated to the
  same chain as the AMD/shared path: asetpts=RTCTIME<+delay_us>,aresample=
  async=1 (was aresample=async=1000:first_pts=0 which has the 0-based-vs-
  wallclock gap -> audio dropped). python3_cuda bundle has soundcard+numpy;
  both bundles share the same ffmpeg.exe. The AMD (RTCTIME) and macOS
  (avfoundation, no soundcard code) filter behavior is unchanged.

- AMD ROCm OpenXR glow (2026-09-02): torch glow is now the ROCm DEFAULT
  (D2S_ROCm_TORCH_GLOW unset/1 = GPU torch glow; 0/false/off = cpu_fallback),
  with the upload upgraded to a zero-copy HIP/Vulkan path:
  VulkanExportableImage gains a tiling kwarg (the ROCm torch glow slots use
  VK_IMAGE_TILING_LINEAR with SAMPLED|TRANSFER_SRC usage - STORAGE requires
  the linear format feature set and is dropped for linear), a row_pitch()
  query (vkGetImageSubresourceLayout), and RocmVulkanImageImporter gains
  image_pointer() (hipImportExternalMemory + hipExternalMemoryGetMappedBuffer
  over the image memory) + copy_tensor_to_image() (synchronous
  hipMemcpy2D with the Vulkan row pitch straight into the shared memory).
  RocmTorchGlowSource prefers the zero-copy write and falls back to the
  mipmapped-array hipMemcpy2DToArray path; one log line reports which
  ("ROCm torch glow upload: zero_copy_hip_buffer" | "..._fallback").
  NVIDIA (vulkan compute backend + cuda array copy) and macOS untouched.
  Diagnosis findings on this machine: glow pipeline works end-to-end on
  ROCm (prewarm, publish, "Glow draw: surround pass executed mode=3"),
  the tool-quad overlays fail against VDXR (RuntimeFailureError in
  enumerate_swapchain_images, CAUGHT - "quad layer render error; skipping
  overlays"; D2S_OPENXR_DISABLE_TOOL_QUADS is the escape hatch), and the
  Quest 3 proximity sensor (mProximityPositive) MUST read worn for VDXR to
  deliver eye frames - software keep-awake (svc power stayon true,
  stay_on_while_plugged_in=7, screen_off_timeout=2147483647) does NOT
  override it; sensor is internal IR (between the lenses). Headless
  --runtime reads settings.yaml directly (no D2S settings-path override);
  capture display must exist (currently only a VITURE/MTT display present,
  Dell U2410 disconnected -> Monitor Index/Identity had to be pointed at the
  VITURE for the diagnosis; .tmp/settings.yaml.bak holds the previous file).

- AMD ROCm OpenXR glow device-drop saga (2026-09-04/05, Quest 2 + VDXR):
  * USER finding (in rocm_torch_glow_source comment): ROCm7 Windows driver
    drops the Vulkan device (~28s) whenever HIP writes Vulkan-imported
    external memory directly (mapped-buffer LINEAR, hipMemcpy2DToArray D2D
    all tested). Glow upload moved to host-staging path
    (hip_compute_host_staging: HIP->host->VulkanHostImage->vkCmdCopy) - the
    direct zero-copy hipMemcpy2D path was additionally REJECTED by a
    readback verifier (readback_mismatch=57600 = whole image -> pitch wrong).
  * A/B isolation on Quest 2: glow OFF (D2S_ROCm_TORCH_GLOW=0 cpu_fallback)
    stable 180s; glow ON -> VDXR xr.end_frame RuntimeFailureError ~30-40s
    into sustained ~70fps presentation (NOT device-lost-initiated; device
    lost is the aftermath of VDXR tearing the session). Glow worker errors
    surface after the runtime failure.
  * Reconnect robustness fixes applied (core_openxr_vulkan.py):
    (1) run_until now tolerates RuntimeFailureError (recreate session) instead
    of letting it kill the presenter thread;
    (2) close() destroys the XrInstance even on device lost - previously the
    instance leaked and every reconnect failed with LimitReachedError
    "loader does not support simultaneous XrInstances".
    Result: app survives the failure + reconnects, but VDXR then refuses
    re-init with "device or resource busy" (streamer holds session resources
    ~6-24s+); full app restart still needed for a clean session.
  * Tool-quad VDXR RuntimeFailureError on enumerate_swapchain_images fixed by
    user (pooled tool-quad swapchains, SAMPLED_BIT, destination_rect).
  * Log noise: Filament "Descriptor bindings updated between render-pass
    begin/end" spam every frame at 70fps (bindings 0-63) - validation-level
    concern, suspect for sustained-load driver faults but unproven.
  * Headset: Quest 2 adb 1WMHHB650V2013; keep-awake (stay_on=7,
    screen_off_timeout=max); Quest OS shows VD client interstitial broadcast
    loop when stream drops ("sendExtendedInterstitialBroadcast
    secondsSinceDetection=N"). Quest 3 proximity sensor never registers worn
    (mProximityPositive=false) - VDXR refuses eye frames; must wear/cover.

## TDR root-cause: undocumented cross-API ordering on ROCm (official HIP doc)
  * Diagnosis grounded in official HIP 7.2.1 "External resource interoperability"
    (rocm.docs.amd.com/projects/HIP/en/docs-7.2.1/how-to/hip_runtime_api/external_interop.html):
    **"access to the buffer should be synchronized between the APIs, e.g. queue
    syncs or semaphores."** The sample signals/walts binary external semaphores
    (hipWaitExternalSemaphoresAsync before the kernel, hipSignalExternalSemaphoresAsync
    after). It also imports the memory with 
    hipExternalMemoryHandleTypeOpaqueWin32Kmt paired to
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT.
  * The AMD/HIP eye path VIOLATED this: runtime_output created TIMELINE external
    semaphores, but hipImportExternalSemaphore is binary-only -> the ROCm importer
    threw "HIP timeline external semaphore is not implemented" -> the whole
    semaphore path fell back to a HIP-only `hipStreamSynchronize` with NO
    Vulkan-side ordering. HIP wrote the imported eye images every frame while the
    Vulkan composer sampled them, unconordinated -> the AMD driver TDR'd
    (~30-40s of sustained presentation; independent of depth-model size because
    the eye upload is still full 4K RGBA at Render Scale 100%).
  * FIX (implemented, not committed): on ROCm the eye path now uses BINARY
    external semaphores (timeline=False) exactly per the doc: HIP waits the
    Vulkan release semaphore -> copy_tensor -> HIP signals the ready semaphore ->
    Vulkan waits ready, samples -> Vulkan signals release. Gated by
    importer.capabilities.producer == "amd-rocm-hip" (_binary_external_semaphores).
    CUDA/NVIDIA path untouched (keeps timeline + wait/signal values).
    Files: runtime_output.py (semaphore type + wait/signal value gating + HIP
    stream drain after the ready signal, mirroring the proven-stable local-viewer
    copy->signal->synchronize pattern so a later depth-frame torch.cuda.synchronize
    does not stall on an in-flight eye copy), rocm_vulkan_interop.py (signal/wait
    accept + ignore timeline-style value=).
  * OPT-IN, AMD-only, DEFAULT OFF: D2S_ROCM_KMT_EXTERNAL_MEMORY=1 switches the
    eye image to VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT +
    hipExternalMemoryHandleTypeOpaqueWin32Kmt (per the HIP sample). Inert unless
    set; NVIDIA path never selects it.
  * Test plan: with D2S_ROCM_TORCH_GLOW=0 AND keep D2S_OPENXR_DISABLE_GLOW_DRAW=1,
    run OpenXR continuous wear 2-3 min. If the TDR is GONE, the missing cross-API
    ordering was the cause. Then re-enable glow (clear DISABLE_GLOW_DRAW, leave
    D2S_ROCm_TORCH_GLOW=1) and confirm glow renders + no crash.
  * Consistency verified headless: rocm_torch_glow_source.py math == shaders/
    d2s_glow_source.comp (8x8 stratified screen-light reduction; glow sample
    y-mirror + max(base_footprint, prefilter_scale=256 glow/surround, 1 veil);
    320x180 RGBA8; temporal mix; 24-value edge lights top8/right4/bottom8/left4).
    Only transport differs (host-staging vs Vulkan compute) - output is
    pixel-equivalent, so visible glow matches NVIDIA path.
  * GLOW transport is already safe: rocm_torch_glow_source.py:320-326 documents
    the AMD "driver drops ~28s on any direct HIP write to Vulkan-imported memory"
    and moved the glow to HIP->host->Vulkan staging->copy_image. The EYES were the
    remaining direct HIP->imported write; the binary-semaphore fix now orders it
    like the proven local viewer (copy->signal->synchronize at ~58 FPS stable).
  * **Live validation (main.py --runtime, OpenXR Link, headset on):** the TDR is
    GONE. Test A (D2S_ROCm_TORCH_GLOW=0, DISABLE_GLOW_DRAW=1): "External binary
    semaphore cross-API sync: active=True", 71-71.2 FPS sustained, rt_depth
    14.9-23.4ms, NO TDR / no 2446ms stall / no device loss. Test B (glow ON,
    DISABLE_GLOW_DRAW cleared, D2S_GLOW_DIAGNOSTIC=1): "Glow draw: surround pass
    executed mode=3 draws=2" 2637 times (glow rendered continuously), last FPS
    sample present_fps=71.8 rt_depth=15.12ms rt_model=7.10ms rt_gpu_depth=17.61ms,
    NO TDR. The run only ended via VDXR "Runtime rejected a frame (client
    interstitial?)" -> session-recreate -> reconnect "device or resource busy",
    which is the HEADSET-STANDBY path ("Headset not detected or in standby"),
    not a glow/crash defect.
  * STARTUP blocker fixed (headless --runtime): settings.yaml had EMPTY
    "Monitor Identity:" and "Stereo Output Identity:" (both nested blocks), and
    the stereo output (Dell U2410 DELF017) is no longer connected. Pointed
    Stereo Output Identity at the current index-2 VITURE (MTT1337, 3840x2160)
    virtual display. Verify with display_info.enumerate_displays().
  * Remaining for completion: user must keep the headset screen awake (Quest
    auto-sleep ~45-60s triggers the VDXR interstitial) and visually confirm the
    glow matches the NVIDIA path (math is identical, but eyeball confirm pending).
  * Model can no longer read images (read_image fails) - verify visuals via logs.

- ROCm OpenXR eye handoff follow-up (2026-09-05, after user "glow not visible"
  report): RocmVulkanOutputAdapter previously inherited the CUDA adapter's
  timeline-semaphore setup, so every AMD session logged "HIP timeline external
  semaphore is not implemented", fell back to synchronized copies, and (in the
  report process) inherited a stale D2S_OPENXR_DISABLE_GLOW_DRAW=1 that made
  glow upload work but never draw. The adapter now creates binary HIP/Vulkan
  semaphores per eye slot, drains HIP before Vulkan waits (same proven pattern
  as the glow worker), consumes ready signals when a frame is dropped before
  sampling, and clears the stale legacy draw-disable while keeping an explicit
  AMD-only D2S_ROCM_DISABLE_GLOW_DRAW=1 switch. Prewarm workers are closed on
  presenter shutdown even when OpenXR never initializes.
  Validation on RX 9060 XT: nine-frame real-GPU eye handoff probe with binary
  semaphores passed; ROCm glow smoke (25s/530 uploads) and 286 focused tests
  passed. OpenXR headset validation still requires a connected/worn headset;
  last attempt reported "HMD form factor unavailable".

- ROCm glow device-loss root confirmed with Virtual Desktop's OpenXR.log
  (2026-09-05 16:46:59): xrEndFrame failed with `VkStatus failure [-4]`
  (VK_ERROR_DEVICE_LOST) at vkQueueSubmit inside the VDXR runtime ~12s after
  the session started with direct HIP->Vulkan glow image copies. Direct HIP
  writes into Vulkan-imported image memory are therefore NOT safe for the glow
  even with binary semaphores and HIP drain on this driver. RocmTorchGlowSource
  now defaults to the small host-staging transport (HIP compute -> CPU ->
  VulkanHostImage -> vkCmdCopy) again, while keeping the locking/lease/shutdown
  fixes from the earlier work. The eye adapter keeps the binary-semaphore
  HIP/Vulkan path (previously stable for 180s with glow off). Smoke on RX 9060
  XT: 424 host-staging glow publications / 46 lease checks / exact readbacks
  over 20s. NVIDIA path untouched.

- ROCm eye memory now uses the official HIP interop KMT handle pair by default
  (Vulkan `OPAQUE_WIN32_KMT_BIT` + `hipExternalMemoryHandleTypeOpaqueWin32Kmt`)
  when the ROCm output adapter creates eye images. D2S_ROCM_KMT_EXTERNAL_MEMORY=0
  reverts to the previous opaque Win32 handle. KMT export handles are not passed
  to CloseHandle. Headless RX 9060 XT probe: 36-frame binary-semaphore eye
  handoff with exact Vulkan readback passed; log shows
  `memory=kmt_win32`. OpenXR headset validation with this handle pair is pending.

- ROCm eye upload switched from HIP->imported-image memory to HIP->exported
  staging buffers + vkCmdCopyBufferToImage (Vulkan owns all image writes). KMT
  handle pair is used for both the image and the transfer buffers; direct HIP
  image writes still lost the Vulkan device after ~35s even with KMT +
  binary semaphores in the 17:00 run. New path logs
  `hip_staging_buffer_to_vulkan_image` and passed 36 real-GPU frames with exact
  Vulkan readback on RX 9060 XT. Headset validation pending.

- Buffer staging also lost the device (17:14 run survived ~3 minutes at 70 FPS
  before VkStatus -4). No HIP/Vulkan shared-memory variant is durable on this
  AMD/VDXR combination. Host staging (17:23) crashed after only ~20 seconds and
  slowed conversion to 26ms/frame, so the default is back to the HIP staging
  buffer path that sustained ~3 minutes. Pure Vulkan host staging is retained
  behind `D2S_ROCM_EYE_HOST_STAGING=1`.

- Direct A/B on VDXR, 1.0 supersampling, controller=none + tool quads disabled:
  no glow -> stable for the full 240s. Re-enabling ROCm torch glow (even with
  draw disabled) crashes within tens of seconds with VkStatus -4 / native
  exception. CPU fallback is stable but does not produce a Projection Composer
  glow image. Conclusion: the ROCm torch glow source/draw path itself is not
  stable on this AMD 26.8.1 + VDXR combination; transport changes do not fix it.
  Recommended temporary stable run:
  D2S_ROCm_TORCH_GLOW=0, D2S_ROCM_DISABLE_GLOW_DRAW=1,
  D2S_CONTROLLER_MODEL=none, D2S_OPENXR_DISABLE_TOOL_QUADS=1.

- ROCm glow now defaults to VulkanGlowSourceComputeBackend when the Vulkan
  context exposes compute queue 1 (D2S_ROCM_GLOW_VULKAN_COMPUTE=auto). HIP only
  feeds an exported buffer; Vulkan compute generates the glow image. Direct
  VDXR run at 1.0 supersampling: visible mode-1 glow drew continuously and no
  device loss occurred for ~2 minutes until the headset session ended. This is
  the first glow-on ROCm path without VkStatus -4. Full 4-minute validation
  with the headset kept awake is still pending.


- AMD media GPU-required fallback (2026-11-12): On RX 9060 XT / ROCm 7.16, NVDEC is unavailable, so `F:\Videos\test.mp4` selects FFmpeg AMF. The current FFmpeg `bgr24` rawvideo pipe returns a host `numpy.ndarray` (not a GPU-resident/zero-copy frame), which previously tripped the runtime's pre-preprocess GPU-residency guard. For Video/Image sources only, allow host input through capture preprocessing so it can upload/convert on the selected GPU; keep the post-preprocess GPU-device guard intact, and keep strict GPU capture enforcement for desktop Monitor/Window sources. Validation: media frame preprocessed to `cuda:0` on AMD ROCm; focused capture/pipeline/runtime tests: 327 passed, 1 skipped. Do not describe the AMF rawvideo pipe as zero-copy.


- AMD media startup bug report follow-up (2026-11-12): the reported fatal log was the pre-fix path that rejected host video frames before GPU preprocessing. Current `PollingCaptureRunner` preserves the source's `Video` mode on `CapturedFrame`; the GPU-required input residency guard exempts only `Video`/`Image`, and the post-preprocess GPU-device guard remains strict. Regression test now covers runner metadata flowing into that guard. Re-tested with `F:\Videos\test.mp4` on RX 9060 XT: decoded input was CPU (`frame_raw_device=cpu`), media guard accepted it, preprocessing produced `(1,3,1080,1920)` on `cuda:0`. Focused capture/media/preprocess/pipeline/runtime-entry suite: 327 passed, 1 skipped. This only validates media->GPU preprocessing, not a live OpenXR headset session. Also fixed playback pacing: FFmpeg AMF/software decode must keep `-re` even after NVDEC is disabled; `PollingCaptureRunner` has no playback pacing. Regression test checks the AMD fallback command retains `-re`.


- Media 4K HDR GPU preprocessing optimization (2026-11-12): the existing Triton media color/HDR conversion was gated to same-size output, so resized NVDEC RGBP and AMF NV12 frames fell through to slower Torch conversion. `capture/preprocess.py` now runs Triton conversion at source resolution on CUDA/ROCm and then GPU antialiased resize, preserving the prior conversion-before-resize order; NV12 and RGBP/BGR24 paths keep their Torch fallback. Added resized GPU-vs-CPU reference tests for PQ/HLG/SDR and NV12. On `F:\Videos\【顶级画质】B站首支4K 120FPS HDR10影片《文字大战》\P1-【4K 120FPS HDR10】文字大战.mp4` (3840x1920 HEVC HDR, target 1920x1080), RTX 2070 NVDEC RGBP repeated preprocess measured old Torch 17.95 ms/frame vs Triton+resize 1.20 ms/frame (~15x); max pixel difference 5.6e-6. RX 9060 XT AMF NV12 host-frame path measured 2.75 vs 2.44 ms/frame (~11%); max pixel difference 1.8e-6 and includes host-to-GPU upload. Triton block-size A/B kept block=256: AMD measured 2.66 ms at 256 vs 2.81/2.87/2.96 ms at 128/512/1024; CUDA differences were negligible. Decoder-only playback at requested 60 FPS averaged 59.8 FPS NVDEC and 61.7 FPS AMF after excluding startup/warm-up; per-read p95 remained ~28-29 ms, so treat cadence as bursty and do not infer smooth presentation. CUDA/ROCm focused suites each: 348 passed, 1 skipped (only source/decode/preprocess/runtime tests). No end-to-end 3D viewer/OpenXR presentation FPS was measured; use presenter `content_fps`/`present_fps` and capture-to-present latency for any 4K60 claim.

- GPU 3D playback optimization (2026-11-13): `stereo_runtime._depth_gaussian_triton` now has a model-weighted fused edge-aware Gaussian kernel for CUDA/ROCm. `anti_alias_depth(..., edge_threshold=...)` attempts this path and falls back to the existing Torch `unfold` implementation if Triton is unavailable or fails. On both RTX 2070 CUDA 12.8 and RX 9060 XT ROCm 7.16, 196x336 7x7 model-depth AA measured about 0.03 ms fused vs 1.11 ms Torch, with max absolute reference error below 3e-7. Added GPU parity coverage in `tests/test_display_antialias.py`.

- ROCm attention default (2026-11-13): `app_runtime.runtime_context` enables `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` for AMD devices before the depth runtime is constructed, unless the user explicitly sets it or disables `D2S_ROCM_AOTRITON`. On the local RX 9060 XT test, Distill-Any-Depth-Base PyTorch inference improved from about 32 ms to 21 ms and image-to-3D benchmark throughput from about 20.8 to 25.2 FPS. This is an app-local process setting; it does not change shell configuration.

- Full local video-to-3D benchmark (2026-11-13, `F:\Videos\2025-11-11 22-05-29.mkv`, 3840x2160, quality_4k, half-SBS): CUDA NVDEC + TensorRT CUDA Graph measured decode 0.38 ms/read, preprocessing 0.49 ms, depth 11.88 ms, synthesis 19.72 ms, total 32.46 ms (~30.8 FPS). ROCm FFmpeg AMF/NV12 + AOTriton PyTorch measured decode 10.14 ms/read, preprocessing 2.16 ms, depth 20.38 ms, synthesis 20.53 ms, total 53.21 ms (~18.8 FPS). Decoder-only reads remain roughly 60-77 FPS, so 4K60 end-to-end is not established; preserve these stage measurements when evaluating further changes.


- Updated local 4K video GPU throughput check (2026-11-13, `F:\Videos\2025-11-11 22-05-29.mkv`, 3840x2160 H.264 60fps; current Base/518, `quality_4k`, 2-layer config; `StereoRuntime.process_openxr_frame`, 10 warmups + 120 repeated-frame samples, no headset presentation/stream encode): CUDA selected native TensorRT, p50 12.98 ms, p95 13.85 ms, p99 13.87 ms (~76.7 mean FPS; p95-equivalent 72.2 FPS). ROCm selected MIGraphX zero-copy graph, p50 11.38 ms, p95 13.30 ms, p99 13.85 ms (~86.0 mean FPS; p95-equivalent 75.2 FPS). The runtime output path reports sub-ms synthesis after warm-up. These are 4K RGB->OpenXR stereo-compute measurements, not full capture-to-present FPS; do not claim end-to-end 4K60 without live presenter evidence.
- Separate paced 60-frame media-source check on the same file: RTX 2070 NVDEC delivered 60 frames at 60.96 measured FPS (mean `grab()` 16.40 ms, p95 27.95 ms); RX 9060 XT rejected NVDEC and correctly selected FFmpeg AMF/NV12, delivered 60 at 69.01 measured FPS (mean 14.49 ms, p95 13.32 ms). AMF raw pipe is CPU-backed; GPU preprocessing follows. Keep media decode and 3D-compute figures separate; current steady-state compute has headroom over 60 FPS, while production capture/presentation and audio still need a live end-to-end run to establish actual smooth 4K60 playback.


- ROCm 4K video 3D compute resolution check (2026-11-13, RX 9060 XT, `F:\Videos\2025-11-11 22-05-29.mkv`, 3840x2160, Base model, `quality_4k` OpenXR compute path): at 392 model-input height the run reported 8.866 ms mean, 10.262 ms p95, and 112.79 mean compute FPS; last GPU-event timing reported 1.226 ms total (0.969 ms depth + 0.213 ms synthesis). This is repeated-frame compute only, not decode-to-present or headset FPS. 392 is a speed/quality tradeoff; do not silently change the user's 518 setting. CUDA/ROCm preprocessing + depth antialias focused tests each pass: 50 passed, 1 skipped.
 Both local Python environments also pass `tests/test_pipeline.py tests/test_runtime_entry.py`: 271 passed each on ROCm (`src/python3`) and CUDA (`src/python3_cuda`); only existing `mss.mss` deprecation warning. `tests/test_capture_media.py tests/test_capture_factory.py` also pass 51/51 in both environments; warnings are the existing `mss.mss` deprecation.

- OpenXR Simulator integration run (2026-11-13, OpenXR-Simulator 1.7.1): launched the simulator by setting `XR_RUNTIME_JSON` only in the test process to `C:\Users\zjuli\AppData\Local\OpenXR-Simulator-v1.7.1\openxr_simulator.json`; the Windows runtime registration was not changed. The runtime identified a PICO 4 profile at 1280x1400 per eye. On RX 9060 XT / ROCm, the 4K HDR HEVC120 media run selected FFmpeg `hevc_amf` and emitted NV12 through a rawvideo pipe, so treat frames as host-backed, not zero-copy. MIGraphX depth and Triton stereo synthesis ran; while the file was active, observed SBS processing was about 45-51 FPS (adaptive target about 56-59 FPS). XR presentation was about 89 FPS (p50 ~12 ms, p95 ~18 ms), which is not the media-processing rate. At source EOF SBS fell to about 19-20 FPS; do not include that tail as steady playback. This simulator run did not establish end-to-end 4K60. Next benchmark should capture `FPSBreakdown` cap/raw/runtime/drop counters and per-stage timings; 45-51 FPS alone does not locate the bottleneck. Candidate areas to measure are AMF's host-frame transfer and synchronous `grab`; a bounded latest-frame prefetch must invalidate queued frames on seek/reopen and coordinate EOF/stop/pause.


- Media quality preservation boundary (2026-11-13): Preserve source video dimensions by default. The capture runtime passes the same UI `RenderSizeConfig` through `create_runtime_context` -> `CaptureConfig` -> media capture; AMF `vpp_amf` resizing is enabled only when that config resolves to a smaller size. `Render Scale: 100%` emits no VPP resize; explicit 85/75/50% settings produce only their corresponding UI-requested sizes. Do not add hidden resolution reduction or lower model-input quality to chase FPS. Regression coverage verifies all four UI scale tiers; media/factory/runtime-entry tests passed 134/134 in both `src/python3` (ROCm) and `src/python3_cuda` (CUDA), with only the existing `mss.mss` deprecation warning. OpenXR Simulator presentation FPS is not the optimization target; report decode, preprocessing, depth, synthesis, and presentation rates separately.


- Media player source/UI refactor (2026-11-14): User-facing capture sources are now `Monitor / Window / File`; saved `Video / Image` configurations migrate to `File`, and media subtype is derived from the path. For `File`, the GUI hides Capture Tool and shows a read-only path plus grouped Open File/Open Disc buttons; media selection happens in the output browser, not a system file picker. The GUI transport was reorganized to a separate seek row and a VLC-like transport row with `-10s`, play/pause and `+10s`. The player overlays are shared through `media_ui.py`; Local Viewer draws the bar/browser into both eye rects, OpenXR uses tool quads, and both consume the same revisioned media state/control files. Folder-derived playlists support current index, selection, previous/next/clear and automatic advance on EOF. Same-monitor Monitor 3D keeps its click-through/cursor path; different input/output Monitor/Window modes map absolute SBS pointer positions back to the captured monitor/window while File mode only consumes player controls.

- HDR P010 stride/color fix (2026-11-14): On RTX 2070, PyNvVideoCodec `OutputColorType.NATIVE` returns a P010 DLPack view shaped `(H*3/2, W)` with a byte-like column stride `(W, 2)`. Passing that view directly to the Triton YUV kernel produced repeated/tiled and magenta/green frames. `_decode_nvdec_frames` now reinterprets the underlying row-major storage as `(H*3/2, W)` with stride `(W, 1)` before cloning, still on GPU. The 4K HDR sample at 60s now produces a coherent red/gold stage frame instead of tiled colors, and the CUDA OpenXR Simulator run showed the corrected image. AMF HDR continues to skip `vpp_amf` because that chain produced uniform black P010 on the tested stack. Preserve the UI-only scaling rule; browser-only neutral backdrops are capped at 720p and do not affect media quality.


- HDR/SDR colour correction (2026-11-14): The P010 stride fix was necessary but not sufficient. PQ normalisation used Hable with a 10000-nit white point (`peak=100` in 100-nit units), making skin and midtones dark and oversaturated. `capture/preprocess.py` and `capture/preprocess_triton.py` now use Hable's standard 11.2 white point (1120 cd/m2). BT.2020/P3 -> BT.709 conversion now compresses out-of-gamut colours toward Rec.709 luminance instead of clipping each channel, preventing hue shifts in saturated highlights. HDR files with missing/`unknown` primaries or matrix tags now default to BT.2020/BT.2020nc. CUDA NVDEC and ROCm AMF both still emit the fused `triton_media_p010_hdr_fused` path; CUDA remains GPU-resident and AMF remains host-decode followed by GPU preprocess. Use the RTX 2070 + AMD RX 9060 XT 4K HDR sample for regression and compare against PotPlayer on the same frame.

- Media overlay control closure (2026-11-14): `browse_close` now clears `_browser_mode`, so the last paused frame returns instead of the browser neutral frame. Added shared `browse_select`; Local Viewer single-click selects and double-click opens, OpenXR first trigger selects and second trigger opens. Media-bar hit testing now supports absolute progress seek, volume, and disabled controls. Local Viewer drags seek/volume at 20 Hz and sends a final value on release while freezing polled state during the drag. Media bars remain as a 6 px low-alpha handle after the 3 s auto-hide; hovering the handle restores the full bar. Pointers inside an expanded bar/browser panel are consumed rather than forwarded to the captured screen/window. OpenXR keeps the ghost handle ray target, throttles joystick scrolling, and chooses one visual hover hand while still allowing both hands to press. Bar and browser texture keys are split so browser RGBA is no longer re-uploaded for every position tick; bar redraw is whole-second only. GUI Open File/Open Disc remain usable for images, while play/seek/volume are disabled when there is no video/audio; `ready=False` is published on stop.

- Video EOF presentation (2026-11-14): `MediaCaptureSource.grab()` previously returned `None` on the EOF transition while the last decoded frame stayed in `_frame`. The presenter could latch that black transition and never receive the retained frame. EOF now returns `_frame` immediately after stopping audio/decoders and marks `playing=False`, `_eof=True`; automatic playlist advance still happens first when a next item exists. Added a regression test.

- Verification (2026-11-14): Full suites pass in both local environments after the colour, overlay, EOF, runtime-settings and depth-normalization work: CUDA `2806 passed, 34 skipped`; ROCm `2770 passed, 70 skipped`. Focused media/OpenXR/UI tests pass on both backends. A live GUI Local Viewer run confirmed the HDR sample renders with natural skin/red-gold/blue-purple colours in SBS; the test runtime was stopped after capture. `main.py --runtime` now falls back to saved `Media Path`, playback position, volume/mute and browser directory when the GUI environment variables are absent; a live probe loaded the 4K HDR file with `zero_copy=true` and `zero_copy_ready=true`.

- ZipDepth EdgePad assets (2026-11-14): Confirmed the public HuggingFace repo `lc700x/ZipDepth-EdgePad` contains `LICENSE`, `README.md`, `zipdepth_edgepad_384.onnx`, `zipdepth_wide_512x288.onnx`, and `zipdepth_wide_672x384.onnx` (about 70 MB total). Local verified staging path is `C:\Users\zjuli\.rea\temp\zipdepth-edgepad-2026-10-08`. The widescreen `672x384` model is the preferred D2S default, `512x288` is the lower-latency option, and square `384x384` is compatibility-only. These are complete depth graphs (single `image` -> `depth` output), not the legacy ZipDepth encoder/decoder split; do not route them through the six-feature encoder branch. License is MIT; preserve Fabio Tosi/Nightfall notices. No repository model weights were added.

- Subagent research queue (2026-11-14): InfiniDepth official-contract review and ZipDepth EdgePad provider integration review were delegated read-only. Current evidence points to the EdgePad models being directly usable as full ONNX graphs on CUDA/TensorRT and ROCm/MIGraphX after a dedicated full-graph provider path; the existing ZipDepth provider's encoder-only contract is not sufficient. Do not claim a model upload failure: the HF API showed all three ONNX files and metadata present.

- InfiniDepth stability audit (2026-11-14): The clearest confirmed bug was `_normalize_depth` accepting `+Inf` into the 98th-percentile bound: a single `+Inf` could collapse an otherwise valid frame to near-black. It now replaces NaN/±Inf before percentile selection, and `tests/test_fp16_nan_guards.py` covers the range-collapse case. Remaining high-priority risks from the official-contract review: InfiniDepth PyTorch/MIGraphX defaults use FP16 while the official reference is FP32; fixed-raster plus generic upsampling does not use the official arbitrary-resolution query path; the provider returns the raw head output without an explicit `pred_depth` vs `pred_disparity` contract; and per-frame percentile normalization has no temporal stabilization. Do not invert depth or add global EMA until near/far direction and scene-cut reset behavior are locked by tests and provider parity evidence.

- OpenXR Glow-off background/visibility fix (2026-11-14): controller precomposition is now allowed only with a transparent (`clear_color[3] == 0`) projection clear. The previous opaque black clear (`0,0,0,1`) plus the screen-underlay destination-alpha blend hid the virtual screen; using an opaque clear with precomposition also left the finite screen quad transparent outside its silhouette, so the runtime backdrop was visible. Default Glow-off now uses the normal screen pass with an opaque black full-viewport clear, then the transparent Filament controller/laser overlay. Unit coverage in `tests/test_openxr_vulkan.py` asserts opaque clears do not enter precomposition. OpenXR Simulator capture with Pico 4 auto render scale (`1843x2016` per eye) showed a visible stereo screen and sampled frame corners at RGB `(0,0,0)`; the live preview used source `3840x1920`, screen footprint `960x556`, and the direct `downsample_lanczos_rcas` path.
- Verification after OpenXR background fix (2026-11-14): focused OpenXR/screen suites pass on both environments (`425 passed, 1 skipped` each); shader manifest and `spirv-val` pass. ROCm full suite: `2773 passed, 70 skipped`. CUDA full suite: `2806 passed, 36 skipped` with one order-dependent `test_quality_scale_gpu_preprocess_matches_final_pipeline_size[100%-capture_size1]` CUDA context failure; the same test passes in isolation and `tests/test_pipeline.py` passes as a file (`206 passed`). The simulator capture after the fix showed visible stereo screen content with all sampled frame corners exactly RGB `(0,0,0)`.
- Media Player GUI/output ownership (2026-11-14): The capture-mode display label `Input File` is now `Media Player` / `媒体播放器` while the persisted value remains `File`. In File mode the main GUI keeps only the source-mode selector and hides every media control/button, including path, open file/disc, transport, seek, volume, and status. Local Viewer and OpenXR own the complete media workflow through the shared media bar/browser: file and disc opening, mounts, directory navigation, playlist controls, and explicit Open (single-click select; double-click or Open activates). Starting File mode with no saved path now opens the output browser as the intended initial state without surfacing a false media error; stale saved paths still report the error. Focused CUDA/ROCm tests pass (`76 passed` on each environment).
- Media Player UI verification (2026-11-14): CUDA and ROCm both pass the integrated media/GUI/Local Viewer/OpenXR suite (`643 passed, 1 skipped` each), including hidden main-GUI media widgets, Media Player localization, empty-path browser startup, Local Viewer browse/open dispatch, and OpenXR browse/open dispatch. Browser preview confirmed the explicit localized Open button at the bottom-right while retaining mounts, path navigation, playlist controls, and double-click activation.
- Local Viewer media overlay invisibility root cause (2026-11-14): media bar/browser/pointer `VulkanFpsOverlay` instances were created before the swapchain but were omitted from `_create_swapchain().available_overlays`, so `on_swapchain_recreated()` never built their framebuffers. `record_rects()` then always returned false (`not self._framebuffers`). The media overlays are now included in swapchain image-usage registration and receive both `on_swapchain_recreated` and `on_swapchain_destroyed`; focused CUDA/ROCm integration suites pass (`733 passed, 1 skipped` each).
- Recent media history and capture-tool UX (2026-11-14): `MediaBrowser` now has a stable JSON history (`media-recent.json`) shared across Local Viewer/OpenXR sessions. Successful media opens prepend the resolved path; the browser shows up to five recent items with a delete `x`, and both viewer and OpenXR dispatch recent-open/recent-remove through the revisioned command channel. History is included in browser texture keys so deletion redraws immediately. The main GUI capture-mode selector still displays the Capture Tool label/dropdown in Media Player mode, but the dropdown is disabled; all media controls remain hidden. The browser also exposes an explicit localized Open button.
- Media playback speed/UI closure (2026-11-15): `MediaCaptureSource` now decodes video on a bounded `MediaFramePump` (default 8 frames, `D2S_MEDIA_FRAME_BUFFER`) instead of letting a slow inference consumer back-pressure the demuxer. The presentation clock remains wall-clock based and stale queued frames are discarded, so a 4K120 source still advances one source second per wall second while the 3D pipeline consumes about 30 FPS. A direct FFmpeg-AMF probe with a deliberately slow consumer measured `playback_fps=60`, consumer `28.55 FPS`, `speed_ratio=1.0` over 10 seconds. Pausing stops the pump but retains the queued frames and last displayed frame for warm resume/seeking. The Local Viewer media bar now follows the reference player hierarchy: times above a slim full-width timeline, a larger accent-ring Play/Pause target, large 64 px secondary controls, text-fit Open File/Open Disc buttons, previous/next playlist controls, and equal outer padding / equal flexible gutters around the three control groups. Volume dragging maps to the inset slider track rather than the whole bar. The shared media inventory now mirrors Oku3D at exactly 59 video and 49 image extensions, with Pillow/OpenCV/FFmpeg still-image fallback. Runtime pipeline media-depth reuse now requires an exact source identity match `(media_generation, source_frame_index, source_pts)` plus matching source-target/render-size/depth-settings keys and a 250 ms maximum age (`D2S_MEDIA_DEPTH_REUSE_MAX_AGE_MS`); changed content, missing identity, seek/reopen, source switch, render-size change, or depth-setting change fails closed. If inference is late for changed content, the pipeline skips that output update rather than substituting stale depth; cursor-only reuse remains a separate gated path.
- Local Viewer player-bar polish (2026-11-15): the media bar now follows the supplied Bilibili-style reference hierarchy while retaining Desktop2Stereo-specific Open File/Open Disc controls: timestamps above a slim full-width scrubber; a prominent 80 px accent-ring Play/Pause control; 64 px previous/next/±10s/mute targets; equal outer margins and equal flexible gutters between the left/utility/transport groups; and an inset volume slider whose click/drag coordinates map to the visible track. Pointer polling supplements GLFW cursor events so controls can still reveal when the viewer uses click-through cursor passthrough.
- Stable FPS plan progress (2026-11-15): Phase 0 baseline is documented in `docs/stable-fps-baseline.md`. Phase 1 exact-depth identity is implemented: media frames carry `(media_generation, source_frame_index, source_pts)` through `PollingCaptureRunner`, the pipeline cache key includes that identity plus source-target/render-size/depth-settings, and changed or ambiguous content fails closed. Debug output now carries the media identity and `media_depth_cache_reason`. A pure `StableCadenceGovernor` policy module (`stereo_runtime/cadence_governor.py`) with 72/60/45/30 hysteresis, queue/depth-age guards, and observe-only default is unit-tested; it is not yet wired to change presentation cadence.
