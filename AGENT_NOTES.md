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
