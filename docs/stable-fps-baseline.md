# Stable FPS Baseline

This document records the Phase 0 evidence for the stable-cadence work. It is a
measurement record, not a claim that the current path meets every target.

## Quality Gate

Depth substitution is not an FPS optimization. A depth profile may be reused only
when the source identity is exact:

- media generation;
- source frame index;
- source PTS/position;
- source target;
- render size;
- depth backend/model/resolution;
- active settings version.

The cache also has a maximum age, but age never makes unrelated content eligible.
Cursor-only reuse remains a separate path with its own change-isolation check.

## Local Viewer / ROCm Baseline

Environment:

- Windows 11;
- AMD Radeon RX 9060 XT, ROCm 7;
- MIGraphX depth backend;
- `Distill-Any-Depth-Base`, depth resolution 518;
- source: 3840x1920 packed stereo, Half-SBS output;
- output display: Dell U2410, 1920x1200 at 59 Hz;
- local viewer with `D2S_MEDIA_FRAME_BUFFER=8`;
- source file: 4K 120 FPS HDR10 HEVC sample.

Observed over repeated 300-presentation metric windows after warm-up:

| Metric | Observed range |
| --- | ---: |
| Display/present FPS | 58.4-59.2 |
| Unique content FPS | 28.8-33.6 |
| Runtime execution p95 | 3.9-4.2 ms |
| Runtime GPU wait p95 | 13.5-13.9 ms |
| Runtime completion p95 | 16.5-17.3 ms |
| Viewer Vulkan total p95 | 1.6-1.8 ms |
| Presentation latency p95 | about 22-24 ms |

Interpretation:

- presentation is stable near the panel cadence;
- the depth/synthesis completion rate is the content-rate limiter;
- the dominant current contributor is GPU depth/synthesis completion and wait,
  not Vulkan presentation or pack;
- media playback itself remains 1.0x wall-clock speed while the 3D pipeline
  consumes a lower unique-frame rate.

That earlier local-viewer run predates the completion-latency and media-pump
expansion below. Its values remain a valid presentation/content baseline, while
the new OpenXR section records the end-to-end completion measurements.

## OpenXR Simulator Observe-Only Baseline

Measured on 2026-11-15 with:

- Windows 11 and AMD Radeon RX 9060 XT / ROCm 7;
- OpenXR Simulator 1.7.1, `1280x1400` per-eye source swapchain;
- D2S `Pico 4 / 4 Ultra` headset profile, MIGraphX depth at 518,
  `Distill-Any-Depth-Base`, Triton synthesis;
- source `3840x1920` HEVC Main10, 120 fps, PQ / BT.2020;
- the 4K HDR sample under `F:\Videos\` starting at 12 seconds.

The simulator runtime reported `target_fps=90` and fresh left/right swapchain
images. Setting the simulator preview preference to 72 did not change the
compositor submission rate, so these runs measure a 90 Hz repeated display with
the D2S cadence target set independently to 72 or 30. This is not a true 72 Hz
panel-refresh test.

### App target 72 Hz

| Setting | Stable content FPS | XR submissions | `rt_total` p95 |
| --- | ---: | ---: | ---: |
| XR Render 1.0 | 24.4-26.2 | 90.0-90.2 | 4.9-5.7 ms |
| XR Render 2.0 | 26.0-26.4 | 89.9-90.2 | 5.5-6.0 ms |

Render scale did not materially change content rate. The XR compositor kept
submitting old eye frames near 90 Hz while D2S produced about 26 unique frames
per second.

### App target 30 Hz with completion timing

| Metric | Observed stable range |
| --- | ---: |
| Content FPS | 22.0-25.4 |
| Media frame-pump FPS | 22.4-28.5 |
| Media raw-read p50 | 39.2-44.1 ms |
| Media raw-read p95 | 42.7-52.8 ms |
| Frame execution p95 | 6.8-8.7 ms |
| GPU wait p95 | 27.1-27.7 ms |
| Runtime completion p95 | 32.9-33.4 ms |
| Capture-to-runtime-ready p95 | 52.2-54.3 ms |
| Media buffer | `0/8` |

At 30 Hz, published-frame periods still alternate around the 33.3 ms budget
(observed 33.3-46.1 ms). Overloaded windows therefore report
`proposed=30 cadence_reason=floor_over_budget`. The governor no longer mistakes
the short enqueue segment for end-to-end cadence.

## Dominant Bottleneck

The current limiter is the host-backed P010 raw-video pipe, not depth or synthesis:

- direct FFmpeg with `hevc_amf` or software `hevc` completed 1440 frames at the
  72 fps filter rate when writing raw P010 to `NUL`;
- a Python `stdout.readinto` reader using the same 22 MB frame size averaged
  about 39 FPS outside the application;
- preallocating 12 reusable P010 buffers raised the standalone reader only to
  about 44 FPS, still below 60/72, so pooling alone cannot close the gap;
 - inside D2S, the bounded pump records 39-44 ms median raw reads and 22-28 FPS;
- the media buffer stays empty, so inference is not accumulating a backlog;
- GPU depth/synthesis execution is only 6.8-8.7 ms p95, while asynchronous GPU
  wait and OpenXR presentation raise completion latency to about 33 ms p95.

AMF completed the decode workload but crashed during forced process teardown in
the standalone benchmark. This is a harness teardown issue, not a decode-rate
shortfall.

Artifact-free options remain:

1. keep a GPU-resident AMF/NVDEC surface and import it into the compute/present
   path instead of materializing every P010 frame through a Python byte pipe;
2. pool and pipeline host buffers as an intermediate fallback, preserving exact
   frame identity and the existing image-quality settings;
3. accept a stable lower unique-content cadence while the compositor repeats the
   last valid stereo frame.

## Cadence Governor

The observe-only policy is enabled with:

```text
D2S_CADENCE_GOVERNOR_OBSERVE=1
D2S_CADENCE_TARGET_HZ=72
```

The governor observes the interval between successfully published runtime frames,
not just the enqueue slice. Its ladder is `72 -> 60 -> 45 -> 30`; a sustained
over-budget period steps down, and the lowest rung reports `floor_over_budget`
instead of claiming stability.

Rolling metadata now includes:

- `stable_cadence_frame_ms` and `stable_cadence_budget_ms`;
- over-budget and headroom streaks;
- depth age, queue depth, and dropped-depth count;
- whether the sample came from the published-frame period or completion fallback;
- independent media pump FPS, raw-read p50/p95, buffer depth, and decode backend.

Control mode remains disabled by default. The 4K HDR profile cannot sustain the
30 Hz floor on this host decode path, so enabling control would only codify a
lower content cadence without removing the host-pipe bottleneck.

## Local Viewer Observe-Only Result

Observed local-viewer run (about 40 seconds, target 59 Hz):

```text
cadence_active=59 cadence_proposed=59 cadence_reason=steady
present_fps=58.05-59.16 content_fps=31.78-32.32
present_unique_fps=31.78-32.32 present_repeat_fps=25.73-27.38
depth_total=1.55ms depth_model=0.94ms synthesis=0.48ms
rt_total_p95=3.58ms runtime_execution_p95=4.12-4.38ms
runtime_gpu_wait_p95=18.15-18.68ms
media_gen=0 source_frame=751 source_pts=24.52 depth_cache=no_cache
```

Changed video frames do not reuse an unrelated depth profile. A cache miss is
reported instead of silently substituting stale depth.

## Next Gate

Before enabling governor control, collect the same metrics on a runtime that can
actually select 72/60 Hz, then repeat the A/B after moving 10-bit media frames to
a GPU-resident decode/import path. Acceptance still requires no stale-depth reuse,
no hidden scaling, source-time 1.0x playback, and no depth or stereo math changes.

## GPU Media Pool Result

Measured 2026-11-15 after replacing the per-frame ROCm tensor allocation and
CPU stream synchronize with a slot-indexed GPU tensor pool:

- decoder: `ffmpeg-d3d11va-gpu`;
- source: 4K 120 fps HDR10 HEVC, start 60 seconds;
- `ring_size=8` and `D2S_MEDIA_FRAME_BUFFER=8`;
- 180-frame direct decoder/import smoke: 8 imported textures and 8 reusable
  tensors, no hot-path CPU synchronize;
- import timing p50 `0.20 ms`, p95 `1.08 ms`;
- local viewer: media read p50 `0.32 ms`, p95 `0.62 ms`, decode pump
  `120.77 fps`;
- OpenXR: media read p50 `0.40-0.44 ms`, p95 `1.41-1.93 ms`, decode pump
  `120.5-120.9 fps`, buffer `8/8`;
- OpenXR content rate improved from the earlier `22-28 fps` to `37-38 fps`;
- a simulator screenshot after the 60-second resume showed the expected
  PQ/BT.2020 -> SDR/BT.709 non-black SBS image.

The remaining OpenXR limit is the depth/runtime completion wait, not media
readback. The generated content cadence still cannot be claimed as 60 fps.

## Screen and Window Pool Result

`WindowsCaptureROCm` now passes a six-slot output pool to the bundled WGC
wrapper. The wrapper uses a dedicated HIP copy stream, waits on the consumer
stream only when a slot wraps, and orders the consumer stream against the copy
before publishing the tensor. No CPU pixel readback or per-frame clone is added.

Validation runs:

- Monitor Local Viewer: exit 0, source `3840x2160`, capture-to-runtime RGB
  preprocessing `0.19 ms`, non-black output;
- Window Local Viewer: exit 0, source `1802x1400`, capture-to-runtime RGB
  preprocessing observed after warmup, non-black output;
- native WGC without HIP import on the same 4K desktop produced about 38 fps,
  so the remaining capture-rate floor is in the WGC/desktop producer on this
  host rather than a CPU readback in D2S.

The OpenXR Simulator media run rendered correctly but its process exited with
Windows status `0xC0000409` when the harness stopped the runtime. The failure
reproduces with tool quads disabled and does not occur in the Local Viewer run;
it is currently treated as an OpenXR Simulator/Filament teardown issue, not a
media decode fallback or a black-frame failure.
