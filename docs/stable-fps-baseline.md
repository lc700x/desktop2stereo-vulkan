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

The current logs do not expose a complete per-stage depth/synthesis breakdown,
so this baseline cannot yet distinguish model execution from depth transport or
synthesis wait with sufficient precision.

## OpenXR Baseline

Historical OpenXR Simulator evidence on the same ROCm machine:

- PICO 4 profile, 1280x1400 per eye;
- simulator selected 72 Hz;
- 4K HDR HEVC source through FFmpeg AMF host-backed decode;
- MIGraphX depth and Triton synthesis;
- active media processing observed around 45-51 FPS;
- XR presentation around 89 FPS in the simulator, which is not the media
  content-processing rate.

This is a preliminary profile, not a final 72/60 cadence comparison. It should be
re-run with the exact frame identity, skip/repeat reasons, and stage breakdown
added by Phases 1 and 4.

## Required Next Measurements

The following are still missing from the rolling status output and must be added
before the cadence governor is enabled:

- source capture timestamp and source PTS;
- depth source PTS, capture frame ID, and depth age;
- depth cache hit/miss reason;
- raw queue and depth queue depth;
- in-flight depth slots;
- `depth_total_ms`;
- `synthesis_ms`;
- `openxr_render_ms`;
- `pack_ms`;
- unique content FPS separated from repeated presentation FPS;
- dropped and repeated counters with reason codes.

## Candidate Cadence Ladder

The governor will start in observe-only mode and evaluate:

- requested headset cadence;
- 72 Hz;
- 60 Hz;
- 45 Hz;
- 30 Hz.

It must step down only after sustained over-budget evidence and step up only
after a sustained headroom window. A lower stable cadence is preferred over an
unstable higher cadence.

## Observe-Only Governor

The governor policy is available behind:

```text
D2S_CADENCE_GOVERNOR_OBSERVE=1
D2S_CADENCE_TARGET_HZ=72
```

In this mode it records `stable_cadence_active_hz`,
`stable_cadence_proposed_hz`, `stable_cadence_reason`,
`stable_cadence_depth_queue`, and `stable_cadence_dropped` in runtime debug
metadata. It does not change presentation or content cadence.

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

The live run confirms that changed video frames do not reuse an unrelated
cached depth profile. The missing cache is expected on continuous motion when
no exact duplicate frame is available; the pipeline reports a skip/miss instead
of silently substituting stale depth.
