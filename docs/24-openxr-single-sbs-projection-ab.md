# OpenXR Single-SBS Projection A/B

## Purpose

Provide an opt-in OpenXR Vulkan presentation layout that stores both eye views in
one full-width, `array_size=1` swapchain image. The two Projection views reference
the same swapchain handle with different image rectangles:

- left view: `(0, 0, eye_width, eye_height)`
- right view: `(eye_width, 0, eye_width, eye_height)`
- both views: `image_array_index = 0`

This preserves distinct left/right eye content. It is not a
`CompositionLayerQuad` with `eyeVisibility=BOTH`; a single BOTH quad would show
the same SBS image to both eyes and is not used for stereo.

The existing per-eye path remains the default. Enable the experimental path with:

```text
D2S_OPENXR_PROJECTION_SBS_SINGLE_SWAPCHAIN=1
```

## Rendering contract

- The target is one `array_size=1` OpenXR swapchain of `2 * eye_width` by
  `eye_height`.
- Vulkan Projection Composer draws each logical eye into its half-image viewport
  and scissor rectangle.
- Filament GLB/panorama backgrounds use the existing Filament multiview HDR
  targets and resolve each eye into its SBS half.
- If multiview/HDR is unavailable or the combined swapchain cannot be created,
  the presenter recreates the validated per-eye swapchains before the first
  frame and records the fallback reason.
- Screen, Glow, laser, controller and panorama passes use the same logical eye
  target rectangle, so background composition remains compatible.
- The Filament controller overlay is bypassed for this layout because its
  native renderer targets one full swapchain extent. The independent Vulkan
  controller proxy is enabled instead; it retains a true two-eye composition
  layer and avoids stretching one eye's controllers across both SBS halves.
- If the Vulkan controller-proxy swapchains cannot be created, the presenter
  silently returns to the existing per-eye Projection layout before the first
  frame and records the fallback reason.

## A/B acceptance

Run the same scene and settings with the flag off and on. Record actual GPU,
eye extents, XR render scale, `xr_present_fps`, `xr_projection_total`,
`xr_end`, `screen_new`, frame-time p50/p95, `projection_layout`,
`projection_swapchains`, `projection_array_size`, and fallback reason.

Acceptance requires different left/right eye content, unchanged background and
screen clarity, no device loss, and a measured frame-rate/frame-time benefit.
The feature remains opt-in until RX 9060 XT measurements establish the benefit.

## 2026-11-27 RX 9060 XT / OpenXR Simulator result

Test input was the same 4K desktop capture scene while the Bilibili video
`BV17Qs86jEAi` played full-screen. OpenXR Simulator ran at 90 Hz with render
scale `1.0`; hole fill was verifiably disabled (`rt_fill=none`,
`rt_hole_fill=none(0/0.00)`). Each row uses only stable 90 Hz presentation
windows; the startup warm-up window is excluded.

| Path | Samples | Projection total median | Projection p50 median | Projection p95 median | XR frame p95 median | Present FPS median |
|---|---:|---:|---:|---:|---:|---:|
| Per-eye + Filament controllers | 4 | 3.815 ms | 3.465 ms | 5.300 ms | 13.670 ms | 90.0 |
| Per-eye + Vulkan proxy | 4 | 3.785 ms | 3.680 ms | 5.180 ms | 14.290 ms | 90.0 |
| Single-SBS + Vulkan proxy | 5 | 1.080 ms | 0.950 ms | 1.760 ms | 13.260 ms | 90.0 |

Against per-eye + Vulkan proxy, the single-SBS path reduced projection total by
`71.5%`, p50 by `74.2%`, and p95 by `66.0%`. Against the full per-eye baseline
with Filament controllers, the reductions were `71.7%`, `72.6%`, and `66.8%`.
The simulator's 90 Hz cap means these numbers demonstrate GPU/submission
headroom, not a higher display rate.

Clarity was held constant. Both paths reported the same `1280x1400` logical eye
target, `640x360` screen sampling image, `667x386` screen footprint, and
`5.76x5.60` source pixels per screen pixel. The screenshot showed distinct
left/right perspective and the B-button guide in both eyes.

The combined SBS swapchain is constrained by the runtime's per-image width
limits. At render scale `2.0`, the required `5120x2800` swapchain failed
`xrCreateSwapchain` validation and the presenter silently fell back to the
per-eye path before the first frame. The working configuration is therefore
render scale `1.0` on the tested runtime.

A `3d_cinema` GLB test also fell back before the first frame because the current
Filament bridge reported multiview/HDR unavailable. Single-SBS is currently a
Default/dark-room optimization; GLB environments need a future native Filament
SBS viewport or another two-eye resolve path.

## Evidence files

- Baseline, per-eye Filament controllers:
  [`runtime-diagnostics/openxr-simulator-per-eye-video-20261127.log`](../runtime-diagnostics/openxr-simulator-per-eye-video-20261127.log)
- Per-eye Vulkan proxy control:
  [`runtime-diagnostics/openxr-simulator-per-eye-proxy-video-20261127.log`](../runtime-diagnostics/openxr-simulator-per-eye-proxy-video-20261127.log)
- Single-SBS Vulkan proxy:
  [`runtime-diagnostics/openxr-simulator-sbs-on-proxy-video-20261127.log`](../runtime-diagnostics/openxr-simulator-sbs-on-proxy-video-20261127.log)
- Render-scale 2.0 pre-frame fallback:
  [`runtime-diagnostics/openxr-simulator-sbs-render2-fallback-20261127.log`](../runtime-diagnostics/openxr-simulator-sbs-render2-fallback-20261127.log)
- Filament GLB pre-frame fallback:
  [`runtime-diagnostics/openxr-simulator-sbs-filament-fallback-20261127.log`](../runtime-diagnostics/openxr-simulator-sbs-filament-fallback-20261127.log)

## Remaining work

1. Validate render scale `1.0` for at least 10 minutes on the physical Quest 2 /
   VDXR path with the final controller-proxy ordering.
2. Add a native Filament eye-rectangle viewport API so GLB environments can
   render directly into the two halves without the currently unavailable
   multiview path.
3. Evaluate a layered `array_size=2` path for render scale `2.0`, where a single
   full-width image exceeds the runtime's swapchain width limit.
4. Keep the silent pre-frame fallback as the release contract. Never fall back
   after images have been submitted and never surface the switch as a runtime
   error.
