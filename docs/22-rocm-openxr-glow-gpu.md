# AMD ROCm OpenXR glow GPU path

Glow generation uses the Vulkan compute backend on the AMD GPU. HIP writes only
exported staging buffers; Vulkan then produces the sampled glow image. Direct
HIP writes into Vulkan-imported image memory make the AMD driver lose the Vulkan
device under sustained OpenXR compositing (Virtual Desktop's OpenXR log reports
`VkStatus failure [-4]` at `xrEndFrame`). NVIDIA's implementation is unchanged.

## Ownership and synchronization

The former ring checked leases with `id(slot)`, while `acquire()` recorded
`id(slot.resource)`. This allowed the worker to overwrite images still held by
OpenXR. Selection, acquisition, and release now share a lock and resource keys.
The currently published image also stays reserved between frame acquisitions.

The worker waits for Vulkan completion before publishing, releases an image
from sampling before reusing it, and only overwrites slots not currently held
by OpenXR. Publication follows the Vulkan barrier to shader-read layout. See the
[Vulkan synchronization specification](https://docs.vulkan.org/spec/latest/chapters/synchronization.html)
for the image access and layout dependency rules.

Shutdown joins the worker before destroying its images. Unclaimed ROCm prewarm
workers are also closed when OpenXR initialization fails.

## Validation

```powershell
src\python3\python.exe -m pytest -q tests/test_rocm_torch_glow_source.py
src\python3\python.exe src\tools\rocm_glow_smoke.py --seconds 120
```

On RX 9060 XT with bundled torch `2.11.0+rocm10.1.0a20260822`, a 20-second
smoke passed 12 exact patterned-image readbacks, 424 glow publications, and 46
checks that retained images stayed unchanged. Diagnostic readbacks exist only
in the smoke tool. Visual/headset validation is required to confirm the
previous device-loss failure no longer appears.

## OpenXR eye handoff

The AMD output adapter writes HIP results into exported Vulkan staging buffers,
then Vulkan runs `vkCmdCopyBufferToImage` so HIP never writes image memory
directly. This lasted longer in headset tests than both direct image copies and
host staging. `D2S_ROCM_EYE_HOST_STAGING=1` opts into pure host staging, but the
GPU buffer path is the default because it sustained ~70 FPS longer in testing.
Host staging passed a 36-frame real-GPU handoff with exact Vulkan readback.

`D2S_OPENXR_DISABLE_GLOW_DRAW=1` disables rendering regardless of upload success.
AMD clears this legacy diagnostic automatically and keeps an explicit AMD-only
`D2S_ROCM_DISABLE_GLOW_DRAW=1` switch; restart any launcher that inherited the
old setting.

## VDXR stability defaults

The normal ROCm OpenXR path keeps the selected GPU Filament controller model
and GPU tool-quad menu visible. For A/B diagnosis, set
`D2S_ROCM_DISABLE_OPENXR_OVERLAYS=1`; this selects the Vulkan controller proxy
and disables tool-quad swapchains. The isolation mode avoids the optional
controller/composition paths while investigating AMD device-loss or stale
layer failures. If an inherited launch environment still sets the isolation
flag, `D2S_ROCM_ENABLE_OPENXR_OVERLAYS=1` explicitly restores the GPU controller
callout and controller-anchored tool panels. The projection quality allocation
remains disabled by default on ROCm; enable it explicitly only for testing.
