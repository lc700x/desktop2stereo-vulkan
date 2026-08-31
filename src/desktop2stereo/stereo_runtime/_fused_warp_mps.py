"""Fused warp+SBS-pack Metal kernel (torch.mps.compile_shader) for the
Vulkan local viewer path.

Replaces the torch node chain (permute/cat/contiguous + separate depth
quantize) with ONE Metal kernel that applies the exact Metal-warp math
(gaussian taps, asymmetric shaping, edge falloff) and writes final
Half-SBS RGBA8 bytes. The packer thread then only moves bytes into the
IOSurface stage / host frame — no per-frame SBS synthesis on the MPS
stream, minimal queue coupling.

Kernel validated pixel-exact against a numpy reference
(max abs diff 0 on random inputs); see tools/ notes in docs/33.

Platform: darwin + MPS only. Kill switch: D2S_VK_FUSED_WARP=0.
"""

from __future__ import annotations

import functools
import os
import sys

import numpy as np

WARP_MSL = r"""
#include <metal_stdlib>
using namespace metal;

static inline float sampf(device float* img, uint W, uint H, float u, float v) {
    float x = clamp(u * (float)W - 0.5f, 0.0f, (float)W - 1.001f);
    float y = clamp(v * (float)H - 0.5f, 0.0f, (float)H - 1.001f);
    int x0 = (int)floor(x), y0 = (int)floor(y);
    int x1 = min(x0 + 1, (int)W - 1), y1 = min(y0 + 1, (int)H - 1);
    float fx = x - (float)x0, fy = y - (float)y0;
    float a = img[y0 * W + x0], b_ = img[y0 * W + x1];
    float c = img[y1 * W + x0], d = img[y1 * W + x1];
    return mix(mix(a, b_, fx), mix(c, d, fx), fy);
}

kernel void warp_pack(
    device uchar* out  [[buffer(0)]],
    device float* col  [[buffer(1)]],
    device float* dep  [[buffer(2)]],
    constant float& eyeOffset     [[buffer(3)]],
    constant float& depthStrength [[buffer(4)]],
    constant float& convergence   [[buffer(5)]],
    constant uint&  srcW          [[buffer(6)]],
    constant uint&  srcH          [[buffer(7)]],
    constant uint&  outW          [[buffer(8)]],
    constant uint&  outH          [[buffer(9)]],
    constant float& smoothTexels  [[buffer(10)]],
    uint idx [[thread_position_in_grid]])
{
    // HALF-SBS contract: output frame is outW x outH (runtime input
    // resolution per the NVIDIA-path contract), each eye squeezed into
    // outW/2; full_sbs passes outW = 2*srcW for unsqueezed eyes.
    uint pw = outW / 2u;           // per-eye width
    uint total = outW * outH * 4u;
    if (idx >= total) return;
    uint py = idx / (outW * 4u);
    uint rem = idx % (outW * 4u);
    uint px = rem / 4u;
    uint comp = rem % 4u;
    if (comp == 3u) { out[idx] = 255u; return; }

    bool left = px < pw;
    uint lx = left ? px : (px - pw);       // coordinate within the eye
    float eye = left ? -eyeOffset : eyeOffset;
    float u = ((float)lx + 0.5f) / (float)pw;   // normalized across the eye
    float v = ((float)py + 0.5f) / (float)outH;

    // 3-tap gaussian depth smoothing. Aperture scales with the eye/output
    // ratio so the EFFECTIVE smoothing matches whatever presentation res --
    // native-res warping otherwise amplifies depth estimation noise into
    // visible edge shimmer that the upscaled-540p era never showed.
    float du = smoothTexels / (float)srcW;
    float d0 = sampf(dep, srcW, srcH, u, v);
    float dm = sampf(dep, srcW, srcH, u - du, v);
    float dp_ = sampf(dep, srcW, srcH, u + du, v);
    float d = clamp(d0 * 0.7f + dm * 0.15f + dp_ * 0.15f, 0.0f, 1.0f);
    float d_shaped = d * (1.0f + 0.35f * (1.0f - d));
    float shift = (d_shaped - convergence) * depthStrength * eye;
    float e0 = smoothstep(0.0f, 0.05f, u);
    float e1 = smoothstep(0.0f, 0.05f, 1.0f - u);
    shift *= e0 * e1;
    float fx = clamp(u + shift, 0.0f, 1.0f);

    float cval = sampf(col + (uint)comp * (uint)(srcW * srcH),
                       srcW, srcH, fx, v)
                 * 255.0f + 0.5f;  // planar CHW channel base
    out[idx] = (uchar)clamp(cval, 0.0f, 255.0f);
}
"""


@functools.lru_cache(maxsize=1)
def _lib():
    import torch

    return torch.mps.compile_shader(WARP_MSL)


def warp_params_from_env() -> tuple[float, float, float]:
    """Mirror macos_metal_viewer's calibration knobs exactly."""
    ipd_uv = float(os.environ.get("D2S_METAL_WARP_IPD", "0.064") or 0.064)
    depth_strength = 0.1 * float(
        os.environ.get("D2S_METAL_WARP_DEPTH_STRENGTH", "4.0") or 4.0
    )
    convergence = float(os.environ.get("D2S_METAL_WARP_CONVERGENCE", "0.0") or 0.0)
    return ipd_uv / 2.0, depth_strength, convergence


def fused_enabled() -> bool:
    """Darwin + Vulkan viewer + not explicitly disabled."""
    return (
        sys.platform == "darwin"
        and os.environ.get("D2S_MAC_VIEWER") == "vulkan"
        and os.environ.get("D2S_VK_FUSED_WARP", "1")
        not in {"0", "false", "off"}
    )


@functools.lru_cache(maxsize=8)
def _smooth_texels_cached(key: tuple[int, int, int]) -> float:
    import os as _os

    raw = _os.environ.get("D2S_WARP_DEPTH_SMOOTH_TEXELS", "")
    try:
        return max(0.0, float(raw))
    except Exception:
        pass
    eye_w, src_w, base = key
    # Scale-invariant default: 1.5 texels at the reference where eye width
    # equals source width; grows proportionally when the packed frame is
    # larger than the source (native-res presentation).
    return 1.5 * (float(eye_w) / float(src_w)) if src_w else 1.5


def warp_smooth_texels(src_w: int, out_w: int) -> float:
    """Depth-smoothing aperture in source texels for the warp kernel."""
    eye_w = max(1, int(out_w) // 2)  # half-SBS per-eye width
    return _smooth_texels_cached((eye_w, int(src_w), int(out_w)))


def pack_target(src_w: int, src_h: int, output_format: str = "half_sbs") -> tuple[int, int]:
    """Frame dims for a given runtime input size and output format.

    Mirrors the NVIDIA local-mode contract: SBS geometry follows the
    RUNTIME input resolution and the selected format -- never the viewer
    window. 1080p in -> half_sbs 1920x1080, full_sbs 3840x1080.
    """
    sw, sh = int(src_w), int(src_h)
    if str(output_format) == "full_sbs":
        return sw * 2, sh
    return sw - sw % 2, sh


def fused_sbs_pack(rgb_f32_chw, depth_f32, host_out=None, out_size=None,
                   output_format: str = "half_sbs"):
    """Run the fused kernel; return (host_view|None, w, h) like
    _pack_sbs_host_frame, or None on any failure (caller falls back).

    Default frame dims follow the NVIDIA local-mode contract: derived from
    the runtime INPUT resolution and ``output_format`` (half_sbs WxH,
    full_sbs 2WxH). ``out_size`` remains as an explicit override."""
    try:
        import torch

        if rgb_f32_chw.dim() == 4 and int(rgb_f32_chw.shape[0]) == 1:
            rgb_f32_chw = rgb_f32_chw.squeeze(0)  # BCHW -> CHW
        if rgb_f32_chw.dim() != 3:
            if os.environ.get("D2S_FUSED_DEBUG"):
                print(f"[fused] skip: dim={rgb_f32_chw.dim()}", flush=True)
            return None
        channels, h, w = (
            int(rgb_f32_chw.shape[0]),
            int(rgb_f32_chw.shape[-2]),
            int(rgb_f32_chw.shape[-1]),
        )
        if channels != 3 or h <= 0 or w <= 0:
            if os.environ.get("D2S_FUSED_DEBUG"):
                print(f"[fused] skip: ch={channels} h={h} w={w}", flush=True)
            return None
        dep = depth_f32
        if dep.dim() == 3:
            dep = dep.squeeze(0)
        ow, oh = (
            (int(out_size[0]), int(out_size[1]))
            if out_size is not None
            else pack_target(w, h, output_format)
        )
        if ow % 2 != 0 or ow < 4 or oh < 4:
            return None  # half-SBS needs an even frame width
        out_t = torch.empty(ow * oh * 4, dtype=torch.uint8, device="mps")
        eo, ds, cv = warp_params_from_env()
        stex = warp_smooth_texels(w, ow)
        _lib().warp_pack(
            out_t, rgb_f32_chw.contiguous(), dep.contiguous(),
            float(eo), float(ds), float(cv),
            int(w), int(h), int(ow), int(oh), float(stex),
        )
        # Half-SBS: reported dims are the FRAME dims (ow x oh), matching the
        # synthesized half_sbs contract the viewer was built around.
        if host_out is not None:
            dst = torch.frombuffer(host_out, dtype=torch.uint8)
            dst.copy_(out_t)
            view = np.frombuffer(host_out, dtype=np.uint8).reshape(oh, ow, 4)
            return view, ow, oh
        host = out_t.cpu().numpy().reshape(oh, ow, 4)
        return host, ow, oh
    except Exception as exc:
        if os.environ.get("D2S_FUSED_DEBUG"):
            print(f"[fused] pack failed: {exc!r}", flush=True)
        return None
