"""macOS Metal local viewer — zero-copy-ish present path (Darwin only).

Replaces the MoltenVK swapchain chain (GPU quantize -> .cpu() -> vkMapMemory ->
vkCmdCopyBufferToImage -> blit) with the proven Desktop2Stereo v2.5 approach:

    SBS float32 (MPS) -> quantize on MPS -> one .cpu() copy -> numpy RGBA
    -> CAMetalLayer MTLTexture replaceRegion -> textured-quad draw

Interface matches VulkanLocalViewer so the runtime can switch viewers without
touching the pipeline: initialize() / present(frame) / poll_events() / close().
"""

from __future__ import annotations

import os
import queue
import struct
import sys
import time
from dataclasses import replace
from typing import Any

from utils.display_info import resolve_glfw_monitor_index

if sys.platform == "darwin":
    import Metal
    import Quartz
from viewer.vulkan_local_viewer import (
    VulkanLocalViewerConfig,
    fit_rect,
    frame_to_rgba_bytes,
    is_exclusive_fullscreen_toggle,
    pack_frame_to_rgba8,
    present_fps_if_due,
)

METAL_SHADER = r"""
#include <metal_stdlib>
using namespace metal;

struct VOut {
    float4 position [[position]];
    float2 uv;
};

vertex VOut quad_vertex(uint vid [[vertex_id]]) {
    float2 positions[4] = {
        float2(-1.0, -1.0),
        float2( 1.0, -1.0),
        float2(-1.0,  1.0),
        float2( 1.0,  1.0),
    };
    // Top row of the uploaded texture must appear at the top of the screen.
    float2 uvs[4] = {
        float2(0.0, 1.0),
        float2(1.0, 1.0),
        float2(0.0, 0.0),
        float2(1.0, 0.0),
    };
    VOut out;
    out.position = float4(positions[vid], 0.0, 1.0);
    out.uv = uvs[vid];
    return out;
}

fragment float4 quad_fragment(
    VOut in [[stage_in]],
    texture2d<float> tex [[texture(0)]],
    sampler smp [[sampler(0)]],
    constant float4 &uv_xform [[buffer(0)]]
) {
    float2 uv = (in.uv - uv_xform.zw) / max(uv_xform.xy, float2(1e-6));
    if (any(uv < 0.0) || any(uv > 1.0)) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }
    return tex.sample(smp, uv);
}
"""

# Deferred stereo warp (ported from proven Desktop2Stereo v2.5 metal_viewer):
# the fragment shader samples the depth texture per-pixel and displaces each
# eye at draw time, so no SBS tensor is ever synthesized or uploaded.
WARP_SHADER = r"""
#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float2 uv;
};

struct Uniforms {
    float eyeOffset;
    float depthStrength;
    float convergence;
    float _pad0;
    float4 viewport;
    float mode;
    float featherEnabled;
    float featherWidth;
    float _pad1;
};

vertex VertexOut vertex_main(uint vid [[vertex_id]]) {
    float2 positions[4] = {
        float2(-1.0, -1.0),
        float2( 1.0, -1.0),
        float2(-1.0,  1.0),
        float2( 1.0,  1.0),
    };
    float2 uvs[4] = {
        float2(0.0, 1.0),
        float2(1.0, 1.0),
        float2(0.0, 0.0),
        float2(1.0, 0.0),
    };
    VertexOut out;
    out.position = float4(positions[vid], 0.0, 1.0);
    out.uv = uvs[vid];
    return out;
}

static float2 displaced_uv(float2 uv, float eye, texture2d<float> depthTex, sampler s, constant Uniforms& u) {
    // 3-tap Gaussian depth smoothing along horizontal parallax
    float2 ds_dir = float2(sign(eye) / float(depthTex.get_width()) * 1.5, 0.0);
    float d0 = depthTex.sample(s, uv).r;
    float dm = depthTex.sample(s, uv - ds_dir).r;
    float dp = depthTex.sample(s, uv + ds_dir).r;
    float d = clamp(d0 * 0.7 + dm * 0.15 + dp * 0.15, 0.0, 1.0);
    // Asymmetric depth shaping: boosts near-object pop ~35%
    float depth_shaped = d * (1.0 + 0.35 * (1.0 - d));
    float shift = (depth_shaped - u.convergence) * u.depthStrength * eye;
    // Edge falloff: reduce parallax at image borders to prevent sampling artifacts
    float edge_falloff = smoothstep(0.0, 0.05, uv.x) * smoothstep(1.0, 0.95, uv.x);
    shift *= edge_falloff;
    return float2(clamp(uv.x + shift, 0.0, 1.0), uv.y);
}

static float3 spectral_r_ultrafast(float t) {
    float3 color1 = float3(0.0, 0.298, 0.651);
    float3 color2 = float3(0.0, 0.5, 0.0);
    float3 color3 = float3(1.0, 0.851, 0.0);
    float3 color4 = float3(0.988, 0.0, 0.0);

    float w1 = max(0.0, 1.0 - abs(t - 0.125) * 4.0);
    float w2 = max(0.0, 1.0 - abs(t - 0.375) * 4.0);
    float w3 = max(0.0, 1.0 - abs(t - 0.625) * 4.0);
    float w4 = max(0.0, 1.0 - abs(t - 0.875) * 4.0);
    float total = w1 + w2 + w3 + w4;
    if (total > 0.0) {
        w1 /= total;
        w2 /= total;
        w3 /= total;
        w4 /= total;
    }
    return color1 * w1 + color2 * w2 + color3 * w3 + color4 * w4;
}

fragment float4 fragment_main(
    VertexOut in [[stage_in]],
    texture2d<float> colorTex [[texture(0)]],
    texture2d<float> depthTex [[texture(1)]],
    constant Uniforms& u [[buffer(0)]]
) {
    constexpr sampler s(address::clamp_to_edge, filter::linear);
    float2 frag = in.position.xy;
    if (frag.x < u.viewport.x || frag.y < u.viewport.y ||
        frag.x >= u.viewport.x + u.viewport.z || frag.y >= u.viewport.y + u.viewport.w) {
        discard_fragment();
    }
    float2 uv = clamp((frag - u.viewport.xy) / u.viewport.zw, 0.0, 1.0);
    int mode = int(u.mode + 0.5);

    if (mode == 0) {
        return colorTex.sample(s, uv);
    }

    if (mode == 7) {
        return colorTex.sample(s, displaced_uv(uv, u.eyeOffset, depthTex, s, u));
    }

    if (mode == 3) {
        float d = depthTex.sample(s, uv).r;
        return float4(spectral_r_ultrafast(d), 1.0);
    }

    float4 color;
    if (mode == 4) {
        float2 luv = displaced_uv(uv, -u.eyeOffset, depthTex, s, u);
        float2 ruv = displaced_uv(uv,  u.eyeOffset, depthTex, s, u);
        float4 lc = colorTex.sample(s, luv);
        float4 rc = colorTex.sample(s, ruv);
        color = float4(lc.r, rc.g, rc.b, 1.0);
    } else if (mode == 5) {
        float eye = (fmod(floor(in.position.y), 2.0) < 1.0) ? -u.eyeOffset : u.eyeOffset;
        color = colorTex.sample(s, displaced_uv(uv, eye, depthTex, s, u));
    } else if (mode == 6) {
        float eye = (fmod(floor(in.position.x), 2.0) < 1.0) ? -u.eyeOffset : u.eyeOffset;
        color = colorTex.sample(s, displaced_uv(uv, eye, depthTex, s, u));
    } else if (mode == 2) {
        float eye = uv.y < 0.5 ? u.eyeOffset : -u.eyeOffset;
        float2 src = float2(uv.x, uv.y < 0.5 ? uv.y * 2.0 : (uv.y - 0.5) * 2.0);
        color = colorTex.sample(s, displaced_uv(src, eye, depthTex, s, u));
    } else {
        float eye = uv.x < 0.5 ? -u.eyeOffset : u.eyeOffset;
        float2 src = float2(uv.x < 0.5 ? uv.x * 2.0 : (uv.x - 0.5) * 2.0, uv.y);
        color = colorTex.sample(s, displaced_uv(src, eye, depthTex, s, u));
    }
    return color;
}
"""


class MetalLocalViewer:
    """GLFW window + CAMetalLayer renderer for packed SBS frames (macOS)."""

    def __init__(self, config: VulkanLocalViewerConfig) -> None:
        self.config = config
        self.glfw = None
        self.window = None
        self.device = None
        self.queue = None
        self.pipeline = None
        self.sampler = None
        self.layer = None
        self._ns_window = None
        self._ns_view = None
        self._staging_texture = None
        self._staging_size = (0, 0)
        self._exclusive_fullscreen = False
        self._windowed_rect = (40, 40, int(config.window_width), int(config.window_height))
        self._target_monitor = None
        self._fps_frames = 0
        self._fps_started = time.perf_counter()
        self._last_drawable_size = (0, 0)
        # Deferred shader-warp state (rgb+depth present path).
        self._warp_pipeline = None
        self._warp_color_tex = None
        self._warp_depth_tex = None
        self._warp_color_size = (0, 0)
        self._warp_depth_size = (0, 0)
        # Warp calibration knobs (v2.5 parity): IPD in UV units and the raw
        # depth-strength ratio fed to the shader's shift formula.
        self._warp_ipd_uv = float(os.environ.get("D2S_METAL_WARP_IPD", "0.064") or 0.064)
        self._warp_depth_ratio = float(
            os.environ.get("D2S_METAL_WARP_DEPTH_STRENGTH", "4.0") or 4.0
        )
        self._warp_convergence = float(
            os.environ.get("D2S_METAL_WARP_CONVERGENCE", "0.0") or 0.0
        )

    # ── setup ──

    def initialize(self) -> None:
        if sys.platform != "darwin":
            raise RuntimeError("MetalLocalViewer is macOS-only")
        try:
            import Metal
            import Quartz
            from AppKit import NSScreen  # noqa: F401
        except Exception as exc:
            raise RuntimeError(f"PyObjC Metal frameworks unavailable: {exc}") from exc

        import glfw

        self.glfw = glfw
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed for Metal local viewer")
        # Same hints as the Vulkan path minus Windows-only tricks.
        glfw.default_window_hints()
        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
        glfw.window_hint(glfw.AUTO_ICONIFY, glfw.FALSE)
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
        glfw.window_hint(glfw.DECORATED, glfw.TRUE)
        glfw.window_hint(glfw.FLOATING, glfw.FALSE)
        glfw.window_hint(glfw.FOCUS_ON_SHOW, glfw.TRUE)

        monitors = glfw.get_monitors() or []
        monitor = (
            monitors[resolve_glfw_monitor_index(self.config.monitor_index, glfw)]
            if monitors
            else None
        )
        self._target_monitor = monitor
        width, height = int(self.config.window_width), int(self.config.window_height)
        x, y = 40, 40
        mx = my = 0
        if monitor is not None:
            mx, my = glfw.get_monitor_pos(monitor)
            x, y = int(mx) + 40, int(my) + 40
            self._windowed_rect = (mx + 40, my + 40, width, height)
        exclusive_fullscreen = bool(self.config.fullscreen and monitor is not None)
        if exclusive_fullscreen:
            mode = glfw.get_video_mode(monitor)
            width, height = int(mode.size.width), int(mode.size.height)
            x, y = int(mx), int(my)
            glfw.window_hint(glfw.DECORATED, glfw.FALSE)

        self.window = glfw.create_window(width, height, self.config.title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("could not create Metal local viewer window")
        if exclusive_fullscreen:
            # macOS: promote the borderless window to a real monitor fullscreen.
            mode = glfw.get_video_mode(monitor)
            glfw.set_window_monitor(
                self.window,
                monitor,
                int(mx),
                int(my),
                int(mode.size.width),
                int(mode.size.height),
                int(mode.refresh_rate),
            )
            glfw.show_window(self.window)
            self._exclusive_fullscreen = True
        glfw.set_window_pos(self.window, x, y)
        glfw.set_key_callback(self.window, self._on_key)

        self._attach_metal_layer()
        self._build_pipeline()

        if exclusive_fullscreen:
            mode = glfw.get_video_mode(self._target_monitor)
            print(
                "[MetalLocalViewer] Display fullscreen active on 2D display: "
                f"{int(mode.size.width)}x{int(mode.size.height)}@{int(mode.refresh_rate)}Hz "
                "(Alt+Enter)",
                flush=True,
            )

    def _glfw_cocoa_window(self):
        if not hasattr(self.glfw, "get_cocoa_window"):
            raise RuntimeError("GLFW Python package lacks get_cocoa_window")
        ptr = self.glfw.get_cocoa_window(self.window)
        if not ptr:
            raise RuntimeError("glfwGetCocoaWindow returned null")
        import objc

        return objc.objc_object(c_void_p=ptr)

    def _attach_metal_layer(self) -> None:
        ns_window = self._glfw_cocoa_window()
        view = ns_window.contentView()
        layer = Quartz.CAMetalLayer.layer()
        self.device = Metal.MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise RuntimeError("No Metal device available")
        self.queue = self.device.newCommandQueue()
        layer.setDevice_(self.device)
        layer.setPixelFormat_(Metal.MTLPixelFormatBGRA8Unorm)
        layer.setFramebufferOnly_(True)
        layer.setContentsScale_(ns_window.backingScaleFactor())
        if hasattr(layer, "setDisplaySyncEnabled_"):
            layer.setDisplaySyncEnabled_(bool(self.config.vsync))
        if hasattr(layer, "setMaximumDrawableCount_"):
            layer.setMaximumDrawableCount_(3)
        if hasattr(layer, "setOpaque_"):
            layer.setOpaque_(True)
        view.setWantsLayer_(True)
        view.setLayer_(layer)
        self._ns_window = ns_window
        self._ns_view = view
        self.layer = layer

    def _build_pipeline(self) -> None:
        library, error = self.device.newLibraryWithSource_options_error_(
            METAL_SHADER, None, None
        )
        if library is None:
            raise RuntimeError(f"Metal shader compile failed: {error}")
        desc = Metal.MTLRenderPipelineDescriptor.alloc().init()
        desc.setVertexFunction_(library.newFunctionWithName_("quad_vertex"))
        desc.setFragmentFunction_(library.newFunctionWithName_("quad_fragment"))
        attachment = desc.colorAttachments().objectAtIndexedSubscript_(0)
        attachment.setPixelFormat_(Metal.MTLPixelFormatBGRA8Unorm)
        self.pipeline, error = self.device.newRenderPipelineStateWithDescriptor_error_(
            desc, None
        )
        if self.pipeline is None:
            raise RuntimeError(f"Metal render pipeline failed: {error}")

        sampler_desc = Metal.MTLSamplerDescriptor.alloc().init()
        sampler_desc.setMinFilter_(Metal.MTLSamplerMinMagFilterLinear)
        sampler_desc.setMagFilter_(Metal.MTLSamplerMinMagFilterLinear)
        sampler_desc.setSAddressMode_(Metal.MTLSamplerAddressModeClampToEdge)
        sampler_desc.setTAddressMode_(Metal.MTLSamplerAddressModeClampToEdge)
        self.sampler = self.device.newSamplerStateWithDescriptor_(sampler_desc)

        # Warp pipeline is optional: if it fails to compile, present() falls
        # back to the packed-SBS quad path.
        warp_library, warp_error = self.device.newLibraryWithSource_options_error_(
            WARP_SHADER, None, None
        )
        if warp_library is not None:
            warp_desc = Metal.MTLRenderPipelineDescriptor.alloc().init()
            warp_desc.setVertexFunction_(warp_library.newFunctionWithName_("vertex_main"))
            warp_desc.setFragmentFunction_(warp_library.newFunctionWithName_("fragment_main"))
            warp_attachment = warp_desc.colorAttachments().objectAtIndexedSubscript_(0)
            warp_attachment.setPixelFormat_(Metal.MTLPixelFormatBGRA8Unorm)
            self._warp_pipeline, warp_pipeline_error = (
                self.device.newRenderPipelineStateWithDescriptor_error_(warp_desc, None)
            )
            if self._warp_pipeline is None:
                print(
                    f"[MetalLocalViewer] Warp pipeline unavailable ({warp_pipeline_error}); "
                    "using packed-SBS path",
                    flush=True,
                )
        else:
            print(
                f"[MetalLocalViewer] Warp shader compile failed ({warp_error}); "
                "using packed-SBS path",
                flush=True,
            )

    # ── staging texture ──

    def _ensure_staging_texture(self, width: int, height: int) -> Any:
        if self._staging_texture is not None and self._staging_size == (width, height):
            return self._staging_texture
        desc = Metal.MTLTextureDescriptor.texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
            Metal.MTLPixelFormatRGBA8Unorm, width, height, False
        )
        desc.setUsage_(Metal.MTLTextureUsageShaderRead)
        desc.setStorageMode_(Metal.MTLStorageModeManaged)
        self._staging_texture = self.device.newTextureWithDescriptor_(desc)
        self._staging_size = (width, height)
        return self._staging_texture

    # ── frame flow ──

    def present(self, frame: Any) -> None:
        if self.window is None or not self._initialized_flag():
            raise RuntimeError("Metal local viewer has not been initialized")
        self.poll_events()

        # Deferred shader-warp fast path: the runtime shipped rgb+depth and
        # the fragment shader does the stereo warp at draw time.
        rgb = getattr(frame, "viewer_rgb", None)
        depth = getattr(frame, "viewer_depth", None)
        if (
            rgb is not None
            and depth is not None
            and self._warp_pipeline is not None
            and os.environ.get("D2S_METAL_SHADER_WARP", "0") == "1"
        ):
            self._present_warp(rgb, depth, bgra=getattr(frame, "viewer_bgra", None))
            return

        # Runtime-packed host frame: pure memcpy, no device sync.
        host_np = getattr(frame, "viewer_frame_np", None)
        if host_np is not None:
            pixels, width, height = host_np
            pack_started = time.perf_counter()
        else:
            # Fast path: accelerator tensors are quantized/packed to HWC RGBA8
            # on their own device and handed to replaceRegion as a buffer view
            # (no numpy layout math, no extra tobytes copy on the host).
            pack_started = time.perf_counter()
            packed = pack_frame_to_rgba8(frame)
            if packed is not None:
                pixels, width, height = packed
            else:
                pixels, width, height = frame_to_rgba_bytes(frame)
        if self.config.on_breakdown_add_time is not None:
            self.config.on_breakdown_add_time(
                "local_present_pack", time.perf_counter() - pack_started
            )
        texture = self._ensure_staging_texture(width, height)

        region = Metal.MTLRegionMake2D(0, 0, width, height)
        texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow_(
            region, 0, pixels, width * 4
        )

        drawable = self.layer.nextDrawable()
        if drawable is None:
            return

        fb_w, fb_h = self.glfw.get_framebuffer_size(self.window)
        scale = self._ns_window.backingScaleFactor()
        dw, dh = int(fb_w * scale), int(fb_h * scale)
        dw, dh = max(1, dw), max(1, dh)
        if self._last_drawable_size != (dw, dh):
            self.layer.setDrawableSize_((dw, dh))
            self._last_drawable_size = (dw, dh)

        # Aspect-fit the SBS texture inside the drawable (letterbox black).
        sx = sy = s = min(dw / width, dh / height)
        su = min(1.0, (width * s) / dw)
        sv = min(1.0, (height * s) / dh)
        off_u = (1.0 - su) / 2.0
        off_v = (1.0 - sv) / 2.0

        command_buffer = self.queue.commandBuffer()
        pass_desc = Metal.MTLRenderPassDescriptor.renderPassDescriptor()
        color = pass_desc.colorAttachments().objectAtIndexedSubscript_(0)
        color.setTexture_(drawable.texture())
        color.setLoadAction_(Metal.MTLLoadActionClear)
        color.setStoreAction_(Metal.MTLStoreActionStore)
        color.setClearColor_(Metal.MTLClearColorMake(0.0, 0.0, 0.0, 1.0))

        encoder = command_buffer.renderCommandEncoderWithDescriptor_(pass_desc)
        encoder.setRenderPipelineState_(self.pipeline)
        encoder.setFragmentTexture_atIndex_(texture, 0)
        encoder.setFragmentSamplerState_atIndex_(self.sampler, 0)
        encoder.setFragmentBytes_length_atIndex_(
            # PyObjC requires a buffer-protocol object for `const void*`;
            # a plain tuple raises "too few values (4) expecting at least 16".
            struct.pack("4f", su, sv, off_u, off_v),
            16,
            0,
        )
        encoder.drawPrimitives_vertexStart_vertexCount_(
            Metal.MTLPrimitiveTypeTriangleStrip, 0, 4
        )
        encoder.endEncoding()
        # PyObjC selector: `presentDrawable:` maps to presentDrawable_.
        command_buffer.presentDrawable_(drawable)
        command_buffer.commit()

        now = time.perf_counter()
        self._fps_frames += 1
        fps = present_fps_if_due(self._fps_frames, now - self._fps_started)
        if fps is not None:
            self._report_present_fps(fps, self._fps_frames)
            self._fps_frames = 0
            self._fps_started = now

    def _initialized_flag(self) -> bool:
        return self.pipeline is not None

    # ── deferred shader-warp present (rgb + depth) ──

    def _ensure_warp_textures(
        self,
        color_size: tuple[int, int],
        depth_size: tuple[int, int],
        *,
        depth_u8: bool = False,
    ) -> None:
        self._warp_depth_format = (
            Metal.MTLPixelFormatR8Unorm if depth_u8 else Metal.MTLPixelFormatR32Float
        )
        if self._warp_color_size != color_size:
            self._warp_color_tex = self._make_texture(
                color_size[0], color_size[1], Metal.MTLPixelFormatRGBA8Unorm
            )
            self._warp_color_size = color_size
        fmt = self._warp_depth_format or Metal.MTLPixelFormatR32Float
        if self._warp_depth_size != depth_size or fmt != getattr(
            self._warp_depth_tex, "pixelFormat", lambda: None
        )():
            self._warp_depth_tex = self._make_texture(
                depth_size[0], depth_size[1], fmt
            )
            self._warp_depth_size = depth_size

    def _make_texture(self, width: int, height: int, pixel_format: Any) -> Any:
        desc = (
            Metal.MTLTextureDescriptor
            .texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
                pixel_format, max(1, int(width)), max(1, int(height)), False
            )
        )
        desc.setUsage_(Metal.MTLTextureUsageShaderRead)
        desc.setStorageMode_(Metal.MTLStorageModeManaged)
        return self.device.newTextureWithDescriptor_(desc)

    @staticmethod
    def _compute_render_size(max_w: float, max_h: float, src_w: float, src_h: float) -> tuple[int, int]:
        if src_w <= 0 or src_h <= 0:
            return 0, 0
        scale = min(max_w / src_w, max_h / src_h)
        return max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))

    def _warp_viewports(
        self, win_w: int, win_h: int, tex_w: int, tex_h: int
    ) -> list[tuple[tuple[int, int, int, int], float]]:
        """Eye viewports + eye offsets, ported from v2.5 _stereo_viewports."""
        ipd_half = self._warp_ipd_uv / 2.0
        mode = str(getattr(self.config, "display_mode", "Half-SBS"))
        fill = bool(getattr(self.config, "fill_16_9", True))

        def sbs_pair(src_w: float, src_h: float, half_w: float) -> list[tuple[tuple[int, int, int, int], float]]:
            render_w, render_h = self._compute_render_size(half_w, win_h, src_w, src_h)
            center_y = win_h / 2.0
            return [
                ((int(win_w / 4.0 - render_w / 2), int(center_y - render_h / 2), render_w, render_h), -ipd_half),
                ((int(3 * win_w / 4.0 - render_w / 2), int(center_y - render_h / 2), render_w, render_h), ipd_half),
            ]

        if mode == "Full-SBS":
            return sbs_pair(tex_w, tex_h, win_w / 2.0) if fill else [
                ((0, 0, win_w // 2, win_h), -ipd_half),
                ((win_w // 2, 0, win_w // 2, win_h), ipd_half),
            ]
        if mode in ("Half-SBS", ""):
            return sbs_pair(tex_w / 2.0, tex_h, win_w / 2.0)
        if mode == "Full-TAB":
            render_w, render_h = self._compute_render_size(win_w, win_h / 2.0, tex_w, tex_h)
            return [
                ((int(win_w / 2.0 - render_w / 2), int(win_h / 4.0 - render_h / 2), render_w, render_h), -ipd_half),
                ((int(win_w / 2.0 - render_w / 2), int(3 * win_h / 4.0 - render_h / 2), render_w, render_h), ipd_half),
            ]
        if mode == "Half-TAB":
            render_w, render_h = self._compute_render_size(win_w, win_h / 2.0, tex_w, tex_h / 2.0)
            return [
                ((int(win_w / 2.0 - render_w / 2), int(win_h / 4.0 - render_h / 2), render_w, render_h), -ipd_half),
                ((int(win_w / 2.0 - render_w / 2), int(3 * win_h / 4.0 - render_h / 2), render_w, render_h), ipd_half),
            ]
        # Mono-ish modes (original / depth map / anaglyph / interleaved).
        render_w, render_h = self._compute_render_size(win_w, win_h, tex_w, tex_h)
        viewport = (int((win_w - render_w) / 2), int((win_h - render_h) / 2), render_w, render_h)
        return [(viewport, ipd_half)]

    @staticmethod
    def _warp_mode_id(display_mode: str) -> int:
        # Shader mode ids ported verbatim from v2.5 _mode_id.
        if display_mode == "Depth Map":
            return 3
        if display_mode == "Anaglyph":
            return 4
        if display_mode == "Interleaved":
            return 5
        if display_mode == "Interleaved-V":
            return 6
        if display_mode in ("Half-TAB", "Full-TAB"):
            return 2
        return 7  # SBS warp (per-eye draws with explicit eye offsets)

    def _present_warp(self, rgb: Any, depth: Any, bgra: Any = None) -> None:
        import torch  # local: module stays importable without torch

        # Zero-copy fast path: sample the captured SCK BGRA texture directly
        # (IOSurface-backed, never touches CPU); release it once the GPU is
        # done. Fallback: device-pack the preprocessed tensor.
        zc_texture = None
        if bgra is not None:
            try:
                zc_texture = bgra.mtl_texture()
            except Exception:
                zc_texture = None
            if zc_texture is None:
                try:
                    bgra.release()
                except Exception:
                    pass

        def _release_bgra(_cb=None):
            try:
                bgra.release()
            except Exception:
                pass

        if zc_texture is not None:
            width = int(zc_texture.width())
            height = int(zc_texture.height())
        else:
            packed = pack_frame_to_rgba8(rgb)
            if packed is None:
                packed = pack_frame_to_rgba8(rgb.detach().float().clamp(0.0, 1.0))
            if packed is None:
                raise RuntimeError("shader-warp path requires an accelerator rgb tensor")
            rgba, width, height = packed

        # Numpy-free depth upload: pull the (already contiguous) CPU tensor
        # and hand pyobjc its raw bytes via the buffer protocol. bytes() on a
        # uint8/fp32 CPU tensor is a single flat copy — no numpy view math.
        depth_t = depth.detach()
        if depth_t.dtype != torch.uint8:
            depth_t = depth_t.float()
        depth_t = depth_t.contiguous().cpu().reshape(-1, depth_t.shape[-1])
        dh, dw = int(depth_t.shape[0]), int(depth_t.shape[1])
        depth_is_u8 = depth_t.dtype == torch.uint8

        self._ensure_warp_textures(
            (width, height), (dw, dh), depth_u8=depth_is_u8
        )
        if zc_texture is not None:
            color_tex = zc_texture
        else:
            color_region = Metal.MTLRegionMake2D(0, 0, width, height)
            self._warp_color_tex.replaceRegion_mipmapLevel_withBytes_bytesPerRow_(
                color_region, 0, rgba, width * 4
            )
            color_tex = self._warp_color_tex
        # ctypes.string_at: C-speed raw copy of the tensor's bytes.
        # bytes(tensor) is NOT viable — torch.__bytes__ loops per element
        # (~400ms for 2MB!); numpy is intentionally avoided here per leg-2.
        import ctypes

        flat = depth_t.reshape(-1)
        depth_payload = ctypes.string_at(flat.data_ptr(), flat.numel())
        depth_bpr = dw * (1 if depth_is_u8 else 4)
        depth_region = Metal.MTLRegionMake2D(0, 0, dw, dh)
        self._warp_depth_tex.replaceRegion_mipmapLevel_withBytes_bytesPerRow_(
            depth_region, 0, depth_payload, depth_bpr
        )
        if os.environ.get("D2S_PREP_TRACE") == "1" and not getattr(
            self, "_warp_trace_once", False
        ):
            self._warp_trace_once = True
            print(
                f"[warp-trace] u8={depth_is_u8} {dw}x{dh} bpr={depth_bpr} "
                f"zcolor={zc_texture is not None} "
                f"depth_fmt={self._warp_depth_format}",
                flush=True,
            )

        drawable = self.layer.nextDrawable()
        if drawable is None:
            if bgra is not None:
                _release_bgra()
            return
        fb_w, fb_h = self.glfw.get_framebuffer_size(self.window)
        scale = self._ns_window.backingScaleFactor()
        dw_px, dh_px = max(1, int(fb_w * scale)), max(1, int(fb_h * scale))
        if self._last_drawable_size != (dw_px, dh_px):
            self.layer.setDrawableSize_((dw_px, dh_px))
            self._last_drawable_size = (dw_px, dh_px)

        command_buffer = self.queue.commandBuffer()
        pass_desc = Metal.MTLRenderPassDescriptor.renderPassDescriptor()
        attachment = pass_desc.colorAttachments().objectAtIndexedSubscript_(0)
        attachment.setTexture_(drawable.texture())
        attachment.setLoadAction_(Metal.MTLLoadActionClear)
        attachment.setStoreAction_(Metal.MTLStoreActionStore)
        attachment.setClearColor_(Metal.MTLClearColorMake(0.0, 0.0, 0.0, 1.0))

        encoder = command_buffer.renderCommandEncoderWithDescriptor_(pass_desc)
        encoder.setRenderPipelineState_(self._warp_pipeline)
        encoder.setFragmentTexture_atIndex_(color_tex, 0)
        encoder.setFragmentTexture_atIndex_(self._warp_depth_tex, 1)

        mode_id = self._warp_mode_id(str(getattr(self.config, "display_mode", "Half-SBS")))
        viewports = self._warp_viewports(dw_px, dh_px, width, height)
        # Per-eye viewport draws always use warp mode 7 (the viewport selects
        # the eye region); single-viewport modes pass their own shader mode.
        mode_value = 7.0 if len(viewports) > 1 else float(mode_id)
        for viewport, eye_offset in viewports:
            # struct.pack: numpy-free uniform payload (12 floats, native LE
            # matches MSL float layout).
            payload = struct.pack(
                "<12f",
                eye_offset,
                0.1 * self._warp_depth_ratio,
                self._warp_convergence,
                0.0,
                float(viewport[0]),
                float(viewport[1]),
                float(viewport[2]),
                float(viewport[3]),
                mode_value,
                0.0,
                0.0,
                0.0,
            )
            encoder.setFragmentBytes_length_atIndex_(payload, len(payload), 0)
            encoder.drawPrimitives_vertexStart_vertexCount_(
                Metal.MTLPrimitiveTypeTriangleStrip, 0, 4
            )
        encoder.endEncoding()
        command_buffer.presentDrawable_(drawable)
        if bgra is not None:
            # Keep the capture buffer alive until the GPU finished sampling.
            try:
                command_buffer.addCompletedHandler_(_release_bgra)
            except Exception:
                _release_bgra()
        command_buffer.commit()

        now = time.perf_counter()
        self._fps_frames += 1
        fps = present_fps_if_due(self._fps_frames, now - self._fps_started)
        if fps is not None:
            self._report_present_fps(fps, self._fps_frames)
            self._fps_frames = 0
            self._fps_started = now

    def _report_present_fps(self, fps: float, frame_count: int) -> None:
        capture_target = (
            self.config.on_sbs_fps(fps, frame_count)
            if self.config.on_sbs_fps is not None
            else None
        )
        show_fps = (
            bool(self.config.show_fps_provider())
            if self.config.show_fps_provider is not None
            else self.config.show_fps
        )
        if not show_fps:
            return
        target_text = (
            f" capture_target={capture_target}" if capture_target is not None else ""
        )
        print(
            f"[MetalLocalViewer] Present FPS: {fps:.1f}{target_text}",
            flush=True,
        )

    # ── window events ──

    def _on_key(self, _window: Any, key: int, _scancode: int, action: int, mods: int) -> None:
        if is_exclusive_fullscreen_toggle(key, action, mods, self.glfw):
            self._set_exclusive_fullscreen(not self._exclusive_fullscreen)

    def _set_exclusive_fullscreen(self, enabled: bool) -> None:
        glfw = self.glfw
        if self.window is None or self._target_monitor is None:
            return
        if enabled:
            if self._exclusive_fullscreen:
                return
            x, y = glfw.get_window_pos(self.window)
            width, height = glfw.get_window_size(self.window)
            self._windowed_rect = (int(x), int(y), int(width), int(height))
            mode = glfw.get_video_mode(self._target_monitor)
            monitor_x, monitor_y = glfw.get_monitor_pos(self._target_monitor)
            glfw.set_window_attrib(self.window, glfw.DECORATED, glfw.FALSE)
            glfw.set_window_pos(self.window, int(monitor_x), int(monitor_y))
            glfw.set_window_size(
                self.window, int(mode.size.width), int(mode.size.height)
            )
            self._exclusive_fullscreen = True
        else:
            if not self._exclusive_fullscreen:
                return
            self._exclusive_fullscreen = False
            x, y, width, height = self._windowed_rect
            glfw.set_window_attrib(self.window, glfw.DECORATED, glfw.TRUE)
            glfw.set_window_pos(self.window, x, y)
            glfw.set_window_size(self.window, width, height)
        print(
            f"[MetalLocalViewer] Display fullscreen: {'on' if enabled else 'off'} "
            "(Alt+Enter)",
            flush=True,
        )

    def poll_events(self) -> None:
        self.glfw.poll_events()
        if self.glfw.window_should_close(self.window):
            raise StopIteration

    def close(self) -> None:
        try:
            self._staging_texture = None
            self.pipeline = None
            self.queue = None
            self.device = None
        finally:
            if self.window is not None and self.glfw is not None:
                self.glfw.destroy_window(self.window)
            if self.glfw is not None and self.config.manage_glfw_lifecycle:
                self.glfw.terminate()
            self.window = None


def run_metal_local_viewer(*, runtime_q: Any, shutdown_event: Any, config: Any) -> None:
    """Metal variant of run_vulkan_local_viewer (macOS Local Viewer)."""
    viewer: MetalLocalViewer | None = None
    preview_logged = False
    try:
        while not shutdown_event.is_set():
            try:
                result, _started = runtime_q.get(timeout=0.05)
            except queue.Empty:
                if viewer is not None:
                    viewer.poll_events()
                continue
            if config.on_breakdown_inc is not None:
                config.on_breakdown_inc("viewer_get", 1)
            while True:
                try:
                    result, _started = runtime_q.get_nowait()
                except queue.Empty:
                    break
                if config.on_breakdown_inc is not None:
                    config.on_breakdown_inc("viewer_get", 1)
                    config.on_breakdown_inc("viewer_drop", 1)
            # Deferred warp: pass the whole result so present() sees
            # viewer_rgb/viewer_depth; host-packed frames likewise; otherwise
            # the packed SBS tensor.
            frame = (
                result
                if (
                    getattr(result, "viewer_rgb", None) is not None
                    or getattr(result, "viewer_frame_np", None) is not None
                )
                else getattr(result, "sbs", None)
            )
            if frame is None:
                continue
            if viewer is None:
                viewer = MetalLocalViewer(config)
                viewer.initialize()
                print("[MetalLocalViewer] Metal local window initialized", flush=True)
            if getattr(config, "window_preview", False) and not preview_logged:
                preview_logged = True
                print(
                    "[MetalLocalViewer] Debug preview window not supported on the "
                    "Metal path yet; main output only",
                    flush=True,
                )
            present_started = time.perf_counter()
            viewer.present(frame)
            if config.on_breakdown_add_time is not None:
                config.on_breakdown_add_time(
                    "local_present", time.perf_counter() - present_started
                )
            if config.on_breakdown_inc is not None:
                config.on_breakdown_inc("local_presented_frame", 1)
    except StopIteration:
        shutdown_event.set()
    finally:
        if viewer is not None:
            viewer.close()


def select_config_for_metal(config: Any) -> Any:
    """Metal viewer accepts the Vulkan config as-is; hook for future fields."""
    return replace(config) if False else config
