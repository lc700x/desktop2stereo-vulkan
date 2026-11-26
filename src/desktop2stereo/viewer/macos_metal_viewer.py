"""macOS Metal local viewer — zero-copy-ish present path (Darwin only).

Replaces the MoltenVK swapchain chain with a native GPU path:

    CoreML/Metal warp-pack -> shared MTLBuffer texture view
    -> CAMetalLayer textured-quad draw

Desktop capture stays GPU-only; local media playback may use a prepared host
RGBA8 frame because file sources do not provide a borrowed CoreML/Metal surface.

Interface matches VulkanLocalViewer so the runtime can switch viewers without
touching the pipeline: initialize() / present(frame) / poll_events() / close().
"""

from __future__ import annotations

import json
import math
import os
import queue
import struct
import sys
import time
from dataclasses import replace
from typing import Any

from utils.display_info import resolve_glfw_monitor_index
from media_ui import (
    MEDIA_BAR_TEXTURE_SIZE,
    MEDIA_BROWSER_TEXTURE_SIZE,
    MEDIA_POINTER_IDLE_SECONDS,
    MediaHoverAnimation,
    build_media_bar_rgba,
    build_media_browser_rgba,
    build_pointer_rgba,
    media_bar_hit,
    media_bar_rect_for_eye,
    media_bar_value_fraction,
    media_browser_hit,
    media_browser_font_scale,
    media_browser_rect_for_eye,
    media_browser_scroll_metrics,
    media_browser_scrollbar_offset,
    media_control_enabled,
)

if sys.platform == "darwin":
    import Metal
    import Quartz
from viewer.vulkan_local_viewer import (
    FPS_REPORT_INTERVAL_SECONDS,
    VulkanLocalViewerConfig,
    fit_rect,
    frame_to_rgba_bytes,
    _has_gpu_present_source,
    is_exclusive_fullscreen_toggle,
    is_media_capture_mode,
    pack_frame_to_rgba8,
    present_fps_if_due,
)
from viewer.direct_sink import native_pack_error_is_recoverable
from viewer.vulkan_fps_overlay import (
    build_fps_panel_rgba,
    overlay_rect_for_eye,
    overlay_size_for_display_mode,
)
from viewer.window_control import (
    pin_macos_viewer_window,
    set_window_mouse_passthrough,
    update_native_cursor_suppression,
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

static float4 blend_cursor(
    float4 color,
    float2 pixel,
    float4 rect,
    texture2d<float> cursorTex,
    sampler smp
) {
    if (rect.z <= 0.0 || rect.w <= 0.0) {
        return color;
    }
    float2 cursor_uv = (pixel - rect.xy) / rect.zw;
    if (all(cursor_uv >= 0.0) && all(cursor_uv < 1.0)) {
        float4 cursor = cursorTex.sample(smp, cursor_uv);
        color.rgb = mix(color.rgb, cursor.rgb, cursor.a);
    }
    return color;
}

static float4 blend_overlay(
    float4 color,
    float2 pixel,
    float4 rect,
    texture2d<float> overlayTex,
    sampler smp
) {
    if (rect.z <= 0.0 || rect.w <= 0.0) {
        return color;
    }
    float2 overlay_uv = (pixel - rect.xy) / rect.zw;
    if (all(overlay_uv >= 0.0) && all(overlay_uv < 1.0)) {
        float4 overlay = overlayTex.sample(smp, overlay_uv);
        color.rgb = mix(color.rgb, overlay.rgb, overlay.a);
    }
    return color;
}

fragment float4 quad_fragment(
    VOut in [[stage_in]],
    texture2d<float> tex [[texture(0)]],
    texture2d<float> cursorTex [[texture(1)]],
    texture2d<float> fpsTex [[texture(2)]],
    texture2d<float> mediaTex [[texture(3)]],
    sampler smp [[sampler(0)]],
    constant float4 &uv_xform [[buffer(0)]],
    constant float4 &cursor_rect_left [[buffer(1)]],
    constant float4 &cursor_rect_right [[buffer(2)]],
    constant float4 &fps_rect_left [[buffer(3)]],
    constant float4 &fps_rect_right [[buffer(4)]],
    constant float4 &media_rect_left [[buffer(5)]],
    constant float4 &media_rect_right [[buffer(6)]]
) {
    float2 uv = (in.uv - uv_xform.zw) / max(uv_xform.xy, float2(1e-6));
    if (any(uv < 0.0) || any(uv > 1.0)) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }
    float4 color = tex.sample(smp, uv);
    color = blend_overlay(color, in.position.xy, fps_rect_left, fpsTex, smp);
    color = blend_overlay(color, in.position.xy, fps_rect_right, fpsTex, smp);
    color = blend_overlay(color, in.position.xy, media_rect_left, mediaTex, smp);
    color = blend_overlay(color, in.position.xy, media_rect_right, mediaTex, smp);
    color = blend_cursor(color, in.position.xy, cursor_rect_left, cursorTex, smp);
    color = blend_cursor(color, in.position.xy, cursor_rect_right, cursorTex, smp);
    return color;
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
    float4 cursorRectLeft;
    float4 cursorRectRight;
    float4 fpsRect;
    float4 mediaRect;
    // x/y are the raw Core ML depth range; z selects raw-depth sampling.
    float4 rawNormalize;
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

static float4 blend_cursor(
    float4 color,
    float2 pixel,
    float4 rect,
    texture2d<float> cursorTex,
    sampler smp
) {
    if (rect.z <= 0.0 || rect.w <= 0.0) {
        return color;
    }
    float2 cursor_uv = (pixel - rect.xy) / rect.zw;
    if (all(cursor_uv >= 0.0) && all(cursor_uv < 1.0)) {
        float4 cursor = cursorTex.sample(smp, cursor_uv);
        color.rgb = mix(color.rgb, cursor.rgb, cursor.a);
    }
    return color;
}

static float4 blend_overlay(
    float4 color,
    float2 pixel,
    float4 rect,
    texture2d<float> overlayTex,
    sampler smp
) {
    if (rect.z <= 0.0 || rect.w <= 0.0) {
        return color;
    }
    float2 overlay_uv = (pixel - rect.xy) / rect.zw;
    if (all(overlay_uv >= 0.0) && all(overlay_uv < 1.0)) {
        float4 overlay = overlayTex.sample(smp, overlay_uv);
        color.rgb = mix(color.rgb, overlay.rgb, overlay.a);
    }
    return color;
}

static float2 displaced_uv(float2 uv, float eye, texture2d<float> depthTex, sampler s, constant Uniforms& u) {
    // Depth is globally antialiased before upload, so one sample is enough.
    float d = depthTex.sample(s, uv).r;
    if (!isfinite(d)) d = 1.0;
    if (u.rawNormalize.z > 0.5) {
        d = clamp((d - u.rawNormalize.x) /
                  max(u.rawNormalize.y - u.rawNormalize.x, 1.0e-6), 0.0, 1.0);
    } else {
        d = clamp(d, 0.0, 1.0);
    }
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
    texture2d<float> cursorTex [[texture(2)]],
    texture2d<float> fpsTex [[texture(3)]],
    texture2d<float> mediaTex [[texture(4)]],
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

    float4 color;
    if (mode == 0) {
        color = colorTex.sample(s, uv);
    } else if (mode == 7) {
        color = colorTex.sample(s, displaced_uv(uv, u.eyeOffset, depthTex, s, u));
    } else if (mode == 3) {
        float d = depthTex.sample(s, uv).r;
        if (!isfinite(d)) d = 1.0;
        if (u.rawNormalize.z > 0.5) {
            d = (d - u.rawNormalize.x) /
                max(u.rawNormalize.y - u.rawNormalize.x, 1.0e-6);
        }
        d = clamp(d, 0.0, 1.0);
        color = float4(spectral_r_ultrafast(d), 1.0);
    } else if (mode == 4) {
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
    color = blend_overlay(color, frag, u.fpsRect, fpsTex, s);
    color = blend_overlay(color, frag, u.mediaRect, mediaTex, s);
    color = blend_cursor(color, frag, u.cursorRectLeft, cursorTex, s);
    color = blend_cursor(color, frag, u.cursorRectRight, cursorTex, s);
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
        self._capture_window_registered = False
        self._cursor_texture = None
        self._virtual_pointer_texture = None
        self._virtual_pointer_uv: tuple[float, float] | None = (
            (0.5, 0.5)
            if bool(getattr(config, "pointer_mapping_enabled", False))
            else None
        )
        self._blank_texture = None
        self._cursor_hide_enabled = False
        self._fps_overlay_texture = None
        self._fps_overlay_size = (0, 0)
        self._fps_overlay_visible = False
        self._media_overlay_texture = None
        self._media_overlay_size = (0, 0)
        self._media_overlay_key = None
        self._media_state: dict[str, Any] = {}
        self._media_state_read_at = 0.0
        self._media_browser_scroll = 0
        self._media_browser_scroll_drag = False
        self._media_browser_hover: str | None = None
        self._media_browser_rects: tuple[tuple[int, int, int, int], ...] = ()
        self._media_bar_hover: str | None = None
        self._media_hover_animation = MediaHoverAnimation()
        self._media_hover_hit: str | None = None
        self._media_hover_progress = 0.0
        self._media_bar_rects: tuple[tuple[int, int, int, int], ...] = ()
        self._media_bar_hit_uv: tuple[float, float] | None = None
        self._media_drag_action: str | None = None
        self._media_drag_eye: int | None = None
        self._media_drag_last_sent_at = 0.0
        self._media_panel_hovered = False
        self._last_eye_rects: tuple[tuple[int, int, int, int], ...] = ()
        self._media_last_cursor_position: tuple[float, float] | None = None
        self._media_last_activity = 0.0
        self._media_passthrough_active = False
        self._staging_texture = None
        self._staging_size = (0, 0)
        self._native_pack_size = (0, 0)
        self._native_pack_slots: list[dict[str, Any]] = []
        self._native_pack_cursor = 0
        self._exclusive_fullscreen = False
        self._windowed_rect = (40, 40, int(config.window_width), int(config.window_height))
        self._target_monitor = None
        self._fps_frames = 0
        self._fps_started = time.perf_counter()
        self._last_drawable_size = (0, 0)
        self._last_present_texture = None
        self._last_present_size = (0, 0)
        self._last_native_frame = None
        self._last_native_entry: dict[str, Any] | None = None
        self._display_refresh_hz = 60.0
        # Keep the Metal OSD metrics compatible with the Vulkan/Windows viewer.
        self._present_total = 0
        self._present_reused = 0
        self._content_fps = 0.0
        self._content_frame_count = 0
        self._reuse_ratio = 0.0
        self._present_window_start = time.perf_counter()
        self._last_latency_ms = 0.0
        self._latency_history: list[float] = []
        self._latency_history_max = 300
        self._latency_anchor = "poll"
        self._latency_stage_history: list[dict[str, float]] = []
        self._latency_stage_history_max = 300
        self._last_present_metrics: dict[str, float] = {}
        self._upstream_fps = 0.0
        self.upstream_fps_provider = None
        self.sys_load_provider = None
        self._load_snapshot: dict[str, Any] = {}
        self._gpu_util = None
        self._gpu_util_max = None
        # Deferred shader-warp state (rgb+depth present path).
        self._warp_pipeline = None
        self._warp_color_tex = None
        self._warp_depth_tex = None
        self._warp_color_size = (0, 0)
        self._warp_depth_size = (0, 0)
        self._drawable_pixel_format = None
        self._source_pixel_format = None
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
        # The SBS output must not activate its own Space when a source app
        # enters native fullscreen. It is mouse-transparent and is pinned by
        # AppKit instead.
        glfw.window_hint(glfw.FOCUS_ON_SHOW, glfw.FALSE)

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
            try:
                mode = glfw.get_video_mode(monitor)
                self._display_refresh_hz = max(20.0, float(mode.refresh_rate))
            except (AttributeError, TypeError, ValueError):
                self._display_refresh_hz = 60.0
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
            self._exclusive_fullscreen = True
        glfw.set_window_pos(self.window, x, y)
        pin_macos_viewer_window(
            self.window,
            fullscreen=exclusive_fullscreen,
            show_in_app_switcher=bool(
                getattr(self.config, "show_in_app_switcher", False)
            ),
        )
        # Match the Vulkan/Windows viewer: the first OSD interval starts after
        # the window, Metal layer, and pipelines are ready.
        self._fps_started = time.perf_counter()
        self._present_window_start = self._fps_started
        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor_pos)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_scroll_callback(self.window, self._on_scroll)

        self._attach_metal_layer()
        self._register_capture_window()
        self._build_pipeline()
        self._cursor_texture = self._new_cursor_texture(1, 1, b"\x00\x00\x00\x00")
        self._blank_texture = self._new_cursor_texture(1, 1, b"\x00\x00\x00\xff")
        self._fps_overlay_texture = self._new_cursor_texture(1, 1, b"\x00\x00\x00\x00")
        self._fps_overlay_size = (1, 1)
        self._media_overlay_texture = self._new_cursor_texture(1, 1, b"\x00\x00\x00\x00")
        self._media_overlay_size = (1, 1)
        if (
            bool(getattr(self.config, "pointer_mapping_enabled", False))
            and is_media_capture_mode(self.config.capture_mode)
        ):
            try:
                pointer_rgba = build_pointer_rgba(48)
                self._virtual_pointer_texture = self._new_cursor_texture(
                    48, 48, pointer_rgba.tobytes()
                )
            except Exception as exc:
                print(
                    "[MetalLocalViewer] Virtual pointer texture unavailable: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
        if self.config.cursor_passthrough:
            self._cursor_hide_enabled = set_window_mouse_passthrough(
                self.window,
                True,
                hide_cursor_overlay=bool(self.config.hide_cursor_overlay),
            )
            self._media_passthrough_active = True
        pin_macos_viewer_window(
            self.window,
            fullscreen=exclusive_fullscreen,
            show_in_app_switcher=bool(
                getattr(self.config, "show_in_app_switcher", False)
            ),
        )

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
        # SCK supplies display-referred sRGB bytes. Keep the render target and
        # staged textures unorm so Metal does not apply another transfer; tag
        # the layer so WindowServer can color-manage those bytes for the display.
        self._drawable_pixel_format = Metal.MTLPixelFormatBGRA8Unorm
        self._source_pixel_format = Metal.MTLPixelFormatRGBA8Unorm
        layer.setDevice_(self.device)
        layer.setPixelFormat_(self._drawable_pixel_format)
        layer.setColorspace_(Quartz.CGColorSpaceCreateWithName(Quartz.kCGColorSpaceSRGB))
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

    def _register_capture_window(self) -> None:
        callback = getattr(self.config, "on_capture_window_exclusion", None)
        if not callable(callback) or self._ns_window is None:
            return
        try:
            window_id = int(self._ns_window.windowNumber())
            self._capture_window_registered = bool(callback(window_id, True))
        except Exception as exc:
            raise RuntimeError(
                f"Could not register Cocoa window for capture exclusion: {exc}"
            ) from exc
        if not self._capture_window_registered:
            raise RuntimeError("ScreenCaptureKit capture exclusion rejected the Cocoa window")

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
        attachment.setPixelFormat_(self._drawable_pixel_format)
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
            warp_attachment.setPixelFormat_(self._drawable_pixel_format)
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
            self._source_pixel_format, width, height, False
        )
        desc.setUsage_(Metal.MTLTextureUsageShaderRead)
        desc.setStorageMode_(Metal.MTLStorageModeManaged)
        self._staging_texture = self.device.newTextureWithDescriptor_(desc)
        self._staging_size = (width, height)
        return self._staging_texture

    def _new_cursor_texture(self, width: int, height: int, rgba: bytes) -> Any:
        texture = self._make_texture(width, height, self._source_pixel_format)
        region = Metal.MTLRegionMake2D(0, 0, width, height)
        texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow_(
            region, 0, rgba, width * 4
        )
        return texture

    def _show_fps_enabled(self) -> bool:
        provider = getattr(self.config, "show_fps_provider", None)
        if callable(provider):
            try:
                return bool(provider())
            except Exception:
                pass
        return bool(getattr(self.config, "show_fps", False))

    def _sync_fps_overlay_visibility(self) -> bool:
        """Read the live GUI setting and update the per-frame draw flag."""
        enabled = self._show_fps_enabled()
        self._fps_overlay_visible = bool(
            enabled and self._fps_overlay_texture is not None
        )
        return self._fps_overlay_visible

    def _read_media_state(self) -> dict[str, Any]:
        path = str(getattr(self.config, "media_state_path", "") or "").strip()
        session = str(getattr(self.config, "media_session_id", "") or "").strip()
        now = time.monotonic()
        state_cache = getattr(self, "_media_state", {})
        if not path or now - float(getattr(self, "_media_state_read_at", 0.0)) < 0.20:
            return state_cache
        self._media_state_read_at = now
        try:
            with open(path, "r", encoding="utf-8") as stream:
                state = json.load(stream)
        except (OSError, ValueError, UnicodeError):
            return state_cache
        if not isinstance(state, dict) or (session and state.get("session") != session):
            return state_cache
        if state != state_cache:
            if not state_cache:
                self._media_last_activity = now
            self._media_state = state
            self._media_overlay_key = None
        return state

    def _media_browser_visible(self) -> bool:
        state = self._read_media_state()
        return (
            str(state.get("mode", "")).casefold() == "browser"
            and bool(state.get("ready", True))
            and bool(state.get("browser_visible"))
        )

    def _media_bar_expanded(self) -> bool:
        last_activity = float(getattr(self, "_media_last_activity", 0.0) or 0.0)
        return (
            self._media_browser_visible()
            or time.monotonic() - last_activity <= MEDIA_POINTER_IDLE_SECONDS
        )

    @staticmethod
    def _media_browser_rect_for_eye(
        eye_rect: tuple[int, int, int, int], display_mode: str = ""
    ) -> tuple[int, int, int, int]:
        return media_browser_rect_for_eye(eye_rect, display_mode)

    def _media_overlay_rects(
        self, eye_rects: tuple[tuple[int, int, int, int], ...]
    ) -> tuple[tuple[float, float, float, float], ...]:
        if self._media_overlay_texture is None:
            return tuple((0.0, 0.0, 0.0, 0.0) for _ in eye_rects) or (
                (0.0, 0.0, 0.0, 0.0),
            )
        if self._media_browser_visible():
            display_mode = getattr(self.config, "display_mode", "")
            rects = tuple(
                self._media_browser_rect_for_eye(rect, display_mode)
                for rect in eye_rects
            )
            self._media_browser_rects = rects
            self._media_bar_rects = ()
        elif self._media_state:
            display_mode = str(getattr(self.config, "display_mode", ""))
            rects = tuple(
                media_bar_rect_for_eye(rect, display_mode)
                for rect in eye_rects
            )
            self._media_bar_rects = rects
            self._media_browser_rects = ()
        else:
            return tuple((0.0, 0.0, 0.0, 0.0) for _ in eye_rects) or (
                (0.0, 0.0, 0.0, 0.0),
            )
        return tuple(
            (
                float(left),
                float(top),
                float(right - left),
                float(bottom - top),
            )
            for left, top, right, bottom in rects
        ) or (
            (0.0, 0.0, 0.0, 0.0),
        )

    def _media_browser_font_scale(
        self, eye_rects: tuple[tuple[int, int, int, int], ...] | None = None
    ) -> float:
        rects = eye_rects or getattr(self, "_last_eye_rects", ())
        if not rects:
            return 1.0
        panel = self._media_browser_rect_for_eye(
            rects[0], getattr(self.config, "display_mode", "")
        )
        return media_browser_font_scale(panel[3] - panel[1])

    def _refresh_media_overlay(
        self, eye_rects: tuple[tuple[int, int, int, int], ...] | None = None
    ) -> None:
        state = self._read_media_state()
        if not state:
            return
        hover = self._media_browser_hover if self._media_browser_visible() else self._media_bar_hover
        animator = getattr(self, "_media_hover_animation", None)
        if animator is None:
            self._media_hover_hit = hover
            self._media_hover_progress = 1.0 if hover else 0.0
        else:
            self._media_hover_hit, self._media_hover_progress = animator.update(
                hover, time.perf_counter()
            )
        lang = "CN" if str(getattr(self.config, "locale", "EN")).upper() == "CN" else "EN"
        if self._media_browser_visible():
            key = (
                "browser",
                state.get("browser_revision"),
                state.get("browser_path"),
                state.get("browser_selection"),
                state.get("browser_error"),
                self._media_browser_scroll,
                self._media_hover_hit,
                round(self._media_hover_progress * 8) / 8,
                lang,
                round(self._media_browser_font_scale(eye_rects), 3),
            )
            if key == self._media_overlay_key:
                return
            rgba = build_media_browser_rgba(
                state,
                width=MEDIA_BROWSER_TEXTURE_SIZE[0],
                height=MEDIA_BROWSER_TEXTURE_SIZE[1],
                hover=self._media_hover_hit,
                hover_progress=self._media_hover_progress,
                scroll_offset=self._media_browser_scroll,
                lang=lang,
                font_scale=self._media_browser_font_scale(eye_rects),
            )
        else:
            key = (
                "bar",
                self._media_bar_expanded(),
                state.get("mode"),
                state.get("ready"),
                state.get("playing"),
                state.get("muted"),
                int(float(state.get("position", 0.0) or 0.0)),
                int(float(state.get("duration", 0.0) or 0.0)),
                round(float(state.get("volume", 1.0) or 0.0), 2),
                state.get("audio_enabled"),
                self._media_hover_hit,
                round(self._media_hover_progress * 8) / 8,
                lang,
            )
            if key == self._media_overlay_key:
                return
            rgba = build_media_bar_rgba(
                state,
                width=MEDIA_BAR_TEXTURE_SIZE[0],
                height=MEDIA_BAR_TEXTURE_SIZE[1],
                hover=self._media_hover_hit,
                hover_progress=self._media_hover_progress,
                lang=lang,
                collapsed=not self._media_bar_expanded(),
            )
        width, height = (int(value) for value in rgba.shape[1::-1])
        if self._media_overlay_texture is None or self._media_overlay_size != (width, height):
            self._media_overlay_texture = self._make_texture(
                width, height, self._source_pixel_format
            )
            self._media_overlay_size = (width, height)
        region = Metal.MTLRegionMake2D(0, 0, width, height)
        self._media_overlay_texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow_(
            region, 0, rgba.tobytes(), width * 4
        )
        self._media_overlay_key = key

    def _sync_media_passthrough(self) -> None:
        if not getattr(self.config, "cursor_passthrough", False) or self.window is None:
            return
        interactive = self._media_browser_visible() or self._media_panel_hovered or bool(
            self._media_drag_action
        )
        want_passthrough = not interactive
        if want_passthrough == getattr(self, "_media_passthrough_active", False):
            return
        set_window_mouse_passthrough(
            self.window,
            want_passthrough,
            hide_cursor_overlay=bool(
                want_passthrough
                and bool(getattr(self.config, "hide_cursor_overlay", False))
            ),
        )
        self._media_passthrough_active = want_passthrough

    def _media_bar_rect_for_eye(
        self, eye_rect: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        return media_bar_rect_for_eye(
            eye_rect, str(getattr(self.config, "display_mode", ""))
        )

    def _media_hit_from_point(self, x: float, y: float):
        panels = self._media_browser_rects if self._media_browser_visible() else self._media_bar_rects
        for index, panel in enumerate(panels):
            left, top, right, bottom = panel
            if not (left <= x < right and top <= y < bottom):
                continue
            u = (x - left) / max(1, right - left)
            v = (y - top) / max(1, bottom - top)
            if self._media_browser_visible():
                hit = media_browser_hit(
                    u,
                    v,
                    self._media_state,
                    scroll_offset=self._media_browser_scroll,
                )
                return ("browser", index, u, v, hit)
            hit = media_bar_hit(
                u,
                v,
                self._media_state,
                collapsed=not self._media_bar_expanded(),
            )
            return ("bar", index, u, v, hit)
        return None

    def _media_bar_hit_from_cursor(self, xpos: float, ypos: float):
        try:
            client_w, client_h = self.glfw.get_window_size(self.window)
            fb_w, fb_h = self.glfw.get_framebuffer_size(self.window)
        except Exception:
            return None
        x = float(xpos) * max(1, int(fb_w)) / max(1, int(client_w))
        y = float(ypos) * max(1, int(fb_h)) / max(1, int(client_h))
        virtual_uv = getattr(self, "_virtual_pointer_uv", None)
        if (
            bool(getattr(self.config, "pointer_mapping_enabled", False))
            and virtual_uv is not None
            and self._last_eye_rects
        ):
            left, top, right, bottom = self._last_eye_rects[0]
            u, v = virtual_uv
            x = left + u * (right - left)
            y = top + v * (bottom - top)
        return self._media_hit_from_point(x, y)

    def _dispatch_media_bar_hit(self, hit: str | None) -> None:
        if not hit:
            return
        if hit.startswith("seek:"):
            try:
                fraction = max(0.0, min(1.0, float(hit.split(":", 1)[1])))
            except (TypeError, ValueError):
                return
            duration = float(self._media_state.get("duration", 0.0) or 0.0)
            if duration > 0.0 and media_control_enabled("seek", self._media_state):
                self._media_state = {**self._media_state, "position": fraction * duration}
                self._media_action("seek", fraction * duration)
            return
        if hit.startswith("volume:"):
            try:
                volume = max(0.0, min(1.0, float(hit.split(":", 1)[1])))
            except (TypeError, ValueError):
                return
            if media_control_enabled("volume", self._media_state):
                self._media_state = {**self._media_state, "volume": volume}
                self._media_action("volume", volume)
            return
        if hit.startswith("seek_relative:"):
            try:
                value = float(hit.split(":", 1)[1])
            except (TypeError, ValueError):
                return
            if media_control_enabled("seek_relative", self._media_state):
                self._media_action("seek_relative", value)
            return
        if hit in {"browse", "browse_disc"}:
            self._media_action("browse", {"mode": "disc"} if hit == "browse_disc" else None)
        elif hit == "media_exit":
            self._media_browser_scroll = 0
            self._media_browser_scroll_drag = False
            self._media_action("media_exit")
        elif hit == "play" and media_control_enabled("play", self._media_state):
            was_playing = bool(self._media_state.get("playing", False))
            action = "pause" if was_playing else "play"
            if self._media_action(action):
                self._media_state = {**self._media_state, "playing": not was_playing}
                self._media_overlay_key = None
        elif hit in {"playlist_previous", "playlist_next"}:
            if media_control_enabled(hit, self._media_state):
                self._media_action(hit)
        elif hit == "mute" and media_control_enabled("mute", self._media_state):
            self._media_action("mute", not bool(self._media_state.get("muted", False)))

    def _update_media_drag(self, u: float) -> None:
        now = time.monotonic()
        if now - self._media_drag_last_sent_at < 0.05:
            return
        action = str(self._media_drag_action or "")
        fraction = media_bar_value_fraction(u, action=action)
        if fraction is None:
            return
        if action == "seek":
            duration = float(self._media_state.get("duration", 0.0) or 0.0)
            if duration <= 0.0 or not media_control_enabled("seek", self._media_state):
                return
            value = fraction * duration
            self._media_state = {**self._media_state, "position": value}
            self._media_action("seek", value)
        elif action == "volume":
            if not media_control_enabled("volume", self._media_state):
                return
            self._media_state = {**self._media_state, "volume": fraction}
            self._media_action("volume", fraction)
        else:
            return
        self._media_drag_last_sent_at = now
        self._media_overlay_key = None

    def _update_media_hover(self, xpos: float, ypos: float) -> None:
        hit = self._media_bar_hit_from_cursor(xpos, ypos)
        previous_hovered = self._media_panel_hovered
        self._media_panel_hovered = hit is not None
        previous_bar = self._media_bar_hover
        previous_browser = self._media_browser_hover
        if hit is None:
            self._media_bar_hover = None
            self._media_browser_hover = None
            if self._media_drag_action and self._media_drag_eye is not None:
                try:
                    client_w, client_h = self.glfw.get_window_size(self.window)
                    fb_w, fb_h = self.glfw.get_framebuffer_size(self.window)
                    x = float(xpos) * max(1, int(fb_w)) / max(1, int(client_w))
                    rect = self._media_bar_rects[self._media_drag_eye]
                    u = (x - rect[0]) / max(1, rect[2] - rect[0])
                    self._media_bar_hit_uv = (u, self._media_bar_hit_uv[1])
                    self._update_media_drag(u)
                except (IndexError, TypeError, ValueError):
                    pass
        elif hit[0] == "bar":
            self._media_bar_hover = hit[4]
            self._media_bar_hit_uv = (hit[2], hit[3])
            self._media_browser_hover = None
            if self._media_bar_hover == "wake":
                self._media_last_activity = time.monotonic()
                self._media_panel_hovered = True
        else:
            self._media_bar_hover = None
            browser_hit = hit[4]
            if (
                self._media_browser_scroll_drag
                and browser_hit is not None
                and browser_hit[0] == "browser_scroll"
            ):
                self._media_browser_scroll = int(browser_hit[1])
                self._media_overlay_key = None
            self._media_browser_hover = (
                f"{browser_hit[0]}:{browser_hit[1]}"
                if browser_hit is not None
                else None
            )
        if (
            previous_hovered != self._media_panel_hovered
            or previous_bar != self._media_bar_hover
            or previous_browser != self._media_browser_hover
        ):
            self._media_overlay_key = None
        if self._media_drag_action and self._media_panel_hovered and hit and hit[0] == "bar":
            self._update_media_drag(hit[2])
        self._sync_media_passthrough()

    def _media_action(self, action: str, value: Any = None) -> bool:
        callback = getattr(self.config, "media_action_callback", None)
        if not callable(callback):
            return False
        return bool(callback(action, value))

    def _dispatch_media_hit(self, hit: tuple[str, Any] | None) -> None:
        if hit is None:
            return
        action, value = hit
        if action == "viewer_stop":
            callback = getattr(self.config, "on_stop_requested", None)
            if callable(callback):
                callback()
        elif action == "entry":
            path = str(value or "")
            self._media_state = {
                **self._media_state,
                "browser_selection": path,
                "browser_revision": int(
                    self._media_state.get("browser_revision", 0) or 0
                ) + 1,
            }
            self._media_overlay_key = None
            self._media_action("browse_open", {"path": path})
        elif action == "recent":
            self._media_action("browse_open", {"path": value})
        elif action == "mount":
            self._media_action("browse", value)
        elif action in {"browse_open", "playlist_add", "playlist_select", "playlist_remove", "recent_remove"}:
            self._media_action(action, {"path": value} if value else None)
        elif action == "browser_scroll":
            self._media_browser_scroll = int(value or 0)
            self._media_overlay_key = None
        elif action == "browser_sort":
            self._media_action(
                "browser_sort",
                {"descending": self._media_state.get("browser_sort_descending") is not True},
            )
        else:
            self._media_action(action, value)

    def _on_cursor_pos(self, _window: Any, xpos: float, ypos: float) -> None:
        position = (float(xpos), float(ypos))
        moved = position != self._media_last_cursor_position
        self._media_last_cursor_position = position
        if moved:
            self._media_last_activity = time.monotonic()
        self._update_virtual_pointer_uv(xpos, ypos)
        self._update_media_hover(xpos, ypos)
        self._refresh_media_overlay()

    def _poll_virtual_pointer(self) -> None:
        """Poll coordinates while the SBS window ignores mouse events."""
        try:
            xpos, ypos = self.glfw.get_cursor_pos(self.window)
            client_w, client_h = self.glfw.get_window_size(self.window)
        except Exception:
            return
        self._on_cursor_pos(self.window, float(xpos), float(ypos))

    def _update_virtual_pointer_uv(self, xpos: float, ypos: float) -> None:
        """Map the complete packed-frame ray to one source coordinate for both eyes."""
        if not bool(getattr(self.config, "pointer_mapping_enabled", False)):
            return
        eye_rects = getattr(self, "_last_eye_rects", ())
        if not eye_rects:
            return
        try:
            client_w, client_h = self.glfw.get_window_size(self.window)
            fb_w, fb_h = self.glfw.get_framebuffer_size(self.window)
            x = float(xpos) * max(1, int(fb_w)) / max(1, int(client_w))
            y = float(ypos) * max(1, int(fb_h)) / max(1, int(client_h))
        except Exception:
            return
        mode = str(getattr(self.config, "display_mode", ""))
        packed_mode = mode.strip().casefold().replace("_", "-")
        frame_left = min(rect[0] for rect in eye_rects)
        frame_top = min(rect[1] for rect in eye_rects)
        frame_right = max(rect[2] for rect in eye_rects)
        frame_bottom = max(rect[3] for rect in eye_rects)
        if not (frame_left <= x < frame_right and frame_top <= y < frame_bottom):
            self._virtual_pointer_uv = None
            return

        # The ray is measured in the complete packed output. A ray at the
        # SBS midpoint therefore maps to source u=0.5 in both eyes, which
        # places the markers at 25% and 75% of the full frame.
        if packed_mode.endswith("sbs"):
            u = (x - frame_left) / max(1, frame_right - frame_left)
            eye_top = max(rect[1] for rect in eye_rects)
            eye_bottom = min(rect[3] for rect in eye_rects)
            v = (y - eye_top) / max(1, eye_bottom - eye_top)
        elif packed_mode.endswith("tab"):
            eye_left = max(rect[0] for rect in eye_rects)
            eye_right = min(rect[2] for rect in eye_rects)
            u = (x - eye_left) / max(1, eye_right - eye_left)
            v = (y - frame_top) / max(1, frame_bottom - frame_top)
        else:
            left, top, right, bottom = eye_rects[0]
            u = (x - left) / max(1, right - left)
            v = (y - top) / max(1, bottom - top)
        self._virtual_pointer_uv = (u, v)

    def _on_mouse_button(self, _window: Any, button: int, action: int, _mods: int) -> None:
        left_button = int(getattr(self.glfw, "MOUSE_BUTTON_LEFT", 0))
        release = int(getattr(self.glfw, "RELEASE", 0))
        if int(action) == release:
            self._media_browser_scroll_drag = False
            if self._media_drag_action is not None and self._media_bar_hit_uv is not None:
                self._media_drag_last_sent_at = 0.0
                self._update_media_drag(self._media_bar_hit_uv[0])
            self._media_drag_action = None
            self._media_drag_eye = None
            self._sync_media_passthrough()
            return
        if int(button) != left_button or int(action) != int(getattr(self.glfw, "PRESS", 1)):
            return
        try:
            xpos, ypos = self.glfw.get_cursor_pos(self.window)
        except Exception:
            return
        self._on_cursor_pos(self.window, xpos, ypos)
        hit = self._media_bar_hit_from_cursor(xpos, ypos)
        if hit is None:
            return
        self._media_last_activity = time.monotonic()
        if hit[0] == "browser":
            browser_hit = hit[4]
            if browser_hit is None:
                self._dispatch_media_hit(("browse_close", None))
            else:
                if browser_hit[0] == "browser_scroll":
                    self._media_browser_scroll_drag = True
                self._dispatch_media_hit(browser_hit)
        else:
            value = hit[4]
            if value == "wake":
                self._media_last_activity = time.monotonic()
                self._media_overlay_key = None
                return
            self._dispatch_media_bar_hit(value)
            if value and value.startswith(("seek:", "volume:")):
                self._media_drag_action = value.split(":", 1)[0]
                self._media_drag_eye = hit[1]
                self._media_drag_last_sent_at = 0.0
                self._media_bar_hit_uv = (hit[2], hit[3])
                self._update_media_drag(hit[2])
        self._media_overlay_key = None
        self._refresh_media_overlay()

    def _on_scroll(self, _window: Any, _xoffset: float, yoffset: float) -> None:
        if not self._media_browser_visible():
            return
        if not self._media_panel_hovered:
            return
        step, _visible, maximum = media_browser_scroll_metrics(self._media_state)
        self._media_browser_scroll = max(
            0,
            min(maximum, self._media_browser_scroll + (-step * 3 if yoffset > 0 else step * 3)),
        )
        self._media_overlay_key = None
        self._refresh_media_overlay()

    def _upload_fps_overlay(self, panel: Any) -> bool:
        if panel is None:
            return False
        try:
            width, height = (int(value) for value in panel.size)
            if width <= 0 or height <= 0:
                return False
            rgba = (
                panel
                if getattr(panel, "mode", None) == "RGBA"
                else panel.convert("RGBA")
            )
            pixels = rgba.tobytes()
            texture = self._fps_overlay_texture
            if texture is None or self._fps_overlay_size != (width, height):
                texture = self._make_texture(width, height, self._source_pixel_format)
                if texture is None:
                    return False
                self._fps_overlay_texture = texture
                self._fps_overlay_size = (width, height)
            region = Metal.MTLRegionMake2D(0, 0, width, height)
            texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow_(
                region, 0, pixels, width * 4
            )
            return True
        except Exception as exc:
            print(
                "[MetalLocalViewer] FPS overlay upload skipped: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return False

    def _fps_overlay_rects(
        self, eye_rects: tuple[tuple[int, int, int, int], ...]
    ) -> tuple[tuple[float, float, float, float], ...]:
        empty = (0.0, 0.0, 0.0, 0.0)
        if (
            not self._fps_overlay_visible
            or self._fps_overlay_texture is None
            or self._fps_overlay_size == (0, 0)
        ):
            return tuple(empty for _ in eye_rects) or (empty,)
        display_size = overlay_size_for_display_mode(
            self._fps_overlay_size,
            getattr(self.config, "display_mode", "Half-SBS"),
        )
        return tuple(
            tuple(
                float(value) for value in overlay_rect_for_eye(eye_rect, display_size)
            )
            for eye_rect in eye_rects
        ) or (empty,)

    def _enable_native_cursor_hiding(self) -> None:
        if not self.config.hide_cursor_overlay or self._cursor_hide_enabled:
            return
        self._cursor_hide_enabled = set_window_mouse_passthrough(
            self.window,
            True,
            hide_cursor_overlay=True,
        )

    def _cursor_overlay_rects(
        self, eye_rects: tuple[tuple[int, int, int, int], ...]
    ) -> tuple[tuple[float, float, float, float], ...]:
        empty = (0.0, 0.0, 0.0, 0.0)
        if (
            is_media_capture_mode(getattr(self.config, "capture_mode", ""))
            and bool(getattr(self.config, "pointer_mapping_enabled", False))
        ):
            return self._virtual_cursor_rects(eye_rects)
        return (empty, empty)

    def _virtual_cursor_rects(
        self, eye_rects: tuple[tuple[int, int, int, int], ...]
    ) -> tuple[tuple[float, float, float, float], ...]:
        empty = (0.0, 0.0, 0.0, 0.0)
        last_activity = float(getattr(self, "_media_last_activity", 0.0) or 0.0)
        if (
            getattr(self, "_virtual_pointer_texture", None) is None
            or getattr(self, "_virtual_pointer_uv", None) is None
            or not eye_rects
            or (
                last_activity > 0.0
                and time.monotonic() - last_activity > MEDIA_POINTER_IDLE_SECONDS
            )
        ):
            return (empty, empty)
        u, v = self._virtual_pointer_uv
        mode = str(getattr(self.config, "display_mode", ""))
        packed_mode = mode.strip().casefold().replace("_", "-")
        pointer_width = 24 if packed_mode == "half-sbs" else 48
        rects = []
        for left, top, right, bottom in eye_rects:
            x = int(round(left + u * (right - left))) - pointer_width // 2
            y = int(round(top + v * (bottom - top))) - 24
            rects.append((float(x), float(y), float(pointer_width), 48.0))
        while len(rects) < 2:
            rects.append(empty)
        return tuple(rects[:2])

    def _cursor_texture_for_present(self) -> Any:
        if bool(getattr(self.config, "pointer_mapping_enabled", False)):
            if self._virtual_pointer_texture is not None:
                return self._virtual_pointer_texture
        return self._cursor_texture

    def _pack_native_coreml_frame(self, frame: Any):
        native_frame = getattr(frame, "viewer_native", None)
        if native_frame is None:
            return None
        output_size = getattr(frame, "output_eye_size", None)
        if not isinstance(output_size, (tuple, list)) or len(output_size) != 2:
            output_size = getattr(native_frame, "source_size", None)
        if not isinstance(output_size, (tuple, list)) or len(output_size) != 2:
            raise RuntimeError("native CoreML frame has no output dimensions")
        output_width, output_height = map(int, output_size)
        output_format = str(getattr(frame, "output_format", "half_sbs") or "half_sbs")
        normalized_format = output_format.strip().casefold().replace("-", "_")
        # Keep the local viewer on the same native DIBR implementation as the
        # original display path. The former low-latency half-SBS shortcut used
        # a center sample and disabled edge AA, which necessarily changed
        # silhouettes and occlusion pixels. Exact display parity takes
        # precedence here, so the configured full-quality parameters are sent
        # through unchanged.
        low_latency = False
        width = output_width * 2 if normalized_format == "full_sbs" else output_width
        height = output_height * 2 if normalized_format == "full_tab" else output_height
        size = (width, height)
        if min(size) < 1:
            raise RuntimeError(f"native CoreML output size is invalid: {size}")

        if size != self._native_pack_size:
            self._native_pack_size = size
            self._native_pack_cursor = 0
            self._native_pack_slots = []
            for _index in range(2):
                buffer = self.device.newBufferWithLength_options_(
                    width * height * 4, Metal.MTLResourceStorageModeShared
                )
                if buffer is None:
                    raise RuntimeError("could not allocate a Metal native-output buffer")
                descriptor = Metal.MTLTextureDescriptor.texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
                    self._source_pixel_format, width, height, False
                )
                descriptor.setUsage_(Metal.MTLTextureUsageShaderRead)
                descriptor.setStorageMode_(Metal.MTLStorageModeShared)
                texture = buffer.newTextureWithDescriptor_offset_bytesPerRow_(
                    descriptor, 0, width * 4
                )
                if texture is None:
                    raise RuntimeError("could not create a texture view of the Metal output buffer")
                self._native_pack_slots.append(
                    {
                        "buffer": buffer,
                        "texture": texture,
                        "busy": False,
                        "wait_event": None,
                        "wait_value": 0,
                    }
                )

        slot = None
        for offset in range(len(self._native_pack_slots)):
            index = (self._native_pack_cursor + offset) % len(self._native_pack_slots)
            candidate = self._native_pack_slots[index]
            if not candidate["busy"]:
                candidate["busy"] = True
                candidate["wait_event"] = None
                candidate["wait_value"] = 0
                self._native_pack_cursor = (index + 1) % len(self._native_pack_slots)
                slot = candidate
                break
        if slot is None:
            try:
                native_frame.release()
            except Exception:
                pass
            object.__setattr__(frame, "viewer_native", None)
            return None

        import objc

        pack_started = time.perf_counter()
        try:
            pack_metrics = native_frame.pack_metal(
                objc.pyobjc_id(slot["buffer"]),
                width * height * 4,
                size,
                output_format,
            )
        except Exception as exc:
            slot["busy"] = False
            try:
                native_frame.release()
            except Exception:
                pass
            object.__setattr__(frame, "viewer_native", None)
            if native_pack_error_is_recoverable(exc):
                debug = getattr(frame, "debug_info", None)
                if isinstance(debug, dict):
                    debug["macos_metal_native_pack_drop"] = 1
                    debug["macos_metal_native_pack_error"] = str(exc)
                on_breakdown_inc = getattr(
                    getattr(self, "config", None), "on_breakdown_inc", None
                )
                if on_breakdown_inc is not None:
                    on_breakdown_inc("viewer_drop", 1)
                return None
            raise
        pack_async = bool(pack_metrics.get("asynchronous", False))
        if pack_async:
            event_pointer = int(pack_metrics.get("completion_event", 0) or 0)
            if event_pointer:
                slot["wait_event"] = objc.objc_object(c_void_p=event_pointer)
                slot["wait_value"] = int(pack_metrics.get("completion_value", 0) or 0)
            else:
                pack_async = False
        if not pack_async:
            try:
                native_frame.release()
            except Exception:
                pass
        object.__setattr__(frame, "viewer_native", None)
        debug = getattr(frame, "debug_info", None)
        if isinstance(debug, dict):
            debug["macos_metal_output_pack_ms"] = (
                time.perf_counter() - pack_started
            ) * 1000.0
            debug["macos_metal_host_image_copies"] = 0
            debug["macos_metal_gpu_output_buffer"] = 1
            debug["macos_metal_half_sbs_single_sample"] = int(
                2 if normalized_format == "half_sbs" and low_latency else 0
            )
            debug["macos_metal_low_latency"] = int(low_latency)
            if isinstance(pack_metrics, dict):
                debug["macos_metal_pack_gpu_ms"] = float(
                    pack_metrics.get("metal_gpu_ms", 0.0) or 0.0
                )
                debug["macos_metal_pack_host_handoff_ms"] = float(
                    pack_metrics.get("host_handoff_copy_ms", 0.0) or 0.0
                )
                debug["macos_metal_pack_warp_threads"] = int(
                    pack_metrics.get("warp_threads", 0) or 0
                )
                debug["macos_metal_pack_async"] = int(pack_async)
        return slot, slot["texture"], width, height

    # ── frame flow ──

    def present(self, frame: Any) -> bool:
        if self.window is None or not self._initialized_flag():
            raise RuntimeError("Metal local viewer has not been initialized")
        self.poll_events()
        # The GUI callback is live and may change while the viewer thread is
        # presenting. Read it for every frame so hiding/showing the panel does
        # not wait for the five-second FPS reporting interval.
        self._sync_fps_overlay_visibility()

        # macOS local presentation is GPU-only: native CoreML/Metal packing
        # feeds the CAMetalLayer, and a host RGBA upload is never accepted.
        gpu_required = (
            sys.platform == "darwin"
            or os.environ.get("D2S_GPU_IMAGE_PIPELINE_REQUIRED", "0") == "1"
        )
        native_frame = getattr(frame, "viewer_native", None)
        if (
            self._media_browser_visible()
            and native_frame is None
            and getattr(frame, "viewer_rgb", None) is None
            and getattr(frame, "viewer_frame_np", None) is None
        ):
            return self.present_media_browser()
        # The shader-warp compatibility path uploads depth through CPU memory.
        # Strict GPU mode uses the native CoreML/Metal packer instead.
        rgb = getattr(frame, "viewer_rgb", None)
        depth = getattr(frame, "viewer_depth", None)
        if (
            rgb is not None
            and depth is not None
            and self._warp_pipeline is not None
            and os.environ.get("D2S_METAL_SHADER_WARP", "0") == "1"
        ):
            if gpu_required:
                raise RuntimeError(
                    "GPU-only Metal presentation cannot use the shader-warp path "
                    "because its depth upload reads back to CPU memory"
                )
            return self._present_warp(
                rgb, depth, bgra=getattr(frame, "viewer_bgra", None)
            )

        if gpu_required and not _has_gpu_present_source(frame, self.config):
            raise RuntimeError(
                "GPU-only Metal presentation requires a native frame or packed media frame"
            )
        # The fragment deferred-warp path is intentionally not used for normal
        # presentation. It is an approximation and does not produce the same
        # pixels as the original layered DIBR display path. The native Metal
        # pack below is the single source of truth for local output.
        pack_started = time.perf_counter()
        native_pack = self._pack_native_coreml_frame(frame)
        native_slot = None
        if native_pack is not None:
            native_slot, texture, width, height = native_pack
        elif gpu_required and getattr(frame, "viewer_frame_np", None) is None:
            if self.config.on_breakdown_inc is not None:
                self.config.on_breakdown_inc("viewer_drop", 1)
            return False

        # Runtime-packed host frame: pure memcpy, no device sync.
        host_np = None if native_slot is not None else getattr(frame, "viewer_frame_np", None)
        if native_slot is not None:
            pixels = None
        elif host_np is not None:
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
        if native_slot is None:
            texture = self._ensure_staging_texture(width, height)
            region = Metal.MTLRegionMake2D(0, 0, width, height)
            texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow_(
                region, 0, pixels, width * 4
            )

        return self._present_packed_texture(
            texture,
            width,
            height,
            native_slot=native_slot,
            native_frame=native_frame,
        )

    def _present_packed_texture(
        self,
        texture: Any,
        width: int,
        height: int,
        *,
        native_slot: dict[str, Any] | None = None,
        native_frame: Any | None = None,
    ) -> bool:
        """Submit a packed GPU texture with the live cursor overlays."""
        drawable = self.layer.nextDrawable()
        if drawable is None:
            if native_frame is not None:
                wait_event = (
                    native_slot.get("wait_event")
                    if native_slot is not None
                    else None
                )
                wait_value = (
                    int(native_slot.get("wait_value", 0) or 0)
                    if native_slot is not None
                    else 0
                )
                if wait_event is not None and wait_value:
                    cleanup = self.queue.commandBuffer()
                    cleanup.encodeWaitForEvent_value_(wait_event, wait_value)

                    def _release_unscheduled_native(
                        _completed,
                        frame=native_frame,
                    ):
                        try:
                            frame.release()
                        except Exception:
                            pass

                    cleanup.addCompletedHandler_(_release_unscheduled_native)
                    cleanup.commit()
                else:
                    try:
                        native_frame.release()
                    except Exception:
                        pass
            if native_slot is not None:
                native_slot["busy"] = False
                native_slot["wait_event"] = None
                native_slot["wait_value"] = 0
            return False

        fb_w, fb_h = self.glfw.get_framebuffer_size(self.window)
        # GLFW already reports framebuffer pixels on macOS. Multiplying by
        # backingScaleFactor again overallocates Retina drawables (for example
        # 1920x1080 becomes 7680x4320) and can make fullscreen presents vanish.
        dw, dh = max(1, int(fb_w)), max(1, int(fb_h))
        if self._last_drawable_size != (dw, dh):
            self.layer.setDrawableSize_((dw, dh))
            self._last_drawable_size = (dw, dh)

        # Aspect-fit the SBS texture inside the drawable (letterbox black).
        scale = min(dw / width, dh / height)
        su = min(1.0, (width * scale) / dw)
        sv = min(1.0, (height * scale) / dh)
        off_u = (1.0 - su) / 2.0
        off_v = (1.0 - sv) / 2.0
        content_left = round(off_u * dw)
        content_top = round(off_v * dh)
        content_width = round(su * dw)
        content_height = round(sv * dh)
        eye_rects = (
            (
                content_left,
                content_top,
                content_left + content_width // 2,
                content_top + content_height,
            ),
            (
                content_left + content_width // 2,
                content_top,
                content_left + content_width,
                content_top + content_height,
            ),
        )
        self._last_eye_rects = eye_rects
        self._refresh_media_overlay()
        media_rects = self._media_overlay_rects(eye_rects)

        submit_started = time.perf_counter()
        command_buffer = self.queue.commandBuffer()
        if native_slot is not None:
            wait_event = native_slot.get("wait_event")
            wait_value = int(native_slot.get("wait_value", 0) or 0)
            if wait_event is not None and wait_value:
                command_buffer.encodeWaitForEvent_value_(wait_event, wait_value)

            def _complete_native_slot(
                _completed,
                slot=native_slot,
                frame=native_frame,
            ):
                slot["busy"] = False
                slot["wait_event"] = None
                slot["wait_value"] = 0
                if frame is not None:
                    try:
                        frame.release()
                    except Exception:
                        pass

            command_buffer.addCompletedHandler_(_complete_native_slot)
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
        # PyObjC requires a buffer-protocol object for `const void*`; a plain
        # tuple raises "too few values (4) expecting at least 16".
        encoder.setFragmentBytes_length_atIndex_(
            struct.pack("4f", su, sv, off_u, off_v), 16, 0
        )
        cursor_rects = self._cursor_overlay_rects(eye_rects)
        fps_rects = self._fps_overlay_rects(eye_rects)
        encoder.setFragmentTexture_atIndex_(self._cursor_texture_for_present(), 1)
        fps_texture = self._fps_overlay_texture
        if fps_texture is None:
            fps_texture = self._cursor_texture
        encoder.setFragmentTexture_atIndex_(fps_texture, 2)
        media_texture = self._media_overlay_texture
        if media_texture is None:
            media_texture = self._cursor_texture
        encoder.setFragmentTexture_atIndex_(media_texture, 3)
        encoder.setFragmentBytes_length_atIndex_(
            struct.pack("4f", *cursor_rects[0]), 16, 1
        )
        encoder.setFragmentBytes_length_atIndex_(
            struct.pack("4f", *cursor_rects[1]), 16, 2
        )
        encoder.setFragmentBytes_length_atIndex_(
            struct.pack("4f", *fps_rects[0]), 16, 3
        )
        encoder.setFragmentBytes_length_atIndex_(
            struct.pack(
                "4f", *(fps_rects[1] if len(fps_rects) > 1 else fps_rects[0])
            ),
            16,
            4,
        )
        encoder.setFragmentBytes_length_atIndex_(
            struct.pack("4f", *media_rects[0]), 16, 5
        )
        encoder.setFragmentBytes_length_atIndex_(
            struct.pack(
                "4f", *(media_rects[1] if len(media_rects) > 1 else media_rects[0])
            ),
            16,
            6,
        )
        encoder.drawPrimitives_vertexStart_vertexCount_(
            Metal.MTLPrimitiveTypeTriangleStrip, 0, 4
        )
        encoder.endEncoding()
        command_buffer.presentDrawable_(drawable)
        self._last_present_metrics = {
            "metal_present_submit_cpu_ms": 0.0,
        }
        command_buffer.commit()
        self._last_present_metrics = {
            "metal_present_submit_cpu_ms": max(
                0.0, (time.perf_counter() - submit_started) * 1000.0
            ),
        }
        self._last_present_texture = texture
        self._last_present_size = (int(width), int(height))
        self._enable_native_cursor_hiding()

        now = time.perf_counter()
        self._fps_frames += 1
        fps = present_fps_if_due(
            self._fps_frames,
            now - self._fps_started,
            interval=FPS_REPORT_INTERVAL_SECONDS,
        )
        if fps is not None:
            self._report_present_fps(fps, self._fps_frames)
            self._fps_frames = 0
            self._fps_started = now
        return True

    def present_cached(self) -> bool:
        """Redraw the last GPU frame so cursor motion is independent of inference."""
        if self._media_browser_visible() and getattr(self, "_blank_texture", None) is not None:
            self.poll_events()
            self._sync_fps_overlay_visibility()
            return self._present_packed_texture(self._blank_texture, 2, 1)
        if (
            self.window is not None
            and self._initialized_flag()
            and getattr(self, "_last_native_frame", None) is not None
        ):
            self.poll_events()
            self._sync_fps_overlay_visibility()
            presented = self._present_native_coreml_warp(None, cached=True)
            if presented is not None:
                return presented
        texture = self._last_present_texture
        width, height = self._last_present_size
        if (
            self.window is None
            or not self._initialized_flag()
        ):
            return False
        if texture is None or width <= 0 or height <= 0:
            texture = getattr(self, "_blank_texture", None)
            if texture is None:
                return False
            width, height = 2, 1
        self.poll_events()
        self._sync_fps_overlay_visibility()
        return self._present_packed_texture(texture, width, height)

    def present_blank(self) -> bool:
        """Present a black SBS player surface while no media frame is available."""
        if (
            self.window is None
            or not self._initialized_flag()
            or getattr(self, "_blank_texture", None) is None
        ):
            return False
        self.poll_events()
        self._sync_fps_overlay_visibility()
        return self._present_packed_texture(self._blank_texture, 2, 1)

    def present_media_browser(self) -> bool:
        """Present the file browser before a media frame is available."""
        if not self._media_browser_visible() or getattr(self, "_blank_texture", None) is None:
            return False
        self.poll_events()
        self._sync_fps_overlay_visibility()
        return self._present_packed_texture(self._blank_texture, 2, 1)

    def _present_cached(self) -> bool:
        """Compatibility alias matching the Vulkan viewer interface."""
        return self.present_cached()

    def cached_present_interval(self) -> float:
        """Return a bounded display cadence for cursor-only redraws."""
        try:
            refresh_hz = float(self._display_refresh_hz)
        except (AttributeError, TypeError, ValueError):
            refresh_hz = 60.0
        return 1.0 / max(20.0, min(240.0, refresh_hz))

    def _initialized_flag(self) -> bool:
        return self.pipeline is not None

    def _retire_native_entry_if_ready(self, entry: dict[str, Any]) -> None:
        """Release a native slot once it is dropped and no Metal command uses it."""
        if (
            entry.get("drop")
            and entry.get("completed")
            and not entry.get("released")
        ):
            entry["released"] = True
            try:
                entry["frame"].release()
            except Exception:
                pass

    def _complete_native_entry(self, entry: dict[str, Any]) -> None:
        """Finish one command-buffer lease on a retained native frame."""
        entry["pending"] = max(0, int(entry.get("pending", 0)) - 1)
        entry["completed"] = entry["pending"] == 0
        self._retire_native_entry_if_ready(entry)

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
                color_size[0], color_size[1], self._source_pixel_format
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
        self,
        win_w: int,
        win_h: int,
        tex_w: int,
        tex_h: int,
        *,
        ipd_uv: float | None = None,
    ) -> list[tuple[tuple[int, int, int, int], float]]:
        """Eye viewports + eye offsets, ported from v2.5 _stereo_viewports."""
        ipd_half = (self._warp_ipd_uv if ipd_uv is None else ipd_uv) / 2.0
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

    def _present_native_coreml_warp(
        self, frame: Any, *, cached: bool = False
    ) -> bool | None:
        """Render from retained Core ML Metal resources without an SBS pack."""
        native_frame = (
            getattr(frame, "viewer_native", None)
            if not cached
            else self._last_native_frame
        )
        if native_frame is None:
            return None
        resources = getattr(native_frame, "metal_resources", None)
        if not callable(resources):
            return None
        try:
            metal_resources = resources()
        except Exception:
            metal_resources = None
        if not metal_resources:
            return None
        color_texture, depth_buffer = metal_resources
        if not cached:
            entry = {
                "frame": native_frame,
                "completed": True,
                "pending": 0,
                "drop": False,
                "released": False,
            }
            previous = self._last_native_entry
            if previous is not None:
                previous["drop"] = True
                self._retire_native_entry_if_ready(previous)
            self._last_native_entry = entry
            self._last_native_frame = native_frame
        else:
            entry = self._last_native_entry
            if entry is None or entry.get("released"):
                return None
        entry["pending"] = int(entry.get("pending", 0)) + 1
        entry["completed"] = False
        debug_info = getattr(frame, "debug_info", None) if frame is not None else None
        if isinstance(debug_info, dict):
            debug_info["macos_metal_deferred_native_warp"] = 1
            debug_info["macos_metal_host_image_copies"] = 0
            debug_info["macos_metal_gpu_output_buffer"] = 0
            debug_info["macos_metal_output_pack_ms"] = 0.0
        presented = self._present_warp(
            None,
            None,
            native_color_texture=color_texture,
            native_depth_buffer=depth_buffer,
            native_frame=native_frame,
            native_release_entry=entry,
        )
        if presented is False:
            self._complete_native_entry(entry)
        if presented and not cached:
            object.__setattr__(frame, "viewer_native", None)
        elif presented is False and not cached:
            entry["drop"] = True
            self._retire_native_entry_if_ready(entry)
            if self._last_native_entry is entry:
                self._last_native_entry = None
                self._last_native_frame = None
        return presented

    def _present_warp(
        self,
        rgb: Any,
        depth: Any,
        bgra: Any = None,
        *,
        native_color_texture: Any = None,
        native_depth_buffer: Any = None,
        native_frame: Any = None,
        native_release_entry: dict[str, Any] | None = None,
    ) -> bool:
        import torch  # local: module stays importable without torch

        # Zero-copy fast path: sample the captured SCK BGRA texture directly
        # (IOSurface-backed, never touches CPU); release it once the GPU is
        # done. Fallback: device-pack the preprocessed tensor.
        zc_texture = native_color_texture
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
            _color_view = color_tex if zc_texture is not None else None
            del _color_view
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
        raw_normalize = (0.0, 1.0, 0.0, 0.0)
        if native_depth_buffer is not None and native_frame is not None:
            dw, dh = int(native_frame.depth_width), int(native_frame.depth_height)
            depth_is_float16 = bool(getattr(native_frame, "depth_is_float16", False))
            depth_format = (
                Metal.MTLPixelFormatR16Float
                if depth_is_float16
                else Metal.MTLPixelFormatR32Float
            )
            depth_desc = (
                Metal.MTLTextureDescriptor
                .texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
                    depth_format, dw, dh, False
                )
            )
            depth_desc.setUsage_(Metal.MTLTextureUsageShaderRead)
            depth_desc.setStorageMode_(Metal.MTLStorageModeShared)
            depth_texture = native_depth_buffer.newTextureWithDescriptor_offset_bytesPerRow_(
                depth_desc,
                0,
                dw * (2 if depth_is_float16 else 4),
            )
            if depth_texture is None:
                return None
            raw_normalize = (
                float(getattr(native_frame, "raw_normalize_lo", 0.0)),
                float(getattr(native_frame, "raw_normalize_hi", 1.0)),
                1.0,
                0.0,
            )
        else:
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
            # Preserve ScreenCaptureKit's display-referred sRGB bytes through
            # the unorm drawable; the CAMetalLayer carries the sRGB colorspace.
            color_tex = zc_texture
        else:
            color_region = Metal.MTLRegionMake2D(0, 0, width, height)
            self._warp_color_tex.replaceRegion_mipmapLevel_withBytes_bytesPerRow_(
                color_region, 0, rgba, width * 4
            )
            color_tex = self._warp_color_tex
        if native_depth_buffer is None:
            # ctypes.string_at: C-speed raw copy of the tensor's bytes.
            # bytes(tensor) loops per element; numpy is intentionally avoided.
            import ctypes

            flat = depth_t.reshape(-1)
            depth_payload = ctypes.string_at(flat.data_ptr(), flat.numel())
            depth_bpr = dw * (1 if depth_is_u8 else 4)
            depth_region = Metal.MTLRegionMake2D(0, 0, dw, dh)
            self._warp_depth_tex.replaceRegion_mipmapLevel_withBytes_bytesPerRow_(
                depth_region, 0, depth_payload, depth_bpr
            )
            depth_texture = self._warp_depth_tex
        native_warp = dict(getattr(native_frame, "warp_config", None) or {})
        if native_frame is not None:
            native_max_disparity = max(
                0.0, float(native_warp.get("max_disparity_px", 48.0))
            )
            native_ipd_uv = native_max_disparity / max(float(width), 1.0)
            warp_depth_strength = max(
                0.0, float(native_warp.get("depth_strength", 1.0))
            )
            warp_convergence = float(native_warp.get("convergence", 0.0))
        else:
            native_ipd_uv = None
            warp_depth_strength = 0.1 * self._warp_depth_ratio
            warp_convergence = self._warp_convergence
        if os.environ.get("D2S_PREP_TRACE") == "1" and not getattr(
            self, "_warp_trace_once", False
        ):
            self._warp_trace_once = True
            trace_depth_u8 = (
                False if native_depth_buffer is not None else depth_is_u8
            )
            trace_depth_bpr = dw * (
                2 if native_depth_buffer is not None and native_frame is not None
                and bool(getattr(native_frame, "depth_is_float16", False))
                else 1 if trace_depth_u8 else 4
            )
            print(
                f"[warp-trace] u8={trace_depth_u8} {dw}x{dh} bpr={trace_depth_bpr} "
                f"zcolor={zc_texture is not None} "
                "depth_fmt="
                f"{getattr(self, '_warp_depth_format', depth_format if native_depth_buffer is not None else '')}",
                flush=True,
            )

        drawable = self.layer.nextDrawable()
        if drawable is None:
            if bgra is not None:
                _release_bgra()
            return False
        fb_w, fb_h = self.glfw.get_framebuffer_size(self.window)
        dw_px, dh_px = max(1, int(fb_w)), max(1, int(fb_h))
        if self._last_drawable_size != (dw_px, dh_px):
            self.layer.setDrawableSize_((dw_px, dh_px))
            self._last_drawable_size = (dw_px, dh_px)

        submit_started = time.perf_counter()
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
        encoder.setFragmentTexture_atIndex_(depth_texture, 1)

        mode_id = self._warp_mode_id(str(getattr(self.config, "display_mode", "Half-SBS")))
        viewports = self._warp_viewports(
            dw_px,
            dh_px,
            width,
            height,
            ipd_uv=native_ipd_uv,
        )
        # Per-eye viewport draws always use warp mode 7 (the viewport selects
        # the eye region); single-viewport modes pass their own shader mode.
        mode_value = 7.0 if len(viewports) > 1 else float(mode_id)
        eye_rects = tuple(
            (
                int(viewport[0]),
                int(viewport[1]),
                int(viewport[0] + viewport[2]),
                int(viewport[1] + viewport[3]),
            )
            for viewport, _eye_offset in viewports
        )
        self._last_eye_rects = eye_rects
        self._refresh_media_overlay()
        media_rects = self._media_overlay_rects(eye_rects)
        cursor_rects = self._cursor_overlay_rects(eye_rects)
        fps_rects = self._fps_overlay_rects(eye_rects)
        encoder.setFragmentTexture_atIndex_(self._cursor_texture_for_present(), 2)
        fps_texture = self._fps_overlay_texture
        if fps_texture is None:
            fps_texture = self._cursor_texture
        encoder.setFragmentTexture_atIndex_(fps_texture, 3)
        media_texture = self._media_overlay_texture
        if media_texture is None:
            media_texture = self._cursor_texture
        encoder.setFragmentTexture_atIndex_(media_texture, 4)
        for index, (viewport, eye_offset) in enumerate(viewports):
            # struct.pack: numpy-free uniform payload (32 floats, native LE
            # matches MSL float layout).
            payload = struct.pack(
                "<32f",
                eye_offset,
                warp_depth_strength,
                warp_convergence,
                0.0,
                float(viewport[0]),
                float(viewport[1]),
                float(viewport[2]),
                float(viewport[3]),
                mode_value,
                0.0,
                0.0,
                0.0,
                *cursor_rects[0],
                *cursor_rects[1],
                *(
                    fps_rects[index]
                    if index < len(fps_rects)
                    else (0.0, 0.0, 0.0, 0.0)
                ),
                *(
                    media_rects[index]
                    if index < len(media_rects)
                    else (0.0, 0.0, 0.0, 0.0)
                ),
                *raw_normalize,
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
        if native_frame is not None:
            def _release_native(_cb=None, _color_view=color_tex):
                # Keep the borrowed IOSurface texture view alive through GPU use.
                del _color_view
                try:
                    if native_release_entry is None:
                        native_frame.release()
                        return
                    self._complete_native_entry(native_release_entry)
                except Exception:
                    pass

            try:
                command_buffer.addCompletedHandler_(_release_native)
            except Exception:
                _release_native()
        self._last_present_metrics = {
            "metal_present_submit_cpu_ms": 0.0,
        }
        try:
            def _record_present_gpu(_cb):
                try:
                    gpu_start = _cb.GPUStartTime
                    gpu_end = _cb.GPUEndTime
                    if callable(gpu_start):
                        gpu_start = gpu_start()
                    if callable(gpu_end):
                        gpu_end = gpu_end()
                    if gpu_end >= gpu_start and gpu_start > 0.0:
                        self._last_present_metrics["metal_present_gpu_ms"] = (
                            gpu_end - gpu_start
                        ) * 1000.0
                except Exception:
                    pass

            command_buffer.addCompletedHandler_(_record_present_gpu)
        except Exception:
            pass
        command_buffer.commit()
        self._last_present_metrics["metal_present_submit_cpu_ms"] = max(
            0.0, (time.perf_counter() - submit_started) * 1000.0
        )
        self._enable_native_cursor_hiding()

        now = time.perf_counter()
        self._fps_frames += 1
        fps = present_fps_if_due(
            self._fps_frames,
            now - self._fps_started,
            interval=FPS_REPORT_INTERVAL_SECONDS,
        )
        if fps is not None:
            self._report_present_fps(fps, self._fps_frames)
            self._fps_frames = 0
            self._fps_started = now
        return True

    def record_present(self, reused: bool) -> None:
        """Count successful submissions for the shared Windows-style OSD."""
        self._present_total += 1
        if reused:
            self._present_reused += 1

    def _refresh_content_rate(self) -> None:
        window_start = getattr(self, "_present_window_start", None)
        if window_start is None:
            return
        now = time.perf_counter()
        elapsed = now - window_start
        if elapsed < 0.5:
            return
        fresh = self._present_total - self._present_reused
        self._content_fps = fresh / elapsed
        self._content_frame_count = fresh
        self._reuse_ratio = (
            self._present_reused / self._present_total
            if self._present_total
            else 0.0
        )
        self._present_total = 0
        self._present_reused = 0
        self._present_window_start = now

    def content_fps(self) -> float:
        return self._content_fps

    def reuse_ratio(self) -> float:
        return self._reuse_ratio

    def latency_ms(self) -> float:
        return self._last_latency_ms

    def avg_latency_ms(self) -> float:
        if not self._latency_history:
            return 0.0
        return sum(self._latency_history) / len(self._latency_history)

    @staticmethod
    def _sample_percentile(samples: list[float], quantile: float) -> float:
        if not samples:
            return 0.0
        position = (len(samples) - 1) * min(1.0, max(0.0, float(quantile)))
        lower = int(position)
        upper = min(lower + 1, len(samples) - 1)
        fraction = position - lower
        return samples[lower] + (samples[upper] - samples[lower]) * fraction

    def _log_present_metrics(self, fps: float, capture_target: int | None) -> None:
        """Emit the same stable latency fields as the Vulkan local viewer."""
        latency = sorted(self._latency_history)
        parts = [
            "backend=metal",
            f"present_fps={float(fps):.2f}",
            f"content_fps={self.content_fps():.2f}",
            f"reuse={self.reuse_ratio():.4f}",
            f"latency_ms={self.latency_ms():.2f}",
            f"avg_latency_ms={self.avg_latency_ms():.2f}",
            f"latency_samples={len(self._latency_history)}",
            f"latency_anchor={self._latency_anchor}",
            f"display_hz={int(self._display_refresh_hz)}",
        ]
        if self._upstream_fps > 0.0:
            parts.append(f"upstream_fps={self._upstream_fps:.2f}")
        if latency:
            parts.extend(
                (
                    f"latency_p50_ms={self._sample_percentile(latency, 0.50):.2f}",
                    f"latency_p95_ms={self._sample_percentile(latency, 0.95):.2f}",
                    f"latency_p99_ms={self._sample_percentile(latency, 0.99):.2f}",
                    f"latency_max_ms={latency[-1]:.2f}",
                )
            )
        for name in (
            "capture_to_runtime_start_ms",
            "capture_to_runtime_ready_ms",
            "capture_wait_ms",
            "sck_callback_to_runtime_ms",
            "runtime_execution_ms",
            "runtime_gpu_wait_ms",
            "runtime_completion_ms",
            "depth_total_ms",
            "depth_preprocess_ms",
            "depth_model_ms",
            "depth_postprocess_ms",
            "synthesis_ms",
            "pack_ms",
            "macos_metal_output_pack_ms",
            "macos_metal_pack_gpu_ms",
            "macos_metal_pack_host_handoff_ms",
            "macos_metal_pack_async",
            "metal_present_submit_cpu_ms",
            "metal_present_gpu_ms",
            "viewer_queue_ms",
            "present_call_ms",
        ):
            samples = sorted(
                item[name]
                for item in self._latency_stage_history
                if name in item
            )
            if samples:
                parts.append(f"{name}_p95={self._sample_percentile(samples, 0.95):.2f}")
        if capture_target is not None:
            parts.append(f"capture_target={int(capture_target)}")
        print(f"[D2S_METRICS] {' '.join(parts)}", flush=True)

    def record_frame_latency(
        self,
        capture_start_time: float | None,
        *,
        stage_timing: dict[str, float] | None = None,
        anchor_time: float | None = None,
    ) -> None:
        """Record frame age using the same capture-to-present metric as Vulkan."""
        if capture_start_time is None:
            return
        try:
            started = float(capture_start_time)
        except (TypeError, ValueError):
            return
        if not math.isfinite(started) or started <= 0.0:
            return
        timing = stage_timing or {}
        try:
            presented_at = float(timing.get("presented_at", time.perf_counter()))
        except (TypeError, ValueError):
            presented_at = time.perf_counter()
        now = time.perf_counter()
        if math.isfinite(presented_at) and started <= presented_at <= now:
            now = presented_at
        label = "poll"
        if os.environ.get("D2S_LATENCY_ANCHOR", "callback").strip().lower() not in {
            "poll",
            "start",
            "legacy",
        }:
            for candidate_label, value in (
                ("sck_callback", timing.get("sck_callback_time")),
                ("capture_return", timing.get("capture_poll_returned_at")),
                ("callback", anchor_time),
            ):
                try:
                    candidate = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(candidate) and started <= candidate <= now:
                    started, label = candidate, candidate_label
                    break
        latency_ms = (now - started) * 1000.0
        if not math.isfinite(latency_ms) or latency_ms <= 0.0 or latency_ms > 10_000.0:
            return
        if label != self._latency_anchor:
            self._latency_history.clear()
        self._latency_anchor = label
        self._last_latency_ms = latency_ms
        self._latency_history.append(latency_ms)
        if len(self._latency_history) > self._latency_history_max:
            del self._latency_history[0]
        if timing:
            sample: dict[str, float] = {}
            for key in (
                "capture_to_runtime_start_ms",
                "capture_to_runtime_ready_ms",
                "capture_wait_ms",
                "sck_callback_to_runtime_ms",
                "runtime_execution_ms",
                "runtime_gpu_wait_ms",
                "runtime_completion_ms",
                "depth_total_ms",
                "depth_preprocess_ms",
                "depth_model_ms",
                "depth_postprocess_ms",
                "synthesis_ms",
                "pack_ms",
                "macos_metal_output_pack_ms",
                "macos_metal_pack_gpu_ms",
                "macos_metal_pack_host_handoff_ms",
                "macos_metal_pack_async",
                "macos_metal_pack_warp_threads",
                "metal_present_submit_cpu_ms",
                "metal_present_gpu_ms",
                "viewer_queue_ms",
                "present_call_ms",
            ):
                try:
                    value = float(timing[key])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(value) and 0.0 <= value < 10_000.0:
                    sample[key] = value
            if sample:
                self._latency_stage_history.append(sample)
                if len(self._latency_stage_history) > self._latency_stage_history_max:
                    del self._latency_stage_history[0]

    def _refresh_system_metrics(self) -> None:
        upstream_provider = getattr(self, "upstream_fps_provider", None)
        if callable(upstream_provider):
            try:
                self._upstream_fps = max(0.0, float(upstream_provider()))
            except (TypeError, ValueError, RuntimeError):
                self._upstream_fps = 0.0
        load_provider = getattr(self, "sys_load_provider", None)
        if callable(load_provider):
            try:
                load = load_provider()
                self._load_snapshot = dict(load) if load else {}
                self._gpu_util = load.get("gpu")
                self._gpu_util_max = load.get("gpu_max")
            except (AttributeError, TypeError, ValueError, RuntimeError):
                self._load_snapshot = {}
                self._gpu_util = None
                self._gpu_util_max = None

    def _report_present_fps(self, fps: float, frame_count: int) -> None:
        self._refresh_content_rate()
        self._refresh_system_metrics()
        capture_target = (
            self.config.on_sbs_fps(fps, frame_count)
            if self.config.on_sbs_fps is not None
            else None
        )
        show_fps = self._show_fps_enabled()
        self._fps_overlay_visible = bool(
            show_fps and self._fps_overlay_texture is not None
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
        if hasattr(self, "_latency_stage_history"):
            self._log_present_metrics(fps, capture_target)
        panel_kwargs = {
            "present_fps": fps,
            "capture_target": capture_target,
        }
        # Keep this conditional so small unit-test doubles and older callers
        # still receive the original two-field call shape.
        if hasattr(self, "_content_fps"):
            panel_kwargs.update(
                content_fps=self.content_fps(),
                reuse_ratio=self.reuse_ratio(),
                latency_ms=self.latency_ms(),
                avg_latency_ms=self.avg_latency_ms(),
                source_fps=self._upstream_fps if self._upstream_fps > 0.0 else None,
                gpu_util=self._gpu_util,
                gpu_util_max=self._gpu_util_max,
            )
        panel = build_fps_panel_rgba(**panel_kwargs)
        if self._upload_fps_overlay(panel):
            self._fps_overlay_visible = True

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
        pin_macos_viewer_window(
            self.window,
            fullscreen=enabled,
            show_in_app_switcher=bool(
                getattr(self.config, "show_in_app_switcher", False)
            ),
        )
        print(
            f"[MetalLocalViewer] Display fullscreen: {'on' if enabled else 'off'} "
            "(Alt+Enter)",
            flush=True,
        )

    def poll_events(self) -> None:
        self._read_media_state()
        self._refresh_media_overlay()
        self.glfw.poll_events()
        self._poll_virtual_pointer()
        self._refresh_media_overlay()
        if self.glfw.window_should_close(self.window):
            raise StopIteration
        if self._exclusive_fullscreen:
            pin_macos_viewer_window(
                self.window,
                fullscreen=True,
                show_in_app_switcher=bool(
                    getattr(self.config, "show_in_app_switcher", False)
                ),
            )
        if self.config.hide_cursor_overlay:
            update_native_cursor_suppression()
        self._sync_media_passthrough()

    def close(self) -> None:
        try:
            if self.config.cursor_passthrough and self.window is not None:
                set_window_mouse_passthrough(
                    self.window,
                    False,
                    hide_cursor_overlay=False,
                )
                self._cursor_hide_enabled = False
            if self._capture_window_registered and self._ns_window is not None:
                try:
                    window_id = int(self._ns_window.windowNumber())
                    callback = getattr(self.config, "on_capture_window_exclusion", None)
                    if callable(callback):
                        callback(window_id, False)
                except Exception as exc:
                    print(
                        "[MetalLocalViewer] Could not remove capture exclusion: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                self._capture_window_registered = False
            self._staging_texture = None
            self._blank_texture = None
            self._last_present_texture = None
            self._last_present_size = (0, 0)
            if self._last_native_entry is not None:
                entry = self._last_native_entry
                entry["drop"] = True
                self._retire_native_entry_if_ready(entry)
            self._last_native_entry = None
            self._last_native_frame = None
            self._fps_overlay_texture = None
            self._fps_overlay_size = (0, 0)
            self._fps_overlay_visible = False
            self._virtual_pointer_texture = None
            self._virtual_pointer_uv = None
            self.pipeline = None
            self.queue = None
            self.device = None
        finally:
            if self.window is not None and self.glfw is not None:
                self.glfw.destroy_window(self.window)
            if self.glfw is not None and self.config.manage_glfw_lifecycle:
                self.glfw.terminate()
            self.window = None


def _has_cached_frame(target: MetalLocalViewer | None) -> bool:
    """Return whether an idle redraw can reuse a packed or native frame."""
    if target is None:
        return False
    return bool(
        getattr(target, "_last_present_texture", None) is not None
        or getattr(target, "_last_native_frame", None) is not None
    )


def run_metal_local_viewer(*, runtime_q: Any, shutdown_event: Any, config: Any) -> None:
    """Metal variant of run_vulkan_local_viewer (macOS Local Viewer)."""
    viewer: MetalLocalViewer | None = None
    preview_logged = False
    next_cached_present_at = 0.0

    def _configure_metrics(target: MetalLocalViewer) -> None:
        packer = getattr(runtime_q, "_d2s_packer", None)
        if packer is not None:
            target.upstream_fps_provider = getattr(
                packer, "take_published_fps", None
            )
        try:
            from utils.sys_load import install_system_load_sampler

            sampler = install_system_load_sampler(cpu=False)
            if sampler is not None:
                target.sys_load_provider = sampler.snapshot
        except (ImportError, RuntimeError):
            pass

    def _cached_present_interval() -> float:
        if viewer is None:
            return 1.0 / 60.0
        provider = getattr(viewer, "cached_present_interval", None)
        if callable(provider):
            try:
                return max(0.004, min(0.05, float(provider())))
            except (TypeError, ValueError):
                pass
        return 1.0 / 60.0

    try:
        viewer = MetalLocalViewer(config)
        viewer.initialize()
        _configure_metrics(viewer)
        viewer.present_blank()
        print("[MetalLocalViewer] Metal local window initialized", flush=True)
        # Same-monitor ScreenCaptureKit must start after the Cocoa window
        # exists, otherwise its initial exclusion list is empty.
        if getattr(config, "on_capture_windows_initialized", None) is not None:
            config.on_capture_windows_initialized()
        while not shutdown_event.is_set():
            queue_timeout = 0.05
            if _has_cached_frame(viewer):
                queue_timeout = min(
                    queue_timeout,
                    max(0.0, next_cached_present_at - time.perf_counter()),
                )
            try:
                result, _started = runtime_q.get(timeout=queue_timeout)
            except queue.Empty:
                if viewer is not None:
                    now = time.perf_counter()
                    cached_present = getattr(viewer, "present_cached", None)
                    if callable(cached_present) and now >= next_cached_present_at:
                        present_started = now
                        presented = cached_present()
                        if presented is not False:
                            viewer.record_present(True)
                        if config.on_breakdown_add_time is not None:
                            config.on_breakdown_add_time(
                                "local_present", time.perf_counter() - present_started
                            )
                        if presented is not False and config.on_breakdown_inc is not None:
                            config.on_breakdown_inc("local_presented_frame", 1)
                            config.on_breakdown_inc("local_reused_presented", 1)
                        next_cached_present_at = time.perf_counter() + _cached_present_interval()
                    else:
                        viewer.poll_events()
                continue
            if config.on_breakdown_inc is not None:
                config.on_breakdown_inc("viewer_get", 1)
            # Media frames carry an ordered source timeline. Only desktop
            # latest-wins queues may discard queued results here; draining an
            # ordered media queue makes playback appear frozen or jump ahead.
            if not bool(getattr(runtime_q, "_d2s_ordered", False)):
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
                    getattr(result, "viewer_native", None) is not None
                    or getattr(result, "viewer_rgb", None) is not None
                    or getattr(result, "viewer_frame_np", None) is not None
                )
                else getattr(result, "sbs", None)
            )
            if frame is None:
                continue
            if viewer is None:
                viewer = MetalLocalViewer(config)
                viewer.initialize()
                _configure_metrics(viewer)
                print("[MetalLocalViewer] Metal local window initialized", flush=True)
            if getattr(config, "window_preview", False) and not preview_logged:
                preview_logged = True
                print(
                    "[MetalLocalViewer] Debug preview window not supported on the "
                    "Metal path yet; main output only",
                    flush=True,
                )
            present_started = time.perf_counter()
            presented = viewer.present(frame)
            presented_at = time.perf_counter()
            if presented is not False:
                debug_info = getattr(result, "debug_info", None)
                stage_timing: dict[str, float] = {}
                if isinstance(debug_info, dict):
                    for key in (
                        "capture_to_runtime_start_ms",
                        "capture_to_runtime_ready_ms",
                        "capture_wait_ms",
                        "sck_callback_to_runtime_ms",
                        "runtime_execution_ms",
                        "runtime_gpu_wait_ms",
                        "runtime_completion_ms",
                        "depth_preprocess_ms",
                        "depth_model_ms",
                        "depth_postprocess_ms",
                        "depth_total_ms",
                        "synthesis_ms",
                        "pack_ms",
                        "metal_synthesis_gpu_ms",
                        "metal_handoff_copy_ms",
                        "macos_metal_output_pack_ms",
                        "macos_metal_pack_gpu_ms",
                        "macos_metal_pack_host_handoff_ms",
                        "macos_metal_pack_async",
                        "macos_metal_pack_warp_threads",
                        "native_coreml_metal_vulkan_alias",
                        "native_coreml_ring_fallback",
                        "native_coreml_host_handoff_copy_count",
                        "native_coreml_gpu_copy_count",
                        "sck_presentation_time",
                        "sck_callback_time",
                        "capture_poll_returned_at",
                    ):
                        try:
                            stage_timing[key] = float(debug_info[key])
                        except (KeyError, TypeError, ValueError):
                            pass
                    ready_at = debug_info.get("viewer_ready_at")
                    try:
                        ready_at = float(ready_at)
                    except (TypeError, ValueError):
                        ready_at = None
                    if ready_at is not None:
                        stage_timing["viewer_ready_at"] = ready_at
                        stage_timing["viewer_queue_ms"] = max(
                            0.0, (present_started - ready_at) * 1000.0
                        )
                present_metrics = getattr(viewer, "_last_present_metrics", None)
                if isinstance(present_metrics, dict):
                    stage_timing.update(present_metrics)
                stage_timing["present_call_ms"] = max(
                    0.0, (presented_at - present_started) * 1000.0
                )
                stage_timing["presented_at"] = presented_at
                viewer.record_frame_latency(
                    _started,
                    stage_timing=stage_timing,
                    anchor_time=(
                        stage_timing.get("sck_callback_time")
                        or stage_timing.get("capture_poll_returned_at")
                    ),
                )
                viewer.record_present(False)
                if str(getattr(config, "capture_mode", "")).casefold() in {
                    "file", "video",
                }:
                    observe_media = getattr(viewer, "observe_media_presentation", None)
                    if callable(observe_media):
                        observe_media(result)
                    on_media_presented = getattr(config, "on_media_presented", None)
                    if callable(on_media_presented):
                        on_media_presented(
                            getattr(result, "debug_info", None) or {},
                            presented_at,
                        )
            if config.on_breakdown_add_time is not None:
                config.on_breakdown_add_time(
                    "local_present", time.perf_counter() - present_started
                )
            if presented is not False and config.on_breakdown_inc is not None:
                config.on_breakdown_inc("local_presented_frame", 1)
                if bool(getattr(result, "depth_complete", False)):
                    config.on_breakdown_inc("local_depth_presented_frame", 1)
            next_cached_present_at = time.perf_counter() + _cached_present_interval()
    except StopIteration:
        shutdown_event.set()
    finally:
        packer = getattr(runtime_q, "_d2s_packer", None)
        if packer is not None:
            # Stop native writes before the viewer unmaps its staging ring.
            packer.stop()
            from utils.queue_utils import clear_nonblocking

            clear_nonblocking(runtime_q)
        if viewer is not None:
            viewer.close()


def select_config_for_metal(config: Any) -> Any:
    """Metal viewer accepts the Vulkan config as-is; hook for future fields."""
    return replace(config) if False else config
