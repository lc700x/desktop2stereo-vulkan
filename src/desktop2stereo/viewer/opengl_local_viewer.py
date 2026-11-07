"""OpenGL fallback viewer (3rd tier after Vulkan -> Metal).

Uses glfw + PyOpenGL to present SBS frames when neither Vulkan nor Metal
are available. This is the same path validated by tests/test_opengl_fallback.py
and tools/opengl_fallback_smoke.py (PBO + FBO + fence).
"""
from __future__ import annotations

import time
import sys
from typing import Any

from viewer.window_control import hide_window_from_capture


class OpenGLLocalViewer:
    """Minimal OpenGL viewer for fallback chain.

    Interface matches VulkanLocalViewer/MetalLocalViewer:
      initialize() / present(frame) / poll_events() / close()
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self.glfw = None
        self.window = None
        self._backend = None
        self._initialized = False
        self._show_after_capture_protection = False

    def initialize(self) -> None:
        try:
            import glfw
        except Exception as exc:
            raise RuntimeError(f"glfw unavailable for OpenGL fallback: {exc}") from exc

        self.glfw = glfw
        if not glfw.init():
            raise RuntimeError("GLFW init failed for OpenGL fallback")

        glfw.default_window_hints()
        # OpenGL context for fallback path
        glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_API)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, True)
        self._show_after_capture_protection = bool(
            sys.platform == "win32" and self.config.exclude_from_capture
        )
        if self._show_after_capture_protection:
            glfw.window_hint(glfw.VISIBLE, False)

        w, h = int(self.config.window_width), int(self.config.window_height)
        self.window = glfw.create_window(w, h, self.config.title, None, None)
        if not self.window:
            raise RuntimeError("OpenGL fallback: could not create GLFW window")

        if self._show_after_capture_protection:
            hide_window_from_capture(self.window)
            print(
                "[OpenGLLocalViewer] using global capture exclusion; "
                "remote desktop capture will not show SBS",
                flush=True,
            )

        glfw.make_context_current(self.window)

        # Lazy import OpenGL backend (validates PBO/FBO/fence support)
        try:
            from streaming.opengl_stream_backend import OpenGLFallbackBackend

            self._backend = OpenGLFallbackBackend(w, h)
            print(
                f"[OpenGLLocalViewer] OpenGL fallback active "
                f"({self._backend.capabilities.context_api} "
                f"interopt={self._backend.capabilities.interop_mode})",
                flush=True,
            )
        except Exception as exc:
            try:
                glfw.destroy_window(self.window)
            except Exception:
                pass
            self.window = None
            raise RuntimeError(f"OpenGL backend init failed: {exc}") from exc

        self._initialized = True
        print("[OpenGLLocalViewer] OpenGL fallback window initialized", flush=True)

    def present(self, frame: Any) -> bool:
        if not self._initialized or self.window is None:
            raise RuntimeError("OpenGL viewer not initialized")
        self.poll_events()
        # frame is SBS RGBA bytes or HWC uint8 numpy; delegate to backend
        # which handles PBO upload + FBO blit. For torch tensors, convert via
        # frame_to_rgba_bytes first (caller already does for Vulkan path; for
        # fallback we accept either).
        try:
            import numpy as np

            if hasattr(frame, "numpy"):
                frame = frame.numpy()
            arr = np.asarray(frame)
            if arr.ndim == 4:
                arr = arr[0]
            if arr.ndim == 3 and arr.shape[0] in (3, 4):
                arr = np.moveaxis(arr, 0, -1)
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            # Ensure RGBA
            if arr.shape[-1] == 3:
                alpha = np.full(arr.shape[:2] + (1,), 255, dtype=np.uint8)
                arr = np.concatenate((arr, alpha), axis=-1)
            self._backend.submit_rgb(arr)
            # Present via glfw swap
            self.glfw.swap_buffers(self.window)
            if self._show_after_capture_protection:
                self.glfw.show_window(self.window)
                self._show_after_capture_protection = False
            return True
        except Exception as exc:
            # Non-fatal per-frame errors should not kill viewer
            print(f"[OpenGLLocalViewer] present failed: {exc}", flush=True)
            return False

    def poll_events(self) -> None:
        if self.glfw and self.window:
            self.glfw.poll_events()
            if self.glfw.window_should_close(self.window):
                raise StopIteration

    def close(self) -> None:
        try:
            if self._backend is not None:
                self._backend.close()
        finally:
            self._backend = None
        try:
            if self.window and self.glfw:
                self.glfw.destroy_window(self.window)
        finally:
            self.window = None
            self._initialized = False
