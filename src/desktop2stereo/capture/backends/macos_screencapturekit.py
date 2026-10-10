from __future__ import annotations

import os
import threading
import time
import ctypes
import numpy as np
import cv2

import objc
from Foundation import NSObject
from Quartz import CoreVideo as CV
from CoreMedia import (
    CMTimeMake,
    CMSampleBufferGetImageBuffer,
    CMSampleBufferGetSampleAttachmentsArray,
)
from AppKit import NSScreen
from ..geometry import match_mss_monitor_to_rect

# Optional Metal zero-copy path (Milestone 2)
try:
    import Metal
    _metal_device = None
    _cv_metal_cache = None

    def _get_metal_device():
        global _metal_device
        if _metal_device is None:
            _metal_device = Metal.MTLCreateSystemDefaultDevice()
        return _metal_device

    def _get_cv_metal_cache():
        global _cv_metal_cache
        if _cv_metal_cache is not None:
            return _cv_metal_cache
        device = _get_metal_device()
        if device is None:
            return None
        # pyobjc 12.x returns (err, cache) via the bridge's out-param
        # metadata; objc.Variable does not exist in this version.
        res = CV.CVMetalTextureCacheCreate(None, None, device, None, None)
        err, cache = res if isinstance(res, tuple) else (res, None)
        if err == 0 and cache is not None:  # kCVReturnSuccess
            _cv_metal_cache = cache
        return _cv_metal_cache
except Exception:
    Metal = None
    _metal_device = None
    _cv_metal_cache = None

    def _get_metal_device():
        return None

    def _get_cv_metal_cache():
        return None

# Load ScreenCaptureKit framework
objc.loadBundle('ScreenCaptureKit', globals(),
    bundle_path=objc.pathForFramework('/System/Library/Frameworks/ScreenCaptureKit.framework'))
import ScreenCaptureKit as SCK

# Module-level cache for shareable content
_sck_content_cache = None
_sck_content_cache_time = 0.0
_SCK_CACHE_TTL = 2.0

def _sck_get_shareable_content(force=False):
    global _sck_content_cache, _sck_content_cache_time
    now = time.time()
    if not force and _sck_content_cache is not None and (now - _sck_content_cache_time) < _SCK_CACHE_TTL:
        return _sck_content_cache

    done = threading.Event()
    result = {}
    def _handler(content, error):
        result['content'] = content
        result['error'] = error
        done.set()

    SCK.SCShareableContent.getShareableContentWithCompletionHandler_(_handler)
    if not done.wait(timeout=10.0):
        raise RuntimeError("Timed out waiting for shareable content")
    if result.get('error'):
        raise RuntimeError(f"Failed to get shareable content: {result['error']}")

    _sck_content_cache = result['content']
    _sck_content_cache_time = now
    return _sck_content_cache

def _sck_find_window(title):
    content = _sck_get_shareable_content()
    windows = content.windows()
    for w in windows:
        wt = w.title()
        if wt is None:
            continue
        if wt == title:
            return w
        owner = w.owningApplication()
        if owner is not None:
            app_name = owner.applicationName()
            if app_name == title:
                return w
    return None

def get_window_info_mac(window_title):
    win = _sck_find_window(window_title)
    if win is None:
        return None
    frame = win.frame()
    return {
        "window_id": int(win.windowID()),
        "left": int(frame.origin.x),
        "top": int(frame.origin.y),
        "width": int(frame.size.width),
        "height": int(frame.size.height),
    }

def get_window_client_bounds_mac(window_title):
    info = get_window_info_mac(window_title)
    if info is None:
        return None, None, None, None
    return info["left"], info["top"], info["width"], info["height"]


class _OwnedSCKFrame:
    """Own a CVPixelBuffer + its CVMetalTexture for zero-copy sampling.

    Holds +1 on both objects; ``release()`` (or GC via ``__del__``) frees
    them, so frames dropped anywhere along the queue chain cannot leak.
    """

    __slots__ = ("texture", "pixel_buffer", "frame_status", "_released")

    def __init__(self, cv_texture, pixel_buffer, frame_status=None):
        self.texture = cv_texture
        self.pixel_buffer = pixel_buffer
        self.frame_status = frame_status
        self._released = False
        CV.CVPixelBufferRetain(pixel_buffer)
        if cv_texture is not None:
            cv_texture.retain()

    def mtl_texture(self):
        """Return the live MTLTexture (BGRA8Unorm), or None after release."""
        if self._released or self.texture is None:
            return None
        return CV.CVMetalTextureGetTexture(self.texture)

    def to_cpu_frame(self):
        """Materialize BGRA bytes only for an explicit native-path fallback."""
        if self._released or self.pixel_buffer is None:
            raise RuntimeError("ScreenCaptureKit frame is no longer available")
        w = CV.CVPixelBufferGetWidth(self.pixel_buffer)
        h = CV.CVPixelBufferGetHeight(self.pixel_buffer)
        bpr = CV.CVPixelBufferGetBytesPerRow(self.pixel_buffer)
        size = bpr * h
        CV.CVPixelBufferLockBaseAddress(self.pixel_buffer, 0)
        try:
            address = CV.CVPixelBufferGetBaseAddress(self.pixel_buffer)
            frame = np.frombuffer(address.as_buffer(size), dtype=np.uint8).reshape(h, bpr)
            if bpr != w * 4:
                return np.ascontiguousarray(frame[:, : w * 4].reshape(h, w, 4))
            return frame.reshape(h, w, 4).copy()
        finally:
            CV.CVPixelBufferUnlockBaseAddress(self.pixel_buffer, 0)

    def release(self):
        if self._released:
            return
        self._released = True
        try:
            if self.texture is not None:
                self.texture.release()
        except Exception:
            pass
        try:
            if self.pixel_buffer is not None:
                CV.CVPixelBufferRelease(self.pixel_buffer)
        except Exception:
            pass

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass


class _SCKFrameReceiver(NSObject):
    def init(self):
        self = objc.super(_SCKFrameReceiver, self).init()
        if self is None:
            return None
        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_mtl_texture = None
        self._latest_owned = None
        self._latest_texture_size = (0, 0)
        self._frame_count = 0
        self._frame_status_counts = {}
        self._texture_diag_logged = False
        self._condition = threading.Condition(self._lock)
        return self

    def stream_didOutputSampleBuffer_ofType_(self, stream, sampleBuffer, outputType):
        if outputType != 0:
            return
        try:
            imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer)
            if imageBuffer is None:
                return
            frame_status = None
            try:
                attachments = CMSampleBufferGetSampleAttachmentsArray(sampleBuffer, False)
                if attachments:
                    status = attachments[0].get(SCK.SCStreamFrameInfoStatus)
                    if status is not None:
                        frame_status = int(status)
            except Exception:
                # Status is diagnostic only; an OS-version-specific attachment
                # shape must never discard an otherwise valid IOSurface.
                frame_status = None
            with self._condition:
                status_key = (
                    str(frame_status)
                    if frame_status is not None
                    else "unknown"
                )
                self._frame_status_counts[status_key] = (
                    self._frame_status_counts.get(status_key, 0) + 1
                )

            # Zero-copy Metal texture path (survey doc milestone 5): every
            # frame is wrapped as an owned CVPixelBuffer+CVMetalTexture pair
            # that downstream stages (warp viewer) can sample directly,
            # skipping the 8MB device->host->device color round-trip.
            # D2S_SCK_ZEROCOPY_TEX=0 reverts to v2.5 parity (CPU readout only;
            # the old D2S_SCK_METAL_TEXTURE_DIAG=1 diagnostic still prints).
            zc_enabled = os.environ.get("D2S_SCK_ZEROCOPY_TEX", "1") != "0"
            native_only = os.environ.get("D2S_SCK_NATIVE_ONLY", "0").strip().lower() in {
                "1", "true", "yes", "on"
            }
            diag_enabled = os.environ.get("D2S_SCK_METAL_TEXTURE_DIAG") == "1"
            if native_only and zc_enabled:
                # Native Core ML creates the CVMetalTexture from this same
                # IOSurface at consumption time. Avoid wrapping it once here
                # and again in the bridge on the SCK callback thread.
                owned = _OwnedSCKFrame(None, imageBuffer, frame_status)
                replaced = None
                with self._condition:
                    replaced = self._latest_owned
                    self._latest_owned = owned
                    self._latest_texture_size = (
                        CV.CVPixelBufferGetWidth(imageBuffer),
                        CV.CVPixelBufferGetHeight(imageBuffer),
                    )
                    self._frame_count += 1
                    self._condition.notify_all()
                if replaced is not None:
                    replaced.release()
                return
            if (
                (zc_enabled or (diag_enabled and self._frame_count == 0))
                and _get_metal_device() is not None
                and _get_cv_metal_cache() is not None
            ):
                try:
                    # Retain pixel buffer for texture lifetime
                    CV.CVPixelBufferRetain(imageBuffer)
                    try:
                        cache = _get_cv_metal_cache()
                        w = CV.CVPixelBufferGetWidth(imageBuffer)
                        h = CV.CVPixelBufferGetHeight(imageBuffer)
                        # BGRA8Unorm == kCVPixelFormatType_32BGRA.
                        # pyobjc 12.x: out-param returns as (err, cvtex).
                        res = CV.CVMetalTextureCacheCreateTextureFromImage(
                            None, cache, imageBuffer, None,
                            Metal.MTLPixelFormatBGRA8Unorm, w, h, 0, None
                        )
                        err, cv_tex = (
                            res if isinstance(res, tuple) else (res, None)
                        )
                        if err == 0 and cv_tex is not None:
                            mtl_tex = CV.CVMetalTextureGetTexture(cv_tex)
                            if mtl_tex is not None:
                                if not self._texture_diag_logged and (
                                    zc_enabled or diag_enabled
                                ):
                                    self._texture_diag_logged = True
                                    print(
                                        "[ScreenCaptureKit] Metal zero-copy texture OK: "
                                        f"{w}x{h} BGRA8Unorm via CVMetalTextureCache "
                                        "(no CPU base-address read)",
                                        flush=True,
                                    )
                                if zc_enabled:
                                    owned = _OwnedSCKFrame(cv_tex, imageBuffer, frame_status)
                                    replaced = None
                                    with self._condition:
                                        replaced = self._latest_owned
                                        self._latest_owned = owned
                                        self._latest_texture_size = (w, h)
                                    # Release AFTER the swap so a concurrent
                                    # taker never sees a freed ref.
                                    if replaced is not None:
                                        replaced.release()
                                else:
                                    cv_tex.release()
                    finally:
                        CV.CVPixelBufferRelease(imageBuffer)
                except Exception as exc:
                    # Never silent again: the pyobjc out-param convention
                    # changed under us once and zero-copy died invisibly.
                    if not getattr(self, "_zc_err_logged", False):
                        self._zc_err_logged = True
                        print(
                            f"[ScreenCaptureKit] zero-copy texture wrap failed "
                            f"(once): {type(exc).__name__}: {exc}",
                            flush=True,
                        )

            # The native Core ML bridge consumes the retained IOSurface
            # directly. Do not map/read the same 8 MB frame on the CPU when
            # the zero-copy lease was established successfully. If the
            # bridge later rejects this frame, it can call to_cpu_frame().
            w = CV.CVPixelBufferGetWidth(imageBuffer)
            h = CV.CVPixelBufferGetHeight(imageBuffer)
            bpr = CV.CVPixelBufferGetBytesPerRow(imageBuffer)
            size = bpr * h

            CV.CVPixelBufferLockBaseAddress(imageBuffer, 0)
            try:
                varlist = CV.CVPixelBufferGetBaseAddress(imageBuffer)
                buf = varlist.as_buffer(size)
                frame = np.frombuffer(buf, dtype=np.uint8).reshape(h, bpr)
                if bpr != w * 4:
                    frame = np.ascontiguousarray(frame[:, :w*4].reshape(h, w, 4))
                else:
                    frame = frame.reshape(h, w, 4).copy()
            finally:
                CV.CVPixelBufferUnlockBaseAddress(imageBuffer, 0)

            with self._condition:
                self._latest_frame = frame
                self._frame_count += 1
                self._condition.notify_all()
            try:
                from utils.residency import mark as _rz_mark

                _rz_mark("capture_frame(host bytes)", frame, "SCK contract")
            except Exception:
                pass
        except Exception:
            pass

    def stream_didStopWithError_(self, stream, error):
        if error is not None:
            print(f"[ScreenCaptureKit] Stream stopped with error: {error}")

    def get_latest_frame(self, timeout=0.1):
        """Return the newest frame, transferring ownership instead of copying.

        Each SCK callback allocates a fresh contiguous array and replaces the
        slot atomically under the lock, so the returned array is never written
        again by this receiver. The extra defensive `.copy()` cost a full-frame
        memcpy per frame; D2S_SCK_LATEST_FRAME_COPY=1 restores it.
        """
        defensive_copy = os.environ.get("D2S_SCK_LATEST_FRAME_COPY") == "1"
        native_only = os.environ.get("D2S_SCK_NATIVE_ONLY", "0").strip().lower() in {
            "1", "true", "yes", "on"
        }
        with self._condition:
            if (
                self._latest_frame is None
                and (not native_only or self._latest_owned is None)
                and timeout > 0
            ):
                self._condition.wait(timeout=timeout)
            # Native-only callers take ownership through take_latest_zero_copy()
            # immediately after this wake-up. Do not expose or copy a CPU frame.
            if native_only:
                return None
            if self._latest_frame is not None:
                if defensive_copy:
                    return self._latest_frame.copy()
                frame = self._latest_frame
                # Detach the slot so a blocked consumer cannot observe a frame
                # twice while a newer one has already replaced it.
                self._latest_frame = None
                return frame
            return None

    def get_latest_mtl_texture(self):
        """Zero-copy path diagnostic: return (MTLTexture, (w,h)) without CPU read."""
        with self._condition:
            return self._latest_mtl_texture, self._latest_texture_size

    def take_latest_zero_copy(self, timeout=0.0):
        """Transfer the newest owned zero-copy frame, or None.

        Ownership of the +1 refs moves to the caller; it must call
        ``release()`` when done drawing (or let GC do it).
        """
        with self._condition:
            if self._latest_owned is None and timeout > 0.0:
                deadline = time.monotonic() + float(timeout)
                while self._latest_owned is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        break
                    self._condition.wait(timeout=remaining)
            owned = self._latest_owned
            self._latest_owned = None
        return owned

    @property
    def frame_count(self):
        return self._frame_count

    @property
    def frame_status_counts(self):
        with self._condition:
            return dict(self._frame_status_counts)

class DesktopGrabber:
    def __init__(self, output_resolution=1080, fps=60, window_title=None,
                capture_mode="Monitor", monitor_index=1, with_cursor=True,
                fps_provider=None):
        # OUTPUT_RESOLUTION may be an int or a (width, height) tuple when
        # "Processing Resolution: Auto"; the grabber needs a target height.
        if isinstance(output_resolution, (tuple, list)):
            output_resolution = int(output_resolution[1])
        self.scaled_height = int(output_resolution)
        self.fps = max(1, min(240, int(fps)))
        self._fps_provider = fps_provider
        self.with_cursor = with_cursor
        # Native ScreenCaptureKit pixel format handed to consumers by grab().
        self.frame_format = "bgra"
        self.window_title = window_title
        self.capture_mode = capture_mode
        self._stream = None
        self._receiver = None
        self._stream_config = None
        self._last_fps_update_request = self.fps
        self._fps_update_error_logged = False
        self._last_frame = None
        self._display = None
        self._window = None
        self.left = 0
        self.top = 0
        self.width = 0
        self.height = 0

        content = _sck_get_shareable_content()
        displays = content.displays()

        if not displays or len(displays) == 0:
            raise RuntimeError(
                "No displays available via ScreenCaptureKit. "
                "Grant Screen Recording permission to Terminal/Python in "
                "System Settings > Privacy & Security > Screen Recording, "
                "then try again."
            )

        if self.capture_mode == "Monitor":
            import mss

            requested_index = int(monitor_index)
            with mss.mss() as capture:
                monitors = capture.monitors
            display_rects = []
            for display in displays:
                frame = display.frame()
                display_rects.append((
                    frame.origin.x,
                    frame.origin.y,
                    frame.size.width,
                    frame.size.height,
                ))
            display_position = match_mss_monitor_to_rect(
                requested_index, monitors, display_rects
            )
            if display_position is None:
                raise RuntimeError(
                    f"MSS monitor {requested_index} could not be matched to a "
                    "ScreenCaptureKit display by bounds"
                )
            self._display = displays[display_position]
            df = self._display.frame()
            print(
                f"[ScreenCaptureKit] MSS monitor {requested_index} -> "
                f"display ID {self._display.displayID()}"
            )
            self.left = int(df.origin.x)
            self.top = int(df.origin.y)
            self.width = self._display.width()
            self.height = self._display.height()
        else:
            self._window = _sck_find_window(self.window_title)
            if self._window is None:
                raise RuntimeError(f"Window '{self.window_title}' not found via ScreenCaptureKit")

            wf = self._window.frame()
            for d in displays:
                df = d.frame()
                if (df.origin.x <= wf.origin.x < df.origin.x + df.size.width and
                    df.origin.y <= wf.origin.y < df.origin.y + df.size.height):
                    self._display = d
                    break
            if self._display is None:
                self._display = displays[0]

            self.left = int(wf.origin.x)
            self.top = int(wf.origin.y)
            self.width = int(wf.size.width)
            self.height = int(wf.size.height)

        self._start_stream()

    def _start_stream(self):
        if self.capture_mode == "Monitor":
            filt = SCK.SCContentFilter.alloc().initWithDisplay_excludingWindows_(
                self._display, [])
        else:
            filt = SCK.SCContentFilter.alloc().initWithDisplay_includingWindows_(
                self._display, [self._window])

        config = SCK.SCStreamConfiguration.alloc().init()
        config.setWidth_(self.width)
        config.setHeight_(self.height)
        config.setShowsCursor_(self.with_cursor)
        config.setPixelFormat_(CV.kCVPixelFormatType_32BGRA)
        config.setMinimumFrameInterval_(CMTimeMake(1, max(1, self.fps)))
        # Keep the SCK producer low-latency. A bounded queue prevents retained
        # IOSurfaces from back-pressuring the display stream. The default
        # depth 3 is the ScreenCaptureKit setting verified on this host;
        # lower values remain an explicit diagnostic override.
        if hasattr(config, "setQueueDepth_"):
            try:
                queue_depth = max(
                    1, min(8, int(os.environ.get("D2S_SCK_QUEUE_DEPTH", "3")))
                )
            except (TypeError, ValueError):
                queue_depth = 3
            config.setQueueDepth_(queue_depth)

        self._receiver = _SCKFrameReceiver.alloc().init()
        self._stream = SCK.SCStream.alloc().initWithFilter_configuration_delegate_(
            filt, config, self._receiver)
        self._stream_config = config

        success, error = self._stream.addStreamOutput_type_sampleHandlerQueue_error_(
            self._receiver, 0, None, None)
        if not success:
            raise RuntimeError(f"Failed to add stream output: {error}")

        done = threading.Event()
        start_result = {}
        def _on_start(error):
            start_result['error'] = error
            done.set()

        self._stream.startCaptureWithCompletionHandler_(_on_start)
        if not done.wait(timeout=10.0):
            raise RuntimeError("Timed out waiting for capture to start")
        if start_result.get('error'):
            raise RuntimeError(f"Failed to start capture: {start_result['error']}")

        self._receiver.get_latest_frame(timeout=2.0)

    def _sync_capture_fps(self):
        provider = self._fps_provider
        if not callable(provider):
            return
        try:
            target_fps = max(1, min(240, int(provider())))
        except (TypeError, ValueError, RuntimeError):
            return
        if target_fps == self._last_fps_update_request:
            return
        self._last_fps_update_request = target_fps
        stream = self._stream
        if stream is None:
            return
        config = SCK.SCStreamConfiguration.alloc().init()
        config.setWidth_(self.width)
        config.setHeight_(self.height)
        config.setShowsCursor_(self.with_cursor)
        config.setPixelFormat_(CV.kCVPixelFormatType_32BGRA)
        config.setMinimumFrameInterval_(CMTimeMake(1, target_fps))
        try:
            if hasattr(config, "setQueueDepth_"):
                try:
                    queue_depth = max(
                        1,
                        min(8, int(os.environ.get("D2S_SCK_QUEUE_DEPTH", "3"))),
                    )
                except (TypeError, ValueError):
                    queue_depth = 3
                config.setQueueDepth_(queue_depth)
            self._stream_config = config

            def _on_update(error):
                if error is not None:
                    if not self._fps_update_error_logged:
                        self._fps_update_error_logged = True
                        print(
                            "[ScreenCaptureKit] Adaptive frame interval update failed; "
                            f"continuing with the previous capture rate: {error}",
                            flush=True,
                        )
                    return
                self.fps = target_fps
                print(
                    f"[ScreenCaptureKit] Adaptive capture target applied: {target_fps} FPS",
                    flush=True,
                )

            stream.updateConfiguration_completionHandler_(config, _on_update)
        except Exception as exc:
            if not self._fps_update_error_logged:
                self._fps_update_error_logged = True
                print(
                    "[ScreenCaptureKit] Adaptive frame interval update unavailable; "
                    f"continuing with the previous capture rate: {exc}",
                    flush=True,
                )

    def _update_window_filter(self):
        if self.capture_mode != "Window":
            return

        win = _sck_find_window(self.window_title)
        if win is None:
            return

        wf = win.frame()
        nl, nt = int(wf.origin.x), int(wf.origin.y)
        nw, nh = int(wf.size.width), int(wf.size.height)

        if nl == self.left and nt == self.top and nw == self.width and nh == self.height:
            return

        self.left, self.top = nl, nt
        self.width, self.height = nw, nh
        self._window = win

        fid = SCK.SCContentFilter.alloc().initWithDisplay_includingWindows_(
            self._display, [win])
        done = threading.Event()
        self._stream.updateContentFilter_completionHandler_(fid, lambda e: done.set())
        done.wait(timeout=3.0)

    def grab(self, output_format="bgra"):
        """Return the newest captured frame.

        Defaults to ``bgra`` so the ScreenCaptureKit native format passes
        through without a per-frame ``cv2.cvtColor``; the tensor preprocess
        path (capture.preprocess) accepts 4-channel BGRA directly. ``bgr``
        remains available for CPU consumers.
        """
        self._update_window_filter()

        frame = self._receiver.get_latest_frame(timeout=1.0 / max(1, self.fps))

        if frame is None:
            if os.environ.get("D2S_SCK_NATIVE_ONLY", "0").strip().lower() in {
                "1", "true", "yes", "on"
            }:
                return None, (int(self.width), int(self.height))
            if self._last_frame is not None:
                return self._last_frame.copy(), self.scaled_height
            h = self.scaled_height
            w = int(h * self.width / max(1, self.height))
            channels = 4 if output_format == "bgra" else 3
            return np.zeros((h, w, channels), dtype=np.uint8), self.scaled_height

        self._last_frame = frame

        if output_format == "bgra":
            return frame, self.scaled_height
        elif output_format == "bgr":
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR), self.scaled_height
        else:
            raise ValueError("output_format must be 'bgr' or 'bgra'")

    def take_latest_zero_copy(self):
        """Transfer the newest owned zero-copy frame (or None).

        Only valid between start() and stop(); returns None otherwise.
        """
        receiver = self._receiver
        if receiver is None:
            return None
        return receiver.take_latest_zero_copy()

    def grab_native_zero_copy(self, timeout=0.0):
        """Atomically wait for and take the newest retained IOSurface."""
        self._sync_capture_fps()
        self._update_window_filter()
        receiver = self._receiver
        if receiver is None:
            return None, (int(self.width), int(self.height))
        owned = receiver.take_latest_zero_copy(timeout=timeout)
        return owned, (int(self.width), int(self.height))

    @property
    def frame_status_counts(self):
        receiver = self._receiver
        if receiver is None:
            return {}
        return receiver.frame_status_counts

    def stop(self):
        if self._stream is not None:
            done = threading.Event()
            self._stream.stopCaptureWithCompletionHandler_(lambda e: done.set())
            done.wait(timeout=5.0)
            self._stream = None
        self._receiver = None
        self._last_frame = None
