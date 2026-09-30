from __future__ import annotations

import ctypes
import os
import threading
import time
from ctypes import wintypes
from dataclasses import replace

from capture.types import FrameCopyMode, capture_frame_from_raw, native_resource_contract


CAPTURE_CURSOR_DELAY_S = 0.2
CAPTURE_GAP_LOG_LIMIT = 3
_PENDING_CAPTURE_GAP_LOGS = []
_PENDING_CAPTURE_GAP_LOCK = threading.Lock()
_CAPTURE_GAP_DEFER_RELEASED = False


def _env_bool(name):
    value = os.environ.get(name)
    if value is None:
        return None
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _env_int(name):
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return None
    return int(value)


def _emit_capture_gap_log(line: str) -> None:
    print(line, flush=True)


def flush_pending_capture_gap_logs() -> None:
    global _CAPTURE_GAP_DEFER_RELEASED
    with _PENDING_CAPTURE_GAP_LOCK:
        pending = list(_PENDING_CAPTURE_GAP_LOGS)
        _PENDING_CAPTURE_GAP_LOGS.clear()
        _CAPTURE_GAP_DEFER_RELEASED = True
    for line in pending:
        _emit_capture_gap_log(line)


def _defer_capture_gap_log(line: str) -> bool:
    if not _env_bool("D2S_DEFER_CAPTURE_GAP_UNTIL_OPENXR_PROJECTION"):
        return False
    with _PENDING_CAPTURE_GAP_LOCK:
        if _CAPTURE_GAP_DEFER_RELEASED:
            return False
        _PENDING_CAPTURE_GAP_LOGS.append(line)
    return True


def _fps_to_minimum_update_interval_ms(fps):
    try:
        fps_value = int(fps)
    except (TypeError, ValueError):
        return None
    if fps_value <= 0:
        return None
    return max(1, int(round(1000.0 / fps_value)))


def _software_throttle_enabled(capture_tool):
    override = _env_bool("D2S_WGC_SOFTWARE_THROTTLE")
    if override is not None:
        return bool(override)
    return capture_tool == "WindowsCaptureCUDA"


def _windows_capture_kwargs(config, capture_tool):
    if config.capture_mode == "Window":
        kwargs = {"window_name": config.window_title}
    else:
        kwargs = {"monitor_index": config.monitor_index}
    if capture_tool == "WindowsCaptureCUDA":
        optional = {
            "minimum_update_interval": _fps_to_minimum_update_interval_ms(getattr(config, "fps", None)),
            "reuse_output_buffer": _env_bool("D2S_WGC_REUSE_OUTPUT_BUFFER"),
            "output_buffer_count": _env_int("D2S_WGC_OUTPUT_BUFFER_COUNT"),
        }
        kwargs.update({key: value for key, value in optional.items() if value is not None})
    return kwargs


def _load_windows_capture(capture_tool):
    if capture_tool == "WindowsCaptureROCm":
        from wc_rocm import WindowsCapture, Frame, InternalCaptureControl
    elif capture_tool == "WindowsCaptureCUDA":
        from wc_cuda import WindowsCapture, Frame, InternalCaptureControl
    else:
        from windows_capture import WindowsCapture, Frame, InternalCaptureControl
    return WindowsCapture, Frame, InternalCaptureControl



def _event_capture_device(capture_tool):
    if capture_tool == "WindowsCaptureCUDA":
        return "cuda"
    if capture_tool == "WindowsCaptureROCm":
        return "rocm"
    return "cpu"


def _copy_frame_buffer(frame_buffer, capture_tool):
    device = _event_capture_device(capture_tool)
    if device in ("cuda", "rocm") and not _env_bool("D2S_WGC_COPY_FRAME_BUFFER"):
        return frame_buffer, FrameCopyMode.GPU_TENSOR, device
    prefer_clone = device in ("cuda", "rocm")
    if prefer_clone and hasattr(frame_buffer, "clone"):
        return frame_buffer.clone(), FrameCopyMode.CLONE, device
    if hasattr(frame_buffer, "copy"):
        return frame_buffer.copy(), FrameCopyMode.COPY, device
    return frame_buffer.clone(), FrameCopyMode.CLONE, device


def _borrow_native_resource(frame_buffer):
    """Borrow an exposed WGC/D3D11 resource without changing CPU compatibility.

    windows-capture packages differ by release. Some expose a D3D11 texture
    through a property or accessor while older releases expose only a CPU
    frame buffer. The resource is retained in CapturedFrame for a downstream
    consumer and is never reported as zero-copy until that consumer verifies
    the same adapter, format and synchronization contract.
    """
    for attribute in (
        "d3d11_texture",
        "native_texture",
        "texture",
        "resource",
        "get_d3d11_texture",
        "get_native_texture",
    ):
        try:
            value = getattr(frame_buffer, attribute, None)
            value = value() if callable(value) else value
        except Exception:
            continue
        if value is None or value is frame_buffer:
            continue
        kind = str(
            getattr(value, "resource_kind", "")
            or getattr(value, "kind", "")
            or type(value).__name__
        ).lower()
        if any(token in kind for token in ("d3d", "texture", "gpu", "dxgi")):
            return value
    return None


def _setup_dpi_awareness():
    from windows_dpi import set_per_monitor_dpi_v2

    set_per_monitor_dpi_v2()


class WindowsCaptureEventRunner:
    def __init__(self, config):
        self.config = config
        self.capture_tool = config.capture_tool or "WindowsCapture"
        self._capture_started_event = threading.Event()
        self._keyboard_thread = None
        self._session = None
        self._control = None
        self._fps_last_log = 0.0
        self._fps_frames = 0
        self._fps_copy_seconds = 0.0
        self._fps_enqueue_seconds = 0.0
        self._fps_handler_seconds = 0.0
        self._last_frame_ts = 0.0
        self._capture_gap_logs = 0
        self._software_frame_due = 0.0
        self._software_pacing_fps = 0
        self._software_limited_frames = 0
        self._replay_lock = threading.Lock()
        self._replay_frame = None
        self._last_frame_emit_ts = 0.0
        self._replay_thread = None

    @property
    def session(self):
        return self._session

    @property
    def control(self):
        return self._control

    def stop(self):
        if self._control is not None:
            self._control.stop()
        elif self._session is not None and hasattr(self._session, "stop"):
            self._session.stop()

    def _accept_software_paced_frame(self, now: float) -> bool:
        if not _software_throttle_enabled(self.capture_tool):
            return True
        try:
            provider = getattr(self.config, "fps_provider", None)
            fps = int(provider() if provider is not None else self.config.fps)
        except (TypeError, ValueError):
            return True
        if fps <= 0:
            return True

        interval = 1.0 / float(fps)
        if fps != self._software_pacing_fps:
            self._software_pacing_fps = fps
            self._software_frame_due = 0.0
        if self._software_frame_due <= 0.0:
            self._software_frame_due = now + interval
            return True

        # Accept callbacks that land slightly before the ideal grid point. This
        # keeps a nominal 120 Hz source at 60 Hz instead of occasionally
        # rejecting both neighboring callbacks because of timer jitter.
        tolerance = min(0.001, interval * 0.1)
        if now < self._software_frame_due - tolerance:
            self._software_limited_frames += 1
            return False

        if now - self._software_frame_due > interval:
            self._software_frame_due = now + interval
        else:
            self._software_frame_due += interval
        return True

    def _target_fps(self) -> int:
        try:
            provider = getattr(self.config, "fps_provider", None)
            return max(1, int(provider() if provider is not None else self.config.fps))
        except (TypeError, ValueError):
            return max(1, int(self.config.fps or 60))

    def _remember_emitted_frame(self, captured_frame, now: float) -> None:
        with self._replay_lock:
            self._replay_frame = captured_frame
            self._last_frame_emit_ts = float(now)

    def _replay_frame_if_due(self, now: float):
        with self._replay_lock:
            captured_frame = self._replay_frame
            if captured_frame is None:
                return None
            interval = 1.0 / float(self._target_fps())
            if float(now) - self._last_frame_emit_ts + 1e-9 < interval:
                return None
            self._last_frame_emit_ts = float(now)
            metadata = dict(captured_frame.metadata)
            metadata["replayed_static_frame"] = True
            return replace(captured_frame, timestamp=float(now), metadata=metadata)

    def _start_static_replay_worker(
        self,
        *,
        shutdown_event,
        on_frame,
        is_paused=None,
        is_hard_idle=None,
    ) -> None:
        def replay_worker():
            while not shutdown_event.is_set():
                interval = 1.0 / float(self._target_fps())
                if shutdown_event.wait(min(0.02, max(0.001, interval * 0.25))):
                    break
                if (is_hard_idle is not None and is_hard_idle()) or (
                    is_paused is not None and is_paused()
                ):
                    continue
                replay_frame = self._replay_frame_if_due(time.perf_counter())
                if replay_frame is not None:
                    on_frame(replay_frame)

        self._replay_thread = threading.Thread(
            target=replay_worker,
            name="CaptureStaticReplay",
            daemon=True,
        )
        self._replay_thread.start()

    def _log_capture_fps(self, now: float) -> None:
        if self.capture_tool != "WindowsCaptureCUDA":
            return
        if not _env_bool("D2S_WGC_CAPTURE_FPS_LOG"):
            return
        if self._fps_last_log <= 0.0:
            self._fps_last_log = now
            self._fps_frames = 0
            return
        self._fps_frames += 1
        elapsed = now - self._fps_last_log
        if elapsed < 1.0:
            return
        fps = self._fps_frames / elapsed
        copy_ms = self._fps_copy_seconds * 1000.0 / max(self._fps_frames, 1)
        enqueue_ms = self._fps_enqueue_seconds * 1000.0 / max(self._fps_frames, 1)
        handler_ms = self._fps_handler_seconds * 1000.0 / max(self._fps_frames, 1)
        print(
            f"[WindowsCaptureCUDA] capture_fps={fps:.1f} frames={self._fps_frames} "
            f"monitor={self.config.monitor_index} mode={self.config.capture_mode} "
            f"copy_ms={copy_ms:.2f} enqueue_ms={enqueue_ms:.2f} handler_ms={handler_ms:.2f} "
            f"software_limited={self._software_limited_frames}",
            flush=True,
        )
        self._fps_last_log = now
        self._fps_frames = 0
        self._fps_copy_seconds = 0.0
        self._fps_enqueue_seconds = 0.0
        self._fps_handler_seconds = 0.0
        self._software_limited_frames = 0

    def _record_capture_timing(self, *, copy_seconds: float, enqueue_seconds: float, handler_seconds: float) -> None:
        if self.capture_tool != "WindowsCaptureCUDA":
            return
        if not _env_bool("D2S_WGC_CAPTURE_FPS_LOG"):
            return
        self._fps_copy_seconds += copy_seconds
        self._fps_enqueue_seconds += enqueue_seconds
        self._fps_handler_seconds += handler_seconds

    def _log_capture_gap(self, now: float, capture_kwargs: dict) -> None:
        if self.capture_tool != "WindowsCaptureCUDA":
            self._last_frame_ts = now
            return
        gap = now - self._last_frame_ts if self._last_frame_ts > 0.0 else 0.0
        self._last_frame_ts = now
        if not _env_bool("D2S_WGC_CAPTURE_GAP_LOG"):
            return
        if gap < 0.5:
            return
        if self._capture_gap_logs >= CAPTURE_GAP_LOG_LIMIT:
            return
        self._capture_gap_logs += 1
        line = (
            f"[CaptureGap] tool={self.capture_tool} mode={self.config.capture_mode} "
            f"monitor={self.config.monitor_index} gap={gap:.2f}s kwargs={capture_kwargs}"
        )
        if not _defer_capture_gap_log(line):
            _emit_capture_gap_log(line)

    def _start_keyboard_worker(self, shutdown_event):
        user32 = ctypes.windll.user32
        user32.ShowCursor.argtypes = [wintypes.BOOL]
        user32.ShowCursor.restype = ctypes.c_int
        user32.keybd_event.argtypes = [ctypes.c_ubyte, ctypes.c_ubyte, wintypes.DWORD, ctypes.c_ulonglong]
        user32.keybd_event.restype = None

        vk_menu = 0x12
        vk_tab = 0x09
        keyeventf_keyup = 0x0002

        def simulate_alt_tab():
            user32.keybd_event(vk_menu, 0, 0, 0)
            user32.keybd_event(vk_tab, 0, 0, 0)
            time.sleep(0.01)
            user32.keybd_event(vk_tab, 0, keyeventf_keyup, 0)
            user32.keybd_event(vk_menu, 0, keyeventf_keyup, 0)
            return True

        def keyboard_worker():
            while not shutdown_event.is_set():
                triggered = self._capture_started_event.wait(timeout=0.1)
                if shutdown_event.is_set():
                    break
                if not triggered:
                    continue
                try:
                    simulate_alt_tab()
                    time.sleep(0.2)
                    simulate_alt_tab()
                    if CAPTURE_CURSOR_DELAY_S:
                        time.sleep(CAPTURE_CURSOR_DELAY_S)
                except Exception as exc:
                    print(f"[keyboard] Exception during action: {exc}")
                finally:
                    break

        self._keyboard_thread = threading.Thread(target=keyboard_worker, name="CursorWorker", daemon=True)
        self._keyboard_thread.start()

    def run(
        self,
        *,
        shutdown_event,
        on_frame,
        on_error=None,
        on_closed=None,
        is_paused=None,
        is_hard_idle=None,
        on_paused=None,
        on_session_update=None,
        on_tick=None,
    ):
        _setup_dpi_awareness()
        WindowsCapture, Frame, InternalCaptureControl = _load_windows_capture(self.capture_tool)
        self._start_keyboard_worker(shutdown_event)
        self._start_static_replay_worker(
            shutdown_event=shutdown_event,
            on_frame=on_frame,
            is_paused=is_paused,
            is_hard_idle=is_hard_idle,
        )
        on_closed_callback = on_closed

        while not shutdown_event.is_set():
            if on_tick is not None:
                on_tick()
            if is_hard_idle is not None and is_hard_idle():
                if on_paused is not None:
                    on_paused("hard_idle")
                time.sleep(0.1)
                continue

            capture_kwargs = _windows_capture_kwargs(self.config, self.capture_tool)
            if self.config.capture_mode != "Window":
                if os.environ.get('D2S_DEBUG', '0') in ('1', 'true', 'yes', 'on'):
                    print(
                        f"[capture_loop] WindowsCapture monitor_index={self.config.monitor_index} "
                        f"tool={self.capture_tool} kwargs={capture_kwargs}",
                        flush=True,
                    )
            cap = WindowsCapture(**capture_kwargs)
            self._session = cap
            self._control = None
            self._software_frame_due = 0.0
            self._software_pacing_fps = 0
            self._software_limited_frames = 0
            if on_session_update is not None:
                on_session_update(self._session, self._control)

            @cap.event
            def on_frame_arrived(frame: Frame, internal_capture_control: InternalCaptureControl):
                self._control = internal_capture_control
                if on_session_update is not None:
                    on_session_update(self._session, self._control)
                capture_start_time = time.perf_counter()
                if shutdown_event.is_set():
                    return
                if (is_hard_idle is not None and is_hard_idle()) or (is_paused is not None and is_paused()):
                    if on_paused is not None:
                        on_paused("paused")
                    return
                if not self._accept_software_paced_frame(capture_start_time):
                    return
                self._log_capture_gap(capture_start_time, capture_kwargs)
                self._log_capture_fps(capture_start_time)
                copy_start_time = time.perf_counter()
                native_resource = _borrow_native_resource(frame.frame_buffer)
                if _env_bool("D2S_WGC_NATIVE_RESOURCE_REQUIRED") and native_resource is None:
                    raise RuntimeError(
                        "WindowsCapture did not expose a D3D11 resource while "
                        "D2S_WGC_NATIVE_RESOURCE_REQUIRED is enabled"
                    )
                raw, copy_mode, frame_raw_device = _copy_frame_buffer(
                    frame.frame_buffer,
                    self.capture_tool,
                )
                enqueue_start_time = time.perf_counter()
                resource_contract = native_resource_contract(native_resource) if native_resource is not None else {
                    "resource_kind": "cpu_frame_buffer",
                    "resource_format": type(frame.frame_buffer).__name__,
                    "resource_width": None,
                    "resource_height": None,
                    "adapter_luid": 0,
                    "adapter_identity": None,
                    "resource_lifecycle": "owned_copy",
                }
                native_output = native_resource is not None
                native_to_cpu = bool(native_output and raw is not native_resource)
                output_frame = native_resource if native_output else raw
                output_copy_mode = FrameCopyMode.NONE if native_output else copy_mode
                output_device = "d3d11" if native_output else frame_raw_device
                captured_frame = capture_frame_from_raw(
                    output_frame,
                    self.config.output_resolution,
                    capture_start_time,
                    config=self.config,
                    copy_mode=output_copy_mode,
                    original_format=(
                        str(resource_contract.get("resource_format"))
                        if native_output else type(frame.frame_buffer).__name__
                    ),
                    frame_raw_device=output_device,
                    native_resource=native_resource,
                    cpu_compat_frame=raw if native_to_cpu else None,
                    metadata={
                        "backend": "windows_capture_event",
                        "native_resource_output": native_output,
                        "capture_gpu": bool(
                            copy_mode is FrameCopyMode.GPU_TENSOR
                            or native_output
                        ),
                        **resource_contract,
                        "gpu_to_cpu": native_to_cpu,
                        "gpu_copy_count": 1 if native_to_cpu else 0,
                        "compatibility_copy_mode": (
                            copy_mode.value if native_to_cpu else None
                        ),
                        "compatibility_frame_retained": bool(native_to_cpu),
                        "zero_copy": bool(
                            copy_mode is FrameCopyMode.GPU_TENSOR
                            and not native_output
                        ),
                        "zero_copy_ready": False,
                        "fallback_reason": (
                            "WGC native resource retained as primary frame; "
                            "CPU compatibility copy is available"
                            if native_to_cpu else None
                        ),
                    },
                )
                self._remember_emitted_frame(captured_frame, capture_start_time)
                on_frame(captured_frame)
                handler_end_time = time.perf_counter()
                self._record_capture_timing(
                    copy_seconds=enqueue_start_time - copy_start_time,
                    enqueue_seconds=handler_end_time - enqueue_start_time,
                    handler_seconds=handler_end_time - capture_start_time,
                )

            @cap.event
            def on_closed():
                if on_closed_callback is not None:
                    on_closed_callback()

            try:
                cap.start()
            except Exception as exc:
                if on_error is not None:
                    on_error(exc)
                else:
                    raise
                time.sleep(0.5)
            finally:
                self._control = None
                self._session = None
                if on_session_update is not None:
                    on_session_update(None, None)

            if shutdown_event.is_set():
                break
            time.sleep(0.1)

        if self._replay_thread is not None:
            self._replay_thread.join(timeout=0.2)
