from __future__ import annotations

from collections import deque
from dataclasses import replace
import os
import time

from utils.queue_utils import clear_nonblocking, drain_latest, put_latest
from xr_viewer.settings_menu import OPENXR_RENDER_SCALE_MAX, OPENXR_RENDER_SCALE_MIN


class RuntimeCallbacks:
    def __init__(
        self,
        context,
        *,
        show_fps: bool = False,
        display_fit_mode: str = "contain",
    ):
        self.context = context
        self.capture_control = None
        self.capture_session = None
        self._controller_saved_depth_strength = max(
            0.0,
            float(getattr(context.stereo_runtime.stereo_config, "depth_strength", 1.0)),
        )
        self._runtime_fps_started = time.perf_counter()
        self._runtime_fps_frames = 0.0
        self._runtime_fps = 0.0
        self._capture_frame_ts = deque(maxlen=120)
        self._show_fps = bool(show_fps)
        self._display_fit_mode = str(display_fit_mode or "contain")
        self.stream_output = None
        self._screen_state_save_error_logged = False

    def set_stream_output(self, output) -> None:
        self.stream_output = output

    def show_fps(self) -> bool:
        return self._show_fps

    def display_fit_mode(self) -> str:
        return self._display_fit_mode

    def stereo_warmup_key(self, rgb_frame):
        return self.context.stereo_warmup_tracker.key_for_frame(rgb_frame)

    def warmup_stereo_once_for_frame(self, rgb_frame):
        self.context.stereo_warmup_tracker.warmup_once_for_frame(rgb_frame)

    def breakdown_inc(self, name, amount=1):
        self.context.fps_breakdown.inc(name, amount)
        if name == "capture":
            now = time.perf_counter()
            for _ in range(max(0, int(amount))):
                self._capture_frame_ts.append(now)
        if name == "runtime":
            now = time.perf_counter()
            self._runtime_fps_frames += float(amount)
            elapsed = now - self._runtime_fps_started
            if elapsed >= 1.0:
                self._runtime_fps = self._runtime_fps_frames / elapsed
                self._runtime_fps_started = now
                self._runtime_fps_frames = 0.0

    def runtime_fps(self):
        return self._runtime_fps

    def capture_fps(self):
        timestamps = tuple(self._capture_frame_ts)
        if len(timestamps) < 2:
            return 0.0
        now = time.perf_counter()
        if now - timestamps[-1] > 1.0:
            return 0.0
        span = timestamps[-1] - timestamps[0]
        return (len(timestamps) - 1) / span if span > 0.0 else 0.0

    def breakdown_add_time(self, name, seconds):
        self.context.fps_breakdown.add_time(name, seconds)

    def breakdown_add_value(self, name, value):
        self.context.fps_breakdown.add_value(name, value)

    def breakdown_set_latest(self, name, value):
        self.context.fps_breakdown.set_latest(name, value)

    def breakdown_add_runtime_timing(self, runtime_result):
        self.context.fps_breakdown.add_runtime_timing(runtime_result)

    def _render_active_for_breakdown(self):
        openxr_state = getattr(self.context, "openxr_state", None)
        run_mode = getattr(self.context, "run_mode", getattr(openxr_state, "run_mode", ""))
        if str(run_mode).strip().lower() != "openxr":
            return True
        render_active = getattr(openxr_state, "render_active", None)
        return render_active is None or render_active.is_set()

    def log_fps_breakdown(self, now=None):
        if self._render_active_for_breakdown():
            self.context.fps_breakdown.log(now)

    def source_stat_inc(self, name, amount=1, **values):
        self.context.source_health.inc(name, amount, **values)

    def source_stat_set(self, **values):
        self.context.source_health.set(**values)

    def log_source_health(self, now=None, force=False):
        self.context.source_health.log(now, force)
        if self._render_active_for_breakdown():
            self.context.fps_breakdown.log(now)

    def openxr_source_paused(self):
        return self.context.openxr_state.source_paused()

    def stop_active_capture_session(self):
        stopped = False
        try:
            if self.capture_control is not None:
                self.capture_control.stop()
                stopped = True
        except Exception:
            pass
        try:
            if not stopped and self.capture_session is not None and hasattr(self.capture_session, "stop"):
                self.capture_session.stop()
                stopped = True
        except Exception:
            pass
        return stopped

    def on_openxr_hard_idle_enter(self):
        self.queue_clear_nonblocking(self.context.raw_q)
        self.queue_clear_nonblocking(self.context.runtime_q)
        self.stop_active_capture_session()

    def openxr_hard_idle_active(self):
        return self.context.openxr_state.hard_idle_active(
            on_enter=self.on_openxr_hard_idle_enter
        )

    def on_openxr_headset_state(self, state: str) -> None:
        """Mirror Vulkan headset availability into capture and inference gates."""
        state = str(state).strip().lower()
        openxr_state = self.context.openxr_state
        if state == "waiting":
            openxr_state.render_active.clear()
            return
        if state == "hard_idle":
            openxr_state.render_active.clear()
            openxr_state.source_active.clear()
            openxr_state.wait_idle_active.set()
            self.context.stereo_runtime.set_inference_active(False)
            self.on_openxr_hard_idle_enter()
            return
        if state == "active":
            was_idle = openxr_state.wait_idle_active.is_set()
            openxr_state.wait_idle_active.clear()
            openxr_state.source_active.set()
            openxr_state.render_active.set()
            self.context.stereo_runtime.set_inference_active(True)
            if was_idle:
                self.queue_clear_nonblocking(self.context.raw_q)
                self.queue_clear_nonblocking(self.context.runtime_q)
                print("[Main] OpenXR headset resumed; inference restarted", flush=True)

    def on_openxr_controller_shortcut(self, action: str, **values) -> bool:
        """Apply renderer-independent depth shortcuts to runtime state."""
        if action == "select_environment_model":
            try:
                from gui.config import save_yaml
                from stereo_runtime.hot_reload import read_yaml

                model = str(values.get("model", "Default") or "Default")
                settings_path = os.path.join(self.context.base_dir, "settings.yaml")
                settings = read_yaml(settings_path)
                settings["Environment Model"] = model
                ok, error = save_yaml(settings_path, settings)
                if not ok:
                    raise OSError(error)
                return True
            except Exception as exc:
                print(f"[OpenXRViewer] room selection save failed: {exc}", flush=True)
                return False
        if action == "persist_openxr_render_scale":
            try:
                numeric = max(
                    OPENXR_RENDER_SCALE_MIN,
                    min(OPENXR_RENDER_SCALE_MAX, float(values.get("value", 1.0))),
                )
                from gui.config import save_yaml
                from stereo_runtime.hot_reload import read_yaml

                settings_path = os.path.join(self.context.base_dir, "settings.yaml")
                settings = read_yaml(settings_path)
                # Canonical name is percentage-oriented and is also read by
                # the main Flet GUI. Keep the legacy key synchronized.
                settings["XR Render"] = numeric
                settings["XR Render Mode"] = "manual"
                settings["OpenXR Render Scale"] = numeric
                ok, error = save_yaml(settings_path, settings)
                if not ok:
                    raise OSError(error)
                return True
            except Exception as exc:
                print(f"[OpenXRViewer] render scale save failed: {exc}", flush=True)
                return False
        if action == "persist_openxr_render_auto":
            try:
                from gui.config import save_yaml
                from stereo_runtime.hot_reload import read_yaml
                from utils.xr_headset_presets import resolve_xr_headset_preset

                settings_path = os.path.join(self.context.base_dir, "settings.yaml")
                settings = read_yaml(settings_path)
                requested = values.get("value")
                if requested is None:
                    requested = resolve_xr_headset_preset(
                        settings.get("XR Headset Model")
                    ).recommended_render_scale
                numeric = max(
                    OPENXR_RENDER_SCALE_MIN,
                    min(
                        OPENXR_RENDER_SCALE_MAX,
                        float(requested),
                    ),
                )
                settings["XR Render Mode"] = "auto"
                settings["XR Render"] = numeric
                settings["OpenXR Render Scale"] = numeric
                ok, error = save_yaml(settings_path, settings)
                if not ok:
                    raise OSError(error)
                return True
            except Exception as exc:
                print(f"[OpenXRViewer] auto render scale save failed: {exc}", flush=True)
                return False
        if action in {"persist_openxr_screen_state", "reset_openxr_screen_state"}:
            try:
                environment = str(values.get("environment", "Default") or "Default").strip()
                if not environment:
                    environment = "Default"
                from gui.config import save_yaml
                from stereo_runtime.hot_reload import read_yaml

                settings_path = os.path.join(self.context.base_dir, "settings.yaml")
                settings = read_yaml(settings_path)
                states = settings.get("OpenXR Screen States", {})
                states = dict(states) if isinstance(states, dict) else {}
                if action == "reset_openxr_screen_state":
                    states.pop(environment, None)
                else:
                    state = values.get("state")
                    if not isinstance(state, dict):
                        return False
                    states[environment] = dict(state)
                settings["OpenXR Screen States"] = states
                ok, error = save_yaml(settings_path, settings)
                if not ok:
                    raise OSError(error)
                self._screen_state_save_error_logged = False
                return True
            except Exception as exc:
                if not self._screen_state_save_error_logged:
                    self._screen_state_save_error_logged = True
                    print(
                        "[OpenXRViewer] screen state save unavailable; "
                        f"keeping the change for this session: {exc}",
                        flush=True,
                    )
                return False
        if action == "persist_openxr_glow_mode":
            try:
                environment = str(
                    values.get("environment", "Default") or "Default"
                ).strip() or "Default"
                mode = str(values.get("mode", "off") or "off").strip().lower()
                mode = {
                    "none": "off",
                    "false": "off",
                    "0": "off",
                    "screen": "glow",
                }.get(mode, mode)
                if mode not in {"off", "surround", "glow", "veil"}:
                    return False
                from gui.config import save_yaml
                from stereo_runtime.hot_reload import read_yaml

                settings_path = os.path.join(self.context.base_dir, "settings.yaml")
                settings = read_yaml(settings_path)
                modes = settings.get("OpenXR Glow Modes", {})
                modes = dict(modes) if isinstance(modes, dict) else {}
                modes[environment] = mode
                settings["OpenXR Glow Modes"] = modes
                ok, error = save_yaml(settings_path, settings)
                if not ok:
                    raise OSError(error)
                return True
            except Exception as exc:
                print(f"[OpenXRViewer] glow mode save failed: {exc}", flush=True)
                return False
        if action == "persist_openxr_glow_transparency":
            try:
                environment = str(
                    values.get("environment", "Default") or "Default"
                ).strip() or "Default"
                transparency = min(
                    1.0,
                    max(0.0, float(values.get("transparency", 0.0))),
                )
                from gui.config import save_yaml
                from stereo_runtime.hot_reload import read_yaml

                settings_path = os.path.join(self.context.base_dir, "settings.yaml")
                settings = read_yaml(settings_path)
                transparencies = settings.get("OpenXR Glow Transparency", {})
                transparencies = (
                    dict(transparencies)
                    if isinstance(transparencies, dict) else {}
                )
                transparencies[environment] = transparency
                settings["OpenXR Glow Transparency"] = transparencies
                ok, error = save_yaml(settings_path, settings)
                if not ok:
                    raise OSError(error)
                return True
            except Exception as exc:
                print(f"[OpenXRViewer] glow transparency save failed: {exc}", flush=True)
                return False
        if action == "set_runtime_setting":
            return self._set_openxr_runtime_setting(
                str(values.get("name", "")), values.get("value"),
                persist=bool(values.get("persist", False)),
            )
        if action == "set_runtime_settings":
            return self._set_openxr_runtime_settings(
                dict(values.get("settings") or {}),
                persist=bool(values.get("persist", False)),
            )
        if action not in {
            "toggle_stereo",
            "reset_depth",
            "adjust_depth_strength",
        }:
            return False
        snapshot = self.context.openxr_state.runtime_settings_snapshot
        current = getattr(snapshot, "depth_strength", None)
        if current is None:
            current = getattr(
                self.context.stereo_runtime.stereo_config,
                "depth_strength",
                self._controller_saved_depth_strength,
            )
        current = max(0.0, float(current))
        if action == "adjust_depth_strength":
            target = max(
                0.0,
                min(10.0, current + float(values.get("delta", 0.0))),
            )
            if target > 0.0:
                self._controller_saved_depth_strength = target
        elif action == "toggle_stereo":
            if current > 0.0:
                self._controller_saved_depth_strength = current
                target = 0.0
            else:
                target = max(0.0, self._controller_saved_depth_strength)
        else:
            target = max(
                0.0,
                float(getattr(
                    self.context.stereo_runtime.stereo_config,
                    "depth_strength",
                    self._controller_saved_depth_strength,
                )),
            )
            self._controller_saved_depth_strength = target
        self.update_openxr_runtime_config(depth_strength=target)
        print(
            f"[OpenXRViewer] controller shortcut {action}: depth_strength={target:.3f}",
            flush=True,
        )
        return True

    def _set_openxr_runtime_setting(self, name: str, value, *, persist: bool) -> bool:
        return self._set_openxr_runtime_settings({name: value}, persist=persist)

    def _set_openxr_runtime_settings(self, values: dict, *, persist: bool) -> bool:
        yaml_keys = {
            "openxr_render_scale": "XR Render",
            "depth_strength": "Depth Strength",
            "cross_eyed": "Cross Eyed",
            "color_brightness": "Color Brightness",
            "color_contrast": "Color Contrast",
            "color_saturation": "Color Saturation",
            "color_gamma": "Color Gamma",
            "color_temperature": "Color Temperature",
            "color_tint": "Color Tint",
            "vulkan_projection_min_lod": "Vulkan Projection Min LOD",
            "vulkan_projection_max_lod": "Vulkan Projection Max LOD",
            "vulkan_projection_mip_lod_bias": "Vulkan Projection MIP LOD Bias",
            "vulkan_projection_rcas_sharpness": "Vulkan Projection RCAS Sharpness",
        }
        if not values or any(name not in yaml_keys for name in values):
            return False
        numeric_values = {
            name: (bool(value) if name == "cross_eyed" else float(value))
            for name, value in values.items()
        }
        if "openxr_render_scale" in numeric_values:
            numeric_values["openxr_render_scale"] = max(
                OPENXR_RENDER_SCALE_MIN,
                min(
                    OPENXR_RENDER_SCALE_MAX,
                    numeric_values["openxr_render_scale"],
                ),
            )
        current = self.context.openxr_state.runtime_settings_snapshot
        snapshot = replace(
            current,
            version=int(current.version) + 1,
            timestamp=time.time(),
            source="openxr_settings_menu",
            **numeric_values,
        )
        self.update_openxr_runtime_config(snapshot=snapshot)
        self.send_settings_snapshot(snapshot)
        if not persist:
            return True
        try:
            from gui.config import save_yaml
            from stereo_runtime.hot_reload import read_yaml

            settings_path = os.path.join(self.context.base_dir, "settings.yaml")
            settings = read_yaml(settings_path)
            for name, numeric in numeric_values.items():
                settings[yaml_keys[name]] = numeric
                if name == "openxr_render_scale":
                    settings["OpenXR Render Scale"] = numeric
            ok, error = save_yaml(settings_path, settings)
            if not ok:
                raise OSError(error)
        except Exception as exc:
            print(f"[OpenXRViewer] settings menu save failed: {exc}", flush=True)
        return True

    def queue_put_latest(self, q, item):
        put_latest(q, item)

    def queue_clear_nonblocking(self, q):
        clear_nonblocking(q)

    def queue_drain_latest(self, q, first_item):
        def on_drop():
            self.source_stat_inc("raw_dropped_stale")
            self.breakdown_inc("raw_dropped_stale")

        return drain_latest(q, first_item, on_drop=on_drop)

    def send_settings_snapshot(self, snapshot):
        presentation_flags = getattr(snapshot, "presentation_flags", None)
        if isinstance(presentation_flags, dict):
            if "show_fps" in presentation_flags:
                self._show_fps = bool(presentation_flags["show_fps"])
            if "display_fit_mode" in presentation_flags:
                self._display_fit_mode = str(
                    presentation_flags["display_fit_mode"] or "contain"
                )
        put_latest(self.context.settings_update_q, snapshot)

    def update_openxr_runtime_config(
        self,
        *,
        snapshot=None,
        depth_ratio=None,
        depth_strength=None,
        convergence=None,
        max_disparity_px=None,
        parallax_preset=None,
        screen_roll=None,
    ):
        self.context.openxr_state.update_runtime_config(
            snapshot=snapshot,
            depth_ratio=depth_ratio,
            depth_strength=depth_strength,
            convergence=convergence,
            max_disparity_px=max_disparity_px,
            parallax_preset=parallax_preset,
            screen_roll=screen_roll,
        )

    def current_openxr_render_config(self):
        return self.context.openxr_state.current_render_config(self.context.stereo_runtime)

    def apply_stereo_hot_reload_if_needed(self):
        reloader = self.context.stereo_hot_reloader
        poll = getattr(reloader, "poll_settings_snapshot_if_needed", None)
        if callable(poll):
            polled = poll(
                runtime=self.context.stereo_runtime,
                active_preset=self.context.stereo_active_preset,
            )
            if polled is None:
                return False
            snapshot, _applied_preset, values = polled
            request_audio_delay = getattr(
                self.stream_output, "request_audio_delay", None
            )
            audio_delay = values.get("audio_delay")
            if callable(request_audio_delay) and audio_delay is not None:
                request_audio_delay(audio_delay)
            self.send_settings_snapshot(snapshot)
            self.update_openxr_runtime_config(snapshot=snapshot)
            log_snapshot = getattr(reloader, "log_settings_snapshot", None)
            if callable(log_snapshot):
                log_snapshot(values, on_mode_log=self.log_stereo_runtime_mode_once)
            return True
        return reloader.apply_if_needed(
            runtime=self.context.stereo_runtime,
            active_preset=self.context.stereo_active_preset,
            on_openxr_config_update=self.update_openxr_runtime_config,
            on_mode_log=self.log_stereo_runtime_mode_once,
        )

    def log_stereo_runtime_mode(self, reason, decision=None, samples=None, motion=None):
        self.context.stereo_runtime_logger.log_mode(
            reason,
            decision=decision,
            samples=samples,
            motion=motion,
        )

    def log_stereo_runtime_mode_once(self, reason="active"):
        self.context.stereo_runtime_logger.log_mode_once(reason)

    def log_fast_plus_fused_runtime_state(self, runtime_result):
        self.context.stereo_runtime_logger.log_fast_plus_fused_runtime_state(runtime_result)

    def capture_session_update(self, session, control):
        self.capture_session = session
        self.capture_control = control

    def put_raw_latest(self, item):
        was_full = self.context.raw_q.full()
        self.queue_put_latest(self.context.raw_q, item)
        return was_full

    def set_runtime_preprocess_backend(self, backend):
        if self.context.fps_breakdown_log:
            self.context.fps_breakdown.set_latest("rt_preprocess_backend", backend)
