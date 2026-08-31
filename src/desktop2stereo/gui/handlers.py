"""GUI Handler Mixin — event handlers, visibility sync, i18n, audio, refresh."""
import os
import asyncio
import logging
import re
import subprocess
import flet as ft
from utils import (
    OS_NAME, DEFAULT_PORT, STEREO_MIX_NAMES,
    DISABLE_TRT_KEYWORDS, DISABLE_COREML_KEYWORDS, DISABLE_OPENVINO_KEYWORDS,
    DISABLE_MIGRAPHX_KEYWORDS,
    get_local_ip,
)
from streaming.audio import (
    find_loopback_audio_devices,
    query_ffmpeg_dshow_audio_devices,
    query_ffmpeg_wasapi_audio_devices,
)
from streaming.wasapi_audio import query_soundcard_loopback_devices
from utils.run_mode import target_fps_setting_key
from utils.xr_headset_presets import display_to_xr_headset, xr_headset_options, xr_headset_to_display
from . import devices as devices_module
from .capture_sources import (
    get_capture_tool_options, get_primary_monitor_index, list_windows,
)
from .config import (
    DEFAULTS, FAMILY_SIZE_TO_MODEL, FAMILY_TO_SIZES,
    environment_display_label, environment_key_from_label,
    get_environment_model_options, load_environment_display_names,
)
from .controls import FONT_SIZE
from .localization import UI_MESSAGES, is_supported_locale
from .paths import BASE_DIR
from .devices import DEVICES

logger = logging.getLogger(__name__)


class GUIHandlerMixin:
    """Mixin providing event handlers, visibility sync, and i18n for Desktop2StereoGUI."""

    # ── locale-dependent converters ──

    def _upscaler_display_options(self):
        t = UI_MESSAGES[self.locale]
        return [t.get("Auto", "Auto"), t.get("Off", "Off")]

    def _upscaler_to_display(self, value):
        value_l = str(value or "").strip().lower()
        if value_l in ("auto", "自动"):
            return UI_MESSAGES[self.locale].get("Off", "Off")
        return UI_MESSAGES[self.locale].get("Off", "Off")

    def _target_fps_to_display(self, value):
        fps = self._parse_int(value, DEFAULTS["Target FPS"])
        if fps <= 0:
            return UI_MESSAGES[self.locale].get("Auto", "Auto")
        return str(fps)

    def _depth_quick_to_display(self, value):
        return UI_MESSAGES[self.locale].get(value, value)

    def _render_policy_options(self):
        t = UI_MESSAGES[self.locale]
        return [t["Native"], t["Scaled"], t["Fixed"], t["Dynamic"]]

    def _render_scale_options(self):
        return [
            "4K / 100%",
            "3K / 85%",
            "2K / 75%",
            "1K / 50%",
        ]

    def _render_scale_to_display(self, value):
        return self._normalize_render_scale_tier(value)

    def _display_to_render_scale(self, value):
        return self._normalize_render_scale_tier(value)

    def _normalize_render_scale_tier(self, value):
        text = str(value or "").strip()
        if text in self._render_scale_options():
            return text
        return DEFAULTS["Render Scale"]

    def _render_policy_to_display(self, value):
        return UI_MESSAGES[self.locale].get("Scaled", "Scaled")

    def _display_to_render_policy(self, value):
        return "scaled"

    def _update_render_size_control_visibility(self, show_render_size):
        self.row6d.visible = show_render_size
        self.row6e.visible = False
        self.row6f.visible = False
        for ctrl in [self.render_policy_label, self.render_policy_dd]:
            ctrl.visible = False
        for ctrl in [self.render_scale_label, self.render_scale_dd]:
            ctrl.visible = show_render_size
        for ctrl in [self.render_fixed_label, self.render_fixed_dd]:
            ctrl.visible = False
        for ctrl in [self.render_max_pixels_label, self.render_max_pixels_dd]:
            ctrl.visible = False
        for ctrl in [self.render_min_dimension_label, self.render_min_dimension_dd]:
            ctrl.visible = False
        for ctrl in [self.render_align_label, self.render_align_dd]:
            ctrl.visible = show_render_size

    def _preset_to_display(self, value):
        mapping = {
            "auto": "Cinema", "cinema": "Cinema",
            "traditional_fastest": "Traditional / Fastest",
            "game_low_latency": "Game / Low Latency",
            "still_image_hq": "Image  / High Quality",
            "debug_export": "Cinema",
        }
        key = mapping.get(str(value or "cinema").strip().lower(), "Cinema")
        return UI_MESSAGES[self.locale].get(key, key) if hasattr(self, "locale") else key

    # ── model handlers ──

    def on_model_family_change(self, e):
        family = e.control.value
        sizes = FAMILY_TO_SIZES.get(family, [])
        self.model_size_dd.options = [s for s in sizes]
        self.model_size_dd.value = sizes[0] if sizes else ""
        self.model_size_dd.update()
        self._on_model_changed()

    def on_model_size_change(self, e):
        self._on_model_changed()

    @property
    def current_model_name(self):
        family = self.depth_model_dd.value
        size = self.model_size_dd.value
        return FAMILY_SIZE_TO_MODEL.get((family, size), family if not size else f"{family}-{size}")

    def _on_model_changed(self):
        model = self.current_model_name
        self._config["Depth Model"] = model
        self.update_depth_resolution_options(model)
        self.auto_enable_optimizers_based_on_device()
        if "CUDA" in self.device_dd.value:
            if devices_module.IS_ROCM:
                self.update_migraphx_visibility_based_on_model(model)
            else:
                self.update_tensorrt_visibility_based_on_model(model)
        elif "MPS" in self.device_dd.value:
            self.update_coreml_visibility_based_on_model(model)
        elif "XPU" in self.device_dd.value:
            self.update_openvino_visibility_based_on_model(model)
        self._fit_window_to_content()

    # ── device / accelerator handlers ──

    def on_device_change(self, e):
        if getattr(self, "_startup_defer_devices", False):
            return
        device_label = e.control.value if e else self.device_dd.value
        self._config["Computing Device"] = self.device_label_to_index.get(device_label, 0)
        if OS_NAME in ("Windows", "Darwin"):
            new_opts = get_capture_tool_options(device_label)
            self.capture_tool_dd.options = [o for o in new_opts]
            saved_capture_tool = self._config.get(
                "Capture Tool", DEFAULTS["Capture Tool"]
            )
            if saved_capture_tool in new_opts:
                self.capture_tool_dd.value = saved_capture_tool
            elif saved_capture_tool in ("", "none", None):
                # Auto selection follows the detected device-specific order.
                self.capture_tool_dd.value = new_opts[0]
            elif self.capture_tool_dd.value not in new_opts:
                self.capture_tool_dd.value = new_opts[0]
            self.capture_tool_dd.update()
        self._update_accelerator_visibility(device_label)
        self._sync_nvfruc_visibility(device_label)
        self.auto_enable_optimizers_based_on_device()
        self._fit_window_to_content()

    def _sync_nvfruc_visibility(self, device_label=None):
        device_label = device_label if device_label is not None else self.device_dd.value
        is_nvidia_cuda = (
            OS_NAME == "Windows"
            and "CUDA" in (device_label or "")
            and not devices_module.IS_ROCM
        )
        self.lossless_cb.visible = is_nvidia_cuda
        self.lossless_cb.disabled = not is_nvidia_cuda

    def _update_accelerator_visibility(self, device_label):
        cuda = "CUDA" in device_label
        rocm = cuda and devices_module.IS_ROCM
        dml = "DirectML" in device_label
        mps = "MPS" in device_label
        xpu = "XPU" in device_label
        other = not (cuda or dml or mps or xpu)
        self.torch_compile_cb.visible = cuda
        self.tensorrt_cb.visible = cuda and not rocm
        self.tensorrt_cb.disabled = False
        self.recompile_trt_cb.visible = self.tensorrt_cb.value if self.tensorrt_cb.visible else False
        self.migraphx_cb.visible = rocm
        self.migraphx_cb.disabled = False
        self.recompile_migraphx_cb.visible = self.migraphx_cb.value if self.migraphx_cb.visible else False
        self.coreml_cb.visible = mps
        self.coreml_cb.disabled = False
        self.recompile_coreml_cb.visible = self.coreml_cb.value if self.coreml_cb.visible else False
        self.openvino_cb.visible = xpu
        self.openvino_cb.disabled = False
        self.recompile_openvino_cb.visible = self.openvino_cb.value if self.openvino_cb.visible else False
        self.fp16_cb.visible = not (dml or mps)
        self.acceleration_label.visible = not (dml or other)
        is_windows_or_mac = OS_NAME in ("Windows", "Darwin")
        self.capture_tool_label.visible = is_windows_or_mac
        self.capture_tool_dd.visible = is_windows_or_mac
        if dml or other:
            self.torch_compile_cb.visible = False
            self.tensorrt_cb.visible = False
            self.recompile_trt_cb.visible = False
            self.migraphx_cb.visible = False
            self.recompile_migraphx_cb.visible = False
            self.coreml_cb.visible = False
            self.recompile_coreml_cb.visible = False
            self.openvino_cb.visible = False
            self.recompile_openvino_cb.visible = False
        current_model = self.current_model_name
        if cuda:
            if rocm:
                self.update_migraphx_visibility_based_on_model(current_model)
            else:
                self.update_tensorrt_visibility_based_on_model(current_model)
        if mps:
            self.update_coreml_visibility_based_on_model(current_model)
        if xpu:
            self.update_openvino_visibility_based_on_model(current_model)

    def _on_trt_toggle(self, e):
        if e is not None:
            self._config["TensorRT"] = bool(self.tensorrt_cb.value)
        self.recompile_trt_cb.visible = self.tensorrt_cb.value
        self._fit_window_to_content()

    def _on_coreml_toggle(self, e):
        if e is not None:
            self._config["CoreML"] = bool(self.coreml_cb.value)
        self.recompile_coreml_cb.visible = self.coreml_cb.value
        self._fit_window_to_content()

    def _on_openvino_toggle(self, e):
        if e is not None:
            self._config["OpenVINO"] = bool(self.openvino_cb.value)
        self.recompile_openvino_cb.visible = self.openvino_cb.value
        self._fit_window_to_content()

    def _on_migraphx_toggle(self, e):
        if e is not None:
            self._config["MIGraphX"] = bool(self.migraphx_cb.value)
        self.recompile_migraphx_cb.visible = self.migraphx_cb.value
        self._fit_window_to_content()

    def _platform_accelerator_values(self, device_label=None, use_control_values=True):
        device_label = device_label or self.device_dd.value or ""
        model_lower = (self.current_model_name or "").lower()
        values = {
            "TensorRT": None,
            "CoreML": None,
            "OpenVINO": None,
            "MIGraphX": None,
        }
        recompile_values = {
            "Recompile TensorRT": False,
            "Recompile CoreML": False,
            "Recompile OpenVINO": False,
            "Recompile MIGraphX": False,
        }

        active_key = None
        recompile_key = None
        active_cb = None
        recompile_cb = None
        disabled_by_model = False

        if "CUDA" in device_label and devices_module.IS_ROCM:
            active_key = "MIGraphX"
            recompile_key = "Recompile MIGraphX"
            active_cb = self.migraphx_cb
            recompile_cb = self.recompile_migraphx_cb
            disabled_by_model = any(kw in model_lower for kw in DISABLE_MIGRAPHX_KEYWORDS)
        elif "CUDA" in device_label:
            active_key = "TensorRT"
            recompile_key = "Recompile TensorRT"
            active_cb = self.tensorrt_cb
            recompile_cb = self.recompile_trt_cb
            disabled_by_model = any(kw in model_lower for kw in DISABLE_TRT_KEYWORDS)
        elif "MPS" in device_label:
            active_key = "CoreML"
            recompile_key = "Recompile CoreML"
            active_cb = self.coreml_cb
            recompile_cb = self.recompile_coreml_cb
            disabled_by_model = any(kw in model_lower for kw in DISABLE_COREML_KEYWORDS)
        elif "XPU" in device_label:
            active_key = "OpenVINO"
            recompile_key = "Recompile OpenVINO"
            active_cb = self.openvino_cb
            recompile_cb = self.recompile_openvino_cb
            disabled_by_model = any(kw in model_lower for kw in DISABLE_OPENVINO_KEYWORDS)

        if active_key is not None:
            if disabled_by_model:
                enabled = False
            elif use_control_values:
                enabled = bool(active_cb.value)
            else:
                saved_value = self._config.get(active_key)
                enabled = (saved_value is None) or bool(saved_value)

            values[active_key] = enabled
            if enabled:
                recompile_values[recompile_key] = bool(recompile_cb.value)

        return values, recompile_values

    def _apply_platform_accelerator_policy(self, update_controls=True):
        values, recompile_values = self._platform_accelerator_values(use_control_values=False)
        self._config.update(values)
        self._config.update(recompile_values)

        if update_controls:
            self.tensorrt_cb.value = bool(values["TensorRT"])
            self.coreml_cb.value = bool(values["CoreML"])
            self.openvino_cb.value = bool(values["OpenVINO"])
            self.migraphx_cb.value = bool(values["MIGraphX"])
            self.recompile_trt_cb.value = recompile_values["Recompile TensorRT"]
            self.recompile_coreml_cb.value = recompile_values["Recompile CoreML"]
            self.recompile_openvino_cb.value = recompile_values["Recompile OpenVINO"]
            self.recompile_migraphx_cb.value = recompile_values["Recompile MIGraphX"]

        self._on_trt_toggle(None)
        self._on_coreml_toggle(None)
        self._on_openvino_toggle(None)
        self._on_migraphx_toggle(None)

    def auto_enable_optimizers_based_on_device(self):
        if getattr(self, "_startup_defer_devices", False):
            return
        if (
            "CUDA" in (self.device_dd.value or "")
            and not devices_module.IS_ROCM
            and self._config.get("torch.compile") is None
        ):
            self.torch_compile_cb.value = True
        self._apply_platform_accelerator_policy()

    def update_tensorrt_visibility_based_on_model(self, model_name):
        if not model_name:
            return
        if not devices_module.IS_ROCM and "CUDA" in self.device_dd.value:
            should_disable = any(kw in model_name.lower() for kw in DISABLE_TRT_KEYWORDS)
            if should_disable:
                self.tensorrt_cb.value = False
                self.tensorrt_cb.disabled = True
                self.recompile_trt_cb.visible = False
            else:
                self.tensorrt_cb.disabled = False

    def update_migraphx_visibility_based_on_model(self, model_name):
        if not model_name:
            return
        if devices_module.IS_ROCM and "CUDA" in self.device_dd.value:
            should_disable = any(kw in model_name.lower() for kw in DISABLE_MIGRAPHX_KEYWORDS)
            if should_disable:
                self.migraphx_cb.value = False
                self.migraphx_cb.disabled = True
                self.recompile_migraphx_cb.visible = False
            else:
                self.migraphx_cb.disabled = False
                self.recompile_migraphx_cb.visible = self.migraphx_cb.value

    def update_coreml_visibility_based_on_model(self, model_name):
        if not model_name:
            return
        if "MPS" in self.device_dd.value:
            should_disable = any(kw in model_name.lower() for kw in DISABLE_COREML_KEYWORDS)
            if should_disable:
                self.coreml_cb.value = False
                self.coreml_cb.disabled = True
                self.recompile_coreml_cb.visible = False
            else:
                self.coreml_cb.disabled = False

    def update_openvino_visibility_based_on_model(self, model_name):
        if not model_name:
            return
        if "XPU" in self.device_dd.value:
            should_disable = any(kw in model_name.lower() for kw in DISABLE_OPENVINO_KEYWORDS)
            if should_disable:
                self.openvino_cb.value = False
                self.openvino_cb.disabled = True
                self.recompile_openvino_cb.visible = False
            else:
                self.openvino_cb.disabled = False

    # ── capture tool / mode handlers ──

    def on_capture_tool_change(self, e):
        self.capture_mode_dd.disabled = False
        self.set_status("", key="")
        self.capture_mode_dd.update()
        self._sync_capture_mode_visibility()

    def _sync_capture_mode_visibility(self):
        if self.capture_mode_key == "Monitor":
            self.monitor_dd.visible = True
            self.window_dd.visible = False
        else:
            self.monitor_dd.visible = False
            self.window_dd.visible = True
        self._safe_update(self.monitor_dd, self.window_dd)
        self._fit_window_to_content()

    def on_capture_mode_change(self, e):
        mode = e.control.value
        texts = UI_MESSAGES[self.locale]
        reverse_map = {texts["Monitor"]: "Monitor", texts["Window"]: "Window"}
        self.capture_mode_key = reverse_map.get(mode, "Monitor")
        self._sync_capture_mode_visibility()
        if self.capture_mode_key == "Window":
            self.refresh_window_list()
        self.update_stereo_monitor_menu()
        self._fit_window_to_content()

    def on_window_selected(self, e):
        label = e.control.value if e else self.window_dd.value
        if not label:
            self.selected_window_name = ""
            self.selected_window_handle = None
            self.selected_window_rect = None
            self._fit_window_to_content()
            return
        m = re.search(r'\[h:(\d+)\]$', label)
        target_handle = int(m.group(1)) if m else None
        display_title = re.sub(r'\s*\[h:\d+\]$', '', label).strip()
        self.selected_window_name = display_title
        for win in self._window_objects:
            wh = win.get("handle") or 0
            if target_handle is not None and wh == target_handle:
                self.selected_window_handle = win.get("handle")
                self.selected_window_rect = win.get("rect")
                break
            elif target_handle is None and win["title"] == display_title:
                self.selected_window_handle = win.get("handle")
                self.selected_window_rect = win.get("rect")
                break
        self._fit_window_to_content()

    def _on_monitor_change(self, e):
        self._missing_monitor_identity = False
        self.update_stereo_monitor_menu()
        self._fit_window_to_content()
        self.set_status(f"{UI_MESSAGES[self.locale]['Selected input monitor:']} {e.control.value}")

    def _on_stereo_monitor_change(self, _e):
        self._missing_stereo_output_identity = False
        self._fit_window_to_content()

    # ── run mode / visibility handlers ──

    def on_xr_headset_change(self, e):
        value = e.control.value if e else self.xr_headset_dd.value
        self._config["XR Headset Model"] = display_to_xr_headset(value)

    def on_env_change(self, e):
        label = e.control.value if e else self.env_model_dd.value
        self.env_key = environment_key_from_label(
            label,
            self.locale,
            getattr(self, "env_model_keys", None),
            getattr(self, "env_model_display_names", None),
        )
        self._config["Environment Model"] = self.env_key

    def _refresh_environment_options(self):
        self.env_model_keys = get_environment_model_options(return_keys=True)
        self.env_model_display_names = load_environment_display_names(self.env_model_keys)
        if getattr(self, "env_key", None) not in self.env_model_keys:
            self.env_key = self.env_model_keys[0] if self.env_model_keys else "None"
        self.env_model_dd.options = [
            environment_display_label(key, self.locale, self.env_model_display_names)
            for key in self.env_model_keys
        ]
        self.env_model_dd.value = environment_display_label(
            self.env_key,
            self.locale,
            self.env_model_display_names,
        )

    def on_run_mode_change(self, e):
        previous_mode = getattr(self, "run_mode_key", "Local Viewer")
        previous_fps_key = target_fps_setting_key(previous_mode)
        self._config[previous_fps_key] = self._target_fps_from_display(
            self.target_fps_dd.value
        )
        label = e.control.value
        texts = UI_MESSAGES[self.locale]
        mode_map = {
            texts.get("Local Viewer", "Local Viewer"): "Local Viewer",
            texts.get("OpenXR Link", "OpenXR Link"): "OpenXR Link",
            texts.get("RTMP Streamer", "RTMP Streamer"): "RTMP Streamer",
            texts.get("MJPEG Streamer", "MJPEG Streamer"): "MJPEG Streamer",
            texts.get("3D Monitor", "3D Monitor"): "3D Monitor",
        }
        self.run_mode_key = mode_map.get(label, "Local Viewer")
        self._config["Run Mode"] = self.run_mode_key
        fps_key = target_fps_setting_key(self.run_mode_key)
        fps_default = DEFAULTS.get(fps_key, DEFAULTS["Target FPS"])
        fps = self._config.get(fps_key, self._config.get("Target FPS", fps_default))
        self.target_fps_dd.value = self._target_fps_to_display(fps)
        if self.run_mode_key in {"MJPEG Streamer", "RTMP Streamer"}:
            self.stream_settings_cb.value = False
        if self.run_mode_key == "RTMP Streamer":
            # Collect every GUI control after changing mode.  The calibration
            # fingerprint must be built from the same values that will be
            # written at run time, otherwise an old local-mode snapshot can
            # prevent a valid network profile from being restored.
            self._collect_config()
            # A completed auto-calibration profile is scoped to the network
            # streamer. Apply it immediately when entering this mode so the
            # visible CRF/bitrate controls cannot retain stale local-mode
            # values (for example CRF=26 after a 38 Mbps calibration).
            self._persist_current_stream_calibration_profile()
        self._sync_visibility()
        self._fit_window_to_content(update=True, resize_window=True)

    def on_stream_settings_change(self, e):
        mode = self.run_mode_key
        streamer_modes = {"MJPEG Streamer", "RTMP Streamer"}
        row_map = self._get_streamer_row_map()
        row_indices = row_map.get(mode, []) if mode in streamer_modes and self.stream_settings_cb.value else []
        self._show_streamer_rows(*row_indices)
        self._fit_window_to_content(update=True, resize_window=True)

    def on_advanced_device_change(self, e):
        self._sync_visibility()
        self._fit_window_to_content(update=True, resize_window=True)
        self._safe_update(self.page)

    def on_render_policy_change(self, e):
        self._sync_visibility()
        self._fit_window_to_content()
        self._safe_update(self.page)

    def _sync_device_advanced_visibility(self, mode):
        advanced = bool(getattr(self, "advanced_device_cb", None) and self.advanced_device_cb.value)
        for row in getattr(self, "_advanced_device_rows", []):
            row.visible = advanced
        show_timing = advanced and mode in ["Local Viewer", "3D Monitor", "OpenXR Link"]
        show_enhance = False
        self.row6b.visible = show_timing
        self.row6c.visible = show_enhance
        show_render_size = advanced and mode in ["Local Viewer", "3D Monitor", "OpenXR Link"]
        self._update_render_size_control_visibility(show_render_size)
        self.target_fps_label.visible = show_timing
        self.target_fps_dd.visible = show_timing
        self.xr_preview_cb.visible = advanced and mode == "OpenXR Link"
        self.local_vsync_cb.visible = advanced and mode in ["Local Viewer", "3D Monitor"]
        self.window_preview_cb.visible = advanced and mode in ["Local Viewer", "3D Monitor"]
        self.upscaler_label.visible = show_enhance
        self.upscaler_dd.visible = show_enhance
        self.upscaler_sharpness_label.visible = show_enhance
        self.upscaler_sharpness_dd.visible = show_enhance

    def _sync_visibility(self):
        mode = self.run_mode_key
        texts = UI_MESSAGES[self.locale]
        mode_reverse = {
            "Local Viewer": texts["Local Viewer"], "OpenXR Link": texts["OpenXR Link"],
            "RTMP Streamer": texts["RTMP Streamer"], "MJPEG Streamer": texts["MJPEG Streamer"],
            "3D Monitor": texts["3D Monitor"],
        }
        self.run_mode_dd.value = mode_reverse.get(mode, texts["Local Viewer"])
        streamer_modes = {"MJPEG Streamer", "RTMP Streamer"}
        is_streamer = mode in streamer_modes
        if hasattr(self, "stream_url_row"):
            self.stream_url_row.visible = is_streamer
        if hasattr(self, "stream_url_container"):
            self.stream_url_container.visible = is_streamer
        self.stream_settings_cb.visible = is_streamer
        is_openxr = mode == "OpenXR Link"
        self.display_mode_label.visible = not is_openxr
        self.display_mode_dd.visible = not is_openxr
        # The target headset preset is shared by every output mode, so keep its
        # dedicated row available independently of the current run mode.
        self.xr_headset_label.visible = True
        self.xr_headset_dd.visible = True
        self._sync_device_advanced_visibility(mode)
        self.row7b.visible = is_openxr
        display_fit_visible = mode in ["Local Viewer", "3D Monitor"]
        if is_openxr:
            self.showfps_cb.visible = False
        else:
            self.showfps_cb.visible = True
        self.display_fit_dd.visible = display_fit_visible
        if mode == "3D Monitor":
            self.display_mode_dd.options = ["Half-SBS", "Full-SBS", "Half-TAB", "Full-TAB"]
        else:
            self.display_mode_dd.options = [
                "Half-SBS", "Full-SBS", "Half-TAB", "Full-TAB",
                "Depth Map", "Anaglyph", "Interleaved", "Mono", "Leia"]
        mon_count = self._get_monitor_count()
        stereo_full = mode in ["Local Viewer", "3D Monitor"] and mon_count > 1
        self.stereo_output_label.visible = stereo_full
        self.stereo_monitor_dd.visible = stereo_full
        if hasattr(self, '_stereo_spacer'):
            self._stereo_spacer.visible = stereo_full
        self._sync_nvfruc_visibility()
        self.row9.visible = (not is_openxr) and (
            stereo_full or display_fit_visible
        )
        row_map = self._get_streamer_row_map()
        row_indices = row_map.get(mode, []) if is_streamer and self.stream_settings_cb.value else []
        self._show_streamer_rows(*row_indices)
        self.update_stereo_monitor_menu()
        self._fit_window_to_content()
        if mode == "Local Viewer":
            self._auto_select_stereo_monitor()
        if mode == "RTMP Streamer":
            if getattr(self, "_startup_defer_audio", False):
                # The slow SoundCard/WASAPI scan runs after the window is shown.
                self.audio_dd.options = []
                self.audio_dd.value = ""
            elif not self.audio_devices:
                self.populate_audio_devices()
                self.auto_select_stereo_mix()
            else:
                self._ensure_audio_device_selected()
        self.update_stream_url()
        self._fit_window_to_content()
        self.page.update()

    def _auto_select_stereo_monitor(self):
        mon_count = self._get_monitor_count()
        if mon_count <= 1:
            self.stereo_monitor_dd.value = ""
            return
        self.update_stereo_monitor_menu()

    # ── theme / language ──

    def on_theme_change(self, e):
        color = e.control.value
        cn_map = {"系统": "system", "蓝色": "blue", "绿色": "green", "红色": "red",
                  "紫色": "purple", "橙色": "orange", "青色": "teal", "粉色": "pink", "灰色": "grey"}
        color = cn_map.get(color, color.lower())
        if OS_NAME == "Windows":
            font = "Microsoft YaHei"
        elif OS_NAME == "Darwin":
            font = "PingFang SC"
        else:
            font = "Noto Sans SC"
        if color == "system":
            self.page.theme_mode = ft.ThemeMode.SYSTEM
            self.page.theme = ft.Theme(color_scheme_seed="blue", font_family=font)
        else:
            self.page.theme = ft.Theme(color_scheme_seed=color, font_family=font)
        self.page.update()

    def on_language_change(self, e):
        lang_display = e.control.value
        _LANG_MAP = {"English": "EN", "简体中文": "CN"}
        lang = _LANG_MAP.get(lang_display, "EN")
        if is_supported_locale(lang):
            self.locale = lang
            os.environ["DESKTOP2STEREO_LOCALE"] = self.locale
            self._config["Language"] = lang
            self.update_ui_texts()
            self._sync_visibility()
            if self._status_key:
                self.set_status(UI_MESSAGES[self.locale].get(self._status_key, self.status_text.value), key=self._status_key)
            t = UI_MESSAGES[self.locale]
            cur = self.theme_dd.value
            cn_map = {"系统": "system", "蓝色": "blue", "绿色": "green", "红色": "red",
                      "紫色": "purple", "橙色": "orange", "青色": "teal", "粉色": "pink", "灰色": "grey"}
            key = cn_map.get(cur, cur.lower() if cur else "system")
            self.theme_dd.value = t.get(key, key) if self.locale == "CN" else key.capitalize()
            self._fit_window_to_content(resize_window=True)
            self.page.update()

    # ── i18n text update ──

    def update_ui_texts(self):
        t = UI_MESSAGES[self.locale]
        self.depth_model_label.value = t["Depth Model:"]
        self.fp16_cb.label = t["FP16"]
        self.depth_resolution_label.value = t["Depth Resolution:"]
        self.convergence_label.value = t["Convergence:"]
        self.dynamic_convergence_label.value = t["Dynamic Convergence:"]
        self.depth_strength_label.value = t["Depth Strength:"]
        self.depth_quick_label.value = t["Depth Quick:"]
        depth_quick_key = self._display_to_depth_quick(self.depth_quick_dd.value)
        self.depth_quick_dd.options = [t["Soft"], t["Standard"], t["Enhanced"]]
        self.depth_quick_dd.value = self._depth_quick_to_display(depth_quick_key)
        self.depth_pop_label.value = t["Depth Pop:"]
        self.foreground_pop_label.value = t["Foreground Pop:"]
        self.midground_pop_label.value = t["Midground Pop:"]
        self.background_pop_label.value = t["Background Pop:"]
        self.antialiasing_label.value = t["Anti-aliasing:"]
        self.stereo_preset_label.value = t["Stereo Mode:"]
        preset_key = self._display_to_preset(self.stereo_preset_dd.value)
        self.stereo_preset_dd.options = [t["Traditional / Fastest"], t["Cinema"], t["Game / Low Latency"], t["Image  / High Quality"]]
        self.stereo_preset_dd.value = self._preset_to_display(preset_key)
        self.stereo_quality_label.value = t["Synthetic View:"]
        self.stereo_quality_label.visible = False
        self.stereo_quality_dd.options = self._stereo_quality_options()
        self.stereo_quality_dd.value = self._stereo_quality_to_display(
            self._stereo_quality_for_preset(preset_key))
        self.stereo_quality_dd.visible = False
        self.parallax_budget_label.value = t["Parallax Budget:"]
        parallax_budget_key = self._display_to_parallax_budget(self.parallax_budget_dd.value)
        self.parallax_budget_dd.options = self._parallax_budget_options()
        self.parallax_budget_dd.value = self._parallax_budget_to_display(parallax_budget_key)
        self.depth_separation_label.value = t["Depth Separation:"]
        depth_separation_key = self._display_to_depth_separation(self.depth_separation_dd.value)
        self.depth_separation_dd.options = self._depth_separation_options()
        self.depth_separation_dd.value = self._depth_separation_to_display(depth_separation_key)
        self.temporal_strength_label.value = t["Temporal Strength:"]
        self.scene_reset_label.value = t["Scene Threshold:"]
        self.edge_dilation_label.value = t["Edge Dilation:"]
        self.mask_feather_label.value = t["Mask Feather:"]
        self.edge_threshold_label.value = t["Edge Threshold:"]
        self.hole_fill_mode_label.value = t["Hole Fill Mode:"]
        hole_fill_mode_key = self._display_to_hole_fill_mode(self.hole_fill_mode_dd.value)
        self.hole_fill_mode_dd.options = self._hole_fill_mode_options()
        self.hole_fill_mode_dd.value = self._hole_fill_mode_to_display(hole_fill_mode_key)
        self.anaglyph_label.value = t["Anaglyph:"]
        self.color_brightness_label.value = t["Color Brightness:"]
        self.color_contrast_label.value = t["Color Contrast:"]
        self.color_saturation_label.value = t["Color Saturation:"]
        self.color_gamma_label.value = t["Color Gamma:"]
        self.color_temperature_label.value = t["Color Temperature:"]
        self.color_tint_label.value = t["Color Tint:"]
        self.projection_min_lod_label.value = t["Projection Min LOD:"]
        self.projection_max_lod_label.value = t["Projection Max LOD:"]
        self.projection_mip_lod_bias_label.value = t["Projection MIP Bias:"]
        self.projection_rcas_sharpness_label.value = t["Projection RCAS:"]
        projection_tooltips = (
            (self.projection_min_lod_label, self.projection_min_lod_dd, t["Projection Min LOD tooltip"]),
            (self.projection_max_lod_label, self.projection_max_lod_dd, t["Projection Max LOD tooltip"]),
            (self.projection_mip_lod_bias_label, self.projection_mip_lod_bias_dd, t["Projection MIP Bias tooltip"]),
            (self.projection_rcas_sharpness_label, self.projection_rcas_sharpness_dd, t["Projection RCAS tooltip"]),
        )
        for label, dropdown, tooltip in projection_tooltips:
            label.tooltip = tooltip
            dropdown.set_tooltip(tooltip)
        self.cross_eyed_cb.label = t["Cross Eyed"]
        self.advanced_stereo_cb.label = t["Advanced Stereo"]
        self.acceleration_label.value = t["Inference Acceleration:"]
        self.torch_compile_cb.label = t["torch.compile"]
        self.tensorrt_cb.label = t["TensorRT"]
        parallel_workers = self._display_to_parallel_inference_workers(
            self.parallel_inference_dd.value
        )
        self.parallel_inference_label.value = t.get("Parallel Inference", "Parallel Inference") + ":"
        self.parallel_inference_dd.options = self._parallel_inference_options()
        self.parallel_inference_dd.value = self._parallel_inference_to_display(parallel_workers)
        self.coreml_cb.label = t["CoreML"]
        self.openvino_cb.label = t["OpenVINO"]
        self.recompile_trt_cb.label = t["Recompile TensorRT"]
        self.recompile_coreml_cb.label = t["Recompile CoreML"]
        self.recompile_openvino_cb.label = t["Recompile OpenVINO"]
        self.computing_device_label.value = t["Computing Device:"]
        self.advanced_device_cb.label = t["Advanced Device Options"]
        self.showfps_cb.label = t["Show FPS"]
        self.capture_tool_label.value = t["Capture Tool:"]
        self.run_mode_label.value = t["Run Mode:"]
        self.stream_settings_cb.label = t.get("Stream Settings", "Stream Settings")
        self.display_mode_label.value = t["Display Mode:"]
        self.xr_headset_label.value = t.get("Headset Model:", "Headset Model:")
        headset_key = display_to_xr_headset(self.xr_headset_dd.value)
        self.xr_headset_dd.options = xr_headset_options(self.locale)
        self.xr_headset_dd.value = xr_headset_to_display(headset_key, self.locale)
        self.xr_preview_cb.label = t.get("XR Preview Window", "XR画面预览窗口" if self.locale == "CN" else "XR Preview Window")
        self.local_vsync_cb.label = t.get("VSync", "VSync")
        self.window_preview_cb.label = t.get("Window Preview", "Window Preview")
        self.target_fps_label.value = t.get("Capture FPS:", "Capture FPS:")
        target_fps_value = self._target_fps_from_display(self.target_fps_dd.value)
        self.target_fps_dd.options = [t["Auto"]] + [str(fps) for fps in range(5, 95, 5)]
        self.target_fps_dd.value = self._target_fps_to_display(target_fps_value)
        self.render_policy_label.value = t["Render Policy:"]
        self.render_policy_dd.options = ["Scaled"]
        self.render_policy_dd.value = "Scaled"
        self.render_scale_label.value = t["Render Scale:"]
        render_scale = self._display_to_render_scale(self.render_scale_dd.value)
        self.render_scale_dd.options = self._render_scale_options()
        self.render_scale_dd.value = self._render_scale_to_display(render_scale)
        self.render_fixed_label.value = t["Render Fixed Size:"]
        self.render_max_pixels_label.value = t["Render Pixel Cap:"]
        self.render_min_dimension_label.value = t["Render Min Side:"]
        self.render_align_label.value = t["Render Align:"]
        self.upscaler_label.value = t.get("Upscaler:", "Upscaler:")
        self.upscaler_sharpness_label.value = t.get("Upscaler Sharpness:", "Sharpness:")
        self.upscaler_dd.options = self._upscaler_display_options()
        self.upscaler_dd.value = self._upscaler_to_display("Off")
        self.upscaler_sharpness_dd.value = "0.00"
        self.stereo_output_label.value = t["Stereo Output:"]
        self.update_stereo_monitor_menu()
        self.theme_label.value = t["Theme:"]
        theme_display = ["System", "Blue", "Green", "Red", "Purple", "Orange", "Teal", "Pink", "Grey"]
        self.theme_dd.options = [t.get(k.lower(), k) for k in theme_display]
        cur = self.theme_dd.value
        cn_map = {"系统": "system", "蓝色": "blue", "绿色": "green", "红色": "red",
                  "紫色": "purple", "橙色": "orange", "青色": "teal", "粉色": "pink", "灰色": "grey"}
        key = cn_map.get(cur, cur.lower() if cur else "system")
        self.theme_dd.value = t.get(key, key) if self.locale == "CN" else key.capitalize()
        display_fit_mode = self._display_to_display_fit(self.display_fit_dd.value)
        self.display_fit_dd.options = self._display_fit_options()
        self.display_fit_dd.value = self._display_fit_to_display(display_fit_mode)
        self.display_fit_dd.set_tooltip(t["tooltip_display_fit"])
        if hasattr(self, "stream_display_fit_label"):
            self.stream_display_fit_label.value = t.get("Stream Fit:", "Stream Fit:")
            self.stream_display_fit_label.tooltip = t["tooltip_display_fit"]
        self.lossless_cb.label = t["Lossless Scaling Support"]
        self.lossless_cb.tooltip = t["tooltip_nvfruc"]
        self.controller_label.value = t["Controller:"]
        self.environment_label.value = t["Environment:"]
        self._refresh_environment_options()
        self.lang_label.value = t["Set Language:"]
        run_mode_texts = {}
        if OS_NAME == "Darwin":
            run_mode_texts = {
                "Local Viewer": t["Local Viewer"], "RTMP Streamer": t["RTMP Streamer"],
                "MJPEG Streamer": t["MJPEG Streamer"],
            }
        else:
            run_mode_texts = {
                "Local Viewer": t["Local Viewer"], "OpenXR Link": t["OpenXR Link"],
                "RTMP Streamer": t["RTMP Streamer"], "MJPEG Streamer": t["MJPEG Streamer"],
            }
            if OS_NAME == "Windows":
                run_mode_texts["3D Monitor"] = t["3D Monitor"]
        self.run_mode_dd.options = [v for v in run_mode_texts.values()]
        self.capture_mode_dd.options = [t["Monitor"], t["Window"]]
        self.capture_mode_dd.value = t["Monitor"] if self.capture_mode_key == "Monitor" else t["Window"]
        self.capture_mode_dd.reapply_width()
        self.stream_url_label.value = t["Streamer URL"]
        self.stream_port_label.value = t["Streamer Port:"]
        self.stream_quality_label.value = t["Stream Quality:"]
        self.stream_proto_label.value = t["Stream Protocol:"]
        self.stream_key_label.value = t["Stream Key"]
        calibration_auto = self._stream_calibration_auto_enabled()
        self.stream_calibration_label.value = t.get(
            "Transmission Profile:", "Transmission Profile:"
        )
        self.stream_calibration_mode_dd.options = [
            t.get("Auto Calibration", "Auto Calibration"),
            t.get("Manual", "Manual"),
        ]
        self.stream_calibration_mode_dd.value = self.stream_calibration_mode_dd.options[
            0 if calibration_auto else 1
        ]
        self.stream_calibration_btn.content.value = t.get(
            "Start Calibration", "Start Calibration"
        )
        self.audio_label.value = t["Stereo Mix"]
        self.crf_label.value = t["CRF"]
        self.audio_delay_label.value = t["Audio Delay"]
        self.preview_btn.content.value = t["Preview"]
        self.refresh_btn.content.value = t["Refresh"]
        self.reset_btn.content.value = t["Reset"]
        self.stop_btn.content.value = t["Stop"]
        self.run_btn.content.value = t["Run"]
        self.report_issue_btn.content.value = t.get("Report issue", "Report bug")
        self.open_log_file_btn.content.value = t.get("Open log file", "Open log")
        self._sync_log_visibility_link()

        def _set_tooltip(ctrl, text):
            if hasattr(ctrl, "set_tooltip"):
                ctrl.set_tooltip(text)
            else:
                try:
                    ctrl.tooltip = text
                except Exception:
                    pass

        _set_tooltip(self.window_dd, t["tooltip_window"])
        for ctrl, key in [
            (self.depth_model_dd, "tooltip_depth_model"),
            (self.model_size_dd, "tooltip_model_size"),
            (self.depth_res_dd, "tooltip_depth_res"),
            (self.convergence_dd, "tooltip_convergence"),
            (self.dynamic_convergence_strength_dd, "tooltip_dynamic_convergence_strength"),
            (self.depth_strength_dd, "tooltip_depth_strength"),
            (self.depth_quick_dd, "tooltip_depth_quick"),
            (self.stereo_preset_dd, "tooltip_stereo_preset"),
            (self.parallax_budget_dd, "tooltip_parallax_budget"),
            (self.depth_separation_dd, "tooltip_depth_separation"),
            (self.temporal_strength_dd, "tooltip_temporal_strength"),
            (self.scene_reset_dd, "tooltip_scene_reset"),
            (self.edge_dilation_dd, "tooltip_edge_dilation"),
            (self.mask_feather_dd, "tooltip_mask_feather"),
            (self.edge_threshold_dd, "tooltip_edge_threshold"),
            (self.hole_fill_mode_dd, "tooltip_hole_fill_mode"),
            (self.anaglyph_dd, "tooltip_anaglyph"),
            (self.parallel_inference_label, "tooltip_parallel_inference"),
            (self.parallel_inference_dd, "tooltip_parallel_inference"),
            (self.cross_eyed_cb, "tooltip_cross_eyed"),
            (self.advanced_stereo_cb, "tooltip_advanced_stereo"),
            (self.depth_pop_dd, "tooltip_depth_pop"),
            (self.foreground_pop_dd, "tooltip_foreground_pop"),
            (self.midground_pop_dd, "tooltip_midground_pop"),
            (self.background_pop_dd, "tooltip_background_pop"),
            (self.antialiasing_dd, "tooltip_antialiasing"),
            (self.device_dd, "tooltip_device"),
            (self.advanced_device_cb, "tooltip_advanced_device_options"),
            (self.capture_tool_dd, "tooltip_capture_tool"),
            (self.run_mode_dd, "tooltip_run_mode"),
            (self.display_mode_dd, "tooltip_display_mode"),
            (self.xr_headset_dd, "tooltip_xr_headset"),
            (self.local_vsync_cb, "tooltip_vsync"),
            (self.window_preview_cb, "tooltip_window_preview"),
            (self.target_fps_dd, "tooltip_target_fps"),
            (self.xr_preview_cb, "tooltip_xr_preview"),
            (self.render_policy_dd, "tooltip_render_policy"),
            (self.render_scale_dd, "tooltip_render_scale"),
            (self.render_fixed_dd, "tooltip_render_fixed_size"),
            (self.render_max_pixels_dd, "tooltip_render_max_pixels"),
            (self.render_min_dimension_dd, "tooltip_render_min_dimension"),
            (self.render_align_dd, "tooltip_render_align"),
            (self.ctrl_model_dd, "tooltip_ctrl_model"),
            (self.env_model_dd, "tooltip_env_model"),
            (self.capture_mode_dd, "tooltip_capture_mode"),
            (self.monitor_dd, "tooltip_monitor"),
            (self.stereo_monitor_dd, "tooltip_stereo_monitor"),
            (self.lang_dd, "tooltip_lang"),
            (self.theme_dd, "tooltip_theme"),
            (self.stream_settings_cb, "tooltip_stream_settings"),
            (self.stream_url_tf, "tooltip_stream_url"),
            (self.preview_btn, "tooltip_stream_preview"),
            (self.stream_quality_dd, "tooltip_stream_quality"),
            (self.stream_proto_dd, "tooltip_stream_proto"),
            (self.audio_dd, "tooltip_audio"),
            (self.stream_port_tf, "tooltip_stream_port"),
            (self.stream_key_tf, "tooltip_stream_key"),
            (self.crf_tf, "tooltip_crf"),
            (self.audio_delay_tf, "tooltip_audio_delay"),
            (self.video_backend_dd, "tooltip_video_backend"),
            (self.stream_calibration_mode_dd, "tooltip_stream_calibration_mode"),
            (self.stream_calibration_btn, "tooltip_stream_calibration_start"),
        ]:
            _set_tooltip(ctrl, t.get(key, UI_MESSAGES["EN"].get(key, key)))
        self._refresh_stream_calibration_status()
        self._auto_align_labels(force=True)

    def _safe_update(self, *controls):
        for c in controls:
            if c is None:
                continue
            try:
                c.update()
            except RuntimeError:
                pass
            except Exception as e:
                logger.warning("_safe_update failed: %s", e)

    # ── stream URL ──

    def _format_stream_url(self, local_ip, protocol, port, stream_key):
        if self.run_mode_key == "MJPEG Streamer":
            return f"http://{local_ip}:{port}/"
        templates = {
            "RTMP": f"rtmp://{local_ip}:{port}/{stream_key}",
            "RTSP": f"rtsp://{local_ip}:{port}/{stream_key}",
            "HLS": f"http://{local_ip}:{port}/{stream_key}/",
            "HLS M3U8": f"http://{local_ip}:{port}/{stream_key}/index.m3u8",
            "WebRTC": f"http://{local_ip}:{port}/{stream_key}/",
        }
        return templates.get(protocol, f"http://{local_ip}:{port}/{stream_key}/")

    async def _refresh_local_ip_async(self):
        try:
            local_ip = await asyncio.to_thread(get_local_ip)
        except Exception:
            local_ip = "127.0.0.1"
        if local_ip and local_ip != getattr(self, "_local_ip_cache", None):
            self._local_ip_cache = local_ip
            self.update_stream_url(resolve_ip=False)
        self._local_ip_task = None

    def _schedule_local_ip_refresh(self):
        task = getattr(self, "_local_ip_task", None)
        if task is not None and not task.done():
            return
        try:
            self._local_ip_task = asyncio.create_task(self._refresh_local_ip_async())
        except RuntimeError:
            self._local_ip_cache = get_local_ip()

    def update_stream_url(self, e=None, resolve_ip=True):
        if self.run_mode_key not in {"MJPEG Streamer", "RTMP Streamer"}:
            return
        protocol = self.stream_proto_dd.value
        port = self.stream_port_tf.value or str(DEFAULT_PORT)
        stream_key = self.stream_key_tf.value or "live"
        local_ip = getattr(self, "_local_ip_cache", "127.0.0.1")
        self.stream_url_tf.value = self._format_stream_url(local_ip, protocol, port, stream_key)
        self._safe_update(self.stream_url_tf)
        self.preview_btn.visible = protocol not in ["RTMP", "RTSP"]
        self._safe_update(self.preview_btn)
        if resolve_ip:
            self._schedule_local_ip_refresh()

    def _on_stream_protocol_change(self, e):
        self.stream_protocol_key = e.control.value
        self._config["Stream Protocol"] = self.stream_protocol_key
        self.update_stream_url()
        self._fit_window_to_content()

    def on_audio_device_change(self, e):
        # Persist an explicit Stereo Mix selection on macOS only (v2.5.0
        # parity: the configured device name is captured by the runtime).
        # Windows/Linux resolve the default output device on every start and
        # must not save a stale name. Placeholder labels are never saved.
        if OS_NAME != "Darwin":
            return
        value = str(getattr(e, "control", None) and e.control.value or "").strip()
        if not value:
            return
        placeholders = (
            "no stereo mix device found",
            "sounddevice not available",
            "no audio capture devices found",
            "no audio devices found",
            "no audio sources found",
            "pacmd not available",
        )
        if value.casefold() in placeholders:
            self._config.pop("Stereo Mix", None)
            return
        self._config["Stereo Mix"] = value

    def _on_stream_key_change(self, e):
        val = e.control.value or ""
        if not re.match(r'^[A-Za-z0-9_-]*$', val) or len(val) > 64:
            self.set_status(UI_MESSAGES[self.locale]["err_stream_key"])
        self._config["Stream Key"] = val
        self.update_stream_url()

    # ── audio ──

    def populate_audio_devices(self):
        if OS_NAME == "Linux":
            self._populate_audio_linux()
        else:
            self._populate_audio_generic()
        self._apply_audio_devices_to_dropdown()

    def _saved_audio_device(self):
        """Return the persisted Stereo Mix device when it is still present.

        The runtime (and the old GUI) may store a bare device name or a
        "soundcard:"/"wasapi:"-prefixed label; match case-insensitively
        against the currently discovered devices so the dropdown reflects the
        saved choice instead of defaulting to the first discovered device.
        """
        saved = str((getattr(self, "_config", None) or {}).get("Stereo Mix", "") or "").strip()
        if not saved:
            return None
        for prefix in ("soundcard:", "wasapi:"):
            if saved.casefold().startswith(prefix):
                saved = saved.split(":", 1)[1].strip()
                break
        if not saved:
            return None
        for device in self.audio_devices or []:
            if str(device).casefold() == saved.casefold():
                return device
        return None

    def _apply_audio_devices_to_dropdown(self):
        if self.audio_devices:
            self.audio_dd.options = [d for d in self.audio_devices]
            # Prefer the persisted Stereo Mix device so the dropdown reflects
            # what the stream will actually capture (macOS persists the
            # selection; Windows/Linux keep the auto-resolved default).
            self.audio_dd.value = self._saved_audio_device() or self.audio_devices[0]
            self.audio_dd.update()
            if self.audio_devices[0] in ["No Stereo Mix device found", "sounddevice not available"]:
                self.set_status(self.audio_devices[0])

    async def populate_audio_devices_after_startup(self):
        """Discover slow audio backends without delaying the first GUI frame."""
        try:
            if OS_NAME == "Linux":
                await asyncio.to_thread(self._populate_audio_linux)
            else:
                await asyncio.to_thread(self._populate_audio_generic)
            if getattr(self, "_closed", False):
                return
            self._apply_audio_devices_to_dropdown()
            self.auto_select_stereo_mix()
            self._fit_window_to_content(update=True, resize_window=True)
        except (asyncio.CancelledError, RuntimeError):
            return

    def _populate_audio_generic(self):
        self.audio_devices = []
        if OS_NAME == "Windows":
            ffmpeg_path = os.path.join(
                BASE_DIR,
                "streaming",
                "rtmp",
                "ffmpeg",
                "bin",
                "ffmpeg.exe",
            )

            def append_virtual_audio_capture(devices):
                for device in devices or []:
                    if (
                        "virtual-audio-capturer" in str(device).casefold()
                        and device not in self.audio_devices
                    ):
                        self.audio_devices.append(device)

            soundcard_devices = query_soundcard_loopback_devices()
            if soundcard_devices:
                self.audio_devices = [f"soundcard:{name}" for name in soundcard_devices]
                append_virtual_audio_capture(
                    query_ffmpeg_dshow_audio_devices(ffmpeg_path)
                )
                logger.info(
                    "Using SoundCard WASAPI loopback audio devices: %s",
                    self.audio_devices,
                )
                return
            wasapi_devices = query_ffmpeg_wasapi_audio_devices(ffmpeg_path)
            if wasapi_devices:
                self.audio_devices = [f"wasapi:{name}" for name in wasapi_devices]
                append_virtual_audio_capture(
                    query_ffmpeg_dshow_audio_devices(ffmpeg_path)
                )
                logger.info(
                    "Using Windows WASAPI loopback audio devices: %s",
                    self.audio_devices,
                )
                return
            dshow_devices = query_ffmpeg_dshow_audio_devices(ffmpeg_path)
            if dshow_devices is not None:
                self.audio_devices = find_loopback_audio_devices(dshow_devices)
                if not self.audio_devices:
                    self.audio_devices = ["virtual-audio-capturer"]
                return
        try:
            import sounddevice as sd

            all_devices = sd.query_devices()
            eligible_names = [
                dev.get("name")
                for dev in all_devices
                if (
                    dev.get("max_input_channels", 0) > 0
                    or dev.get("max_output_channels", 0) > 0
                )
            ]
            found = find_loopback_audio_devices(eligible_names)
            if not found and OS_NAME == "Darwin":
                logger.info("No audio capture devices found on MacOS. Recommended tools: BlackHole, Virtual Desktop Streamer, Loopback")
                self.audio_devices = ["No audio capture devices found"]
            elif not found and OS_NAME == "Windows":
                self.audio_devices = ["virtual-audio-capturer"]
            else:
                self.audio_devices = found or ["No Stereo Mix device found"]
        except ImportError:
            self.audio_devices = ["sounddevice not available"]
        except Exception as e:
            self.audio_devices = [f"Error: {e}"]

    def _populate_audio_linux(self):
        self.audio_devices = []
        try:
            result = subprocess.run(["pacmd", "list-sources"], capture_output=True, text=True, check=True)
            sources = []
            for block in result.stdout.split("index:")[1:]:
                m = re.search(r"name:\s*<(.+?)>", block)
                if m:
                    sources.append(m.group(1))
            self.audio_devices = sources or ["No audio sources found"]
        except Exception:
            self.audio_devices = ["pacmd not available"]

    def auto_select_stereo_mix(self):
        if not self.audio_devices:
            return
        # A persisted macOS selection wins over the first-discovered loopback
        # (e.g. a silent BlackHole); Windows/Linux keep their auto behavior.
        saved = self._saved_audio_device()
        if saved is not None:
            self.audio_dd.value = saved
            self.audio_dd.update()
            return
        for dev in self.audio_devices:
            dl = dev.lower()
            if dl.startswith(("soundcard:", "wasapi:")):
                self.audio_dd.value = dev
                self.audio_dd.update()
                return
            for mix in STEREO_MIX_NAMES:
                if mix in dl and "audio stereo input" not in dl:
                    self.audio_dd.value = dev
                    self.audio_dd.update()
                    return
        for dev in self.audio_devices:
            if "virtual-audio-capturer" in dev.lower():
                self.audio_dd.value = dev
                self.audio_dd.update()
                return
        self.audio_dd.value = "No Stereo Mix device found"
        self.audio_dd.update()

    def _ensure_audio_device_selected(self):
        if not self.audio_devices:
            return
        self.audio_dd.options = [device for device in self.audio_devices]
        if self.audio_dd.value in self.audio_devices:
            return
        self.auto_select_stereo_mix()

    # ── refresh ──

    def refresh_monitor_and_window(self, e=None):
        self.populate_monitors()
        if self.capture_mode_key == "Window":
            self.refresh_window_list()
        if self.run_mode_key == "RTMP Streamer":
            self.populate_audio_devices()
            self.auto_select_stereo_mix()
        self.update_stereo_monitor_menu()
        self._sync_visibility()
        self._fit_window_to_content()
        if self.capture_mode_key == "Monitor" and self.monitor_dd.value:
            self.set_status(f"{UI_MESSAGES[self.locale]['Selected input monitor:']} {self.monitor_dd.value}")
        elif self.capture_mode_key == "Window" and self.selected_window_name:
            self.set_status(f"{UI_MESSAGES[self.locale]['Selected input window:']} {self.selected_window_name}")

    def refresh_window_list(self):
        try:
            windows = list_windows()
            if not windows:
                self.window_dd.options = []
                self.window_dd.update()
                return
            self._window_objects = windows
            win_labels = [f"{w['title']} [h:{w['handle'] or 0}]" for w in windows]
            self.window_dd.options = win_labels
            if self.selected_window_name:
                labels_by_title = [lbl for lbl in win_labels if lbl.startswith(self.selected_window_name + " [")]
                selected_handle_str = f"[h:{self.selected_window_handle or 0}]"
                match = next((lbl for lbl in labels_by_title if selected_handle_str in lbl), None)
                if match:
                    self.window_dd.value = match
                elif labels_by_title:
                    self.window_dd.value = labels_by_title[0]
                    self.on_window_selected(None)
                else:
                    self.window_dd.value = win_labels[0] if windows else ""
                    self.on_window_selected(None)
            elif windows:
                self.window_dd.value = win_labels[0]
                self.on_window_selected(None)
            self.window_dd.update()
        except Exception as e:
            self.set_status(UI_MESSAGES[self.locale].get("err_refresh_window", "Failed to refresh window list: {}").format(e))

    # ── stereo preset / advanced stereo ──

    def on_stereo_preset_change(self, e=None):
        preset = self._display_to_preset(self.stereo_preset_dd.value)
        if self._apply_stereo_preset_values(preset):
            self._schedule_stereo_hot_save()
        else:
            self.on_stereo_hot_param_change(e)

    def _apply_stereo_preset_values(self, preset):
        values = self._stereo_preset_gui_values(preset)
        if not values:
            return False
        self.stereo_quality_dd.value = self._stereo_quality_to_display(values["quality"])
        self.parallax_budget_dd.value = self._parallax_budget_to_display(values["parallax_budget"])
        self.depth_strength_dd.value = f"{values['depth_strength']:.2f}"
        self.depth_quick_dd.value = self._depth_quick_to_display(values["depth_quick"])
        self.convergence_dd.value = f"{values['convergence']:.2f}"
        self.dynamic_convergence_strength_dd.value = f"{values.get('dynamic_convergence_strength', 0.0):.2f}"
        self.temporal_strength_dd.value = f"{values['temporal_strength']:.2f}"
        self.scene_reset_dd.value = f"{values['scene_reset_threshold']:.2f}"
        self.depth_pop_dd.value = f"{values['depth_pop']:.1f}"
        self.depth_separation_dd.value = self._depth_separation_to_display(values["depth_separation"])
        self.foreground_pop_dd.value = f"{values['foreground_pop']:.2f}"
        self.midground_pop_dd.value = f"{values['midground_pop']:.2f}"
        self.background_pop_dd.value = f"{values['background_pop']:.2f}"
        self.antialiasing_dd.value = str(values["antialiasing"])
        self.edge_dilation_dd.value = str(values["edge_dilation"])
        self.mask_feather_dd.value = str(values["mask_feather_radius"])
        self.hole_fill_mode_dd.value = self._hole_fill_mode_to_display(values["hole_fill_mode"])
        self.edge_threshold_dd.value = f"{values['edge_threshold']:.2f}"
        self.cross_eyed_cb.value = bool(values.get("cross_eyed", False))
        self.color_brightness_dd.value = f"{values.get('color_brightness', 1.0):.1f}"
        self.color_contrast_dd.value = f"{values.get('color_contrast', 1.0):.1f}"
        self.color_saturation_dd.value = f"{values.get('color_saturation', 1.0):.1f}"
        self.color_gamma_dd.value = f"{values.get('color_gamma', 1.0):.1f}"
        self.color_temperature_dd.value = str(int(values.get('color_temperature', 0.0)))
        self.color_tint_dd.value = str(int(values.get('color_tint', 0.0)))
        for ctrl in (
            self.stereo_quality_dd, self.parallax_budget_dd, self.depth_strength_dd, self.depth_quick_dd,
            self.convergence_dd, self.dynamic_convergence_strength_dd, self.temporal_strength_dd,
            self.scene_reset_dd, self.depth_pop_dd, self.depth_separation_dd, self.foreground_pop_dd,
            self.midground_pop_dd, self.background_pop_dd, self.antialiasing_dd, self.edge_dilation_dd,
            self.mask_feather_dd, self.edge_threshold_dd, self.hole_fill_mode_dd, self.cross_eyed_cb,
            self.color_brightness_dd, self.color_contrast_dd, self.color_saturation_dd, self.color_gamma_dd,
            self.color_temperature_dd, self.color_tint_dd,
        ):
            self._safe_update(ctrl)
        return True

    def on_depth_separation_change(self, e=None):
        key = self._display_to_depth_separation(self.depth_separation_dd.value)
        foreground, midground, background = self._depth_separation_values(key)
        self.foreground_pop_dd.value = f"{foreground:.2f}"
        self.midground_pop_dd.value = f"{midground:.2f}"
        self.background_pop_dd.value = f"{background:.2f}"
        self._safe_update(self.foreground_pop_dd, self.midground_pop_dd, self.background_pop_dd)
        self._schedule_stereo_hot_save()

    def on_advanced_stereo_change(self, e):
        self._sync_advanced_stereo_visibility()
        self._fit_window_to_content(update=True, resize_window=True)
        self._safe_update(self.page)

    def _sync_advanced_stereo_visibility(self):
        visible = bool(getattr(self, "advanced_stereo_cb", None) and self.advanced_stereo_cb.value)
        for row in getattr(self, "_advanced_stereo_rows", []):
            row.visible = visible

    def on_depth_quick_change(self, value):
        quick = value if isinstance(value, str) else getattr(getattr(value, "control", None), "value", self.depth_quick_dd.value)
        strength = self._depth_strength_for_quick(self._display_to_depth_quick(quick))
        self.depth_strength_dd.value = f"{strength:.2f}"
        self._safe_update(self.depth_strength_dd)
        self._schedule_stereo_hot_save()
