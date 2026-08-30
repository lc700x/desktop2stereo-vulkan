"""Desktop2Stereo Flet GUI — main application class combining all mixins.

Mixins:
  GUIBuilderMixin  — UI construction, layout sizing, data population
  GUIHandlerMixin  — event handlers, visibility sync, i18n, audio, refresh
  GUIConfigMixin   — config read/write, stereo preset data, hot-param save
  GUIProcessMixin  — subprocess lifecycle, ESC monitoring, URL actions
"""
import os
import asyncio
import logging
from .flet_runtime import ensure_vendored_flet_view

ensure_vendored_flet_view()

import flet as ft
from utils import VERSION, OS_NAME, bootstrap_settings
from .builders import GUIBuilderMixin
from .handlers import GUIHandlerMixin
from .config_mgr import GUIConfigMixin
from .process import GUIProcessMixin, _setup_console_logging
from .config import DEFAULTS
from .controls import S
from .paths import BASE_DIR, GUI_READY_FILE, LOG_DIR
from .localization import UI_MESSAGES


logger = logging.getLogger(__name__)


def _write_gui_ready_flag():
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(GUI_READY_FILE, "w", encoding="utf-8") as ready_file:
            ready_file.write("ready\n")
    except Exception:
        logger.exception("Failed to write GUI ready flag")


class Desktop2StereoGUI(
    GUIBuilderMixin,
    GUIHandlerMixin,
    GUIConfigMixin,
    GUIProcessMixin,
):
    """Flet GUI for Desktop2Stereo — full equivalent of tk ConfigGUI."""
    def __init__(self, page: ft.Page):
        self.page = page
        self._loop = None
        self.locale = "EN"
        self._config = {}
        self.run_mode_key = DEFAULTS.get("Run Mode", "Local Viewer")
        self.capture_mode_key = DEFAULTS.get("Capture Mode", "Monitor")
        self.stream_protocol_key = DEFAULTS.get("Stream Protocol", "WebRTC")
        self.selected_window_name = ""
        self.selected_window_handle = None
        self.selected_window_rect = None
        self._window_objects = []
        self.process = None
        self._starting = False
        self._proc_lock = None
        self.monitor_label_to_index = {}
        self.monitor_label_to_display = {}
        self._missing_monitor_identity = False
        self._missing_stereo_output_identity = False
        self.device_label_to_index = {}
        self._esc_down = None
        self._esc_stopped = False
        self._closed = False
        self._cancel_starting = False
        self._stopping = False
        self._labels_aligned = False
        self._status_key = ""
        self._local_ip_cache = "127.0.0.1"
        self._local_ip_task = None
        self.gui_log_handler = None
        self._log_poll_task = None
        self._resize_repaint_task = None
        self._progress_log_spans = {}
        self._calibration_run_requested = False
        self._calibration_poll_task = None
        self._calibration_dialog = None
        self._display_refresh_warning_dialog = None
        self._display_refresh_warning_payload = None
        self._pending_display_refresh_warnings = []
        self._calibration_active = False
        self._calibration_previous_target_value = None
        self.audio_devices = []
        self._startup_defer_audio = True
        self._audio_startup_task = None
        self._startup_defer_devices = True
        self._device_startup_task = None

    async def setup(self):
        self.gui_log_handler = _setup_console_logging()
        self._loop = asyncio.get_running_loop()
        self._proc_lock = asyncio.Lock()
        self._hot_save_task = None

        self.page.title = f"Desktop2Stereo v{VERSION}"
        self.page.window.icon = os.path.join(BASE_DIR, "icon.ico")
        self.page.padding = S(24)
        self.page.horizontal_alignment = ft.CrossAxisAlignment.STRETCH
        if OS_NAME == "Windows":
            font = "Microsoft YaHei"
        elif OS_NAME == "Darwin":
            font = "PingFang SC"
        else:
            font = "Noto Sans SC"
        self.page.theme = ft.Theme(color_scheme_seed="blue", font_family=font)
        self.page.spacing = 0
        self.page.theme_mode = ft.ThemeMode.SYSTEM
        self.page.window.min_width = S(520)
        self.page.window.min_height = S(300)

        # Build UI
        self.build_ui()
        self.page.update()
        self._log_poll_task = asyncio.create_task(self._poll_log_queue())
        self._auto_align_labels()
        self.page.on_close = self._on_page_close
        self.page.on_resize = self._on_page_resize
        self._startup_fit_armed = True
        self._startup_fit_task = None

        # Populate monitors first so apply_config can select the saved monitor
        self.monitor_label_to_index = self.populate_monitors()
        self.page.update()

        # Load config
        self._config = DEFAULTS.copy()
        if os.path.exists(os.path.join(BASE_DIR, "settings.yaml")):
            try:
                cfg = bootstrap_settings(os.path.join(BASE_DIR, "settings.yaml"), os_name=OS_NAME)
                if cfg:
                    self._config.update(cfg)
                    self._yaml_loaded = True
                    self.locale = self._config.get("Language", "EN")
                    os.environ["DESKTOP2STEREO_LOCALE"] = self.locale
                    self.apply_config(self._config)
                    self.set_status(UI_MESSAGES[self.locale]["Loaded settings.yaml at startup"],
                                    key="Loaded settings.yaml at startup")
            except Exception as e:
                self.apply_config(self._config)
                self.set_status(
                    f"{UI_MESSAGES[self.locale]['Failed to load settings.yaml:']} {e}")
        else:
            self.apply_config(self._config)

        self.page.update()

        # Now that config is applied (which sets monitor_dd.value), populate windows
        # and select the saved window if in Window capture mode
        self.refresh_window_list()
        self.update_stereo_monitor_menu()
        self._sync_visibility()
        self._fit_window_to_content()

        # Apply the monitor selection to the UI
        if hasattr(self, 'monitor_dd') and self.monitor_dd.value:
            self.monitor_dd.update()
        self.page.update()

        self.on_device_change(None)
        self._refresh_stream_calibration_status()
        self.auto_enable_optimizers_based_on_device()
        self.page.on_keyboard_event = self._on_key
        self._esc_task = asyncio.ensure_future(self._esc_poll_task())
        self._set_log_panel_visible(self._config.get("Show Log Panel", DEFAULTS["Show Log Panel"]), update=False)
        # Size the window to the fitted content BEFORE showing it. Applying the
        # size after visible=True lets the Flet client fall back to its own
        # default / restored width on the first render, which is why the
        # hide-log window opened wider than its fitted width on first launch.
        self._fit_window_to_content(update=False, resize_window=True)
        self.page.window.visible = True
        self.page.update()
        self._device_startup_task = asyncio.create_task(
            self.populate_devices_after_startup()
        )
        self._startup_defer_audio = False
        if self.run_mode_key == "RTMP Streamer":
            self._audio_startup_task = asyncio.create_task(
                self.populate_audio_devices_after_startup()
            )
        self._schedule_startup_fit()
        await asyncio.sleep(0)
        self._fit_window_to_content(update=True, resize_window=True)
        self._signal_gui_ready()
        asyncio.create_task(self._prepare_startup_after_window_visible())

    def _signal_gui_ready(self):
        _write_gui_ready_flag()

    async def _prepare_startup_after_window_visible(self):
        self.set_status(
            UI_MESSAGES[self.locale].get(
                "Preparing Flet package...",
                "Preparing Flet desktop client...",
            ),
            key="Preparing Flet package...",
        )
        try:
            await asyncio.to_thread(ensure_vendored_flet_view)
            self.set_status(
                UI_MESSAGES[self.locale].get(
                    "Startup preparation complete",
                    "Startup preparation complete.",
                ),
                key="Startup preparation complete",
            )
            # Startup logging is done: release the file so it is not locked
            # while the GUI idles before the first run.
            self._release_file_log_handler_if_idle()
        except Exception as exc:
            logger.exception("Startup preparation failed")
            message = UI_MESSAGES[self.locale].get(
                "Startup preparation failed: {}",
                "Startup preparation failed: {}",
            )
            self.set_status(message.format(exc))


def main():
    """Entry point for the GUI application."""
    _setup_console_logging()
    ft.run(_async_main)


async def _async_main(page: ft.Page):
    # Keep the native window visible to avoid hide/show ordering races. Replace
    # Flet's large blank default frame with a compact startup surface while
    # synchronous configuration and hardware discovery are still running.
    page.window.width = S(520)
    page.window.height = S(300)
    page.padding = S(24)
    page.add(
        ft.Container(
            content=ft.Column(
                [
                    ft.ProgressRing(width=S(28), height=S(28)),
                    ft.Text("Desktop2Stereo is starting...", size=S(14)),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=S(12),
            ),
            expand=True,
            alignment=ft.Alignment.CENTER,
        )
    )
    page.update()
    await asyncio.sleep(0.1)
    # The launcher only needs confirmation that a visible GUI surface exists;
    # compute/audio discovery and full menu construction continue afterward.
    _write_gui_ready_flag()
    app = Desktop2StereoGUI(page)
    await app.setup()


if __name__ == "__main__":
    main()
