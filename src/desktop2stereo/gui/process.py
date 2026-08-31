"""GUI Process Mixin — subprocess lifecycle, ESC monitoring, URL actions."""
import os
import re
import sys
import base64
import time
import asyncio
import ctypes
import datetime
import json
import logging
import platform
import queue
import subprocess
import traceback
import flet as ft
from utils import OS_NAME, DEFAULT_PORT, shutdown_event, read_yaml
from . import devices as devices_module
from .config import DEFAULTS, DEFAULT_MODEL_LIST, default_base_depth_model, save_yaml
from .paths import (
    BASE_DIR,
    DIAG_LOG,
    LOG_DIR,
    LOG_FILE,
    STOP_REQUEST_FILE,
    STREAM_CALIBRATION_PROFILE_FILE,
    STREAM_CALIBRATION_STATE_FILE,
)
from .capture_sources import get_primary_monitor_index, list_windows
from .localization import UI_MESSAGES
from .log_handler import GuiLogHandler
from utils.logging_setup import _NoisyThirdPartyDebugFilter
from utils.run_mode import target_fps_setting_key
from streaming.stream_calibration import (
    build_calibration_fingerprint,
    calibration_fingerprint_matches,
    recommended_crf_for_bitrate,
)
from streaming.stream_session import supports_network_calibration

# ── module-level console helpers ──

_NOISY_CONSOLE_PREFIXES = (
    "[NativeUtil] sogou_native_util_pc loaded successfully",
    "[warmup] same version",
    "[INFO] [flet] Session was garbage collected:",
)
_DEBUG_CONSOLE_PREFIXES = (
    "[debug]",
    "debug:",
)
_FLET_LOGGER_NAMES = ("flet", "flet_desktop", "flet_controls", "flet_transport")
_FLET_MESSAGE_PREFIXES = ("[flet]", "[flet_desktop]", "[flet_controls]", "[flet_transport]")
_PROGRESS_PERCENT_RE = re.compile(r"(?P<percent>\d{1,3}(?:\.\d+)?)%")
_TQDM_AMOUNT_RE = re.compile(r"\|\s*(?P<completed>[^/\[\]|]+)\s*/\s*(?P<total>[^\[\]|]+)")
_TQDM_TIMING_RE = re.compile(r"\[(?P<elapsed>[^<,\]]+)(?:<(?P<eta>[^,\]]+))?(?:,\s*(?P<speed>[^\]]+))?\]")
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_VULKAN_DESCRIPTOR_DETAIL_RE = re.compile(
    r"^Descriptor set \(handle=\d+\) binding=\d+ was set between begin/endRenderPass$"
)
_MEDIAMTX_LEVEL_RE = re.compile(
    r"^\[MediaMTX\].*?\s(?P<level>INF|WAR|ERR|DBG)\s"
)
_FILAMENT_FRAME_DETAIL_RE = re.compile(
    r"^\[FilamentBridge\] (?:"
    r"acquired eye=\d+ index=\d+ image=\w+ result=1|"
    r"begin eye=\d+ renderer=\w+ swapchain=\w+ active=1|"
    r"end eye=\d+"
    r")$"
)
_LOG_FILE_LINE_RE = re.compile(r"^\[(?P<asctime>\d\d:\d\d:\d\d)\] \[(?P<level>[A-Z]+)\] \[(?P<name>[^\]]+)\] (?P<message>.*)$")
_LEGACY_LOG_FILE_LINE_RE = re.compile(r"^\[(?P<asctime>\d\d:\d\d:\d\d)\] \[(?P<name>[^\]]+)\] (?P<message>.*)$")
_PROGRESS_PREFIX = "[D2S_PROGRESS] "
_STATUS_PREFIX = "[D2S_STATUS] "
_BACKEND_STATUS_PREFIX = "[D2S_BACKEND_STATUS] "
_DISPLAY_REFRESH_WARNING_PREFIX = "[D2S_DISPLAY_REFRESH_WARNING] "
_ASYNCIO_SHUTDOWN_UNRAISABLE_MODULES = (
    "asyncio.base_subprocess",
    "asyncio.proactor_events",
)
_ASYNCIO_SHUTDOWN_UNRAISABLE_MESSAGES = (
    "Event loop is closed",
    "I/O operation on closed pipe",
)
_GRACEFUL_PROCESS_STOP_TIMEOUT_S = 8.0
_asyncio_shutdown_noise_filter_installed = False
_console_logging_installed = False
_gui_log_handler = None
_file_log_handler = None
logger = logging.getLogger(__name__)
status_logger = logging.getLogger("status")
child_logger = logging.getLogger("child")


class FirewallProbeError(RuntimeError):
    """Raised when Windows Firewall rules cannot be inspected reliably."""


def _parse_firewall_block_output(output):
    """Parse the compact JSON emitted by the Windows firewall probe."""
    text = str(output or "").strip()
    if not text:
        return []
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return []
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _detect_windows_firewall_blocks(executable=None):
    """Return enabled inbound block rules targeting the supplied executable."""
    if OS_NAME != "Windows":
        return []
    executable = os.path.normcase(os.path.abspath(executable or sys.executable))
    probe = r'''
$exe = [Environment]::GetEnvironmentVariable('D2S_FIREWALL_EXE')
$matches = @(
  Get-NetFirewallRule -Direction Inbound -Action Block -Enabled True -ErrorAction SilentlyContinue |
    ForEach-Object {
      $rule = $_
      $app = $rule | Get-NetFirewallApplicationFilter -ErrorAction SilentlyContinue
      if ($app.Program -and ([IO.Path]::GetFullPath($app.Program) -ieq $exe)) {
        $port = $rule | Get-NetFirewallPortFilter -ErrorAction SilentlyContinue
        [PSCustomObject]@{
          Protocol = [string]$port.Protocol
          LocalPort = [string]$port.LocalPort
          DisplayName = [string]$rule.DisplayName
          Name = [string]$rule.Name
        }
      }
    }
)
if ($matches.Count -eq 0) {
  Write-Output '[]'
} else {
  $matches | ConvertTo-Json -Compress
}
'''
    env = os.environ.copy()
    env["D2S_FIREWALL_EXE"] = executable
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", probe],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise FirewallProbeError(
            "Windows Firewall detection timed out after 20 seconds."
        ) from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise FirewallProbeError(
            f"Windows Firewall detection could not start: {exc}"
        ) from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or f"PowerShell exit code {result.returncode}"
        raise FirewallProbeError(f"Windows Firewall detection failed: {detail}")
    return _parse_firewall_block_output(result.stdout)


def _remove_windows_firewall_blocks(executable=None):
    """Remove only matching inbound block rules; return (success, error)."""
    if OS_NAME != "Windows":
        return True, ""
    executable = os.path.normcase(os.path.abspath(executable or sys.executable))
    command = r'''
$exe = [Environment]::GetEnvironmentVariable('D2S_FIREWALL_EXE')
Get-NetFirewallRule -Direction Inbound -Action Block -Enabled True -ErrorAction SilentlyContinue |
  ForEach-Object {
    $rule = $_
    $app = $rule | Get-NetFirewallApplicationFilter -ErrorAction SilentlyContinue
    if ($app.Program -and ([IO.Path]::GetFullPath($app.Program) -ieq $exe)) {
      Remove-NetFirewallRule -Name $rule.Name -ErrorAction Stop
    }
  }
'''
    env = os.environ.copy()
    env["D2S_FIREWALL_EXE"] = executable
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            env=env,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    if result.returncode == 0:
        return True, ""

    # Firewall rule changes normally require elevation. Retry the same narrow
    # rule deletion through one standard Windows UAC prompt.
    encoded = base64.b64encode(command.encode("utf-16le")).decode("ascii")
    elevated = (
        "$p = Start-Process powershell.exe -Verb RunAs -Wait -PassThru "
        f"-ArgumentList @('-NoProfile','-NonInteractive','-EncodedCommand','{encoded}'); "
        "exit $p.ExitCode"
    )
    try:
        elevated_result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", elevated],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            env=env,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    if elevated_result.returncode != 0:
        return False, (
            elevated_result.stderr.strip()
            or result.stderr.strip()
            or "Windows UAC authorization was cancelled"
        )
    return True, ""


def _is_asyncio_shutdown_unraisable(unraisable):
    exc = getattr(unraisable, "exc_value", None)
    if str(exc) not in _ASYNCIO_SHUTDOWN_UNRAISABLE_MESSAGES:
        return False
    obj = getattr(unraisable, "object", None)
    module = getattr(obj, "__module__", "")
    qualname = getattr(obj, "__qualname__", "")
    return module in _ASYNCIO_SHUTDOWN_UNRAISABLE_MODULES and qualname.endswith(".__del__")


def _log_item_from_file_line(line: str):
    text = str(line or "").rstrip("\r\n")
    match = _LOG_FILE_LINE_RE.match(text)
    if match:
        level_name = match.group("level")
        levelno = getattr(logging, level_name, logging.INFO)
        return levelno, match.group("name"), match.group("asctime"), text
    legacy_match = _LEGACY_LOG_FILE_LINE_RE.match(text)
    if legacy_match:
        asctime = legacy_match.group("asctime")
        name = legacy_match.group("name")
        message = legacy_match.group("message")
        return logging.INFO, name, asctime, f"[{asctime}] [INFO] [{name}] {message}"
    return None


def _read_log_file_items():
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as file:
            return [item for line in file if (item := _log_item_from_file_line(line)) is not None]
    except OSError:
        return []


def _install_asyncio_shutdown_noise_filter():
    """Suppress known Windows asyncio transport __del__ noise during GUI shutdown."""
    global _asyncio_shutdown_noise_filter_installed
    if _asyncio_shutdown_noise_filter_installed or not hasattr(sys, "unraisablehook"):
        return
    previous_hook = sys.unraisablehook

    def _desktop2stereo_unraisable_hook(unraisable):
        if _is_asyncio_shutdown_unraisable(unraisable):
            return
        previous_hook(unraisable)

    sys.unraisablehook = _desktop2stereo_unraisable_hook
    _asyncio_shutdown_noise_filter_installed = True


def _is_key_console_output(data):
    text = str(data or "").strip()
    if not text:
        return True
    lower = text.lower()
    if any(text.startswith(prefix) for prefix in _NOISY_CONSOLE_PREFIXES):
        return False
    if any(lower.startswith(prefix) for prefix in _DEBUG_CONSOLE_PREFIXES):
        return False
    if lower.startswith("[diag]") and not any(token in lower for token in ("error", "failed", "exception", "exited")):
        return False
    return True


class _FletInfoAsDebugFilter(logging.Filter):
    def filter(self, record):
        if record.levelno == logging.INFO and (
            record.name in _FLET_LOGGER_NAMES
            or record.getMessage().startswith(_FLET_MESSAGE_PREFIXES)
        ):
            record.levelno = logging.DEBUG
            record.levelname = "DEBUG"
        return True


class _HideStructuredProgressFilter(logging.Filter):
    def filter(self, record):
        return _PROGRESS_PREFIX not in record.getMessage()


class _CpuOperationAsCriticalFilter(logging.Filter):
    def filter(self, record):
        text = f"{record.name} {record.getMessage()}".lower()
        if "cpu" in text:
            record.levelno = logging.CRITICAL
            record.levelname = "CRITICAL"
        return True


def _disable_flet_logging():
    for name in _FLET_LOGGER_NAMES:
        logging.getLogger(name).disabled = True


def _setup_file_log_handler():
    """Create (or re-create) the file handler that appends to the log file."""
    global _file_log_handler
    if _file_log_handler is None:
        handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.addFilter(_FletInfoAsDebugFilter())
        handler.addFilter(_CpuOperationAsCriticalFilter())
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", "%H:%M:%S"
        ))
        logging.getLogger().addHandler(handler)
        _file_log_handler = handler
    return _file_log_handler


def _release_file_log_handler():
    """Close and detach the file handler so the log file is free for the OS.

    The next run re-creates it in append mode; only a GUI restart truncates
    the file. Between runs the GUI logs to console and the GUI queue only.
    """
    global _file_log_handler
    handler = _file_log_handler
    _file_log_handler = None
    if handler is not None:
        try:
            logging.getLogger().removeHandler(handler)
            handler.close()
        except Exception:
            pass


def _setup_console_logging():
    """Configure console logging, file logging, and GUI log queue."""
    global _console_logging_installed, _gui_log_handler
    _disable_flet_logging()
    if _console_logging_installed:
        _install_asyncio_shutdown_noise_filter()
        return _gui_log_handler

    os.makedirs(LOG_DIR, exist_ok=True)
    try:
        open(LOG_FILE, "w", encoding="utf-8").close()
    except Exception:
        pass

    try:
        persistent_paths = {
            os.path.abspath(LOG_FILE),
            os.path.abspath(STREAM_CALIBRATION_PROFILE_FILE),
        }
        for name in os.listdir(LOG_DIR):
            path = os.path.join(LOG_DIR, name)
            if os.path.isfile(path) and os.path.abspath(path) not in persistent_paths:
                try:
                    os.remove(path)
                except Exception:
                    pass
    except Exception:
        pass

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    console_stream = sys.__stderr__ or sys.stderr or open(os.devnull, "w", encoding="utf-8")
    console_handler = logging.StreamHandler(console_stream)
    console_handler.setLevel(logging.DEBUG)
    console_handler.addFilter(_FletInfoAsDebugFilter())
    console_handler.addFilter(_CpuOperationAsCriticalFilter())
    console_handler.addFilter(_NoisyThirdPartyDebugFilter())
    console_handler.addFilter(_HideStructuredProgressFilter())
    console_handler.setFormatter(logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", "%H:%M:%S"
    ))

    _setup_file_log_handler()

    gui_handler = GuiLogHandler(maxlen=2000)
    gui_handler.setLevel(logging.DEBUG)
    gui_handler.addFilter(_FletInfoAsDebugFilter())
    gui_handler.addFilter(_CpuOperationAsCriticalFilter())
    gui_handler.addFilter(_NoisyThirdPartyDebugFilter())
    gui_handler.setFormatter(logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", "%H:%M:%S"
    ))

    root.addHandler(console_handler)
    root.addHandler(gui_handler)
    _gui_log_handler = gui_handler

    class _StreamToLogger:
        def __init__(self, stream_logger, level):
            self.stream_logger = stream_logger
            self.level = level
            self.original = (sys.__stdout__ if level < logging.ERROR else sys.__stderr__) or console_stream
            self._buffer = ""

        def write(self, data):
            if not data:
                return 0
            self._buffer += str(data)
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                self._log(line.rstrip("\r"))
            return len(data)

        def flush(self):
            if self._buffer.strip():
                self._log(self._buffer.strip())
            self._buffer = ""

        def isatty(self):
            try:
                return self.original.isatty()
            except Exception:
                return False

        def fileno(self):
            return self.original.fileno()

        def _log(self, line):
            if line and _is_key_console_output(line):
                self.stream_logger.log(self.level, line)

    _install_asyncio_shutdown_noise_filter()
    sys.stdout = _StreamToLogger(logging.getLogger("stdout"), logging.INFO)
    sys.stderr = _StreamToLogger(logging.getLogger("stderr"), logging.ERROR)
    logger.info("Desktop2Stereo log started %s", datetime.datetime.now().isoformat(timespec="seconds"))
    _console_logging_installed = True
    return gui_handler


def _set_console_quick_edit(enabled: bool):
    """Toggle Windows console Quick Edit mode when a real console is attached."""
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.GetStdHandle.restype = ctypes.c_void_p
        kernel32.GetStdHandle.argtypes = [ctypes.c_uint32]
        STD_INPUT_HANDLE = -10
        ENABLE_QUICK_EDIT_MODE = 0x0040
        ENABLE_EXTENDED_FLAGS = 0x0080
        h_stdin = kernel32.GetStdHandle(STD_INPUT_HANDLE)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(h_stdin, ctypes.byref(mode)):
            mode.value |= ENABLE_EXTENDED_FLAGS
            if enabled:
                mode.value |= ENABLE_QUICK_EDIT_MODE
            else:
                mode.value &= ~ENABLE_QUICK_EDIT_MODE
            kernel32.SetConsoleMode(h_stdin, mode)
    except Exception:
        pass


# Disable Quick Edit while the worker is running.
_set_console_quick_edit(False)


class GUIProcessMixin:
    """Mixin providing process lifecycle, ESC monitoring, and status for Desktop2StereoGUI."""

    def set_status(self, msg, key=None):
        self.status_text.value = msg
        if key is not None:
            self._status_key = key
        if msg:
            status_logger.info(msg)
        self._safe_update(self.status_text)

    def _set_backend_status(self, payload):
        """Display read-only runtime backend telemetry in the GUI footer."""
        control = getattr(self, "backend_status_text", None)
        if control is None or not isinstance(payload, dict):
            return
        depth = payload.get("depth_backend") or "unknown"
        stereo = payload.get("stereo_backend") or "unknown"
        fallback = "是" if payload.get("fallback") else "否"
        gpu_to_cpu = "是" if payload.get("gpu_to_cpu") else "否"
        zero_copy = "是" if payload.get("zero_copy") else "否"
        copies = payload.get("gpu_copy_count", 0)
        resource_kind = payload.get("resource_kind") or "unknown"
        resource_format = payload.get("resource_format") or "unknown"
        directml_mode = payload.get("directml_resource_mode")
        reasons = payload.get("fallback_reasons") or []
        reason_text = "; ".join(str(item) for item in reasons if item)
        if len(reason_text) > 220:
            reason_text = reason_text[:217] + "..."
        text = (
            f"深度={depth} | 合成={stereo} | 回退={fallback} | "
            f"CPU回读={gpu_to_cpu} | GPU复制={copies} | 零回读={zero_copy} | "
            f"资源={resource_kind}/{resource_format}"
        )
        if directml_mode:
            text += f" | DirectML资源={directml_mode}"
        if reason_text:
            text += f" | 原因={reason_text}"
        control.value = text
        control.visible = True
        bar = getattr(self, "_backend_status_bar", None)
        if bar is not None:
            bar.visible = True
            self._safe_update(control, bar)
        else:
            self._safe_update(control)

    def _set_running_ui(self, running: bool):
        # Keep Stop enabled and avoid updating it across startup/status refreshes.
        # Re-sending its disabled state while the mouse is held cancels Flet's
        # in-progress click gesture, so a long hold would never reach on_click.
        changed_controls = []
        if self.run_btn.disabled != running:
            self.run_btn.disabled = running
            changed_controls.append(self.run_btn)
        if self.stop_btn.disabled:
            self.stop_btn.disabled = False
            changed_controls.append(self.stop_btn)
        calibration_button = getattr(self, "stream_calibration_btn", None)
        if (
            calibration_button is not None
            and calibration_button.disabled != running
        ):
            calibration_button.disabled = running
            changed_controls.append(calibration_button)
        self._safe_update(*changed_controls)

    def _show_display_refresh_warning(self, payload: dict) -> None:
        if getattr(self, "_display_refresh_warning_dialog", None) is not None:
            current = getattr(self, "_display_refresh_warning_payload", None)
            pending = getattr(self, "_pending_display_refresh_warnings", None)
            if pending is None:
                pending = []
                self._pending_display_refresh_warnings = pending
            if payload != current and payload not in pending:
                pending.append(dict(payload))
            return
        try:
            refresh_hz = int(payload.get("refresh_hz", 0) or 0)
            kind = str(payload.get("kind", "output") or "output")
            if kind == "input_capture_target":
                capture_target = int(payload.get("capture_target", 0) or 0)
                if refresh_hz <= 0 or capture_target <= 0:
                    return
            else:
                sbs_fps = float(payload.get("sbs_fps", 0.0) or 0.0)
                if refresh_hz <= 0 or sbs_fps <= 0.0:
                    return
        except (TypeError, ValueError, AttributeError):
            return
        messages = UI_MESSAGES[self.locale]
        if kind == "input_capture_target":
            title = messages.get(
                "Input display refresh warning",
                "Input display refresh warning",
            )
            body = messages.get(
                "input_refresh_warning_body",
                "The input display is running at {refresh_hz} Hz, below the "
                "dynamic capture target of {capture_target} FPS. Increase the "
                "input display refresh rate or lower the capture target manually.",
            ).format(
                refresh_hz=refresh_hz,
                capture_target=capture_target,
            )
        else:
            title = messages.get(
                "Display refresh warning",
                "Display refresh warning",
            )
            body = messages.get(
                "display_refresh_warning_body",
                "The SBS output display is running at {refresh_hz} Hz, below the "
                "measured {sbs_fps:.1f} FPS or the recommended 60 Hz minimum. "
                "Increase the display refresh rate in Windows or the GPU control panel.",
            ).format(refresh_hz=refresh_hz, sbs_fps=sbs_fps)
        continuing = messages.get(
            "display_refresh_warning_continuing",
            "Desktop2Stereo will continue running; this warning does not stop output.",
        )
        self._display_refresh_warning_payload = dict(payload)
        self._display_refresh_warning_dialog = ft.AlertDialog(
            modal=True,
            title=ft.Row(
                [
                    ft.Icon(ft.Icons.WARNING_AMBER_ROUNDED, color=ft.Colors.ORANGE),
                    ft.Text(title),
                ],
                spacing=10,
            ),
            content=ft.Column(
                [
                    ft.Text(body, size=16, weight=ft.FontWeight.BOLD),
                    ft.Text(continuing, color=ft.Colors.GREEN),
                ],
                width=520,
                height=170,
                spacing=16,
            ),
            actions=[
                ft.Button(
                    content=ft.Text(messages.get("Close", "Close")),
                    on_click=self._close_display_refresh_warning,
                )
            ],
        )
        self.page.show_dialog(self._display_refresh_warning_dialog)

    def _close_display_refresh_warning(self, _event=None) -> None:
        if getattr(self, "_display_refresh_warning_dialog", None) is None:
            return
        try:
            self.page.pop_dialog()
        except Exception:
            pass
        self._display_refresh_warning_dialog = None
        self._display_refresh_warning_payload = None
        pending = getattr(self, "_pending_display_refresh_warnings", None) or []
        if pending:
            self._show_display_refresh_warning(pending.pop(0))

    def _stream_calibration_auto_enabled(self) -> bool:
        value = str(getattr(self.stream_calibration_mode_dd, "value", "") or "")
        return value.casefold().startswith("auto") or value.startswith("自动")

    def _stream_calibration_profile_status(self) -> str:
        """Return current, missing, or stale for the saved calibration profile."""
        if not self._stream_calibration_auto_enabled():
            return "current"
        try:
            with open(STREAM_CALIBRATION_PROFILE_FILE, "r", encoding="utf-8") as file:
                profile = json.load(file)
            if not calibration_fingerprint_matches(profile.get("fingerprint"), self._config):
                return "stale"
            if profile.get("stability", "stable") != "stable":
                return "missing"
            valid = all(
                int(profile.get(key, 0) or 0) > 0
                for key in ("fps", "target_mbps", "peak_mbps")
            )
            return "current" if valid else "missing"
        except (OSError, ValueError, TypeError, AttributeError):
            return "missing"

    def _stream_calibration_profile_is_current(self) -> bool:
        return self._stream_calibration_profile_status() == "current"

    def _persist_current_stream_calibration_profile(self) -> None:
        """Restore a completed profile if shutdown raced GUI result handling."""
        if getattr(self, "stream_calibration_mode_dd", None) is None:
            return
        if not self._stream_calibration_auto_enabled():
            return
        try:
            with open(STREAM_CALIBRATION_PROFILE_FILE, "r", encoding="utf-8") as file:
                profile = json.load(file)
            if profile.get("stability", "stable") != "stable":
                return
            if not calibration_fingerprint_matches(profile.get("fingerprint"), self._config):
                return
            fps = int(profile["fps"])
            target = int(profile["target_mbps"])
            peak = int(profile["peak_mbps"])
            crf = recommended_crf_for_bitrate(target)
            if min(fps, target, peak) <= 0:
                return
        except (OSError, ValueError, TypeError, KeyError):
            return

        changed = (
            int(self._config.get("Stream Target FPS", 0) or 0) != fps
            or int(self._config.get("Stream Target Bitrate Mbps", 0) or 0) != target
            or int(self._config.get("Stream Peak Bitrate Mbps", 0) or 0) != peak
            or int(self._config.get("CRF", 0) or 0) != crf
            or not bool(self._config.get("Use Stream Calibration", False))
        )
        if not changed:
            return
        self._config.update({
            "Stream Target FPS": fps,
            "Use Stream Calibration": True,
            "Stream Target Bitrate Mbps": target,
            "Stream Peak Bitrate Mbps": peak,
            "CRF": crf,
        })
        target_control = getattr(self, "target_fps_dd", None)
        if (
            target_control is not None
            and target_fps_setting_key(getattr(self, "run_mode_key", "Local Viewer"))
            == "Stream Target FPS"
        ):
            target_control.value = self._target_fps_to_display(fps)
        crf_control = getattr(self, "crf_tf", None)
        if crf_control is not None:
            crf_control.value = str(crf)
        ok, error = save_yaml(os.path.join(BASE_DIR, "settings.yaml"), self._config)
        if not ok:
            logger.warning("[StreamCalibration] Failed to restore saved profile: %s", error)

    def _refresh_stream_calibration_status(self) -> None:
        control = getattr(self, "stream_calibration_status", None)
        if control is None:
            return
        # Controls such as the selected monitor resolution tier are derived
        # during GUI startup and are not guaranteed to be present in the
        # YAML loaded before the widgets are applied.  Refresh the effective
        # config before comparing a persisted profile, otherwise a valid
        # profile is incorrectly reported as needing recalibration after a
        # restart.
        collect_config = getattr(self, "_collect_config", None)
        if callable(collect_config):
            collect_config()
        self._persist_current_stream_calibration_profile()
        warning = getattr(self, "stream_calibration_warning", None)
        warning_row = getattr(self, "stream_calibration_warning_row", None)
        result = getattr(self, "stream_calibration_result", None)
        result_row = getattr(self, "stream_calibration_result_row", None)
        recalibrate_hint = getattr(self, "stream_calibration_recalibrate_hint", None)
        recalibrate_hint_row = getattr(self, "stream_calibration_recalibrate_hint_row", None)

        def clear_warning():
            if warning is not None:
                warning.value = ""
                warning.visible = False
            if warning_row is not None:
                warning_row.visible = False

        def clear_result():
            if result is not None:
                result.value = ""
                result.visible = False
            if result_row is not None:
                result_row.visible = False
            if recalibrate_hint is not None:
                recalibrate_hint.value = ""
                recalibrate_hint.visible = False
            if recalibrate_hint_row is not None:
                recalibrate_hint_row.visible = False

        def show_result(
            fps,
            target,
            peak,
            measured=None,
            network_max=None,
            stable=True,
        ):
            network_bitrate = (
                int(round(float(network_max or 0.0)))
                or int(round(float(measured or 0.0)))
                or target
            )
            if result is not None:
                result.value = UI_MESSAGES[self.locale].get(
                    "calibration_result_stable" if stable else "calibration_result_limited",
                    "Stable network limit: {network_max} Mbps, safe bitrate: "
                    "{safe_target} Mbps, {fps} FPS.",
                ).format(
                    fps=fps,
                    network_max=network_bitrate,
                    safe_target=target,
                )
                result.color = ft.Colors.GREEN if stable else ft.Colors.ORANGE
                result.visible = True
            if result_row is not None:
                result_row.visible = True
            if recalibrate_hint is not None:
                recalibrate_hint.value = UI_MESSAGES[self.locale].get(
                    "calibration_recalibrate_hint",
                    "Recalibrate after changing the router, Wi-Fi band or headset position; headset/browser/system upgrades; output resolution, codec or quality; or the PC GPU, driver or performance mode.",
                )
                recalibrate_hint.visible = True
            if recalibrate_hint_row is not None:
                recalibrate_hint_row.visible = True

        def show_warning(message):
            if warning is not None:
                warning.value = message
                warning.visible = True
            if warning_row is not None:
                warning_row.visible = True

        def refit_visible_rows():
            fit_window = getattr(self, "_fit_window_to_content", None)
            if callable(fit_window):
                fit_window(update=False, resize_window=True)

        try:
            with open(STREAM_CALIBRATION_PROFILE_FILE, "r", encoding="utf-8") as file:
                profile = json.load(file)
            clear_result()
            if profile.get("stability", "stable") != "stable":
                control.value = UI_MESSAGES[self.locale].get("Not calibrated", "Not calibrated")
                control.color = ft.Colors.ORANGE
                clear_warning()
                show_result(
                    int(profile.get("fps", 0)),
                    int(profile.get("target_mbps", 0)),
                    int(profile.get("peak_mbps", 0)),
                    profile.get("measured_bitrate_mbps"),
                    profile.get("network_max_mbps"),
                    stable=False,
                )
                self._safe_update(control)
                self._safe_update(
                    warning, warning_row, result, result_row,
                    recalibrate_hint, recalibrate_hint_row,
                )
                refit_visible_rows()
                return
            clear_warning()
            show_result(
                int(profile.get("fps", 0)),
                int(profile.get("target_mbps", 0)),
                int(profile.get("peak_mbps", 0)),
                profile.get("measured_bitrate_mbps"),
                profile.get("network_max_mbps"),
                stable=True,
            )
            control.value = UI_MESSAGES[self.locale].get(
                "calibration_profile_summary", "{fps} FPS · {target} Mbps"
            ).format(
                fps=int(profile.get("fps", 0)),
                target=int(profile.get("target_mbps", 0)),
            )
            control.color = ft.Colors.GREEN
        except (OSError, ValueError, TypeError):
            saved = getattr(self, "_config", {}) or {}
            saved_fps = int(saved.get("Stream Target FPS", 0) or 0)
            saved_target = int(saved.get("Stream Target Bitrate Mbps", 0) or 0)
            if (
                bool(saved.get("Use Stream Calibration", False))
                and saved_fps > 0
                and saved_target > 0
            ):
                control.value = UI_MESSAGES[self.locale].get(
                    "calibration_profile_summary", "{fps} FPS · {target} Mbps"
                ).format(fps=saved_fps, target=saved_target)
                control.color = ft.Colors.GREEN
                clear_warning()
                show_result(
                    saved_fps,
                    saved_target,
                    int(saved.get("Stream Peak Bitrate Mbps", 0) or 0),
                    stable=True,
                )
                self._safe_update(control)
                self._safe_update(
                    result, result_row,
                    recalibrate_hint, recalibrate_hint_row,
                )
                refit_visible_rows()
                return
            clear_warning()
            clear_result()
            control.value = UI_MESSAGES[self.locale].get(
                "Not calibrated", "Not calibrated"
            )
            control.color = ft.Colors.GREY
        self._safe_update(control)
        self._safe_update(
            warning, warning_row, result, result_row,
            recalibrate_hint, recalibrate_hint_row,
        )
        refit_visible_rows()

    def start_stream_calibration(self, _event=None) -> None:
        if self._starting or (self.process and self.process.returncode is None):
            self.set_status(UI_MESSAGES[self.locale]["A thread already running!"])
            return
        if not supports_network_calibration(self.run_mode_key, self.stream_proto_dd.value):
            self.set_status(UI_MESSAGES[self.locale].get(
                "calibration_requires_advanced",
                "Automatic calibration requires Advanced Streaming with WebRTC.",
            ))
            return
        if int(self.stream_port_tf.value or DEFAULT_PORT) >= 65535:
            self.set_status(UI_MESSAGES[self.locale].get(
                "calibration_port_unavailable",
                "Automatic calibration needs the port immediately after the WebRTC port.",
            ))
            return
        ok, error = self._validate_config_before_run()
        if not ok:
            self.set_status(error)
            return
        try:
            os.remove(STREAM_CALIBRATION_STATE_FILE)
        except FileNotFoundError:
            pass
        self._calibration_previous_target_value = self.target_fps_dd.value
        self.target_fps_dd.value = self._target_fps_to_display(0)
        self._calibration_active = True
        self._calibration_run_requested = True
        self._show_stream_calibration_dialog()
        self.save_and_run(None)
        previous = getattr(self, "_calibration_poll_task", None)
        if previous is not None and not previous.done():
            previous.cancel()
        self._calibration_poll_task = asyncio.create_task(
            self._poll_stream_calibration()
        )

    def _show_stream_calibration_dialog(self) -> None:
        port = min(65535, int(self.stream_port_tf.value or DEFAULT_PORT) + 1)
        local_ip = getattr(self, "_local_ip_cache", "127.0.0.1")
        self._calibration_dialog_url = ft.Text(
            f"http://{local_ip}:{port}/", selectable=True, color=ft.Colors.BLUE
        )
        self._calibration_dialog_stage = ft.Text(UI_MESSAGES[self.locale].get(
            "calibration_waiting", "Waiting for the headset to open the test page..."
        ))
        self._calibration_dialog_detail = ft.Text("", selectable=True)
        self._calibration_dialog_progress = ft.ProgressBar(value=0.0)
        self._calibration_dialog_firewall_hint = ft.Text(
            UI_MESSAGES[self.locale].get(
                "calibration_firewall_manual_hint",
                "If the headset does not respond when opening port {port}, click Detect Firewall Rules.",
            ).format(port=port),
            color=ft.Colors.ORANGE,
        )
        self._calibration_firewall_btn = ft.Button(
            content=ft.Text(UI_MESSAGES[self.locale].get(
                "Detect Firewall Rules", "Detect Firewall Rules"
            )),
            on_click=self.check_stream_calibration_firewall,
        )
        self._calibration_dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(UI_MESSAGES[self.locale].get(
                "Automatic Network and Performance Calibration",
                "Automatic Network and Performance Calibration",
            )),
            content=ft.Column(
                [
                    ft.Text(UI_MESSAGES[self.locale].get(
                        "calibration_open_headset_url",
                        "Open this address in the headset browser and keep the page visible:",
                    )),
                    self._calibration_dialog_url,
                    self._calibration_dialog_stage,
                    self._calibration_dialog_progress,
                    self._calibration_dialog_detail,
                    self._calibration_dialog_firewall_hint,
                ],
                width=520,
                height=250,
                spacing=12,
            ),
            actions=[
                self._calibration_firewall_btn,
                ft.Button(
                    content=ft.Text(UI_MESSAGES[self.locale].get(
                        "Cancel Calibration", "Cancel Calibration"
                    )),
                    on_click=self._cancel_stream_calibration,
                ),
            ],
        )
        self.page.show_dialog(self._calibration_dialog)

    def check_stream_calibration_firewall(self, _event=None) -> None:
        task = getattr(self, "_calibration_firewall_task", None)
        if task is not None and not task.done():
            return
        self._calibration_firewall_task = asyncio.create_task(
            self._check_stream_calibration_firewall_async()
        )

    async def _check_stream_calibration_firewall_async(self) -> None:
        button = getattr(self, "_calibration_firewall_btn", None)
        hint = getattr(self, "_calibration_dialog_firewall_hint", None)
        checking = UI_MESSAGES[self.locale].get(
            "calibration_firewall_checking",
            "Checking Windows Firewall rules...",
        )
        if button is not None:
            button.disabled = True
        if hint is not None:
            hint.value = checking
        self.set_status(checking)
        self._safe_update(button, hint)
        try:
            firewall_blocks = await asyncio.to_thread(
                _detect_windows_firewall_blocks
            )
            if not firewall_blocks:
                message = UI_MESSAGES[self.locale].get(
                    "calibration_firewall_no_blocks",
                    "No inbound block rules were found for the bundled Python.",
                )
            else:
                protocols = sorted({
                    str(item.get("Protocol", "")).upper()
                    for item in firewall_blocks
                    if item.get("Protocol")
                })
                rule_names = sorted({
                    str(item.get("DisplayName", "")).strip()
                    for item in firewall_blocks
                    if item.get("DisplayName")
                })
                protocol_text = "/".join(protocols) or "TCP/UDP"
                logger.warning(
                    "[StreamCalibration] Windows Firewall inbound block detected: protocols=%s rules=%s",
                    protocol_text,
                    ", ".join(rule_names) or "(unnamed)",
                )
                removed, remove_error = await asyncio.to_thread(
                    _remove_windows_firewall_blocks
                )
                if not removed:
                    message = UI_MESSAGES[self.locale].get(
                        "calibration_firewall_remove_failed",
                        "Windows Firewall is blocking the bundled Python inbound connection ({protocol}), and the rule could not be removed. Run Desktop2Stereo as administrator and try again.",
                    ).format(protocol=protocol_text)
                    logger.error(
                        "[StreamCalibration] Failed to remove Windows Firewall block: %s",
                        remove_error,
                    )
                else:
                    message = UI_MESSAGES[self.locale].get(
                        "calibration_firewall_removed",
                        "Matching Windows Firewall block rules were removed. Open the headset page again.",
                    )
                    logger.info(
                        "[StreamCalibration] Removed Windows Firewall inbound block rules for %s",
                        sys.executable,
                    )
        except FirewallProbeError as exc:
            message = UI_MESSAGES[self.locale].get(
                "calibration_firewall_probe_failed",
                "Windows Firewall detection failed: {error}",
            ).format(error=exc)
            logger.error("[StreamCalibration] %s", message)
        finally:
            if button is not None:
                button.disabled = False
        if hint is not None:
            hint.value = message
        self.set_status(message)
        self._safe_update(button, hint)

    def _cancel_stream_calibration(self, _event=None) -> None:
        self._calibration_run_requested = False
        self._restore_precalibration_target()
        self.stop_process()
        self._close_stream_calibration_dialog()

    def _close_stream_calibration_dialog(self, _event=None) -> None:
        if self._calibration_dialog is not None:
            try:
                self.page.pop_dialog()
            except Exception:
                pass
            self._calibration_dialog = None

    def _restore_precalibration_target(self) -> None:
        previous = self._calibration_previous_target_value
        self._calibration_previous_target_value = None
        self._calibration_active = False
        if previous is None:
            return
        self.target_fps_dd.value = previous
        self._collect_config()
        save_yaml(os.path.join(BASE_DIR, "settings.yaml"), self._config)

    async def _poll_stream_calibration(self) -> None:
        try:
            while not getattr(self, "_closed", False):
                await asyncio.sleep(0.5)
                try:
                    with open(STREAM_CALIBRATION_STATE_FILE, "r", encoding="utf-8") as file:
                        state = json.load(file)
                except (OSError, ValueError):
                    if self.process is None and not self._starting:
                        return
                    continue
                status = str(state.get("status", "waiting_receiver"))
                tier = state.get("tier") or {}
                progress = float(state.get("stage_progress", 0.0) or 0.0)
                overall = (
                    float(state.get("tier_index", 0)) + progress
                ) / max(1.0, float(state.get("tier_count", 1)))
                if getattr(self, "_calibration_dialog_progress", None) is not None:
                    self._calibration_dialog_progress.value = min(1.0, overall)
                    stage_key = {
                        "waiting_receiver": "calibration_waiting",
                        "settling": "calibration_settling",
                        "testing": "calibration_testing",
                        "reconnecting": "calibration_reconnecting",
                        "complete": "calibration_complete",
                    }.get(status, "calibration_waiting")
                    self._calibration_dialog_stage.value = UI_MESSAGES[self.locale].get(
                        stage_key, status
                    ).format(fps=int(tier.get("fps", 0) or 0))
                    sender = state.get("sender") or {}
                    sender_detail = (
                        f"{int(tier.get('fps', 0) or 0)} FPS · "
                        f"{int(tier.get('target_mbps', 0) or 0)} Mbps · "
                        f"send {float(sender.get('submitted_fps', 0.0) or 0.0):.1f} FPS · "
                        f"samples {int(state.get('receiver_samples', 0) or 0)}"
                    )
                    receiver = state.get("receiver_latest") or {}
                    receiver_detail = UI_MESSAGES[self.locale].get(
                        "calibration_receiver_metrics",
                        "decode {decoded} FPS · bitrate {bitrate} Mbps · dropped {dropped} · freeze {freeze} · lost {lost} · jitter {jitter} ms",
                    ).format(
                        decoded=float(receiver.get("decoded_fps", 0.0) or 0.0),
                        bitrate=float(receiver.get("bitrate_mbps", 0.0) or 0.0),
                        dropped=int(receiver.get("dropped_frames", 0) or 0),
                        freeze=int(receiver.get("freeze_count", 0) or 0),
                        lost=int(receiver.get("packets_lost", 0) or 0),
                        jitter=float(receiver.get("jitter_buffer_ms", 0.0) or 0.0),
                    )
                    self._calibration_dialog_detail.value = f"{sender_detail}\n{receiver_detail}"
                    self._safe_update(
                        self._calibration_dialog_progress,
                        self._calibration_dialog_stage,
                        self._calibration_dialog_detail,
                    )
                if status == "complete":
                    await self._apply_stream_calibration_profile()
                    return
        except asyncio.CancelledError:
            return

    async def _apply_stream_calibration_profile(self) -> None:
        try:
            with open(STREAM_CALIBRATION_PROFILE_FILE, "r", encoding="utf-8") as file:
                profile = json.load(file)
            fps = int(profile["fps"])
            target = int(profile["target_mbps"])
            peak = int(profile["peak_mbps"])
            crf = recommended_crf_for_bitrate(target)
            network_max = int(
                profile.get("network_max_mbps", 0)
                or round(float(profile.get("measured_bitrate_mbps", 0.0) or 0.0))
                or target
            )
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self.set_status(f"Calibration result error: {exc}")
            return
        if profile.get("stability", "stable") != "stable":
            await self._async_stop()
            self._restore_precalibration_target()
            message = UI_MESSAGES[self.locale].get(
                "calibration_bandwidth_insufficient",
                "Bandwidth is insufficient for the current resolution; lower the resolution and recalibrate.",
            )
            self._refresh_stream_calibration_status()
            self.set_status(message)
            if self._calibration_dialog is not None:
                self._calibration_dialog.actions = [ft.Button(
                    content=ft.Text(UI_MESSAGES[self.locale].get("Close", "Close")),
                    on_click=self._close_stream_calibration_dialog,
                )]
                self._calibration_dialog_progress.value = 1.0
                self._calibration_dialog_detail.value = message
                self._safe_update(self._calibration_dialog)
            return
        # Calibration is a temporary run. Stop MediaMTX/FFmpeg/the calibration
        # HTTP server before applying and presenting the final profile.
        await self._async_stop()
        self.target_fps_dd.value = str(fps)
        self._calibration_previous_target_value = None
        self._calibration_active = False
        self.stream_calibration_mode_dd.value = UI_MESSAGES[self.locale].get(
            "Auto Calibration", "Auto Calibration"
        )
        self._config["Stream Target FPS"] = fps
        self._config["Use Stream Calibration"] = True
        self._config["Stream Target Bitrate Mbps"] = target
        self._config["Stream Peak Bitrate Mbps"] = peak
        self._config["CRF"] = crf
        crf_control = getattr(self, "crf_tf", None)
        if crf_control is not None:
            crf_control.value = str(crf)
        self._collect_config()
        ok, error = save_yaml(os.path.join(BASE_DIR, "settings.yaml"), self._config)
        if not ok:
            self.set_status(UI_MESSAGES[self.locale]["failed_save_yaml"].format(error))
            return
        self._refresh_stream_calibration_status()
        self.set_status(UI_MESSAGES[self.locale].get(
            "calibration_applied", "Calibration applied: {fps} FPS, {target} Mbps"
        ).format(fps=fps, target=target))
        if self._calibration_dialog is not None:
            self._calibration_dialog.actions = [ft.Button(
                content=ft.Text(UI_MESSAGES[self.locale].get("Close", "Close")),
                on_click=self._close_stream_calibration_dialog,
            )]
            self._calibration_dialog_progress.value = 1.0
            self._calibration_dialog_detail.value = UI_MESSAGES[self.locale].get(
                "calibration_result",
                "Stable network limit: {network_max} Mbps · safe bitrate: "
                "{target} Mbps · peak {peak} Mbps · {fps} FPS",
            ).format(
                fps=fps,
                target=target,
                peak=peak,
                network_max=network_max,
            )
            self._safe_update(self._calibration_dialog)

    def _diag(self, msg, error=False):
        os.makedirs(LOG_DIR, exist_ok=True)
        level_name = "ERROR" if error else "INFO"
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        lines = str(msg).splitlines() or [""]
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                for line in lines:
                    f.write(f"[{timestamp}] [{level_name}] [diag] {line}\n")
        except Exception:
            pass
        if error:
            try:
                original = getattr(sys.stdout, "original", sys.stdout)
                for line in lines:
                    original.write(f"[{timestamp}] [{level_name}] [diag] {line}\n")
                original.flush()
            except Exception:
                pass

    # ── save & run ──

    def _validate_config_before_run(self):
        if self.run_mode_key == "Local Viewer" and self._get_monitor_count() <= 1:
            return False, UI_MESSAGES[self.locale]["Local Viewer requires a second display"]
        if self.capture_mode_key == "Monitor" and (
            getattr(self, "_missing_monitor_identity", False)
            or not self.monitor_dd.value
        ):
            return False, UI_MESSAGES[self.locale]["Selected input display is unavailable"]
        if self.run_mode_key in {"Local Viewer", "3D Monitor"} and (
            getattr(self, "_missing_stereo_output_identity", False)
            or not self.stereo_monitor_dd.value
        ):
            # 3D Monitor with a single display has no second output monitor;
            # cursor passthrough keeps the system cursor visible over the
            # fullscreen SBS window covering the captured display.
            if self.run_mode_key == "3D Monitor" and self._get_monitor_count() <= 1:
                pass
            else:
                return False, UI_MESSAGES[self.locale][
                    "Selected stereo output display is unavailable"
                ]
        try:
            port_val = int(self.stream_port_tf.value) if self.stream_port_tf.value else DEFAULT_PORT
            if not (1 <= port_val <= 65535):
                return False, UI_MESSAGES[self.locale]["Invalid port number (1-65535)"]
        except ValueError:
            return False, UI_MESSAGES[self.locale]["Invalid port number (1-65535)"]
        try:
            crf_val = int(self.crf_tf.value) if self.crf_tf.value else DEFAULTS["CRF"]
            if not (0 <= crf_val <= 51):
                return False, UI_MESSAGES[self.locale]["err_crf"]
        except ValueError:
            return False, UI_MESSAGES[self.locale]["err_crf"]
        try:
            delay_val = float(self.audio_delay_tf.value) if self.audio_delay_tf.value else DEFAULTS["Audio Delay"]
            if not (-10 <= delay_val <= 10):
                return False, UI_MESSAGES[self.locale]["err_audio_delay"]
        except ValueError:
            return False, UI_MESSAGES[self.locale]["err_audio_delay"]
        sk = self.stream_key_tf.value or "live"
        if not re.match(r'^[A-Za-z0-9_-]+$', sk) or len(sk) > 64:
            return False, UI_MESSAGES[self.locale]["err_stream_key"]
        if self.capture_mode_key == "Window":
            if not self.selected_window_name:
                return False, UI_MESSAGES[self.locale]["Please select a window before running in Window capture mode"]
            windows = list_windows()
            exists = any(
                (w.get("handle") is not None and w["handle"] == self.selected_window_handle)
                or (w.get("handle") is None and w["title"] == self.selected_window_name)
                for w in windows)
            if not exists:
                return False, UI_MESSAGES[self.locale]["The selected window no longer exists. Please refresh and select a valid window."]
        return True, ""

    def save_and_run(self, e):
        if self._starting or (self.process and self.process.returncode is None):
            self.set_status(UI_MESSAGES[self.locale]["A thread already running!"])
            self.page.update()
            return
        if (
            supports_network_calibration(self.run_mode_key, self.stream_proto_dd.value)
            and self._stream_calibration_auto_enabled()
            and not self._calibration_run_requested
        ):
            # Collect the current controls before comparing the fingerprint;
            # unsaved GUI changes must invalidate the previous profile too.
            self._collect_config()
            profile_status = self._stream_calibration_profile_status()
            if profile_status != "current":
                logger.info(
                    "[StreamCalibration] Automatic calibration required before run: profile %s",
                    profile_status,
                )
                self.start_stream_calibration()
                return
        ok, err = self._validate_config_before_run()
        if not ok:
            self.set_status(err)
            return
        self._starting = True
        self._cancel_starting = False
        self._esc_stopped = False
        self._stopping = False
        # Re-attach the file handler (append mode) for this run's log output;
        # it was released after the previous run so the file stayed free.
        _setup_file_log_handler()
        _set_console_quick_edit(False)
        self._set_log_panel_visible(self._config.get("Show Log Panel", DEFAULTS["Show Log Panel"]))
        self._set_running_ui(True)
        self._collect_config()
        if (
            self.run_mode_key == "RTMP Streamer"
            and self._config.get("Display Mode") == "Full-SBS"
        ):
            status_logger.warning(
                UI_MESSAGES[self.locale]["full_sbs_stream_advisory"]
            )
        ok, err = save_yaml(os.path.join(BASE_DIR, "settings.yaml"), self._config)
        if not ok:
            self.set_status(UI_MESSAGES[self.locale]["failed_save_yaml"].format(err))
            self._starting = False
            self._set_running_ui(False)
            return
        self.set_status(UI_MESSAGES[self.locale]["Countdown"], key="Countdown")
        self.page.update()
        asyncio.create_task(self._countdown_and_run(0.5))

    async def _countdown_and_run(self, seconds):
        self._diag("_countdown_and_run scheduled")
        try:
            if self.process and self.process.returncode is None:
                self.set_status(UI_MESSAGES[self.locale]["A thread already running!"])
                self._diag("already running, return")
                return
            if seconds > 0:
                await asyncio.sleep(seconds)
            if self._cancel_starting:
                self._cancel_starting = False
                self._diag("cancelled, return")
                return
            status_logger.info(UI_MESSAGES[self.locale].get("Starting Desktop2Stereo...", "Starting Desktop2Stereo...").format(self.run_mode_key))
            shutdown_event.clear()
            try:
                if os.path.exists(STOP_REQUEST_FILE):
                    os.remove(STOP_REQUEST_FILE)
            except Exception:
                pass
            child_args = [
                sys.executable,
                "-u",
                "-X",
                "faulthandler",
                os.path.join(BASE_DIR, "main.py"),
                "--runtime",
            ]
            child_env = os.environ.copy()
            child_env["DESKTOP2STEREO_LOCALE"] = self.locale
            child_env["PYTHONIOENCODING"] = "utf-8"
            if OS_NAME == "Darwin":
                from utils.vulkan_env import apply_macos_vulkan_env

                apply_macos_vulkan_env(child_env)
            child_env["D2S_STOP_REQUEST_FILE"] = STOP_REQUEST_FILE
            calibration_requested = self._calibration_run_requested
            self._calibration_run_requested = False
            if calibration_requested:
                child_env["D2S_STREAM_CALIBRATE"] = "1"
            if self.run_mode_key == "OpenXR Link":
                child_env.setdefault("D2S_FPS_BREAKDOWN", "1")
                child_env.setdefault("D2S_OPENXR_DEBUG", "1")
                child_env.setdefault("D2S_OPENXR_ASYNC_EFFECTS", "0")
            if OS_NAME == "Windows":
                self.process = await asyncio.create_subprocess_exec(
                    *child_args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                    env=child_env,
                )
            else:
                self.process = await asyncio.create_subprocess_exec(
                    *child_args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    start_new_session=True,
                    env=child_env,
                )
            self._diag(f"process started, pid={self.process.pid}, log={LOG_FILE}")
            asyncio.create_task(self._pump_child_output(self.process))
            self.set_status(UI_MESSAGES[self.locale]["Running"], key="Running")
            self.page.update()
            asyncio.create_task(self._monitor_process_task())
            self._diag("monitor_task created")
            for _ in range(8):
                await asyncio.sleep(1)
                if self.process and self.process.returncode is not None:
                    self._diag(f"process exited during wait, code={self.process.returncode}")
                    break
            self._config["Recompile TensorRT"] = False
            self._config["Recompile MIGraphX"] = False
            self._config["Recompile CoreML"] = False
            self._config["Recompile OpenVINO"] = False
            save_yaml(os.path.join(BASE_DIR, "settings.yaml"), self._config)
        except Exception as e:
            self._diag(f"_countdown_and_run failed:\n{traceback.format_exc()}", error=True)
            if self._calibration_active:
                self._restore_precalibration_target()
                self._close_stream_calibration_dialog()
            self.set_status(UI_MESSAGES[self.locale]["err_start_failed"].format(e))
            self.page.update()
        finally:
            self._starting = False

    async def _pump_child_output(self, proc):
        try:
            stream = proc.stdout
            if stream is None:
                return
            pending = ""
            while True:
                raw = await stream.read(4096)
                if not raw:
                    break
                try:
                    pending += raw.decode("utf-8", errors="replace")
                except Exception:
                    pending += repr(raw)
                lines = pending.splitlines(keepends=True)
                if lines and not lines[-1].endswith(("\n", "\r")):
                    pending = lines[-1]
                    lines = lines[:-1]
                else:
                    pending = ""
                for line in lines:
                    self._log_child_line(line.rstrip("\r\n"))
            if pending.strip():
                self._log_child_line(pending.strip())
        except Exception as e:
            logger.exception("_pump_child_output exception: %s", e)
            self._diag(f"_pump_child_output exception: {e}\n{traceback.format_exc()}", error=True)

    def _log_child_line(self, line):
        text = str(line or "").strip()
        if not text:
            return
        if _VULKAN_DESCRIPTOR_DETAIL_RE.match(text):
            if not getattr(self, "_vulkan_descriptor_summary_logged", False):
                self._vulkan_descriptor_summary_logged = True
                child_logger.info(
                    "[VulkanValidation] Descriptor bindings were updated between "
                    "render-pass begin/end; repeated binding details suppressed."
                )
            return
        if _FILAMENT_FRAME_DETAIL_RE.match(text):
            if not getattr(self, "_filament_frame_summary_logged", False):
                self._filament_frame_summary_logged = True
                child_logger.info(
                    "[FilamentBridge] Eye acquire/render/end frame diagnostics "
                    "are active; repeated per-frame details suppressed."
                )
            return
        # stdout and stderr are merged by the child process pipe. A producer
        # can write the next tagged record before the previous record reaches
        # this reader, so split known records before level classification.
        fps_marker = text.find("[FPSBreakdown]", 1)
        if fps_marker > 0:
            self._log_child_line(text[:fps_marker])
            self._log_child_line(text[fps_marker:])
            return
        lower = text.lower()
        mediamtx_level = _MEDIAMTX_LEVEL_RE.match(text)
        if text.startswith(_DISPLAY_REFRESH_WARNING_PREFIX):
            payload_text = text[len(_DISPLAY_REFRESH_WARNING_PREFIX):].strip()
            try:
                self._show_display_refresh_warning(json.loads(payload_text))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                child_logger.warning("Invalid display refresh warning payload: %s", exc)
        elif text.startswith(_BACKEND_STATUS_PREFIX):
            payload_text = text[len(_BACKEND_STATUS_PREFIX):].strip()
            try:
                self._set_backend_status(json.loads(payload_text))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                child_logger.warning("Invalid backend status payload: %s", exc)
        elif text.startswith(_STATUS_PREFIX):
            status_message = text[len(_STATUS_PREFIX):].strip()
            # set_status() owns both GUI mutation and status logging. Logging
            # here as well produced duplicate consecutive status records.
            self.set_status(status_message)
        elif text.startswith("[FPSBreakdown]"):
            child_logger.debug(text)
        elif mediamtx_level is not None:
            level = mediamtx_level.group("level")
            if level == "ERR":
                child_logger.error(text)
            elif level == "WAR":
                child_logger.warning(text)
            elif level == "DBG":
                child_logger.debug(text)
            else:
                child_logger.info(text)
        elif any(token in lower for token in ("traceback", "exception", "error", "failed", "exited with code")):
            child_logger.error(text)
        elif any(token in lower for token in ("warning", "warn")):
            child_logger.warning(text)
        else:
            child_logger.info(text)

    async def _monitor_process_task(self):
        proc = self.process
        if not proc:
            self._diag("monitor_task: proc is None, return")
            return
        self._diag(f"monitor_task started, pid={proc.pid}")
        try:
            await proc.wait()
            self._diag(f"proc.wait returned, rc={proc.returncode}")
        except Exception as e:
            self._diag(f"proc.wait() exception: {e}", error=True)
        finally:
            self._diag(f"finally: process is proc={self.process is proc}, returncode={proc.returncode}")
            if self.process is proc:
                self.process = None
            self._starting = False
            if self._calibration_active:
                try:
                    with open(STREAM_CALIBRATION_STATE_FILE, "r", encoding="utf-8") as file:
                        calibration_complete = json.load(file).get("status") == "complete"
                except (OSError, ValueError):
                    calibration_complete = False
                if not calibration_complete:
                    self._restore_precalibration_target()
                    self._close_stream_calibration_dialog()
            code = proc.returncode if proc else None
            if code and code != 0:
                self._diag(f"child exited rc={code}; see {LOG_FILE} for details", error=True)
                self.set_status(UI_MESSAGES[self.locale]["exited_with_code"].format(code))
            else:
                self.set_status(UI_MESSAGES[self.locale]["Stopped"], key="Stopped")
            _set_console_quick_edit(True)
            self._set_running_ui(False)
            self._diag("monitor_task done, status updated")

    # ── stop ──

    def stop_process(self, e=None):
        process = getattr(self, "process", None)
        running = bool(getattr(self, "_starting", False)) or (
            process is not None and getattr(process, "returncode", None) is None
        )
        if not running:
            return
        future = asyncio.run_coroutine_threadsafe(self._async_stop(), self._loop)
        future.add_done_callback(lambda f: f.exception() if f.exception() else None)

    async def _on_page_close(self, e=None):
        self._closed = True
        if hasattr(self, '_esc_task') and self._esc_task and not self._esc_task.done():
            self._esc_task.cancel()
        if hasattr(self, '_log_poll_task') and self._log_poll_task and not self._log_poll_task.done():
            self._log_poll_task.cancel()
        # The browser can reach the completed state just before the user
        # closes the GUI.  The polling task may not have applied the profile
        # to settings.yaml yet, so finish that persistence step synchronously
        # during shutdown instead of losing the selected FPS/bitrates.
        if self._calibration_active:
            try:
                with open(STREAM_CALIBRATION_STATE_FILE, "r", encoding="utf-8") as file:
                    calibration_complete = json.load(file).get("status") == "complete"
            except (OSError, ValueError, TypeError):
                calibration_complete = False
            if calibration_complete:
                await self._apply_stream_calibration_profile()
        if self._calibration_poll_task and not self._calibration_poll_task.done():
            self._calibration_poll_task.cancel()
        await self._async_stop()

    async def _kill_process_tree(self, proc, pid):
        if OS_NAME == "Windows":
            try:
                # Kill the tree while the parent PID still exists. Killing the
                # parent first can orphan MediaMTX and FFmpeg before taskkill
                # has a chance to discover its descendants.
                tree_kill = await asyncio.create_subprocess_exec(
                    "taskkill", "/f", "/t", "/pid", str(pid),
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await tree_kill.wait()
            except Exception:
                pass
        else:
            try:
                import signal
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except Exception:
                pass
        if proc.returncode is None:
            try:
                proc.kill()
            except Exception:
                pass

    async def _async_stop(self):
        if self._stopping:
            if self._closed and self.process and self.process.returncode is None:
                proc = self.process
                self.process = None
                await self._kill_process_tree(proc, proc.pid)
            return
        self._stopping = True
        self._esc_stopped = True
        self._esc_down = None
        self._cancel_starting = True

        if self._proc_lock is not None:
            shutdown_event.set()
            saved_pid = None
            proc = None
            force_kill = False
            async with self._proc_lock:
                proc = self.process
                if proc and proc.returncode is None:
                    saved_pid = proc.pid
                    force_kill = self._closed
                    if not force_kill:
                        try:
                            if OS_NAME == "Windows":
                                os.makedirs(LOG_DIR, exist_ok=True)
                                with open(STOP_REQUEST_FILE, "w", encoding="utf-8") as f:
                                    f.write(str(saved_pid))
                            elif OS_NAME == "Darwin":
                                import signal
                                os.makedirs(LOG_DIR, exist_ok=True)
                                with open(STOP_REQUEST_FILE, "w", encoding="utf-8") as f:
                                    f.write(str(saved_pid))
                                # Signal only the runtime process, NOT its
                                # whole group: MediaMTX and FFmpeg run in the
                                # child's process group, so killpg(SIGINT)
                                # shuts the RTSP server down first and the
                                # FFmpeg publisher dies with "Broken pipe" /
                                # "End of file" errors. The runtime handles
                                # SIGINT itself and stops FFmpeg before
                                # MediaMTX during teardown. Windows/Linux
                                # keep their existing stop behavior.
                                os.kill(saved_pid, signal.SIGINT)
                            else:
                                import signal
                                os.killpg(os.getpgid(saved_pid), signal.SIGINT)
                        except Exception:
                            self._diag(f"graceful stop failed:\n{traceback.format_exc()}", error=True)
                            try:
                                proc.terminate()
                            except Exception:
                                self._diag(f"proc.terminate() failed:\n{traceback.format_exc()}", error=True)
                self.process = None

            if saved_pid and proc:
                exited_cleanly = False
                if force_kill:
                    await self._kill_process_tree(proc, saved_pid)
                else:
                    try:
                        await asyncio.wait_for(
                            proc.wait(), timeout=_GRACEFUL_PROCESS_STOP_TIMEOUT_S
                        )
                        exited_cleanly = True
                    except asyncio.TimeoutError:
                        exited_cleanly = False
                    except Exception:
                        self._diag(f"proc.wait() exception:\n{traceback.format_exc()}", error=True)
                        exited_cleanly = True
                    if not exited_cleanly:
                        await self._kill_process_tree(proc, saved_pid)
        status_logger.info(UI_MESSAGES[self.locale].get("Runtime stopped", "Stopped"))
        self._starting = False
        self.set_status(UI_MESSAGES[self.locale]["Stopped"], key="Stopped")
        _set_console_quick_edit(True)
        if not self._closed:
            self._set_running_ui(False)
        # Release the log file so it is not locked by the GUI between runs.
        # The next run re-adds the file handler in append mode.
        _release_file_log_handler()

    def _release_file_log_handler_if_idle(self):
        """Free the log file once startup logging is done, unless a run is active."""
        running = getattr(self, "_starting", False) or (
            getattr(self, "process", None) is not None
            and getattr(self.process, "returncode", None) is None
        )
        if not running:
            _release_file_log_handler()

    def _show_log_panel(self):
        panel = getattr(self, "log_panel", None)
        if panel is None:
            return
        panel.visible = True
        if getattr(self, "log_body", None) is not None:
            self.log_body.visible = True
        if getattr(self, "log_toggle_btn", None) is not None:
            self.log_toggle_btn.content.value = "▼"
        if getattr(self, "log_title", None) is not None:
            self.log_title.value = UI_MESSAGES[self.locale].get("Log panel running title", "Running - live log")
            self.log_title.color = None
        self._sync_log_visibility_link()
        self._fit_window_to_content(update=False, resize_window=True)
        self._safe_update(panel, getattr(self, "log_toggle_btn", None), getattr(self, "log_title", None), getattr(self, "log_visibility_link", None))
        try:
            self.page.update()
        except RuntimeError:
            pass

    def _sync_log_visibility_link(self):
        link = getattr(self, "log_visibility_link", None)
        if link is None:
            return
        visible = bool(getattr(getattr(self, "log_panel", None), "visible", False))
        key = "Hide log panel link" if visible else "Show log panel link"
        link.value = UI_MESSAGES[self.locale].get(key, UI_MESSAGES["EN"][key])

    def _set_log_panel_visible(self, visible, save=False, update=True):
        panel = getattr(self, "log_panel", None)
        if panel is None:
            return
        panel.visible = bool(visible)
        self._sync_log_visibility_link()
        # Re-layout within the current window size. Do NOT resize the native
        # window: clicking Run shows the log panel and would otherwise clobber
        # the user's manually chosen size. The explicit "Hide log window" link
        # still fits the window via _resize_window_after_log_visibility_change.
        self._fit_window_to_content(update=update, resize_window=False)
        if save:
            path = os.path.join(BASE_DIR, "settings.yaml")
            cfg = self._config.copy()
            if os.path.exists(path):
                loaded = read_yaml(path) or {}
                cfg.update(loaded)
            cfg["Show Log Panel"] = panel.visible
            self._config.update(cfg)
            save_yaml(path, cfg)
        if update:
            self._safe_update(panel, getattr(self, "log_visibility_link", None))

    def on_log_visibility_link(self, e=None):
        self._set_log_panel_visible(not getattr(self.log_panel, "visible", False), save=True)
        asyncio.create_task(self._resize_window_after_log_visibility_change())

    async def _resize_window_after_log_visibility_change(self):
        await asyncio.sleep(0)
        self._fit_window_to_content(update=True, resize_window=True)
        await asyncio.sleep(0.5)
        self.page.window.max_width = None
        try:
            self.page.window.update()
        except RuntimeError:
            pass

    def _log_color(self, levelno, text=""):
        lower = str(text or "").lower()
        if levelno >= logging.ERROR:
            return ft.Colors.RED
        if levelno >= logging.WARNING:
            return ft.Colors.ORANGE
        if "100%" in lower or "complete" in lower or "finished" in lower or "cache hit" in lower:
            return ft.Colors.GREEN
        if any(token in lower for token in ("download", "model.safetensors", "model download", "depth model")):
            return ft.Colors.BLUE
        if levelno < logging.INFO:
            return ft.Colors.GREY
        return None

    def _format_gui_log_line(self, item):
        return item[3]

    def _log_emoji(self, logger_name, levelno):
        if logger_name != "status":
            return ""
        if levelno >= logging.ERROR:
            return "❌ "
        if levelno >= logging.WARNING:
            return "⚠️ "
        return "✅ "
    def _selected_log_filter(self):
        return getattr(getattr(self, "log_level_dd", None), "value", "ALL") or "ALL"

    def _log_item_matches_filter(self, item):
        value = self._selected_log_filter()
        levelno, name = item[0], item[1]
        if value == "ALL":
            return True
        if value == "STATUS":
            return name == "status"
        if value == "DEBUG":
            return levelno == logging.DEBUG
        if value == "INFO":
            return levelno == logging.INFO
        if value == "WARNING":
            return levelno == logging.WARNING
        if value == "ERROR":
            return levelno == logging.ERROR
        if value == "CRITICAL":
            return levelno == logging.CRITICAL
        return True

    def _make_log_span(self, item, line=None):
        levelno, name, _, _ = item
        line = self._format_gui_log_line(item) if line is None else line
        color = self._log_color(levelno, line)
        style_kwargs = {
            "size": 12,
            "weight": ft.FontWeight.BOLD if name == "status" else ft.FontWeight.NORMAL,
        }
        if color is not None:
            style_kwargs["color"] = color
        return ft.TextSpan(text=f"{line}\n", style=ft.TextStyle(**style_kwargs))

    def _append_log_span(self, span):
        log_text = getattr(self, "log_text", None)
        if log_text is None:
            return
        if log_text.spans is None:
            log_text.spans = []
        log_text.spans.append(span)
        if len(log_text.spans) > 1000:
            kept_spans = log_text.spans[-500:]
            kept_ids = {id(item) for item in kept_spans}
            log_text.spans = kept_spans
            progress_spans = getattr(self, "_progress_log_spans", {})
            self._progress_log_spans = {
                key: value for key, value in progress_spans.items() if id(value) in kept_ids
            }

    def _progress_event(self, item):
        levelno, name, _, formatted = item
        if levelno >= logging.ERROR or name not in ("child", "stdout"):
            return None
        parsed = _LOG_FILE_LINE_RE.match(formatted)
        message = parsed.group("message") if parsed else formatted
        text = _ANSI_RE.sub("", message).replace("\r", "").strip()
        if _PROGRESS_PREFIX not in text:
            return self._tqdm_progress_event(text)
        text = text[text.index(_PROGRESS_PREFIX):]
        try:
            return json.loads(text[len(_PROGRESS_PREFIX):])
        except Exception:
            return None

    def _tqdm_progress_event(self, text):
        match = _PROGRESS_PERCENT_RE.search(text)
        if not match or "|" not in text or "/" not in text:
            return None
        desc = text[:match.start()].strip(" :-|") or "Progress"
        amount = _TQDM_AMOUNT_RE.search(text)
        timing = _TQDM_TIMING_RE.search(text)
        return {
            "desc": desc,
            "percent": max(0.0, min(100.0, float(match.group("percent")))),
            "downloaded": amount.group("completed").strip() if amount else "",
            "size": amount.group("total").strip() if amount else "",
            "speed": (timing.group("speed") or "").strip() if timing else "",
            "eta": (timing.group("eta") or "00:00").strip() if timing else "",
        }

    def _update_download_progress(self, data):
        panel = getattr(self, "download_progress_panel", None)
        if panel is None:
            return
        percent = data.get("percent")
        desc = str(data.get("desc") or "Download")
        completed = data.get("downloaded") or ""
        total = data.get("size") or ""
        speed = data.get("speed") or ""
        eta = data.get("eta") or ""
        value = 0.0 if percent is None else max(0.0, min(1.0, float(percent) / 100.0))
        done = percent is not None and float(percent) >= 100.0
        color = ft.Colors.GREEN if done else ft.Colors.BLUE
        panel.visible = True
        self.download_progress_title.value = desc
        self.download_progress_title.color = color
        self.download_progress_percent.value = "?" if percent is None else f"{float(percent):.1f}%"
        self.download_progress_percent.color = color
        self.download_progress_bar.value = value
        self.download_progress_bar.color = color
        self.download_progress_detail.value = f"{completed} / {total}  {speed}  ETA {eta}".strip()
        self.download_progress_detail.color = ft.Colors.GREEN if done else ft.Colors.GREY

    def _progress_log_line(self, item):
        levelno, name, _, formatted = item
        if levelno >= logging.WARNING or name not in ("child", "stdout"):
            return None
        marker = "] "
        message = formatted.split(marker, 2)[-1] if marker in formatted else formatted
        text = _ANSI_RE.sub("", message).replace("\r", "").strip()
        match = _PROGRESS_PERCENT_RE.search(text)
        if not match or not any(token in text for token in ("|", "/", "it/s", "ETA", "Downloading", "Exporting", "Building", "Saving", "Runtime preparation")):
            return None
        percent = max(0.0, min(100.0, float(match.group("percent"))))
        desc = text[:match.start()].strip(" :-|")
        if not desc:
            desc = "Progress"
        key = re.sub(r"\s+", " ", desc)
        return key, f"{desc} {percent:5.1f}%"

    def _append_log_item(self, item):
        if not self._log_item_matches_filter(item):
            return
        event = self._progress_event(item)
        if event is not None:
            self._update_download_progress(event)
            return
        progress = self._progress_log_line(item)
        if progress is not None:
            key, line = progress
            progress_spans = getattr(self, "_progress_log_spans", {})
            existing = progress_spans.get(key)
            if existing is not None:
                existing.text = f"{line}\n"
                return
            span = self._make_log_span(item, line)
            progress_spans[key] = span
            self._progress_log_spans = progress_spans
            self._append_log_span(span)
            return
        self._append_log_span(self._make_log_span(item))

    async def _poll_log_queue(self):
        while not self._closed:
            handler = getattr(self, "gui_log_handler", None)
            log_text = getattr(self, "log_text", None)
            if handler is None or log_text is None:
                await asyncio.sleep(0.1)
                continue
            changed = False
            try:
                for _ in range(100):
                    item = handler.queue.get_nowait()
                    try:
                        self._append_log_item(item)
                        if item[0] >= logging.ERROR:
                            self._set_log_problem_state()
                    except Exception:
                        logger.exception("GUI log queue item skipped")
                    changed = True
            except queue.Empty:
                pass
            if changed:
                self._fit_window_to_content(update=False)
                self._safe_update(
                    getattr(self, "log_viewport", None),
                    getattr(self, "log_scroll_row", None),
                    log_text,
                    getattr(self, "download_progress_panel", None),
                    getattr(self, "log_title", None),
                    getattr(self, "report_issue_btn", None),
                )
            await asyncio.sleep(0.1)

    def _set_log_problem_state(self):
        if getattr(self, "log_title", None) is None:
            return
        self.log_title.value = UI_MESSAGES[self.locale].get("Log panel error title", "Issue detected - check logs")
        self.log_title.color = ft.Colors.RED
        if getattr(self, "report_issue_btn", None) is not None:
            self.report_issue_btn.visible = True

    def on_log_toggle(self, e=None):
        self.log_body.visible = not self.log_body.visible
        self.log_toggle_btn.content.value = "▼" if self.log_body.visible else "▶"
        self._fit_window_to_content()
        self._safe_update(self.log_toggle_btn, self.log_body)

    def on_log_level_filter(self, e=None):
        self.log_text.spans = []
        self._progress_log_spans.clear()
        if getattr(self, "download_progress_panel", None) is not None:
            self.download_progress_panel.visible = False
        items = _read_log_file_items()
        handler = getattr(self, "gui_log_handler", None)
        if not items and handler is not None:
            value = getattr(getattr(self, "log_level_dd", None), "value", "ALL")
            items = list(getattr(handler, "status_cache", [])) if value == "STATUS" else list(getattr(handler, "cache", []))
            if value == "ALL" and not any(item[1] == "status" for item in items):
                items.extend(getattr(handler, "status_cache", []))
        for item in items:
            self._append_log_item(item)
        self._safe_update(
            getattr(self, "log_viewport", None),
            getattr(self, "log_scroll_row", None),
            self.log_text,
        )

    def on_log_clear(self, e=None):
        self.log_text.spans = []
        self._progress_log_spans.clear()
        handler = getattr(self, "gui_log_handler", None)
        if handler is not None:
            handler.cache.clear()
            if hasattr(handler, "status_cache"):
                handler.status_cache.clear()
            while True:
                try:
                    handler.queue.get_nowait()
                except queue.Empty:
                    break
        if getattr(self, "log_title", None) is not None:
            self.log_title.value = UI_MESSAGES[self.locale].get("Log panel title", "Run Log")
            self.log_title.color = None
        self._safe_update(
            getattr(self, "log_viewport", None),
            getattr(self, "log_scroll_row", None),
            self.log_text,
            getattr(self, "download_progress_panel", None),
            getattr(self, "log_title", None),
        )

    def on_report_issue(self, e=None):
        handler = getattr(self, "gui_log_handler", None)
        try:
            lines = [
                "=== Desktop2Stereo Bug Report ===",
                f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"OS: {platform.platform()}",
                f"Device: {getattr(getattr(self, 'device_dd', None), 'value', '')}",
                f"Run Mode: {getattr(self, 'run_mode_key', '')}",
                f"Depth Model: {getattr(self, 'current_model_name', '')}",
                "",
                "=== Last log lines ===",
            ]
            if handler is not None:
                for item in list(handler.cache)[-200:]:
                    lines.append(self._format_gui_log_line(item))
            lines.extend(["", "=== Config ===", json.dumps(
                {k: v for k, v in getattr(self, "_config", {}).items() if k != "Model List"},
                indent=2,
                ensure_ascii=False,
            )])
            text = "\n".join(lines)
            try:
                import pyperclip
                pyperclip.copy(text)
            except ImportError:
                if OS_NAME == "Windows":
                    subprocess.run("clip", input=text, text=True, shell=True)
                elif OS_NAME == "Darwin":
                    subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=False)
                else:
                    raise RuntimeError("pyperclip is required to copy the log")
            self.set_status(UI_MESSAGES[self.locale].get("Bug report copied to clipboard!", "Bug report copied to clipboard!"))
        except Exception as exc:
            logger.exception("Failed to build bug report")
            self.set_status(str(exc))

    def on_open_log_file(self, e=None):
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            if not os.path.exists(LOG_FILE):
                open(LOG_FILE, "a", encoding="utf-8").close()
            if OS_NAME == "Windows":
                os.startfile(LOG_FILE)
            elif OS_NAME == "Darwin":
                subprocess.Popen(["open", LOG_FILE])
            else:
                subprocess.Popen(["xdg-open", LOG_FILE])
            self.set_status(UI_MESSAGES[self.locale].get("Opening log file", "Opening log file"))
        except Exception as exc:
            logger.exception("Failed to open log file")
            self.set_status(str(exc))

    # ── reset ──

    def reset_defaults(self, e):
        current_locale = self.locale
        current_device_label = self.device_dd.value
        current_device_idx = self.device_label_to_index.get(current_device_label, DEFAULTS["Computing Device"])
        current_primary = get_primary_monitor_index()
        is_nvidia_cuda = "CUDA" in (current_device_label or "") and not devices_module.IS_ROCM
        dynamic_defaults = DEFAULTS.copy()
        dynamic_defaults["Monitor Index"] = current_primary
        dynamic_defaults["Depth Model"] = (
            "Distill-Any-Depth-Base"
            if "Distill-Any-Depth-Base" in DEFAULT_MODEL_LIST
            else default_base_depth_model()
        )
        # Reset uses the Cinema preset with hole filling disabled.
        dynamic_defaults["Hole Fill Mode"] = "none"
        dynamic_defaults["Run Mode"] = getattr(self, "run_mode_key", "Local Viewer")
        dynamic_defaults["XR Preview Window"] = False
        if is_nvidia_cuda:
            dynamic_defaults["torch.compile"] = True
            dynamic_defaults["TensorRT"] = True
        self.apply_config(dynamic_defaults, keep_optional=False)
        self.locale = current_locale
        self.lang_dd.value = "English" if current_locale == "EN" else "简体中文"
        self.device_dd.value = current_device_label
        self._config["Language"] = current_locale
        self._config["Computing Device"] = current_device_idx
        self.update_ui_texts()
        self._sync_visibility()
        self.on_device_change(None)
        self.auto_enable_optimizers_based_on_device()
        self._fit_window_to_content(update=True, resize_window=True)

    # ── URL actions ──

    def preview_in_browser(self, e):
        try:
            import webbrowser
            url = self.stream_url_tf.value
            if not url.startswith(("http://", "https://")):
                self.set_status(UI_MESSAGES[self.locale]["invalid_url_scheme"].format(url))
                return
            webbrowser.open(url)
            self.set_status(f"{UI_MESSAGES[self.locale]['Opening URL in browser']}: {url}")
        except Exception as ex:
            self.set_status(UI_MESSAGES[self.locale]["error_preview"].format(ex))

    def copy_url_to_clipboard(self, e):
        url = self.stream_url_tf.value
        if url:
            try:
                import pyperclip
                pyperclip.copy(url)
            except ImportError:
                if OS_NAME == "Windows":
                    subprocess.run("clip", input=url, text=True, shell=True)
                elif OS_NAME == "Darwin":
                    subprocess.run("pbcopy", input=url, text=True)
            self.set_status(UI_MESSAGES[self.locale]["url_copied"], key="url_copied")
            asyncio.create_task(self._fade_status(2.0))

    async def _fade_status(self, delay):
        await asyncio.sleep(delay)
        self.set_status("", key="")

    # ── ESC long-press monitoring ──

    VK_ESC = 0x1B

    async def _esc_poll_task(self):
        if OS_NAME != "Windows":
            return
        user32 = ctypes.windll.user32
        try:
            while not self._closed:
                await asyncio.sleep(0.2)
                if self._closed:
                    break
                if user32.GetAsyncKeyState(self.VK_ESC) & 0x8000:
                    if self._esc_down is None:
                        self._esc_down = time.time()
                    elif not self._esc_stopped and (time.time() - self._esc_down >= 3.0):
                        self._esc_stopped = True
                        self._esc_down = None
                        self.set_status(UI_MESSAGES[self.locale]["esc_stop"])
                        asyncio.ensure_future(self._async_stop())
                else:
                    if self._esc_down is not None:
                        self._esc_down = None
                        self._esc_stopped = False
        except asyncio.CancelledError:
            pass

    def _on_key(self, e: ft.KeyboardEvent):
        if e.key != "Esc" or self._esc_stopped or OS_NAME == "Windows":
            return
        now = time.time()
        if self._esc_down is None:
            self._esc_down = now
            asyncio.create_task(self._esc_watch_task())
        elif now - self._esc_down >= 3.0:
            self._esc_stopped = True
            self._esc_down = None
            self.set_status(UI_MESSAGES[self.locale]["esc_stop"])
            asyncio.ensure_future(self._async_stop())

    async def _esc_watch_task(self):
        try:
            for _ in range(60):
                await asyncio.sleep(0.05)
                if self._esc_down is None or self._esc_stopped or self._closed:
                    return
                if time.time() - self._esc_down >= 3.0:
                    self._esc_stopped = True
                    self._esc_down = None
                    self.set_status(UI_MESSAGES[self.locale]["esc_stop"])
                    asyncio.ensure_future(self._async_stop())
                    return
        except asyncio.CancelledError:
            pass
