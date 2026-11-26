from __future__ import annotations

import logging
import sys

from gui import process as gui_process


def test_gui_log_cleanup_preserves_launcher_pid_file(tmp_path) -> None:
    pid_file = tmp_path / "gui.pid"
    stale_file = tmp_path / "stale.log"
    pid_file.write_text("123\n", encoding="utf-8")
    stale_file.write_text("old\n", encoding="utf-8")

    original = {
        "LOG_DIR": gui_process.LOG_DIR,
        "LOG_FILE": gui_process.LOG_FILE,
        "STREAM_CALIBRATION_PROFILE_FILE": gui_process.STREAM_CALIBRATION_PROFILE_FILE,
        "GUI_PID_FILE": getattr(gui_process, "GUI_PID_FILE", None),
        "installed": gui_process._console_logging_installed,
        "handler": gui_process._file_log_handler,
        "gui_handler": gui_process._gui_log_handler,
        "async_filter": gui_process._asyncio_shutdown_noise_filter_installed,
        "root_handlers": list(logging.getLogger().handlers),
        "root_level": logging.getLogger().level,
        "root_disabled": logging.getLogger().disabled,
        "unraisablehook": sys.unraisablehook,
        "flet_disabled": {
            name: logging.getLogger(name).disabled for name in gui_process._FLET_LOGGER_NAMES
        },
    }
    try:
        gui_process.LOG_DIR = str(tmp_path)
        gui_process.LOG_FILE = str(tmp_path / "desktop2stereo.log")
        gui_process.STREAM_CALIBRATION_PROFILE_FILE = str(tmp_path / "profile.json")
        gui_process.GUI_PID_FILE = str(pid_file)
        gui_process._console_logging_installed = False
        gui_process._file_log_handler = None

        gui_process._setup_console_logging()

        assert pid_file.is_file()
        assert not stale_file.exists()
    finally:
        gui_process._release_file_log_handler()
        for root, level in original.items():
            if root == "installed":
                gui_process._console_logging_installed = level
            elif root == "handler":
                gui_process._file_log_handler = level
            elif root == "gui_handler":
                gui_process._gui_log_handler = level
            elif root == "async_filter":
                gui_process._asyncio_shutdown_noise_filter_installed = level
            elif root == "root_handlers":
                logging.getLogger().handlers = level
            elif root == "root_level":
                logging.getLogger().setLevel(level)
            elif root == "root_disabled":
                logging.getLogger().disabled = level
            elif root == "unraisablehook":
                sys.unraisablehook = level
            elif root == "flet_disabled":
                for name, disabled in level.items():
                    logging.getLogger(name).disabled = disabled
            else:
                setattr(gui_process, root, level)
