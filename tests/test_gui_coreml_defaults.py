from __future__ import annotations

from types import SimpleNamespace

import pytest

from gui.config import default_coreml_enabled
from gui.process import GUIProcessMixin
from utils.bootstrap import _normalize_legacy_settings


def test_coreml_default_is_macos_only() -> None:
    assert default_coreml_enabled("Darwin") is True
    assert default_coreml_enabled("Windows") is False
    assert default_coreml_enabled("Linux") is False


def test_legacy_settings_use_coreml_default_on_macos() -> None:
    assert _normalize_legacy_settings({}, os_name="Darwin")["CoreML"] is True
    assert _normalize_legacy_settings({}, os_name="Windows")["CoreML"] is False


def test_reset_defaults_enables_coreml_on_macos(monkeypatch) -> None:
    import gui.process as gui_process

    class Harness(GUIProcessMixin):
        def __init__(self):
            self.locale = "EN"
            self.device_dd = SimpleNamespace(value="MPS: Apple Silicon")
            self.device_label_to_index = {"MPS: Apple Silicon": 1}
            self.run_mode_key = "Local Viewer"
            self.lang_dd = SimpleNamespace(value=None)
            self._config = {}
            self.applied = None

        def apply_config(self, config, keep_optional=False):
            self.applied = config

        def update_ui_texts(self):
            pass

        def _sync_visibility(self):
            pass

        def on_device_change(self, _event):
            pass

        def auto_enable_optimizers_based_on_device(self):
            pass

        def _fit_window_to_content(self, **_kwargs):
            pass

    monkeypatch.setattr(gui_process, "OS_NAME", "Darwin")
    monkeypatch.setattr(gui_process, "get_primary_monitor_index", lambda: 1)
    target = Harness()

    target.reset_defaults(None)

    assert target.applied is not None
    assert target.applied["CoreML"] is True


@pytest.mark.parametrize("is_rocm", [False, True])
def test_reset_defaults_keeps_coreml_disabled_on_windows(monkeypatch, is_rocm) -> None:
    import gui.process as gui_process

    class Harness:
        locale = "EN"
        device_dd = SimpleNamespace(value="CUDA 0: test")
        device_label_to_index = {"CUDA 0: test": 0}
        run_mode_key = "Local Viewer"
        lang_dd = SimpleNamespace(value=None)
        _config = {}

        def apply_config(self, config, keep_optional=False):
            self.applied = config

        def update_ui_texts(self):
            pass

        def _sync_visibility(self):
            pass

        def on_device_change(self, _event):
            pass

        def auto_enable_optimizers_based_on_device(self):
            pass

        def _fit_window_to_content(self, **_kwargs):
            pass

    monkeypatch.setattr(gui_process, "OS_NAME", "Windows")
    monkeypatch.setattr(gui_process.devices_module, "IS_ROCM", is_rocm)
    monkeypatch.setattr(gui_process, "get_primary_monitor_index", lambda: 1)
    target = Harness()

    GUIProcessMixin.reset_defaults(target, None)

    assert target.applied["CoreML"] is False
    assert target.applied["MIGraphX"] is is_rocm
    assert target.applied["TensorRT"] is (not is_rocm)
