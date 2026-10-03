from types import SimpleNamespace
from string import Formatter

import pytest

from gui.localization import MESSAGE_CATALOGS, _STATUS_MESSAGES, translate_status
from gui.process import GUIProcessMixin


@pytest.mark.parametrize("key", _STATUS_MESSAGES)
def test_status_messages_translate_both_ways(key):
    en = MESSAGE_CATALOGS["EN"][key]
    cn = MESSAGE_CATALOGS["CN"][key]
    fields = {field: "model_fp16_294x518.onnx" for _, field, _, _ in Formatter().parse(en)
              if field is not None}
    english = en.format(**fields)
    chinese = cn.format(**fields)
    assert translate_status(english, "CN") == chinese
    assert translate_status(chinese, "EN") == ("Download" if key == "download" else english)


def test_formatted_status_refresh_preserves_arguments_and_replaces_stale_key():
    class Harness(GUIProcessMixin):
        locale = "EN"
        status_text = SimpleNamespace(value="")

        def _safe_update(self, *controls):
            pass

    gui = Harness()
    gui.set_status(MESSAGE_CATALOGS["EN"]["Running"], key="Running")
    gui.set_status("Calibration result error: missing profile.json")
    gui.locale = "CN"
    gui._refresh_status_display()
    assert gui.status_text.value == "校准结果错误：missing profile.json"
    assert gui._status_key is None
    gui.set_status(MESSAGE_CATALOGS["CN"]["exited_with_code"].format(77))
    gui.locale = "EN"
    gui._refresh_status_display()
    assert gui.status_text.value == MESSAGE_CATALOGS["EN"]["exited_with_code"].format(77)
    gui.set_status("", key="")
    gui.locale = "CN"
    gui._refresh_status_display()
    assert gui.status_text.value == ""


def test_unknown_diagnostic_is_preserved():
    diagnostic = "VkStatus -4: driver detail C:/models/model.onnx"
    assert translate_status(diagnostic, "CN") == diagnostic


@pytest.mark.parametrize("label,detail", [
    ("Selected input monitor:", "1"),
    ("Selected input window:", "My video - 01"),
    ("Failed to load settings.yaml:", "C:/settings.yaml"),
])
def test_status_prefixes_refresh_without_translating_user_data(label, detail):
    en = MESSAGE_CATALOGS["EN"][label] + " " + detail
    cn = MESSAGE_CATALOGS["CN"][label] + " " + detail
    assert translate_status(en, "CN") == cn
    assert translate_status(cn, "EN") == en


def test_progress_refresh_translates_labels_and_preserves_measurements():
    gui = GUIProcessMixin()
    gui.locale = "CN"
    for name in ("panel", "title", "percent", "bar", "detail"):
        setattr(gui, f"download_progress_{name}", SimpleNamespace(value="", visible=False))
    gui._update_download_progress({
        "desc": "Exporting ONNX: model.onnx", "percent": 50,
        "downloaded": "5 steps", "size": "10 steps", "speed": "1.0 steps/s", "eta": "00:05",
    })
    assert gui.download_progress_title.value == "正在导出 ONNX：model.onnx"
    assert "5 步 / 10 步" in gui.download_progress_detail.value
    assert "预计剩余时间 00:05" in gui.download_progress_detail.value
    gui.locale = "EN"
    gui._update_download_progress(gui._download_progress_payload)
    assert gui.download_progress_title.value == "Exporting ONNX: model.onnx"
    assert "ETA 00:05" in gui.download_progress_detail.value
