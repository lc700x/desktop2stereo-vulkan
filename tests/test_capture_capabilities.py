from __future__ import annotations

from capture import capabilities


def test_darwin_probe_reports_implemented_capture_backends(monkeypatch) -> None:
    monkeypatch.setattr(capabilities.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        capabilities,
        "_module_available",
        lambda name: name in {"ScreenCaptureKit", "Quartz"},
    )

    report = capabilities.probe_capture_capabilities()

    assert report["platform"] == "Darwin"
    assert report["implemented"]["monitor_capture"] is True
    assert report["implemented"]["window_capture"] is True
    assert report["screen_capture_kit"]["available"] is True
    assert report["core_graphics"]["available"] is True
    assert report["hardware_verified"] is False
    assert "未实现" not in report["status"]


def test_darwin_probe_degrades_cleanly_without_apple_modules(monkeypatch) -> None:
    monkeypatch.setattr(capabilities.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(capabilities, "_module_available", lambda _name: False)

    report = capabilities.probe_capture_capabilities()

    assert report["implemented"]["monitor_capture"] is False
    assert report["screen_capture_kit"]["available"] is False
    assert report["core_graphics"]["available"] is False
