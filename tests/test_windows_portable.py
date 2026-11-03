from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts" / "tools" / "build_windows_portable.py"
SPEC = importlib.util.spec_from_file_location("d2s_windows_portable_builder", BUILDER_PATH)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def _seed_moonlight_app(root: Path, *, include_nested_web_asset: bool = True) -> Path:
    app_root = root / "app"
    server_root = app_root / "streaming" / "moonlight_server"
    files = (
        "sunshine.exe",
        "zlib1.dll",
        "d2s_amd_encoder.dll",
        "LICENSE-Sunshine.txt",
        "NOTICE-Sunshine.txt",
        "sunshine-d2s-2026.914.233613.patch",
        "assets/apps.json",
        "assets/web/index.html",
    )
    for name in files:
        path = server_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode("utf-8"))
    if include_nested_web_asset:
        web_asset = server_root / "assets" / "web" / "assets" / "theme.js"
        web_asset.parent.mkdir(parents=True, exist_ok=True)
        web_asset.write_text("window.theme = {};", encoding="utf-8")
    return app_root


def test_copy_application_excludes_sunshine_config_and_macos_payload(tmp_path, monkeypatch):
    source_app = _seed_moonlight_app(tmp_path / "source")
    server_root = source_app / "streaming" / "moonlight_server"
    (server_root / "config").mkdir()
    (server_root / "config" / "sunshine_state.json").write_text("paired client data", encoding="utf-8")
    (server_root / "macos" / "Sunshine.app").mkdir(parents=True)

    monkeypatch.setattr(builder, "APP_SOURCE", source_app)
    stage = tmp_path / "stage"
    builder.copy_application(stage)

    packaged_app = stage / "src" / "desktop2stereo"
    packaged_server = packaged_app / "streaming" / "moonlight_server"
    assert not (packaged_server / "config").exists()
    assert not (packaged_server / "macos").exists()
    manifest = builder.validate_moonlight_assets(packaged_app)
    assert manifest["file_count"] == 9
    assert manifest["files"]["assets/web/assets/theme.js"]


def test_moonlight_asset_validation_rejects_missing_nested_web_asset(tmp_path, monkeypatch):
    source_app = _seed_moonlight_app(tmp_path / "source")
    packaged_app = _seed_moonlight_app(tmp_path / "package", include_nested_web_asset=False)
    monkeypatch.setattr(builder, "APP_SOURCE", source_app)

    with pytest.raises(RuntimeError, match="assets/web/assets/theme.js"):
        builder.validate_moonlight_assets(packaged_app)
