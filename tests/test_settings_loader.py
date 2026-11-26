from __future__ import annotations

from utils.settings import load_settings, read_yaml


def test_missing_settings_file_returns_defaults(tmp_path) -> None:
    missing = tmp_path / "settings.yaml"

    assert read_yaml(str(missing)) == {}
    assert load_settings(str(missing)) == {}
