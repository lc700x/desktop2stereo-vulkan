import os

import yaml


def read_yaml(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}
    except UnicodeDecodeError:
        try:
            with open(path, "r", encoding="gbk") as f:
                return yaml.safe_load(f) or {}
        except Exception as exc:
            print(f"Failed to load settings.yaml with GBK encoding: {exc}")
            return {}


def load_settings(path="settings.yaml"):
    settings_path = path if os.path.isabs(path) else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", path))
    return read_yaml(settings_path)
