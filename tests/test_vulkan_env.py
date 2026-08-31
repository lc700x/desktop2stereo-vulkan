"""Unit tests for macOS Vulkan loader/ICD environment helper (ported)."""

import sys

from utils.vulkan_env import (
    apply_macos_vulkan_env,
    macos_vulkan_icd_files,
    macos_vulkan_library_dirs,
)


def test_library_dirs_and_icd_files_return_lists() -> None:
    assert isinstance(macos_vulkan_library_dirs(), list)
    assert isinstance(macos_vulkan_icd_files(), list)


def test_apply_macos_vulkan_env_is_noop_off_darwin(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    env = {"PATH": "/usr/bin"}
    result = apply_macos_vulkan_env(env)
    assert result is env
    assert "DYLD_FALLBACK_LIBRARY_PATH" not in env
    assert "VK_DRIVER_FILES" not in env


def test_apply_macos_vulkan_env_prepends_loader_dirs(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(
        "utils.vulkan_env.macos_vulkan_library_dirs",
        lambda: ["/opt/homebrew/lib"],
    )
    monkeypatch.setattr(
        "utils.vulkan_env.macos_vulkan_icd_files",
        lambda: ["/opt/homebrew/share/vulkan/icd.d/MoltenVK_icd.json"],
    )
    env = {"PATH": "/usr/bin", "DYLD_FALLBACK_LIBRARY_PATH": "/usr/lib"}
    result = apply_macos_vulkan_env(env)
    assert result["DYLD_FALLBACK_LIBRARY_PATH"].startswith("/opt/homebrew/lib:")
    assert result["VK_DRIVER_FILES"].endswith("MoltenVK_icd.json")
    assert result["VK_ICD_FILENAMES"].endswith("MoltenVK_icd.json")
