"""Vulkan loader/ICD discovery helpers for the current platform.

On macOS there is no system Vulkan; applications go through MoltenVK.  The
pip ``vulkan`` package resolves its loader with a bare ``dlopen("libvulkan.dylib")``
and dyld only searches ``DYLD_LIBRARY_PATH`` plus the legacy fallback paths
(``~/lib``, ``/usr/local/lib``, ``/usr/lib``) -- never ``/opt/homebrew/lib`` or a
user-installed LunarG SDK tree.  When spawning the runtime subprocess we
therefore point the child at whatever loader/ICD exists on this machine.
"""
import glob
import os
import sys


def _home() -> str:
    return os.path.expanduser("~")


def macos_vulkan_library_dirs() -> list[str]:
    """Directories that may contain libvulkan*.dylib / libMoltenVK.dylib."""
    sdk = os.environ.get("VULKAN_SDK", "")
    candidates = [
        os.path.join(sdk, "lib"),
        os.path.join(sdk, "macOS", "lib"),
        "/opt/homebrew/lib",
        "/usr/local/lib",
        os.path.join(_home(), "VulkanSDK", "*", "macOS", "lib"),
        os.path.join(_home(), "VulkanSDK", "*", "lib"),
    ]
    dirs: list[str] = []
    for candidate in candidates:
        matches = sorted(glob.glob(candidate)) if "*" in candidate else [candidate]
        for path in matches:
            if os.path.isdir(path) and path not in dirs:
                if any(
                    os.path.exists(os.path.join(path, name))
                    for name in ("libvulkan.1.dylib", "libvulkan.dylib", "libMoltenVK.dylib")
                ):
                    dirs.append(path)
    return dirs


def macos_vulkan_icd_files() -> list[str]:
    """MoltenVK ICD manifests the Vulkan loader should be told about."""
    sdk = os.environ.get("VULKAN_SDK", "")
    patterns = [
        os.path.join(sdk, "share", "vulkan", "icd.d", "*.json"),
        os.path.join(sdk, "macOS", "share", "vulkan", "icd.d", "*.json"),
        "/opt/homebrew/etc/vulkan/icd.d/*.json",
        "/opt/homebrew/share/vulkan/icd.d/*.json",
        "/usr/local/etc/vulkan/icd.d/*.json",
        "/usr/local/share/vulkan/icd.d/*.json",
        os.path.join(_home(), ".config", "vulkan", "icd.d", "*.json"),
        os.path.join(_home(), "VulkanSDK", "*", "macOS", "share", "vulkan", "icd.d", "*.json"),
    ]
    files: list[str] = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            if os.path.isfile(path) and path not in files:
                files.append(path)
    return files


def apply_macos_vulkan_env(env: dict) -> dict:
    """Extend a child-process env so macOS can resolve the Vulkan loader/ICD.

    No-op on other platforms and when no Vulkan runtime is installed.
    """
    if sys.platform != "darwin":
        return env
    library_dirs = macos_vulkan_library_dirs()
    if library_dirs:
        fallback = env.get("DYLD_FALLBACK_LIBRARY_PATH", "")
        parts = [part for part in fallback.split(":") if part]
        for directory in reversed(library_dirs):
            if directory not in parts:
                parts.insert(0, directory)
        env["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(parts)
    icd_files = macos_vulkan_icd_files()
    if icd_files:
        # VK_DRIVER_FILES is the current name; VK_ICD_FILENAMES is the
        # legacy alias still honoured by older loaders.
        joined = ":".join(icd_files)
        env.setdefault("VK_DRIVER_FILES", joined)
        env.setdefault("VK_ICD_FILENAMES", joined)
    return env
