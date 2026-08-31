"""Vulkan instance portability enumeration tests (ported macOS behavior)."""

import sys
from unittest import mock

from viewer import vulkan_context


class _StopAfterInstanceCreate(Exception):
    pass


class _CreateInfo:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _fake_vk(extensions, layer_props=()):
    from types import SimpleNamespace

    vk = mock.MagicMock()
    vk.VkApplicationInfo = _CreateInfo
    vk.VkInstanceCreateInfo = _CreateInfo
    vk.vkEnumerateInstanceLayerProperties.return_value = list(layer_props)
    vk.vkEnumerateInstanceExtensionProperties.return_value = [
        SimpleNamespace(extensionName=name) for name in extensions
    ]

    def _create_instance(create_info, allocator):
        vk._created_create_info = create_info
        raise _StopAfterInstanceCreate

    vk.vkCreateInstance.side_effect = _create_instance
    return vk


def _run_create(monkeypatch, extensions):
    vk = _fake_vk(extensions)
    monkeypatch.setattr(vulkan_context, "_import_vulkan", lambda: vk)
    monkeypatch.setattr(vulkan_context, "_loader_api_version", lambda v: 0x400000)
    try:
        vulkan_context.VulkanContext.create()
    except _StopAfterInstanceCreate:
        pass
    return vk._created_create_info


def test_portability_enumeration_enabled_on_darwin(monkeypatch) -> None:
    if sys.platform != "darwin":
        return  # behavior is darwin-gated
    create_info = _run_create(
        monkeypatch,
        ["VK_KHR_portability_enumeration"],
    )
    assert create_info.flags == 0x00000001
    names = create_info.ppEnabledExtensionNames
    assert "VK_KHR_portability_enumeration" in names


def test_portability_flag_zero_when_extension_absent(monkeypatch) -> None:
    create_info = _run_create(monkeypatch, [])
    assert create_info.flags == 0


def test_moltenvk_hint_in_import_error(monkeypatch) -> None:
    # Force the vulkan import to fail so _import_vulkan raises.
    import builtins

    real_import = builtins.__import__

    def _blocking_import(name, *args, **kwargs):
        if name == "vulkan" or name.startswith("vulkan."):
            raise OSError("dlopen failed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocking_import)
    try:
        with mock.patch.dict(sys.modules, {"vulkan": None}):
            try:
                vulkan_context._import_vulkan()
                raise AssertionError("expected VulkanUnavailableError")
            except vulkan_context.VulkanUnavailableError as exc:
                message = str(exc)
                if sys.platform == "darwin":
                    assert "brew install molten-vk" in message
                else:
                    assert "brew install molten-vk" not in message
    finally:
        monkeypatch.setattr(builtins, "__import__", real_import)
