"""Regression coverage for the output-preserving Local Viewer present path.

These tests use a recording fake Vulkan device so they run without a GPU,
swapchain or window. They assert that the fast path preserves transfer
geometry while cutting per-frame CPU work.
"""
from __future__ import annotations

from collections import Counter
from types import SimpleNamespace
from typing import Any

import pytest

import viewer.vulkan_local_viewer as lvm
from viewer.vulkan_local_viewer import (
    blit_destinations_cover_target,
    image_layout_barrier_sync,
)


class _Struct:
    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)


class _Ffi:
    NULL = None

    def __init__(self) -> None:
        self.new_calls = 0
        self.addressof_calls = 0

    def new(self, _declaration: str) -> list[int]:
        self.new_calls += 1
        return [0]

    def addressof(self, value: Any) -> int:
        self.addressof_calls += 1
        return id(value)


class _FakeVk:
    VK_SUCCESS = 0
    VK_TIMEOUT = 2
    VK_NOT_READY = 1
    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT = 1 << 0
    VK_PIPELINE_STAGE_TRANSFER_BIT = 1 << 1
    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT = 1 << 2
    VK_ACCESS_TRANSFER_READ_BIT = 1 << 3
    VK_ACCESS_TRANSFER_WRITE_BIT = 1 << 4
    VK_ACCESS_MEMORY_READ_BIT = 1 << 5
    VK_ACCESS_MEMORY_WRITE_BIT = 1 << 6
    VK_IMAGE_LAYOUT_UNDEFINED = 0
    VK_IMAGE_LAYOUT_GENERAL = 1
    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL = 6
    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL = 7
    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR = 1000001002
    VK_IMAGE_ASPECT_COLOR_BIT = 1
    VK_FORMAT_B8G8R8A8_SRGB = 50
    VK_FORMAT_R8G8B8A8_SRGB = 43
    VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = 42
    VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER = 43
    VK_STRUCTURE_TYPE_SUBMIT_INFO = 44
    VK_STRUCTURE_TYPE_PRESENT_INFO_KHR = 45
    VK_FILTER_LINEAR = 1

    class VkTimeout(Exception):
        pass

    def __init__(self) -> None:
        self.ffi = _Ffi()
        self.structs: Counter[str] = Counter()
        self.barriers: list[tuple[int, int, int, int]] = []
        self.clears = 0
        self.blits: list[Any] = []
        self.copies: list[Any] = []
        self.submits: list[Any] = []

    def _struct(self, name: str, **fields: Any) -> _Struct:
        self.structs[name] += 1
        return _Struct(**fields)

    def VkCommandBufferBeginInfo(self, **kw): return self._struct("begin", **kw)
    def VkImageSubresourceLayers(self, **kw): return self._struct("layers", **kw)
    def VkImageSubresourceRange(self, **kw): return self._struct("range", **kw)
    def VkImageMemoryBarrier(self, **kw):
        value = self._struct("barrier", **kw)
        value.srcAccessMask = kw.get("srcAccessMask", 0)
        value.dstAccessMask = kw.get("dstAccessMask", 0)
        return value
    def VkBufferImageCopy(self, **kw): return self._struct("copy", **kw)
    def VkOffset3D(self, **kw): return self._struct("offset", **kw)
    def VkExtent3D(self, **kw): return self._struct("extent", **kw)
    def VkClearColorValue(self, **kw): return self._struct("clear", **kw)
    def VkImageBlit(self, **kw): return self._struct("blit", **kw)
    def VkImageCopy(self, **kw): return self._struct("image_copy", **kw)
    def VkSubmitInfo(self, **kw): return self._struct("submit", **kw)
    def VkPresentInfoKHR(self, **kw): return self._struct("present", **kw)

    def vkWaitForFences(self, *_args): return None
    def vkResetFences(self, *_args): return None
    def vkResetCommandBuffer(self, *_args): return None
    def vkBeginCommandBuffer(self, *_args): return None
    def vkEndCommandBuffer(self, *_args): return None
    def vkCmdCopyBufferToImage(self, *_args): return None
    def vkCmdCopyImage(self, _cmd, _src, _src_layout, _dst, _dst_layout, _count, copies):
        self.copies.append(copies[0])
    def vkQueueSubmit(self, _queue, _count, submits, _fence):
        self.submits.append(submits[0])
        return None

    def vkCmdPipelineBarrier(self, _cmd, src_stage, dst_stage, *_args):
        barrier = _args[6][0]
        self.barriers.append(
            (src_stage, dst_stage, barrier.srcAccessMask, barrier.dstAccessMask)
        )

    def vkCmdClearColorImage(self, *_args):
        self.clears += 1

    def vkCmdBlitImage(self, _cmd, _src, _src_layout, _dst, _dst_layout, _count, blits, _filter):
        self.blits.append(blits[0])


class _Harness:
    def __init__(self, *, input_size: tuple[int, int] = (3840, 2160), fit_mode: str = "stretch") -> None:
        self.vk = _FakeVk()
        self.device_lookups: Counter[bytes] = Counter()
        self.acquire_calls = 0
        self.present_calls = 0
        config = SimpleNamespace(
            vsync=False,
            display_fit_enabled=True,
            display_fit_mode_provider=None,
            display_fit_mode=fit_mode,
            display_mode="Half-SBS",
            input_size=input_size,
            capture_mode="",
            window_title="",
            preview_monitor_index=0,
            monitor_index=1,
            on_breakdown_inc=None,
            on_breakdown_add_time=None,
        )

        def acquire(_device, _swapchain, _timeout, _semaphore, _fence, index_output):
            self.acquire_calls += 1
            index_output[0] = 0
            return 0

        def queue_present(_queue, _info):
            self.present_calls += 1
            return 0

        self.functions = {
            b"vkAcquireNextImageKHR": acquire,
            b"vkQueuePresentKHR": queue_present,
        }

        def device_function(name: bytes, _signature: str):
            self.device_lookups[name] += 1
            return self.functions[name]

        self.owner = SimpleNamespace(
            vk=self.vk,
            device=object(),
            fence=object(),
            queue=object(),
            command_buffer=object(),
            swapchain=object(),
            image_available=object(),
            render_finished=[object(), object(), object()],
            swap_images=[object(), object(), object()],
            _swap_image_initialized=[True, True, True],
            extent=(3840, 2160),
            source_format=self.vk.VK_FORMAT_R8G8B8A8_SRGB,
            swapchain_format=self.vk.VK_FORMAT_R8G8B8A8_SRGB,
            config=config,
            _last_presentation_geometry=None,
            _presentation_geometry_reported=False,
            _source=None,
            _direct_sources=[],
            _cursor_texture_overlay=None,
            _fps_overlay=None,
            _device_function=device_function,
            is_swapchain_out_of_date=lambda _result: False,
            is_swapchain_recreate_result=lambda _result: False,
        )
        self.source = lvm._TransferSource.__new__(lvm._TransferSource)
        self.source.owner = self.owner
        self.source.size = (3840, 2160)
        self.source.format = self.owner.source_format
        self.source.image = object()
        self.source.buffer = object()
        self.source.memory = object()
        self.source.capacity = 3840 * 2160 * 4
        self.source._image_initialized = True
        self.source._cuda_active = True
        self.source._cuda_importer = SimpleNamespace(
            copy_tensor_to_buffer=lambda *_a: None,
            signal_semaphore=lambda *_a: None,
        )
        self.source._cuda_ready = SimpleNamespace(semaphore=object())
        self.source._external_buffer = SimpleNamespace(buffer=object())
        self.source._rocm_interop = False
        self.source._mapped = None
        self.source._mapped_view = None
        self.source._slow_present_count = 0
        self.source._input_monitor_size_cache = None
        self.source.invalidate_present_caches()
        self.pixels = SimpleNamespace(is_cuda=True)

    def present(self, *, reuse: bool = False) -> bool:
        return bool(self.source.present(self.pixels, reuse=reuse))


# ── pure helpers ──────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("regions", "extent", "expected"),
    [
        ((((0, 0, 0, 0), (0, 0, 3840, 2160)),), (3840, 2160), True),
        (
            (
                ((0, 0, 0, 0), (0, 0, 1920, 2160)),
                ((0, 0, 0, 0), (1920, 0, 3840, 2160)),
            ),
            (3840, 2160),
            True,
        ),
        (
            (((0, 0, 0, 0), (0, 540, 1920, 1620)),),
            (3840, 2160),
            False,
        ),
        (
            (
                ((0, 0, 0, 0), (0, 0, 1920, 2160)),
                ((0, 0, 0, 0), (1921, 0, 3840, 2160)),
            ),
            (3840, 2160),
            False,
        ),
        ((), (3840, 2160), False),
        ((((0, 0, 0, 0), (0, 0, 4000, 2160)),), (3840, 2160), False),
    ],
)
def test_blit_coverage_is_exact(regions, extent, expected) -> None:
    assert blit_destinations_cover_target(regions, extent) is expected


def test_precise_barriers_use_transfer_stages_for_presentation_layouts() -> None:
    vk = _FakeVk()
    transfer = vk.VK_PIPELINE_STAGE_TRANSFER_BIT
    all_commands = vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT
    pairs = (
        (vk.VK_IMAGE_LAYOUT_UNDEFINED, vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL),
        (vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL),
        (vk.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL),
        (vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, vk.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR),
    )
    for old, new in pairs:
        src_stage, _src_access, dst_stage, _dst_access = image_layout_barrier_sync(vk, old, new)
        assert dst_stage == transfer
        assert src_stage in (transfer, vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT)
    # External CUDA/HIP ownership stays conservative.
    assert image_layout_barrier_sync(vk, vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                    vk.VK_IMAGE_LAYOUT_GENERAL)[2] == all_commands


# ── present path ──────────────────────────────────────────────────────────

def test_fast_present_preserves_full_frame_transfer_and_skips_redundant_clear(monkeypatch) -> None:
    monkeypatch.setenv("D2S_LOCAL_VIEWER_FAST_PRESENT", "0")
    legacy = _Harness()
    assert legacy.present() is True
    legacy_blits = [
        (tuple((o.x, o.y, o.z) for o in b.srcOffsets),
         tuple((o.x, o.y, o.z) for o in b.dstOffsets))
        for b in legacy.vk.blits
    ]
    assert legacy.vk.clears == 1

    monkeypatch.setenv("D2S_LOCAL_VIEWER_FAST_PRESENT", "1")
    fast = _Harness()
    assert fast.present() is True
    fast_blits = [
        (tuple((o.x, o.y, o.z) for o in b.srcOffsets),
         tuple((o.x, o.y, o.z) for o in b.dstOffsets))
        for b in fast.vk.blits
    ]
    assert fast_blits == []
    assert len(fast.vk.copies) == 1
    assert fast.vk.copies[0].extent.width == 3840
    assert fast.vk.copies[0].extent.height == 2160
    assert fast.vk.copies[0].extent.depth == 1
    assert fast.vk.clears == 0
    assert legacy_blits == [
        (((0, 0, 0), (3840, 2160, 1)), ((0, 0, 0), (3840, 2160, 1)))
    ]
    # Both paths keep the same number of layout transitions.
    assert len(fast.vk.barriers) == len(legacy.vk.barriers)


def test_fast_present_caches_structs_and_device_functions_across_frames(monkeypatch) -> None:
    monkeypatch.setenv("D2S_LOCAL_VIEWER_FAST_PRESENT", "1")
    h = _Harness()
    for _ in range(5):
        assert h.present() is True
    assert h.device_lookups[b"vkAcquireNextImageKHR"] == 1
    assert h.device_lookups[b"vkQueuePresentKHR"] == 1
    assert h.vk.structs["submit"] == 1
    assert h.vk.structs["present"] == 1
    assert h.vk.structs["blit"] == 0
    assert h.vk.structs["image_copy"] == 1
    assert h.vk.structs["copy"] == 1
    assert h.vk.structs["begin"] == 1
    assert h.vk.structs["layers"] == 1
    assert h.vk.ffi.new_calls == 1
    assert h.vk.ffi.addressof_calls == 1
    assert h.vk.clears == 0
    assert h.acquire_calls == h.present_calls == 5

    monkeypatch.setenv("D2S_LOCAL_VIEWER_FAST_PRESENT", "0")
    legacy = _Harness()
    for _ in range(5):
        assert legacy.present() is True
    assert legacy.device_lookups[b"vkAcquireNextImageKHR"] == 5
    assert legacy.vk.structs["submit"] == 5
    assert legacy.vk.structs["present"] == 5
    assert legacy.vk.structs["blit"] == 5
    assert legacy.vk.structs["image_copy"] == 0
    assert legacy.vk.structs["copy"] == 5
    assert legacy.vk.structs["begin"] == 5
    assert legacy.vk.structs["layers"] == 5
    assert legacy.vk.clears == 5
    # Same number of barrier commands; only their cffi allocations are cached.
    assert len(h.vk.barriers) == len(legacy.vk.barriers) == 5 * 5
    assert h.vk.structs["barrier"] == 5
    assert legacy.vk.structs["barrier"] == 25


def test_fast_present_keeps_clear_for_letterboxed_geometry(monkeypatch) -> None:
    monkeypatch.setenv("D2S_LOCAL_VIEWER_FAST_PRESENT", "1")
    h = _Harness(input_size=(3840, 1080), fit_mode="contain")
    assert h.present() is True
    assert h.vk.clears == 1
    assert h.vk.copies == []


def test_geometry_change_rebuilds_blits(monkeypatch) -> None:
    monkeypatch.setenv("D2S_LOCAL_VIEWER_FAST_PRESENT", "1")
    h = _Harness()
    assert h.present() is True
    assert h.vk.structs["blit"] == 0
    assert h.present() is True
    assert h.vk.structs["blit"] == 0  # cached copy path
    h.owner.config.display_fit_mode = "contain"
    h.owner.config.input_size = (3840, 1080)
    assert h.present() is True
    # The letterboxed layout needs two separate eye blits, so two structs.
    assert h.vk.structs["blit"] == 2
    assert h.vk.structs["image_copy"] == 1
    assert h.vk.clears == 1  # letterboxed now


def test_reuse_and_fresh_frames_do_not_thrash_geometry_cache(monkeypatch) -> None:
    """Alternating reuse/fresh frames must keep their own packed-blit geometry."""
    monkeypatch.setenv("D2S_LOCAL_VIEWER_FAST_PRESENT", "1")
    h = _Harness()
    assert h.present() is True  # fresh GPU-buffer frame
    assert h.present(reuse=True) is True
    assert h.present() is True
    assert h.present(reuse=True) is True
    # Fresh frames use one cached copy; reuse frames use two eye blits.
    assert h.vk.structs["blit"] == 2
    assert len(h.source._geometry_cache) == 2
    assert all(len(value[2]) in (1, 2) for value in h.source._geometry_cache.values())


def test_format_mismatch_uses_blit_for_fresh_and_reused_frames(monkeypatch) -> None:
    monkeypatch.setenv("D2S_LOCAL_VIEWER_FAST_PRESENT", "1")
    h = _Harness()
    h.owner.swapchain_format = h.vk.VK_FORMAT_B8G8R8A8_SRGB

    for reuse in (False, True, False, True):
        assert h.present(reuse=reuse) is True

    assert h.vk.copies == []
    assert h.vk.structs["image_copy"] == 0
    # Fresh GPU-buffer frames use one packed blit; reused frames retain two
    # eye blits so their SBS layout and per-eye mapping are unchanged.
    assert len(h.vk.blits) == 6
    assert len(h.source._geometry_cache) == 2


def test_swapchain_invalidation_drops_cached_submit_structs(monkeypatch) -> None:
    monkeypatch.setenv("D2S_LOCAL_VIEWER_FAST_PRESENT", "1")
    h = _Harness()
    assert h.present() is True
    assert h.source._submit_cache
    h.source.invalidate_present_caches()
    assert h.source._submit_cache is None
    assert h.source._present_info_cache is None
    assert h.source._geometry_cache is None
    assert h.present() is True
    assert h.vk.structs["submit"] == 2


def test_precise_barriers_switch_restores_legacy_stage_masks(monkeypatch) -> None:
    monkeypatch.setenv("D2S_LOCAL_VIEWER_FAST_PRESENT", "1")
    monkeypatch.setenv("D2S_LOCAL_VIEWER_PRECISE_BARRIERS", "0")
    off = _Harness()
    assert off.present() is True
    assert all(
        src == off.vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT
        and dst == off.vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT
        for src, dst, _access, _barrier in off.vk.barriers
    )

    monkeypatch.setenv("D2S_LOCAL_VIEWER_PRECISE_BARRIERS", "1")
    on = _Harness()
    assert on.present() is True
    assert any(
        dst == on.vk.VK_PIPELINE_STAGE_TRANSFER_BIT
        for _src, dst, _access, _barrier in on.vk.barriers
    )
