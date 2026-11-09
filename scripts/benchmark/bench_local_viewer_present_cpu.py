"""Measure the CPU/GPU-command cost of one Local Viewer present() call.

Uses the real PyVulkan cffi struct constructors (no device needed) with
recording no-op Vulkan entry points. This isolates the per-frame Python/cffi
record overhead that D2S_LOCAL_VIEWER_FAST_PRESENT removes, and counts the GPU
commands each path issues. It is NOT an end-to-end FPS measurement: run the
real Local Viewer for that (see the A/B notes in the task report).

Usage:
    src/python3_cuda/python.exe scripts/benchmark/bench_local_viewer_present_cpu.py
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "desktop2stereo"))

import vulkan as _vkmod  # noqa: E402
import viewer.vulkan_local_viewer as lvm  # noqa: E402

FRAMES = 3000
WARMUP = 200


class _RecordingVk:
    """Real cffi structs, no-op Vulkan entry points with call counters."""

    def __init__(self) -> None:
        self.ffi = _vkmod.ffi
        self.calls: Counter[str] = Counter()
        self.structs: Counter[str] = Counter()
        self.barriers: list[tuple[int, int]] = []
        self.blits: list[Any] = []
        self.copies: list[Any] = []

    def __getattr__(self, name: str) -> Any:
        value = getattr(_vkmod, name)
        if name.startswith("Vk") and name[2:3].isupper():
            def build(**kwargs: Any) -> Any:
                self.structs[name] += 1
                return value(**kwargs)

            return build
        return value

    def _noop(self, name: str):
        self.calls[name] += 1
        return None

    def vkWaitForFences(self, *_a): return self._noop("wait_fences")
    def vkResetFences(self, *_a): return self._noop("reset_fences")
    def vkResetCommandBuffer(self, *_a): return self._noop("reset_cmd")
    def vkBeginCommandBuffer(self, *_a): return self._noop("begin_cmd")
    def vkEndCommandBuffer(self, *_a): return self._noop("end_cmd")
    def vkCmdCopyBufferToImage(self, *_a): return self._noop("copy_buffer_to_image")
    def vkCmdCopyImage(self, _cmd, _src, _sl, _dst, _dl, _n, copies):
        self.calls["copy_image"] += 1
        self.copies.append(copies[0])
    def vkQueueSubmit(self, _q, _n, _submits, _fence): return self._noop("queue_submit")

    def vkCmdPipelineBarrier(self, _cmd, src_stage, dst_stage, *_a):
        self.calls["pipeline_barrier"] += 1
        self.barriers.append((src_stage, dst_stage))

    def vkCmdClearColorImage(self, *_a):
        self.calls["clear_color_image"] += 1

    def vkCmdBlitImage(self, _cmd, _src, _sl, _dst, _dl, _n, blits, _filter):
        self.calls["blit_image"] += 1
        self.blits.append(blits[0])


class _Harness:
    def __init__(self) -> None:
        self.vk = _RecordingVk()
        self.device_lookups: Counter[bytes] = Counter()
        handles = [_vkmod.ffi.cast("VkSemaphore", i + 1) for i in range(4)]
        images = [_vkmod.ffi.cast("VkImage", 100 + i) for i in range(4)]
        self.images = images
        self.handles = handles
        config = SimpleNamespace(
            vsync=False,
            display_fit_enabled=True,
            display_fit_mode_provider=None,
            display_fit_mode="stretch",
            display_mode="Half-SBS",
            input_size=(3840, 2160),
            capture_mode="",
            window_title="",
            preview_monitor_index=0,
            monitor_index=1,
            on_breakdown_inc=None,
            on_breakdown_add_time=None,
        )

        def acquire(_d, _s, _t, _sem, _fence, index_output):
            index_output[0] = 0
            return 0

        def queue_present(_q, _info):
            return 0

        functions = {
            b"vkAcquireNextImageKHR": acquire,
            b"vkQueuePresentKHR": queue_present,
        }

        def device_function(name: bytes, _signature: str):
            self.device_lookups[name] += 1
            return functions[name]

        self.owner = SimpleNamespace(
            vk=self.vk,
            device=object(),
            fence=_vkmod.ffi.cast("VkFence", 7),
            queue=_vkmod.ffi.cast("VkQueue", 8),
            command_buffer=_vkmod.ffi.cast("VkCommandBuffer", 9),
            swapchain=_vkmod.ffi.cast("VkSwapchainKHR", 10),
            image_available=handles[0],
            render_finished=handles[1:],
            swap_images=images[1:],
            _swap_image_initialized=[True, True, True],
            extent=(3840, 2160),
            config=config,
            _last_presentation_geometry=None,
            _presentation_geometry_reported=False,
            _source=None,
            _direct_sources=[],
            _cursor_texture_overlay=None,
            _fps_overlay=None,
            _device_function=device_function,
            is_swapchain_out_of_date=lambda _r: False,
            is_swapchain_recreate_result=lambda _r: False,
        )
        self.source = lvm._TransferSource.__new__(lvm._TransferSource)
        source = self.source
        source.owner = self.owner
        source.size = (3840, 2160)
        source.image = images[0]
        source.buffer = _vkmod.ffi.cast("VkBuffer", 200)
        source.memory = object()
        source.capacity = 3840 * 2160 * 4
        source._image_initialized = True
        source._cuda_active = True
        source._cuda_importer = SimpleNamespace(
            copy_tensor_to_buffer=lambda *_a: None,
            signal_semaphore=lambda *_a: None,
        )
        source._cuda_ready = SimpleNamespace(semaphore=handles[1])
        source._external_buffer = SimpleNamespace(
            buffer=_vkmod.ffi.cast("VkBuffer", 201)
        )
        source._rocm_interop = False
        source._mapped = None
        source._mapped_view = None
        source._slow_present_count = 0
        source._input_monitor_size_cache = None
        source.invalidate_present_caches()
        self.pixels = SimpleNamespace(is_cuda=True)

    def present(self) -> bool:
        return bool(self.source.present(self.pixels))


def run_case(label: str, *, fast: bool, precise: bool) -> dict:
    os.environ["D2S_LOCAL_VIEWER_FAST_PRESENT"] = "1" if fast else "0"
    os.environ["D2S_LOCAL_VIEWER_PRECISE_BARRIERS"] = "1" if precise else "0"
    h = _Harness()
    for _ in range(WARMUP):
        assert h.present() is True
    h.vk.calls.clear()
    h.vk.structs.clear()
    h.vk.barriers.clear()
    h.vk.blits.clear()
    h.vk.copies.clear()
    h.device_lookups.clear()
    samples: list[float] = []
    for _ in range(FRAMES):
        started = time.perf_counter()
        assert h.present() is True
        samples.append((time.perf_counter() - started) * 1000.0)
    ordered = sorted(samples)
    return {
        "label": label,
        "frames": FRAMES,
        "mean_ms": round(statistics.fmean(samples), 5),
        "p50_ms": round(ordered[len(ordered) // 2], 5),
        "p95_ms": round(ordered[int(len(ordered) * 0.95)], 5),
        "max_ms": round(ordered[-1], 5),
        "gpu_commands_per_frame": {
            name: round(count / FRAMES, 3)
            for name, count in sorted(h.vk.calls.items())
        },
        "struct_allocs_total": dict(sorted(h.vk.structs.items())),
        "device_fn_lookups_total": {
            name.decode(): count for name, count in h.device_lookups.items()
        },
        "transfer_stage_barriers": sum(
            1 for src, dst in h.vk.barriers
            if dst == _vkmod.VK_PIPELINE_STAGE_TRANSFER_BIT
        ),
        "blit_count": len(h.vk.blits),
        "copy_count": len(h.vk.copies),
    }


def main() -> int:
    cases = [
        run_case("legacy", fast=False, precise=False),
        run_case("fast_precise", fast=True, precise=True),
        run_case("fast_legacy_barriers", fast=True, precise=False),
    ]
    legacy_mean = cases[0]["mean_ms"]
    for case in cases[1:]:
        case["cpu_reduction_vs_legacy_pct"] = round(
            (1.0 - case["mean_ms"] / legacy_mean) * 100.0, 2
        )
    report = {
        "environment": {
            "python": sys.version.split()[0],
            "frames": FRAMES,
            "warmup": WARMUP,
            "note": "CPU record path only; no GPU device, capture or swapchain present.",
        },
        "cases": cases,
    }
    out = ROOT / "outputs" / "visual_regression" / "local_viewer_present_cpu.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nreport: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
