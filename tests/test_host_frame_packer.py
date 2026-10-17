"""Host frame packer gate tests (ported macOS pipeline)."""

import sys
import queue
import threading
from types import SimpleNamespace

from viewer.direct_sink import DirectSinkRegistry
from viewer.host_frame_packer import maybe_install_local_viewer_packer
from viewer.host_frame_packer import HostFramePacker


def test_packer_gated_off_off_darwin(monkeypatch) -> None:
    if sys.platform == "darwin":
        monkeypatch.delenv("D2S_VIEWER_HOST_FRAME", raising=False)
        monkeypatch.setenv("D2S_VIEWER_HOST_FRAME", "0")
    kwargs: dict = {"runtime_q": object()}
    installed = maybe_install_local_viewer_packer(
        pipeline_q=object(),
        local_viewer_kwargs=kwargs,
        os_name="Windows" if sys.platform != "win32" else "Linux",
    )
    assert installed is False


class _DirectSource:
    size = (4, 2)
    direct_view = bytearray(size[0] * size[1] * 4)

    def __init__(self):
        self.claimed = False
        self.completed = False

    def claim_direct(self):
        if self.claimed or self.completed:
            return False
        self.claimed = True
        return True

    def _release_direct(self):
        self.claimed = False

    def _mark_direct_submitted(self):
        self.claimed = False
        self.completed = True

    def _complete_direct(self):
        self.completed = False


def test_direct_sink_claim_is_single_use_until_fence_completion(monkeypatch) -> None:
    monkeypatch.setattr("viewer.direct_sink.sys.platform", "darwin")
    source = _DirectSource()
    registry = DirectSinkRegistry()
    registry.register(source)

    claimed, view = registry.acquire(4, 2)
    assert claimed is source
    assert view is source.direct_view
    assert registry.acquire(4, 2) == (None, None)

    source._mark_direct_submitted()
    assert registry.acquire(4, 2) == (None, None)
    source._complete_direct()
    claimed, view = registry.acquire(4, 2)
    assert claimed is source
    assert view is source.direct_view


def test_direct_sink_release_makes_claimed_slot_available(monkeypatch) -> None:
    monkeypatch.setattr("viewer.direct_sink.sys.platform", "darwin")
    source = _DirectSource()
    registry = DirectSinkRegistry()
    registry.register(source)

    claimed, _ = registry.acquire(4, 2)
    assert claimed is source
    registry.release(source)
    claimed, _ = registry.acquire(4, 2)
    assert claimed is source


def test_dropping_a_queue_item_releases_its_direct_source() -> None:
    from utils.queue_utils import _release_item

    source = _DirectSource()
    assert source.claim_direct()

    _release_item(SimpleNamespace(viewer_frame_direct=source))

    assert source.claimed is False


def test_native_packing_runs_on_worker_and_releases_frame_after_write(monkeypatch) -> None:
    from viewer.direct_sink import DIRECT_SINK

    source = _DirectSource()
    registry = DirectSinkRegistry()
    monkeypatch.setattr("viewer.direct_sink.sys.platform", "darwin")
    registry.register(source)
    monkeypatch.setattr(DIRECT_SINK, "pack_native", registry.pack_native)
    calls = []

    class Native:
        source_size = (4, 2)

        def pack(self, view, size, output_format):
            calls.append((threading.current_thread().name, size, output_format))
            assert view is source.direct_view
            assert calls[-1][0] == "HostFramePacker"

        def release(self):
            calls.append("release")

    result = SimpleNamespace(viewer_native=Native(), output_format="half_sbs")
    in_q, out_q = queue.Queue(), queue.Queue(maxsize=1)
    in_q.put((result, 42.0))
    packer = HostFramePacker(in_q, out_q)
    packer.start()
    try:
        completed, stamp = out_q.get(timeout=2)
    finally:
        packer.stop()

    assert completed is result and stamp == 42.0
    assert calls == [("HostFramePacker", (4, 2), "half_sbs"), "release"]
    assert result.viewer_native is None
    assert result.viewer_frame_direct is source
    assert source.claimed


def test_native_pack_failure_returns_staging_slot(monkeypatch) -> None:
    import pytest

    monkeypatch.setattr("viewer.direct_sink.sys.platform", "darwin")
    registry = DirectSinkRegistry()
    source = _DirectSource()
    registry.register(source)

    def fail(*_args):
        raise RuntimeError("GPU failure")

    with pytest.raises(RuntimeError, match="GPU failure"):
        registry.pack_native(SimpleNamespace(pack=fail), (4, 2), "half_sbs")
    assert registry.acquire(4, 2)[0] is source


def test_native_metal_alias_is_compared_to_cpu_handoff_then_reused(monkeypatch) -> None:
    monkeypatch.setattr("viewer.direct_sink.sys.platform", "darwin")
    source = _DirectSource()
    source._metal_buffer = object()
    source._metal_buffer_capacity = len(source.direct_view)
    registry = DirectSinkRegistry()
    registry.register(source)

    class Native:
        last_pack_metrics = {}

        def pack(self, destination, _size, _format):
            destination[:] = b"reference" + bytes(len(destination) - 9)
            self.last_pack_metrics = {"metal_buffer_alias": False}

        def pack_metal(self, _metal, _capacity, _size, _format):
            source.direct_view[:] = b"reference" + bytes(len(source.direct_view) - 9)
            self.last_pack_metrics = {"metal_buffer_alias": True}

    frame = Native()
    assert registry.pack_native(frame, (4, 2), "half_sbs") is source
    assert registry._metal_alias_verified is True
    assert source.direct_view[:9] == b"reference"
    source._release_direct()
    assert registry.pack_native(frame, (4, 2), "half_sbs") is source


def test_native_pack_uses_mapped_handoff_when_direct_ring_is_exhausted(monkeypatch) -> None:
    monkeypatch.setattr("viewer.direct_sink.sys.platform", "darwin")
    registry = DirectSinkRegistry()
    fallback = SimpleNamespace(size=(4, 2), _mapped_view=bytearray(4 * 2 * 4))

    class Native:
        last_pack_metrics = {}

        def pack(self, destination, _size, _format):
            destination[:] = bytes(range(len(destination)))
            self.last_pack_metrics = {"metal_buffer_alias": False}

    frame = Native()
    source = registry.pack_native(
        frame,
        (4, 2),
        "half_sbs",
        fallback_source=fallback,
    )

    assert source is fallback
    assert fallback._mapped_view == bytes(range(4 * 2 * 4))
    assert frame.last_pack_metrics["metal_vulkan_ring_fallback"] is True
    assert registry._ring_fallback_count == 1


def test_packer_replaces_queued_frame_without_waiting_for_viewer() -> None:
    released = []
    out_q = queue.Queue(maxsize=1)
    out_q.put(SimpleNamespace(viewer_native=SimpleNamespace(release=lambda: released.append(1))))
    packer = HostFramePacker(queue.Queue(), out_q)
    newest = object()

    packer._forward(newest)

    assert out_q.get_nowait() is newest
    assert released == [1]
