from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

pytest.importorskip("wc_rocm")
torch = pytest.importorskip("torch")

from capture import media_d3d11


class _FakeStream:
    def __init__(self, name: str):
        self.name = name
        self.wait_streams: list[object] = []
        self.synchronize_count = 0

    def wait_stream(self, other):
        self.wait_streams.append(other)

    def synchronize(self):
        self.synchronize_count += 1


class _FakeSemaphore:
    def __init__(self):
        self.waits: list[tuple[int, object]] = []
        self.signals: list[tuple[int, object]] = []

    def wait(self, value, stream):
        self.waits.append((value, stream))

    def signal(self, value, stream):
        self.signals.append((value, stream))

    def close(self):
        pass


class _FakeTexture:
    def __init__(self, shared_handle, width, height, device_id=0):
        self.shared_handle = shared_handle
        self.width = width
        self.height = height
        self.device_id = device_id
        self.copies: list[tuple[object, object]] = []
        self.closed = False

    def copy_to_tensor(self, tensor, stream):
        self.copies.append((tensor, stream))

    def close(self):
        self.closed = True


def _frame(*, slot=0, width=8, height=4, handle=0x100, fence_value=1):
    return SimpleNamespace(
        slot=slot,
        width=width,
        height=height,
        texture_shared_handle=handle,
        ready_fence_handle=0x200,
        consumed_fence_handle=0x300,
        ready_fence_value=fence_value,
    )


def test_d3d11_importer_reuses_slot_tensors_and_uses_stream_backpressure(monkeypatch):
    import_stream = _FakeStream("import")
    consumer_stream = _FakeStream("consumer")
    textures: list[_FakeTexture] = []

    def make_texture(*args):
        texture = _FakeTexture(*args)
        textures.append(texture)
        return texture

    monkeypatch.setattr(media_d3d11, "HipD3D11BgraTexture", make_texture)
    monkeypatch.setattr(
        media_d3d11.torch.cuda,
        "stream",
        lambda _stream: nullcontext(),
    )
    monkeypatch.setattr(
        media_d3d11.torch.cuda,
        "current_stream",
        lambda _device_id: consumer_stream,
    )

    importer = object.__new__(media_d3d11.D3D11VideoFrameImporter)
    importer.device_id = 0
    importer._stream = import_stream
    importer._textures = {}
    importer._tensors = {}
    importer._texture_keys = {}
    importer._ready_semaphore = _FakeSemaphore()
    importer._consumed_semaphore = _FakeSemaphore()
    importer._allocate_tensor = lambda _height, _width: SimpleNamespace(
        data_ptr=lambda: 0x1234
    )

    first = importer.import_frame(_frame(fence_value=1))
    second = importer.import_frame(_frame(fence_value=2))

    assert first is second
    assert len(importer._tensors) == 1
    assert len(textures) == 1
    assert len(textures[0].copies) == 2
    assert consumer_stream.wait_streams == [import_stream, import_stream]
    # The second import has to wait before it overwrites the reused slot.
    assert import_stream.wait_streams == [consumer_stream]
    assert import_stream.synchronize_count == 0

    importer.close()
    assert import_stream.synchronize_count == 1
    assert textures[0].closed is True
    assert importer._tensors == {}


def test_d3d11_importer_rebinds_texture_when_a_slot_changes(monkeypatch):
    import_stream = _FakeStream("import")
    consumer_stream = _FakeStream("consumer")
    textures: list[_FakeTexture] = []

    def make_texture(*args):
        texture = _FakeTexture(*args)
        textures.append(texture)
        return texture

    monkeypatch.setattr(media_d3d11, "HipD3D11BgraTexture", make_texture)
    monkeypatch.setattr(media_d3d11.torch.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(
        media_d3d11.torch.cuda,
        "current_stream",
        lambda _device_id: consumer_stream,
    )

    importer = object.__new__(media_d3d11.D3D11VideoFrameImporter)
    importer.device_id = 0
    importer._stream = import_stream
    importer._textures = {}
    importer._tensors = {}
    importer._texture_keys = {}
    importer._ready_semaphore = _FakeSemaphore()
    importer._consumed_semaphore = _FakeSemaphore()
    importer._allocate_tensor = lambda _height, _width: SimpleNamespace(
        data_ptr=lambda: 0x1234
    )

    importer.import_frame(_frame(slot=0, handle=0x100))
    importer.import_frame(_frame(slot=0, handle=0x101))

    assert len(textures) == 2
    assert textures[0].closed is True
    assert textures[1].shared_handle == 0x101
    assert len(importer._tensors) == 1
