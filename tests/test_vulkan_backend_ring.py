from __future__ import annotations

from types import SimpleNamespace

import torch

import stereo_runtime.vulkan_backend as backend_module
from stereo_runtime.vulkan_backend import VulkanStereoImageComputeBackend
from stereo_runtime.vulkan_stereo_pass import VulkanLayeredStereoParams


def test_host_input_ring_waits_only_when_its_slot_is_reused(monkeypatch) -> None:
    waits: list[int] = []
    created_buffers = []
    submissions = []

    class FakeVk:
        VK_IMAGE_LAYOUT_GENERAL = 1
        VK_ACCESS_SHADER_WRITE_BIT = 2
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT = 4

    class FakeContext:
        frame_context_count = 3
        closed = False
        queue_family_index = 0
        vk = FakeVk()
        device_info = SimpleNamespace(name="fake-vulkan")

        @staticmethod
        def wait_for_timeline(value):
            waits.append(int(value))

        @staticmethod
        def register_image_state(_image, _state):
            return None

    class FakeBuffer:
        def __init__(self, context, size):
            self.context = context
            self.size = int(size)
            self.writes = 0
            self.closed = False
            created_buffers.append(self)

        def write_bytes(self, _payload):
            self.writes += 1

        def close(self):
            self.closed = True

    class FakePass:
        input_buffer_sizes = {"rgb": 48, "depth": 16}

        def __init__(self, context, **_kwargs):
            self.context = context
            self.closed = False

        def submit(self, rgb, depth, left, right, **kwargs):
            timeline = len(submissions) + 1
            submissions.append((rgb, depth, left, right, kwargs))
            return timeline

        def close(self):
            self.closed = True

    monkeypatch.setattr(backend_module, "VulkanStorageBuffer", FakeBuffer)
    monkeypatch.setattr(backend_module, "VulkanStereoImagePass", FakePass)

    context = FakeContext()
    backend = VulkanStereoImageComputeBackend(context)
    rgb = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    depth = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    left = SimpleNamespace(context=context, width=2, height=2, image="left")
    right = SimpleNamespace(context=context, width=2, height=2, image="right")

    slots = []
    for _index in range(4):
        _timeline, debug = backend.submit_to_images(
            rgb,
            depth,
            left,
            right,
            params=VulkanLayeredStereoParams(),
        )
        slots.append(debug["vulkan_input_ring_slot"])

    assert slots == [0, 1, 2, 0]
    assert len(created_buffers) == 6
    assert waits == [1]
    assert submissions[0][0:2] == tuple(backend._host_input_slots[0])
    assert submissions[3][0:2] == tuple(backend._host_input_slots[0])
    assert all(item[4]["signal_semaphore"] is None for item in submissions)
    assert all(item[4]["wait_semaphore"] is None for item in submissions)

    backend.close()
    assert waits[-1] == 4
    assert all(buffer.closed for buffer in created_buffers)


def test_cuda_input_ring_reuses_slots_with_device_side_semaphores(monkeypatch) -> None:
    waits: list[int] = []
    submissions = []
    events = []

    class FakeVk:
        VK_IMAGE_LAYOUT_GENERAL = 1
        VK_ACCESS_SHADER_WRITE_BIT = 2
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT = 4

    class FakeContext:
        frame_context_count = 3
        closed = False
        queue_family_index = 0
        vk = FakeVk()
        device_info = SimpleNamespace(name="fake-vulkan")

        @staticmethod
        def wait_for_timeline(value):
            waits.append(int(value))

        @staticmethod
        def register_image_state(_image, _state):
            return None

    class FakePass:
        input_buffer_sizes = {"rgb": 48, "depth": 16}

        def __init__(self, context, **_kwargs):
            self.context = context

        def submit(self, rgb, depth, left, right, **kwargs):
            timeline = len(submissions) + 1
            submissions.append((rgb, depth, left, right, kwargs))
            return timeline

        def close(self):
            return None

    class FakeImporter:
        def wait_semaphore(self, semaphore, *, stream):
            events.append(("wait", semaphore.label, stream))

        def copy_tensor_to_buffer(self, _tensor, buffer, *, stream):
            events.append(("copy", buffer.label, stream))

        def signal_semaphore(self, semaphore, *, stream):
            events.append(("signal", semaphore.label, stream))

        def close(self):
            events.append(("close",))

    class FakeSemaphore:
        def __init__(self, label):
            self.label = label
            self.semaphore = f"vk:{label}"

        def close(self):
            return None

    monkeypatch.setattr(backend_module, "VulkanStereoImagePass", FakePass)
    monkeypatch.setattr(
        backend_module.torch.cuda,
        "current_stream",
        lambda **_kwargs: SimpleNamespace(cuda_stream=77),
    )
    monkeypatch.setattr(
        VulkanStereoImageComputeBackend,
        "_validate_inputs",
        staticmethod(lambda _rgb, _depth: (2, 2)),
    )

    context = FakeContext()
    backend = VulkanStereoImageComputeBackend(context)

    def ensure_cuda_inputs(_height, _width):
        if backend._cuda_importer is not None:
            return
        backend._cuda_importer = FakeImporter()
        backend._cuda_input_slots = tuple(
            (
                SimpleNamespace(
                    context=context, size=48, label=f"rgb-{index}", close=lambda: None
                ),
                SimpleNamespace(
                    context=context, size=16, label=f"depth-{index}", close=lambda: None
                ),
            )
            for index in range(3)
        )
        backend._cuda_input_ready = tuple(
            FakeSemaphore(f"ready-{index}") for index in range(3)
        )
        backend._cuda_input_released = tuple(
            FakeSemaphore(f"released-{index}") for index in range(3)
        )

    backend._ensure_cuda_inputs = ensure_cuda_inputs
    tensor = SimpleNamespace(
        device=SimpleNamespace(type="cuda"),
        dtype=torch.float32,
        is_contiguous=lambda: True,
    )
    left = SimpleNamespace(context=context, width=2, height=2, image="left")
    right = SimpleNamespace(context=context, width=2, height=2, image="right")

    for _index in range(4):
        backend.submit_to_images(
            tensor,
            tensor,
            left,
            right,
            params=VulkanLayeredStereoParams(),
        )

    assert waits == []
    assert events.count(("wait", "released-0", 77)) == 1
    assert events.count(("wait", "released-1", 77)) == 0
    assert submissions[0][4]["wait_semaphore"] == "vk:ready-0"
    assert submissions[0][4]["signal_semaphore"] == "vk:released-0"
    assert submissions[3][4]["wait_semaphore"] == "vk:ready-0"
    assert submissions[3][4]["signal_semaphore"] == "vk:released-0"

    backend.close()
    assert waits == [4]


def test_prebound_input_dispatch_elides_host_upload_and_output_readback(monkeypatch) -> None:
    submissions = []

    class FakeVk:
        VK_IMAGE_LAYOUT_GENERAL = 1
        VK_ACCESS_SHADER_WRITE_BIT = 2
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT = 4

    class FakeContext:
        frame_context_count = 3
        closed = False
        queue_family_index = 0
        vk = FakeVk()
        device_info = SimpleNamespace(name="fake-vulkan")

        @staticmethod
        def register_image_state(_image, _state):
            return None

    class FakePass:
        input_buffer_sizes = {"rgb": 48, "depth": 16}
        depth_gaussian_enabled = False

        def __init__(self, context, **_kwargs):
            self.context = context

        def submit(self, rgb, depth, left, right, **kwargs):
            submissions.append((rgb, depth, left, right, kwargs))
            return len(submissions)

        def close(self):
            return None

    monkeypatch.setattr(backend_module, "VulkanStereoImagePass", FakePass)

    context = FakeContext()
    backend = VulkanStereoImageComputeBackend(context)
    rgb_buffer = SimpleNamespace(context=context, size=48)
    depth_buffer = SimpleNamespace(context=context, size=16)
    left = SimpleNamespace(context=context, width=2, height=2, image="left")
    right = SimpleNamespace(context=context, width=2, height=2, image="right")

    timeline, debug = backend.submit_prebound_to_images(
        width=2,
        height=2,
        rgb_buffer=rgb_buffer,
        depth_buffer=depth_buffer,
        left_eye=left,
        right_eye=right,
        params=VulkanLayeredStereoParams(),
        wait_semaphore="ready",
        signal_semaphore="released",
    )

    assert timeline == 1
    assert submissions[0][:4] == (rgb_buffer, depth_buffer, left, right)
    assert submissions[0][4]["wait_semaphore"] == "ready"
    assert submissions[0][4]["signal_semaphore"] == "released"
    assert debug["vulkan_input_path"] == "prebound_vulkan_buffer"
    assert debug["vulkan_input_upload_elided"] is True
    assert debug["vulkan_input_upload_ms"] == 0.0
    assert debug["vulkan_readback"] == "none"
    assert debug["vulkan_zero_cpu_readback"] is True
    assert debug["vulkan_zero_copy"] is False
    assert debug["vulkan_zero_copy_reason"] == "prebound_input_buffer_provenance_unverified"
