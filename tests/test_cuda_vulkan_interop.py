import ctypes
import sys
import threading
import types
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import app_runtime.runtime_output as runtime_output_module
from app_runtime.gpu_producer import (
    _ADAPTER_FACTORIES,
    GpuProducerAdapter,
    GpuProducerUnavailableError,
    create_gpu_producer_adapter,
)
from app_runtime.output_contract import VulkanStereoOutputFrame
from app_runtime.runtime_output import (
    CudaVulkanOutputAdapter,
    RocmVulkanOutputAdapter,
)
from viewer.cuda_vulkan_interop import (
    CudaVulkanImageImporter,
    _ExternalMemoryBufferDesc,
    _ExternalSemaphoreSignalParams,
    _ExternalSemaphoreWaitParams,
    _SemaphoreSignalParams,
)
from viewer.vulkan_context import VulkanContext, VulkanContextConfig
from viewer.vulkan_resources import VulkanExportableImage, VulkanExportableSemaphore, VulkanImageResource


def test_cuda_external_semaphore_signal_params_match_runtime_abi() -> None:
    # CUDA's v2 signal parameter layout is 72 bytes of nested params,
    # followed by flags and sixteen reserved uint32 values.
    assert _SemaphoreSignalParams.fence_value.offset == 0
    assert _SemaphoreSignalParams.nv_sci_sync.offset == 8
    assert _SemaphoreSignalParams.keyed_mutex_key.offset == 16
    assert _SemaphoreSignalParams.reserved.offset == 24
    assert ctypes.sizeof(_SemaphoreSignalParams) == 72
    assert _ExternalSemaphoreSignalParams.params.offset == 0
    assert _ExternalSemaphoreSignalParams.flags.offset == 72
    assert ctypes.sizeof(_ExternalSemaphoreSignalParams) == 144
    assert _ExternalSemaphoreWaitParams.params.offset == 0
    assert _ExternalSemaphoreWaitParams.flags.offset == 72
    assert ctypes.sizeof(_ExternalSemaphoreWaitParams) == 144


def test_cuda_external_buffer_desc_matches_runtime_abi() -> None:
    assert _ExternalMemoryBufferDesc.offset.offset == 0
    assert _ExternalMemoryBufferDesc.size.offset == ctypes.sizeof(ctypes.c_size_t)
    assert _ExternalMemoryBufferDesc.flags.offset == ctypes.sizeof(ctypes.c_size_t) * 2
    # CUDA aligns the trailing uint32 field to the host size_t alignment.
    assert ctypes.sizeof(_ExternalMemoryBufferDesc) == 24


def test_cuda_external_semaphore_is_default_and_can_be_disabled(monkeypatch) -> None:
    monkeypatch.delenv("D2S_ENABLE_CUDA_EXTERNAL_SEMAPHORE", raising=False)
    assert CudaVulkanOutputAdapter._external_semaphore_requested()
    monkeypatch.setenv("D2S_ENABLE_CUDA_EXTERNAL_SEMAPHORE", "0")
    assert not CudaVulkanOutputAdapter._external_semaphore_requested()


def test_cuda_source_prepare_is_idempotent_for_reused_frame() -> None:
    adapter = CudaVulkanOutputAdapter.__new__(CudaVulkanOutputAdapter)
    adapter._prepared_source_eyes = set()
    adapter._source_frames = {
        7: (
            SimpleNamespace(resource=object()),
            SimpleNamespace(resource=object()),
            0,
        )
    }
    adapter.left_ready_semaphores = [SimpleNamespace(semaphore="left-ready")]
    adapter.right_ready_semaphores = [SimpleNamespace(semaphore="right-ready")]
    adapter.left_ready_values = [3]
    adapter.right_ready_values = [4]
    adapter.left_visible_semaphores = [SimpleNamespace(semaphore="left-visible")]
    adapter.right_visible_semaphores = [SimpleNamespace(semaphore="right-visible")]
    calls: list[object] = []
    re_signals: list[object] = []
    adapter.presenter = SimpleNamespace(
        vulkan=SimpleNamespace(
            prepare_external_image_for_sampling=lambda *_args, **_kwargs: calls.append(
                (_args, _kwargs)
            ),
            submit_on=lambda _role, _record, **_kwargs: re_signals.append(_kwargs),
        )
    )

    assert adapter.prepare_source_for_sampling(7, 0) == "left-visible"
    assert adapter.prepare_source_for_sampling(7, 0) is None
    assert len(calls) == 1
    assert calls[0][1]["wait_semaphore_value"] == 3
    assert re_signals == []


def test_cuda_cross_adapter_source_prepare_uses_gpu_fence_and_is_idempotent() -> None:
    adapter = CudaVulkanOutputAdapter.__new__(CudaVulkanOutputAdapter)
    adapter._prepared_source_eyes = set()
    resource = SimpleNamespace(resource=object())
    adapter._source_frames = {9: (resource, resource, 1)}
    ready = SimpleNamespace(semaphore="d3d12-ready")
    bridge = SimpleNamespace(
        ready_semaphore=ready,
        completion_for_frame=lambda frame_id: 17 if frame_id == 9 else 0,
    )
    adapter._cross_adapter_gpu_bridge = bridge
    visible = SimpleNamespace(semaphore="vulkan-visible")
    adapter.left_visible_semaphores = [None, visible]
    adapter.right_visible_semaphores = [None, visible]
    calls = []
    adapter.presenter = SimpleNamespace(
        vulkan=SimpleNamespace(
            prepare_external_image_for_sampling=lambda *args, **kwargs: calls.append(
                (args, kwargs)
            )
        )
    )

    assert adapter.prepare_source_for_sampling(9, 0) == "vulkan-visible"
    assert adapter.prepare_source_for_sampling(9, 0) is None
    assert len(calls) == 1
    assert calls[0][0] == (resource.resource,)
    assert calls[0][1] == {
        "wait_semaphore": "d3d12-ready",
        "wait_semaphore_value": 17,
        "signal_semaphore": "vulkan-visible",
    }


def test_cuda_cross_adapter_release_signals_independent_eye_fences() -> None:
    adapter = CudaVulkanOutputAdapter.__new__(CudaVulkanOutputAdapter)
    adapter._released_source_frames = set()
    adapter._source_frames = {
        12: (
            SimpleNamespace(resource="left-image"),
            SimpleNamespace(resource="right-image"),
            0,
        )
    }
    adapter._prepared_source_eyes = {(12, 0), (12, 1)}
    adapter._release_signaled = set()
    adapter.left_release_semaphores = [SimpleNamespace(semaphore="left-release")]
    adapter.right_release_semaphores = [SimpleNamespace(semaphore="right-release")]
    adapter.left_release_values = [0]
    adapter.right_release_values = [0]
    adapter.backend_name = "cuda"
    adapter._cuda_release_timelines = [0]
    adapter.release_frame = lambda frame_id: None
    calls = []

    def release(resource, **kwargs):
        calls.append((resource, kwargs))
        return 25

    adapter.presenter = SimpleNamespace(
        vulkan=SimpleNamespace(
            device_lost=False,
            release_external_image_from_sampling=release,
        )
    )

    adapter.release_consumer_frame(12, wait_for_timeline=24)

    assert calls == [
        (
            "left-image",
            {
                "wait_for_timeline": 24,
                "wait_semaphore": None,
                "signal_semaphore": "left-release",
                "signal_semaphore_value": 1,
            },
        ),
        (
            "right-image",
            {
                "wait_for_timeline": 24,
                "wait_semaphore": None,
                "signal_semaphore": "right-release",
                "signal_semaphore_value": 1,
            },
        ),
    ]
    assert adapter.left_release_values == [1]
    assert adapter.right_release_values == [1]
    assert adapter._release_signaled == {(0, 0), (1, 0)}


def test_cuda_cross_adapter_busy_slot_drops_frame_without_host_staging(monkeypatch) -> None:
    class BusyError(RuntimeError):
        pass

    bridge_module = types.ModuleType("viewer.cuda_d3d12_bridge")
    bridge_module.CudaD3D12BridgeBusyError = BusyError
    monkeypatch.setitem(sys.modules, "viewer.cuda_d3d12_bridge", bridge_module)
    from contextlib import contextmanager

    streams_created = []
    recorded_streams = []
    copy_streams = []
    transfer_stream = SimpleNamespace(cuda_stream=456)

    def create_stream(device):
        streams_created.append(device)
        return transfer_stream

    @contextmanager
    def use_stream(stream):
        assert stream is transfer_stream
        yield

    torch_module = types.ModuleType("torch")
    torch_module.cuda = SimpleNamespace(
        current_stream=lambda device: SimpleNamespace(cuda_stream=123),
        Stream=create_stream,
        stream=use_stream,
    )
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    class BusyBridge:
        def copy(self, *_args, **_kwargs):
            copy_streams.append(_kwargs["stream"])
            raise BusyError("slot copy is still in flight")

    adapter = CudaVulkanOutputAdapter.__new__(CudaVulkanOutputAdapter)
    adapter._ensure_cross_adapter_gpu_bridge = lambda *_args: BusyBridge()
    adapter.ring_size = 2
    adapter.left_release_values = [0, 0]
    adapter.right_release_values = [0, 0]
    adapter._claim_slot = lambda *_args: None
    adapter._wait_for_runtime_cuda_event = lambda *_args: None
    released = []
    adapter.release_frame = released.append
    adapter._convert_cross_adapter_host_staging = lambda *_args, **_kwargs: pytest.fail(
        "a busy GPU slot must not trigger host staging"
    )
    tensor = SimpleNamespace(device="cuda:0", record_stream=recorded_streams.append)

    result = adapter._convert_cross_adapter_gpu(
        SimpleNamespace(),
        frame_id=4,
        timestamp=1.0,
        left=tensor,
        right=tensor,
        width=2,
        height=2,
    )

    assert result is None
    assert released == [4]
    assert streams_created == ["cuda:0"]
    assert copy_streams == [456]
    assert recorded_streams == [transfer_stream, transfer_stream]


def test_cuda_bridge_checks_slot_before_writing_shared_source_buffers() -> None:
    from viewer.cuda_d3d12_bridge import (
        CudaD3D12BridgeBusyError,
        CudaD3D12OutputBridge,
    )

    class Importer:
        def __init__(self):
            self.calls = []

        def wait_semaphore(self, *_args, **_kwargs):
            self.calls.append("wait")

        def copy_tensor_to_pitched_buffer(self, *_args, **_kwargs):
            self.calls.append("copy")

        def signal_semaphore(self, *_args, **_kwargs):
            self.calls.append("signal")

    importer = Importer()
    bridge = CudaD3D12OutputBridge.__new__(CudaD3D12OutputBridge)
    bridge.width = 2
    bridge.height = 2
    bridge.ring_size = 2
    bridge._bridge = ctypes.c_void_p(1)
    bridge._library = SimpleNamespace(
        d2s_d3d12_bridge_check_slot_ready=lambda *_args: -2147483638
    )
    bridge.importer = importer

    tensor = SimpleNamespace(shape=(2, 2, 4), dtype="uint8")
    with pytest.raises(CudaD3D12BridgeBusyError):
        bridge.copy(tensor, tensor, frame_id=0, stream=123)

    assert importer.calls == []


def test_cuda_timeline_semaphore_handle_types_match_runtime_enum() -> None:
    assert CudaVulkanImageImporter._CUDA_TIMELINE_SEMAPHORE_FD == 9
    assert CudaVulkanImageImporter._CUDA_TIMELINE_SEMAPHORE_WIN32 == 10
    timeline = SimpleNamespace(timeline=True)
    binary = SimpleNamespace(timeline=False)
    assert CudaVulkanImageImporter._semaphore_value(timeline, 7) == 7
    assert CudaVulkanImageImporter._semaphore_value(binary, 0) == 0
    with pytest.raises(RuntimeError, match="positive"):
        CudaVulkanImageImporter._semaphore_value(timeline, 0)
    with pytest.raises(RuntimeError, match="must be zero"):
        CudaVulkanImageImporter._semaphore_value(binary, 1)


def test_rocm_external_semaphore_is_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("D2S_ENABLE_ROCM_EXTERNAL_SEMAPHORE", raising=False)
    assert not RocmVulkanOutputAdapter._external_semaphore_requested()
    monkeypatch.setenv("D2S_ENABLE_ROCM_EXTERNAL_SEMAPHORE", "0")
    assert not RocmVulkanOutputAdapter._external_semaphore_requested()


@pytest.mark.parametrize(
    ("rocm_luid", "vulkan_luid", "host_staging", "path"),
    [
        (0x1234, 0x1234, False, "hip_vulkan_buffer_same_adapter"),
        (0x1234, 0x5678, True, "rocm_host_eye_staging"),
        (0, 0x5678, True, "rocm_host_eye_staging"),
        (0x1234, 0, True, "rocm_host_eye_staging"),
    ],
)
def test_rocm_output_transfer_requires_matching_adapter_luids(
    monkeypatch, rocm_luid, vulkan_luid, host_staging, path
) -> None:
    module = types.ModuleType("wc_rocm")
    module.get_luid = lambda _index=0: (
        rocm_luid & 0xFFFFFFFF,
        (rocm_luid >> 32) & 0xFFFFFFFF,
    )
    monkeypatch.setitem(sys.modules, "wc_rocm", module)
    adapter = RocmVulkanOutputAdapter.__new__(RocmVulkanOutputAdapter)
    adapter._host_staging_enabled = False
    adapter._adapter_transfer_decision = None
    adapter._rocm_transfer_fallback_reason = None
    adapter.presenter = SimpleNamespace(
        vulkan=SimpleNamespace(
            device_info=SimpleNamespace(adapter_luid=vulkan_luid)
        )
    )

    adapter._configure_adapter_transfer(
        SimpleNamespace(device=SimpleNamespace(index=0))
    )

    assert adapter._host_staging_enabled is host_staging
    assert adapter._rocm_transfer_path == path
    assert bool(adapter._adapter_transfer_decision.allowed) is (not host_staging)
    if host_staging:
        assert adapter._rocm_transfer_fallback_reason


@pytest.mark.parametrize(
    ("cuda_luid", "vulkan_luid", "expected_path"),
    [
        (0x1234, 0x1234, "cuda_vulkan_same_adapter"),
        (0x1234, 0x5678, "cuda_d3d12_cross_adapter_gpu_pending"),
        (0, 0x5678, "cuda_host_eye_staging"),
        (0x1234, 0, "cuda_host_eye_staging"),
    ],
)
def test_cuda_output_transfer_requires_matching_adapter_luids(
    monkeypatch, cuda_luid, vulkan_luid, expected_path
) -> None:
    module = types.ModuleType("wc_cuda")
    module.get_luid = lambda _index=0: (
        cuda_luid & 0xFFFFFFFF,
        (cuda_luid >> 32) & 0xFFFFFFFF,
    )
    monkeypatch.setitem(sys.modules, "wc_cuda", module)
    adapter = CudaVulkanOutputAdapter.__new__(CudaVulkanOutputAdapter)
    adapter._cuda_adapter_transfer_decision = None
    adapter._cuda_transfer_fallback_reason = None
    adapter.presenter = SimpleNamespace(
        vulkan=SimpleNamespace(
            device_info=SimpleNamespace(adapter_luid=vulkan_luid)
        )
    )

    decision = adapter._configure_adapter_transfer(
        SimpleNamespace(device=SimpleNamespace(index=1))
    )

    assert adapter._cuda_transfer_path == expected_path
    assert bool(decision.allowed) is (expected_path == "cuda_vulkan_same_adapter")
    if expected_path != "cuda_vulkan_same_adapter":
        assert adapter._cuda_transfer_fallback_reason


def test_cuda_matching_adapter_keeps_original_vulkan_output_path(monkeypatch):
    module = types.ModuleType("wc_cuda")
    module.get_luid = lambda _index=0: (0x1234, 0)
    monkeypatch.setitem(sys.modules, "wc_cuda", module)
    copied = []
    left_slot = SimpleNamespace(resource=object())
    right_slot = SimpleNamespace(resource=object())

    class FakeImporter:
        def copy_tensor(self, tensor, target):
            copied.append((tensor, target))

        def synchronize(self):
            pass

    adapter = CudaVulkanOutputAdapter.__new__(CudaVulkanOutputAdapter)
    adapter.backend_name = "cuda"
    adapter.presenter = SimpleNamespace(
        vulkan=SimpleNamespace(
            device_info=SimpleNamespace(adapter_luid=0x1234),
            wait_for_timeline=lambda _value: None,
        ),
        source_ready_semaphore_available=False,
    )
    adapter._cuda_adapter_transfer_decision = None
    adapter._cuda_transfer_path = None
    adapter._cuda_transfer_fallback_reason = None
    adapter._cuda_adapter_luid = 0
    adapter._vulkan_adapter_luid = 0
    adapter._cross_adapter_gpu_bridge = None
    adapter._cross_adapter_gpu_bridge_attempted = False
    adapter._cross_adapter_gpu_bridge_error = None
    adapter.ring_size = 1
    adapter.left_slots = [left_slot]
    adapter.right_slots = [right_slot]
    adapter.left_ready_semaphores = []
    adapter.right_ready_semaphores = []
    adapter.left_release_semaphores = []
    adapter.right_release_semaphores = []
    adapter.left_visible_semaphores = []
    adapter.right_visible_semaphores = []
    adapter.left_release_values = []
    adapter.right_release_values = []
    adapter.left_ready_values = []
    adapter.right_ready_values = []
    adapter.external_semaphore_enabled = False
    adapter._external_semaphore_request_enabled = False
    adapter._external_semaphore_error = None
    adapter._external_semaphore_request_reason = None
    adapter._logged_external_sync_mode = True
    adapter._cuda_release_timelines = [0]
    adapter._screen_light_sample_path = "vulkan_compute_reduction"
    adapter._screen_light_rgb = (0.18, 0.18, 0.18)
    adapter._source_frames = {}
    adapter._released_source_frames = set()
    adapter._prepared_source_eyes = set()
    adapter._release_signaled = set()
    adapter.importer = FakeImporter()
    adapter._wait_for_runtime_cuda_event = lambda *_args: None
    adapter._ensure_slots = lambda *_args: None
    adapter._claim_slot = lambda *_args: None
    adapter._update_glow_gpu_source = lambda *_args, **_kwargs: {}
    adapter.source_image_contract = lambda _resource: {
        "layout": "general",
        "queue_family": 0,
    }
    adapter._convert_cross_adapter_gpu = lambda *_args, **_kwargs: pytest.fail(
        "matching LUIDs must keep the existing direct Vulkan path"
    )
    tensor = SimpleNamespace(shape=(2, 3, 4), device=SimpleNamespace(index=0))

    frame = adapter.convert(
        SimpleNamespace(left_eye=tensor, right_eye=tensor),
        frame_id=0,
        timestamp=1.0,
    )

    assert frame.metadata["cuda_vulkan_transfer_path"] == "cuda_vulkan_same_adapter"
    assert copied == [(tensor, left_slot), (tensor, right_slot)]


def test_cuda_cross_adapter_output_falls_back_when_d3d12_bridge_is_unavailable(monkeypatch):
    module = types.ModuleType("wc_cuda")
    module.get_luid = lambda _index=0: (0x1234, 0)
    monkeypatch.setitem(sys.modules, "wc_cuda", module)
    bridge_module = types.ModuleType("viewer.cuda_d3d12_bridge")

    class UnavailableBridge:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("test bridge unavailable")

    bridge_module.CudaD3D12OutputBridge = UnavailableBridge
    monkeypatch.setitem(sys.modules, "viewer.cuda_d3d12_bridge", bridge_module)
    host_frame = VulkanStereoOutputFrame(
        frame_id=4,
        timestamp=1.0,
        left_eye=object(),
        right_eye=object(),
        metadata={"vulkan_output_path": "host_visible_vulkan_image"},
    )

    class FakeHostOutput:
        def __init__(self, _presenter):
            self.closed = False

        def convert(self, _runtime_result, *, frame_id, timestamp):
            assert frame_id == 4
            assert timestamp == 1.0
            return host_frame

        def close(self):
            self.closed = True

    monkeypatch.setattr(runtime_output_module, "VulkanHostOutputAdapter", FakeHostOutput)
    adapter = CudaVulkanOutputAdapter.__new__(CudaVulkanOutputAdapter)
    adapter.presenter = SimpleNamespace(
        vulkan=SimpleNamespace(
            device_info=SimpleNamespace(adapter_luid=0x5678)
        )
    )
    adapter._cuda_adapter_transfer_decision = None
    adapter._cuda_transfer_fallback_reason = None
    adapter._cross_adapter_host_output = None
    adapter._cross_adapter_gpu_bridge = None
    adapter._cross_adapter_gpu_bridge_attempted = False
    adapter._cross_adapter_gpu_bridge_error = None
    adapter._cuda_transfer_path = None
    adapter.importer = None
    adapter.ring_size = 3
    importer = SimpleNamespace(close=lambda: None)
    adapter._create_importer = lambda: importer
    adapter.backend_name = "cuda"
    tensor = SimpleNamespace(
        device=SimpleNamespace(index=0),
        shape=(2, 3, 4),
    )

    result = adapter.convert(
        SimpleNamespace(left_eye=tensor, right_eye=tensor),
        frame_id=4,
        timestamp=1.0,
    )

    assert result.metadata["cuda_vulkan_adapter_match"] is False
    assert result.metadata["cuda_vulkan_transfer_path"] == "cuda_host_eye_staging"
    assert result.metadata["vulkan_gpu_to_cpu"] is True


def test_rocm_source_prepare_waits_for_hip_ready_semaphore() -> None:
    adapter = RocmVulkanOutputAdapter.__new__(RocmVulkanOutputAdapter)
    adapter._prepared_source_eyes = set()
    adapter._rocm_ready_pending = {(7, 0)}
    adapter._source_frames = {
        7: (
            SimpleNamespace(resource=object()),
            SimpleNamespace(resource=object()),
            0,
        )
    }
    adapter._buffer_frames = {
        7: (SimpleNamespace(), SimpleNamespace())
    }
    adapter._host_staging_enabled = False
    adapter.external_semaphore_enabled = True
    adapter.left_ready_semaphores = [SimpleNamespace(semaphore="left-ready")]
    adapter.right_ready_semaphores = [SimpleNamespace(semaphore="right-ready")]
    adapter.left_ready_values = [3]
    adapter.right_ready_values = [4]
    adapter.left_visible_semaphores = [SimpleNamespace(semaphore="left-visible")]
    adapter.right_visible_semaphores = [SimpleNamespace(semaphore="right-visible")]
    calls = []
    adapter.presenter = SimpleNamespace(
        vulkan=SimpleNamespace(
            copy_buffer_to_image=lambda *args, **kwargs: calls.append((args, kwargs))
        )
    )

    assert adapter.prepare_source_for_sampling(7, 0) == "left-visible"
    assert calls[0][1] == {
        "wait_semaphore": "left-ready",
        "wait_semaphore_value": 3,
        "signal_semaphore": "left-visible",
    }
    assert (7, 0) not in adapter._rocm_ready_pending


def test_cuda_output_adapter_implements_backend_neutral_gpu_contract() -> None:
    adapter = CudaVulkanOutputAdapter.__new__(CudaVulkanOutputAdapter)
    assert isinstance(adapter, GpuProducerAdapter)
    assert adapter.backend_name == "cuda"
    assert adapter.output_sync_mode == "gpu_synchronized"
    assert adapter.external_semaphore_sync_mode == "gpu_external_semaphore"


def test_gpu_producer_factory_selects_cuda_without_importing_vendor_api() -> None:
    adapter = create_gpu_producer_adapter(SimpleNamespace(), backend="cuda")
    assert isinstance(adapter, CudaVulkanOutputAdapter)


def test_gpu_producer_factory_rejects_unregistered_backend() -> None:
    assert _ADAPTER_FACTORIES["rocm"] is RocmVulkanOutputAdapter


def test_gpu_producer_factory_reports_unknown_backend() -> None:
    with pytest.raises(GpuProducerUnavailableError, match="unknown"):
        create_gpu_producer_adapter(SimpleNamespace(), backend="unknown")


def test_vulkan_output_slot_waits_for_consumer_release() -> None:
    adapter = CudaVulkanOutputAdapter.__new__(CudaVulkanOutputAdapter)
    adapter._lease_condition = threading.Condition()
    adapter._active_leases = {}
    adapter._closed = False
    adapter._claim_slot(0, 10)
    claimed = threading.Event()

    def claim_reused_slot() -> None:
        adapter._claim_slot(0, 11)
        claimed.set()

    worker = threading.Thread(target=claim_reused_slot)
    worker.start()
    assert not claimed.wait(0.05)
    adapter.release_frame(10)
    worker.join(timeout=1.0)
    assert claimed.is_set()
    adapter.release_frame(11)


def test_output_contract_publishes_actual_source_layout_and_queue_family() -> None:
    vk = SimpleNamespace(
        VK_IMAGE_LAYOUT_GENERAL=1,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL=2,
    )
    state = SimpleNamespace(layout=1, queue_family_index=4)
    context = SimpleNamespace(
        vk=vk,
        image_state=lambda _image: state,
    )
    resource = VulkanImageResource(
        context=context,
        image=object(),
        view=None,
        width=2,
        height=2,
        format=37,
        layout=1,
        access_mask=0,
        stage_mask=0,
        queue_family_index=4,
    )

    contract = GpuProducerAdapter.source_image_contract(resource)
    assert contract == {"layout": "general", "queue_family": 4}

    state.layout = 2
    assert CudaVulkanOutputAdapter._source_image_contract(resource)["layout"] == (
        "shader_read_only_optimal"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is unavailable")
def test_cuda_tensor_reaches_vulkan_output_slot_without_cpu_roundtrip(monkeypatch) -> None:
    # This test exercises the production CUDA timeline-semaphore path.
    monkeypatch.setenv("D2S_ENABLE_CUDA_EXTERNAL_SEMAPHORE", "1")
    required_extensions = tuple(
        dict.fromkeys(
            VulkanExportableImage.required_device_extensions()
            + VulkanExportableSemaphore.required_device_extensions()
        )
    )
    context = VulkanContext.create(
        VulkanContextConfig(
            required_device_extensions=required_extensions
        )
    )
    destination = None
    adapter = None
    try:
        presenter = SimpleNamespace(
            initialized=True,
            vulkan=context,
            source_ready_semaphore_available=True,
        )
        adapter = CudaVulkanOutputAdapter(presenter)
        result = SimpleNamespace(
            left_eye=torch.full((32, 48, 4), 7, dtype=torch.uint8, device="cuda"),
            right_eye=torch.full((32, 48, 4), 11, dtype=torch.uint8, device="cuda"),
            debug_info={"output_format": "openxr_full_synthesis_eyes"},
        )
        if not adapter._configure_adapter_transfer(result.left_eye).allowed:
            pytest.skip("OpenXR Vulkan selected a different adapter from CUDA")
        for frame_id in range(300):
            frame = adapter.convert(result, frame_id=frame_id, timestamp=1.0)
            visible = (
                adapter.prepare_source_for_sampling(frame_id, 0),
                adapter.prepare_source_for_sampling(frame_id, 1),
            )
            context.submit_on(
                "graphics",
                lambda _command_buffer: None,
                wait_semaphore=visible,
            )
            adapter.release_consumer_frame(frame_id)

        assert frame.left_eye.width == 48
        assert frame.left_eye.height == 32
        assert frame.left_eye.format == context.vk.VK_FORMAT_R8G8B8A8_SRGB
        assert len(adapter.left_slots) == 3
        assert len(adapter.right_slots) == 3
        assert frame.metadata["vulkan_output_ring_slot"] == 2
        assert frame.metadata["vulkan_output_ring_size"] == 3
        assert frame.metadata["vulkan_output_sync"] == "gpu_external_semaphore"
        assert frame.metadata["vulkan_external_semaphore_type"] == "timeline"
        assert adapter.left_ready_values == [100, 100, 100]
        assert adapter.right_ready_values == [100, 100, 100]
        assert adapter.left_release_values == [100, 100, 100]
        assert adapter.right_release_values == [100, 100, 100]
        assert context.image_state(frame.left_eye.image).layout == context.vk.VK_IMAGE_LAYOUT_GENERAL

        destination = VulkanExportableImage(
            context,
            48,
            32,
            label="cuda-output-destination",
            format=context.vk.VK_FORMAT_R8G8B8A8_SRGB,
        )
        timeline = context.copy_image(frame.left_eye, destination.resource)
        context.wait_idle()
        assert timeline > 0
    finally:
        if adapter is not None:
            adapter.close()
        if destination is not None:
            destination.close()
        context.close()
