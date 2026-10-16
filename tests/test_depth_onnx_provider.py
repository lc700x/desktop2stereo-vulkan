import sys
import threading
from pathlib import Path

from path_config import APP_ROOT

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from stereo_runtime.depth_provider import (
    DepthProviderConfig,
    DepthProviderInfo,
    DistillAnyDepthBase518,
    GenericAutoDepthProvider,
    TorchDepthProvider,
    create_depth_provider,
    estimate_depth,
    _normalize_depth,
)
from stereo_runtime.depth_onnx_provider import DistillPreprocessor, ModelOnnxPreprocessor, _preprocess_distill_rgb, estimate_depth_onnx_cuda
from stereo_runtime.providers.nvidia.tensorrt_ort import estimate_depth_nvidia_chain


class FakeTorchProvider:
    def __init__(self, **kwargs):
        self.info = DepthProviderInfo(
            provider="fake",
            model_name="Distill-Any-Depth-Base",
            model_id="fake",
            depth_resolution=518,
            cache_dir=str(kwargs.get("cache_dir") or ""),
            load_mode="test",
            depth_backend="pytorch_cpu",
            runtime="fake",
        )

    def predict(self, rgb):
        return torch.zeros(rgb.shape[0], 1, rgb.shape[-2], rgb.shape[-1])


def test_normalize_depth_uses_beta_percentile_bounds():
    values = torch.full((100,), 0.5)
    values[0] = 0.0
    values[1] = 0.1
    values[2] = 0.2
    values[97] = 0.8
    values[98] = 0.9
    values[99] = 1.0
    values = values.view(1, 1, 10, 10)

    normalized = _normalize_depth(values)

    mid = normalized[0, 0, 5, 0]
    assert 0.45 < float(mid) < 0.55
    assert float(normalized.min()) == 0.0
    assert float(normalized.max()) == 1.0


def test_nvidia_provider_falls_back_when_onnx_missing(monkeypatch, tmp_path):
    import stereo_runtime.providers.nvidia.tensorrt_ort as provider_module

    monkeypatch.setattr(provider_module, "TorchDepthProvider", FakeTorchProvider)
    rgb = torch.zeros(1, 3, 8, 8)
    missing = tmp_path / "missing.onnx"

    depth, info = estimate_depth_nvidia_chain(
        rgb,
        device="cpu",
        onnx_path=missing,
        prefer_onnx=True,
        allow_pytorch_fallback=True,
        local_files_only=True,
    )

    assert depth.shape == (1, 1, 8, 8)
    assert info.depth_backend == "pytorch_cpu"
    assert info.onnx_path == str(missing)
    assert "FileNotFoundError" in (info.fallback_reason or "")


def test_nvidia_provider_requires_tensorrt_when_requested(tmp_path):
    rgb = torch.zeros(1, 3, 8, 8)
    missing = tmp_path / "missing.onnx"

    try:
        estimate_depth_nvidia_chain(
            rgb,
            device="cpu",
            onnx_path=missing,
            require_tensorrt=True,
        )
    except RuntimeError as exc:
        assert "tensorrt" in str(exc)
    else:
        raise AssertionError("expected TensorRT requirement to fail")


def test_onnx_cuda_provider_falls_back_when_onnx_missing(monkeypatch, tmp_path):
    import stereo_runtime.depth_onnx_provider as provider_module

    monkeypatch.setattr(provider_module, "TorchDepthProvider", FakeTorchProvider)
    rgb = torch.zeros(1, 3, 8, 8)
    missing = tmp_path / "missing.onnx"

    depth, info = estimate_depth_onnx_cuda(
        rgb,
        device="cpu",
        onnx_path=missing,
        allow_pytorch_fallback=True,
        local_files_only=True,
    )

    assert depth.shape == (1, 1, 8, 8)
    assert info.depth_backend == "pytorch_cpu"
    assert info.onnx_path == str(missing)
    assert "FileNotFoundError" in (info.fallback_reason or "")


def test_create_depth_provider_falls_back_to_pytorch_when_ort_missing(monkeypatch, tmp_path):
    import stereo_runtime.depth_provider as provider_module

    monkeypatch.setattr(provider_module, "_onnxruntime_available", lambda: False)

    provider = create_depth_provider(
        DepthProviderConfig(
            backend="onnx_cuda",
            device="cpu",
            cache_dir=tmp_path,
            local_files_only=True,
        )
    )

    assert isinstance(provider, DistillAnyDepthBase518)
    assert isinstance(provider, TorchDepthProvider)
    assert provider.info.depth_backend == "pytorch_cpu"


def test_create_depth_provider_requires_ort_when_fallback_disabled(monkeypatch, tmp_path):
    import stereo_runtime.depth_provider as provider_module

    monkeypatch.setattr(provider_module, "_onnxruntime_available", lambda: False)

    try:
        create_depth_provider(
            DepthProviderConfig(
                backend="onnx_cuda",
                device="cpu",
                cache_dir=tmp_path,
                local_files_only=True,
                allow_pytorch_fallback=False,
            )
        )
    except RuntimeError as exc:
        assert "ONNX Runtime is not installed" in str(exc)
    else:
        raise AssertionError("expected missing ONNX Runtime to fail when fallback is disabled")

def test_create_depth_provider_supports_persistent_pytorch_provider(tmp_path):
    provider = create_depth_provider(
        DepthProviderConfig(
            backend="pytorch_cuda",
            device="cpu",
            cache_dir=tmp_path,
            local_files_only=True,
        )
    )

    assert isinstance(provider, DistillAnyDepthBase518)
    assert isinstance(provider, TorchDepthProvider)
    assert provider.info.depth_backend == "pytorch_cpu"


def test_create_depth_provider_supports_generic_pytorch_provider(tmp_path):
    provider = create_depth_provider(
        DepthProviderConfig(
            backend="pytorch",
            model_id="Intel/dpt-large",
            model_name="dpt-large",
            device="cpu",
            cache_dir=tmp_path,
            local_files_only=True,
        )
    )

    assert isinstance(provider, GenericAutoDepthProvider)
    assert provider.info.model_id == "Intel/dpt-large"
    assert provider.info.runtime == "transformers-generic"


def test_generic_provider_predict_profile_with_fake_transformers_model(monkeypatch, tmp_path):
    class Output:
        predicted_depth = torch.arange(16, dtype=torch.float32).view(1, 4, 4)

    class FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, pixel_values):
            return Output()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return FakeModel()

    import transformers

    monkeypatch.setattr(transformers, "AutoModelForDepthEstimation", FakeAutoModel)
    provider = GenericAutoDepthProvider(
        model_id="Intel/dpt-large",
        model_name="dpt-large",
        device="cpu",
        cache_dir=tmp_path,
        local_files_only=True,
        depth_resolution=8,
        patch_size=1,
    )

    result = provider.predict_profile(torch.zeros(1, 3, 8, 8))

    assert result.depth.shape == (1, 1, 8, 8)
    assert float(result.depth.min()) >= 0.0
    assert float(result.depth.max()) <= 1.0


def test_create_depth_provider_supports_native_tensorrt(monkeypatch, tmp_path):
    from stereo_runtime.providers.nvidia.tensorrt_native import NativeTensorRtDepthProvider

    class Paths:
        trt_fp16_path = tmp_path / "model.trt"

        def onnx_path_for_dtype(self, dtype_name):
            return tmp_path / "model.onnx"

        def trt_path_for_dtype(self, dtype_name):
            return self.trt_fp16_path

    class Artifacts:
        selected_onnx_path = tmp_path / "model.onnx"
        paths = Paths()

    import stereo_runtime.providers.nvidia.tensorrt_native as native_module

    monkeypatch.setattr(native_module, "_prepare_accelerated_artifacts", lambda *args, **kwargs: Artifacts())
    engine_path = tmp_path / "model.trt"
    provider = create_depth_provider(
        DepthProviderConfig(
            backend="tensorrt_native",
            device="cuda",
            cache_dir=tmp_path,
            build_engine=True,
            depth_resolution=336,
        )
    )

    assert isinstance(provider, NativeTensorRtDepthProvider)
    assert provider.info.depth_backend == "tensorrt_native"
    assert provider.onnx_path.name == "model_fp16_294x518.onnx"
    assert provider.engine_path.name == "model_fp16_294x518.trt"
    assert provider.build_engine is True
    assert provider.info.depth_resolution == 336
    assert provider._preprocessor.input_size(2160, 3840) == (196, 336)

    provider._ensure_artifacts_for_input(768, 1024)
    assert provider.onnx_path == tmp_path / "model.onnx"
    assert provider.engine_path == engine_path


def test_native_tensorrt_infers_large_metadata_from_model_path(tmp_path):
    from stereo_runtime.providers.nvidia.tensorrt_native import NativeTensorRtDepthProvider

    model_dir = tmp_path / "models--xingyang1--Distill-Any-Depth-Large-hf"
    onnx_path = model_dir / "model_fp16_294x518.onnx"
    engine_path = model_dir / "model_fp16_294x518.trt"
    provider = NativeTensorRtDepthProvider(
        device="cuda",
        onnx_path=onnx_path,
        engine_path=engine_path,
    )

    assert provider.info.model_id == "xingyang1/Distill-Any-Depth-Large-hf"
    assert provider.info.model_name == "Distill-Any-Depth-Large"


def test_native_tensorrt_keeps_explicit_metadata(tmp_path):
    from stereo_runtime.providers.nvidia.tensorrt_native import NativeTensorRtDepthProvider

    model_dir = tmp_path / "models--xingyang1--Distill-Any-Depth-Large-hf"
    provider = NativeTensorRtDepthProvider(
        device="cuda",
        onnx_path=model_dir / "model_fp16_294x518.onnx",
        engine_path=model_dir / "model_fp16_294x518.trt",
        model_id="custom/model",
        model_name="Custom Model",
    )

    assert provider.info.model_id == "custom/model"
    assert provider.info.model_name == "Custom Model"


def test_estimate_depth_uses_configured_provider(monkeypatch):
    class Provider:
        info = DepthProviderInfo(
            provider="fake",
            model_name="fake",
            model_id="fake",
            depth_resolution=518,
            cache_dir="",
            load_mode="test",
            depth_backend="fake",
            runtime="test",
        )

        def predict(self, rgb):
            return torch.ones(rgb.shape[0], 1, rgb.shape[-2], rgb.shape[-1])

    import stereo_runtime.depth_provider as provider_module

    monkeypatch.setattr(provider_module, "create_depth_provider", lambda config=None: Provider())
    depth, info = estimate_depth(torch.zeros(1, 3, 4, 5), {"backend": "fake"})

    assert depth.shape == (1, 1, 4, 5)
    assert info.depth_backend == "fake"


def test_distill_preprocessor_matches_reference():
    rgb = torch.linspace(0, 1, steps=3 * 12 * 16, dtype=torch.float32).view(1, 3, 12, 16)
    device = torch.device("cpu")
    dtype = torch.float32

    expected = _preprocess_distill_rgb(rgb, device=device, dtype=dtype)
    actual = DistillPreprocessor(device=device, dtype=dtype)(rgb)

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("explicit_size", [False, True])
def test_onnx_preprocessor_reuses_normalized_capture_rgb(explicit_size):
    rgb = torch.rand(1, 3, 12, 16)
    preprocessor = DistillPreprocessor(device=torch.device("cpu"), dtype=torch.float32)
    apply = (lambda x: preprocessor.prepare(x, height=14, width=28)) if explicit_size else preprocessor
    expected = apply(rgb)
    rgb._d2s_preprocess_backend = "triton_bgr_resize_norm"

    assert preprocessor._prepare_rgb(rgb) is rgb
    assert torch.equal(apply(rgb), expected)


def test_onnx_preprocessor_clamps_unmarked_input():
    rgb = torch.tensor([-1.0, 0.5, 2.0]).view(1, 3, 1, 1)
    preprocessor = DistillPreprocessor(device=torch.device("cpu"), dtype=torch.float32)
    assert torch.equal(preprocessor._prepare_rgb(rgb), rgb.clamp(0, 1))


def test_infinidepth_onnx_preprocessor_uses_patch_16_without_normalization():
    rgb = torch.linspace(0, 1, steps=3 * 12 * 18, dtype=torch.float32).view(1, 3, 12, 18)
    preprocessor = ModelOnnxPreprocessor(
        model_id="lc700x/InfiniDepth-Base",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    actual = preprocessor(rgb)

    assert actual.shape[-2] % 16 == 0
    assert actual.shape[-1] % 16 == 0
    assert float(actual.min()) >= 0.0
    assert float(actual.max()) <= 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/ROCm is unavailable")
def test_distill_preprocessor_uses_antialiased_bilinear_on_cuda():
    import torch.nn.functional as F

    rgb = torch.rand((1, 3, 64, 80), device="cuda")
    preprocessor = ModelOnnxPreprocessor(
        model_id="lc700x/Distill-Any-Depth-Small-hf",
        device=torch.device("cuda"),
        dtype=torch.float32,
        fixed_input_size=(16, 16),
    )

    actual = preprocessor(rgb)
    expected = F.interpolate(
        rgb, size=(16, 16), mode="bilinear", align_corners=False, antialias=True
    )
    mean = torch.tensor((0.485, 0.456, 0.406), device="cuda").view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device="cuda").view(1, 3, 1, 1)

    assert torch.allclose(actual, (expected - mean) / std, atol=1e-6, rtol=0.0)

def test_onnx_provider_defers_load_until_frame_artifact_size_is_known(tmp_path):
    from stereo_runtime.depth_onnx_provider import OnnxCudaDepthProvider

    provider = OnnxCudaDepthProvider(
        device="cpu",
        cache_dir=tmp_path,
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        model_name="Distill-Any-Depth-Base",
        onnx_dtype="fp32",
    )

    assert provider.load() is None
    assert provider.onnx_path.name.startswith("model_fp16_")
    assert provider._session is None


def test_onnx_provider_prepares_artifact_for_first_frame_shape(monkeypatch, tmp_path):
    from stereo_runtime.depth_onnx_provider import OnnxCudaDepthProvider

    calls = {}
    selected = tmp_path / "models--lc700x--Distill-Any-Depth-Base-hf" / "model_fp16_392x518.onnx"

    class Artifacts:
        selected_onnx_path = selected

    def fake_prepare(*args, **kwargs):
        calls.update(kwargs)
        selected.parent.mkdir(parents=True, exist_ok=True)
        selected.write_bytes(b"onnx")
        return Artifacts()

    import stereo_runtime.depth_onnx_provider as provider_module

    monkeypatch.setattr(provider_module, "_prepare_accelerated_artifacts", fake_prepare)
    provider = OnnxCudaDepthProvider(
        device="cpu",
        cache_dir=tmp_path,
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        model_name="Distill-Any-Depth-Base",
    )

    provider._ensure_artifacts_for_input(768, 1024)

    assert calls["input_size"] == (392, 518)
    assert provider.onnx_path == selected
    assert provider._preprocessor.fixed_input_size == (392, 518)


def test_distill_preprocessor_can_use_fixed_tensorrt_input_size():
    rgb = torch.zeros(1, 3, 2160, 1920)
    preprocessor = DistillPreprocessor(
        device=torch.device("cpu"),
        dtype=torch.float32,
        fixed_input_size=(294, 518),
    )

    tensor = preprocessor(rgb)

    assert tensor.shape == (1, 3, 294, 518)
    assert preprocessor.input_size(2160, 1920) == (294, 518)


def test_native_tensorrt_defers_load_until_frame_artifact_size_is_known(tmp_path):
    from stereo_runtime.providers.nvidia.tensorrt_native import NativeTensorRtDepthProvider

    provider = NativeTensorRtDepthProvider(
        device="cuda",
        cache_dir=tmp_path,
        onnx_dtype="fp32",
    )

    assert provider.load() is None
    assert provider.onnx_path.name.startswith("model_fp16_")
    assert provider._engine is None


def test_native_tensorrt_preserves_requested_execution_slots(tmp_path):
    from stereo_runtime.providers.nvidia.tensorrt_native import NativeTensorRtDepthProvider

    provider = NativeTensorRtDepthProvider(
        device="cuda",
        cache_dir=tmp_path,
        execution_slot_count=3,
    )

    assert provider.execution_slot_count == 3


def test_native_tensorrt_load_prints_compact_engine_path(monkeypatch, tmp_path, capsys):
    import stereo_runtime.providers.nvidia.tensorrt_native as native_module

    class FakeEngine:
        input_image_size = (294, 518)

        def __init__(self, engine_path, *, device, dtype, execution_slot_count):
            self.engine_path = engine_path

    engine_path = tmp_path / "models--lc700x--Distill-Any-Depth-Base-hf" / "model.trt"
    onnx_path = tmp_path / "model.onnx"
    engine_path.parent.mkdir()
    engine_path.write_bytes(b"trt")
    provider = native_module.NativeTensorRtDepthProvider(
        device="cuda",
        onnx_path=onnx_path,
        engine_path=engine_path,
    )

    monkeypatch.setattr(native_module, "ensure_tensorrt_dll_path", lambda: [tmp_path / "dlls"])
    monkeypatch.setattr(native_module, "NativeTensorRtEngine", FakeEngine)

    provider.load()

    output = capsys.readouterr().out.strip()
    compact_path = Path(engine_path.parent.name) / engine_path.name
    assert output == f"[TensorRT] native provider loaded: engine={compact_path}"
    assert str(tmp_path) not in output
    assert " onnx=" not in output
    assert " dll_dirs=" not in output


def test_native_tensorrt_provider_uses_engine_static_input_size(monkeypatch, tmp_path):
    import stereo_runtime.providers.nvidia.tensorrt_native as native_module

    captured = {}

    class FakeEngine:
        input_image_size = (294, 518)

        def __call__(self, tensor):
            captured["tensor_shape"] = tuple(tensor.shape)
            return torch.zeros(1, 1, 294, 518, dtype=torch.float32)

    class FakePreprocessor:
        fixed_input_size = None

        def __call__(self, rgb):
            assert self.fixed_input_size == (294, 518)
            return torch.zeros(1, 3, *self.fixed_input_size, dtype=torch.float32)

    provider = native_module.NativeTensorRtDepthProvider(
        device="cuda",
        cache_dir=tmp_path,
        engine_path=tmp_path / "model.trt",
    )
    provider._engine = FakeEngine()
    provider._preprocessor = FakePreprocessor()
    provider._artifact_input_size = (294, 518)
    monkeypatch.setattr(provider, "_ensure_artifacts_for_input", lambda height, width: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    result = provider.predict_profile(torch.zeros(1, 3, 2160, 1920))

    assert captured["tensor_shape"] == (1, 3, 294, 518)
    assert result.depth.shape == (1, 1, 2160, 1920)


def test_native_tensorrt_provider_reports_execution_slot(monkeypatch, tmp_path):
    import stereo_runtime.providers.nvidia.tensorrt_native as native_module

    calls = []

    class FakeEngine:
        input_image_size = (2, 2)
        pipeline_slot_count = 2

        def run_with_slot(self, tensor, *, synchronize):
            calls.append(("run", synchronize, tuple(tensor.shape)))
            return torch.zeros(1, 1, 2, 2, dtype=torch.float32), 1

        def slot_acquire_wait_ms(self, slot_index):
            assert slot_index == 1
            return 7.5

        def mark_slot_available_after_current_stream(self, slot_index):
            calls.append(("release", slot_index))

    class FakePreprocessor:
        fixed_input_size = (2, 2)

        def __call__(self, rgb):
            return torch.zeros(1, 3, 2, 2, dtype=torch.float32)

    provider = native_module.NativeTensorRtDepthProvider(
        device="cuda",
        cache_dir=tmp_path,
        engine_path=tmp_path / "model.trt",
    )
    provider._engine = FakeEngine()
    provider._preprocessor = FakePreprocessor()
    provider._artifact_input_size = (2, 2)
    monkeypatch.setattr(provider, "_ensure_artifacts_for_input", lambda height, width: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    result = provider.predict_profile(torch.zeros(1, 3, 4, 4))

    assert result.execution_slot == 1
    assert result.execution_slot_count == 2
    assert result.slot_wait_ms == 7.5
    assert calls == [("run", False, (1, 3, 2, 2)), ("release", 1)]


def test_native_tensorrt_engine_skips_slot_with_unfinished_event():
    from stereo_runtime.providers.nvidia.tensorrt_native import (
        NativeTensorRtEngine,
        _NativeTensorRtExecutionSlot,
    )

    class Event:
        def query(self):
            return False

    engine = object.__new__(NativeTensorRtEngine)
    engine._slots = [
        _NativeTensorRtExecutionSlot(0, object(), available_event=Event()),
        _NativeTensorRtExecutionSlot(1, object()),
    ]
    engine._next_slot_index = 0

    slot = engine._acquire_slot()

    assert slot.index == 1
    assert engine._next_slot_index == 0


def test_native_tensorrt_engine_reserves_slot_atomically():
    from stereo_runtime.providers.nvidia.tensorrt_native import (
        NativeTensorRtEngine,
        _NativeTensorRtExecutionSlot,
    )

    engine = object.__new__(NativeTensorRtEngine)
    engine._slots = [_NativeTensorRtExecutionSlot(0, object())]
    engine._next_slot_index = 0
    engine._slot_lock = threading.Lock()

    slot = engine._acquire_slot()

    assert slot.reserved is True
    assert engine._slots[0].reserved is True


def test_native_tensorrt_engine_records_slot_wait(monkeypatch):
    import stereo_runtime.providers.nvidia.tensorrt_native as native_module
    from stereo_runtime.providers.nvidia.tensorrt_native import (
        NativeTensorRtEngine,
        _NativeTensorRtExecutionSlot,
    )

    class Event:
        def __init__(self):
            self.ready = False

        def query(self):
            return self.ready

        def synchronize(self):
            self.ready = True

    event = Event()
    engine = object.__new__(NativeTensorRtEngine)
    engine._slots = [_NativeTensorRtExecutionSlot(0, object(), available_event=event)]
    engine._next_slot_index = 0
    engine._slot_lock = threading.Lock()
    ticks = iter((10.0, 10.025))
    monkeypatch.setattr(native_module.time, "perf_counter", lambda: next(ticks))

    slot = engine._acquire_slot()

    assert abs(slot.last_acquire_wait_ms - 25.0) < 0.001
    assert abs(engine.slot_acquire_wait_ms(0) - 25.0) < 0.001


def test_native_tensorrt_engine_uses_independent_runtime_and_engine_per_slot(monkeypatch, tmp_path):
    from stereo_runtime.providers.nvidia.tensorrt_native import NativeTensorRtEngine

    created_runtimes = []

    class FakeLogger:
        ERROR = 1

        def __init__(self, level):
            self.level = level

    class FakeContext:
        pass

    class FakeEngine:
        num_io_tensors = 2

        def __init__(self, runtime_index):
            self.runtime_index = runtime_index

        def create_execution_context(self):
            return FakeContext()

        def get_tensor_name(self, index):
            return ("input", "output")[index]

        def get_tensor_mode(self, name):
            return "input" if name == "input" else "output"

        def get_tensor_shape(self, name):
            return (1, 3, 294, 518)

    class FakeRuntime:
        def __init__(self, logger):
            self.index = len(created_runtimes)
            created_runtimes.append(self)

        def deserialize_cuda_engine(self, serialized):
            assert serialized == b"trt"
            return FakeEngine(self.index)

    class FakeTrt:
        Logger = FakeLogger
        Runtime = FakeRuntime

        class TensorIOMode:
            INPUT = "input"

    engine_path = tmp_path / "model.trt"
    engine_path.write_bytes(b"trt")
    monkeypatch.setitem(sys.modules, "tensorrt", FakeTrt)
    monkeypatch.setattr(torch.cuda, "Stream", lambda *, device: object())

    engine = NativeTensorRtEngine(engine_path, execution_slot_count=3)

    assert engine.pipeline_slot_count == 3
    assert len({id(slot.runtime) for slot in engine._slots}) == 3
    assert len({id(slot.engine) for slot in engine._slots}) == 3
    assert [slot.engine.runtime_index for slot in engine._slots] == [0, 1, 2]


def test_native_tensorrt_engine_falls_back_when_extra_engine_is_unavailable(monkeypatch, tmp_path, capsys):
    from stereo_runtime.providers.nvidia.tensorrt_native import NativeTensorRtEngine

    class FakeLogger:
        ERROR = 1

        def __init__(self, level):
            self.level = level

    class FakeEngine:
        num_io_tensors = 2

        def create_execution_context(self):
            return object()

        def get_tensor_name(self, index):
            return ("input", "output")[index]

        def get_tensor_mode(self, name):
            return "input" if name == "input" else "output"

        def get_tensor_shape(self, name):
            return (1, 3, 294, 518)

    class FakeRuntime:
        calls = 0

        def __init__(self, logger):
            pass

        def deserialize_cuda_engine(self, serialized):
            FakeRuntime.calls += 1
            return FakeEngine() if FakeRuntime.calls == 1 else None

    class FakeTrt:
        Logger = FakeLogger
        Runtime = FakeRuntime

        class TensorIOMode:
            INPUT = "input"

    engine_path = tmp_path / "model.trt"
    engine_path.write_bytes(b"trt")
    monkeypatch.setitem(sys.modules, "tensorrt", FakeTrt)
    monkeypatch.setattr(torch.cuda, "Stream", lambda *, device: object())

    engine = NativeTensorRtEngine(engine_path, execution_slot_count=2)

    assert engine.pipeline_slot_count == 1
    assert "execution slot 2/2 unavailable; using 1 slot(s)" in capsys.readouterr().out


def test_native_tensorrt_engine_close_waits_for_slot_event():
    from stereo_runtime.providers.nvidia.tensorrt_native import (
        NativeTensorRtEngine,
        _NativeTensorRtExecutionSlot,
    )

    calls = []

    class Event:
        def synchronize(self):
            calls.append("sync")

    engine = object.__new__(NativeTensorRtEngine)
    engine._slots = [
        _NativeTensorRtExecutionSlot(0, object(), available_event=Event()),
    ]
    engine.context = object()
    engine.engine = object()
    engine.runtime = object()

    engine.close()

    assert calls == ["sync"]
    assert engine._slots == []
    assert engine.context is None


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="NVIDIA CUDA is required")
def test_native_tensorrt_cuda_graph_replays_changed_input(tmp_path):
    from stereo_runtime.providers.nvidia.tensorrt_native import NativeTensorRtEngine, ensure_tensorrt_dll_path

    ensure_tensorrt_dll_path()
    trt = pytest.importorskip("tensorrt")
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    input_tensor = network.add_input("input", trt.float32, (1, 3, 4, 6))
    output = network.add_activation(input_tensor, trt.ActivationType.RELU).get_output(0)
    output.name = "predicted_depth"
    network.mark_output(output)
    serialized = builder.build_serialized_network(network, builder.create_builder_config())
    assert serialized is not None
    engine_path = tmp_path / "relu.trt"
    engine_path.write_bytes(serialized)
    engine = NativeTensorRtEngine(engine_path, dtype=torch.float32)
    try:
        for offset in (-1.0, 2.0):
            tensor = torch.linspace(-2, 2, 72, device="cuda").reshape(1, 3, 4, 6) + offset
            output, slot = engine.run_graph_with_slot(tensor)
            actual = output.clone()
            engine.mark_slot_available_after_current_stream(slot)
            torch.cuda.synchronize()
            assert torch.equal(actual, tensor.relu())
    finally:
        engine.close()
