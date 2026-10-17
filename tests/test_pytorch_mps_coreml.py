"""CoreML provider plumbing tests (ported macOS behavior)."""

from types import SimpleNamespace

from stereo_runtime.depth_provider import (
    DepthProviderConfig,
    DepthProviderInfo,
    GenericAutoDepthProvider,
    create_depth_provider,
)
from stereo_runtime.providers.apple.pytorch_mps import (
    CoreMLEngine,
    GenericAutoDepthMpsProvider,
    create_pytorch_mps_provider,
    is_coreml_available,
)


def test_is_coreml_available_returns_bool() -> None:
    assert isinstance(is_coreml_available(), bool)


def test_native_coreml_startup_does_not_load_a_second_python_model(monkeypatch) -> None:
    from stereo_runtime.providers.apple.native import NativeCoreMLIOBridge

    calls = []
    bridge = SimpleNamespace(predict=lambda buffer, frame_id: (buffer, frame_id))
    monkeypatch.setattr("stereo_runtime.providers.apple.pytorch_mps.sys.platform", "darwin")
    monkeypatch.setenv("D2S_COREML_NATIVE_IO", "1")
    monkeypatch.setattr(NativeCoreMLIOBridge, "try_create", lambda *_: (bridge, None))
    monkeypatch.setattr(CoreMLEngine, "_load_python_model", lambda _: calls.append("load"))

    engine = CoreMLEngine("cached.mlpackage", "cpu")
    engine.enable_native_io("cached.mlpackage", 336, 196)

    assert engine.predict_native("pixel_buffer", 7) == ("pixel_buffer", 7)
    assert calls == []


def test_python_coreml_fallback_loads_once_on_first_prediction(monkeypatch) -> None:
    import numpy as np
    import torch

    calls = []
    model = SimpleNamespace(predict=lambda _: {"depth": np.ones((1, 2, 2), dtype=np.float32)})

    def load(_engine):
        calls.append("load")
        return model

    monkeypatch.setattr(CoreMLEngine, "_load_python_model", load)
    engine = CoreMLEngine("cached.mlpackage", "cpu")
    assert calls == []
    for _ in range(2):
        result = engine(torch.zeros(1, 3, 2, 2))
        assert bool((result.predicted_depth == 1).all())
    assert calls == ["load"]


def test_generic_provider_accepts_coreml_flags() -> None:
    provider = GenericAutoDepthMpsProvider(
        model_id="xingyang1/Distill-Any-Depth-Small-hf",
        device="mps",
        cache_dir="/tmp/d2s-models",
        local_files_only=True,
        use_coreml=True,
        recompile_coreml=False,
    )
    assert provider.use_coreml is True
    assert provider.recompile_coreml is False
    assert CoreMLEngine.__module__ == "stereo_runtime.providers.apple.pytorch_mps"


def test_create_pytorch_mps_provider_forwards_coreml_flags() -> None:
    provider = create_pytorch_mps_provider(
        model_id="xingyang1/Distill-Any-Depth-Small-hf",
        device="mps",
        cache_dir="/tmp/d2s-models",
        local_files_only=True,
        depth_upsample="bilinear",
        depth_upsample_edge_strength=0.35,
        use_coreml=True,
        recompile_coreml=True,
    )
    assert isinstance(provider, GenericAutoDepthMpsProvider)
    assert provider.use_coreml is True
    assert provider.recompile_coreml is True


def test_coreml_load_does_not_preload_transformers(monkeypatch) -> None:
    """A cached CoreML model must not load the PyTorch model at startup."""
    import stereo_runtime.providers.apple.pytorch_mps as mps_module

    monkeypatch.setattr(mps_module.sys, "platform", "darwin")

    def fail_transformers_load(_provider):
        raise AssertionError("CoreML startup must not load Transformers")

    monkeypatch.setattr(
        GenericAutoDepthProvider, "load", fail_transformers_load
    )
    provider = create_pytorch_mps_provider(
        model_id="xingyang1/Distill-Any-Depth-Small-hf",
        device="cpu",
        cache_dir="/tmp/d2s-models",
        local_files_only=True,
        use_coreml=True,
    )

    assert isinstance(provider, GenericAutoDepthMpsProvider)
    assert provider.load() is None


def test_depth_provider_config_carries_coreml_fields() -> None:
    cfg = DepthProviderConfig(
        model_id="xingyang1/Distill-Any-Depth-Small-hf",
        device="mps",
        cache_dir="/tmp/d2s-models",
        local_files_only=True,
        use_coreml=True,
        recompile_coreml=False,
    )
    assert cfg.use_coreml is True
    assert cfg.recompile_coreml is False


def test_create_depth_provider_mps_backend_returns_mps_provider() -> None:
    cfg = DepthProviderConfig(
        model_id="xingyang1/Distill-Any-Depth-Small-hf",
        device="mps",
        cache_dir="/tmp/d2s-models",
        local_files_only=True,
        use_coreml=False,
    )
    provider = create_depth_provider(cfg)
    assert isinstance(provider, GenericAutoDepthMpsProvider)
    assert provider.use_coreml is False


def test_coreml_provider_info_reports_active_backend_without_fallback():
    provider = object.__new__(GenericAutoDepthMpsProvider)
    provider.info = DepthProviderInfo(
        provider="test",
        model_name="test",
        model_id="test",
        depth_resolution=336,
        cache_dir="/tmp",
    )

    provider._set_coreml_info()

    assert provider.info.depth_backend == "coreml"
    assert provider.info.runtime == "coreml"
    assert provider.info.execution_provider == "Apple Core ML"
    assert provider.info.fallback_reason is None


def test_coreml_engine_sanitizes_nonfinite_output():
    import numpy as np
    import torch

    engine = object.__new__(CoreMLEngine)
    engine.device = torch.device("mps")
    engine.model = SimpleNamespace(
        predict=lambda _inputs: {
            "depth": np.array([[[np.nan, np.inf], [-np.inf, 0.5]]], dtype=np.float32)
        }
    )

    output = engine(torch.zeros(1, 3, 2, 2))

    assert output.nonfinite_count == 3
    assert output.finite_depth is True
    assert bool(torch.isfinite(output.predicted_depth).all())
