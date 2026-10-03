import sys
import types
from types import SimpleNamespace

import pytest
import torch

from stereo_runtime.depth_provider import DepthProviderConfig, create_depth_provider
from stereo_runtime.providers.amd import (
    GenericTorchRocmDepthProvider,
    MIGraphXDepthProvider,
    TorchRocmDepthProvider,
)
import stereo_runtime.providers.amd.migraphx as migraphx_provider


@pytest.mark.parametrize("profile_sync", [False, True])
def test_migraphx_factory_preserves_profile_sync(monkeypatch, profile_sync):
    provider = SimpleNamespace()
    monkeypatch.setattr(
        "stereo_runtime.providers.amd.create_migraphx_rocm_provider", lambda **kwargs: provider
    )
    result = create_depth_provider(DepthProviderConfig(backend="migraphx_rocm", profile_sync=profile_sync))
    assert result is provider
    assert result.profile_sync is profile_sync


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_migraphx_profile_does_not_synchronize_normal_inference(monkeypatch, tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU timing requires a CUDA/ROCm device")
    provider = MIGraphXDepthProvider(device=device, onnx_path=tmp_path / "model_fp16_2x3.onnx")
    predicted = torch.arange(6, dtype=torch.float32, device=device).reshape(1, 2, 3)

    class Engine:
        input_image_size = (2, 3)

        def __call__(self, tensor):
            return predicted

    monkeypatch.setattr(provider, "load", lambda: Engine())
    # An already-submitted model output is enough to exercise preprocessing,
    # postprocessing and stream timing without model downloads or MIGraphX.
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: pytest.fail("device-wide synchronization"))
    result = provider.predict_profile(torch.zeros(1, 3, 8, 12, device=device))
    assert result.depth.shape == (1, 1, 8, 12)
    if device == "cuda":
        torch.cuda.current_stream().synchronize()
        events = result.cuda_timing_events
        assert {"depth_pre_start", "depth_pre_end", "depth_model_start", "depth_model_end",
                "depth_post_start", "depth_post_end"} <= events.keys()
        assert events["depth_pre_start"].elapsed_time(events["depth_post_end"]) >= 0
    else:
        assert result.cuda_timing_events == {}
    assert torch.isfinite(result.depth).all()
    assert result.depth.min() == 0
    assert result.depth.max() == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="explicit profiling requires CUDA/ROCm")
def test_migraphx_explicit_profile_sync_still_waits(monkeypatch, tmp_path):
    provider = MIGraphXDepthProvider(device="cuda", onnx_path=tmp_path / "model_fp16_2x3.onnx")
    provider.profile_sync = True
    predicted = torch.arange(6, dtype=torch.float32, device="cuda").reshape(1, 2, 3)

    class Engine:
        input_image_size = (2, 3)

        def __call__(self, tensor):
            return predicted

    monkeypatch.setattr(provider, "load", lambda: Engine())
    synchronize = torch.cuda.synchronize
    calls = []

    def sync(device):
        calls.append(device)
        synchronize(device)

    monkeypatch.setattr(torch.cuda, "synchronize", sync)
    result = provider.predict_profile(torch.zeros(1, 3, 8, 12, device="cuda"))
    assert calls == [provider.device] * 5
    assert result.depth.shape == (1, 1, 8, 12)


def test_create_pytorch_rocm_provider_marks_backend():
    provider = create_depth_provider(
        DepthProviderConfig(
            backend="pytorch_rocm",
            device="cuda",
            local_files_only=True,
            prefer_tensorrt=True,
            prefer_onnx=True,
        )
    )

    assert isinstance(provider, TorchRocmDepthProvider)
    assert provider.info.depth_backend == "pytorch_rocm"
    assert provider.info.runtime == "transformers-rocm"
    assert provider.info.execution_provider == "ROCm PyTorch"
    assert provider.info.output_device == "cuda"


def test_create_pytorch_rocm_provider_supports_generic_models():
    provider = create_depth_provider(
        DepthProviderConfig(
            backend="amd_rocm",
            model_id="apple/DepthPro-hf",
            model_name="DepthPro-Large",
            device="cuda",
            local_files_only=True,
            depth_resolution=518,
            patch_size=14,
        )
    )

    assert isinstance(provider, GenericTorchRocmDepthProvider)
    assert provider.info.model_id == "apple/DepthPro-hf"
    assert provider.info.model_name == "DepthPro-Large"
    assert provider.info.depth_backend == "pytorch_rocm"


def test_create_migraphx_rocm_provider_falls_back_to_pytorch_rocm(monkeypatch):
    monkeypatch.setattr(migraphx_provider, "is_rocm_torch_available", lambda: True)
    monkeypatch.setattr(migraphx_provider, "is_migraphx_available", lambda: False)

    provider = create_depth_provider(
        DepthProviderConfig(
            backend="migraphx_rocm",
            device="cuda",
            local_files_only=True,
            allow_pytorch_fallback=True,
            prefer_tensorrt=False,
            prefer_onnx=False,
        )
    )

    assert isinstance(provider, TorchRocmDepthProvider)
    assert provider.info.depth_backend == "pytorch_rocm"
    assert provider.info.fallback_reason == "migraphx is not installed"


def test_create_migraphx_rocm_provider_preserves_model_id_and_size(monkeypatch, tmp_path):
    monkeypatch.setattr(migraphx_provider, "is_rocm_torch_available", lambda: True)
    monkeypatch.setattr(migraphx_provider, "is_migraphx_available", lambda: True)

    provider = create_depth_provider(
        DepthProviderConfig(
            backend="migraphx_rocm",
            model_id="apple/DepthPro-hf",
            model_name="DepthPro-Large",
            device="cpu",
            cache_dir=tmp_path,
            onnx_path=tmp_path / "model_fp16_868x1540.onnx",
            engine_path=tmp_path / "model_fp16_868x1540.mgx",
            depth_resolution=1536,
            patch_size=14,
            allow_pytorch_fallback=False,
        )
    )

    assert isinstance(provider, MIGraphXDepthProvider)
    assert provider.info.model_id == "apple/DepthPro-hf"
    assert provider.info.model_name == "DepthPro-Large"
    assert provider.info.depth_resolution == 1536
    assert provider.preprocessor.input_size(2160, 3840) == (868, 1540)


def test_create_migraphx_rocm_provider_resolves_missing_onnx_before_rebuild(monkeypatch, tmp_path):
    monkeypatch.setattr(migraphx_provider, "is_rocm_torch_available", lambda: True)
    monkeypatch.setattr(migraphx_provider, "is_migraphx_available", lambda: True)
    generated_onnx = tmp_path / "generated.onnx"
    generated_onnx.write_bytes(b"onnx")
    generated_graph = tmp_path / "generated.mgx"
    calls = {}

    def prepare_model_artifacts(model_id, **kwargs):
        calls["model_id"] = model_id
        calls["kwargs"] = kwargs
        return SimpleNamespace(
            selected_onnx_path=generated_onnx,
            selected_migraphx_path=None,
            paths=SimpleNamespace(migraphx_fp16_path=generated_graph),
        )

    import stereo_runtime.model_artifacts as model_artifacts

    monkeypatch.setattr(model_artifacts, "prepare_model_artifacts", prepare_model_artifacts)
    missing_onnx = tmp_path / "model_fp16_196x336.onnx"
    graph_path = tmp_path / "model_fp16_196x336.mgx"

    provider = migraphx_provider.create_migraphx_rocm_provider(
        model_id="depth-anything/Video-Depth-Anything-Small",
        model_name="Video-Depth-Anything-Small",
        device="cpu",
        cache_dir=tmp_path,
        onnx_path=missing_onnx,
        graph_path=graph_path,
        build_graph=True,
        depth_resolution=336,
        patch_size=14,
        allow_pytorch_fallback=False,
    )

    assert calls["model_id"] == "depth-anything/Video-Depth-Anything-Small"
    assert calls["kwargs"]["export_width"] == 336
    assert provider.onnx_path == generated_onnx
    assert provider.graph_path == graph_path


def test_build_migraphx_graph_uses_fp8_then_saves(monkeypatch, tmp_path):
    calls = []

    class Program:
        def compile(self, target, offload_copy=False):
            calls.append(("compile", target, offload_copy))

    fake_mx = types.SimpleNamespace(
        parse_onnx=lambda path: calls.append(("parse", path)) or Program(),
        get_target=lambda name: calls.append(("target", name)) or name,
        autocast_fp8=lambda prog: calls.append("fp8"),
        quantize_fp16=lambda prog: calls.append("fp16"),
        save=lambda prog, path: calls.append(("save", path)),
    )
    monkeypatch.setitem(sys.modules, "migraphx", fake_mx)
    onnx_path = tmp_path / "model.onnx"
    graph_path = tmp_path / "model.mgx"
    onnx_path.write_bytes(b"onnx")

    assert migraphx_provider.build_migraphx_graph(onnx_path, graph_path) == graph_path
    assert "fp8" in calls
    assert "fp16" not in calls
    assert ("compile", "gpu", False) in calls


def test_build_migraphx_graph_falls_back_to_fp16(monkeypatch, tmp_path):
    calls = []

    class Program:
        def compile(self, target, offload_copy=False):
            calls.append(("compile", target, offload_copy))

    def fail_fp8(_prog):
        calls.append("fp8")
        raise RuntimeError("fp8 unsupported")

    fake_mx = types.SimpleNamespace(
        parse_onnx=lambda path: Program(),
        get_target=lambda name: name,
        autocast_fp8=fail_fp8,
        quantize_fp16=lambda prog: calls.append("fp16"),
        save=lambda prog, path: None,
    )
    monkeypatch.setitem(sys.modules, "migraphx", fake_mx)
    onnx_path = tmp_path / "model.onnx"
    graph_path = tmp_path / "model.mgx"
    onnx_path.write_bytes(b"onnx")

    migraphx_provider.build_migraphx_graph(onnx_path, graph_path)

    assert calls[:2] == ["fp8", "fp16"]


def test_build_migraphx_graph_force_fp32_skips_quantization(monkeypatch, tmp_path):
    calls = []

    class Program:
        def compile(self, target, offload_copy=False):
            calls.append(("compile", target, offload_copy))

    fake_mx = types.SimpleNamespace(
        parse_onnx=lambda path: Program(),
        get_target=lambda name: name,
        autocast_fp8=lambda prog: calls.append("fp8"),
        quantize_fp16=lambda prog: calls.append("fp16"),
        save=lambda prog, path: None,
    )
    monkeypatch.setitem(sys.modules, "migraphx", fake_mx)
    onnx_path = tmp_path / "model.onnx"
    graph_path = tmp_path / "model.mgx"
    onnx_path.write_bytes(b"onnx")

    migraphx_provider.build_migraphx_graph(onnx_path, graph_path, force_fp32=True)

    assert "fp8" not in calls
    assert "fp16" not in calls
    assert ("compile", "gpu", False) in calls
