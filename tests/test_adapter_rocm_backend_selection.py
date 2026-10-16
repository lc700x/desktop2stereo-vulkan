from __future__ import annotations

import torch

from stereo_runtime.adapter import runtime_config_from_d2s_settings


def test_rocm_respects_disabled_migraphx_setting(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.version, "hip", "7.0.0")

    config = runtime_config_from_d2s_settings(
        {
            "Depth Model": "Distill-Any-Depth-Base",
            "MIGraphX": False,
        },
        cache_dir=tmp_path,
        device="cuda",
    )

    assert config.depth_backend == "pytorch_rocm"
    assert config.build_migraphx_graph is False


def test_rocm_uses_migraphx_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.version, "hip", "7.0.0")

    config = runtime_config_from_d2s_settings(
        {
            "Depth Model": "Distill-Any-Depth-Base",
            "MIGraphX": True,
            "Depth CUDA Graph": True,
        },
        cache_dir=tmp_path,
        device="cuda",
    )

    assert config.depth_backend == "migraphx_rocm"
    assert config.build_migraphx_graph is True
    assert config.use_cuda_graph is False


def test_nvidia_ignores_stale_migraphx_toggle_when_tensorrt_is_enabled(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.version, "hip", None)

    config = runtime_config_from_d2s_settings(
        {
            "Depth Model": "Distill-Any-Depth-Small",
            "MIGraphX": True,
            "TensorRT": True,
        },
        cache_dir=tmp_path,
        device="cuda",
    )

    assert config.depth_backend == "tensorrt_native"


def test_macos_ignores_stale_windows_gpu_backend_toggles(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.version, "hip", None)

    config = runtime_config_from_d2s_settings(
        {
            "Depth Model": "Distill-Any-Depth-Small",
            "MIGraphX": True,
            "TensorRT": True,
        },
        cache_dir=tmp_path,
        device="mps",
    )

    assert config.depth_backend == "auto"
