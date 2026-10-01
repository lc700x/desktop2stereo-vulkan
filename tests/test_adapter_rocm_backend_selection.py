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
        },
        cache_dir=tmp_path,
        device="cuda",
    )

    assert config.depth_backend == "migraphx_rocm"
    assert config.build_migraphx_graph is True
