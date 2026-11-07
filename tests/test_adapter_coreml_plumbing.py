"""CoreML settings plumbing tests through the runtime adapter (ported)."""

import pytest
import torch

from stereo_runtime.adapter import (
    StereoRuntimeConfig,
    depth_provider_config_from_runtime,
    runtime_config_from_d2s_settings,
)


def test_settings_map_coreml_flags() -> None:
    config = runtime_config_from_d2s_settings(
        {"CoreML": "true", "Recompile CoreML": "1", "Depth Model": "xingyang1/Distill-Any-Depth-Small-hf"},
        cache_dir="/tmp/d2s-models",
        device="mps",
    )
    assert config.use_coreml is True
    assert config.recompile_coreml is True


def test_settings_default_coreml_flags_off() -> None:
    config = runtime_config_from_d2s_settings(
        {"Depth Model": "xingyang1/Distill-Any-Depth-Small-hf"},
        cache_dir="/tmp/d2s-models",
        device="mps",
    )
    assert config.use_coreml is False
    assert config.recompile_coreml is False


def test_depth_aa_defaults_to_two_and_preserves_explicit_zero() -> None:
    default_config = runtime_config_from_d2s_settings(
        {"Depth Model": "xingyang1/Distill-Any-Depth-Small-hf"},
        cache_dir="/tmp/d2s-models",
        device="cuda",
    )
    explicit_config = runtime_config_from_d2s_settings(
        {
            "Depth Model": "xingyang1/Distill-Any-Depth-Small-hf",
            "Anti-aliasing": 0,
            "Depth Antialias Strength": 0.0,
        },
        cache_dir="/tmp/d2s-models",
        device="cuda",
    )

    assert default_config.depth_antialias_strength == 2.0
    assert explicit_config.depth_antialias_strength == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA CUDA is required for the CUDA graph flag mapping")
def test_settings_map_cuda_graph_flag() -> None:
    config = runtime_config_from_d2s_settings(
        {"Depth Model": "xingyang1/Distill-Any-Depth-Small-hf", "Depth CUDA Graph": "true"},
        cache_dir="/tmp/d2s-models",
        device="cuda",
    )
    assert config.use_cuda_graph is True


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="NVIDIA CUDA is required for the CUDA graph flag mapping",
)
def test_moonlight_tensorrt_enables_cuda_graph_unless_explicitly_disabled() -> None:
    settings = {
        "Depth Model": "xingyang1/Distill-Any-Depth-Small-hf",
        "Run Mode": "Moonlight Streamer",
        "TensorRT": True,
    }
    default_config = runtime_config_from_d2s_settings(
        settings,
        cache_dir="/tmp/d2s-models",
        device="cuda",
    )
    disabled_config = runtime_config_from_d2s_settings(
        {**settings, "Depth CUDA Graph": False},
        cache_dir="/tmp/d2s-models",
        device="cuda",
    )

    assert default_config.use_cuda_graph is True
    assert disabled_config.use_cuda_graph is False


def test_runtime_config_forwards_to_depth_provider_config() -> None:
    config = StereoRuntimeConfig(
        model_id="xingyang1/Distill-Any-Depth-Small-hf",
        cache_dir="/tmp/d2s-models",
        device="mps",
        use_coreml=True,
        recompile_coreml=False,
    )
    dpc = depth_provider_config_from_runtime(config)
    assert dpc.use_coreml is True
    assert dpc.recompile_coreml is False


def test_cuda_graph_uses_two_inference_slots_without_changing_depth_config() -> None:
    config = StereoRuntimeConfig(
        model_id="xingyang1/Distill-Any-Depth-Small-hf",
        cache_dir="/tmp/d2s-models",
        device="cuda",
        depth_backend="tensorrt_native",
        use_cuda_graph=True,
        parallel_inference=False,
        parallel_inference_workers=1,
    )

    depth_config = depth_provider_config_from_runtime(config)

    assert depth_config.execution_slot_count == 2
    assert depth_config.depth_upsample == config.depth_upsample
