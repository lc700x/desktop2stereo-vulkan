"""CoreML settings plumbing tests through the runtime adapter (ported)."""

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
