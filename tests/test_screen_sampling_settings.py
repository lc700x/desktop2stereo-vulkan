from __future__ import annotations

from stereo_runtime.adapter import runtime_config_from_d2s_settings
from utils.xr_headset_presets import DEFAULT_XR_HEADSET_MODEL


def test_screen_sampling_visual_regression_settings_are_program_controlled() -> None:
    config = runtime_config_from_d2s_settings(
        {
            "Depth Model": "Distill-Any-Depth-Base",
            "OpenXR Visual Regression Directory": "artifacts/screen-mip",
        },
        cache_dir="models",
        device="cpu",
        depth_only=True,
    )

    assert config.openxr_visual_regression_dir == "artifacts/screen-mip"


def test_stream_output_quality_uses_selected_headset_and_pico4_default() -> None:
    config = runtime_config_from_d2s_settings(
        {
            "Depth Model": "Distill-Any-Depth-Base",
            "Run Mode": "RTMP Streamer",
            "Vulkan Projection Min LOD": 0.5,
            "Vulkan Projection Max LOD": 1.5,
            "Vulkan Projection MIP LOD Bias": -0.7,
        },
        cache_dir="models",
        device="cpu",
        depth_only=False,
    )

    assert DEFAULT_XR_HEADSET_MODEL == "Pico 4 / 4 Ultra"
    # The encoder owns the final canvas. Leaving headset-tier output quality
    # enabled here can inflate a Quality-Scale 4K source to 6528x3672 before
    # the encoder shrinks it back, undoing the whole render-scale saving.
    assert config.output_quality_enabled is False
    # The selected headset tier and LOD settings are still carried through.
    assert config.output_headset_tier_k == 4
    assert config.output_min_lod == 0.5
    assert config.output_max_lod == 1.5
    assert config.output_mip_lod_bias == -0.7


def test_local_viewer_keeps_output_quality_enabled() -> None:
    config = runtime_config_from_d2s_settings(
        {"Depth Model": "Distill-Any-Depth-Base", "Run Mode": "Local Viewer"},
        cache_dir="models",
        device="cpu",
        depth_only=False,
    )
    assert config.output_quality_enabled is True
    assert config.output_headset_tier_k == 4


def test_local_output_quality_uses_the_same_selected_headset_target() -> None:
    config = runtime_config_from_d2s_settings(
        {
            "Depth Model": "Distill-Any-Depth-Base",
            "Run Mode": "3D Monitor",
            "Monitor Index": 1,
            "Stereo Output": 3,
        },
        cache_dir="models",
        device="cpu",
        depth_only=False,
    )

    assert config.output_headset_tier_k == 4


def test_openxr_keeps_native_source_for_runtime_projection_resolution() -> None:
    config = runtime_config_from_d2s_settings(
        {
            "Depth Model": "Distill-Any-Depth-Base",
            "Run Mode": "OpenXR Link",
            "XR Headset Model": "Meta Quest 2",
        },
        cache_dir="models",
        device="cpu",
        depth_only=False,
    )

    assert config.output_quality_enabled is False
