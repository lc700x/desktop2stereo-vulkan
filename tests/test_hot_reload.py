from __future__ import annotations

from types import SimpleNamespace

from stereo_runtime.adapter import StereoRuntimeConfig
from stereo_runtime.hot_reload import (
    StereoHotReloader,
    clamp_depth_pop_hot_reload,
    hot_reload_runtime_settings_snapshot,
    hot_reload_value_snapshot,
    runtime_stereo_overrides,
    to_bool_hot_reload,
)
from stereo_runtime.settings_snapshot import RuntimeSettingsSnapshot, SnapshotChangeClass


def make_config(**overrides):
    values = {
        "stereo_quality": "fast_plus",
        "depth_strength": 1.0,
        "convergence": 0.5,
        "output_format": "half_sbs",
        "max_disparity_px": None,
        "parallax_preset": "standard",
        "temporal": True,
        "depth_pop": 1.0,
        "foreground_shift_scale": 1.0,
        "midground_shift_scale": 1.0,
        "background_shift_scale": 1.0,
        "dynamic_convergence_enabled": False,
        "dynamic_convergence_strength": 0.0,
        "dynamic_convergence_target": 0.5,
        "dynamic_convergence_alpha": 0.85,
        "depth_antialias_strength": 0.25,
        "edge_threshold": 0.1,
        "edge_dilation": 2,
        "mask_feather_radius": 3,
        "hole_fill": "edge_aware",
        "hole_fill_mode": "balanced",
        "hole_fill_radius": 3,
        "hole_fill_strength": 1.0,
        "screen_edge_mask_suppression": 0.0,
        "cross_eyed": False,
        "anaglyph_method": "dubois",
        "debug_output": False,
        "fused": True,
        "temporal_strength": 0.5,
        "auto_reset_temporal": True,
        "scene_reset_threshold": 0.2,
        "stereo_preset": "cinema",
        "mode": "cinema",
        "export_height": 294,
        "export_width": 518,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_hot_reload_value_snapshot_parses_expected_fields():
    config = make_config()
    settings = {
        "Depth Strength": "1.25",
        "Convergence": "0.75",
        "Stereo Quality": "Quality",
        "Stereo Preset": "Game / Low Latency",
        "Runtime Quality Mode": "Still Image / HQ",
        "Display Mode": "Full-SBS",
        "Max Disparity Px": "88",
        "Parallax Preset": "strong",
        "Temporal Strength": "0",
        "Scene Reset Threshold": "0.3",
        "Depth Pop": "9.0",
        "Foreground Pop": "1.20",
        "Midground Pop": "1.10",
        "Background Pop": "0.90",
        "Depth Antialias Strength": "0.8",
        "Edge Dilation": "3",
        "Mask Feather Radius": "4",
        "Hole Fill Mode": "Soft / Low Ghost",
        "Hole Fill Radius": "5",
        "Hole Fill Strength": "0.4",
        "Edge Threshold": "0.2",
        "Anaglyph Method": "gray",
        "Cross Eyed": "yes",
        "Screen Edge Mask Suppression": "2",
        "Vulkan Projection Min LOD": "0.15",
        "Vulkan Projection Max LOD": "0.45",
        "Vulkan Projection MIP LOD Bias": "-0.25",
        "Vulkan Projection RCAS Sharpness": "0.65",
        "XR Headset Model": "Pimax Crystal / Light",
        "Debug Stereo Output": "yes",
        "Show FPS": "yes",
        "Display Fit Mode": "cover",
        "Audio Delay": "0.25",
    }

    values = hot_reload_value_snapshot(settings, config)

    assert values["depth_strength"] == 1.25
    assert values["presentation_flags"] == {
        "show_fps": True,
        "display_fit_mode": "cover",
    }
    assert values["convergence"] == 0.75
    assert "ipd" not in values
    assert "ipd_mm" not in values
    assert "stereo_scale" not in values
    assert "max_shift_ratio" not in values
    assert values["stereo_quality"] == "quality_4k"
    assert values["stereo_preset"] == "game_low_latency"
    assert values["runtime_quality_mode"] == "image"
    assert values["mode"] == "image"
    assert values["output_format"] == "full_sbs"
    assert values["max_disparity_px"] == 88.0
    assert values["parallax_preset"] == "strong"
    assert values["temporal"] is False
    assert values["scene_reset_threshold"] == 0.3
    assert ("reset_" + "cooldown" + "_frames") not in values
    assert values["depth_pop"] == 5.0
    assert values["foreground_shift_scale"] == 1.2
    assert values["midground_shift_scale"] == 1.1
    assert values["background_shift_scale"] == 0.9
    assert values["depth_antialias_strength"] == 0.8
    assert values["edge_dilation"] == 3
    assert values["mask_feather_radius"] == 4
    assert values["hole_fill_mode"] == "balanced"
    assert values["hole_fill_radius"] == 1
    assert values["hole_fill_strength"] == 0.6
    assert values["edge_threshold"] == 0.2
    assert values["anaglyph_method"] == "gray"
    assert values["cross_eyed"] is True
    assert values["screen_edge_mask_suppression"] == 2
    assert values["vulkan_projection_min_lod"] == 0.15
    assert values["vulkan_projection_max_lod"] == 0.45
    assert values["vulkan_projection_mip_lod_bias"] == -0.25
    assert values["vulkan_projection_rcas_sharpness"] == 0.65
    assert values["output_headset_tier_k"] == 8
    assert values["debug_output"] is True
    assert values["debug_flags"] == {"debug_output": True}
    assert values["audio_delay"] == 0.25


def test_runtime_stereo_overrides_maps_runtime_config():
    runtime = SimpleNamespace(config=make_config())

    overrides = runtime_stereo_overrides(runtime)

    assert overrides["backend"] == "fast_plus"
    assert overrides["depth_strength"] == 1.0
    assert overrides["temporal"] is True
    assert overrides["temporal_strength"] == 0.5
    assert overrides["auto_reset_temporal"] is True
    assert overrides["scene_reset_threshold"] == 0.2
    assert ("reset_" + "cooldown" + "_frames") not in overrides
    assert overrides["mask_feather_radius"] == 3
    assert overrides["hole_fill"] == "edge_aware"
    assert overrides["hole_fill_mode"] == "balanced"
    assert overrides["hole_fill_radius"] == 3
    assert overrides["hole_fill_strength"] == 1.0
    assert overrides["cross_eyed"] is False
    assert overrides["debug_output"] is False
    assert overrides["fused"] is True
    assert overrides["output_headset_tier_k"] == 4
    assert overrides["output_min_lod"] == 0.0
    assert overrides["output_max_lod"] == 0.35
    assert overrides["output_mip_lod_bias"] == -0.35
    assert overrides["output_rcas_sharpness"] == 0.5


def test_headset_and_rcas_hot_reload_update_common_output_quality() -> None:
    snapshot = RuntimeSettingsSnapshot(
        version=2,
        timestamp=1.0,
        output_headset_tier_k=2,
        vulkan_projection_min_lod=0.25,
        vulkan_projection_max_lod=1.25,
        vulkan_projection_mip_lod_bias=-0.5,
        vulkan_projection_rcas_sharpness=0.75,
    )

    assert snapshot.classify() is SnapshotChangeClass.HOT_RELOAD
    assert snapshot.to_config_updates()["output_headset_tier_k"] == 2
    assert snapshot.to_config_updates()["output_min_lod"] == 0.25
    assert snapshot.to_config_updates()["output_max_lod"] == 1.25
    assert snapshot.to_config_updates()["output_mip_lod_bias"] == -0.5
    assert snapshot.to_config_updates()["output_rcas_sharpness"] == 0.75


def test_hot_reload_hole_fill_off_disables_fill_and_ignores_stale_strength() -> None:
    config = make_config()

    values = hot_reload_value_snapshot(
        {
            "Hole Fill Mode": "Off / No Fill",
            "Hole Fill Radius": 3,
            "Hole Fill Strength": 1.0,
        },
        config,
    )

    assert values["hole_fill_mode"] == "none"
    assert values["hole_fill_radius"] == 0
    assert values["hole_fill_strength"] == 0.0

    runtime = SimpleNamespace(
        config=make_config(
            hole_fill="edge_aware",
            hole_fill_mode="none",
            hole_fill_radius=0,
            hole_fill_strength=0.0,
        )
    )
    assert runtime_stereo_overrides(runtime)["hole_fill"] == "none"


def test_hot_reload_depth_provider_rebuild_fields_only_when_changed():
    config = make_config(
        model_id="Distill-Any-Depth-Base",
        depth_backend="pytorch_cuda",
        profile_sync=False,
    )

    unchanged = hot_reload_value_snapshot(
        {
            "Depth Model": "Distill-Any-Depth-Base",
            "TensorRT": False,
            "Depth Profile Sync": False,
            "Depth Resolution": 518,
        },
        config,
    )
    changed = hot_reload_runtime_settings_snapshot(
        {
            "Depth Model": "DepthPro-Large",
            "TensorRT": True,
            "Depth Profile Sync": True,
            "Depth Resolution": 756,
        },
        config,
        version=31,
        timestamp=1.0,
    )

    assert "model_id" not in unchanged
    assert "depth_backend" not in unchanged
    assert "profile_sync" not in unchanged
    assert "export_height" not in unchanged
    assert "export_width" not in unchanged
    assert changed.model_id == "DepthPro-Large"
    assert changed.depth_backend == "tensorrt_native"
    assert changed.profile_sync is True
    assert changed.export_height == 429
    assert changed.export_width == 756
    assert changed.classify() is SnapshotChangeClass.PIPELINE_REBUILD


def test_null_backend_placeholders_do_not_request_rebuild_against_auto_config():
    # The runtime config defaults depth_backend to "auto" (resolved per
    # platform, e.g. PyTorch MPS on macOS). settings.yaml saves
    # "TensorRT: null"/"MIGraphX: null" placeholders; these must not force
    # the legacy "pytorch_cuda" fallback, which would diverge from the
    # runtime default and raise RuntimeSettingsPipelineRebuildRequired on the
    # first hot reload, killing the pipeline thread.
    config = make_config(model_id="Distill-Any-Depth-Small", depth_backend="auto")

    snapshot = hot_reload_runtime_settings_snapshot(
        {
            "Depth Model": "Distill-Any-Depth-Small",
            "TensorRT": None,
            "MIGraphX": None,
        },
        config,
        version=7,
        timestamp=1.0,
    )

    assert snapshot.depth_backend is None
    assert snapshot.classify() is not SnapshotChangeClass.PIPELINE_REBUILD


def test_hot_reload_ignores_run_mode_as_runtime_quality_mode():
    config = make_config(mode="movie")

    values = hot_reload_value_snapshot({"Run Mode": "OpenXR Link"}, config)
    snapshot = hot_reload_runtime_settings_snapshot({"Run Mode": "OpenXR Link"}, config, version=22, timestamp=1.0)

    assert "runtime_quality_mode" not in values
    assert "mode" not in values
    assert snapshot.runtime_quality_mode is None


def test_hot_reload_explicit_temporal_toggles_override_positive_values():
    config = make_config(temporal=True, auto_reset_temporal=True, temporal_strength=0.7, scene_reset_threshold=0.22)
    settings = {
        "Temporal": "false",
        "Temporal Strength": "0.7",
        "Auto Scene Reset": "false",
        "Scene Reset Threshold": "0.22",
    }

    values = hot_reload_value_snapshot(settings, config)
    snapshot = hot_reload_runtime_settings_snapshot(settings, config, version=21, timestamp=1.0)

    assert values["temporal"] is False
    assert values["temporal_strength"] == 0.7
    assert values["auto_reset_temporal"] is False
    assert values["scene_reset_threshold"] == 0.22
    assert snapshot.temporal is False
    assert snapshot.auto_reset_temporal is False


def test_hot_reload_bool_and_depth_pop_helpers():
    assert to_bool_hot_reload(True) is True
    assert to_bool_hot_reload("on") is True
    assert to_bool_hot_reload(None) is False
    assert clamp_depth_pop_hot_reload(-5.0) == -0.9
    assert clamp_depth_pop_hot_reload(10.0) == 5.0


def test_hot_reload_fast_quality_disables_temporal_and_postprocess():
    config = make_config(stereo_quality="fast", temporal_strength=0.7, depth_pop=0.5, depth_antialias_strength=2.0)
    settings = {
        "Stereo Quality": "fast",
        "Synthetic View": "fast",
        "Temporal Strength": "0.7",
        "Scene Reset Threshold": "0.22",
        "Depth Pop": "0.5",
        "Depth Antialias Strength": "2.0",
    }

    values = hot_reload_value_snapshot(settings, config)

    assert values["stereo_quality"] == "fast"
    assert values["temporal"] is False
    assert values["temporal_strength"] == 0.0
    assert values["auto_reset_temporal"] is False
    assert values["scene_reset_threshold"] == 0.0
    assert ("reset_" + "cooldown" + "_frames") not in values
    assert values["depth_pop"] == 0.0
    assert values["depth_antialias_strength"] == 0.0


def test_hot_reload_builds_runtime_settings_snapshot():
    config = make_config()
    settings = {
        "Depth Strength": "1.25",
        "Synthetic View": "HQ",
        "Stereo Mode Preset": "Still Image / HQ",
        "Display Mode": "Half-TAB",
        "Max Disparity PX": "96",
        "Parallax Budget Preset": "comfort",
        "Temporal Strength": "0.3",
        "Scene Reset Threshold": "0.4",
        "Screen Edge Mask Suppression": "3",
        "Debug Stereo Output": "true",
    }

    snapshot = hot_reload_runtime_settings_snapshot(
        settings,
        config,
        version=12,
        timestamp=3.5,
    )

    assert snapshot.version == 12
    assert snapshot.timestamp == 3.5
    assert snapshot.source == "settings_yaml_hot_reload"
    assert snapshot.depth_strength == 1.25
    assert not hasattr(snapshot, "ipd_mm")
    assert snapshot.stereo_quality == "hq_4k"
    assert snapshot.stereo_preset == "still_image_hq"
    assert snapshot.output_format == "half_tab"
    assert snapshot.max_disparity_px == 96.0
    assert snapshot.parallax_preset == "comfort"
    assert snapshot.temporal is True
    assert snapshot.temporal_strength == 0.3
    assert snapshot.auto_reset_temporal is True
    assert snapshot.scene_reset_threshold == 0.4
    assert not hasattr(snapshot, "reset_" + "cooldown" + "_frames")
    assert snapshot.screen_edge_mask_suppression == 3
    assert snapshot.debug_output is True
    assert snapshot.debug_flags == {"debug_output": True}
    assert snapshot.classify() is SnapshotChangeClass.HOT_RELOAD


def test_hot_reload_poll_returns_snapshot_without_applying_runtime(tmp_path):
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text("", encoding="utf-8")
    runtime = SimpleNamespace(
        config=StereoRuntimeConfig(
            model_id="Distill-Any-Depth-Base",
            cache_dir="models",
            stereo_quality="fast_plus",
            stereo_preset="cinema",
        ),
        apply_settings_snapshot=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("poll must not apply runtime settings directly")
        ),
    )
    reloader = StereoHotReloader(
        settings_path=str(settings_path),
        interval_s=0.0,
        read_settings=lambda _path: {
            "Stereo Quality": "Quality",
            "Stereo Preset": "Game / Low Latency",
            "Depth Strength": "2.0",
        },
        clock=lambda: 1.0,
    )

    polled = reloader.poll_settings_snapshot_if_needed(
        runtime=runtime,
        active_preset="cinema",
    )

    assert polled is not None
    snapshot, applied_preset, values = polled
    assert snapshot.source == "settings_yaml_hot_reload"
    assert snapshot.stereo_quality == "quality_4k"
    assert snapshot.stereo_preset == "game_low_latency"
    assert snapshot.depth_strength == 2.0
    assert applied_preset == "game_low_latency"
    assert values["stereo_preset"] == "game_low_latency"
    assert reloader.poll_settings_snapshot_if_needed(runtime=runtime, active_preset="cinema") is None


def test_hot_reload_pushes_all_openxr_stereo_controls(tmp_path):
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text("", encoding="utf-8")
    runtime_config = StereoRuntimeConfig(
        model_id="Distill-Any-Depth-Base",
        cache_dir="models",
        stereo_quality="fast",
        stereo_preset="cinema",
    )
    runtime = SimpleNamespace(
        config=runtime_config,
        stereo_config=SimpleNamespace(output_format="half_sbs"),
        apply_settings_snapshot=lambda snapshot, active_preset=None: (
            setattr(runtime, "applied_snapshot", snapshot),
            setattr(runtime, "applied_active_preset", active_preset),
        ),
        configure_stereo=lambda stereo_config, reset_temporal=False: setattr(runtime, "stereo_config", stereo_config),
    )
    pushed = {}
    reloader = StereoHotReloader(
        settings_path=str(settings_path),
        interval_s=0.0,
        read_settings=lambda _path: {
            "Stereo Quality": "fast",
            "Stereo Preset": "Game / Low Latency",
            "Depth Strength": "2.0",
            "Convergence": "0.25",
            "Display Mode": "Full-TAB",
            "Max Disparity Px": "72",
            "Parallax Preset": "standard",
            "Screen Edge Mask Suppression": "4",
            "Debug Stereo Output": "on",
        },
        clock=lambda: 1.0,
    )

    assert reloader.apply_if_needed(
        runtime=runtime,
        active_preset="cinema",
        on_openxr_config_update=lambda **kwargs: pushed.update(kwargs),
        on_mode_log=lambda _reason: None,
    )

    assert pushed["snapshot"] is runtime.applied_snapshot
    assert runtime.applied_snapshot.source == "settings_yaml_hot_reload"
    assert not hasattr(runtime.applied_snapshot, "ipd_mm")
    assert runtime.applied_snapshot.depth_strength == 2.0
    assert runtime.applied_snapshot.convergence == 0.25
    assert not hasattr(runtime.applied_snapshot, "stereo_scale")
    assert not hasattr(runtime.applied_snapshot, "max_shift_ratio")
    assert runtime.applied_snapshot.stereo_quality == "fast"
    assert runtime.applied_snapshot.stereo_preset == "game_low_latency"
    assert runtime.applied_active_preset == "game_low_latency"
    assert runtime.applied_snapshot.output_format == "full_tab"
    assert runtime.applied_snapshot.max_disparity_px == 72.0
    assert runtime.applied_snapshot.parallax_preset == "standard"
    assert runtime.applied_snapshot.screen_edge_mask_suppression == 4
    assert runtime.applied_snapshot.debug_output is True
    assert runtime.applied_snapshot.debug_flags == {"debug_output": True}


def test_hot_reload_fallback_filters_snapshot_only_fields(tmp_path):
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text("", encoding="utf-8")
    runtime = SimpleNamespace(
        config=StereoRuntimeConfig(
            model_id="Distill-Any-Depth-Base",
            cache_dir="models",
            stereo_quality="fast_plus",
            stereo_preset="cinema",
        ),
        stereo_config=SimpleNamespace(output_format="half_sbs"),
        configure_stereo=lambda stereo_config, reset_temporal=False: setattr(runtime, "stereo_config", stereo_config),
    )
    pushed = {}
    reloader = StereoHotReloader(
        settings_path=str(settings_path),
        interval_s=0.0,
        read_settings=lambda _path: {
            "Stereo Quality": "Quality",
            "Stereo Preset": "Still Image / HQ",
            "Debug Stereo Output": "yes",
            "Display Mode": "Full-SBS",
        },
        clock=lambda: 1.0,
    )

    assert reloader.apply_if_needed(
        runtime=runtime,
        active_preset="cinema",
        on_openxr_config_update=lambda **kwargs: pushed.update(kwargs),
        on_mode_log=lambda _reason: None,
    )

    assert runtime.config.stereo_quality == "quality_4k"
    assert runtime.config.stereo_preset == "still_image_hq"
    assert runtime.config.debug_output is True
    assert not hasattr(runtime.config, "debug_flags")
    assert runtime.stereo_config.backend == "quality_4k"
    assert pushed["snapshot"].stereo_quality == "quality_4k"
    assert pushed["snapshot"].stereo_preset == "still_image_hq"
    assert pushed["snapshot"].debug_flags == {"debug_output": True}


def test_hot_reload_auto_preset_preserves_active_preset(tmp_path):
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text("", encoding="utf-8")
    runtime_config = StereoRuntimeConfig(
        model_id="Distill-Any-Depth-Base",
        cache_dir="models",
        stereo_quality="fast_plus",
        stereo_preset="auto",
    )
    runtime = SimpleNamespace(
        config=runtime_config,
        stereo_config=SimpleNamespace(output_format="half_sbs"),
        apply_settings_snapshot=lambda snapshot, active_preset=None: (
            setattr(runtime, "applied_snapshot", snapshot),
            setattr(runtime, "applied_active_preset", active_preset),
        ),
        configure_stereo=lambda stereo_config, reset_temporal=False: setattr(runtime, "stereo_config", stereo_config),
    )
    reloader = StereoHotReloader(
        settings_path=str(settings_path),
        interval_s=0.0,
        read_settings=lambda _path: {"Stereo Preset": "auto"},
        clock=lambda: 1.0,
    )

    assert reloader.apply_if_needed(
        runtime=runtime,
        active_preset="game_low_latency",
        on_openxr_config_update=lambda **_kwargs: None,
        on_mode_log=lambda _reason: None,
    )

    assert runtime.applied_snapshot.stereo_preset == "auto"
    assert runtime.applied_active_preset == "game_low_latency"
