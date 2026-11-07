import sys
from pathlib import Path

from path_config import APP_ROOT

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from stereo_runtime.openxr_render import OpenXRRenderConfig
from stereo_runtime.presets import (
    AutoModeRuntime,
    AutoModeSignals,
    PARALLAX_BUDGET_PRESETS,
    PRESET_CHOICES,
    auto_detection_required,
    auto_mode_scores,
    classify_auto_mode,
    openxr_config_for_auto_mode,
    openxr_config_for_preset,
    preset_summary,
    stereo_config_for_auto_mode,
    stereo_config_for_preset,
)
from stereo_runtime.synthesis import StereoConfig


def test_preset_choices_are_public_and_stable():
    assert PRESET_CHOICES == ("auto", "traditional_fastest", "cinema", "game_low_latency", "still_image_hq", "debug_export")


def test_parallax_budget_presets_are_public():
    assert PARALLAX_BUDGET_PRESETS["standard"][1080] == 48.0
    assert PARALLAX_BUDGET_PRESETS["extreme"][2160] == 160.0


def test_stereo_presets_map_to_expected_modes():
    traditional = stereo_config_for_preset("Traditional / Fastest")
    cinema = stereo_config_for_preset("cinema")
    game = stereo_config_for_preset("game")
    still = stereo_config_for_preset("still image / hq")
    debug = stereo_config_for_preset("debug")

    assert isinstance(traditional, StereoConfig)
    assert traditional.backend == "fast"
    assert traditional.temporal is False
    assert traditional.temporal_strength == 0.0
    assert traditional.parallax_preset == "standard"
    assert not hasattr(traditional, "max_shift_ratio")
    assert traditional.edge_dilation == 0
    assert traditional.depth_antialias_strength == 0.0

    assert isinstance(cinema, StereoConfig)
    assert cinema.backend == "quality_4k"
    assert cinema.temporal is False
    assert cinema.temporal_strength == 0.0
    assert cinema.auto_reset_temporal is False
    assert cinema.hole_fill == "none"
    assert cinema.hole_fill_mode == "none"
    assert cinema.hole_fill_radius == 0
    assert cinema.hole_fill_strength == 0.0
    assert cinema.parallax_preset == "standard"
    assert not hasattr(cinema, "ipd_mm")
    assert not hasattr(cinema, "stereo_scale")
    for config in (traditional, cinema, game, still, debug):
        assert config.depth_strength == 0.48
        assert config.convergence == 0.5
        assert config.parallax_preset == "standard"
        assert config.foreground_shift_scale == 1.0

    assert game.backend == "fast_plus"
    assert game.hole_fill == "none"
    assert game.hole_fill_mode == "none"
    assert game.hole_fill_radius == 0
    assert game.hole_fill_strength == 0.0
    assert cinema.temporal_strength == 0.0
    assert game.depth_strength == cinema.depth_strength

    assert still.backend == "hq_4k"
    assert still.layers == 3
    assert still.temporal is False
    assert still.auto_reset_temporal is False
    assert still.hole_fill_mode == "quality"
    assert still.hole_fill_radius == 3
    assert still.hole_fill_strength == 1.0

    assert debug.debug_output is True
    assert debug.depth_strength >= cinema.depth_strength
    assert debug.hole_fill_mode == "quality"


def test_preset_output_format_and_overrides():
    config = stereo_config_for_preset("cinema", output_format="full_sbs", overrides={"depth_strength": 2.25})
    assert config.output_format == "full_sbs"
    assert config.depth_strength == 2.25
    runtime_override = stereo_config_for_preset("cinema", overrides={"parallax_preset": "strong"})
    assert runtime_override.parallax_preset == "strong"

    with pytest.raises(ValueError, match="unknown config override"):
        stereo_config_for_preset("cinema", overrides={"not_a_field": 1})


def test_stereo_presets_keep_moonlight_depth_parameters_across_modes():
    configs = (
        stereo_config_for_preset("traditional_fastest"),
        stereo_config_for_preset("cinema"),
        stereo_config_for_preset("game_low_latency"),
        stereo_config_for_preset("still_image_hq"),
        stereo_config_for_preset("debug_export"),
    )

    assert {
        (config.depth_strength, config.convergence, config.parallax_preset)
        for config in configs
    } == {(0.48, 0.5, "standard")}


def test_openxr_presets_map_shared_stereo_params():
    traditional = openxr_config_for_preset("traditional_fastest")
    openxr = openxr_config_for_preset("game_low_latency", screen_roll=0.5)
    cinema = openxr_config_for_preset("cinema")

    assert isinstance(traditional, OpenXRRenderConfig)
    assert traditional.parallax_preset == "standard"
    assert traditional.output_mode == "rgb_depth"
    assert cinema.output_mode == "full_synthesis_eyes"
    assert not hasattr(traditional, "max_shift_ratio")
    assert isinstance(openxr, OpenXRRenderConfig)
    assert openxr.screen_roll == 0.5
    assert openxr.depth_strength == cinema.depth_strength == 0.48
    assert openxr.parallax_preset == "standard"
    for config in (traditional, openxr, cinema, openxr_config_for_preset("still_image_hq"), openxr_config_for_preset("debug_export")):
        assert config.convergence == 0.5


def test_auto_mode_classifier_priority():
    export = classify_auto_mode(AutoModeSignals(user_export_action=True, frame_motion_score=1.0))
    still = classify_auto_mode(AutoModeSignals(still_duration_s=2.0, frame_motion_score=0.01))
    game = classify_auto_mode(AutoModeSignals(gpu_3d_util=0.7, input_activity=0.8))
    cinema = classify_auto_mode(AutoModeSignals(frame_motion_score=0.05))

    assert export.preset == "debug_export"
    assert still.preset == "still_image_hq"
    assert game.preset == "game_low_latency"
    assert cinema.preset == "cinema"


def test_auto_mode_config_helpers_return_decision_and_config():
    decision, stereo = stereo_config_for_auto_mode(AutoModeSignals(gpu_3d_util=0.65, input_activity=0.8), output_format="half_tab")
    openxr_decision, openxr = openxr_config_for_auto_mode(AutoModeSignals(openxr_active=True), screen_roll=0.25)

    assert decision.preset == "game_low_latency"
    assert stereo.output_format == "half_tab"
    assert stereo.temporal_strength <= 0.7
    assert openxr_decision.preset == "cinema"
    assert openxr.screen_roll == 0.25


def test_preset_summary_is_serializable_shape():
    summary = preset_summary()
    assert set(summary) == {"traditional_fastest", "cinema", "game_low_latency", "still_image_hq", "debug_export"}
    assert summary["traditional_fastest"]["stereo"]["backend"] == "fast"
    assert summary["cinema"]["stereo"]["backend"] == "quality_4k"
    assert "depth_strength" in summary["cinema"]["openxr"]


def test_presets_do_not_control_depth_provider_or_model_paths():
    forbidden = {
        "cache_dir",
        "depth_backend",
        "depth_resolution",
        "engine_path",
        "model_id",
        "model_name",
        "onnx_path",
        "trt_cache_dir",
    }
    summary = preset_summary()
    for preset in summary.values():
        assert forbidden.isdisjoint(preset["stereo"])
        assert forbidden.isdisjoint(preset["openxr"])


def test_auto_mode_scores_use_behavior_before_process_name():
    generic_game = auto_mode_scores(AutoModeSignals(gpu_3d_util=0.75, input_activity=0.8, foreground_process="UnknownApp.exe"))
    video = auto_mode_scores(AutoModeSignals(gpu_video_decode_util=0.35, gpu_3d_util=0.05, input_activity=0.02, idle_seconds=8.0, audio_active=True))

    assert generic_game["game"] >= 5.0
    assert generic_game["game"] > generic_game["video"]
    assert video["video"] >= 4.0
    assert video["video"] > video["game"]


def test_auto_mode_runtime_requires_consecutive_samples_and_holds():
    runtime = AutoModeRuntime()
    game_signals = AutoModeSignals(gpu_3d_util=0.75, input_activity=0.9, fullscreen=True, target_fps=120.0)

    for _ in range(3):
        decision = runtime.update(game_signals, dt_s=0.5)
        assert decision.preset == "cinema"
        assert "confirming game_low_latency" in decision.reason

    decision = runtime.update(game_signals, dt_s=0.5)
    assert decision.preset == "game_low_latency"
    assert runtime.state.active_preset == "game_low_latency"

    still_signals = AutoModeSignals(gpu_3d_util=0.0, gpu_video_decode_util=0.0, input_activity=0.0, idle_seconds=40.0, still_duration_s=2.0)
    held = runtime.update(still_signals, dt_s=0.5)
    assert held.preset == "game_low_latency"
    assert "holding game_low_latency" in held.reason


def test_manual_presets_bypass_auto_runtime_contract():
    manual = stereo_config_for_preset("game_low_latency", output_format="half_sbs")
    auto_default = stereo_config_for_preset("auto", output_format="half_sbs")

    assert manual.backend == "fast_plus"
    assert manual.hole_fill == "none"
    assert auto_default.backend == "quality_4k"
    assert auto_default.hole_fill == "none"
    assert auto_default.hole_fill_mode == "none"
    assert auto_detection_required("auto") is True
    assert auto_detection_required("traditional_fastest") is False
    assert auto_detection_required("cinema") is False
    assert auto_detection_required("game_low_latency") is False
