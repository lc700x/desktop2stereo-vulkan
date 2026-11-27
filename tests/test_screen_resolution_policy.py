import pytest

from utils.screen_resolution_policy import (
    ScreenSamplingPlanStabilizer,
    build_output_sampling_plan,
    build_projection_screen_sampling_plan,
    build_screen_sampling_plan,
    classify_input_resolution,
)


@pytest.mark.parametrize(
    ("width", "height", "tier"),
    (
        (1920, 1080, 1),
        (2560, 1440, 2),
        (3840, 2160, 4),
        (2560, 1600, 2),
        (3264, 1836, 2),
        (3440, 1440, 2),
    ),
)
def test_input_resolution_uses_nearest_standard_tier(width, height, tier):
    assert classify_input_resolution(width, height) == tier


@pytest.mark.parametrize(
    ("source", "headset", "recommended", "effective", "filter_scale", "upscale_scale", "mode"),
    (
        ((1920, 1080), 2, 2, 2, 1.0, 2.0, "upscale_easu"),
        ((2560, 1440), 4, 4, 4, 1.0, 2.0, "upscale_easu"),
        ((3840, 2160), 8, 8, 8, 1.0, 2.0, "upscale_easu"),
        ((3840, 2160), 2, 8, 2, 2.0, 1.0, "downsample_lanczos_rcas"),
        ((1920, 1080), 8, 2, 2, 1.0, 2.0, "upscale_easu"),
        ((2560, 1440), 2, 4, 2, 1.0, 1.0, "native_mip"),
        ((3264, 1836), 4, 4, 4, 1.0, 2.0, "upscale_easu"),
    ),
)
def test_input_headset_matrix(
    source, headset, recommended, effective, filter_scale, upscale_scale, mode
):
    plan = build_screen_sampling_plan(*source, headset)
    assert plan.recommended_headset_tier_k == recommended
    assert plan.effective_tier_k == effective
    assert plan.filter_scale == pytest.approx(filter_scale)
    assert plan.upscale_scale == pytest.approx(upscale_scale)
    assert plan.mode == mode


def test_invalid_resolution_is_rejected():
    with pytest.raises(ValueError):
        build_screen_sampling_plan(0, 1080, 2)


def test_shared_headset_target_uses_the_existing_input_headset_matrix():
    upscale = build_output_sampling_plan(
        1920, 1080, headset_tier_k=4
    )
    native = build_output_sampling_plan(
        3840, 2160, headset_tier_k=4
    )

    assert (upscale.mode, upscale.target_width, upscale.target_height) == (
        "upscale_easu", 3840, 2160
    )
    assert (native.mode, native.target_width, native.target_height) == (
        "native_mip", 3840, 2160
    )


def test_projection_plan_uses_visible_footprint_instead_of_quest2_2k_tier():
    plan = build_projection_screen_sampling_plan(3840, 2160, 3000, 1687)

    assert plan.mode == "downsample_lanczos_rcas"
    assert plan.quality_size == (3008, 1692)
    assert plan.quality_width > 2048
    assert plan.quality_width / plan.quality_height == pytest.approx(3840 / 2160, rel=1e-3)


def test_projection_plan_preserves_1920x1200_aspect_ratio():
    plan = build_projection_screen_sampling_plan(1920, 1200, 2400, 1500)

    assert plan.mode == "upscale_easu"
    assert plan.quality_size == (2432, 1520)
    assert plan.quality_width / plan.quality_height == pytest.approx(16 / 10)


def test_projection_oversample_increases_minified_screen_quality_without_inventing_detail():
    minified = build_projection_screen_sampling_plan(
        3840, 2160, 667, 386, oversample=1.5
    )
    already_upscaled = build_projection_screen_sampling_plan(
        1920, 1200, 2400, 1500, oversample=1.5
    )

    assert minified.mode == "downsample_lanczos_rcas"
    assert minified.quality_size == (1024, 576)
    assert minified.filter_scale == pytest.approx(3840 / 1024)
    assert already_upscaled.mode == "upscale_easu"
    assert already_upscaled.quality_size == (2432, 1520)


def test_projection_plan_keeps_native_mode_across_small_pose_footprint_jitter():
    for footprint_width in (1893, 1918, 1975):
        plan = build_projection_screen_sampling_plan(
            1920, 800, footprint_width, 1000
        )
        assert plan.mode == "native_mip"
        assert plan.quality_size == (1920, 800)


def test_projection_sampling_plan_requires_persistent_candidate_before_switch():
    active = build_projection_screen_sampling_plan(1920, 1080, 1500, 844)
    candidate = build_projection_screen_sampling_plan(1920, 1080, 2400, 1350)
    alternate = build_projection_screen_sampling_plan(1920, 1080, 1600, 900)
    stabilizer = ScreenSamplingPlanStabilizer()

    assert stabilizer.select(active, candidate, 0.0) == active
    assert stabilizer.select(active, alternate, 0.1) == active
    assert stabilizer.select(active, candidate, 0.2) == active
    for index in range(1, 7):
        assert stabilizer.select(active, candidate, 0.2 + index * 0.1) == active
    assert stabilizer.select(active, candidate, 0.9) == candidate


def test_projection_sampling_plan_switches_immediately_on_source_change():
    active = build_projection_screen_sampling_plan(1920, 1080, 1500, 844)
    candidate = build_projection_screen_sampling_plan(1280, 720, 1500, 844)
    stabilizer = ScreenSamplingPlanStabilizer()

    assert stabilizer.select(active, candidate, 0.0) == candidate
