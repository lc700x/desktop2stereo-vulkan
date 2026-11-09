from __future__ import annotations

import pytest

from stereo_runtime.render_size import (
    RenderSizeConfig,
    RenderSizePolicy,
    render_size_config_from_settings,
    resolve_render_size,
    runtime_output_size_text,
)


def test_render_size_config_from_settings_parses_gui_fields():
    config = render_size_config_from_settings(
        {
            "Render Size Policy": "scaled",
            "Render Scale": "2K / 75%",
            "Render Fixed Width": "1600",
            "Render Fixed Height": "900",
            "Render Max Pixels": "2073600",
            "Render Min Dimension": "540",
            "Render Align": "1",
        }
    )

    assert config == RenderSizeConfig(
        policy=RenderSizePolicy.SCALED,
        scale_factor="75%",
        fixed_width=1600,
        fixed_height=900,
        max_pixels=2073600,
        min_dimension=540,
        align=1,
    )


def test_render_size_config_from_settings_defaults_invalid_policy_to_scaled():
    config = render_size_config_from_settings({"Render Size Policy": "unknown"})

    assert config.policy is RenderSizePolicy.SCALED
    assert config.align == 1


def test_resolve_render_size_native_aligns_capture_size():
    config = RenderSizeConfig(policy=RenderSizePolicy.NATIVE, align=16)

    assert resolve_render_size((1919, 1079), config) == (1904, 1072)


@pytest.mark.parametrize(
    ("capture_size", "tier", "expected"),
    [
        ((3840, 2160), "1K / 50%", (1920, 1072)),
        ((2160, 3840), "1K / 50%", (1072, 1920)),
        ((3840, 2400), "2K / 75%", (2880, 1792)),
        ((4096, 2160), "3K / 85%", (3472, 1824)),
        ((3840, 1600), "2K / 75%", (2880, 1200)),
    ],
)
def test_resolve_render_size_scaled_preserves_legacy_tiers_and_alignment(capture_size, tier, expected):
    config = RenderSizeConfig(policy=RenderSizePolicy.SCALED, scale_factor=tier, align=16)

    assert resolve_render_size(capture_size, config) == expected


@pytest.mark.parametrize("capture_size", [
    (1280, 720), (1920, 1080), (2560, 1440), (3840, 2160), (3440, 1440), (1080, 1920), (320, 200),
])
@pytest.mark.parametrize("percent", ["100%", "85%", "75%", "50%"])
def test_resolve_render_size_scales_every_input_without_minimum_side(capture_size, percent):
    config = RenderSizeConfig(scale_factor=percent, min_dimension=480)
    factor = int(percent[:-1]) / 100.0
    expected = tuple(round(side * factor) for side in capture_size)
    assert resolve_render_size(capture_size, config) == expected


@pytest.mark.parametrize("legacy_value", ["0.75", "2K", "3K", "1K", "42%", None])
def test_render_scale_rejects_legacy_numeric_and_short_aliases(legacy_value):
    config = render_size_config_from_settings({"Render Size Policy": "scaled", "Render Scale": legacy_value})

    assert config.scale_factor == "100%"
    assert resolve_render_size((3840, 2160), config) == (3840, 2160)


def test_resolve_render_size_fixed_uses_configured_size():
    config = RenderSizeConfig(policy=RenderSizePolicy.FIXED, fixed_width=1280, fixed_height=720, align=16)

    assert resolve_render_size((3840, 2160), config) == (1280, 720)


def test_resolve_render_size_dynamic_caps_pixels():
    config = RenderSizeConfig(
        policy=RenderSizePolicy.DYNAMIC,
        max_pixels=1280 * 720,
        min_dimension=360,
        align=16,
    )

    width, height = resolve_render_size((3840, 2160), config)

    assert width <= 1280
    assert height <= 720
    assert width % 16 == 0
    assert height % 16 == 0


def test_runtime_output_size_text_validates_size():
    assert runtime_output_size_text((1920, 1080)) == "1920x1080"
    assert runtime_output_size_text(None) == "unknown"
    with pytest.raises(ValueError):
        runtime_output_size_text((0, 1080))


@pytest.mark.parametrize(("old", "new"), [
    ("4K / 100%", "100%"), ("3K / 85%", "85%"), ("2K / 75%", "75%"), ("1K / 50%", "50%"),
])
def test_render_scale_roundtrips_old_settings_as_percentages(tmp_path, old, new):
    import yaml

    settings = {"Render Scale": old, "Render Size Policy": "scaled", "Depth Resolution": 336}
    original = yaml.safe_dump(settings)
    path = tmp_path / "settings.yaml"
    path.write_text(original, encoding="utf-8")
    config = render_size_config_from_settings(yaml.safe_load(path.read_text(encoding="utf-8")))
    assert config.scale_factor == new
    assert path.read_text(encoding="utf-8") == original
    settings["Render Scale"] = config.scale_factor
    path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    reloaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert render_size_config_from_settings(reloaded) == config
    assert reloaded["Depth Resolution"] == 336
