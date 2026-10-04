from utils.xr_headset_presets import (
    DEFAULT_XR_HEADSET_MODEL,
    XR_HEADSET_HORIZONTAL_FOV_DEG,
    display_to_xr_headset,
    resolve_xr_headset_preset,
    xr_headset_options,
    xr_headset_to_display,
)


def test_xr_headset_presets_match_reference_screen_sizes():
    pico = resolve_xr_headset_preset("Pico 4 / 4 Ultra")
    quest = resolve_xr_headset_preset("Meta Quest 3")
    xreal = resolve_xr_headset_preset("XREAL Air / Air 2 / Pro")

    assert DEFAULT_XR_HEADSET_MODEL == "Pico 4 / 4 Ultra"
    assert XR_HEADSET_HORIZONTAL_FOV_DEG == 60.0
    assert pico.resolution_tier_k == 4
    assert resolve_xr_headset_preset("Pimax Crystal / Light").resolution_tier_k == 8
    assert resolve_xr_headset_preset("Meta Quest 2").resolution_tier_k == 2
    assert resolve_xr_headset_preset("Meta Quest 2").recommended_render_scale == 1.24
    assert quest.recommended_render_scale == 1.24
    assert resolve_xr_headset_preset("Meta Quest 3S").recommended_render_scale == 1.09
    assert resolve_xr_headset_preset("Meta Quest Pro").recommended_render_scale == 1.23
    assert pico.recommended_render_scale == 1.44
    assert resolve_xr_headset_preset("Pico Neo 3").recommended_render_scale == 1.25
    assert (pico.distance_m, pico.width_m, pico.height_m, pico.diagonal_in) == (20.0, 23.09, 12.99, 1043)
    assert (quest.distance_m, quest.width_m, quest.height_m, quest.diagonal_in) == (1.3, 1.50, 0.84, 68)
    assert (xreal.distance_m, xreal.width_m, xreal.height_m, xreal.diagonal_in) == (4.0, 4.62, 2.60, 209)


def test_xr_headset_dropdown_options_are_localized_and_save_stable_keys():
    en_value = xr_headset_to_display("XREAL Air / Air 2 / Pro", "EN")
    cn_value = xr_headset_to_display("XREAL Air / Air 2 / Pro", "CN")

    assert en_value == "XREAL Air / Air 2 / Pro"
    assert cn_value == "XREAL Air / Air 2 / Pro"
    assert display_to_xr_headset(cn_value) == "XREAL Air / Air 2 / Pro"
    assert "Pico 4 / 4 Ultra" in xr_headset_options("CN")
    assert "Meta Quest 3S" in xr_headset_options("EN")
