from __future__ import annotations

import math
from dataclasses import dataclass


DEFAULT_XR_HEADSET_MODEL = "Pico 4 / 4 Ultra"
XR_HEADSET_HORIZONTAL_FOV_DEG = 60.0


@dataclass(frozen=True)
class XRHeadsetPreset:
    key: str
    category: str
    display_name: str
    distance_m: float
    # Application sampling tier, not a claim about a vendor's exact panel
    # specification. It is selected by the GUI headset model.
    resolution_tier_k: int
    # Multiplier of the OpenXR runtime's recommended eye image size used by
    # the Headset optimized mode. The runtime's recommended/max extents still
    # clamp the actual swapchain dimensions.
    recommended_render_scale: float = 1.0

    @property
    def width_m(self) -> float:
        return round(self.distance_m * 2.0 * math.tan(math.radians(XR_HEADSET_HORIZONTAL_FOV_DEG) * 0.5), 2)

    @property
    def height_m(self) -> float:
        return round(self.width_m * 9.0 / 16.0, 2)

    @property
    def diagonal_in(self) -> int:
        return round(math.hypot(self.width_m, self.height_m) / 0.0254)


XR_HEADSET_PRESETS = (
    # Use published panel-to-default-eye-buffer ratios in Headset optimized
    # mode where vendors document both values. Other models retain the runtime
    # recommendation; projection sampling adapts to the live swapchain.
    XRHeadsetPreset("Meta Quest 2", "vr", "Meta Quest 2", 1.3, 2, 1.24),
    XRHeadsetPreset("Meta Quest 3", "vr", "Meta Quest 3", 1.3, 4, 1.24),
    XRHeadsetPreset("Meta Quest 3S", "vr", "Meta Quest 3S", 1.3, 4, 1.09),
    XRHeadsetPreset("Meta Quest Pro", "vr", "Meta Quest Pro", 1.1, 4, 1.23),
    # PICO documents 2160px/eye against a 1504px default eye buffer for PICO 4.
    XRHeadsetPreset("Pico 4 / 4 Ultra", "vr", "Pico 4 / 4 Ultra", 20.0, 4, 1.44),
    # Neo 3's 1832x1920 panel/eye and ~1500x1500 default buffer give a
    # geometric-mean scale of about 1.25.
    XRHeadsetPreset("Pico Neo 3", "vr", "Pico Neo 3", 1.5, 4, 1.25),
    XRHeadsetPreset("HTC VIVE XR Elite", "vr", "HTC VIVE XR Elite", 20.0, 4),
    XRHeadsetPreset("HTC VIVE Focus 3 / Vision", "vr", "HTC VIVE Focus 3 / Vision", 1.5, 4),
    XRHeadsetPreset("HTC VIVE Pro / Cosmos", "vr", "HTC VIVE Pro / Cosmos", 1.5, 2),
    XRHeadsetPreset("Valve Index", "vr", "Valve Index", 1.5, 2),
    XRHeadsetPreset("Pimax Crystal / Light", "vr", "Pimax Crystal / Light", 1.0, 8),
    XRHeadsetPreset("Pimax 8K/5K/Artisan", "vr", "Pimax 8K/5K/Artisan", 1.5, 8),
    XRHeadsetPreset("DPVR E4 / E3", "vr", "DPVR E4 / E3", 1.0, 4),
    XRHeadsetPreset("Sony PS VR2", "vr", "Sony PS VR2", 2.0, 4),
    XRHeadsetPreset("Apple Vision Pro", "vr", "Apple Vision Pro", 1.5, 8),
    XRHeadsetPreset("Varjo Aero / XR-3", "vr", "Varjo Aero / XR-3", 1.5, 8),
    XRHeadsetPreset("Viture One / Pro / Lite", "ar", "Viture One / Pro / Lite", 3.0, 2),
    XRHeadsetPreset("RayNeo Air 2 / 3 / 2s / Plus", "ar", "RayNeo Air 2 / 3 / 2s / Plus", 4.0, 2),
    XRHeadsetPreset("XREAL Air / Air 2 / Pro", "ar", "XREAL Air / Air 2 / Pro", 4.0, 2),
    XRHeadsetPreset("Rokid Max / Max Pro / AR Lite", "ar", "Rokid Max / Max Pro / AR Lite", 6.0, 2),
    XRHeadsetPreset("Huawei Vision Glass", "ar", "Huawei Vision Glass", 4.0, 2),
    XRHeadsetPreset("Thunderobot Air 1S", "ar", "Thunderobot Air 1S", 4.0, 2),
)

_PRESETS_BY_KEY = {preset.key: preset for preset in XR_HEADSET_PRESETS}
_PRESETS_BY_DISPLAY = {preset.display_name: preset for preset in XR_HEADSET_PRESETS}
_PRESET_ALIASES = {
    "雷鸟 Air 2 / 3 / 2s / Plus": "RayNeo Air 2 / 3 / 2s / Plus",
    "雷神 Air 1S": "Thunderobot Air 1S",
    "华为 Vision Glass": "Huawei Vision Glass",
}

def resolve_xr_headset_preset(value: str | None) -> XRHeadsetPreset:
    text = str(value or "").strip()
    text = _PRESET_ALIASES.get(text, text)
    preset = _PRESETS_BY_KEY.get(text) or _PRESETS_BY_DISPLAY.get(text)
    return preset or _PRESETS_BY_KEY[DEFAULT_XR_HEADSET_MODEL]


def format_xr_headset_option(preset: XRHeadsetPreset, locale: str = "EN") -> str:
    return preset.display_name


def xr_headset_options(locale: str = "EN") -> list[str]:
    return [format_xr_headset_option(preset, locale) for preset in XR_HEADSET_PRESETS]


def xr_headset_to_display(value: str | None, locale: str = "EN") -> str:
    return format_xr_headset_option(resolve_xr_headset_preset(value), locale)


def display_to_xr_headset(value: str | None) -> str:
    text = str(value or "").strip()
    for preset in XR_HEADSET_PRESETS:
        if text == preset.key or text == preset.display_name:
            return preset.key
        if f"/ {preset.display_name} -" in text:
            return preset.key
    return resolve_xr_headset_preset(text).key
