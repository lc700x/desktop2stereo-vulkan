from __future__ import annotations

from utils.display_info import (
    DisplayInfo,
    _windows_instance_key,
    _windows_monitor_instance_id,
    classify_windows_output_technology,
    display_identity_record,
    match_display_index,
    resolve_display_capture_index,
    resolve_glfw_monitor_index,
    resolve_windows_fullscreen_policy,
)


class _Size:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height


class _Mode:
    def __init__(self, width: int, height: int) -> None:
        self.size = _Size(width, height)


class _Glfw:
    monitors = ["primary", "second", "third"]
    geometry = {
        "primary": ((0, 0), (3840, 2160)),
        "second": ((3840, 0), (1920, 1080)),
        "third": ((5760, 0), (3840, 2160)),
    }

    @classmethod
    def get_monitors(cls):
        return cls.monitors

    @classmethod
    def get_monitor_pos(cls, monitor):
        return cls.geometry[monitor][0]

    @classmethod
    def get_video_mode(cls, monitor):
        return _Mode(*cls.geometry[monitor][1])

    @staticmethod
    def get_monitor_name(_monitor):
        return b"Generic PnP Monitor"


DISPLAYS = [
    DisplayInfo(3, 1, 0, 0, 3840, 2160, True, "hp-id", "HP Z27s", model="HP Z27s"),
    DisplayInfo(1, 2, 3840, 0, 1920, 1080, False, "dell-id", "DELL E2215HV", model="DELL E2215HV"),
    DisplayInfo(2, 3, 5760, 0, 3840, 2160, False, "viture-id", "VITURE", model="VITURE"),
]


def test_glfw_mapping_uses_display_identity_instead_of_mss_order() -> None:
    assert resolve_glfw_monitor_index(2, _Glfw, DISPLAYS) == 2
    assert resolve_glfw_monitor_index(1, _Glfw, DISPLAYS) == 1
    assert resolve_glfw_monitor_index(3, _Glfw, DISPLAYS) == 0


def test_display_matching_prefers_model_and_serial_for_duplicate_models() -> None:
    target = DisplayInfo(2, 2, 1920, 0, 1920, 1080, model="VITURE", serial="B")
    candidates = [
        DisplayInfo(1, 1, 0, 0, 1920, 1080, model="VITURE", serial="A"),
        DisplayInfo(2, 2, 1920, 0, 1920, 1080, model="VITURE", serial="B"),
    ]

    assert match_display_index(target, candidates) == 1


def test_display_matching_falls_back_to_geometry_when_names_are_generic() -> None:
    target = DisplayInfo(2, 3, 5760, 0, 3840, 2160, name="Generic PnP Monitor")
    candidates = [
        DisplayInfo(1, 1, 0, 0, 3840, 2160, name="Generic PnP Monitor"),
        DisplayInfo(2, 2, 5760, 0, 3840, 2160, name="Generic PnP Monitor"),
    ]

    assert match_display_index(target, candidates) == 1


def test_saved_identity_survives_capture_index_reordering() -> None:
    previous = DisplayInfo(
        2,
        3,
        5760,
        0,
        3840,
        2160,
        stable_id="viture-id",
        model="VITURE",
        serial="2290649088",
    )
    reordered = [
        DisplayInfo(1, 1, 0, 0, 3840, 2160, stable_id="hp-id", model="HP Z27s"),
        DisplayInfo(2, 2, 3840, 0, 1920, 1080, stable_id="dell-id", model="DELL"),
        DisplayInfo(
            3,
            3,
            5760,
            0,
            3840,
            2160,
            stable_id="viture-id",
            model="VITURE",
            serial="2290649088",
        ),
    ]

    identity = display_identity_record(previous)
    assert "display_number" not in identity
    assert "device_display_number" not in identity
    assert resolve_display_capture_index(2, identity, reordered) == 3


def test_missing_saved_identity_never_falls_back_to_stale_index() -> None:
    missing = {"stable_id": "disconnected-id", "model": "Disconnected"}
    assert resolve_display_capture_index(2, missing, DISPLAYS) is None


def test_legacy_numeric_display_selection_remains_supported() -> None:
    assert resolve_display_capture_index(2, None, DISPLAYS) == 2


def test_windows_output_technology_classification() -> None:
    assert classify_windows_output_technology(10) == "physical"
    assert classify_windows_output_technology(0x80000000) == "physical"
    assert classify_windows_output_technology(15) == "remote"
    assert classify_windows_output_technology(16) == "indirect"
    assert classify_windows_output_technology(17) == "virtual"
    assert classify_windows_output_technology(0xFFFFFFFF) == "unknown"


def test_windows_monitor_path_converts_to_pnp_instance_id() -> None:
    path = r"\\?\DISPLAY#MTT1337#1&33320f0&0&UID256#{device-interface-guid}"
    assert _windows_monitor_instance_id(path) == (
        r"DISPLAY\MTT1337\1&33320f0&0&UID256"
    )


def test_windows_fullscreen_policy_uses_dwm_for_physical_output() -> None:
    display = DisplayInfo(
        1, 1, 0, 0, 3840, 2160, display_kind="physical"
    )
    policy, target = resolve_windows_fullscreen_policy(1, [display])
    assert policy == "capture_compatible"
    assert target is display


def test_windows_fullscreen_policy_uses_dwm_for_virtual_output() -> None:
    display = DisplayInfo(
        2, 2, 3840, 0, 3840, 2400, display_kind="virtual"
    )
    policy, target = resolve_windows_fullscreen_policy(2, [display])
    assert policy == "capture_compatible"
    assert target is display


def test_windows_fullscreen_policy_falls_back_to_dwm_when_unmatched() -> None:
    policy, target = resolve_windows_fullscreen_policy(9, DISPLAYS)
    assert policy == "capture_compatible"
    assert target is None


def test_windows_mss_and_wmi_ids_normalize_to_the_same_instance() -> None:
    mss_id = r"\\?\DISPLAY#MTT1337#1&33320f0&0&UID256#{device-interface-guid}"
    wmi_id = r"DISPLAY\MTT1337\1&33320f0&0&UID256_0"

    assert _windows_instance_key(mss_id) == _windows_instance_key(wmi_id)
