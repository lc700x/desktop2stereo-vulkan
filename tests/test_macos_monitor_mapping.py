from capture.geometry import match_mss_monitor_to_rect


MSS_MONITORS = [
    {"left": 0, "top": 0, "width": 5760, "height": 1200},
    {"left": 0, "top": 0, "width": 1920, "height": 1080},  # iMac
    {"left": 1920, "top": 0, "width": 1920, "height": 1200},  # DELL
    {"left": 3840, "top": 0, "width": 1920, "height": 1080},  # VITURE
]

SCK_RECTS = [
    (3840, 0, 1920, 1080),  # VITURE is first in ScreenCaptureKit order.
    (0, 0, 1920, 1080),
    (1920, 0, 1920, 1200),
]


def test_sck_order_does_not_change_mss_input_monitor_selection():
    assert match_mss_monitor_to_rect(1, MSS_MONITORS, SCK_RECTS) == 1


def test_sck_monitor_mapping_does_not_fall_back_when_bounds_do_not_match():
    assert match_mss_monitor_to_rect(
        1,
        MSS_MONITORS,
        [(3840, 0, 1920, 1080), (1920, 0, 1920, 1200)],
    ) is None
