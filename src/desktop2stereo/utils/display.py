import platform
import re


OS_NAME = platform.system()


def get_fps(window_title="", monitor_index=None):
    """Return monitor refresh rate for the target monitor.
    If window_title is set, finds the monitor containing that window.
    If monitor_index is set (mss 1-based), uses that monitor directly.
    Falls back to primary monitor. Returns 60 if detection fails."""
    try:
        if OS_NAME == "Windows":
            return _get_fps_windows(window_title, monitor_index)
        elif OS_NAME == "Darwin":
            return _get_fps_macos(window_title, monitor_index)
        else:
            return _get_fps_linux(window_title, monitor_index)
    except Exception:
        return 60


def get_monitor_size(monitor_index=None):
    """Return (width, height) for an mss monitor index, falling back to primary."""
    try:
        import mss
        with mss.MSS() as sct:
            if monitor_index is None or monitor_index <= 0 or monitor_index >= len(sct.monitors):
                monitor_index = 1
            mon = sct.monitors[monitor_index]
            return int(mon["width"]), int(mon["height"])
    except Exception:
        return 3840, 2160


def compute_output_resolution(setting_value, display_mode, input_monitor_index, stereo_monitor_index, use_stereo_monitor=True):
    """Compute the source processing size used before depth inference.

    Integer values keep the legacy meaning of source eye height. WxH values and
    Auto describe the captured source / single-eye processing size regardless
    of the packed display mode. Full-SBS and Full-TAB expand only when the two
    completed eye images are packed for display; Half modes compress only at
    that same final packing stage.

    When ``setting_value`` is Auto, Local Viewer/OpenXR/streamers follow the
    captured input monitor. Same-monitor Local Viewer stereo output considers
    the input and output dimensions, avoiding an expensive pre-upscale before
    depth/stereo when the packed image is presented fullscreen.
    """
    explicit_size = _parse_resolution_size(setting_value)
    if explicit_size is not None:
        out_w, out_h = explicit_size
        return _source_processing_size(out_w, out_h)

    try:
        if isinstance(setting_value, str):
            value = setting_value.strip()
            if value and value.lower() != "auto":
                parsed = int(value)
                if parsed > 0:
                    return _even_height(parsed)
        elif setting_value:
            parsed = int(setting_value)
            if parsed > 0:
                return _even_height(parsed)
    except (TypeError, ValueError):
        pass

    in_w, in_h = get_monitor_size(input_monitor_index)
    if use_stereo_monitor and stereo_monitor_index:
        out_w, out_h = get_monitor_size(stereo_monitor_index)
        source_w, source_h = _fit_source_to_output(in_w, in_h, out_w, out_h)
    else:
        source_w, source_h = in_w, in_h
    return _source_processing_size(source_w, source_h)


def _fit_source_to_output(input_width, input_height, output_width, output_height):
    in_w = max(1, int(input_width))
    in_h = max(1, int(input_height))
    out_w = max(1, int(output_width))
    out_h = max(1, int(output_height))
    scale = min(1.0, float(out_w) / float(in_w), float(out_h) / float(in_h))
    return _even_width(round(in_w * scale)), _even_height(round(in_h * scale))


def _parse_resolution_size(setting_value):
    if isinstance(setting_value, (tuple, list)) and len(setting_value) == 2:
        try:
            width = int(setting_value[0])
            height = int(setting_value[1])
            if width > 0 and height > 0:
                return width, height
        except (TypeError, ValueError):
            return None

    if not isinstance(setting_value, str):
        return None

    value = setting_value.strip().lower()
    match = re.fullmatch(r"(\d+)\s*[x*×]\s*(\d+)", value)
    if not match:
        return None

    width = int(match.group(1))
    height = int(match.group(2))
    if width <= 0 or height <= 0:
        return None
    return width, height


def _source_processing_size(source_width, source_height):
    return _even_width(source_width), _even_height(source_height)


def _even_width(value):
    return max(2, (int(value) // 2) * 2)


def _even_height(value):
    return max(2, (int(value) // 2) * 2)


def _get_device_name_from_mss_monitor(monitor_index):
    """Map mss monitor index (1-based) to win32api device name by matching rects."""
    import win32api
    import mss
    with mss.MSS() as sct:
        if monitor_index is None or monitor_index >= len(sct.monitors):
            monitor_index = 1
        target = sct.monitors[monitor_index]
        tl, tt, tr, tb = target['left'], target['top'], target['left'] + target['width'], target['top'] + target['height']

    monitors = win32api.EnumDisplayMonitors()
    for hmon, hdc, rect in monitors:
        if rect[0] == tl and rect[1] == tt:
            mi = win32api.GetMonitorInfo(hmon)
            return mi['Device']
    # Fallback: match by overlap
    for hmon, hdc, rect in monitors:
        if rect[0] <= tl < rect[2] and rect[1] <= tt < rect[3]:
            mi = win32api.GetMonitorInfo(hmon)
            return mi['Device']
    return win32api.EnumDisplayDevices(None, 0).DeviceName


def _get_fps_windows(window_title, monitor_index):
    import win32api
    import win32gui
    import mss

    device_name = None

    if window_title:
        hwnd = win32gui.FindWindow(None, window_title)
        if hwnd:
            r = win32gui.GetWindowRect(hwnd)
            wx, wy = (r[0] + r[2]) // 2, (r[1] + r[3]) // 2
            with mss.MSS() as sct:
                for i, mon in enumerate(sct.monitors):
                    if i == 0:
                        continue
                    if mon['left'] <= wx < mon['left'] + mon['width'] and mon['top'] <= wy < mon['top'] + mon['height']:
                        device_name = _get_device_name_from_mss_monitor(i)
                        break

    if device_name is None and monitor_index is not None:
        try:
            device_name = _get_device_name_from_mss_monitor(monitor_index)
        except Exception:
            pass

    if device_name is None:
        try:
            device_name = win32api.EnumDisplayDevices(None, 0).DeviceName
        except Exception:
            return 60

    try:
        settings = win32api.EnumDisplaySettings(device_name, -1)
        return settings.DisplayFrequency
    except Exception:
        return 60


def _get_display_id_for_window_macos(window_title):
    """Find the CGDirectDisplayID that contains the given window."""
    try:
        from Quartz import (
            CGWindowListCopyWindowInfo,
            kCGWindowListOptionOnScreenOnly,
            kCGNullWindowID,
            CGGetOnlineDisplayList,
            CGDisplayBounds,
        )
    except ImportError:
        return None

    info = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
    if not info:
        return None

    max_displays = 16
    display_ids, count = CGGetOnlineDisplayList(max_displays, None, None)
    display_bounds = {}
    for d_id in display_ids[:count]:
        bounds = CGDisplayBounds(d_id)
        display_bounds[d_id] = bounds

    for win_dict in info:
        name = win_dict.get('kCGWindowName', '')
        if window_title in name:
            bounds = win_dict.get('kCGWindowBounds', {})
            wx = bounds.get('X', 0) + bounds.get('Width', 0) / 2
            wy = bounds.get('Y', 0) + bounds.get('Height', 0) / 2
            for d_id, db in display_bounds.items():
                if (db.origin.x <= wx < db.origin.x + db.size.width and
                    db.origin.y <= wy < db.origin.y + db.size.height):
                    return d_id
            return display_ids[0] if count > 0 else None
    return None


def _get_fps_macos(window_title, monitor_index):
    try:
        from Quartz import (
            CGGetOnlineDisplayList,
            CGDisplayCopyDisplayMode,
            CGDisplayModeGetRefreshRate,
            CGDisplayBounds,
        )
        import mss
    except ImportError:
        return 60

    max_displays = 16
    display_ids, count = CGGetOnlineDisplayList(max_displays, None, None)
    if count == 0:
        return 60

    display_id = None

    if window_title:
        display_id = _get_display_id_for_window_macos(window_title)

    if display_id is None and monitor_index is not None and monitor_index > 0:
        with mss.MSS() as sct:
            if monitor_index < len(sct.monitors):
                tx, ty = sct.monitors[monitor_index]['left'], sct.monitors[monitor_index]['top']
                for d_id in display_ids[:count]:
                    bounds = CGDisplayBounds(d_id)
                    if abs(int(bounds.origin.x) - tx) <= 1 and abs(int(bounds.origin.y) - ty) <= 1:
                        display_id = d_id
                        break

    if display_id is None:
        display_id = display_ids[0]

    mode = CGDisplayCopyDisplayMode(display_id)
    if mode:
        hz = CGDisplayModeGetRefreshRate(mode)
        if hz > 0:
            return int(round(hz))
    return 60


def _get_fps_linux(window_title, monitor_index):
    import subprocess
    import re
    import mss

    target_left, target_top = None, None

    if window_title:
        try:
            result = subprocess.run(
                ['wmctrl', '-lG'],
                capture_output=True, text=True, timeout=3
            )
            for line in result.stdout.split('\n'):
                if window_title in line:
                    parts = line.split(None, 7)
                    if len(parts) >= 6:
                        wx = int(parts[2]) + int(parts[4]) // 2
                        wy = int(parts[3]) + int(parts[5]) // 2
                        with mss.MSS() as sct:
                            for i, mon in enumerate(sct.monitors):
                                if i == 0:
                                    continue
                                if (mon['left'] <= wx < mon['left'] + mon['width'] and
                                        mon['top'] <= wy < mon['top'] + mon['height']):
                                    target_left, target_top = mon['left'], mon['top']
                                    break
                    break
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    if target_left is None and monitor_index is not None and monitor_index > 0:
        with mss.MSS() as sct:
            if monitor_index < len(sct.monitors):
                target_left = sct.monitors[monitor_index]['left']
                target_top = sct.monitors[monitor_index]['top']

    try:
        result = subprocess.run(
            ['xrandr', '--current'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return 60

        current_is_target = (target_left is None)
        best_rate = None

        for line in result.stdout.split('\n'):
            out_match = re.match(r'^(\S+)\s+connected', line)
            if out_match:
                current_is_target = False
                pos_match = re.search(r'(\d+)x(\d+)\+(\d+)\+(\d+)', line)
                if pos_match:
                    ox, oy = int(pos_match.group(3)), int(pos_match.group(4))
                    is_primary = 'primary' in line
                    if target_left is not None:
                        current_is_target = (ox == target_left and oy == target_top)
                    elif is_primary:
                        current_is_target = True
                continue

            if current_is_target or target_left is None:
                rm = re.search(r'(\d+(?:\.\d+)?)\s*\*\+?', line)
                if rm:
                    rate = int(round(float(rm.group(1))))
                    if target_left is not None and current_is_target:
                        return rate
                    if best_rate is None:
                        best_rate = rate

        if best_rate is not None:
            return best_rate

        # Final fallback: any active mode
        for line in result.stdout.split('\n'):
            rm = re.search(r'(\d+(?:\.\d+)?)\s*\*', line)
            if rm:
                return int(round(float(rm.group(1))))
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return 60
