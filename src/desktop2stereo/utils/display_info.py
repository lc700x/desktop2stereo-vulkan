from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
import os
import re
import subprocess
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class DisplayInfo:
    capture_index: int
    display_number: int
    left: int
    top: int
    width: int
    height: int
    is_primary: bool = False
    stable_id: str | None = None
    name: str | None = None
    manufacturer: str | None = None
    model: str | None = None
    serial: str | None = None
    device_display_number: int | None = None
    output_technology: int | None = None
    display_kind: str = "unknown"
    display_kind_source: str = "unknown"
    monitor_device_path: str | None = None

    @property
    def rect(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.left + self.width, self.top + self.height

    @property
    def label_name(self) -> str | None:
        return self.model or self.name

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def enumerate_displays(os_name: str | None = None) -> list[DisplayInfo]:
    platform_name = os_name or _platform_name()
    displays = _mss_displays()
    if not displays:
        return []
    if platform_name == "Windows":
        displays = _enrich_windows(displays)
    elif platform_name == "Darwin":
        displays = _enrich_macos(displays)
    else:
        displays = _enrich_linux(displays)
    return _assign_display_numbers(displays, platform_name)


def display_identity_record(
    display: DisplayInfo | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Return the stable, serializable fields used to find a display again."""
    if display is None:
        return None
    values = display.as_dict() if isinstance(display, DisplayInfo) else dict(display)
    keys = (
        "stable_id",
        "name",
        "manufacturer",
        "model",
        "serial",
        "left",
        "top",
        "width",
        "height",
    )
    return {key: values[key] for key in keys if values.get(key) is not None}


def resolve_display_capture_index(
    saved_index: Any,
    identity: Mapping[str, Any] | None,
    displays: Iterable[DisplayInfo] | None = None,
) -> int | None:
    """Resolve a saved identity, returning None when that display is unavailable."""
    try:
        fallback_index = max(1, int(saved_index))
    except (TypeError, ValueError):
        fallback_index = 1
    if not isinstance(identity, Mapping) or not identity:
        return fallback_index

    known = list(displays) if displays is not None else enumerate_displays()
    if not known:
        return None

    def integer(key: str, fallback: int = 0) -> int:
        try:
            return int(identity.get(key, fallback))
        except (TypeError, ValueError):
            return fallback

    target = DisplayInfo(
        capture_index=fallback_index,
        # Display numbers are transient; identity matching must never reuse them.
        display_number=0,
        left=integer("left"),
        top=integer("top"),
        width=integer("width"),
        height=integer("height"),
        stable_id=str(identity.get("stable_id") or "").strip() or None,
        name=str(identity.get("name") or "").strip() or None,
        manufacturer=str(identity.get("manufacturer") or "").strip() or None,
        model=str(identity.get("model") or "").strip() or None,
        serial=str(identity.get("serial") or "").strip() or None,
    )
    matched = match_display_index(target, known)
    return int(known[matched].capture_index) if matched is not None else None


def resolve_glfw_monitor_index(
    capture_index: int,
    glfw: Any,
    displays: Iterable[DisplayInfo] | None = None,
) -> int:
    glfw_monitors = list(glfw.get_monitors() or [])
    if not glfw_monitors:
        return 0
    known = list(displays) if displays is not None else enumerate_displays()
    target = next((item for item in known if item.capture_index == int(capture_index)), None)
    if target is None:
        return _clamp_monitor_index(capture_index, len(glfw_monitors))

    candidates: list[DisplayInfo] = []
    for index, monitor in enumerate(glfw_monitors, start=1):
        left, top = glfw.get_monitor_pos(monitor)
        mode = glfw.get_video_mode(monitor)
        width, height = int(mode.size.width), int(mode.size.height)
        mapped = _display_by_geometry(known, int(left), int(top), width, height)
        if mapped is not None:
            candidates.append(mapped)
            continue
        raw_name = glfw.get_monitor_name(monitor)
        if isinstance(raw_name, bytes):
            raw_name = raw_name.decode("utf-8", errors="replace")
        candidates.append(
            DisplayInfo(
                capture_index=index,
                display_number=index,
                left=int(left),
                top=int(top),
                width=width,
                height=height,
                name=str(raw_name or "").strip() or None,
            )
        )

    matched = match_display_index(target, candidates)
    if matched is not None:
        return matched
    return _clamp_monitor_index(capture_index, len(glfw_monitors))


def match_display_index(target: DisplayInfo, candidates: Iterable[DisplayInfo]) -> int | None:
    candidate_list = list(candidates)
    stable_id = _normalized_text(target.stable_id)
    if stable_id:
        for index, candidate in enumerate(candidate_list):
            if _normalized_text(candidate.stable_id) == stable_id:
                return index

    target_model = _normalized_text(target.model or target.name)
    target_serial = _normalized_text(target.serial)
    if target_model:
        same_model = [
            (index, candidate)
            for index, candidate in enumerate(candidate_list)
            if _normalized_text(candidate.model or candidate.name) == target_model
        ]
        if target_serial:
            for index, candidate in same_model:
                if _normalized_text(candidate.serial) == target_serial:
                    return index
        if len(same_model) == 1:
            return same_model[0][0]
        for index, candidate in same_model:
            if candidate.rect == target.rect:
                return index
        for index, candidate in same_model:
            if candidate.display_number == target.display_number:
                return index

    for index, candidate in enumerate(candidate_list):
        if candidate.rect == target.rect:
            return index
    return None


def _mss_displays() -> list[DisplayInfo]:
    try:
        import mss

        with mss.mss() as capture:
            monitors = list(capture.monitors[1:])
    except Exception:
        return []
    return [
        DisplayInfo(
            capture_index=index,
            display_number=index,
            left=int(monitor["left"]),
            top=int(monitor["top"]),
            width=int(monitor["width"]),
            height=int(monitor["height"]),
            is_primary=bool(monitor.get("is_primary", False)),
            stable_id=str(monitor.get("unique_id") or "").strip() or None,
            name=str(monitor.get("name") or "").strip() or None,
        )
        for index, monitor in enumerate(monitors, start=1)
    ]


def _enrich_windows(displays: list[DisplayInfo]) -> list[DisplayInfo]:
    metadata = _windows_wmi_metadata()
    display_config = _windows_display_config_metadata()

    # Build geometry-based lookup from display_config source rects.
    # mss uses (left, top, width, height); display_config stores
    # (left, top, right, bottom) from QueryDisplayConfig source modes.
    config_rect_to_number: dict[tuple[int, int, int, int], int] = {}
    for number, cfg in display_config.items():
        r = cfg.get("rect")
        if r is not None:
            config_rect_to_number[r] = number

    # Also build win32api lookup if available (same rect format).
    win32_rect_to_number: dict[tuple[int, int, int, int], int] = {}
    try:
        import win32api

        for handle, _hdc, rect in win32api.EnumDisplayMonitors():
            info = win32api.GetMonitorInfo(handle)
            number = _windows_display_number(info.get("Device", ""))
            if number is not None:
                win32_rect_to_number[tuple(int(v) for v in rect)] = number
    except Exception:
        pass

    enriched = []
    for display in displays:
        monitor_metadata = metadata.get(_windows_instance_key(display.stable_id), {})
        # Match display to Windows display number by rect.
        # display.rect = (left, top, left+width, top+height) = (l, t, r, b)
        device_display_number = (
            win32_rect_to_number.get(display.rect)
            or config_rect_to_number.get(display.rect)
        )
        target_metadata = display_config.get(device_display_number, {})
        wmi_model = monitor_metadata.get("model")
        friendly_name = target_metadata.get("monitor_friendly_name")
        mss_name = display.name or ""
        is_generic = mss_name.lower().replace(" ", "").startswith("genericpnp")
        # mss EDID > display config friendly > WMI model
        if is_generic:
            display_name = friendly_name or wmi_model or mss_name
        else:
            display_name = mss_name or friendly_name or wmi_model
        enriched.append(
            replace(
                display,
                stable_id=monitor_metadata.get("stable_id") or display.stable_id,
                name=display_name or mss_name,
                manufacturer=monitor_metadata.get("manufacturer"),
                model=wmi_model or friendly_name,
                serial=monitor_metadata.get("serial"),
                device_display_number=device_display_number,
                output_technology=target_metadata.get("output_technology"),
                display_kind=target_metadata.get("display_kind", "unknown"),
                display_kind_source=target_metadata.get(
                    "display_kind_source", "unknown"
                ),
                monitor_device_path=target_metadata.get("monitor_device_path"),
            )
        )
    return enriched


def resolve_windows_fullscreen_policy(
    capture_index: int,
    displays: Iterable[DisplayInfo] | None = None,
) -> tuple[str, DisplayInfo | None]:
    """Choose capture-compatible DWM for physical or unclassified targets."""
    known = list(displays) if displays is not None else enumerate_displays("Windows")
    target = next(
        (item for item in known if item.capture_index == int(capture_index)),
        None,
    )
    if target is None or target.display_kind in {"physical", "unknown"}:
        return "capture_compatible", target
    return "exclusive", target


def classify_windows_output_technology(value: Any) -> str:
    """Classify a QueryDisplayConfig output target without name heuristics."""
    try:
        technology = int(value) & 0xFFFFFFFF
    except (TypeError, ValueError):
        return "unknown"
    if technology == 17:
        return "virtual"
    if technology == 16:
        return "indirect"
    if technology == 15:
        return "remote"
    if technology == 0x80000000 or 0 <= technology <= 14:
        return "physical"
    return "unknown"


def _windows_monitor_instance_id(monitor_device_path: str | None) -> str | None:
    text = str(monitor_device_path or "").strip()
    if not text:
        return None
    if text.startswith("\\\\?\\"):
        text = text[4:]
    text = text.split("#{", 1)[0]
    return text.replace("#", "\\") or None


def _windows_parent_device_id(monitor_device_path: str | None) -> str | None:
    if os.name != "nt":
        return None
    instance_id = _windows_monitor_instance_id(monitor_device_path)
    if not instance_id:
        return None
    try:
        import ctypes
        from ctypes import wintypes

        cfgmgr32 = ctypes.windll.cfgmgr32
        device = wintypes.ULONG()
        parent = wintypes.ULONG()
        if int(cfgmgr32.CM_Locate_DevNodeW(ctypes.byref(device), instance_id, 0)) != 0:
            return None
        if int(cfgmgr32.CM_Get_Parent(ctypes.byref(parent), device, 0)) != 0:
            return None
        size = wintypes.ULONG()
        if int(cfgmgr32.CM_Get_Device_ID_Size(ctypes.byref(size), parent, 0)) != 0:
            return None
        output = ctypes.create_unicode_buffer(int(size.value) + 1)
        if int(cfgmgr32.CM_Get_Device_IDW(parent, output, len(output), 0)) != 0:
            return None
        return str(output.value or "").strip() or None
    except Exception:
        return None


def _windows_virtual_display_parent(
    monitor_device_path: str | None,
) -> tuple[bool | None, str | None]:
    """Identify virtual adapters that advertise a physical connector type."""
    parent_id = _windows_parent_device_id(monitor_device_path)
    if not parent_id:
        return None, None
    try:
        import winreg

        key_path = rf"SYSTEM\CurrentControlSet\Enum\{parent_id}"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
            values: dict[str, Any] = {}
            for name in ("HardwareID", "DeviceDesc", "FriendlyName", "Mfg", "Service"):
                try:
                    values[name] = winreg.QueryValueEx(key, name)[0]
                except OSError:
                    continue
    except Exception:
        values = {}

    fields = [parent_id]
    for value in values.values():
        if isinstance(value, (list, tuple)):
            fields.extend(str(item) for item in value)
        else:
            fields.append(str(value))
    evidence = " ".join(fields).casefold()
    virtual_tokens = (
        "virtual display",
        "virtualdisplay",
        "mttvdd",
        "iddsample",
        "indirect display",
        "spacedesk",
        "parsec",
        "usbmmidd",
    )
    is_root_display = parent_id.upper().startswith("ROOT\\DISPLAY\\")
    is_virtual = is_root_display or any(token in evidence for token in virtual_tokens)
    return is_virtual, parent_id


def _windows_display_config_metadata() -> dict[int, dict[str, Any]]:
    """Return active Windows display targets keyed by their DISPLAY number."""
    if os.name != "nt":
        return {}
    try:
        import ctypes
        from ctypes import wintypes

        class LUID(ctypes.Structure):
            _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]

        class RATIONAL(ctypes.Structure):
            _fields_ = [("Numerator", wintypes.UINT), ("Denominator", wintypes.UINT)]

        class REGION(ctypes.Structure):
            _fields_ = [("cx", wintypes.UINT), ("cy", wintypes.UINT)]

        class PATH_SOURCE_INFO(ctypes.Structure):
            _fields_ = [
                ("adapterId", LUID),
                ("id", wintypes.UINT),
                ("modeInfoIdx", wintypes.UINT),
                ("statusFlags", wintypes.UINT),
            ]

        class PATH_TARGET_INFO(ctypes.Structure):
            _fields_ = [
                ("adapterId", LUID),
                ("id", wintypes.UINT),
                ("modeInfoIdx", wintypes.UINT),
                ("outputTechnology", wintypes.UINT),
                ("rotation", wintypes.UINT),
                ("scaling", wintypes.UINT),
                ("refreshRate", RATIONAL),
                ("scanLineOrdering", wintypes.UINT),
                ("targetAvailable", wintypes.BOOL),
                ("statusFlags", wintypes.UINT),
            ]

        class PATH_INFO(ctypes.Structure):
            _fields_ = [
                ("sourceInfo", PATH_SOURCE_INFO),
                ("targetInfo", PATH_TARGET_INFO),
                ("flags", wintypes.UINT),
            ]

        class VIDEO_SIGNAL_INFO(ctypes.Structure):
            _fields_ = [
                ("pixelRate", ctypes.c_uint64),
                ("hSyncFreq", RATIONAL),
                ("vSyncFreq", RATIONAL),
                ("activeSize", REGION),
                ("totalSize", REGION),
                ("videoStandard", wintypes.UINT),
                ("scanLineOrdering", wintypes.UINT),
            ]

        class TARGET_MODE(ctypes.Structure):
            _fields_ = [("targetVideoSignalInfo", VIDEO_SIGNAL_INFO)]

        class POINTL(ctypes.Structure):
            _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]

        class SOURCE_MODE(ctypes.Structure):
            _fields_ = [
                ("width", wintypes.UINT),
                ("height", wintypes.UINT),
                ("pixelFormat", wintypes.UINT),
                ("position", POINTL),
            ]

        class DESKTOP_IMAGE_INFO(ctypes.Structure):
            _fields_ = [
                ("PathSourceSize", POINTL),
                ("DesktopImageRegion", wintypes.RECT),
                ("DesktopImageClip", wintypes.RECT),
            ]

        class MODE_UNION(ctypes.Union):
            _fields_ = [
                ("targetMode", TARGET_MODE),
                ("sourceMode", SOURCE_MODE),
                ("desktopImageInfo", DESKTOP_IMAGE_INFO),
            ]

        class MODE_INFO(ctypes.Structure):
            _anonymous_ = ("mode",)
            _fields_ = [
                ("infoType", wintypes.UINT),
                ("id", wintypes.UINT),
                ("adapterId", LUID),
                ("mode", MODE_UNION),
            ]

        class DEVICE_INFO_HEADER(ctypes.Structure):
            _fields_ = [
                ("type", wintypes.UINT),
                ("size", wintypes.UINT),
                ("adapterId", LUID),
                ("id", wintypes.UINT),
            ]

        class SOURCE_DEVICE_NAME(ctypes.Structure):
            _fields_ = [
                ("header", DEVICE_INFO_HEADER),
                ("viewGdiDeviceName", wintypes.WCHAR * 32),
            ]

        class TARGET_DEVICE_NAME(ctypes.Structure):
            _fields_ = [
                ("header", DEVICE_INFO_HEADER),
                ("flags", wintypes.UINT),
                ("outputTechnology", wintypes.UINT),
                ("edidManufactureId", wintypes.USHORT),
                ("edidProductCodeId", wintypes.USHORT),
                ("connectorInstance", wintypes.UINT),
                ("monitorFriendlyDeviceName", wintypes.WCHAR * 64),
                ("monitorDevicePath", wintypes.WCHAR * 128),
            ]

        user32 = ctypes.windll.user32
        qdc_only_active_paths = 0x00000002
        error_insufficient_buffer = 122
        path_count = wintypes.UINT()
        mode_count = wintypes.UINT()
        for _attempt in range(3):
            result = int(
                user32.GetDisplayConfigBufferSizes(
                    qdc_only_active_paths,
                    ctypes.byref(path_count),
                    ctypes.byref(mode_count),
                )
            )
            if result != 0:
                return {}
            paths = (PATH_INFO * max(1, int(path_count.value)))()
            modes = (MODE_INFO * max(1, int(mode_count.value)))()
            result = int(
                user32.QueryDisplayConfig(
                    qdc_only_active_paths,
                    ctypes.byref(path_count),
                    paths,
                    ctypes.byref(mode_count),
                    modes,
                    None,
                )
            )
            if result == error_insufficient_buffer:
                continue
            if result != 0:
                return {}
            break
        else:
            return {}

        result_by_number: dict[int, dict[str, Any]] = {}
        for idx, path in enumerate(paths[: int(path_count.value)]):
            source = SOURCE_DEVICE_NAME()
            source.header.type = 1
            source.header.size = ctypes.sizeof(SOURCE_DEVICE_NAME)
            source.header.adapterId = path.sourceInfo.adapterId
            source.header.id = path.sourceInfo.id
            if int(user32.DisplayConfigGetDeviceInfo(ctypes.byref(source))) != 0:
                continue
            display_number = _windows_display_number(source.viewGdiDeviceName)
            if display_number is None:
                continue

            target = TARGET_DEVICE_NAME()
            target.header.type = 2
            target.header.size = ctypes.sizeof(TARGET_DEVICE_NAME)
            target.header.adapterId = path.targetInfo.adapterId
            target.header.id = path.targetInfo.id
            target_result = int(
                user32.DisplayConfigGetDeviceInfo(ctypes.byref(target))
            )
            technology = int(path.targetInfo.outputTechnology) & 0xFFFFFFFF
            monitor_device_path = (
                str(target.monitorDevicePath or "").strip() or None
                if target_result == 0
                else None
            )
            display_kind = classify_windows_output_technology(technology)
            display_kind_source = "output_technology"
            virtual_parent, parent_id = _windows_virtual_display_parent(
                monitor_device_path
            )
            if virtual_parent is True:
                display_kind = "virtual"
                display_kind_source = f"pnp_parent:{parent_id}"
            friendly_name = (
                str(target.monitorFriendlyDeviceName or "").strip() or None
                if target_result == 0
                else None
            )
            source_rect = None
            if (
                0 <= int(path.sourceInfo.modeInfoIdx) < int(mode_count.value)
                and int(modes[int(path.sourceInfo.modeInfoIdx)].infoType) == 1
            ):
                sm = modes[int(path.sourceInfo.modeInfoIdx)].sourceMode
                source_rect = (
                    int(sm.position.x),
                    int(sm.position.y),
                    int(sm.position.x) + int(sm.width),
                    int(sm.position.y) + int(sm.height),
                )
            result_by_number[display_number] = {
                "output_technology": technology,
                "display_kind": display_kind,
                "display_kind_source": display_kind_source,
                "monitor_device_path": monitor_device_path,
                "monitor_friendly_name": friendly_name,
                "rect": source_rect,
            }
        return result_by_number
    except Exception:
        return {}


def _windows_wmi_metadata() -> dict[str, dict[str, str | None]]:
    try:
        import pythoncom
        import win32com.client
    except Exception:
        return {}
    pythoncom.CoInitialize()
    service = None
    rows = None
    row = None
    try:
        service = win32com.client.GetObject(r"winmgmts:\\.\root\wmi")
        rows = service.ExecQuery("SELECT * FROM WmiMonitorID")
        result = {}
        for row in rows:
            stable_id = str(row.InstanceName).removesuffix("_0")
            result[_windows_instance_key(stable_id)] = {
                "stable_id": stable_id,
                "manufacturer": _decode_wmi_chars(row.ManufacturerName),
                "model": _decode_wmi_chars(row.UserFriendlyName),
                "serial": _decode_wmi_chars(row.SerialNumberID),
            }
        return result
    except Exception:
        return {}
    finally:
        row = None
        rows = None
        service = None
        pythoncom.CoUninitialize()


def _enrich_linux(displays: list[DisplayInfo]) -> list[DisplayInfo]:
    try:
        output = subprocess.check_output(
            ["xrandr", "--listactivemonitors"],
            stderr=subprocess.DEVNULL,
            timeout=2,
            text=True,
            encoding="utf-8",
        )
    except Exception:
        return displays
    pattern = re.compile(r"^\s*\d+:\s+[^ ]*\s+(\d+)/\d+x(\d+)/\d+([+-]\d+)([+-]\d+)\s+(.+)$")
    metadata = []
    for line in output.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        width, height, left, top, name = match.groups()
        metadata.append((int(left), int(top), int(width), int(height), name.strip()))
    return [
        replace(
            display,
            stable_id=f"xrandr:{item[4]}",
            name=item[4],
            model=item[4],
        )
        if (item := _metadata_by_geometry(metadata, display)) is not None
        else display
        for display in displays
    ]


def _enrich_macos(displays: list[DisplayInfo]) -> list[DisplayInfo]:
    try:
        import AppKit
        import Quartz
    except Exception:
        return displays
    metadata = []
    for screen in AppKit.NSScreen.screens():
        display_id = int(screen.deviceDescription().get("NSScreenNumber", 0))
        bounds = Quartz.CGDisplayBounds(display_id)
        left, top = int(bounds.origin.x), int(bounds.origin.y)
        width, height = int(bounds.size.width), int(bounds.size.height)
        name = str(getattr(screen, "localizedName", lambda: "")() or "").strip() or None
        vendor = int(Quartz.CGDisplayVendorNumber(display_id))
        model_number = int(Quartz.CGDisplayModelNumber(display_id))
        serial = int(Quartz.CGDisplaySerialNumber(display_id))
        stable_id = f"cg:{vendor}:{model_number}:{serial or display_id}"
        metadata.append((left, top, width, height, stable_id, name, vendor, model_number, serial))
    enriched = []
    for display in displays:
        item = _metadata_by_geometry(metadata, display)
        if item is None:
            enriched.append(display)
            continue
        enriched.append(
            replace(
                display,
                stable_id=item[4],
                name=item[5],
                manufacturer=str(item[6]),
                model=item[5] or str(item[7]),
                serial=str(item[8]) if item[8] else None,
            )
        )
    return enriched


def _assign_display_numbers(displays: list[DisplayInfo], platform_name: str) -> list[DisplayInfo]:
    if platform_name == "Windows":
        ordered = sorted(
            displays,
            key=lambda item: item.device_display_number or item.capture_index,
        )
    else:
        ordered = sorted(displays, key=lambda item: item.display_number)
    return [replace(display, display_number=index) for index, display in enumerate(ordered, start=1)]


def _display_by_geometry(
    displays: Iterable[DisplayInfo], left: int, top: int, width: int, height: int
) -> DisplayInfo | None:
    return next(
        (
            display
            for display in displays
            if display.left == left
            and display.top == top
            and display.width == width
            and display.height == height
        ),
        None,
    )


def _metadata_by_geometry(metadata: Iterable[tuple], display: DisplayInfo) -> tuple | None:
    return next(
        (
            item
            for item in metadata
            if tuple(item[:4]) == (display.left, display.top, display.width, display.height)
        ),
        None,
    )


def _windows_instance_key(value: str | None) -> str:
    text = str(value or "").upper().replace("#", "\\")
    marker = "DISPLAY\\"
    if marker in text:
        text = text[text.index(marker):]
    text = text.split("\\{", 1)[0].removesuffix("_0")
    return text


def _windows_display_number(device_name: str) -> int | None:
    suffix = str(device_name).upper().rsplit("DISPLAY", 1)[-1]
    return int(suffix) if suffix.isdigit() else None


def _decode_wmi_chars(values: Any) -> str | None:
    text = "".join(chr(int(value)) for value in values if int(value)).strip()
    return text or None


def _normalized_text(value: str | None) -> str:
    return " ".join(str(value or "").casefold().split())


def _clamp_monitor_index(monitor_index: int, monitor_count: int) -> int:
    return min(max(0, int(monitor_index) - 1), max(0, int(monitor_count) - 1))


def _platform_name() -> str:
    if os.name == "nt":
        return "Windows"
    if sys_platform := os.environ.get("D2S_PLATFORM_NAME"):
        return sys_platform
    import platform

    return platform.system()
