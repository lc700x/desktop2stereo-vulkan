"""Run one short, instrumented OpenXR Link leg on the connected headset.

Starts the real processing runtime with the current settings.yaml and emits
[FPSBreakdown] lines every few seconds. Extra environment overrides are passed
with --set KEY=VALUE so expensive OpenXR stages can be isolated without
editing settings.yaml.

Usage:
    src/python3_cuda/python.exe scripts/benchmark/run_openxr_profile.py \
        --seconds 40 --label control > outputs/openxr_ab/control.log 2>&1
    src/python3_cuda/python.exe scripts/benchmark/run_openxr_profile.py \
        --seconds 40 --label glow_off --set D2S_OPENXR_GLOW_MODE=off \
        > outputs/openxr_ab/glow_off.log 2>&1
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "src" / "desktop2stereo"
sys.path.insert(0, str(APP))
os.chdir(APP)
os.environ["PYTHONPATH"] = str(APP)

if sys.platform == "win32":
    from windows_dpi import set_per_monitor_dpi_v2

    set_per_monitor_dpi_v2()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seconds", type=float, default=40.0)
parser.add_argument("--label", default="")
parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
parser.add_argument("--interval", default="5")
args = parser.parse_args()

# Deterministic, clean process environment (mirrors run_windows_cuda.bat).
for name in (
    "D2S_FILAMENT_CONTROLLER_EYE_DIAGNOSTIC", "D2S_FILAMENT_EYE_DIAGNOSTIC",
    "D2S_FILAMENT_MULTIVIEW_LAYER_READBACK", "D2S_FILAMENT_MULTIVIEW",
    "D2S_FILAMENT_SHADER_DUMP_DIR", "D2S_FILAMENT_MULTIVIEW_PROJECTION_DIAGNOSTIC",
    "D2S_FILAMENT_PROJECTION_ONLY", "D2S_OPENXR_PROJECTION_ARRAY_EYE_DIAGNOSTIC",
    "D2S_OPENXR_SCREEN_QUAD_EYE_DIAGNOSTIC", "D2S_VULKAN_PROJECTION_COMPOSER_EYE_DIAGNOSTIC",
    "D2S_VULKAN_PROJECTION_QUALITY_CHAIN", "D2S_ROCM_DISABLE_OPENXR_OVERLAYS",
    "D2S_ROCM_ENABLE_OPENXR_OVERLAYS", "D2S_DEBUG", "D2S_BENCHMARK",
):
    os.environ.pop(name, None)

os.environ["D2S_FPS_BREAKDOWN"] = "1"
os.environ["D2S_FPS_BREAKDOWN_INTERVAL"] = args.interval
os.environ["D2S_FPS_BREAKDOWN_LIMIT"] = "0"
os.environ["D2S_CRASH_LOG_DIR"] = str(ROOT / "outputs" / "openxr_ab")

overrides: dict[str, str] = {}
for item in args.set:
    if "=" not in item:
        raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
    key, value = item.split("=", 1)
    os.environ[key] = value
    overrides[key] = value

import utils  # noqa: E402
from utils.settings import load_settings  # noqa: E402

settings = load_settings()

from stereo_runtime.hot_reload import StereoHotReloader  # noqa: E402

StereoHotReloader.poll_settings_snapshot_if_needed = lambda *_a, **_kw: None

try:
    from capture.backends.windows_capture_event import WindowsCaptureEventRunner  # noqa: E402

    WindowsCaptureEventRunner._start_keyboard_worker = lambda *_a, **_kw: None
except Exception:
    pass

from app_runtime.runtime_entry import run_processing_runtime  # noqa: E402

print(
    f"[XR-A/B] label={args.label or 'unnamed'} seconds={args.seconds} "
    f"overrides={overrides or '{}'}",
    flush=True,
)
print(
    "[XR-A/B] settings "
    f"run_mode={settings.get('Run Mode')} model={settings.get('Depth Model')} "
    f"depth_resolution={settings.get('Depth Resolution')} "
    f"render_scale={settings.get('Render Scale')} "
    f"xr_render={settings.get('XR Render')} xr_mode={settings.get('XR Render Mode')} "
    f"controller={settings.get('Controller Model')} "
    f"environment={settings.get('Environment Model')}",
    flush=True,
)
print(f"[XR-A/B] utils RUN_MODE={utils.RUN_MODE} DEVICE_INFO={utils.DEVICE_INFO}", flush=True)

try:
    rc = run_processing_runtime(max_seconds=args.seconds)
except Exception as exc:
    print(f"[XR-A/B] FAILED {type(exc).__name__}: {exc}", flush=True)
    raise
print(f"[XR-A/B] finished rc={rc}", flush=True)
raise SystemExit(int(rc or 0))
