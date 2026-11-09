"""Run one deterministic Local Viewer A/B leg on the real GPU/window.

Mirrors the production launch (same settings.yaml, same working directory,
same DPI setup) but starts the processing runtime directly and stops it after
a fixed wall-clock duration. Both legs therefore have identical startup and
shutdown behaviour; only the presenter optimization switches differ.

Usage:
    src/python3_cuda/python.exe scripts/benchmark/run_local_viewer_ab.py \
        --seconds 75 --legacy --label baseline > outputs/live_ab/baseline.log 2>&1
    src/python3_cuda/python.exe scripts/benchmark/run_local_viewer_ab.py \
        --seconds 75 --label optimized > outputs/live_ab/optimized.log 2>&1
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

# DPI awareness is first-wins per process; do this before any module import,
# exactly like desktop2stereo/main.py.
if sys.platform == "win32":
    from windows_dpi import set_per_monitor_dpi_v2

    set_per_monitor_dpi_v2()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seconds", type=float, default=75.0)
parser.add_argument("--legacy", action="store_true", help="disable both presenter optimizations")
parser.add_argument("--label", default="")
args = parser.parse_args()

# Clear diagnostic overrides that a reused shell may have left behind, matching
# src/run_windows_cuda.bat's preflight.
for name in (
    "D2S_FILAMENT_CONTROLLER_EYE_DIAGNOSTIC", "D2S_FILAMENT_EYE_DIAGNOSTIC",
    "D2S_FILAMENT_MULTIVIEW_LAYER_READBACK", "D2S_FILAMENT_MULTIVIEW",
    "D2S_FILAMENT_SHADER_DUMP_DIR", "D2S_FILAMENT_MULTIVIEW_PROJECTION_DIAGNOSTIC",
    "D2S_FILAMENT_PROJECTION_ONLY", "D2S_OPENXR_PROJECTION_ARRAY_EYE_DIAGNOSTIC",
    "D2S_OPENXR_SCREEN_QUAD_EYE_DIAGNOSTIC", "D2S_OPENXR_DISABLE_TOOL_QUADS",
    "D2S_OPENXR_VULKAN_MULTIVIEW_EYE_DIAGNOSTIC",
    "D2S_VULKAN_PROJECTION_COMPOSER_EYE_DIAGNOSTIC",
    "D2S_VULKAN_PROJECTION_QUALITY_CHAIN", "D2S_ROCM_DISABLE_OPENXR_OVERLAYS",
    "D2S_ROCM_ENABLE_OPENXR_OVERLAYS", "D2S_CONTROLLER_MODEL",
    "D2S_FPS_BREAKDOWN", "D2S_BENCHMARK", "D2S_DEBUG",
):
    os.environ.pop(name, None)

switch = "0" if args.legacy else "1"
os.environ["D2S_LOCAL_VIEWER_FAST_PRESENT"] = switch
os.environ["D2S_LOCAL_VIEWER_PRECISE_BARRIERS"] = switch
os.environ["D2S_CRASH_LOG_DIR"] = str(ROOT / "outputs" / "live_ab")

import utils  # noqa: E402
from utils.settings import load_settings  # noqa: E402

settings = load_settings()

# Freeze settings so hot reload cannot introduce a mid-run confound.
from stereo_runtime.hot_reload import StereoHotReloader  # noqa: E402

StereoHotReloader.poll_settings_snapshot_if_needed = lambda *_a, **_kw: None

# No global keyboard hook is needed for an unattended benchmark.
try:
    from capture.backends.windows_capture_event import WindowsCaptureEventRunner  # noqa: E402

    WindowsCaptureEventRunner._start_keyboard_worker = lambda *_a, **_kw: None
except Exception:
    pass

from app_runtime.runtime_entry import run_processing_runtime  # noqa: E402

print(
    f"[A/B] label={args.label or 'unnamed'} legacy={args.legacy} "
    f"fast_present={switch} precise_barriers={switch} seconds={args.seconds}",
    flush=True,
)
print(
    "[A/B] settings "
    f"run_mode={settings.get('Run Mode')} "
    f"model={settings.get('Depth Model')} "
    f"depth_resolution={settings.get('Depth Resolution')} "
    f"stereo_quality={settings.get('Stereo Quality')} "
    f"display_mode={settings.get('Display Mode')} "
    f"render_scale={settings.get('Render Scale')} "
    f"vsync={settings.get('VSync')} show_fps={settings.get('Show FPS')} "
    f"capture={settings.get('Capture Tool')} "
    f"monitor={settings.get('Monitor Index')}",
    flush=True,
)
print(f"[A/B] utils RUN_MODE={utils.RUN_MODE} SHOW_FPS={utils.SHOW_FPS} "
      f"TARGET_FPS={utils.TARGET_FPS} DEVICE_INFO={utils.DEVICE_INFO}", flush=True)

try:
    rc = run_processing_runtime(max_seconds=args.seconds)
except Exception as exc:
    print(f"[A/B] FAILED {type(exc).__name__}: {exc}", flush=True)
    raise
print(f"[A/B] finished rc={rc}", flush=True)
raise SystemExit(int(rc or 0))
