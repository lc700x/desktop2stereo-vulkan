"""Compare two OpenXR [FPSBreakdown] legs side by side.

Usage:
    src/python3_cuda/python.exe scripts/benchmark/compare_xr_breakdown.py \
        outputs/openxr_ab/graph_on.log outputs/openxr_ab/graph_off.log [--skip 1]

Rates are per-second; timings are milliseconds. The first window is warm-up
and skipped by default.
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_xr_breakdown import load  # noqa: E402

FIELDS = (
    "cap", "raw", "runtime", "xr_loop", "xr_present_fps", "screen_new",
    "screen_reuse", "xr_no_render", "sys_gpu", "rt_slot_wait", "rt_total",
    "rt_depth", "rt_model", "rt_synth", "rt_loop", "source_lat", "xr_frame",
    "xr_frame_p95", "rt_total_p95", "rt_slot_wait_p95", "xr_projection_total",
    "xr_projection_layer", "controller_eye0", "controller_eye1", "xr_wait",
    "xr_end", "vk_composer_submit", "vk_out_convert", "quad_total",
)
HIGHER_BETTER = {
    "cap", "raw", "runtime", "xr_loop", "xr_present_fps", "screen_new",
}
LOWER_BETTER = {
    "rt_slot_wait", "rt_total", "rt_depth", "rt_loop", "source_lat",
    "xr_frame", "xr_frame_p95", "rt_total_p95", "rt_slot_wait_p95",
    "xr_projection_total", "xr_projection_layer", "controller_eye0",
    "controller_eye1", "xr_wait", "xr_end", "vk_composer_submit",
    "vk_out_convert", "quad_total",
}


def value_map(rows: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in FIELDS:
        values = [float(r[key]) for r in rows if isinstance(r.get(key), float)]
        if values:
            out[key] = statistics.fmean(values)
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--skip", type=int, default=1)
    args = parser.parse_args(argv)

    a_rows, b_rows = load(Path(args.baseline))[args.skip:], load(Path(args.candidate))[args.skip:]
    if not a_rows or not b_rows:
        print(f"no samples after skipping {args.skip}: {len(a_rows)} / {len(b_rows)}")
        return 1
    a, b = value_map(a_rows), value_map(b_rows)
    print(f"baseline : {args.baseline} ({len(a_rows)} windows)")
    print(f"candidate: {args.candidate} ({len(b_rows)} windows)")
    print(f"\n{'metric':<26} {'baseline':>11} {'candidate':>11} {'delta':>10}  verdict")
    print("-" * 72)
    for key in FIELDS:
        if key not in a or key not in b:
            continue
        delta = b[key] - a[key]
        if key in HIGHER_BETTER:
            verdict = "better" if delta > 0.5 else "worse" if delta < -0.5 else "~same"
        elif key in LOWER_BETTER:
            verdict = "better" if delta < -0.05 else "worse" if delta > 0.05 else "~same"
        else:
            verdict = "context"
        print(f"{key:<26} {a[key]:>11.2f} {b[key]:>11.2f} {delta:>+10.2f}  {verdict}")
    print("-" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
