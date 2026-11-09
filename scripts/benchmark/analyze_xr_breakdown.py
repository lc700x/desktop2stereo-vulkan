"""Rank the real per-frame stages from [FPSBreakdown] log lines.

Usage:
    src/python3_cuda/python.exe scripts/benchmark/analyze_xr_breakdown.py \
        src/desktop2stereo/logs/desktop2stereo.log [--top 25] [--last 5]

OpenXR emits one long key=value line per sampling window. This tool averages
those windows, separates rates from millisecond stages, and prints the stages
sorted by cost so the actual bottleneck is obvious instead of buried.
"""
from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path

KV = re.compile(r"(\w+)=([-\w.:,]*)")
RATE_KEYS = (
    "cap", "raw", "overwrite", "runtime", "xr_loop", "xr_input_fps",
    "xr_present_fps", "xr_should", "xr_no_render", "screen_new",
    "screen_reuse", "screen_replay", "rt_overwrite", "capture_target",
)


TIMED_KEYS: set[str] = set()


def load(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "[FPSBreakdown]" not in line:
            continue
        payload = line.split("[FPSBreakdown]", 1)[1]
        row: dict[str, float | str] = {}
        for key, raw in KV.findall(payload):
            if raw.endswith("ms"):
                try:
                    row[key] = float(raw[:-2])
                    TIMED_KEYS.add(key)
                except ValueError:
                    continue
                continue
            try:
                row[key] = float(raw)
            except ValueError:
                if key in ("xr_path", "quad_unavail", "ba", "wall_mask"):
                    row[key] = raw
        if row:
            rows.append(row)
    return rows


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log")
    parser.add_argument("--top", type=int, default=22)
    parser.add_argument("--last", type=int, default=5, help="use only the final N windows")
    args = parser.parse_args()
    path = Path(args.log)
    rows = load(path)[-args.last:]
    if not rows:
        print(f"No [FPSBreakdown] rows in {path}")
        return 1
    print(f"source: {path}  windows used: {len(rows)} (of total requested)")
    paths = {str(r["xr_path"]) for r in rows if "xr_path" in r}
    print(f"xr_path: {sorted(paths)}\n")

    print(f"{'rate':<28} {'mean':>9} {'min':>9} {'max':>9}")
    print("-" * 60)
    for key in RATE_KEYS:
        values = [float(r[key]) for r in rows if isinstance(r.get(key), float)]
        if not values:
            continue
        print(f"{key:<28} {statistics.fmean(values):>9.2f} {min(values):>9.2f} {max(values):>9.2f}")

    SUMMARY = re.compile(
        r"(_p\d+|^xr_frame$|^rt_total$|^rt_loop$|^rt_call$|^source_lat$|"
        r"max$|_max$|vk_convert|rt_slot_wait|xr_projection_p)"
    )
    timed: list[tuple[str, float, float]] = []
    summaries: list[tuple[str, float, float]] = []
    for key in TIMED_KEYS:
        values = [float(r[key]) for r in rows if isinstance(r.get(key), float)]
        if not values:
            continue
        entry = (key, statistics.fmean(values), max(values))
        (summaries if SUMMARY.search(key) else timed).append(entry)
    timed.sort(key=lambda item: -item[1])
    print(f"\n{'LEAF stage (ms)':<40} {'mean':>9} {'max':>9}")
    print("-" * 60)
    for key, mean_ms, max_ms in timed[: args.top]:
        print(f"{key:<40} {mean_ms:>9.2f} {max_ms:>9.2f}")
    summaries.sort(key=lambda item: -item[1])
    print(f"\n{'frame/percentile (ms)':<40} {'mean':>9} {'max':>9}")
    print("-" * 60)
    for key, mean_ms, max_ms in summaries[:10]:
        print(f"{key:<40} {mean_ms:>9.2f} {max_ms:>9.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv))
