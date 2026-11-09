"""Compare two Local Viewer A/B runs from their [D2S_METRICS] log lines.

Usage:
    src/python3_cuda/python.exe scripts/benchmark/compare_local_viewer_ab.py \
        baseline.log optimized.log [skip_warmup_samples=3]

Capture each run with the GUI in Local Viewer mode (Show FPS enabled so the
metrics lines are emitted every 5 s), for example:

    set D2S_LOCAL_VIEWER_FAST_PRESENT=0
    set D2S_LOCAL_VIEWER_PRECISE_BARRIERS=0
    rem ... start the app, let it run 60 s, stop, copy the log ...
    set D2S_LOCAL_VIEWER_FAST_PRESENT=1
    set D2S_LOCAL_VIEWER_PRECISE_BARRIERS=1
    rem ... repeat with identical settings and desktop content ...

Only compare runs with the same settings, resolution, model and desktop
content. Present FPS alone can be misleading: also check content_fps, reuse
and latency_p95_ms.
"""
from __future__ import annotations

import math
import re
import statistics
import sys
from pathlib import Path

KV = re.compile(r"(\w+)=([-\w.]+)")

FIELDS = (
    "present_fps",
    "content_fps",
    "reuse",
    "latency_ms",
    "avg_latency_ms",
    "latency_p50_ms",
    "latency_p95_ms",
    "present_call_ms_p95",
    "viewer_queue_ms_p95",
    "runtime_execution_ms_p95",
    "capture_to_runtime_ready_ms_p95",
    "sys_gpu",
    "display_hz",
)

HIGHER_IS_BETTER = {"present_fps", "content_fps"}
LOWER_IS_BETTER = {
    "reuse",
    "latency_ms",
    "avg_latency_ms",
    "latency_p50_ms",
    "latency_p95_ms",
    "present_call_ms_p95",
    "viewer_queue_ms_p95",
    "runtime_execution_ms_p95",
    "capture_to_runtime_ready_ms_p95",
}


def load(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "[D2S_METRICS]" not in line:
            continue
        parsed: dict[str, float] = {}
        for key, raw in KV.findall(line.split("[D2S_METRICS]", 1)[1]):
            try:
                parsed[key] = float(raw)
            except ValueError:
                continue
        if parsed:
            rows.append(parsed)
    return rows


def mean(rows: list[dict[str, float]], field: str) -> float:
    values = [row[field] for row in rows if field in row]
    return statistics.fmean(values) if values else math.nan


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    baseline_path, optimized_path = Path(argv[1]), Path(argv[2])
    skip = int(argv[3]) if len(argv) > 3 else 3
    baseline, optimized = load(baseline_path)[skip:], load(optimized_path)[skip:]
    if not baseline or not optimized:
        print(f"No [D2S_METRICS] samples after skipping {skip}: "
              f"{baseline_path}={len(baseline)} {optimized_path}={len(optimized)}")
        return 1
    print(f"baseline : {baseline_path} ({len(baseline)} samples, skipped {skip} warmup)")
    print(f"optimized: {optimized_path} ({len(optimized)} samples, skipped {skip} warmup)")
    print(f"{'metric':<38} {'baseline':>16} {'optimized':>16} {'delta':>9}  verdict")
    print("-" * 96)
    for field in FIELDS:
        rows_a = [row[field] for row in baseline if field in row]
        rows_b = [row[field] for row in optimized if field in row]
        if not rows_a or not rows_b:
            continue
        before, after = statistics.fmean(rows_a), statistics.fmean(rows_b)
        sd_a = statistics.pstdev(rows_a) if len(rows_a) > 1 else 0.0
        sd_b = statistics.pstdev(rows_b) if len(rows_b) > 1 else 0.0
        # ~95% interval for the difference of the two means.
        noise = 2.0 * math.sqrt(sd_a ** 2 / len(rows_a) + sd_b ** 2 / len(rows_b))
        delta = after - before
        if field not in HIGHER_IS_BETTER and field not in LOWER_IS_BETTER:
            verdict = "context"
        elif abs(delta) <= noise:
            verdict = "within noise"
        elif field in HIGHER_IS_BETTER:
            verdict = "better" if delta > 0 else "worse"
        else:
            verdict = "better" if delta < 0 else "worse"
        print(
            f"{field:<38} {before:>8.2f}±{sd_a:<6.2f} {after:>8.2f}±{sd_b:<6.2f} "
            f"{delta:>+9.2f}  {verdict} (95% noise ±{noise:.2f})"
        )
    print("-" * 96)
    # The viewer alternates between fresh-content and cached-reuse phases.
    # Mixing them into one mean produces huge variance, so compare each phase
    # separately as well.
    print("\nPhase-matched windows (reuse splits the two presentation modes):")
    phases = (
        ("fresh  (reuse < 0.10)", lambda r: r.get("reuse", 0.0) < 0.10),
        ("cached (reuse > 0.50)", lambda r: r.get("reuse", 0.0) > 0.50),
    )
    phase_fields = (
        "present_fps", "content_fps", "latency_p95_ms",
        "present_call_ms_p95", "viewer_queue_ms_p95", "runtime_execution_ms_p95",
        "sys_gpu",
    )
    for phase_name, predicate in phases:
        a = [row for row in baseline if predicate(row)]
        b = [row for row in optimized if predicate(row)]
        print(f"  {phase_name}: baseline n={len(a)}, optimized n={len(b)}")
        if not a or not b:
            continue
        for field in phase_fields:
            before, after = mean(a, field), mean(b, field)
            if math.isnan(before) or math.isnan(after):
                continue
            print(f"    {field:<34} {before:>8.2f} -> {after:>8.2f}  ({after - before:+.2f})")
    print(
        "\nNote: 5 s aggregates with high run-to-run variance. 'within noise' means\n"
        "the observed delta is smaller than the ~95% interval of the difference\n"
        "of the two means; it is NOT evidence of a regression or an improvement.\n"
        "Only trust consistent results across repeated interleaved A/B runs."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
