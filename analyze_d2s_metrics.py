"""Parse Desktop2Stereo [D2S_METRICS] lines and summarise playback health.

Answers the question the on-screen panel cannot: is a "60 FPS" run actually
delivering new frames, and how much latency was it carrying?

Usage:
    python3 analyze_d2s_metrics.py [path/to/desktop2stereo.log]
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

DEFAULT_LOG = (
    Path(__file__).resolve().parent
    / "src"
    / "desktop2stereo"
    / "logs"
    / "desktop2stereo.log"
)
KV = re.compile(r"(\w+)=([\w.]+)")


def load(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "[D2S_METRICS]" not in line:
            continue
        payload = line.split("[D2S_METRICS]", 1)[1]
        parsed: dict = {}
        for key, value in KV.findall(payload):
            try:
                parsed[key] = float(value)
            except ValueError:
                continue
        if parsed:
            parsed["_raw"] = line.strip()
            rows.append(parsed)
    return rows


def stat(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    ordered = sorted(values)
    avg = sum(ordered) / len(ordered)
    p95 = ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)]
    return avg, p95, ordered[-1]


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else DEFAULT_LOG
    rows = load(path)
    if not rows:
        print(f"No [D2S_METRICS] lines found in {path}")
        print("Run the app with 'Show FPS' enabled; metrics are written every report interval.")
        return 1

    print(f"source: {path}")
    print(f"samples: {len(rows)}\n")

    pres, cont, reuse, lat, avg_lat = [], [], [], [], []
    for row in rows:
        pres.append(row.get("present_fps", 0.0))
        cont.append(row.get("content_fps", 0.0))
        reuse.append(row.get("reuse", 0.0))
        lat.append(row.get("latency_ms", 0.0))
        avg_lat.append(row.get("avg_latency_ms", 0.0))

    p_avg, p_p95, p_max = stat(pres)
    c_avg, c_p95, c_max = stat(cont)
    r_avg, r_p95, r_max = stat(reuse)
    l_avg, l_p95, l_max = stat(lat)
    a_avg, a_p95, a_max = stat(avg_lat)

    print("present_fps   avg {:6.1f}   p95 {:6.1f}   max {:6.1f}".format(p_avg, p_p95, p_max))
    print("content_fps   avg {:6.1f}   p95 {:6.1f}   max {:6.1f}".format(c_avg, c_p95, c_max))
    print("reuse         avg {:6.1%}  p95 {:6.1%}  max {:6.1%}".format(r_avg, r_p95, r_max))
    print("latency_ms    avg {:6.1f}   p95 {:6.1f}   max {:6.1f}".format(l_avg, l_p95, l_max))
    print("avg_latency   avg {:6.1f}   p95 {:6.1f}   max {:6.1f}".format(a_avg, a_p95, a_max))
    print()

    hz = rows[-1].get("display_hz", 0.0)
    print(f"display refresh: {hz:.0f} Hz")
    if hz and p_avg > hz * 1.05:
        print(f"  note: present_fps exceeds refresh; frames are being shown more than once")
    print()

    total = rows[-1].get("present_total", 0.0)
    reuse_total = rows[-1].get("reuse_total", 0.0)
    if total > 0:
        print(f"cumulative reuse: {reuse_total:.0f}/{total:.0f} = {reuse_total / total:.1%}")
    print()

    worst = max(rows, key=lambda r: r.get("reuse", 0.0))
    print("worst sample (highest reuse):")
    print(f"  present {worst.get('present_fps', 0):.1f} fps  "
          f"content {worst.get('content_fps', 0):.1f} fps  "
          f"reuse {worst.get('reuse', 0):.1%}  "
          f"latency {worst.get('latency_ms', 0):.0f} ms")
    print()

    if r_avg > 0.30:
        verdict = "STUTTER CONFIRMED: most presents replay a cached frame."
        advice = "Lower 'Depth Resolution', or enable parallel inference."
    elif r_avg > 0.10:
        verdict = "MILD REUSE: some frames repeat; motion may look uneven."
        advice = "Try lowering 'Depth Resolution' one step."
    else:
        verdict = "REUSE LOW: frame delivery looks healthy."
        advice = "Stutter has another cause (vsync/present timing, capture cadence)."
    print(verdict)
    print(f"next: {advice}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
