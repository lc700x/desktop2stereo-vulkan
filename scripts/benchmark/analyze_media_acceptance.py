"""Summarize source-order and clock evidence, not headset acoustic/video measurements."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from itertools import pairwise
from pathlib import Path


def distribution(values: list[float]) -> dict:
    values = sorted(value for value in values if math.isfinite(value))
    if not values:
        return {}
    def percentile(fraction: float) -> float:
        index = fraction * (len(values) - 1)
        lower = int(index)
        return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (index - lower)
    return {
        "count": len(values), "mean": statistics.mean(values), "p50": percentile(0.5),
        "p95": percentile(0.95), "min": values[0], "max": values[-1],
    }


def source_pairs(rows: list[dict]) -> list[tuple[dict, dict]]:
    return [(a, b) for a, b in pairwise(rows) if a.get("media_generation") == b.get("media_generation")]


def summarize_segment(rows: list[dict]) -> dict:
    pairs = source_pairs(rows)
    deltas = Counter(b["source_frame_index"] - a["source_frame_index"] for a, b in pairs)
    expected = Counter()
    anomalies = []
    pts_anomalies = []
    for a, b in pairs:
        fps = b.get("media_playback_fps") or 0
        source_fps = b.get("media_source_fps") or 0
        step = max(1, round(source_fps / fps)) if fps > 0 and source_fps > 0 else None
        if step is not None:
            expected[step] += 1
            delta = b["source_frame_index"] - a["source_frame_index"]
            if delta != step and not (b.get("media_terminal_partial") and 0 < delta < step):
                anomalies.append({"t": b["t"], "from": a["source_frame_index"],
                                  "to": b["source_frame_index"], "expected_step": step})
            pts_delta = b["source_pts"] - a["source_pts"]
            expected_pts_delta = step / source_fps
            tolerance = max(0.002, 0.12 / source_fps)
            terminal = b.get("media_terminal_partial") and 0 < pts_delta <= expected_pts_delta + tolerance
            if not terminal and abs(pts_delta - expected_pts_delta) > tolerance:
                pts_anomalies.append({"t": b["t"], "pts_delta_s": pts_delta,
                                      "expected_pts_delta_s": expected_pts_delta})
    span = (rows[-1].get("presentation_at_relative_s", rows[-1]["t"])
            - rows[0].get("presentation_at_relative_s", rows[0]["t"])) if len(rows) > 1 else 0.0
    intervals = []
    tick_counts = Counter()
    for a, b in pairs:
        start = a.get("predicted_present_time_s", a["t"])
        end = b.get("predicted_present_time_s", b["t"])
        delta = end - start
        intervals.append(delta * 1000.0)
        if b.get("refresh_hz"):
            tick_counts[round(delta * b["refresh_hz"])] += 1
    return {
        "frames": len(rows), "duration_s": span,
        "terminal_partial_frames": sum(bool(row.get("media_terminal_partial")) for row in rows),
        "fresh_fps": (len(rows) - 1) / span if span > 0 else None,
        "source_pts_speed": (rows[-1]["source_pts"] - rows[0]["source_pts"]) / span if span > 0 else None,
        "source_steps": dict(deltas), "expected_source_steps": dict(expected), "anomalies": anomalies,
        "pts_anomalies": pts_anomalies,
        "presentation_interval_ms": distribution(intervals), "display_ticks": dict(tick_counts),
        "depth_age_ms": distribution([row["capture_to_runtime_ready_ms"] for row in rows
                                      if row.get("capture_to_runtime_ready_ms") is not None]),
        "queue_latency_ms": distribution([row["capture_to_runtime_start_ms"] for row in rows
                                          if row.get("capture_to_runtime_start_ms") is not None]),
        "runtime_completion_ms": distribution([row["runtime_completion_ms"] for row in rows
                                               if row.get("runtime_completion_ms") is not None]),
        "render_sizes": sorted({str(row["render_size"]) for row in rows if row.get("render_size")}),
        "eye_image_sizes": sorted({str(row["eye_image_sizes"]) for row in rows if row.get("eye_image_sizes")}),
        "packed_sizes": sorted({str(row["media_packed_output_size"]) for row in rows
                                if row.get("media_packed_output_size")}),
        "runtime_to_present_ms": distribution([row["runtime_to_present_ms"] for row in rows
                                               if row.get("runtime_to_present_ms") is not None]),
        "deadline_error_ms": distribution([
            (row["predicted_present_time_s"] - row["media_presentation_deadline"]) * 1000.0 for row in rows
            if row.get("predicted_present_time_s") is not None and row.get("media_presentation_deadline") is not None
        ]),
    }


def summarize_run(data: dict, warmup_s: float = 10.0) -> dict:
    rows = [row for row in data.get("presents", []) if row.get("source_frame_index") is not None
            and row.get("source_pts") is not None]
    if not rows:
        return {"status": "no_presentation", "frames": 0, "rc": data.get("rc")}
    generations = sorted({row["media_generation"] for row in rows})
    if len(generations) > 1:
        videos = {}
        for generation in generations:
            subset = dict(data)
            for collection in ("presents", "captures"):
                subset[collection] = [row for row in data.get(collection, [])
                                      if row.get("media_generation") == generation]
            subset["observations"] = [row for row in data.get("observations", [])
                                      if row.get("source_identity") and row["source_identity"][0] == generation]
            subset["scenario_events"] = [row for row in data.get("scenario_events", [])
                                         if row.get("media_generation") == generation]
            subset["media"] = data.get("media_sessions", {}).get(str(generation), {})
            videos[str(generation)] = summarize_run(subset, warmup_s)
        return {"status": "multi_video", "backend": data.get("backend"), "mode": data.get("mode"),
                "gpu": data.get("gpu"), "rc": data.get("rc"), "videos": videos,
                "requested_media_paths": data.get("requested_media_paths")}
    media = data.get("media_sessions", {}).get(str(generations[0]), data.get("media"))
    events = data.get("scenario_events", [])
    applied = [event for event in events if event["action"] == "control_applied"]
    pauses = [event.get("clock_t") if event.get("clock_t") is not None else event["t"]
              for event in applied if event.get("command") == "pause"]
    resumes = [event.get("clock_t") if event.get("clock_t") is not None else event["t"]
               for event in applied if event.get("command") == "play"]
    if not applied:
        pauses = [event["t"] for event in events if event["action"] == "pause"]
        resumes = [event["t"] for event in events if event["action"] == "play"]
    bounds = [(rows[0]["t"] + warmup_s, pauses[0] - 0.5 if pauses else rows[-1]["t"] + 0.001)]
    if resumes:
        bounds.append((resumes[0] + 3.0, rows[-1]["t"] + 0.001))
    segments = []
    for start, end in bounds:
        selected = [row for row in rows if start <= row["t"] < end]
        if len(selected) > 1:
            segments.append(summarize_segment(selected))
    steady_observations = [row for row in data.get("observations", []) if row.get("playing") is not False
                           and any(start <= row["t"] < end for start, end in bounds)]
    audio = [row.get("audio_sync") or {} for row in steady_observations]
    audio_counter_deltas = []
    for start, end in bounds:
        samples = [row["audio_sync"] for row in steady_observations
                   if start <= row["t"] < end and row.get("audio_sync")]
        if samples:
            audio_counter_deltas.append({key: samples[-1].get(key, 0) - samples[0].get(key, 0)
                                         for key in ("underruns", "device_underflows")})
    offsets = [row["offset_ms"] for row in audio if row.get("offset_ms") is not None]
    paused = [row for row in data.get("observations", []) if pauses and resumes
              and pauses[0] + 1.0 <= row["t"] <= resumes[0] - 0.5 and row.get("playing") is False]
    pause_identities = {tuple(row["source_identity"][:3]) for row in paused if row.get("source_identity")}
    eof = next((event for event in events if event["action"] == "eof"), None)
    pause_duration = sum(max(0.0, resume - pause) for pause, resume in zip(pauses, resumes))
    duration_evidence = {}
    if eof is not None:
        expected = eof["duration"] - rows[0]["source_pts"]
        actual = eof["t"] - rows[0].get("presentation_at_relative_s", rows[0]["t"]) - pause_duration
        duration_evidence = {"source_duration_s": expected, "playing_duration_s": actual,
                             "difference_ms": (actual - expected) * 1000.0}
    return {
        "status": "measured" if segments else "insufficient_steady_samples",
        "backend": data.get("backend"), "mode": data.get("mode"), "gpu": data.get("gpu"),
        "settings": data.get("settings"), "media": media,
        "rc": data.get("rc"), "total_frames": len(rows),
        "whole_playing_source_speed": (
            (rows[-1]["source_pts"] - rows[0]["source_pts"])
            / max(1e-9, rows[-1].get("presentation_at_relative_s", rows[-1]["t"])
                  - rows[0].get("presentation_at_relative_s", rows[0]["t"]) - pause_duration)
        ),
        "duration_evidence": duration_evidence,
        "pause_clock": "source_epoch" if any(event.get("clock_t") is not None for event in applied)
        else "control_applied" if applied else "control_request_fallback",
        "display_feedback_clocks": dict(Counter(row.get("media_feedback_clock") for row in rows
                                                if row.get("media_feedback_clock"))),
        "warmup_s": warmup_s, "all_source_steps": summarize_segment(rows)["source_steps"], "segments": segments,
        "audio_clock_offset_ms": distribution(offsets), "audio_abs_clock_offset_ms": distribution([abs(x) for x in offsets]),
        "decode_backends": dict(Counter(row["decode_backend"] for row in steady_observations
                                        if row.get("decode_backend"))),
        "decode_fallback_reasons": sorted({row["decode_fallback_reason"] for row in steady_observations
                                           if row.get("decode_fallback_reason")}),
        "audio_errors": sorted({row["audio_error"] for row in steady_observations if row.get("audio_error")}),
        "steady_audio_counter_deltas": audio_counter_deltas,
        "pcm_underruns": max((row.get("underruns", 0) for row in audio), default=None),
        "device_underflows": max((row.get("device_underflows", 0) for row in audio), default=None),
        "pause": {"samples": len(paused), "source_identities": [list(value) for value in sorted(pause_identities)],
                  "layers": dict(Counter(row["layers"] for row in paused if "layers" in row))},
        "xr_session_states": dict(Counter(row.get("session_state") for row in data.get("observations", [])
                                          if "session_state" in row)),
        "capture_overwrites": sum(bool(row.get("overwritten")) for row in data.get("captures", [])),
        "limits": "Clock-ledger evidence only; DAC excludes remote audio transport and acoustic output. "
                  "Non-integral video/display ratios require pulldown, not uniform frame intervals.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmup", type=float, default=10.0)
    args = parser.parse_args()
    result = {str(path.resolve()): summarize_run(json.loads(path.read_text(encoding="utf-8")), args.warmup)
              for path in args.inputs}
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
