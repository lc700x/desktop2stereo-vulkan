"""Replay small-person 60 FPS clips through native CoreML + Metal SBS pack.

Usage:
    src/python3/bin/python scripts/validate_person_motion_models.py \
        /Users/nick/Downloads/dibr-edge-review/temporal-contours/input \
        /Users/nick/Downloads/dibr-edge-review/temporal-contours/production
"""
from __future__ import annotations

import json
import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/desktop2stereo"))
sys.path.insert(0, str(ROOT / "scripts"))

from validate_sbs_antialias import pixel_buffer_for


def stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "mean_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
    return {
        "count": len(ordered),
        "mean_ms": float(statistics.fmean(ordered)),
        "p95_ms": float(np.percentile(ordered, 95)),
        "p99_ms": float(np.percentile(ordered, 99)),
    }


def run_clip(bridge, stereo, source: Path, output: Path, model: str,
             resolution: int, max_frames: int | None = None) -> dict:
    from stereo_runtime.depth_provider import _model_input_size
    from stereo_runtime.runtime import _configure_native_coreml_warp

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open input video: {source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width % 2:
        raise ValueError("SBS eye width must be even")
    output.mkdir(parents=True, exist_ok=True)
    full_writer = cv2.VideoWriter(
        str(output / "full_sbs.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps, (width * 2, height),
    )
    half_writer = cv2.VideoWriter(
        str(output / "half_sbs.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps, (width, height),
    )
    zoom_size = min(320, width, height)
    left_zoom_writer = cv2.VideoWriter(
        str(output / "left_contour_zoom.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps, (zoom_size * 3, zoom_size * 3),
    )
    right_zoom_writer = cv2.VideoWriter(
        str(output / "right_contour_zoom.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps, (zoom_size * 3, zoom_size * 3),
    )
    if not all(w.isOpened() for w in (full_writer, half_writer,
                                      left_zoom_writer, right_zoom_writer)):
        raise RuntimeError("cannot create SBS review movies")

    input_height, input_width = _model_input_size(height, width, resolution, 14)
    timings: list[float] = []
    inference_wall_ms: list[float] = []
    inference_model_ms: list[float] = []
    full_pack_ms: list[float] = []
    half_pack_ms: list[float] = []
    normalize_ranges: list[tuple[float, float]] = []
    raw_normalize_ranges: list[tuple[float, float]] = []
    normalize_series: list[dict[str, float | int]] = []
    silhouette_centroids: list[dict[str, float | int]] = []
    frame_id = 0
    try:
        while True:
            if max_frames is not None and frame_id >= max_frames:
                break
            ok, bgr = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pixel_buffer = pixel_buffer_for(rgb)
            frame = None
            start = time.perf_counter()
            try:
                inference_start = time.perf_counter()
                frame = bridge.predict(pixel_buffer, frame_id)
                inference_wall_ms.append((time.perf_counter() - inference_start) * 1000.0)
                inference_model_ms.append(
                    float(frame.preprocess_ms + frame.model_ms + frame.postprocess_ms)
                )
                pixel_buffer = None  # Native frame slot retains the source pixel buffer.
                warp_debug = _configure_native_coreml_warp(
                    frame, stereo, width=width, height=height
                )
                full_bytes = bytearray(width * 2 * height * 4)
                half_bytes = bytearray(width * height * 4)
                pack_start = time.perf_counter()
                frame.pack(full_bytes, (width * 2, height), "full_sbs")
                full_pack_ms.append((time.perf_counter() - pack_start) * 1000.0)
                pack_start = time.perf_counter()
                frame.pack(half_bytes, (width, height), "half_sbs")
                half_pack_ms.append((time.perf_counter() - pack_start) * 1000.0)
                full_rgba = np.frombuffer(full_bytes, np.uint8).reshape(height, width * 2, 4)
                half_rgba = np.frombuffer(half_bytes, np.uint8).reshape(height, width, 4)
                full_writer.write(cv2.cvtColor(full_rgba[..., :3], cv2.COLOR_RGB2BGR))
                half_writer.write(cv2.cvtColor(half_rgba[..., :3], cv2.COLOR_RGB2BGR))
                # Center each zoom on the current bright subject in the fixed
                # camera input, keeping both eye contours visible through motion.
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                ys, xs = np.nonzero(gray > 220)
                if xs.size:
                    center_x = int(round(float(xs.mean())))
                    center_y = int(round(float(ys.mean())))
                    centroid = {
                        "frame_id": frame_id, "x": center_x, "y": center_y,
                        "area_px": int(xs.size),
                    }
                else:
                    center_x, center_y = width // 2, height // 2
                    centroid = {"frame_id": frame_id, "x": center_x,
                                "y": center_y, "area_px": 0}
                silhouette_centroids.append(centroid)
                x0 = max(0, min(width - zoom_size, center_x - zoom_size // 2))
                y0 = max(0, min(height - zoom_size, center_y - zoom_size // 2))
                for eye, writer in ((full_rgba[:, :width, :3], left_zoom_writer),
                                    (full_rgba[:, width:, :3], right_zoom_writer)):
                    crop = eye[y0:y0 + zoom_size, x0:x0 + zoom_size]
                    zoom = cv2.resize(crop, (zoom_size * 3, zoom_size * 3),
                                      interpolation=cv2.INTER_NEAREST)
                    writer.write(cv2.cvtColor(zoom, cv2.COLOR_RGB2BGR))
                if frame_id % max(1, int(round(source_fps))) == 0:
                    key = output / "frames" / f"frame-{frame_id:04d}"
                    key.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(rgb).save(key / "input.png")
                    Image.fromarray(full_rgba[..., :3]).save(key / "full_sbs.png")
                    Image.fromarray(half_rgba[..., :3]).save(key / "half_sbs.png")
                    Image.fromarray(full_rgba[:, :width, :3]).save(key / "full_left.png")
                    Image.fromarray(full_rgba[:, width:, :3]).save(key / "full_right.png")
                    Image.fromarray(half_rgba[:, :width // 2, :3]).save(key / "half_left.png")
                    Image.fromarray(half_rgba[:, width // 2:, :3]).save(key / "half_right.png")
                    depth_bytes = bytearray(width * height * 4)
                    frame.pack(depth_bytes, (width, height), "depth_map")
                    depth_rgba = np.frombuffer(depth_bytes, np.uint8).reshape(height, width, 4)
                    Image.fromarray(depth_rgba[..., :3]).save(key / "render_depth.png")
                timings.append((time.perf_counter() - start) * 1000.0)
                normalize_ranges.append((float(frame.normalize_lo), float(frame.normalize_hi)))
                raw_normalize_ranges.append((float(frame.raw_normalize_lo),
                                             float(frame.raw_normalize_hi)))
                normalize_series.append({
                    "rgb_frame_id": frame_id,
                    "depth_frame_id": frame.frame_id,
                    "raw_lo": float(frame.raw_normalize_lo),
                    "raw_hi": float(frame.raw_normalize_hi),
                    "render_lo": float(frame.normalize_lo),
                    "render_hi": float(frame.normalize_hi),
                    "range_history_reset": int(frame.normalization_history_reset),
                })
            finally:
                if frame is not None:
                    frame.release()
                pixel_buffer = None
            frame_id += 1
            if frame_id % 60 == 0:
                print(f"{model} {source.stem}: {frame_id}/{expected_frames}", flush=True)
    finally:
        capture.release()
        full_writer.release()
        half_writer.release()
        left_zoom_writer.release()
        right_zoom_writer.release()

    elapsed = sum(timings) / 1000.0
    ranges = np.asarray(normalize_ranges, dtype=np.float64)
    raw_ranges = np.asarray(raw_normalize_ranges, dtype=np.float64)
    raw_steps = np.max(np.abs(np.diff(raw_ranges, axis=0)), axis=1) if len(raw_ranges) > 1 else np.zeros(0)
    render_steps = np.max(np.abs(np.diff(ranges, axis=0)), axis=1) if len(ranges) > 1 else np.zeros(0)
    frame_diagnostics_path = output / "frame_diagnostics.jsonl"
    with frame_diagnostics_path.open("w") as diagnostics_file:
        for index, entry in enumerate(normalize_series):
            if index < len(silhouette_centroids):
                entry["rgb_silhouette"] = silhouette_centroids[index]
            diagnostics_file.write(json.dumps(entry, separators=(",", ":")) + "\n")
    result = {
        "input": str(source),
        "model": model,
        "resolution": resolution,
        "model_input_size": [input_width, input_height],
        "source_size": [width, height],
        "source_fps": source_fps,
        "frames": frame_id,
        "validation_frame_limit": max_frames,
        "processing_fps": frame_id / elapsed if elapsed else 0.0,
        "frame_processing_ms": stats(timings),
        "timing_ms": {
            "inference_wall": stats(inference_wall_ms),
            "inference_reported": stats(inference_model_ms),
            "full_pack": stats(full_pack_ms),
            "half_pack": stats(half_pack_ms),
        },
        "normalize_raw_lo_range": [float(raw_ranges[:, 0].min()), float(raw_ranges[:, 0].max())],
        "normalize_raw_hi_range": [float(raw_ranges[:, 1].min()), float(raw_ranges[:, 1].max())],
        "normalize_render_lo_range": [float(ranges[:, 0].min()), float(ranges[:, 0].max())],
        "normalize_render_hi_range": [float(ranges[:, 1].min()), float(ranges[:, 1].max())],
        "normalization_step_abs_max_p50": {
            "raw": float(np.percentile(raw_steps, 50)) if raw_steps.size else 0.0,
            "render": float(np.percentile(render_steps, 50)) if render_steps.size else 0.0,
        },
        "normalization_step_abs_max_p90": {
            "raw": float(np.percentile(raw_steps, 90)) if raw_steps.size else 0.0,
            "render": float(np.percentile(render_steps, 90)) if render_steps.size else 0.0,
        },
        "rgb_depth_frame_ids_match": all(
            entry["rgb_frame_id"] == entry["depth_frame_id"]
            for entry in normalize_series
        ),
        "normalization_history_reset_count": sum(
            int(entry["range_history_reset"]) for entry in normalize_series
        ),
        "max_disparity_px": warp_debug.get("native_coreml_max_disparity_px"),
        "presentation_commit_rate_tested": False,
        "renderer": "NativeCoreMLIOBridge production Metal warp + shared FXAA pack",
        "full_sbs_movie": str(output / "full_sbs.mp4"),
        "half_sbs_movie": str(output / "half_sbs.mp4"),
        "left_contour_zoom_movie": str(output / "left_contour_zoom.mp4"),
        "right_contour_zoom_movie": str(output / "right_contour_zoom.mp4"),
        "frames_dir": str(output / "frames"),
        "frame_diagnostics": str(frame_diagnostics_path),
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--model", choices=("small336", "base518"))
    parser.add_argument("--clip", choices=("person-05pct-60fps",
                                             "person-10pct-60fps",
                                             "person-20pct-60fps"))
    parser.add_argument("--input-video", type=Path,
                        help="also accept a specific 60 FPS source clip")
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    source_dir, output_root = args.source_dir.expanduser(), args.output_root.expanduser()
    from stereo_runtime.adapter import runtime_config_from_d2s_settings, stereo_config_from_runtime
    from stereo_runtime.providers.apple.native import coreml_io
    from utils.bootstrap import bootstrap_settings

    settings_path = ROOT / "src/desktop2stereo/settings.yaml"
    model_cases = (
        ("small336", "Distill-Any-Depth-Small", 336,
         "xingyang1-Distill-Any-Depth-Small-hf"),
        ("base518", "Distill-Any-Depth-Base", 518,
         "lc700x-Distill-Any-Depth-Base-hf"),
    )
    reports = []
    for model_tag, model_name, resolution, folder in model_cases:
        if args.model and model_tag != args.model:
            continue
        settings = dict(bootstrap_settings(str(settings_path), os_name="Darwin"))
        settings["Depth Model"] = model_name
        settings["Depth Resolution"] = resolution
        config = runtime_config_from_d2s_settings(
            settings,
            cache_dir=str(ROOT / "src/desktop2stereo/models"),
            device="mps",
            depth_only=False,
        )
        stereo = stereo_config_from_runtime(config)
        sources = ([args.input_video.expanduser()] if args.input_video else
                   sorted(source_dir.glob("person-*-60fps.mp4")))
        for source in sources:
            if args.clip and source.stem != args.clip:
                continue
            input_width, input_height = _model_size(source, resolution)
            model_path = (ROOT / "src/desktop2stereo/models/coreml" / folder
                          / f"model_fp32_{input_height}x{input_width}.mlpackage")
            if not model_path.exists():
                raise FileNotFoundError(model_path)
            bridge = coreml_io.NativeCoreMLIOBridge(model_path, input_width, input_height)
            output = output_root / model_tag / source.stem
            try:
                report = run_clip(bridge, stereo, source, output, model_name,
                                  resolution, args.max_frames)
            finally:
                bridge.close()
            reports.append(report)
            print(json.dumps(report), flush=True)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")


def _model_size(source: Path, resolution: int) -> tuple[int, int]:
    capture = cv2.VideoCapture(str(source))
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    from stereo_runtime.depth_provider import _model_input_size

    return _model_input_size(height, width, resolution, 14)[::-1]


if __name__ == "__main__":
    main()
