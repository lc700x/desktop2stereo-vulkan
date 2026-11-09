"""Measure Local Viewer GPU cost across Quality Scale tiers (real CUDA device).

Times the two resolution-dependent stages with CUDA-synchronised wall clock
(so CPU launch overhead is included), plus the runtime's pure-GPU warp delta:
  * capture preprocess (BGRA uint8 4K -> scaled RGB float, Triton area resize)
  * stereo synthesis (fused Triton direct-SBS warp when eligible)

Model-grid depth antialias is deliberately disabled (D2S_SBS_AA=0,
depth_antialias_strength=0): in the real pipeline it runs at the depth model's
grid, not at render resolution, so it does not scale with Quality Scale. A
smooth depth ramp is used instead of random noise, which would trigger a
pathological edge-aware workload.

Runs each tier with the fixed-canvas output-upscale cap ON (default) and OFF
(legacy headset-tier EASU), so the Quality Scale trap and its fix are visible
in measured GPU milliseconds. Depth inference, desktop capture, NVENC, network
and headset composition are deliberately excluded.

Usage:
    src/python3_cuda/python.exe scripts/benchmark/bench_quality_scale.py
    src/python3_cuda/python.exe scripts/benchmark/bench_quality_scale.py --iters 40 --scales 100%,85%
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "desktop2stereo"))

import torch  # noqa: E402

from stereo_runtime.output_quality import (  # noqa: E402
    output_quality_requires_eye_images,
    output_sampling_plan_for_config,
    output_upscale_capped,
)
from stereo_runtime.render_size import RenderSizeConfig, resolve_render_size  # noqa: E402
from stereo_runtime.synthesis import StereoConfig, synthesize_stereo  # noqa: E402

CAPTURE_SIZE = (3840, 2160)
SCALES = ("100%", "85%", "75%", "50%")


def stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean_ms": round(statistics.fmean(values), 4),
        "p50_ms": round(ordered[len(ordered) // 2], 4),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 4),
        "max_ms": round(ordered[-1], 4),
    }


def gpu_time(fn: Callable[[], Any], *, warmup: int, iters: int) -> tuple[dict[str, float], Any]:
    """Synchronised wall time per call.

    Plain CUDA event pairs around a Python-heavy call report a bogus ~140 ms
    span on this driver because the start event can be ordered against work
    from a previous asynchronous iteration. Draining the queue before each
    sample gives a stable, conservative per-frame number that includes CPU
    launch overhead. Relative comparisons across tiers are what matter here.
    """
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()
    samples: list[float] = []
    last = None
    for _ in range(iters):
        torch.cuda.synchronize()
        started = time.perf_counter()
        last = fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    return stats(samples), last


def make_config() -> StereoConfig:
    return StereoConfig(
        backend="quality_4k",
        layers=2,
        output_format="half_sbs",
        temporal=False,
        hole_fill="none",
        fused=True,
        refine=False,
        debug_output=False,
        cross_eyed=False,
        output_quality_enabled=True,
        output_headset_tier_k=4,
        output_min_lod=0.0,
        output_max_lod=0.35,
        output_mip_lod_bias=-0.35,
        output_rcas_sharpness=0.5,
        depth_strength=0.48,
        convergence=0.5,
        depth_antialias_strength=0.0,
        edge_dilation=1,
        mask_feather_radius=1,
    )


def run_case(scale: str, *, cap: bool, warmup: int, iters: int) -> dict:
    from capture.preprocess import capture_frame_to_rgb, prepare_rgb_for_stereo_runtime

    os.environ["D2S_CAP_OUTPUT_UPSCALE"] = "1" if cap else "0"
    # Model-grid AA is not part of the render-resolution cost under test.
    os.environ["D2S_SBS_AA"] = "0"
    size = resolve_render_size(CAPTURE_SIZE, RenderSizeConfig(scale_factor=scale))
    width, height = size
    config = make_config()

    raw = torch.empty(
        (CAPTURE_SIZE[1], CAPTURE_SIZE[0], 4), dtype=torch.uint8, device="cuda"
    )
    # A gradient instead of a flat fill so the resize cannot be optimised away
    # and the area kernel does real work.
    column = torch.arange(CAPTURE_SIZE[0], device="cuda", dtype=torch.int32) % 256
    raw[..., 0] = column.to(torch.uint8)
    raw[..., 1] = 64
    raw[..., 2] = 240
    raw[..., 3] = 255

    def preprocess():
        rgb = capture_frame_to_rgb(raw, size, device="cuda", output="tensor")
        return prepare_rgb_for_stereo_runtime(rgb, device="cuda")

    pre_stats, runtime_rgb = gpu_time(preprocess, warmup=warmup, iters=iters)
    rgb = runtime_rgb.contiguous()
    # Depth is already at render resolution in the real pipeline (the depth
    # provider upsamples it). A smooth ramp mirrors real model output.
    row = torch.linspace(0.0, 1.0, height, device="cuda").view(1, 1, height, 1)
    column_depth = torch.linspace(0.0, 1.0, width, device="cuda").view(1, 1, 1, width)
    depth = (0.25 + 0.5 * column_depth + 0.2 * row).clamp(0.0, 1.0).float()

    def synthesis():
        return synthesize_stereo(rgb, depth, config, sbs_only=True)

    syn_stats, result = gpu_time(synthesis, warmup=warmup, iters=iters)
    plan = output_sampling_plan_for_config(config, width, height)
    debug = dict(result.debug_info or {})
    events = dict(getattr(result, "cuda_timing_events", {}) or {})
    warp_gpu_ms = None
    if "synth_depth_shift" in events and "synth_warp" in events:
        warp_gpu_ms = round(
            events["synth_depth_shift"].elapsed_time(events["synth_warp"]), 4
        )
    total = pre_stats["mean_ms"] + syn_stats["mean_ms"]
    case = {
        "scale": scale,
        "cap": cap,
        "render_size": list(size),
        "render_pixels": width * height,
        "preprocess": pre_stats,
        "synthesis": syn_stats,
        "total_mean_ms": round(total, 4),
        "fps_from_gpu_total": round(1000.0 / total, 2) if total > 0 else 0.0,
        "output_quality_requires_eye_images": bool(
            output_quality_requires_eye_images(config, width, height)
        ),
        "output_quality_plan": None if plan is None else {
            "mode": plan.mode,
            "target": [plan.target_width, plan.target_height],
            "target_pixels": plan.target_width * plan.target_height,
        },
        "warp_gpu_ms": warp_gpu_ms,
        "direct_sbs_ms": debug.get("direct_sbs_ms", 0),
        "synthesis_total_wall_ms": debug.get("synthesis_total_ms", 0),
        "warp_backend": str(debug.get("warp_composite_backend", "unknown")),
        "output_quality_mode": str(debug.get("output_quality_mode", "unknown")),
        "output_quality_applied": debug.get("output_quality_applied", 0),
        "sbs_shape": list(result.sbs.shape),
    }
    del raw, column, rgb, depth, runtime_rgb, result
    gc.collect()
    torch.cuda.empty_cache()
    return case


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--scales", default=",".join(SCALES))
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "visual_regression" / "quality_scale_gpu.json",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")
    scales = [item.strip() for item in args.scales.split(",") if item.strip()]

    print(f"device={torch.cuda.get_device_name(0)} torch={torch.__version__} "
          f"capture={CAPTURE_SIZE[0]}x{CAPTURE_SIZE[1]}", flush=True)
    print(f"default output_upscale_capped={output_upscale_capped()}", flush=True)

    cases: list[dict] = []
    for cap in (True, False):
        for scale in scales:
            label = f"scale={scale:<5} cap={'ON ' if cap else 'OFF'}"
            try:
                case = run_case(scale, cap=cap, warmup=args.warmup, iters=args.iters)
                cases.append(case)
                print(
                    f"{label} size={case['render_size']} "
                    f"pre={case['preprocess']['mean_ms']:.3f}ms "
                    f"synth={case['synthesis']['mean_ms']:.3f}ms "
                    f"total={case['total_mean_ms']:.3f}ms "
                    f"fps={case['fps_from_gpu_total']:.1f} "
                    f"backend={case['warp_backend']}",
                    flush=True,
                )
            except Exception as exc:
                cases.append({"scale": scale, "cap": cap, "error": f"{type(exc).__name__}: {exc}"})
                print(f"{label} ERROR {type(exc).__name__}: {exc}", flush=True)

    by_scale: dict[str, dict] = {}
    for case in cases:
        if "error" in case:
            continue
        entry = by_scale.setdefault(case["scale"], {})
        entry["cap_on" if case["cap"] else "cap_off"] = case
    comparison = {}
    for scale, pair in by_scale.items():
        if "cap_on" in pair and "cap_off" in pair:
            on, off = pair["cap_on"], pair["cap_off"]
            comparison[scale] = {
                "speedup_x": round(off["total_mean_ms"] / max(on["total_mean_ms"], 1e-6), 2),
                "saved_ms": round(off["total_mean_ms"] - on["total_mean_ms"], 4),
            }

    report = {
        "environment": {
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "capture_size": list(CAPTURE_SIZE),
            "warmup": args.warmup,
            "iters": args.iters,
            "note": "CUDA-synchronised wall time for preprocess + synthesis only "
                    "(includes CPU launch overhead). Model-grid depth AA disabled; "
                    "smooth depth ramp. Excludes capture, depth inference, present, "
                    "NVENC, network and headset composition. warp_gpu_ms is the "
                    "runtime's pure-GPU event delta for the fused warp stage.",
        },
        "cases": cases,
        "cap_speedup_by_scale": comparison,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2), flush=True)
    print(f"report: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
