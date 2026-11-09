from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = ROOT / "src" / "desktop2stereo"
sys.path.insert(0, str(APP_ROOT))

from stereo_runtime.output import OUTPUT_FORMAT_CHOICES


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p90_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "max_ms": 0.0,
        }
    sorted_values = sorted(values)

    def percentile(q: float) -> float:
        index = min(len(sorted_values) - 1, round((len(sorted_values) - 1) * q))
        return float(sorted_values[index])

    return {
        "count": len(values),
        "mean_ms": float(statistics.mean(values)),
        "median_ms": float(statistics.median(values)),
        "p90_ms": percentile(0.90),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "max_ms": float(max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--out", default="outputs/end_to_end_4k/end_to_end.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", choices=["fast", "fast_plus", "quality_4k", "hq_4k"], default="quality_4k")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument(
        "--output-format",
        choices=OUTPUT_FORMAT_CHOICES,
        action="append",
        default=None,
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--depth-backend", choices=["tensorrt_native", "onnx_cuda_dlpack", "onnx_cuda_iobinding", "pytorch_cuda"], default="tensorrt_native")
    parser.add_argument("--onnx", default=None)
    parser.add_argument("--trt-engine", default=None)
    parser.add_argument("--no-fused", action="store_true")
    parser.add_argument("--depth-strength", type=float, default=0.48)
    parser.add_argument("--convergence", type=float, default=0.5)
    parser.add_argument("--temporal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temporal-strength", type=float, default=0.75)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--auto-reset-temporal", action="store_true")
    parser.add_argument("--scene-reset-threshold", type=float, default=0.22)
    parser.add_argument("--depth-antialias-strength", type=float, default=2.0)
    parser.add_argument("--edge-dilation", type=int, default=2)
    parser.add_argument("--edge-threshold", type=float, default=0.04)
    parser.add_argument("--cross-eyed", action="store_true")
    parser.add_argument("--anaglyph-method", choices=["red_cyan", "green_magenta", "amber_blue", "gray"], default="red_cyan")
    args = parser.parse_args()

    import torch

    from stereo_runtime.depth_provider import DepthProviderConfig, create_depth_provider
    from stereo_runtime.io import load_rgb
    from stereo_runtime.synthesis import StereoConfig, synthesize_stereo

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    rgb = load_rgb(args.rgb, device=device)
    output_formats = args.output_format or ["half_sbs", "full_sbs"]

    depth_config = DepthProviderConfig(
        backend=args.depth_backend,
        device=device,
        onnx_path=args.onnx,
        engine_path=args.trt_engine,
        cache_dir=APP_ROOT / "models",
        use_dlpack=args.depth_backend == "onnx_cuda_dlpack",
        use_cuda_graph=args.cuda_graph,
        execution_slot_count=2 if args.cuda_graph else 1,
    )
    provider = create_depth_provider(depth_config)
    if hasattr(provider, "load"):
        provider.load()

    def sync() -> None:
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

    report = {
        "rgb": str(args.rgb),
        "input_shape": list(rgb.shape),
        "device": str(device),
        "depth_backend": args.depth_backend,
        "provider": provider.info.to_report(),
        "stereo_backend": args.backend,
        "layers": args.layers,
        "cuda_graph": args.cuda_graph,
        "processing_config": {
            "depth_strength": args.depth_strength,
            "convergence": args.convergence,
            "temporal": args.temporal,
            "temporal_strength": args.temporal_strength,
            "depth_antialias_strength": args.depth_antialias_strength,
            "output_quality_enabled": False,
        },
        "formats": {},
    }

    for output_format in output_formats:
        config = StereoConfig(
            backend=args.backend,
            layers=args.layers,
            output_format=output_format,
            depth_strength=args.depth_strength,
            convergence=args.convergence,
            temporal=args.temporal,
            temporal_strength=args.temporal_strength,
            auto_reset_temporal=args.auto_reset_temporal,
            scene_reset_threshold=args.scene_reset_threshold,
            depth_antialias_strength=args.depth_antialias_strength,
            edge_dilation=args.edge_dilation,
            edge_threshold=args.edge_threshold,
            cross_eyed=args.cross_eyed,
            anaglyph_method=args.anaglyph_method,
            debug_output=False,
            fused=not args.no_fused,
        )
        depth_times: list[float] = []
        synthesis_times: list[float] = []
        total_times: list[float] = []
        output_shape = None
        synthesis_debug = {}

        with torch.inference_mode():
            for iteration in range(args.warmup + args.iters):
                sync()
                total_start = time.perf_counter()

                depth_start = time.perf_counter()
                depth = provider.predict(rgb)
                sync()
                depth_ms = (time.perf_counter() - depth_start) * 1000.0

                synth_start = time.perf_counter()
                result = synthesize_stereo(rgb, depth, config)
                sync()
                synthesis_ms = (time.perf_counter() - synth_start) * 1000.0

                total_ms = (time.perf_counter() - total_start) * 1000.0
                output_shape = list(result.sbs.shape)
                synthesis_debug = {
                    key: value
                    for key, value in result.debug_info.items()
                    if isinstance(value, (float, int, str))
                }
                if iteration >= args.warmup:
                    depth_times.append(depth_ms)
                    synthesis_times.append(synthesis_ms)
                    total_times.append(total_ms)
                print(
                    f"[{output_format}] iter={iteration + 1} depth_ms={depth_ms:.3f} synthesis_ms={synthesis_ms:.3f} total_ms={total_ms:.3f}",
                    flush=True,
                )

        total_summary = summarize(total_times)
        mean_ms = total_summary["mean_ms"]
        median_ms = total_summary["median_ms"]
        report["formats"][output_format] = {
            "output_shape": output_shape,
            "synthesis_debug": synthesis_debug,
            "depth": summarize(depth_times),
            "synthesis": summarize(synthesis_times),
            "total": total_summary,
            "mean_fps": 1000.0 / mean_ms if mean_ms else 0.0,
            "median_fps": 1000.0 / median_ms if median_ms else 0.0,
            "p95_frame_budget_fps": 1000.0 / total_summary["p95_ms"] if total_summary["p95_ms"] else 0.0,
            "p99_frame_budget_fps": 1000.0 / total_summary["p99_ms"] if total_summary["p99_ms"] else 0.0,
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] wrote {out}", flush=True)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

