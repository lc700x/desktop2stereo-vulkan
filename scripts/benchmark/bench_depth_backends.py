from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = ROOT / "src" / "desktop2stereo"
sys.path.insert(0, str(APP_ROOT))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def summarize_ms(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p90_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
        }
    return {
        "count": len(values),
        "mean_ms": float(statistics.mean(values)),
        "median_ms": float(statistics.median(values)),
        "p90_ms": float(percentile(values, 0.9)),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
    }


def _sync(device) -> None:
    import torch

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _provider_for(
    *,
    model_name: str,
    backend: str,
    device,
    cache_dir: Path,
    onnx_dtype: str,
    build_engine: bool,
    force_rebuild: bool,
    local_files_only: bool,
    allow_pytorch_fallback: bool,
) -> Any:
    from stereo_runtime.depth_provider import DepthProviderConfig, create_depth_provider
    from stereo_runtime.model_artifacts import artifact_paths_for_model
    from stereo_runtime.model_registry import ModelRegistry

    spec = ModelRegistry.default().get(model_name)
    paths = artifact_paths_for_model(
        spec.name,
        cache_dir=cache_dir,
        export_height=294,
        export_width=518,
    )
    is_migraphx = backend in {"migraphx", "migraphx_rocm", "rocm_migraphx"}
    is_tensorrt = backend in {"tensorrt", "tensorrt_ort", "tensorrt_native", "native_tensorrt"}
    onnx_path = paths.onnx_path_for_dtype(onnx_dtype) if onnx_dtype in {"fp16", "fp32"} else paths.onnx_fp16_path
    engine_path = paths.migraphx_fp16_path if is_migraphx else paths.trt_fp16_path
    patch_size = 32 if spec.family in {"zipdepth", "depth-pro-2"} else 14
    config = DepthProviderConfig(
        backend=backend,
        model_id=spec.model_id,
        model_name=spec.name,
        device=device,
        cache_dir=cache_dir,
        onnx_path=onnx_path if (is_migraphx or is_tensorrt) and onnx_path.is_file() else None,
        engine_path=engine_path if (is_migraphx or is_tensorrt) and engine_path.is_file() else None,
        onnx_dtype=onnx_dtype,
        local_files_only=local_files_only,
        build_engine=build_engine,
        force_rebuild=force_rebuild,
        prefer_tensorrt=is_tensorrt,
        prefer_native_tensorrt=backend in {"tensorrt_native", "native_tensorrt"},
        allow_pytorch_fallback=allow_pytorch_fallback,
        depth_resolution=spec.default_resolution,
        patch_size=patch_size,
    )
    return create_depth_provider(config)


def _write_csv(path: Path, results: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for model_name, model_results in results["models"].items():
        for backend, result in model_results.items():
            timing = result.get("timings", {})
            profile = result.get("profile", {})
            provider = result.get("provider") or {}
            rows.append(
                {
                    "model": model_name,
                    "backend": backend,
                    "status": result.get("status"),
                    "setup_ms": result.get("setup_ms"),
                    "mean_ms": timing.get("mean_ms"),
                    "median_ms": timing.get("median_ms"),
                    "p90_ms": timing.get("p90_ms"),
                    "preprocess_ms": profile.get("preprocess", {}).get("mean_ms"),
                    "model_ms": profile.get("model", {}).get("mean_ms"),
                    "postprocess_ms": profile.get("postprocess", {}).get("mean_ms"),
                    "model_profile": provider.get("model_profile"),
                    "zero_copy_output": provider.get("zero_copy_output"),
                    "host_readback_bytes": provider.get("host_readback_bytes"),
                    "error": result.get("error"),
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["model", "backend"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark D2S depth providers without stale provider imports.")
    parser.add_argument("--rgb", required=True)
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="Model name or Hugging Face id. Defaults to the two Distill baselines.",
    )
    parser.add_argument(
        "--backend",
        action="append",
        default=None,
        choices=[
            "auto",
            "pytorch_cuda",
            "pytorch_rocm",
            "migraphx",
            "migraphx_rocm",
            "tensorrt_native",
            "tensorrt",
            "onnx_cuda",
            "onnx_cuda_iobinding",
        ],
    )
    parser.add_argument("--out-dir", default="outputs/depth_backend_bench")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", default=str(ROOT / "models"))
    parser.add_argument("--onnx-dtype", choices=["auto", "fp16", "fp32"], default="auto")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--input-height", type=int, default=None, help="Resize the benchmark frame before inference.")
    parser.add_argument("--input-width", type=int, default=None, help="Resize the benchmark frame before inference.")
    parser.add_argument("--build-engine", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--allow-pytorch-fallback", action="store_true")
    args = parser.parse_args()

    import torch
    from stereo_runtime.io import load_rgb
    from stereo_runtime.model_registry import ModelRegistry

    models = args.model or [
        "Distill-Any-Depth-Small",
        "Distill-Any-Depth-Base",
    ]
    backends = args.backend or ["auto"]
    device = torch.device(
        args.device
        if args.device == "cpu" or torch.cuda.is_available()
        else "cpu"
    )
    rgb = load_rgb(args.rgb, device=device)
    if args.input_height or args.input_width:
        height = int(args.input_height or rgb.shape[-2])
        width = int(args.input_width or rgb.shape[-1])
        rgb = torch.nn.functional.interpolate(
            rgb,
            size=(height, width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    results: dict[str, Any] = {
        "rgb": str(args.rgb),
        "environment": {
            "python": sys.executable,
            "torch": torch.__version__,
            "torch_cuda": getattr(torch.version, "cuda", None),
            "torch_hip": getattr(torch.version, "hip", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "models": {},
    }

    for model_name in models:
        spec = ModelRegistry.default().get(model_name)
        model_results: dict[str, Any] = {}
        for backend in backends:
            print(f"[bench] model={spec.name} backend={backend}", flush=True)
            timings: list[float] = []
            preprocess: list[float] = []
            model_times: list[float] = []
            postprocess: list[float] = []
            profile_raw: list[dict[str, Any]] = []
            provider_report: dict[str, Any] | None = None
            output_shape: tuple[int, ...] | None = None
            status = "ok"
            error: str | None = None
            setup_ms = 0.0
            try:
                _sync(device)
                start = time.perf_counter()
                provider = _provider_for(
                    model_name=spec.name,
                    backend=backend,
                    device=device,
                    cache_dir=cache_dir,
                    onnx_dtype=args.onnx_dtype,
                    build_engine=args.build_engine,
                    force_rebuild=args.force_rebuild,
                    local_files_only=args.offline,
                    allow_pytorch_fallback=args.allow_pytorch_fallback,
                )
                load = getattr(provider, "load", None)
                if callable(load):
                    load()
                _sync(device)
                setup_ms = (time.perf_counter() - start) * 1000.0
                provider_report = provider.info.to_report() if hasattr(provider, "info") else None

                for iteration in range(args.warmup + args.iters):
                    _sync(device)
                    start = time.perf_counter()
                    result = provider.predict_profile(rgb)
                    _sync(device)
                    elapsed = (time.perf_counter() - start) * 1000.0
                    depth = result.depth
                    output_shape = tuple(int(value) for value in depth.shape)
                    profile = result.to_report()
                    if iteration >= args.warmup:
                        timings.append(elapsed)
                        preprocess.append(float(profile["preprocess_ms"]))
                        model_times.append(float(profile["model_ms"]))
                        postprocess.append(float(profile["postprocess_ms"]))
                        profile_raw.append(profile)
                    if hasattr(provider, "info"):
                        provider_report = provider.info.to_report()
                    print(f"  iter={iteration + 1}/{args.warmup + args.iters} {elapsed:.3f} ms", flush=True)
            except Exception as exc:  # noqa: BLE001 - benchmark must record every backend failure
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
                print(f"  failed: {error}", flush=True)

            model_results[backend] = {
                "status": status,
                "error": error,
                "provider": provider_report,
                "output_shape": output_shape,
                "setup_ms": float(setup_ms),
                "timings": summarize_ms(timings),
                "profile": {
                    "preprocess": summarize_ms(preprocess),
                    "model": summarize_ms(model_times),
                    "postprocess": summarize_ms(postprocess),
                },
                "profile_raw": profile_raw,
            }
        results["models"][spec.name] = model_results

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "depth_backend_bench.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(out_dir / "depth_backend_bench.csv", results)
    print(f"[done] wrote {json_path} and {out_dir / 'depth_backend_bench.csv'}", flush=True)


if __name__ == "__main__":
    main()
