from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "desktop2stereo"))

DEFAULT_MODELS = [
    "Distill-Any-Depth-Small",
    "Distill-Any-Depth-Base",
    "DepthPro2-S",
    "DepthPro2-B",
    "DepthPro2-L",
    "DepthPro2-H",
    "MoGe3-L",
    "MoGe3-G",
    "FlashDepth-S",
    "FlashDepth-L",
    "FlashDepth-Full",
]


def _save_rgb(path: Path, tensor) -> None:
    import cv2

    image = tensor.detach()
    if image.ndim == 4:
        image = image[0]
    image = image.clamp(0.0, 1.0).mul(255.0).byte().permute(1, 2, 0).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def _save_depth(path: Path, tensor) -> None:
    import cv2

    depth = tensor.detach()
    if depth.ndim == 4:
        depth = depth[0]
    depth = depth[0].clamp(0.0, 1.0).mul(255.0).byte().cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), depth)


def _sync(device) -> None:
    import torch

    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return float(ordered[index])


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p90_ms": 0.0}
    return {
        "count": len(values),
        "mean_ms": float(statistics.fmean(values)),
        "p50_ms": float(statistics.median(values)),
        "p90_ms": _percentile(values, 0.9),
    }


def _load_frame(
    *,
    image: Path | None,
    video: Path | None,
    source_time: float,
    width: int,
    height: int,
    device,
):
    import cv2
    import numpy as np
    import torch

    if image is not None:
        frame_bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise FileNotFoundError(f"could not read image: {image}")
    elif video is not None:
        capture = cv2.VideoCapture(str(video))
        try:
            if not capture.isOpened():
                raise FileNotFoundError(f"could not open video: {video}")
            capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, source_time) * 1000.0)
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                raise RuntimeError(f"could not decode frame at {source_time:.3f}s from {video}")
        finally:
            capture.release()
    else:
        raise ValueError("one of image or video is required")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (int(width), int(height)), interpolation=cv2.INTER_AREA)
    return (
        torch.from_numpy(np.ascontiguousarray(frame_rgb))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .div_(255.0)
        .to(device)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate depth/Half-SBS samples and timing evidence for D2S depth providers."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--video", type=Path)
    parser.add_argument("--source-time", type=float, default=0.0)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "depth-model-eval")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--device", default="mps" if platform.system() == "Darwin" else "cuda")
    parser.add_argument(
        "--backend",
        choices=["auto", "coreml", "pytorch_mps", "pytorch_cuda", "pytorch_rocm"],
        default="coreml" if platform.system() == "Darwin" else "pytorch_cuda",
    )
    parser.add_argument("--compute-units", choices=["auto", "gpu", "ane"], default="gpu")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--depth-resolution", type=int, required=True)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    import torch
    from stereo_runtime.depth_provider import DepthProviderConfig, create_depth_provider
    from stereo_runtime.model_registry import ModelRegistry
    from stereo_runtime.synthesis import StereoConfig, synthesize_stereo

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable; refusing CPU fallback")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested but is unavailable; refusing CPU fallback")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb = _load_frame(
        image=args.image,
        video=args.video,
        source_time=args.source_time,
        width=args.width,
        height=args.height,
        device=device,
    )
    _save_rgb(out_dir / "source_rgb.png", rgb)
    report: dict[str, object] = {
        "source": str(args.image or args.video),
        "source_time_s": args.source_time,
        "source_size": [args.width, args.height],
        "backend": args.backend,
        "device": str(device),
        "depth_resolution": args.depth_resolution,
        "compute_units": args.compute_units,
        "models": {},
    }

    os.environ["D2S_COREML_COMPUTE_UNITS"] = args.compute_units
    registry = ModelRegistry.default()
    for model_name in args.models:
        spec = registry.get(model_name)
        print(f"[sample] {spec.name} resolution={args.depth_resolution}", flush=True)
        provider = None
        try:
            use_coreml = args.backend == "coreml"
            backend = "auto" if use_coreml else args.backend
            provider = create_depth_provider(
                DepthProviderConfig(
                    backend=backend,
                    model_id=spec.model_id,
                    model_name=spec.name,
                    device=device,
                    cache_dir=args.cache_dir,
                    depth_resolution=args.depth_resolution,
                    local_files_only=args.offline,
                    allow_pytorch_fallback=False,
                    use_coreml=use_coreml,
                )
            )
            setup_start = time.perf_counter()
            provider.load()
            _sync(device)
            setup_ms = (time.perf_counter() - setup_start) * 1000.0

            elapsed = []
            profile_rows = []
            depth = None
            for iteration in range(max(1, args.warmup + args.iters)):
                _sync(device)
                started = time.perf_counter()
                profile = provider.predict_profile(rgb)
                _sync(device)
                if iteration >= args.warmup:
                    elapsed.append((time.perf_counter() - started) * 1000.0)
                    profile_rows.append(profile.to_report())
                depth = profile.depth
            if depth is None:
                raise RuntimeError("provider returned no depth")
            depth_min = float(depth.min().item())
            depth_max = float(depth.max().item())
            depth_mean = float(depth.float().mean().item())
            nonzero_depth = int(torch.count_nonzero(depth).item())
            nonfinite_count = int((~torch.isfinite(depth)).sum().item())
            if (
                nonfinite_count
                or nonzero_depth == 0
                or depth_min < 0.0
                or depth_max > 1.0
                or depth_max <= depth_min
            ):
                raise RuntimeError(
                    f"invalid depth output: min={depth_min} max={depth_max} "
                    f"nonzero={nonzero_depth} nonfinite={nonfinite_count}"
                )

            _save_depth(out_dir / f"{spec.name}_{args.depth_resolution}_depth.png", depth)
            stereo = synthesize_stereo(
                rgb,
                depth,
                StereoConfig(
                    backend="quality_4k",
                    output_format="half_sbs",
                    temporal=False,
                    hole_fill="edge_aware",
                    occlusion=True,
                    symmetric=True,
                    fused=False,
                    max_disparity_px=48.0,
                ),
            )
            _sync(device)
            _save_rgb(
                out_dir / f"{spec.name}_{args.depth_resolution}_half_sbs.png",
                stereo.sbs,
            )
            report["models"][spec.name] = {
                "status": "ok",
                "provider": provider.info.to_report(),
                "setup_ms": setup_ms,
                "elapsed": _summary(elapsed),
                "preprocess": _summary([float(row["preprocess_ms"]) for row in profile_rows]),
                "model": _summary([float(row["model_ms"]) for row in profile_rows]),
                "postprocess": _summary([float(row["postprocess_ms"]) for row in profile_rows]),
                "profile_raw": profile_rows,
                "depth_shape": list(depth.shape),
                "half_sbs_shape": list(stereo.sbs.shape),
                "depth_min": depth_min,
                "depth_max": depth_max,
                "depth_mean": depth_mean,
                "nonzero_depth": nonzero_depth,
                "finite_depth": bool(torch.isfinite(depth).all().item()),
                "nonfinite_count": nonfinite_count,
            }
        except Exception as exc:  # noqa: BLE001 - matrix evidence must retain every failure
            report["models"][spec.name] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"[sample] failed {spec.name}: {type(exc).__name__}: {exc}", flush=True)
        finally:
            if provider is not None:
                close = getattr(provider, "close", None)
                if callable(close):
                    close()
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()

    report_path = out_dir / "sample_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
