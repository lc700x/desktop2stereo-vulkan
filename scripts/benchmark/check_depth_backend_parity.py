from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "desktop2stereo"))


def _spearman(a, b) -> float:
    import torch

    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    finite = torch.isfinite(a) & torch.isfinite(b)
    a = a[finite]
    b = b[finite]
    if a.numel() < 2:
        return 0.0
    rank_a = torch.argsort(torch.argsort(a)).float()
    rank_b = torch.argsort(torch.argsort(b)).float()
    rank_a -= rank_a.mean()
    rank_b -= rank_b.mean()
    denom = rank_a.norm() * rank_b.norm()
    return float((rank_a * rank_b).sum() / denom.clamp_min(1e-8))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a compiled depth backend with its PyTorch reference.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", required=True, choices=["migraphx_rocm", "tensorrt_native"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", default=str(ROOT / "models"))
    parser.add_argument("--width", type=int, default=None, help="Defaults to the compiled artifact width.")
    parser.add_argument("--height", type=int, default=None, help="Defaults to the compiled artifact height.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from stereo_runtime.depth_provider import DepthProviderConfig, create_depth_provider
    from stereo_runtime.io import load_rgb
    from stereo_runtime.model_artifacts import artifact_paths_for_model
    from stereo_runtime.model_registry import ModelRegistry

    spec = ModelRegistry.default().get(args.model)
    cache_dir = Path(args.cache_dir)
    paths = artifact_paths_for_model(
        spec.name,
        cache_dir=cache_dir,
        export_height=294,
        export_width=518,
    )
    height = int(args.height or paths.export_height)
    width = int(args.width or paths.export_width)
    rgb = load_rgb(args.image, device=torch.device(args.device))
    rgb = F.interpolate(
        rgb,
        size=(height, width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    reference = create_depth_provider(
        DepthProviderConfig(
            backend="pytorch_rocm" if torch.version.hip else "pytorch_cuda",
            model_id=spec.model_id,
            model_name=spec.name,
            device=args.device,
            cache_dir=cache_dir,
            depth_resolution=max(height, width),
            patch_size=32 if spec.family in {"depth-pro-2", "zipdepth"} else 14,
            local_files_only=True,
            allow_pytorch_fallback=False,
        )
    )
    candidate = create_depth_provider(
        DepthProviderConfig(
            backend=args.backend,
            model_id=spec.model_id,
            model_name=spec.name,
            device=args.device,
            cache_dir=cache_dir,
            onnx_path=paths.onnx_fp16_path,
            engine_path=paths.migraphx_fp16_path if args.backend == "migraphx_rocm" else paths.trt_fp16_path,
            local_files_only=True,
            allow_pytorch_fallback=False,
            depth_resolution=max(height, width),
            patch_size=32 if spec.family in {"depth-pro-2", "zipdepth"} else 14,
        )
    )
    reference.load()
    candidate.load()
    reference_depth = reference.predict_profile(rgb).depth
    candidate_depth = candidate.predict_profile(rgb).depth
    if torch.cuda.is_available():
        torch.cuda.synchronize(rgb.device)
    if tuple(reference_depth.shape) != tuple(candidate_depth.shape):
        candidate_depth = F.interpolate(
            candidate_depth,
            size=reference_depth.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    diff = (reference_depth - candidate_depth).abs()
    result = {
        "model": spec.name,
        "backend": args.backend,
        "reference_backend": reference.info.depth_backend,
        "candidate_backend": candidate.info.depth_backend,
        "shape": list(reference_depth.shape),
        "finite": bool(torch.isfinite(reference_depth).all() and torch.isfinite(candidate_depth).all()),
        "spearman": _spearman(reference_depth, candidate_depth),
        "reference_stats": {
            "min": float(reference_depth.min()),
            "mean": float(reference_depth.mean()),
            "max": float(reference_depth.max()),
        },
        "candidate_stats": {
            "min": float(candidate_depth.min()),
            "mean": float(candidate_depth.mean()),
            "max": float(candidate_depth.max()),
        },
        "normalized_mae": float(diff.mean()),
        "max_abs": float(diff.max()),
        "reference_provider": reference.info.to_report(),
        "candidate_provider": candidate.info.to_report(),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
