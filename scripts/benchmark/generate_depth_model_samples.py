from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "desktop2stereo"))


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate depth maps and Half-SBS samples from a source frame.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--out-dir", default="outputs/depth-model-eval-20261128/samples")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", default=str(ROOT / "models"))
    parser.add_argument(
        "--models",
        nargs="*",
        default=[
            "Distill-Any-Depth-Small",
            "Distill-Any-Depth-Base",
            "DepthPro2-S",
            "MoGe3-L",
            "FlashDepth-S",
            "FlashDepth-Full",
        ],
    )
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from stereo_runtime.depth_provider import DepthProviderConfig, create_depth_provider
    from stereo_runtime.io import load_rgb
    from stereo_runtime.model_registry import ModelRegistry
    from stereo_runtime.synthesis import StereoConfig, synthesize_stereo

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb = load_rgb(args.image, device=torch.device(args.device))
    rgb = F.interpolate(
        rgb,
        size=(args.height, args.width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    _save_rgb(out_dir / "source_rgb.png", rgb)
    report: dict[str, object] = {
        "source": str(args.image),
        "source_size": [args.width, args.height],
        "models": {},
    }

    registry = ModelRegistry.default()
    for model_name in args.models:
        spec = registry.get(model_name)
        print(f"[sample] {spec.name}", flush=True)
        provider = create_depth_provider(
            DepthProviderConfig(
                backend="pytorch_rocm" if torch.version.hip else "pytorch_cuda",
                model_id=spec.model_id,
                model_name=spec.name,
                device=args.device,
                cache_dir=Path(args.cache_dir),
                depth_resolution=518,
                patch_size=32 if spec.family in {"depth-pro-2", "zipdepth"} else 14,
                local_files_only=True,
                allow_pytorch_fallback=False,
            )
        )
        provider.load()
        start = time.perf_counter()
        profile = provider.predict_profile(rgb)
        if profile.depth.is_cuda:
            torch.cuda.synchronize(rgb.device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        depth = profile.depth
        _save_depth(out_dir / f"{spec.name}_depth.png", depth)

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
                max_disparity_px=48.0,
            ),
        )
        _save_rgb(out_dir / f"{spec.name}_half_sbs.png", stereo.sbs)
        report["models"][spec.name] = {
            "provider": provider.info.to_report(),
            "depth_report": profile.to_report(),
            "elapsed_ms": elapsed_ms,
            "depth_shape": list(depth.shape),
            "half_sbs_shape": list(stereo.sbs.shape),
            "finite_depth": bool(torch.isfinite(depth).all().item()),
            "nonfinite_count": int((~torch.isfinite(depth)).sum().item()),
        }
        close = getattr(provider, "close", None)
        if callable(close):
            close()
        del provider
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (out_dir / "sample_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[done] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
