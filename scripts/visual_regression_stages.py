from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = REPO_ROOT / "src" / "desktop2stereo"
sys.path.insert(0, str(APP_ROOT))

from stereo_runtime.stage_visual_regression import (  # noqa: E402
    StageVisualRegressionConfig,
    run_stage_visual_regression_from_paths,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Save per-stage Vulkan/CUDA stereo visual regression images."
    )
    parser.add_argument("--rgb", required=True, help="Captured RGB frame path")
    parser.add_argument("--depth", help="Depth image path; proxy depth is generated when omitted")
    parser.add_argument("--output-dir", required=True, help="Directory for stage images and metrics")
    parser.add_argument("--cuda-reference-dir", help="Directory containing CUDA left_eye.png/right_eye.png")
    parser.add_argument("--skip-cuda", action="store_true", help="Do not run CUDA/Triton synthesis")
    parser.add_argument("--cuda-device", default="cuda", help="CUDA device for the reference run")
    parser.add_argument("--depth-strength", type=float, default=0.25)
    parser.add_argument("--convergence", type=float, default=0.0)
    parser.add_argument("--max-disparity-px", type=float)
    parser.add_argument("--parallax-preset", default="standard")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hole-fill-strength", type=float, default=0.6)
    parser.add_argument("--hole-fill-radius", type=int, default=1)
    parser.add_argument("--mask-feather-radius", type=int, default=3)
    parser.add_argument("--edge-dilation", type=int, default=2)
    parser.add_argument("--edge-threshold", type=float, default=0.04)
    parser.add_argument("--hole-fill-mode", default="balanced")
    parser.add_argument("--no-occlusion", action="store_true")
    args = parser.parse_args()

    config = StageVisualRegressionConfig(
        depth_strength=args.depth_strength,
        convergence=args.convergence,
        max_disparity_px=args.max_disparity_px,
        parallax_preset=args.parallax_preset,
        layers=args.layers,
        occlusion=not args.no_occlusion,
        edge_threshold=args.edge_threshold,
        edge_dilation=args.edge_dilation,
        mask_feather_radius=args.mask_feather_radius,
        hole_fill_radius=args.hole_fill_radius,
        hole_fill_strength=args.hole_fill_strength,
        hole_fill_mode=args.hole_fill_mode,
    )
    result = run_stage_visual_regression_from_paths(
        rgb_path=args.rgb,
        depth_path=args.depth,
        output_dir=args.output_dir,
        config=config,
        run_cuda=not args.skip_cuda,
        cuda_device=args.cuda_device,
        cuda_reference_dir=args.cuda_reference_dir,
    )
    print(json.dumps(result.get("comparison", {}), ensure_ascii=False, indent=2))
    print(f"visual regression output: {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
