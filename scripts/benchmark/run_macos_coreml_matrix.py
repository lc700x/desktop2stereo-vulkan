"""Run the macOS Core ML depth/SBS and Local Viewer evidence matrix."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NEW_MODELS = [
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
BASELINES = {
    336: "Distill-Any-Depth-Small",
    518: "Distill-Any-Depth-Base",
}
MODEL_RESOLUTIONS = {
    "DepthPro2-S": [320, 512],
    "DepthPro2-B": [320, 512],
    "DepthPro2-L": [320, 512],
    "DepthPro2-H": [320, 512],
    "MoGe3-L": [336, 518],
    "MoGe3-G": [336, 518],
    "FlashDepth-S": [336, 518],
    "FlashDepth-L": [336, 518],
    "FlashDepth-Full": [336, 518],
}
VIEWER_RESOLUTIONS = {
    "MoGe3-L": 336,
    "MoGe3-G": 336,
}


def _viewer_ok(log_path: Path, model: str) -> bool:
    if not log_path.is_file():
        return False
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if "[A/B] finished rc=0" not in text or "[D2S_METRICS]" not in text:
        return False
    if any(
        marker in text
        for marker in (
            "[process_runtime_loop] Fatal:",
            "[CoreMLNativeIO] prediction failed",
            "[A/B] FAILED",
        )
    ):
        return False
    statuses = []
    for line in text.splitlines():
        if "[D2S_BACKEND_STATUS]" not in line:
            continue
        try:
            statuses.append(json.loads(line.split("[D2S_BACKEND_STATUS]", 1)[1]))
        except ValueError:
            continue
    return bool(statuses) and any(
        status.get("depth_backend") == "coreml"
        and not status.get("fallback")
        and bool(status.get("coreml_input_shared"))
        and bool(status.get("coreml_output_zero_copy"))
        for status in statuses
    )


def _run(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[matrix] " + " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    print(f"[matrix] rc={process.returncode} log={log_path}", flush=True)
    return process.returncode


def _offline_ok(report_path: Path, model: str) -> bool:
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return report.get("models", {}).get(model, {}).get("status") == "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "macos_coreml_matrix")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "models")
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=None,
        help="Explicit override; defaults to each model's training-aligned profile.",
    )
    parser.add_argument("--models", nargs="*", default=NEW_MODELS)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--viewer-seconds", type=float, default=60.0)
    parser.add_argument("--min-free-gb", type=float, default=3.0)
    parser.add_argument("--compute-units", choices=["gpu"], default="gpu")
    parser.add_argument("--skip-offline", action="store_true")
    parser.add_argument("--skip-viewer", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    media = args.media.expanduser().resolve(strict=True)
    if not media.is_file():
        parser.error("--media must name a regular file")
    output_root = args.output_root.resolve()
    offline_root = output_root / "offline"
    viewer_root = output_root / "local_viewer"
    failures: list[dict[str, str]] = []

    if not args.skip_offline:
        baseline_jobs = [(model, resolution) for resolution, model in BASELINES.items()]
        for model, resolution in baseline_jobs:
            free_gb = shutil.disk_usage(ROOT).free / (1024 ** 3)
            if free_gb < args.min_free_gb:
                failures.append(
                    {
                        "stage": "offline",
                        "model": model,
                        "resolution": str(resolution),
                        "error": f"insufficient disk: {free_gb:.1f} GiB free",
                    }
                )
                continue
            report_path = offline_root / model / str(resolution) / "sample_report.json"
            if not args.force and _offline_ok(report_path, model):
                continue
            command = [
                sys.executable,
                str(ROOT / "scripts" / "benchmark" / "generate_depth_model_samples.py"),
                "--video",
                str(media),
                "--out-dir",
                str(report_path.parent),
                "--backend",
                "coreml",
                "--device",
                "mps",
                "--compute-units",
                args.compute_units,
                "--cache-dir",
                str(args.cache_dir),
                "--depth-resolution",
                str(resolution),
                "--models",
                model,
                "--warmup",
                str(args.warmup),
                "--iters",
                str(args.iters),
            ]
            rc = _run(command, report_path.parent / "run.log")
            if rc != 0 or not _offline_ok(report_path, model):
                failures.append({"stage": "offline", "model": model, "resolution": str(resolution)})

        for model in args.models:
            resolutions = args.resolutions or MODEL_RESOLUTIONS.get(model, [336, 518])
            for resolution in resolutions:
                free_gb = shutil.disk_usage(ROOT).free / (1024 ** 3)
                if free_gb < args.min_free_gb:
                    failures.append(
                        {
                            "stage": "offline",
                            "model": model,
                            "resolution": str(resolution),
                            "error": f"insufficient disk: {free_gb:.1f} GiB free",
                        }
                    )
                    continue
                report_path = offline_root / model / str(resolution) / "sample_report.json"
                if not args.force and _offline_ok(report_path, model):
                    continue
                command = [
                    sys.executable,
                    str(ROOT / "scripts" / "benchmark" / "generate_depth_model_samples.py"),
                    "--video",
                    str(media),
                    "--out-dir",
                    str(report_path.parent),
                    "--backend",
                    "coreml",
                    "--device",
                    "mps",
                    "--compute-units",
                    args.compute_units,
                    "--cache-dir",
                    str(args.cache_dir),
                    "--depth-resolution",
                    str(resolution),
                    "--models",
                    model,
                    "--warmup",
                    str(args.warmup),
                    "--iters",
                    str(args.iters),
                ]
                rc = _run(command, report_path.parent / "run.log")
                if rc != 0 or not _offline_ok(report_path, model):
                    failures.append(
                        {"stage": "offline", "model": model, "resolution": str(resolution)}
                    )

    if not args.skip_viewer:
        for model in args.models:
            model_resolutions = args.resolutions or MODEL_RESOLUTIONS.get(model, [336, 518])
            selected_resolution = VIEWER_RESOLUTIONS.get(model)
            override_ok = (
                selected_resolution is not None
                and selected_resolution in model_resolutions
                and _offline_ok(
                    offline_root / model / str(selected_resolution) / "sample_report.json",
                    model,
                )
            )
            if not override_ok:
                selected_resolution = None
                for resolution in sorted(model_resolutions, reverse=True):
                    report_path = offline_root / model / str(resolution) / "sample_report.json"
                    if _offline_ok(report_path, model):
                        selected_resolution = resolution
                        break
            if selected_resolution is None:
                failures.append({"stage": "viewer", "model": model, "resolution": "none"})
                continue
            log_path = viewer_root / model / "run.log"
            command = [
                sys.executable,
                str(ROOT / "scripts" / "benchmark" / "run_local_viewer_ab.py"),
                "--seconds",
                str(args.viewer_seconds),
                "--label",
                f"macos-coreml-{model}",
                "--model",
                model,
                "--resolution",
                str(selected_resolution),
                "--media",
                str(media),
                "--coreml",
                "on",
                "--compute-units",
                args.compute_units,
                "--interval",
                "5",
            ]
            rc = _run(command, log_path)
            if rc != 0 or not _viewer_ok(log_path, model):
                failures.append(
                    {"stage": "viewer", "model": model, "resolution": str(selected_resolution)}
                )

    summary = {
        "media": str(media),
        "models": args.models,
        "resolutions": {
            model: args.resolutions or MODEL_RESOLUTIONS.get(model, [336, 518])
            for model in args.models
        },
        "compute_units": args.compute_units,
        "failures": failures,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "matrix_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report_command = [
        sys.executable,
        str(ROOT / "scripts" / "benchmark" / "report_macos_coreml_matrix.py"),
        "--input",
        str(output_root),
    ]
    subprocess.run(report_command, cwd=ROOT, check=False)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
