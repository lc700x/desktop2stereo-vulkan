"""Build Markdown, JSON, CSV and contact-sheet evidence for Core ML matrix runs."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METRIC = re.compile(r"(\w+)=([-\w.]+)")
FIELDS = (
    "present_fps",
    "content_fps",
    "latency_p50_ms",
    "latency_p95_ms",
    "depth_total_ms_p95",
    "depth_model_ms_p95",
    "macos_metal_output_pack_ms_p95",
    "present_call_ms_p95",
    "viewer_queue_ms_p95",
    "runtime_execution_ms_p95",
    "capture_to_runtime_ready_ms_p95",
)
BASELINE_MODELS = {
    336: "Distill-Any-Depth-Small",
    518: "Distill-Any-Depth-Base",
}


def _portable_path(value: str | Path | None) -> str | None:
    if not value:
        return None
    path = Path(value)
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return path.name


def _offline_rows(root: Path) -> list[dict]:
    rows = []
    for report_path in sorted((root / "offline").glob("*/*/sample_report.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for model, result in report.get("models", {}).items():
            provider = result.get("provider") or {}
            plan = provider.get("coreml_compute_plan") or {}
            selected = provider.get("coreml_compute_units")
            candidate = (plan.get("candidates") or {}).get(selected, {})
            row = {
                "model": model,
                "resolution": report.get("depth_resolution"),
                "status": result.get("status"),
                "error": result.get("error"),
                "setup_ms": result.get("setup_ms"),
                "elapsed_p50_ms": (result.get("elapsed") or {}).get("p50_ms"),
                "elapsed_p90_ms": (result.get("elapsed") or {}).get("p90_ms"),
                "model_p50_ms": (result.get("model") or {}).get("p50_ms"),
                "depth_min": result.get("depth_min"),
                "depth_max": result.get("depth_max"),
                "depth_mean": result.get("depth_mean"),
                "nonzero_depth": result.get("nonzero_depth"),
                "compute_units": selected,
                "gpu_preferred_ops": (candidate.get("preferred") or {}).get(
                    "MLGPUComputeDevice", 0
                ),
                "ane_preferred_ops": (candidate.get("preferred") or {}).get(
                    "MLNeuralEngineComputeDevice", 0
                ),
                "cpu_only_ops": candidate.get("cpu_only_operator_count"),
                "finite_depth": result.get("finite_depth"),
                "artifact": _portable_path(provider.get("coreml_artifact_path")),
                "report_path": _portable_path(report_path),
            }
            baseline_model = (
                BASELINE_MODELS[336]
                if int(report.get("depth_resolution") or 0) <= 400
                else BASELINE_MODELS[518]
            )
            row["baseline_model"] = baseline_model
            rows.append(row)
    baseline_p50 = {
        (row["model"], int(row["resolution"])): row.get("elapsed_p50_ms")
        for row in rows
        if row["model"] in BASELINE_MODELS.values() and row.get("elapsed_p50_ms")
    }
    for row in rows:
        baseline = baseline_p50.get(
            (row["baseline_model"], 336 if int(row["resolution"] or 0) <= 400 else 518)
        )
        row["baseline_p50_ms"] = baseline
        candidate = row.get("elapsed_p50_ms")
        row["latency_ratio_vs_baseline"] = (
            float(candidate) / float(baseline)
            if candidate and baseline
            else None
        )
    return rows


def _metrics(log_path: Path) -> dict[str, float]:
    rows = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "[D2S_METRICS]" not in line:
            continue
        row = {}
        for key, value in METRIC.findall(line.split("[D2S_METRICS]", 1)[1]):
            try:
                row[key] = float(value)
            except ValueError:
                continue
        if row:
            rows.append(row)
    # Short viewer runs can contain only the first three metric windows.
    # Drop warmup rows only when enough steady-state data remains.
    if len(rows) > 4:
        rows = rows[3:]
    return {
        field: float(statistics.fmean(row[field] for row in rows if field in row))
        for field in FIELDS
        if any(field in row for row in rows)
    }


def _local_row(log_path: Path) -> dict:
    result: dict = _metrics(log_path)
    statuses = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "[D2S_BACKEND_STATUS]" not in line:
            continue
        try:
            statuses.append(json.loads(line.split("[D2S_BACKEND_STATUS]", 1)[1]))
        except ValueError:
            continue
    status = statuses[-1] if statuses else {}
    result.update(
        {
            "depth_backend": status.get("depth_backend"),
            "zero_copy": bool(status.get("zero_copy")),
            "coreml_input_shared": bool(status.get("coreml_input_shared")),
            "coreml_output_zero_copy": bool(status.get("coreml_output_zero_copy")),
            "fallback": bool(status.get("fallback")),
        }
    )
    return result


def _label(image, text: str):
    import cv2

    height, width = image.shape[:2]
    label_height = max(28, height // 16)
    canvas = cv2.copyMakeBorder(
        image,
        label_height,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(20, 20, 20),
    )
    cv2.putText(
        canvas,
        text,
        (8, max(20, label_height - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.45, width / 900.0),
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _contact_sheet(input_root: Path, image_root: Path) -> list[str]:
    import cv2
    import numpy as np

    output = input_root / "contact_sheets"
    output.mkdir(parents=True, exist_ok=True)
    written = []
    baselines = {"Distill-Any-Depth-Small", "Distill-Any-Depth-Base"}
    models = sorted(
        path.name
        for path in (image_root / "offline").glob("*")
        if path.is_dir() and path.name not in baselines
    )
    for model in models:
        columns = []
        model_dir = image_root / "offline" / model
        resolutions = sorted(
            int(path.name)
            for path in model_dir.iterdir()
            if path.is_dir() and path.name.isdigit()
        )
        for resolution in resolutions:
            baseline_resolution = 336 if resolution <= 400 else 518
            baseline = (
                "Distill-Any-Depth-Small"
                if baseline_resolution == 336
                else "Distill-Any-Depth-Base"
            )
            cells = []
            for source_model, source_resolution, caption in (
                (baseline, baseline_resolution, f"{baseline}@{baseline_resolution}"),
                (model, resolution, f"{model}@{resolution}"),
            ):
                for kind in ("depth", "half_sbs"):
                    path = (
                        image_root
                        / "offline"
                        / source_model
                        / str(source_resolution)
                        / f"{source_model}_{source_resolution}_{kind}.png"
                    )
                    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                    if image is None:
                        continue
                    cells.append(_label(image, f"{caption} {kind}"))
            if cells:
                width = min(image.shape[1] for image in cells)
                resized = [
                    cv2.resize(image, (width, round(image.shape[0] * width / image.shape[1])))
                    for image in cells
                ]
                columns.append(np.vstack(resized))
        if columns:
            width = min(column.shape[1] for column in columns)
            columns = [
                cv2.resize(column, (width, round(column.shape[0] * width / column.shape[1])))
                for column in columns
            ]
            sheet = np.hstack(columns)
            path = output / f"{model}_depth_half_sbs.jpg"
            cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
            written.append(str(path.relative_to(input_root)))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    root = args.input.resolve()
    root.mkdir(parents=True, exist_ok=True)
    offline = _offline_rows(root)
    local = {}
    for log_path in sorted((root / "local_viewer").glob("*/run.log")):
        local[log_path.parent.name] = _local_row(log_path)
    sheets = _contact_sheet(root, root)
    result = {
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "offline": offline,
        "local_viewer": local,
        "contact_sheets": sheets,
    }
    (root / "report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (root / "report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(offline[0]) if offline else ["model"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(offline)

    lines = [
        "# macOS Core ML Depth Matrix",
        "",
        f"- Platform: `{platform.platform()}`",
        f"- Machine: `{platform.machine()}`",
        "",
        "## Offline Depth and Half-SBS",
        "",
        "| Model | Resolution | Status | Units | Setup ms | Depth p50/p90 ms | Model p50 ms | Finite |",
        "|---|---:|---|---|---:|---:|---:|---|",
    ]
    for row in offline:
        lines.append(
            f"| {row['model']} | {row['resolution']} | {row['status']} | "
            f"{row['compute_units'] or '-'} | {row['setup_ms'] or 0:.1f} | "
            f"{row['elapsed_p50_ms'] or 0:.1f}/{row['elapsed_p90_ms'] or 0:.1f} | "
            f"{row['model_p50_ms'] or 0:.1f} | {row['finite_depth']} |"
        )
    lines.extend(
        [
            "",
            "Latency ratios below compare each candidate against the Distill model at the same tier:",
            "",
            "| Candidate | Resolution | Candidate p50 ms | Baseline | Baseline p50 ms | Ratio |",
            "|---|---:|---:|---|---:|---:|",
        ]
    )
    for row in offline:
        if row["model"] in BASELINE_MODELS.values():
            continue
        ratio = row.get("latency_ratio_vs_baseline")
        if ratio is None:
            line = (
                f"| {row['model']} | {row['resolution']} | - | "
                f"{row['baseline_model']} | - | - |"
            )
        else:
            line = (
                f"| {row['model']} | {row['resolution']} | "
                f"{row.get('elapsed_p50_ms') or 0:.1f} | {row['baseline_model']} | "
                f"{row.get('baseline_p50_ms') or 0:.1f} | {ratio:.2f}x |"
            )
        lines.append(line)
    lines.extend(["", "## Local Viewer", ""])
    lines.append(
        "| Model | Backend | Zero Copy | Present FPS | Content FPS | "
        "Latency p50/p95 ms | Depth p95 ms | Runtime p95 ms |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for model, row in sorted(local.items()):
        lines.append(
            f"| {model} | {row.get('depth_backend') or '-'} | "
            f"{'yes' if row.get('zero_copy') else 'no'} | "
            f"{row.get('present_fps', 0):.1f} | {row.get('content_fps', 0):.1f} | "
            f"{row.get('latency_p50_ms', 0):.1f}/{row.get('latency_p95_ms', 0):.1f} | "
            f"{row.get('depth_total_ms_p95', 0):.1f} | "
            f"{row.get('runtime_execution_ms_p95', 0):.1f} |"
        )
    lines.extend(["", "## Contact Sheets", ""])
    lines.extend(f"- [{Path(path).name}]({path})" for path in sheets)
    local_accelerator_only = bool(local) and all(
        row.get("depth_backend") == "coreml"
        and row.get("zero_copy")
        and not row.get("fallback")
        for row in local.values()
    )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            (
                "- Local Viewer ran the Metal zero-copy path; every row reports Core ML on GPU with no fallback."
                if local_accelerator_only
                else "- Local Viewer logs are incomplete or include a non-Core-ML/fallback row."
            ),
            "- MoGe-3 is much slower than the other families and is best tested at 336 for live use.",
            "- A first MoGe3-G@518 conversion includes one-time graph compilation and is not a steady-state frame latency.",
        ]
    )
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[report] wrote {root / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
