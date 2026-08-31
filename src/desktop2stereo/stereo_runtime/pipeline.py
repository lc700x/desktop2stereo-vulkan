from __future__ import annotations

import json
import os
import platform
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, is_dataclass, replace
from pathlib import Path
from typing import Callable

from capture.types import CapturedFrame, compatibility_frame

from .render_size import RenderSizeConfig, resolve_render_size, runtime_output_size_text
from .settings_snapshot import RuntimeSettingsPipelineRebuildRequired, RuntimeSettingsRestartRequired, RuntimeSettingsSnapshot


@dataclass
class _ParallelDepthJob:
    frame_id: int
    future: object
    frame_rgb: object
    runtime_rgb: object
    frame_raw: object
    size: object
    capture_start_time: float
    captured_frame: object
    render_size: object
    process_latency: float
    source_target_changed: bool = False
    render_size_changed: bool = False


class _ParallelDepthScheduler:
    """Bounded depth-only workers; presentation remains on the caller thread."""

    def __init__(self, runtime, max_workers: int = 2):
        self.runtime = runtime
        self.worker_count = max(1, min(int(max_workers), 3))
        self.executor = ThreadPoolExecutor(max_workers=self.worker_count,
                                            thread_name_prefix="d2s-depth")
        self.next_frame_id = 0
        self.pending: list[_ParallelDepthJob] = []
        self.dropped = 0
        self.effective_limit = self.worker_count

    def can_submit(self) -> bool:
        return len(self.pending) < self.effective_limit

    def set_effective_limit(self, limit: int) -> bool:
        resolved = max(1, min(self.worker_count, int(limit)))
        if resolved == self.effective_limit:
            return False
        self.effective_limit = resolved
        self.trim(resolved)
        return True

    def submit(self, runtime_rgb, **metadata) -> _ParallelDepthJob:
        frame_id = self.next_frame_id
        self.next_frame_id += 1
        future = self.executor.submit(self.runtime.predict_openxr_depth, runtime_rgb)
        job = _ParallelDepthJob(frame_id=frame_id, future=future, runtime_rgb=runtime_rgb, **metadata)
        self.pending.append(job)
        return job

    def pop_ready(self, *, block: bool = False) -> _ParallelDepthJob | None:
        if not self.pending:
            return None
        job = self.pending[0]
        if not job.future.done() and not block:
            return None
        try:
            job.depth_profile = job.future.result()
        except Exception:
            self.pending.pop(0)
            raise
        self.pending.pop(0)
        return job

    def trim(self, limit: int) -> None:
        limit = max(1, min(self.effective_limit, int(limit)))
        while len(self.pending) > limit:
            # Drop only work that has not started; cancelling a running CUDA
            # launch is unsafe and the bounded queue prevents accumulation.
            index = next((i for i, item in enumerate(self.pending) if not item.future.running()), None)
            if index is None:
                break
            job = self.pending.pop(index)
            if job.future.cancel():
                self.dropped += 1

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)

_OPENXR_FULL_SYNTHESIS_PRESETS = {"cinema", "game_low_latency", "still_image_hq", "debug_export"}


def _env_flag(name: str) -> bool:
    return str(os.environ.get(name, "0") or "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _runtime_diag_stage() -> str:
    if _env_flag("D2S_RUNTIME_DROP_ONLY"):
        return "raw"
    return str(os.environ.get("D2S_RUNTIME_DIAG_STAGE", "full") or "full").strip().lower()


def _save_preprocess_image_diagnostic(
    frame_rgb,
    *,
    capture_size,
    render_size,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Save the exact RGB image consumed by inference for one-shot comparison."""
    import cv2
    import numpy as np

    backend = str(getattr(frame_rgb, "_d2s_preprocess_backend", "unknown"))
    value = frame_rgb.detach() if hasattr(frame_rgb, "detach") else frame_rgb
    if getattr(value, "ndim", 0) == 4:
        value = value[0]
    if getattr(value, "ndim", 0) != 3:
        raise ValueError(
            "preprocess diagnostic requires a 3D RGB frame, "
            f"got {getattr(value, 'shape', None)}"
        )
    if int(value.shape[0]) in (3, 4):
        value = (
            value[:3].permute(1, 2, 0)
            if hasattr(value, "permute")
            else np.moveaxis(value[:3], 0, -1)
        )
    else:
        value = value[..., :3]
    if hasattr(value, "cpu"):
        value = value.cpu().numpy()
    image_rgb = np.asarray(value)
    if np.issubdtype(image_rgb.dtype, np.floating):
        image_rgb = np.rint(np.clip(image_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

    capture_width, capture_height = int(capture_size[0]), int(capture_size[1])
    render_width, render_height = int(render_size[0]), int(render_size[1])
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    stem = (
        f"preprocess_{capture_width}x{capture_height}_to_"
        f"{render_width}x{render_height}_{stamp}"
    )
    image_path = destination / f"{stem}.png"
    manifest_path = destination / f"{stem}.json"
    if not cv2.imwrite(str(image_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to save preprocess diagnostic image: {image_path}")
    manifest_path.write_text(
        json.dumps(
            {
                "stage": "after_capture_preprocess_before_inference",
                "capture_size": [capture_width, capture_height],
                "render_size": [render_width, render_height],
                "image_size": [int(image_rgb.shape[1]), int(image_rgb.shape[0])],
                "preprocess_backend": backend,
                "image": image_path.name,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return image_path, manifest_path


def _rcas_reference(image_rgb, sharpness: float = 0.35):
    """Small CPU reference for comparing RCAS after area downsampling."""
    import numpy as np

    image = np.asarray(image_rgb, dtype=np.float32) * (1.0 / 255.0)
    padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode="edge")
    center = padded[1:-1, 1:-1]
    up = padded[:-2, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    down = padded[2:, 1:-1]
    minimum = np.minimum(np.minimum(up, left), np.minimum(right, down))
    maximum = np.maximum(np.maximum(up, left), np.maximum(right, down))
    hit_min = minimum / np.maximum(4.0 * maximum, 1.0e-6)
    hit_max = (1.0 - maximum) / np.minimum(4.0 * minimum - 4.0, -1.0e-6)
    lobe = np.maximum(-hit_min, hit_max)
    lobe = np.maximum(-0.1875, np.minimum(lobe, 0.0))
    lobe *= 2.0 ** (-max(0.0, float(sharpness)))
    result = (lobe * (up + left + right + down) + center) / (4.0 * lobe + 1.0)
    return np.rint(np.clip(result, 0.0, 1.0) * 255.0).astype(np.uint8)


def _save_downsample_filter_comparison(
    frame_raw,
    *,
    capture_size,
    render_size,
    output_dir: str | Path,
) -> Path | None:
    """Export multiple downsample candidates from the same untouched raw frame."""
    import cv2
    import numpy as np

    capture_width, capture_height = int(capture_size[0]), int(capture_size[1])
    render_width, render_height = int(render_size[0]), int(render_size[1])
    if capture_width <= render_width or capture_height <= render_height:
        return None
    value = frame_raw.detach() if hasattr(frame_raw, "detach") else frame_raw
    if getattr(value, "ndim", 0) == 4:
        value = value[0]
    if hasattr(value, "cpu"):
        value = value.cpu().numpy()
    raw = np.asarray(value)
    if raw.ndim != 3 or raw.shape[-1] not in (3, 4):
        raise ValueError(f"downsample comparison requires HWC BGR/BGRA input, got {raw.shape}")
    source_rgb = cv2.cvtColor(
        raw,
        cv2.COLOR_BGRA2RGB if raw.shape[-1] == 4 else cv2.COLOR_BGR2RGB,
    )
    target = (render_width, render_height)
    area = cv2.resize(source_rgb, target, interpolation=cv2.INTER_AREA)
    lanczos4 = cv2.resize(source_rgb, target, interpolation=cv2.INTER_LANCZOS4)
    bicubic = cv2.resize(source_rgb, target, interpolation=cv2.INTER_CUBIC)
    area_rcas = _rcas_reference(area, 0.35)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    comparison_dir = Path(output_dir) / (
        f"downsample_compare_{capture_width}x{capture_height}_to_"
        f"{render_width}x{render_height}_{stamp}"
    )
    comparison_dir.mkdir(parents=True, exist_ok=True)
    images = {
        "00_source_rgb.png": source_rgb,
        "01_area.png": area,
        "02_lanczos4.png": lanczos4,
        "03_bicubic.png": bicubic,
        "04_area_rcas_035.png": area_rcas,
    }
    for name, image_rgb in images.items():
        path = comparison_dir / name
        if not cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"failed to save downsample comparison image: {path}")
    (comparison_dir / "manifest.json").write_text(
        json.dumps(
            {
                "stage": "same_raw_capture_downsample_comparison",
                "capture_size": [capture_width, capture_height],
                "render_size": [render_width, render_height],
                "candidates": list(images),
                "area_rcas_sharpness": 0.35,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return comparison_dir


@dataclass(frozen=True)
class RuntimePipelineContext:
    shutdown_event: object
    raw_q: object
    runtime_q: object
    time_sleep: float
    run_mode: str
    openxr_runtime_direct: bool
    stereo_active_preset: str | None
    device: object
    use_cudart: bool
    thread_latencies: dict
    stereo_runtime: object
    capture_frame_to_rgb: Callable
    prepare_rgb_for_stereo_runtime: Callable
    current_openxr_render_config: Callable[[], object]
    is_hard_idle: Callable[[], bool]
    is_source_paused: Callable[[], bool]
    log_source_health: Callable[[], None]
    source_stat_inc: Callable[..., None]
    breakdown_inc: Callable[..., None]
    breakdown_add_time: Callable[..., None]
    breakdown_add_runtime_timing: Callable[..., None]
    set_preprocess_backend: Callable[[str], None]
    queue_clear: Callable[[object], None]
    queue_drain_latest: Callable[[object, object], object]
    queue_put_latest: Callable[[object, object], None]
    log_stereo_runtime_mode_once: Callable[[], None]
    apply_stereo_hot_reload_if_needed: Callable[[], None]
    warmup_stereo_once_for_frame: Callable[[object], None]
    log_fast_plus_fused_runtime_state: Callable[[object], None]
    runtime_ready_event: object | None = None
    application_runtime_target: str | None = None
    output_transport: str | None = None
    settings_update_q: object | None = None
    render_size_config: RenderSizeConfig | None = None
    runtime_config: object | None = None
    openxr_presenter_pressure: Callable[[], bool] | None = None


def _drain_latest_nowait(q: object | None):
    if q is None:
        return None
    latest = None
    while True:
        try:
            latest = q.get_nowait()
        except queue.Empty:
            return latest


def _openxr_full_synthesis_enabled(ctx: RuntimePipelineContext) -> bool:
    if ctx.run_mode != "OpenXR":
        return False
    if not ctx.openxr_runtime_direct:
        return True
    return str(ctx.stereo_active_preset or "").strip().lower() in _OPENXR_FULL_SYNTHESIS_PRESETS


def _openxr_realtime_synthesis_config(config):
    if config is None:
        return None
    updates = {}
    if bool(getattr(config, "temporal", False)):
        updates.update(temporal=False, temporal_strength=0.0, auto_reset_temporal=False)
    return replace(config, **updates) if updates and is_dataclass(config) else config


def _active_preset_for_snapshot(settings_snapshot, active_preset):
    snapshot_preset = getattr(settings_snapshot, "stereo_preset", None)
    if snapshot_preset == "auto":
        return active_preset
    return snapshot_preset or active_preset


def _apply_latest_settings_snapshot(ctx: RuntimePipelineContext):
    settings_snapshot = _drain_latest_nowait(ctx.settings_update_q)
    if settings_snapshot is None:
        return None
    change_class = ctx.stereo_runtime.apply_settings_snapshot(
        settings_snapshot,
        active_preset=_active_preset_for_snapshot(settings_snapshot, ctx.stereo_active_preset),
    )
    ctx.source_stat_inc(
        "settings_updates",
        last_settings_version=int(settings_snapshot.version),
        last_settings_change_class=change_class.value,
    )
    return change_class


def _captured_frame_uses_cuda_capture(captured_frame: CapturedFrame | None) -> bool:
    return str(getattr(captured_frame, "capture_tool", "") or "").strip().lower() == "windowscapturecuda"


def _enable_openxr_depth_cuda_graph_if_needed(
    ctx: RuntimePipelineContext,
    openxr_full_synthesis: bool,
    captured_frame: CapturedFrame | None = None,
) -> None:
    if not openxr_full_synthesis:
        return
    if _captured_frame_uses_cuda_capture(captured_frame):
        config = getattr(ctx.stereo_runtime, "config", None)
        if config is not None and bool(getattr(config, "use_cuda_graph", False)):
            # WindowsCaptureCUDA owns a blocking capture stream. Rebuilding the
            # provider removes any graph captured before the capture backend was selected.
            ctx.stereo_runtime.apply_settings_snapshot(
                RuntimeSettingsSnapshot(version=int(time.time() * 1000), timestamp=time.time(), use_cuda_graph=False),
                active_preset=ctx.stereo_active_preset,
            )
            ctx.source_stat_inc("openxr_depth_cuda_graph_disabled_cuda_capture")
        ctx.source_stat_inc("openxr_depth_cuda_graph_skipped_cuda_capture")
        return
    config = getattr(ctx.stereo_runtime, "config", None)
    if config is None or bool(getattr(config, "use_cuda_graph", False)):
        return
    provider = getattr(ctx.stereo_runtime, "depth_provider", None)
    if getattr(provider, "_cuda_graph_disabled_reason", None):
        return
    backend = str(getattr(config, "depth_backend", "auto") or "auto").strip().lower()
    if backend not in {"auto", "tensorrt_native"}:
        return
    ctx.stereo_runtime.apply_settings_snapshot(
        RuntimeSettingsSnapshot(version=int(time.time() * 1000), timestamp=time.time(), use_cuda_graph=True),
        active_preset=ctx.stereo_active_preset,
    )
    ctx.source_stat_inc("openxr_depth_cuda_graph_enabled")


def _unpack_raw_queue_item(item):
    if isinstance(item, CapturedFrame):
        return compatibility_frame(item), item.target_height, item.timestamp, item
    frame_raw, size, capture_start_time = item
    return frame_raw, size, capture_start_time, None

def _runtime_depth_backend(runtime) -> str:
    try:
        report = runtime.provider_report()
    except Exception:
        report = {}
    backend = report.get("depth_backend") if isinstance(report, dict) else None
    if not backend:
        backend = getattr(getattr(runtime, "config", None), "depth_backend", "")
    return str(backend or "").strip().lower()


def _runtime_consumer_adapter_luid(runtime) -> int:
    candidates = (
        getattr(runtime, "_vulkan_stereo_backend", None),
        getattr(runtime, "_vulkan_context", None),
        getattr(getattr(runtime, "config", None), "adapter_luid", None),
    )
    for candidate in candidates:
        value = getattr(candidate, "adapter_luid", candidate)
        try:
            value = int(value or 0)
        except (TypeError, ValueError):
            value = 0
        if value:
            return value
    return 0


def _prepare_frame_input(ctx, captured_frame, frame_raw):
    if captured_frame is None or _runtime_depth_backend(ctx.stereo_runtime) != "directml":
        return frame_raw
    from .providers.directml_resource import prepare_directml_input

    prepared, decision = prepare_directml_input(
        captured_frame,
        consumer_adapter_luid=_runtime_consumer_adapter_luid(ctx.stereo_runtime),
        allow_cpu_fallback=True,
    )
    if decision is not None:
        captured_frame.metadata.update({
            "directml_resource_mode": decision.mode,
            "directml_resource_reason": decision.reason,
            "directml_consumer_adapter_luid": decision.consumer_adapter_luid,
            "directml_gpu_to_cpu": decision.gpu_to_cpu,
            "directml_gpu_copy_count": decision.gpu_copy_count,
            "directml_zero_copy_ready": decision.zero_copy_ready,
            "directml_fallback_reason": (
                decision.reason if decision.mode == "cpu_compat" else None
            ),
        })
    return prepared


def _resolve_pipeline_render_size(size, config: RenderSizeConfig | None):
    if config is None:
        return size
    if isinstance(size, (tuple, list)) and len(size) == 2:
        return resolve_render_size((int(size[0]), int(size[1])), config)
    return size


def _reset_runtime_temporal_state(runtime) -> None:
    temporal_state = getattr(runtime, "temporal_state", None)
    reset_stereo = getattr(temporal_state, "reset_stereo", None)
    if callable(reset_stereo):
        reset_stereo()


def _source_target_key(captured_frame: CapturedFrame | None):
    if captured_frame is None:
        return ("legacy",)
    metadata = captured_frame.metadata if isinstance(captured_frame.metadata, dict) else {}
    metadata_parts = tuple(
        (key, str(metadata[key]))
        for key in ("source_id", "source_key", "capture_source", "capture_target", "target_id", "window_handle", "hwnd")
        if key in metadata and metadata[key] is not None
    )
    return (
        "captured",
        str(captured_frame.capture_mode or ""),
        int(captured_frame.monitor_index or 0),
        str(captured_frame.window_title or ""),
        metadata_parts,
    )


def _append_temporal_reset_reason(debug_info: dict, reason: str) -> None:
    current = debug_info.get("temporal_reset_reason")
    if not current:
        debug_info["temporal_reset_reason"] = reason
        return
    reasons = [part.strip() for part in str(current).split(",") if part.strip()]
    if reason not in reasons:
        reasons.append(reason)
    debug_info["temporal_reset_reason"] = ",".join(reasons)


def _rgb_size_text(frame) -> str:
    shape = tuple(getattr(frame, "shape", ()))
    if len(shape) == 4:
        return f"{int(shape[3])}x{int(shape[2])}"
    if len(shape) == 3 and shape[0] in (3, 4):
        return f"{int(shape[2])}x{int(shape[1])}"
    if len(shape) == 3:
        return f"{int(shape[1])}x{int(shape[0])}"
    return "unknown"


def _capture_zero_copy(captured_frame: CapturedFrame | None):
    if captured_frame is None:
        return None
    if "zero_copy" in captured_frame.metadata:
        return bool(captured_frame.metadata["zero_copy"])
    return captured_frame.copy_mode.value == "none"


def _capture_copy_mode(captured_frame: CapturedFrame | None):
    if captured_frame is None:
        return None
    metadata = captured_frame.metadata if isinstance(captured_frame.metadata, dict) else {}
    if captured_frame.cpu_compat_frame is not None:
        return str(metadata.get("compatibility_copy_mode") or "cpu_compat")
    return captured_frame.copy_mode.value


def _capture_debug_fields(captured_frame: CapturedFrame | None, frame_rgb) -> dict:
    fields = {
        "preprocess_device_origin": getattr(frame_rgb, "_d2s_preprocess_device_origin", None),
        "preprocess_device_output": getattr(frame_rgb, "_d2s_preprocess_device_output", None),
        "preprocess_device_transfer": getattr(frame_rgb, "_d2s_preprocess_device_transfer", None),
        "preprocess_input_kind": getattr(frame_rgb, "_d2s_preprocess_input_kind", None),
        "capture_copy_mode": getattr(frame_rgb, "_d2s_capture_copy_mode", _capture_copy_mode(captured_frame)),
        "capture_zero_copy": getattr(frame_rgb, "_d2s_capture_zero_copy", _capture_zero_copy(captured_frame)),
    }
    if captured_frame is not None:
        metadata = captured_frame.metadata if isinstance(captured_frame.metadata, dict) else {}
        fields.update(
            capture_tool=captured_frame.capture_tool,
            capture_mode=captured_frame.capture_mode,
            capture_frame_raw_device=captured_frame.frame_raw_device,
            capture_frame_raw_type=captured_frame.frame_raw_type,
            capture_frame_raw_dtype=captured_frame.frame_raw_dtype,
            capture_size=captured_frame.capture_size or (
                (
                    metadata.get("resource_width"),
                    metadata.get("resource_height"),
                )
                if metadata.get("resource_width") is not None
                or metadata.get("resource_height") is not None
                else None
            ),
            capture_resource_kind=metadata.get("resource_kind"),
            capture_resource_format=metadata.get("resource_format"),
            capture_adapter_luid=metadata.get("adapter_luid"),
            capture_adapter_uuid=metadata.get("adapter_uuid"),
            capture_pci_bdf=metadata.get("pci_bdf"),
            capture_adapter_identity=metadata.get("adapter_identity"),
            capture_gpu_to_cpu=metadata.get("gpu_to_cpu"),
            capture_gpu_copy_count=metadata.get("gpu_copy_count"),
            capture_zero_copy_ready=metadata.get("zero_copy_ready"),
            capture_resource_lifecycle=metadata.get("resource_lifecycle"),
            capture_native_resource_output=metadata.get("native_resource_output"),
            capture_compatibility_frame_retained=metadata.get("compatibility_frame_retained"),
            capture_compatibility_copy_mode=metadata.get("compatibility_copy_mode"),
            directml_resource_mode=metadata.get("directml_resource_mode"),
            directml_resource_reason=metadata.get("directml_resource_reason"),
            directml_consumer_adapter_luid=metadata.get("directml_consumer_adapter_luid"),
            directml_gpu_to_cpu=metadata.get("directml_gpu_to_cpu"),
            directml_gpu_copy_count=metadata.get("directml_gpu_copy_count"),
            directml_zero_copy_ready=metadata.get("directml_zero_copy_ready"),
            directml_fallback_reason=metadata.get("directml_fallback_reason"),
            capture_fallback_reason=metadata.get("fallback_reason"),
        )
    return {key: value for key, value in fields.items() if value is not None}


def _attach_capture_debug(runtime_result, captured_frame: CapturedFrame | None, frame_rgb) -> None:
    debug_info = getattr(runtime_result, "debug_info", None)
    if isinstance(debug_info, dict):
        debug_info.update(_capture_debug_fields(captured_frame, frame_rgb))


def _attach_pipeline_debug(
    runtime_result,
    *,
    capture_size,
    render_size,
    run_mode,
    render_size_config,
    application_runtime_target=None,
    output_transport=None,
) -> None:
    debug_info = getattr(runtime_result, "debug_info", None)
    if not isinstance(debug_info, dict):
        return
    debug_info["capture_size"] = _size_debug_text(capture_size)
    debug_info["render_size"] = _size_debug_text(render_size)
    debug_info["application_runtime_target"] = _application_target_debug_label(
        run_mode,
        application_runtime_target=application_runtime_target,
    )
    transport = _transport_debug_label(run_mode, output_transport=output_transport)
    debug_info["transport"] = transport
    debug_info["output_transport"] = transport
    if render_size_config is not None:
        debug_info["render_size_policy"] = render_size_config.policy.value
        debug_info["stereo_render_scale"] = render_size_config.scale_factor


def _application_target_debug_label(run_mode, *, application_runtime_target=None) -> str:
    if application_runtime_target:
        return str(application_runtime_target)
    if run_mode == "OpenXR":
        return "openxr"
    if run_mode in {"MJPEG", "RTMP", "Streamer", "MJPEG Streamer", "RTMP Streamer"}:
        return "network_stream"
    return "local_viewer"


def _transport_debug_label(run_mode, *, output_transport=None) -> str:
    if output_transport:
        return str(output_transport)
    return "openxr_swapchain" if run_mode == "OpenXR" else "local_window"


def _size_debug_text(size) -> str:
    if isinstance(size, (tuple, list)) and len(size) == 2:
        try:
            return runtime_output_size_text((int(size[0]), int(size[1])))
        except (TypeError, ValueError):
            pass
    return str(size)


def _run_depth_only(runtime, runtime_rgb) -> None:
    load = getattr(runtime, "load", None)
    if callable(load):
        load()
    predict = getattr(runtime, "_predict_depth_profile")
    predict(runtime_rgb)


def _synchronize_runtime_device(runtime_rgb) -> None:
    device = getattr(runtime_rgb, "device", None)
    if getattr(device, "type", None) != "cuda":
        return
    try:
        import torch

        torch.cuda.synchronize(device)
    except Exception:
        return


def _runtime_sync_after_frame_enabled(ctx: RuntimePipelineContext) -> bool:
    value = str(os.environ.get("D2S_RUNTIME_SYNC_AFTER_FRAME", "0") or "0").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off", "auto"}:
        return False
    return False


def _runtime_result_cuda_device(runtime_result):
    for name in ("left_eye", "right_eye", "depth", "source_rgb"):
        tensor = getattr(runtime_result, name, None)
        device = getattr(tensor, "device", None)
        if getattr(device, "type", None) == "cuda":
            return device
    return None


def _attach_cuda_ready_event(runtime_result):
    if getattr(runtime_result, "cuda_ready_event", None) is not None:
        return runtime_result
    device = _runtime_result_cuda_device(runtime_result)
    if device is None:
        return runtime_result
    try:
        import torch

        event = torch.cuda.Event(blocking=False)
        event.record(torch.cuda.current_stream(device))
        if is_dataclass(runtime_result):
            return replace(runtime_result, cuda_ready_event=event)
        setattr(runtime_result, "cuda_ready_event", event)
    except Exception:
        return runtime_result
    return runtime_result


def _cuda_event_ready(event) -> bool:
    if event is None:
        return True
    query = getattr(event, "query", None)
    if not callable(query):
        return True
    try:
        return bool(query())
    except Exception:
        return True


def _runtime_supports_parallel_cuda_pending(ctx: RuntimePipelineContext) -> bool:
    runtime_config = getattr(ctx, "runtime_config", None)
    if runtime_config is not None and not bool(
        getattr(runtime_config, "parallel_inference", False)
    ):
        return False
    if ctx.run_mode not in {"OpenXR", "Viewer"}:
        return False
    runtime = ctx.stereo_runtime
    provider = getattr(runtime, "depth_provider", None)
    if int(getattr(provider, "pipeline_slot_count", 1)) < 2:
        return False
    if bool(getattr(getattr(runtime, "config", None), "profile_sync", False)):
        return False
    resolved_backend = str(
        getattr(runtime, "_resolved_stereo_compute_backend", "") or ""
    ).strip().lower()
    if resolved_backend not in {"triton", "cuda", "cuda_triton"}:
        # On macOS the "vulkan" label just means "not triton": synthesis runs
        # the torch-fast backend on MPS (the Vulkan fused kernel path is
        # unavailable without a GLSL compiler), so the presenter-side consumer
        # lease concern does not apply. Opt in via D2S_RUNTIME_PARALLEL_MPS=1.
        import sys as _sys

        # "" is the steady state on the torch-fast path: the resolver only
        # runs for the fused/vulkan kernel attempt, which macOS never takes.
        if not resolved_backend and _sys.platform == "darwin":
            resolved_backend = "vulkan"
        is_mps_mac = _sys.platform == "darwin" and resolved_backend == "vulkan"
        if not (is_mps_mac and os.environ.get("D2S_RUNTIME_PARALLEL_MPS", "0") == "1"):
            # Vulkan deferred stereo has a separate presenter-side consumer lease;
            # keep its depth queue single-pending until that path is made safe.
            return False
    if ctx.run_mode == "OpenXR":
        realtime_config = _openxr_realtime_synthesis_config(
            getattr(runtime, "stereo_config", None)
        )
        if bool(getattr(realtime_config, "temporal", False)):
            return False
        convergence = getattr(realtime_config, "convergence", None)
        if str(getattr(type(convergence), "__module__", "")).startswith("torch"):
            return False
    return True


def _runtime_pending_depth_limit(ctx: RuntimePipelineContext | None = None) -> int:
    raw = str(os.environ.get("D2S_RUNTIME_PENDING_CUDA_DEPTH", "auto") or "auto").strip().lower()
    requested: int | None
    try:
        requested = max(1, int(raw))
    except ValueError:
        requested = None
    if ctx is None or not _runtime_supports_parallel_cuda_pending(ctx):
        return 1
    provider = getattr(ctx.stereo_runtime, "depth_provider", None)
    slot_count = max(1, int(getattr(provider, "pipeline_slot_count", 1)))
    if requested is None:
        runtime_config = getattr(ctx, "runtime_config", None)
        requested = max(1, min(
            slot_count,
            int(getattr(runtime_config, "parallel_inference_workers", 2) or 2),
        ))
    return min(slot_count, requested)


def _runtime_pending_cuda_wait_s(ctx) -> float:
    raw = os.environ.get("D2S_RUNTIME_PENDING_CUDA_WAIT_MS")
    if raw is None:
        raw = "0"
    try:
        return max(0.0, float(str(raw).strip()) / 1000.0)
    except ValueError:
        return 0.0


def _runtime_max_algorithm_fps(ctx: RuntimePipelineContext) -> float:
    default = "30" if ctx.run_mode == "OpenXR" else "0"
    raw = os.environ.get("D2S_RUNTIME_MAX_ALGO_FPS", default)
    try:
        return max(0.0, float(str(raw).strip()))
    except ValueError:
        return 0.0


def _runtime_motion_gate_enabled(ctx: RuntimePipelineContext) -> bool:
    # Keep the legacy default: frame reuse remains available as an explicit
    # optimization, but must not throttle normal SBS updates by default.
    default = "0"
    return str(os.environ.get("D2S_RUNTIME_MOTION_GATE", default) or default).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _runtime_parallel_adaptive_backoff_enabled() -> bool:
    value = str(os.environ.get("D2S_RUNTIME_PARALLEL_ADAPTIVE_BACKOFF", "0") or "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _runtime_presenter_backpressure_enabled(ctx: RuntimePipelineContext) -> bool:
    if ctx.run_mode != "OpenXR" or not callable(
        getattr(ctx, "openxr_presenter_pressure", None)
    ):
        return False
    value = str(
        os.environ.get("D2S_RUNTIME_PRESENTER_BACKPRESSURE", "1") or "1"
    ).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _runtime_motion_threshold(name: str, default: float) -> float:
    try:
        return max(0.0, float(str(os.environ.get(name, default)).strip()))
    except ValueError:
        return default


def _motion_sample(frame_rgb):
    shape = tuple(getattr(frame_rgb, "shape", ()))
    if len(shape) < 2:
        return None
    try:
        import torch
        if isinstance(frame_rgb, torch.Tensor):
            sample = frame_rgb.detach()
            if sample.ndim == 4:
                sample = sample[0]
            if sample.ndim == 3 and sample.shape[0] in (3, 4):
                h, w = int(sample.shape[1]), int(sample.shape[2])
                sample = sample[:, ::max(1, h // 18), ::max(1, w // 32)]
            elif sample.ndim >= 2:
                h, w = int(sample.shape[0]), int(sample.shape[1])
                sample = sample[::max(1, h // 18), ::max(1, w // 32)]
            is_integer = not torch.is_floating_point(sample)
            # Capture/preprocess backends may reuse the same tensor storage for
            # the next frame. Keep the motion reference independent of it.
            sample = sample.to(dtype=torch.float32).clone()
            return sample / 255.0 if is_integer else sample
    except Exception:
        pass
    try:
        import numpy as np
        sample = np.asarray(frame_rgb)
        if sample.ndim == 4:
            sample = sample[0]
        if sample.ndim == 3 and sample.shape[0] in (3, 4):
            h, w = int(sample.shape[1]), int(sample.shape[2])
            sample = sample[:, ::max(1, h // 18), ::max(1, w // 32)]
        elif sample.ndim >= 2:
            h, w = int(sample.shape[0]), int(sample.shape[1])
            sample = sample[::max(1, h // 18), ::max(1, w // 32)]
        is_integer = np.issubdtype(sample.dtype, np.integer)
        sample = sample.astype("float32", copy=True)
        return sample / 255.0 if is_integer else sample
    except Exception:
        return None


def _motion_score(previous, current) -> float | None:
    if previous is None or current is None:
        return None
    if tuple(getattr(previous, "shape", ())) != tuple(getattr(current, "shape", ())):
        return None
    try:
        import torch
        if isinstance(current, torch.Tensor):
            return float((current - previous).abs().mean().item())
    except Exception:
        pass
    try:
        import numpy as np
        return float(np.mean(np.abs(current - previous)))
    except Exception:
        return None


def _is_fatal_runtime_preparation_error(exc: Exception) -> bool:
    if isinstance(exc, FileNotFoundError):
        return True
    text = str(exc or "").lower()
    if not text:
        return False
    fatal_markers = (
        "unable to resolve infinidepth weights",
        "model directory not found",
        "onnx artifact not found",
        "tensorrt engine not found",
        "download completed but model directory was not found",
        "model weights resolved but model directory was not found",
    )
    return any(marker in text for marker in fatal_markers)


def _cuda_elapsed_ms(events: dict, start: str, end: str) -> float | None:
    first = events.get(start)
    second = events.get(end)
    if first is None or second is None:
        return None
    elapsed_time = getattr(first, "elapsed_time", None)
    if not callable(elapsed_time):
        return None
    try:
        return float(elapsed_time(second))
    except Exception:
        return None


def _add_cuda_event_timings(ctx: RuntimePipelineContext, runtime_result) -> None:
    events = getattr(runtime_result, "cuda_timing_events", None)
    if not isinstance(events, dict) or not events:
        return
    for name, start, end in (
        ("rt_gpu_depth", "start", "depth"),
        ("rt_gpu_depth_preprocess", "depth_pre_start", "depth_pre_end"),
        ("rt_gpu_depth_model", "depth_model_start", "depth_model_end"),
        ("rt_gpu_depth_normalize", "depth_norm_start", "depth_norm_end"),
        ("rt_gpu_depth_upsample", "depth_upsample_start", "depth_upsample_end"),
        ("rt_gpu_depth_postprocess", "depth_post_start", "depth_post_end"),
        ("rt_gpu_synth", "depth", "synthesis"),
        ("rt_gpu_synth_scene", "synth_start", "synth_scene"),
        ("rt_gpu_synth_depth_postprocess", "synth_scene", "synth_depth_postprocess"),
        ("rt_gpu_synth_shift_response", "synth_depth_postprocess", "synth_shift_response"),
        ("rt_gpu_synth_depth_shift", "synth_scene", "synth_depth_shift"),
        ("rt_gpu_synth_warp", "synth_depth_shift", "synth_warp"),
        ("rt_gpu_synth_occ", "synth_warp", "synth_occlusion"),
        ("rt_gpu_synth_fill", "synth_occlusion", "synth_hole_fill"),
        ("rt_gpu_synth_refine", "synth_hole_fill", "synth_refine"),
        ("rt_gpu_synth_temporal", "synth_refine", "synth_temporal"),
        ("rt_gpu_synth_output_depth", "synth_temporal", "synth_output_depth"),
        ("rt_gpu_synth_sbs", "synth_output_depth", "synth_sbs"),
        ("rt_gpu_pack", "synthesis", "pack"),
        ("rt_gpu_openxr_pack", "openxr_pack_start", "openxr_pack"),
        ("rt_gpu_total", "start", "end"),
    ):
        elapsed_ms = _cuda_elapsed_ms(events, start, end)
        if elapsed_ms is not None:
            ctx.breakdown_add_time(name, elapsed_ms / 1000.0)


class RuntimePipelineLoop:
    def __init__(self, context: RuntimePipelineContext):
        self.context = context
        self._logged_rgb_shape = False
        self._logged_drop_only = False
        self._last_render_size = None
        self._last_source_target_key = None
        self._has_source_target_key = False
        self._last_cuda_ready_event = None
        self._pending_runtime_items = []
        self._last_runtime_motion_sample = None
        self._last_algorithm_output_time = 0.0
        self._prepared = False
        self._consecutive_runtime_errors = 0
        self._dual_pending_cooldown_until = 0.0
        self._parallel_depth_scheduler = None
        self._parallel_backoff_until = 0.0
        self._parallel_recovery_after = 0.0
        self._presenter_backpressure_active = False
        self._preprocess_diagnostic_saved = False
        self._preprocess_diagnostic_first_frame_time = None
        self._backend_status_emitted = False

    def _emit_backend_status_once(self, debug_info: dict | None) -> None:
        if self._backend_status_emitted or not isinstance(debug_info, dict):
            return
        runtime = self.context.stereo_runtime
        provider_report = {}
        try:
            provider_report = dict(runtime.provider_report())
        except Exception as exc:
            provider_report = {"fallback_reason": f"provider_report_failed: {type(exc).__name__}: {exc}"}
        attempts = provider_report.get("attempts") or []
        capture_reason = debug_info.get("capture_fallback_reason")
        directml_reason = debug_info.get("directml_fallback_reason") or debug_info.get(
            "directml_resource_reason"
        )
        stereo_reason = str(
            getattr(runtime, "_stereo_compute_backend_reason", "") or ""
        )
        fallback_reasons = []
        if provider_report.get("fallback_reason"):
            fallback_reasons.append(str(provider_report["fallback_reason"]))
        if capture_reason:
            fallback_reasons.append(str(capture_reason))
        if directml_reason and str(directml_reason) not in fallback_reasons:
            fallback_reasons.append(str(directml_reason))
        if stereo_reason and stereo_reason not in {"selected_by_priority", "explicit_request"}:
            fallback_reasons.append(stereo_reason)
        gpu_copy_count = debug_info.get("capture_gpu_copy_count")
        try:
            gpu_copy_count = int(gpu_copy_count) if gpu_copy_count is not None else 0
        except (TypeError, ValueError):
            gpu_copy_count = 0
        gpu_to_cpu = bool(debug_info.get("capture_gpu_to_cpu", False))
        directml_gpu_to_cpu = bool(debug_info.get("directml_gpu_to_cpu", False))
        zero_copy = bool(
            debug_info.get("capture_zero_copy", False)
            and debug_info.get("capture_zero_copy_ready", False)
            and not gpu_to_cpu
        )
        payload = {
            "os": platform.system(),
            "device": str(getattr(getattr(runtime, "config", None), "device", "")),
            "capture_mode": debug_info.get("capture_mode"),
            "capture_tool": debug_info.get("capture_tool"),
            "depth_backend": provider_report.get(
                "depth_backend",
                debug_info.get("depth_backend_resolved", debug_info.get("runtime_depth_backend", "unknown")),
            ),
            "stereo_backend": debug_info.get(
                "stereo_compute_backend",
                getattr(runtime, "_resolved_stereo_compute_backend", "unknown"),
            ),
            "stereo_backend_reason": stereo_reason or "not_reported",
            "fallback": bool(attempts or fallback_reasons),
            "fallback_reasons": fallback_reasons,
            "adapter_luid": debug_info.get("capture_adapter_luid"),
            "adapter_uuid": debug_info.get("capture_adapter_uuid"),
            "pci_bdf": debug_info.get("capture_pci_bdf"),
            "adapter_identity": debug_info.get("capture_adapter_identity"),
            "gpu_vendor": debug_info.get("gpu_vendor")
            or provider_report.get("gpu_vendor")
            or getattr(getattr(runtime, "config", None), "gpu_vendor", "unknown"),
            "resource_kind": debug_info.get("capture_resource_kind"),
            "resource_format": debug_info.get("capture_resource_format"),
            "capture_size": debug_info.get("capture_size"),
            "gpu_to_cpu": gpu_to_cpu or directml_gpu_to_cpu,
            "gpu_copy_count": gpu_copy_count,
            "directml_gpu_copy_count": debug_info.get("directml_gpu_copy_count", 0),
            "directml_resource_mode": debug_info.get("directml_resource_mode"),
            "directml_zero_copy_ready": debug_info.get("directml_zero_copy_ready"),
            "zero_copy": zero_copy and bool(
                debug_info.get("directml_zero_copy_ready", True)
            ),
            "zero_copy_ready": bool(debug_info.get("capture_zero_copy_ready", False)),
            "directml_zero_copy_ready": bool(
                debug_info.get("directml_zero_copy_ready", False)
            ),
            "directml_resource_mode": debug_info.get("directml_resource_mode"),
        }
        try:
            print(
                "[D2S_BACKEND_STATUS] "
                + json.dumps(payload, ensure_ascii=False, sort_keys=True),
                flush=True,
            )
            self._backend_status_emitted = True
        except (TypeError, ValueError) as exc:
            print(
                "[D2S_BACKEND_STATUS] "
                + json.dumps(
                    {"fallback": True, "fallback_reasons": [f"serialization: {exc}"]},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            self._backend_status_emitted = True

    def _presenter_pressure_active(self) -> bool:
        ctx = self.context
        active = False
        if _runtime_presenter_backpressure_enabled(ctx):
            try:
                active = bool(ctx.openxr_presenter_pressure())
            except Exception:
                active = False
        self._presenter_backpressure_active = active
        return active

    def _pending_depth_limit(self) -> int:
        limit = _runtime_pending_depth_limit(self.context)
        if limit > 1 and time.perf_counter() < self._dual_pending_cooldown_until:
            return 1
        if limit > 1 and self._presenter_pressure_active():
            return 1
        scheduler = getattr(self, "_parallel_depth_scheduler", None)
        if limit > 1 and scheduler is not None:
            return min(limit, scheduler.effective_limit)
        return limit

    @staticmethod
    def _parallel_slot_wait_backoff_ms() -> float:
        try:
            return max(0.0, float(os.environ.get("D2S_RUNTIME_PARALLEL_SLOT_WAIT_BACKOFF_MS", "12")))
        except (TypeError, ValueError):
            return 12.0

    @staticmethod
    def _parallel_backoff_duration_s() -> float:
        try:
            return max(0.1, float(os.environ.get("D2S_RUNTIME_PARALLEL_BACKOFF_S", "1.5")))
        except (TypeError, ValueError):
            return 1.5

    def _parallel_reduce_for_pressure(self, reason: str) -> None:
        scheduler = getattr(self, "_parallel_depth_scheduler", None)
        if scheduler is None or scheduler.effective_limit <= 1 or not _runtime_parallel_adaptive_backoff_enabled():
            return
        now = time.perf_counter()
        changed = scheduler.set_effective_limit(scheduler.effective_limit - 1)
        self._parallel_backoff_until = now + self._parallel_backoff_duration_s()
        self._parallel_recovery_after = self._parallel_backoff_until
        if changed:
            self.context.source_stat_inc(
                "runtime_parallel_backoff",
                reason=reason,
                active_workers=scheduler.effective_limit,
            )
            self.context.breakdown_inc("runtime_parallel_backoff")
            print(
                f"[RuntimePipeline] Parallel depth backoff: reason={reason} "
                f"effective_workers={scheduler.effective_limit}/{scheduler.worker_count}",
                flush=True,
            )

    def _parallel_recover_if_ready(self) -> None:
        scheduler = getattr(self, "_parallel_depth_scheduler", None)
        if scheduler is None or scheduler.effective_limit >= scheduler.worker_count or not _runtime_parallel_adaptive_backoff_enabled():
            return
        now = time.perf_counter()
        if now < self._parallel_recovery_after:
            return
        if scheduler.set_effective_limit(scheduler.effective_limit + 1):
            self._parallel_backoff_until = 0.0
            self._parallel_recovery_after = now + self._parallel_backoff_duration_s()
            self.context.source_stat_inc(
                "runtime_parallel_recovery",
                active_workers=scheduler.effective_limit,
            )
            self.context.breakdown_inc("runtime_parallel_recovery")
            print(
                f"[RuntimePipeline] Parallel depth recovery: effective_workers="
                f"{scheduler.effective_limit}/{scheduler.worker_count}",
                flush=True,
            )

    def _rebuild_after_consecutive_failures(self) -> None:
        threshold_text = os.environ.get("D2S_RUNTIME_REBUILD_AFTER_ERRORS", "3")
        try:
            threshold = max(1, int(threshold_text))
        except (TypeError, ValueError):
            threshold = 3
        if self._consecutive_runtime_errors < threshold:
            return
        rebuild = getattr(self.context.stereo_runtime, "_rebuild_depth_provider", None)
        if not callable(rebuild):
            self.context.source_stat_inc("runtime_rebuild_unavailable")
            self._consecutive_runtime_errors = 0
            return
        try:
            rebuild()
            reset_temporal = getattr(self.context.stereo_runtime, "reset_temporal", None)
            if callable(reset_temporal):
                reset_temporal()
            self.context.source_stat_inc("runtime_adapter_rebuilds")
            print(
                "[RuntimePipeline] depth provider rebuilt after consecutive failures",
                flush=True,
            )
        except Exception as rebuild_error:
            self.context.source_stat_inc(
                "runtime_rebuild_errors",
                last_error=f"{type(rebuild_error).__name__}: {rebuild_error}",
            )
            print(
                f"[RuntimePipeline] Provider rebuild failed: {type(rebuild_error).__name__}: {rebuild_error}",
                flush=True,
            )
        finally:
            self._consecutive_runtime_errors = 0

    def _ensure_parallel_depth_scheduler(self) -> None:
        if getattr(self, "_parallel_depth_scheduler", None) is not None:
            return
        ctx = self.context
        runtime_config = getattr(ctx, "runtime_config", None)
        provider = getattr(ctx.stereo_runtime, "depth_provider", None)
        requested_workers = max(1, min(
            3,
            int(getattr(runtime_config, "parallel_inference_workers", 2) or 2),
        ))
        if (
            ctx.run_mode not in {"OpenXR", "Viewer"}
            or not bool(getattr(runtime_config, "parallel_inference", False))
            or min(int(getattr(provider, "pipeline_slot_count", 1)), requested_workers) < 2
            or bool(getattr(getattr(ctx.stereo_runtime, "config", None), "profile_sync", False))
        ):
            return
        try:
            workers = min(int(getattr(provider, "pipeline_slot_count", 1)), requested_workers)
            self._parallel_depth_scheduler = _ParallelDepthScheduler(ctx.stereo_runtime, workers)
            ctx.source_stat_inc("runtime_parallel_workers", active_workers=workers)
            print(
                f"[RuntimePipeline] Parallel depth scheduler enabled: workers={workers} slots="
                f"{int(getattr(provider, 'pipeline_slot_count', 1))}",
                flush=True,
            )
        except Exception as exc:
            self._parallel_depth_scheduler = None
            ctx.source_stat_inc("runtime_parallel_backend_fallback", last_error=str(exc))
            print(f"[RuntimePipeline] Parallel depth scheduler unavailable: {type(exc).__name__}: {exc}", flush=True)

    def prepare(self) -> None:
        if self._prepared:
            return
        load = getattr(self.context.stereo_runtime, "load", None)
        if callable(load):
            load()
        self._ensure_parallel_depth_scheduler()
        self._prepared = True

    def _publish_runtime_item(self, item) -> None:
        ctx = self.context
        runtime_result, capture_start_time, process_latency, runtime_latency, pending_since = item
        queue_put_start_time = time.perf_counter()
        _add_cuda_event_timings(ctx, runtime_result)
        try:
            runtime_q_was_full = ctx.runtime_q.full()
        except Exception:
            runtime_q_was_full = False
        ctx.queue_put_latest(ctx.runtime_q, (runtime_result, capture_start_time))
        ready_event = getattr(ctx, "runtime_ready_event", None)
        if ready_event is not None:
            ready_event.set()
        if runtime_q_was_full:
            ctx.breakdown_inc("runtime_overwrite")
            ctx.source_stat_inc("runtime_overwrite")
            if _runtime_pending_depth_limit(ctx) > 1:
                self._dual_pending_cooldown_until = time.perf_counter() + 1.0
                self._parallel_reduce_for_pressure("output_queue_full")
                ctx.breakdown_inc("runtime_pending_cuda_backoff")
                ctx.source_stat_inc("runtime_pending_cuda_backoff")
        ctx.breakdown_add_time("rt_put", time.perf_counter() - queue_put_start_time)
        if pending_since is not None:
            ctx.breakdown_add_time("rt_pending_age", time.perf_counter() - pending_since)
        self._last_algorithm_output_time = time.perf_counter()
        ctx.source_stat_inc(
            "runtime_frames",
            last_runtime_ts=time.perf_counter(),
            last_process_latency=process_latency,
            last_runtime_latency=runtime_latency,
        )
        ctx.breakdown_inc("runtime")

    def _publish_ready_pending_items(self) -> int:
        if not self._pending_runtime_items:
            return 0
        ready_index = None
        for index in range(len(self._pending_runtime_items) - 1, -1, -1):
            pending_result = self._pending_runtime_items[index][0]
            if _cuda_event_ready(getattr(pending_result, "cuda_ready_event", None)):
                ready_index = index
                break
        if ready_index is None:
            pending_limit = self._pending_depth_limit()
            if len(self._pending_runtime_items) > pending_limit:
                self._pending_runtime_items[:] = self._pending_runtime_items[-pending_limit:]
            return 0
        item = self._pending_runtime_items[ready_index]
        self._pending_runtime_items[:] = self._pending_runtime_items[ready_index + 1:]
        self._publish_runtime_item(item)
        return 1

    def _publish_ready_pending_items_until(self, timeout_s: float) -> int:
        published = self._publish_ready_pending_items()
        if published:
            ctx = self.context
            ctx.source_stat_inc("runtime_pending_cuda_wait")
            ctx.breakdown_inc("runtime_pending_cuda_wait")
            ctx.breakdown_add_time("rt_pending_wait", 0.0)
            return published
        if timeout_s <= 0.0 or not self._pending_runtime_items:
            return published
        ctx = self.context
        wait_start = time.perf_counter()
        deadline = wait_start + timeout_s
        while self._pending_runtime_items and not ctx.shutdown_event.is_set():
            remaining = deadline - time.perf_counter()
            if remaining <= 0.0:
                break
            time.sleep(min(max(remaining, 0.0), max(float(ctx.time_sleep), 0.0005), 0.0005))
            published = self._publish_ready_pending_items()
            if published:
                ctx.source_stat_inc("runtime_pending_cuda_wait")
                ctx.breakdown_inc("runtime_pending_cuda_wait")
                ctx.breakdown_add_time("rt_pending_wait", time.perf_counter() - wait_start)
                return published
        ctx.breakdown_add_time("rt_pending_wait", time.perf_counter() - wait_start)
        return 0

    def _retain_latest_raw_while_cuda_pending(self) -> None:
        """Leave raw_q populated so CUDA completion can consume it immediately."""
        ctx = self.context
        # raw_q is a latest-frame overwrite queue. Removing its current item
        # here creates an avoidable empty interval: when the CUDA event becomes
        # ready, the runtime has to wait for the following capture tick. Keep
        # the slot occupied instead; the capture producer will atomically
        # replace it with newer content while preserving low-latency semantics.
        ctx.source_stat_inc("runtime_pending_cuda_inflight")
        ctx.breakdown_inc("runtime_pending_cuda_inflight")
        time.sleep(min(max(float(ctx.time_sleep), 0.0005), 0.001))

    def run(self) -> None:
        ctx = self.context
        while not ctx.shutdown_event.is_set():
            ctx.log_source_health()
            try:
                diag_stage = _runtime_diag_stage()
                if ctx.shutdown_event.is_set():
                    break
                if ctx.is_hard_idle():
                    self._pending_runtime_items.clear()
                    self._last_cuda_ready_event = None
                    ctx.queue_clear(ctx.raw_q)
                    ctx.queue_clear(ctx.runtime_q)
                    time.sleep(0.1)
                    continue
                if ctx.is_source_paused():
                    self._pending_runtime_items.clear()
                    self._last_cuda_ready_event = None
                    ctx.queue_clear(ctx.raw_q)
                    ctx.queue_clear(ctx.runtime_q)
                    ctx.source_stat_inc("runtime_dropped_paused")
                    time.sleep(0.01)
                    continue
                self._publish_ready_pending_items()
                if len(self._pending_runtime_items) >= self._pending_depth_limit():
                    if self._publish_ready_pending_items_until(_runtime_pending_cuda_wait_s(ctx)):
                        continue
                    self._retain_latest_raw_while_cuda_pending()
                    continue

                _apply_latest_settings_snapshot(ctx)

                frame_raw, size, capture_start_time, captured_frame = _unpack_raw_queue_item(
                    ctx.queue_drain_latest(
                        ctx.raw_q,
                        ctx.raw_q.get(timeout=min(ctx.time_sleep, 0.01)),
                    )
                )
                ctx.source_stat_inc("raw_get", last_raw_get_ts=time.perf_counter())
                ctx.breakdown_inc("raw_get")

                if diag_stage == "raw":
                    if not self._logged_drop_only:
                        self._logged_drop_only = True
                        print("[RuntimePipeline] diag_stage=raw: dropping raw frames before GPU work", flush=True)
                    ctx.source_stat_inc("runtime_diag_raw")
                    ctx.breakdown_inc("runtime_diag_raw")
                    continue

                if len(self._pending_runtime_items) >= self._pending_depth_limit() and not _cuda_event_ready(self._last_cuda_ready_event):
                    ctx.source_stat_inc("runtime_drop_cuda_inflight")
                    ctx.breakdown_inc("runtime_drop_cuda_inflight")
                    continue
                self._last_cuda_ready_event = None

                if ctx.is_source_paused():
                    ctx.queue_clear(ctx.raw_q)
                    ctx.queue_clear(ctx.runtime_q)
                    ctx.source_stat_inc("runtime_dropped_paused")
                    time.sleep(0.01)
                    continue

                loop_start_time = time.perf_counter()

                process_start_time = time.perf_counter()
                render_size = _resolve_pipeline_render_size(size, ctx.render_size_config)
                render_size_changed = self._last_render_size is not None and render_size != self._last_render_size
                source_target_key = _source_target_key(captured_frame)
                source_target_changed = (
                    self._has_source_target_key
                    and source_target_key != self._last_source_target_key
                )
                if render_size_changed:
                    _reset_runtime_temporal_state(ctx.stereo_runtime)
                if source_target_changed:
                    _reset_runtime_temporal_state(ctx.stereo_runtime)
                self._last_render_size = render_size
                self._last_source_target_key = source_target_key
                self._has_source_target_key = True
                frame_input = _prepare_frame_input(ctx, captured_frame, frame_raw)
                frame_rgb = ctx.capture_frame_to_rgb(
                    frame_input,
                    render_size,
                    device=ctx.device,
                    use_torch=ctx.use_cudart,
                    output="tensor",
                    frame_raw_device=captured_frame.frame_raw_device if captured_frame else None,
                    capture_copy_mode=_capture_copy_mode(captured_frame),
                    capture_zero_copy=_capture_zero_copy(captured_frame),
                )
                if not self._logged_rgb_shape and os.environ.get('D2S_DEBUG', '0') in ('1', 'true', 'yes', 'on'):
                    self._logged_rgb_shape = True
                    print(
                        f"[process_runtime_loop] rgb={_rgb_size_text(frame_rgb)}",
                        flush=True,
                    )
                ctx.breakdown_add_time("rt_cap2rgb", time.perf_counter() - process_start_time)
                ctx.set_preprocess_backend(
                    str(getattr(frame_rgb, "_d2s_preprocess_backend", "unknown"))
                )
                if _env_flag("D2S_PREPROCESS_IMAGE_DIAGNOSTIC") and not self._preprocess_diagnostic_saved:
                    now = time.perf_counter()
                    if self._preprocess_diagnostic_first_frame_time is None:
                        self._preprocess_diagnostic_first_frame_time = now
                    try:
                        delay_s = max(
                            0.0,
                            float(os.environ.get("D2S_PREPROCESS_IMAGE_DIAGNOSTIC_DELAY_S", "3.0")),
                        )
                    except (TypeError, ValueError):
                        delay_s = 3.0
                    if now - self._preprocess_diagnostic_first_frame_time >= delay_s:
                        try:
                            output_dir = os.environ.get(
                                "D2S_PREPROCESS_IMAGE_DIAGNOSTIC_DIR",
                                str(Path(__file__).resolve().parents[2] / "artifacts"),
                            )
                            image_path, manifest_path = _save_preprocess_image_diagnostic(
                                frame_rgb,
                                capture_size=size,
                                render_size=render_size,
                                output_dir=output_dir,
                            )
                            comparison_dir = _save_downsample_filter_comparison(
                                frame_raw,
                                capture_size=size,
                                render_size=render_size,
                                output_dir=output_dir,
                            )
                            self._preprocess_diagnostic_saved = True
                            print(
                                "[RuntimePipeline] preprocess image diagnostic saved: "
                                f"image={image_path} manifest={manifest_path} "
                                f"comparison={comparison_dir or 'not_required'}",
                                flush=True,
                            )
                        except Exception as exc:
                            self._preprocess_diagnostic_saved = True
                            print(
                                "[RuntimePipeline] preprocess image diagnostic failed: "
                                f"{type(exc).__name__}: {exc}",
                                flush=True,
                            )
                if diag_stage == "preprocess":
                    ctx.source_stat_inc("runtime_diag_preprocess")
                    ctx.breakdown_inc("runtime_diag_preprocess")
                    continue
                process_latency = process_start_time - capture_start_time
                ctx.thread_latencies["capture"] = process_latency

                if ctx.is_source_paused():
                    ctx.queue_clear(ctx.raw_q)
                    ctx.queue_clear(ctx.runtime_q)
                    ctx.source_stat_inc("runtime_dropped_paused")
                    time.sleep(0.01)
                    continue

                runtime_start_time = time.perf_counter()
                prepare_start_time = time.perf_counter()
                runtime_rgb = ctx.prepare_rgb_for_stereo_runtime(frame_rgb, device=ctx.device)
                ctx.breakdown_add_time("rt_prepare", time.perf_counter() - prepare_start_time)
                # Native TensorRT may create its engine lazily after the first
                # input shape is known. Re-check here so slot_count=2 can
                # activate the worker pool after that lazy load.
                self._ensure_parallel_depth_scheduler()
                if ctx.run_mode == "OpenXR" and _runtime_motion_gate_enabled(ctx):
                    motion_sample = _motion_sample(runtime_rgb)
                    motion = _motion_score(self._last_runtime_motion_sample, motion_sample)
                    now_for_gate = time.perf_counter()
                    max_fps = _runtime_max_algorithm_fps(ctx)
                    min_interval = 1.0 / max_fps if max_fps > 0.0 else 0.0
                    small_threshold = _runtime_motion_threshold("D2S_RUNTIME_MOTION_SMALL_THRESHOLD", 0.003)
                    force_threshold = _runtime_motion_threshold("D2S_RUNTIME_MOTION_FORCE_THRESHOLD", 0.02)
                    too_soon = min_interval > 0.0 and (now_for_gate - self._last_algorithm_output_time) < min_interval
                    weak_motion = motion is not None and motion < small_threshold
                    strong_motion = motion is not None and motion >= force_threshold
                    if self._last_runtime_motion_sample is not None and (weak_motion or (too_soon and not strong_motion)):
                        ctx.source_stat_inc("runtime_motion_skip")
                        ctx.breakdown_inc("runtime_motion_skip")
                        continue
                    self._last_runtime_motion_sample = motion_sample
                if diag_stage == "prepare":
                    ctx.source_stat_inc("runtime_diag_prepare")
                    ctx.breakdown_inc("runtime_diag_prepare")
                    continue
                if diag_stage == "depth":
                    depth_start_time = time.perf_counter()
                    _run_depth_only(ctx.stereo_runtime, runtime_rgb)
                    ctx.breakdown_add_time("rt_call", time.perf_counter() - depth_start_time)
                    ctx.source_stat_inc("runtime_diag_depth")
                    ctx.breakdown_inc("runtime_diag_depth")
                    continue

                # A native Intel capture source may already have produced a
                # DepthProfileResult from the borrowed D3D11 texture. Keep that
                # result attached to the capture frame and bypass the tensor
                # worker pool; the RGB compatibility frame is still used by
                # the existing stereo synthesis path.
                native_depth_profile = None
                if captured_frame is not None:
                    native_depth_profile = captured_frame.metadata.get(
                        "native_depth_profile"
                    )
                # Submit only the depth stage to the bounded worker pool when
                # native capture did not provide a result. The caller then
                # consumes the oldest completed job in order.
                depth_profile = native_depth_profile
                if depth_profile is None and self._parallel_depth_scheduler is not None:
                    scheduler = self._parallel_depth_scheduler
                    self._parallel_recover_if_ready()
                    admission_limit = self._pending_depth_limit()
                    if scheduler.can_submit() and len(scheduler.pending) < admission_limit:
                        scheduler.submit(
                            runtime_rgb,
                            frame_rgb=frame_rgb,
                            frame_raw=frame_raw,
                            size=size,
                            capture_start_time=capture_start_time,
                            captured_frame=captured_frame,
                            render_size=render_size,
                            process_latency=process_latency,
                            source_target_changed=source_target_changed,
                            render_size_changed=render_size_changed,
                        )
                    elif (
                        scheduler.can_submit()
                        and admission_limit < scheduler.effective_limit
                    ):
                        ctx.source_stat_inc("runtime_presenter_backpressure")
                        ctx.breakdown_inc("runtime_presenter_backpressure")
                    scheduler.trim(self._pending_depth_limit())
                    job = scheduler.pop_ready(block=False)
                    if job is None:
                        ctx.source_stat_inc("runtime_parallel_pending", pending_depth=len(scheduler.pending))
                        ctx.breakdown_inc("runtime_parallel_pending")
                        continue
                    frame_rgb = job.frame_rgb
                    frame_raw = job.frame_raw
                    size = job.size
                    capture_start_time = job.capture_start_time
                    captured_frame = job.captured_frame
                    render_size = job.render_size
                    process_latency = job.process_latency
                    source_target_changed = job.source_target_changed
                    render_size_changed = job.render_size_changed
                    runtime_rgb = job.runtime_rgb
                    depth_profile = job.depth_profile
                    if float(getattr(depth_profile, "slot_wait_ms", 0.0)) >= self._parallel_slot_wait_backoff_ms():
                        self._parallel_reduce_for_pressure("tensorrt_slot_wait")
                    ctx.source_stat_inc("runtime_parallel_processed")
                    ctx.source_stat_inc("runtime_parallel_reorder_wait", frame_reorder_wait=0)
                    parallel_frame_id = job.frame_id
                else:
                    parallel_frame_id = None
                ctx.log_stereo_runtime_mode_once()
                ctx.apply_stereo_hot_reload_if_needed()
                _apply_latest_settings_snapshot(ctx)
                openxr_full_synthesis = ctx.run_mode == "OpenXR" and _openxr_full_synthesis_enabled(ctx)
                _enable_openxr_depth_cuda_graph_if_needed(ctx, openxr_full_synthesis, captured_frame)
                runtime_call_start_time = time.perf_counter()
                if ctx.run_mode == "OpenXR":
                    original_stereo_config = None
                    openxr_config = ctx.current_openxr_render_config()
                    if openxr_full_synthesis:
                        original_stereo_config = getattr(ctx.stereo_runtime, "stereo_config", None)
                        realtime_config = _openxr_realtime_synthesis_config(original_stereo_config)
                        if realtime_config is not original_stereo_config:
                            ctx.stereo_runtime.stereo_config = realtime_config
                        if openxr_config is not None and is_dataclass(openxr_config):
                            openxr_config = replace(
                                openxr_config,
                                output_mode="full_synthesis_eyes",
                            )
                    try:
                        # Keep the Vulkan request attached to the runtime
                        # result so the Presenter can dispatch directly into
                        # its own external images and semaphores.
                        runtime_result = ctx.stereo_runtime.process_openxr_frame(
                            runtime_rgb,
                            openxr_config,
                            depth_profile=depth_profile,
                        )
                    finally:
                        if original_stereo_config is not None:
                            ctx.stereo_runtime.stereo_config = original_stereo_config
                else:
                    runtime_result = ctx.stereo_runtime.process_rgb_frame(
                        runtime_rgb,
                        skip_sbs_output=False,
                        depth_profile=depth_profile,
                        pixel_buffer=getattr(captured_frame, "sck_zero_copy", None),
                    )
                ctx.breakdown_add_time("rt_call", time.perf_counter() - runtime_call_start_time)
                _attach_pipeline_debug(
                    runtime_result,
                    capture_size=size,
                    render_size=render_size,
                    run_mode=ctx.run_mode,
                    render_size_config=ctx.render_size_config,
                    application_runtime_target=ctx.application_runtime_target,
                    output_transport=ctx.output_transport,
                )
                debug_info = getattr(runtime_result, "debug_info", None)
                if isinstance(debug_info, dict):
                    debug_info["parallel_inference_enabled"] = int(
                        _runtime_pending_depth_limit(ctx) > 1
                    )
                    debug_info["parallel_inference_pending_limit"] = int(
                        self._pending_depth_limit()
                    )
                    debug_info["parallel_inference_workers"] = int(
                        getattr(self._parallel_depth_scheduler, "worker_count", 0)
                    )
                    if self._parallel_depth_scheduler is not None:
                        debug_info["parallel_inference_effective_workers"] = int(
                            self._parallel_depth_scheduler.effective_limit
                        )
                        debug_info["parallel_inference_backoff"] = int(
                            self._parallel_depth_scheduler.effective_limit
                            < self._parallel_depth_scheduler.worker_count
                        )
                        debug_info["parallel_inference_adaptive_backoff"] = int(
                            _runtime_parallel_adaptive_backoff_enabled()
                        )
                        debug_info["parallel_inference_presenter_backpressure"] = int(
                            self._presenter_backpressure_active
                        )
                        debug_info["parallel_inference_frame_id"] = parallel_frame_id
                        debug_info["parallel_inference_pending"] = int(
                            len(self._parallel_depth_scheduler.pending)
                        )
                        debug_info["parallel_inference_dropped"] = int(
                            self._parallel_depth_scheduler.dropped
                        )
                    if render_size_changed:
                        _append_temporal_reset_reason(debug_info, "render_size_changed")
                    if source_target_changed:
                        _append_temporal_reset_reason(debug_info, "source_target_changed")
                    if ctx.application_runtime_target:
                        debug_info["application_runtime_target"] = ctx.application_runtime_target
                    if ctx.output_transport:
                        debug_info["output_transport"] = ctx.output_transport
                _attach_capture_debug(runtime_result, captured_frame, frame_rgb)
                if isinstance(debug_info, dict) and captured_frame is not None:
                    native_profile = captured_frame.metadata.get("native_depth_profile")
                    if native_profile is not None:
                        native_debug = getattr(native_profile, "cuda_timing_events", None)
                        if not isinstance(native_debug, dict):
                            native_debug = getattr(native_profile, "debug", {})
                        debug_info["native_depth_inference"] = 1
                        debug_info["native_depth_backend"] = captured_frame.metadata.get(
                            "native_depth_backend", "openvino_d3d11_remote"
                        )
                        if isinstance(native_debug, dict):
                            debug_info["native_depth_capture_gpu"] = int(
                                bool(native_debug.get("capture_gpu", True))
                            )
                            debug_info["native_depth_gpu_to_cpu"] = int(
                                bool(native_debug.get("gpu_to_cpu", True))
                            )
                            debug_info["native_depth_input_zero_copy"] = int(
                                bool(native_debug.get("input_zero_copy", False))
                            )
                            debug_info["native_depth_input_gpu_to_cpu"] = int(
                                bool(native_debug.get("input_gpu_to_cpu", True))
                            )
                            debug_info["native_depth_output_device"] = str(
                                native_debug.get("output_device", "cpu")
                            )
                            debug_info["native_depth_output_gpu_to_cpu"] = int(
                                bool(native_debug.get("output_gpu_to_cpu", True))
                            )
                            debug_info["native_depth_zero_copy"] = int(
                                bool(native_debug.get("zero_copy", False))
                            )
                self._emit_backend_status_once(debug_info)
                ctx.breakdown_add_runtime_timing(runtime_result)
                ctx.log_fast_plus_fused_runtime_state(runtime_result)
                if runtime_result.depth is None:
                    ctx.queue_clear(ctx.runtime_q)
                    ctx.source_stat_inc("runtime_none")
                    continue
                ctx.warmup_stereo_once_for_frame(runtime_rgb)
                if diag_stage == "runtime_sync" or _runtime_sync_after_frame_enabled(ctx):
                    sync_start_time = time.perf_counter()
                    _synchronize_runtime_device(runtime_rgb)
                    ctx.breakdown_add_time("rt_sync", time.perf_counter() - sync_start_time)
                    ctx.source_stat_inc("runtime_gpu_sync")
                if diag_stage in {"runtime", "runtime_sync"}:
                    ctx.source_stat_inc("runtime_diag_runtime")
                    ctx.breakdown_inc("runtime_diag_runtime")
                    continue
                runtime_result = _attach_cuda_ready_event(runtime_result)
                self._last_cuda_ready_event = getattr(runtime_result, "cuda_ready_event", None)
                runtime_latency = time.perf_counter() - runtime_start_time
                ctx.thread_latencies["resize"] = process_latency
                ctx.thread_latencies["runtime"] = runtime_latency

                if not _cuda_event_ready(self._last_cuda_ready_event):
                    self._pending_runtime_items.append(
                        (runtime_result, capture_start_time, process_latency, runtime_latency, time.perf_counter())
                    )
                    pending_limit = self._pending_depth_limit()
                    if len(self._pending_runtime_items) > pending_limit:
                        self._pending_runtime_items[:] = self._pending_runtime_items[-pending_limit:]
                    ctx.breakdown_inc("runtime_pending_cuda")
                    ctx.source_stat_inc("runtime_pending_cuda")
                    ctx.breakdown_add_time("rt_loop", time.perf_counter() - loop_start_time)
                    continue

                self._publish_runtime_item(
                    (runtime_result, capture_start_time, process_latency, runtime_latency, None)
                )
                self._consecutive_runtime_errors = 0
                ctx.breakdown_add_time("rt_loop", time.perf_counter() - loop_start_time)

            except queue.Empty:
                ctx.source_stat_inc("raw_queue_empty")
                continue
            except (RuntimeSettingsPipelineRebuildRequired, RuntimeSettingsRestartRequired):
                raise
            except Exception as exc:
                self._consecutive_runtime_errors += 1
                fatal_error = _is_fatal_runtime_preparation_error(exc)
                ctx.source_stat_inc(
                    "runtime_errors",
                    last_error=f"process_runtime_loop {type(exc).__name__}: {exc}",
                )
                ctx.source_stat_inc("runtime_inference_failures")
                ctx.breakdown_inc("runtime_inference_failures")
                self._rebuild_after_consecutive_failures()
                if fatal_error:
                    ctx.source_stat_inc("runtime_fatal_errors")
                    print(f"[process_runtime_loop] Fatal: {type(exc).__name__}: {exc}", flush=True)
                    try:
                        ctx.shutdown_event.set()
                    except Exception:
                        pass
                    break
                print(f"[process_runtime_loop] Error: {type(exc).__name__}: {exc}", flush=True)
                time.sleep(0.05)
                continue
        if self._parallel_depth_scheduler is not None:
            self._parallel_depth_scheduler.close()
            self._parallel_depth_scheduler = None
