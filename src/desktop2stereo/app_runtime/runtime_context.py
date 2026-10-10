from __future__ import annotations

import os
import queue
import time
from dataclasses import dataclass

from capture import CaptureConfig
from capture.session import CaptureSessionCallbacks
from capture.source_health import SourceHealth
from stereo_runtime import StereoRuntime, runtime_config_from_d2s_settings
from stereo_runtime.adapter import preset_for_runtime_mode, stereo_config_from_runtime
from stereo_runtime.hot_reload import StereoHotReloader
from stereo_runtime.openxr_state import OpenXRStateController
from stereo_runtime.pipeline import RuntimePipelineContext
from stereo_runtime.presets import normalize_preset
from stereo_runtime.render_size import RenderSizeConfig
from stereo_runtime.session_helpers import StereoRuntimeLogger, StereoWarmupTracker
from utils.breakdown import FPSBreakdown, fps_breakdown_default
from utils.run_mode import resolve_run_mode


@dataclass
class AppRuntimeContext:
    base_dir: str
    use_cudart: bool
    time_sleep: float
    openxr_runtime_direct: bool
    raw_q: queue.Queue
    runtime_q: queue.Queue
    # Ordered queue used when NvFRUC emits real and generated frames.
    presentation_q: queue.Queue
    settings_update_q: queue.Queue
    runtime_config: object
    stereo_runtime: StereoRuntime
    stereo_auto_enabled: bool
    stereo_active_preset: str
    stereo_still_duration_s: float
    stereo_last_auto_ts: float
    stereo_hot_reloader: StereoHotReloader
    stereo_warmup_tracker: StereoWarmupTracker
    stereo_runtime_logger: StereoRuntimeLogger
    openxr_state: OpenXRStateController
    source_health: SourceHealth
    fps_breakdown_log: bool
    fps_breakdown: FPSBreakdown
    thread_latencies: dict
    capture_config: CaptureConfig
    render_size_config: RenderSizeConfig
    application_runtime_target: str
    output_transport: str
    # True when the GUI requested the process-local NVIDIA NvFRUC stage.
    nvfruc_frame_generation: bool


def initial_stereo_preset_state(config):
    raw_preset = config.stereo_preset
    if raw_preset is not None:
        preset = normalize_preset(raw_preset)
        return False, "cinema" if preset == "auto" else preset
    if config.mode == "auto":
        return False, "cinema"
    return False, preset_for_runtime_mode(config.mode)


def env_flag(name, default="0"):
    return str(os.environ.get(name, default) or default).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _runtime_contract_from_settings(settings: dict, *, os_name: str):
    run_mode_config = resolve_run_mode(
        settings.get("Run Mode", "Local Viewer"),
        os_name=os_name,
        fix_viewer_aspect=bool(settings.get("Fix Viewer Aspect", True)),
        lossless_scaling_support=bool(
            settings.get(
                "NVIDIA Frame Generation",
                settings.get("Lossless Scaling Support", False),
            )
        ),
    )
    if run_mode_config.run_mode == "OpenXR":
        return "openxr", "openxr_swapchain"
    if run_mode_config.stream_mode:
        return "network_stream", "encoded_stream"
    if run_mode_config.use_3d_monitor:
        return "local_display", "local_fullscreen"
    return "local_display", "local_window"


def create_runtime_context(
    *,
    file_path,
    settings,
    cache_path,
    device,
    device_info,
    output_resolution,
    render_size_config,
    fps,
    window_title,
    capture_mode,
    monitor_index,
    capture_tool,
    os_name,
    run_mode,
    depth_strength,
    convergence,
    capture_fps_provider=None,
):
    base_dir = os.path.dirname(os.path.abspath(file_path))
    nvfruc_frame_generation = bool(
        settings.get(
            "NVIDIA Frame Generation",
            settings.get("Lossless Scaling Support", False),
        )
    )
    use_cudart = "CUDA" in device_info and "ZLUDA" not in device_info
    time_sleep = 1.0 / fps
    openxr_runtime_direct = str(
        os.environ.get("D2S_OPENXR_RUNTIME_DIRECT", "1") or "1"
    ).strip().lower() not in ("0", "false", "no", "off")

    raw_q = queue.Queue(maxsize=1)
    runtime_q = queue.Queue(maxsize=1)
    presentation_q = queue.Queue(maxsize=6)
    # Consumers must preserve generated frame ordering instead of coalescing.
    presentation_q._d2s_ordered = True
    settings_update_q = queue.Queue(maxsize=1)
    runtime_config = runtime_config_from_d2s_settings(
        settings,
        cache_dir=cache_path,
        device=str(device),
        depth_only=False,
    )
    stereo_runtime = StereoRuntime(runtime_config)
    stereo_auto_enabled, stereo_active_preset = initial_stereo_preset_state(runtime_config)
    stereo_runtime.configure_stereo(stereo_config_from_runtime(runtime_config), reset_temporal=True)

    openxr_state = OpenXRStateController(
        run_mode=run_mode,
        depth_strength=depth_strength,
        convergence=convergence,
    )
    source_health_log = env_flag(
        "D2S_SOURCE_HEALTH_LOG",
        os.environ.get("D2S_OPENXR_DEBUG", "0"),
    )
    application_runtime_target, output_transport = _runtime_contract_from_settings(settings, os_name=os_name)
    source_health = SourceHealth(
        enabled=source_health_log,
        run_mode=run_mode,
        raw_q=raw_q,
        runtime_q=runtime_q,
        source_active=openxr_state.source_active.is_set,
        render_active=openxr_state.render_active.is_set,
        idle_active=openxr_state.wait_idle_active.is_set,
    )
    fps_breakdown_log = env_flag(
        "D2S_FPS_BREAKDOWN",
        fps_breakdown_default(settings, run_mode),
    )

    return AppRuntimeContext(
        base_dir=base_dir,
        use_cudart=use_cudart,
        time_sleep=time_sleep,
        openxr_runtime_direct=openxr_runtime_direct,
        raw_q=raw_q,
        runtime_q=runtime_q,
        presentation_q=presentation_q,
        settings_update_q=settings_update_q,
        runtime_config=runtime_config,
        stereo_runtime=stereo_runtime,
        stereo_auto_enabled=stereo_auto_enabled,
        stereo_active_preset=stereo_active_preset,
        stereo_still_duration_s=0.0,
        stereo_last_auto_ts=time.perf_counter(),
        stereo_hot_reloader=StereoHotReloader(settings_path=os.path.join(base_dir, "settings.yaml")),
        stereo_warmup_tracker=StereoWarmupTracker(
            stereo_runtime,
            run_mode=run_mode,
            openxr_runtime_direct=openxr_runtime_direct,
            active_preset=stereo_active_preset,
        ),
        stereo_runtime_logger=StereoRuntimeLogger(
            stereo_runtime,
            active_preset_getter=lambda: stereo_active_preset,
        ),
        openxr_state=openxr_state,
        source_health=source_health,
        fps_breakdown_log=fps_breakdown_log,
        fps_breakdown=FPSBreakdown(enabled=fps_breakdown_log, target_fps=fps),
        thread_latencies={
            "capture": 0.0,
            "resize": 0.0,
            "runtime": 0.0,
            "render": 0.0,
            "total": 0.0,
        },
        capture_config=CaptureConfig(
            output_resolution=output_resolution,
            fps=fps,
            window_title=window_title,
            capture_mode=capture_mode,
            monitor_index=monitor_index,
            capture_tool=capture_tool,
            os_name=os_name,
            fps_provider=capture_fps_provider,
        ),
        render_size_config=render_size_config or RenderSizeConfig(),
        application_runtime_target=application_runtime_target,
        output_transport=output_transport,
        nvfruc_frame_generation=nvfruc_frame_generation,
    )


def build_capture_callbacks(
    *,
    raw_q,
    shutdown_event,
    queue_clear,
    inc_source_stat,
    inc_breakdown,
    put_raw_latest,
    is_paused,
    is_hard_idle,
    on_session_update,
    on_tick,
):
    return CaptureSessionCallbacks(
        clear_raw_queue=lambda: queue_clear(raw_q),
        inc_source_stat=inc_source_stat,
        inc_breakdown=inc_breakdown,
        put_raw_latest=put_raw_latest,
        is_shutdown=shutdown_event.is_set,
        is_paused=is_paused,
        is_hard_idle=is_hard_idle,
        on_session_update=on_session_update,
        on_tick=on_tick,
    )


def build_runtime_pipeline_context(
    *,
    shutdown_event,
    app_context: AppRuntimeContext,
    run_mode,
    device,
    capture_frame_to_rgb,
    prepare_rgb_for_stereo_runtime,
    current_openxr_render_config,
    is_hard_idle,
    is_source_paused,
    log_source_health,
    source_stat_inc,
    breakdown_inc,
    breakdown_add_time,
    breakdown_add_runtime_timing,
    set_preprocess_backend,
    queue_clear,
    queue_drain_latest,
    queue_put_latest,
    log_stereo_runtime_mode_once,
    apply_stereo_hot_reload_if_needed,
    warmup_stereo_once_for_frame,
    log_fast_plus_fused_runtime_state,
    runtime_ready_event=None,
    openxr_presenter_pressure=None,
):
    return RuntimePipelineContext(
        shutdown_event=shutdown_event,
        raw_q=app_context.raw_q,
        runtime_q=app_context.runtime_q,
        time_sleep=app_context.time_sleep,
        run_mode=run_mode,
        openxr_runtime_direct=app_context.openxr_runtime_direct,
        stereo_active_preset=getattr(app_context, "stereo_active_preset", None),
        device=device,
        use_cudart=app_context.use_cudart,
        thread_latencies=app_context.thread_latencies,
        stereo_runtime=app_context.stereo_runtime,
        runtime_config=app_context.runtime_config,
        capture_frame_to_rgb=capture_frame_to_rgb,
        prepare_rgb_for_stereo_runtime=prepare_rgb_for_stereo_runtime,
        current_openxr_render_config=current_openxr_render_config,
        is_hard_idle=is_hard_idle,
        is_source_paused=is_source_paused,
        log_source_health=log_source_health,
        source_stat_inc=source_stat_inc,
        breakdown_inc=breakdown_inc,
        breakdown_add_time=breakdown_add_time,
        breakdown_add_runtime_timing=breakdown_add_runtime_timing,
        set_preprocess_backend=set_preprocess_backend,
        queue_clear=queue_clear,
        queue_drain_latest=queue_drain_latest,
        queue_put_latest=queue_put_latest,
        log_stereo_runtime_mode_once=log_stereo_runtime_mode_once,
        apply_stereo_hot_reload_if_needed=apply_stereo_hot_reload_if_needed,
        warmup_stereo_once_for_frame=warmup_stereo_once_for_frame,
        log_fast_plus_fused_runtime_state=log_fast_plus_fused_runtime_state,
        runtime_ready_event=runtime_ready_event,
        openxr_presenter_pressure=openxr_presenter_pressure,
        application_runtime_target=getattr(app_context, "application_runtime_target", None),
        output_transport=getattr(app_context, "output_transport", None),
        settings_update_q=app_context.settings_update_q,
        render_size_config=app_context.render_size_config,
        capture_size_hint=(
            tuple(int(value) for value in app_context.capture_config.output_resolution)
            if isinstance(app_context.capture_config.output_resolution, (tuple, list))
            and len(app_context.capture_config.output_resolution) == 2
            else None
        ),
    )
