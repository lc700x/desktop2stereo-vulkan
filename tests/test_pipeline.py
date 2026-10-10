import json
import queue
import sys as sys_module
import threading
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from path_config import APP_ROOT

from path_config import APP_ROOT

from capture.types import CapturedFrame
from stereo_runtime.render_size import RenderSizeConfig
import stereo_runtime.pipeline as pipeline_module

from stereo_runtime.pipeline import (
    _ParallelDepthScheduler,
    _prepare_frame_input,
    RuntimePipelineLoop,
    _resolve_pipeline_render_size,
    _unpack_raw_queue_item,
    _enable_openxr_depth_cuda_graph_if_needed,
    _motion_sample,
    _motion_score,
    _save_downsample_filter_comparison,
    _save_preprocess_image_diagnostic,
    _runtime_motion_gate_enabled,
    _runtime_pending_depth_limit,
    _runtime_parallel_adaptive_backoff_enabled,
    _cuda_event_ready,
    _attach_capture_debug,
    _native_coreml_capture_enabled,
)


def test_pipeline_uses_captured_dimensions_when_render_scale_is_4k():
    captured = CapturedFrame(
        frame=np.zeros((1200, 1920, 3), dtype=np.uint8),
        target_height=2160,
        timestamp=1.0,
        capture_size=(1920, 1200),
    )
    _frame, source_size, _timestamp, _captured = _unpack_raw_queue_item(captured)
    config = RenderSizeConfig(scale_factor="4K / 100%")

    assert source_size == (1920, 1200)
    assert _resolve_pipeline_render_size(source_size, config) == (1920, 1200)


def test_attach_capture_debug_marks_depth_complete_for_current_frame():
    result = SimpleNamespace(
        depth=np.zeros((2, 2), dtype=np.float32),
        depth_finite=True,
        debug_info={},
        timing={"depth_nonfinite_count": 0},
    )
    captured = CapturedFrame(
        frame=np.zeros((2, 2, 3), dtype=np.uint8),
        target_height=2,
        timestamp=1.0,
        metadata={"capture_frame_id": 42},
    )

    _attach_capture_debug(result, captured, SimpleNamespace())

    assert result.capture_frame_id == 42
    assert result.depth_frame_id == 42
    assert result.depth_complete is True
    assert result.debug_info["depth_finite"] == 1
    assert result.debug_info["depth_nonfinite_count"] == 0


def test_native_coreml_capture_is_enabled_for_macos_network_stream(monkeypatch):
    monkeypatch.setattr("stereo_runtime.pipeline.platform.system", lambda: "Darwin")
    monkeypatch.setenv("D2S_MAC_STREAM_NATIVE_IO", "1")

    assert _native_coreml_capture_enabled(
        SimpleNamespace(run_mode="RTMP Streamer", application_runtime_target="network_stream")
    )
    assert not _native_coreml_capture_enabled(
        SimpleNamespace(run_mode="RTMP Streamer", application_runtime_target="local_viewer")
    )
    assert _native_coreml_capture_enabled(
        SimpleNamespace(run_mode="Local Viewer", application_runtime_target="local_viewer")
    )


@pytest.mark.parametrize(
    ("source_size", "scale", "expected"),
    [
        ((1920, 1200), "4K / 100%", (1920, 1200)),
        ((3840, 2160), "4K / 100%", (3840, 2160)),
        ((3840, 2160), "1K / 50%", (1920, 1080)),
        ((3840, 2400), "1K / 50%", (1920, 1200)),
    ],
)
def test_scaled_render_size_preserves_source_aspect_ratio(source_size, scale, expected):
    assert _resolve_pipeline_render_size(
        source_size,
        RenderSizeConfig(scale_factor=scale),
    ) == expected


def test_save_preprocess_image_diagnostic_exports_exact_pre_inference_rgb(tmp_path):
    import cv2
    import numpy as np

    frame = np.zeros((2, 3, 3), dtype=np.float32)
    frame[..., 0] = 1.0
    frame[..., 1] = 0.5

    image_path, manifest_path = _save_preprocess_image_diagnostic(
        frame,
        capture_size=(3840, 2160),
        render_size=(3, 2),
        output_dir=tmp_path,
    )

    saved_rgb = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    assert saved_rgb.shape == (2, 3, 3)
    assert tuple(saved_rgb[0, 0]) == (255, 128, 0)
    manifest = manifest_path.read_text(encoding="utf-8")
    assert '"stage": "after_capture_preprocess_before_inference"' in manifest
    assert '"capture_size": [' in manifest
    assert '"render_size": [' in manifest


def test_save_downsample_filter_comparison_exports_same_frame_candidates(tmp_path):
    import cv2
    import numpy as np

    frame_bgra = np.zeros((4, 6, 4), dtype=np.uint8)
    frame_bgra[..., 0] = np.arange(6, dtype=np.uint8) * 30
    frame_bgra[..., 1] = 80
    frame_bgra[..., 2] = 160
    frame_bgra[..., 3] = 255

    comparison_dir = _save_downsample_filter_comparison(
        frame_bgra,
        capture_size=(6, 4),
        render_size=(3, 2),
        output_dir=tmp_path,
    )

    assert comparison_dir is not None
    assert cv2.imread(str(comparison_dir / "00_source_rgb.png")).shape[:2] == (4, 6)
    for name in (
        "01_area.png",
        "02_lanczos4.png",
        "03_bicubic.png",
        "04_area_rcas_035.png",
    ):
        assert cv2.imread(str(comparison_dir / name)).shape[:2] == (2, 3)
    assert (comparison_dir / "manifest.json").is_file()


def test_parallel_depth_scheduler_enforces_effective_submission_limit():
    scheduler = object.__new__(_ParallelDepthScheduler)
    scheduler.worker_count = 3
    scheduler.effective_limit = 3
    scheduler.pending = [object(), object()]
    scheduler.trim = lambda _limit: None

    assert scheduler.can_submit() is True
    assert scheduler.set_effective_limit(2) is True
    assert scheduler.can_submit() is False
    assert scheduler.set_effective_limit(1) is True
    assert scheduler.effective_limit == 1


def test_parallel_depth_scheduler_passes_native_capture_to_coreml_worker():
    class Runtime:
        def predict_openxr_depth(
            self, frame, *, native_capture=None, capture_frame_id=None
        ):
            return frame, native_capture, capture_frame_id

    scheduler = _ParallelDepthScheduler(Runtime(), max_workers=1)
    native_capture = object()
    try:
        job = scheduler.submit(
            "meta-frame",
            native_capture=native_capture,
            capture_frame_id=42,
            frame_rgb=None,
            frame_raw=None,
            size=(1920, 1080),
            capture_start_time=1.0,
            captured_frame=None,
            render_size=(1920, 1080),
            process_latency=0.0,
        )
        assert job.future.result(timeout=1.0) == (
            "meta-frame",
            native_capture,
            42,
        )
    finally:
        scheduler.close()


def test_parallel_scheduler_backoff_and_recovery_supports_three_workers(monkeypatch):
    monkeypatch.setenv("D2S_RUNTIME_PARALLEL_ADAPTIVE_BACKOFF", "1")
    class Scheduler:
        worker_count = 3
        effective_limit = 3

        def set_effective_limit(self, limit):
            self.effective_limit = max(1, min(self.worker_count, limit))
            return True

    events = []
    loop = object.__new__(RuntimePipelineLoop)
    loop.context = SimpleNamespace(
        source_stat_inc=lambda name, **kwargs: events.append((name, kwargs)),
        breakdown_inc=lambda *args, **kwargs: None,
    )
    loop._parallel_depth_scheduler = Scheduler()
    loop._parallel_backoff_until = 0.0
    loop._parallel_recovery_after = 0.0
    monkeypatch.setattr("stereo_runtime.pipeline.time.perf_counter", lambda: 10.0)
    monkeypatch.setenv("D2S_RUNTIME_PARALLEL_BACKOFF_S", "1.5")

    loop._parallel_reduce_for_pressure("test")
    assert loop._parallel_depth_scheduler.effective_limit == 2
    loop._parallel_reduce_for_pressure("test")
    assert loop._parallel_depth_scheduler.effective_limit == 1

    monkeypatch.setattr("stereo_runtime.pipeline.time.perf_counter", lambda: 11.6)
    loop._parallel_recover_if_ready()
    assert loop._parallel_depth_scheduler.effective_limit == 2
    monkeypatch.setattr("stereo_runtime.pipeline.time.perf_counter", lambda: 13.2)
    loop._parallel_recover_if_ready()
    assert loop._parallel_depth_scheduler.effective_limit == 3
    assert [name for name, _kwargs in events] == [
        "runtime_parallel_backoff",
        "runtime_parallel_backoff",
        "runtime_parallel_recovery",
        "runtime_parallel_recovery",
    ]


def test_parallel_adaptive_backoff_can_be_disabled(monkeypatch):
    monkeypatch.setenv("D2S_RUNTIME_PARALLEL_ADAPTIVE_BACKOFF", "0")
    assert _runtime_parallel_adaptive_backoff_enabled() is False

    class Scheduler:
        worker_count = 2
        effective_limit = 2

    events = []
    loop = object.__new__(RuntimePipelineLoop)
    loop.context = SimpleNamespace(
        source_stat_inc=lambda name, **kwargs: events.append(name),
        breakdown_inc=lambda *args, **kwargs: None,
    )
    loop._parallel_depth_scheduler = Scheduler()
    loop._parallel_reduce_for_pressure("test")

    assert loop._parallel_depth_scheduler.effective_limit == 2
    assert events == []


@pytest.mark.parametrize("run_mode", ["Viewer", "Local Viewer"])
def test_coreml_viewer_parallel_pending_defaults_to_two_on_macos(monkeypatch, run_mode):
    monkeypatch.setattr(sys_module, "platform", "darwin")
    monkeypatch.setattr(pipeline_module.platform, "system", lambda: "Darwin")
    monkeypatch.delenv("D2S_RUNTIME_PENDING_CUDA_DEPTH", raising=False)
    monkeypatch.delenv("D2S_RUNTIME_PARALLEL_MPS", raising=False)
    runtime = SimpleNamespace(
        depth_provider=SimpleNamespace(pipeline_slot_count=2),
        config=SimpleNamespace(profile_sync=False),
        _resolved_stereo_compute_backend="",
    )
    context = SimpleNamespace(
        run_mode=run_mode,
        runtime_config=SimpleNamespace(
            parallel_inference=True,
            parallel_inference_workers=2,
            use_coreml=True,
        ),
        stereo_runtime=runtime,
    )

    assert _runtime_pending_depth_limit(context) == 2


def test_local_viewer_parallel_requires_native_coreml_on_macos(monkeypatch):
    monkeypatch.setattr(sys_module, "platform", "darwin")
    monkeypatch.setattr(pipeline_module.platform, "system", lambda: "Darwin")
    monkeypatch.delenv("D2S_RUNTIME_PENDING_CUDA_DEPTH", raising=False)
    monkeypatch.delenv("D2S_RUNTIME_PARALLEL_MPS", raising=False)
    context = SimpleNamespace(
        run_mode="Local Viewer",
        runtime_config=SimpleNamespace(
            parallel_inference=True,
            parallel_inference_workers=2,
            use_coreml=False,
        ),
        stereo_runtime=SimpleNamespace(
            depth_provider=SimpleNamespace(pipeline_slot_count=2),
            config=SimpleNamespace(profile_sync=False),
            _resolved_stereo_compute_backend="",
        ),
    )

    assert _runtime_pending_depth_limit(context) == 1


def test_coreml_viewer_parallel_pending_respects_explicit_mps_disable(monkeypatch):
    monkeypatch.setattr(sys_module, "platform", "darwin")
    monkeypatch.setattr(pipeline_module.platform, "system", lambda: "Darwin")
    monkeypatch.delenv("D2S_RUNTIME_PENDING_CUDA_DEPTH", raising=False)
    monkeypatch.setenv("D2S_RUNTIME_PARALLEL_MPS", "0")
    runtime = SimpleNamespace(
        depth_provider=SimpleNamespace(pipeline_slot_count=2),
        config=SimpleNamespace(profile_sync=False),
        _resolved_stereo_compute_backend="",
    )
    context = SimpleNamespace(
        run_mode="Viewer",
        runtime_config=SimpleNamespace(
            parallel_inference=True,
            parallel_inference_workers=2,
            use_coreml=True,
        ),
        stereo_runtime=runtime,
    )

    assert _runtime_pending_depth_limit(context) == 1


def test_coreml_viewer_parallel_enables_adaptive_backoff_by_default(monkeypatch):
    monkeypatch.delenv("D2S_RUNTIME_PARALLEL_ADAPTIVE_BACKOFF", raising=False)
    context = SimpleNamespace(
        run_mode="Viewer",
        runtime_config=SimpleNamespace(
            parallel_inference=True,
            parallel_inference_workers=2,
            use_coreml=True,
        ),
        stereo_runtime=SimpleNamespace(
            depth_provider=SimpleNamespace(pipeline_slot_count=2)
        ),
    )

    assert _runtime_parallel_adaptive_backoff_enabled(context) is True


def test_coreml_viewer_parallel_adaptive_backoff_can_be_disabled(monkeypatch):
    monkeypatch.setenv("D2S_RUNTIME_PARALLEL_ADAPTIVE_BACKOFF", "0")
    context = SimpleNamespace(
        run_mode="Viewer",
        runtime_config=SimpleNamespace(
            parallel_inference=True,
            parallel_inference_workers=2,
            use_coreml=True,
        ),
        stereo_runtime=SimpleNamespace(
            depth_provider=SimpleNamespace(pipeline_slot_count=2)
        ),
    )

    assert _runtime_parallel_adaptive_backoff_enabled(context) is False


def test_presenter_pressure_temporarily_limits_parallel_depth_and_recovers(monkeypatch):
    monkeypatch.delenv("D2S_RUNTIME_PRESENTER_BACKPRESSURE", raising=False)
    pressure = {"active": True}
    loop = object.__new__(RuntimePipelineLoop)
    loop.context = _dual_pending_context()
    loop.context.openxr_presenter_pressure = lambda: pressure["active"]
    loop._parallel_depth_scheduler = SimpleNamespace(effective_limit=2)
    loop._dual_pending_cooldown_until = 0.0
    loop._presenter_backpressure_active = False

    assert loop._pending_depth_limit() == 1
    assert loop._presenter_backpressure_active is True

    pressure["active"] = False
    assert loop._pending_depth_limit() == 2
    assert loop._presenter_backpressure_active is False


def test_output_queue_cooldown_applies_with_parallel_scheduler(monkeypatch):
    loop = object.__new__(RuntimePipelineLoop)
    loop.context = _dual_pending_context()
    loop.context.openxr_presenter_pressure = lambda: False
    loop._parallel_depth_scheduler = SimpleNamespace(effective_limit=2)
    loop._dual_pending_cooldown_until = 11.0
    loop._presenter_backpressure_active = False
    monkeypatch.setattr("stereo_runtime.pipeline.time.perf_counter", lambda: 10.0)

    assert loop._pending_depth_limit() == 1


def test_openxr_motion_gate_matches_legacy_default(monkeypatch):
    monkeypatch.delenv("D2S_RUNTIME_MOTION_GATE", raising=False)

    assert _runtime_motion_gate_enabled(SimpleNamespace(run_mode="OpenXR")) is False


def test_motion_gate_can_be_enabled_explicitly(monkeypatch):
    monkeypatch.setenv("D2S_RUNTIME_MOTION_GATE", "1")

    assert _runtime_motion_gate_enabled(SimpleNamespace(run_mode="OpenXR")) is True


def test_motion_gate_can_be_disabled_explicitly(monkeypatch):
    monkeypatch.setenv("D2S_RUNTIME_MOTION_GATE", "0")

    assert _runtime_motion_gate_enabled(SimpleNamespace(run_mode="OpenXR")) is False


def test_motion_sample_is_snapshot_when_input_storage_is_reused():
    import torch

    frame = torch.zeros((1, 3, 36, 64), dtype=torch.float32)
    previous = _motion_sample(frame)
    frame.fill_(1.0)
    current = _motion_sample(frame)

    assert _motion_score(previous, current) == 1.0


def test_openxr_pipeline_keeps_deferred_vulkan_request_path():
    source = (APP_ROOT / "stereo_runtime/pipeline.py").read_text(
        encoding="utf-8"
    )

    assert "ctx.stereo_runtime.process_openxr_frame(" in source
    assert "openxr_result_from_stereo_result" not in source
    assert 'output_mode="full_synthesis_eyes"' in source


def test_publish_runtime_item_marks_inference_pipeline_ready():
    ready = threading.Event()
    runtime_q = queue.Queue(maxsize=1)
    loop = object.__new__(RuntimePipelineLoop)
    loop.context = SimpleNamespace(
        runtime_q=runtime_q,
        runtime_ready_event=ready,
        queue_put_latest=lambda q, item: q.put_nowait(item),
        breakdown_inc=lambda *args, **kwargs: None,
        breakdown_add_time=lambda *args, **kwargs: None,
        source_stat_inc=lambda *args, **kwargs: None,
    )

    loop._publish_runtime_item((object(), 1.0, 0.01, 0.02, None))

    assert ready.is_set()
    assert runtime_q.qsize() == 1


def test_pending_cuda_retains_latest_raw_frame(monkeypatch):
    raw_q = queue.Queue(maxsize=1)
    raw_q.put_nowait("latest-frame")
    stats = []
    breakdown = []
    sleeps = []
    loop = object.__new__(RuntimePipelineLoop)
    loop.context = SimpleNamespace(
        raw_q=raw_q,
        time_sleep=1.0 / 60.0,
        source_stat_inc=lambda name, *args, **kwargs: stats.append(name),
        breakdown_inc=lambda name, *args, **kwargs: breakdown.append(name),
    )
    monkeypatch.setattr(
        "stereo_runtime.pipeline.time.sleep",
        lambda seconds: sleeps.append(seconds),
    )

    loop._retain_latest_raw_while_cuda_pending()

    assert raw_q.get_nowait() == "latest-frame"
    assert stats == ["runtime_pending_cuda_inflight"]
    assert breakdown == ["runtime_pending_cuda_inflight"]
    assert sleeps == [0.001]


def test_cuda_ready_event_uses_gpu_waitable_event(monkeypatch):
    import torch

    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    event = SimpleNamespace(wait=lambda: None, query=lambda: False)

    assert _cuda_event_ready(event) is True


def test_rocm_ready_event_keeps_query_pending_path(monkeypatch):
    import torch

    monkeypatch.setattr(torch.version, "hip", "7.0", raising=False)
    event = SimpleNamespace(wait=lambda: None, query=lambda: False)

    assert _cuda_event_ready(event) is False


def _dual_pending_context(
    *, backend="cuda_triton", slots=2, temporal=False, workers=2, run_mode="OpenXR"
):
    runtime = SimpleNamespace(
        depth_provider=SimpleNamespace(pipeline_slot_count=slots),
        config=SimpleNamespace(profile_sync=False),
        stereo_config=SimpleNamespace(temporal=temporal, convergence=0.0),
        _resolved_stereo_compute_backend=backend,
    )
    return SimpleNamespace(
        run_mode=run_mode,
        openxr_runtime_direct=False,
        stereo_active_preset="cinema",
        stereo_runtime=runtime,
        runtime_config=SimpleNamespace(parallel_inference=True, parallel_inference_workers=workers),
    )


def test_openxr_safe_dual_slot_defaults_to_two_pending(monkeypatch):
    monkeypatch.delenv("D2S_RUNTIME_PENDING_CUDA_DEPTH", raising=False)

    assert _runtime_pending_depth_limit(_dual_pending_context()) == 2


@pytest.mark.parametrize(
    ("run_mode", "platform", "use_coreml", "expected_workers"),
    [("Viewer", "linux", False, 2), ("Local Viewer", "darwin", True, 1)],
)
def test_viewer_depth_scheduler_uses_measured_worker_policy(
    monkeypatch, run_mode, platform, use_coreml, expected_workers
) -> None:
    monkeypatch.setattr(sys_module, "platform", platform)
    monkeypatch.setattr(
        pipeline_module.platform,
        "system",
        lambda: "Darwin" if platform == "darwin" else "Linux",
    )
    monkeypatch.delenv("D2S_RUNTIME_PARALLEL_MPS", raising=False)
    events = []
    runtime = SimpleNamespace(
        depth_provider=SimpleNamespace(pipeline_slot_count=2),
        config=SimpleNamespace(profile_sync=False),
        predict_openxr_depth=lambda _frame: None,
    )
    loop = RuntimePipelineLoop(
        SimpleNamespace(
            run_mode=run_mode,
            runtime_config=SimpleNamespace(
                parallel_inference=True,
                parallel_inference_workers=2,
                use_coreml=use_coreml,
            ),
            stereo_runtime=runtime,
            source_stat_inc=lambda name, **values: events.append((name, values)),
        )
    )

    loop._ensure_parallel_depth_scheduler()

    if expected_workers == 1:
        assert loop._parallel_depth_scheduler is None
        assert events == []
        assert loop._parallel_mps_serial_notice_emitted is True
    else:
        assert loop._parallel_depth_scheduler is not None
        assert loop._parallel_depth_scheduler.worker_count == expected_workers
        assert events == [
            ("runtime_parallel_workers", {"active_workers": expected_workers})
        ]
        loop._parallel_depth_scheduler.close()


def test_local_viewer_coreml_parallel_can_be_explicitly_enabled(monkeypatch) -> None:
    monkeypatch.setattr(sys_module, "platform", "darwin")
    monkeypatch.setattr(pipeline_module.platform, "system", lambda: "Darwin")
    monkeypatch.setenv("D2S_RUNTIME_PARALLEL_MPS", "1")
    events = []
    runtime = SimpleNamespace(
        depth_provider=SimpleNamespace(pipeline_slot_count=2),
        config=SimpleNamespace(profile_sync=False),
        predict_openxr_depth=lambda _frame: None,
    )
    loop = RuntimePipelineLoop(
        SimpleNamespace(
            run_mode="Local Viewer",
            runtime_config=SimpleNamespace(
                parallel_inference=True,
                parallel_inference_workers=2,
                use_coreml=True,
            ),
            stereo_runtime=runtime,
            source_stat_inc=lambda name, **values: events.append((name, values)),
        )
    )

    loop._ensure_parallel_depth_scheduler()

    assert loop._parallel_depth_scheduler is not None
    assert loop._parallel_depth_scheduler.worker_count == 2
    assert events == [("runtime_parallel_workers", {"active_workers": 2})]
    loop._parallel_depth_scheduler.close()


def test_viewer_dual_pending_supports_ordered_temporal_synthesis(monkeypatch):
    monkeypatch.delenv("D2S_RUNTIME_PENDING_CUDA_DEPTH", raising=False)

    assert _runtime_pending_depth_limit(
        _dual_pending_context(run_mode="Viewer", temporal=True)
    ) == 2


def test_openxr_temporal_pipeline_keeps_existing_single_pending_limit(monkeypatch):
    monkeypatch.delenv("D2S_RUNTIME_PENDING_CUDA_DEPTH", raising=False)

    assert _runtime_pending_depth_limit(_dual_pending_context(temporal=True)) == 1


def test_openxr_three_slot_defaults_to_three_pending(monkeypatch):
    monkeypatch.delenv("D2S_RUNTIME_PENDING_CUDA_DEPTH", raising=False)

    assert _runtime_pending_depth_limit(_dual_pending_context(slots=3, workers=3)) == 3


def test_openxr_dual_pending_can_be_forced_back_to_one(monkeypatch):
    monkeypatch.setenv("D2S_RUNTIME_PENDING_CUDA_DEPTH", "1")

    assert _runtime_pending_depth_limit(_dual_pending_context()) == 1


def test_openxr_dual_pending_requires_explicit_experimental_override(monkeypatch):
    monkeypatch.setenv("D2S_RUNTIME_PENDING_CUDA_DEPTH", "2")

    assert _runtime_pending_depth_limit(_dual_pending_context()) == 2


def test_vulkan_deferred_pipeline_remains_single_pending(monkeypatch):
    monkeypatch.delenv("D2S_RUNTIME_PENDING_CUDA_DEPTH", raising=False)

    assert _runtime_pending_depth_limit(_dual_pending_context(backend="vulkan")) == 1


def test_unisolated_depth_provider_remains_single_pending(monkeypatch):
    monkeypatch.delenv("D2S_RUNTIME_PENDING_CUDA_DEPTH", raising=False)

    assert _runtime_pending_depth_limit(_dual_pending_context(slots=1)) == 1


def test_dual_pending_backs_off_when_presenter_has_not_consumed_output():
    runtime_q = queue.Queue(maxsize=1)
    runtime_q.put_nowait("previous")
    loop = object.__new__(RuntimePipelineLoop)
    loop.context = _dual_pending_context()
    loop.context.runtime_q = runtime_q
    loop.context.runtime_ready_event = None
    loop.context.queue_put_latest = lambda q, item: (q.get_nowait(), q.put_nowait(item))
    loop.context.breakdown_inc = lambda *args, **kwargs: None
    loop.context.breakdown_add_time = lambda *args, **kwargs: None
    loop.context.source_stat_inc = lambda *args, **kwargs: None
    loop._dual_pending_cooldown_until = 0.0
    loop._last_algorithm_output_time = 0.0

    loop._publish_runtime_item((object(), 1.0, 0.01, 0.02, None))

    assert loop._pending_depth_limit() == 1


def test_pipeline_rebuilds_provider_after_consecutive_failures(monkeypatch):
    calls = []

    class Runtime:
        def _rebuild_depth_provider(self):
            calls.append("rebuild")

        def reset_temporal(self):
            calls.append("reset")

    stats = []
    loop = object.__new__(RuntimePipelineLoop)
    loop.context = SimpleNamespace(
        stereo_runtime=Runtime(),
        source_stat_inc=lambda name, *args, **kwargs: stats.append(name),
    )
    loop._consecutive_runtime_errors = 3
    monkeypatch.setenv("D2S_RUNTIME_REBUILD_AFTER_ERRORS", "3")

    loop._rebuild_after_consecutive_failures()

    assert calls == ["rebuild", "reset"]
    assert "runtime_adapter_rebuilds" in stats
    assert loop._consecutive_runtime_errors == 0


def test_pipeline_does_not_rebuild_before_threshold(monkeypatch):
    calls = []

    class Runtime:
        def _rebuild_depth_provider(self):
            calls.append("rebuild")

    loop = object.__new__(RuntimePipelineLoop)
    loop.context = SimpleNamespace(
        stereo_runtime=Runtime(),
        source_stat_inc=lambda *args, **kwargs: None,
    )
    loop._consecutive_runtime_errors = 2
    monkeypatch.setenv("D2S_RUNTIME_REBUILD_AFTER_ERRORS", "3")

    loop._rebuild_after_consecutive_failures()

    assert calls == []
    assert loop._consecutive_runtime_errors == 2


def test_prepare_frame_input_uses_directml_native_bridge_and_records_decision(monkeypatch):
    from utils.display_info import DisplayInfo

    monkeypatch.setattr(
        "utils.display_info.enumerate_displays",
        lambda: [
            DisplayInfo(
                capture_index=1,
                display_number=1,
                left=0,
                top=0,
                width=1920,
                height=1080,
                stable_id="cg:1552:41036:4251086178",
                name="iMac",
                manufacturer="1552",
                model="iMac",
                serial="4251086178",
            ),
            DisplayInfo(
                capture_index=2,
                display_number=2,
                left=1920,
                top=0,
                width=1920,
                height=1080,
                stable_id="cg:2198:39711:3271970828",
                name="VITURE",
                manufacturer="2198",
                model="VITURE",
                serial="3271970828",
            ),
        ],
    )
    class SharedResource:
        adapter_luid = 0x10
        format = "BGRA8"
        width = 16
        height = 8
        shared_handle = 1

        def to_directml_tensor(self):
            return "dml-tensor"

    resource = SharedResource()
    captured = CapturedFrame(
        resource,
        8,
        1.0,
        native_resource=resource,
        cpu_compat_frame="cpu-frame",
    )
    runtime = SimpleNamespace(
        provider_report=lambda: {"depth_backend": "directml"},
        config=SimpleNamespace(adapter_luid=0x10),
        _vulkan_stereo_backend=None,
        _vulkan_context=None,
    )
    ctx = SimpleNamespace(stereo_runtime=runtime)

    assert _prepare_frame_input(ctx, captured, "cpu-frame") == "dml-tensor"
    assert captured.metadata["directml_resource_mode"] == "shared"
    assert captured.metadata["directml_zero_copy_ready"] is True


def test_backend_status_log_reports_explicit_resource_fallback(capsys):
    runtime = SimpleNamespace(
        provider_report=lambda: {
            "depth_backend": "cpu",
            "attempts": [{"backend": "tensorrt", "status": "failed"}],
            "fallback_reason": "TensorRT unavailable",
        },
        config=SimpleNamespace(device="cpu"),
        _resolved_stereo_compute_backend="torch",
        _stereo_compute_backend_reason="vulkan_unavailable",
    )
    loop = object.__new__(RuntimePipelineLoop)
    loop.context = SimpleNamespace(stereo_runtime=runtime)
    loop._backend_status_emitted = False

    loop._emit_backend_status_once({
        "capture_mode": "monitor",
        "capture_tool": "WindowsCapture",
        "capture_adapter_luid": 7,
        "capture_resource_kind": "d3d11_texture",
        "capture_resource_format": "BGRA8",
        "capture_gpu_to_cpu": True,
        "capture_gpu_copy_count": 1,
        "capture_zero_copy": False,
        "capture_zero_copy_ready": False,
    })

    line = capsys.readouterr().out.strip()
    assert line.startswith("[D2S_BACKEND_STATUS] ")
    payload = json.loads(line.split(" ", 1)[1])
    assert payload["depth_backend"] == "cpu"
    assert payload["gpu_to_cpu"] is True
    assert payload["gpu_copy_count"] == 1
    assert payload["zero_copy"] is False
    assert "TensorRT unavailable" in payload["fallback_reasons"]
    loop._emit_backend_status_once({})
    assert capsys.readouterr().out == ""


def test_cuda_capture_disables_previously_enabled_openxr_cuda_graph():
    snapshots = []

    class Runtime:
        config = SimpleNamespace(depth_backend="tensorrt_native", use_cuda_graph=True)

        def apply_settings_snapshot(self, snapshot, *, active_preset):
            snapshots.append(snapshot)
            self.config.use_cuda_graph = snapshot.use_cuda_graph

    stats = []
    ctx = SimpleNamespace(
        stereo_runtime=Runtime(),
        stereo_active_preset="quality_4k",
        source_stat_inc=lambda name, *args, **kwargs: stats.append(name),
    )

    _enable_openxr_depth_cuda_graph_if_needed(
        ctx,
        True,
        SimpleNamespace(capture_tool="WindowsCaptureCUDA"),
    )

    assert [snapshot.use_cuda_graph for snapshot in snapshots] == [False]
    assert ctx.stereo_runtime.config.use_cuda_graph is False
    assert "openxr_depth_cuda_graph_disabled_cuda_capture" in stats
