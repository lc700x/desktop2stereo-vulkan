from utils.breakdown import FPSBreakdown, fps_breakdown_default


def test_fps_breakdown_defaults_off_for_every_run_mode(monkeypatch):
    monkeypatch.delenv("D2S_OPENXR_DEBUG", raising=False)

    assert fps_breakdown_default(
        {"Run Mode": "Local Viewer", "Show FPS": True}, "Viewer"
    ) == "0"
    assert fps_breakdown_default(
        {"Run Mode": "Local Viewer", "Show FPS": False}, "Viewer"
    ) == "0"
    assert fps_breakdown_default(
        {"Run Mode": "RTMP Streamer", "Show FPS": True}, "Viewer"
    ) == "0"
    assert fps_breakdown_default(
        {"Run Mode": "Moonlight Streamer", "Show FPS": True}, "Viewer"
    ) == "0"


def test_fps_breakdown_preserves_openxr_debug_default(monkeypatch):
    monkeypatch.setenv("D2S_OPENXR_DEBUG", "0")

    assert fps_breakdown_default(
        {"Run Mode": "Local Viewer", "Show FPS": True}, "Viewer"
    ) == "0"


def test_fps_breakdown_logs_five_times_at_fifteen_second_intervals(capsys):
    breakdown = FPSBreakdown(enabled=True, target_fps=60)
    start = breakdown.last_log
    breakdown.inc("capture", 10)

    breakdown.log(now=start + 14.99)
    assert "[FPSBreakdown]" not in capsys.readouterr().out

    breakdown.log(now=start + 15.0)
    first = capsys.readouterr().out
    assert first.count("[FPSBreakdown]") == 1
    assert "sample=1" in first
    assert "window=15.00s" in first

    for index in range(2, 6):
        breakdown.log(now=start + 15.0 * index)
        assert capsys.readouterr().out.count("[FPSBreakdown]") == 1

    breakdown.log(now=start + 90.0)
    assert "[FPSBreakdown]" not in capsys.readouterr().out


def test_fps_breakdown_honors_explicit_log_limit(monkeypatch, capsys):
    monkeypatch.setenv("D2S_FPS_BREAKDOWN_LIMIT", "2")
    monkeypatch.setenv("D2S_FPS_BREAKDOWN_INTERVAL", "1")
    breakdown = FPSBreakdown(enabled=True, target_fps=60)
    start = breakdown.last_log

    for index in range(1, 4):
        breakdown.log(now=start + index)

    assert capsys.readouterr().out.count("[FPSBreakdown]") == 2


def test_fps_breakdown_reports_normalized_hole_fill_mode(capsys):
    breakdown = FPSBreakdown(enabled=True, target_fps=60)
    start = breakdown.last_log
    breakdown.set_latest("rt_backend", "quality_4k")
    breakdown.set_latest("rt_hole_fill_mode", "quality")
    breakdown.set_latest("rt_hole_fill_radius", 3)
    breakdown.set_latest("rt_hole_fill_strength", 1.0)

    breakdown.log(now=start + 15.0)

    output = capsys.readouterr().out
    assert "rt_backend=quality_4k" in output
    assert "rt_hole_fill=quality(3/1.00)" in output


def test_fps_breakdown_reports_no_fill_execution_path(capsys):
    breakdown = FPSBreakdown(enabled=True, target_fps=60)
    start = breakdown.last_log
    breakdown.set_latest("rt_sbs_backend", "openxr_triton_no_fill_fused_rgba_u8")
    breakdown.set_latest("rt_openxr_prewarp_backend", "triton_no_fill_fused_rgba_u8")
    breakdown.set_latest("rt_no_fill_fused_reason", "used")
    breakdown.set_latest("rt_temporal_enabled", 0)
    breakdown.set_latest("rt_refine_enabled", 0)
    breakdown.set_latest("rt_occlusion_enabled", 1)

    breakdown.log(now=start + 15.0)

    output = capsys.readouterr().out
    assert "rt_prewarp=triton_no_fill_fused_rgba_u8" in output
    assert "rt_no_fill=used" in output
    assert "rt_temporal=0" in output
    assert "rt_refine=0" in output
    assert "rt_occlusion=1" in output


def test_fps_breakdown_reports_parallel_slot_wait(capsys):
    breakdown = FPSBreakdown(enabled=True, target_fps=60)
    start = breakdown.last_log
    runtime_result = type(
        "Result",
        (),
        {
            "timing": {"depth_slot_wait_ms": 18.25},
            "debug_info": {
                "parallel_inference_workers": 2,
                "parallel_inference_effective_workers": 1,
                "parallel_inference_backoff": 1,
            },
        },
    )()
    breakdown.add_runtime_timing(runtime_result)
    breakdown.log(now=start + 15.0)

    output = capsys.readouterr().out
    assert "rt_slot_wait=18.25ms" in output
    assert "rt_parallel_effective_workers=1" in output
    assert "rt_parallel_backoff=1" in output


def test_fps_breakdown_reports_bounded_openxr_timing_percentiles(capsys):
    breakdown = FPSBreakdown(enabled=True, target_fps=60)
    start = breakdown.last_log
    for value_ms in (10.0, 20.0, 30.0):
        breakdown.add_time("openxr_frame_total", value_ms / 1000.0)
        breakdown.add_time("openxr_projection_total", value_ms / 1000.0)
        breakdown.add_time("openxr_vulkan_composer_queue_submit", value_ms / 1000.0)
        breakdown.add_time("openxr_vulkan_composer_fence_wait", value_ms / 1000.0)
        breakdown.add_time("openxr_vulkan_output_convert", value_ms / 1000.0)
    for value_ms in (12.0, 20.0, 30.0):
        runtime_result = type(
            "Result",
            (),
            {"timing": {"total_ms": value_ms, "depth_slot_wait_ms": value_ms / 2.0}, "debug_info": {}},
        )()
        breakdown.add_runtime_timing(runtime_result)

    breakdown.log(now=start + 15.0)

    output = capsys.readouterr().out
    assert "xr_frame_p50=20.00ms" in output
    assert "xr_frame_p95=29.00ms" in output
    assert "xr_frame_max=30.00ms" in output
    assert "vk_submit_p95=29.00ms" in output
    assert "rt_total_p50=20.00ms" in output
    assert "rt_slot_wait_p95=14.50ms" in output


def test_fps_breakdown_reports_stable_cadence_and_media_depth_identity(capsys):
    breakdown = FPSBreakdown(enabled=True, target_fps=72)
    start = breakdown.last_log
    runtime_result = type(
        "Result",
        (),
        {
            "timing": {
                "depth_total_ms": 18.0,
                "synthesis_ms": 5.0,
                "openxr_render_ms": 7.0,
                "pack_ms": 2.0,
            },
            "debug_info": {
                "media_generation": 4,
                "source_frame_index": 88,
                "source_pts": 2.933333,
                "media_depth_reused": 1,
                "media_depth_age_ms": 12.5,
                "media_depth_cache_reason": "exact_match",
                "runtime_execution_ms": 11.0,
                "runtime_gpu_wait_ms": 21.0,
                "runtime_completion_ms": 32.0,
                "capture_to_runtime_start_ms": 4.0,
                "capture_to_runtime_ready_ms": 36.0,
                "stable_cadence_active_hz": 72,
                "stable_cadence_proposed_hz": 60,
                "stable_cadence_reason": "sustained_over_budget",
                "stable_cadence_depth_queue": 2,
                "stable_cadence_dropped": 3,
                "stable_cadence_frame_ms": 40.0,
                "stable_cadence_budget_ms": 33.333,
                "stable_cadence_over_budget_streak": 2,
                "stable_cadence_headroom_streak": 0,
                "stable_cadence_age_ms": 52.0,
                "stable_cadence_source": "period",
            },
        },
    )()

    breakdown.add_runtime_timing(runtime_result)
    breakdown.log(now=start + 15.0)

    output = capsys.readouterr().out
    assert "rt_depth=18.00ms" in output
    assert "rt_synth=5.00ms" in output
    assert "openxr_render=7.00ms" in output
    assert "media_gen=4" in output
    assert "source_frame=88" in output
    assert "depth_reused=1" in output
    assert "depth_age_ms=12.50" in output
    assert "depth_cache=exact_match" in output
    assert "frame_exec=11.00ms" in output
    assert "frame_exec_p95=11.00ms" in output
    assert "frame_gpu_wait=21.00ms" in output
    assert "frame_completion=32.00ms" in output
    assert "capture_to_start=4.00ms" in output
    assert "capture_ready=36.00ms" in output
    assert "capture_ready_p95=36.00ms" in output
    assert "cadence_active=72" in output
    assert "cadence_proposed=60" in output
    assert "cadence_reason=sustained_over_budget" in output
    assert "cadence_frame=40.00ms" in output
    assert "cadence_budget=33.33ms" in output
    assert "cadence_over=2" in output
    assert "cadence_headroom=0" in output
    assert "cadence_age=52.00ms" in output
    assert "cadence_source=period" in output


def test_fps_breakdown_reports_gc_and_optional_gpu_metrics(capsys):
    breakdown = FPSBreakdown(enabled=True, target_fps=60)
    start = breakdown.last_log
    breakdown.set_latest("sys_gpu", 81)
    breakdown.set_latest("sys_gpu_max", 97)

    breakdown.log(now=start + 15.0)

    output = capsys.readouterr().out
    assert "sys_gpu=81" in output
    assert "sys_gpu_max=97" in output
    assert "gc_window=" in output
    assert "gc_max=" in output
