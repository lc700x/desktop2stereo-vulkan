from utils.breakdown import FPSBreakdown


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
