from __future__ import annotations

from dataclasses import dataclass

import os
import threading
import time


@dataclass(frozen=True)
class OpenXRAsyncValidation:
    passed: bool
    missing: tuple[str, ...]
    failed: tuple[str, ...]


LATEST_KEYS = {
    "rt_backend",
    "rt_depth_backend",
    "rt_depth_slot",
    "rt_depth_slot_count",
    "rt_output_format",
    "rt_output_dtype",
    "rt_output_pack",
    "rt_sbs_backend",
    "rt_openxr_prewarp_backend",
    "rt_no_fill_fused_reason",
    "rt_temporal_enabled",
    "rt_refine_enabled",
    "rt_occlusion_enabled",
    "rt_occ_backend",
    "rt_fill_backend",
    "rt_hole_fill_mode",
    "rt_hole_fill_radius",
    "rt_hole_fill_strength",
    "rt_depth_strength",
    "rt_convergence",
    "rt_max_disparity_px",
    "rt_resolved_max_disparity_px",
    "rt_parallax_preset",
    "rt_parallax_budget_preset",
    "rt_depth_total_ms",
    "rt_depth_model_ms",
    "rt_depth_slot_wait_ms",
    "rt_synthesis_ms",
    "rt_total_ms",
    "openxr_async_effects_enabled",
    "parallel_inference_enabled",
    "parallel_inference_pending_limit",
    "parallel_inference_workers",
    "parallel_inference_effective_workers",
    "parallel_inference_backoff",
    "parallel_inference_adaptive_backoff",
    "parallel_inference_pending",
    "parallel_inference_dropped",
    "openxr_vulkan_projection_quality_chain_requested",
}


def _breakdown_log_interval() -> float:
    try:
        return max(float(os.environ.get("D2S_FPS_BREAKDOWN_INTERVAL", "15.0")), 0.1)
    except (TypeError, ValueError):
        return 15.0


def _breakdown_log_limit() -> int:
    try:
        return max(int(os.environ.get("D2S_FPS_BREAKDOWN_LIMIT", "5")), 0)
    except (TypeError, ValueError):
        return 5


class FPSBreakdown:
    def __init__(self, *, enabled: bool, target_fps: int | float):
        self.enabled = enabled
        self.target_fps = target_fps
        self.lock = threading.Lock()
        self.stats = {
            "capture": 0,
            "raw_get": 0,
            "runtime": 0,
            "viewer_get": 0,
            "viewer_drop": 0,
            "local_presented_frame": 0,
            "loops": 0,
            "update_ms": 0.0,
            "render_ms": 0.0,
            "swap_ms": 0.0,
            "wait_ms": 0.0,
            "openxr_poll_ms": 0.0,
            "openxr_upload_ms": 0.0,
            "update_count": 0,
            "render_count": 0,
            "swap_count": 0,
            "wait_count": 0,
            "openxr_poll_count": 0,
            "openxr_upload_count": 0,
        }
        self.last_log = time.perf_counter()
        self.log_count = 0
        self.log_limit = _breakdown_log_limit()
        self.log_delay = _breakdown_log_interval()

    def inc(self, name: str, amount: int | float = 1) -> None:
        if not self.enabled:
            return
        with self.lock:
            self.stats[name] = self.stats.get(name, 0) + amount

    def add_time(self, name: str, seconds: float) -> None:
        if not self.enabled:
            return
        with self.lock:
            self.stats[f"{name}_ms"] = self.stats.get(f"{name}_ms", 0.0) + seconds * 1000.0
            self.stats[f"{name}_count"] = self.stats.get(f"{name}_count", 0) + 1

    def add_value(self, name: str, value: float) -> None:
        if not self.enabled:
            return
        with self.lock:
            self.stats[f"{name}_total"] = self.stats.get(f"{name}_total", 0.0) + float(value)
            self.stats[f"{name}_count"] = self.stats.get(f"{name}_count", 0) + 1

    def set_latest(self, name: str, value) -> None:
        if not self.enabled:
            return
        with self.lock:
            self.stats[name] = value

    def _validate_openxr_async_stats(self, stats) -> OpenXRAsyncValidation:
        missing = []
        failed = []
        screen_present = (
            stats.get("openxr_new_screen_frame", 0)
            + stats.get("openxr_reused_screen_frame", 0)
            + stats.get("openxr_projection_screen_present", 0)
        )
        if screen_present <= 0:
            missing.append("screen_present")
        if stats.get("openxr_quad_layer_failed", 0) > 0:
            failed.append("quad_layer_failed")
        if stats.get("openxr_d3d11_pbo_readback", 0) > 0:
            failed.append("d3d11_pbo_readback")
        if stats.get("openxr_no_renderable", 0) > 0 and screen_present <= 0:
            failed.append("no_renderable_without_quad_reuse")
        effects_enabled = bool(stats.get("openxr_async_effects_enabled", True))
        if effects_enabled and stats.get("openxr_effect_submit_count", 0) <= 0 and stats.get("openxr_effect_source_reused_safe", 0) <= 0:
            missing.append("effect_submit_or_safe_reuse")
        if stats.get("openxr_background_layer_failed", 0) > 0 and screen_present <= 0:
            failed.append("background_failure_blocked_screen")
        return OpenXRAsyncValidation(not missing and not failed, tuple(missing), tuple(failed))

    def validate_openxr_async(self) -> OpenXRAsyncValidation:
        with self.lock:
            stats = dict(self.stats)
        return self._validate_openxr_async_stats(stats)

    def add_runtime_timing(self, runtime_result) -> None:
        if not self.enabled:
            return
        timing = getattr(runtime_result, "timing", None) or {}
        debug = getattr(runtime_result, "debug_info", None) or {}
        with self.lock:
            for key in ("depth_total_ms", "depth_model_ms", "depth_slot_wait_ms", "synthesis_ms", "pack_ms", "total_ms",
                        "sbs_host_ms", "sbs_host_device_ms", "sbs_host_copy_ms", "sbs_host_numpy_ms",
                        "packer_ms"):
                value = timing.get(key)
                if value is not None:
                    self.stats[f"rt_{key}"] = float(value)
            self.stats["rt_backend"] = str(debug.get("backend", "unknown"))
            self.stats["rt_depth_backend"] = str(debug.get("runtime_depth_backend", "unknown"))
            self.stats["rt_depth_slot"] = (
                debug.get("runtime_depth_execution_slot")
                if debug.get("runtime_depth_execution_slot") is not None
                else "n/a"
            )
            self.stats["rt_depth_slot_count"] = int(
                debug.get("runtime_depth_execution_slot_count", 1) or 1
            )
            self.stats["rt_output_format"] = str(
                getattr(runtime_result, "output_format", None) or debug.get("runtime_output_format", "unknown")
            )
            self.stats["rt_output_dtype"] = str(
                getattr(runtime_result, "output_dtype", None) or debug.get("runtime_output_dtype", "unknown")
            )
            self.stats["rt_output_pack"] = str(
                getattr(runtime_result, "output_pack_backend", None) or debug.get("runtime_output_pack_backend", "n/a")
            )
            if "sbs_backend" in debug:
                self.stats["rt_sbs_backend"] = str(debug.get("sbs_backend"))
            if "openxr_prewarp_backend" in debug:
                self.stats["rt_openxr_prewarp_backend"] = str(debug.get("openxr_prewarp_backend"))
            if "openxr_no_fill_fused_reason" in debug:
                self.stats["rt_no_fill_fused_reason"] = str(debug.get("openxr_no_fill_fused_reason"))
            for debug_key, stat_key in (
                ("runtime_temporal_enabled", "rt_temporal_enabled"),
                ("runtime_refine_enabled", "rt_refine_enabled"),
                ("runtime_occlusion_enabled", "rt_occlusion_enabled"),
            ):
                if debug_key in debug:
                    self.stats[stat_key] = int(bool(debug.get(debug_key)))
            if "occlusion_mask_backend" in debug:
                self.stats["rt_occ_backend"] = str(debug.get("occlusion_mask_backend"))
            if "hole_fill_backend" in debug:
                self.stats["rt_fill_backend"] = str(debug.get("hole_fill_backend"))
            if "hole_fill_mode" in debug:
                self.stats["rt_hole_fill_mode"] = str(debug.get("hole_fill_mode"))
            if "hole_fill_radius" in debug:
                self.stats["rt_hole_fill_radius"] = int(debug.get("hole_fill_radius", 0))
            if "hole_fill_strength" in debug:
                self.stats["rt_hole_fill_strength"] = float(debug.get("hole_fill_strength", 0.0))
            if "parallel_inference_enabled" in debug:
                self.stats["parallel_inference_enabled"] = int(debug.get("parallel_inference_enabled", 0))
            if "parallel_inference_pending_limit" in debug:
                self.stats["parallel_inference_pending_limit"] = int(debug.get("parallel_inference_pending_limit", 1))
            if "parallel_inference_workers" in debug:
                self.stats["parallel_inference_workers"] = int(debug.get("parallel_inference_workers", 0))
            if "parallel_inference_effective_workers" in debug:
                self.stats["parallel_inference_effective_workers"] = int(debug.get("parallel_inference_effective_workers", 0))
            if "parallel_inference_backoff" in debug:
                self.stats["parallel_inference_backoff"] = int(debug.get("parallel_inference_backoff", 0))
            if "parallel_inference_adaptive_backoff" in debug:
                self.stats["parallel_inference_adaptive_backoff"] = int(debug.get("parallel_inference_adaptive_backoff", 0))
            if "parallel_inference_pending" in debug:
                self.stats["parallel_inference_pending"] = int(debug.get("parallel_inference_pending", 0))
            if "parallel_inference_dropped" in debug:
                self.stats["parallel_inference_dropped"] = int(debug.get("parallel_inference_dropped", 0))
            if "openxr_vulkan_projection_quality_chain_requested" in debug:
                self.stats["openxr_vulkan_projection_quality_chain_requested"] = int(
                    bool(debug.get("openxr_vulkan_projection_quality_chain_requested", 0))
                )
            if "fast_plus_fused_backend" in debug:
                self.stats["rt_fast_plus_fused_backend"] = str(debug.get("fast_plus_fused_backend"))
            if "fast_plus_fused_skip" in debug:
                self.stats["rt_fast_plus_fused_skip"] = str(debug.get("fast_plus_fused_skip"))
            if "fast_plus_fused_temporal_bypass" in debug:
                self.stats["rt_fast_plus_fused_temporal_bypass"] = str(debug.get("fast_plus_fused_temporal_bypass"))
            for debug_key, stat_key in (
                ("depth_strength", "rt_depth_strength"),
                ("convergence", "rt_convergence"),
                ("max_disparity_px", "rt_max_disparity_px"),
                ("resolved_max_disparity_px", "rt_resolved_max_disparity_px"),
                ("parallax_preset", "rt_parallax_preset"),
                ("parallax_budget_preset", "rt_parallax_budget_preset"),
            ):
                if debug_key in debug:
                    self.stats[stat_key] = str(debug.get(debug_key))

    def log(self, now: float | None = None) -> None:
        if not self.enabled:
            return
        now = time.perf_counter() if now is None else now
        if self.log_limit > 0 and self.log_count >= self.log_limit:
            return
        elapsed = now - self.last_log
        if elapsed < self.log_delay:
            return
        with self.lock:
            self.log_count += 1
            stats = dict(self.stats)
            for key in list(self.stats.keys()):
                if key in LATEST_KEYS:
                    continue
                self.stats[key] = 0.0 if key.endswith("_ms") else 0
            self.last_log = now

        def rate(name: str) -> float:
            return stats.get(name, 0) / elapsed

        def avg_ms(name: str) -> float:
            count = stats.get(f"{name}_count", 0)
            return stats.get(f"{name}_ms", 0.0) / count if count else 0.0

        def avg_value(name: str) -> float:
            count = stats.get(f"{name}_count", 0)
            return stats.get(f"{name}_total", 0.0) / count if count else 0.0

        quad_unavailable = ",".join(
            f"{key[len('openxr_quad_unavailable_'):]}:{value / elapsed:.1f}"
            for key, value in sorted(stats.items())
            if key.startswith("openxr_quad_unavailable_") and value
        ) or "none"

        async_validation = self._validate_openxr_async_stats(stats)
        async_missing = ",".join(async_validation.missing) or "none"
        async_failed = ",".join(async_validation.failed) or "none"

        print(
            "[FPSBreakdown] "
            f"sample={self.log_count} window={elapsed:.2f}s "
            f"target={self.target_fps}Hz "
            f"capture_target={int(stats.get('adaptive_capture_target_fps', self.target_fps))}Hz "
            f"cap={rate('capture'):.1f} raw={rate('raw_get'):.1f} "
            f"overwrite={rate('raw_overwritten'):.1f} drain_drop={rate('raw_dropped_stale'):.1f} "
            f"runtime={rate('runtime'):.1f} rt_overwrite={rate('runtime_overwrite'):.1f} "
            f"rt_motion_skip={rate('runtime_motion_skip'):.1f} "
            f"rt_backpressure_drop={rate('runtime_drop_backpressure'):.1f} "
            f"rt_cuda_inflight_drop={rate('runtime_drop_cuda_inflight'):.1f} "
            f"rt_pending_cuda={rate('runtime_pending_cuda'):.1f} "
            f"rt_pending_publish={rate('runtime_pending_cuda_wait'):.1f} "
            f"rt_pending_inflight={rate('runtime_pending_cuda_inflight'):.1f} "
            f"rt_pending_wait={avg_ms('rt_pending_wait'):.2f}ms "
            f"rt_pending_age={avg_ms('rt_pending_age'):.2f}ms "
            f"rt_parallel={int(stats.get('parallel_inference_enabled', 0))} "
            f"rt_parallel_workers={int(stats.get('parallel_inference_workers', 0))} "
            f"rt_parallel_effective_workers={int(stats.get('parallel_inference_effective_workers', 0))} "
            f"rt_parallel_backoff={int(stats.get('parallel_inference_backoff', 0))} "
            f"rt_parallel_adaptive={int(stats.get('parallel_inference_adaptive_backoff', 0))} "
            f"rt_presenter_bp={rate('runtime_presenter_backpressure'):.1f} "
            f"rt_pending_limit={int(stats.get('parallel_inference_pending_limit', 1))} "
            f"viewer_get={rate('viewer_get'):.1f} "
            f"viewer_drop={rate('viewer_drop'):.1f} "
            f"local_present={rate('local_presented_frame'):.1f} "
            f"local_present_ms={avg_ms('local_present'):.2f}ms "
            f"local_present_pack_ms={avg_ms('local_present_pack'):.2f}ms "
            f"screen_new={rate('openxr_new_screen_frame'):.1f} "
            f"screen_reuse={rate('openxr_reused_screen_frame'):.1f} "
            f"screen_proj={rate('openxr_projection_screen_present'):.1f} "
            f"screen_quad_requested={int(bool(stats.get('openxr_screen_quad_reprojection_requested', False)))} "
            f"screen_quad_active={int(bool(stats.get('openxr_screen_quad_reprojection_active', False)))} "
            f"screen_quad_new={rate('openxr_screen_quad_new'):.1f} "
            f"screen_quad_reuse={rate('openxr_screen_quad_reuse'):.1f} "
            f"screen_quad_fallback={rate('openxr_screen_quad_fallback'):.1f} "
            f"vk_composer_requested={int(bool(stats.get('openxr_vulkan_projection_composer_requested', False)))} "
            f"vk_composer_active={int(bool(stats.get('openxr_vulkan_projection_composer_active', False)))} "
            f"vk_quality_chain={int(bool(stats.get('openxr_vulkan_projection_quality_chain_requested', False)))} "
            f"vk_composer_frame={int(stats.get('openxr_vulkan_projection_composer_frame_id', -1))} "
            f"vk_composer_fallback={rate('openxr_vulkan_projection_composer_fallback'):.1f} "
            f"vk_rcas={rate('openxr_vulkan_composer_rcas'):.1f} "
            f"vk_rcas_skip={rate('openxr_vulkan_composer_rcas_skip'):.1f} "
            f"vk_mip={rate('openxr_vulkan_composer_mip'):.1f} "
            f"vk_mip_skip={rate('openxr_vulkan_composer_mip_skip'):.1f} "
            f"vk_mip_tpl_hit={rate('openxr_vulkan_mip_template_hit'):.1f} "
            f"vk_mip_tpl_new={rate('openxr_vulkan_mip_template_new'):.1f} "
            f"vk_rp_tpl_hit={rate('openxr_vulkan_render_pass_template_hit'):.1f} "
            f"vk_rp_tpl_new={rate('openxr_vulkan_render_pass_template_new'):.1f} "
            f"vk_barrier_tpl_hit={rate('openxr_vulkan_image_barrier_template_hit'):.1f} "
            f"vk_barrier_tpl_new={rate('openxr_vulkan_image_barrier_template_new'):.1f} "
            f"xr_path={stats.get('openxr_projection_path', 'uninitialized')} "
            f"screen_age={avg_value('openxr_screen_frame_age_frames'):.2f}f "
            f"screen_quality_failed={rate('openxr_screen_quality_failed'):.1f} "
            f"source_lat={avg_ms('openxr_source_latency'):.2f}ms "
            f"loop={rate('loops'):.1f} "
            f"xr_loop={rate('openxr_loop'):.1f} "
            f"xr_input_fps={rate('openxr_input_sample'):.1f} "
            f"xr_present_fps={rate('openxr_presented_frame'):.1f} "
            f"xr_should={rate('openxr_should_render'):.1f} "
            f"xr_no_render={rate('openxr_no_render'):.1f} "
            f"xr_no_fresh={rate('openxr_no_fresh'):.1f} "
            f"xr_no_renderable={rate('openxr_no_renderable'):.1f} "
            f"update={avg_ms('update'):.2f}ms "
            f"render={avg_ms('render'):.2f}ms "
            f"post={avg_ms('post'):.2f}ms "
            f"swap={avg_ms('swap'):.2f}ms "
            f"wait={avg_ms('wait'):.2f}ms "
            f"openxr_poll={avg_ms('openxr_poll'):.2f}ms "
            f"openxr_upload={avg_ms('openxr_upload'):.2f}ms "
            f"eye_total={avg_ms('runtime_eye_total'):.2f}ms "
            f"eye_tensor={avg_ms('runtime_eye_tensor'):.2f}ms "
            f"eye_image={avg_ms('runtime_eye_image'):.2f}ms "
            f"eye_d3d11={avg_ms('runtime_eye_d3d11'):.2f}ms "
            f"eye_mipmap={avg_ms('runtime_eye_mipmap'):.2f}ms "
            f"fx_total={avg_ms('runtime_effect_source_total'):.2f}ms "
            f"fx_enabled={int(bool(stats.get('openxr_async_effects_enabled', True)))} "
            f"fx_tensor={avg_ms('runtime_effect_source_tensor'):.2f}ms "
            f"fx_upload={avg_ms('runtime_effect_source_upload'):.2f}ms "
            f"fx_submit={avg_ms('openxr_effect_submit'):.2f}ms "
            f"fx_age={avg_value('openxr_effect_ready_age_frames'):.2f}f "
            f"fx_age_failed={rate('openxr_effect_ready_age_record_failed'):.1f} "
            f"fx_ready={rate('openxr_effect_source_ready_publish'):.1f} "
            f"fx_safe={rate('openxr_effect_source_safe_publish'):.1f} "
            f"fx_promote_reuse={rate('openxr_effect_source_promote_reuse'):.1f} "
            f"fx_promote_failed={rate('openxr_effect_source_promote_failed'):.1f} "
            f"fx_source_reuse={rate('openxr_screen_effect_source_reuse'):.1f} "
            f"fx_safe_reuse={rate('openxr_effect_source_reused_safe'):.1f} "
            f"fx_skip={rate('openxr_effect_source_interval_skip'):.1f} "
            f"fx_overwrite={rate('openxr_effect_submit_overwrite'):.1f} "
            f"fx_budget_skip={rate('openxr_effect_submit_budget_skip'):.1f} "
            f"fx_submit_failed={rate('openxr_effect_submit_failed'):.1f} "
            f"fx_ds_render={rate('openxr_glow_downsample_render'):.1f} "
            f"fx_ds_reuse={rate('openxr_glow_downsample_reuse'):.1f} "
            f"fx_ds_prewarm={rate('openxr_effect_downsample_prewarm'):.1f} "
            f"fx_ds_prewarm_ms={avg_ms('openxr_effect_downsample_prewarm'):.2f}ms "
            f"fx_ds_failed={rate('openxr_glow_downsample_failed'):.1f},"
            f"prewarm:{rate('openxr_effect_downsample_prewarm_failed'):.1f},"
            f"prewarm_skip:{rate('openxr_effect_downsample_prewarm_skip'):.1f} "
            f"fx_entry_failed=bg:{rate('openxr_screen_background_effect_failed'):.1f},"
            f"fg:{rate('openxr_screen_foreground_effect_failed'):.1f} "
            f"light_ds={rate('openxr_screen_light_downsample_source'):.1f} "
            f"light_reuse={rate('openxr_screen_light_source_reuse'):.1f} "
            f"light_source_failed={rate('openxr_screen_light_source_failed'):.1f} "
            f"light_bind_failed={rate('openxr_screen_light_bind_failed'):.1f} "
            f"wall_mask="
            f"loaded:{rate('openxr_wall_light_mask_loaded'):.1f},"
            f"missing:{rate('openxr_wall_light_mask_missing'):.1f},"
            f"disabled:{rate('openxr_wall_light_mask_disabled'):.1f},"
            f"failed:{rate('openxr_wall_light_mask_failed'):.1f} "
            f"xr_poll0={avg_ms('openxr_poll_no_upload'):.2f}ms "
            f"xr_events={avg_ms('openxr_events'):.2f}ms "
            f"xr_wait={avg_ms('openxr_wait_frame'):.2f}ms "
            f"xr_cmd={avg_ms('openxr_presenter_commands'):.2f}ms "
            f"swapchain_wait={avg_ms('openxr_swapchain_wait'):.2f}ms "
            f"xr_acquire_pair={avg_ms('openxr_projection_acquire_pair'):.2f}ms "
            f"xr_wait_l={avg_ms('openxr_projection_wait_eye0'):.2f}ms "
            f"xr_wait_r={avg_ms('openxr_projection_wait_eye1'):.2f}ms "
            f"xr_shared={avg_ms('openxr_projection_shared_prepare'):.2f}ms "
            f"filament_lock_wait={avg_ms('openxr_filament_lock_wait'):.2f}ms "
            f"vk_composer_fallback_prepare={avg_ms('openxr_vulkan_composer_fallback_prepare'):.2f}ms "
            f"mv_source_prepare={avg_ms('openxr_filament_multiview_source_prepare'):.2f}ms "
            f"mv_source_wait={avg_ms('openxr_filament_multiview_source_wait'):.2f}ms "
            f"mv_state={avg_ms('openxr_filament_multiview_state'):.2f}ms "
            f"mv_queue={avg_ms('openxr_filament_multiview_queue'):.2f}ms "
            f"mv_finish_wait={avg_ms('openxr_filament_multiview_finish_wait'):.2f}ms "
            f"eye0_queue={avg_ms('openxr_filament_eye0_queue'):.2f}ms "
            f"eye0_source_prepare={avg_ms('openxr_filament_eye0_source_prepare'):.2f}ms "
            f"eye0_state={avg_ms('openxr_filament_eye0_state'):.2f}ms "
            f"eye0_deferred={avg_ms('openxr_filament_eye0_deferred_submit'):.2f}ms "
            f"eye0_finish_wait={avg_ms('openxr_filament_eye0_finish_wait'):.2f}ms "
            f"eye1_queue={avg_ms('openxr_filament_eye1_queue'):.2f}ms "
            f"eye1_source_prepare={avg_ms('openxr_filament_eye1_source_prepare'):.2f}ms "
            f"eye1_state={avg_ms('openxr_filament_eye1_state'):.2f}ms "
            f"eye1_deferred={avg_ms('openxr_filament_eye1_deferred_submit'):.2f}ms "
            f"eye1_finish_wait={avg_ms('openxr_filament_eye1_finish_wait'):.2f}ms "
            f"controller_eye0={avg_ms('openxr_filament_controller_overlay_eye0'):.2f}ms "
            f"controller_eye1={avg_ms('openxr_filament_controller_overlay_eye1'):.2f}ms "
            f"stereo_finish_wait={avg_ms('openxr_filament_stereo_finish_wait'):.2f}ms "
            f"filament_drain={avg_ms('openxr_filament_completion_drain'):.2f}ms "
            f"xr_release_pair={avg_ms('openxr_projection_release_pair'):.2f}ms "
            f"xr_projection_total={avg_ms('openxr_projection_total'):.2f}ms "
            f"xr_projection_layer={avg_ms('openxr_projection_layer'):.2f}ms "
            f"screen_quad_upload={avg_ms('openxr_screen_quad_upload'):.2f}ms "
            f"vk_composer_source={avg_ms('openxr_vulkan_composer_source_prepare'):.2f}ms "
            f"vk_composer_draw={avg_ms('openxr_vulkan_composer_draw_prepare'):.2f}ms "
            f"vk_composer_submit={avg_ms('openxr_vulkan_composer_submit'):.2f}ms "
            f"vk_composer_fence={avg_ms('openxr_vulkan_composer_fence_wait'):.2f}ms "
            f"vk_composer_record={avg_ms('openxr_vulkan_composer_record'):.2f}ms "
            f"vk_composer_queue={avg_ms('openxr_vulkan_composer_queue_submit'):.2f}ms "
            f"vk_out_convert={avg_ms('openxr_vulkan_output_convert'):.2f}ms "
            f"vk_input_wait={avg_ms('openxr_vulkan_input_slot_wait'):.2f}ms "
            f"vk_input_upload={avg_ms('openxr_vulkan_input_upload'):.2f}ms "
            f"xr_pred={avg_ms('openxr_predicted_period'):.2f}ms "
            f"xr_submit={avg_ms('openxr_submit_frame'):.2f}ms "
            f"xr_begin={avg_ms('openxr_begin_frame'):.2f}ms "
            f"xr_sync={avg_ms('openxr_sync_actions'):.2f}ms "
            f"xr_pose={avg_ms('openxr_controller_pose'):.2f}ms "
            f"xr_input={avg_ms('openxr_controller_input'):.2f}ms "
            f"xr_controls={avg_ms('openxr_controls'):.2f}ms "
            f"input_trigger_failed={rate('openxr_input_trigger_failed'):.1f} "
            f"xr_poll1={avg_ms('openxr_poll_upload'):.2f}ms "
            f"d3d11_upload={avg_ms('openxr_d3d11_upload'):.2f}ms "
            f"d3d11_pbo={rate('openxr_d3d11_pbo_readback'):.1f} "
            f"xr_locate={avg_ms('openxr_locate_views'):.2f}ms "
            f"xr_locate_call={avg_ms('openxr_locate_call'):.2f}ms "
            f"xr_view_prepare={avg_ms('openxr_view_prepare'):.2f}ms "
            f"xr_output_lock={avg_ms('openxr_output_lock'):.2f}ms "
            f"xr_overlay_metrics={avg_ms('openxr_overlay_metrics'):.2f}ms "
            f"xr_locate_total={avg_ms('openxr_locate_total'):.2f}ms "
            f"xr_render={avg_ms('openxr_render_eyes'):.2f}ms "
            f"proj_skip={rate('openxr_projection_layer_skipped'):.1f} "
            f"projection_failed={rate('openxr_projection_render_failed'):.1f} "
            f"overlay_failed={rate('openxr_overlay_render_failed'):.1f} "
            f"controller_failed={rate('openxr_controller_render_failed'):.1f} "
            f"laser_failed={rate('openxr_laser_render_failed'):.1f} "
            f"quad_update={avg_ms('openxr_quad_update'):.2f}ms "
            f"quad_upload={avg_ms('openxr_quad_upload'):.2f}ms "
            f"quad_total={avg_ms('openxr_quad_total'):.2f}ms "
            f"xr_output_commit={avg_ms('openxr_output_commit'):.2f}ms "
            f"xr_output_release={avg_ms('openxr_output_release'):.2f}ms "
            f"xr_layer_pointers={avg_ms('openxr_layer_pointers'):.2f}ms "
            f"xr_layer_assembly={avg_ms('openxr_layer_assembly'):.2f}ms "
            f"quad_reuse={rate('openxr_quad_reused_screen_frame'):.1f} "
            f"quad_failed={rate('openxr_quad_layer_failed'):.1f} "
            f"quad_unavail={quad_unavailable} "
            f"background={avg_ms('openxr_background'):.2f}ms "
            f"bg_upload={avg_ms('openxr_background_upload'):.2f}ms "
            f"bg_age={avg_value('openxr_background_safe_age_frames'):.2f}f "
            f"bg_reuse={rate('openxr_background_reuse'):.1f} "
            f"bg_path=layer:{rate('openxr_background_layer'):.1f},"
            f"upload:{rate('openxr_background_layer_upload'):.1f},"
            f"budget_skip:{rate('openxr_background_upload_budget_skip'):.1f},"
            f"upload_failed:{rate('openxr_background_layer_upload_failed'):.1f},"
            f"fallback:{rate('openxr_background_projection_fallback'):.1f},"
            f"layer_failed:{rate('openxr_background_layer_failed'):.1f},"
            f"panorama:{rate('openxr_background_panorama'):.1f},"
            f"env:{rate('openxr_background_env_model'):.1f},"
            f"env_failed:{rate('openxr_background_env_model_failed'):.1f},"
            f"idle:{rate('openxr_background_idle'):.1f} "
            f"xr_layers={avg_ms('openxr_layers'):.2f}ms "
            f"layer_count={rate('openxr_layer_count'):.1f} "
            f"xr_no_layers={avg_ms('openxr_render_no_layers'):.2f}ms "
            f"xr_end={avg_ms('openxr_end_frame'):.2f}ms "
            f"xr_frame={avg_ms('openxr_frame_total'):.2f}ms "
            f"rt_loop={avg_ms('rt_loop'):.2f}ms "
            f"rt_cap2rgb={avg_ms('rt_cap2rgb'):.2f}ms "
            f"rt_prepare={avg_ms('rt_prepare'):.2f}ms "
            f"pre={stats.get('rt_preprocess_backend', 'unknown')} "
            f"rt_call={avg_ms('rt_call'):.2f}ms "
            f"rt_put={avg_ms('rt_put'):.2f}ms "
            f"rt_gpu_total={avg_ms('rt_gpu_total'):.2f}ms "
            f"rt_gpu_depth={avg_ms('rt_gpu_depth'):.2f}ms "
            f"rt_gpu_depth_pre={avg_ms('rt_gpu_depth_preprocess'):.2f}ms "
            f"rt_gpu_depth_model={avg_ms('rt_gpu_depth_model'):.2f}ms "
            f"rt_gpu_depth_norm={avg_ms('rt_gpu_depth_normalize'):.2f}ms "
            f"rt_gpu_depth_up={avg_ms('rt_gpu_depth_upsample'):.2f}ms "
            f"rt_gpu_depth_post={avg_ms('rt_gpu_depth_postprocess'):.2f}ms "
            f"rt_gpu_synth={avg_ms('rt_gpu_synth'):.2f}ms "
            f"rt_gpu_syn_scene={avg_ms('rt_gpu_synth_scene'):.2f}ms "
            f"rt_gpu_syn_depth_post={avg_ms('rt_gpu_synth_depth_postprocess'):.2f}ms "
            f"rt_gpu_syn_shift_response={avg_ms('rt_gpu_synth_shift_response'):.2f}ms "
            f"rt_gpu_syn_shift={avg_ms('rt_gpu_synth_depth_shift'):.2f}ms "
            f"rt_gpu_syn_warp={avg_ms('rt_gpu_synth_warp'):.2f}ms "
            f"rt_gpu_syn_occ={avg_ms('rt_gpu_synth_occ'):.2f}ms "
            f"rt_gpu_syn_fill={avg_ms('rt_gpu_synth_fill'):.2f}ms "
            f"rt_gpu_syn_refine={avg_ms('rt_gpu_synth_refine'):.2f}ms "
            f"rt_gpu_syn_temporal={avg_ms('rt_gpu_synth_temporal'):.2f}ms "
            f"rt_gpu_syn_out_depth={avg_ms('rt_gpu_synth_output_depth'):.2f}ms "
            f"rt_gpu_syn_sbs={avg_ms('rt_gpu_synth_sbs'):.2f}ms "
            f"rt_gpu_pack={avg_ms('rt_gpu_pack'):.2f}ms "
            f"rt_gpu_openxr_pack={avg_ms('rt_gpu_openxr_pack'):.2f}ms "
            f"rt_backend={stats.get('rt_backend', 'unknown')} "
            f"rt_depth={stats.get('rt_depth_total_ms', 0.0):.2f}ms "
            f"rt_model={stats.get('rt_depth_model_ms', 0.0):.2f}ms "
            f"rt_slot_wait={stats.get('rt_depth_slot_wait_ms', 0.0):.2f}ms "
            f"rt_synth={stats.get('rt_synthesis_ms', 0.0):.2f}ms "
            f"rt_total={stats.get('rt_total_ms', 0.0):.2f}ms "
            f"rt_sbs_host={stats.get('rt_sbs_host_ms', 0.0):.2f}ms "
            f"rt_sbs_host_copy={stats.get('rt_sbs_host_copy_ms', 0.0):.2f}ms "
            f"rt_sbs_host_numpy={stats.get('rt_sbs_host_numpy_ms', 0.0):.2f}ms "
            f"rt_sbs_host_presync={stats.get('rt_sbs_host_presync_ms', 0.0):.2f}ms "
            f"packer={stats.get('packer_ms', 0.0):.2f}ms "
            f"rt_depth_backend={stats.get('rt_depth_backend', 'unknown')} "
            f"rt_depth_slot={stats.get('rt_depth_slot', 'n/a')}/{stats.get('rt_depth_slot_count', 1)} "
            f"rt_out={stats.get('rt_output_dtype', 'unknown')} "
            f"rt_pack={stats.get('rt_output_pack', 'n/a')} "
            f"rt_sbs={stats.get('rt_sbs_backend', 'unknown')} "
            f"rt_prewarp={stats.get('rt_openxr_prewarp_backend', 'n/a')} "
            f"rt_no_fill={stats.get('rt_no_fill_fused_reason', 'n/a')} "
            f"rt_temporal={int(stats.get('rt_temporal_enabled', 0))} "
            f"rt_refine={int(stats.get('rt_refine_enabled', 0))} "
            f"rt_occlusion={int(stats.get('rt_occlusion_enabled', 0))} "
            f"rt_depth_strength={stats.get('rt_depth_strength', 'n/a')} "
            f"rt_convergence={stats.get('rt_convergence', 'n/a')} "
            f"rt_max_disp={stats.get('rt_resolved_max_disparity_px', stats.get('rt_max_disparity_px', 'n/a'))} "
            f"rt_parallax={stats.get('rt_parallax_budget_preset', stats.get('rt_parallax_preset', 'n/a'))} "
            f"rt_occ={stats.get('rt_occ_backend', 'n/a')} "
            f"rt_fill={stats.get('rt_fill_backend', 'n/a')} "
            f"rt_hole_fill={stats.get('rt_hole_fill_mode', 'n/a')}"
            f"({stats.get('rt_hole_fill_radius', 'n/a')}/{float(stats.get('rt_hole_fill_strength', 0.0)):.2f}) "
            f"rt_fused={stats.get('rt_fast_plus_fused_backend', 'n/a')} "
            f"rt_fused_skip={stats.get('rt_fast_plus_fused_skip', 'n/a')} "
            f"rt_fused_temporal_bypass={stats.get('rt_fast_plus_fused_temporal_bypass', 'n/a')} "
            f"openxr_async_ok={int(async_validation.passed)} "
            f"openxr_async_missing={async_missing} "
            f"openxr_async_failed={async_failed}",
            flush=True,
        )
