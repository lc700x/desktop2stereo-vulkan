from __future__ import annotations

from dataclasses import dataclass

from streaming.config import resolve_streaming_config
from stereo_runtime.depth_settings import resolve_depth_settings
from stereo_runtime.model_capabilities import model_name_mapping
from stereo_runtime.render_size import RenderSizeConfig
from viewer.settings import resolve_viewer_settings

from .capture_tool import resolve_capture_tool
from .run_mode import resolve_run_mode
from utils.display_info import enumerate_displays


@dataclass(frozen=True)
class RuntimeExports:
    model_mapping: dict
    stream_quality: int
    stream_port: int
    local_ip: str
    run_mode: str
    stream_mode: bool
    use_3d_monitor: bool
    lossless_scaling_support: bool
    model: str
    model_id: str
    all_models: dict
    cache_path: str
    depth_resolution: int
    device_id: int
    fp16: bool
    monitor_index: int
    display_mode: str
    stereo_display_index: int | None
    stereo_display_selection: bool
    output_resolution: int | tuple[int, int]
    render_size_config: RenderSizeConfig
    show_fps: bool
    depth_strength: float
    convergence: float
    capture_mode: str
    window_title: str | None
    target_fps: int
    fps: int
    foreground_scale: float
    aa_strength: float
    use_torch_compile: bool
    use_tensorrt: bool
    recompile_trt: bool
    use_migraphx: bool
    recompile_migraphx: bool
    use_coreml: bool
    recompile_coreml: bool
    use_openvino: bool
    recompile_openvino: bool
    capture_tool: str
    fill_16_9: bool
    local_vsync: bool
    upscaler: str
    upscaler_sharpness: float
    fix_viewer_aspect: bool
    stereo_mix_device: str | None
    stream_key: str
    audio_delay: float
    crf: int
    language: str
    controller_help_rows: list
    environment_help_rows: list
    controller_model: str
    environment_model: str
    xr_headset_model: str
    openxr_screen_width: float
    openxr_screen_distance: float
    xr_preview_window: bool


def resolve_runtime_exports(settings: dict, *, os_name: str) -> RuntimeExports:
    model_mapping = model_name_mapping()
    streaming_config = resolve_streaming_config(settings)
    display_count = len(enumerate_displays()) if os_name == "Windows" else None
    depth_settings = resolve_depth_settings(settings, os_name=os_name)
    viewer_settings = resolve_viewer_settings(settings)
    run_mode_config = resolve_run_mode(
        settings["Run Mode"],
        os_name=os_name,
        fix_viewer_aspect=settings["Fix Viewer Aspect"],
        lossless_scaling_support=settings.get(
            "NVIDIA Frame Generation",
            settings.get("Lossless Scaling Support", False),
        ),
        monitor_count=display_count,
        capture_mode=settings.get("Capture Mode"),
        input_monitor_index=viewer_settings.monitor_index,
        output_monitor_index=viewer_settings.stereo_display_index,
    )

    return RuntimeExports(
        model_mapping=model_mapping,
        stream_quality=streaming_config.stream_quality,
        stream_port=streaming_config.stream_port,
        local_ip=streaming_config.local_ip,
        run_mode=run_mode_config.run_mode,
        stream_mode=run_mode_config.stream_mode,
        use_3d_monitor=run_mode_config.use_3d_monitor,
        lossless_scaling_support=run_mode_config.lossless_scaling_support,
        model=depth_settings.model,
        model_id=depth_settings.model_id,
        all_models=depth_settings.all_models,
        cache_path=depth_settings.cache_path,
        depth_resolution=depth_settings.depth_resolution,
        device_id=depth_settings.device_id,
        fp16=depth_settings.fp16,
        monitor_index=viewer_settings.monitor_index,
        display_mode=viewer_settings.display_mode,
        stereo_display_index=viewer_settings.stereo_display_index,
        stereo_display_selection=viewer_settings.stereo_display_selection,
        output_resolution=viewer_settings.output_resolution,
        render_size_config=viewer_settings.render_size_config,
        show_fps=viewer_settings.show_fps,
        depth_strength=viewer_settings.depth_strength,
        convergence=viewer_settings.convergence,
        capture_mode=viewer_settings.capture_mode,
        window_title=viewer_settings.window_title,
        target_fps=viewer_settings.target_fps,
        fps=viewer_settings.fps,
        foreground_scale=depth_settings.foreground_scale,
        aa_strength=depth_settings.aa_strength,
        use_torch_compile=depth_settings.use_torch_compile,
        use_tensorrt=depth_settings.use_tensorrt,
        recompile_trt=depth_settings.recompile_trt,
        use_migraphx=depth_settings.use_migraphx,
        recompile_migraphx=depth_settings.recompile_migraphx,
        use_coreml=depth_settings.use_coreml,
        recompile_coreml=depth_settings.recompile_coreml,
        use_openvino=depth_settings.use_openvino,
        recompile_openvino=depth_settings.recompile_openvino,
        capture_tool=resolve_capture_tool(settings["Capture Tool"]),
        fill_16_9=viewer_settings.fill_16_9,
        local_vsync=viewer_settings.local_vsync,
        upscaler=viewer_settings.upscaler,
        upscaler_sharpness=viewer_settings.upscaler_sharpness,
        fix_viewer_aspect=run_mode_config.fix_viewer_aspect,
        stereo_mix_device=streaming_config.stereo_mix_device,
        stream_key=streaming_config.stream_key,
        audio_delay=streaming_config.audio_delay,
        crf=streaming_config.crf,
        language=viewer_settings.language,
        controller_help_rows=viewer_settings.controller_help_rows,
        environment_help_rows=viewer_settings.environment_help_rows,
        controller_model=viewer_settings.controller_model,
        environment_model=viewer_settings.environment_model,
        xr_headset_model=viewer_settings.xr_headset_model,
        openxr_screen_width=viewer_settings.openxr_screen_width,
        openxr_screen_distance=viewer_settings.openxr_screen_distance,
        xr_preview_window=viewer_settings.xr_preview_window,
    )
