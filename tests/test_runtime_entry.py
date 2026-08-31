import ast
import json
from pathlib import Path


from path_config import APP_ROOT, PROJECT_ROOT


ROOT = PROJECT_ROOT
RUNTIME_ENTRY = APP_ROOT / "app_runtime/runtime_entry.py"


def _load_environment_resolver():
    source = RUNTIME_ENTRY.read_text(encoding="utf-8")
    module = ast.parse(source)
    resolver = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_resolve_filament_environment_paths"
    )
    namespace = {"Path": Path, "json": json}
    exec(compile(ast.Module(body=[resolver], type_ignores=[]), str(RUNTIME_ENTRY), "exec"), namespace)
    return namespace["_resolve_filament_environment_paths"]


def test_stop_request_is_consumed_only_by_target_runtime(tmp_path: Path) -> None:
    from app_runtime.runtime_entry import _consume_stop_request

    request = tmp_path / "stop.request"
    request.write_text("4321", encoding="utf-8")

    assert not _consume_stop_request(str(request), expected_pid=1234)
    assert request.is_file()
    assert _consume_stop_request(str(request), expected_pid=4321)
    assert not request.exists()


def test_stop_request_watcher_sets_runtime_event(tmp_path: Path) -> None:
    import threading

    from app_runtime.runtime_entry import _watch_stop_request

    request = tmp_path / "stop.request"
    stopped = threading.Event()
    watcher = threading.Thread(
        target=_watch_stop_request,
        kwargs={
            "stop_request_path": str(request),
            "expected_pid": 4321,
            "stop_event": stopped,
            "poll_interval": 0.01,
        },
    )
    watcher.start()
    request.write_text("4321", encoding="utf-8")
    watcher.join(timeout=1.0)

    assert stopped.is_set()
    assert not watcher.is_alive()
    assert not request.exists()


def test_legacy_streamer_normalizes_to_mjpeg() -> None:
    from utils.run_mode import normalize_run_mode, resolve_run_mode

    assert normalize_run_mode("Legacy Streamer") == "MJPEG Streamer"
    resolved = resolve_run_mode(
        "Legacy Streamer",
        os_name="Windows",
        fix_viewer_aspect=False,
        lossless_scaling_support=True,
    )
    assert resolved.run_mode == "Viewer"
    assert resolved.stream_mode == "MJPEG"


def test_legacy_gpu_streamer_normalizes_to_advanced_stream_mode() -> None:
    from utils.run_mode import normalize_run_mode, resolve_run_mode

    assert normalize_run_mode("NVIDIA Streamer") == "RTMP Streamer"
    resolved = resolve_run_mode(
        "GPU Streamer",
        os_name="Windows",
        fix_viewer_aspect=False,
        lossless_scaling_support=True,
    )
    assert resolved.run_mode == "Viewer"
    assert resolved.stream_mode == "RTMP"
    assert resolved.fix_viewer_aspect


def test_network_stream_session_policy_uses_gpu_backends_in_advanced_mode(monkeypatch) -> None:
    from streaming.stream_session import (
        NetworkStreamSessionConfig,
        is_network_stream_mode,
        resolve_network_video_backend,
        supports_network_calibration,
    )

    assert is_network_stream_mode("RTMP Streamer")
    assert not is_network_stream_mode("GPU Streamer")
    assert not is_network_stream_mode("MJPEG Streamer")
    assert supports_network_calibration("RTMP Streamer", "WebRTC")
    assert not supports_network_calibration("GPU Streamer", "WebRTC")
    assert not supports_network_calibration("GPU Streamer", "RTSP")
    config = NetworkStreamSessionConfig.from_settings(
        {"Stream Protocol": "WebRTC", "Streamer Port": 1122}, fps=30
    )
    assert config.port == 1122
    assert config.fps == 30

    # Advanced Auto keeps a distinct lazy chain so vendor-native GPU encoding
    # is attempted before Vulkan. Explicit Vulkan remains unchanged.
    # The vendor chain is a Windows/CUDA/ROCm concept; pin the platform so
    # the assertions hold regardless of the host running the suite.
    monkeypatch.setattr("streaming.stream_session.sys.platform", "win32")
    nvidia_auto = resolve_network_video_backend(
        "RTMP Streamer", "auto", device_info="NVIDIA RTX 3090"
    )
    assert nvidia_auto.backend == "auto"
    assert nvidia_auto.reason.startswith(
        "Auto capability chain: NVIDIA -> Vulkan"
    )
    intel_auto = resolve_network_video_backend(
        "RTMP Streamer", "auto", device_info="Intel Arc"
    )
    assert intel_auto.backend == "auto"
    assert intel_auto.reason.startswith(
        "Auto capability chain: Intel -> Vulkan"
    )
    assert resolve_network_video_backend(
        "RTMP Streamer", "vulkan", device_info="NVIDIA RTX 3090"
    ).backend == "vulkan"
    # macOS has no vendor/Vulkan encoder: Auto must land on the FFmpeg
    # backend (VideoToolbox/libx264) instead of h264_vulkan.
    monkeypatch.setattr("streaming.stream_session.sys.platform", "darwin")
    mac_auto = resolve_network_video_backend(
        "RTMP Streamer", "auto", device_info="Apple Silicon (MPS)"
    )
    assert mac_auto.backend == "ffmpeg"
    assert "VideoToolbox" in mac_auto.reason


def test_only_windows_3d_display_is_excluded_from_capture() -> None:
    from app_runtime.runtime_entry import _exclude_local_output_from_capture

    assert not _exclude_local_output_from_capture(
        {"Run Mode": "Local Viewer"},
        os_name="Windows",
    )
    assert _exclude_local_output_from_capture(
        {"Run Mode": "3D Monitor"},
        os_name="Windows",
    )


def test_stream_and_non_windows_outputs_remain_capturable() -> None:
    from app_runtime.runtime_entry import _exclude_local_output_from_capture

    for run_mode in ("RTMP Streamer", "MJPEG Streamer", "Legacy Streamer"):
        assert not _exclude_local_output_from_capture(
            {"Run Mode": run_mode},
            os_name="Windows",
        )
    assert not _exclude_local_output_from_capture(
        {"Run Mode": "Local Viewer"},
        os_name="Linux",
    )


def test_direct_stream_output_uses_uint8_nvenc_and_fps_provider() -> None:
    source = RUNTIME_ENTRY.read_text(encoding="utf-8")
    stream_branch = source[source.index("elif configured_run_mode in {"):]

    uint8_enable = source.index(
        'os.environ["D2S_RUNTIME_OUTPUT_UINT8"] = "1"'
    )
    context_create = source.index("context = create_runtime_context(")
    assert uint8_enable < context_create
    assert "prefer_nvenc=" in stream_branch
    assert "NetworkStreamSessionConfig.from_settings" in stream_branch
    assert "display_mode=stream_config.display_mode" in stream_branch
    assert "supports_network_calibration" in stream_branch
    assert '"NVIDIA" in str(DEVICE_INFO).upper()' in stream_branch
    assert "show_fps_provider=callbacks.show_fps" in stream_branch
    assert "observe_sbs_fps if adaptive_capture_rate.enabled else None" in stream_branch


def test_openxr_starts_after_inference_load_and_first_ready_output() -> None:
    source = RUNTIME_ENTRY.read_text(encoding="utf-8")

    prepare = source.index("pipeline.prepare()")
    capture_start = source.index("capture_thread.start()")
    wait_ready = source.index(
        "_wait_for_runtime_ready(runtime_ready_event, pipeline_thread)"
    )
    presenter_create = source.index("presenter = OpenXrVulkanPresenter(")

    assert prepare < capture_start < wait_ready < presenter_create


def test_openxr_filament_screen_geometry_follows_gui_headset_model() -> None:
    from app_runtime.runtime_entry import _openxr_filament_config

    config = _openxr_filament_config(
        {
            "Environment Model": "Default",
            "XR Headset Model": "Pico 4 / 4 Ultra",
        }
    )

    assert config["filament_screen_distance"] == 20.0
    assert config["filament_screen_width"] == 23.09
    assert "headset_model" not in config
    assert "render_scale" not in config


def test_openxr_projection_config_is_separate_from_filament_config() -> None:
    from app_runtime.runtime_entry import _openxr_filament_config, _openxr_projection_config

    settings = {
        "Environment Model": "Default",
        "XR Headset Model": "Pico 4 / 4 Ultra",
        "Controller Model": "PICO",
        "Render Scale": "1K / 50%",
    }
    projection = _openxr_projection_config(settings)
    filament = _openxr_filament_config(settings)

    assert projection["headset_model"] == "Pico 4 / 4 Ultra"
    assert projection["render_scale"] == 1.0
    assert "render_scale" not in filament
    assert "headset_model" not in filament


def test_openxr_render_scale_keeps_non_4k_processing_inputs_at_native_projection_size():
    from app_runtime.runtime_entry import _resolve_openxr_render_scale

    settings = {"Render Scale": "1K / 50%"}

    assert _resolve_openxr_render_scale(settings, (1920, 1080)) == 1.0
    assert _resolve_openxr_render_scale(settings, (2560, 1440)) == 1.0
    assert _resolve_openxr_render_scale(settings, 1080) == 1.0


def test_openxr_render_scale_does_not_follow_inference_scale_for_4k_input():
    from app_runtime.runtime_entry import _resolve_openxr_render_scale

    assert _resolve_openxr_render_scale({"Render Scale": "1K / 50%"}, (3840, 2160)) == 1.0
    assert _resolve_openxr_render_scale({"Render Scale": "2K / 75%"}, 2160) == 1.0


def test_openxr_render_scale_uses_dedicated_persisted_setting(monkeypatch):
    from app_runtime.runtime_entry import _resolve_openxr_render_scale

    monkeypatch.delenv("D2S_OPENXR_RENDER_SCALE", raising=False)
    assert _resolve_openxr_render_scale({"OpenXR Render Scale": 0.5}) == 0.5
    assert _resolve_openxr_render_scale({"OpenXR Render Scale": 2.0}) == 2.0
    assert _resolve_openxr_render_scale({"OpenXR Render Scale": 8.0}) == 2.0

def test_openxr_filament_color_defaults_come_from_common_json() -> None:
    from app_runtime.runtime_entry import _openxr_filament_config

    config = _openxr_filament_config({"Environment Model": "Default"})

    assert config["filament_scene_exposure_ev"] == 2.0
    assert config["filament_skybox_brightness"] == 1.0
    assert config["filament_ambient_light_intensity_lux"] == 30000.0
    assert config["filament_controller_ambient_light_intensity_lux"] == 8000.0
    assert config["filament_controller_light_intensity_candela"] == 2000.0
    assert config["filament_controller_head_light_weight"] == 0.85
    assert config["filament_controller_top_light_weight"] == 0.6
    assert config["filament_controller_screen_light_enabled"] is True
    assert config["filament_controller_screen_light_intensity_lux"] == 500.0
    assert config["filament_controller_screen_light_sample_hz"] == 12.0
    assert config["filament_environment_screen_light_enabled"] is False
    assert config["filament_environment_screen_light_intensity_candela"] == 120.0
    assert config["filament_environment_screen_light_sample_hz"] == 12.0
    assert config["filament_glow_sample_hz"] == 30.0
    assert config["filament_glow_smoothing_seconds"] == 0.10


def test_openxr_environment_uses_selected_folder_and_profile_glb(tmp_path: Path) -> None:
    resolver = _load_environment_resolver()
    room = tmp_path / "xr_viewer/environments/3D_Artemis"
    room.mkdir(parents=True)
    (room / "profile.json").write_text(
        json.dumps({"glb": "environment-custom.glb"}),
        encoding="utf-8",
    )
    (room / "environment-custom.glb").write_bytes(b"glTF")

    glb_path, profile_path, panorama_path = resolver(
        {"Environment Model": "3D_Artemis"},
        tmp_path,
    )

    assert glb_path == room / "environment-custom.glb"
    assert profile_path == room / "profile.json"
    assert panorama_path is None


def test_openxr_environment_missing_profile_falls_back_to_default(
    tmp_path: Path,
) -> None:
    resolver = _load_environment_resolver()
    default = tmp_path / "xr_viewer/environments/Default"
    default.mkdir(parents=True)
    (default / "profile.json").write_text(
        json.dumps({"glb": None}),
        encoding="utf-8",
    )

    glb_path, profile_path, panorama_path = resolver(
        {"Environment Model": "MissingRoom"},
        tmp_path,
    )

    assert glb_path is None
    assert profile_path == default / "profile.json"
    assert panorama_path is None


def test_openxr_environment_without_selection_uses_default(tmp_path: Path) -> None:
    resolver = _load_environment_resolver()
    default = tmp_path / "xr_viewer/environments/Default"
    default.mkdir(parents=True)
    (default / "profile.json").write_text(
        json.dumps({"glb": None}),
        encoding="utf-8",
    )

    glb_path, profile_path, panorama_path = resolver({}, tmp_path)

    assert glb_path is None
    assert profile_path == default / "profile.json"
    assert panorama_path is None


def test_openxr_environment_resolves_hdr_panorama_image(tmp_path: Path) -> None:
    resolver = _load_environment_resolver()
    room = tmp_path / "xr_viewer/environments/hdr_room"
    room.mkdir(parents=True)
    (room / "profile.json").write_text(
        json.dumps({"environment_type": "panorama", "background": {"image": "room.hdr"}, "glb": None}),
        encoding="utf-8",
    )
    (room / "room.hdr").write_bytes(b"HDR")

    glb_path, profile_path, panorama_path = resolver(
        {"Environment Model": "hdr_room"}, tmp_path
    )

    assert glb_path is None
    assert profile_path == room / "profile.json"
    assert panorama_path == room / "room.hdr"
